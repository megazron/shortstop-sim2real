import math

import pytest

from sim2real_gap import HomingGate, VelocityRelay, deadband_audit, mrad_per_cycle, plan_home

CONT = (0, 2, 4, 6)


def test_deadband_audit_verdicts():
    assert deadband_audit(1.0, 0.305)["verdict"] == "CAUSE"
    assert abs(deadband_audit(1.0, 0.305)["ratio"] - 3.28) < 0.01
    assert deadband_audit(0.3, 0.305)["verdict"] == "TIGHT"
    assert deadband_audit(0.1, 0.305)["verdict"] == "UNRELATED"
    assert abs(mrad_per_cycle(0.05, 18.0) - 2.78) < 0.01
    with pytest.raises(ValueError):
        deadband_audit(0.0, 0.3)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def relay(**kw):
    c = Clock()
    args = dict(kp=0.5, vmax_rad_s=0.05, deadband_rad=math.radians(0.25), watchdog_s=0.5,
                continuous_idx=CONT, clock=c)
    args.update(kw)
    return VelocityRelay(**args), c


def test_deadband_suppresses_correction_but_not_feedforward():
    r, c = relay()
    actual = [0.0] * 7
    tgt = [0.001] * 7                      # 0.057 deg, inside the 0.25 deg deadband
    v = r.step(actual, target=tgt, target_vel=[0.02] * 7)
    assert all(abs(x - 0.02) < 1e-12 for x in v)     # feedforward passes, correction is zero
    v = r.step(actual, target=tgt, target_vel=None)
    r2, _ = relay()
    v2 = r2.step(actual, target=tgt)
    assert all(x == 0.0 for x in v2)                  # no ff, error inside band -> zero


def test_watchdog_zeros_speed():
    r, c = relay()
    r.step([0.0] * 7, target=[1.0] * 7)
    c.t += 0.6
    assert r.step([0.0] * 7) == [0.0] * 7
    assert "watchdog" in r.last_reason


def test_clip_is_on_speed_not_error():
    r, c = relay(kp=10.0)
    v = r.step([0.0] * 7, target=[2.0] * 7)
    assert all(abs(x - 0.05) < 1e-12 for x in v)
    v = r.step([0.0] * 7, target=[-2.0] * 7)
    assert all(abs(x + 0.05) < 1e-12 for x in v)


def test_feedforward_from_differentiated_setpoint_stream():
    r, c = relay(deadband_rad=1.0)          # huge deadband: correction always off
    r.step([0.0] * 7, target=[0.0] * 7)
    c.t += 0.1
    v = r.step([0.0] * 7, target=[0.002] * 7)   # 0.002 rad in 0.1 s = 0.02 rad/s
    assert all(abs(x - 0.02) < 1e-9 for x in v)


def test_monotonic_clock_injection_and_rate():
    r, c = relay()
    for _ in range(11):
        r.step([0.0] * 7, target=[0.0] * 7)
        c.t += 0.05
    assert abs(r.achieved_rate_hz - 20.0) < 1e-6
    r.record_send_latency(26.0)
    assert r.latency_summary_ms()["avg"] == 26.0


def test_terminal_overshoot_refused_with_zero_eps():
    with pytest.raises(ValueError):
        VelocityRelay(0.5, 0.05, 0.001, 0.5, CONT, terminal_overshoot=True, terminal_overshoot_rad=0.0)


def test_terminal_overshoot_moves_setpoint_in_direction_of_travel():
    r, c = relay(terminal_overshoot=True, terminal_overshoot_rad=math.radians(0.3), deadband_rad=0.0, kp=1.0)
    v = r.step([0.0] * 7, target=[0.01] * 7)
    # error seen by the law = 0.01 + 0.3deg
    assert all(abs(x - (0.01 + math.radians(0.3))) < 1e-9 for x in v)


def test_continuous_seam_in_relay_error():
    r, c = relay(kp=1.0, vmax_rad_s=10.0, deadband_rad=0.0)
    v = r.step([math.radians(170)] + [0.0] * 6, target=[math.radians(-170)] + [0.0] * 6)
    assert v[0] > 0 and abs(v[0] - math.radians(20)) < 1e-9


def test_homing_gate_refuses_and_accepts():
    g = HomingGate(0.05, CONT)
    q = [0.0] * 7
    ok, j, err = g.check([0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0], q)
    assert not ok and j == 3 and abs(err - 0.06) < 1e-12
    assert "REFUSING" in g.explain([0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0], q)
    ok, _, _ = g.check([0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0], q)
    assert ok
    # a continuous joint across the seam is a small difference, not a large one
    ok, _, err = g.check([math.radians(179)] + [0.0] * 6, [math.radians(-179)] + [0.0] * 6)
    assert ok and abs(math.degrees(err) - 2.0) < 1e-9


def test_plan_home_velocities_never_exceed_vmax():
    now = [0.0] * 7
    home = [0.5, -0.2, 0.1, 1.0, 0.0, -0.4, 0.05]
    plan = plan_home(now, home, vmax_rad_s=0.05, rate_hz=20.0, continuous_idx=CONT)
    assert abs(plan["duration_s"] - 20.0) < 1e-9
    for p in plan["points"]:
        assert max(abs(v) for v in p["velocities"]) <= 0.05 + 1e-12
    last = plan["points"][-1]
    assert max(abs(a - b) for a, b in zip(last["positions"], home)) < 1e-9
    assert last["velocities"] == [0.0] * 7
