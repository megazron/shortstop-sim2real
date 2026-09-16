"""Command line for sim2real-gap-kit.

    sim2real-gap synth -o runs.json                    # synthetic runs from a hidden EPS
    sim2real-gap fit runs.json -o model.json           # fit, run controls, write the model
    sim2real-gap correct model.json --arm left --from 0,0,0,0,0,0,0 --to 0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5
    sim2real-gap deadband --deadband-deg 1.0 --park-deg 0.305 [--rate-hz 18 --vmax 0.05]
    sim2real-gap selftest
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from .deadband import audit
from .fit import fit_runs, load_runs
from .model import NotCalibrated, TerminalOffsetModel

DIRS = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.replace(" ", "").split(",") if x]


def _ints(s: str) -> tuple:
    return tuple(int(x) for x in s.replace(" ", "").split(",") if x)


def synth_runs(n_reps: int = 3, eps_deg: dict | None = None, noise_deg: float = 0.02,
               n_joints: int = 7, continuous_idx=(0, 2, 4, 6), seed: int = 1) -> dict:
    """Runs from a hidden per-arm EPS: six axis directions x n_reps, each arm."""
    eps_deg = eps_deg or {"left": 0.3052, "right": 0.3059}
    rng = np.random.default_rng(seed)
    runs = []
    for arm, e_deg in eps_deg.items():
        e = math.radians(e_deg)
        q0 = rng.uniform(-1.0, 1.0, n_joints)
        for d in DIRS:
            # a fixed joint-space displacement per direction, so the same "axis" repeats
            disp = rng.uniform(0.15, 0.6, n_joints) * rng.choice([-1, 1], n_joints)
            disp[rng.integers(0, n_joints)] *= 0.001      # one joint barely moves
            for _ in range(n_reps):
                start = q0 + rng.normal(0, 0.01, n_joints)
                target = start + disp
                s = np.sign(disp)
                ach = target - s * e * (np.abs(disp) > e) + rng.normal(0, math.radians(noise_deg), n_joints)
                runs.append({"arm": arm, "direction": d, "q_start": start.tolist(),
                             "q_target": target.tolist(), "q_achieved": ach.tolist()})
    return {"runs": runs, "_synth": {"hidden_eps_deg": eps_deg, "noise_deg": noise_deg,
                                     "continuous_idx": list(continuous_idx)}}


def cmd_synth(a):
    d = synth_runs(a.reps, noise_deg=a.noise_deg)
    with open(a.out, "w") as f:
        json.dump(d, f, indent=1)
    print("wrote %d synthetic runs to %s (hidden EPS %s deg)" % (len(d["runs"]), a.out, d["_synth"]["hidden_eps_deg"]))
    return 0


def cmd_fit(a):
    runs = load_runs(a.runs)
    rep = fit_runs(runs, continuous_idx=_ints(a.continuous), seed=a.seed, source=a.runs)
    print(rep.summary())
    if a.out:
        try:
            m = rep.to_model(force=a.force)
        except ValueError as e:
            print("\nNOT WRITTEN: %s" % e)
            return 1
        m.save(a.out)
        print("\nmodel written to %s" % a.out)
    return 0 if rep.controls_passed else 1


def cmd_correct(a):
    try:
        m = TerminalOffsetModel.load(a.model, force=a.force)
        cmd, applied, why = m.correct(a.arm, _floats(a.q_from), _floats(a.q_to))
    except NotCalibrated as e:
        print("refused: %s" % e)
        return 1
    print("command:", ", ".join("%.6f" % x for x in cmd))
    print("applied:", applied)
    print(why)
    return 0


def cmd_deadband(a):
    r = audit(a.deadband_deg, a.park_deg, a.rate_hz, a.vmax)
    print("%s -- %s" % (r["verdict"], r["text"]))
    return 0


def cmd_selftest(a):
    d = synth_runs(3)
    rep = fit_runs(d["runs"], continuous_idx=(0, 2, 4, 6))
    ok = rep.controls_passed
    for arm, af in rep.arms.items():
        hid = d["_synth"]["hidden_eps_deg"][arm]
        got = math.degrees(af.eps_rad)
        good = abs(got - hid) / hid < 0.05
        ok &= good
        print("  %-5s hidden %.4f deg  fitted %.4f deg  %s" % (arm, hid, got, "OK" if good else "*** FAILED ***"))
    m = rep.to_model()
    q0 = [math.radians(170.0)] + [0.0] * 6
    q1 = [math.radians(-170.0)] + [0.0] * 6
    cmd, _, _ = m.correct("left", q0, q1)
    seam = (cmd[0] - q1[0]) > 0
    ok &= seam
    print("  continuous seam +170 -> -170 overshoots positively      %s" % ("OK" if seam else "*** FAILED ***"))
    print("selftest %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sim2real-gap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("synth", help="write synthetic runs from a hidden EPS")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--noise-deg", type=float, default=0.02)
    s.set_defaults(fn=cmd_synth)
    f = sp.add_parser("fit", help="fit EPS from runs and run the controls")
    f.add_argument("runs")
    f.add_argument("-o", "--out")
    f.add_argument("--continuous", default="0,2,4,6", help="indices of continuous joints (Gen3: 0,2,4,6)")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--force", action="store_true", help="write the model even if controls fail (loudly)")
    f.set_defaults(fn=cmd_fit)
    c = sp.add_parser("correct", help="the joint target to command so the arm arrives")
    c.add_argument("model")
    c.add_argument("--arm", required=True)
    c.add_argument("--from", dest="q_from", required=True)
    c.add_argument("--to", dest="q_to", required=True)
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cmd_correct)
    d = sp.add_parser("deadband", help="is the deadband the cause of the park error?")
    d.add_argument("--deadband-deg", type=float, required=True)
    d.add_argument("--park-deg", type=float, required=True)
    d.add_argument("--rate-hz", type=float)
    d.add_argument("--vmax", type=float, help="rad/s")
    d.set_defaults(fn=cmd_deadband)
    t = sp.add_parser("selftest", help="fit a hidden EPS end to end")
    t.set_defaults(fn=cmd_selftest)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
