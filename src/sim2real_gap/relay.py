"""A velocity relay for a position stream arriving over a slow link.

    speed[j] = clip(v_ff[j] + kp * wrap(target[j] - actual[j]), -vmax, +vmax)

The clip is on SPEED, not on error: a large error produces a long slow move,
never a fast one. The deadband suppresses the CORRECTION only. The feedforward
comes from the sender's velocities when it supplies them, otherwise from
differentiating the setpoint stream.

TWO TRAPS THIS CLASS EXISTS TO AVOID, both paid for on hardware.

1. Applying a position law twice. Senders that publish an INCREMENTAL setpoint
   (cur + v*dt) put the target ~1.7 mrad ahead of the measured position. A
   pure position law with a 17 mrad deadband sees an error an order of
   magnitude inside the band and commands exactly zero, forever: the arm never
   moves, the setpoint never advances because it tracks the measured position,
   and homing "converges" at its starting error. Take the setpoint's own RATE
   as the command; use the position error only as a correction.

2. time.time() for intervals. Under WSL the wall clock steps backwards when it
   resyncs with the host; the first run of the reference node logged a send
   latency of -2321 ms. Every interval here comes from an injected monotonic
   clock (default time.monotonic).
"""
from __future__ import annotations

import time
from typing import Callable, Iterable, Sequence

from .model import joint_travel


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class VelocityRelay:
    def __init__(self, kp: float, vmax_rad_s: float, deadband_rad: float,
                 watchdog_s: float, continuous_idx: Iterable[int] = (),
                 terminal_overshoot: bool = False, terminal_overshoot_rad: float = 0.0,
                 clock: Callable[[], float] = time.monotonic):
        if vmax_rad_s <= 0:
            raise ValueError("vmax must be positive")
        if deadband_rad < 0 or watchdog_s <= 0 or kp < 0:
            raise ValueError("deadband >= 0, watchdog > 0, kp >= 0")
        if terminal_overshoot and not terminal_overshoot_rad > 0:
            raise ValueError(
                "terminal_overshoot is on but terminal_overshoot_rad is 0: refusing to run "
                "'on' with no measured overshoot. Fit a model first (sim2real-gap fit).")
        self.kp = float(kp)
        self.vmax = float(vmax_rad_s)
        self.deadband = float(deadband_rad)
        self.watchdog_s = float(watchdog_s)
        self.continuous_idx = tuple(continuous_idx)
        self.terminal_overshoot = bool(terminal_overshoot)
        self.overshoot_rad = float(terminal_overshoot_rad)
        self.clock = clock
        self._target = None
        self._target_vel = None
        self._target_t = None
        self._prev_target = None
        self._prev_target_t = None
        self._step_times: list[float] = []
        self.send_latencies_ms: list[float] = []
        self.last_reason = "no target yet"
        self.last_err: list[float] | None = None

    # ------------------------------------------------------------- inputs
    def set_target(self, target: Sequence[float], velocities: Sequence[float] | None = None,
                   now: float | None = None) -> None:
        now = self.clock() if now is None else now
        if self._target is not None:
            self._prev_target, self._prev_target_t = self._target, self._target_t
        self._target = [float(x) for x in target]
        self._target_vel = None if velocities is None else [float(v) for v in velocities]
        self._target_t = now

    def record_send_latency(self, ms: float) -> None:
        self.send_latencies_ms.append(float(ms))
        if len(self.send_latencies_ms) > 1000:
            del self.send_latencies_ms[:-1000]

    # ------------------------------------------------------------- the law
    def _overshot(self, target: Sequence[float], actual: Sequence[float]) -> list[float]:
        if not self.terminal_overshoot:
            return list(target)
        d = joint_travel(actual, target, self.continuous_idx)
        e = self.overshoot_rad
        return [t + (e if x > 0 else -e) if abs(x) > e else t for t, x in zip(target, d)]

    def step(self, actual: Sequence[float], now: float | None = None,
             target: Sequence[float] | None = None,
             target_vel: Sequence[float] | None = None) -> list[float]:
        """Speeds to command this cycle. Pass `target` to set it in the same call."""
        now = self.clock() if now is None else now
        if target is not None:
            self.set_target(target, target_vel, now)
        self._step_times.append(now)
        if len(self._step_times) > 200:
            del self._step_times[:-200]
        n = len(actual)
        if self._target is None:
            self.last_reason = "no target yet"
            return [0.0] * n
        age = now - self._target_t
        if age > self.watchdog_s:
            self.last_reason = "watchdog (%.2f s since last target)" % age
            return [0.0] * n
        tgt = self._overshot(self._target, actual)
        err = joint_travel(actual, tgt, self.continuous_idx)
        self.last_err = err
        if self._target_vel is not None:
            v_ff = [_clip(v, -self.vmax, self.vmax) for v in self._target_vel]
        elif self._prev_target is not None and (self._target_t - self._prev_target_t) > 1e-6:
            dt = self._target_t - self._prev_target_t
            step = joint_travel(self._prev_target, self._target, self.continuous_idx)
            v_ff = [_clip(s / dt, -self.vmax, self.vmax) for s in step]
        else:
            v_ff = [0.0] * n
        speeds = []
        for e, ff in zip(err, v_ff):
            corr = self.kp * e if abs(e) > self.deadband else 0.0
            speeds.append(_clip(ff + corr, -self.vmax, self.vmax))
        self.last_reason = "tracking"
        return speeds

    # ------------------------------------------------------------- stats
    @property
    def achieved_rate_hz(self) -> float | None:
        """Loop rate over the last <=200 steps. Log THIS, never the configured rate."""
        t = self._step_times
        if len(t) < 2 or t[-1] <= t[0]:
            return None
        return (len(t) - 1) / (t[-1] - t[0])

    def latency_summary_ms(self) -> dict | None:
        L = self.send_latencies_ms
        if not L:
            return None
        return {"min": min(L), "avg": sum(L) / len(L), "max": max(L), "n": len(L)}
