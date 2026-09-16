"""Refuse to relay sim onto real until they agree, then get there slowly.

A sim->real relay replays the simulation's joint angles onto the metal. Any
difference it does not catch at enable time is commanded as a JUMP. On the
reference rig the sim home and the real home once differed by 1.94 rad at the
wrist, and one session started with the left arm 55 deg out on joint 4. The
gate is what stood between those numbers and a moving arm.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from .model import joint_travel


class HomingGate:
    def __init__(self, tolerance_rad: float = 0.05, continuous_idx: Iterable[int] = ()):
        if tolerance_rad <= 0:
            raise ValueError("tolerance must be positive")
        self.tolerance_rad = float(tolerance_rad)
        self.continuous_idx = tuple(continuous_idx)

    def check(self, q_sim: Sequence[float], q_real: Sequence[float]):
        """(ok, worst_joint, worst_err_rad). ok is False if ANY joint exceeds tolerance."""
        d = joint_travel(q_real, q_sim, self.continuous_idx)
        worst = max(range(len(d)), key=lambda i: abs(d[i]))
        err = abs(d[worst])
        return err <= self.tolerance_rad, worst, err

    def explain(self, q_sim, q_real) -> str:
        ok, j, err = self.check(q_sim, q_real)
        if ok:
            return "sim and real agree: worst joint %d at %.3f deg (gate %.3f deg)" % (
                j + 1, math.degrees(err), math.degrees(self.tolerance_rad))
        return ("REFUSING to enable the relay: joint %d differs by %.2f deg (gate %.3f deg). "
                "Enabling now would command that difference as a jump. Home the real arm "
                "with plan_home() first." % (j + 1, math.degrees(err), math.degrees(self.tolerance_rad)))


def plan_home(q_now: Sequence[float], q_home: Sequence[float], vmax_rad_s: float,
              rate_hz: float, continuous_idx: Iterable[int] = ()) -> dict:
    """A synchronised linear ramp from q_now to q_home, velocities filled.

    The slowest joint sets the duration so every joint's speed stays <= vmax.
    Points carry `velocities` so a feedforward relay executes the law exactly
    once rather than re-deriving it from a setpoint stream.
    """
    if vmax_rad_s <= 0 or rate_hz <= 0:
        raise ValueError("vmax and rate must be positive")
    d = joint_travel(q_now, q_home, continuous_idx)
    dur = max(abs(x) for x in d) / vmax_rad_s
    n = max(1, int(math.ceil(dur * rate_hz)))
    dt = dur / n if n else 0.0
    vel = [x / dur if dur > 0 else 0.0 for x in d]
    pts = []
    for k in range(n + 1):
        s = k / n
        pts.append({
            "t": k * dt,
            "positions": [float(a) + s * x for a, x in zip(q_now, d)],
            "velocities": list(vel) if k < n else [0.0] * len(d),
        })
    return {"duration_s": dur, "dt": dt, "points": pts, "travel_rad": d}
