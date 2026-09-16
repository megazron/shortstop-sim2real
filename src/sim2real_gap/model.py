"""The terminal-offset model: one EPS per arm, applied in joint space.

    q_command = q_target + EPS * sign(travel)      per joint

A joint parks EPS short of wherever it is sent, on the side it arrived from.
Overshooting by EPS in the direction of travel makes it park on the target.
The model makes no claim about direction in Cartesian space, so there is no
step length, no basis and no domain to refuse outside of. The only thing it
refuses is a joint whose travel is smaller than EPS, where "the direction it
was going" is a guess.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from typing import Iterable, Sequence

FORMAT = "sim2real-gap-kit/terminal-offset/1"


class NotCalibrated(Exception):
    """No usable model: missing file, failed controls, or an arm never fitted."""


def wrap_pi(a: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def joint_travel(q_from: Sequence[float], q_to: Sequence[float],
                 continuous_idx: Iterable[int] = ()) -> list[float]:
    """q_to - q_from, wrapped across +/-pi ONLY on the continuous joints.

    A continuous joint going +170 -> -170 deg travelled +20 deg the short way
    round; naive subtraction says -340 and puts an overshoot on the wrong side
    of the seam. A joint with hard stops must NOT wrap: for it -340 is real.
    """
    cont = set(continuous_idx)
    if len(q_from) != len(q_to):
        raise ValueError("q_from and q_to differ in length")
    out = []
    for i, (a, b) in enumerate(zip(q_from, q_to)):
        d = float(b) - float(a)
        out.append(wrap_pi(d) if i in cont else d)
    return out


class TerminalOffsetModel:
    """EPS per arm plus which joints are continuous."""

    def __init__(self, eps_rad: dict[str, float], continuous_idx: Iterable[int] = (),
                 controls_passed: bool = True, not_validated: Sequence[str] = (),
                 source: str | None = None, notes: str = ""):
        if not eps_rad:
            raise NotCalibrated("a model needs at least one arm")
        for arm, e in eps_rad.items():
            if not (e >= 0.0) or not math.isfinite(e):
                raise ValueError("EPS for %r must be a finite non-negative radian value, got %r" % (arm, e))
        self.eps_rad = {str(k): float(v) for k, v in eps_rad.items()}
        self.continuous_idx = tuple(int(i) for i in continuous_idx)
        self.controls_passed = bool(controls_passed)
        self.not_validated = list(not_validated)
        self.source = source
        self.notes = notes

    # ----------------------------------------------------------------- basics
    def arms(self) -> list[str]:
        return sorted(self.eps_rad)

    def eps(self, arm: str) -> float:
        """EPS for one arm. An unknown arm is refused, never defaulted to zero."""
        try:
            return self.eps_rad[arm]
        except KeyError:
            raise NotCalibrated("no terminal-offset model for arm %r (fitted arms: %s)"
                                % (arm, ", ".join(self.arms()))) from None

    def joint_travel(self, q_from, q_to):
        return joint_travel(q_from, q_to, self.continuous_idx)

    # ------------------------------------------------------------- correction
    def correct(self, arm: str, q_from: Sequence[float], q_to: Sequence[float],
                min_travel: float | None = None):
        """The joint target to COMMAND so the arm ARRIVES at `q_to`.

        Returns (q_command, applied, why). `applied[i]` is False for a joint
        whose travel is <= the floor (EPS by default): it is left alone.
        """
        e = self.eps(arm)
        floor = e if min_travel is None else float(min_travel)
        trav = self.joint_travel(q_from, q_to)
        cmd, applied = [], []
        for t, target in zip(trav, q_to):
            if abs(t) <= floor:
                cmd.append(float(target))
                applied.append(False)
            else:
                cmd.append(float(target) + (e if t > 0 else -e))
                applied.append(True)
        n = sum(applied)
        why = ("overshot %d of %d joints by %.4f deg in the direction of travel; "
               "%d moved <= %.4f deg and were left alone"
               % (n, len(cmd), math.degrees(e), len(cmd) - n, math.degrees(floor)))
        return cmd, applied, why

    def predict(self, arm: str, q_from: Sequence[float], q_cmd: Sequence[float],
                min_travel: float | None = None) -> list[float]:
        """Where the arm will actually STOP if commanded `q_cmd` (inverse of correct)."""
        e = self.eps(arm)
        floor = e if min_travel is None else float(min_travel)
        trav = self.joint_travel(q_from, q_cmd)
        out = []
        for t, c in zip(trav, q_cmd):
            if abs(t) <= floor:
                out.append(float(c))
            else:
                out.append(float(c) - (e if t > 0 else -e))
        return out

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "units": "rad",
            "eps_rad": dict(self.eps_rad),
            "eps_deg": {k: math.degrees(v) for k, v in self.eps_rad.items()},
            "continuous_idx": list(self.continuous_idx),
            "controls_passed": self.controls_passed,
            "not_validated": list(self.not_validated),
            "source": self.source,
            "fitted_on": _dt.date.today().isoformat(),
            "notes": self.notes,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def from_dict(cls, d: dict, force: bool = False) -> "TerminalOffsetModel":
        if d.get("format") != FORMAT:
            raise NotCalibrated("not a %s file (format=%r)" % (FORMAT, d.get("format")))
        if not d.get("controls_passed") and not force:
            raise NotCalibrated(
                "the model's controls did not pass. A correction whose controls "
                "failed is a systematic error nobody checked. Re-fit, or load with force=True.")
        return cls(d["eps_rad"], d.get("continuous_idx", ()),
                   controls_passed=bool(d.get("controls_passed")),
                   not_validated=d.get("not_validated", ()),
                   source=d.get("source"), notes=d.get("notes", ""))

    @classmethod
    def load(cls, path: str, force: bool = False) -> "TerminalOffsetModel":
        try:
            with open(path) as f:
                d = json.load(f)
        except FileNotFoundError:
            raise NotCalibrated("%s does not exist. Run `sim2real-gap fit runs.json -o %s`."
                                % (path, path)) from None
        return cls.from_dict(d, force=force)
