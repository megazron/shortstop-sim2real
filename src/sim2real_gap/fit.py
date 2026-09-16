"""Fit EPS from recorded runs, and refuse to write a model whose controls fail.

INPUT FORMAT -- a JSON list of runs:

    {"arm": "left",
     "q_target":   [7 floats, rad],        where the joints were told to go
     "q_achieved": [7 floats, rad],        where they actually stopped
     "q_start":    [7 floats, rad],        optional; gives the direction of travel
     "direction":  "+X",                    optional; axis label, enables axis hold-out
     "ee_target":  [x, y, z],  "ee_achieved": [x, y, z]   optional, metres}

WHAT IS FITTED. Per joint, e = wrap(q_achieved - q_target). If the arm parks
EPS short on the side it came from, |e| clusters at one magnitude and the
sign follows -sign(travel). EPS is the median |e| over joints that moved.

CONTROLS, all of which must pass before `to_model()` will hand you a model
with controls_passed=True:
  1. identity is on the table: the uncorrected error is reported beside the
     corrected one, and the held-out corrected score must beat it.
  2. held out, never fit: by AXIS when directions are labelled (+X and -X
     together -- the hardest hold-out available), else K-fold by run.
  3. scrambled signs: flipping the sign of each joint's correction at random
     must score no better than 0.9x identity. If destroying the sign structure
     does not destroy the model, the model was not using it.
  4. noise monotonicity: injecting noise at three levels must degrade the
     held-out score monotonically. A metric that cannot be made worse cannot
     be trusted when it is good.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .model import TerminalOffsetModel, wrap_pi

Jacobian = Callable[[str, Sequence[float]], np.ndarray]   # (arm, q) -> (3 x n)

CLUSTER_HALF_WIDTH_DEG = 0.05


def _errors(run: dict, cont: set[int]) -> np.ndarray:
    t, a = np.asarray(run["q_target"], float), np.asarray(run["q_achieved"], float)
    e = a - t
    for i in cont:
        e[i] = wrap_pi(e[i])
    return e


def _travel(run: dict, cont: set[int]) -> np.ndarray | None:
    if "q_start" not in run:
        return None
    s, t = np.asarray(run["q_start"], float), np.asarray(run["q_target"], float)
    d = t - s
    for i in cont:
        d[i] = wrap_pi(d[i])
    return d


def _signs(run: dict, cont: set[int]) -> np.ndarray:
    """Direction of travel per joint: from q_start when known, else inferred from the error."""
    d = _travel(run, cont)
    e = _errors(run, cont)
    if d is None:
        return -np.sign(e)
    s = np.sign(d)
    s[s == 0] = -np.sign(e[s == 0])
    return s


def _moved(run: dict, cont: set[int], floor: float) -> np.ndarray:
    d = _travel(run, cont)
    if d is None:
        return np.ones(len(run["q_target"]), bool)
    return np.abs(d) > floor


def _eps_from(runs: list[dict], cont: set[int]) -> float:
    E = np.concatenate([np.abs(_errors(r, cont)) for r in runs])
    eps0 = float(np.median(E))
    keep = np.concatenate([np.abs(_errors(r, cont))[_moved(r, cont, eps0)] for r in runs])
    return float(np.median(keep)) if keep.size else eps0


def _residuals(runs: list[dict], cont: set[int], eps: float, sign_flip: np.ndarray | None = None,
               noise: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Per-joint residual after correction: e + s*eps on joints that moved, e otherwise."""
    out = []
    for r in runs:
        e = _errors(r, cont)
        if noise > 0 and rng is not None:
            e = e + rng.normal(0.0, noise, e.shape)
        s = _signs(r, cont)
        if sign_flip is not None:
            s = s * sign_flip
        m = _moved(r, cont, eps)
        res = e + np.where(m, s * eps, 0.0)
        out.append(res)
    return np.concatenate(out)


def _rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(v)))) if v.size else float("nan")


def _cart_rms_mm(runs, cont, eps, jac: Jacobian | None, arm: str, sign_flip=None) -> float | None:
    if jac is None:
        return None
    acc = []
    for r in runs:
        e = _errors(r, cont)
        s = _signs(r, cont)
        if sign_flip is not None:
            s = s * sign_flip
        m = _moved(r, cont, eps)
        res = e + np.where(m, s * eps, 0.0)
        J = np.asarray(jac(arm, r["q_target"]), float)[:3]
        acc.append(J @ res)
    return 1000.0 * _rms(np.concatenate(acc))


def _groups(runs: list[dict]) -> list[list[int]]:
    """Axis groups when directions are labelled, else K-fold by run."""
    if all("direction" in r for r in runs):
        by = {}
        for i, r in enumerate(runs):
            by.setdefault(str(r["direction"]).strip("+-").upper(), []).append(i)
        if len(by) >= 2:
            return list(by.values())
    k = min(5, len(runs))
    return [list(range(i, len(runs), k)) for i in range(k)] if k > 1 else [[]]


@dataclass
class ArmFit:
    arm: str
    n_runs: int
    eps_rad: float
    identity_rms_deg: float
    corrected_rms_deg: float
    held_out_rms_deg: float
    scrambled_rms_deg: float
    noise_curve_deg: list[float]
    cluster_fraction: float
    sign_positive_fraction: float
    sign_agreement: float | None
    held_out_scheme: str
    identity_cart_mm: float | None = None
    held_out_cart_mm: float | None = None
    controls: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(v[0] for v in self.controls.values())


@dataclass
class FitReport:
    arms: dict[str, ArmFit]
    continuous_idx: tuple
    source: str | None = None

    @property
    def controls_passed(self) -> bool:
        return all(a.passed for a in self.arms.values())

    def arms_agree(self, tol_deg: float = 0.01):
        e = [math.degrees(a.eps_rad) for a in self.arms.values()]
        return (max(e) - min(e)) <= tol_deg if len(e) > 1 else None

    def to_model(self, force: bool = False) -> TerminalOffsetModel:
        if not self.controls_passed and not force:
            failed = [f"{arm}:{k}" for arm, a in self.arms.items() for k, v in a.controls.items() if not v[0]]
            raise ValueError("controls failed (%s); no model written. Use force=True to override, loudly."
                             % ", ".join(failed))
        nv = ["how EPS varies with speed (fit at whatever speed the runs used)",
              "poses far from the recorded ones (the model claims no pose dependence; verify)"]
        return TerminalOffsetModel({k: v.eps_rad for k, v in self.arms.items()}, self.continuous_idx,
                                   controls_passed=self.controls_passed, not_validated=nv,
                                   source=self.source,
                                   notes="fitted by sim2real-gap-kit; see FitReport.summary()")

    def summary(self) -> str:
        L = []
        for arm, a in self.arms.items():
            L.append("arm %-6s  %d runs  EPS = %.4f deg  (%s hold-out)" % (arm, a.n_runs, math.degrees(a.eps_rad), a.held_out_scheme))
            L.append("  |e| cluster within +/-%.2f deg of median: %.0f%%   sign +: %.0f%%%s"
                     % (CLUSTER_HALF_WIDTH_DEG, 100 * a.cluster_fraction, 100 * a.sign_positive_fraction,
                        ("   sign follows -travel: %.0f%%" % (100 * a.sign_agreement)) if a.sign_agreement is not None else ""))
            L.append("  joint RMS  identity %.4f deg  corrected %.4f  HELD OUT %.4f  scrambled %.4f"
                     % (a.identity_rms_deg, a.corrected_rms_deg, a.held_out_rms_deg, a.scrambled_rms_deg))
            if a.identity_cart_mm is not None:
                L.append("  EE RMS     identity %.2f mm  held out %.2f mm" % (a.identity_cart_mm, a.held_out_cart_mm))
            L.append("  noise curve (held-out deg at 0/0.5/1/2 x EPS noise): %s" % ", ".join("%.4f" % x for x in a.noise_curve_deg))
            for k, (ok, why) in a.controls.items():
                L.append("  [%s] %-22s %s" % ("PASS" if ok else "FAIL", k, why))
        ag = self.arms_agree()
        if ag is not None:
            L.append("arms agree within 0.01 deg: %s" % ("yes" if ag else "NO -- fit and apply per arm"))
        L.append("controls %s" % ("PASSED -- model may be written" if self.controls_passed else "FAILED -- no model"))
        return "\n".join(L)


def fit_runs(runs: list[dict], continuous_idx: Iterable[int] = (), jacobian: Jacobian | None = None,
             seed: int = 0, source: str | None = None) -> FitReport:
    cont = set(int(i) for i in continuous_idx)
    rng = np.random.default_rng(seed)
    by_arm: dict[str, list[dict]] = {}
    for r in runs:
        for k in ("arm", "q_target", "q_achieved"):
            if k not in r:
                raise ValueError("every run needs %r" % k)
        if len(r["q_target"]) != len(r["q_achieved"]):
            raise ValueError("q_target and q_achieved differ in length")
        by_arm.setdefault(str(r["arm"]), []).append(r)
    arms = {}
    for arm, R in by_arm.items():
        eps = _eps_from(R, cont)
        nj = len(R[0]["q_target"])
        allE = np.concatenate([_errors(r, cont) for r in R])
        absE = np.abs(allE)
        med = float(np.median(absE))
        cluster = float(np.mean(np.abs(absE - med) <= math.radians(CLUSTER_HALF_WIDTH_DEG)))
        pos = float(np.mean(allE > 0))
        agree = None
        if all("q_start" in r for r in R):
            s_true = np.concatenate([np.sign(_travel(r, cont)) for r in R])
            m = s_true != 0
            agree = float(np.mean(np.sign(allE[m]) == -s_true[m])) if m.any() else None
        identity = _residuals(R, cont, 0.0)
        corrected = _residuals(R, cont, eps)
        # held out
        groups = _groups(R)
        scheme = "by axis" if all("direction" in r for r in R) and len(groups) >= 2 and len(groups) < len(R) else "k-fold by run"
        ho, sc, ho_cart, id_cart = [], [], [], []
        for g in groups:
            test = [R[i] for i in g]
            train = [R[i] for i in range(len(R)) if i not in g]
            if not train or not test:
                continue
            e_tr = _eps_from(train, cont)
            ho.append(_residuals(test, cont, e_tr))
            flips = rng.choice([-1.0, 1.0], size=nj)
            while nj > 1 and np.all(flips == 1.0):
                flips = rng.choice([-1.0, 1.0], size=nj)
            sc.append(_residuals(test, cont, e_tr, sign_flip=flips))
            if jacobian is not None:
                ho_cart.append(_cart_rms_mm(test, cont, e_tr, jacobian, arm))
                id_cart.append(_cart_rms_mm(test, cont, 0.0, jacobian, arm))
        ho_rms = _rms(np.concatenate(ho)) if ho else _rms(corrected)
        sc_rms = _rms(np.concatenate(sc)) if sc else float("nan")
        # scrambled null: median over several random flips
        sc_trials = []
        for _ in range(9):
            flips = rng.choice([-1.0, 1.0], size=nj)
            sc_trials.append(_rms(_residuals(R, cont, eps, sign_flip=flips)))
        sc_rms = float(np.median(sc_trials + [sc_rms])) if not math.isnan(sc_rms) else float(np.median(sc_trials))
        # noise injection
        curve = [ho_rms]
        for lvl in (0.5, 1.0, 2.0):
            acc = []
            for g in groups:
                test = [R[i] for i in g]
                train = [R[i] for i in range(len(R)) if i not in g]
                if not train or not test:
                    continue
                acc.append(_residuals(test, cont, _eps_from(train, cont), noise=lvl * eps, rng=rng))
            curve.append(_rms(np.concatenate(acc)) if acc else float("nan"))
        id_rms, ho_deg, sc_deg = _rms(identity), ho_rms, sc_rms
        controls = {
            "beats identity": (ho_deg < 0.7 * id_rms,
                               "held out %.4f vs identity %.4f deg (needs < 0.7x)" % (math.degrees(ho_deg), math.degrees(id_rms))),
            "held out never fit": (bool(ho), "%s, %d folds" % (scheme, len([g for g in groups if g]))),
            "scrambled signs fail": (sc_deg >= 0.9 * id_rms,
                                     "scrambled %.4f vs identity %.4f deg (must be >= 0.9x)" % (math.degrees(sc_deg), math.degrees(id_rms))),
            "noise degrades": (all(curve[i] <= curve[i + 1] + 1e-12 for i in range(len(curve) - 1)),
                               "held-out deg: " + " -> ".join("%.4f" % math.degrees(c) for c in curve)),
            "enough runs": (len(R) >= 3, "%d runs (need >= 3)" % len(R)),
        }
        arms[arm] = ArmFit(
            arm=arm, n_runs=len(R), eps_rad=eps,
            identity_rms_deg=math.degrees(id_rms), corrected_rms_deg=math.degrees(_rms(corrected)),
            held_out_rms_deg=math.degrees(ho_deg), scrambled_rms_deg=math.degrees(sc_deg),
            noise_curve_deg=[math.degrees(c) for c in curve],
            cluster_fraction=cluster, sign_positive_fraction=pos, sign_agreement=agree,
            held_out_scheme=scheme,
            identity_cart_mm=(float(np.mean(id_cart)) if id_cart else None),
            held_out_cart_mm=(float(np.mean(ho_cart)) if ho_cart else None),
            controls=controls)
    return FitReport(arms=arms, continuous_idx=tuple(sorted(cont)), source=source)


def load_runs(path: str) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict) and "runs" in d:
        d = d["runs"]
    if not isinstance(d, list):
        raise ValueError("runs file must be a JSON list of runs (or {'runs': [...]})")
    return d
