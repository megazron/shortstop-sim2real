import math

import numpy as np
import pytest

from sim2real_gap import fit_runs
from sim2real_gap.cli import synth_runs
from sim2real_gap.fit import _residuals, _eps_from

CONT = (0, 2, 4, 6)


def test_fit_recovers_hidden_eps_within_5pct():
    d = synth_runs(3)
    rep = fit_runs(d["runs"], CONT)
    for arm, af in rep.arms.items():
        hid = d["_synth"]["hidden_eps_deg"][arm]
        assert abs(math.degrees(af.eps_rad) - hid) / hid < 0.05
    assert rep.controls_passed
    assert rep.arms_agree() is True


def test_axis_hold_out_used_when_directions_labelled():
    rep = fit_runs(synth_runs(2)["runs"], CONT)
    assert all(a.held_out_scheme == "by axis" for a in rep.arms.values())


def test_scrambled_sign_control_destroys_model():
    runs = [r for r in synth_runs(3)["runs"] if r["arm"] == "left"]
    eps = _eps_from(runs, set(CONT))
    ident = np.sqrt(np.mean(_residuals(runs, set(CONT), 0.0) ** 2))
    corr = np.sqrt(np.mean(_residuals(runs, set(CONT), eps) ** 2))
    flips = np.array([-1.0] * 7)
    scr = np.sqrt(np.mean(_residuals(runs, set(CONT), eps, sign_flip=flips) ** 2))
    assert corr < 0.3 * ident
    assert scr > 1.5 * ident       # fully wrong sign doubles the error


def test_noise_monotonicity_control():
    rep = fit_runs(synth_runs(3)["runs"], CONT, seed=3)
    for af in rep.arms.values():
        c = af.noise_curve_deg
        assert c == sorted(c)
        assert af.controls["noise degrades"][0]


def test_runs_without_signal_fail_controls_and_refuse_model():
    rng = np.random.default_rng(0)
    runs = []
    for i in range(12):
        t = rng.uniform(-1, 1, 7)
        runs.append({"arm": "left", "direction": "+XYZ"[i % 3] if False else ["+X", "-X", "+Y", "-Y", "+Z", "-Z"][i % 6],
                     "q_start": (t - 0.3).tolist(), "q_target": t.tolist(),
                     "q_achieved": (t + rng.normal(0, 0.01, 7)).tolist()})
    rep = fit_runs(runs, CONT)
    assert not rep.controls_passed
    with pytest.raises(ValueError):
        rep.to_model()
    m = rep.to_model(force=True)
    assert m.controls_passed is False


def test_kfold_when_no_directions():
    runs = [dict(r) for r in synth_runs(2)["runs"]]
    for r in runs:
        r.pop("direction")
    rep = fit_runs(runs, CONT)
    assert all(a.held_out_scheme == "k-fold by run" for a in rep.arms.values())
    assert rep.controls_passed


def test_missing_key_rejected():
    with pytest.raises(ValueError):
        fit_runs([{"arm": "left", "q_target": [0] * 7}], CONT)
