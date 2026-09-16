import math

import pytest

from sim2real_gap import NotCalibrated, TerminalOffsetModel

CONT = (0, 2, 4, 6)
EPS = math.radians(0.3052)


def model():
    return TerminalOffsetModel({"left": EPS, "right": math.radians(0.3059)}, CONT)


def test_continuous_seam_travels_short_way_and_overshoots_positively():
    m = model()
    q0 = [math.radians(170.0)] + [0.0] * 6
    q1 = [math.radians(-170.0)] + [0.0] * 6
    t = m.joint_travel(q0, q1)
    assert abs(math.degrees(t[0]) - 20.0) < 1e-9
    cmd, applied, _ = m.correct("left", q0, q1)
    assert applied[0] and (cmd[0] - q1[0]) > 0 and abs(abs(cmd[0] - q1[0]) - EPS) < 1e-12


def test_seam_other_way():
    m = model()
    q0 = [math.radians(-170.0)] + [0.0] * 6
    q1 = [math.radians(170.0)] + [0.0] * 6
    assert abs(math.degrees(m.joint_travel(q0, q1)[0]) + 20.0) < 1e-9
    cmd, _, _ = m.correct("left", q0, q1)
    assert (cmd[0] - q1[0]) < 0


def test_non_continuous_joint_does_not_wrap():
    m = model()
    q0 = [0.0, math.radians(170.0)] + [0.0] * 5
    q1 = [0.0, math.radians(-170.0)] + [0.0] * 5
    assert abs(math.degrees(m.joint_travel(q0, q1)[1]) + 340.0) < 1e-9


def test_correct_then_predict_round_trip():
    m = model()
    q0 = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7]
    q1 = [0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3]
    cmd, applied, _ = m.correct("right", q0, q1)
    assert all(applied)
    back = m.predict("right", q0, cmd)
    assert max(abs(a - b) for a, b in zip(back, q1)) < 1e-12


def test_joint_moving_less_than_eps_is_untouched():
    m = model()
    q0 = [0.0] * 7
    q1 = [EPS * 0.5, EPS * 2, 0.0, -EPS * 0.5, -EPS * 2, 0.0, 1.0]
    cmd, applied, why = m.correct("left", q0, q1)
    assert applied == [False, True, False, False, True, False, True]
    assert cmd[0] == q1[0] and cmd[2] == q1[2] and cmd[3] == q1[3]
    assert "left alone" in why


def test_unknown_arm_refused():
    with pytest.raises(NotCalibrated):
        model().eps("middle")
    with pytest.raises(NotCalibrated):
        model().correct("middle", [0.0] * 7, [1.0] * 7)


def test_model_with_failed_controls_refused(tmp_path):
    m = TerminalOffsetModel({"left": EPS}, CONT, controls_passed=False)
    p = tmp_path / "m.json"
    m.save(str(p))
    with pytest.raises(NotCalibrated):
        TerminalOffsetModel.load(str(p))
    forced = TerminalOffsetModel.load(str(p), force=True)
    assert forced.eps("left") == EPS


def test_save_load_round_trip(tmp_path):
    p = tmp_path / "m.json"
    model().save(str(p))
    m2 = TerminalOffsetModel.load(str(p))
    assert m2.eps_rad == model().eps_rad and m2.continuous_idx == CONT


def test_missing_file_is_not_calibrated(tmp_path):
    with pytest.raises(NotCalibrated):
        TerminalOffsetModel.load(str(tmp_path / "nope.json"))
