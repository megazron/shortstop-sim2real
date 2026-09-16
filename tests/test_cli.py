import json

from sim2real_gap.cli import main


def test_cli_end_to_end(tmp_path, capsys):
    runs = tmp_path / "runs.json"
    model = tmp_path / "model.json"
    assert main(["synth", "-o", str(runs)]) == 0
    assert main(["fit", str(runs), "-o", str(model)]) == 0
    d = json.load(open(model))
    assert d["controls_passed"] and set(d["eps_rad"]) == {"left", "right"}
    assert main(["correct", str(model), "--arm", "left", "--from", "0,0,0,0,0,0,0",
                 "--to", "0.5,-0.5,0.5,-0.5,0.5,-0.5,0.5"]) == 0
    out = capsys.readouterr().out
    assert "overshot 7 of 7" in out
    assert main(["deadband", "--deadband-deg", "1.0", "--park-deg", "0.305"]) == 0
    assert "CAUSE" in capsys.readouterr().out
    assert main(["selftest"]) == 0
