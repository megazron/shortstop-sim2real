#!/usr/bin/env python3
"""Write a runs.json you can fit, from a hidden EPS you choose.

    python3 examples/make_synthetic_runs.py runs.json --eps-deg 0.3
"""
import argparse
import json

from sim2real_gap.cli import synth_runs

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--eps-deg", type=float, default=0.3052)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
d = synth_runs(a.reps, eps_deg={"left": a.eps_deg, "right": a.eps_deg + 0.0007})
json.dump(d, open(a.out, "w"), indent=1)
print("wrote", len(d["runs"]), "runs")
