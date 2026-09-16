"""sim2real-gap-kit: measure, model and close the joint-space sim-to-real gap.

The model is one number per arm: every joint parks EPS radians short of its
target, on the side it came from. The compensation is to overshoot each joint
by EPS in its direction of travel. See README.md for where that came from.
"""
from .model import NotCalibrated, TerminalOffsetModel, wrap_pi
from .fit import FitReport, fit_runs
from .deadband import audit as deadband_audit, mrad_per_cycle
from .relay import VelocityRelay
from .homing import HomingGate, plan_home

__all__ = [
    "NotCalibrated", "TerminalOffsetModel", "wrap_pi",
    "FitReport", "fit_runs",
    "deadband_audit", "mrad_per_cycle",
    "VelocityRelay",
    "HomingGate", "plan_home",
]
__version__ = "0.1.0"
