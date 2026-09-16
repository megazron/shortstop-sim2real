"""Is the controller's deadband the CAUSE of the terminal error?

A proportional term that is switched off inside a deadband parks the joint
wherever it enters the band, on the side it came from. If the measured park
error is well inside the deadband, the deadband is not hiding the error, it
park -- 3.3x -- and the fix was upstream of any compensation.
"""
from __future__ import annotations

import math


def mrad_per_cycle(vmax_rad_s: float, rate_hz: float) -> float:
    """How far a joint travels per control cycle at vmax, in milliradians."""
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    return 1000.0 * vmax_rad_s / rate_hz


def audit(deadband_deg: float, measured_park_deg: float,
          rate_hz: float | None = None, vmax_rad_s: float | None = None) -> dict:
    """Verdict on a deadband against a measured terminal (park) error.

    Returns {"verdict": "CAUSE"|"TIGHT"|"OK"|"UNRELATED", "ratio", "recommend_deg", "text"}.
    """
    if deadband_deg <= 0 or measured_park_deg < 0:
        raise ValueError("deadband must be > 0 and park error >= 0")
    ratio = deadband_deg / measured_park_deg if measured_park_deg > 0 else float("inf")
    rec = round(measured_park_deg / 3.0, 3) if measured_park_deg > 0 else deadband_deg
    rec = max(rec, 0.05)
    lines = []
    if measured_park_deg == 0:
        verdict = "OK"
        lines.append("no measured park error; nothing to attribute to the deadband.")
    elif ratio >= 1.5:
        verdict = "CAUSE"
        lines.append("the deadband (%.3f deg) is %.1fx the measured park error (%.3f deg). "
                     "The proportional term is OFF over the whole range in which the joint "
                     "stops short, so the deadband is producing that error, not hiding it."
                     % (deadband_deg, ratio, measured_park_deg))
        lines.append("Narrow it to <= %.3f deg (park/3) and re-measure before adding any "
                     "downstream compensation. Compensating for a controller that stops "
                     "before it arrives is a workaround; closing the deadband is the fix." % rec)
    elif ratio >= 0.8:
        verdict = "TIGHT"
        lines.append("the deadband (%.3f deg) is about the size of the park error (%.3f deg): "
                     "it is contributing. Try <= %.3f deg." % (deadband_deg, measured_park_deg, rec))
    else:
        verdict = "UNRELATED"
        lines.append("the deadband (%.3f deg) is smaller than the park error (%.3f deg); the error "
                     "comes from somewhere else (gravity sag, link latency, missing integral term)."
                     % (deadband_deg, measured_park_deg))
    lines.append("Do not go to zero: on a networked 12-20 Hz loop a deadband under ~0.1 deg "
                 "lets the P term chase encoder noise and the joints buzz. 0.25 deg was the "
                 "settled value on the reference rig.")
    if rate_hz and vmax_rad_s:
        step = mrad_per_cycle(vmax_rad_s, rate_hz)
        lines.append("at vmax %.3f rad/s and %.1f Hz a joint moves %.2f mrad per cycle against a "
                     "%.2f mrad deadband." % (vmax_rad_s, rate_hz, step, math.radians(deadband_deg) * 1000))
    return {"verdict": verdict, "ratio": ratio, "recommend_deg": rec, "text": " ".join(lines)}
