#!/usr/bin/env python3
"""Regenerate every figure in docs/img/ from the kit's own code and the rig numbers.

    python3 docs/make_figures.py

Needs matplotlib. The joint-error histogram and the relay traces come from
`sim2real_gap` itself (synthetic runs, the real VelocityRelay), so the figures
cannot drift from the code.
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from sim2real_gap.cli import synth_runs  # noqa: E402
from sim2real_gap.fit import _errors  # noqa: E402
from sim2real_gap.relay import VelocityRelay  # noqa: E402

OUT = os.path.join(HERE, "img")
os.makedirs(OUT, exist_ok=True)

INK = "#1f2933"
ACC = "#0b6e4f"       # primary accent (green)
ACC2 = "#c8553d"      # contrast accent (rust)
GREY = "#8a94a6"
LIGHT = "#e6ebf0"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": LIGHT, "grid.linewidth": 0.8, "legend.frameon": False,
    "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})


def save(fig, name, svg=False):
    fig.savefig(os.path.join(OUT, name + ".png"))
    if svg:
        fig.savefig(os.path.join(OUT, name + ".svg"))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. gain vs offset
def fig_gain_vs_offset():
    L = np.linspace(0, 150, 301)
    gain = 0.065 * L            # 93.5 % of travel -> 6.5 % short
    offset = np.full_like(L, 7.2)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(L, gain, color=ACC2, lw=2.2, label="gain model: arm travels 93.5 % of the command")
    ax.plot(L, offset, color=ACC, lw=2.2, label="offset model: arm stops 7.2 mm short")
    ax.scatter([100], [7.2], s=90, color=INK, zorder=5)
    ax.annotate("the measurement:\n36 runs, all 100 mm, 7.2 mm RMS\nboth models fit it exactly",
                xy=(100, 7.2), xytext=(104, 11.2), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.axvline(20, color=GREY, lw=1, ls="--")
    ax.plot([20, 20], [1.3, 7.2], color=INK, lw=1.2)
    ax.scatter([20, 20], [1.3, 7.2], s=40, color=[ACC2, ACC], zorder=5)
    ax.annotate("at 20 mm the two models\ndiffer by a factor of five:\n1.3 mm vs 7.0 mm",
                xy=(20, 4.2), xytext=(38, 8.3), fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.set_xlabel("commanded move length (mm)")
    ax.set_ylabel("predicted terminal error (mm)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 13)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Two models that agree at 100 mm and nowhere else")
    save(fig, "gain_vs_offset")


# ---------------------------------------------------------------- 2. joint errors
def fig_joint_errors():
    d = synth_runs(n_reps=6, noise_deg=0.02, seed=3)
    cont = {0, 2, 4, 6}
    errs = np.concatenate([_errors(r, cont) for r in d["runs"]])
    deg = np.degrees(errs)
    deg = deg[np.abs(deg) < 0.6]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(deg, bins=np.arange(-0.45, 0.46, 0.02), color=ACC, alpha=0.9, edgecolor="white")
    for x in (-0.305, 0.305):
        ax.axvline(x, color=ACC2, lw=1.4, ls="--")
    ax.text(0.305, ax.get_ylim()[1] * 0.92, "+EPS", color=ACC2, ha="left", fontsize=9)
    ax.text(-0.305, ax.get_ylim()[1] * 0.92, "-EPS", color=ACC2, ha="right", fontsize=9)
    ax.text(0.0, ax.get_ylim()[1] * 0.55,
            "on the rig, 211 of 252 recorded\nper-joint errors sat at ±0.25–0.30°\nwith the sign a coin flip:\n"
            "a bound, not a spread",
            ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=LIGHT))
    ax.set_xlabel("per-joint terminal error, achieved − target (deg)")
    ax.set_ylabel("count")
    ax.set_title("Per-joint terminal errors are bimodal (synthetic runs, hidden EPS 0.305°)")
    save(fig, "joint_errors")


# ---------------------------------------------------------------- 3. held out
def fig_held_out():
    labels = ["identity\n(no correction)", "joint model\nfitted", "joint model\nheld-out axis",
              "scrambled-sign\ncontrol", "Cartesian 3×3\nfitted", "Cartesian 3×3\nheld-out axis"]
    vals = [7.24, 1.83, 1.88, 7.19, 0.98, 93.5]
    cols = [GREY, ACC, ACC, GREY, ACC2, ACC2]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    bars = ax.bar(labels, vals, color=cols, width=0.62)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 200)
    ax.set_ylabel("end-effector RMS error (mm, log scale)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, "%.2f" % v if v < 10 else "%.1f" % v,
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(7.24, color=GREY, lw=1, ls=":")
    ax.text(5.42, 7.9, "identity", color=GREY, fontsize=8, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_title("One parameter survives the hold-out the nine-parameter fit fails")
    ax.text(0.5, -0.32, "held-out axis: fit on ±Y and ±Z runs, score on ±X, and so on round the three axes",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=GREY)
    save(fig, "held_out")


# ---------------------------------------------------------------- 4. pipeline (SVG)
def fig_pipeline():
    W, H = 1180, 420
    f = ["<?xml version='1.0' encoding='UTF-8'?>",
         f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' font-family='Helvetica, Arial, sans-serif'>",
         f"<rect width='{W}' height='{H}' fill='white'/>",
         "<defs><marker id='a' markerWidth='10' markerHeight='10' refX='9' refY='5' orient='auto'>"
         f"<path d='M0,0 L10,5 L0,10 z' fill='{INK}'/></marker></defs>"]

    def box(x, y, w, h, title, lines, fill="#f4f7f5", stroke=ACC, mono=False):
        f.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{stroke}' stroke-width='1.6'/>")
        f.append(f"<text x='{x + w / 2}' y='{y + 24}' text-anchor='middle' font-size='15' font-weight='bold' fill='{INK}'>{title}</text>")
        for i, ln in enumerate(lines):
            fam = "Menlo, Consolas, monospace" if mono else "Helvetica, Arial, sans-serif"
            f.append(f"<text x='{x + w / 2}' y='{y + 46 + i * 17}' text-anchor='middle' font-size='12' fill='{INK}' font-family='{fam}'>{ln}</text>")

    def arrow(x1, y1, x2, y2, label=None):
        f.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{INK}' stroke-width='1.6' marker-end='url(#a)'/>")
        if label:
            f.append(f"<text x='{(x1 + x2) / 2}' y='{min(y1, y2) - 10}' text-anchor='middle' font-size='10' fill='{GREY}'>{label}</text>")

    # top row: measurement -> fit -> model -> correct
    box(20, 40, 190, 120, "1. Record runs", ["command a move, wait,", "log q_target / q_achieved,", "optional axis label", "(both arms separately)"])
    arrow(210, 100, 258, 100, "runs.json")
    box(260, 30, 250, 140, "2. sim2real-gap fit", ["EPS = median |park error|", "controls, all must pass:", "identity on the table", "held out on an unseen AXIS", "scrambled signs must fail", "noise must degrade it"], mono=False)
    arrow(510, 100, 558, 100, "model.json")
    box(560, 40, 190, 120, "3. TerminalOffsetModel", ["eps per arm, controls_passed,", "not_validated list", "refuses unknown arms", "refuses failed controls"])
    arrow(750, 100, 798, 100)
    box(800, 40, 220, 120, "4. correct(q_from, q_to)", ["overshoot every joint by EPS", "in its direction of travel;", "wrap continuous joints;", "leave a joint that barely moved"], mono=False)

    # bottom row: sim -> homing gate -> relay -> arm
    box(20, 250, 190, 110, "Simulation / planner", ["joint setpoint stream", "(sim -> real seam)"], fill="#f7f7f7", stroke=GREY)
    arrow(210, 305, 258, 305, "q_sim, v_sim")
    box(260, 240, 220, 130, "HomingGate", ["|wrap(q_sim − q_real)| ≤ 0.05 rad", "on every joint, or REFUSE", "plan_home(): slow ramp with", "velocities filled in"], fill="#fdf3ef", stroke=ACC2)
    arrow(480, 305, 528, 305, "enabled")
    box(530, 230, 300, 150, "VelocityRelay", ["speed = clip(v_ff + kp·err, ±vmax)", "clip on SPEED, not on error", "deadband gates the correction only", "watchdog → zero speed", "monotonic clock; achieved rate logged"], mono=True)
    arrow(830, 305, 878, 305, "joint speeds")
    box(880, 250, 140, 110, "Real arm", ["high-level velocity", "command over the", "slow link"], fill="#f7f7f7", stroke=GREY)

    # vertical link: corrected target feeds the relay
    f.append(f"<path d='M910 160 L910 200 L680 200 L680 228' fill='none' stroke='{INK}' stroke-width='1.6' marker-end='url(#a)'/>")
    f.append(f"<text x='795' y='194' text-anchor='middle' font-size='11' fill='{GREY}'>corrected targets (or terminal_overshoot inside the relay)</text>")
    f.append(f"<text x='{W / 2}' y='{H - 14}' text-anchor='middle' font-size='12' fill='{GREY}'>Top: measure once, fit once, write a model only when its controls pass.  Bottom: the live path, gated before it can move metal.</text>")
    f.append("</svg>")
    with open(os.path.join(OUT, "pipeline.svg"), "w") as fh:
        fh.write("\n".join(f))
    print("wrote pipeline")


# ---------------------------------------------------------------- 5. relay step
def fig_relay_step():
    """Simulate the real VelocityRelay driving a first-order joint.

    Case A: an INCREMENTAL sender (setpoint = measured + v*dt) with a pure
    position law (no feedforward): the error stays inside the deadband and the
    joint never moves.  Case B: the same sender, relay takes the setpoint's own
    rate as feedforward: the joint arrives.
    """
    rate, dt = 20.0, 0.05
    vmax, kp, deadband = 0.05, 0.5, math.radians(1.0)
    goal, T = 0.20, 8.0
    n = int(T / dt)
    t = np.arange(n) * dt

    def run(feedforward: bool):
        q = 0.0
        clock = [0.0]
        relay = VelocityRelay(kp=kp, vmax_rad_s=vmax, deadband_rad=deadband, watchdog_s=0.5,
                              clock=lambda: clock[0])
        setp = 0.0
        qs, sps, errs = [], [], []
        for _ in range(n):
            # incremental sender: one cycle of travel ahead of the measured position
            v_cmd = vmax if setp < goal else 0.0
            setp = min(goal, q + v_cmd * dt)
            relay.set_target([setp], velocities=([v_cmd] if feedforward else None), now=clock[0])
            # a sender that gives no velocities: suppress the rate the relay would infer,
            # so this case is the pure position law the docstring warns about
            if not feedforward:
                relay._prev_target = None
            speed = relay.step([q], now=clock[0])[0]
            q += speed * dt                    # ideal velocity-tracking joint
            qs.append(q); sps.append(setp); errs.append(setp - q)
            clock[0] += dt
        return np.array(qs), np.array(sps), np.array(errs)

    qa, sa, ea = run(False)
    qb, sb, eb = run(True)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=False)
    for ax, (q, s, e), title, col in zip(
            axes, [(qa, sa, ea), (qb, sb, eb)],
            ["A. position law only: the stall", "B. feedforward + correction: arrives"], [ACC2, ACC]):
        ax.plot(t, np.degrees(s), color=GREY, lw=1.2, ls="--", label="setpoint (incremental sender)")
        ax.plot(t, np.degrees(q), color=col, lw=2.2, label="joint position")
        ax.axhline(math.degrees(goal), color=INK, lw=0.8, ls=":")
        ax.text(T - 0.1, math.degrees(goal) + 0.25, "goal", ha="right", fontsize=8.5)
        ax.set_xlabel("time (s)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 0.93))
    axes[0].set_ylabel("angle (deg)")
    axes[0].text(0.55, 0.62, "error = setpoint − measured\n= one cycle of travel ≈ 0.14°\ninside the 1.0° deadband\n→ commanded speed is 0, forever",
                 transform=axes[0].transAxes, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=LIGHT))
    # inset: error vs deadband, case A
    ins = axes[0].inset_axes([0.56, 0.14, 0.4, 0.26])
    ins.axhspan(-1.0, 1.0, color=LIGHT)
    ins.plot(t, np.degrees(ea), color=ACC2, lw=1.4)
    ins.set_ylim(-1.3, 1.3)
    ins.set_title("error vs deadband (deg)", fontsize=7.5)
    ins.tick_params(labelsize=6.5)
    ins.grid(False)
    fig.suptitle("The double position law, reproduced with the kit's own VelocityRelay (20 Hz, vmax 0.05 rad/s, kp 0.5, deadband 1.0°)",
                 fontsize=10, fontweight="bold")
    save(fig, "relay_step")


# ---------------------------------------------------------------- 6. deadband
def fig_deadband():
    park = 0.305
    step_mrad = 1000.0 * 0.05 / 18.0             # 2.78 mrad per cycle
    step_deg = math.degrees(step_mrad / 1000.0)  # 0.159 deg
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    rows = [("1.0° deadband (as shipped)", 1.0, ACC2), ("0.25° deadband (settled value)", 0.25, ACC)]
    ys = [1.7, 0.0]
    for i, (name, db, col) in enumerate(rows):
        y = ys[i]
        ax.barh(y, 2 * db, left=-db, height=0.5, color=col, alpha=0.18)
        ax.plot([-db, -db], [y - 0.25, y + 0.25], color=col, lw=2)
        ax.plot([db, db], [y - 0.25, y + 0.25], color=col, lw=2)
        ax.text(db + 0.03, y + 0.16, "P term off inside", fontsize=8, color=col)
        # the arm parks here
        ax.scatter([-park, park], [y, y], s=60, color=INK, zorder=5)
        ax.annotate("", xy=(park, y - 0.32), xytext=(-park, y - 0.32), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
        ax.text(0, y - 0.45, "measured park: ±0.305°", ha="center", fontsize=8.5)
        # one cycle of travel
        ax.plot([db - step_deg, db], [y + 0.32, y + 0.32], color=INK, lw=3)
        ax.text(db - step_deg / 2, y + 0.4, "one cycle at vmax\n%.2f mrad = %.2f°" % (step_mrad, step_deg), ha="center", fontsize=7.5)
    ax.set_xlim(-1.15, 1.35)
    ax.set_ylim(-0.7, 2.45)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("joint error, target − actual (deg)")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_title("The 1.0° band is 3.3× the error it leaves behind; 0.25° still swallows one cycle of noise at 18 Hz")
    save(fig, "deadband")


if __name__ == "__main__":
    fig_gain_vs_offset()
    fig_joint_errors()
    fig_held_out()
    fig_pipeline()
    fig_relay_step()
    fig_deadband()
