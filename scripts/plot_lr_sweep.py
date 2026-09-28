"""Figure for REPORT.md section 1: the 1,000-update peak-learning-rate sweep.

Run from the repository root:  uv run python scripts/plot_lr_sweep.py
Reads runs/lr_sweep/lr_*/metrics.jsonl plus the first 1,000 updates of
runs/baseline_3e-4 (identical settings, alpha_max = 3e-4), and writes
report_assets/fig2_lr_sweep.png.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# Distinct hues so neighbouring learning rates stay separable. The chosen run and
# its closest rival get the two strongest colours (blue vs orange).
RUNS = [  # label, peak LR, colour
    ("1e-4", 1e-4, "#e87ba4"),
    ("3e-4", 3e-4, "#eda100"),
    ("1e-3", 1e-3, "#1baf7a"),
    ("3e-3", 3e-3, "#2a78d6"),
    ("1e-2", 1e-2, "#eb6834"),
]
CHOSEN = "3e-3"
MAX_STEP = 1000
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
COLOR = {label: color for label, _, color in RUNS}


def load(label):
    # the 3e-4 point is the baseline's own first 1,000 updates: same seeds and schedule
    run_dir = Path("runs/baseline_3e-4") if label == "3e-4" else Path("runs/lr_sweep") / f"lr_{label}"
    path = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [r for r in rows if r["step"] <= MAX_STEP]


def val_points(rows):
    return [(r["step"], r["val_loss"]) for r in rows if r.get("val_loss") is not None]


def style(ax, title, ylabel, xlabel="optimizer update"):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def lw(label):
    return 2.6 if label == CHOSEN else 1.6


def main():
    data = {label: load(label) for label, _, _ in RUNS}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    (ax_val, ax_zoom), (ax_grad, ax_final) = axes

    # (a) validation loss, all runs, linear scale
    end_offsets = {"1e-3": 11, "1e-2": 0, "3e-3": -11}  # three final values within 0.07
    for label, _, color in RUNS:
        xs, ys = zip(*val_points(data[label]))
        ax_val.plot(xs, ys, color=color, lw=lw(label), marker="o", ms=5, zorder=3 if label == CHOSEN else 2)
        ax_val.annotate(f"{label}  {ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(8, end_offsets.get(label, 0)),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=INK if label == CHOSEN else MUTED,
                        fontweight="bold" if label == CHOSEN else "normal")
    ax_val.set_xlim(150, 1230)
    ax_val.set_xticks(range(200, 1001, 200))
    style(ax_val, "(a) Validation loss, all five rates", "cross-entropy (nats/token)")

    # (b) zoom: gap to the chosen run for the three best rates
    ref = dict(val_points(data[CHOSEN]))
    ax_zoom.axhspan(-0.01, 0.01, color="#f0efec", zorder=0)
    ax_zoom.axhline(0, color=COLOR[CHOSEN], lw=lw(CHOSEN))
    ax_zoom.text(1030, -0.014, "3e-3 (reference)", va="top", fontsize=8.5, color=INK, fontweight="bold")
    ax_zoom.text(210, 0.013, "±0.01 treated as a tie", fontsize=8, color=MUTED)
    for label in ("1e-3", "1e-2"):
        xs, ys = zip(*[(s, v - ref[s]) for s, v in val_points(data[label])])
        ax_zoom.plot(xs, ys, color=COLOR[label], lw=lw(label), marker="o", ms=5)
        ax_zoom.annotate(f"{label}  {ys[-1]:+.3f}", (xs[-1], ys[-1]), xytext=(8, 5 if label == "1e-2" else 0),
                         textcoords="offset points", va="center", fontsize=8.5, color=MUTED)
    ax_zoom.set_xlim(150, 1230)
    ax_zoom.set_ylim(-0.05, 0.37)
    ax_zoom.set_xticks(range(200, 1001, 200))
    style(ax_zoom, "(b) Zoom: validation-loss gap to 3e-3", "Δ cross-entropy vs 3e-3 (nats/token)")

    # (c) pre-clip gradient norm, log scale with plain tick labels
    for label, _, color in RUNS:
        rows = data[label]
        ax_grad.plot([r["step"] for r in rows], [r["grad_norm"] for r in rows], color=color,
                     lw=lw(label) - 0.3, alpha=0.95, zorder=3 if label == CHOSEN else 2)
    ax_grad.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax_grad.text(1000, 1.07, "clip threshold M = 1.0", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    ax_grad.set_yscale("log")
    ax_grad.yaxis.set_major_locator(FixedLocator([0.07, 0.1, 0.2, 0.3, 0.5, 1, 2]))
    ax_grad.yaxis.set_minor_locator(NullLocator())
    ax_grad.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    style(ax_grad, "(c) Pre-clip global gradient norm (log scale)", "L2 norm")

    # (d) final validation loss vs learning rate
    lrs = [lr for _, lr, _ in RUNS]
    finals = [data[label][-1]["val_loss"] for label, _, _ in RUNS]
    ax_final.plot(lrs, finals, color=AXIS, lw=1.5, zorder=1)
    for (label, lr, color), final in zip(RUNS, finals):
        chosen = label == CHOSEN
        ax_final.scatter([lr], [final], s=110 if chosen else 60, color=color, zorder=2,
                         edgecolor="white", linewidth=2)
        ax_final.annotate(f"{final:.3f}" + ("\n(chosen)" if chosen else ""), (lr, final),
                          xytext=(0, -24 if chosen else 11), textcoords="offset points", ha="center",
                          fontsize=9, color=INK if chosen else MUTED,
                          fontweight="bold" if chosen else "normal")
    ax_final.set_xscale("log")
    ax_final.xaxis.set_minor_locator(NullLocator())
    ax_final.set_xticks(lrs, [label for label, _, _ in RUNS])
    ax_final.set_xlim(6e-5, 1.6e-2)
    ax_final.set_ylim(1.75, 2.95)
    style(ax_final, "(d) Validation loss at update 1,000 vs peak LR", "cross-entropy (nats/token)",
          xlabel="peak learning rate (log scale)")

    handles = [Line2D([], [], color=c, lw=lw(l), marker="o", ms=5, label=f"lr {l}" + ("  (chosen)" if l == CHOSEN else ""))
               for l, _, c in RUNS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Peak-learning-rate sweep: 1,000 updates per run, s_c = 9999, all other settings fixed",
                 x=0.01, y=0.995, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.5, w_pad=2.5)
    out = Path("report_assets/fig2_lr_sweep.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
