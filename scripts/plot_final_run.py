"""Figure for REPORT.md section 2: the final 10,000-update run vs the baseline.

Run from the repository root:  uv run python scripts/plot_final_run.py
Reads runs/baseline_3e-4 and runs/final_3e-3, writes
report_assets/fig3_final_run_vs_baseline.png.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

RUNS = [  # directory, label, colour, standardized eval (seed 42, 100 batches)
    ("runs/baseline_3e-4", "baseline  α_max = 3e-4", "#eda100", "runs/baseline_3e-4/eval.json"),
    ("runs/final_3e-3", "final  α_max = 3e-3", "#2a78d6", "report_assets/final_model_eval.json"),
]
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def load(run_dir):
    return [json.loads(line) for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines() if line]


def smooth(values, k=10):
    v = np.asarray(values, dtype=float)
    return np.convolve(v, np.ones(k) / k, mode="valid")


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)
    ax.set_xlabel("optimizer update", color=MUTED, fontsize=9.5)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xticks(range(0, 10001, 2000))


def plain_log_ticks(ax, ticks):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def main():
    data = {d: load(d) for d, *_ in RUNS}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    (ax_loss, ax_gap), (ax_lr, ax_grad) = axes

    # (a) loss curves: smoothed training loss (thin) + 20-batch validation (markers)
    for d, label, color, eval_path in RUNS:
        rows = data[d]
        steps = [r["step"] for r in rows]
        ax_loss.plot(steps[9:], smooth([r["train_loss"] for r in rows]), color=color, lw=1, alpha=0.55)
        val = [(r["step"], r["val_loss"]) for r in rows if r.get("val_loss") is not None]
        xs, ys = zip(*val)
        ax_loss.plot(xs, ys, color=color, lw=2, marker="o", ms=3)
        final = json.loads(Path(eval_path).read_text())["mean_cross_entropy"]
        ax_loss.annotate(f"std. eval {final:.3f}", (10000, ys[-1]), xytext=(8, 0), textcoords="offset points",
                         va="center", fontsize=9, color=INK, fontweight="bold")
    ax_loss.set_ylim(1.35, 3.0)
    ax_loss.set_xlim(0, 12300)
    style(ax_loss, "(a) Validation (markers) and training loss (thin)", "cross-entropy (nats/token)")

    # (b) validation gap baseline - final
    base = {r["step"]: r["val_loss"] for r in data["runs/baseline_3e-4"] if r.get("val_loss") is not None}
    fin = {r["step"]: r["val_loss"] for r in data["runs/final_3e-3"] if r.get("val_loss") is not None}
    steps = sorted(set(base) & set(fin))
    gap = [base[s] - fin[s] for s in steps]
    ax_gap.plot(steps, gap, color=INK, lw=1.8, marker="o", ms=3)
    ax_gap.axhline(0, color=AXIS, lw=1)
    reach = min(s for s, v in fin.items() if v <= base[10000])
    for s in (1000, 5000, 10000):
        ax_gap.annotate(f"update {s:,}: {base[s] - fin[s]:.3f}", (s, base[s] - fin[s]),
                        xytext=(8, 10) if s < 10000 else (0, 12), textcoords="offset points",
                        ha="left" if s < 10000 else "right", fontsize=9, color=MUTED)
    ax_gap.set_ylim(-0.05, 1.05)
    style(ax_gap, "(b) Validation-loss advantage of 3e-3 (baseline − final)", "Δ cross-entropy (nats/token)")
    ax_gap.text(9900, 0.75, f"3e-3 matches the baseline's final\nvalidation loss at update {reach:,}\n"
                f"({10000 / reach:.1f}× fewer updates)", ha="right", fontsize=9.5, color=INK)

    # (c) learning-rate schedule
    for d, label, color, _ in RUNS:
        rows = data[d]
        ax_lr.plot([r["step"] for r in rows], [r["lr"] for r in rows], color=color, lw=2)
    plain_log_ticks(ax_lr, [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    ax_lr.axvline(200, color=AXIS, lw=1, ls="--")
    ax_lr.text(260, 1.2e-5, "end of warmup (200)", fontsize=8.5, color=MUTED)
    style(ax_lr, "(c) Learning-rate schedule (log scale): warmup + cosine to 3e-5", "learning rate")

    # (d) gradient norm, 100-update mean
    for d, label, color, _ in RUNS:
        rows = data[d]
        steps_all = [r["step"] for r in rows]
        ax_grad.plot(steps_all[9:], smooth([r["grad_norm"] for r in rows]), color=color, lw=1.8)
    ax_grad.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax_grad.text(10000, 1.06, "clip threshold M = 1.0", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    plain_log_ticks(ax_grad, [0.1, 0.15, 0.2, 0.3, 0.5, 1, 2])
    style(ax_grad, "(d) Pre-clip gradient norm (log scale, 100-update mean)", "L2 norm")

    handles = [Line2D([], [], color=c, lw=2.2, marker="o", ms=4, label=l) for _, l, c, _ in RUNS]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Full 10,000-update runs (655M sampled tokens each): only the peak learning rate differs",
                 x=0.01, y=0.995, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.5, w_pad=2.5)
    fig.savefig("report_assets/fig3_final_run_vs_baseline.png", dpi=150, facecolor="white")
    print("saved report_assets/fig3_final_run_vs_baseline.png")


if __name__ == "__main__":
    main()
