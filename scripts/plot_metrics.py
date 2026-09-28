"""Generic 4-panel training-curve plot for one or more run directories.

Used for Figure 1 (fixed-batch overfit check):
  uv run python scripts/plot_metrics.py runs/overfit --labels overfit \
      --output report_assets/fig1_overfit_sanity_check.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Jupyter/Kaggle export MPLBACKEND=module://matplotlib_inline..., which a bare venv
# can't import; matplotlib validates it at import time, so override it first.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
import matplotlib.pyplot as plt


def load_metrics(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot training curves from one or more run directories.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", default="report_assets/training_curves.png")
    parser.add_argument("--x-axis", choices=["step", "tokens_seen"], default="step")
    args = parser.parse_args(argv)
    labels = args.labels or [Path(d).name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        parser.error("--labels must match the number of run directories")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("train_loss", "training loss", True),
        ("val_loss", "validation loss", True),
        ("lr", "learning rate", False),
        ("grad_norm", "pre-clip gradient norm", True),
    ]
    for run_dir, label in zip(args.run_dirs, labels):
        records = load_metrics(run_dir)
        for ax, (key, title, log_scale) in zip(axes.flat, panels):
            points = [(r[args.x_axis], r[key]) for r in records if r.get(key) is not None]
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, label=label, marker="o" if key == "val_loss" else None, markersize=3)
            ax.set_title(title)
            ax.set_xlabel(args.x_axis.replace("_", " "))
            if log_scale:
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
