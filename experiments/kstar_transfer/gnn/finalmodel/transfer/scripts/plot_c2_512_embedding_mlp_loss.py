#!/usr/bin/env python3
"""Draw five-seed C2-512 fixed-embedding MLP probe convergence."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tapd_mplconfig")
import matplotlib
matplotlib.use("Agg")
import numpy as np

STYLE = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(STYLE))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis

configure_paper_style()
import matplotlib.pyplot as plt

ROOT = Path("/home/zxy/project/TAPD")
RUN_ROOT = ROOT / "baselinemodels/pub0818/gnn/finalmodel/transfer/c2_512_embedding_mlp_probe_20260903"
OUTPUT = ROOT / "baselinemodels/pub0818/gnn/finalmodel/figure/kstar_transfer/archive/c2_512_seed43/summary/c2_512_embedding_mlp_five_seed_loss"
SEEDS = (202601, 202602, 202603, 202604, 202605)

def load(seed: int):
    path = RUN_ROOT / f"run{SEEDS.index(seed)+1:02d}_seed{seed}" / "probe_loss_history.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.array([float(row["train_loss"]) for row in rows]), np.array([float(row["val_loss"]) for row in rows])

def stats(curves):
    n = min(map(len, curves))
    values = np.vstack([curve[:n] for curve in curves])
    log_values = np.log10(np.clip(values, 1e-12, None))
    mean = 10 ** log_values.mean(axis=0)
    sd = log_values.std(axis=0, ddof=1)
    return np.arange(1, n + 1), mean, 10 ** (np.log10(mean)-sd), 10 ** (np.log10(mean)+sd)

def main():
    curves = [load(seed) for seed in SEEDS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor="white")
    for ax, index, title in zip(axes, (0, 1), ("Train loss", "Validation loss")):
        epoch, mean, lower, upper = stats([item[index] for item in curves])
        ax.fill_between(epoch, lower, upper, color=COLORS["purple"], alpha=0.18, linewidth=0)
        ax.plot(epoch, mean, color=COLORS["purple"], lw=2.8, label="C2-512 embedding MLP")
        ax.set(xlim=(-2, 502), yscale="log", xlabel="Epoch", ylabel="MSE loss")
        ax.set_title(title, fontsize=32, pad=18)
        style_axis(ax)
    axes[1].legend(loc="upper right", fontsize=20, handlelength=2.4)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.88, wspace=0.30)
    save_dual(fig, OUTPUT)
    plt.close(fig)

if __name__ == "__main__":
    main()
