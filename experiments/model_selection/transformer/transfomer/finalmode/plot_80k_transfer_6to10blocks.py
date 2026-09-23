#!/usr/bin/env python3
"""Draw finalmode-80k 6--10-block OOD parity plots in TAPD paper style."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

SKILL_SCRIPTS = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis

configure_paper_style()
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INFERENCE = ROOT / "80k/transfer_6to10blocks_inference"
OUTPUT = ROOT / "figure/transfer_6to10blocks"
BLOCK_COLORS = {6: COLORS["blue"], 7: COLORS["green"], 8: COLORS["purple"], 9: COLORS["red"], 10: COLORS["amber"]}


def draw(seed: int) -> None:
    stem = f"seed{seed}_transfer_6to10blocks"
    with (INFERENCE / f"{stem}_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    true = np.asarray([float(row["y_true"]) for row in rows])
    pred = np.asarray([float(row["y_pred"]) for row in rows])
    blocks = np.asarray([int(row["block_count"]) for row in rows])
    result = json.loads((INFERENCE / f"{stem}.json").read_text(encoding="utf-8"))
    rel = np.abs(pred - true) / np.clip(np.abs(true), 1e-12, None) * 100
    lower, upper = min(float(true.min()), float(pred.min())) * 0.86, max(float(true.max()), float(pred.max())) * 1.14
    order = np.random.default_rng(seed).permutation(len(true))
    fig, ax = plt.subplots(figsize=(10.5, 10.5), facecolor="white")
    ax.scatter(true[order], pred[order], s=60, c=[BLOCK_COLORS[int(value)] for value in blocks[order]],
               alpha=0.60, edgecolors="#FFFFFF", linewidths=0.7, zorder=2)
    ax.plot((lower, upper), (lower, upper), "--", color=COLORS["black"], lw=2.8, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper),
           xlabel=r"True $\chi N^*$", ylabel=r"Predicted $\chi N^*$")
    ax.set_box_aspect(1)
    ax.text(0.035, 0.955, "\n".join((rf"$R^2={result['R2']:.4f}$", rf"$MAPE={result['MAPE']:.3f}\%$",
                                       rf"$RMSE={result['RMSE']:.2f}$", rf"$MAE={result['MAE']:.3f}$")),
            transform=ax.transAxes, ha="left", va="top", fontsize=26)
    style_axis(ax)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=10, markerfacecolor=BLOCK_COLORS[count],
                      markeredgecolor="#FFFFFF", markeredgewidth=0.7, alpha=0.8, label=f"{count} blocks")
               for count in sorted(BLOCK_COLORS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.985), ncol=5,
               fontsize=16, columnspacing=0.8, handletextpad=0.30, markerscale=0.85)
    inset = ax.inset_axes([0.60, 0.115, 0.34, 0.285])
    p995, mean_error, p95 = float(np.percentile(rel, 99.5)), float(rel.mean()), float(np.percentile(rel, 95))
    inset.hist(rel, bins=np.linspace(0, p995, 28), density=True, color=COLORS["density_green"], alpha=0.82, edgecolor="none")
    inset.axvline(mean_error, color=COLORS["mean_orange"], ls="--", lw=2.4)
    inset.axvline(p95, color="0.35", ls=":", lw=2.4)
    inset.set(xlim=(0, p995 * 1.04), xlabel=r"$|error|$ (%)", ylabel="Probability density")
    style_axis(inset)
    inset.tick_params(axis="both", which="major", labelsize=13, length=7, width=1.6)
    inset.tick_params(axis="both", which="minor", length=3.5, width=1.2)
    for spine in inset.spines.values(): spine.set_linewidth(1.6)
    inset.text(0.96, 0.94, f"mean={mean_error:.2f}%\n95th={p95:.2f}%", transform=inset.transAxes,
               ha="right", va="top", fontsize=16)
    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.15, top=0.90)
    save_dual(fig, OUTPUT / f"transfer_6to10blocks_parity_seed{seed}", tight=False)
    plt.close(fig)


def main() -> None:
    for seed in range(202601, 202606):
        draw(seed)


if __name__ == "__main__":
    main()
