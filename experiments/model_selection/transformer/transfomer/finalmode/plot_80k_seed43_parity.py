#!/usr/bin/env python3
"""Draw paper-style parity figures for finalmode-80k seed43 test inference."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SKILL_SCRIPTS = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis

configure_paper_style()
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
INFERENCE = ROOT / "80k/seed43_independent_2k_inference"
OUTPUT = ROOT / "figure"


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    true = np.asarray([float(row["true"]) for row in rows])
    pred = np.asarray([float(row["pred"]) for row in rows])
    if np.any(true <= 0) or np.any(pred <= 0):
        raise ValueError(f"Log-scale parity requires positive predictions: {path}")
    return true, pred


def plot_parity(seed: int) -> None:
    stem = f"seed{seed}_seed43_independent"
    true, pred = load_predictions(INFERENCE / f"{stem}_predictions.csv")
    metric = json.loads((INFERENCE / f"{stem}.json").read_text(encoding="utf-8"))
    rel = np.abs(pred - true) / np.clip(np.abs(true), 1e-12, None) * 100
    mean_error = float(rel.mean())
    p95_error = float(np.percentile(rel, 95))
    upper = float(np.percentile(rel, 99.5))
    low = min(float(true.min()), float(pred.min())) * 0.86
    high = max(float(true.max()), float(pred.max())) * 1.14

    fig, ax = plt.subplots(figsize=(10.5, 10.5), facecolor="white")
    ax.scatter(true, pred, s=120, color=COLORS["parity_blue"], alpha=0.60,
               edgecolors="#FFFFFF", linewidths=0.7, zorder=2)
    ax.plot((low, high), (low, high), "--", color=COLORS["black"], lw=2.8, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
           xlabel=r"True $\chi N^*$", ylabel=r"Predicted $\chi N^*$")
    ax.set_box_aspect(1)
    ax.set_title("Prediction on log scale", fontsize=32, pad=18)
    ax.text(0.035, 0.955,
            "\n".join((rf"$R^2={metric['r2']:.4f}$", rf"$MAPE={metric['mape_percent']:.3f}\%$",
                       rf"$RMSE={metric['rmse']:.2f}$", rf"$MAE={metric['mae']:.3f}$")),
            transform=ax.transAxes, ha="left", va="top", fontsize=26)
    style_axis(ax)

    inset = ax.inset_axes([0.60, 0.115, 0.34, 0.285])
    inset.hist(rel, bins=np.linspace(0, upper, 28), density=True, color=COLORS["density_green"],
               alpha=0.82, edgecolor="none")
    inset.axvline(mean_error, color=COLORS["mean_orange"], ls="--", lw=2.4)
    inset.axvline(p95_error, color="0.35", ls=":", lw=2.4)
    inset.set(xlim=(0, upper * 1.04), xlabel=r"$|error|$ (%)", ylabel="Probability density")
    style_axis(inset)
    inset.tick_params(axis="both", which="major", labelsize=13, length=7, width=1.6)
    inset.tick_params(axis="both", which="minor", length=3.5, width=1.2)
    for spine in inset.spines.values():
        spine.set_linewidth(1.6)
    inset.text(0.96, 0.94, f"mean={mean_error:.2f}%\n95th={p95_error:.2f}%",
               transform=inset.transAxes, ha="right", va="top", fontsize=16)
    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.15, top=0.90)
    save_dual(fig, OUTPUT / f"parity_seed{seed}_seed43_independent", tight=False)
    plt.close(fig)


def main() -> None:
    for seed in range(202601, 202606):
        plot_parity(seed)


if __name__ == "__main__":
    main()
