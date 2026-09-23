#!/usr/bin/env python3
"""Redraw the corrected GNN, MLP, and Transformer data-size curves."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tapd-corrected-scaling")
SKILL_SCRIPTS = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis

configure_paper_style()
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

SPECS = {
    "gnn": {
        "summary": HERE / "results/gnn/summary/c2_block_stratified_nested_validation_summary.csv",
        "x": "train_size", "mean": "development_validation_mape_mean_percent",
        "sd": "development_validation_mape_sd_percent", "scale": 1000.0,
        "ylabel": "Validation MAPE (%)", "output": HERE / "figures/gnn_block_stratified_nested_validation_mape", "figsize": (5666/600, 4698/600), "inset": [0.43, 0.45, 0.53, 0.40],
    },
    "mlp": {
        "summary": HERE / "results/mlp/development_evaluation_seed42/nested_training_size_seed42_summary.csv",
        "x": "train_size_k", "mean": "evaluation_mape_mean_percent",
        "sd": "evaluation_mape_sd_percent", "scale": 1.0,
        "ylabel": "Held-out MAPE (%)", "output": HERE / "figures/mlp_block_stratified_nested_heldout_mape", "figsize": (5666/600, 4698/600), "inset": [0.43, 0.45, 0.53, 0.40],
    },
    "transformer": {
        "summary": HERE / "results/transformer_cls_correct/summary/validation_summary.csv",
        "x": "size_k", "mean": "validation_MAPE_mean", "sd": "validation_MAPE_sd", "scale": 1.0,
        "ylabel": "Development-validation MAPE (%)", "output": HERE / "figures/transformer_block_stratified_nested_validation_mape", "figsize": (6285/600, 5085/600), "inset": [0.60, 0.55, 0.36, 0.36],
    },
}


def load(spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Path(spec["summary"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.asarray([float(row[spec["x"]]) / spec["scale"] * 1000 for row in rows])
    mean = np.asarray([float(row[spec["mean"]]) for row in rows])
    sd = np.asarray([float(row[spec["sd"]]) for row in rows])
    order = np.argsort(x)
    x, mean, sd = x[order], mean[order], sd[order]
    expected = np.arange(8_000, 80_001, 8_000)
    # Transformer includes a preregistered 70k diagnostic; the publication curve uses the common 8k grid.
    keep = np.isin(x.astype(int), expected)
    x, mean, sd = x[keep], mean[keep], sd[keep]
    if len(x) != 10 or not np.array_equal(x.astype(int), expected):
        raise RuntimeError(f"Expected complete 8k--80k curve, got {x.tolist()}")
    return x, mean, sd


def draw_series(ax, x, mean, sd, *, inset: bool = False) -> None:
    ax.errorbar(
        x, mean, yerr=sd, color=COLORS["blue"], marker="o",
        markersize=5 if inset else 7, markerfacecolor=COLORS["blue"],
        markeredgecolor="#FFFFFF", markeredgewidth=0.7 if inset else 0.8,
        linewidth=2.0 if inset else 2.8, elinewidth=1.4 if inset else 2.0,
        capsize=3 if inset else 4, capthick=1.2 if inset else 1.8, zorder=2,
    )
    selected = int(np.flatnonzero(x == 80_000)[0])
    ax.scatter(
        [x[selected]], [mean[selected]], s=58 if inset else 115,
        color=COLORS["orange"], edgecolors=COLORS["black"],
        linewidths=0.9 if inset else 1.2, zorder=4,
    )


def plot(model: str, spec: dict) -> tuple[Path, Path]:
    x, mean, sd = load(spec)
    fig, ax = plt.subplots(figsize=spec["figsize"], facecolor="white")
    draw_series(ax, x, mean, sd)
    low, high = float(np.min(mean - sd)), float(np.max(mean + sd))
    span = max(high - low, 0.2)
    ax.set_xlim(4_000, 84_000)
    ax.set_ylim(max(0.0, low - 0.10 * span), high + 0.10 * span)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(value)}" for value in x], rotation=42, ha="right")
    ax.set_xlabel("Training samples")
    ax.set_ylabel(spec["ylabel"])
    style_axis(ax)

    start = int(np.flatnonzero(x == 24_000)[0])
    inset = ax.inset_axes(spec["inset"])
    draw_series(inset, x[start:], mean[start:], sd[start:], inset=True)
    inset_low = max(0.0, float(np.min(mean[start:] - sd[start:])))
    inset_high = float(np.max(mean[start:] + sd[start:]))
    inset_span = max(inset_high - inset_low, 0.05)
    inset.set_xlim(22_000, 82_000)
    inset.set_ylim(max(0.0, inset_low - 0.08 * inset_span), inset_high + 0.08 * inset_span)
    inset.set_xticks([24_000, 40_000, 56_000, 72_000, 80_000])
    style_axis(inset)
    inset.tick_params(axis="both", which="major", labelsize=13, length=7, width=1.6)
    inset.tick_params(axis="both", which="minor", length=3.5, width=1.2)
    for spine in inset.spines.values():
        spine.set_linewidth(1.6)

    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.19, top=0.97)
    output = Path(spec["output"])
    png, svg = save_dual(fig, output, tight=False)
    plt.close(fig)
    print(f"{model}: {png}\n{model}: {svg}")
    return png, svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*SPECS, "all"], default="all")
    args = parser.parse_args()
    models = SPECS if args.model == "all" else {args.model: SPECS[args.model]}
    for model, spec in models.items():
        plot(model, spec)


if __name__ == "__main__":
    main()
