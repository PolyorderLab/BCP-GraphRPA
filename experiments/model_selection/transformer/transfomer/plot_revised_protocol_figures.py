#!/usr/bin/env python3
"""Publication figures for the frozen 80k/20k Transformer protocol.

OFAT and interaction panels use only validation MAPE.  Final ablation panels
and parity use the post-freeze 2k test and are never used for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tapd-transformer-revised")

import numpy as np

SKILL_SCRIPTS = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis

configure_paper_style()
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

LABELS = {
    "batch_size": "Batch size", "lr": "Learning rate", "dropout": "Dropout",
    "weight_decay": "Weight decay", "model_size": r"$d_{model}$/heads",
    "num_layers": "Encoder layers", "ff_mult": "FF multiplier",
    "readout": "Readout", "position_encoding": "Position encoding",
}
ORDERS = {
    "batch_size": [16, 32, 40, 64, 128],
    "lr": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "dropout": [0.0, 0.05, 0.1, 0.2],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "model_size": [[64, 2], [128, 4], [192, 6], [256, 8]],
    "num_layers": [1, 2, 3, 4], "ff_mult": [2, 4, 6],
    "readout": ["cls", "mean", "cls_mean", "attention"],
    "position_encoding": ["none", "learnable", "sinusoidal"],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value_label(family: str, value) -> str:
    if family == "model_size":
        return f"{value[0]}/{value[1]}"
    if family in {"lr", "weight_decay"}:
        return "0" if float(value) == 0 else f"{float(value):.0e}"
    if family == "position_encoding":
        return {"none": "None", "learnable": "Learned", "sinusoidal": "Sinusoidal"}[value]
    if family == "readout" and value == "cls_mean":
        return "CLS+mean"
    return str(value).capitalize()


def order_value(family: str, value) -> int:
    values = ORDERS[family]
    return values.index(value)


def ofat_family_rows(rows: list[dict[str, str]], family: str) -> list[dict]:
    selected = []
    for row in rows:
        cfg = json.loads(row["config_json"])
        value = [cfg["d_model"], cfg["num_heads"]] if family == "model_size" else cfg[family]
        if cfg.get("varied_family") == family:
            selected.append({**row, "value": value})
        elif row["config_id"] == "anchor":
            selected.append({**row, "value": value})
    unique = {json.dumps(item["value"]): item for item in selected}
    return sorted(unique.values(), key=lambda item: order_value(family, item["value"]))


def error_line(ax, labels, means, sds, color, xlabel, rotation=0):
    x = np.arange(len(labels))
    ax.errorbar(x, means, yerr=sds, color=color, marker="o", ms=10,
                markeredgecolor=COLORS["black"], markeredgewidth=0.8,
                capsize=6, capthick=2.2, lw=2.8)
    ax.set_xticks(x, labels, rotation=rotation, ha="right" if rotation else "center")
    ax.set(xlabel=xlabel, ylabel="Validation MAPE (%)", xlim=(-0.45, len(labels) - 0.55),
           ylim=(0, max(np.asarray(means) + np.asarray(sds)) * 1.17))
    style_axis(ax)


def plot_ofat(root: Path, figures: Path):
    rows = read_csv(root / "ofat/summary/config_summary.csv")
    groups = [
        ("transformer_ofat_optimization", ["batch_size", "lr", "dropout", "weight_decay"], 2),
        ("transformer_ofat_architecture", ["model_size", "num_layers", "ff_mult", "readout", "position_encoding"], 3),
    ]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["teal"], COLORS["purple"], COLORS["green"]]
    for stem, families, ncols in groups:
        fig, axes = plt.subplots(2, ncols, figsize=(7.4 * ncols, 6.3 * 2))
        axes = np.asarray(axes).ravel()
        for ax, family, color in zip(axes, families, colors):
            items = ofat_family_rows(rows, family)
            error_line(ax, [value_label(family, item["value"]) for item in items],
                       [float(item["validation_MAPE_mean"]) for item in items],
                       [float(item["validation_MAPE_sd"]) for item in items], color, LABELS[family],
                       rotation=28 if family in {"readout", "position_encoding"} else 0)
        for ax in axes[len(families):]:
            ax.axis("off")
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.98, hspace=0.43, wspace=0.34)
        save_dual(fig, figures / stem)
        plt.close(fig)


def plot_interactions(root: Path, figures: Path):
    rows = read_csv(root / "interactions/summary/config_summary.csv")
    specs = [("lr_x_batch", "LR / batch"),
             ("size_x_layers", r"Size / layers"),
             ("dropout_x_weight_decay", "Dropout / decay")]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 20))
    for ax, (prefix, xlabel), color in zip(axes, specs, [COLORS["blue"], COLORS["orange"], COLORS["teal"]]):
        items = [row for row in rows if row["config_id"].startswith(prefix + "__")]
        items.sort(key=lambda row: row["config_id"])
        labels = [row["config_id"].split("__", 1)[1].replace("__", "\n") for row in items]
        error_line(ax, labels, [float(row["validation_MAPE_mean"]) for row in items],
                   [float(row["validation_MAPE_sd"]) for row in items], color, xlabel)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.08, top=0.985, hspace=0.56)
    save_dual(fig, figures / "transformer_interaction_checks")
    plt.close(fig)


def grouped_test_rows(root: Path) -> dict[str, list[dict[str, str]]]:
    groups = defaultdict(list)
    for row in read_csv(root / "final_test/per_run.csv"):
        key = f"{row['representation']}::{row['architecture']}"
        groups[key].append(row)
    return groups


def mean_sd(rows: list[dict[str, str]]) -> tuple[float, float]:
    values = np.asarray([float(row["MAPE"]) for row in rows])
    return float(values.mean()), float(values.std(ddof=1))


def bar_panel(ax, labels, values, errors, colors, ylabel="Test MAPE (%)"):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, yerr=errors, color=colors, edgecolor=COLORS["black"], linewidth=1.2,
                  capsize=7, error_kw={"lw": 2.2, "capthick": 2.2})
    ax.set(xlabel="", ylabel=ylabel, xticks=x, xticklabels=labels,
           xlim=(-0.6, len(labels) - 0.4), ylim=(0, max(np.asarray(values) + np.asarray(errors)) * 1.24))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value * 1.035, f"{value:.2f}",
                ha="center", va="bottom", fontsize=20)
    style_axis(ax)


def plot_final_ablation(root: Path, figures: Path):
    groups = grouped_test_rows(root)
    rep_keys = ["species_fraction_only::transformer", "no_shared_junction::transformer", "full_topology::transformer"]
    rep_labels = ["Species +\nfraction", "No shared\njunction", "Full\ntopology"]
    fig, ax = plt.subplots(figsize=(10.5, 8))
    rep_values = [mean_sd(groups[key]) for key in rep_keys]
    bar_panel(ax, rep_labels, [item[0] for item in rep_values], [item[1] for item in rep_values],
              [COLORS["purple"], COLORS["orange"], COLORS["blue"]])
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.15, top=0.96)
    save_dual(fig, figures / "transformer_representation_ablation")
    plt.close(fig)


def plot_parity(prediction_files: list[Path], figures: Path, stem: str):
    predictions = [read_csv(path) for path in prediction_files]
    names = [row["base_name"] for row in predictions[0]]
    y_true = np.asarray([float(row["true"]) for row in predictions[0]])
    y_pred = np.mean([[float(row["pred"]) for row in run] for run in predictions], axis=0)
    if any([row["base_name"] for row in run] != names for run in predictions[1:]):
        raise RuntimeError("Frozen-test prediction files are not aligned by sample name.")
    residual = y_pred - y_true
    mae = float(np.abs(residual).mean())
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mape = float(np.mean(np.abs(residual) / np.clip(np.abs(y_true), 1e-12, None)) * 100)
    r2 = float(1 - np.sum(residual ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    error = np.abs(residual) / np.clip(np.abs(y_true), 1e-12, None) * 100
    mean_error, p95, upper = float(error.mean()), float(np.percentile(error, 95)), float(np.percentile(error, 99.5))
    lims = (min(y_true.min(), y_pred.min()) * 0.86, max(y_true.max(), y_pred.max()) * 1.14)
    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    ax.scatter(y_true, y_pred, s=120, color=COLORS["parity_blue"], alpha=0.60,
               edgecolors="#FFFFFF", linewidths=0.7, zorder=2)
    ax.plot(lims, lims, "--", color=COLORS["black"], lw=2.8, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims,
           xlabel=r"True $\chi N^*$", ylabel=r"Predicted $\chi N^*$")
    ax.set_box_aspect(1)
    style_axis(ax)
    ax.text(0.035, 0.955, f"$R^2$ = {r2:.4f}\n$MAPE$ = {mape:.3f}%\n$RMSE$ = {rmse:.2f}\n$MAE$ = {mae:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=26)
    inset = ax.inset_axes([0.60, 0.115, 0.34, 0.285])
    inset.hist(error, bins=np.linspace(0, upper, 28), density=True, color=COLORS["density_green"], alpha=0.82, edgecolor="none")
    inset.axvline(mean_error, color=COLORS["mean_orange"], ls="--", lw=2.4)
    inset.axvline(p95, color="0.35", ls=":", lw=2.4)
    inset.set(xlim=(0, upper * 1.04), xlabel=r"$|error|$ (%)", ylabel="Probability density")
    style_axis(inset)
    inset.tick_params(which="major", labelsize=13, length=7, width=1.6)
    inset.tick_params(which="minor", length=3.5, width=1.2)
    inset.xaxis.label.set_size(17); inset.yaxis.label.set_size(17)
    inset.text(0.96, 0.94, f"mean={mean_error:.2f}%\n95th={p95:.2f}%", transform=inset.transAxes,
               ha="right", va="top", fontsize=16)
    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.15, top=0.90)
    save_dual(fig, figures / stem, tight=False)
    plt.close(fig)


def plot_test_parity(root: Path, figures: Path):
    prediction_files = sorted((root / "final_test").glob("full_topology__seed*_predictions.csv"))
    if len(prediction_files) != 5:
        raise RuntimeError("Need all five frozen-test prediction files before plotting ensemble parity.")
    plot_parity(prediction_files, figures, "transformer_ensemble_parity")


def plot_single_seed_parity(root: Path, figures: Path):
    prediction_file = root / "final_test/full_topology__seed202604_predictions.csv"
    if not prediction_file.is_file():
        raise RuntimeError("Missing the frozen-test prediction file for seed 202604.")
    plot_parity([prediction_file], figures, "transformer_best_single_seed_202604_parity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    validation_figures = args.root / "figures/validation_20k"
    test_figures = args.root / "figures/final_test_2k"
    plot_ofat(args.root, validation_figures)
    plot_interactions(args.root, validation_figures)
    plot_final_ablation(args.root, test_figures)
    plot_test_parity(args.root, test_figures)
    plot_single_seed_parity(args.root, test_figures)


if __name__ == "__main__":
    main()
