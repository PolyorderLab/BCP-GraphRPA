#!/usr/bin/env python3
"""Publication figures for the systematic PolymerType Transformer study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tapd")

import numpy as np

SKILL_SCRIPTS = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis

configure_paper_style()
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def family_data(rows, family):
    items = [row for row in rows if row["family"] == family]
    return items, np.asarray([float(row["MAPE_mean"]) for row in items]), np.asarray(
        [float(row["MAPE_sd"]) for row in items]
    )


LABELS = {
    "batch_size": "Batch size", "lr": "Learning rate",
    "model_size": r"$d_{model}$/heads", "num_layers": "Encoder layers",
    "ff_mult": "FF multiplier", "dropout": "Dropout", "readout": "Readout",
    "position_encoding": "Position encoding", "weight_decay": "Weight decay",
    "development_pool": "Development samples",
}


def display_value(family, raw):
    value = json.loads(raw)
    if family == "model_size":
        return f"{value[0]}/{value[1]}"
    if family == "lr" or family == "weight_decay":
        return f"{float(value):.0e}" if float(value) else "0"
    if family == "development_pool":
        return value.upper()
    if family == "position_encoding" and value == "learnable":
        return "Learned"
    if family == "readout" and value == "cls_mean":
        return "CLS+mean"
    return str(value).capitalize()


def sensitivity_panel(ax, rows, family, color):
    items, means, sds = family_data(rows, family)
    x = np.arange(len(items))
    labels = [display_value(family, row["value"]) for row in items]
    ax.errorbar(x, means, yerr=sds, color=color, marker="o", ms=9,
                markeredgecolor=COLORS["black"], markeredgewidth=0.8,
                capsize=6, capthick=2.2, lw=2.8)
    ax.set_xticks(x, labels, rotation=25 if family in {"readout", "position_encoding"} else 0,
                  ha="right" if family in {"readout", "position_encoding"} else "center")
    ax.set_xlabel(LABELS[family])
    ax.set_ylabel("MAPE (%)")
    ax.set_xlim(-0.45, len(items) - 0.55)
    upper = max(means + sds) * 1.15
    ax.set_ylim(0, upper)
    style_axis(ax)


def plot_ofat_groups(root, figures):
    rows = read_csv(root / "ofat/summary/family_results.csv")
    groups = [
        ("transformer_ofat_optimization", ["batch_size", "lr", "dropout", "weight_decay"]),
        ("transformer_ofat_architecture", [
            "model_size", "num_layers", "ff_mult", "readout", "position_encoding"
        ]),
    ]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["teal"], COLORS["purple"], COLORS["green"]]
    for stem, families in groups:
        ncols = 2 if len(families) == 4 else 3
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.4 * ncols, 6.3 * nrows))
        axes = np.asarray(axes).reshape(-1)
        for ax, family, color in zip(axes, families, colors):
            sensitivity_panel(ax, rows, family, color)
        for ax in axes[len(families):]:
            ax.axis("off")
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.98,
                            hspace=0.42, wspace=0.34)
        save_dual(fig, figures / stem)
        plt.close(fig)

    items, means, sds = family_data(rows, "development_pool")
    x = np.asarray([int(json.loads(row["value"]).replace("k", "")) * 1000 for row in items])
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.errorbar(x, means, yerr=sds, color=COLORS["blue"], marker="o", ms=10,
                markeredgecolor=COLORS["black"], markeredgewidth=0.8,
                capsize=6, capthick=2.2)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{v//1000}k" for v in x])
    ax.set(xlabel="Development samples", ylabel="MAPE (%)", ylim=(0, max(means+sds)*1.15))
    style_axis(ax)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.97)
    save_dual(fig, figures / "transformer_data_scaling")
    plt.close(fig)


def plot_interactions(root, figures):
    rows = read_csv(root / "interactions/summary/config_summary.csv")
    specs = [
        ("lr_x_batch", "LR / batch"),
        ("size_x_layers", r"$d_{model}$/heads / layers"),
        ("dropout_x_weight_decay", "Dropout / weight decay"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.6))
    for ax, (prefix, xlabel), color in zip(
        axes, specs, [COLORS["blue"], COLORS["orange"], COLORS["teal"]]
    ):
        items = [row for row in rows if row["config_id"].startswith(prefix + "__")]
        means = np.asarray([float(row["MAPE_mean"]) for row in items])
        sds = np.asarray([float(row["MAPE_sd"]) for row in items])
        labels = [row["config_id"].split("__", 1)[1].replace("__", "/") for row in items]
        x = np.arange(len(items))
        ax.errorbar(x, means, yerr=sds, color=color, marker="o", ms=9,
                    markeredgecolor=COLORS["black"], markeredgewidth=0.8,
                    capsize=6, capthick=2.2)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set(xlabel=xlabel, ylabel="MAPE (%)", xlim=(-0.45, len(items)-0.55),
               ylim=(0, max(means+sds)*1.16))
        style_axis(ax)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.25, top=0.97, wspace=0.34)
    save_dual(fig, figures / "transformer_interaction_checks")
    plt.close(fig)


def bar_with_error(ax, labels, means, sds, colors, ylabel="MAPE (%)", log=False):
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sds, color=colors, edgecolor=COLORS["black"],
                  linewidth=1.2, capsize=7, error_kw={"lw": 2.2, "capthick": 2.2})
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(labels)-0.4)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
        lower = min(means) / 1.55
        upper = max(np.asarray(means) + np.asarray(sds)) * 1.45
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(0, max(np.asarray(means)+np.asarray(sds))*1.24)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, value * (1.16 if log else 1.03),
                f"{value:.2f}", ha="center", va="bottom", fontsize=20)
    style_axis(ax)


def plot_final_results(root, figures):
    rep = read_csv(root / "final_summary/representation_summary.csv")
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    rep_order = ["species_fraction_only", "no_shared_junction", "full_topology"]
    rep_labels = ["Species +\nfraction", "No shared\njunction", "Full\ntopology"]
    items = [next(row for row in rep if row["representation"] == key) for key in rep_order]
    bar_with_error(ax, rep_labels, [float(r["MAPE_mean"]) for r in items],
                   [float(r["MAPE_sd"]) for r in items],
                   [COLORS["purple"], COLORS["orange"], COLORS["blue"]])
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.96)
    save_dual(fig, figures / "transformer_representation_ablation")
    plt.close(fig)

    comparison = read_csv(root / "final_summary/model_comparison_summary.csv")
    order = ["GNN", "Transformer"]
    items = [next(row for row in comparison if row["model"] == key) for key in order]
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    bar_with_error(ax, order, [float(r["MAPE_mean"]) for r in items],
                   [float(r["MAPE_sd"]) for r in items],
                   [COLORS["blue"], COLORS["orange"]])
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.96)
    save_dual(fig, figures / "gnn_transformer_comparison")
    plt.close(fig)

    order_rows = read_csv(root / "final_summary/cross_order_summary.csv")
    order = ["canonical", "reversed", "randomized"]
    items = [next(row for row in order_rows if row["order_variant"] == key) for key in order]
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    bar_with_error(ax, ["Canonical", "Reversed", "Randomized"],
                   [float(r["MAPE_mean"]) for r in items],
                   [float(r["MAPE_sd"]) for r in items],
                   [COLORS["blue"], COLORS["orange"], COLORS["purple"]], log=True)
    ax.set_ylabel("MAPE (%) — log scale")
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.16, top=0.96)
    save_dual(fig, figures / "transformer_same_checkpoint_order_robustness")
    plt.close(fig)


def plot_parity(root, figures):
    rows = read_csv(root / "final_summary/transformer_ensemble_predictions.csv")
    y_true = np.asarray([float(row["true"]) for row in rows])
    y_pred = np.asarray([float(row["pred"]) for row in rows])
    metrics = json.loads((root / "final_summary/transformer_ensemble_metrics.json").read_text())
    error = np.abs(y_pred-y_true) / np.clip(np.abs(y_true), 1e-12, None) * 100
    mean_error, p95 = float(error.mean()), float(np.percentile(error, 95))
    upper_error = float(np.percentile(error, 99.5))
    lims = (min(y_true.min(), y_pred.min())*0.86, max(y_true.max(), y_pred.max())*1.14)
    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    ax.scatter(y_true, y_pred, s=120, color=COLORS["parity_blue"], alpha=0.60,
               edgecolors="#FFFFFF", linewidths=0.7, zorder=2)
    ax.plot(lims, lims, "--", color=COLORS["black"], lw=2.8, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims,
           xlabel=r"True $\chi N^*$", ylabel=r"Predicted $\chi N^*$")
    ax.set_box_aspect(1)
    style_axis(ax)
    ax.text(0.035, 0.955,
            f"$R^2$ = {metrics['R2']:.4f}\n$MAPE$ = {metrics['MAPE']:.3f}%\n"
            f"$RMSE$ = {metrics['RMSE']:.2f}\n$MAE$ = {metrics['MAE']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=26)
    inset = ax.inset_axes([0.60, 0.115, 0.34, 0.285])
    inset.hist(error, bins=np.linspace(0, upper_error, 28), density=True,
               color=COLORS["density_green"], alpha=0.82, edgecolor="none")
    inset.axvline(mean_error, color=COLORS["mean_orange"], ls="--", lw=2.4)
    inset.axvline(p95, color="0.35", ls=":", lw=2.4)
    inset.set(xlim=(0, upper_error*1.04), xlabel=r"$|error|$ (%)",
              ylabel="Probability density")
    style_axis(inset)
    inset.tick_params(which="major", labelsize=13, length=7, width=1.6)
    inset.tick_params(which="minor", length=3.5, width=1.2)
    inset.xaxis.label.set_size(17); inset.yaxis.label.set_size(17)
    inset.text(0.96, 0.94, f"mean={mean_error:.2f}%\n95th={p95:.2f}%",
               transform=inset.transAxes, ha="right", va="top", fontsize=16)
    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.15, top=0.94)
    save_dual(fig, figures / "transformer_ensemble_parity", tight=False)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True)
    args = parser.parse_args()
    root = Path(args.batch_root)
    figures = root / "figures"
    plot_ofat_groups(root, figures)
    plot_interactions(root, figures)
    plot_final_results(root, figures)
    plot_parity(root, figures)
    print(figures)


if __name__ == "__main__":
    main()
