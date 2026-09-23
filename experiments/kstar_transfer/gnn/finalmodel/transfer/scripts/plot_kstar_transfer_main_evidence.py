#!/usr/bin/env python3
"""Draw the five-seed k* transfer test evidence and label learning curves."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tapd_mplconfig")

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

PROJECT_ROOT = Path("/home/zxy/project/TAPD")
TRANSFER_ROOT = PROJECT_ROOT / "baselinemodels/pub0818/gnn/finalmodel/transfer"
CURVE_ROOT = TRANSFER_ROOT / "kstar_label_curve_stratified_20260913"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "baselinemodels/pub0818/gnn/finalmodel/figure/kstar_transfer/main_evidence"
)
STYLE_ROOT = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(STYLE_ROOT))

from paper_style import COLORS, configure_paper_style, save_dual, style_axis  # noqa: E402

configure_paper_style()
import matplotlib.pyplot as plt  # noqa: E402

SEEDS = (202601, 202602, 202603, 202604, 202605)
SIZES = (100, 250, 500, 1000, 2000)
METHODS = (
    "frozen_encoder",
    "frozen_embedding_mlp",
    "finetune_encoder",
    "two_stage",
    "scratch",
)
LABELS = {
    "frozen_encoder": "Frozen +\nlinear",
    "frozen_embedding_mlp": "Frozen +\nMLP",
    "finetune_encoder": "End-to-end\nfine-tuning",
    "two_stage": "Two-stage",
    "scratch": "Scratch",
}
LEGEND_LABELS = {
    "frozen_encoder": "Frozen encoder + linear decoder",
    "frozen_embedding_mlp": "Frozen embedding + nonlinear MLP",
    "finetune_encoder": "End-to-end fine-tuning",
    "two_stage": "Two-stage transfer",
    "scratch": "Scratch",
}
METHOD_COLORS = {
    "frozen_encoder": COLORS["purple"],
    "frozen_embedding_mlp": COLORS["teal"],
    "finetune_encoder": COLORS["orange"],
    "two_stage": COLORS["green"],
    "scratch": COLORS["blue"],
}
METHOD_MARKERS = {
    "frozen_encoder": "o",
    "frozen_embedding_mlp": "s",
    "finetune_encoder": "^",
    "two_stage": "D",
    "scratch": "P",
}


def read_mape(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = float(json.loads(path.read_text(encoding="utf-8"))["MAPE"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid MAPE in {path}: {value}")
    return value


def full_result_path(method: str, seed: int) -> Path:
    run_id = SEEDS.index(seed) + 1
    if method == "frozen_embedding_mlp":
        return (
            TRANSFER_ROOT
            / "c2_512_embedding_mlp_probe_20260903"
            / f"run{run_id:02d}_seed{seed}"
            / "metrics.json"
        )
    if method == "scratch":
        root = TRANSFER_ROOT / "scratch_c2_512"
        retry = root / f"scratch_c2_512_retry_run{run_id:02d}_seed{seed}"
        initial = root / f"scratch_c2_512_run{run_id:02d}_seed{seed}"
        directory = retry if retry.is_dir() else initial
        return directory / "seed43_metrics.json"
    directory = TRANSFER_ROOT / method / f"{method}_run{run_id:02d}_seed{seed}"
    return directory / "seed43_metrics.json"


def load_full_results():
    return {
        method: np.asarray([read_mape(full_result_path(method, seed)) for seed in SEEDS])
        for method in METHODS
    }


def load_curve_results():
    results = {}
    for method in METHODS:
        method_values = []
        for size in SIZES:
            values = []
            for seed in SEEDS:
                path = CURVE_ROOT / method / f"n{size}" / f"seed{seed}" / "metrics.json"
                values.append(read_mape(path))
            method_values.append(values)
        results[method] = np.asarray(method_values, dtype=float)
    return results


def write_tables(full_results, curve_results=None):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "kstar_transfer_full_data_five_seed_mape.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "seed", "MAPE_percent"])
        for method in METHODS:
            for seed, value in zip(SEEDS, full_results[method], strict=True):
                writer.writerow([method, seed, f"{value:.10g}"])
    if curve_results is None:
        return
    with (OUTPUT_ROOT / "kstar_transfer_label_curve_five_seed_mape.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "train_size", "seed", "MAPE_percent", "mean", "sample_sd"]
        )
        for method in METHODS:
            for size, row in zip(SIZES, curve_results[method], strict=True):
                mean, sd = float(row.mean()), float(row.std(ddof=1))
                for seed, value in zip(SEEDS, row, strict=True):
                    writer.writerow(
                        [method, size, seed, f"{value:.10g}", f"{mean:.10g}", f"{sd:.10g}"]
                    )


def format_log_number(value, _position):
    return f"{value:g}"


def draw_full_panel(ax, full_results):
    offsets = np.linspace(-0.16, 0.16, len(SEEDS))
    for x, method in enumerate(METHODS):
        values = full_results[method]
        color = METHOD_COLORS[method]
        ax.scatter(
            x + offsets,
            values,
            s=125,
            color=color,
            alpha=0.82,
            edgecolors="#FFFFFF",
            linewidths=0.8,
            zorder=3,
        )
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        ax.errorbar(
            x,
            mean,
            yerr=sd,
            fmt="D",
            ms=10,
            color=COLORS["black"],
            markerfacecolor="white",
            markeredgewidth=2.0,
            capsize=7,
            capthick=2.2,
            elinewidth=2.2,
            zorder=4,
        )
    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(METHODS) - 0.5)
    ax.set_ylim(0.09, 5.5)
    ax.set_xticks(range(len(METHODS)), [LABELS[method] for method in METHODS])
    ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5]))
    ax.yaxis.set_major_formatter(FuncFormatter(format_log_number))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title(r"Full data ($n_{\rm train}=80{,}000$)", fontsize=32, pad=18)
    ax.text(
        0.965,
        0.955,
        "points: five seeds\ndiamond: mean $\\pm$ sample SD",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=20,
    )
    style_axis(ax)
    ax.tick_params(axis="x", labelsize=18, pad=10)


def draw_curve_panel(ax, curve_results):
    for method in METHODS:
        values = curve_results[method]
        mean = values.mean(axis=1)
        sd = values.std(axis=1, ddof=1)
        color = METHOD_COLORS[method]
        ax.fill_between(SIZES, np.maximum(mean - sd, 1e-4), mean + sd, color=color, alpha=0.13)
        ax.plot(
            SIZES,
            mean,
            color=color,
            marker=METHOD_MARKERS[method],
            markersize=8,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.8,
            label=LEGEND_LABELS[method],
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(85, 2350)
    all_values = np.concatenate([curve_results[method].reshape(-1) for method in METHODS])
    ax.set_ylim(max(0.05, all_values.min() * 0.72), all_values.max() * 1.38)
    ax.xaxis.set_major_locator(FixedLocator(SIZES))
    ax.xaxis.set_major_formatter(FuncFormatter(format_log_number))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(format_log_number))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Number of labeled training samples")
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("Block-stratified learning curve", fontsize=32, pad=18)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        fontsize=18,
        handlelength=2.0,
        columnspacing=1.1,
    )
    style_axis(ax)
    ax.tick_params(axis="x", labelsize=21)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Render the complete full-data dot panel while learning-curve jobs are running.",
    )
    args = parser.parse_args()
    full_results = load_full_results()
    if args.full_only:
        write_tables(full_results)
        fig, ax = plt.subplots(figsize=(10.5, 8.5))
        draw_full_panel(ax, full_results)
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.20, top=0.88)
        save_dual(fig, OUTPUT_ROOT / "kstar_transfer_full_data_five_seed_mape")
        plt.close(fig)
        return

    curve_results = load_curve_results()
    write_tables(full_results, curve_results)
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 7.4))
    draw_full_panel(axes[0], full_results)
    draw_curve_panel(axes[1], curve_results)
    axes[0].text(-0.14, 1.06, r"$\it{a}$", transform=axes[0].transAxes, fontsize=32)
    axes[1].text(-0.14, 1.06, r"$\it{b}$", transform=axes[1].transAxes, fontsize=32)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.31, top=0.88, wspace=0.32)
    save_dual(fig, OUTPUT_ROOT / "kstar_transfer_main_evidence")
    plt.close(fig)


if __name__ == "__main__":
    main()
