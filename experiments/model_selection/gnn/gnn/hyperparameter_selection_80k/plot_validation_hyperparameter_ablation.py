"""Plot the completed five-panel GNN OFAT validation study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


SKILL_SCRIPTS = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis


configure_paper_style()

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "runs_100k_fixed80_20_validation_only"
SUMMARY = RUN_ROOT / "summary" / "hyperparameter_validation_summary.csv"
SELECTED = RUN_ROOT / "summary" / "selected_by_validation_mape.json"
OUTPUT_STEM = RUN_ROOT / "figures" / "gnn_ofat_validation_mape"
PANEL_SPECS = (
    ("batch_size", "(a) Batch size", "Batch size", False),
    ("learning_rate", "(b) Learning rate", "Learning rate", True),
    ("hidden_dim", "(c) Hidden dimension", "Hidden dimension", False),
    ("num_layers", "(d) GNN depth", "Number of GNN layers", False),
    ("dec_layers", "(e) Decoder depth", "Number of decoder layers", False),
)


def _format_rate(value: float, _position: float) -> str:
    exponent = int(np.round(np.log10(value)))
    return rf"$10^{{{exponent}}}$"


def _panel_data(frame: pd.DataFrame, dimension: str):
    panel = frame.loc[frame["dimension"] == dimension].copy()
    panel["x_numeric"] = panel["value_label"].astype(float)
    panel = panel.sort_values("x_numeric")
    return (
        panel["x_numeric"].to_numpy(float),
        panel["validation_mape_mean_percent"].to_numpy(float),
        panel["validation_mape_sd_percent"].to_numpy(float),
    )


def _set_limits(ax, plot_x: np.ndarray, y: np.ndarray, err: np.ndarray, log_x: bool) -> None:
    low = float(np.min(y - err))
    high = float(np.max(y + err))
    span = max(high - low, 0.2)
    ax.set_ylim(max(0.0, low - 0.18 * span), high + 0.22 * span)
    if log_x:
        ax.set_xlim(float(np.min(plot_x)) / 1.65, float(np.max(plot_x)) * 1.65)
    else:
        ax.set_xlim(-0.42, len(plot_x) - 0.58)


def _add_gnn_depth_inset(
    ax, plot_x: np.ndarray, y: np.ndarray, err: np.ndarray, selected_index: int
) -> None:
    """Resolve the compressed three-to-five-layer region in panel (d)."""
    local = np.arange(len(plot_x)) >= 2
    inset = ax.inset_axes([0.47, 0.45, 0.47, 0.46])
    inset.errorbar(
        plot_x[local],
        y[local],
        yerr=err[local],
        color=COLORS["blue"],
        marker="o",
        markersize=5.5,
        markerfacecolor=COLORS["blue"],
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.7,
        linewidth=2.2,
        elinewidth=1.6,
        capsize=3,
        capthick=1.4,
        zorder=2,
    )
    inset.scatter(
        [plot_x[selected_index]],
        [y[selected_index]],
        s=75,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=1.0,
        zorder=4,
    )
    inset.set_xlim(1.72, 4.28)
    inset.set_ylim(0.35, 1.40)
    inset.set_xticks(plot_x[local])
    inset.set_xticklabels(["3", "4", "5"])
    inset.set_yticks([0.5, 1.0])
    inset.set_title("Layers 3–5", fontsize=17, pad=4)
    style_axis(inset)
    inset.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.5)
    inset.tick_params(axis="both", which="minor", length=3, width=1.0)

    # A compact dashed box identifies the magnified range without drawing long
    # connector lines across the steep one-to-two-layer trend.
    ax.plot(
        [1.72, 4.28, 4.28, 1.72, 1.72],
        [0.35, 0.35, 1.40, 1.40, 0.35],
        color=COLORS["black"],
        linestyle=(0, (3, 2)),
        linewidth=1.5,
        alpha=0.70,
        zorder=3,
    )


def _plot_panel(ax, frame: pd.DataFrame, selected: dict[str, str], spec) -> None:
    dimension, title, xlabel, log_x = spec
    x, y, err = _panel_data(frame, dimension)
    plot_x = x if log_x else np.arange(len(x), dtype=float)

    ax.errorbar(
        plot_x,
        y,
        yerr=err,
        color=COLORS["blue"],
        marker="o",
        markersize=7,
        markerfacecolor=COLORS["blue"],
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.8,
        linewidth=2.8,
        elinewidth=2.0,
        capsize=4,
        capthick=1.8,
        zorder=2,
        label="5-seed mean ± SD",
    )

    selected_x = float(selected[dimension])
    selected_index = int(np.flatnonzero(np.isclose(x, selected_x))[0])
    selected_plot_x = selected_x if log_x else float(selected_index)
    ax.scatter(
        [selected_plot_x],
        [y[selected_index]],
        s=115,
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=1.2,
        zorder=4,
        label="Selected",
    )

    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=10))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_rate))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xticks(plot_x)
        ax.set_xticklabels([f"{value:g}" for value in x])

    _set_limits(ax, plot_x, y, err, log_x)
    ax.set_title(title, fontsize=32, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation MAPE (%)")
    style_axis(ax)
    if dimension == "num_layers":
        _add_gnn_depth_inset(ax, plot_x, y, err, selected_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM)
    args = parser.parse_args()
    frame = pd.read_csv(SUMMARY)
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))["selected_values"]

    # Keep all panels at the same size: three across the top and two centered
    # beneath them. A six-column grid gives each axis an equal two-column span.
    fig = plt.figure(figsize=(16.0, 10.76), facecolor="white")
    grid = fig.add_gridspec(2, 6, wspace=0.85, hspace=0.78)
    axes = (
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    )

    for ax, spec in zip(axes, PANEL_SPECS):
        _plot_panel(ax, frame, selected, spec)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_items = dict(zip(labels, handles))
    axes[0].legend(
        [legend_items["5-seed mean ± SD"], legend_items["Selected"]],
        ["5-seed mean ± SD", "Selected"],
        loc="upper center",
        bbox_to_anchor=(0.61, 0.99),
        ncol=1,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.095)
    save_dual(fig, args.output_stem, tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
