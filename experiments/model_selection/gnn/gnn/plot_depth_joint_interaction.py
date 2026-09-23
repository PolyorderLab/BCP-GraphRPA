#!/usr/bin/env python3
"""Plot the S01 encoder-by-decoder depth interaction on validation data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter


SKILL_SCRIPTS = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis  # noqa: E402


configure_paper_style()

HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "depth_joint_grid_100k_current_ofat"
SUMMARY = RESULT_ROOT / "summary" / "depth_joint_grid_validation_summary.csv"
OUTPUT_STEM = RESULT_ROOT / "figures" / "depth_interaction_validation_mape"

SERIES_STYLE = {
    1: {"color": COLORS["blue"], "marker": "o", "linestyle": "-"},
    2: {"color": COLORS["teal"], "marker": "s", "linestyle": "--"},
    3: {"color": COLORS["purple"], "marker": "^", "linestyle": ":"},
}


def _load_complete_grid(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {(encoder, decoder) for encoder in range(2, 6) for decoder in range(1, 4)}
    observed = set(zip(frame["num_layers"].astype(int), frame["dec_layers"].astype(int)))
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise RuntimeError(f"Incomplete E2–E5 × D1–D3 grid; missing={missing}, extra={extra}")
    if not (frame["n_seeds"].astype(int) == 5).all():
        raise RuntimeError("Every depth cell must contain five seeds")
    return frame


def _format_log_tick(value: float, _position: float) -> str:
    return f"{value:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM)
    args = parser.parse_args()

    frame = _load_complete_grid(args.summary)
    fig, ax = plt.subplots(figsize=(10.5, 8.5), facecolor="white")

    for decoder in (1, 2, 3):
        series = frame.loc[frame["dec_layers"] == decoder].sort_values("num_layers")
        style = SERIES_STYLE[decoder]
        ax.errorbar(
            series["num_layers"],
            series["validation_mape_mean_percent"],
            yerr=series["validation_mape_sd_percent"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=9,
            markerfacecolor=style["color"],
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.9,
            linewidth=2.8,
            elinewidth=2.0,
            capsize=5,
            capthick=1.8,
            label=f"D{decoder}",
            zorder=2,
        )

    e4d1 = frame.loc[(frame["num_layers"] == 4) & (frame["dec_layers"] == 1)].iloc[0]
    e5d1 = frame.loc[(frame["num_layers"] == 5) & (frame["dec_layers"] == 1)].iloc[0]
    ax.scatter(
        [4],
        [e4d1.validation_mape_mean_percent],
        s=190,
        facecolors="none",
        edgecolors=COLORS["black"],
        linewidths=2.2,
        zorder=4,
    )
    ax.scatter(
        [5],
        [e5d1.validation_mape_mean_percent],
        s=310,
        marker="*",
        color=COLORS["orange"],
        edgecolors=COLORS["black"],
        linewidths=1.2,
        zorder=5,
    )

    ax.annotate(
        "OFAT E4D1",
        xy=(4, e4d1.validation_mape_mean_percent),
        xytext=(3.38, 0.53),
        fontsize=21,
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["black"],
            "linewidth": 1.8,
            "shrinkA": 3,
            "shrinkB": 7,
        },
        zorder=6,
    )
    ax.annotate(
        "Selected E5D1",
        xy=(5, e5d1.validation_mape_mean_percent),
        xytext=(4.48, 0.43),
        fontsize=21,
        color=COLORS["orange"],
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["orange"],
            "linewidth": 1.8,
            "shrinkA": 3,
            "shrinkB": 9,
        },
        zorder=6,
    )

    ax.set_yscale("log")
    ax.set_xlim(1.75, 5.25)
    ax.set_ylim(0.36, 16.5)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xticklabels(["E2", "E3", "E4", "E5"])
    ax.set_xlabel("Number of GNN/encoder layers")
    ax.set_ylabel("Validation MAPE (%)")
    ax.set_title("Encoder–decoder depth interaction", fontsize=32, pad=14)
    ax.text(
        0.035,
        0.045,
        "mean ± SD, $n=5$ seeds",
        transform=ax.transAxes,
        fontsize=20,
        ha="left",
        va="bottom",
        color="0.25",
    )
    ax.legend(
        title="Decoder depth",
        loc="upper right",
        ncol=1,
        frameon=False,
        handlelength=2.1,
        borderaxespad=0.7,
    )

    style_axis(ax)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_log_tick))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.15, top=0.90)
    save_dual(fig, args.output_stem, tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
