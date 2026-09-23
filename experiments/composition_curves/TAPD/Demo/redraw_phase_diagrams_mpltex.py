"""Redraw publication phase diagrams with Matplotlib + mpltex."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpltex
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


ROOT = Path("/home/zxy/project/TAPD")
OUT_DIR = ROOT / "Demo" / "outcome" / "phase_diagrams"
PLOT_DATA = OUT_DIR / "plot_data.json"
M1_M2_DATA = OUT_DIR / "m1_m2_comparison_data.json"
RPA_LINE_WIDTH = 1.85
ML_MARKER_SIZE = 11
GRID_ML_MARKER_SIZE = 12
LOCAL_ML_STEP = 4
LEGEND_KW = {
    "loc": "upper center",
    "frameon": False,
    "fancybox": False,
    "framealpha": 0.94,
    "edgecolor": "0.15",
    "borderpad": 0.28,
    "handlelength": 1.8,
    "handletextpad": 0.45,
    "columnspacing": 0.9,
}

COLORS = {
    "AB": "#2E86AB",
    "ABA": "#A23B72",
    "ABAB": "#F18F01",
    "ABABA": "#C73E1D",
    "AB2": "#3A7D44",
    "AB3": "#9B59B6",
    "M1": "#E67E22",
    "M2": "#1ABC9C",
}


def load_data(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def series(data: dict, topo: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    values = data[topo][kind]
    return np.asarray(values["fA"], dtype=float), np.asarray(values["chiN"], dtype=float)


def apply_publication_axes(ax, xlabel=True, ylabel=True, grid=False):
    ax.grid(grid)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.45)
    ax.tick_params(axis="both", which="major", direction="in", width=1.2, length=6, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", width=1.0, length=3, top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    if xlabel:
        ax.set_xlabel(r"$f_A$")
    if ylabel:
        ax.set_ylabel(r"$\chi N$")


def add_inside_legend(ax, ncols=2, fontsize=12):
    legend = ax.legend(ncols=ncols, fontsize=fontsize, **LEGEND_KW)
    legend.get_frame().set_linewidth(0.8)
    return legend


def add_topology_label(ax, label, color="black"):
    ax.text(
        0.5,
        0.94,
        label,
        transform=ax.transAxes,
        color=color,
        fontsize=17,
        fontstyle="italic",
        ha="center",
        va="top",
    )


def add_stacked_legend_below_label(ax):
    legend = ax.legend(
        ncols=1,
        fontsize=11.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        frameon=False,
        borderpad=0.0,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.25,
    )
    return legend


def save_png(fig, path: Path, dpi: int = 300):
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def use_local_fonts():
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["axes.titlesize"] = 13


def plot_rpa(ax, x, y, color, label="RPA", lw=RPA_LINE_WIDTH, step=1):
    ax.plot(x, y, color=color, lw=lw, ls="-", label=label, zorder=2)


def plot_ml(ax, x, y, color, label="ML", ms=ML_MARKER_SIZE, step=1):
    x = x[::step]
    y = y[::step]
    ax.scatter(
        x,
        y,
        color=color,
        s=ms,
        marker="o",
        edgecolor="black",
        linewidth=0.25,
        alpha=0.9,
        label=label,
        zorder=3,
    )


@mpltex.acs_decorator
def plot_rpa_ml_comparison_all(data: dict, output_path: Path):
    use_local_fonts()
    fig, ax = plt.subplots(figsize=(4.2, 3.7), constrained_layout=True)
    topologies = ["AB", "ABA", "ABAB", "ABABA"]

    for topo in topologies:
        color = COLORS[topo]
        if "rpa" in data.get(topo, {}):
            x, y = series(data, topo, "rpa")
            plot_rpa(ax, x, y, color, label=f"{topo} RPA")
        if "ml" in data.get(topo, {}):
            x, y = series(data, topo, "ml")
            plot_ml(ax, x, y, color, label=f"{topo} ML")

    apply_publication_axes(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 210.0)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticks([0, 50, 100, 150, 200])
    add_inside_legend(ax, ncols=2, fontsize=8.0)
    save_png(fig, output_path)


@mpltex.acs_decorator
def plot_rpa_ml_comparison_grid(data: dict, output_path: Path):
    use_local_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.9), constrained_layout=True)
    topologies = ["AB", "ABA", "ABAB", "ABABA"]

    for ax, topo in zip(axes.ravel(), topologies):
        color = COLORS[topo]
        if "rpa" in data.get(topo, {}):
            x, y = series(data, topo, "rpa")
            plot_rpa(ax, x, y, color, label="RPA")
        if "ml" in data.get(topo, {}):
            x, y = series(data, topo, "ml")
            plot_ml(ax, x, y, color, label="ML", ms=GRID_ML_MARKER_SIZE)
        apply_publication_axes(ax)
        add_topology_label(ax, topo)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 550.0)
        ax.set_yticks([0, 100, 200, 300, 400, 500])

        add_stacked_legend_below_label(ax)
    save_png(fig, output_path)


@mpltex.acs_decorator
def plot_rpa_ab_series(data: dict, output_path: Path):
    use_local_fonts()
    fig, ax = plt.subplots(figsize=(4.2, 3.7), constrained_layout=True)
    topologies = ["AB", "AB2", "AB3"]

    for topo in topologies:
        color = COLORS[topo]
        if "rpa" in data.get(topo, {}):
            x, y = series(data, topo, "rpa")
            plot_rpa(ax, x, y, color, label=f"{topo} RPA")
        if "ml" in data.get(topo, {}):
            x, y = series(data, topo, "ml")
            plot_ml(ax, x, y, color, label=f"{topo} ML")

    apply_publication_axes(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 210.0)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticks([0, 50, 100, 150, 200])
    add_inside_legend(ax, ncols=2, fontsize=8.5)
    save_png(fig, output_path)


@mpltex.acs_decorator
def plot_rpa_ab_series_grid(data: dict, output_path: Path):
    use_local_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9), constrained_layout=True)
    topologies = ["AB", "AB2", "AB3"]

    for ax, topo in zip(axes, topologies):
        color = COLORS[topo]
        if "rpa" in data.get(topo, {}):
            x, y = series(data, topo, "rpa")
            plot_rpa(ax, x, y, color, label="RPA")
        if "ml" in data.get(topo, {}):
            x, y = series(data, topo, "ml")
            plot_ml(ax, x, y, color, label="ML", ms=GRID_ML_MARKER_SIZE)
        apply_publication_axes(ax)
        add_topology_label(ax, topo)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 550.0)
        ax.set_yticks([0, 100, 200, 300, 400, 500])

        add_stacked_legend_below_label(ax)
    save_png(fig, output_path)


@mpltex.acs_decorator
def plot_m1_m2_comparison(data: dict, output_path: Path):
    use_local_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.7), constrained_layout=True)

    for ax, topo in zip(axes, ["M1", "M2"]):
        color = COLORS[topo]
        if "rpa" in data.get(topo, {}):
            x, y = series(data, topo, "rpa")
            plot_rpa(ax, x, y, color, label="RPA")
        if "ml" in data.get(topo, {}):
            x, y = series(data, topo, "ml")
            plot_ml(ax, x, y, color, label="ML", step=LOCAL_ML_STEP)
        apply_publication_axes(ax)
        ax.text(0.06, 0.10, rf"${topo}$", transform=ax.transAxes, color="black", fontsize=20, fontstyle="italic")
        ax.set_xlim(0.11, 0.19)
        ax.set_ylim(50.0, 155.0)
        ax.set_yticks([50, 75, 100, 125, 150])

        add_inside_legend(ax, ncols=2, fontsize=11.5)
    save_png(fig, output_path)


@mpltex.acs_decorator
def plot_m1_m2_global(data: dict, output_path: Path):
    use_local_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)

    for ax, topo in zip(axes, ["M1", "M2"]):
        color = COLORS[topo]
        if "rpa" in data.get(topo, {}):
            x, y = series(data, topo, "rpa")
            plot_rpa(ax, x, y, color, label="RPA")
        if "ml" in data.get(topo, {}):
            x, y = series(data, topo, "ml")
            plot_ml(ax, x, y, color, label="ML")
        apply_publication_axes(ax)
        ax.text(0.08, 0.12, rf"${topo}$", transform=ax.transAxes, color="black", fontsize=17, fontstyle="italic")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 210.0)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_yticks([0, 50, 100, 150, 200])

        add_inside_legend(ax, ncols=2, fontsize=11.5)
    save_png(fig, output_path)


def redraw_from_data(data_path: Path, m1_m2_data_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(data_path)
    m1_m2_data = load_data(m1_m2_data_path) if m1_m2_data_path.exists() else data

    plot_m1_m2_comparison(m1_m2_data, output_dir / "m1_m2_comparison_publication.png")
    plot_m1_m2_global(data, output_dir / "m1_m2_global_publication.png")
    plot_rpa_ab_series_grid(data, output_dir / "rpa_ab_series_grid_publication.png")
    plot_rpa_ab_series(data, output_dir / "rpa_ab_series_publication.png")
    plot_rpa_ml_comparison_all(data, output_dir / "rpa_ml_comparison_all_publication.png")
    plot_rpa_ml_comparison_grid(data, output_dir / "rpa_ml_comparison_grid_publication.png")


def main() -> None:
    redraw_from_data(PLOT_DATA, M1_M2_DATA, OUT_DIR)


if __name__ == "__main__":
    main()
