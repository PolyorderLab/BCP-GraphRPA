"""Compute and plot Shi 2021 AB/B2A2/(B2A2)2 spinodal series."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpltex
import numpy as np
from matplotlib.ticker import AutoMinorLocator

ROOT = Path("/home/zxy/project/TAPD")
SRC_DIR = ROOT / "src"
MAINMODEL_DIR = ROOT / "mainmodel"
OUT_DIR = MAINMODEL_DIR / "phase_diagrams"
COORD_DIR = OUT_DIR / "coordinate_points"
CHECKPOINT = (
    MAINMODEL_DIR
    / "five_seeds"
    / "vn_free_branch_degree_tuned_from_ablation_rerun_5runs_20260628_134001"
    / "VN_FreeBranchDegree"
    / "VN_FreeBranchDegree_run03_seed202603"
    / "best_model.pth"
)

sys.path.append(str(ROOT))
sys.path.append(str(SRC_DIR))

from mainmodel.infer_phase_diagrams_from_rpa import MainModelChiNPredictor  # noqa: E402
from src.rpa_calculator_julia import compute_rpa_phase_boundary  # noqa: E402


TOPOLOGIES = ["AB", "B2A2", "(B2A2)2"]
LABELS = {
    "AB": "AB",
    "B2A2": r"$B_2A_2$",
    "(B2A2)2": r"$(B_2A_2)_2$",
}
COLORS = {
    "AB": "#1F3A5F",
    "B2A2": "#D45087",
    "(B2A2)2": "#C49A00",
}
FA_RANGE = (0.01, 0.99)
NUM_POINTS = 100


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["topology", "method", "point_index", "fA", "chiN"])
        writer.writeheader()
        for topo in TOPOLOGIES:
            for method in ["rpa", "ml"]:
                f_values = data[topo][method]["fA"]
                chi_values = data[topo][method]["chiN"]
                for idx, (f_a, chi_n) in enumerate(zip(f_values, chi_values), start=1):
                    writer.writerow(
                        {
                            "topology": topo,
                            "method": method.upper(),
                            "point_index": idx,
                            "fA": f"{f_a:.12g}",
                            "chiN": f"{chi_n:.12g}",
                        }
                    )


def use_publication_style() -> None:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11.5


def apply_publication_axes(ax) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.45)
    ax.tick_params(axis="both", which="major", direction="in", width=1.2, length=6, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", width=1.0, length=3, top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_xlabel(r"$f_A$")
    ax.set_ylabel(r"$\chi N$")


def series(data: dict, topo: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    values = data[topo][kind]
    return np.asarray(values["fA"], dtype=float), np.asarray(values["chiN"], dtype=float)


def plot_rpa(ax, x, y, color, label):
    ax.plot(x, y, color=color, lw=1.85, ls="-", label=label, zorder=2)


def plot_ml(ax, x, y, color, label, marker_size=11):
    ax.scatter(
        x,
        y,
        color=color,
        s=marker_size,
        marker="o",
        edgecolor="black",
        linewidth=0.25,
        alpha=0.9,
        label=label,
        zorder=3,
    )


def add_inside_legend(ax, ncols=3, fontsize=11.5):
    legend = ax.legend(
        ncols=ncols,
        fontsize=fontsize,
        loc="upper center",
        frameon=False,
        fancybox=False,
        framealpha=0.94,
        edgecolor="0.15",
        borderpad=0.28,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    legend.get_frame().set_linewidth(0.8)


def add_topology_label(ax, label):
    ax.text(
        0.5,
        0.94,
        label,
        transform=ax.transAxes,
        color="black",
        fontsize=17,
        fontstyle="italic",
        ha="center",
        va="top",
    )


@mpltex.acs_decorator
def plot_series(data: dict, output_path: Path) -> None:
    use_publication_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.7), constrained_layout=True)

    for topo in TOPOLOGIES:
        color = COLORS[topo]
        x, y = series(data, topo, "rpa")
        plot_rpa(ax, x, y, color, label=f"{LABELS[topo]} RPA")
        x, y = series(data, topo, "ml")
        plot_ml(ax, x, y, color, label=f"{LABELS[topo]} ML")

    apply_publication_axes(ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 210.0)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticks([0, 50, 100, 150, 200])
    add_inside_legend(ax, ncols=2, fontsize=8.5)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


@mpltex.acs_decorator
def plot_grid(data: dict, output_path: Path) -> None:
    use_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9), constrained_layout=True)

    for ax, topo in zip(axes, TOPOLOGIES):
        color = COLORS[topo]
        x, y = series(data, topo, "rpa")
        plot_rpa(ax, x, y, color, label="RPA")
        x, y = series(data, topo, "ml")
        plot_ml(ax, x, y, color, label="ML", marker_size=12)
        apply_publication_axes(ax)
        add_topology_label(ax, LABELS[topo])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 550.0)
        ax.set_yticks([0, 100, 200, 300, 400, 500])
        ax.legend(
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

    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def compute_data(num_points: int, f_range: tuple[float, float], julia_threads: int) -> dict:
    predictor = MainModelChiNPredictor(CHECKPOINT)
    payload = {}

    for topo in TOPOLOGIES:
        f_values, rpa_chi = compute_rpa_phase_boundary(
            topo,
            fA_range=f_range,
            num_points=num_points,
            chiN0=15.0,
            julia_threads=julia_threads,
        )
        ml_chi = predictor.predict_batch(topo, f_values)
        payload[topo] = {
            "rpa": {"fA": f_values.tolist(), "chiN": rpa_chi.tolist()},
            "ml": {"fA": f_values.tolist(), "chiN": ml_chi.tolist()},
        }

    payload["parameters"] = {
        "source": "Shi 2021 Fig. 3 topology logic",
        "AB": {"A": "fA", "B": "1-fA"},
        "B2A2": {"A1": "fA/2", "A2": "fA/2", "B1": "(1-fA)/2", "B2": "(1-fA)/2"},
        "(B2A2)2": {
            "A1": "fA/4",
            "A2": "fA/4",
            "A3": "fA/4",
            "A4": "fA/4",
            "B1": "(1-fA)/4",
            "B2": "(1-fA)/4",
            "B3": "(1-fA)/4",
            "B4": "(1-fA)/4",
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--f-min", type=float, default=FA_RANGE[0])
    parser.add_argument("--f-max", type=float, default=FA_RANGE[1])
    parser.add_argument("--julia-threads", type=int, default=112)
    parser.add_argument("--from-data", action="store_true")
    args = parser.parse_args()

    data_path = OUT_DIR / "shi_b2a2_series_data.json"
    csv_path = COORD_DIR / "shi_b2a2_series_coordinates.csv"
    series_path = OUT_DIR / "shi_b2a2_series_publication.png"
    grid_path = OUT_DIR / "shi_b2a2_series_grid_publication.png"

    if args.from_data and data_path.exists():
        payload = load_json(data_path)
    else:
        payload = compute_data(args.num_points, (args.f_min, args.f_max), args.julia_threads)
        write_json(data_path, payload)
        write_csv(csv_path, payload)
        print(f"Saved: {data_path}")
        print(f"Saved: {csv_path}")

    plot_series(payload, series_path)
    plot_grid(payload, grid_path)


if __name__ == "__main__":
    main()
