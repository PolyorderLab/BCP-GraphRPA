"""Compute and plot the BTABAT spinodal line with RPA and mainmodel ML."""

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
import numpy as np

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


TOPOLOGY = "BTABAT"
FA_RANGE = (0.01, 0.99)
NUM_POINTS = 100
PLOT_COLOR = "#8C564B"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["topology", "method", "fA", "chiN"])
        writer.writeheader()
        for method in ["rpa", "ml"]:
            f_values = data[TOPOLOGY][method]["fA"]
            chi_values = data[TOPOLOGY][method]["chiN"]
            for f_a, chi_n in zip(f_values, chi_values):
                writer.writerow(
                    {
                        "topology": TOPOLOGY,
                        "method": method.upper(),
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


def plot_spinodal(data: dict, output_path: Path) -> None:
    use_publication_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.7), constrained_layout=True)
    color = PLOT_COLOR

    rpa = data[TOPOLOGY]["rpa"]
    ml = data[TOPOLOGY]["ml"]
    ax.plot(rpa["fA"], rpa["chiN"], color=color, lw=1.85, label="RPA")
    ax.scatter(
        ml["fA"],
        ml["chiN"],
        color=color,
        s=11,
        marker="o",
        edgecolor="black",
        linewidth=0.25,
        alpha=0.9,
        label="ML",
        zorder=3,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.45)
    ax.tick_params(axis="both", which="major", direction="in", width=1.2, length=6, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", width=1.0, length=3, top=True, right=True)
    ax.minorticks_on()

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 210.0)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xlabel(r"$f_A$")
    ax.set_ylabel(r"$\chi N$")
    ax.text(0.06, 0.065, r"$B_TABA_T$", transform=ax.transAxes, color="black", fontsize=15, fontstyle="italic")
    legend = ax.legend(
        ncols=2,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_local_spinodal(
    data: dict,
    output_path: Path,
    f_range: tuple[float, float] = (0.10, 0.30),
) -> None:
    use_publication_style()
    fig, ax = plt.subplots(figsize=(5.7, 3.7), constrained_layout=True)
    color = PLOT_COLOR
    selected_x_values = []

    for method, label in [("rpa", "RPA"), ("ml", "ML")]:
        values = data[TOPOLOGY][method]
        x = np.asarray(values["fA"], dtype=float)
        y = np.asarray(values["chiN"], dtype=float)
        start_idx = int(np.searchsorted(x, f_range[0], side="left"))
        end_idx = int(np.searchsorted(x, f_range[1], side="left"))
        end_idx = min(end_idx, len(x) - 1)
        selected = slice(start_idx, end_idx + 1)
        selected_x_values.append(x[selected])
        if method == "rpa":
            ax.plot(x[selected], y[selected], color=color, lw=1.85, label=label)
        else:
            ax.scatter(
                x[selected],
                y[selected],
                color=color,
                s=11,
                marker="o",
                edgecolor="black",
                linewidth=0.25,
                alpha=0.9,
                label=label,
                zorder=3,
            )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.45)
    ax.tick_params(axis="both", which="major", direction="in", width=1.2, length=6, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", width=1.0, length=3, top=True, right=True)
    ax.minorticks_on()

    selected_x = np.concatenate(selected_x_values)
    ax.set_xlim(float(selected_x.min()), float(selected_x.max()))
    ax.set_ylim(25.0, 170.0)
    ax.set_xticks([0.11, 0.15, 0.19, 0.23, 0.27, 0.30])
    ax.set_yticks([25, 50, 75, 100, 125, 150])
    ax.set_xlabel(r"$f_A$")
    ax.set_ylabel(r"$\chi N$")
    ax.text(0.06, 0.11, r"$B_TABA_T$", transform=ax.transAxes, color="black", fontsize=17, fontstyle="italic")
    legend = ax.legend(
        ncols=2,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--f-min", type=float, default=FA_RANGE[0])
    parser.add_argument("--f-max", type=float, default=FA_RANGE[1])
    parser.add_argument("--julia-threads", type=int, default=112)
    args = parser.parse_args()

    f_range = (args.f_min, args.f_max)
    f_values, rpa_chi = compute_rpa_phase_boundary(
        TOPOLOGY,
        fA_range=f_range,
        num_points=args.num_points,
        chiN0=15.0,
        julia_threads=args.julia_threads,
    )

    predictor = MainModelChiNPredictor(CHECKPOINT)
    ml_chi = predictor.predict_batch(TOPOLOGY, f_values)

    payload = {
        TOPOLOGY: {
            "rpa": {"fA": f_values.tolist(), "chiN": rpa_chi.tolist()},
            "ml": {"fA": f_values.tolist(), "chiN": ml_chi.tolist()},
            "parameters": {
                "fA1": "fA/4",
                "fA2": "fA/4",
                "fA3": "fA/2",
                "fB1": "0.15*(1-fA)",
                "fB2": "0.15*(1-fA)",
                "fB3": "0.70*(1-fA)",
                "tauB2": 0.15,
                "xi": "log(14/3)",
            },
        }
    }

    json_path = OUT_DIR / "btabat_spinodal_data.json"
    csv_path = COORD_DIR / "btabat_spinodal_coordinates.csv"
    png_path = OUT_DIR / "btabat_spinodal_publication.png"
    write_json(json_path, payload)
    write_csv(csv_path, payload)
    plot_spinodal(payload, png_path)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
