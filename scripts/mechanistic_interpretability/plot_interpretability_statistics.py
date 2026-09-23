#!/usr/bin/env python3
"""Draw the four-panel mechanistic-interpretability figure directly with Matplotlib."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
HISTORY_RESULTS = HERE.parent / "figure" / "results_legal_growth_histories"
OUTPUT = HERE.parent / "figure" / "abtab2_interpretability_statistics"
EDGE_ORDER = ["A1", "B1", "A2", "B2", "B3"]
COLORS = {
    "blue": "#4C93C3",
    "orange": "#E9780D",
    "teal": "#1CB39E",
    "purple": "#9B75CC",
    "red": "#E15759",
    "ink": "#222222",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


def representative_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = pd.read_csv(RESULTS / "abtab2_rpa_edge_influence.csv")
    deletion = pd.read_csv(RESULTS / "abtab2_gnnexplainer_edge_summary.csv")
    edges = edges[np.isclose(edges.f_A, 0.5) & np.isclose(edges.rho, 1.0)].copy()
    deletion = deletion[np.isclose(deletion.f_A, 0.5) & np.isclose(deletion.rho, 1.0)].copy()
    for frame in (edges, deletion):
        frame["edge"] = pd.Categorical(frame.edge, EDGE_ORDER, ordered=True)
        frame.sort_values("edge", inplace=True)
    assert edges.edge.astype(str).tolist() == EDGE_ORDER
    assert deletion.edge.astype(str).tolist() == EDGE_ORDER
    return edges, deletion


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_color(COLORS["ink"])


def panel_a(ax: plt.Axes, edges: pd.DataFrame) -> None:
    x = np.arange(len(edges))
    width = 0.22
    specs = [
        ("I_self", r"$I_{\mathrm{self}}$", COLORS["blue"]),
        ("I_junction", r"$I_{\mathrm{junction}}$", COLORS["orange"]),
        ("I_separated", r"$I_{\mathrm{separated}}$", COLORS["teal"]),
    ]
    for offset, (column, label, color) in zip((-width, 0, width), specs):
        ax.bar(x + offset, edges[column], width * 0.92, color=color,
               edgecolor=COLORS["ink"], linewidth=0.55, label=label)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set(xticks=x, xticklabels=EDGE_ORDER, xlabel="Block edge",
           ylabel=r"Signed RPA component $I_e$", ylim=(-12.5, 8.5))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3,
              columnspacing=1.0, handletextpad=0.4)
    style_axis(ax)


def panel_b(ax: plt.Axes, edges: pd.DataFrame, deletion: pd.DataFrame) -> None:
    x = np.arange(len(edges))
    width = 0.34
    ax.bar(x - width / 2, -edges.I_total, width, color=COLORS["purple"],
           edgecolor=COLORS["ink"], linewidth=0.55)
    ax.set(xticks=x, xticklabels=EDGE_ORDER, xlabel="Block edge",
           ylabel=r"Aggregate RPA influence $-I_e^{\mathrm{RPA}}$", ylim=(0, 10))
    style_axis(ax)

    ax2 = ax.twinx()
    ax2.bar(x + width / 2, deletion.deletion_mean, width, color=COLORS["red"],
            edgecolor=COLORS["ink"], linewidth=0.55)
    ax2.set_ylabel(r"Deletion effect $F_e$")
    ax2.set_ylim(0, 5)
    style_axis(ax2)
    handles = [
        Patch(facecolor=COLORS["purple"], edgecolor=COLORS["ink"],
              label=r"aggregate $-I_e^{\mathrm{RPA}}$"),
        Patch(facecolor=COLORS["red"], edgecolor=COLORS["ink"],
              label=r"deletion $F_e$"),
    ]
    ax.legend(handles=handles, loc="upper right")


def panel_c(ax: plt.Axes) -> None:
    probe = pd.read_csv(RESULTS / "abtab2_layerwise_probe.csv")
    specs = [
        ("S_AA", r"$S_{AA}$", COLORS["blue"]),
        ("S_BB", r"$S_{BB}$", COLORS["orange"]),
        ("S_AB", r"$S_{AB}$", COLORS["teal"]),
        ("q_star", r"$k^*$", COLORS["purple"]),
        ("chiN_star", r"$\chi N^*$", COLORS["red"]),
    ]
    for concept, label, color in specs:
        rows = probe[probe.concept.eq(concept)].sort_values("layer")
        ax.plot(rows.layer, rows.R2_grouped_by_rho, "o-", color=color,
                markerfacecolor="white", markeredgewidth=0.9, linewidth=1.6,
                markersize=4.2, label=label)
    ax.axhline(0, color="#888888", linewidth=0.7)
    ax.set_xlabel("GINE layer")
    ax.set_ylabel(r"PCA--ridge probe $R^2$ (LO-$\rho$)")
    ax.set_xticks(range(6), ["Input", "GINE-1", "GINE-2", "GINE-3", "GINE-4", "GINE-5"])
    ax.set_ylim(min(-1.35, probe.R2_grouped_by_rho.min() - 0.05), 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=5,
              columnspacing=0.55, handlelength=1.3, handletextpad=0.25)
    style_axis(ax)


def panel_d(ax: plt.Axes) -> None:
    summary = json.loads((HISTORY_RESULTS / "summary.json").read_text(encoding="utf-8"))
    rows = [row for row in summary["summary"] if row["layer"] > 0]
    assert len(rows) == 5 and {row["n_pairs"] for row in rows} == {16065}
    values = np.array([row["spearman_mean_embedding_vs_RPA_path"] for row in rows])
    labels = [f"GINE-{row['layer']}" for row in rows]
    y = np.arange(len(rows))
    colors = [COLORS["blue"]] * 4 + [COLORS["red"]]
    bars = ax.barh(y, values, height=0.58, color=colors,
                   edgecolor=COLORS["ink"], linewidth=0.55)
    ax.set(yticks=y, yticklabels=labels, xlabel=r"Path Spearman $\rho_s$",
           ylabel="GINE layer", xlim=(0, 1), ylim=(-0.6, 4.6))
    for bar, value in zip(bars, values):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=8.5)
    style_axis(ax)


def main() -> None:
    configure_style()
    edges, deletion = representative_rows()
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.0))
    panel_a(axes[0, 0], edges)
    panel_b(axes[0, 1], edges, deletion)
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.17, 1.10, f"({label})", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.09, top=0.94,
                        wspace=0.42, hspace=0.42)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"saved {OUTPUT.with_suffix('.png')} and {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
