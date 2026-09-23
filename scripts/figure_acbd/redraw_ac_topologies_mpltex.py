#!/usr/bin/env python3
"""Redraw the Figure 2 a/c topology panels in the TAPD paper style."""

from __future__ import annotations

from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA = ROOT / "data/figure_acbd/architecture/train"
OOD_DATA = ROOT / "data/figure_acbd/architecture/ood"
OUT_STEM = ROOT / "results/figures/figure_acbd/ac_representative_topologies"


from paper_style import (  # noqa: E402
    TOPOLOGY_COLORS,
    configure_paper_style,
    draw_topology,
    save_dual,
)


# Preserve the six training topologies used in the existing a panel.
TRAIN_SAMPLES = [
    "3blocks_sample_27020",
    "3blocks_sample_01839",
    "4blocks_sample_19583",
    "5blocks_sample_12582",
    "5blocks_sample_00369",
    "5blocks_sample_04529",
]

# Preserve the five representative OOD samples selected for the existing c panel.
OOD_SAMPLES = [
    "6blocks_sample_00137",
    "7blocks_sample_00203",
    "8blocks_sample_00050",
    "9blocks_sample_00005",
    "10blocks_sample_00286",
]


def load_blocks(directory: Path, stem: str) -> list[dict]:
    with (directory / f"{stem}.yml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["system"]["components"][0]["blocks"]


def graph_layout(blocks: list[dict]) -> dict[str, tuple[float, float]]:
    graph = nx.Graph()
    graph.add_edges_from(tuple(block["ends"]) for block in blocks)
    raw = nx.kamada_kawai_layout(graph, weight=None)
    nodes = sorted(raw)
    points = np.asarray([raw[node] for node in nodes], dtype=float)
    points -= points.mean(axis=0)
    _, _, vh = np.linalg.svd(points, full_matrices=False)
    points = points @ vh.T
    if np.ptp(points[:, 0]) < np.ptp(points[:, 1]):
        points = points[:, ::-1]
    points[:, 0] /= max(np.max(np.abs(points[:, 0])), 1e-9)
    points[:, 1] /= max(np.max(np.abs(points[:, 1])), 1e-9)
    points[:, 1] *= 0.72
    return {node: tuple(points[index]) for index, node in enumerate(nodes)}


def reference_a_layout(blocks: list[dict]) -> dict[str, tuple[float, float]]:
    """Reproduce the deterministic tree layout used by the original a images."""
    adjacency: dict[str, list[str]] = {}
    for block in blocks:
        n1, n2 = block["ends"]
        adjacency.setdefault(n1, []).append(n2)
        adjacency.setdefault(n2, []).append(n1)
    root = max(sorted(adjacency), key=lambda node: len(adjacency[node]))
    positions: dict[str, tuple[float, float]] = {root: (0.0, 0.0)}

    def walk(node: str, parent: str | None, start: float, end: float, depth: int) -> None:
        children = [child for child in adjacency[node] if child != parent]
        if not children:
            return
        if len(children) == 1:
            angles = [(start + end) / 2.0]
        else:
            step = (end - start) / max(1, len(children) - 1)
            angles = [start + index * step for index in range(len(children))]
        radius = 1.05 + 0.18 * depth
        x0, y0 = positions[node]
        for child, angle in zip(sorted(children), angles):
            positions[child] = (x0 + radius * math.cos(angle),
                                y0 + radius * math.sin(angle))
            spread = max(math.pi / 5.0,
                         (end - start) / max(2.4, len(children) + 1.0))
            walk(child, node, angle - spread, angle + spread, depth + 1)

    degree = len(adjacency[root])
    initial_angles = [0.0] if degree <= 1 else [
        2.0 * math.pi * index / degree for index in range(degree)
    ]
    for neighbor, angle in zip(sorted(adjacency[root]), initial_angles):
        positions[neighbor] = (math.cos(angle), math.sin(angle))
        walk(neighbor, root, angle - math.pi / 4.0, angle + math.pi / 4.0, 1)

    points = np.asarray(list(positions.values()), dtype=float)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    points -= center
    scale = max(np.max(np.abs(points[:, 0])), np.max(np.abs(points[:, 1])) / 0.72, 1e-9)
    points /= scale
    return {node: tuple(points[index]) for index, node in enumerate(positions)}


def draw_chain(
    ax,
    blocks: list[dict],
    positions: dict[str, tuple[float, float]],
    *,
    reference_a: bool = False,
) -> None:
    # Small alternating bends reproduce the smooth, hand-drawn reference while
    # retaining the exact graph connectivity and A/B identity of every block.
    edges = []
    for index, block in enumerate(blocks):
        curvature = 0.16 if reference_a else 0.075 * (-1.0 if index % 2 else 1.0)
        edges.append((*block["ends"], str(block["segment"]), curvature))

    collection_start = len(ax.collections)
    patch_start = len(ax.patches)
    draw_topology(ax, positions, edges)

    # The supplied reference is line-only: retain rounded paths, omit all nodes,
    # and use the requested 5.8 pt line rather than the helper's standalone scale.
    for patch in ax.patches[patch_start:]:
        patch.set_linewidth(5.8)
    for collection in list(ax.collections[collection_start:]):
        collection.remove()

    ax.set(xlim=(-1.18, 1.18), ylim=(-0.88, 0.88), aspect="equal")
    ax.axis("off")


def add_grid_axes(fig, panel_bottom: float) -> list:
    panel_left, panel_width, panel_height = 0.10, 0.865, 0.427
    gap_x, gap_y = 0.018, 0.035
    cell_width = (panel_width - 2 * gap_x) / 3
    cell_height = (panel_height - gap_y) / 2
    axes = []
    for index in range(6):
        row, col = divmod(index, 3)
        left = panel_left + col * (cell_width + gap_x)
        bottom = panel_bottom + (1 - row) * (cell_height + gap_y)
        axes.append(fig.add_axes([left, bottom, cell_width, cell_height]))
    return axes


def main() -> None:
    configure_paper_style()
    # Exactly match the current b/d canvas: 12 x 16.2 in at 600 dpi = 7200 x 9720 px.
    fig = plt.figure(figsize=(12.0, 16.2), facecolor="white")
    axes_a = add_grid_axes(fig, panel_bottom=0.553)
    axes_c = add_grid_axes(fig, panel_bottom=0.100)

    for ax, stem in zip(axes_a, TRAIN_SAMPLES):
        blocks = load_blocks(TRAIN_DATA, stem)
        draw_chain(ax, blocks, reference_a_layout(blocks), reference_a=True)

    for ax, stem in zip(axes_c[:5], OOD_SAMPLES):
        blocks = load_blocks(OOD_DATA, stem)
        draw_chain(ax, blocks, graph_layout(blocks))

    # The sixth c-panel cell is intentionally reserved for the sole A/B legend.
    legend_ax = axes_c[5]
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color=TOPOLOGY_COLORS["A"], lw=5.8,
               solid_capstyle="round", label="A block"),
        Line2D([0], [0], color=TOPOLOGY_COLORS["B"], lw=5.8,
               solid_capstyle="round", label="B block"),
    ]
    legend_ax.legend(handles=handles, loc="lower right", frameon=False,
                     fontsize=24, handlelength=2.3, handletextpad=0.7,
                     bbox_to_anchor=(1.0, 0.20), borderaxespad=0.0,
                     labelspacing=0.75)

    save_dual(fig, OUT_STEM, tight=False)
    plt.close(fig)
    print(f"Saved {OUT_STEM}.png/.svg")


if __name__ == "__main__":
    main()
