"""Draw M1, M2, and BTABAT architecture schematics in a unified thick-line style."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch


OUT_DIR = Path("/home/zxy/project/TAPD/Demo/outcome/phase_diagrams")
A_COLOR = "#1248ff"
B_COLOR = "#ff7600"
NODE_COLOR = "#ad5100"
LINE_WIDTH = 5.8
NODE_SIZE = 46
LABEL_SIZE = 23
LEGEND_SIZE = 27
TITLE_SIZE = 25
DRAW_SEGMENT_LABELS = False

matplotlib.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
    }
)

M1_FORMULA = r"$\mathrm{M}_1:\mathrm{A}_1(\mathrm{B}_1)\mathrm{T}(\mathrm{B}_3)\mathrm{T}\mathrm{B}_2\mathrm{A}_2$"
M2_FORMULA = r"$\mathrm{M}_2:\mathrm{A}_1(\mathrm{B}_3)\mathrm{T}\mathrm{B}_2\mathrm{A}_2$"
BTABAT_FORMULA = r"$(\mathrm{B}_{\mathrm{T}})\mathrm{AB}(\mathrm{A}_{\mathrm{T}})$"
BTABAT_FULL_FORMULA = (
    r"$\mathrm{M}_3:"
    r"\mathrm{A}_1(\mathrm{B}_1)\mathrm{T}\mathrm{B}_2\mathrm{A}_2"
    r"\mathrm{T}\mathrm{A}_3\mathrm{B}_3$"
)


def bezier(ax, p0, p1, p2, p3, color, label=None, label_t=0.5, label_offset=(0, 0), rotation=None):
    path = MplPath([p0, p1, p2, p3], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    patch = PathPatch(path, facecolor="none", edgecolor=color, lw=LINE_WIDTH, capstyle="round", joinstyle="round")
    ax.add_patch(patch)
    if not DRAW_SEGMENT_LABELS:
        return
    if label:
        x, y = cubic_point(p0, p1, p2, p3, label_t)
        if rotation is None:
            dx, dy = cubic_tangent(p0, p1, p2, p3, label_t)
            rotation = math.degrees(math.atan2(dy, dx))
        ax.text(
            x + label_offset[0],
            y + label_offset[1],
            label,
            color=color,
            fontsize=LABEL_SIZE,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.4},
        )


def cubic_point(p0, p1, p2, p3, t):
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def cubic_tangent(p0, p1, p2, p3, t):
    u = 1 - t
    dx = 3 * u**2 * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t**2 * (p3[0] - p2[0])
    dy = 3 * u**2 * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t**2 * (p3[1] - p2[1])
    return dx, dy


def setup_figure():
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=222)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def setup_axis(ax):
    ax.set_aspect("equal")
    ax.axis("off")


def add_block_legend(
    ax, *, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncols=2,
    fontsize=LEGEND_SIZE,
):
    handles = [
        Line2D([0], [0], color=A_COLOR, lw=LINE_WIDTH, label="A block"),
        Line2D([0], [0], color=B_COLOR, lw=LINE_WIDTH, label="B block"),
    ]
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncols=ncols,
        frameon=False,
        fontsize=fontsize,
        handlelength=2.4,
        columnspacing=3.0,
        handletextpad=0.8,
    )


def finish(fig, ax, path, include_legend=False, legend_kwargs=None):
    if include_legend:
        add_block_legend(ax, **(legend_kwargs or {}))
    ax.margins(0.10)
    fig.savefig(path, dpi=222, transparent=False, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(path.with_suffix(".svg"), transparent=False, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {path}")


def nodes(ax, pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.scatter(xs, ys, s=NODE_SIZE, color=NODE_COLOR, zorder=5, edgecolors="none")


def architecture_title(ax, text, fontsize=TITLE_SIZE):
    ax.text(
        0.5,
        1.03,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontfamily="STIXGeneral",
        color="black",
    )


def figure_title(fig, x, y, text):
    fig.text(
        x,
        y,
        text,
        ha="left",
        va="top",
        fontsize=TITLE_SIZE,
        fontfamily="STIXGeneral",
        color="black",
    )


def draw_m1_architecture(ax, *, show_nodes=True):
    c = (0.0, 0.0)
    b1_end = (-0.26, 0.43)
    b2_end = (1.18, 0.66)
    b3_end = (-1.34, 0.86)
    a2_end = (1.78, 1.05)
    a1_end = (-0.04, -0.76)

    bezier(ax, c, (-0.44, 0.18), (-0.88, 0.52), b3_end, B_COLOR, "B3", 0.55, (0.00, 0.12), -23)
    bezier(ax, c, (-0.08, 0.14), (-0.17, 0.29), b1_end, B_COLOR, "B1", 0.53, (-0.12, 0.04), -60)
    bezier(ax, c, (0.34, 0.21), (0.78, 0.44), b2_end, B_COLOR, "B2", 0.58, (0.02, 0.13), 38)
    bezier(ax, b2_end, (1.40, 0.80), (1.60, 0.94), a2_end, A_COLOR, "A2", 0.50, (0.00, 0.13), 25)
    bezier(ax, c, (0.05, -0.22), (0.04, -0.54), a1_end, A_COLOR, "A1", 0.55, (0.13, 0.00), 83)
    if show_nodes:
        nodes(ax, [c, b1_end, b2_end, b3_end, a2_end, a1_end])


def draw_m1(path: Path):
    fig, ax = setup_figure()
    architecture_title(ax, M1_FORMULA)
    draw_m1_architecture(ax)
    finish(fig, ax, path, include_legend=True)


def draw_m2_architecture(ax, *, show_nodes=True):
    c = (0.0, 0.0)
    b1_end = (-0.95, 0.38)
    b2_end = (0.95, 0.42)
    a2_end = (1.55, 0.88)
    a1_end = (-0.02, -1.00)

    bezier(ax, c, (-0.25, 0.05), (-0.58, 0.22), b1_end, B_COLOR, "B1", 0.50, (-0.02, 0.11), -25)
    bezier(ax, c, (0.28, 0.10), (0.62, 0.25), b2_end, B_COLOR, "B2", 0.54, (0.02, 0.13), 28)
    bezier(ax, b2_end, (1.16, 0.58), (1.38, 0.76), a2_end, A_COLOR, "A2", 0.54, (0.00, 0.13), 32)
    bezier(ax, c, (0.06, -0.30), (0.04, -0.70), a1_end, A_COLOR, "A1", 0.55, (0.13, 0.00), 86)
    if show_nodes:
        nodes(ax, [c, b1_end, b2_end, a2_end, a1_end])


def draw_m2(path: Path):
    fig, ax = setup_figure()
    architecture_title(ax, M2_FORMULA)
    draw_m2_architecture(ax)
    finish(fig, ax, path, include_legend=True)


def draw_m1_m2(path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=222)
    for ax in axes:
        setup_axis(ax)
    architecture_title(axes[0], M1_FORMULA)
    architecture_title(axes[1], M2_FORMULA)
    draw_m1_architecture(axes[0])
    draw_m2_architecture(axes[1])
    handles = [
        Line2D([0], [0], color=A_COLOR, lw=LINE_WIDTH, label="A block"),
        Line2D([0], [0], color=B_COLOR, lw=LINE_WIDTH, label="B block"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.00),
        ncols=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.4,
        columnspacing=3.0,
        handletextpad=0.8,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.23, wspace=0.08)
    fig.savefig(path, dpi=222, transparent=False, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {path}")


def draw_btabat_architecture(ax):
    t_a = (-0.58, 0.40)
    ab = (0.00, 0.05)
    t_b = (0.48, -0.28)
    a1_end = (0.58, -0.92)
    a3_end = (-0.60, 1.30)
    b2_end = (-1.30, 0.22)
    b3_end = (1.85, -0.10)

    bezier(ax, t_b, (0.50, -0.50), (0.54, -0.75), a1_end, A_COLOR, "A1", 0.58, (0.13, 0.00), 82)
    bezier(ax, t_b, (0.28, -0.14), (0.12, -0.02), ab, A_COLOR, "A2", 0.54, (-0.08, -0.11), -54)
    bezier(ax, ab, (-0.16, 0.16), (-0.35, 0.30), t_a, B_COLOR, "B1", 0.54, (0.07, 0.10), 41)
    bezier(ax, t_a, (-0.86, 0.30), (-1.08, 0.22), b2_end, B_COLOR, "B2", 0.54, (0.00, 0.12), 12)
    bezier(ax, t_a, (-0.66, 0.70), (-0.70, 1.06), a3_end, A_COLOR, "A3", 0.50, (-0.13, 0.00), 87)
    bezier(ax, t_b, (0.88, -0.18), (1.34, -0.08), b3_end, B_COLOR, "B3", 0.56, (0.00, 0.13), 6)


def draw_btabat(path: Path):
    fig, ax = setup_figure()
    architecture_title(ax, BTABAT_FORMULA, fontsize=36)
    draw_btabat_architecture(ax)
    finish(
        fig, ax, path, include_legend=True,
        legend_kwargs={
            "loc": "upper right", "bbox_to_anchor": (0.98, 0.93),
            "ncols": 1, "fontsize": 20,
        },
    )


def draw_m1_m2_btabat(path: Path):
    """Draw the three node-free architectures with one legend at far right."""
    fig, axes = plt.subplots(1, 3, figsize=(27.0, 6.5), dpi=222)
    for ax in axes:
        setup_axis(ax)

    draw_m1_architecture(axes[0], show_nodes=False)
    draw_m2_architecture(axes[1], show_nodes=False)
    draw_btabat_architecture(axes[2])
    for ax in axes:
        ax.margins(0.10)

    handles = [
        Line2D([0], [0], color=A_COLOR, lw=LINE_WIDTH, label="A block"),
        Line2D([0], [0], color=B_COLOR, lw=LINE_WIDTH, label="B block"),
    ]
    fig.legend(
        handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.55),
        ncols=1, frameon=False, fontsize=20, handlelength=2.4,
        handletextpad=0.8, labelspacing=0.7,
    )
    fig.subplots_adjust(left=0.015, right=0.86, top=0.82, bottom=0.07, wspace=0.12)
    panel_width = (0.86 - 0.015) / 3.0
    title_centers = [0.015 + panel_width * (index + 0.5) for index in range(3)]
    for center, title in zip(title_centers, [M1_FORMULA, M2_FORMULA, BTABAT_FULL_FORMULA]):
        fig.text(
            center, 0.92, title, ha="center", va="center",
            fontsize=36, fontfamily="STIXGeneral", color="black",
        )
    fig.savefig(path, dpi=222, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(path.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {path}")


def draw_legend(path: Path):
    fig, ax = plt.subplots(figsize=(6.0, 0.9), dpi=222)
    ax.axis("off")
    handles = [
        Line2D([0], [0], color=A_COLOR, lw=LINE_WIDTH, label="A block"),
        Line2D([0], [0], color=B_COLOR, lw=LINE_WIDTH, label="B block"),
    ]
    ax.legend(
        handles=handles,
        loc="center",
        ncols=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handlelength=2.4,
        columnspacing=3.0,
        handletextpad=0.8,
    )
    fig.savefig(path, dpi=222, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_m1(OUT_DIR / "M1_structure.png")
    draw_m2(OUT_DIR / "M2_structure.png")
    draw_m1_m2(OUT_DIR / "M1_M2_structure.png")
    draw_btabat(OUT_DIR / "BTABAT_structure.png")
    draw_m1_m2_btabat(OUT_DIR / "M1_M2_BTABAT_structure.png")
    draw_legend(OUT_DIR / "structure_legend.png")


if __name__ == "__main__":
    main()
