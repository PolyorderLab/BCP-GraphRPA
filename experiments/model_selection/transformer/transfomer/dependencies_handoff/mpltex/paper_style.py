"""Fixed publication style for TAPD figures; requires mpltex + Matplotlib."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# Keep these semantic names stable across all paper figures.
COLORS = {
    "orange": "#E87817",
    "teal": "#18B89C",
    "blue": "#4C92C3",
    "green": "#58B85D",
    "purple": "#9C7ACC",
    "red": "#DF5A5A",
    "amber": "#FF9A3D",
    "black": "#222222",
    # Fixed semantic colors for single-series parity plots and their
    # relative-error probability-density inset.
    "parity_blue": "#0F4D92",
    "density_green": "#1F7A4A",
    "mean_orange": "#C95F00",
}

# Palette and geometry for all topology/molecular-architecture schematics.
TOPOLOGY_COLORS = {"A": "#1045FF", "B": "#FF7900", "junction": "#A94700"}
TOPOLOGY_LINEWIDTH = 7.0
TOPOLOGY_NODE_SIZE = 120
TOPOLOGY_TITLE_SIZE = 42
TOPOLOGY_LEGEND_SIZE = 40


def configure_paper_style() -> None:
    """Load mpltex and apply the non-negotiable paper-wide rcParams."""
    # mpltex is intentionally an explicit dependency of every plot using this skill.
    import mpltex  # noqa: F401

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.linewidth": 2.6,
        # mpltex or a previously active style may hide the top/right spines.
        # Publication axes in TAPD always require an explicit four-sided frame.
        "axes.spines.left": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "axes.labelsize": 30,
        "axes.labelcolor": "#000000",
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 12,
        "ytick.major.size": 12,
        "xtick.major.width": 2.6,
        "ytick.major.width": 2.6,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 6,
        "ytick.minor.size": 6,
        "xtick.minor.width": 2.0,
        "ytick.minor.width": 2.0,
        "lines.linewidth": 2.8,
        "lines.markersize": 7,
        "legend.fontsize": 24,
        "legend.frameon": False,
        "svg.fonttype": "none",
    })


def style_axis(ax) -> None:
    """Style one main or inset axis, including log axes after their scale is set."""
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(2.6)
    ax.tick_params(which="major", direction="in", top=True, right=True,
                   length=12, width=2.6, labelsize=24, colors="#000000")
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=6, width=2.0, colors="#000000")


def style_distribution_axis(ax) -> None:
    """Use the lighter, sparse-tick frame for compact violin/distribution plots."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.6)
    ax.tick_params(which="major", direction="in", top=True, right=True,
                   length=8, width=1.6, colors="#000000")
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=4, width=1.0, colors="#000000")
    # Categorical positions are integer-spaced; 0.5 gives exactly one minor
    # tick between neighbours. Linear y axes likewise get one midpoint tick.
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def save_dual(fig, output_stem: str | Path, *, tight: bool = True) -> tuple[Path, Path]:
    """Save publication PNG (600 dpi) and text-preserving SVG.

    Use ``tight=False`` for square parity plots so their exported canvas stays square.
    """
    for index, ax in enumerate(fig.axes):
        # Axis-free topology schematics are the only intentional exception.
        if not ax.axison:
            continue
        hidden = [name for name, spine in ax.spines.items() if not spine.get_visible()]
        if hidden:
            raise RuntimeError(
                f"Axis {index} is missing required frame spines: {', '.join(hidden)}. "
                "Call style_axis(ax) (or style_distribution_axis(ax)) before save_dual()."
            )
    stem = Path(output_stem)
    if stem.suffix.lower() in {".png", ".svg"}:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    save_kwargs = {"facecolor": "white"}
    if tight:
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.03})
    fig.savefig(png, dpi=600, **save_kwargs)
    fig.savefig(svg, **save_kwargs)
    return png, svg


def new_topology_figure():
    """Create the fixed blank canvas for one centered topology schematic."""
    configure_paper_style()
    fig, ax = plt.subplots(figsize=(9.6, 7.47), facecolor="white")
    ax.set(xlim=(-1.15, 1.15), ylim=(-0.85, 0.85), aspect="equal")
    ax.axis("off")
    # Keep the title and legend bands unoccupied by the topology itself.
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.17, top=0.84)
    return fig, ax


def draw_topology(ax, positions: dict, edges) -> None:
    """Draw A/B block edges and junction nodes.

    Each edge is `(start, end, block_type, curvature)`; curvature may be 0.
    `positions` maps node identifiers to `(x, y)` tuples.
    """
    for start, end, block_type, curvature in edges:
        if block_type not in {"A", "B"}:
            raise ValueError(f"Unknown block type: {block_type!r}; use 'A' or 'B'.")
        patch = FancyArrowPatch(
            positions[start], positions[end], arrowstyle="-",
            connectionstyle=f"arc3,rad={curvature}",
            linewidth=TOPOLOGY_LINEWIDTH, color=TOPOLOGY_COLORS[block_type],
            capstyle="round", joinstyle="round", shrinkA=0, shrinkB=0, zorder=1,
        )
        ax.add_patch(patch)
    xs, ys = zip(*positions.values())
    ax.scatter(xs, ys, s=TOPOLOGY_NODE_SIZE, color=TOPOLOGY_COLORS["junction"],
               edgecolors="none", zorder=2)


def add_topology_title(fig, title: str) -> None:
    """Add one centered STIX topology formula/title in the reserved upper band."""
    fig.text(0.5, 0.93, title, ha="center", va="center", fontsize=TOPOLOGY_TITLE_SIZE,
             color="#000000")


def add_topology_legend(fig) -> None:
    """Add the fixed centered A/B block legend in the reserved lower band."""
    handles = [
        Line2D([0], [0], color=TOPOLOGY_COLORS["A"], lw=TOPOLOGY_LINEWIDTH,
               solid_capstyle="butt", label="A block"),
        Line2D([0], [0], color=TOPOLOGY_COLORS["B"], lw=TOPOLOGY_LINEWIDTH,
               solid_capstyle="butt", label="B block"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=2,
               frameon=False, fontsize=TOPOLOGY_LEGEND_SIZE, handlelength=2.5,
               handletextpad=0.55, columnspacing=2.4, alignment="center")
