"""Compose the mpltex topology and violin columns into manuscript Figure 2."""

from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



from paper_style import configure_paper_style, save_dual  # noqa: E402


configure_paper_style()


def trim_white_margin(image: Image.Image, *, padding: int = 90) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"))
    ink = np.any(pixels < 248, axis=2)
    if not ink.any():
        return image
    rows, columns = np.where(ink)
    left = max(0, int(columns.min()) - padding)
    top = max(0, int(rows.min()) - padding)
    right = min(image.width, int(columns.max()) + padding + 1)
    bottom = min(image.height, int(rows.max()) + padding + 1)
    return image.crop((left, top, right, bottom))


def main() -> None:
    source_root = Path(__file__).resolve().parents[2] / "results/figures/figure_acbd"
    topology_path = source_root / "ac_representative_topologies.png"
    violin_path = source_root / "bd_response_distributions.png"

    topology = Image.open(topology_path).convert("RGB")
    violin = Image.open(violin_path).convert("RGB")
    if topology.size != violin.size:
        raise ValueError(f"Source columns must have equal dimensions: {topology.size} != {violin.size}")

    topology = trim_white_margin(topology)
    violin = trim_white_margin(violin)

    fig = plt.figure(figsize=(15.2, 10.2), facecolor="white")
    grid = fig.add_gridspec(1, 2, left=0.018, right=0.995, bottom=0.018, top=0.988,
                            wspace=0.012, width_ratios=(1, 1))
    axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    for axis, panel in zip(axes, (topology, violin)):
        axis.imshow(panel, interpolation="lanczos")
        axis.set_axis_off()

    # Match the supplied reference: bold parenthesized labels outside panel content.
    label_style = dict(fontsize=32, fontweight="bold", ha="left", va="top", color="black")
    fig.text(0.002, 0.998, "(a)", **label_style)
    fig.text(0.500, 0.998, "(b)", **label_style)
    fig.text(0.002, 0.505, "(c)", **label_style)
    fig.text(0.500, 0.505, "(d)", **label_style)

    output = source_root / "figure_acbd_four_panel"
    png, svg = save_dual(fig, output, tight=False)
    fig.savefig(output.with_suffix(".pdf"), dpi=600, facecolor="white")
    plt.close(fig)

    if not png.is_file() or not svg.is_file():
        raise RuntimeError("Figure 2 export failed")


if __name__ == "__main__":
    main()
