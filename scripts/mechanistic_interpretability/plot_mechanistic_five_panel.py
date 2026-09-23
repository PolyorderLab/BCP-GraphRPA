"""Compose the original topology image with four mpltex-styled data panels."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
STYLE_DIR = CODE / 'src/plotting'
sys.path.insert(0, str(STYLE_DIR))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis

configure_paper_style()
import matplotlib.pyplot as plt
import numpy as np

RESULTS = CODE / 'data/mechanistic_interpretability/rpa'
HISTORY_RESULTS = CODE / 'data/mechanistic_interpretability/legal_histories'
OUTPUT = CODE / 'results/figures/mechanistic_interpretability/fig5_mechanistic_interpretability'
TOPOLOGY_IMAGE = CODE / 'data/mechanistic_interpretability/topology.png'


def main():
    source = HERE / 'plot_interpretability_statistics.py'
    spec = importlib.util.spec_from_file_location('original_mechanistic_panels', source)
    plots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plots)
    plots.RESULTS = RESULTS
    plots.HISTORY_RESULTS = HISTORY_RESULTS
    panel_colors = dict(COLORS)
    panel_colors['ink'] = COLORS['black']
    plots.COLORS = panel_colors
    plots.style_axis = style_axis
    edges, deletion = plots.representative_rows()

    fig = plt.figure(figsize=(26, 13.5), facecolor='white')
    grid = fig.add_gridspec(2, 3, width_ratios=(0.82, 1, 1),
                            left=0.055, right=0.965, bottom=0.09, top=0.94,
                            wspace=0.46, hspace=0.42)
    axes = {
        'a': fig.add_subplot(grid[:, 0]),
        'b': fig.add_subplot(grid[0, 1]),
        'c': fig.add_subplot(grid[1, 1]),
        'd': fig.add_subplot(grid[0, 2]),
        'e': fig.add_subplot(grid[1, 2]),
    }
    image = plt.imread(TOPOLOGY_IMAGE)
    rgb = image[..., :3]
    background = (rgb.min(axis=2) > 0.86) & (np.ptp(rgb, axis=2) < 0.08)
    image = image.copy()
    image[background, :3] = 1.0
    occupied = np.any(image[..., :3] < 0.98, axis=2)
    rows, cols = np.where(occupied)
    axes['a'].imshow(image[rows.min():rows.max() + 1, cols.min():cols.max() + 1])
    axes['a'].axis('off')
    plots.panel_a(axes['b'], edges)
    plots.panel_b(axes['c'], edges, deletion)
    plots.panel_c(axes['d'])
    plots.panel_d(axes['e'])
    for line in axes['d'].lines:
        if line.get_marker() not in (None, 'None', ''):
            line.set_linewidth(2.8)
            line.set_markersize(7)
            line.set_markeredgewidth(1.0)
    for text in axes['e'].texts:
        text.set_fontsize(24)
    for label, ax in axes.items():
        box = ax.get_position()
        fig.text(box.x0 - 0.025, box.y1 + 0.012, f'({label})', fontsize=32,
                 fontweight='bold', va='bottom', ha='left')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_dual(fig, OUTPUT, tight=False)
    fig.savefig(OUTPUT.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    print(f'saved {OUTPUT.with_suffix(".png")}, {OUTPUT.with_suffix(".svg")} and {OUTPUT.with_suffix(".pdf")}')


if __name__ == '__main__':
    main()
