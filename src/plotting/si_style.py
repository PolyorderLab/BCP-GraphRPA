"""Supplementary-figure typography and export settings."""
from pathlib import Path
import logging
import matplotlib as mpl
from matplotlib.text import Text
from paper_style import save_dual as _save_dual

WIDTHS = {'in_domain_all_seed_parity': 120, 'in_domain_signed_relative_error': 120, 'gnn_training_histories': 118, 'mlp_training_histories': 118, 'final_transformer_training_histories': 118, 'ood_all_seed_parity': 128.1257, 'abtab2_grid_robustness': 145, 'kstar_transfer_n100_all_seed_parity': 130.7462, 'kstar_transfer_all_seed_parity': 130.7485, 'kstar_transfer_signed_relative_error': 130.7429, 'gnn_hyperparameter_screening': 130.7429, 'gnn_representation_ablation': 130.7429, 'gnn_training_size': 75, 'descriptor_screening_a': 130.7429, 'descriptor_screening_b': 130.7429, 'descriptor_screening_c': 130.7429, 'descriptor_selection_signals': 87, 'mlp_hyperparameter_screening': 100, 'mlp_training_size': 62, 'descriptor_ablation': 85, 'transformer_hyperparameter_screening': 130.7429, 'transformer_training_size': 75}


class _FontTimestampFilter(logging.Filter):
    """Ignore known zero timestamps in Matplotlib's bundled Computer Modern fonts."""

    def filter(self, record):
        return record.getMessage() not in {
            "'created' timestamp seems very low; regarding as unix timestamp",
            "'modified' timestamp seems very low; regarding as unix timestamp",
        }


_font_timestamp_filter = _FontTimestampFilter()

def configure_si_math():
    font_logger = logging.getLogger('fontTools.ttLib.tables._h_e_a_d')
    if _font_timestamp_filter not in font_logger.filters:
        font_logger.addFilter(_font_timestamp_filter)
    mpl.rcParams.update({'font.family': 'Arial', 'mathtext.fontset': 'custom',
                         'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
                         'mathtext.bf': 'Arial:bold', 'text.usetex': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def compact_training_style(fig):
    """Scale the original 10.5-inch style to the native 75-mm notebook canvas."""
    from matplotlib.lines import Line2D
    from matplotlib.collections import Collection, PathCollection
    from matplotlib.patches import Patch
    from matplotlib.axes import Axes
    from matplotlib.legend import Legend
    from matplotlib.ticker import MaxNLocator
    scale = fig.get_figwidth() / 10.5
    axes = fig.findobj(Axes)
    # Materialize tick artists before scaling, including inset ticks.
    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        for axis in (ax.xaxis, ax.yaxis):
            axis.get_major_ticks()
            axis.get_minor_ticks()
            axis.labelpad *= scale
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick.set_pad(tick.get_pad() * scale)
    for artist in fig.findobj():
        if isinstance(artist, Text):
            artist.set_fontsize(artist.get_fontsize() * scale)
        elif isinstance(artist, Line2D):
            artist.set_linewidth(artist.get_linewidth() * scale)
            artist.set_markersize(artist.get_markersize() * scale)
            artist.set_markeredgewidth(artist.get_markeredgewidth() * scale)
        elif isinstance(artist, Collection):
            artist.set_linewidth(artist.get_linewidth() * scale)
            if isinstance(artist, PathCollection):
                artist.set_sizes(artist.get_sizes() * scale**2)
        elif isinstance(artist, Patch):
            artist.set_linewidth(artist.get_linewidth() * scale)
    for legend in fig.findobj(Legend):
        legend.axes.legend(handles=legend.legend_handles,
                           labels=[t.get_text() for t in legend.get_texts()],
                           fontsize=legend.get_texts()[0].get_fontsize(),
                           loc='upper right', bbox_to_anchor=(.99, .99),
                           handlelength=2.6, frameon=False)

def style_trend_grid(fig, axes, row_labels, *, target, output_stem):
    """One pair of axis titles for the six SI prediction/error grids.

    Rows retain their identities and scales. Per-panel numerical summaries
    belong in the tables; zero/mean/percentile guide lines remain as trends.
    """
    import numpy as np
    axes = np.asarray(axes)
    stem = Path(output_stem).stem
    scale = WIDTHS[stem] / 25.4 / fig.get_figwidth()
    # S11 uses the same final title size as the S19 panel subtitles.  Its
    # centered PDF export enlarges text by about 1.5%, so 6.0646 pt here
    # yields the measured 6.153 pt target in the manuscript artwork.
    title_pt = 6.0646 if stem == 'kstar_transfer_signed_relative_error' else 6
    tick_size, title_size, label_size = 5.5 / scale, title_pt / scale, 6.5 / scale
    if fig.legends:
        grid_top = .865
    elif stem == 'kstar_transfer_signed_relative_error':
        # Leave enough top clearance for the enlarged seed headings.
        grid_top = .920
    else:
        grid_top = .925
    fig.subplots_adjust(left=.145, right=.985, bottom=.09, top=grid_top,
                        wspace=.20, hspace=.22)
    row_titles = []
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(which='major', labelsize=tick_size, pad=2 / scale,
                           length=2.3 / scale, width=.55 / scale)
            ax.tick_params(which='minor', length=1.2 / scale, width=.4 / scale)
            ax.tick_params(axis='y', labelleft=(col == 0))
            for spine in ax.spines.values():
                spine.set_linewidth(.55 / scale)
            if row == 0:
                ax.set_title(f'Seed {202601 + col}', fontsize=title_size, pad=5 / scale)
        box = axes[row, 0].get_position()
        row_titles.append(fig.text(.073, (box.y0 + box.y1) / 2, row_labels[row],
                 rotation=90, ha='right', va='center', fontsize=title_size))
    if target == 'error':
        xlabel, ylabel = 'Relative error (%)', 'Probability density'
    else:
        symbol = r'$(\chi N)^{\ast}$' if target == 'chi' else r'$k^{\ast}$'
        xlabel, ylabel = 'RPA ' + symbol, 'Predicted ' + symbol
    shared_x = fig.supxlabel(xlabel, x=.565, y=.012, fontsize=label_size, va='top')
    shared_y = fig.supylabel(ylabel, x=.025, y=(.09 + grid_top) / 2,
                            fontsize=label_size, ha='right')
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(title_size)
        fig._suptitle.set_y(.995)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(title_size)
    # Anchor shared titles to actual tick-label bounds, not distant canvas edges.
    # Gaps are measured at the final SI insertion size (points).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap = 2.0 / scale * fig.dpi / 72
    inverse = fig.transFigure.inverted()
    xlabels = [t for ax in axes[-1, :] for t in
               [*ax.get_xticklabels(), ax.xaxis.get_offset_text()]
               if t.get_visible() and t.get_text()]
    ylabels = [t for ax in axes[:, 0] for t in ax.get_yticklabels()
               if t.get_visible() and t.get_text()]
    bottom = min(t.get_window_extent(renderer).y0 for t in xlabels)
    shared_x.set_y(inverse.transform((0, bottom - gap))[1])
    left = min(t.get_window_extent(renderer).x0 for t in ylabels)
    for text in row_titles:
        text.set_x(inverse.transform((left - gap, 0))[0])
    row_left = min(t.get_window_extent(renderer).x0 for t in row_titles)
    shared_y.set_x(inverse.transform((row_left - gap, 0))[0])

def save_dual(fig, output_stem, *, tight=True):
    stem = Path(output_stem)
    if stem.stem in WIDTHS:
        for text in fig.findobj(Text):
            text.set_text(text.get_text().replace('^*', r'^{\ast}').replace(r'\%', '%'))
    result = _save_dual(fig, output_stem, tight=tight)
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white',
                **({'bbox_inches': 'tight', 'pad_inches': .03} if tight else {}))
    return result

# Preserve the approved outer canvas; Arial glyph bounds may differ slightly.
REFERENCE_ASPECT = {'in_domain_all_seed_parity': 0.5654166666666667, 'in_domain_signed_relative_error': 0.5633333333333334, 'gnn_training_histories': 0.4125, 'mlp_training_histories': 0.36041666666666666, 'final_transformer_training_histories': 0.36125, 'ood_all_seed_parity': 0.5933333333333334, 'abtab2_grid_robustness': 0.4429166666666667, 'kstar_transfer_n100_all_seed_parity': 1.0776829815895823, 'kstar_transfer_all_seed_parity': 1.0554089709762533, 'kstar_transfer_signed_relative_error': 0.935, 'gnn_hyperparameter_screening': 0.6729166666666667, 'gnn_representation_ablation': 0.36875, 'gnn_training_size': 0.8091666666666667, 'descriptor_screening_a': 0.9, 'descriptor_screening_b': 0.6791666666666667, 'descriptor_screening_c': 0.5733333333333334, 'descriptor_selection_signals': 0.88125, 'mlp_hyperparameter_screening': 0.6729166666666667, 'mlp_training_size': 0.8095833333333333, 'descriptor_ablation': 0.96, 'transformer_hyperparameter_screening': 0.73875, 'transformer_training_size': 0.8095833333333333}

# Limited-label signed errors use the full-data error-grid layout.
WIDTHS['kstar_transfer_n100_signed_relative_error'] = WIDTHS['kstar_transfer_signed_relative_error']
REFERENCE_ASPECT['kstar_transfer_n100_signed_relative_error'] = REFERENCE_ASPECT['kstar_transfer_signed_relative_error']
