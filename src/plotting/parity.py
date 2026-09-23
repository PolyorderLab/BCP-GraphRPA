"""
Nature-style parity plot utilities for regression models.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['legend.frameon'] = False


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    max_ae = np.max(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    return {"R²": r2, "RMSE": rmse, "MAE": mae, "MaxAE": max_ae, "MAPE": mape}


def save_pub_fig(fig, filename, dpi=300):
    """Save publication-quality figure as both SVG and PNG."""
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")


def plot_parity(
    y_true,
    y_pred,
    save_path,
    title="",
    xlabel=r"True $\chi N^*$",
    ylabel=r"Predicted $\chi N^*$",
    lims=None,
    show_stats=True,
    stats_metrics=None,
    show_maxae=False,
    figsize=(5, 4.5),
    fontsize=11,
    color=None,
):
    """
    Create a Nature-style parity plot.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Path to save figure (without extension)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        lims: Tuple of (xmin, xmax) for axis limits, auto-computed if None
        show_stats: Whether to show metrics text box
        stats_metrics: Dict of metrics to display, computed if None
        show_maxae: Whether to include MaxAE in stats
        figsize: Figure size
        fontsize: Font size for labels
        color: Scatter plot color, defaults to PALETTE["blue_main"]
    """
    if lims is None:
        lims = [0, max(y_true.max(), y_pred.max()) * 1.05]

    if stats_metrics is None:
        stats_metrics = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if color is None:
        color = PALETTE["blue_main"]
    ax.scatter(
        y_true, y_pred,
        c=color,
        alpha=0.6,
        edgecolors='white',
        linewidths=0.5,
        s=40,
        rasterized=True
    )

    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_aspect('equal', adjustable='box')

    if title:
        ax.set_title(title, fontsize=fontsize, pad=10)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    if show_stats:
        stats_lines = [
            f'$R^2$ = {stats_metrics["R²"]:.4f}',
            f'$MAPE$ = {stats_metrics["MAPE"]:.2f}%',
            f'$RMSE$ = {stats_metrics["RMSE"]:.2f}',
        ]
        if show_maxae:
            stats_lines.append(f'MaxAE = {stats_metrics["MaxAE"]:.2f}')

        stats_text = '\n'.join(stats_lines)
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9)
        )

    ax.tick_params(width=1.2, labelsize=10)

    plt.tight_layout()

    save_pub_fig(fig, save_path, dpi=300)
    plt.close()

    return stats_metrics


def plot_parity_log(
    y_true,
    y_pred,
    save_path,
    title="",
    xlabel=r"True $\chi N^*$",
    ylabel=r"Predicted $\chi N^*$",
    fontsize=11,
    xlim_min=None,
    usetex=False,
    show_stats=True,
    stats_metrics=None,
    figsize=(5, 4.5),
    lims=None,
    color=None,
):
    """
    Create a log-scale parity plot (for data spanning multiple orders of magnitude).
    If usetex=True, use LaTeX for text rendering.
    """
    if usetex:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
    else:
        plt.rcParams['text.usetex'] = False

    if stats_metrics is None:
        stats_metrics = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if color is None:
        color = PALETTE["blue_main"]
    ax.scatter(
        y_true, y_pred,
        c=color,
        alpha=0.6,
        edgecolors='white',
        linewidths=0.5,
        s=40,
        rasterized=True
    )

    if lims is None:
        min_val = xlim_min if xlim_min is not None else min(y_true.min(), y_pred.min()) * 0.8
        max_val = max(y_true.max(), y_pred.max()) * 1.2
        lims = [min_val, max_val]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7, label='y = x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if title:
        ax.set_title(title, fontsize=fontsize, pad=10)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    if show_stats:
        stats_lines = [
            f'$R^2$ = {stats_metrics["R²"]:.4f}',
            f'$MAPE$ = {stats_metrics["MAPE"]:.2f}%',
            f'$RMSE$ = {stats_metrics["RMSE"]:.2f}',
        ]
        stats_text = '\n'.join(stats_lines)
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9)
        )

    ax.tick_params(width=1.2, labelsize=10)

    plt.tight_layout()

    save_pub_fig(fig, save_path, dpi=300)
    plt.close()

    plt.rcParams['text.usetex'] = False

    return stats_metrics
