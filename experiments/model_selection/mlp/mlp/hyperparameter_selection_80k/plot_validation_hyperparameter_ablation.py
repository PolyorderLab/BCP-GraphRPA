"""Plot validation-fold hyperparameter ablations for the fixed 80k/20k search."""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

SKILL_SCRIPTS = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from paper_style import COLORS, configure_paper_style, save_dual, style_axis


configure_paper_style()

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "runs_100k_fixed80_20_validation_only"
SUMMARY = RUN_ROOT / "summary" / "hyperparameter_validation_summary.csv"
FIGURES = RUN_ROOT / "figures"

PLOTS = {
    "batch_size": ("01_validation_mape_vs_batch_size", "Batch size", False),
    "lr": ("02_validation_mape_vs_learning_rate", "Learning rate", True),
    "hidden_dim": ("03_validation_mape_vs_hidden_dimension", "Hidden dimension", False),
    "depth": ("04_validation_mape_vs_depth", "Network depth", False),
    "dropout": ("05_validation_mape_vs_dropout", "Dropout", False),
}
SELECTED = {"batch_size": "256", "lr": "0.0005", "hidden_dim": "256", "depth": "3", "dropout": "0.0"}


def _numeric_labels(values):
    return [float(str(v)) for v in values]


def plot_one(frame, dimension, output_name, xlabel, log_x):
    x = _numeric_labels(frame["value_label"])
    y = frame["validation_mape_mean_percent"].to_numpy(float)
    err = frame["validation_mape_sd_percent"].to_numpy(float)
    order = sorted(range(len(x)), key=lambda i: x[i])
    x = [x[i] for i in order]
    y = y[order]
    err = err[order]

    fig, ax = plt.subplots(figsize=(10.5, 8.5), constrained_layout=True)
    ax.errorbar(
        x, y, yerr=err, color=COLORS["blue"], marker="o", markersize=9,
        markerfacecolor=COLORS["blue"], markeredgecolor="#FFFFFF", markeredgewidth=1.0,
        linewidth=2.8, capsize=5, capthick=2.0, label="5-seed mean ± SD",
    )
    selected = float(SELECTED[dimension])
    selected_index = x.index(selected)
    ax.scatter([selected], [y[selected_index]], s=150, color=COLORS["orange"],
               edgecolors=COLORS["black"], linewidths=1.2, zorder=4, label="selected")
    ax.annotate(
        f"selected = {SELECTED[dimension]}", (selected, y[selected_index]),
        xytext=(10, 18), textcoords="offset points", fontsize=20,
        color=COLORS["black"], arrowprops={"arrowstyle": "-", "lw": 1.5, "color": COLORS["black"]},
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation MAPE (%)")
    ax.set_title("Fixed 80k/20k validation hyperparameter selection", pad=16)
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.10, y=0.16)
    style_axis(ax)
    FIGURES.mkdir(parents=True, exist_ok=True)
    save_dual(fig, FIGURES / output_name)
    plt.close(fig)


def main():
    data = pd.read_csv(SUMMARY)
    for dimension, (output_name, xlabel, log_x) in PLOTS.items():
        plot_one(data[data["dimension"] == dimension].copy(), dimension, output_name, xlabel, log_x)


if __name__ == "__main__":
    main()
