#!/usr/bin/env python3
"""Aggregate and draw publication figures for the four C2-to-k* protocols.

All figures are written as 600-dpi PNG plus text-preserving SVG under
``baselinemodels/pub0818/gnn/finalmodel/figure``.  Per-seed figures are grouped
by figure type and transfer mode, while cross-mode summaries and tabular data
are stored in ``summary`` and ``tables``.  By default the script requires five
complete seed43 runs for every requested mode.  The legacy prediction fallback
exists only for validating the plotting pipeline before the fresh seed43
inference is available and must be enabled explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tapd_mplconfig")

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path("/home/zxy/project/TAPD")
TRANSFER_ROOT = PROJECT_ROOT / "baselinemodels/pub0818/gnn/finalmodel/transfer"
FIGURE_ROOT = PROJECT_ROOT / "baselinemodels/pub0818/gnn/finalmodel/figure"
KSTAR_ROOT = FIGURE_ROOT / "kstar_transfer/mainline_seed43"
LOSS_ROOT = KSTAR_ROOT / "loss"
PARITY_ROOT = KSTAR_ROOT / "parity"
SUMMARY_ROOT = KSTAR_ROOT / "summary"
TABLE_ROOT = KSTAR_ROOT / "tables"
STYLE_ROOT = Path("/home/zxy/.codex/skills/mpltex-skill/scripts")
sys.path.insert(0, str(STYLE_ROOT))

from paper_style import (  # noqa: E402
    COLORS,
    configure_paper_style,
    save_dual,
    style_axis,
    style_distribution_axis,
)

configure_paper_style()
import matplotlib.pyplot as plt  # noqa: E402  # must follow paper-style setup

SEEDS = (202601, 202602, 202603, 202604, 202605)
MODES = ("scratch", "frozen_encoder", "finetune_encoder", "two_stage")
MODE_COLORS = {
    "scratch": COLORS["blue"],
    "frozen_encoder": COLORS["purple"],
    "finetune_encoder": COLORS["orange"],
    "two_stage": COLORS["green"],
}
MODE_LABELS = {
    "scratch": "Scratch",
    "frozen_encoder": "Frozen encoder",
    "finetune_encoder": "Fine-tuned encoder",
    "two_stage": "Two-stage",
}
COMPACT_MODE_LABELS = {
    "scratch": "Scratch",
    "frozen_encoder": "Frozen",
    "finetune_encoder": "Fine-tuned",
    "two_stage": "Two-stage",
}
METRICS = ("MAPE", "RMSE", "MAE", "R2", "MaxAE")
METRIC_LABELS = {
    "MAPE": "MAPE (%)",
    "RMSE": "RMSE",
    "MAE": "MAE",
    "R2": r"$R^2$",
    "MaxAE": "MaxAE",
}


def run_dir(mode: str, run_id: int, seed: int) -> Path:
    if mode == "scratch":
        # Seeds 202601--202604 were relaunched after an intentional archival
        # of their interrupted attempts; seed 202605 completed in the initial
        # directory.  These are the only C2-512 random-init artifacts.
        c2_scratch_root = TRANSFER_ROOT / "scratch_c2_512"
        retry = c2_scratch_root / f"scratch_c2_512_retry_run{run_id:02d}_seed{seed}"
        initial = c2_scratch_root / f"scratch_c2_512_run{run_id:02d}_seed{seed}"
        if retry.is_dir():
            return retry
        if initial.is_dir():
            return initial
        raise FileNotFoundError(
            f"Missing completed C2-512 scratch run for seed {seed}: "
            f"checked {retry} and {initial}"
        )
    return TRANSFER_ROOT / mode / f"{mode}_run{run_id:02d}_seed{seed}"


def load_loss(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    json_path = path / "loss_history.json"
    csv_path = path / "loss_history.csv"
    if json_path.is_file():
        rows = json.loads(json_path.read_text(encoding="utf-8"))
    elif csv_path.is_file():
        with csv_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise FileNotFoundError(f"Missing loss history under {path}")
    if not rows:
        raise RuntimeError(f"Empty loss history: {path}")
    epochs = np.asarray([int(row.get("epoch", index + 1)) for index, row in enumerate(rows)])
    train = np.asarray([float(row["train_loss"]) for row in rows])
    validation = np.asarray(
        [float(row.get("val_loss", row.get("validation_loss"))) for row in rows]
    )
    return epochs, train, validation


def prediction_path(path: Path, allow_legacy: bool) -> tuple[Path, bool]:
    seed43 = path / "seed43_predictions.npz"
    if seed43.is_file():
        return seed43, True
    if allow_legacy:
        legacy = path / "test_predictions.npz"
        if legacy.is_file():
            warnings.warn(
                f"Using legacy non-seed43 predictions for plotting-pipeline validation: {legacy}",
                stacklevel=2,
            )
            return legacy, False
    raise FileNotFoundError(f"Missing seed43_predictions.npz under {path}")


def load_predictions(path: Path, allow_legacy: bool):
    source, is_seed43 = prediction_path(path, allow_legacy)
    with np.load(source) as payload:
        y_true = np.asarray(payload["y_true"], dtype=float).reshape(-1)
        y_pred = np.asarray(payload["y_pred"], dtype=float).reshape(-1)
    if y_true.size != y_pred.size or not y_true.size:
        raise RuntimeError(f"Invalid prediction arrays in {source}")
    if np.any(y_true <= 0) or np.any(y_pred <= 0):
        raise RuntimeError(f"Log-log parity requires positive values: {source}")
    return y_true, y_pred, is_seed43


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    absolute = np.abs(error)
    residual = float(np.sum(error**2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAPE": float(np.mean(absolute / np.clip(np.abs(y_true), 1e-12, None)) * 100),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(absolute)),
        "R2": float(1.0 - residual / total) if total > 0 else float("nan"),
        "MaxAE": float(np.max(absolute)),
    }


def available_runs(modes: tuple[str, ...], allow_incomplete: bool):
    records = []
    missing = []
    for mode in modes:
        for run_id, seed in enumerate(SEEDS, start=1):
            path = run_dir(mode, run_id, seed)
            if (path / "loss_history.json").is_file() or (path / "loss_history.csv").is_file():
                records.append((mode, run_id, seed, path))
            else:
                missing.append(str(path))
    if missing and not allow_incomplete:
        raise FileNotFoundError("Incomplete transfer results:\n" + "\n".join(missing))
    return records


def finish_ordinary(fig, axes, stem: Path, *, distribution: bool = False) -> None:
    styler = style_distribution_axis if distribution else style_axis
    for axis in np.atleast_1d(axes).ravel():
        styler(axis)
        if distribution:
            # Five compact categorical panels need readable, non-overlapping
            # protocol names; this is the documented compact-distribution
            # exception to the full-page 24 pt tick convention.
            axis.tick_params(axis="x", which="major", labelsize=16)
            axis.tick_params(axis="y", which="major", labelsize=18)
    save_dual(fig, stem)
    plt.close(fig)


def plot_seed_loss(mode: str, seed: int, path: Path) -> None:
    epochs, train, validation = load_loss(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor="white")
    color = MODE_COLORS[mode]
    for axis, values, title in zip(
        axes, (train, validation), ("Train loss", "Validation loss")
    ):
        axis.plot(epochs, np.clip(values, 1e-12, None), color=color, lw=2.8)
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE loss")
        axis.set_title(title, fontsize=28, pad=12)
        axis.margins(x=0.025)
    fig.suptitle(
        f"{MODE_LABELS[mode]} — seed {seed}", fontsize=30, y=0.985
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.84, wspace=0.30)
    finish_ordinary(
        fig,
        axes,
        LOSS_ROOT / mode / f"kstar_transfer_loss_{mode}_seed{seed}",
    )


def parity_metrics_block(metrics: dict[str, float]) -> str:
    return (
        f'$R^2$ = {metrics["R2"]:.4f}\n'
        f'$MAPE$ = {metrics["MAPE"]:.3f}%\n'
        f'$RMSE$ = {metrics["RMSE"]:.4f}\n'
        f'$MAE$ = {metrics["MAE"]:.4f}'
    )


def plot_seed_parity(
    mode: str, seed: int, path: Path, allow_legacy: bool
) -> tuple[dict[str, float], bool]:
    y_true, y_pred, is_seed43 = load_predictions(path, allow_legacy)
    metrics = compute_metrics(y_true, y_pred)
    relative = np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-12, None) * 100
    mean_error = float(relative.mean())
    p95_error = float(np.percentile(relative, 95))
    upper = max(float(np.percentile(relative, 99.5)), 1e-8)
    lower_limit = 0.86 * min(float(y_true.min()), float(y_pred.min()))
    upper_limit = 1.14 * max(float(y_true.max()), float(y_pred.max()))
    limits = (lower_limit, upper_limit)

    fig, axis = plt.subplots(figsize=(10.5, 10.5), facecolor="white")
    axis.scatter(
        y_true,
        y_pred,
        s=120,
        color=COLORS["parity_blue"],
        alpha=0.60,
        edgecolors="#FFFFFF",
        linewidths=0.7,
        zorder=2,
        rasterized=True,
    )
    axis.plot(limits, limits, "--", color=COLORS["black"], lw=2.8, zorder=3)
    axis.set(
        xscale="log",
        yscale="log",
        xlim=limits,
        ylim=limits,
        xlabel=r"True $k^*R_g$",
        ylabel=r"Predicted $k^*R_g$",
    )
    axis.set_box_aspect(1)
    axis.set_title("Prediction on log scale", fontsize=32, pad=18)
    style_axis(axis)
    axis.text(
        0.035,
        0.955,
        parity_metrics_block(metrics),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=26,
        linespacing=1.18,
    )
    inset = axis.inset_axes([0.60, 0.115, 0.34, 0.285])
    inset.hist(
        relative,
        bins=np.linspace(0, upper, 28),
        density=True,
        color=COLORS["density_green"],
        alpha=0.82,
        edgecolor="none",
    )
    inset.axvline(mean_error, color=COLORS["mean_orange"], ls="--", lw=2.4)
    inset.axvline(p95_error, color="0.35", ls=":", lw=2.4)
    inset.set(
        xlim=(0, upper * 1.04),
        xlabel=r"$|error|$ (%)",
        ylabel="Probability density",
    )
    style_axis(inset)
    inset.xaxis.label.set_size(17)
    inset.yaxis.label.set_size(17)
    inset.tick_params(
        axis="both", which="major", labelsize=13, length=7, width=1.6
    )
    inset.tick_params(
        axis="both", which="minor", labelsize=11, length=3.5, width=1.2
    )
    inset.text(
        0.96,
        0.94,
        f"mean={mean_error:.2f}%\n95th={p95_error:.2f}%",
        transform=inset.transAxes,
        ha="right",
        va="top",
        fontsize=16,
    )
    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.15, top=0.90)
    suffix = "seed43" if is_seed43 else "legacy_test_dryrun"
    save_dual(
        fig,
        PARITY_ROOT
        / mode
        / f"kstar_transfer_parity_{mode}_seed{seed}_{suffix}",
        tight=False,
    )
    plt.close(fig)
    return metrics, is_seed43


def plot_metric_boxplots(
    metric_rows: list[dict[str, object]], modes: tuple[str, ...], is_seed43: bool
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(24, 7.8), facecolor="white")
    for axis, metric in zip(axes, METRICS):
        values = [
            [float(row[metric]) for row in metric_rows if row["mode"] == mode]
            for mode in modes
        ]
        positions = np.arange(1, len(modes) + 1)
        boxes = axis.boxplot(
            values,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showmeans=True,
            boxprops={"edgecolor": "#333333", "linewidth": 0.9},
            medianprops={"color": COLORS["black"], "linewidth": 1.6},
            whiskerprops={"color": "#333333", "linewidth": 1.0},
            capprops={"color": "#333333", "linewidth": 1.0},
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": COLORS["black"],
                "markersize": 5.5,
            },
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": "white",
                "markeredgecolor": COLORS["black"],
            },
        )
        for patch, mode in zip(boxes["boxes"], modes):
            patch.set_facecolor(MODE_COLORS[mode])
            patch.set_alpha(0.42)
        for position, mode, samples in zip(positions, modes, values):
            offsets = np.linspace(-0.10, 0.10, len(samples))
            axis.scatter(
                position + offsets,
                samples,
                s=38,
                color=MODE_COLORS[mode],
                edgecolors="white",
                linewidths=0.55,
                alpha=0.90,
                zorder=3,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [COMPACT_MODE_LABELS[mode] for mode in modes], rotation=20, ha="right"
        )
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.set_xlim(0.55, len(modes) + 0.45)
        values_flat = np.concatenate([np.asarray(item, dtype=float) for item in values])
        spread = max(float(np.ptp(values_flat)), abs(float(values_flat.mean())) * 0.04, 1e-8)
        axis.set_ylim(float(values_flat.min()) - 0.14 * spread, float(values_flat.max()) + 0.14 * spread)
        axis.set_title(METRIC_LABELS[metric], fontsize=27, pad=12)
    # The abbreviated x-axis labels already encode the colors and methods, so
    # a second large legend would only consume publication space.
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.25, top=0.91, wspace=0.55)
    suffix = "seed43" if is_seed43 else "legacy_test_dryrun"
    finish_ordinary(
        fig,
        axes,
        SUMMARY_ROOT / f"kstar_transfer_{suffix}_five_metric_boxplots",
        distribution=True,
    )


def stack_curves(records, mode: str, column: int):
    curves = []
    for record_mode, _, _, path in records:
        if record_mode != mode:
            continue
        curves.append(load_loss(path)[column])
    if not curves:
        return None
    length = min(map(len, curves))
    return np.vstack([curve[:length] for curve in curves])


def plot_mean_sd_loss(records, modes: tuple[str, ...]) -> None:
    # This is the direct protocol comparison: frozen encoder is retained in
    # the per-seed panels and metric summary, while the convergence panel
    # contrasts the baseline with the two trainable-transfer protocols.
    comparison_modes = tuple(
        mode for mode in ("scratch", "finetune_encoder", "two_stage") if mode in modes
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor="white")
    for mode in comparison_modes:
        for axis, column in zip(axes, (1, 2)):
            data = stack_curves(records, mode, column)
            if data is None:
                continue
            epochs = np.arange(1, data.shape[1] + 1)
            # Loss spans orders of magnitude, so aggregate in log10 space as in
            # the source main-line convergence figure. ddof=1 is the requested
            # sample SD (the historical script used population SD).
            log_data = np.log10(np.clip(data, 1e-12, None))
            log_mean = log_data.mean(axis=0)
            log_sample_sd = (
                log_data.std(axis=0, ddof=1)
                if data.shape[0] > 1
                else np.zeros_like(log_mean)
            )
            mean = 10**log_mean
            lower = 10 ** (log_mean - log_sample_sd)
            upper = 10 ** (log_mean + log_sample_sd)
            axis.plot(
                epochs,
                np.clip(mean, 1e-12, None),
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
                lw=2.8,
            )
            axis.fill_between(
                epochs, lower, upper, color=MODE_COLORS[mode], alpha=0.16, linewidth=0
            )
    for axis_index, (axis, title) in enumerate(
        zip(axes, ("Train loss", "Validation loss"))
    ):
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE loss")
        axis.set_title(title, fontsize=28, pad=12)
        axis.set_xlim(-2, 202)
        axis.axvline(50, color="0.30", linestyle="--", linewidth=2.2, zorder=1)
        if axis_index == 0:
            axis.text(
                52,
                0.955,
                "Encoder unfreezing /\njoint fine-tuning",
                transform=axis.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=15,
                linespacing=1.05,
            )
    axes[1].legend(loc="upper right", fontsize=18, handlelength=2.4)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.88, wspace=0.30)
    finish_ordinary(
        fig,
        axes,
        SUMMARY_ROOT / "kstar_transfer_five_seed_loss_mean_sample_sd",
    )


def write_metric_summary(rows: list[dict[str, object]], is_seed43: bool) -> None:
    if not is_seed43:
        return
    path = TABLE_ROOT / "kstar_transfer_seed43_per_seed_metrics.csv"
    fieldnames = ["mode", "run_id", "seed", *METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in rows])


def remove_dryrun_figures_after_final_seed43() -> None:
    """Remove only this script's explicitly named legacy validation artifacts."""
    for pattern in (
        "kstar_transfer_*_legacy_test_dryrun.png",
        "kstar_transfer_*_legacy_test_dryrun.svg",
        "kstar_transfer_legacy_test_dryrun_five_metric_boxplots.png",
        "kstar_transfer_legacy_test_dryrun_five_metric_boxplots.svg",
    ):
        for path in PARITY_ROOT.glob(f"*/{pattern}"):
            path.unlink()
        for path in SUMMARY_ROOT.glob(pattern):
            path.unlink()


def main(args) -> None:
    modes = tuple(args.modes)
    for directory in (SUMMARY_ROOT, TABLE_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        (LOSS_ROOT / mode).mkdir(parents=True, exist_ok=True)
        (PARITY_ROOT / mode).mkdir(parents=True, exist_ok=True)
    records = available_runs(modes, args.allow_incomplete)
    if not records:
        raise RuntimeError("No completed transfer runs found")

    metric_rows = []
    all_seed43 = True
    for mode, run_id, seed, path in records:
        plot_seed_loss(mode, seed, path)
        try:
            metrics, is_seed43 = plot_seed_parity(
                mode, seed, path, args.allow_legacy_predictions
            )
        except FileNotFoundError:
            if not args.allow_incomplete:
                raise
            warnings.warn(f"Skipping missing parity inputs under {path}", stacklevel=2)
            continue
        all_seed43 &= is_seed43
        metric_rows.append(
            {"mode": mode, "run_id": run_id, "seed": seed, **metrics}
        )

    plot_mean_sd_loss(records, modes)
    expected_metrics = len(modes) * len(SEEDS)
    if len(metric_rows) == expected_metrics:
        plot_metric_boxplots(metric_rows, modes, all_seed43)
        write_metric_summary(metric_rows, all_seed43)
        if all_seed43:
            remove_dryrun_figures_after_final_seed43()
    elif not args.allow_incomplete:
        raise RuntimeError(
            f"Expected {expected_metrics} metric rows, found {len(metric_rows)}"
        )
    print(
        f"Created figures for {len(records)} loss histories and {len(metric_rows)} parity runs "
        f"in {FIGURE_ROOT}; seed43_only={all_seed43}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--allow-legacy-predictions",
        action="store_true",
        help="Dry-run only: fall back to legacy test_predictions.npz.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
