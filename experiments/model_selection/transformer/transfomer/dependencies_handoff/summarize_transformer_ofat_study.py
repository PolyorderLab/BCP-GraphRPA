#!/usr/bin/env python3
"""Create auditable final summaries for the systematic Transformer study."""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plotting import compute_metrics

SEEDS = [202601, 202602, 202603, 202604, 202605]
METRICS = ["MAPE", "RMSE", "MAE", "R2", "MaxAE"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return (
        [row["base_name"] for row in rows],
        np.asarray([float(row["true"]) for row in rows]),
        np.asarray([float(row["pred"]) for row in rows]),
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    output = []
    for label, items in grouped.items():
        result = {key: label, "n": len(items)}
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def find_run(root, run_name):
    matches = list(root.glob(f"**/{run_name}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {run_name} under {root}; got {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True)
    parser.add_argument(
        "--gnn-root",
        default=str(
            ROOT / "baselinemodels/outcome/experiment_batches/"
            "polymer_type_strict_100k_leakage_free_20260805/gnn_leakage_free_5seeds"
        ),
    )
    args = parser.parse_args()
    root = Path(args.batch_root)
    out = root / "final_summary"
    out.mkdir(parents=True, exist_ok=True)

    frozen_root = root / "frozen_100k/runs"
    representation_rows = []
    full_predictions = []
    for representation in ("species_fraction_only", "no_shared_junction", "full_topology"):
        for seed in SEEDS:
            run_dir = find_run(frozen_root, f"{representation}__seed{seed}")
            metrics = read_json(run_dir / "metrics.json")
            representation_rows.append(
                {"representation": representation, "seed": seed,
                 **{metric: metrics[metric] for metric in METRICS}}
            )
            if representation == "full_topology":
                full_predictions.append(read_predictions(run_dir / "predictions.csv"))
    write_csv(out / "representation_per_seed.csv", representation_rows)
    representation_summary = summarize(representation_rows, "representation")
    write_csv(out / "representation_summary.csv", representation_summary)

    reference_names, reference_true, _ = full_predictions[0]
    for names, true, _ in full_predictions[1:]:
        if names != reference_names or not np.allclose(true, reference_true):
            raise ValueError("Transformer prediction files are not sample-aligned")
    ensemble_pred = np.mean(np.stack([item[2] for item in full_predictions]), axis=0)
    raw = compute_metrics(reference_true, ensemble_pred)
    ensemble_metrics = {
        "MAPE": float(raw["MAPE"]), "RMSE": float(raw["RMSE"]),
        "MAE": float(raw["MAE"]), "R2": float(raw["R²"]),
        "MaxAE": float(raw["MaxAE"]),
    }
    with (out / "transformer_ensemble_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["base_name", "true", "pred"])
        writer.writerows(zip(reference_names, reference_true, ensemble_pred))
    (out / "transformer_ensemble_metrics.json").write_text(
        json.dumps(ensemble_metrics, indent=2), encoding="utf-8"
    )

    full_by_seed = {
        row["seed"]: row for row in representation_rows
        if row["representation"] == "full_topology"
    }
    cross_rows, shift_rows = [], []
    for seed in SEEDS:
        seed_root = root / f"cross_order_evaluation/seed{seed}"
        manifest = read_json(seed_root / "evaluation_manifest.json")
        grouped = defaultdict(list)
        for result in manifest["results"]:
            grouped[result["order_variant"]].append(result)
        for variant, items in grouped.items():
            cross_rows.append({
                "seed": seed, "order_variant": variant,
                "n_serializations": len(items),
                **{metric: statistics.mean(float(item[metric]) for item in items)
                   for metric in METRICS},
            })
        _, _, canonical = read_predictions(seed_root / "canonical/predictions.csv")
        for variant in ("reversed", "randomized"):
            paths = ([seed_root / "reversed/predictions.csv"] if variant == "reversed"
                     else sorted(seed_root.glob("randomized_seed*/predictions.csv")))
            shifts = []
            for path in paths:
                names, true, pred = read_predictions(path)
                if names != reference_names or not np.allclose(true, reference_true):
                    raise ValueError(f"Cross-order file not aligned: {path}")
                shifts.append(np.abs(pred - canonical) / np.clip(np.abs(canonical), 1e-12, None) * 100)
            shift = np.mean(np.stack(shifts), axis=0)
            shift_rows.append({
                "seed": seed, "order_variant": variant, "n_serializations": len(paths),
                "mean_prediction_shift_percent": float(np.mean(shift)),
                "p95_prediction_shift_percent": float(np.percentile(shift, 95)),
                "max_prediction_shift_percent": float(np.max(shift)),
            })
    write_csv(out / "cross_order_per_seed.csv", cross_rows)
    cross_summary = summarize(cross_rows, "order_variant")
    write_csv(out / "cross_order_summary.csv", cross_summary)
    write_csv(out / "prediction_shift_per_seed.csv", shift_rows)
    shift_summary = []
    for variant in ("reversed", "randomized"):
        items = [row for row in shift_rows if row["order_variant"] == variant]
        result = {"order_variant": variant, "n": len(items)}
        for metric in (
            "mean_prediction_shift_percent", "p95_prediction_shift_percent",
            "max_prediction_shift_percent",
        ):
            values = [row[metric] for row in items]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_sd"] = statistics.stdev(values)
        shift_summary.append(result)
    write_csv(out / "prediction_shift_summary.csv", shift_summary)

    gnn_root = Path(args.gnn_root)
    with (gnn_root / "per_seed_metrics.csv").open(encoding="utf-8") as f:
        gnn_rows_raw = list(csv.DictReader(f))
    comparison_rows = []
    for seed in SEEDS:
        gnn_row = next(row for row in gnn_rows_raw if int(row["seed"]) == seed)
        comparison_rows.append(
            {"model": "GNN", "seed": seed,
             **{metric: float(gnn_row[metric]) for metric in METRICS}}
        )
        comparison_rows.append(
            {"model": "Transformer", "seed": seed,
             **{metric: float(full_by_seed[seed][metric]) for metric in METRICS}}
        )
        gnn_names, gnn_true, _ = read_predictions(gnn_root / f"seed{seed}/predictions.csv")
        if gnn_names != reference_names or not np.allclose(gnn_true, reference_true):
            raise ValueError(f"GNN seed {seed} is not sample/target aligned")
    write_csv(out / "model_comparison_per_seed.csv", comparison_rows)
    comparison_summary = summarize(comparison_rows, "model")
    write_csv(out / "model_comparison_summary.csv", comparison_summary)
    alignment = {
        "aligned": True, "sample_count": len(reference_names),
        "base_name_order_identical": True, "targets_identical": True,
        "selection_set": str(
            (ROOT / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed42").resolve()
        ),
        "selection_set_role": "selection/reporting; not an untouched final test",
        "gnn_root": str(gnn_root.resolve()),
    }
    (out / "gnn_transformer_alignment.json").write_text(
        json.dumps(alignment, indent=2), encoding="utf-8"
    )

    combined = {
        "representation": representation_summary,
        "cross_order": cross_summary,
        "prediction_shift": shift_summary,
        "transformer_ensemble": ensemble_metrics,
        "model_comparison": comparison_summary,
        "alignment": alignment,
    }
    (out / "summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
