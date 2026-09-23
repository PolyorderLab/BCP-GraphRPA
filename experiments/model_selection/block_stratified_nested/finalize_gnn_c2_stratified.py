#!/usr/bin/env python3
"""Accept epoch-200 C2 confirmations and build the final stratified summary."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from statistics import mean, median, stdev


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE / "results/gnn"
DIAGNOSTICS = RESULTS / "diagnostics/nonconverged_epoch200"
ARCHIVE = RESULTS / "diagnostics/original_early_stopped"
FINALMODEL = ROOT / "baselinemodels/pub0818/gnn/finalmodel"
CASES = (
    (8, 202602), (8, 202603), (16, 202604),
    (24, 202602), (24, 202604), (24, 202605),
    (32, 202601), (32, 202604), (40, 202602),
    (48, 202602), (48, 202604), (48, 202605),
    (64, 202601), (72, 202605),
)
SEEDS = (202601, 202602, 202603, 202604, 202605)
SEEDS_BY_SIZE = {64: (202601, 202602, 202603, 202604, 202606)}


def mape(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["development_validation_mape_percent"])


def main() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    audit = []
    for size_k, seed in CASES:
        canonical_id = f"C2_nested_train__{size_k}k__seed{seed}"
        confirmation_id = f"{canonical_id}__epoch200_confirmation"
        canonical = RESULTS / "runs" / canonical_id
        confirmation = DIAGNOSTICS / "runs" / confirmation_id
        archived = ARCHIVE / canonical_id
        if not archived.exists():
            shutil.move(str(canonical), str(archived))
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.copytree(confirmation, canonical)
        metric_path = canonical / "metrics.json"
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        metrics["run_id"] = canonical_id
        metrics["replacement_source"] = str(confirmation)
        metrics["replacement_reason"] = "canonical early stopping terminated before convergence; accepted complete 200-epoch confirmation"
        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        config_path = RESULTS / "configs" / f"{canonical_id}.json"
        config = json.loads((DIAGNOSTICS / "configs" / f"{confirmation_id}.json").read_text(encoding="utf-8"))
        config["run_id"] = canonical_id
        config["accepted_epoch200_confirmation"] = True
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        audit.append({
            "size_k": size_k,
            "seed": seed,
            "original_mape": mape(archived / "metrics.json"),
            "replacement_mape": mape(metric_path),
            "original_archive": str(archived),
            "replacement_source": str(confirmation),
        })

    rows = []
    per_run = []
    for size_k in range(8, 81, 8):
        values = []
        for seed in SEEDS_BY_SIZE.get(size_k, SEEDS):
            if size_k == 80:
                metric_path = FINALMODEL / f"C2_nested_train__80k__seed{seed}" / "metrics.json"
                source = "existing C2 finalmodel endpoint"
            else:
                metric_path = RESULTS / "runs" / f"C2_nested_train__{size_k}k__seed{seed}" / "metrics.json"
                source = "block-stratified nested C2 run"
            value = mape(metric_path)
            values.append(value)
            per_run.append({"train_size": size_k * 1000, "train_size_label": f"{size_k}k", "seed": seed, "development_validation_mape_percent": value, "source": source, "metrics_path": str(metric_path)})
        rows.append({
            "train_size": size_k * 1000,
            "train_size_label": f"{size_k}k",
            "common_validation_size": 2000,
            "n_seeds": len(values),
            "development_validation_mape_mean_percent": mean(values),
            "development_validation_mape_sd_percent": stdev(values),
            "development_validation_mape_median_percent": median(values),
            "development_validation_mape_per_seed_percent": json.dumps(values),
            "endpoint_source": "existing C2 finalmodel" if size_k == 80 else "block-stratified nested C2",
        })
    summary = RESULTS / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    for path, payload in (
        (summary / "c2_block_stratified_nested_validation_summary.csv", rows),
        (summary / "c2_block_stratified_nested_validation_per_run.csv", per_run),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
            writer.writeheader()
            writer.writerows(payload)
    (RESULTS / "diagnostics/replacement_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (RESULTS / "COMPLETED.json").write_text(json.dumps({"status": "complete", "nested_training_runs": 45, "existing_finalmodel_endpoint_runs": 5, "epoch200_replacements": len(audit)}, indent=2), encoding="utf-8")
    print(json.dumps({"accepted_replacements": len(audit), "summary": str(summary / "c2_block_stratified_nested_validation_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
