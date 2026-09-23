#!/usr/bin/env python3
"""Evaluate frozen training-pool checkpoints on one common 2k set.

The 2k set is used here as a shared evaluation set for the pre-specified
training-pool-size ablation.  After this analysis it must not be described as
an untouched final test set; final model reporting requires a new sealed set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HERE = Path(__file__).resolve().parent
MLP_ROOT = HERE.parent
PROJECT = HERE.parents[3]
CHECKPOINT_DIR = HERE / "checkpoints"
TEST_SOURCE = PROJECT / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed42"
TEST_CSV = MLP_ROOT / "final_test_evaluation/test_descriptor_matrix.csv"
OUTPUT_DIR = HERE / "training_pool_common_2k_evaluation"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
POOL_SIZES_K = tuple(range(10, 101, 10))
SEEDS = (202601, 202602, 202603, 202604, 202605)

sys.path.insert(0, str(HERE))
from validation_ablation_worker import MLP  # noqa: E402


def calculate_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true = 10.0 ** y_true_log
    y_pred = 10.0 ** y_pred_log
    relative = np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-12, None) * 100.0
    return {
        "evaluation_mape_percent": float(relative.mean()),
        "evaluation_relative_error_p95_percent": float(np.percentile(relative, 95)),
        "evaluation_rmse_log10": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "evaluation_mae_log10": float(mean_absolute_error(y_true_log, y_pred_log)),
        "evaluation_r2_log10": float(r2_score(y_true_log, y_pred_log)),
        "evaluation_rmse_raw": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "evaluation_mae_raw": float(mean_absolute_error(y_true, y_pred)),
        "evaluation_r2_raw": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Missing descriptor matrix {TEST_CSV}; generate it from {TEST_SOURCE} first."
        )
    test = pd.read_csv(TEST_CSV)
    if len(test) != 2000 or test["sample_id"].nunique() != 2000:
        raise ValueError("The common evaluation matrix must contain 2,000 unique samples.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []

    for pool_size_k in POOL_SIZES_K:
        for seed in SEEDS:
            experiment_id = f"train_pool__{pool_size_k}k__seed{seed}"
            checkpoint_path = CHECKPOINT_DIR / f"{experiment_id}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            features = list(payload["features"])
            x = test[features].to_numpy(np.float32)
            x = (
                x - np.asarray(payload["scaler_mean"], dtype=np.float32)
            ) / np.asarray(payload["scaler_scale"], dtype=np.float32)
            y_true_log = test["log10_chiN_star"].to_numpy(np.float32)

            model = MLP(
                len(features),
                int(payload["hidden_dim"]),
                int(payload["depth"]),
                float(payload["dropout"]),
            )
            model.load_state_dict(payload["model_state_dict"])
            model.to(device).eval()
            with torch.no_grad():
                y_pred_log = (
                    model(torch.from_numpy(x).to(device)).cpu().numpy().ravel()
                )

            metrics = calculate_metrics(y_true_log, y_pred_log)
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "pool_size_k": pool_size_k,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path),
                    "evaluation_source": str(TEST_SOURCE),
                    "evaluation_size": len(test),
                    **metrics,
                }
            )
            np.savez(
                PREDICTION_DIR / f"{experiment_id}.npz",
                sample_id=test["sample_id"].astype(str).to_numpy(),
                y_true_log=y_true_log,
                y_pred_log=y_pred_log,
            )

    per_seed = pd.DataFrame(rows).sort_values(["pool_size_k", "seed"])
    if len(per_seed) != len(POOL_SIZES_K) * len(SEEDS):
        raise RuntimeError("Incomplete common-set evaluation.")
    summary = (
        per_seed.groupby("pool_size_k", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            evaluation_mape_mean_percent=("evaluation_mape_percent", "mean"),
            evaluation_mape_sd_percent=("evaluation_mape_percent", "std"),
        )
        .sort_values("pool_size_k")
    )
    per_seed.to_csv(OUTPUT_DIR / "training_pool_common_2k_per_seed.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "training_pool_common_2k_summary.csv", index=False)
    protocol = {
        "purpose": "common-set evaluation for the pre-specified training-pool-size ablation",
        "evaluation_source": str(TEST_SOURCE),
        "evaluation_descriptor_matrix": str(TEST_CSV),
        "evaluation_size": 2000,
        "pool_sizes_k": list(POOL_SIZES_K),
        "seeds": list(SEEDS),
        "checkpoint_policy": "existing validation-selected checkpoints; no retraining",
        "important_status": (
            "This 2k set is no longer an untouched final test after this analysis. "
            "A new sealed test set is required for final model reporting."
        ),
    }
    (OUTPUT_DIR / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
