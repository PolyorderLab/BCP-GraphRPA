#!/usr/bin/env python3
"""Evaluate the five frozen joint-selected MLPs on independent seed43 data."""
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
SEARCH_ROOT = MLP_ROOT / "hyperparameter_selection_80k"
TEST_DIR = (
    PROJECT
    / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed43_independent"
)
OUTPUT_DIR = HERE / "final_test_seed43_independent"
TEST_CSV = OUTPUT_DIR / "test_descriptor_matrix.csv"
FEATURES = [
    "SLE", "r_deg", "CV_arm", "n_leaves", "M_AB", "IR",
    "D_f", "ISC", "CAF", "tau_mean", "f_A",
]
SEEDS = (202601, 202602, 202603, 202604, 202605)

sys.path.insert(0, str(SEARCH_ROOT))
from mlp_validation_worker import MLP as Network  # noqa: E402

sys.path.insert(0, str(MLP_ROOT / "descriptor_selection_80k"))
from analyze_descriptors import extract_split  # noqa: E402


def metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true = 10.0 ** y_true_log
    y_pred = 10.0 ** y_pred_log
    relative = np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-12, None) * 100.0
    return {
        "test_mape_percent": float(relative.mean()),
        "test_relative_error_p95_percent": float(np.percentile(relative, 95)),
        "test_rmse_log10": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "test_mae_log10": float(mean_absolute_error(y_true_log, y_pred_log)),
        "test_r2_log10": float(r2_score(y_true_log, y_pred_log)),
        "test_rmse_raw": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_mae_raw": float(mean_absolute_error(y_true, y_pred)),
        "test_r2_raw": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    status = json.loads((HERE / "training_status.json").read_text(encoding="utf-8"))
    if status["completed"] != 5 or status["failed"]:
        raise RuntimeError("The five joint-selected checkpoints are not frozen and complete.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_CSV.exists():
        extract_split(TEST_DIR, "test", workers=8, limit=None).to_csv(TEST_CSV, index=False)
    test = pd.read_csv(TEST_CSV)
    if len(test) != 2000 or test["sample_id"].nunique() != 2000:
        raise ValueError("Expected exactly 2,000 unique independent test samples.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    for seed in SEEDS:
        checkpoint = HERE / "checkpoints" / f"joint_selected__seed{seed}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_features = list(payload["features"])
        if checkpoint_features != FEATURES:
            raise ValueError(f"Unexpected feature order in {checkpoint}")

        x = test[FEATURES].to_numpy(np.float32)
        x = (
            x - np.asarray(payload["scaler_mean"], dtype=np.float32)
        ) / np.asarray(payload["scaler_scale"], dtype=np.float32)
        y_true_log = test["log10_chiN_star"].to_numpy(np.float32)

        model = Network(
            len(FEATURES),
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

        rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "n_test": len(test),
                **metrics(y_true_log, y_pred_log),
            }
        )
        np.savez(
            OUTPUT_DIR / f"test_predictions_seed{seed}.npz",
            sample_id=test["sample_id"].astype(str).to_numpy(),
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
        )

    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(OUTPUT_DIR / "joint_selected_seed43_test_metrics.csv", index=False)
    summary = {
        "configuration": {
            "batch_size": 256,
            "lr": 0.0005,
            "hidden_dim": 256,
            "depth": 3,
            "dropout": 0.0,
            "weight_decay": 1e-5,
            "features": FEATURES,
        },
        "test_source_dir": str(TEST_DIR),
        "test_descriptor_matrix": str(TEST_CSV),
        "test_size": 2000,
        "n_seeds": 5,
        "test_mape_mean_percent": float(frame["test_mape_percent"].mean()),
        "test_mape_sd_percent": float(frame["test_mape_percent"].std(ddof=1)),
        "test_mape_per_seed_percent": frame["test_mape_percent"].tolist(),
        "selection_status": (
            "joint configuration and all five checkpoints were frozen before "
            "accessing the independent seed43 test set"
        ),
    }
    (OUTPUT_DIR / "joint_selected_seed43_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
