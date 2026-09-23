"""Evaluate the frozen validation-selected MLP checkpoints on the sealed 2k test set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
SEARCH = HERE.parent / "hyperparameter_selection_80k"
RUN_ROOT = SEARCH / "runs_100k_fixed80_20_validation_only"
PROJECT = HERE.parents[3]
TEST_DIR = PROJECT / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed42"
TEST_CSV = HERE / "test_descriptor_matrix.csv"
OUT_CSV = HERE / "selected_checkpoint_test_metrics.csv"
OUT_JSON = HERE / "selected_checkpoint_test_summary.json"
FEATURES = ["SLE", "r_deg", "CV_arm", "n_leaves", "M_AB", "IR", "D_f", "ISC", "CAF", "tau_mean", "f_A"]
SEEDS = (202601, 202602, 202603, 202604, 202605)

sys.path.insert(0, str(SEARCH))
from mlp_validation_worker import MLP  # noqa: E402
sys.path.insert(0, str(SEARCH.parent / "descriptor_selection_80k"))
from analyze_descriptors import extract_split  # noqa: E402


def metrics(y_true_log, y_pred_log):
    y_true, y_pred = 10.0**y_true_log, 10.0**y_pred_log
    rel = np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-12, None) * 100.0
    return {
        "test_mape_percent": float(rel.mean()),
        "test_relative_error_p95_percent": float(np.percentile(rel, 95)),
        "test_rmse_log10": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "test_mae_log10": float(mean_absolute_error(y_true_log, y_pred_log)),
        "test_r2_log10": float(r2_score(y_true_log, y_pred_log)),
        "test_rmse_raw": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_mae_raw": float(mean_absolute_error(y_true, y_pred)),
        "test_r2_raw": float(r2_score(y_true, y_pred)),
    }


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    if not TEST_CSV.exists():
        test = extract_split(TEST_DIR, "test", workers=8, limit=None)
        test.to_csv(TEST_CSV, index=False)
    test = pd.read_csv(TEST_CSV, usecols=[*FEATURES, "log10_chiN_star"])
    if len(test) != 2000:
        raise ValueError(f"Expected 2,000 final-test rows, found {len(test)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in SEEDS:
        checkpoint = RUN_ROOT / "checkpoints" / f"hyper__batch_size__256__seed{seed}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
        scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float32)
        x = ((test[FEATURES].to_numpy(np.float32) - scaler_mean) / scaler_scale).astype(np.float32)
        y = test["log10_chiN_star"].to_numpy(np.float32)
        model = MLP(len(FEATURES), payload["hidden_dim"], payload["depth"], payload["dropout"])
        model.load_state_dict(payload["model_state_dict"])
        model.to(device).eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy().ravel()
        row = {"seed": seed, "checkpoint": str(checkpoint), "n_test": len(test), **metrics(y, pred)}
        rows.append(row)
        np.savez(HERE / f"seed{seed}_test_predictions.npz", y_true_log=y, y_pred_log=pred)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False)
    summary = {
        "test_source_dir": str(TEST_DIR),
        "test_descriptor_matrix": str(TEST_CSV),
        "test_size": len(test),
        "selection_status": "hyperparameters frozen from fixed 20k validation fold before this evaluation",
        "hyperparameters": {"batch_size": 256, "lr": 0.0005, "hidden_dim": 256, "depth": 3, "dropout": 0.0},
        "n_seeds": len(rows),
        "test_mape_mean_percent": float(frame["test_mape_percent"].mean()),
        "test_mape_sd_percent": float(frame["test_mape_percent"].std(ddof=1)),
        "test_mape_per_seed_percent": frame["test_mape_percent"].tolist(),
        "per_seed_metrics_csv": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
