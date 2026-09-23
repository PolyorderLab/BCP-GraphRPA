"""Reconstructed frozen-MLP OOD inference from the archived evaluation protocol."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/model_selection/mlp/mlp/descriptor_selection_80k'))
from src.models.mlp_validation_worker import MLP
from analyze_descriptors import _record_from_yml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/block_extrapolation_rerun/mlp')
    parser.add_argument('--seed', type=int, choices=range(202601, 202606))
    args = parser.parse_args()
    paths = sorted(args.test_dir.glob('*blocks_sample_*.yml'))
    counts = [int(p.stem.split('blocks_')[0]) for p in paths]
    if len(paths) != 2500 or any(counts.count(n) != 500 for n in range(6, 11)):
        raise ValueError('Expected 500 samples for each block count 6--10')
    frame = pd.DataFrame([_record_from_yml(str(p)) for p in paths])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / 'descriptor_matrix.csv', index=False)
    rows = []
    for seed in ([args.seed] if args.seed else range(202601, 202606)):
        checkpoint = ROOT / f'checkpoints/mlp/seed{seed}/model.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        scale = np.asarray(payload['scaler_scale'], dtype=np.float32)
        if not np.all(np.isfinite(scale) & (scale > 0)):
            raise ValueError('Invalid checkpoint scaler')
        x = (frame[payload['features']].to_numpy(np.float32) - np.asarray(payload['scaler_mean'], np.float32)) / scale
        if not np.isfinite(x).all():
            raise ValueError('Non-finite OOD descriptors')
        model = MLP(len(payload['features']), payload['hidden_dim'], payload['depth'], payload['dropout'])
        model.load_state_dict(payload['model_state_dict'])
        model.eval()
        with torch.no_grad():
            pred_log = model(torch.from_numpy(x)).numpy().ravel()
        true_log = frame.log10_chiN_star.to_numpy(np.float32)
        true, pred = 10.0**true_log, 10.0**pred_log
        error = pred - true
        metrics = dict(seed=seed, MAPE=float(np.mean(np.abs(error) / true) * 100),
                       RMSE=float(np.sqrt(np.mean(error**2))), MAE=float(np.mean(np.abs(error))),
                       R2=float(1 - np.sum(error**2) / np.sum((true - true.mean())**2)))
        np.savez_compressed(args.output_dir / f'seed{seed}_ood_predictions.npz',
                            sample_id=frame.sample_id.to_numpy(dtype=str), block_count=counts,
                            y_true=true, y_pred=pred, y_true_log=true_log, y_pred_log=pred_log)
        rows.append(metrics)
        print(json.dumps(metrics))
    pd.DataFrame(rows).to_csv(args.output_dir / 'per_seed_overall_metrics.csv', index=False)


if __name__ == '__main__':
    main()
