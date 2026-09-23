#!/usr/bin/env python3
"""Train one GNN run on the fixed development split; never reads final test."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mainmodel.scripts import train_single_chin_supplementary_confirmation as source
from src.data_utils import load_and_process_dataset
from src.plotting import compute_metrics


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval(); pred_log, true_log = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_log.append(model(batch).detach().cpu().numpy().reshape(-1))
            true_log.append(batch.y.detach().cpu().numpy().reshape(-1))
    pred_log, true_log = np.concatenate(pred_log), np.concatenate(true_log)
    true, pred = 10 ** true_log, 10 ** pred_log
    return pred_log, true_log, compute_metrics(true, pred)


def load_cached_development_data(data_dir: str, cache_dir: Path):
    """Parse one development pool once and share its size-specific cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(data_dir).resolve().name
    cache = cache_dir / f'{dataset_name}_chiN_data.pt'
    lock = cache_dir / f'{dataset_name}_chiN_data.lock'
    while True:
        if cache.exists():
            return torch.load(cache, map_location='cpu', weights_only=False)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(2)
            continue
        try:
            dataset = source.load_and_process_dataset(data_dir, task_type='chiN')
            tmp = cache.with_suffix('.tmp.pt')
            torch.save(dataset, tmp)
            os.replace(tmp, cache)
            return dataset
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-json', type=Path, required=True)
    parser.add_argument('--split-indices', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = json.loads(args.config_json.read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg['seed']))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    split = np.load(args.split_indices)
    cache_dir = Path(cfg.get('dataset_cache_dir', args.split_indices.parent / 'dataset_cache'))
    source_data = load_cached_development_data(cfg['train_data_dir'], cache_dir)
    train_data = source.build_wrapped_dataset(Subset(source_data, split['train_index']), cfg)
    val_data = source.build_wrapped_dataset(Subset(source_data, split['val_index']), cfg)
    generator = torch.Generator(); generator.manual_seed(int(cfg['seed']))
    train_loader = DataLoader(train_data, batch_size=int(cfg['batch_size']), shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=int(cfg['batch_size']), shuffle=False)
    model = source.build_model(cfg, train_data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['learning_rate']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    best, best_epoch, stale, history = float('inf'), 0, 0, []
    minimum_epochs = max(120, int(cfg.get('minimum_epochs', 120)))
    if int(cfg['epochs']) < minimum_epochs:
        raise ValueError('epochs must be at least minimum_epochs (120 or greater)')
    checkpoint = args.output_dir / 'best_model.pth'
    started = time.time()
    for epoch in range(1, int(cfg['epochs']) + 1):
        model.train(); train_total = 0.0
        for batch in train_loader:
            batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch.y); loss.backward(); optimizer.step()
            train_total += float(loss.item()) * batch.num_graphs
        model.eval(); val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device); loss = criterion(model(batch), batch.y)
                val_total += float(loss.item()) * batch.num_graphs
        train_loss, val_loss = train_total / len(train_data), val_total / len(val_data)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        scheduler.step(val_loss)
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0; torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if epoch >= minimum_epochs and stale >= int(cfg.get('early_stopping_patience', 30)):
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    pred_log, true_log, metrics = evaluate(model, val_loader, device)
    np.savez_compressed(args.output_dir / 'validation_predictions.npz', y_pred_log=pred_log, y_true_log=true_log)
    (args.output_dir / 'loss_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    result = {**cfg, 'minimum_epochs': minimum_epochs, 'best_epoch': best_epoch, 'epochs_trained': len(history), 'best_validation_mse_log10': best,
              'validation_mape_percent': float(metrics['MAPE']), 'validation_rmse': float(metrics['RMSE']),
              'validation_mae': float(metrics['MAE']), 'validation_r2': float(metrics['R²']),
              'elapsed_seconds': time.time() - started, 'final_test_accessed': False}
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'run_id': cfg['run_id'], 'validation_mape_percent': result['validation_mape_percent'], 'best_epoch': result['best_epoch']}), flush=True)


if __name__ == '__main__':
    main()
