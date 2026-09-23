#!/usr/bin/env python3
"""Resume-safe validation-only OFAT search for the publication GNN."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, subprocess, sys, time
from collections import deque
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[4]
SEEDS = (202601, 202602, 202603, 202604, 202605)
REFERENCE = {'hidden_dim': 256, 'learning_rate': 5e-3, 'batch_size': 32, 'num_layers': 5, 'dec_layers': 1}
GRIDS = {
    'batch_size': (16, 32, 64, 128, 256),
    'learning_rate': (1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2),
    'hidden_dim': (16, 32, 64, 128, 256),
    'num_layers': (1, 2, 3, 4, 5),
    'dec_layers': (1, 2, 3, 4),
}


def prepare_split(path: Path):
    train, val = train_test_split(np.arange(100_000), test_size=0.2, random_state=42)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = np.load(path)
        if not (np.array_equal(old['train_index'], train) and np.array_equal(old['val_index'], val)):
            raise RuntimeError('Existing split is not the declared random_state=42 split.')
    else: np.savez(path, train_index=train, val_index=val)
    return {'random_state': 42, 'train_size': len(train), 'validation_size': len(val)}


def build_tasks():
    unique = {}
    for dimension, values in GRIDS.items():
        for value in values:
            c = dict(REFERENCE); c[dimension] = value
            key = '_'.join(f'{k}={c[k]}' for k in sorted(c))
            unique.setdefault(key, {'config': c, 'dimensions': []})['dimensions'].append([dimension, value])
    tasks = []
    for key, record in unique.items():
        for seed in SEEDS:
            c = dict(record['config']); c.update({
                'run_id': f'hyper__{key}__seed{seed}', 'seed': seed, 'dimensions': record['dimensions'],
                'train_data_dir': str(ROOT / 'TrainData/data/chiNlimited/n_train_100k'),
                'test_data_dir': 'SEALED: not available to validation worker', 'epochs': 200, 'minimum_epochs': 120,
                'early_stopping_patience': 30, 'val_fraction': .2, 'split_random_state': 42,
                'optimizer': 'Adam', 'loss': 'MSE on log10(chiN*)', 'graph_builder': 'virtual_node',
                'node_builder': 'free_branch_degree', 'edge_mode': 'full', 'condition_key': 'S01_full_virtualnode_free_branch_degree_full_edge',
                'aliases': ['A1', 'B1', 'C1'], 'group_labels': ['full model hyperparameter search'],
                'condition_label': 'full virtual-node baseline', 'graph_label': 'original bidirectional + virtual node',
                'node_features_label': '[is_free_end, is_branch_point, degree]', 'edge_features_label': '[f, A/B]',
            })
            tasks.append(c)
    return tasks


def summarize(run_root: Path, out: Path):
    rows = [json.loads(p.read_text()) for p in sorted(run_root.glob('*/metrics.json'))]
    grouped = {}
    for row in rows:
        for dimension, value in row['dimensions']:
            grouped.setdefault((dimension, str(value)), []).append(row)
    summary = []
    for (dimension, value), items in grouped.items():
        if len(items) != len(SEEDS): continue
        scores = [float(x['validation_mape_percent']) for x in items]
        summary.append({'dimension': dimension, 'value_label': value, 'n_seeds': len(items),
                        'validation_mape_mean_percent': mean(scores), 'validation_mape_sd_percent': stdev(scores),
                        'validation_mape_per_seed_percent': json.dumps(scores)})
    summary.sort(key=lambda r: (r['dimension'], float(r['validation_mape_mean_percent']), float(r['validation_mape_sd_percent'])))
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'hyperparameter_validation_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ['dimension']); w.writeheader(); w.writerows(summary)
    selected = {d: min((r for r in summary if r['dimension'] == d), key=lambda r: (r['validation_mape_mean_percent'], r['validation_mape_sd_percent'])) for d in GRIDS} if len({r['dimension'] for r in summary}) == len(GRIDS) else {}
    (out / 'selected_by_validation_mape.json').write_text(json.dumps({'selection_metric': 'five-seed mean original-scale validation MAPE', 'selected_values': {k:v['value_label'] for k,v in selected.items()}, 'details': selected, 'final_test_status': 'sealed'}, indent=2))


def discover_existing_workers(output_dir: Path, tasks_by_id: dict):
    """Adopt live validation workers so a scheduler can change capacity safely."""
    adopted = []
    proc_root = Path('/proc')
    for proc_dir in proc_root.iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            raw = (proc_dir / 'cmdline').read_bytes().split(b'\0')
            argv = [x.decode(errors='replace') for x in raw if x]
            if not any(Path(x).name == 'gnn_validation_worker.py' for x in argv):
                continue
            config_path = Path(argv[argv.index('--config-json') + 1]).resolve()
            worker_output = Path(argv[argv.index('--output-dir') + 1]).resolve()
            if config_path.parent != (output_dir / 'configs').resolve():
                continue
            run_id = worker_output.name
            if run_id not in tasks_by_id:
                continue
            env_raw = (proc_dir / 'environ').read_bytes().split(b'\0')
            env = dict(x.decode(errors='replace').split('=', 1) for x in env_raw if b'=' in x)
            gpu = env.get('CUDA_VISIBLE_DEVICES', '').split(',')[0]
            if not gpu:
                continue
            state = (proc_dir / 'stat').read_text().split()[2]
            if state == 'Z':
                continue
            adopted.append({'task': tasks_by_id[run_id], 'gpu': gpu, 'proc': None,
                            'pid': int(proc_dir.name), 'log': None, 'adopted': True})
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return adopted


def adopted_worker_alive(pid: int) -> bool:
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except (FileNotFoundError, PermissionError, IndexError):
        return False


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'runs_100k_fixed80_20_validation_only'); p.add_argument('--gpus', default='0,1,2,3'); p.add_argument('--slots-per-gpu', type=int, default=1); p.add_argument('--max-retries', type=int, default=1); p.add_argument('--prepare-only', action='store_true'); p.add_argument('--adopt-existing', action='store_true'); a = p.parse_args()
    split = prepare_split(a.output_dir / 'fixed_80k_20k_split_indices.npz'); configs, runs, logs = (a.output_dir / n for n in ('configs', 'runs', 'logs'))
    for d in (configs, runs, logs): d.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(); (a.output_dir / 'manifest.json').write_text(json.dumps(tasks, indent=2)); (a.output_dir / 'protocol.json').write_text(json.dumps({'development_data': str(ROOT / 'TrainData/data/chiNlimited/n_train_100k'), 'split': split, 'seeds': SEEDS, 'reference': REFERENCE, 'grids': GRIDS, 'final_test': str(ROOT / 'TrainData/data/chiNlimited/n_test_2k_leakage_free_seed42'), 'final_test_policy': 'not read in this phase'}, indent=2))
    for task in tasks: (configs / f"{task['run_id']}.json").write_text(json.dumps(task, indent=2))
    if a.prepare_only:
        print(f'Prepared {len(tasks)} runs ({len({t["run_id"].rsplit("__seed", 1)[0] for t in tasks})} unique configurations).')
        return
    tasks_by_id = {t['run_id']: t for t in tasks}
    active = discover_existing_workers(a.output_dir, tasks_by_id) if a.adopt_existing else []
    active_ids = {x['task']['run_id'] for x in active}
    pending = deque(t for t in tasks if t['run_id'] not in active_ids and not (runs / t['run_id'] / 'metrics.json').exists()); retries = {}
    if active:
        print(f'adopted={len(active)} existing workers', flush=True)
    slots = [g for g in a.gpus.split(',') for _ in range(a.slots_per_gpu)]
    worker = Path(__file__).resolve().parents[1] / 'gnn_validation_worker.py'
    while pending or active:
        # Treat each repeated entry in ``slots`` as an independent capacity
        # slot.  The previous set-based filter collapsed duplicate GPU IDs,
        # so ``--slots-per-gpu 2`` silently behaved like one worker/GPU.
        used_counts = {g: sum(x['gpu'] == g for x in active) for g in set(slots)}
        free = []
        for g in slots:
            if used_counts.get(g, 0) > 0:
                used_counts[g] -= 1
            else:
                free.append(g)
        while pending and free:
            task, gpu = pending.popleft(), free.pop(0); run = runs / task['run_id']; run.mkdir(exist_ok=True)
            log = (logs / f"{task['run_id']}.log").open('a')
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpu
            cmd = [sys.executable, str(worker), '--config-json', str(configs / f"{task['run_id']}.json"), '--split-indices', str(a.output_dir / 'fixed_80k_20k_split_indices.npz'), '--output-dir', str(run), '--device', 'cuda:0']
            active.append({'task': task, 'gpu': gpu, 'proc': subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), 'log': log})
        time.sleep(3); remaining=[]
        for item in active:
            if item.get('adopted'):
                if adopted_worker_alive(item['pid']): remaining.append(item); continue
                code = 0 if (runs / item['task']['run_id'] / 'metrics.json').exists() else 1
            else:
                code=item['proc'].poll()
                if code is None: remaining.append(item); continue
                item['log'].close()
            rid=item['task']['run_id']
            if code and retries.get(rid, 0) < a.max_retries: retries[rid]=retries.get(rid,0)+1; pending.append(item['task'])
            elif code: raise SystemExit(f'Failed: {rid}; see {logs / (rid + ".log")}')
        active=remaining; print(f'completed={len(tasks)-len(pending)-len(active)}/{len(tasks)} active={len(active)}', flush=True)
    summarize(runs, a.output_dir / 'summary')


if __name__ == '__main__': main()
