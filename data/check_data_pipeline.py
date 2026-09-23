"""Run small sampling and loader checks using the bundled example data."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_utils import load_and_process_dataset
import numpy as np

def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / 'data/splits' / script), *map(str, args)],
                   check=True, capture_output=True, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})

with tempfile.TemporaryDirectory() as temporary:
    base = Path(temporary)
    run('unified_sampling.py', '--source-dir', ROOT / 'data/example/n_train_1k',
        '--output-dir', base, '--train-per-block', 3, '--pool-per-block', 2)
    assert len(list((base / 'development').glob('*.mat'))) == 12
    for seed in (42, 43):
        extra = ['--exclude-dir', base / 'seed42'] if seed == 43 else []
        run('build_leakage_free_test_2k.py', '--train-dir', base / 'development',
            '--pool-dir', base / 'n_test_pool', '--output-dir', base / f'seed{seed}',
            '--seed', seed, '--per-block', 1, *extra)
    assert not ({p.name for p in (base / 'seed42').glob('*.mat')} &
                {p.name for p in (base / 'seed43').glob('*.mat')})
    period = load_and_process_dataset(str(base / 'seed43'), task_type='k')
    wave = load_and_process_dataset(str(base / 'seed43'), task_type='k_wavevector')
    assert len(period) == len(wave) == 4
    for p, w in zip(period, wave):
        assert np.isclose(float(p.y.item() + w.y.item()), np.log10(2*np.pi), atol=1e-6)
print('Pool selection, exclusion and real MAT/YAML loading passed.')
