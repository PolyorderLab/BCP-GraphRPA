"""Check the actual worker stopping branches without loading Torch or training models."""
import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
WORKERS = [
    ROOT / 'src/models/gnn_validation_worker.py',
    ROOT / 'experiments/model_selection/gnn/gnn/gnn_validation_worker.py',
    ROOT / 'experiments/model_selection/gnn/gnn/training_size_c2_nested_common_val_seed42/gnn_external_validation_worker.py',
    ROOT / 'experiments/training_size/gnn/gnn/training_size_c2_nested_common_val_seed42/gnn_external_validation_worker.py',
]

for path in WORKERS:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    index = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'minimum_epochs' for t in n.targets))
    setup = compile(ast.Module(body=main.body[index:index + 2], type_ignores=[]), str(path), 'exec')
    branch = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                  and n.test.left.id in ('val_loss', 'validation_loss'))
    # Run the worker's checkpoint/stopping branch inside a small synthetic loss sequence.
    loop = ast.parse('for epoch, val_loss in enumerate(losses, 1):\n validation_loss = val_loss\n').body[0]
    loop.body.append(branch)
    code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(path), 'exec')
    for cfg, losses, expected_stop, expected_best in [
        ({'epochs': 200}, [2.] * 9 + [1.] * 191, 120, 10),
        ({'epochs': 200, 'minimum_epochs': 0}, [1.] * 200, 120, 1),
        ({'epochs': 200}, [2.] * 118 + [1.] * 82, 149, 119),
        ({'epochs': 200, 'minimum_epochs': 150}, [1.] * 200, 150, 1),
    ]:
        saved = []
        ns = dict(cfg=cfg, losses=losses, best=float('inf'), best_epoch=0, stale=0,
                  checkpoint='unused', model=SimpleNamespace(state_dict=lambda: {}),
                  torch=SimpleNamespace(save=lambda *_: saved.append(True)))
        exec(setup, ns)
        exec(code, ns)
        assert (ns['epoch'], ns['best_epoch']) == (expected_stop, expected_best), path
        assert saved
    for cfg in ({'epochs': 119}, {'epochs': 120, 'minimum_epochs': 150}):
        try:
            exec(setup, {'cfg': cfg})
        except ValueError:
            pass
        else:
            raise AssertionError(f'{path}: accepted an epoch budget below the minimum')
print('PASS: four GNN workers enforce the epoch floor, patience and all-epoch checkpoint selection.')
