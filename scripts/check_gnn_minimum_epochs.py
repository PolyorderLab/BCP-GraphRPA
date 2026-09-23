"""Check the actual nested-size worker expressions without importing PyTorch."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
workers = list((root/'experiments').rglob('gnn_external_validation_worker.py'))
assert len(workers) == 2
assert workers[0].read_bytes() == workers[1].read_bytes()
for path in workers:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'minimum_epochs' for t in n.targets))
    stop = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and 'stale' in ast.unparse(n.test) and 'minimum_epochs' in ast.unparse(n.test))
    budget = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and any(isinstance(b, ast.Raise) for b in n.body) and 'minimum_epochs' in ast.unparse(n.test))
    for cfg, expected in [({},120),({'minimum_epochs':0},120),({'minimum_epochs':120},120),({'minimum_epochs':150},150)]:
        ns={'cfg':cfg}
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(path),'exec'),ns)
        assert ns['minimum_epochs']==expected
        for epoch in [1,119,expected-1,expected,200]:
            ns.update(epoch=epoch,stale=30)
            assert eval(compile(ast.Expression(stop.test),str(path),'eval'),ns)==(epoch>=expected)
        cfg['epochs']=119
        assert eval(compile(ast.Expression(budget.test),str(path),'eval'),ns)
    launcher=path.with_name('run_nested_training_size.py').read_text(encoding='utf-8')
    assert '"minimum_epochs": 120' in launcher
print('Both launchers/workers enforce >=120 epochs; early-stop and budget checks passed.')
