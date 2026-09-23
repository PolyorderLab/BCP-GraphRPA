"""Check the supplied conversion function without loading unavailable graph dependencies."""
import ast
import math
from pathlib import Path

import numpy as np

tree = ast.parse(Path(__file__).with_name('data_utils.py').read_text(encoding='utf-8-sig'))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'period_label_to_dimensionless_wavevector')
namespace = {'np': np, 'math': math}
exec(compile(ast.Module(body=[function], type_ignores=[]), 'data_utils.py', 'exec'), namespace)
convert = namespace[function.name]
assert np.allclose(convert([2 * math.pi, math.pi]), [1, 2])
for invalid in (0, -1):
    try:
        convert(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonpositive period was accepted')
print('Label conversion checks passed; this is not a full loader test.')
