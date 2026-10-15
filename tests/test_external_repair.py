import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark_mysr.external.solvers import dsr_expression, gplearn_expression
from benchmark_mysr.external.units import input_units


def test_deep_native_program_exports_without_recursion():
    function = SimpleNamespace(name='sin', arity=1)
    p = SimpleNamespace(program=[function] * 3000 + [0])
    expression = gplearn_expression(p)
    assert expression == 'sin(' * 3000 + 'x0' + ')' * 3000


def test_prefix_export_preserves_argument_order_and_full_float():
    sub = SimpleNamespace(name='sub', arity=2)
    p = SimpleNamespace(program=[sub, 1, sub, 0, 0.12345678912345678])
    assert gplearn_expression(p) == 'sub(x1,sub(x0,0.12345678912345678))'
    with pytest.raises(ValueError, match='incomplete'):
        gplearn_expression(SimpleNamespace(program=[sub, 0]))


def test_dsr_string_expression_is_normalized_before_variable_conversion():
    import sympy as sp
    p = SimpleNamespace(sympy_expr='Add(x1,Mul(2,x2))')
    got = sp.sympify(dsr_expression(p))
    assert got == sp.Symbol('x0') + 2 * sp.Symbol('x1')


def test_dsr_export_does_not_simplify_invalid_candidates_or_call_pretty_printer():
    class Native:
        traversal = [SimpleNamespace(name='log', arity=1),
                     SimpleNamespace(name='const', arity=0, input_var=None, value=0.0)]
        @property
        def sympy_expr(self):
            raise ZeroDivisionError('upstream pretty printing failed')
    assert dsr_expression(Native()) == 'log(0.0)'


def test_constant_fit_penalty_handles_huge_finite_and_nonfinite_residuals():
    from benchmark_mysr.external.solvers import constant_fit_residual
    from scipy.optimize import least_squares
    huge = np.array([1e200, -1e200, np.inf, -np.inf, np.nan, 4.0])
    r = constant_fit_residual(huge, 2.0)
    assert r.tolist() == [1e10, -1e10, 1e10, -1e10, 1e10, 2.0]
    assert np.isfinite(r @ r)
    # Exercise the real trust-region/SVD path, not just sanitization output.
    fit = least_squares(lambda c: constant_fit_residual(
        np.array([1e200 * c[0], -1e200 * c[0], c[0] - 2]), 1.0),
        np.ones(1), max_nfev=200)
    assert np.isfinite(fit.x).all() and np.isfinite(fit.fun).all()


def test_public_units_skip_constants_and_follow_dummy_column_positions(tmp_path):
    info = {'task': {'properties': ['V, F', 'C, F', 'V, F', 'V, F'],
                     'si-derived_units': ['$J$', '$kg$', '$rad$', '$m$'],
                     'sympy_eq_str': 'DO_NOT_USE_REFERENCE_EQUATION'}}
    published = tmp_path/'raw/srsd-published-data/easy'
    dummy = tmp_path/'raw/srsd-dummy-variable-data/easy'
    published.mkdir(parents=True)
    (published/'supp_info.json').write_text(json.dumps(info))
    for split in ['train', 'val', 'test']:
        for root in [published, dummy]:
            (root/split).mkdir(parents=True, exist_ok=True)
        source = np.array([[1, 2, 9], [3, 4, 8], [5, 6, 7]])
        augmented = np.column_stack((source[:, 1], np.array([20, 21, 22]), source[:, 0], source[:, 2]))
        np.savetxt(published/split/'task.txt', source)
        np.savetxt(dummy/split/'task.txt', augmented)
    assert input_units(str(tmp_path), 'srsd-feynman-easy-task') == ['$rad$', '$m$']
    assert input_units(str(tmp_path), 'srsd-dummy-variable-easy-task') == ['$m$', None, '$rad$']
    augmented[:, 0] = source[:, 0]
    np.savetxt(dummy/'val/task.txt', augmented)
    input_units.cache_clear()
    with pytest.raises(ValueError, match='differs across splits'):
        input_units(str(tmp_path), 'srsd-dummy-variable-easy-task')


def test_failed_test_prediction_retains_validation_selected_expression(tmp_path, monkeypatch):
    import sys
    import benchmark_mysr.external.worker as worker
    import benchmark_mysr.external.solvers as solvers
    for split, x in [('train', 1), ('validation', 1), ('test', 0)]:
        (tmp_path/f'{split}.csv').write_text(f'row_id,x_0,target,target_clean\na,{x},1,1\n')
    request = {'method': 'gplearn', 'data': str(tmp_path), 'source_root': str(tmp_path)}
    path = tmp_path/'request.json'
    path.write_text(json.dumps(request))
    def predict(X):
        return np.where(X[:, 0] == 0, np.inf, 1)
    monkeypatch.setattr(solvers, 'prepare', lambda *a: None)
    monkeypatch.setattr(solvers, 'fit', lambda *a: ([('chosen', 1, predict)], {}))
    monkeypatch.setattr(sys, 'argv', ['worker', str(path)])
    monkeypatch.chdir(tmp_path)
    assert worker.main() == 1
    result = json.loads((tmp_path/'worker-result.json').read_text())
    assert result['selected_expression'] == 'chosen'
    assert result['validation_score']['mse'] == 0
    assert result['full_frontier'] == 'frontier.json'
    assert 'test_score' not in result
