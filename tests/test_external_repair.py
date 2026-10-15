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


def test_gradient_consistency_handles_zero_nonfinite_and_overflow_rows():
    from benchmark_mysr.external.solvers import stable_gradient_consistency

    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data, dtype=np.float32)
        def cpu(self):
            return self
        def numpy(self):
            return self.data

    def unsafe_native(tensor):
        raw = tensor.numpy()
        with np.errstate(all='ignore'):
            normalized = raw / np.linalg.norm(raw, axis=1)[:, None]
        return np.linalg.eig(normalized.T @ normalized)

    zero = Tensor([[0, 0], [0, 0]])
    with pytest.raises(np.linalg.LinAlgError):
        unsafe_native(zero)
    score, direction = stable_gradient_consistency(zero, unsafe_native)
    assert score == 0 and np.array_equal(direction, [0, 0])
    mixed = Tensor([[1e30, 1e30], [0, 0], [np.inf, 1], [np.nan, 0]])
    score, direction = stable_gradient_consistency(mixed, unsafe_native)
    assert score == pytest.approx(0.25)
    assert np.isfinite(direction).all()
    ordinary = Tensor([[1, 0], [0, 1]])
    sentinel = object()
    assert stable_gradient_consistency(ordinary, lambda actual: sentinel) is sentinel


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
    assert result['test_prediction_count'] == 1
    assert result['test_prediction_finite_count'] == 0
    assert np.isinf(np.load(tmp_path/result['test_prediction_artifact'])['prediction']).all()
    assert result['prediction_failure_stage'] == 'held_out_test'


def test_pysr_checkpoint_excludes_warmup_and_uses_backup_only_for_bad_primary(tmp_path):
    from benchmark_mysr.external.checkpoints import pysr_checkpoint
    warmup = tmp_path/'pysr-warmup/run'
    warmup.mkdir(parents=True)
    (warmup/'hall_of_fame.csv').write_text('Complexity,Loss,Equation\n1,0,999\n')
    assert pysr_checkpoint(tmp_path) is None
    native = tmp_path/'pysr-output/run'
    native.mkdir(parents=True)
    primary = native/'hall_of_fame.csv'
    backup = native/'hall_of_fame.csv.bak'
    primary.write_text('Complexity,Loss,Equation\n3,0,x0 + 2\n')
    backup.write_text('Complexity,Loss,Equation\n1,1,x0\n')
    assert pysr_checkpoint(tmp_path) == primary
    primary.write_text('Complexity,Loss,Equation\n3,0,')
    assert pysr_checkpoint(tmp_path) == backup
    (tmp_path/'pysr-output/ambiguous').mkdir()
    assert pysr_checkpoint(tmp_path) is None


def test_timeout_recovery_retains_expression_even_when_test_domain_is_invalid(tmp_path, monkeypatch):
    import os
    import sys
    import benchmark_mysr.external.runner as runner
    for split in ('train', 'validation', 'test'):
        (tmp_path/f'{split}.csv').write_text('row_id,x_0,target\na,1,1\n')
    script = tmp_path/'worker.py'
    script.write_text('''import json, pathlib, sys, time
r = json.loads(pathlib.Path(sys.argv[-1]).read_text())
def dump(name, value):
    pathlib.Path(name).write_text(json.dumps(value))
if r.get('recover_candidates'):
    dump('frontier.json', [{'expression': 'log(x0)', 'status': 'valid'}])
    dump('worker-result.json', {'status': 'invalid_output', 'failure_status': 'ValueError',
         'selected_expression': 'log(x0)', 'complexity': 2, 'full_frontier': 'frontier.json',
         'train_score': {'mse': 0., 'nrmse': 0.}, 'validation_score': {'mse': 0., 'nrmse': 0.}})
else:
    dump('stage.json', {'stage': 'search'})
    dump('native-frontier.json', [{'expression': 'log(x0)'}])
    time.sleep(30)
''')
    launcher = tmp_path/'python'
    launcher.write_text('#!/bin/sh\nexec '+sys.executable+' '+str(script)+' "$@"\n')
    launcher.chmod(0o700)
    monkeypatch.setattr(runner, 'environment', lambda *a: (launcher, dict(os.environ)))
    request = {'method': 'ai_feynman_2', 'data': str(tmp_path), 'task_id': 'fixture',
               'seed': 1, 'budget': {'search_seconds': 0, 'memory_gib': 1, 'evaluations': 100}}
    result = runner.run(request, tmp_path/'out', tmp_path)
    assert result['status'] == 'search_timeout'
    assert result['partial_result_status'] == 'invalid_output'
    assert result['selected_expression'] == 'log(x0)'
    assert result['validation_score']['mse'] == 0
    assert result['test_score'] is None
    assert json.loads((tmp_path/'out'/result['full_frontier']).read_text())[0]['expression'] == 'log(x0)'
