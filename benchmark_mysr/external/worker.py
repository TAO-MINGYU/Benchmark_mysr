"""One solver run in its pinned environment; invoked by the supervisor."""
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def dump(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def read_split(path):
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        fields = [s for s in reader.fieldnames if s.startswith('x_')]
        rows = list(reader)
    X = np.array([[float(row[c]) for c in fields] for row in rows])
    y = np.array([float(row['target']) for row in rows])
    clean = np.array([float(row.get('target_clean', row['target'])) for row in rows])
    if not len(rows) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Empty/nonfinite input data')
    return X, y, clean


def score(y, pred):
    with np.errstate(all='ignore'):
        p = np.broadcast_to(np.asarray(pred, dtype=float).reshape(-1), y.shape)
        mse = float(np.mean((p - y) ** 2))
        scale = float(np.std(y))
        nrmse = float(np.sqrt(mse) / max(scale, 1e-15))
    if not np.isfinite(p).all() or not np.isfinite(nrmse):
        raise ValueError('Nonfinite predictions/score')
    return {'mse': mse, 'nrmse': nrmse}


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    from benchmark_mysr.external.solvers import NotApplicable, prepare, fit
    started = time.monotonic()
    try:
        if request['method'] == 'tf4sr' and not request.get('task_id', '').startswith('srsd-'):
            raise NotApplicable('TF4SR is registered for compatible SRSD tabular tasks only.')
        if request['method'] == 'ai_feynman_2' and not request.get('ground_truth_available', False):
            raise NotApplicable('AI-Feynman is registered for known-expression tasks only.')
        train = read_split(Path(request['data']) / 'train.csv')
        val = read_split(Path(request['data']) / 'validation.csv')
        if request.get('recover_candidates'):
            import sympy as sp
            symbols = [sp.Symbol(f'x{j}') for j in range(train[0].shape[1])]
            candidates = []
            for row in json.loads(Path(request['recover_candidates']).read_text()):
                expr = sp.sympify(row['expression'].replace('^', '**'))
                if expr.free_symbols - set(symbols):
                    continue
                f = sp.lambdify(symbols, expr, 'numpy')
                candidates.append((str(expr), int(sp.count_ops(expr))+1, lambda Z, f=f: f(*Z.T)))
            details = {'frontier_kind': 'root_native_pareto_checkpoint_at_timeout',
                       'evaluation_count': None, 'evaluation_semantics': 'not_exposed'}
            search_time = 0.0
            startup_time = time.monotonic()-started
        else:
            api = prepare(request['method'], request['source_root'])
            startup_time = time.monotonic()-started
            dump('stage.json', {'stage': 'search', 'time': time.time(), 'startup_seconds': startup_time})
            search_start = time.monotonic()
            candidates, details = fit(request['method'], api, train[0], train[1], request)
            search_time = time.monotonic() - search_start
        dump('stage.json', {'stage': 'scoring', 'time': time.time(), 'search_seconds': search_time})
        scoring_started = time.monotonic()
        # Select using validation only. No test data is loaded until selection is frozen.
        frontier, best, best_key = [], None, None
        for index, (expression, complexity, predict) in enumerate(candidates):
            row = {'expression': expression, 'complexity': int(complexity), 'candidate_index': index}
            try:
                row['train_score'] = score(train[1], predict(train[0]))
                row['validation_score'] = score(val[1], predict(val[0]))
                key = (row['validation_score']['mse'], complexity, expression)
                if best_key is None or key < best_key:
                    best, best_key = index, key
                row['status'] = 'valid'
            except Exception as exc:
                row.update(status='invalid', reason=str(exc))
            frontier.append(row)
        dump('frontier.json', frontier)
        if best is None:
            raise ValueError('No candidate has finite train and validation predictions')
        selected = dict(frontier[best])
        test = read_split(Path(request['data']) / 'test.csv')
        predict = candidates[best][2]
        prediction = np.broadcast_to(np.asarray(predict(test[0])).reshape(-1), test[1].shape)
        selected['test_score'] = score(test[1], prediction)
        selected['test_clean_score'] = score(test[2], prediction)
        np.savez_compressed('test-predictions.npz', prediction=prediction)
        result = {'status': 'success', 'failure_status': None,
                  'selected_expression': selected['expression'], 'complexity': selected['complexity'],
                  'complexity_definition': 'native_node_count_or_sympy_ops_plus_one_not_cross_method_identical',
                  'train_score': selected['train_score'], 'validation_score': selected['validation_score'],
                  'test_score': selected['test_score'], 'test_clean_score': selected['test_clean_score'],
                  'full_frontier': 'frontier.json', 'frontier_count': len(frontier),
                  'frontier_sha256': hashlib.sha256(Path('frontier.json').read_bytes()).hexdigest(),
                  'search_seconds': search_time, 'startup_seconds': startup_time,
                  'scoring_seconds': time.monotonic()-scoring_started, 'details': details,
                  'recovery_status': 'not_scored_symbolically',
                  'cpu_affinity': sorted(os.sched_getaffinity(0)),
                  'test_numeric_agreement': bool(np.allclose(prediction, test[2], rtol=1e-6, atol=1e-8)) if request.get('ground_truth_available') else None}
        dump('worker-result.json', result)
    except NotApplicable as exc:
        dump('worker-result.json', {'status': 'not_applicable', 'failure_status': 'not_applicable', 'reason': str(exc)})
    except Exception as exc:
        traceback.print_exc()
        dump('worker-result.json', {'status': 'solver_error', 'failure_status': type(exc).__name__, 'reason': str(exc)})
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
