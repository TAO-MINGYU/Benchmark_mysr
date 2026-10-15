"""Retain completed native programs for scoring after a budget timeout.

Snapshots contain native tokens and constants, never a newly searched model.
Recovery uses the same native prediction functions, including protected GP ops.
"""
import math
import csv
import time
from pathlib import Path

import numpy as np


def pysr_checkpoint(output):
    """Find the actual search HOF, excluding the synthetic compilation warmup."""
    directories = sorted(p for p in (Path(output) / 'pysr-output').glob('*') if p.is_dir())
    if len(directories) != 1:
        return None
    for name in ('hall_of_fame.csv', 'hall_of_fame.csv.bak'):
        path = directories[0] / name
        try:
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if rows and all(row.get('Equation') and int(row['Complexity']) > 0 for row in rows):
                return path
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None


def recover_pysr(path, n_features):
    """Use native PySR exporters on a completed CSV snapshot, with no search."""
    from pysr.export_sympy import pysr2sympy
    from pysr.export_numpy import sympy2numpy
    import sympy as sp
    names = [f'x{i}' for i in range(n_features)]
    symbols = [sp.Symbol(name) for name in names]
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    candidates = []
    for row in rows:
        expr = pysr2sympy(row['Equation'], feature_names_in=names)
        if expr.free_symbols - set(symbols):
            raise ValueError('PySR checkpoint contains an absent feature')
        candidates.append((str(expr), int(row['Complexity']), sympy2numpy(expr, symbols)))
    return candidates, {'evaluation_count': None,
                        'evaluation_semantics': 'native_max_evals_enforced_actual_count_not_exposed',
                        'frontier_kind': 'native_pysr_csv_at_timeout', 'additional_search': False}


class NativeCheckpoint:
    def __init__(self, request, interval=10):
        self.request = request
        self.interval = interval
        self.last = float('-inf')

    def write(self, programs, evaluations, force=False):
        now = time.monotonic()
        if not force and now - self.last < self.interval:
            return
        rows = []
        if self.request['method'] == 'dsr':
            from dso.program import Program
            for p in programs:
                constants = [float(c) for c in p.get_constants()]
                if not p.invalid and all(math.isfinite(c) for c in constants):
                    rows.append({'tokens': [int(t) for t in p.tokens], 'constants': constants})
            library = [str(x) for x in Program.library.names]
        else:
            library = None
            seen = set()
            for p in programs:
                tokens = []
                for node in p.program:
                    if hasattr(node, 'arity'):
                        tokens.append(('function', node.name))
                    elif isinstance(node, (int, np.integer)):
                        tokens.append(('variable', int(node)))
                    else:
                        tokens.append(('constant', float(node)))
                key = tuple(tokens)
                if key not in seen and all(kind != 'constant' or math.isfinite(value)
                                           for kind, value in tokens):
                    seen.add(key)
                    rows.append({'tokens': tokens})
        from .worker import dump
        dump('native-candidates.json', {'schema': 'native-token-checkpoint-v1',
             'method': self.request['method'], 'seed': self.request['seed'],
             'budget': self.request['budget'], 'library': library,
             'evaluation_count': evaluations, 'candidates': rows,
             'scope': 'completed_native_programs_at_last_checkpoint_no_additional_search'})
        self.last = time.monotonic()


def recover(request, api, X, y):
    import json
    payload = json.loads(Path(request['recover_native_candidates']).read_text())
    if (payload['schema'] != 'native-token-checkpoint-v1' or
            any(payload[k] != request[k] for k in ('method', 'seed', 'budget'))):
        raise ValueError('Native checkpoint identity differs from run request')
    candidates = []
    if request['method'] == 'dsr':
        from dso.program import Program, from_tokens
        from .solvers import dsr_config, dsr_expression
        model = api(config=dsr_config(request))
        config = model.config
        config['task']['dataset'] = (X, y)
        model.set_config(config)
        model.setup()
        if [str(x) for x in Program.library.names] != payload['library']:
            raise ValueError('Native DSR token library differs from checkpoint')
        for row in payload['candidates']:
            p = from_tokens(np.asarray(row['tokens'], dtype=np.int32), skip_cache=True)
            p.set_constants(row['constants'])
            candidates.append((dsr_expression(p), len(p.traversal), p.execute))
        semantics = 'native_sampled_programs_at_last_checkpoint'
    elif request['method'] == 'gplearn':
        from gplearn._program import _Program
        from gplearn.functions import _function_map
        from gplearn.fitness import _fitness_map
        from .solvers import gplearn_expression
        functions = [_function_map[n] for n in ('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt')]
        arities = {n: [f for f in functions if f.arity == n] for n in (1, 2)}
        for row in payload['candidates']:
            tokens = [_function_map[value] if kind == 'function' else value
                      for kind, value in row['tokens']]
            p = _Program(function_set=functions, arities=arities, init_depth=(2, 6),
                         init_method='half and half', n_features=X.shape[1],
                         const_range=(-1, 1), metric=_fitness_map['mean absolute error'],
                         p_point_replace=0.05, parsimony_coefficient=0.001,
                         random_state=np.random.RandomState(request['seed']), program=tokens)
            candidates.append((gplearn_expression(p), p.length_, p.execute))
        semantics = 'generated_programs_at_last_completed_generation'
    else:
        raise ValueError('Native token recovery is unsupported for this method')
    return candidates, {'evaluation_count': payload['evaluation_count'],
                        'evaluation_semantics': semantics,
                        'frontier_kind': payload['scope'], 'additional_search': False}
