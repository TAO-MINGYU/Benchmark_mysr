"""Native external solver APIs. Only training observations enter these functions."""
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


class NotApplicable(Exception):
    pass


class InvalidOutput(ValueError):
    """A native solver returned no usable prediction or expression."""


def prepare(method, source_root):
    """Import heavyweight libraries before the search clock starts."""
    if method == 'pysr':
        from pysr import PySRRegressor
        return PySRRegressor
    if method == 'operon':
        from pyoperon.sklearn import SymbolicRegressor
        return SymbolicRegressor
    if method == 'gplearn':
        from gplearn.genetic import SymbolicRegressor
        return SymbolicRegressor
    if method == 'dsr':
        import collections
        import collections.abc
        # Upstream still uses the pre-Python-3.10 alias. Keep the fix local.
        if not hasattr(collections, 'Mapping'):
            collections.Mapping = collections.abc.Mapping
        for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex, 'object': object}.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        sys.path.insert(0, str(Path(source_root) / 'dsr-pytorch' / 'dso'))
        import torch
        torch.set_num_threads(1)
        from dso.task.regression.sklearn import DeepSymbolicRegressor
        import dso.execute as execute
        if execute.cyfunc is None:
            execute.cython_execute = execute.python_execute
        return DeepSymbolicRegressor
    if method == 'ai_feynman_2':
        import torch
        torch.set_num_threads(1)
        from aifeynman import run_aifeynman
        return run_aifeynman
    if method == 'tf4sr':
        sys.path.insert(0, str(Path(source_root) / 'tf4sr'))
        import torch
        torch.set_num_threads(1)
        from model.transformer_model import TransformerModel
        model = TransformerModel(enc_type='mix', nb_samples=50, max_nb_var=7,
                                 d_model=256, vocab_size=20, seq_length=30,
                                 h=4, N_enc=4, N_dec=8, dropout=0.25)
        weights = Path(source_root) / 'tf4sr/best_model_weights/mix_label_smoothing/model_weights.pt'
        state = torch.load(weights, map_location='cpu', weights_only=True)
        model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()})
        model.eval()
        return model
    raise ValueError(method)


def fit(method, api, X, y, request):
    seed, budget = request['seed'], request['budget']
    random.seed(seed)
    np.random.seed(seed)
    if method in ('dsr', 'ai_feynman_2', 'tf4sr'):
        import torch
        torch.manual_seed(seed)
    limit = budget['evaluations']
    seconds = budget['search_seconds']
    names = [f'x{i}' for i in range(X.shape[1])]
    candidates = []
    details = {'evaluation_count': None, 'evaluation_semantics': 'not_exposed',
               'comparison_track': 'native_default', 'frontier_kind': 'native_exposed_candidates'}
    search_started = time.monotonic()
    def search_finished():
        # Export/validation are scoring work and have no non-protocol time cap.
        from .worker import dump
        details['native_search_seconds'] = time.monotonic() - search_started
        dump('stage.json', {'stage': 'scoring', 'time': time.time(),
                           'search_seconds': details['native_search_seconds']})
    if method == 'pysr':
        m = api(niterations=1000000, populations=1, population_size=100,
                max_evals=limit, timeout_in_seconds=seconds, parallelism='serial',
                deterministic=True, random_state=seed, progress=False, verbosity=0,
                binary_operators=['+', '-', '*', '/'], unary_operators=['sin', 'cos', 'exp', 'log', 'sqrt'],
                output_directory=str(Path.cwd() / 'pysr-output'))
        m.fit(X, y, variable_names=names)
        search_finished()
        for idx, row in m.equations_.iterrows():
            candidates.append((str(row['sympy_format']), int(row['complexity']),
                               lambda Z, i=idx: m.predict(Z, index=i)))
        details['evaluation_semantics'] = 'native_max_evals_enforced_actual_count_not_exposed'
    elif method == 'operon':
        m = api(population_size=min(1000, max(20, limit // 5)), generations=1000000,
                max_evaluations=limit, max_time=max(1, int(seconds)), n_threads=1,
                random_state=seed, objectives=['r2', 'length'],
                allowed_symbols='add,sub,mul,div,sin,cos,exp,log,sqrt,constant,variable')
        # PyOperon's post-search Pareto statistics must not discard an entire
        # front because one native model overflows. Shared scoring rejects it.
        import pyoperon.sklearn as operon_api
        original_mse = operon_api.mean_squared_error
        def finite_mse(y_true, y_pred, *a, **kw):
            if not np.isfinite(y_pred).all():
                return float('inf')
            return original_mse(y_true, y_pred, *a, **kw)
        operon_api.mean_squared_error = finite_mse
        try:
            m.fit(X, y)
        finally:
            operon_api.mean_squared_error = original_mse
        search_finished()
        for row in m.pareto_front_:
            tree = row['tree']
            candidates.append((m.get_model_string(tree, precision=17, names=names), int(row['length']),
                               lambda Z, t=tree: m.evaluate_model(t, Z)))
        details.update(evaluation_count=int(m.stats_['evaluation_count']),
                       evaluation_semantics='native_fitness_calls', native_stats=m.stats_)
    elif method == 'gplearn':
        pop = min(1000, max(20, limit // 5))
        generations = max(1, limit // pop)
        m = api(population_size=pop, generations=generations, random_state=seed,
                n_jobs=1, low_memory=False, function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt'),
                feature_names=names)
        from .checkpoints import NativeCheckpoint
        import gplearn.genetic as genetic
        checkpoint = NativeCheckpoint(request)
        original_evolve = genetic._parallel_evolve
        def evolve(*a, **kw):
            population = original_evolve(*a, **kw)
            retained = [p for g in m._programs for p in (g or []) if p is not None]
            evaluations = pop * (len(m.run_details_['generation']) + 1)
            checkpoint.write(retained + population, evaluations)
            return population
        genetic._parallel_evolve = evolve
        try:
            m.fit(X, y)
        finally:
            genetic._parallel_evolve = original_evolve
        search_finished()
        # gplearn has no HOF: preserve every surviving program exposed by its history.
        programs = {gplearn_expression(p): p for generation in m._programs for p in (generation or []) if p is not None}
        programs[gplearn_expression(m._program)] = m._program
        for expr, p in programs.items():
            candidates.append((gplearn_expression(p), p.length_, p.execute))
        details.update(evaluation_count=pop * len(m.run_details_['generation']),
                       evaluation_semantics='generated_programs_excludes_internal_prediction_calls',
                       frontier_kind='all_retained_native_programs_no_native_hof')
    elif method == 'dsr':
        from dso.program import Program
        from dso.train import Trainer
        from .checkpoints import NativeCheckpoint
        checkpoint = NativeCheckpoint(request)
        m = api(config=dsr_config(request))
        original_step = Trainer.run_one_step
        def step(trainer, *a, **kw):
            result = original_step(trainer, *a, **kw)
            checkpoint.write(list(Program.cache.values()), int(trainer.nevals))
            return result
        Trainer.run_one_step = step
        try:
            m.fit(X, y)
        finally:
            Trainer.run_one_step = original_step
        search_finished()
        for p in list(Program.cache.values()):
            if not p.invalid:
                # DSO symbols are one-indexed; replace simultaneously.
                candidates.append((dsr_expression(p), len(p.traversal), p.execute))
        details.update(evaluation_count=int(m.trainer.nevals), evaluation_semantics='native_sampled_programs',
                       frontier_kind='all_cached_valid_programs', execution_backend='native_python_fallback_when_cyfunc_absent')
    elif method == 'ai_feynman_2':
        if not request.get('ground_truth_available', False):
            raise NotApplicable('Pre-registered physics/known-expression tasks only; black-box excluded.')
        local_entrypoints(Path(sys.executable))
        Path('input').mkdir(exist_ok=True)
        np.savetxt('input/train.txt', np.column_stack((X, y)))
        from aifeynman.get_pareto import ParetoSet
        from benchmark_mysr.external.worker import dump
        original_add = ParetoSet.add
        root_set = [None]
        def checkpoint_add(self, point):
            if root_set[0] is None:
                root_set[0] = self
            result = original_add(self, point)
            if self is root_set[0]:
                dump('native-frontier.json', [{'expression': str(p[2]), 'native_complexity': float(p[0])} for p in self.get_pareto_points()])
            return result
        # S_run_aifeynman creates the root set before any recursive call.
        original_init = ParetoSet.__init__
        def root_init(self):
            original_init(self)
            if root_set[0] is None:
                root_set[0] = self
        ParetoSet.__init__ = root_init
        ParetoSet.add = checkpoint_add
        api(str(Path('input').resolve()) + '/', 'train.txt', BF_try_time=min(60, max(1, int(seconds / 10))),
            BF_ops_file_type='14ops.txt', polyfit_deg=3, NN_epochs=100, test_percentage=0)
        search_finished()
        import sympy as sp
        symbols = sp.symbols(' '.join(names), seq=True)
        paths = list(Path('results').glob('solution*'))
        for path in paths:
            for line in path.read_text().splitlines():
                fields = line.split()
                if not fields:
                    continue
                try:
                    expr = sp.sympify(fields[-1].replace('^', '**'))
                    if expr.free_symbols - set(symbols):
                        continue
                    f = sp.lambdify(symbols, expr, 'numpy')
                    candidates.append((str(expr), int(sp.count_ops(expr)) + 1, lambda Z, f=f: f(*Z.T)))
                except (ValueError, TypeError, SyntaxError):
                    continue
        details['evaluation_semantics'] = 'no_native_evaluation_cap_wall_and_rss_limited'
    elif method == 'tf4sr':
        candidates, extra = fit_tf4sr(api, X, y, request)
        details.update(extra)
    if not candidates:
        raise InvalidOutput('Solver returned no exportable candidates')
    return candidates, details


def dsr_config(request):
    return {'task': {'task_type': 'regression', 'function_set': ['add','sub','mul','div','sin','cos','exp','log','const']},
            'experiment': {'seed': request['seed'], 'device': 'cpu', 'logdir': None},
            'training': {'n_samples': request['budget']['evaluations'],
                         'batch_size': min(100, request['budget']['evaluations']),
                         'n_cores_batch': 1, 'verbose': False},
            'gp_meld': {'run_gp_meld': False}}


def dsr_expression(program):
    if hasattr(program, 'traversal'):
        # Upstream's pretty printer may return text when SymPy simplification
        # fails (e.g. an invalid candidate containing log(0)). Export the exact
        # native tree without simplifying it; shared scoring handles validity.
        stack = []
        for token in reversed(program.traversal):
            if token.arity == 0:
                if token.input_var is not None:
                    stack.append(f'x{token.input_var}')
                else:
                    stack.append(repr(float(token.value)))
                continue
            if len(stack) < token.arity:
                raise ValueError('Incomplete native DSR traversal')
            args = [stack.pop() for _ in range(token.arity)]
            if token.name in ('add', 'sub', 'mul', 'div'):
                operator = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[token.name]
                stack.append(f'({args[0]}{operator}{args[1]})')
            elif token.name in ('sin', 'cos', 'exp', 'log'):
                stack.append(f'{token.name}({args[0]})')
            else:
                raise ValueError(f'Unsupported native DSR token: {token.name}')
        if len(stack) != 1:
            raise ValueError('Malformed native DSR traversal')
        return stack[0]
    import sympy as sp
    expr = sp.sympify(program.sympy_expr, evaluate=False)
    expr = expr.xreplace({s: sp.Symbol(f'x{int(str(s)[1:])-1}')
                         for s in expr.free_symbols
                         if str(s).startswith('x') and str(s)[1:].isdigit()})
    return str(expr)


def fit_tf4sr(model, X, y, request):
    import sympy as sp
    import torch
    from scipy.optimize import least_squares
    from model._utils import is_tree_complete, translate_integers_into_tokens
    from datasets._utils import from_sequence_to_sympy
    if not request.get('task_id', '').startswith('srsd-'):
        raise NotApplicable('TF4SR is registered for SRSD tabular tasks only, not ODE or legacy/black-box.')
    if X.shape[1] > 6:
        raise NotApplicable('Pretrained TF4SR supports at most six input variables.')
    valid = np.all(X > 0, axis=1) & (y != 0)
    if valid.sum() < 50:
        raise NotApplicable('TF4SR requires 50 training rows with positive inputs and nonzero targets.')
    idx = np.random.choice(np.flatnonzero(valid), 50, replace=False)
    xscale = 10 ** np.mean(np.log10(X[idx]), axis=0)
    # Public unit metadata (no formulas) preserves angular coordinates.
    units = request.get('input_units', [])
    if units and len(units) != X.shape[1]:
        raise ValueError('TF4SR input units must match the materialized feature columns')
    for j, unit in enumerate(units):
        if unit == '$rad$':
            xscale[j] = 1.0
    yscale = 10 ** np.mean(np.log10(np.abs(y[idx])))
    data = np.zeros((50, 7), dtype=np.float32)
    data[:, 0] = y[idx] / yscale
    data[:, 1:X.shape[1]+1] = X[idx] / xscale
    with torch.no_grad():
        enc = model.encoder(torch.tensor(data)[None, :, :, None])
        n = model.decoder.positional_encoding.seq_length
        out = torch.zeros((1, n+1), dtype=torch.int64)
        out[:, 0] = 1
        for i in range(n):
            mask = (out[:, :-1] == 0)[:, None, None, :] | torch.triu(torch.ones(n, n), diagonal=1).bool()
            dec = model.decoder(target_seq=out[:, :-1], mask_dec=mask, output_enc=enc)
            out[:, i+1] = torch.argmax(model.last_layer(dec)[:, i], axis=-1)
            if is_tree_complete(out[0, 1:]):
                break
    if not is_tree_complete(out[0, 1:]):
        raise InvalidOutput('TF4SR returned an incomplete decoded expression')
    expr = sp.sympify(from_sequence_to_sympy(translate_integers_into_tokens(out[0])))
    substitutions = {}
    constants = []
    for s in expr.free_symbols:
        if str(s) == 'C':
            constants.append(s)
        elif str(s).startswith('x') and str(s)[1:].isdigit():
            j = int(str(s)[1:]) - 1
            if j >= X.shape[1]:
                raise InvalidOutput('TF4SR predicted an absent feature')
            substitutions[s] = sp.Symbol(f'x{j}') / float(xscale[j])
    expr = yscale * expr.xreplace(substitutions)
    symbols = [sp.Symbol(f'x{j}') for j in range(X.shape[1])]
    if constants:
        f = sp.lambdify(symbols + constants, expr, 'numpy')
        def residual(c):
            with np.errstate(all='ignore'):
                v = np.broadcast_to(f(*X.T, *c), y.shape) - y
            return np.nan_to_num(v / max(float(np.std(y)), 1e-15), nan=1e10, posinf=1e10, neginf=-1e10)
        opt = least_squares(residual, np.ones(len(constants)), max_nfev=200)
        expr = expr.subs(dict(zip(constants, opt.x)))
    f = sp.lambdify(symbols, expr, 'numpy')
    return [(str(expr), int(sp.count_ops(expr))+1, lambda Z: f(*Z.T))], {
        'evaluation_count': 1, 'evaluation_semantics': 'one_greedy_decode_plus_train_only_constant_fit',
        'frontier_kind': 'single_native_decoding', 'training_rows_used_for_decode': 50,
        'constant_fit': 'shared_C_train_only_max_nfev_200',
        'preprocessing': 'official_log_scale_positive_training_rows; inverse_scale_export',
        'pretraining_overlap': 'not_audited_do_not_claim_unseen_formula_recovery'}


def gplearn_expression(program):
    # Emit a prefix program in one pass; native trees can exceed Python's
    # recursion limit. Avoid recursive string construction and quadratic copies.
    parts, remaining = [], []
    completed = False
    for node in program.program:
        if completed:
            raise ValueError('Native program has trailing nodes')
        if hasattr(node, 'arity'):
            if node.arity <= 0:
                raise ValueError('Native function has invalid arity')
            parts.append(node.name + '(')
            remaining.append(node.arity)
            continue
        parts.append(f'x{node}' if isinstance(node, (int, np.integer)) else repr(float(node)))
        while remaining:
            remaining[-1] -= 1
            if remaining[-1]:
                parts.append(',')
                break
            remaining.pop()
            parts.append(')')
        else:
            completed = True
    if not completed or remaining:
        raise ValueError('Native program is incomplete')
    return ''.join(parts)


def local_entrypoints(python):
    """Preserve native Fortran launcher bodies while relocating their interpreter."""
    target = Path.cwd() / 'native-bin'
    target.mkdir(exist_ok=True)
    for original in python.parent.glob('feynman*'):
        if not original.is_file():
            continue
        content = original.read_text()
        if not content.startswith('#!'):
            raise ValueError(f'Unexpected AI-Feynman entrypoint: {original.name}')
        body = content.split('\n', 1)[1]
        entry = target / original.name
        entry.write_text('#!' + str(python) + '\n' + body)
        entry.chmod(0o700)
    os.environ['PATH'] = str(target) + os.pathsep + os.environ.get('PATH', '')


def warmup_pysr(api, n_features):
    """Compile the same operator/type path on synthetic data before search timing."""
    X = np.linspace(0.2, 1.0, 32*n_features).reshape(32, n_features)
    y = X[:, 0]**2 + X[:, 0]
    model = api(niterations=2, populations=1, population_size=100, max_evals=1000,
                ncycles_per_iteration=1, parallelism='serial', deterministic=True,
                random_state=0, progress=False, verbosity=0,
                binary_operators=['+', '-', '*', '/'],
                unary_operators=['sin', 'cos', 'exp', 'log', 'sqrt'],
                output_directory=str(Path.cwd()/'pysr-warmup'))
    model.fit(X, y, variable_names=[f'x{i}' for i in range(n_features)])
