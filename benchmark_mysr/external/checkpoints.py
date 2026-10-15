"""Retain completed native programs for scoring after a budget timeout.

Snapshots contain native tokens and constants, never a newly searched model.
Recovery uses the same native prediction functions, including protected GP ops.
"""
import math
import csv
import ast
import hashlib
import pickle
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


def pysr_csv_predictor(expression, n_features):
    """Evaluate native CSV operations without symbolic rewrites or code eval."""
    binary = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
              ast.Div: np.divide, ast.Pow: np.power}
    unary = {ast.UAdd: np.positive, ast.USub: np.negative}
    functions = {'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
                 'log': np.log, 'sqrt': np.sqrt}
    names = {f'x{i}': i for i in range(n_features)}
    def build(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = np.float64(node.value)
            return lambda X: value
        if isinstance(node, ast.Name) and node.id in names:
            index = names[node.id]
            return lambda X: X[:, index]
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            operation, left, right = binary[type(node.op)], build(node.left), build(node.right)
            return lambda X: operation(left(X), right(X))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            operation, child = unary[type(node.op)], build(node.operand)
            return lambda X: operation(child(X))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and len(node.args) == 1 and not node.keywords):
            operation, child = functions[node.func.id], build(node.args[0])
            return lambda X: operation(child(X))
        raise ValueError('PySR checkpoint contains an absent feature or unsupported operation')
    function = build(ast.parse(expression.replace('^', '**'), mode='eval').body)
    return lambda X: np.broadcast_to(function(X), (X.shape[0],))


def recover_pysr(path, n_features):
    """Replay actual search CSV models without Julia or symbolic conversion.

    Keep native expression text/node counts and ordinary NumPy operator domains.
    In particular, nested roots need no potentially expensive symbolic rewrite.
    """
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    candidates = []
    for row in rows:
        candidates.append((row['Equation'], int(row['Complexity']),
                           pysr_csv_predictor(row['Equation'], n_features)))
    return candidates, {'evaluation_count': None,
                        'evaluation_semantics': 'native_max_evals_enforced_actual_count_not_exposed',
                        'frontier_kind': 'native_pysr_csv_at_timeout', 'additional_search': False,
                        'prediction_semantics': 'native_csv_operation_order_numpy_float64_no_symbolic_rewrite'}


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
             'comparison_protocol_sha256': self.request.get('comparison_protocol_sha256'),
             'evaluation_count': evaluations, 'candidates': rows,
             'scope': 'completed_native_programs_at_last_checkpoint_no_additional_search'})
        self.last = time.monotonic()


class OperonCheckpoint:
    """Serialize native trees only at safe, completed-generation callbacks."""
    def __init__(self,request,interval=10):
        self.request,self.interval,self.last=request,interval,float('-inf')

    def write(self,algorithm,variables):
        if time.monotonic()-self.last<self.interval:
            return
        trees=[individual.Genotype for individual in algorithm.BestFront]
        if not trees:
            return
        raw=pickle.dumps(trees,protocol=pickle.HIGHEST_PROTOCOL)
        checksum=hashlib.sha256(raw).hexdigest()
        # A killed write must leave the previous JSON/tree pair recoverable.
        # Publish each blob under a fresh immutable name, then its JSON pointer.
        path=Path(f'operon-pareto-{int(algorithm.Generation)}-{checksum[:16]}.pkl')
        temp=path.with_suffix('.pkl.tmp');temp.write_bytes(raw);temp.replace(path)
        from .worker import dump
        dump('native-candidates.json',{'schema':'native-operon-checkpoint-v1',
             'method':'operon','seed':self.request['seed'],'budget':self.request['budget'],
             'comparison_protocol_sha256':self.request.get('comparison_protocol_sha256'),
             'pickle_file':path.name,'pickle_sha256':checksum,
             'variables':variables,'generation':int(algorithm.Generation),'evaluation_count':None,
             'scope':'native_pareto_at_last_completed_generation_no_additional_search'})
        self.last=time.monotonic()


def recover(request, api, X, y):
    import json
    payload = json.loads(Path(request['recover_native_candidates']).read_text())
    schema = 'native-operon-checkpoint-v1' if request['method']=='operon' else 'native-token-checkpoint-v1'
    if (payload['schema'] != schema or
            any(payload[k] != request[k] for k in ('method', 'seed', 'budget')) or
            payload.get('comparison_protocol_sha256') != request.get('comparison_protocol_sha256')):
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
    elif request['method'] == 'operon':
        import pyoperon as op
        parent=Path(request['recover_native_candidates']).resolve().parent
        path=(parent/payload['pickle_file']).resolve()
        if path.parent!=parent:
            raise ValueError('Native Operon checkpoint escapes its own artifact directory')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=payload['pickle_sha256']:
            raise ValueError('Native Operon tree checkpoint checksum differs')
        # Only our own same-run, identity/hash-checked native checkpoint is read.
        trees=pickle.loads(raw)
        model=api()
        model.variables_={int(key):value for key,value in payload['variables'].items()}
        for tree in trees:
            values=model.evaluate_model(tree,X)
            if np.isfinite(values).all():
                scale,offset=op.FitLeastSquares(values,y)
                nodes=tree.Nodes
                if scale!=1:
                    nodes += [op.Node.Constant(scale),op.Node.Mul()]
                if offset!=0:
                    nodes += [op.Node.Constant(offset),op.Node.Add()]
                tree=op.Tree(nodes).UpdateNodes()
            candidates.append((model.get_model_string(tree,precision=17,names=[f'x{i}' for i in range(X.shape[1])]),
                               len(tree.Nodes),lambda Z,t=tree:model.evaluate_model(t,Z)))
        semantics='native_fitness_count_not_exposed_at_checkpoint'
    elif request['method'] == 'gplearn':
        from gplearn._program import _Program
        from gplearn.functions import _function_map
        from gplearn.fitness import _fitness_map
        from .solvers import gplearn_expression
        from .solvers import gplearn_function_map
        mapping = gplearn_function_map(request)
        functions = list(mapping.values())
        arities = {n: [f for f in functions if f.arity == n] for n in (1, 2)}
        for row in payload['candidates']:
            tokens = [mapping[value] if kind == 'function' else value
                      for kind, value in row['tokens']]
            p = _Program(function_set=functions, arities=arities, init_depth=(2, 6),
                         init_method='half and half', n_features=X.shape[1],
                         const_range=(-1, 1), metric=_fitness_map['mse' if request.get('comparison_protocol') else 'mean absolute error'],
                         p_point_replace=0.05, parsimony_coefficient=0.001,
                         random_state=np.random.RandomState(request['seed']), program=tokens)
            candidates.append((gplearn_expression(p), p.length_, p.execute))
        semantics = 'generated_programs_at_last_completed_generation'
    else:
        raise ValueError('Native token recovery is unsupported for this method')
    return candidates, {'evaluation_count': payload['evaluation_count'],
                        'evaluation_semantics': semantics,
                        'frontier_kind': payload['scope'], 'additional_search': False}
