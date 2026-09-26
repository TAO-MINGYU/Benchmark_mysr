"""Trajectory-isolated finite-difference ODE identification extension.

No reference equation is used to construct training or validation targets.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from .worker import dump


def differentiated(solution, start=0, stop=None):
    if not solution['success']:
        raise ValueError('Source trajectory integration failed')
    t = np.asarray(solution['t'], dtype=float)[start:stop]
    X = np.asarray(solution['y'], dtype=float).T[start:stop]
    if len(t) < 5 or not np.all(np.diff(t) > 0) or not np.isfinite(X).all():
        raise ValueError('Invalid or insufficient trajectory observations')
    # Differentiate each split independently and discard edge estimates.
    derivative = np.gradient(X, t, axis=0, edge_order=2)
    return X[1:-1], derivative[1:-1]


def build(source, system_index, output, clean_source):
    source, clean_source, output = Path(source), Path(clean_source), Path(output)
    system = json.loads(source.read_text())[system_index]
    clean = json.loads(clean_source.read_text())[system_index]
    if len(system['solutions']) != 1 or len(system['solutions'][0]) != 2:
        raise ValueError('This registered extension requires one parameter set and two initial conditions')
    first, second = system['solutions'][0]
    cut = int(len(first['t']) * 0.7)
    train = differentiated(first, stop=cut)
    val = differentiated(first, start=cut)
    test = differentiated(second)
    clean_test = differentiated(clean['solutions'][0][1])
    # Test uses the entire second clean trajectory for primary derivative accuracy;
    # observed second-trajectory row counts are retained as metadata only.
    output.mkdir(parents=True, exist_ok=True)
    dirs = []
    for component in range(system['dim']):
        dest = output / f'component-{component}'
        dest.mkdir(exist_ok=True)
        for split, (X, dy) in [('train',train),('validation',val),('test',clean_test)]:
            with (dest/f'{split}.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['row_id']+[f'x_{i}' for i in range(X.shape[1])]+['target','target_clean','target_observed'])
                for i, (x, target) in enumerate(zip(X,dy[:,component])):
                    writer.writerow([f'{split}-{i}',*x,target,target,target])
        dirs.append(str(dest))
    meta={'system_id':system['id'], 'system_index':system_index, 'dimension':system['dim'],
          'task_id':f'odebench-{system["id"]}', 'component_dirs':dirs,
          'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'clean_source_sha256':hashlib.sha256(clean_source.read_bytes()).hexdigest(),
          'policy':'first_IC_70_30_train_validation_second_clean_IC_test; independent_split_finite_differences_drop_edges',
          'metric':'clean_heldout_trajectory_derivative_prediction_not_trajectory_rollout',
          'observed_test_rows':len(test[0]), 'ground_truth_available':True}
    dump(output/'ode-data.json',meta)
    return meta
