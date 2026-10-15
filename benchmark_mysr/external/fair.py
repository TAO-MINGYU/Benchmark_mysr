"""Frozen common-compute protocol and train-only numerical coordinate system."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

PROTOCOL = Path(__file__).resolve().parents[2]/'manifests/formal/fair-comparison-v2.json'


def protocol(request=None):
    raw = PROTOCOL.read_bytes()
    value = json.loads(raw)
    if request is not None and request.get('comparison_protocol_sha256') != hashlib.sha256(raw).hexdigest():
        raise ValueError('Frozen comparison protocol checksum differs from the request')
    return value


def enabled(request):
    return request.get('comparison_protocol') == 'common-cpu-train-scale-v2'


class TrainScale:
    """Positive multiplicative factors; no offsets, test inputs or clean targets."""
    def __init__(self, X, y, units=()):
        if units and len(units) != X.shape[1]:
            raise ValueError('Units must align with the training feature columns')
        self.x = np.max(np.abs(X), axis=0).astype(np.float64)
        self.x[self.x == 0] = 1.0
        for j, unit in enumerate(units):
            if unit == '$rad$':
                self.x[j] = 1.0
        self.y = float(np.max(np.abs(y))) or 1.0

    def inputs(self, X):
        return np.asarray(X, dtype=np.float64)/self.x

    def targets(self, y):
        return np.asarray(y, dtype=np.float64)/self.y

    def candidates(self, candidates):
        output = []
        for expression, complexity, predict in candidates:
            original = re.sub(r'\bx(\d+)\b', lambda m: f'(x{m[1]}/{float(self.x[int(m[1])])!r})', expression)
            output.append((f'({self.y!r})*({original})', complexity,
                           lambda Z, f=predict: self.y*np.asarray(f(self.inputs(Z)))))
        return output

    def record(self):
        return {'kind': 'train_max_abs_multiplicative_no_center_v1',
                'input_scales': self.x.tolist(), 'target_scale': self.y,
                'fit_split': 'train_observed_only', 'native_complexity_excludes_shared_preprocessor': True}


def run_fit(requests, offset, expected_sha256, output_root, env_root, sources):
    from .runner import run
    with Path(requests).open('rb') as stream:
        stream.seek(offset)
        raw = stream.readline()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Frozen fit request changed')
    item = json.loads(raw)
    request = item['request']
    policy = protocol(request)
    if request['budget'] != policy['tracks'][request['resource_track']]:
        raise ValueError('Comparison budget differs from its frozen track')
    request['source_root'] = str(sources)
    relative = Path(item['relative_output'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Comparison output must stay inside its fresh run root')
    return run(request, Path(output_root).resolve()/relative, env_root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--offset', type=int, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--env-root', type=Path, required=True)
    p.add_argument('--sources', type=Path, required=True)
    args = p.parse_args()
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('A comparison fit requires exactly one assigned CPU')
    result = run_fit(args.requests, args.offset, args.sha256, args.output_root, args.env_root, args.sources)
    print(json.dumps({'status': result['status'], 'method': result['method'],
                      'task_id': result['task_id'], 'seed': result['seed']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
