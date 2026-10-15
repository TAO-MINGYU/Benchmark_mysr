"""Replay one frozen failed native fit into a separate corrective result tree."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_selected(entry, output_root, env_root, sources):
    from .runner import run
    primary = Path(entry['primary_result'])
    if digest(primary) != entry['primary_sha256']:
        raise ValueError('Selected primary result changed before correction')
    request_path = primary.parent / 'request.json'
    if digest(request_path) != entry['request_sha256']:
        raise ValueError('Selected scientific request changed before correction')
    baseline = Path(entry['original_baseline_result'])
    if digest(baseline) != entry['original_baseline_sha256']:
        raise ValueError('Original baseline changed before correction')
    request = json.loads(request_path.read_text())
    before = json.loads(primary.read_text())
    for key in ('method', 'task_id', 'variant', 'resource_track', 'seed', 'budget'):
        if request[key] != before[key]:
            raise ValueError('Primary result/request identity differs: ' + key)
    if before['data_hashes'] != entry['data_hashes']:
        raise ValueError('Primary data hashes differ from frozen selection')
    request['source_root'] = str(sources)
    request['expected_data_hashes'] = entry['data_hashes']
    relative = Path(entry['relative_output'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Correction output must stay inside its dedicated root')
    output_root = Path(output_root).resolve()
    output = output_root / relative
    if output.resolve() == primary.parent.resolve():
        raise ValueError('Correction must not overwrite its primary attempt')
    return run(request, output, env_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--env-root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    args = parser.parse_args()
    entry = json.loads(args.selection.read_text())
    result = run_selected(entry, args.output_root, args.env_root, args.sources)
    print(json.dumps({'status': result['status'], 'method': result['method'],
                      'task_id': result['task_id'], 'seed': result['seed']}), flush=True)
    # A preserved negative outcome is a completed fit, not a launcher failure.
    # Scientific success and unresolved solver faults are audited separately.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
