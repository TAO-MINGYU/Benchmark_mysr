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
    recovery = entry.get('recovery')
    if recovery:
        if before['status'] != 'search_timeout':
            raise ValueError('Checkpoint-only correction requires an original search timeout')
        path = Path(recovery['path'])
        if digest(path) != recovery['sha256']:
            raise ValueError('Selected native checkpoint changed before recovery')
        field = recovery['request_field']
        allowed = {'dsr': 'recover_native_candidates', 'gplearn': 'recover_native_candidates',
                   'ai_feynman_2': 'recover_candidates', 'pysr': 'recover_pysr_candidates'}
        if field != allowed.get(request['method']):
            raise ValueError('Recovery checkpoint type differs from selected method')
        request[field] = str(path)
        request['recovery_input_sha256'] = recovery['sha256']
    relative = Path(entry['relative_output'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Correction output must stay inside its dedicated root')
    output_root = Path(output_root).resolve()
    output = output_root / relative
    if output.resolve() == primary.parent.resolve():
        raise ValueError('Correction must not overwrite its primary attempt')
    result = run(request, output, env_root)
    if recovery and result.get('correction_operation') != 'existing_native_checkpoint_recovery':
        from .worker import dump
        # A post-budget scoring replay must not relabel its original search
        # as having finished normally. Its CPU/RSS record measures recovery.
        result.update(partial_result_status=result['status'], status=before['status'],
                      failure_status='timeout', primary_result=str(primary),
                      primary_result_sha256=entry['primary_sha256'],
                      correction_operation='existing_native_checkpoint_recovery',
                      resource_scope='checkpoint_scoring_only; original_search_resources_in_primary')
        dump(output/'result.json', result)
    return result


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
