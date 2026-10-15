import hashlib
import json

import pytest

from benchmark_mysr.external.repair import run_selected


def test_failed_fit_replay_preserves_budget_and_rejects_changed_evidence(tmp_path, monkeypatch):
    from benchmark_mysr.external import runner
    primary = tmp_path / 'primary'
    primary.mkdir()
    request = {'method': 'tf4sr', 'task_id': 'srsd-fixture', 'variant': 'clean',
               'resource_track': 'constrained_resource', 'seed': 860,
               'budget': {'evaluations': 20000, 'search_seconds': 120, 'memory_gib': 8},
               'source_root': '/old-sources', 'input_units': [[0, 1]], 'data': '/frozen-data'}
    result = {**request, 'status': 'solver_error', 'data_hashes': {'train': 'frozen'}}
    (primary / 'request.json').write_text(json.dumps(request))
    (primary / 'result.json').write_text(json.dumps(result))
    baseline = tmp_path / 'baseline.json'
    baseline.write_text('{}')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    entry = {'primary_result': str(primary / 'result.json'),
             'primary_sha256': sha(primary / 'result.json'),
             'request_sha256': sha(primary / 'request.json'),
             'original_baseline_result': str(baseline), 'original_baseline_sha256': sha(baseline),
             'data_hashes': result['data_hashes'], 'relative_output': 'tabular/tf4sr/fixture'}
    calls = []

    def replay(scientific, output, environment):
        calls.append((scientific, output, environment))
        assert scientific['budget'] == request['budget']
        assert scientific['seed'] == request['seed']
        assert scientific['input_units'] == request['input_units']
        assert scientific['data'] == request['data']
        assert scientific['source_root'] == '/verified-sources'
        assert output != primary
        return {'status': 'invalid_output'}

    monkeypatch.setattr(runner, 'run', replay)
    assert run_selected(entry, tmp_path / 'corrections', '/environment', '/verified-sources') == {
        'status': 'invalid_output'}
    assert len(calls) == 1
    with pytest.raises(ValueError, match='dedicated root'):
        run_selected(dict(entry, relative_output='../primary'), tmp_path / 'corrections',
                     '/environment', '/verified-sources')
    (primary / 'request.json').write_text(json.dumps(dict(request, seed=861)))
    with pytest.raises(ValueError, match='scientific request changed'):
        run_selected(entry, tmp_path / 'corrections', '/environment', '/verified-sources')
    assert len(calls) == 1


def test_checkpoint_replay_preserves_identity_and_rejects_changed_checkpoint(tmp_path, monkeypatch):
    from benchmark_mysr.external import runner
    request = {'method': 'pysr', 'task_id': 'fixture', 'variant': 'clean',
               'resource_track': 'constrained_resource', 'seed': 7,
               'budget': {'evaluations': 20000, 'search_seconds': 120, 'memory_gib': 8}}
    primary = tmp_path/'primary'
    primary.mkdir()
    result = {**request, 'data_hashes': {'train': 'bytes'}, 'status': 'search_timeout'}
    (primary/'request.json').write_text(json.dumps(request))
    (primary/'result.json').write_text(json.dumps(result))
    checkpoint = primary/'native.csv'
    checkpoint.write_text('Complexity,Loss,Equation\n1,0,1\n')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    entry = {'primary_result': str(primary/'result.json'), 'primary_sha256': sha(primary/'result.json'),
             'request_sha256': sha(primary/'request.json'), 'original_baseline_result': str(primary/'result.json'),
             'original_baseline_sha256': sha(primary/'result.json'), 'data_hashes': result['data_hashes'],
             'relative_output': 'recovered', 'recovery': {'path': str(checkpoint),
             'sha256': sha(checkpoint), 'request_field': 'recover_pysr_candidates'}}
    calls = []
    (tmp_path/'corrections/recovered').mkdir(parents=True)
    monkeypatch.setattr(runner, 'run', lambda scientific, *a: calls.append(scientific) or {'status': 'success'})
    result = run_selected(entry, tmp_path/'corrections', '/env', '/sources')
    assert result['status'] == 'search_timeout'
    assert result['partial_result_status'] == 'success'
    assert all(calls[0][key] == request[key] for key in request)
    assert calls[0]['recover_pysr_candidates'] == str(checkpoint)
    assert calls[0]['recovery_input_sha256'] == sha(checkpoint)
    checkpoint.write_text('changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        run_selected(entry, tmp_path/'corrections', '/env', '/sources')
    assert len(calls) == 1
