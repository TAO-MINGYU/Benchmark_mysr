import errno
from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import sys
import pytest

spec = importlib.util.spec_from_file_location('node_pool', Path(__file__).resolve().parents[1]/'scripts/external/node_pool.py')
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def test_enospc_replica_preserves_authoritative_journal_and_returns(tmp_path):
    journal = tmp_path/'events.jsonl'
    journal.write_text('{"event":"completed"}\n')
    future = Future()
    future.set_exception(OSError(errno.ENOSPC, 'No space left on device'))
    assert pool.finish_publication(future, tmp_path, 'status_replica') is False
    assert journal.read_text() == '{"event":"completed"}\n'
    error = json.loads((tmp_path/'publication-errors.jsonl').read_text())
    assert error['operation'] == 'status_replica'
    successful = Future()
    successful.set_result(None)
    assert pool.finish_publication(successful, tmp_path, 'status_replica') is True


def test_programming_error_is_not_hidden_as_storage_failure(tmp_path):
    future = Future()
    future.set_exception(ValueError('wrong identity'))
    with pytest.raises(ValueError, match='wrong identity'):
        pool.finish_publication(future, tmp_path, 'status_replica')


def test_pool_runs_all_units_when_first_status_replica_fills_disk(tmp_path, monkeypatch):
    manifest = tmp_path/'index.json'
    manifest.write_text(json.dumps({'fits': [{'method': 'gplearn', 'offset': i, 'sha256': 'frozen'}
                                            for i in range(2)]}))
    config = {'pool_output': str(tmp_path/'replica'), 'local_pool_root': str(tmp_path/'journal'),
              'minimum_available_memory_gib': 0, 'bundles': 'unused',
              'campaigns': [{'node': 'carbon', 'kind': 'comparison_fit', 'method': 'gplearn',
                             'code': str(tmp_path), 'environment': 'unused',
                             'manifest': str(manifest), 'output': str(tmp_path/'results')}]}
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv('SLURM_JOB_ID', '888')
    monkeypatch.setattr(pool.socket, 'gethostname', lambda: 'carbon')
    monkeypatch.setattr(pool.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(pool.subprocess, 'check_output', lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(pool, 'command', lambda *a: [
        sys.executable, '-c', 'print(\'{"status":"success"}\')'])
    monkeypatch.setattr(sys, 'argv', ['pool', '--config', str(config_path), '--node', 'carbon', '--workers', '1'])
    original_write = pool.write_json
    failures = []

    def flaky_write(path, value):
        if path == tmp_path/'replica/carbon/888/status.json' and not failures:
            failures.append(path)
            raise OSError(errno.ENOSPC, 'Injected replica failure')
        original_write(path, value)

    monkeypatch.setattr(pool, 'write_json', flaky_write)
    assert pool.main() == 0
    final = json.loads((tmp_path/'replica/carbon/888/status.json').read_text())
    assert final['completed'] == 2 and final['queued'] == final['active'] == 0
    events = [json.loads(line) for line in (tmp_path/'journal/888/events.jsonl').read_text().splitlines()]
    assert [row['index'] for row in events if row['event'] == 'completed'] == [0, 1]
    assert failures


def test_pool_covers_all_units_without_duplicating_ragged_groups():
    groups = [[('pysr', i) for i in range(315)], [('dsr', i) for i in range(314)], []]
    units = list(pool.interleave(groups))
    assert len(units) == len(set(units)) == 629
    assert units[:4] == [('pysr', 0), ('dsr', 0), ('pysr', 1), ('dsr', 1)]
    assert units[-1] == ('pysr', 314)


def test_pool_command_preserves_campaign_identity_and_pins_one_cpu():
    entry = dict(method='pysr', kind='ode', manifest='/frozen/manifest.json', index=7, output='/prior/results')
    cmd = pool.command(entry, dict(supervisor_python='/env/python', archive='/corpus'), 511, '/same/env', '/same/sources')
    assert cmd[:3] == ['taskset', '-c', '511']
    assert cmd[cmd.index('--index')+1] == '7'
    assert cmd[cmd.index('--output-root')+1] == '/prior/results'
    assert cmd[-1] == '--ode'


def test_pool_command_carries_only_explicit_retry_categories():
    entry = dict(method='gplearn', kind='tabular', manifest='/m', index=0, output='/o')
    cmd = pool.command(entry, dict(supervisor_python='/p', archive='/a',
                                   retry_statuses=['startup_timeout']), 1, '/e', '/s')
    assert cmd[-2:] == ['--retry-status', 'startup_timeout']


def test_stale_lock_cleanup_keeps_completed_results(tmp_path):
    import os
    import time
    unfinished = tmp_path/'unfinished'; unfinished.mkdir()
    complete = tmp_path/'complete'; complete.mkdir()
    (unfinished/'running.lock').write_text('')
    (complete/'running.lock').write_text('')
    (complete/'result.json').write_text('{}')
    old = time.time() - 1000
    os.utime(unfinished/'running.lock', (old, old))
    os.utime(complete/'running.lock', (old, old))
    removed = pool.clear_stale_locks([tmp_path])
    assert str(unfinished/'running.lock') in removed
    assert not (unfinished/'running.lock').exists()
    assert (complete/'running.lock').exists()


@pytest.mark.parametrize('kind', ['tabular', 'fit'])
def test_pool_executes_each_workunit_once_on_distinct_logical_cpus(tmp_path, kind):
    import json
    import os
    import socket
    import subprocess
    import sys
    import pytest
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip('requires two allowed CPUs')
    code = tmp_path/'code'
    package = code/'benchmark_mysr/external'
    package.mkdir(parents=True)
    (code/'benchmark_mysr/__init__.py').touch()
    (package/'__init__.py').touch()
    (package/'stage.py').write_text('print("/same/staged/root")\n')
    body = '''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--index',type=int)
p.add_argument('--selection',type=Path)
p.add_argument('--output-root',type=Path)
a,_=p.parse_known_args()
if a.selection is not None:a.index=json.loads(a.selection.read_text())['index']
time.sleep(0.1)
with (a.output_root/f'{a.index}.json').open('x') as f:
 json.dump({'affinity':sorted(os.sched_getaffinity(0))},f)
print(json.dumps({'status':'success'}),flush=True)
'''
    (package/'campaign.py').write_text(body)
    (package/'repair.py').write_text(body)
    output = tmp_path/'results'; output.mkdir()
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'workunits':[{'method_id':'fake'}]*3,
                                    'fits':[{'method':'fake'}]*3}))
    selections=tmp_path/'selection';selections.mkdir()
    for i in range(3):(selections/f'fake-{i}.json').write_text(json.dumps({'index':i}))
    config = tmp_path/'config.json'
    node = socket.gethostname().split('.')[0]
    config.write_text(json.dumps({'pool_output':str(tmp_path/'pool'), 'bundles':'/unused',
        'local_pool_root':str(tmp_path/'local-pool'),
        'archive':'/unused', 'supervisor_python':sys.executable,
        'fit_selection_root':str(selections),
        'minimum_available_memory_gib':0, 'campaigns':[{'node':node,
            'code':str(code), 'environment':'fake', 'method':'fake', 'kind':kind,
            'manifest':str(manifest), 'output':str(output)}]}))
    subprocess.run([sys.executable, pool.__file__, '--config', str(config),
                    '--node', node, '--workers','2'], check=True,
                   env=dict(os.environ,SLURM_JOB_ID='test'), timeout=15)
    results = [json.loads((output/f'{i}.json').read_text()) for i in range(3)]
    assert all(len(r['affinity'])==1 for r in results)
    assert results[0]['affinity'] != results[1]['affinity']
    status = json.loads((tmp_path/'pool'/node/'test/status.json').read_text())
    assert status['completed']==3 and status['failed_workunits']==0
    if kind == 'fit':
        assert status['fit_status_counts'] == {'success': 3}
