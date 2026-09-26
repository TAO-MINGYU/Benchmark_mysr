import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('node_pool', Path(__file__).resolve().parents[1]/'scripts/external/node_pool.py')
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


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


def test_pool_executes_each_workunit_once_on_distinct_logical_cpus(tmp_path):
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
    (package/'campaign.py').write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--index',type=int)
p.add_argument('--output-root',type=Path)
a,_=p.parse_known_args()
time.sleep(0.1)
with (a.output_root/f'{a.index}.json').open('x') as f:
 json.dump({'affinity':sorted(os.sched_getaffinity(0))},f)
''')
    output = tmp_path/'results'; output.mkdir()
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'workunits':[{'method_id':'fake'}]*3}))
    config = tmp_path/'config.json'
    node = socket.gethostname().split('.')[0]
    config.write_text(json.dumps({'pool_output':str(tmp_path/'pool'), 'bundles':'/unused',
        'local_pool_root':str(tmp_path/'local-pool'),
        'archive':'/unused', 'supervisor_python':sys.executable,
        'minimum_available_memory_gib':0, 'campaigns':[{'node':node,
            'code':str(code), 'environment':'fake', 'method':'fake', 'kind':'tabular',
            'manifest':str(manifest), 'output':str(output)}]}))
    subprocess.run([sys.executable, pool.__file__, '--config', str(config),
                    '--node', node, '--workers','2'], check=True,
                   env=dict(os.environ,SLURM_JOB_ID='test'), timeout=15)
    results = [json.loads((output/f'{i}.json').read_text()) for i in range(3)]
    assert all(len(r['affinity'])==1 for r in results)
    assert results[0]['affinity'] != results[1]['affinity']
    status = json.loads((tmp_path/'pool'/node/'test/status.json').read_text())
    assert status['completed']==3 and status['failed_workunits']==0
