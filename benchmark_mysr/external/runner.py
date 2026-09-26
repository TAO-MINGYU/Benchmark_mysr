"""Process-group supervision, immutable run requests and bounded resource use."""
import argparse
import hashlib
import json
import os
import resource
import signal
import shutil
import tempfile
import socket
import subprocess
import sys
import time
from pathlib import Path

from .worker import dump

ENVIRONMENTS = {'pysr': 'env_1_pysr', 'operon': 'parallel_operon', 'dsr': 'parallel_dsr',
                'ai_feynman_2': 'parallel_ai_feynman', 'gplearn': 'parallel_gplearn', 'tf4sr': 'parallel_tf4sr'}
BUDGETS = {'constrained_resource': {'evaluations': 20000, 'search_seconds': 120, 'memory_gib': 8},
           'capability_ceiling': {'evaluations': 500000, 'search_seconds': 900, 'memory_gib': 32}}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(method, env_root, code_root):
    prefix = Path(env_root) / ENVIRONMENTS[method]
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','JULIA_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS','PYTHON_JULIACALL_THREADS'):
        env[name] = '1'
    env.update(PYTHONPATH=str(code_root), PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='',
               MPLCONFIGDIR=str(Path.cwd() / 'mpl-cache'))
    env['PATH'] = str(prefix / 'bin') + os.pathsep + env.get('PATH','')
    if method == 'pysr':
        for k in ('PYTHON_JULIAPKG_EXE', 'PYTHON_JULIACALL_EXE'):
            env[k] = str(prefix / 'opt/julia-1.10.3/bin/julia')
        for k in ('PYTHON_JULIAPKG_PROJECT', 'PYTHON_JULIACALL_PROJECT'):
            env[k] = str(prefix / 'julia_depot/environments/pyjuliapkg')
        env.update(JULIA_DEPOT_PATH=str(prefix / 'julia_depot'), PYTHON_JULIAPKG_OFFLINE='yes',
                   JULIA_PKG_OFFLINE='true', PYTHON_JULIACALL_HANDLE_SIGNALS='yes')
    return prefix / 'bin/python', env


def run(request, output, env_root, startup_seconds=300, scoring_seconds=180):
    import psutil
    output = Path(output).resolve()
    request = dict(request)
    request['data'] = str(Path(request['data']).resolve())
    request['data_hashes'] = {split: digest(Path(request['data']) / f'{split}.csv') for split in ('train','validation','test')}
    code_root = Path(__file__).resolve().parents[2]
    metadata_path = Path(request['data'])/'manifest.json'
    if metadata_path.exists():
        checksums = json.loads(metadata_path.read_text()).get('checksums', {})
        if checksums and checksums != request['data_hashes']:
            raise ValueError('Input split checksums differ from the materialized manifest')
    request['adapter_hashes'] = {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')}
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    final = output / 'result.json'
    if final.exists():
        prior = json.loads(final.read_text())
        if prior['request_sha256'] != request_hash:
            raise ValueError(f'Refusing to reuse changed request at {output}')
        return prior
    # Prevent concurrent jobs writing the same run.
    lock = output / 'running.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    proc = None
    scratch = tempfile.TemporaryDirectory(prefix='mysr-solver-')
    working = Path(scratch.name)
    try:
        dump(output / 'request.json', request)
        python, env = environment(request['method'], env_root, code_root)
        env['MPLCONFIGDIR'] = str(working / 'mpl-cache')
        begun, peak, cpu, timed_stage = time.monotonic(), 0, 0.0, None
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        with (output/'stdout.log').open('w') as stdout, (output/'stderr.log').open('w') as stderr:
            proc = subprocess.Popen([str(python), '-m', 'benchmark_mysr.external.worker', str(output/'request.json')],
                                    cwd=working, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            watch = psutil.Process(proc.pid)
            stage, stage_start = 'startup', begun
            stage_info = {}
            while proc.poll() is None:
                stage_file = working / 'stage.json'
                if stage_file.exists():
                    info = json.loads(stage_file.read_text())
                    if info['stage'] != stage:
                        stage, stage_start = info['stage'], time.monotonic()
                        stage_info.update(info)
                processes = []
                try:
                    processes = [watch] + watch.children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
                rss, used = 0, 0.0
                for p in processes:
                    try:
                        rss += p.memory_info().rss
                        t = p.cpu_times()
                        used += t.user + t.system
                    except psutil.NoSuchProcess:
                        pass
                peak, cpu = max(peak,rss), max(cpu,used)
                limit = {'startup': startup_seconds, 'search': request['budget']['search_seconds'], 'scoring': scoring_seconds}[stage]
                if rss > request['budget']['memory_gib'] * 1024**3:
                    timed_stage = 'memory_limit'
                elif time.monotonic()-stage_start > limit + (2 if stage=='search' else 0):
                    timed_stage = stage + '_timeout'
                if timed_stage:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
                time.sleep(0.2)
            proc.wait()
            # Solvers may have left Fortran or Julia descendants after exiting.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_accounted = (usage_after.ru_utime + usage_after.ru_stime - usage_before.ru_utime - usage_before.ru_stime)
        shutil.copytree(working, output, dirs_exist_ok=True)
        payload_path = output/'worker-result.json'
        result = json.loads(payload_path.read_text()) if payload_path.exists() else {}
        if timed_stage or not result:
            result.update(status=timed_stage or 'crash', failure_status=timed_stage or f'exit_{proc.returncode}')
        result.update(schema_version='external-run-v1', request_sha256=request_hash,
                      method=request['method'], task_id=request['task_id'], seed=request['seed'],
                      variant=request.get('variant'), resource_track=request.get('resource_track'),
                      budget=request['budget'], data_hashes=request['data_hashes'], adapter_hashes=request['adapter_hashes'],
                      hostname=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                      elapsed_seconds=time.monotonic()-begun, peak_memory_bytes=peak, cpu_seconds_sampled=cpu, cpu_seconds_accounted=cpu_accounted,
                      resource_measurement='waited_child_cpu_accounting_and_process_tree_rss_sampled_200ms_no_slurm_memory_claim',
                      stage=stage_info, formal_claim=False,
                      evidence_scope='registered_external_native_solver_run_pending_campaign_audit')
        checkpoint = output/'native-frontier.json'
        if timed_stage == 'search_timeout' and checkpoint.exists() and not request.get('recover_candidates'):
            recovery_request = dict(request, recover_candidates=str(checkpoint))
            recovered = run(recovery_request, output/'recovery', env_root, startup_seconds, scoring_seconds)
            result['partial_result_status'] = recovered['status']
            result['partial_result'] = 'recovery/result.json'
            result['native_frontier'] = 'native-frontier.json'
            if recovered['status'] == 'success':
                for key in ('selected_expression','train_score','validation_score','test_score','test_clean_score','complexity','details'):
                    result[key] = recovered.get(key)
                result['full_frontier'] = 'recovery/frontier.json'
            result['peak_memory_bytes'] = max(peak, recovered['peak_memory_bytes'])
            result['cpu_seconds_sampled'] += recovered['cpu_seconds_sampled']
            result['cpu_seconds_accounted'] += recovered['cpu_seconds_accounted']
        for key in ('selected_expression','train_score','validation_score','test_score','complexity'):
            result.setdefault(key, None)
        result.setdefault('full_frontier', [])
        result['elapsed_seconds'] = time.monotonic()-begun
        result.update(method_id=request['method'], runtime=result['elapsed_seconds'],
                      cpu_time=result['cpu_seconds_accounted'], peak_memory=result['peak_memory_bytes'],
                      evaluations=result.get('details',{}).get('evaluation_count'),
                      evaluation_semantics=result.get('details',{}).get('evaluation_semantics','not_observed'),
                      environment_prefix=str(python.parent.parent), source_root=request.get('source_root'))
        result.setdefault('startup_seconds', stage_info.get('startup_seconds'))
        detail = result.get('failure_status')
        status = result['status']
        result['failure_status'] = ('timeout' if status.endswith('_timeout') else
                                    status if status in ('success','not_applicable','memory_limit','crash') else 'invalid_output')
        if detail:
            result['failure_detail'] = detail
        if status == 'not_applicable':
            result['failure_reason'] = result.get('reason')
            result['evaluations'] = 0
        from benchmark_mysr.methods import validate_result_record
        validate_result_record(result)
        dump(final, result)
        return result
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        scratch.cleanup()
        lock.unlink(missing_ok=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--request', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--env-root', type=Path, required=True)
    a=ap.parse_args()
    r=run(json.loads(a.request.read_text()), a.output, a.env_root)
    print(json.dumps(r, sort_keys=True))
    return 0 if r['status'] in ('success','not_applicable') else 1


if __name__ == '__main__':
    raise SystemExit(main())
