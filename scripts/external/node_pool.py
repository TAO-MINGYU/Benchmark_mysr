#!/usr/bin/env python3
"""Run frozen external work units on distinct logical CPUs in one Slurm job."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import time


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def interleave(groups):
    groups = [deque(g) for g in groups]
    while any(groups):
        for group in groups:
            if group:
                yield group.popleft()


def available_memory_gib():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) / 1024**2


def physical_cpus(cpus, topology=Path('/sys/devices/system/cpu')):
    """Use at most one allowed hardware thread per physical core."""
    seen, chosen = set(), []
    for cpu in cpus:
        directory = topology/f'cpu{cpu}'/'topology'
        key = tuple((directory/name).read_text().strip() for name in ('physical_package_id','core_id'))
        if key not in seen:
            seen.add(key)
            chosen.append(cpu)
    return chosen


def clear_stale_locks(roots, minimum_age_seconds=600):
    removed = []
    now = time.time()
    for root in roots:
        for lock in Path(root).rglob('running.lock'):
            if now - lock.stat().st_mtime < minimum_age_seconds:
                continue
            if not (lock.parent / 'result.json').exists():
                lock.unlink()
                removed.append(str(lock))
    return removed


def command(entry, config, cpu, env_root, sources):
    if entry['kind'] == 'comparison_fit':
        return ['taskset', '-c', str(cpu), config['supervisor_python'], '-m',
                'benchmark_mysr.external.fair', '--requests', config['requests'],
                '--offset', str(entry['request_offset']), '--sha256', entry['request_sha256'],
                '--output-root', entry['output'], '--env-root', str(env_root), '--sources', str(sources)]
    if entry['kind'] == 'fit':
        return ['taskset', '-c', str(cpu), config['supervisor_python'], '-m',
                'benchmark_mysr.external.repair', '--selection',
                str(Path(config['fit_selection_root']) / f"{entry['method']}-{entry['index']}.json"),
                '--output-root', entry['output'], '--env-root', str(env_root),
                '--sources', str(sources)]
    args = ['taskset', '-c', str(cpu), config['supervisor_python'], '-m',
            'benchmark_mysr.external.campaign', '--manifest', entry['manifest'],
            '--method', entry['method'], '--index', str(entry['index']),
            '--output-root', entry['output'], '--archive', config['archive'],
            '--sources', str(sources), '--env-root', str(env_root)]
    if entry['kind'] == 'ode':
        args.append('--ode')
    if config.get('selection_root'):
        args.extend(['--selection-manifest',str(Path(config['selection_root']) /
                     f"{entry['kind']}-{entry['method']}-{entry['index']}.json")])
    for status in config.get('retry_statuses', []):
        args.extend(['--retry-status', status])
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--node', required=True)
    parser.add_argument('--workers', type=int, default=512)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('comparison_requests_sha256'):
        def file_sha(path):
            digest=hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block)
            return digest.hexdigest()
        if file_sha(config['requests'])!=config['comparison_requests_sha256']:
            raise ValueError('Comparison requests changed before launch')
        if any(file_sha(entry['manifest'])!=config['comparison_index_sha256'] for entry in config['campaigns']):
            raise ValueError('Comparison dispatch index changed before launch')
    cpus = sorted(os.sched_getaffinity(0))
    if config.get('physical_cores_only'):
        cpus = physical_cpus(cpus)
    if args.workers > len(cpus):
        raise ValueError(f'{args.workers} workers requested but only {len(cpus)} CPUs allowed')
    if socket.gethostname().split('.')[0] != args.node:
        raise ValueError('Node differs from registered allocation')
    remote = Path(config['pool_output']) / args.node / os.environ['SLURM_JOB_ID']
    remote.mkdir(parents=True, exist_ok=False)
    output = Path(config.get('local_pool_root', f'/tmp/mysr-external-pools-{os.getuid()}')) / os.environ['SLURM_JOB_ID']
    output.mkdir(parents=True, exist_ok=False)
    assigned = [entry for entry in config['campaigns'] if entry['node'] == args.node]
    stale_locks = (clear_stale_locks(sorted({entry['output'] for entry in assigned}))
                   if config.get('resume_stale_locks') else [])
    if config.get('supervisor_bundles'):
        stage_script = Path(assigned[0]['code'])/'benchmark_mysr/external/stage.py'
        def stage_extra(name):
            return Path(subprocess.check_output([
                '/usr/bin/python3', str(stage_script), '--bundles', config['supervisor_bundles'],
                '--cache', f'/tmp/mysr-external-{os.getuid()}', '--name', name], text=True).strip()) / name
        supervisor = stage_extra('env_1_mysr')
        config['supervisor_python'] = str(supervisor/'bin/python')
        subprocess.run([config['supervisor_python'], '-c',
                        'import numpy,psutil; print("local supervisor",numpy.__version__,psutil.__version__)'], check=True)
        for entry in assigned:
            entry['original_code'] = entry['code']
            entry['code'] = str(stage_extra('adapter_'+entry['adapter_commit'][:7]))
    environments, sources = {}, None
    groups = []
    manifests = {}
    for entry in assigned:
        code = Path(entry['code'])
        for name in (entry['environment'], 'sources'):
            root = subprocess.check_output([
                '/usr/bin/python3', str(code/'benchmark_mysr/external/stage.py'),
                '--bundles', config['bundles'], '--cache', f'/tmp/mysr-external-{os.getuid()}',
                '--name', name], text=True).strip()
            if name == 'sources':
                sources = root
            else:
                environments[entry['method']] = root
        if entry['manifest'] not in manifests:
            manifests[entry['manifest']] = json.loads(Path(entry['manifest']).read_text())
        manifest = manifests[entry['manifest']]
        if entry['kind'] in ('fit','comparison_fit'):
            units = [u for u in manifest['fits'] if u['method'] == entry['method']]
        else:
            units = [u for u in manifest['workunits'] if u['method_id'] == entry['method']]
        groups.append([dict(entry, index=i, **({'request_offset':u['offset'], 'request_sha256':u['sha256']}
                                             if entry['kind']=='comparison_fit' else {}))
                       for i,u in enumerate(units)])
    queue = deque(interleave(groups))
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    write_json(output/'allocation.json', {
        'node': args.node, 'job_id': os.environ['SLURM_JOB_ID'], 'cpus': cpus[:args.workers],
        'workers': args.workers, 'config_sha256': config_hash, 'unit_count': len(queue),
        'single_thread_per_fit': True,
        'physical_cores_only': config.get('physical_cores_only', False),
        'resource_epoch': config.get('resource_epoch', 'full-node-smt-v1'),
        'minimum_available_memory_gib': config['minimum_available_memory_gib'],
        'started_at': time.time(), 'node_local_logs': str(output),
        'stale_locks_removed': stale_locks,
    })
    shutil.copy2(output/'allocation.json', remote/'allocation.json')
    publisher = ThreadPoolExecutor(max_workers=1)
    log_publisher = ThreadPoolExecutor(max_workers=4)
    log_copies = []
    publication = None
    last_publication = 0

    def publish(status, event_text):
        write_json(remote/'status.json', status)
        temp = remote/'events.jsonl.tmp'
        temp.write_text(event_text)
        temp.replace(remote/'events.jsonl')
    free = deque(cpus[:args.workers])
    active, completed, failures = {}, 0, 0
    fit_statuses = Counter()
    stop = False

    def stopping(*unused):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    with (output/'events.jsonl').open('a', buffering=1) as events:
        while queue or active:
            for cpu, item in list(active.items()):
                proc, entry, log, start = item
                if proc.poll() is not None:
                    events.write(json.dumps({'event': 'completed', 'time': time.time(),
                        'cpu': cpu, 'method': entry['method'], 'kind': entry['kind'],
                        'index': entry['index'], 'returncode': proc.returncode,
                        'elapsed': time.monotonic()-start})+'\n')
                    completed += 1
                    failures += proc.returncode != 0
                    log.close()
                    if entry['kind'] in ('fit','comparison_fit'):
                        try:
                            with Path(log.name).open('rb') as handle:
                                handle.seek(max(0, Path(log.name).stat().st_size - 4096))
                                summary = json.loads(handle.read().splitlines()[-1])
                            fit_statuses[summary['status']] += 1
                        except (OSError, ValueError, KeyError, IndexError):
                            fit_statuses['unavailable'] += 1
                    log_copies.append(log_publisher.submit(shutil.copy2, log.name, remote/Path(log.name).name))
                    del active[cpu]
                    free.append(cpu)
            memory = available_memory_gib()
            started = 0
            while (not stop and queue and free and
                   memory >= config['minimum_available_memory_gib'] and
                   started < config.get('starts_per_second', 8)):
                entry, cpu = queue.popleft(), free.popleft()
                env = dict(os.environ, PYTHONPATH=entry['code'], OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', JULIA_NUM_THREADS='1',
                           NUMEXPR_NUM_THREADS='1', PYTHON_JULIACALL_THREADS='1')
                log = (output/f"{entry['kind']}-{entry['method']}-{entry['index']}.log").open('a')
                proc = subprocess.Popen(command(entry, config, cpu, environments[entry['method']], sources),
                                        cwd=entry['code'], env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
                active[cpu] = proc, entry, log, time.monotonic()
                events.write(json.dumps({'event': 'started', 'time': time.time(),
                    'cpu': cpu, 'pid': proc.pid, 'method': entry['method'],
                    'kind': entry['kind'], 'index': entry['index'], 'code': entry['code']})+'\n')
                started += 1
            status = {'time': time.time(), 'active': len(active),
                'queued': len(queue), 'completed': completed, 'failed_workunits': failures,
                'fit_status_counts': dict(fit_statuses),
                'available_memory_gib': memory, 'stopping': stop,
                'active_workunits': [{'cpu': cpu, 'pid': x[0].pid, 'method': x[1]['method'],
                    'kind': x[1]['kind'], 'index': x[1]['index']} for cpu, x in active.items()]}
            write_json(output/'status.json', status)
            final = stop or not (queue or active)
            if final and publication is not None:
                publication.result()
            if final or ((publication is None or publication.done()) and time.monotonic()-last_publication >= 10):
                if publication is not None:
                    publication.result()
                publication = publisher.submit(publish, status, (output/'events.jsonl').read_text())
                last_publication = time.monotonic()
            if stop:
                # Slurm owns the cgroup, including detached native solver descendants.
                # Leave interruption evidence for explicit, audited resume preparation.
                publication.result()
                publisher.shutdown()
                log_publisher.shutdown()
                for copy in log_copies:
                    copy.result()
                return 143
            if queue or active:
                time.sleep(1)
    publication.result()
    publisher.shutdown()
    log_publisher.shutdown()
    for copy in log_copies:
        copy.result()
    if config.get('finalize_comparison'):
        env = dict(os.environ,PYTHONPATH=assigned[0]['code'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        result = subprocess.run([config['supervisor_python'],'-m','benchmark_mysr.external.fair_finalize',
            '--requests',config['requests'],'--output-root',assigned[0]['output'],
            '--report-root',config['comparison_report_root'],'--workers',str(min(args.workers,16)),
            '--expected-count',str(config['expected_fit_count']),
            '--expected-ode',str(config['expected_ode_count']),
            '--requests-sha256',config['comparison_requests_sha256']],env=env)
        if result.returncode:
            failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
