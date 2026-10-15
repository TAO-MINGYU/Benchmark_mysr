"""Audit a complete comparison epoch and construct independent-ODE summaries."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .fair import protocol
from .worker import dump, read_split


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def audit_one(args):
    item, output_root, job, hashes = args
    request = item['request']
    path = Path(output_root)/item['relative_output']/'result.json'
    row = {'result':str(path),'relative_output':item['relative_output'],
           'method':request['method'],'variant':request['variant'],
           'resource_track':request['resource_track'],'task_id':request['task_id'],
           'component':request.get('component'),'dimension':request.get('ode_dimension'),
           'seed':request['seed'],'issues':[],'usable':False}
    row['family'] = ('ode' if request.get('component') is not None else
                     'srsd_dummy' if request['task_id'].startswith('srsd-dummy-') else
                     'srsd' if request['task_id'].startswith('srsd-') else
                     'legacy_classic' if request['task_id'].startswith('legacy-') else
                     'srbench_ground_truth' if request.get('ground_truth_available') else 'srbench_black_box')
    try:
        result = json.loads(path.read_text())
        for key in ('method','variant','resource_track','task_id','seed','budget',
                    'comparison_protocol','comparison_protocol_sha256'):
            if result.get(key) != request[key]:
                row['issues'].append('identity_changed:'+key)
        if result['data_hashes'] != request['expected_data_hashes']:
            row['issues'].append('data_hashes_changed')
        if result['adapter_hashes'] != hashes or result['slurm_job_id'] != job:
            row['issues'].append('unverified_writer_or_source')
        if result['status'] in ('solver_error','crash') or result.get('partial_result_status') in ('solver_error','crash'):
            row['issues'].append('implementation_or_runtime_fault_requires_review')
        row['status'] = result['status']
        row['usable'] = bool(result.get('selected_expression')) and all(
            isinstance(result.get(key),dict) and all(
                isinstance(result[key].get(metric),(int,float)) and math.isfinite(result[key][metric])
                and result[key][metric]>=0 for metric in ('mse','nrmse'))
            for key in ('train_score','validation_score','test_score'))
        if result['status']=='success' and not row['usable']:
            row['issues'].append('success_without_complete_finite_scores')
        frontier = result.get('full_frontier')
        if isinstance(frontier,str):
            if sha(path.parent/frontier) != result.get('frontier_sha256'):
                row['issues'].append('frontier_sha256_mismatch')
        artifact = result.get('test_prediction_artifact')
        if artifact:
            prediction = np.load(path.parent/artifact,allow_pickle=False)['prediction']
            finite = int(np.isfinite(prediction).sum())
            if finite != result['test_prediction_finite_count'] or prediction.size != result['test_prediction_count']:
                row['issues'].append('prediction_counts_mismatch')
            if row['usable']:
                _, observed, clean = read_split(Path(request['data'])/'test.csv')
                if prediction.shape != observed.shape or finite != prediction.size:
                    row['issues'].append('usable_prediction_shape_or_finiteness')
                for key,target in [('test_score',observed),('test_clean_score',clean)]:
                    mse = float(np.mean((prediction-target)**2))
                    nrmse = float(np.sqrt(mse)/max(float(np.std(target)),1e-15))
                    if not np.allclose([mse,nrmse],[result[key]['mse'],result[key]['nrmse']],rtol=1e-10,atol=1e-14):
                        row['issues'].append('saved_prediction_metric_mismatch:'+key)
        elif row['usable']:
            row['issues'].append('usable_without_saved_predictions')
        row.update(test_nrmse=result['test_score']['nrmse'] if row['usable'] else None,
                   selected_expression=result.get('selected_expression'),
                   cpu_time=result['cpu_time'],component_serial_elapsed_seconds=result['runtime'],
                   peak_memory=result['peak_memory'],full_frontier=frontier)
    except Exception as exc:
        row['issues'].append(type(exc).__name__+':'+str(exc))
    return row


def aggregate_one(args):
    relative, rows, output_root = args
    rows = sorted(rows,key=lambda r:r['component'])
    count = rows[0]['dimension']
    complete = len(rows)==count and [r['component'] for r in rows]==list(range(count))
    usable = complete and all(r['usable'] and not r['issues'] for r in rows)
    result = {'schema':'fair-ode-aggregate-v2','method':rows[0]['method'],
              'task_id':rows[0]['task_id'],'variant':rows[0]['variant'],
              'resource_track':rows[0]['resource_track'],'seed':rows[0]['seed'],
              'dimension':count,'components':rows,'complete':complete,'prediction_usable':usable,
              'metric':'macro_mean_of_component_nrmse_on_clean_independent_trajectory',
              'test_nrmse':sum(r['test_nrmse'] for r in rows)/count if usable else None,
              'selected_expression':[r.get('selected_expression') for r in rows],
              'component_serial_elapsed_seconds':sum(r.get('component_serial_elapsed_seconds',0) for r in rows),
              'cpu_time':sum(r.get('cpu_time',0) for r in rows),'formal_claim':False}
    destination = Path(output_root)/relative/'result.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    dump(destination,result)
    return complete


def finalize(requests, output_root, report_root, workers=16, expected_count=None,
             requests_sha256=None, expected_ode=None):
    if requests_sha256 is not None and sha(requests)!=requests_sha256:
        raise ValueError('Complete comparison request manifest checksum changed')
    root = Path(__file__).parent
    hashes = {path.name:sha(path) for path in root.glob('*.py')}
    policy = protocol()
    report_root = Path(report_root)
    report_root.mkdir(parents=True,exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID')
    counts, strata, groups, problems = Counter(), defaultdict(Counter), defaultdict(list), []
    total, usable = 0, 0
    with Path(requests).open() as stream, (report_root/'records.jsonl').open('w') as out:
        entries = (json.loads(line) for line in stream)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(audit_one,((item,output_root,job,hashes) for item in entries),chunksize=16):
                out.write(json.dumps(row,allow_nan=False)+'\n')
                total += 1
                usable += row['usable']
                counts[row.get('status','missing')] += 1
                key = (row['method'],row['family'],row['variant'],row['resource_track'])
                strata[key]['records'] += 1
                if row.get('status')!='not_applicable':
                    strata[key]['applicable'] += 1
                strata[key]['usable'] += row['usable']
                for t in policy['failure_metrics']['nrmse_thresholds']:
                    strata[key]['nrmse_le_'+str(t)] += bool(row['usable'] and row['test_nrmse']<=t)
                if row['issues']:
                    problems.append({'result':row['result'],'issues':row['issues']})
                if row['component'] is not None:
                    groups[str(Path(row['relative_output']).parent)].append(row)
            complete_systems = sum(pool.map(aggregate_one,((key,rows,output_root) for key,rows in groups.items()),chunksize=16))
    if expected_count is not None and total!=expected_count:
        problems.append({'issues':['expected_native_records_mismatch'],'expected':expected_count,'actual':total})
    if expected_ode is not None and len(groups)!=expected_ode:
        problems.append({'issues':['expected_ode_system_records_mismatch'],'expected':expected_ode,'actual':len(groups)})
    summary = {'native_records':total,'usable_predictions':usable,'statuses':dict(counts),
               'issues':problems,'ode_aggregates':len(groups),'complete_ode_aggregates':complete_systems,
               'strata':[{'method':key[0],'family':key[1],'variant':key[2],'resource_track':key[3],**value}
                         for key,value in sorted(strata.items())],
               'external_audit_passed':not problems and complete_systems==len(groups),
               'formal_comparison_ready':False,'mysr_results_present':False,
               'scope':'Full frozen external epoch only; MySR must use the same data/protocol and resource epoch.'}
    dump(report_root/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--report-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--expected-count',type=int,required=True)
    p.add_argument('--expected-ode',type=int,required=True)
    p.add_argument('--requests-sha256',required=True)
    a=p.parse_args()
    summary=finalize(a.requests,a.output_root,a.report_root,a.workers,a.expected_count,
                     a.requests_sha256,a.expected_ode)
    print(json.dumps({key:value for key,value in summary.items() if key not in ('strata','issues')},indent=2),flush=True)
    return 0 if summary['external_audit_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
