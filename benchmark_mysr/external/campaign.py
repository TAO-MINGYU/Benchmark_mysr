"""Execute registered tabular or system-level ODE work units, with resume."""
import argparse
import json
import os
import hashlib
from pathlib import Path

from .runner import BUDGETS, run
from .worker import dump
from .units import input_units


def workunit(manifest, method, index, output, archive, sources, env_root, ode=False,
             retry_statuses=(), selection=None):
    units=[u for u in manifest['workunits'] if u['method_id']==method]
    item=units[index]
    selected = None
    if selection is not None:
        if selection['workunit_id'] != item['workunit_id']:
            raise ValueError('Repair selection does not match workunit identity')
        selected = {(r['variant'],r['resource_track'],r['seed'],r.get('component')):r
                    for r in selection['records']}
        if len(selected) != len(selection['records']):
            raise ValueError('Duplicate selected repair records')

    def verify_prior(row):
        prior = Path(row['baseline_result'])
        if hashlib.sha256(prior.read_bytes()).hexdigest() != row['baseline_sha256']:
            raise ValueError('Baseline result changed after repair selection')
        return row['data_hashes']
    root=Path(output)/method
    if ode:
        from .ode import build
        system_index=item['system_index']
        root=root/item['condition']/f'system-{system_index+1}'
        meta=build(archive/item['file'],system_index,root/'data',
                   archive/'raw/odebench/odebench_noiseless_nodrop_150pts.json')
        task=meta['task_id']
        variants=[item['condition']]
    else:
        task=item['task_id']
        root=root/task
        variants=manifest['variants']
    summary=[]
    for variant in variants:
        for track in manifest['resource_tracks']:
            for seed in manifest['formal_search_seeds']:
                key = variant,track,seed
                if selected is not None and not any(k[:3] == key for k in selected):
                    continue
                out=root/variant/track/str(seed)
                budget=dict(BUDGETS[track])
                request={'method':method,'task_id':task,'seed':seed,'variant':variant,'resource_track':track,
                         'budget':budget,'source_root':str(sources),'ground_truth_available':item.get('ground_truth_available',True)}
                request['retry_statuses'] = list(retry_statuses)
                if ode:
                    components=[]
                    # A system-level budget is divided equally between its components.
                    for j,data in enumerate(meta['component_dirs']):
                        row = selected.get((*key,j)) if selected is not None else None
                        if selected is not None and row is None:
                            prior = Path(selection['baseline_workunit_root'])/variant/track/str(seed)/f'component-{j}'
                            destination = out/f'component-{j}'
                            destination.parent.mkdir(parents=True,exist_ok=True)
                            if not destination.exists():
                                destination.symlink_to(prior, target_is_directory=True)
                            components.append(json.loads((prior/'result.json').read_text()))
                            continue
                        part=dict(request, data=data, component=j, ode_policy=meta['policy'])
                        part['budget']=dict(budget, evaluations=max(1,budget['evaluations']//meta['dimension']),
                                            search_seconds=budget['search_seconds']/meta['dimension'])
                        if row is not None:
                            part['expected_data_hashes'] = verify_prior(row)
                        components.append(run(part,out/f'component-{j}',env_root))
                    statuses = [c['status'] for c in components]
                    status = ('success' if all(s == 'success' for s in statuses) else
                              'not_applicable' if all(s == 'not_applicable' for s in statuses) else
                              'timeout' if any(s.endswith('_timeout') for s in statuses) else 'incomplete_system')
                    r={'status':status,
                       'task_id':task,'method':method,'variant':variant,'seed':seed,'resource_track':track,
                       'components':[{'component':j,'status':c['status'],'result':f'component-{j}/result.json'} for j,c in enumerate(components)],
                       'budget':budget,'metric':meta['metric'],'formal_claim':False}
                    r.update(selected_expression=[c.get('selected_expression') for c in components],
                             full_frontier=[f'component-{j}/' + c['full_frontier'] if isinstance(c.get('full_frontier'),str) else None for j,c in enumerate(components)],
                             runtime=sum(c['runtime'] for c in components), cpu_time=sum(c['cpu_time'] for c in components),
                             peak_memory=max(c['peak_memory'] for c in components),
                             evaluations=sum(c['evaluations'] for c in components) if all(c['evaluations'] is not None for c in components) else None,
                             evaluation_semantics='sum_of_component_native_counts_when_available',
                             failure_status=status if status in ('success','not_applicable','timeout') else 'invalid_output',
                             failure_reason='; '.join(c.get('reason','') for c in components),
                             complexity=[c.get('complexity') for c in components],
                             train_score=[c.get('train_score') for c in components],
                             validation_score=[c.get('validation_score') for c in components],
                             test_score=[c.get('test_score') for c in components])
                    from benchmark_mysr.methods import validate_result_record
                    validate_result_record(r)
                    dump(out/'result.json',r)
                else:
                    row = selected.get((*key,None)) if selected is not None else None
                    if selected is not None and row is None:
                        raise ValueError('Invalid tabular selection component')
                    request.update(data=str(archive/item['data_path']/variant),
                                   input_units=input_units(str(archive),task) if method == 'tf4sr' else [])
                    if row is not None:
                        request['expected_data_hashes'] = verify_prior(row)
                    r=run(request,out,env_root)
                summary.append({'variant':variant,'resource_track':track,'seed':seed,'status':r['status']})
                dump(root/'progress.json',{'workunit':item,'completed_count':len(summary),'records':summary})
                print(json.dumps(summary[-1]),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--env-root',type=Path,required=True)
    p.add_argument('--ode',action='store_true')
    p.add_argument('--retry-status', action='append', default=[])
    p.add_argument('--selection-manifest',type=Path)
    a=p.parse_args()
    workunit(json.loads(a.manifest.read_text()),a.method,a.index,a.output_root,a.archive,a.sources,a.env_root,a.ode,
             a.retry_status, json.loads(a.selection_manifest.read_text()) if a.selection_manifest else None)


if __name__=='__main__':
    main()
