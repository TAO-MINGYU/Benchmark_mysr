"""Execute registered tabular or system-level ODE work units, with resume."""
import argparse
import json
import os
from pathlib import Path

from .runner import BUDGETS, run
from .worker import dump


def input_units(archive, task):
    if not task.startswith('srsd-'):
        return []
    for difficulty in ('easy','medium','hard'):
        marker=f'-{difficulty}-'
        if marker in task:
            key=task.split(marker,1)[1]
            metadata=json.loads((archive/f'raw/srsd-published-data/{difficulty}/supp_info.json').read_text())
            return metadata[key].get('si-derived_units',[])[1:]
    return []


def workunit(manifest, method, index, output, archive, sources, env_root, ode=False):
    units=[u for u in manifest['workunits'] if u['method_id']==method]
    item=units[index]
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
                out=root/variant/track/str(seed)
                budget=dict(BUDGETS[track])
                request={'method':method,'task_id':task,'seed':seed,'variant':variant,'resource_track':track,
                         'budget':budget,'source_root':str(sources),'ground_truth_available':item.get('ground_truth_available',True)}
                if ode:
                    components=[]
                    # A system-level budget is divided equally between its components.
                    for j,data in enumerate(meta['component_dirs']):
                        part=dict(request, data=data, component=j, ode_policy=meta['policy'])
                        part['budget']=dict(budget, evaluations=max(1,budget['evaluations']//meta['dimension']),
                                            search_seconds=budget['search_seconds']/meta['dimension'])
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
                    request.update(data=str(archive/item['data_path']/variant),input_units=input_units(archive,task))
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
    a=p.parse_args()
    workunit(json.loads(a.manifest.read_text()),a.method,a.index,a.output_root,a.archive,a.sources,a.env_root,a.ode)


if __name__=='__main__':
    main()
