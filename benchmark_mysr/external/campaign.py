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
                    r={'status':'success' if all(c['status']=='success' for c in components) else 'incomplete_system',
                       'task_id':task,'method':method,'variant':variant,'seed':seed,'resource_track':track,
                       'components':[{'component':j,'status':c['status'],'result':f'component-{j}/result.json'} for j,c in enumerate(components)],
                       'budget':budget,'metric':meta['metric'],'formal_claim':False}
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
