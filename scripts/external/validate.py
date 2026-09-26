"""Small real fits on both nodes using disjoint pilot seeds."""
import json
import os
from pathlib import Path
from benchmark_mysr.external.runner import run
from benchmark_mysr.external.ode import build


def main():
    root=Path(os.environ['VALIDATION_OUTPUT'])
    archive=Path(os.environ['EXTERNAL_ARCHIVE'])
    method=os.environ['METHOD']
    source=os.environ['STAGED_SOURCES']
    envroot=os.environ['STAGED_ENV_ROOT']
    task='legacy-classic-nguyen_1'
    base=archive/'processed/material-A'/task
    if method=='tf4sr':
        task='srsd-feynman-easy-feynman-i.12.1'
        base=archive/'processed/material-B'/task
    request={'method':method,'seed':104729,'budget':{'evaluations':200,'search_seconds':35,'memory_gib':8},
             'task_id':task,'ground_truth_available':True,'source_root':source}
    for label,variant in [('clean','clean'),('repeat','clean'),('noisy','output_noise_05')]:
        r=run(dict(request,data=str(base/variant),variant=variant),root/method/label,envroot)
        print(json.dumps({'method':method,'case':label,'status':r['status'],'partial':r.get('partial_result_status')}),flush=True)
    if method in ('pysr','operon','dsr','gplearn','ai_feynman_2'):
        meta=build(archive/'raw/odebench/odebench_noiseless_nodrop_150pts.json',0,root/method/'ode-data',
                   archive/'raw/odebench/odebench_noiseless_nodrop_150pts.json')
        r=run(dict(request,data=meta['component_dirs'][0],task_id=meta['task_id']),root/method/'ode',envroot)
        print(json.dumps({'method':method,'case':'ode','status':r['status']}),flush=True)


if __name__=='__main__':
    main()
