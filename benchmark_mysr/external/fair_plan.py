"""Snapshot the entire existing external matrix for a new comparison epoch."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from .fair import PROTOCOL, protocol
from .ode import build
from .units import input_units
from .worker import dump


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ode_case(args):
    archive, run, condition, file, index = args
    output = run/'data/ode'/condition/f'system-{index+1}'
    meta = build(archive/file,index,output,archive/'raw/odebench/odebench_noiseless_nodrop_150pts.json')
    return condition,index,meta


def prepare(tabular, ode, archive, run, workers=8):
    archive, run = Path(archive).resolve(), Path(run).resolve()
    run.mkdir(parents=True,exist_ok=True)
    if (run/'requests.jsonl').exists() or (run/'data').exists():
        raise ValueError('Refusing to overwrite an existing comparison preparation')
    policy = protocol()
    protocol_sha = digest(PROTOCOL)
    tasks = {item['task_id']:item for item in tabular['workunits']}
    cases = {}
    for task, item in tasks.items():
        units = input_units(str(archive),task) if task.startswith('srsd-') else []
        for variant in tabular['variants']:
            source = archive/item['data_path']/variant
            dest = run/'data/tabular'/task/variant
            dest.mkdir(parents=True)
            expected = json.loads((source/'manifest.json').read_text()).get('checksums',{})
            checksums = {}
            for split in ('train','validation','test'):
                name = split+'.csv'
                shutil.copy2(source/name,dest/name)
                checksums[split] = digest(dest/name)
            if expected and expected != checksums:
                raise ValueError('Tabular corpus split checksum mismatch: '+task)
            shutil.copy2(source/'manifest.json',dest/'manifest.json')
            cases[task,variant] = {'data':str(dest),'expected_data_hashes':checksums,'input_units':units}
    conditions = {item['condition']:item['file'] for item in ode['workunits']}
    # Original registration grouped systems into 30 arrays; expand all systems,
    # not only the incomplete/failed workunits of any continuation manifest.
    count = ode['system_count']
    args = [(archive,run,condition,file,index)
            for condition,file in conditions.items() for index in range(count)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        systems = {(condition,index):meta for condition,index,meta in pool.map(ode_case,args)}
    system_cases = {}
    for (condition,index), meta in systems.items():
        for component, path in enumerate(meta['component_dirs']):
            checksums = {split:digest(Path(path)/(split+'.csv')) for split in ('train','validation','test')}
            system_cases[condition,index,component] = {'data':path,'expected_data_hashes':checksums,
                'input_units':[],'ode_policy':meta['policy'],'component':component,
                'ode_dimension':meta['dimension'],'system_index':index,'system_id':meta['system_id']}
    index_rows, counts = [], Counter()
    with (run/'requests.jsonl').open('wb') as stream:
        def emit(method,kind,task,variant,track,seed,case,relative,ground_truth):
            request = {**case,'method':method,'task_id':task,'variant':variant,
                'resource_track':track,'seed':seed,'budget':policy['tracks'][track],
                'ground_truth_available':ground_truth,'comparison_protocol':policy['protocol_id'],
                'comparison_protocol_sha256':protocol_sha}
            raw = (json.dumps({'request':request,'relative_output':relative},sort_keys=True)+'\n').encode()
            index_rows.append({'method':method,'kind':kind,'offset':stream.tell(),
                               'sha256':hashlib.sha256(raw).hexdigest(),'relative_output':relative})
            stream.write(raw)
            counts[kind,method] += 1
        for item in tabular['workunits']:
            method, task = item['method_id'], item['task_id']
            for variant in tabular['variants']:
                for track in tabular['resource_tracks']:
                    for seed in tabular['formal_search_seeds']:
                        emit(method,'tabular',task,variant,track,seed,cases[task,variant],
                             f'tabular/{method}/{task}/{variant}/{track}/{seed}',item.get('ground_truth_available',True))
        for method in ode['methods']:
            for (condition,index), meta in sorted(systems.items()):
                for track in ode['resource_tracks']:
                    for seed in ode['formal_search_seeds']:
                        for component in range(meta['dimension']):
                            emit(method,'ode',meta['task_id'],condition,track,seed,
                                system_cases[condition,index,component],
                                f'ode/{method}/{condition}/system-{index+1}/{condition}/{track}/{seed}/component-{component}',True)
    if len({row['relative_output'] for row in index_rows}) != len(index_rows):
        raise ValueError('Duplicate matrix identities')
    expected_tabular = tabular['run_count']
    expected_ode = (sum(meta['dimension'] for meta in systems.values())*len(ode['methods'])*
                    len(ode['resource_tracks'])*len(ode['formal_search_seeds']))
    assert sum(v for (kind,_),v in counts.items() if kind=='tabular') == expected_tabular
    assert sum(v for (kind,_),v in counts.items() if kind=='ode') == expected_ode
    dump(run/'fits-index.json',{'schema':'complete-comparison-fit-index-v2','fits':index_rows})
    shutil.copy2(PROTOCOL,run/'comparison-protocol.json')
    summary = {'native_fits':len(index_rows),'tabular_fits':expected_tabular,'ode_component_fits':expected_ode,
        'tabular_tasks':len(tasks),'ode_systems':count,'ode_conditions':len(conditions),
        'system_level_ode_slots':ode['run_count'],'top_level_slots':expected_tabular+ode['run_count'],
        'by_kind_method':{kind+':'+method:value for (kind,method),value in counts.items()},
        'data_cases':len(cases)+len(system_cases),'requests_sha256':digest(run/'requests.jsonl'),
        'index_sha256':digest(run/'fits-index.json'),'protocol_sha256':protocol_sha,
        'seeds':tabular['formal_search_seeds'],'variants':tabular['variants'],
        'methods':tabular['methods'],'mysr_execution_included':False,'fair_mysr_contract_frozen':True,
        'source_archive':str(archive),'previous_results_reused':False}
    dump(run/'matrix-summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tabular-manifest',type=Path,required=True)
    p.add_argument('--ode-manifest',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    print(json.dumps(prepare(json.loads(a.tabular_manifest.read_text()),json.loads(a.ode_manifest.read_text()),
                             a.archive,a.run_root,a.workers),indent=2),flush=True)


if __name__=='__main__':
    main()
