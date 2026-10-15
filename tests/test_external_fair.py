import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmark_mysr.external.fair import PROTOCOL, TrainScale, protocol, run_fit
from benchmark_mysr.external.solvers import dsr_config, legal_tf4sr_tokens


def request(method='gplearn', track='constrained_resource'):
    return {'method': method, 'task_id': 'fixture', 'variant': 'clean', 'seed': 7,
            'resource_track': track, 'budget': protocol()['tracks'][track],
            'comparison_protocol': protocol()['protocol_id'],
            'comparison_protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}


def test_train_scaling_handles_large_tiny_zero_and_radian_columns():
    X = np.array([[1e30, 0, 2], [2e30, 0, -3]])
    y = np.array([1e-30, -2e-30])
    scale = TrainScale(X, y, ['kg', None, '$rad$'])
    assert np.allclose(scale.inputs(X)[:, 0], [.5, 1])
    assert np.array_equal(scale.inputs(X)[:, 1:], X[:, 1:])
    assert np.allclose(scale.targets(y), [.5, -1])
    assert scale.record()['fit_split'] == 'train_observed_only'
    candidates = scale.candidates([('x0+x2', 3, lambda Z: Z[:, 0]+Z[:, 2])])
    assert np.allclose(candidates[0][2](X), 2e-30*(X[:, 0]/2e30+X[:, 2]))
    assert 'np.float' not in candidates[0][0]


def test_worker_fits_scaled_observed_training_only_and_scores_original_units(tmp_path, monkeypatch):
    import benchmark_mysr.external.worker as worker
    import benchmark_mysr.external.solvers as solvers
    for name, rows in [('train', [(2,20),(4,40)]), ('validation', [(6,60)]), ('test', [(8,80)])]:
        (tmp_path/(name+'.csv')).write_text('row_id,x_0,target,target_clean\n'+''.join(
            f'{i},{x},{y},999\n' for i,(x,y) in enumerate(rows)))
    req = dict(request(), data=str(tmp_path), source_root=str(tmp_path))
    path = tmp_path/'request.json'
    path.write_text(json.dumps(req))
    def fit(method, api, X, y, req):
        assert np.array_equal(X[:, 0], [.5,1])
        assert np.array_equal(y, [.5,1])
        assert not (tmp_path/'test-predictions.npz').exists()
        return [('x0',1,lambda Z:Z[:,0])], {}
    monkeypatch.setattr(solvers,'prepare',lambda *a:None)
    monkeypatch.setattr(solvers,'fit',fit)
    monkeypatch.setattr(sys,'argv',['worker',str(path)])
    monkeypatch.chdir(tmp_path)
    assert worker.main()==0
    result=json.loads((tmp_path/'worker-result.json').read_text())
    assert result['test_score']['mse']==0
    assert result['test_clean_score']['mse']>0
    assert np.load(tmp_path/'test-predictions.npz')['prediction'].tolist()==[80]
    assert result['details']['preprocessing']['input_scales']==[4]


@pytest.mark.parametrize('track,seconds',[('constrained_resource',120),('capability_ceiling',900)])
def test_shared_budget_has_no_incomparable_evaluation_or_memory_cap(track, seconds):
    req=request('dsr',track)
    assert req['budget']=={'search_seconds':seconds,'evaluations':None,'memory_gib':None}
    config=dsr_config(req)
    assert config['training']['n_samples']>10**12
    assert config['training']['batch_size']==100
    assert config['training']['n_cores_batch']==1
    assert 'sqrt' in config['task']['function_set']


def test_dsr_exports_native_sqrt_without_symbolic_rewriting():
    from types import SimpleNamespace
    from benchmark_mysr.external.solvers import dsr_expression
    program = SimpleNamespace(traversal=[SimpleNamespace(name='sqrt',arity=1),
                SimpleNamespace(name='x1',arity=0,input_var=0)])
    assert dsr_expression(program)=='sqrt(x0)'


def test_indexed_fit_rejects_changed_protocol_budget_or_request(tmp_path, monkeypatch):
    from benchmark_mysr.external import runner
    calls=[]
    monkeypatch.setattr(runner,'run',lambda req,*a:calls.append(req) or {'status':'success'})
    req=request()
    item={'relative_output':'tabular/gplearn/fixture','request':req}
    raw=(json.dumps(item)+'\n').encode()
    path=tmp_path/'requests.jsonl';prefix=b'{}\n';path.write_bytes(prefix+raw)
    digest=hashlib.sha256(raw).hexdigest()
    assert run_fit(path,len(prefix),digest,tmp_path/'out','/env','/sources')['status']=='success'
    assert calls[0]['source_root']=='/sources'
    with pytest.raises(ValueError,match='request changed'):
        run_fit(path,0,digest,tmp_path/'out','/env','/sources')
    req['budget']=dict(req['budget'],search_seconds=1)
    changed=(json.dumps(item)+'\n').encode();path.write_bytes(changed)
    with pytest.raises(ValueError,match='budget differs'):
        run_fit(path,0,hashlib.sha256(changed).hexdigest(),tmp_path/'out','/env','/sources')


def test_pysr_fair_fit_keeps_native_search_but_bypasses_symbolic_export(tmp_path, monkeypatch):
    import benchmark_mysr.external.solvers as solvers
    seen=[]
    class Native:
        def __init__(self,**kw):
            seen.append(kw)
        def get_hof(self,*a,**kw):
            raise AssertionError('Symbolic export must not rewrite native numerical trees')
        def fit(self,X,y,**kw):
            out=tmp_path/'pysr-output/run';out.mkdir(parents=True)
            (out/'hall_of_fame.csv').write_text('Complexity,Loss,Equation\n3,0,x0*2\n')
            assert self.get_hof(None) is None
    original=Native.get_hof
    monkeypatch.chdir(tmp_path)
    candidates,details=solvers.fit('pysr',Native,np.array([[1.],[2.]]),np.array([2.,4.]),request('pysr'))
    assert Native.get_hof is original
    assert seen[0]['max_evals'] is None and seen[0]['precision']==64
    assert seen[0]['populations']==protocol()['methods']['pysr']['populations']
    assert np.array_equal(candidates[0][2](np.array([[3.]])),[6])
    assert json.loads((tmp_path/'stage.json').read_text())['stage']=='scoring'


def test_operon_receives_nonzero_constant_optimization_and_mse(monkeypatch):
    import benchmark_mysr.external.solvers as solvers
    class Captured(Exception): pass
    received={}
    def native(**kw):
        received.update(kw)
        raise Captured
    with pytest.raises(Captured):
        solvers.fit('operon',native,np.ones((3,1)),np.ones(3),request('operon'))
    assert received['optimizer_iterations']==10
    assert received['objectives']==['mse','length']
    assert received['max_evaluations']>10**12 and received['n_threads']==1


def test_tf_greedy_mask_forbids_missing_variables_padding_and_unfinished_trees():
    vocabulary=['add','mul','sin','cos','log','exp','neg','inv','sqrt','sq','cb','C']+[f'x{i}' for i in range(1,7)]
    legal=legal_tf4sr_tokens(vocabulary,1,1,0)
    assert not legal[0] and not legal[1]
    assert legal[vocabulary.index('C')+2] and legal[vocabulary.index('x1')+2]
    assert not any(legal[vocabulary.index(f'x{i}')+2] for i in range(2,7))
    assert not legal[vocabulary.index('add')+2]
    assert not legal[vocabulary.index('sin')+2]
    slots=1
    for step in range(30):
        legal=legal_tf4sr_tokens(vocabulary,3,slots,29-step)
        token=next(token for i,token in enumerate(vocabulary,2) if legal[i])
        slots+=(2 if token in ('add','mul') else 0 if token=='C' or token.startswith('x') else 1)-1
        if slots==0:break
    assert slots==0


def test_physical_cpu_selection_avoids_smt_siblings(tmp_path):
    import importlib.util
    path=Path(__file__).parents[1]/'scripts/external/node_pool.py'
    spec=importlib.util.spec_from_file_location('pool_physical_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for cpu,core in [(0,0),(1,1),(2,0),(3,1)]:
        topology=tmp_path/f'cpu{cpu}'/'topology';topology.mkdir(parents=True)
        (topology/'core_id').write_text(str(core));(topology/'physical_package_id').write_text('0')
    assert module.physical_cpus([0,1,2,3],tmp_path)==[0,1]


def test_final_audit_accepts_measured_invalidity_but_flags_runtime_faults(tmp_path):
    from benchmark_mysr.external.fair_finalize import audit_one
    req=dict(request(),expected_data_hashes={'train':'a','validation':'b','test':'c'})
    item={'request':req,'relative_output':'tabular/gplearn/fixture'}
    path=tmp_path/item['relative_output'];path.mkdir(parents=True)
    result={**{key:req[key] for key in ('method','variant','resource_track','task_id','seed','budget',
            'comparison_protocol','comparison_protocol_sha256')},'status':'invalid_output',
            'data_hashes':req['expected_data_hashes'],'slurm_job_id':'job','adapter_hashes':{},
            'selected_expression':None,'cpu_time':1.,'runtime':2.,'peak_memory':100,'full_frontier':[]}
    (path/'result.json').write_text(json.dumps(result))
    row=audit_one((item,str(tmp_path),'job',{}))
    assert not row['issues'] and not row['usable']
    result['partial_result_status']='solver_error'
    (path/'result.json').write_text(json.dumps(result))
    assert 'implementation_or_runtime_fault_requires_review' in audit_one((item,str(tmp_path),'job',{}))['issues']


def test_ode_summary_requires_all_independent_components(tmp_path):
    from benchmark_mysr.external.fair_finalize import aggregate_one
    common={'method':'pysr','task_id':'odebench-1','variant':'noiseless',
            'resource_track':'constrained_resource','seed':7,'dimension':2,'issues':[],
            'usable':True,'test_nrmse':.1,'selected_expression':'x0','cpu_time':1.,
            'component_serial_elapsed_seconds':2.}
    rows=[dict(common,component=0),dict(common,component=1,test_nrmse=.3)]
    assert aggregate_one(('complete',rows,tmp_path))
    result=json.loads((tmp_path/'complete/result.json').read_text())
    assert result['prediction_usable'] and result['test_nrmse']==pytest.approx(.2)
    assert not aggregate_one(('missing',rows[:1],tmp_path))
    assert not json.loads((tmp_path/'missing/result.json').read_text())['prediction_usable']


def test_interrupted_operon_checkpoint_keeps_previous_json_tree_pair(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmark_mysr.external.checkpoints import OperonCheckpoint
    from benchmark_mysr.external import worker
    monkeypatch.chdir(tmp_path)
    checkpoint=OperonCheckpoint(request('operon'),interval=0)
    algorithm=SimpleNamespace(Generation=1,BestFront=[SimpleNamespace(Genotype=[1.])])
    checkpoint.write(algorithm,{1:'X1'})
    prior=json.loads((tmp_path/'native-candidates.json').read_text())
    algorithm.Generation=2;algorithm.BestFront[0].Genotype=[2.]
    def interrupted(*a,**kw):raise InterruptedError('killed before publishing JSON')
    monkeypatch.setattr(worker,'dump',interrupted)
    with pytest.raises(InterruptedError):checkpoint.write(algorithm,{1:'X1'})
    assert json.loads((tmp_path/'native-candidates.json').read_text())==prior
    assert hashlib.sha256((tmp_path/prior['pickle_file']).read_bytes()).hexdigest()==prior['pickle_sha256']
