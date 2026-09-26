import json
from pathlib import Path

import numpy as np
import pytest
from benchmark_mysr.external.worker import read_split, score
from benchmark_mysr.external.ode import build
from benchmark_mysr.external.runner import environment


def test_observed_target_is_training_input(tmp_path):
    p=tmp_path/'train.csv'
    p.write_text('row_id,x_0,target,target_clean,target_observed\na,2,9,4,9\n')
    X,y,clean=read_split(p)
    assert y.tolist()==[9]
    assert clean.tolist()==[4]
    assert X.tolist()==[[2]]


def test_invalid_prediction_rejected_and_scalar_broadcast():
    assert score(np.ones(3),1)['mse']==0
    with pytest.raises(ValueError):
        score(np.ones(3),[np.inf]*3)


def test_ode_split_before_differentiation(tmp_path):
    t=np.arange(20,dtype=float)
    first={'success':True,'t':t.tolist(),'y':[(t**2).tolist()]}
    second={'success':True,'t':t.tolist(),'y':[(2*t).tolist()]}
    system={'id':1,'dim':1,'solutions':[[first,second]],'eq':'FORBIDDEN_REFERENCE_EQUATION'}
    source=tmp_path/'source.json';source.write_text(json.dumps([system]))
    meta=build(source,0,tmp_path/'out',source)
    train=read_split(Path(meta['component_dirs'][0])/'train.csv')
    assert np.allclose(train[1],2*np.arange(1,13))
    first['y'][0][14:]=[1e8]*6
    source.write_text(json.dumps([system]))
    meta2=build(source,0,tmp_path/'out2',source)
    train2=read_split(Path(meta2['component_dirs'][0])/'train.csv')
    assert np.array_equal(train[1],train2[1])
    test=read_split(Path(meta['component_dirs'][0])/'test.csv')
    assert np.allclose(test[1],2)


def test_environment_threads_and_pysr_project(tmp_path):
    python,env=environment('pysr',tmp_path,tmp_path/'code')
    assert str(python).endswith('env_1_pysr/bin/python')
    assert env['JULIA_NUM_THREADS']=='1'
    assert env['CUDA_VISIBLE_DEVICES']==''
    assert env['PYTHON_JULIACALL_PROJECT'].startswith(str(tmp_path))


def test_worker_selects_validation_before_loading_test(tmp_path, monkeypatch):
    import sys
    import benchmark_mysr.external.worker as worker
    import benchmark_mysr.external.solvers as solvers
    for split, target in [('train',0),('validation',1),('test',0)]:
        (tmp_path/f'{split}.csv').write_text(f'row_id,x_0,target,target_clean\na,2,{target},{target}\n')
    request={'method':'gplearn','data':str(tmp_path),'source_root':str(tmp_path), 'ground_truth_available':False}
    path=tmp_path/'request.json'; path.write_text(json.dumps(request))
    original=worker.read_split
    reads=[]
    def read(p):
        reads.append(Path(p).name)
        return original(p)
    def fit(*args):
        assert reads==['train.csv','validation.csv']
        return [('zero',1,lambda X:np.zeros(len(X))),('one',1,lambda X:np.ones(len(X)))],{}
    monkeypatch.setattr(worker,'read_split',read)
    monkeypatch.setattr(solvers,'prepare',lambda *a:None)
    monkeypatch.setattr(solvers,'fit',fit)
    monkeypatch.setattr(sys,'argv',['worker',str(path)])
    monkeypatch.chdir(tmp_path)
    assert worker.main()==0
    result=json.loads((tmp_path/'worker-result.json').read_text())
    assert result['selected_expression']=='one'
    assert result['test_score']['mse']==1
    assert result['test_numeric_agreement'] is None


def test_supervisor_kills_timed_out_worker_and_resumes(tmp_path, monkeypatch):
    import benchmark_mysr.external.runner as runner
    import psutil
    for split in ('train','validation','test'):
        (tmp_path/f'{split}.csv').write_text('dummy')
    fake=tmp_path/'fake-python'
    fake.write_text('#!/usr/bin/env python3\nimport time,os,json\nopen("stage.json","w").write(json.dumps({"stage":"search","time":time.time()}))\nopen("pid","w").write(str(os.getpid()))\ntime.sleep(60)\n')
    fake.chmod(0o755)
    monkeypatch.setattr(runner,'environment',lambda *a:(fake,dict(__import__('os').environ)))
    req={'method':'gplearn','data':str(tmp_path),'task_id':'test','seed':1,'budget':{'search_seconds':0.01,'evaluations':1,'memory_gib':1}}
    result=runner.run(req,tmp_path/'out',tmp_path,startup_seconds=5)
    assert result['status']=='search_timeout'
    assert not psutil.pid_exists(int((tmp_path/'out/pid').read_text()))
    assert runner.run(req,tmp_path/'out',tmp_path)==result
    req['seed']=2
    with pytest.raises(ValueError,match='changed request'):
        runner.run(req,tmp_path/'out',tmp_path)


def test_seed_streams_are_disjoint():
    p=Path(__file__).resolve().parents[1]/'manifests/formal/seed-ledger-v1.json'
    d=json.loads(p.read_text())
    assert not set(d['pilot_seeds']) & set(d['formal_search_seeds'])
