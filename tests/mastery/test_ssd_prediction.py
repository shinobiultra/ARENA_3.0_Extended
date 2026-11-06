"""Independent practice/zero fixtures only; never evaluate the adopted T12 case."""
from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat

import nbformat
import pytest
import torch
from mastery_hub.units.ssd_prediction import lesson

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=deepcopy(lesson.CASE)

@pytest.fixture(autouse=True)
def cpu_and_no_public_dispatch(monkeypatch):
    old=torch.get_num_threads();torch.set_num_threads(1)
    execute=lesson._execute_experiment
    def guarded(*args,**kwargs):
        assert lesson.CASE['id']=='fixture-only-zero-filter', 'TEST MUST NOT EXECUTE PUBLIC CASE'
        return execute(*args,**kwargs)
    monkeypatch.setattr(lesson,'_execute_experiment',guarded)
    yield
    torch.set_num_threads(old)

@pytest.fixture
def diagnostic_case(monkeypatch):
    case=deepcopy(PUBLIC)
    case.update(id='fixture-only-zero-filter',X=[[0.,0.]]*3,
        B=[[1.,0.],[0.,1.],[1.,-1.]],C=[[1.,1.],[1.,-1.],[.5,.25]],
        initial_state=[[0.,0.],[0.,0.]],transition_probabilities=[.5,.75],readout_weights=[.25,-.5])
    monkeypatch.setattr(lesson,'CASE',case)
    monkeypatch.setattr(lesson,'_CASE_DEFINITION',json.dumps(case,sort_keys=True,allow_nan=False))
    return case


def generator():
    spec=importlib.util.spec_from_file_location('ssd_generator',ROOT/'scripts/build_ssd_prediction_notebook.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def commit(prediction=.1,source=None):
    source=lesson.REFERENCE_METHODS if source is None else source
    return lesson.commit_prediction(lesson.freeze_inputs(source),source,prediction,.6,'Fixture forecast','unknown')


def measured_fixture():
    assert lesson.CASE['id']=='fixture-only-zero-filter'
    return lesson.run_experiment(commit(),lesson.REFERENCE_METHODS)


def test_exact_public_raw_inputs_without_evaluation():
    assert lesson.SCHEMA=='arena-blind-ssd-prediction' and lesson.VERSION==1
    assert PUBLIC['transition_probabilities']==[.82,.94]
    assert PUBLIC['X']==[[1.,-.25],[-.5,.8],[0.,-.6],[.75,0.],[-1.,.5],[.25,-1.],[0.,.3],[-.25,0.],[.5,-.4],[0.,.7],[-.75,0.],[.4,-.2]]
    assert PUBLIC['B']==[[1.,0.],[0.,1.],[1.,-1.],[.5,1.],[-1.,.5],[1.,1.],[0.,-1.],[-.5,1.],[1.,.25],[.25,-1.],[-1.,-.5],[.5,.75]]
    assert PUBLIC['C']==[[.5,1.],[1.,-.5],[-.5,.25],[1.,1.],[.25,-1.],[-1.,.5],[1.,-.25],[.5,.5],[-.75,1.],[1.,.25],[.25,-.5],[.8,-.6]]
    assert PUBLIC['readout_weights']==[.7,-.4] and PUBLIC['initial_state']==[[0.,0.],[0.,0.]]
    primary=lesson._case();transfer=lesson._case('transfer')
    assert {k for k in primary if primary[k]!=transfer[k]}=={'variant','log_decay_scale'}
    assert transfer['log_decay_scale']==.5
    assert not {'answer','delta_readout','result'}&PUBLIC.keys()


@pytest.mark.parametrize('stage',lesson.STAGES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_practice_and_split_basis_controls(stage,dtype):
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        namespace=lesson.method_namespace(lesson.REFERENCE_METHODS)
        assert lesson.practice_test(namespace,stage)=='Practice '+stage+' passed (different inputs).'
    finally:torch.set_default_dtype(old)


@pytest.mark.parametrize('method',lesson.METHODS)
def test_hand_worked_one_state(method):
    namespace=lesson.method_namespace(lesson.REFERENCE_METHODS)
    X=torch.tensor([[2.],[3.],[5.]],dtype=torch.float64)
    A=torch.full_like(X,.5).log();B=C=torch.ones_like(X);initial=torch.ones((1,1),dtype=torch.float64)
    args=(X,A,B,C,initial,.5)
    y,state=namespace['ssd_'+method](*args,4) if method=='chunked' else namespace['ssd_'+method](*args)
    expected=torch.tensor([[2.25],[3.5625],[5.890625]],dtype=torch.float64)
    assert torch.equal(y,expected) if method=='recurrent' else torch.allclose(y,expected,atol=1e-12)
    assert torch.allclose(state[:,:,0],expected,atol=1e-12)


@pytest.mark.parametrize('gamma',[0.,.2,1.])
@pytest.mark.parametrize('chunk',[1,2,4,9])
def test_gamma_zero_and_identity_padding_last_real_state(gamma,chunk):
    ns=lesson.method_namespace(lesson.REFERENCE_METHODS);args=lesson.practice_fixture()
    expected=lesson._scalar_oracle(*(v.tolist() for v in args),gamma)
    for name in lesson.METHODS:
        y,states=ns['ssd_'+name](*args,gamma,chunk) if name=='chunked' else ns['ssd_'+name](*args,gamma)
        assert lesson._equal(y.tolist(),expected[0]) and lesson._equal(states.tolist(),expected[1])
        assert torch.isfinite(states).all() and states.shape[0]==3
        if gamma==0:assert torch.equal(states,args[0][:,:,None]*args[2][:,None,:])


@pytest.mark.parametrize('bad',['shape','empty','nan','positive_A','float32','gamma_bool','gamma_nan','gamma_negative','chunk_bool','chunk_zero'])
def test_invalid_inputs_rejected(bad):
    ns=lesson.method_namespace(lesson.REFERENCE_METHODS);args=list(lesson.practice_fixture());gamma=.4;chunk=2
    if bad=='shape':args[2]=args[2][:-1]
    elif bad=='empty':args=[v[:0] for v in args[:4]]+[args[4]]
    elif bad=='nan':args[0][0,0]=float('nan')
    elif bad=='positive_A':args[1][0,0]=1.
    elif bad=='float32':args[0]=args[0].float()
    elif bad=='gamma_bool':gamma=True
    elif bad=='gamma_nan':gamma=float('nan')
    elif bad=='gamma_negative':gamma=-.1
    elif bad=='chunk_bool':chunk=True
    else:chunk=0
    with pytest.raises(ValueError):ns['ssd_chunked'](*args,gamma,chunk)


def test_freeze_commit_import_never_evaluate_numerical_methods(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('NUMERICAL PUBLIC EVALUATION FORBIDDEN')
    for name in ('method_namespace','_scalar_oracle','practice_test','_execute_experiment'):
        monkeypatch.setattr(lesson,name,forbidden)
    identity=lesson.freeze_inputs(lesson.REFERENCE_METHODS)
    text=lesson.commit_prediction(identity,lesson.REFERENCE_METHODS,-100.,.5,'No numerical work')
    assert lesson.read_record(text)['prediction']==-100.
    assert identity['inputs']['X']['shape']==[12,2]
    assert identity['inputs']['initial_state']['shape']==[2,2]
    # Independent import also creates definitions only.
    spec=importlib.util.spec_from_file_location('ssd_import_only',lesson.__file__)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.CASE==PUBLIC


@pytest.mark.parametrize('prediction',[100.,-100.,0.])
def test_signed_unbounded_forecast_and_exposure(prediction):
    text=commit(prediction);out=lesson.read_record(text);out['prediction']=99
    assert lesson.read_record(text)['prediction']==prediction
    for exposure in ('seen','unknown','self_reported_not_seen'):
        out=lesson.read_record(text);out['prior_exposure']=exposure
        assert lesson.read_record(lesson._seal(out))['prior_exposure']==exposure


@pytest.mark.parametrize('prediction',[True,None,float('nan'),float('inf')])
def test_invalid_forecasts(prediction):
    with pytest.raises(ValueError):commit(prediction)


@pytest.mark.parametrize('mutation',['source','helper','tests','runtime','input_hash','shape','dtype','case'])
def test_stale_identity_rejected_before_execution(monkeypatch,mutation):
    out=lesson.read_record(commit());source=lesson.REFERENCE_METHODS
    if mutation=='source':source+='\n# changed source'
    elif mutation in ('helper','tests'):out['identity'][mutation+'_sha256']='0'*64
    elif mutation=='runtime':out['identity']['environment']['torch']='fictional'
    elif mutation=='input_hash':out['identity']['inputs']['X']['sha256']='0'*64
    elif mutation=='shape':out['identity']['inputs']['X']['shape']=[2,12]
    elif mutation=='dtype':out['identity']['inputs']['X']['dtype']='torch.float32'
    else:out['case']['X'][0][0]=99
    monkeypatch.setattr(lesson,'method_namespace',lambda *a:pytest.fail('execution reached'))
    with pytest.raises(ValueError):lesson.run_experiment(lesson._seal(out),source)


def test_strict_schema_json_source_and_size():
    text=commit();out=lesson.read_record(text)
    for key,value in [('schema','arena-blind-sage-prediction'),('schema_version',True),('extra',1)]:
        changed=deepcopy(out);changed[key]=value
        with pytest.raises(ValueError):lesson.read_record(lesson._seal(changed))
    with pytest.raises(ValueError,match='duplicate'):lesson.read_record('{"schema":1,"schema":2}')
    with pytest.raises(ValueError):lesson.read_record('{"n":NaN}')
    with pytest.raises(ValueError):lesson.read_record(' '*(lesson.RECORD_LIMIT+1))
    for source in (lesson.REFERENCE_METHODS+'\nx=1',lesson.REFERENCE_METHODS.replace('def ssd_recurrent','@print(1)\ndef ssd_recurrent'),lesson.REFERENCE_METHODS.replace('gamma):','gamma=print(1)):',1)):
        with pytest.raises(ValueError):lesson.freeze_inputs(source)
    assert lesson.reveal_methods(text)==lesson.REFERENCE_METHODS
    with pytest.raises(ValueError):lesson.reveal_methods('{}')


def test_non_cpu_runtime_rejected():
    old=torch.get_default_device()
    try:
        torch.set_default_device('meta')
        with pytest.raises(ValueError):lesson.freeze_inputs(lesson.REFERENCE_METHODS)
    finally:torch.set_default_device(old)


def test_full_fixture_lifecycle_and_replay_isolation(diagnostic_case,tmp_path,monkeypatch):
    receipt=measured_fixture();observation=lesson.replay(receipt)
    assert observation['condition_count']==14 and len(observation['controls'])==8 and all(observation['controls'].values())
    assert observation['delta_readout']==0
    attempt=lesson.reflect(receipt,'Zero fixture effect',-.2,.5,'Fixture transfer')
    identity=lesson.freeze_inputs(lesson.REFERENCE_METHODS,'transfer')
    transfer=lesson.commit_transfer(attempt,identity,lesson.REFERENCE_METHODS)
    transfer_receipt=lesson.run_experiment(transfer,lesson.REFERENCE_METHODS)
    def forbidden(*a,**k):raise AssertionError('fresh numerical entrypoint reached')
    for name in ('run_experiment','freeze_inputs','method_namespace','environment_identity','practice_test','_scalar_oracle','_execute_experiment'):
        monkeypatch.setattr(lesson,name,forbidden)
    assert lesson.replay(transfer_receipt)['delta_readout']==0
    assert lesson.read_attempt(attempt)['receipt']==receipt
    assert lesson.reflect(receipt,'Stored only',.2,.5,'No experiment')
    for text,writer,reader,name in ((receipt,lesson.export_record,lesson.read_record,'receipt'),(attempt,lesson.export_attempt,lesson.read_attempt,'attempt'),(transfer_receipt,lesson.export_record,lesson.read_record,'transfer')):
        path=tmp_path/(name+'.json');writer(text,path);reader(path.read_text())
        assert path.read_text()==text and stat.S_IMODE(path.stat().st_mode)==0o600
        with pytest.raises(FileExistsError):writer(text,path)
    import matplotlib
    matplotlib.use('Agg')
    fig=lesson.plot_receipt(receipt)
    assert len(fig.axes)==5
    import matplotlib.pyplot as plt
    plt.close(fig)
    changed=lesson.read_attempt(attempt);changed['receipt_sha256']='0'*64
    with pytest.raises(ValueError):lesson.read_attempt(lesson._seal(changed))
    changed=lesson.read_record(transfer);changed['prediction']=99
    with pytest.raises(ValueError):lesson.read_record(lesson._seal(changed))


@pytest.mark.parametrize('mutation',['state','output','readout','hash','shape','count','boolcount','delta','error','control','extra'])
def test_record_array_and_score_tampering_rejected(diagnostic_case,mutation):
    out=lesson.read_record(measured_fixture());obs=out['observation'];result=obs['results']['curve_3']['recurrent']
    if mutation in ('state','output','readout'):
        key={'state':'states','output':'outputs','readout':'readouts'}[mutation]
        if mutation=='state':result[key][0][0][0]+=.1
        elif mutation=='output':result[key][0][0]+=.1
        else:result[key][0]+=.1
        result[key+'_sha256']=lesson._tensor_hash(result[key])
    elif mutation=='hash':result['states_sha256']='0'*64
    elif mutation=='shape':result['outputs'].pop()
    elif mutation=='count':obs['condition_count']=15
    elif mutation=='boolcount':obs['condition_count']=True
    elif mutation=='delta':obs['delta_readout']=.1
    elif mutation=='error':obs['absolute_forecast_error']=9
    elif mutation=='control':obs['controls']['chunk_padding']=False
    else:obs['unknown']=True
    with pytest.raises(ValueError):lesson.read_record(lesson._seal(out))


@pytest.mark.parametrize('mutation',['input','runtime','global_case','wrong_readout','assembly','assembly_mutation','wrong_state','deadline','controls'])
def test_runner_retains_failure_and_actual_arrays(diagnostic_case,monkeypatch,mutation):
    namespace=lesson.method_namespace(lesson.REFERENCE_METHODS)
    monkeypatch.setattr(lesson,'practice_test',lambda *a:True)
    original=namespace['ssd_recurrent'];original_assembly=namespace['assemble_result'];old_threads=torch.get_num_threads()
    if mutation in ('input','runtime','global_case','wrong_state'):
        def corrupt(*args):
            output,states=original(*args)
            if mutation=='input':args[0][0,0]+=.1
            elif mutation=='runtime':torch.set_num_threads(2)
            elif mutation=='global_case':lesson.CASE['X'][0][0]+=.1
            else:states=states+1
            return output,states
        namespace['ssd_recurrent']=corrupt
    elif mutation=='wrong_readout':namespace['readout']=lambda outputs,w:outputs.sum(-1)+1
    elif mutation in ('assembly','assembly_mutation'):
        def assembly(*args):
            out=original_assembly(*args)
            if mutation=='assembly':out['recurrent']['readouts'][0]+=1
            else:args[0].add_(1)
            return out
        namespace['assemble_result']=assembly
    elif mutation=='deadline':
        clock=iter([0.,0.,999.,999.,999.,999.]);monkeypatch.setattr(lesson.time,'monotonic',lambda:next(clock,999.))
    else:
        old_checks=lesson._control_checks
        monkeypatch.setattr(lesson,'_control_checks',lambda *a:{**old_checks(*a),'forced_failure':False})
    monkeypatch.setattr(lesson,'method_namespace',lambda *a:namespace)
    text=commit()
    try:
        with pytest.raises(lesson.ExperimentFailure) as error:lesson.run_experiment(text,lesson.REFERENCE_METHODS)
    finally:torch.set_num_threads(old_threads)
    record=lesson.read_record(error.value.record_text);assert record['status']=='failed' and record['commitment']==text
    failure=lesson.replay_failure(error.value.record_text)
    assert failure['stage'] and failure['error_type'] and 'absolute_forecast_error' not in failure
    if mutation!='deadline':assert failure['measured']
    if mutation=='assembly':assert failure['returned_assemblies']['curve_0']['recurrent']['readouts'][0]==1
    if mutation=='controls':assert len(failure['results'])==14 and failure['controls']['forced_failure'] is False


def test_transfer_parent_and_reflection_guards(diagnostic_case):
    receipt=measured_fixture();attempt=lesson.reflect(receipt,'Reflection',.2,.5,'Transfer')
    with pytest.raises(ValueError):lesson.commit_prediction(lesson.freeze_inputs(lesson.REFERENCE_METHODS),lesson.REFERENCE_METHODS,.2,.5,'reason',parent_attempt=attempt)
    with pytest.raises(ValueError):lesson.commit_prediction(lesson.freeze_inputs(lesson.REFERENCE_METHODS,'transfer'),lesson.REFERENCE_METHODS,.2,.5,'reason',variant='transfer')
    changed=lesson.read_attempt(attempt);changed['transfer_case']['X'][0][0]=9
    with pytest.raises(ValueError):lesson.read_attempt(lesson._seal(changed))


def test_generator_and_blank_visible_method_sources(tmp_path):
    gen=generator();expected=gen.make_notebook();assert gen.source_fingerprint(expected)==gen.source_fingerprint(gen.make_notebook())
    assert sum(c.metadata.get('ssd_method',False) for c in expected.cells)==5
    assert sum('NotImplementedError' in c.source for c in expected.cells)==5
    code=[c for c in expected.cells if c.cell_type=='code']
    for cell in code:
        compile(cell.source,'<cell>','exec');assert cell.outputs==[] and cell.execution_count is None
    ids={c.id:i for i,c in enumerate(expected.cells)}
    assert ids['forecast']<ids['commit']<ids['measure']<ids['transfer']
    for cell in expected.cells[:ids['commit']]:
        assert 'run_experiment(' not in cell.source and 'reveal_methods(' not in cell.source
    assert 'three equivalent computations' in expected.cells[0].source
    assert 'no monotonicity' in '\n'.join(c.source.lower() for c in expected.cells)
    path=tmp_path/'fresh.ipynb';assert gen.synchronize(path,expected,check=False)
    before=path.read_bytes();assert gen.synchronize(path,expected,check=True) and path.read_bytes()==before
    actual=nbformat.read(ROOT/'mastery_hub/units/ssd_prediction/exercises.ipynb',as_version=4)
    assert gen.source_fingerprint(actual)==gen.source_fingerprint(expected)


@pytest.mark.parametrize('saved',['source','outputs','count','execution','widgets','trusted'])
def test_generator_preserves_learner_edits_and_execution(tmp_path,saved):
    gen=generator();old=gen.make_notebook();path=tmp_path/'work.ipynb'
    if saved=='source':old.cells[4].source+='\nLearner notes'
    elif saved=='outputs':old.cells[1].outputs=[nbformat.v4.new_output('stream',name='stdout',text='saved')]
    elif saved=='count':old.cells[1].execution_count=1
    elif saved=='execution':old.cells[1].metadata['execution']={'iopub.status.busy':'saved'}
    elif saved=='widgets':old.metadata['widgets']={'saved':1}
    else:old.cells[1].metadata['trusted']=True
    nbformat.write(old,path)
    if saved=='trusted':
        raw=json.loads(path.read_text());raw['cells'][1]['metadata']['trusted']=True
        path.write_text(json.dumps(raw))
    before=path.read_bytes()
    expected=gen.make_notebook();expected.cells[0].source+='\nChanged generation'
    assert not gen.synchronize(path,expected,check=False) and path.read_bytes()==before


def test_largest_unicode_source_record_roundtrip(diagnostic_case):
    source=lesson.REFERENCE_METHODS+'\n# '+('界'*85000)
    assert len(source)<100000
    text=commit(source=source);receipt=lesson.run_experiment(text,source)
    attempt=lesson.reflect(receipt,'Large fixture reflection',.2,.5,'Transfer')
    transfer=lesson.commit_transfer(attempt,lesson.freeze_inputs(source,'transfer'),source)
    assert lesson.read_record(transfer)['method_source']==source
    assert lesson.read_attempt(attempt)['receipt']==receipt
    transfer_receipt=lesson.run_experiment(transfer,source)
    assert lesson.read_record(transfer_receipt)['status']=='observed'


@pytest.mark.parametrize('invalid',['outputs','states'])
def test_partial_return_retains_valid_partner(diagnostic_case,monkeypatch,invalid):
    ns=lesson.method_namespace(lesson.REFERENCE_METHODS);original=ns['ssd_recurrent']
    def bad(*args):
        output,states=original(*args)
        if invalid=='outputs':output=output.float()
        else:states=states*float('nan')
        return output,states
    ns['ssd_recurrent']=bad
    monkeypatch.setattr(lesson,'method_namespace',lambda *a:ns)
    monkeypatch.setattr(lesson,'practice_test',lambda *a:True)
    with pytest.raises(lesson.ExperimentFailure) as error:lesson.run_experiment(commit(),lesson.REFERENCE_METHODS)
    saved=lesson.replay_failure(error.value.record_text)['measured']['curve_0']['recurrent']
    valid='states' if invalid=='outputs' else 'outputs'
    assert set(saved)=={valid,valid+'_sha256'}


def test_visible_source_stubs_complete_to_exact_reference_methods():
    source_lines=lesson.REFERENCE_METHODS.splitlines(True)
    trees=ast.parse(lesson.REFERENCE_METHODS)
    answers={node.name:''.join(source_lines[node.lineno-1:node.end_lineno])+'\n' for node in trees.body}
    namespace={'lesson':lesson}
    for cell in lesson.build_cells():
        if not cell.get('metadata',{}).get('ssd_method'):continue
        name=cell['id'];stub=lesson._stub(name)
        assert stub in cell['source']
        completed=cell['source'].replace(stub,answers[name])
        exec(compile(completed,'<completed practice source>','exec'),namespace)
    joined='\n\n'.join(namespace[stage.upper()+'_SOURCE'] for stage in lesson.STAGES)
    assert ast.dump(ast.parse(joined))==ast.dump(ast.parse(lesson.REFERENCE_METHODS))
    # Freezing completed source still performs no target evaluation.
    lesson.freeze_inputs(joined)


def test_single_factor_transfer_binds_primary_methods_and_runtime(diagnostic_case):
    receipt=measured_fixture();attempt=lesson.reflect(receipt,'Fixture',.2,.5,'Same operator source')
    changed=lesson.REFERENCE_METHODS+'\n# Changed after primary'
    with pytest.raises(ValueError,match='preserve primary method'):
        lesson.commit_transfer(attempt,lesson.freeze_inputs(changed,'transfer'),changed)
    identity=lesson.freeze_inputs(lesson.REFERENCE_METHODS,'transfer')
    identity['environment']['torch']='changed-runtime'
    with pytest.raises(ValueError,match='preserve primary source and runtime'):
        lesson.commit_transfer(attempt,identity,lesson.REFERENCE_METHODS)


@pytest.mark.parametrize('gamma',[0.,.4,1.])
@pytest.mark.parametrize('chunk',[2,4,7])
def test_strong_decay_masks_future_logs_before_exponentiation(gamma,chunk):
    """Distinct short fixture: future +1000 prefixes must never reach exp."""
    ns=lesson.method_namespace(lesson.REFERENCE_METHODS)
    X=torch.tensor([[1.,-2.],[.5,3.],[-1.,.25],[2.,-.5]],dtype=torch.float64)
    A=torch.tensor([[-1000.,-800.],[-1100.,-900.],[-1200.,-1000.],[-900.,-1100.]],dtype=torch.float64)
    B=torch.tensor([[1.,.5],[-.5,1.],[2.,-1.],[.25,.75]],dtype=torch.float64)
    C=torch.tensor([[.5,-1.],[1.,.25],[-.5,2.],[1.,-.5]],dtype=torch.float64)
    initial=torch.tensor([[.2,-.3],[.4,.1]],dtype=torch.float64)
    expected_y,expected_states=ns['ssd_recurrent'](X,A,B,C,initial,gamma)
    for method in ('quadratic','chunked'):
        args=(X,A,B,C,initial,gamma)
        y,states=ns['ssd_'+method](*args,chunk) if method=='chunked' else ns['ssd_'+method](*args)
        assert bool(torch.isfinite(y).all()) and bool(torch.isfinite(states).all())
        assert torch.allclose(y,expected_y,atol=1e-12,rtol=1e-12)
        assert torch.allclose(states,expected_states,atol=1e-12,rtol=1e-12)
        assert torch.equal(states[-1],X[-1,:,None]*B[-1,None,:])
