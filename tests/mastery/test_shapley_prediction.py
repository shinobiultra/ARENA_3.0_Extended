"""Non-target fixtures only; the eight-player denominator comparison is never run."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat

import nbformat
import pytest
import torch

from mastery_hub.units.shapley_prediction import lesson

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_thread():
    before=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def generator():
    spec=importlib.util.spec_from_file_location('shapley_generator',ROOT/'scripts/build_shapley_prediction_notebook.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def commitment(prediction=.1):
    # Hashes only; no numerical student methods or target utility evaluations.
    source=lesson.REFERENCE_METHODS
    return lesson.commit_prediction(lesson.freeze_inputs(source),source,prediction,.6,'Invented fixture forecast','unknown')


def fabricated_receipt():
    # Completely fabricated zero-state tables to exercise record paths; no model fits.
    parent=commitment()
    result={'states':[[0.,0.,0.] for _ in range(256)],'utilities':[0.]*256,
            'shapley':[0.]*8,'marginals7':[0.]*8,'loo':[0.]*8}
    result['states_sha256']=lesson._tensor_hash(result['states'])
    results={k:deepcopy(result) for k in lesson.CONDITIONS}
    out={'schema':lesson.SCHEMA,'schema_version':1,'status':'observed','commitment':parent,
         'commitment_sha256':hashlib.sha256(parent.encode()).hexdigest(),
         'observation':{'kind':'fresh_cpu_training','results':results,'controls':lesson._control_checks(results),
                        'delta_phi7':0.,'absolute_forecast_error':.1,'elapsed_seconds':1.,
                        'nonzero_lr_fits':1024,'zero_lr_checks':512}}
    return lesson._seal(out)


def test_fixed_single_intervention_case():
    assert lesson.CASE['steps']==40 and lesson.CASE['lr']==.2
    assert lesson.CASE['dtype']=='torch.float64' and lesson.CASE['threads']==1
    assert lesson.CASE['train_x'][0]==lesson.CASE['train_x'][1]==lesson.CASE['train_x'][6]
    assert lesson.CASE['train_y'][0]==lesson.CASE['train_y'][1]!=lesson.CASE['train_y'][6]
    assert lesson.CASE['policies']==['fixed_n','subset_mean']
    assert len(lesson.CASE['train_x'])==len(lesson.CASE['val_x'])==8
    assert lesson.CASE['evidence_class']=='analytic_toy'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('stage',['training','utility','shapley','strata','assembly'])
def test_independent_non_target_fixtures(dtype,stage):
    before=torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        ns=lesson.method_namespace(lesson.REFERENCE_METHODS)
        assert lesson.practice_test(ns,stage)==stage+': independent practice fixture passed'
    finally:
        torch.set_default_dtype(before)


def test_practice_catches_wrong_denominator_and_input_mutation():
    ns=lesson.method_namespace(lesson.REFERENCE_METHODS)
    train=ns['train_coalitions']
    ns['train_coalitions']=lambda x,y,m,s,lr,policy:train(x,y,m,s,lr,'fixed_n')
    with pytest.raises(AssertionError):lesson.practice_test(ns,'training')
    def corrupt(x,y,m,s,lr,p):
        result=train(x,y,m,s,lr,p)
        x.add_(.1)
        return result
    ns['train_coalitions']=corrupt
    with pytest.raises(AssertionError):lesson.practice_test(ns,'training')


def test_notebook_is_blank_and_freeze_commit_precede_run():
    cells=lesson.build_cells()
    assert cells==lesson.build_cells()
    nb=nbformat.v4.new_notebook(cells=[nbformat.from_dict(c) for c in cells])
    nbformat.validate(nb)
    assert len({c['id'] for c in cells})==len(cells)
    order={c['id']:i for i,c in enumerate(cells)}
    assert order['shapley-freeze']<order['shapley-commit']<order['shapley-run']
    for cell in cells:
        source=cell['source']
        assert '<img' not in source and 'base64' not in source and 'from_pretrained' not in source
        assert 'pip install' not in source
        if cell['cell_type']=='code':
            compile(source,'<cell>','exec')
            assert cell['outputs']==[] and cell['execution_count'] is None
        if order[cell['id']]<order['shapley-commit']:
            assert 'run_experiment(' not in source and 'reveal_methods(' not in source
            assert 'delta_phi7' not in source
    assert sum('NotImplementedError' in c['source'] for c in cells)==5
    assert 'No target result' in cells[0]['source']
    # Verify the generated method-text join uses actual newlines, not literal escapes.
    values={f'{s}_SOURCE':lesson._stub(n) for s,n in zip(('TRAINING','UTILITY','SHAPLEY','STRATA','ASSEMBLY'),lesson.METHOD_NAMES,strict=True)}
    fake=type('NoTarget',(),{'freeze_inputs':lambda self,s:None})()
    exec(next(c['source'] for c in cells if c['id']=='shapley-freeze'),{'lesson':fake,**values})


def test_freeze_performs_no_method_execution(monkeypatch):
    monkeypatch.setattr(lesson,'method_namespace',lambda *a,**k:(_ for _ in ()).throw(AssertionError('no methods')))
    identity=lesson.freeze_inputs(lesson.REFERENCE_METHODS)
    assert identity['inputs']['coalition_masks']['shape']==[256,8]
    assert identity['inputs']['train_x']['dtype']=='torch.float64'


@pytest.mark.parametrize('prediction',[-100.,100.,0.])
def test_unbounded_finite_numeric_forecasts_and_immutability(prediction):
    text=commitment(prediction)
    out=lesson.read_record(text)
    out['prediction']=3.
    assert lesson.read_record(text)['prediction']==prediction
    assert out['assessment']=='local_self_report_no_unseen_certification'
    for exposure in ('unknown','seen','self_reported_not_seen'):
        out['prior_exposure']=exposure
        assert lesson.read_record(lesson._seal(out))['prior_exposure']==exposure


@pytest.mark.parametrize('prediction',[True,None,float('nan'),float('inf')])
def test_invalid_forecasts_rejected(prediction):
    with pytest.raises(ValueError):commitment(prediction)


def test_schema_duplicate_keys_nonfinite_and_size_bounds():
    text=commitment()
    for schema in ('arena-extension-guided-practice','arena-blind-circuit-prediction'):
        out=lesson.read_record(text);out['schema']=schema
        with pytest.raises(ValueError):lesson.read_record(lesson._seal(out))
    with pytest.raises(ValueError,match='duplicate'):lesson.read_record('{"schema":1,"schema":2}')
    with pytest.raises(ValueError):lesson.read_record('{"x":NaN,"seal":"x"}')
    with pytest.raises(ValueError,match='bounded'):lesson.read_record(' '*2_000_001)
    for bad in [lesson.REFERENCE_METHODS+'\nx=1',lesson.REFERENCE_METHODS.replace('def train_coalitions(', '@print(1)\ndef train_coalitions(')]:
        with pytest.raises(ValueError):lesson.freeze_inputs(bad)
    with pytest.raises(ValueError):lesson.reveal_methods('{}')
    assert lesson.reveal_methods(text)==lesson.REFERENCE_METHODS


def test_meta_default_device_rejected_before_training():
    before=torch.get_default_device()
    try:
        torch.set_default_device('meta')
        with pytest.raises(ValueError,match='CPU default device'):lesson.freeze_inputs(lesson.REFERENCE_METHODS)
    finally:
        torch.set_default_device(before)


@pytest.mark.parametrize('mutation',['method','helper','tests','environment','data','mask_order','dtype'])
def test_stale_identity_rejected_before_namespace(monkeypatch,mutation):
    text=commitment()
    out=lesson.read_record(text)
    source=lesson.REFERENCE_METHODS
    if mutation=='method':source=source.replace('weights =','weights  =',1)
    elif mutation in ('helper','tests'):out['identity'][mutation+'_sha256']='0'*64
    elif mutation=='environment':out['identity']['environment']['torch']='fictional'
    elif mutation=='data':out['identity']['inputs']['train_x']['sha256']='0'*64
    elif mutation=='mask_order':out['identity']['inputs']['coalition_masks']['sha256']='0'*64
    else:out['identity']['inputs']['train_x']['dtype']='torch.float32'
    monkeypatch.setattr(lesson,'method_namespace',lambda *a,**k:(_ for _ in ()).throw(AssertionError('training reached')))
    with pytest.raises(ValueError):lesson.run_experiment(lesson._seal(out),source)


@pytest.mark.parametrize('mutation',['state_hash','state_value','shape','utility','credit','stratum','loo','order','nan','counts','error','control','extra'])
def test_malformed_observations_rejected(mutation):
    out=lesson.read_record(fabricated_receipt());obs=out['observation'];result=obs['results']['fixed_n']
    if mutation=='state_hash':result['states_sha256']='0'*64
    elif mutation=='state_value':result['states'][1][0]=.1
    elif mutation=='shape':result['states'][0].pop()
    elif mutation=='utility':result['utilities'][1]=.1
    elif mutation=='credit':result['shapley'][1]=1e-8
    elif mutation=='stratum':result['marginals7'][1]=1e-8
    elif mutation=='loo':result['loo'][1]=1e-8
    elif mutation=='order':result['states'][0],result['states'][1]=[1.,0.,0.],[0.,0.,0.]
    elif mutation=='nan':result['utilities'][1]=float('nan')
    elif mutation=='counts':obs['nonzero_lr_fits']=True
    elif mutation=='error':obs['absolute_forecast_error']=.2
    elif mutation=='control':obs['controls']['full_same']=False
    else:obs['unrecognized']=1
    with pytest.raises(ValueError):lesson.read_record(lesson._seal(out))


def test_replay_reflection_private_export_and_no_current_model_work(tmp_path,monkeypatch):
    receipt=fabricated_receipt()
    for name in ('run_experiment','freeze_inputs','method_namespace','environment_identity','practice_test'):
        monkeypatch.setattr(lesson,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('model work forbidden')))
    observation=lesson.replay(receipt)
    observation['delta_phi7']=9
    assert lesson.replay(receipt)['delta_phi7']==0
    attempt=lesson.reflect(receipt,'The dummy fixture matched','Change one learning rate','Expect a difference',.5,'Different gradients accumulate')
    assert lesson.read_attempt(attempt)['receipt']==receipt
    for text,writer,reader in [(receipt,lesson.export_record,lesson.read_record),(attempt,lesson.export_attempt,lesson.read_attempt)]:
        path=tmp_path/('attempt.json' if writer==lesson.export_attempt else 'receipt.json')
        writer(text,path)
        assert stat.S_IMODE(path.stat().st_mode)==0o600
        assert path.read_text()==text
        reader(path.read_text())
        with pytest.raises(FileExistsError):writer(text,path)
    import matplotlib
    matplotlib.use('Agg')
    fig=lesson.plot_receipt(receipt)
    assert len(fig.axes)==2
    import matplotlib.pyplot as plt
    plt.close(fig)
    broken=lesson.read_attempt(attempt);broken['receipt_sha256']='0'*64
    with pytest.raises(ValueError):lesson.read_attempt(lesson._seal(broken))


def _stubbed_run(monkeypatch,train,assembly=None):
    source=lesson.REFERENCE_METHODS;text=commitment()
    ns=lesson.method_namespace(source)
    ns['train_coalitions']=train
    ns['coalition_utilities']=lambda states,*args:states.new_zeros(256)
    if assembly is not None:ns['assemble_result']=assembly
    monkeypatch.setattr(lesson,'method_namespace',lambda *a,**k:ns)
    monkeypatch.setattr(lesson,'practice_test',lambda *a,**k:'fabricated stub fixture')
    return text,source


def test_actual_states_and_utility_are_bound_and_failures_preserved(monkeypatch,tmp_path):
    def train(x,y,masks,*args):
        states=x.new_ones((256,3));states[0]=0
        return states
    def false_assembly(states,utilities,n,player):
        return {'states':[[0.,0.,0.]]*256,'utilities':[0.]*256,'shapley':[0.]*8,'marginals7':[0.]*8,'loo':[0.]*8}
    text,source=_stubbed_run(monkeypatch,train,false_assembly)
    with pytest.raises(lesson.ExperimentFailure,match='actual fitted states') as failure:
        lesson.run_experiment(text,source)
    record=failure.value.record_text
    out=lesson.read_record(record)
    assert out['status']=='failed' and out['commitment']==text
    saved=lesson.replay_failure(record)
    assert saved['measured']['fixed_n']['states'][1]==[1.,1.,1.]
    assert 'utilities' in saved['measured']['fixed_n']
    assert 'absolute_forecast_error' not in saved
    lesson.export_record(record,tmp_path/'failure.json')
    with pytest.raises(ValueError):lesson.replay(record)
    with pytest.raises(ValueError):lesson.reflect(record,'x','q','p',.5,'r')


def test_actual_input_mutation_fails_with_measured_states(monkeypatch):
    def train(x,y,masks,*args):
        x[0,0]+=.1
        return x.new_zeros((256,3))
    text,source=_stubbed_run(monkeypatch,train)
    with pytest.raises(lesson.ExperimentFailure,match='frozen inputs') as failed:
        lesson.run_experiment(text,source)
    assert lesson.replay_failure(failed.value.record_text)['measured']['fixed_n']['states'][0]==[0.,0.,0.]


def test_generator_preserves_edit_outputs_and_records(tmp_path):
    gen=generator();expected=gen.make_notebook();path=tmp_path/'work.ipynb'
    assert gen.synchronize(path,expected,check=False)
    old=nbformat.read(path,as_version=4)
    code=next(c for c in old.cells if c.cell_type=='code')
    code.outputs=[nbformat.v4.new_output('stream',name='stdout',text='private observation')]
    code.execution_count=1
    nbformat.write(old,path);before=path.read_bytes()
    assert gen.synchronize(path,expected,check=True)
    assert path.read_bytes()==before
    newer=deepcopy(expected);newer.cells[0].source+='\nnew generated teaching text'
    assert not gen.synchronize(path,newer,check=False)
    assert path.read_bytes()==before
    old.cells[1].source+='\n# learner edit'
    nbformat.write(old,path);before=path.read_bytes()
    assert not gen.synchronize(path,expected,check=False)
    assert path.read_bytes()==before


def test_generated_method_text_executes_with_exact_newline_join():
    import ast
    functions={n.name:ast.get_source_segment(lesson.REFERENCE_METHODS,n)+'\n' for n in ast.parse(lesson.REFERENCE_METHODS).body}
    namespace={'lesson':lesson}
    for cell,stage,name in zip([c for c in lesson.build_cells() if c['id'] in [f'shapley-{s}' for s in ('training','utility','shapley','strata','assembly')]],
                               ('training','utility','shapley','strata','assembly'),lesson.METHOD_NAMES,strict=True):
        source=cell['source'].replace(lesson._stub(name),functions[name])
        exec(source,namespace)
    freeze=next(c['source'] for c in lesson.build_cells() if c['id']=='shapley-freeze')
    exec(freeze,namespace)
    assert tuple(n.name for n in ast.parse(namespace['METHOD_SOURCE']).body)==lesson.METHOD_NAMES


def test_largest_method_and_unicode_reason_writer_reader_parity():
    source=lesson.REFERENCE_METHODS+'\n#'+('x'*(99_000-len(lesson.REFERENCE_METHODS)-2))
    text=lesson.commit_prediction(lesson.freeze_inputs(source),source,100.,.5,'🧠'*20_000)
    assert len(text.encode())<=2_000_000
    assert lesson.read_record(text)['method_source']==source
    record=lesson.read_record(fabricated_receipt())
    record['commitment']=text
    record['commitment_sha256']=hashlib.sha256(text.encode()).hexdigest()
    record['observation']['absolute_forecast_error']=100.
    receipt=lesson._seal(record)
    lesson.read_record(receipt)
    attempt=lesson.reflect(receipt,'🧠'*20_000,'q'*20_000,'p'*20_000,.5,'r'*20_000)
    assert len(attempt.encode())<=3_000_000
    assert lesson.read_attempt(attempt)['receipt']==receipt


def test_complete_false_controls_keep_all_actual_tables_without_score(monkeypatch):
    def zeros(x,y,masks,*args):return x.new_zeros((256,3))
    text,source=_stubbed_run(monkeypatch,zeros)
    original=lesson._control_checks
    def false_control(results):
        checks=original(results);checks['full_same']=False
        return checks
    monkeypatch.setattr(lesson,'_control_checks',false_control)
    with pytest.raises(lesson.ExperimentFailure,match='control failure') as failed:
        lesson.run_experiment(text,source)
    payload=lesson.replay_failure(failed.value.record_text)
    assert len(payload['measured'])==len(payload['results'])==6
    assert payload['controls']['full_same'] is False
    assert 'delta_phi7' not in payload and 'absolute_forecast_error' not in payload


@pytest.mark.parametrize('stage',['utility','assembly'])
def test_stage_runtime_mutations_rejected_before_next_training(monkeypatch,stage):
    def zeros(x,y,masks,*args):return x.new_zeros((256,3))
    text,source=_stubbed_run(monkeypatch,zeros)
    # Capture the already stubbed namespace without calling target training.
    ns=lesson.method_namespace(source)
    key='coalition_utilities' if stage=='utility' else 'assemble_result'
    original=ns[key]
    before=torch.get_default_dtype()
    def mutate(*args):
        result=original(*args) if stage=='assembly' else args[0].new_zeros(256)
        torch.set_default_dtype(torch.float32 if before==torch.float64 else torch.float64)
        return result
    ns[key]=mutate
    try:
        with pytest.raises(lesson.ExperimentFailure,match='frozen environment') as failure:
            lesson.run_experiment(text,source)
        assert list(lesson.replay_failure(failure.value.record_text)['measured'])==['fixed_n']
    finally:
        torch.set_default_dtype(before)


def test_overbudget_training_preserves_returned_actual_states(monkeypatch):
    clock=[0.]
    def late(x,y,masks,*args):
        clock[0]=121.
        return x.new_zeros((256,3))
    text,source=_stubbed_run(monkeypatch,late)
    monkeypatch.setattr(lesson.time,'monotonic',lambda:clock[0])
    with pytest.raises(lesson.ExperimentFailure,match='CPU budget exceeded') as failed:
        lesson.run_experiment(text,source)
    record=lesson.replay_failure(failed.value.record_text)
    assert record['elapsed_seconds']==121.
    assert record['measured']['fixed_n']['states'][0]==[0.,0.,0.]
    assert record['results']=={}
