"""Non-target fixtures only: never solve the adopted three-dimensional SAGE case."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat

import nbformat
import pytest
import torch

from mastery_hub.units.sage_prediction import lesson

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=deepcopy(lesson.CASE)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def diagnostic_case(monkeypatch):
    """Explicit invented zero predictor; all coefficients zero, not the public target."""
    case=deepcopy(PUBLIC)
    case.update(id='fixture-only-zero-predictor',covariance=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                truth=[1.,0.,0.],model=[0.,0.,0.])
    monkeypatch.setattr(lesson,'CASE',case)
    monkeypatch.setattr(lesson,'_CASE_DEFINITION',json.dumps(case,sort_keys=True,allow_nan=False))
    return case


def generator():
    spec=importlib.util.spec_from_file_location('sage_generator',ROOT/'scripts/build_sage_prediction_notebook.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def commitment(prediction=.1,source=None):
    source=lesson.REFERENCE_METHODS if source is None else source
    return lesson.commit_prediction(lesson.freeze_inputs(source),source,prediction,.6,'Invented fixture forecast','unknown')


def fabricated_receipt():
    """Use only with diagnostic_case: hand-worked zeros, without target method calls."""
    assert lesson.CASE['id']=='fixture-only-zero-predictor'
    parent=commitment()
    results={}
    case=lesson._case()
    for label in lesson.CONDITIONS:
        condition=lesson._condition(case,label)
        result={'coefficients':[[0.,0.,0.] for _ in range(8)],
                'risks':[1.+condition['noise_variance']]*8,'game':[0.]*8,
                'shapley':[0.]*3,'marginals_b':[0.]*3}
        for key in ('coefficients','risks','game'):
            result[key+'_sha256']=lesson._tensor_hash(result[key])
        results[label]=result
    return lesson._seal({'schema':lesson.SCHEMA,'schema_version':1,'status':'observed','commitment':parent,
                         'commitment_sha256':hashlib.sha256(parent.encode()).hexdigest(),
                         'observation':{'kind':'fresh_cpu_analytic','results':results,'controls':lesson._control_checks(results,case),
                                        'delta_phi_b':0.,'absolute_forecast_error':.1,'elapsed_seconds':1.,
                                        'condition_count':10,'coalition_count':80}})


def test_public_case_is_exact_and_no_outcome_is_stored():
    assert lesson.CASE['covariance']==[[1.,.6,-.3],[.6,1.,.14],[-.3,.14,1.]]
    assert lesson.CASE['truth']==[1.,-.5,.25]
    assert lesson.CASE['model']==[.75,.25,-.5]
    assert lesson.CASE['noise_variance']==.25
    assert lesson.CASE['features']==['A','B','C'] and lesson.CASE['player']==1
    assert lesson.SCHEMA=='arena-blind-sage-prediction' and lesson.VERSION==1
    assert lesson.CASE['dtype']=='torch.float64' and lesson.CASE['threads']==1
    transfer=lesson._case('transfer')
    assert transfer['covariance']==[[1.,.3,-.15],[.3,1.,.07],[-.15,.07,1.]]
    assert {k for k in transfer if transfer[k]!=lesson._case()[k]}=={'variant','covariance'}
    assert 'transfer_conditional' not in lesson.CONDITIONS
    assert not any(k in lesson.CASE for k in ('answer','result','shapley','delta_phi_b'))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('stage',lesson.STAGES)
def test_independent_non_target_practice_fixtures(dtype,stage):
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        namespace=lesson.method_namespace(lesson.REFERENCE_METHODS)
        assert lesson.practice_test(namespace,stage)==stage+': independent practice fixture passed'
    finally:
        torch.set_default_dtype(previous)


def test_practice_rejects_wrong_rule_wrong_risk_and_mutation():
    namespace=lesson.method_namespace(lesson.REFERENCE_METHODS)
    original=namespace['restriction_coefficients']
    namespace['restriction_coefficients']=lambda cov,b,rule:original(cov,b,'marginal')
    with pytest.raises(AssertionError):lesson.practice_test(namespace,'restriction')
    namespace=lesson.method_namespace(lesson.REFERENCE_METHODS)
    original=namespace['risk_game']
    def corrupt(coeff,cov,truth,noise):
        result=original(coeff,cov,truth,noise)
        coeff[0,0]+=.1
        return result
    namespace['risk_game']=corrupt
    with pytest.raises(AssertionError):lesson.practice_test(namespace,'risk')


def test_notebook_blank_ordered_and_no_target_solution_before_commit():
    cells=lesson.build_cells();assert cells==lesson.build_cells()
    notebook=nbformat.v4.new_notebook(cells=[nbformat.from_dict(c) for c in cells]);nbformat.validate(notebook)
    order={c['id']:i for i,c in enumerate(cells)}
    assert len(order)==len(cells)
    assert order['sage-freeze']<order['sage-commit']<order['sage-run']<order['sage-reflection']
    assert cells[0]['source'].startswith('# One predictor, two missing-input rules')
    assert 'No target result' in cells[0]['source']
    assert sum('NotImplementedError' in c['source'] for c in cells)==5
    for cell in cells:
        source=cell['source']
        assert '<img' not in source and 'base64' not in source and 'from_pretrained' not in source and 'pip install' not in source
        if cell['cell_type']=='code':
            compile(source,'<cell>','exec')
            assert cell['outputs']==[] and cell['execution_count'] is None
        if order[cell['id']]<order['sage-commit']:
            assert 'run_experiment(' not in source and 'reveal_methods(' not in source
            assert 'delta_phi_b' not in source
    assert 'transfer' not in next(c['source'] for c in cells if c['id']=='sage-run')


def test_freeze_and_commit_execute_no_numerical_methods(monkeypatch):
    for name in ('method_namespace','_reference_credits','_strata','_validate_coefficients'):
        monkeypatch.setattr(lesson,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('numerical work reached')))
    identity=lesson.freeze_inputs(lesson.REFERENCE_METHODS)
    assert identity['inputs']['covariance']['shape']==[3,3]
    assert identity['inputs']['coalition_masks']['shape']==[8,3]
    text=lesson.commit_prediction(identity,lesson.REFERENCE_METHODS,100.,.5,'No method evaluation')
    assert lesson.read_record(text)['prediction']==100.


@pytest.mark.parametrize('prediction',[-100.,100.,0.])
def test_signed_finite_forecast_and_immutable_return_value(prediction):
    text=commitment(prediction);out=lesson.read_record(text)
    out['prediction']=9
    assert lesson.read_record(text)['prediction']==prediction
    assert out['assessment']=='local_self_report_no_unseen_certification'
    for exposure in ('unknown','seen','self_reported_not_seen'):
        out['prior_exposure']=exposure
        assert lesson.read_record(lesson._seal(out))['prior_exposure']==exposure


@pytest.mark.parametrize('prediction',[True,None,float('nan'),float('inf')])
def test_invalid_forecast_rejected(prediction):
    with pytest.raises(ValueError):commitment(prediction)


def test_strict_json_schema_source_and_size_bounds():
    text=commitment()
    for schema in ('arena-extension-guided-practice','arena-blind-shapley-prediction'):
        out=lesson.read_record(text);out['schema']=schema
        with pytest.raises(ValueError):lesson.read_record(lesson._seal(out))
    with pytest.raises(ValueError,match='duplicate'):lesson.read_record('{"schema":1,"schema":2}')
    with pytest.raises(ValueError):lesson.read_record('{"value":NaN,"seal":"bad"}')
    with pytest.raises(ValueError,match='bounded'):lesson.read_record(' '*(lesson.RECORD_LIMIT+1))
    for source in (lesson.REFERENCE_METHODS+'\nx=1',lesson.REFERENCE_METHODS.replace('def restriction_coefficients(', '@print(1)\ndef restriction_coefficients(')):
        with pytest.raises(ValueError):lesson.freeze_inputs(source)
    with pytest.raises(ValueError):lesson.reveal_methods('{}')
    assert lesson.reveal_methods(text)==lesson.REFERENCE_METHODS
    out=lesson.read_record(text);out['schema_version']=True
    with pytest.raises(ValueError):lesson.read_record(lesson._seal(out))


@pytest.mark.parametrize('mutation',['method','helper','tests','environment','covariance','mask_order','dtype','case'])
def test_stale_identity_rejected_before_numerical_namespace(monkeypatch,mutation):
    text=commitment();out=lesson.read_record(text);source=lesson.REFERENCE_METHODS
    if mutation=='method':source=source.replace('coefficients =','coefficients  =',1)
    elif mutation in ('helper','tests'):out['identity'][mutation+'_sha256']='0'*64
    elif mutation=='environment':out['identity']['environment']['torch']='fictional'
    elif mutation=='covariance':out['identity']['inputs']['covariance']['sha256']='0'*64
    elif mutation=='mask_order':out['identity']['inputs']['coalition_masks']['sha256']='0'*64
    elif mutation=='dtype':out['identity']['inputs']['truth']['dtype']='torch.float32'
    else:out['case']['truth'][0]=99
    monkeypatch.setattr(lesson,'method_namespace',lambda *a,**k:(_ for _ in ()).throw(AssertionError('numerical namespace reached')))
    with pytest.raises(ValueError):lesson.run_experiment(lesson._seal(out),source)


def test_non_cpu_default_device_rejected():
    previous=torch.get_default_device()
    try:
        torch.set_default_device('meta')
        with pytest.raises(ValueError,match='CPU default device'):lesson.freeze_inputs(lesson.REFERENCE_METHODS)
    finally:
        torch.set_default_device(previous)


@pytest.mark.parametrize('mutation',['coefficient','coefficient_hash','risk','game','credit','stratum','shape','counts','bool_count','error','control','extra'])
def test_malformed_observed_records_rejected(diagnostic_case,mutation):
    out=lesson.read_record(fabricated_receipt());obs=out['observation'];result=obs['results']['conditional']
    if mutation=='coefficient':
        result['coefficients'][1][0]=.1;result['coefficients_sha256']=lesson._tensor_hash(result['coefficients'])
    elif mutation=='coefficient_hash':result['coefficients_sha256']='0'*64
    elif mutation=='risk':
        result['risks'][0]+=.1;result['risks_sha256']=lesson._tensor_hash(result['risks'])
    elif mutation=='game':
        result['game'][1]=.1;result['game_sha256']=lesson._tensor_hash(result['game'])
    elif mutation=='credit':result['shapley'][1]=.1
    elif mutation=='stratum':result['marginals_b'][1]=.1
    elif mutation=='shape':result['coefficients'][0].pop()
    elif mutation=='counts':obs['coalition_count']=81
    elif mutation=='bool_count':obs['condition_count']=True
    elif mutation=='error':obs['absolute_forecast_error']=.2
    elif mutation=='control':obs['controls']['empty_full_agreement']=False
    else:obs['unrecognized']=1
    with pytest.raises(ValueError):lesson.read_record(lesson._seal(out))


def test_model_free_replay_reflection_transfer_and_exclusive_export(diagnostic_case,tmp_path,monkeypatch):
    receipt=fabricated_receipt()
    for name in ('run_experiment','freeze_inputs','method_namespace','environment_identity','practice_test'):
        monkeypatch.setattr(lesson,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('numerical execution reached')))
    obs=lesson.replay(receipt);obs['delta_phi_b']=9
    assert lesson.replay(receipt)['delta_phi_b']==0
    attempt=lesson.reflect(receipt,'Fixture contrast matched the invented zero predictor',100.,.5,'Invented transfer forecast')
    assert lesson.read_attempt(attempt)['receipt']==receipt
    # Freeze identity is a stored fixture value, not a call to the patched entry point.
    primary=lesson.read_record(lesson.read_record(receipt)['commitment'])
    identity=deepcopy(primary['identity']);case=lesson._case('transfer')
    identity['case_sha256']=lesson.digest(case)
    identity['inputs']={k:{'shape':shape,'dtype':case['dtype'],'sha256':lesson._tensor_hash(rows)} for k,(shape,rows) in lesson._layout(case).items()}
    transfer=lesson.commit_transfer(attempt,identity,lesson.REFERENCE_METHODS)
    parsed=lesson.read_record(transfer)
    assert parsed['case']['variant']=='transfer' and parsed['prediction']==100.
    assert parsed['parent_attempt']==attempt
    for text,writer,reader,name in ((receipt,lesson.export_record,lesson.read_record,'receipt'),(attempt,lesson.export_attempt,lesson.read_attempt,'attempt'),(transfer,lesson.export_record,lesson.read_record,'transfer')):
        path=tmp_path/(name+'.json');writer(text,path)
        assert stat.S_IMODE(path.stat().st_mode)==0o600 and path.read_text()==text
        reader(path.read_text())
        with pytest.raises(FileExistsError):writer(text,path)
    import matplotlib
    matplotlib.use('Agg')
    figure=lesson.plot_receipt(receipt);assert len(figure.axes)==2
    import matplotlib.pyplot as plt
    plt.close(figure)
    broken=lesson.read_attempt(attempt);broken['receipt_sha256']='0'*64
    with pytest.raises(ValueError):lesson.read_attempt(lesson._seal(broken))
    broken=lesson.read_record(transfer);broken['prediction']=99.
    with pytest.raises(ValueError):lesson.read_record(lesson._seal(broken))


def test_primary_cannot_run_transfer_or_recurse_through_transfer_receipt(diagnostic_case):
    receipt=fabricated_receipt()
    attempt=lesson.reflect(receipt,'Fixture reflection',.2,.5,'Fixture transfer reason')
    with pytest.raises(ValueError):lesson.commit_prediction(lesson.freeze_inputs(lesson.REFERENCE_METHODS),lesson.REFERENCE_METHODS,.2,.5,'r',parent_attempt=attempt)
    with pytest.raises(ValueError):lesson.commit_prediction(lesson.freeze_inputs(lesson.REFERENCE_METHODS,'transfer'),lesson.REFERENCE_METHODS,.2,.5,'r',variant='transfer')
    broken=lesson.read_attempt(attempt);broken['transfer_case']['covariance'][0][0]=3
    with pytest.raises(ValueError):lesson.read_attempt(lesson._seal(broken))


def test_full_runner_on_invented_zero_fixture_never_uses_public_inputs(diagnostic_case,monkeypatch):
    source=lesson.REFERENCE_METHODS;parent=commitment()
    original=lesson.method_namespace
    def non_target_namespace(*args,**kwargs):
        ns=original(*args,**kwargs);function=ns['restriction_coefficients']
        def guarded(covariance,model,rule):
            assert covariance.tolist()!=PUBLIC['covariance'], 'public target must not run in fixtures'
            return function(covariance,model,rule)
        ns['restriction_coefficients']=guarded
        return ns
    monkeypatch.setattr(lesson,'method_namespace',non_target_namespace)
    text=lesson.run_experiment(parent,source)
    result=lesson.replay(text)
    assert result['delta_phi_b']==0 and result['condition_count']==10
    assert all(result['controls'].values())


def _stubbed_run(monkeypatch,restriction,assembly=None):
    source=lesson.REFERENCE_METHODS;parent=commitment();ns=lesson.method_namespace(source)
    ns['restriction_coefficients']=restriction
    if assembly is not None:ns['assemble_result']=assembly
    monkeypatch.setattr(lesson,'method_namespace',lambda *a,**k:ns)
    monkeypatch.setattr(lesson,'practice_test',lambda *a,**k:'fabricated independent fixture')
    return parent,source,ns


def test_false_assembly_retains_actual_arrays_and_failure_exports(diagnostic_case,monkeypatch,tmp_path):
    def zero(cov,b,rule):return cov.new_zeros((8,3))
    def false_assembly(coeff,risks,game,n,player):
        return {'coefficients':[[1.,1.,1.]]*8,'risks':risks.tolist(),'game':game.tolist(),
                'shapley':[0.]*3,'marginals_b':[0.]*3}
    parent,source,ns=_stubbed_run(monkeypatch,zero,false_assembly)
    with pytest.raises(lesson.ExperimentFailure,match='actual measured arrays') as caught:
        lesson.run_experiment(parent,source)
    text=caught.value.record_text;failed=lesson.replay_failure(text)
    assert lesson.read_record(text)['commitment']==parent
    assert failed['measured']['conditional']['coefficients'][1]==[0.,0.,0.]
    assert failed['measured']['conditional']['risks']==[1.25]*8
    assert failed['results']=={} and 'absolute_forecast_error' not in failed
    lesson.export_record(text,tmp_path/'failure.json')
    with pytest.raises(ValueError):lesson.replay(text)
    with pytest.raises(ValueError):lesson.reflect(text,'x',.1,.5,'r')


@pytest.mark.parametrize('stage',['restriction','risk','assembly'])
def test_input_mutation_rejected_and_measurements_retained(diagnostic_case,monkeypatch,stage):
    def zero(cov,b,rule):return cov.new_zeros((8,3))
    parent,source,ns=_stubbed_run(monkeypatch,zero)
    name={'restriction':'restriction_coefficients','risk':'risk_game','assembly':'assemble_result'}[stage]
    original=ns[name]
    def mutate(*args):
        result=original(*args);args[0][0,0]+=.1
        return result
    ns[name]=mutate
    with pytest.raises(lesson.ExperimentFailure,match='mutated actual input arrays') as failed:
        lesson.run_experiment(parent,source)
    assert 'conditional' in lesson.replay_failure(failed.value.record_text)['measured']


@pytest.mark.parametrize('stage',['restriction','risk','assembly'])
def test_runtime_mutation_rejected_after_every_call(diagnostic_case,monkeypatch,stage):
    def zero(cov,b,rule):return cov.new_zeros((8,3))
    parent,source,ns=_stubbed_run(monkeypatch,zero)
    name={'restriction':'restriction_coefficients','risk':'risk_game','assembly':'assemble_result'}[stage]
    original=ns[name];previous=torch.get_default_dtype()
    def mutate(*args):
        result=original(*args)
        torch.set_default_dtype(torch.float32 if previous==torch.float64 else torch.float64)
        return result
    ns[name]=mutate
    try:
        with pytest.raises(lesson.ExperimentFailure,match='frozen inputs, helper, tests or environment') as failed:
            lesson.run_experiment(parent,source)
        assert list(lesson.replay_failure(failed.value.record_text)['measured'])==['conditional']
    finally:
        torch.set_default_dtype(previous)


def test_case_mutation_after_return_still_preserves_failure(diagnostic_case,monkeypatch):
    def mutate(cov,b,rule):
        result=cov.new_zeros((8,3));lesson.CASE['truth'][0]=99
        return result
    parent,source,ns=_stubbed_run(monkeypatch,mutate)
    with pytest.raises(lesson.ExperimentFailure,match='frozen inputs') as failed:
        lesson.run_experiment(parent,source)
    assert lesson.replay_failure(failed.value.record_text)['measured']['conditional']['coefficients'][0]==[0.,0.,0.]


def test_overbudget_call_preserves_returned_actual_arrays(diagnostic_case,monkeypatch):
    clock=[0.]
    def late(cov,b,rule):
        clock[0]=121.;return cov.new_zeros((8,3))
    parent,source,ns=_stubbed_run(monkeypatch,late)
    monkeypatch.setattr(lesson.time,'monotonic',lambda:clock[0])
    with pytest.raises(lesson.ExperimentFailure,match='CPU budget exceeded') as failed:
        lesson.run_experiment(parent,source)
    record=lesson.replay_failure(failed.value.record_text)
    assert record['elapsed_seconds']==121. and record['measured']['conditional']['coefficients'][0]==[0.,0.,0.]


def test_false_controls_keep_all_actual_tables_unscored(diagnostic_case,monkeypatch):
    original=lesson._control_checks
    def false(results,case):
        controls=original(results,case);controls['empty_full_agreement']=False
        return controls
    monkeypatch.setattr(lesson,'_control_checks',false)
    with pytest.raises(lesson.ExperimentFailure,match='control failure') as failed:
        lesson.run_experiment(commitment(),lesson.REFERENCE_METHODS)
    payload=lesson.replay_failure(failed.value.record_text)
    assert len(payload['measured'])==len(payload['results'])==10
    assert payload['controls']['empty_full_agreement'] is False
    assert 'delta_phi_b' not in payload and 'absolute_forecast_error' not in payload


def test_largest_writer_reader_roundtrip_uses_diagnostic_only(diagnostic_case):
    source=lesson.REFERENCE_METHODS+'\n#'+('x'*(99_000-len(lesson.REFERENCE_METHODS)-2))
    text=lesson.commit_prediction(lesson.freeze_inputs(source),source,100.,.5,'🧠'*20_000)
    assert lesson.read_record(text)['method_source']==source
    payload=lesson.read_record(fabricated_receipt())
    payload['commitment']=text;payload['commitment_sha256']=hashlib.sha256(text.encode()).hexdigest()
    payload['observation']['absolute_forecast_error']=100.
    receipt=lesson._seal(payload);lesson.read_record(receipt)
    attempt=lesson.reflect(receipt,'🧠'*20_000,100.,.5,'r'*20_000)
    transfer=lesson.commit_transfer(attempt,lesson.freeze_inputs(source,'transfer'),source)
    assert len(transfer.encode())<=lesson.RECORD_LIMIT
    assert lesson.read_record(transfer)['parent_attempt']==attempt
    assert len(attempt.encode())<=lesson.ATTEMPT_LIMIT


def test_generator_preserves_edits_outputs_widgets_records_and_readonly_check(tmp_path):
    gen=generator();expected=gen.make_notebook();path=tmp_path/'work.ipynb'
    assert gen.synchronize(path,expected,check=False)
    old=nbformat.read(path,as_version=4);code=next(c for c in old.cells if c.cell_type=='code')
    code.outputs=[nbformat.v4.new_output('stream',name='stdout',text='private observation')];code.execution_count=1
    old.metadata['widgets']={'private':'state'}
    nbformat.write(old,path);before=path.read_bytes()
    assert gen.synchronize(path,expected,check=True) and path.read_bytes()==before
    newer=deepcopy(expected);newer.cells[0].source+='\nnew teaching text'
    assert not gen.synchronize(path,newer,check=False) and path.read_bytes()==before
    old.cells[1].source+='\n# learner edit';nbformat.write(old,path);before=path.read_bytes()
    assert not gen.synchronize(path,expected,check=False) and path.read_bytes()==before
    malformed=tmp_path/'malformed.ipynb';malformed.write_text('private broken JSON')
    assert not gen.synchronize(malformed,expected,check=False) and malformed.read_text()=='private broken JSON'
    missing=tmp_path/'missing.ipynb';assert not gen.synchronize(missing,expected,check=True) and not missing.exists()


def test_generated_method_join_and_practice_cells_are_executable_without_target():
    import ast
    functions={n.name:ast.get_source_segment(lesson.REFERENCE_METHODS,n)+'\n' for n in ast.parse(lesson.REFERENCE_METHODS).body}
    namespace={'lesson':lesson}
    for cell,stage,name in zip([c for c in lesson.build_cells() if c['id'] in [f'sage-{s}' for s in lesson.STAGES]],lesson.STAGES,lesson.METHOD_NAMES,strict=True):
        source=cell['source'].replace(lesson._stub(name),functions[name]);exec(source,namespace)
    freeze=next(c['source'] for c in lesson.build_cells() if c['id']=='sage-freeze')
    exec(freeze,namespace)
    assert tuple(n.name for n in ast.parse(namespace['METHOD_SOURCE']).body)==lesson.METHOD_NAMES


@pytest.mark.parametrize('kind',['widgets','execution','trusted'])
def test_stale_generator_preserves_metadata_without_code_outputs(tmp_path,kind):
    gen=generator();old=gen.make_notebook();path=tmp_path/'state.ipynb'
    if kind=='widgets':old.metadata['widgets']={'state':'private widget data'}
    elif kind=='execution':old.cells[1].metadata['execution']={'iopub.status.busy':'fixture timestamp'}
    else:old.cells[1].metadata['trusted']=True
    path.write_text(json.dumps(old));before=path.read_bytes()
    newer=gen.make_notebook();newer.cells[0].source+='\nnew generation'
    assert not gen.synchronize(path,newer,check=False)
    assert path.read_bytes()==before


def test_invalid_returned_allocations_are_retained_unscored(diagnostic_case,monkeypatch):
    def zero(cov,b,rule):return cov.new_zeros((8,3))
    parent,source,namespace=_stubbed_run(monkeypatch,zero)
    original=namespace['assemble_result']
    def incorrect(*args):
        result=original(*args);result['shapley'][1]=.1
        return result
    namespace['assemble_result']=incorrect
    with pytest.raises(lesson.ExperimentFailure,match='six-ordering') as failed:
        lesson.run_experiment(parent,source)
    failure=lesson.replay_failure(failed.value.record_text)
    assert failure['allocations']['conditional']['shapley'][1]==.1
    assert failure['results']=={} and 'delta_phi_b' not in failure


@pytest.mark.parametrize('kind',['shape','model_shape','nonfinite','singular','rule'])
def test_restriction_rejects_invalid_non_target_inputs(kind):
    function=lesson.method_namespace(lesson.REFERENCE_METHODS)['restriction_coefficients']
    covariance=torch.tensor([[2.,.1],[.1,1.]],dtype=torch.float64)
    model=torch.tensor([.2,.4],dtype=torch.float64);rule='conditional'
    if kind=='shape':covariance=covariance[:1]
    elif kind=='model_shape':model=model[:,None]
    elif kind=='nonfinite':covariance[0,0]=float('nan')
    elif kind=='singular':covariance[:]=1
    else:rule='unrecognized'
    with pytest.raises(ValueError):function(covariance,model,rule)
