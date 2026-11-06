"""Blind variant structure, records, and independent core-math fixtures; no target run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import stat

import nbformat
import pytest

from mastery_hub.units.circuit_prediction import lesson

ROOT = Path(__file__).resolve().parents[2]


def generator():
    path = ROOT / 'scripts/build_circuit_prediction_notebook.py'
    spec = importlib.util.spec_from_file_location('circuit_generator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_record():
    # Entirely fabricated record inputs; no target model or benchmark execution.
    import hashlib
    method = 'fixture method text'
    mask=[[True]*19 for _ in range(7)]
    planted=[[u==0 and v<7 for v in range(19)] for u in range(7)]
    def tensor(shape,dtype,sha='0'*64):
        return {'shape':shape,'dtype':dtype,'sha256':sha}
    identity={'case_sha256':lesson.digest(lesson.CASE),
        'method_sha256':hashlib.sha256(method.encode()).hexdigest(),
        'helper_sha256':'0'*64,'plumbing_sha256':'0'*64,'environment':{'fixture':'runtime'},
        'sender_order':['input']+[f'{l}.{h}' for l in range(2) for h in range(3)],
        'receiver_order':[f'{l}.{h}<{c}>' for l in range(2) for c in 'qkv' for h in range(3)]+['logits'],
        'tensors':{'clean':tensor([64,17],'torch.int64'),'corrupt':tensor([64,17],'torch.int64'),
            'valid_edges':tensor([7,19],'torch.bool',hashlib.sha256(bytes(v for row in mask for v in row)).hexdigest()),
            'planted_edges':tensor([7,19],'torch.bool',hashlib.sha256(bytes(v for row in planted for v in row)).hexdigest())},
        'model_metadata':{label:[{'name':'','training':False,'config':{'fixture':True}}]
            for label in ('gain10','gain1','gain1_no_noise')},
        'ordered_model_states':{label:[dict(name=n,shape=shape,dtype=dtype,sha256='0'*64)
            for n,shape,dtype in lesson.state_layout()] for label in ('gain10','gain1','gain1_no_noise')}}
    return lesson.commit_prediction(identity,method,.1,.6,'Fixture causal explanation','unknown')


def fixture_receipt():
    import hashlib
    text=fixture_record()
    record=lesson.read_record(text)
    planted=[[u==0 and v<7 for v in range(19)] for u in range(7)]
    scores=[[float(v) for v in row] for row in planted]
    result={'auc':{m:1. for m in ('exact','EAP','IG10')},
        'scores':{m:deepcopy(scores) for m in ('exact','EAP','IG10')},
        'task_accuracy':1.,'answer_logprob':-1.}
    record.update(status='observed',commitment_sha256=hashlib.sha256(text.encode()).hexdigest(),
        observation={'kind':'fresh_cpu_inference','results':{k:deepcopy(result) for k in ('gain10','gain1','gain1_no_noise')},
            'delta_eap_auc':0.,'elapsed_seconds':1.,'sender_order':record['identity']['sender_order'],
            'receiver_order':record['identity']['receiver_order'],'valid_mask':[[True]*19 for _ in range(7)],
            'planted_mask':planted,'absolute_forecast_error':.1})
    receipt=lesson._seal(record)
    lesson.read_record(receipt)
    return receipt


def test_case_contract_is_single_output_gain_intervention():
    case = lesson.CASE
    assert case['gains'] == [10., 1.] and case['sharpness'] == 20 and case['sigma'] == .03
    assert case['noise_seed'] == case['data_seed'] == 0
    assert case['n'] == 64 and case['ig_steps'] == 10 and case['threads'] == 1
    assert 'logprob' in case['metric'] and case['budget_seconds'] == 120
    assert case['dtype'] == 'torch.float32'
    assert len(lesson.PLANTED) == 7


def test_notebook_no_target_answers_or_outputs_before_commit():
    cells = lesson.build_cells()
    nb = nbformat.v4.new_notebook(cells=[nbformat.from_dict(c) for c in cells])
    nbformat.validate(nb)
    assert len({c['id'] for c in cells}) == len(cells)
    assert cells == lesson.build_cells()
    run = next(i for i,c in enumerate(cells) if c['metadata']['stage'] == 'run')
    commit = next(i for i,c in enumerate(cells) if c['id'] == 'circuit-commit')
    freeze = next(i for i,c in enumerate(cells) if c['id'] == 'circuit-freeze')
    assert freeze < commit < run
    for i,cell in enumerate(cells):
        text = cell['source']
        assert 'pip install' not in text and 'from_pretrained' not in text
        assert '<img' not in text and 'base64' not in text
        if cell['cell_type'] == 'code':
            compile(text, '<cell>', 'exec')
            assert cell['outputs'] == [] and cell['execution_count'] is None
        if i < commit:
            assert 'run_experiment(' not in text and 'reveal_methods(' not in text
            assert 'auc of about' not in text.lower() and 'collapses' not in text.lower()
            assert 'delta_eap_auc' not in text and 'plot_receipt(' not in text
    assert sum(c['cell_type'] == 'code' and 'NotImplementedError' in c['source'] for c in cells) == 4
    for stage in ('compiler','scores','auc','assembly'):
        relevant = next(c for c in cells if c['id'] == 'circuit-'+stage)
        assert 'practice_test(' in relevant['source']
        # Core implementations are legible multiline cell contents.
        assert "_SOURCE = '''\ndef " in relevant['source']
    assert 'No target result' in cells[0]['source']


@pytest.mark.parametrize('stage', ['compiler','scores','auc','assembly'])
def test_core_math_independent_practice_fixture(stage):
    ns = lesson.practice_namespace(lesson.REFERENCE_METHODS)
    assert lesson.practice_test(ns,stage).endswith('passed')


def test_wrong_sign_and_tie_handling_are_caught():
    ns = lesson.practice_namespace(lesson.REFERENCE_METHODS)
    ns['patching_scores'] = lambda *a: __import__('torch').tensor([[-3.,2.,0.]])
    with pytest.raises(AssertionError):
        lesson.practice_test(ns,'scores')
    ns = lesson.practice_namespace(lesson.REFERENCE_METHODS)
    ns['roc_auc'] = lambda *a: 1.
    with pytest.raises(AssertionError):
        lesson.practice_test(ns,'auc')


def test_commitment_is_immutable_and_roundtrips():
    text = fixture_record()
    record = lesson.read_record(text)
    assert record['status'] == 'committed'
    record['prediction'] = -.5
    assert lesson.read_record(text)['prediction'] == .1
    damaged = json.loads(text)
    damaged['prediction'] = .9
    with pytest.raises(ValueError,match='seal'):
        lesson.read_record(json.dumps(damaged))
    for exposure in ('unknown','seen','self_reported_not_seen'):
        record = lesson.read_record(text)
        record['prior_exposure'] = exposure
        assert lesson.read_record(lesson._seal(record))['prior_exposure'] == exposure
    assert 'unseen_certification' in record['assessment']


@pytest.mark.parametrize('value', [None, True, float('nan'), 2., -2.])
def test_invalid_numeric_forecasts_rejected(value):
    record = lesson.read_record(fixture_record())
    record['prediction'] = value
    with pytest.raises(ValueError):
        lesson.read_record(lesson._seal(record))


def test_schema_separation_and_solution_gate():
    for schema,version in [('arena-extension-guided-practice',1),(lesson.SCHEMA,999),('quadratic',1)]:
        record = lesson.read_record(fixture_record())
        record.update(schema=schema,schema_version=version)
        with pytest.raises(ValueError):
            lesson.read_record(lesson._seal(record))
    with pytest.raises(ValueError):
        lesson.reveal_methods('{}')
    assert 'def eap_scores' in lesson.reveal_methods(fixture_record())


def test_stale_method_fails_before_prepare(monkeypatch):
    record = lesson.read_record(fixture_record())
    record['identity'].update(helper_sha256=lesson._file(lesson.__file__),
        plumbing_sha256=lesson._file(lesson.UTILS),environment={'fixture':'runtime'})
    monkeypatch.setattr(lesson,'environment_identity',lambda:{'fixture':'runtime'})
    def forbidden(*args):
        raise AssertionError('compilation / model work must not start')
    monkeypatch.setattr(lesson,'prepare',forbidden)
    with pytest.raises(ValueError,match='stale'):
        lesson.run_experiment(lesson._seal(record),'edited method')
    record['identity']['helper_sha256'] = '0'*64
    with pytest.raises(ValueError,match='stale'):
        lesson.run_experiment(lesson._seal(record),record['method_source'])


def test_stale_tensor_or_weight_rejected_before_forward(monkeypatch):
    record = lesson.read_record(fixture_record())
    record['identity'].update(helper_sha256=lesson._file(lesson.__file__),
        plumbing_sha256=lesson._file(lesson.UTILS),environment={'fixture':'runtime'})
    monkeypatch.setattr(lesson,'environment_identity',lambda:{'fixture':'runtime'})
    monkeypatch.setattr(lesson,'prepare',lambda _: ({'different':'ordered weight hash'},))
    with pytest.raises(ValueError,match='stale tensors'):
        lesson.run_experiment(lesson._seal(record),record['method_source'])


def test_replay_and_export_do_not_evaluate_models_or_overwrite(tmp_path,monkeypatch):
    receipt = fixture_receipt()
    def forbidden(*args):
        raise AssertionError('replay must not execute methods or instantiate a model')
    for name in ('prepare','namespace','environment_identity','run_experiment','_plumbing'):
        monkeypatch.setattr(lesson,name,forbidden)
    observed = lesson.replay(receipt)
    observed['delta_eap_auc'] = 99
    assert lesson.replay(receipt)['delta_eap_auc'] == 0.
    path = lesson.export_record(receipt,tmp_path/'private.json')
    assert path.read_text() == receipt
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        lesson.export_record(receipt,path)
    assert path.read_text() == receipt


def test_generator_preserves_learner_work_and_saved_receipts(tmp_path):
    gen = generator()
    fresh = gen.make_notebook()
    path = tmp_path/'exercises.ipynb'
    assert gen.synchronize(path,fresh,check=False)
    assert gen.synchronize(path,fresh,check=True)
    old = nbformat.read(path,as_version=4)
    code = next(c for c in old.cells if c.cell_type=='code')
    code.source += '\n# My filled solution and receipt\n'
    code.execution_count = 1
    code.outputs = [nbformat.v4.new_output('stream',name='stdout',text='private saved receipt')]
    nbformat.write(old,path)
    before = path.read_bytes()
    assert not gen.synchronize(path,fresh,check=False)
    assert path.read_bytes() == before
    # Same source with saved output is current and must remain byte-exact.
    nbformat.write(fresh,path)
    old = nbformat.read(path,as_version=4)
    code = next(c for c in old.cells if c.cell_type=='code')
    code.execution_count = 1
    code.outputs = [nbformat.v4.new_output('stream',name='stdout',text='stored forecast')]
    nbformat.write(old,path)
    before = path.read_bytes()
    assert gen.synchronize(path,fresh,check=True)
    assert gen.synchronize(path,fresh,check=False)
    assert path.read_bytes() == before


@pytest.mark.parametrize('change', ['version_bool','extra','case_bool','missing_identity','incomplete_observation',
    'bad_parent','bad_mask','nan_score','wrong_auc','wrong_delta','late_receipt','state_dtype','state_shape'])
def test_strict_schema_rejects_resealed_malformed_records(change):
    record=lesson.read_record(fixture_receipt())
    if change=='version_bool': record['schema_version']=True
    if change=='extra': record['unexpected']='field'
    if change=='case_bool': record['case']['noise_seed']=False
    if change=='missing_identity': del record['identity']['tensors']
    if change=='incomplete_observation': record['observation']={'delta_eap_auc':100}
    if change=='bad_parent': record['commitment_sha256']='1'*64
    if change=='bad_mask':
        record['observation']['planted_mask'][0][0]=False
        record['observation']['planted_mask'][0][8]=True
    if change=='nan_score': record['observation']['results']['gain1']['scores']['EAP'][0][0]=float('nan')
    if change=='wrong_auc': record['observation']['results']['gain1']['auc']['EAP']=.8
    if change=='wrong_delta': record['observation']['delta_eap_auc']=.3
    if change=='late_receipt': record['observation']['elapsed_seconds']=121.
    if change=='state_dtype': record['identity']['ordered_model_states']['gain10'][0]['dtype']='torch.float64'
    if change=='state_shape': record['identity']['ordered_model_states']['gain10'][0]['shape']=[1]
    with pytest.raises(ValueError): lesson.read_record(lesson._seal(record))


def test_duplicate_json_keys_rejected():
    text=fixture_record()
    text=text.replace('"prediction": 0.1','"prediction": 0.1, "prediction": 0.1')
    with pytest.raises(ValueError,match='duplicate'):
        lesson.read_record(text)


def test_incomplete_compiler_or_noise_fails_immediately():
    ns=lesson.practice_namespace(lesson.REFERENCE_METHODS)
    def stub(*a): raise NotImplementedError('unimplemented')
    ns['compile_induction']=stub
    with pytest.raises(NotImplementedError): lesson.practice_test(ns,'compiler')
    ns=lesson.practice_namespace(lesson.REFERENCE_METHODS)
    ns['add_weight_noise']=lambda model,*args:model
    with pytest.raises(AssertionError): lesson.practice_test(ns,'compiler')


def test_reflection_archive_preserves_receipt_and_transfer(tmp_path,monkeypatch):
    receipt=fixture_receipt()
    attempt=lesson.reflect(receipt,'The control changes my explanation.','Change the seed with fixed metric.',
        'The paired AUC difference will be near zero.',.6,'A second seed changes the attention-noise paths.')
    out=lesson.read_attempt(attempt)
    assert out['receipt']==receipt
    assert 'near zero' in out['transfer_prediction']
    def forbidden(*a): raise AssertionError('archive must not run model')
    monkeypatch.setattr(lesson,'prepare',forbidden)
    path=lesson.export_attempt(attempt,tmp_path/'attempt.json')
    assert lesson.read_attempt(path.read_text())==out
    with pytest.raises(FileExistsError): lesson.export_attempt(attempt,path)
    with pytest.raises(ValueError): lesson.reflect(receipt,'','q','p',.5,'r')
    out['receipt']=out['receipt']+' '
    with pytest.raises(ValueError): lesson.read_attempt(lesson._seal(out))


def runtime_stub_case():
    """Non-target model/graph stand-ins; never construct a compiled transformer."""
    from types import SimpleNamespace
    import torch as t
    record=lesson.read_record(fixture_record())
    truth=t.tensor([[u==0 and v<7 for v in range(19)] for u in range(7)],dtype=t.bool)
    class Config:
        use_attn_scale=True
        def to_dict(self): return {'use_attn_scale':self.use_attn_scale,'dtype':t.float32}
    class Model(t.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg=Config()
            self.weight=t.nn.Parameter(t.zeros(1,dtype=t.float32),requires_grad=False)
        def forward(self, tokens):
            return t.zeros(64,17,16,dtype=t.float32)
    class Graph:
        def __init__(self,model):
            self.model=model
            self.edge_mask=t.ones(7,19,dtype=t.bool)
            self.sender_names=list(record['identity']['sender_order'])
            self.receiver_names=list(record['identity']['receiver_order'])
            self.n_layers,self.n_heads,self.logits=2,3,18
        def sender_outputs(self,tokens): return t.zeros(1,dtype=t.float32)
        def mask_from_edges(self,edges): return truth.clone()
    models={label:Model() for label in ('gain10','gain1','gain1_no_noise')}
    graphs={label:Graph(model) for label,model in models.items()}
    clean=t.zeros(64,17,dtype=t.int64)
    corrupt=t.ones(64,17,dtype=t.int64)
    def scorer(*args): return truth.float()
    ns=lesson.practice_namespace(lesson.REFERENCE_METHODS)
    ns.update(patching_scores=scorer,eap_scores=scorer,eap_ig_scores=scorer)
    prepared=(record['identity'],ns,models,graphs,clean,corrupt)
    return record,prepared


def test_run_rejects_scorer_overrun_before_successful_receipt(monkeypatch):
    import torch as t
    record,prepared=runtime_stub_case()
    now=[0.]
    monkeypatch.setattr(lesson.time,'monotonic',lambda:now[0])
    def too_late(*args):
        now[0]=121.
        return t.zeros(7,19,dtype=t.float32)
    prepared[1]['eap_ig_scores']=too_late
    monkeypatch.setattr(lesson,'assert_current',lambda *args:(record,prepared))
    with pytest.raises(TimeoutError,match='cooperative'):
        lesson.run_experiment(fixture_record(),'fixture method text')


@pytest.mark.parametrize('mutation',['clean','corrupt','valid_mask','sender_order','receiver_order',
    'model_weights','other_condition_weights','corrupt_cache','graph_model','graph_metadata'])
def test_student_scorer_mutations_rejected(monkeypatch,mutation):
    import torch as t
    record,prepared=runtime_stub_case()
    _,ns,models,graphs,clean,corrupt=prepared
    original=ns['patching_scores']
    def mutating(graph,tokens,cache,metric):
        value=original(graph,tokens,cache,metric)
        if mutation=='clean': tokens[0,0]=2
        if mutation=='corrupt': corrupt[0,0]=2
        if mutation=='valid_mask': graph.edge_mask[0,8]=False
        if mutation=='sender_order': graph.sender_names.reverse()
        if mutation=='receiver_order': graph.receiver_names.reverse()
        if mutation=='model_weights': graph.model.weight.add_(1)
        if mutation=='other_condition_weights': models['gain1'].weight.add_(1)
        if mutation=='corrupt_cache': cache.add_(1)
        if mutation=='graph_model': graph.model=models['gain1']
        if mutation=='graph_metadata': graph.logits=0
        return value
    ns['patching_scores']=mutating
    monkeypatch.setattr(lesson,'assert_current',lambda *args:(record,prepared))
    with pytest.raises(ValueError,match='mutated frozen|graph model changed'):
        lesson.run_experiment(fixture_record(),'fixture method text')


@pytest.mark.parametrize('fabrication',['score_table','task_accuracy','answer_logprob'])
def test_student_assembly_cannot_fabricate_measurements(monkeypatch,fabrication):
    record,prepared=runtime_stub_case()
    ns=prepared[1]
    original=ns['assemble_result']
    def fabricated(*args):
        value=original(*args)
        if fabrication=='score_table':
            # A completely different table with a consistent chance AUC would
            # pass standalone record validation, but is not the actual scores.
            value['scores']={m:[[0.]*19 for _ in range(7)] for m in ('exact','EAP','IG10')}
            value['auc']={m:.5 for m in ('exact','EAP','IG10')}
        elif fabrication=='task_accuracy': value['task_accuracy']=0.
        else: value['answer_logprob']=0.
        return value
    ns['assemble_result']=fabricated
    monkeypatch.setattr(lesson,'assert_current',lambda *args:(record,prepared))
    with pytest.raises(ValueError,match='assembled scores|assembled task behavior'):
        lesson.run_experiment(fixture_record(),'fixture method text')


def test_assembly_mutation_of_model_is_rejected(monkeypatch):
    record,prepared=runtime_stub_case()
    ns=prepared[1]
    original=ns['assemble_result']
    def mutating(scores,graph,*args):
        value=original(scores,graph,*args)
        graph.model.weight.add_(1)
        return value
    ns['assemble_result']=mutating
    monkeypatch.setattr(lesson,'assert_current',lambda *args:(record,prepared))
    with pytest.raises(ValueError,match='mutated frozen'):
        lesson.run_experiment(fixture_record(),'fixture method text')


def test_captured_scores_and_earlier_tables_are_not_mutable_aliases(monkeypatch):
    record,prepared=runtime_stub_case()
    ns=prepared[1]
    shared=prepared[3]['gain10'].mask_from_edges(lesson.PLANTED).float()
    calls=[0]
    def aliased(*args):
        calls[0]+=1
        shared.copy_(prepared[3]['gain10'].mask_from_edges(lesson.PLANTED).float()*calls[0])
        return shared
    for name in ('patching_scores','eap_scores','eap_ig_scores'): ns[name]=aliased
    original=ns['assemble_result']
    old_tables=[]
    def assembled(*args):
        # Mutate a previously returned table during the next condition. The
        # helper must have retained its own copy of that completed condition.
        if old_tables: old_tables[-1]['scores']['exact'][0][0]=999.
        value=original(*args)
        old_tables.append(value)
        return value
    ns['assemble_result']=assembled
    monkeypatch.setattr(lesson,'assert_current',lambda *args:(record,prepared))
    receipt=lesson.run_experiment(fixture_record(),'fixture method text')
    results=lesson.replay(receipt)['results']
    assert results['gain10']['scores']['exact'][0][0]==1.
    assert results['gain10']['scores']['EAP'][0][0]==2.
    assert results['gain10']['scores']['IG10'][0][0]==3.
    assert results['gain1']['scores']['exact'][0][0]==4.


def test_notebook_method_text_join_and_practice_execution():
    import ast
    cells=lesson.build_cells()
    ns={'lesson':lesson}
    for stage,names in [('compiler',('compile_select_aggregate','compile_induction','add_weight_noise')),
        ('scores',('patching_scores','eap_scores','eap_ig_scores')),('auc',('roc_auc',)),('assembly',('assemble_result',))]:
        source='\n\n'.join(ast.get_source_segment(lesson.REFERENCE_METHODS,node)
            for node in ast.parse(lesson.REFERENCE_METHODS).body if isinstance(node,ast.FunctionDef) and node.name in names)
        ns[stage.upper()+'_SOURCE']=source
        # Use the exact generated immediate-test lines, replacing only filled source.
        cell=next(c for c in cells if c['id']=='circuit-'+stage)
        lines=cell['source'].split("'''",2)[2]
        exec(lines,ns)
    freeze=next(c['source'] for c in cells if c['id']=='circuit-freeze')
    exec(freeze.split('\n# The code')[0],ns)
    tree=ast.parse(ns['METHOD_SOURCE'])
    assert {n.name for n in tree.body}==set(lesson.METHOD_NAMES)


@pytest.mark.parametrize('dtype_name',['float32','float64'])
def test_practice_fixtures_tolerate_external_default_dtype(dtype_name):
    import torch as t
    previous=t.get_default_dtype()
    try:
        t.set_default_dtype(getattr(t,dtype_name))
        ns=lesson.practice_namespace(lesson.REFERENCE_METHODS)
        for stage in ('compiler','scores','auc','assembly'):
            lesson.practice_test(ns,stage)
    finally:
        t.set_default_dtype(previous)


@pytest.mark.parametrize('setting',['dtype','threads','deterministic','default_device','model_config','training'])
def test_student_runtime_environment_or_config_changes_rejected(monkeypatch,setting):
    import torch as t
    record,prepared=runtime_stub_case()
    ns=prepared[1]
    original=ns['patching_scores']
    dtype=t.get_default_dtype()
    threads=t.get_num_threads()
    deterministic=t.are_deterministic_algorithms_enabled()
    default_device=t.get_default_device()
    calls=[]
    def mutating(graph,*args):
        value=original(graph,*args)
        if setting=='dtype': t.set_default_dtype(t.float64 if dtype==t.float32 else t.float32)
        if setting=='threads': t.set_num_threads(2 if threads==1 else 1)
        if setting=='deterministic': t.use_deterministic_algorithms(not deterministic)
        if setting=='default_device': t.set_default_device('meta')
        if setting=='model_config': graph.model.cfg.use_attn_scale=False
        if setting=='training': graph.model.eval()
        return value
    def forbidden(*args):
        calls.append('following scorer called')
        raise AssertionError('mutation must fail before another scorer')
    ns['patching_scores']=mutating
    ns['eap_scores']=forbidden
    monkeypatch.setattr(lesson,'assert_current',lambda *args:(record,prepared))
    try:
        with pytest.raises(ValueError,match='mutated frozen'):
            lesson.run_experiment(fixture_record(),'fixture method text')
        assert not calls
    finally:
        t.set_default_device(default_device)
        t.set_default_dtype(dtype)
        t.set_num_threads(threads)
        t.use_deterministic_algorithms(deterministic)


def test_prepare_rejects_non_cpu_default_before_compiler(monkeypatch):
    import torch as t
    device=t.get_default_device();dtype=t.get_default_dtype();threads=t.get_num_threads()
    monkeypatch.setattr(lesson,'namespace',lambda *args:(_ for _ in ()).throw(AssertionError('compiler ran')))
    try:
        t.set_num_threads(1);t.set_default_dtype(t.float32);t.set_default_device('meta')
        with pytest.raises(ValueError,match='CPU default device'):
            lesson.prepare('fixture')
    finally:
        t.set_default_device(device);t.set_default_dtype(dtype);t.set_num_threads(threads)


def test_historical_v1_receipts_remain_strict_replay_only():
    import hashlib
    record=lesson.read_record(fixture_receipt())
    record['schema_version']=1
    del record['identity']['model_metadata']
    committed={k:v for k,v in record.items() if k not in ('observation','commitment_sha256')}
    committed['status']='committed'
    committed_text=lesson._seal(committed)
    record['commitment_sha256']=hashlib.sha256(committed_text.encode()).hexdigest()
    historic=lesson._seal(record)
    assert lesson.replay(historic)['delta_eap_auc']==0.
    with pytest.raises(ValueError,match='replay-only'):
        lesson.assert_current(committed_text,'fixture method text')
    record['observation']['delta_eap_auc']=.5
    with pytest.raises(ValueError): lesson.replay(lesson._seal(record))


def test_current_commitment_requires_frozen_module_config():
    record=lesson.read_record(fixture_record())
    assert record['schema_version']==2
    del record['identity']['model_metadata']
    with pytest.raises(ValueError,match='identity'):
        lesson.commit_prediction(record['identity'],record['method_source'],.1,.6,'fixture reason')



def test_freeze_real_compiled_fixture_configuration_without_any_forward(monkeypatch):
    import torch as t
    from transformer_lens import HookedTransformer
    dtype=t.get_default_dtype();device=t.get_default_device();threads=t.get_num_threads()
    def forbidden(*args,**kwargs): raise AssertionError('freeze must never run a model forward')
    monkeypatch.setattr(HookedTransformer,'forward',forbidden)
    try:
        t.set_num_threads(1);t.set_default_dtype(t.float32);t.set_default_device('cpu')
        identity=lesson.freeze_inputs(lesson.REFERENCE_METHODS)
        for modules in identity['model_metadata'].values():
            config=modules[0]['config']
            assert config['attn_scale']['python_type']=='numpy.float64'
            assert config['initializer_range']['python_type']=='numpy.float64'
            assert config['dtype']=={'python_type':'dtype','value':'torch.float32'}
        record=lesson.commit_prediction(identity,lesson.REFERENCE_METHODS,0.,.5,
            'No-forward engineering freeze fixture only; no scientific outcome is evaluated.', 'seen')
        assert lesson.read_record(record)['schema_version']==2
    finally:
        t.set_default_device(device);t.set_default_dtype(dtype);t.set_num_threads(threads)


def test_configuration_numpy_scalars_are_finite_and_type_bound():
    import numpy as np
    value=lesson._configuration_value({'scale':np.float64(2.)})
    assert value=={'scale':{'python_type':'numpy.float64','value':2.}}
    assert lesson.digest(value)!=lesson.digest({'scale':2.})
    with pytest.raises(ValueError): lesson._configuration_value(np.float64('nan'))
