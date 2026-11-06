"""Source/record/generator checks plus DISTINCT four-node DEV fixture arithmetic.

No target graph construction, activation, score, path, intervention or metric call.
Fabricated measurement dictionaries below are four-node exposed fixtures, never
sealed as successful target receipts or written as empirical evidence.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat

import nbformat
import pytest

from mastery_hub.units.graph_prediction import lesson

ROOT = Path(__file__).resolve().parents[2]


def generator():
    spec=importlib.util.spec_from_file_location('graph_generator',ROOT/'scripts/build_graph_prediction_notebook.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def fixed_cpu_runtime():
    import torch as t
    t.set_num_threads(1);t.set_default_dtype(t.float32);t.set_default_device('cpu')


def commitment():
    # Definitions only: no function body is executed by freeze/commit.
    source=lesson.REFERENCE_METHODS
    return lesson.commit_prediction(lesson.freeze_inputs(source),source,False,.5,
                                   'Fabricated record-format fixture, no empirical target forecast.','unknown')


def test_exact_candidate_and_protocol_contract():
    case=lesson.CASE
    assert case['edge_weights']==[[0,1,.92],[1,2,1.08],[2,3,.96],[3,6,1.17],[2,6,.27],
                                  [0,4,.41],[4,5,.33],[5,6,.24],[1,5,.16]]
    assert case['threshold']==.25 and case['ig_steps']==16 and case['random_seed']==0
    assert case['path_fraction']==.8 and case['control_margin']==.5 and case['max_depth']==4
    assert case['dtype']=='torch.float32' and case['threads']==1 and case['device']=='cpu'
    protocol=json.loads(lesson.PROTOCOL.read_text())
    assert protocol['accepted'] is False and protocol['empirical_blind_run_exists'] is False
    assert protocol['organism']['edge_weights']==case['edge_weights']
    assert 'declared_planted_path_indices' not in protocol['organism']
    assert not any('path_indices' in k or 'ground_truth' in k for k in case)
    assert set(case['roles'])=={'TRAIN','DEV','TEST','target'}


def test_blank_notebook_order_and_actual_code_wiring():
    cells=lesson.build_cells();assert cells==lesson.build_cells()
    assert len({c['id'] for c in cells})==len(cells)
    nbformat.validate(nbformat.v4.new_notebook(cells=[nbformat.from_dict(c) for c in cells]))
    ids=[c['id'] for c in cells]
    assert ids.index('graph-freeze')<ids.index('graph-commit')<ids.index('graph-reveal')<ids.index('graph-run')
    before=cells[:ids.index('graph-commit')]
    for cell in cells:
        source=cell['source']
        assert 'from_pretrained' not in source and 'pip install' not in source and '<img' not in source
        if cell['cell_type']=='code':
            tree=ast.parse(source);compile(tree,'<blank notebook>','exec')
            assert cell['outputs']==[] and cell['execution_count'] is None
            # Target execution/reveal is explicitly commented out in shipped cells.
            assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                           and n.func.attr in {'run_experiment','reveal_methods'} for n in ast.walk(tree))
    for cell in before:
        if cell['cell_type']=='code':
            assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                           and n.func.id in lesson.METHOD_NAMES for n in ast.walk(ast.parse(cell['source'])))
    for stage in ('graph','scores','selection','interventions','assembly'):
        source=cells[ids.index('graph-'+stage)]['source']
        assert '_SOURCE = ' in source and 'NotImplementedError' in source and 'practice_test' in source
    code='\n'.join(c['source'] for c in cells if c['cell_type']=='code')
    tree=ast.parse(code)
    assigned={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
              if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
              and n.targets[0].id in {'prediction','confidence','reasoning','observation','reflection',
                                     'transfer_prediction','transfer_confidence','transfer_reasoning'}}
    assert assigned=={'prediction':None,'confidence':None,'reasoning':'','observation':None,
                      'reflection':'','transfer_prediction':None,'transfer_confidence':None,'transfer_reasoning':''}
    assert 'ground_truth_path' not in code and 'declared_planted' not in code


def test_freeze_commit_reveal_do_not_execute_methods(monkeypatch):
    def forbidden(*a,**k):
        raise AssertionError('student namespace must not be executed')
    monkeypatch.setattr(lesson,'namespace',forbidden)
    monkeypatch.setattr(lesson,'_path_from_edges',forbidden)
    text=commitment();out=lesson.read_record(text)
    assert out['status']=='committed' and type(text) is str
    assert lesson.assert_current(text,lesson.REFERENCE_METHODS)['identity']==out['identity']
    assert lesson.reveal_methods(text)==lesson.REFERENCE_METHODS
    assert set(out['identity']['raw_inputs'])=={'weights','clean','corrupt'}
    assert out['identity']['raw_inputs']['weights']['shape']==[7,7]
    copy=lesson.read_record(text);copy['prediction']=True
    assert lesson.read_record(text)['prediction'] is False


@pytest.mark.parametrize('injected',[
    'make_graph({})\n', 'x = make_graph({})\n', '@make_graph({})\ndef a():\n pass\n',
    'def make_graph(x=activations({})):\n pass\n',
    'def make_graph(x: scalar_metric({})):\n pass\n'])
def test_freeze_rejects_import_time_execution(injected):
    with pytest.raises((ValueError,SyntaxError)):
        lesson.freeze_inputs(injected)


@pytest.mark.parametrize('field', ['prediction','confidence','reasoning','prior_exposure','identity'])
def test_commit_record_rejects_malformed_fields(field):
    record=lesson.read_record(commitment())
    record[field]={'prediction':1,'confidence':float('inf'),'reasoning':'',
                   'prior_exposure':'unseen_certified','identity':{}}[field]
    with pytest.raises(ValueError):
        lesson.read_record(lesson._seal(record))


@pytest.mark.parametrize('schema',['arena-blind-circuit-prediction','arena-guided-graph','historical'])
def test_foreign_and_guided_records_rejected(schema):
    record=lesson.read_record(commitment());record['schema']=schema
    with pytest.raises(ValueError):
        lesson.read_record(lesson._seal(record))


def test_unknown_nonfinite_duplicate_boolean_case_and_stale_source(monkeypatch):
    text=commitment();record=lesson.read_record(text)
    for mutate in [lambda r:r.update(extra=0),lambda r:r['case']['clean_inputs'].__setitem__(0,True),
                   lambda r:r['identity']['raw_inputs']['clean']['shape'].__setitem__(0,True)]:
        changed=deepcopy(record);mutate(changed)
        with pytest.raises(ValueError):
            lesson.read_record(lesson._seal(changed))
    with pytest.raises(ValueError):
        lesson.read_record(text.replace('"schema":','"schema":"duplicate","schema":',1))
    with pytest.raises(ValueError):
        lesson.assert_current(text,lesson.REFERENCE_METHODS+'\n')
    old=lesson._file
    monkeypatch.setattr(lesson,'_file',lambda path:'0'*64 if Path(path)==lesson.TESTS else old(path))
    with pytest.raises(ValueError):
        lesson.assert_current(text,lesson.REFERENCE_METHODS)


def test_failure_envelope_and_exclusive_private_exports(tmp_path):
    text=commitment();record=lesson.read_record(text)
    record.update(status='failed',commitment_sha256=lesson.sha_text(text),
                  failure={'stage':'fixture-schema-only','error_type':'FixtureFailure',
                           'message':'fabricated validation fixture, no target attempt','elapsed_seconds':0.})
    failed=lesson._seal(record)
    assert lesson.read_record(failed)['status']=='failed'
    with pytest.raises(ValueError):
        lesson.replay(failed)
    path=lesson.export_record(failed,tmp_path/'fixture-attempt.json')
    assert stat.S_IMODE(path.stat().st_mode)==0o600 and path.read_text()==failed
    with pytest.raises(FileExistsError):
        lesson.export_record(failed,path)
    assert path.read_text()==failed
    record['failure']['invented']='field'
    with pytest.raises(ValueError):
        lesson.read_record(lesson._seal(record))


@pytest.mark.parametrize('stage',['graph','scores','selection','interventions','assembly'])
def test_distinct_independent_practice_arithmetic(stage):
    ns=lesson.practice_namespace(lesson.REFERENCE_METHODS)
    make=ns['make_graph']
    def only_distinct(case):
        assert len(case['node_order'])==4 and case['node_order'][0]=='practice-input'
        assert case['edge_weights'] != lesson.CASE['edge_weights']
        return make(case)
    ns['make_graph']=only_distinct
    assert lesson.practice_test(ns,stage).endswith('passed')


def measurement_fixture():
    """Explicit anonymous four-node DEV values, not target-compatible observations."""
    case={'node_order':['a','b','c','d'],'edge_weights':[[0,1,2.],[1,2,.5],[0,3,.125],[2,3,.25]],
          'source_index':0,'target_index':2,'max_depth':2,'threshold':.5,'random_seed':0,
          'budget_seconds':120,'path_fraction':.8,'control_margin':.5}
    exact=[[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]
    ig=[[v*.5 for v in row] for row in exact]
    selected=[[0,1,.5],[1,2,.5]]
    pool=[(0,3,.125),(2,3,.25)]
    sampled=lesson.random.Random(0).sample(pool,k=2)
    obs={'kind':'fresh_cpu_graph_interventions','elapsed_seconds':0.,'scores':{'exact':exact,'IG16':ig},
         'selected_edges':selected,'path':[0,1,2],'path_drops':[1.,1.],
         'metrics':{'full':1.,'corrupt':0.,'path_only':1.,'path_removed':0.,'selected':1.,'random':0.,'reversed':0.},
         'ratios':{'faithfulness':1.,'completeness':1.,'minimality':1.},
         'control_edges':{'random':[list(v) for v in sampled],'reversed':[[1,0,.5],[2,1,.5]]},
         'margins':{'random':1.,'reversed':1.},'gates':{'path':True,'random':True,'reversed':True},'predicate':True}
    return case,obs


def test_measurement_validation_uses_distinct_fixture_and_retains_rejections():
    case,obs=measurement_fixture()
    lesson._validate_measurements(obs,case,check_path_choice=True)
    obs['metrics']['random']=.6;obs['margins']['random']=.4
    obs['gates']['random']=False;obs['predicate']=False
    lesson._validate_measurements(obs,case)
    with pytest.raises(ValueError):
        lesson._validate_measurements(obs,lesson.CASE)  # Wrong organism rejected before any target search.


@pytest.mark.parametrize('mutation',[
    lambda o:o.update(extra=0),lambda o:o['metrics'].update(corrupt=1.),
    lambda o:o['scores']['IG16'][0].__setitem__(1,float('inf')),
    lambda o:o['scores']['exact'][0].__setitem__(1,True),
    lambda o:o['ratios'].update(faithfulness=True),lambda o:o['control_edges'].pop('random'),
    lambda o:o['control_edges']['random'].pop(),lambda o:o['path'].__setitem__(0,False),
    lambda o:o.update(path=[]),lambda o:o['gates'].update(random=1),
    lambda o:o['selected_edges'][0].__setitem__(2,.51),
    lambda o:o.update(kind='historical_guided'),
    lambda o:o['control_edges']['random'].__setitem__(0,[0,1,.5]),
    lambda o:o['control_edges']['reversed'][0].__setitem__(2,-.5),
    lambda o:o['path_drops'].__setitem__(0,.9)])
def test_distinct_measurement_mutation_rejection(mutation):
    case,obs=measurement_fixture();mutation(obs)
    with pytest.raises(ValueError):
        lesson._validate_measurements(obs,case)


def test_distinct_fixture_no_path_and_insufficient_pool_fail():
    case,obs=measurement_fixture()
    with pytest.raises(ValueError):
        lesson._path_from_edges([],0,2,2)
    case['edge_weights']=case['edge_weights'][:2]
    with pytest.raises(ValueError):
        lesson._validate_measurements(obs,case)


def test_replay_no_student_execution_or_path_search(monkeypatch):
    _,obs=measurement_fixture()
    monkeypatch.setattr(lesson,'read_record',lambda text:{'status':'observed','observation':deepcopy(obs)})
    def forbidden(*a,**k):
        raise AssertionError('no replay execution')
    monkeypatch.setattr(lesson,'namespace',forbidden)
    monkeypatch.setattr(lesson,'_path_from_edges',forbidden)
    monkeypatch.setattr(lesson,'environment_identity',forbidden)
    assert lesson.replay('mocked distinct fixture')==obs


@pytest.mark.parametrize('saved',['current-executed','edited','filled','stale-executed','malformed','unknown-metadata'])
def test_generator_preserves_learner_work_byte_exact(tmp_path,saved):
    module=generator();expected=module.make_notebook();path=tmp_path/'exercises.ipynb'
    if saved=='malformed':
        path.write_text('{not a notebook')
    else:
        old=deepcopy(expected)
        code=next(c for c in old.cells if c.cell_type=='code')
        if saved in ('current-executed','stale-executed'):
            code.outputs=[nbformat.v4.new_output('stream',name='stdout',text='LEARNER OUTPUT')]
            code.execution_count=7
        if saved in ('edited','stale-executed'):
            code.source+='\n# learner source edit\n'
        if saved=='filled':
            next(c for c in old.cells if c.id=='graph-commit').source=next(c for c in old.cells if c.id=='graph-commit').source.replace('prediction = None','prediction = True')
        if saved=='unknown-metadata':
            old.metadata['learner_record']={'keep':'exact'}
        nbformat.write(old,path)
    before=path.read_bytes()
    module.synchronize(path,expected,check=True)
    assert path.read_bytes()==before
    module.synchronize(path,expected,check=False)
    assert path.read_bytes()==before


def test_generator_check_missing_and_stale_pristine(tmp_path):
    module=generator();expected=module.make_notebook();path=tmp_path/'exercises.ipynb'
    assert module.synchronize(path,expected,check=True) is False and not path.exists()
    assert module.synchronize(path,expected,check=False) is True
    old=deepcopy(expected);old.cells[0].source+='\nOld generation\n'
    old.metadata['graph_prediction'][module.BASELINE_KEY]=module.source_fingerprint(old)
    nbformat.write(old,path);before=path.read_bytes()
    assert module.synchronize(path,expected,check=True) is False and path.read_bytes()==before
    assert module.synchronize(path,expected,check=False) is True
    assert module.source_fingerprint(nbformat.read(path,as_version=4))==module.source_fingerprint(expected)


def test_run_gate_is_explicit_and_freeze_calls_no_namespace():
    tree=ast.parse(Path(lesson.__file__).read_text())
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'namespace',*lesson.METHOD_NAMES}
                   for n in ast.walk(funcs['freeze_inputs']))
    run=ast.get_source_segment(Path(lesson.__file__).read_text(),funcs['run_experiment'])
    assert 'authorized is not True' in run and '_ATTEMPTED' in run
    assert "status='failed'" in run and 'except Exception as error' in run
    assert 'check_path_choice=True' in run


@pytest.mark.parametrize('kind',['duplicate-json','metadata-collision','invalid-utf8','non-object'])
def test_generator_malformed_inputs_preserved(tmp_path,kind):
    module=generator();expected=module.make_notebook();path=tmp_path/'work.ipynb'
    if kind=='duplicate-json':
        raw=nbformat.writes(expected).replace('"nbformat": 4','"nbformat": 3,"nbformat": 4')
        path.write_text(raw)
    elif kind=='metadata-collision':
        old=deepcopy(expected);old.metadata['graph_prediction']='learner malformed metadata'
        nbformat.write(old,path)
    elif kind=='invalid-utf8':
        path.write_bytes(bytes([255,254,0]))
    else:
        path.write_text('[]')
    before=path.read_bytes()
    assert module.synchronize(path,expected,check=True) is False
    assert module.synchronize(path,expected,check=False) is False
    assert path.read_bytes()==before


@pytest.mark.parametrize('source_field',['helper_sha256','protocol_sha256','tests_sha256'])
def test_current_rejects_changed_bound_source(source_field,monkeypatch):
    text=commitment();original=lesson.freeze_inputs
    def changed(source):
        identity=original(source);identity[source_field]='0'*64
        return identity
    monkeypatch.setattr(lesson,'freeze_inputs',changed)
    with pytest.raises(ValueError):
        lesson.assert_current(text,lesson.REFERENCE_METHODS)


def test_noncanonical_text_and_runtime_change_rejected(monkeypatch):
    text=commitment()
    with pytest.raises(ValueError):
        lesson.read_record(' '+text)
    original=lesson.environment_identity
    def changed():
        env=original();env['torch']+='-changed';return env
    monkeypatch.setattr(lesson,'environment_identity',changed)
    with pytest.raises(ValueError):
        lesson.assert_current(text,lesson.REFERENCE_METHODS)


# Review regression: these four wrong methods passed all v1 practice stages.
# The oracle must reject them using distinct exposed DEV examples only.
@pytest.mark.parametrize(('mutant','stage','before','after'),[
 ('half-exact','scores','    if type(steps) is not int or steps <= 0:',
  '    return exact_edge_effects(graph)/2\n    if type(steps) is not int or steps <= 0:'),
 ('one-midpoint','scores','    if type(steps) is not int or steps <= 0:',
  '    steps=1\n    if type(steps) is not int or steps <= 0:'),
 ('omit-corrupt','assembly',"denominator = metrics['full']-metrics['corrupt']",
  "denominator = metrics['full']"),
 ('max-drop','assembly',"'minimality':min(path_result['drops'])/denominator",
  "'minimality':max(path_result['drops'])/denominator"),
 ('absolute-denominator','assembly',"denominator = metrics['full']-metrics['corrupt']",
  "denominator = abs(metrics['full']-metrics['corrupt'])")])
def test_practice_oracle_rejects_review_mutants(mutant,stage,before,after):
    source=lesson.REFERENCE_METHODS.replace(before,after)
    assert source!=lesson.REFERENCE_METHODS
    ns=lesson.practice_namespace(source)
    make=ns['make_graph']
    def distinct_only(case):
        assert len(case['node_order'])==4
        assert case['node_order'][0]=='practice-input'
        return make(case)
    ns['make_graph']=distinct_only
    with pytest.raises(AssertionError):
        lesson.practice_test(ns,stage)


@pytest.mark.parametrize('field',['kind','elapsed_seconds','extra'])
def test_fake_assembly_rejects_unknown_and_recorder_collisions(field):
    _,obs=measurement_fixture()
    assembled={k:deepcopy(v) for k,v in obs.items() if k not in ('kind','elapsed_seconds')}
    captured={k:deepcopy(assembled[k]) for k in ('scores','selected_edges','path','path_drops','metrics','control_edges')}
    bound=lesson._bind_assembly(assembled,captured,123.)
    assert bound['elapsed_seconds']==123. and bound['kind']=='fresh_cpu_graph_interventions'
    assembled[field]=0. if field=='elapsed_seconds' else 'forged-DEV-fixture'
    with pytest.raises(ValueError):
        lesson._bind_assembly(assembled,captured,123.)


def test_fake_assembly_rejects_missing_fields_and_does_not_alias():
    _,obs=measurement_fixture()
    assembled={k:deepcopy(v) for k,v in obs.items() if k not in ('kind','elapsed_seconds')}
    captured={k:deepcopy(assembled[k]) for k in ('scores','selected_edges','path','path_drops','metrics','control_edges')}
    bound=lesson._bind_assembly(assembled,captured,123.)
    assembled['ratios']['minimality']=999.
    assert bound['ratios']['minimality']==1.
    assembled.pop('margins')
    with pytest.raises(ValueError):
        lesson._bind_assembly(assembled,captured,123.)
