"""Source/DEV engineering only: no pretrained model, RNG, cache or course experiment."""
from pathlib import Path
from types import SimpleNamespace
import ast
import os
import hashlib
import importlib.util
import json
import re
import shutil
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
FOLDER = REPO / 'mastery_hub/units/acdc_own_method'
SOURCE_ROOT = next(p for p in Path(__file__).resolve().parents if (p/'chapter8_automated_circuits').is_dir())

@pytest.fixture
def generator():
    spec = importlib.util.spec_from_file_location('acdc_addon_generator', Path(os.environ.get('ACDC_DEV_GENERATOR', FOLDER/'generate.py')))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def source_namespace(generator):
    namespace = {}
    for source in (generator.GRAPH, generator.METHOD, generator.CONTROLS, generator.TRANSFER, generator.FIXTURE):
        exec(compile(source, '<visible inline ACDC source>', 'exec'), namespace)
    return namespace

@pytest.fixture
def dev(source_namespace):
    # Explicit test-only import. No import/construction in generator or its default import.
    import torch as t
    ns = source_namespace
    sender, callback = ns['make_own_methods'](t)
    oracle, rows, summarize = ns['make_controls'](t)
    return t, ns, sender, callback, oracle, rows, summarize

def test_notebook_exact_and_blank(generator):
    protocol = json.loads((FOLDER/'protocol.json').read_text())
    raw = (FOLDER/'exercises.ipynb').read_bytes()
    assert raw == generator.generated_bytes(protocol)
    book = json.loads(raw)
    ids = [c['id'] for c in book['cells']]
    assert len(ids) == len(set(ids))
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            ast.parse(''.join(cell['source']))
    assert ids.index('acdc-own-forecast') < ids.index('acdc-own-graph-stub')
    assert ids[ids.index('acdc-own-kernel-test')+1] == 'acdc-own-controls-reflection'
    text = ''.join(''.join(c['source']) for c in book['cells'])
    for claim in ('model runtime pending', 'not an unseen test', 'Try It Yourself', 'full-vocabulary', 'UNSET'):
        assert claim.lower() in text.lower()
    for source in (generator.GRAPH, generator.METHOD, generator.CONTROLS):
        assert any(source in ''.join(c['source']) for c in book['cells'] if c['cell_type'] == 'markdown')
    assert any(generator.TRANSFER == ''.join(c['source']) for c in book['cells'])

def test_protocol_decisions_are_declared(generator):
    protocol = json.loads((FOLDER/'protocol.json').read_text())
    assert protocol['parity_tolerances'] is None
    assert protocol['actual_model_budget'] == 0
    assert protocol['existing_protocol'] == dict(discovery_n=64, discovery_seed=0, heldout_n=64, heldout_seed=1, tau=.01, random_n=20, random_seed=4)
    assert [d['id'] for d in protocol['decisions']] == [f'ACDC-D{i:02}' for i in range(1,9)]
    assert protocol['decisions'][3]['status'] == 'UNSET_future_adoption_required'
    assert protocol['decisions'][5]['status'] == 'DEV_only_actual_model_budget_zero'

def staged_protocol(folder, sources=None):
    """Copy the protocol with the lesson-file pins replaced, so generator behaviour is tested alone."""
    protocol = json.loads((FOLDER/'protocol.json').read_text())
    protocol['sources'] = sources or {}
    (folder/'protocol.json').write_text(json.dumps(protocol))

def test_graph_indices_without_tensors(source_namespace):
    ns=source_namespace
    g=ns['own_graph'](12,12)
    assert (len(g.SENDER_NAMES),len(g.RECEIVER_NAMES),sum(map(sum,g.valid))) == (157,445,32491)
    assert g.head_sender(1,0)==14 and g.mlp_sender(0)==13
    assert g.head_receiver(0,'v',11)==35 and g.mlp_receiver(0)==36
    order=ns['own_gpt2_receiver_order'](g)
    assert order[:5] == [444,443,g.head_receiver(11,'q',11),g.head_receiver(11,'k',11),g.head_receiver(11,'v',11)]
    assert sorted(order)==list(range(445))
    owner=ns['own_gpt2_receiver_owner'](g)
    assert owner[444] is None and owner[g.head_receiver(0,'k',2)]==g.head_sender(0,2)
    for args in [(0,1),(1,0),(True,1),(1.,1)]:
        with pytest.raises(ValueError): ns['own_graph'](*args)

def test_hand_sender_arithmetic_and_kernel(dev):
    t,ns,sender,callback,oracle,_,_=dev
    ns['check_sender_dev'](t,sender)
    ns['check_kernel_dev'](t,sender,callback,oracle)
    assert list(__import__('inspect').signature(callback).parameters) == ['model','clean_tokens','corrupt_out','in_circuit','edge_mask','graph']

@pytest.mark.parametrize('case',['type','shape','cache','graph'])
def test_malformed_inputs_fail_before_forward(dev,case):
    t,ns,sender,callback,_,_,_=dev
    g=ns['own_graph'](1,1); model,clean,corrupt=ns['make_hand_fixture'](t,t.float64)
    edges=t.tensor(g.valid,dtype=t.bool); cache=sender(model,corrupt,g); before=model.calls
    mask=edges.clone()
    if case=='type': mask=mask.float()
    if case=='shape': mask=mask[:1]
    if case=='cache': cache=cache[:,:,:1]
    if case=='graph': g=ns['own_graph'](2,1)
    with pytest.raises((ValueError,TypeError)): callback(model,clean,cache,mask,edges,g)
    assert model.calls==before

def test_oracle_does_not_depend_on_sender_under_test(dev):
    t,ns,sender,callback,oracle,_,_=dev
    g=ns['own_graph'](1,1); model,clean,corrupt=ns['make_hand_fixture'](t,t.float64)
    edges=t.tensor(g.valid,dtype=t.bool); wrong_cache=sender(model,corrupt,g)
    wrong_cache[:,:,1] += 1  # Deliberately wrong projected sender; oracle must not self-validate.
    mask=edges.clone(); mask[1,g.LOGITS_RECEIVER]=False
    wrong=callback(model,clean,wrong_cache,mask,edges,g)
    independent=oracle(model,clean,corrupt,1,g.LOGITS_RECEIVER,g)
    assert not t.equal(wrong,independent)

def test_hand_driver_restores_owned_hooks_on_exception(dev):
    t,ns,*_=dev
    model,clean,_=ns['make_hand_fixture'](t,t.float64)
    def boom(x,hook): raise RuntimeError('deliberate DEV exception')
    before=model.cfg.__dict__.copy()
    with pytest.raises(RuntimeError,match='deliberate'): model.run_with_hooks(clean,[('blocks.0.hook_q_input',boom)])
    assert model._active_hooks==() and model.cfg.__dict__==before
    # Failing inside the actual callback must unwind its owned driver context too.
    _,callback=ns['make_own_methods'](t)
    g=ns['own_graph'](1,1); edges=t.tensor(g.valid,dtype=t.bool)
    corrupt=ns['make_own_methods'](t)[0](model,clean,g)
    original=model.run_with_hooks
    def inject_failure(tokens,fwd_hooks):
        return original(tokens,fwd_hooks + [('blocks.0.hook_q_input',boom)])
    model.run_with_hooks=inject_failure
    with pytest.raises(RuntimeError,match='deliberate'): callback(model,clean,corrupt,edges,edges,g)
    assert model._active_hooks==() and model.cfg.__dict__==before

def test_full_vocabulary_and_complement_metric(dev):
    t,_,_,_,_,rows,_=dev
    logits=t.tensor([[0.,0.,0.],[0.,0.,0.]],dtype=t.float64)
    values=rows(logits,t.tensor([0,1]),t.tensor([0,1]))
    assert t.equal(values,t.tensor([0.,-2/3],dtype=t.float64))
    # A year-only softmax gives [0,-1], a different estimand.
    assert values[1].item()!=-1

def row_records(t, full=(2.,10.), empty=(0.,2.), selected=(1.,2.)):
    def row(v): return {'row_ids':('DEV-fixed-a','DEV-fixed-b'),'values':t.tensor(v,dtype=t.float64)}
    result={'full':row(full),'empty':row(empty),'selected':row(selected)}
    result.update({f'random_{i}':row((0.,2.)) for i in range(20)})
    return result

def test_ratio_of_means_common_denominator(dev):
    t,_,_,_,_,_,summarize=dev
    records=row_records(t); result=summarize(records)
    assert result['denominator']==5
    assert result['recovery']['selected']==.1  # mean row ratios would be .25
    assert result['raw_means']['full']==6 and result['raw_means']['empty']==1
    assert all(result['recovery'][f'random_{i}']==0 for i in range(20))

@pytest.mark.parametrize('case',['zero','negative','nan','row_ids','missing','shape'])
def test_row_failure_keeps_original_records(dev,case):
    t,_,_,_,_,_,summarize=dev
    records=row_records(t)
    if case=='zero': records=row_records(t,full=(0.,2.))
    if case=='negative': records=row_records(t,full=(-1.,1.))
    if case=='nan': records['full']['values'][0]=float('nan')
    if case=='row_ids': records['random_0']['row_ids']=('DEV-fixed-b','DEV-fixed-a')
    if case=='missing': del records['random_19']
    if case=='shape': records['selected']['values']=t.tensor([[1.,2.]])
    snapshot={k:(v['row_ids'],v['values'].clone()) for k,v in records.items()}
    with pytest.raises(ValueError): summarize(records)
    for key,(ids,values) in snapshot.items():
        assert records[key]['row_ids']==ids
        assert t.allclose(records[key]['values'],values,equal_nan=True)

def test_future_tolerance_none_fails_before_model(source_namespace):
    class NoModel:
        def __getattribute__(self,name): raise AssertionError('model touched before tolerance gate')
    prepared=SimpleNamespace(parity_tolerances=None,model=NoModel())
    with pytest.raises(RuntimeError,match='UNSET'): source_namespace['require_prepared_runtime'](prepared)
    with pytest.raises(RuntimeError,match='UNSET'): source_namespace['bind_own_transfer'](
        circuit_logits=None,sender_outputs=None,prepared=prepared,kl_divergence=None,
        acdc=None,receiver_order=None,receiver_owner=None,random_circuit=None)

def marker_prepared(ns,sender):
    # Declaration/control-flow markers only; not a loaded model or numerical task.
    cfg=SimpleNamespace(n_layers=12,n_heads=12,use_split_qkv_input=True,use_hook_mlp_in=True)
    def cohort(prefix,seed):
        return SimpleNamespace(row_ids=tuple(f'{prefix}-{i}' for i in range(64)),seed=seed,
            clean_tokens=tuple(range(64)),corrupt_out=object(),answers=object(),full_clean_logprobs=object())
    return SimpleNamespace(parity_tolerances={'atol':0.,'rtol':0.},runtime_authorized=True,
        known_clean_hook_context=True,model=SimpleNamespace(cfg=cfg),graph=ns['own_graph'](12,12),
        edges=object(),discovery=cohort('DEV-D',0),heldout=cohort('DEV-H',1),
        random_seed=4,random_generator=object(),sender_callback=sender)


@pytest.mark.parametrize('mutation',['forecast','metadata','outputs','execution_count','format'])
def test_no_clobber_edited_learner(tmp_path,generator,mutation):
    folder=tmp_path/'practice'; folder.mkdir()
    staged_protocol(folder)
    assert generator.generate(folder,SOURCE_ROOT)=='created'
    target=folder/'exercises.ipynb'; book=json.loads(target.read_bytes())
    if mutation=='forecast': book['cells'][1]['source'].append('My forecast: ...\n')
    if mutation=='metadata': book['metadata']['learner']='answer'
    code=next(c for c in book['cells'] if c['cell_type']=='code')
    if mutation=='outputs': code['outputs']=[{'output_type':'stream','name':'stdout','text':['answer']}]
    if mutation=='execution_count': code['execution_count']=1
    raw=(json.dumps(book,indent=3 if mutation=='format' else 1)+'\n').encode()
    target.write_bytes(raw); stat=target.stat().st_mtime_ns
    with pytest.raises(ValueError,match='edited'): generator.generate(folder,SOURCE_ROOT)
    with pytest.raises(ValueError,match='edited'): generator.generate(folder,SOURCE_ROOT,True)
    assert target.read_bytes()==raw and target.stat().st_mtime_ns==stat

def test_exact_noop_check_missing_and_bad_pin(tmp_path,generator):
    folder=tmp_path/'practice'; folder.mkdir()
    pinned = tmp_path/'root'; pinned.mkdir(); (pinned/'x.txt').write_bytes(b'x')
    staged_protocol(folder, {'x.txt': {'sha256': hashlib.sha256(b'x').hexdigest(), 'size_bytes': 1}})
    with pytest.raises(ValueError,match='missing'): generator.generate(folder,pinned,True)
    assert not (folder/'exercises.ipynb').exists()
    assert generator.generate(folder,pinned)=='created'
    path=folder/'exercises.ipynb'; stat=path.stat().st_mtime_ns
    assert generator.generate(folder,pinned)=='unchanged'
    assert generator.generate(folder,pinned,True)=='unchanged'
    assert path.stat().st_mtime_ns==stat
    protocol=json.loads((folder/'protocol.json').read_text())
    protocol['sources']['x.txt']['sha256']='0'*64
    (folder/'protocol.json').write_text(json.dumps(protocol))
    path.unlink()
    with pytest.raises(ValueError,match='pinned original'): generator.generate(folder,pinned)
    assert not path.exists()

def test_symlink_and_late_collision_refused(tmp_path,generator,monkeypatch):
    folder=tmp_path/'practice'; folder.mkdir(); staged_protocol(folder)
    other=tmp_path/'learner'; other.write_text('keep')
    target=folder/'exercises.ipynb'; target.symlink_to(other)
    with pytest.raises(ValueError,match='symlink'): generator.generate(folder,SOURCE_ROOT)
    assert other.read_text()=='keep'
    target.unlink(); original=Path.open
    def race(path,*args,**kwargs):
        if path==target and args and args[0]=='xb':
            with original(path,'wb') as stream: stream.write(b'late learner')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',race)
    with pytest.raises(FileExistsError): generator.generate(folder,SOURCE_ROOT)
    assert target.read_bytes()==b'late learner'

def test_local_links_resolve_without_http():
    from urllib.parse import unquote,urlsplit
    book=json.loads((FOLDER/'exercises.ipynb').read_bytes())
    texts=[(FOLDER/'README.md').read_text()]+[''.join(c['source']) for c in book['cells'] if c['cell_type']=='markdown']
    for text in texts:
        for link in re.findall(r'\]\(([^)]+)\)',text):
            parts=urlsplit(link)
            if parts.scheme or not parts.path: continue
            path=(FOLDER/unquote(parts.path)).resolve()
            # Staged candidate and canonical product share the same repo-relative layout.
            rel=path.relative_to(REPO) if path.is_relative_to(REPO) else None
            assert path.is_file() or (rel is not None and (SOURCE_ROOT/rel).is_file()),link


def test_backend_operations_early_bound(dev,monkeypatch):
    t,ns,sender,callback,oracle,rows,summarize=dev
    g=ns['own_graph'](1,1); model,clean,corrupt=ns['make_hand_fixture'](t,t.float64)
    cache=sender(model,corrupt,g); edges=t.tensor(g.valid,dtype=t.bool)
    def bad(*a,**k): raise AssertionError('late backend lookup')
    monkeypatch.setattr(t,'einsum',bad); monkeypatch.setattr(t,'stack',bad)
    monkeypatch.setattr(t,'zeros_like',bad)
    # Both callbacks retain their originally bound operation objects.
    assert tuple(sender(model,clean,g).shape)==(2,2,3,2)
    assert tuple(callback(model,clean,cache,edges,edges,g).shape)==(2,3)


def test_mask_conversion_never_casts_prepared_data(dev):
    t,ns,sender,callback,*_=dev
    g=ns['own_graph'](1,1); model,clean,corrupt=ns['make_hand_fixture'](t,t.float64)
    edges=t.tensor(g.valid,dtype=t.bool); cache=sender(model,corrupt,g)
    before=model.calls; model.W_O=model.W_O.float()
    with pytest.raises(ValueError,match='dtype'): callback(model,clean,cache,edges,edges,g)
    assert model.calls==before and cache.dtype==t.float64 and model.W_O.dtype==t.float32


def test_visible_metric_subtest(dev):
    t,ns,_,_,_,rows,summarize=dev
    ns['check_rows_dev'](t,rows,summarize)


def test_no_scientific_import_or_rng_in_visible_source(generator):
    forbidden={'rand','randn','randperm','manual_seed','Generator','from_pretrained','load_model','cuda'}
    for source in (generator.GRAPH,generator.METHOD,generator.CONTROLS,generator.TRANSFER,generator.FIXTURE):
        tree=ast.parse(source)
        imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        assert all((n.module if isinstance(n,ast.ImportFrom) else n.names[0].name).split('.')[0] in {'types','math'} for n in imports)
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree))
    oracle=next(n for n in ast.walk(ast.parse(generator.CONTROLS)) if isinstance(n,ast.FunctionDef) and n.name=='raw_one_edge_oracle')
    assert not any(isinstance(n,ast.Name) and n.id=='own_sender_outputs' for n in ast.walk(oracle))

# V2 regressions: only literal CPU DEV values. Runtime declarations here are
# synthetic wiring sentinels; these zero tolerances are NOT adopted GPT-2 tolerances.
def dev_prepared_v2(t, ns, dtype):
    graph = ns['own_graph'](12,12)
    class NumericalMarker:
        def __init__(self):
            self.cfg=SimpleNamespace(n_layers=12,n_heads=12,d_vocab=3,device='cpu',
                use_split_qkv_input=True,use_hook_mlp_in=True)
            self.W_O=t.tensor([[[[1.,2.],[0.,1.]]]],dtype=dtype)
        def named_parameters(self): return [('DEV-W_O',self.W_O)]
        def named_buffers(self): return []
    model=NumericalMarker()
    def sender(*args): raise AssertionError('DEV binding never computes a cache')
    def cohort(prefix,seed):
        clean=t.tensor([[11,13],[17,19]]*32,dtype=t.int64)
        corrupt=t.tensor([[23,29],[31,37]]*32,dtype=t.int64)
        cache=t.tensor([[[[.25,.5]]*157]*2]*64,dtype=dtype)
        return SimpleNamespace(row_ids=tuple(f'{prefix}-{i}' for i in range(64)),seed=seed,
            clean_tokens=clean,corrupt_tokens=corrupt,corrupt_out=cache,
            cache_binding=(model,corrupt,sender,'2'*64),
            answers=t.tensor([1,2]*32,dtype=t.int64),
            full_clean_logprobs=t.tensor([[[-1.,-2.,-3.]]]*64,dtype=dtype),
            metric=lambda logits,answers:t.tensor([.5,.25]*32,dtype=dtype))
    prepared=SimpleNamespace(parity_tolerances={'atol':0.,'rtol':0.},runtime_authorized=True,
        known_clean_hook_context=True,model=model,graph=graph,edges=t.tensor(graph.valid,dtype=t.bool),
        discovery=cohort('DEV-discovery',0),heldout=cohort('DEV-heldout',1),
        random_seed=4,random_generator=object(),sender_callback=sender,backend=t,
        hook_state=lambda:(),identity=dict(model_revision='literal DEV marker, no pretrained model',
            graph_source_sha256='1'*64,sender_source_sha256='2'*64,circuit_source_sha256='3'*64))
    return prepared,sender


def dev_bind_v2(t,ns,dtype,callback=None,kl=None,random=None):
    prepared,sender=dev_prepared_v2(t,ns,dtype);calls=[]
    def own(*args):
        calls.append(args)
        return t.tensor([[.25,-.5,.75]]*64,dtype=dtype)
    wired=ns['bind_own_transfer'](circuit_logits=callback or own,sender_outputs=sender,prepared=prepared,
        kl_divergence=kl or (lambda *args:t.tensor(.125,dtype=dtype)),
        acdc=lambda *args,**kwargs:('DEV-selected',0),
        receiver_order=ns['own_gpt2_receiver_order'],receiver_owner=ns['own_gpt2_receiver_owner'],
        random_circuit=random or (lambda edges,k,generator:edges.clone()))
    return prepared,wired,calls


@pytest.mark.parametrize('dtype_name',['float32','float64'])
def test_v2_same_callback_and_exact_single_capture(source_namespace,dtype_name):
    import torch as t
    p,wired,calls=dev_bind_v2(t,source_namespace,getattr(t,dtype_name))
    assert calls==[] and wired.discovery_kl(p.edges)==.125
    row=wired.heldout_metric(p.edges)
    assert len(calls)==2 and calls[0][1] is p.discovery.clean_tokens
    assert calls[1][1] is p.heldout.clean_tokens
    assert row['values'].tolist()==[.5,.25]*32
    assert wired.attempts[-1]['status']=='complete' and wired.attempts[-1]['completed']==1


@pytest.mark.parametrize('mutation',['clean_replace','clean_inplace','cache_replace','cache_inplace',
    'corrupt_replace','answer_replace','anchor_replace','rows','identity','flag','weights','graph_fn','hook'])
def test_v2_bound_drift_rejected_before_callback(source_namespace,mutation):
    import torch as t
    p,wired,calls=dev_bind_v2(t,source_namespace,t.float64)
    if mutation=='clean_replace': p.discovery.clean_tokens=p.discovery.clean_tokens.clone()
    if mutation=='clean_inplace': p.discovery.clean_tokens[0,0]+=1
    if mutation=='cache_replace': p.discovery.corrupt_out=p.discovery.corrupt_out.clone()
    if mutation=='cache_inplace': p.discovery.corrupt_out[0,0,0,0]+=1
    if mutation=='corrupt_replace': p.discovery.corrupt_tokens=p.discovery.corrupt_tokens.clone()
    if mutation=='answer_replace': p.heldout.answers=p.heldout.answers.clone()
    if mutation=='anchor_replace': p.discovery.full_clean_logprobs=p.discovery.full_clean_logprobs.clone()
    if mutation=='rows': p.discovery.row_ids=tuple(reversed(p.discovery.row_ids))
    if mutation=='identity': p.identity['sender_source_sha256']='f'*64
    if mutation=='flag': p.model.cfg.use_split_qkv_input=False
    if mutation=='weights': p.model.W_O[0,0,0,0]+=1
    if mutation=='graph_fn': p.graph.head_sender=lambda *args:1
    if mutation=='hook': p.hook_state=lambda:('unexpected',)
    with pytest.raises(ValueError): wired.discovery_kl(p.edges)
    assert calls==[]


@pytest.mark.parametrize('bad',['nan','shape','dtype','infinite'])
def test_v2_invalid_logits_preserve_failing_return(source_namespace,bad):
    import torch as t
    value=t.tensor([[.25,-.5,.75]]*64,dtype=t.float64)
    if bad=='nan': value[0,0]=float('nan')
    if bad=='infinite': value[0,0]=float('inf')
    if bad=='shape': value=value[:1]
    if bad=='dtype': value=value.float()
    p,wired,_=dev_bind_v2(t,source_namespace,t.float64,callback=lambda *args:value)
    with pytest.raises(source_namespace['TransferFailure']) as caught: wired.evaluate(p.edges)
    record=caught.value.attempt
    assert record['completed']==0 and record['attempted']==1
    capture=next(item for item in record['captures'] if item['label']=='callback:logits')
    assert capture['returned'] is value and capture['snapshot'] is not None
    assert record['primary_failure']['type']=='ValueError'


def test_v2_nan_kl_stops_discovery(source_namespace):
    import torch as t
    p,wired,calls=dev_bind_v2(t,source_namespace,t.float64,kl=lambda *args:t.tensor(float('nan'),dtype=t.float64))
    with pytest.raises(ValueError,match='nonfinite'): wired.discovery_kl(p.edges)
    assert len(calls)==1


@pytest.mark.parametrize('fail_call',[1,3,4,10,23])
def test_v2_partial_rows_survive_late_failure_and_sink_failure(source_namespace,fail_call):
    import torch as t
    count=[0];returned=[]
    def callback(*args):
        count[0]+=1
        if count[0]==fail_call: raise RuntimeError('literal DEV late callback failure')
        value=t.tensor([[.25,-.5,.75]]*64,dtype=t.float32);returned.append(value);return value
    p,wired,_=dev_bind_v2(t,source_namespace,t.float32,callback=callback)
    def bad_sink(attempt): raise OSError('literal DEV storage failure')
    p.attempt_sink=bad_sink
    with pytest.raises(source_namespace['TransferFailure']) as caught: wired.evaluate(p.edges)
    attempt=caught.value.attempt
    assert attempt is wired.attempts[-1] and attempt['completed']==fail_call-1 and attempt['attempted']==fail_call
    assert len(attempt['rows'])==fail_call-1 and count[0]==fail_call
    assert attempt['primary_failure']['message']=='literal DEV late callback failure'
    assert 'storage failure' in attempt['secondary_failures'][-1]
    for row in attempt['rows'].values():
        assert row['values'].tolist()==[.5,.25]*32
        assert row['row_ids']==p.heldout.row_ids and row['mask'].dtype==t.bool
    assert attempt['status']=='failed'


def test_v2_bad_metric_and_capture_failure_keeps_earlier_rows(source_namespace):
    import torch as t
    p,sender=dev_prepared_v2(t,source_namespace,t.float64);count=[0]
    class NoClone:
        def detach(self): raise OSError('literal DEV capture failure')
    bad=NoClone()
    def metric(*args):
        count[0]+=1
        return t.tensor([.5,.25]*32,dtype=t.float64) if count[0]<4 else bad
    p.heldout.metric=metric
    wired=source_namespace['bind_own_transfer'](circuit_logits=lambda *args:t.tensor([[.25,-.5,.75]]*64,dtype=t.float64),
        sender_outputs=sender,prepared=p,kl_divergence=None,acdc=None,
        receiver_order=None,receiver_owner=None,random_circuit=lambda edges,*args:edges.clone())
    with pytest.raises(source_namespace['TransferFailure']) as caught: wired.evaluate(p.edges)
    attempt=caught.value.attempt
    assert attempt['completed']==3 and attempt['condition']=='random_0'
    assert attempt['captures'][-1]['returned'] is bad
    assert attempt['captures'][-1]['snapshot'] is None and 'capture failure' in attempt['secondary_failures'][0]


@pytest.mark.parametrize('bad',['nan','shape','dtype'])
def test_v2_bad_metric_return_is_retained(source_namespace,bad):
    import torch as t
    p,sender=dev_prepared_v2(t,source_namespace,t.float64)
    value=t.tensor([.5,.25]*32,dtype=t.float64)
    if bad=='nan': value[0]=float('nan')
    if bad=='shape': value=value[:,None]
    if bad=='dtype': value=value.float()
    p.heldout.metric=lambda *args:value
    wired=source_namespace['bind_own_transfer'](circuit_logits=lambda *args:t.tensor([[.25,-.5,.75]]*64,dtype=t.float64),
        sender_outputs=sender,prepared=p,kl_divergence=None,acdc=None,
        receiver_order=None,receiver_owner=None,random_circuit=None)
    with pytest.raises(source_namespace['TransferFailure']) as caught: wired.heldout_metric(p.edges)
    assert caught.value.attempt['captures'][-1]['returned'] is value
    assert caught.value.attempt['completed']==0


def test_v2_subnormal_denominator_fails_without_epsilon(dev):
    t,ns,_,_,_,_,summarize=dev
    def row(value): return dict(row_ids=('DEV-subnormal',),values=t.tensor([value],dtype=t.float64))
    records={'full':row(5e-324),'empty':row(0.),'selected':row(1.)}
    records.update({f'random_{i}':row(0.) for i in range(20)})
    before=records['full']['values'].clone()
    with pytest.raises(ValueError,match='nonfinite'): summarize(records)
    assert t.equal(before,records['full']['values']) and before.item()>0


@pytest.mark.parametrize('dtype_name,large',[('float32',3e38),('float64',1e308)])
def test_v2_finite_raw_values_nonfinite_mean_rejected(dev,dtype_name,large):
    t,ns,_,_,_,_,summarize=dev
    def row(value): return dict(row_ids=('DEV-overflow-a','DEV-overflow-b'),values=t.tensor([value,value],dtype=getattr(t,dtype_name)))
    records={'full':row(large),'empty':row(0.),'selected':row(large)}
    records.update({f'random_{i}':row(0.) for i in range(20)})
    assert bool(t.isfinite(records['full']['values']).all())
    with pytest.raises(ValueError,match='nonfinite'): summarize(records)


def test_v2_cross_position_literal_oracle(dev):
    t,ns,sender,callback,oracle,_,_=dev
    ns['check_cross_position_dev'](t,sender,callback,oracle)


@pytest.mark.parametrize('mutation',['last_position','head_index','layer_offset','sign','stale'])
def test_v2_oracle_kills_method_mutations(generator,source_namespace,mutation):
    import torch as t
    source=generator.METHOD
    if mutation=='last_position':
        source=source.replace('return x - einsum("bpsd,sh->bphd", diff, removed[:, start:start + graph.n_heads])',
            'out = x.clone(); out[:, -1:] -= einsum("bpsd,sh->bphd", diff[:, -1:], removed[:, start:start + graph.n_heads]); return out')
    if mutation=='head_index': source=source.replace('model.W_O[hook.layer()]','model.W_O[hook.layer()].flip(0)')
    if mutation=='layer_offset': source=source.replace('start = graph.head_sender(hook.layer(), 0)','start = graph.head_sender(0, 0)')
    if mutation=='sign': source=source.replace('x - einsum','x + einsum')
    if mutation=='stale':
        source=source.replace('diff = zeros_like(corrupt_out)', 'diff = own_sender_outputs(model, clean_tokens, graph) - corrupt_out')
        source=source.replace('diff[:, :, 0] = x - corrupt_out[:, :, 0]', 'pass')
        source=source.replace('diff[:, :, start:start + graph.n_heads] = out - corrupt_out[:, :, start:start + graph.n_heads]', 'pass')
        source=source.replace('diff[:, :, u] = x - corrupt_out[:, :, u]', 'pass')
    assert source!=generator.METHOD
    ns=dict(source_namespace);exec(compile(source,'<intentional DEV mutation>','exec'),ns)
    sender,callback=ns['make_own_methods'](t)
    oracle=ns['make_controls'](t)[0]
    with pytest.raises(AssertionError): ns['check_cross_position_dev'](t,sender,callback,oracle)


def test_v2_plot_names_and_invalid_cells(dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t,ns,*_=dev;g=ns['own_graph'](1,1);edges=t.tensor(g.valid,dtype=t.bool)
    selected=edges.clone();selected[0,0]=False
    summary={'raw_means':{key:0. for key in ['full','empty','selected']+[f'random_{i}' for i in range(20)]}}
    fig=ns['plot_own_result'](plt,selected,summary,g,edges)
    axis=fig.axes[0];array=axis.images[0].get_array()
    assert set(array.flatten())=={-1.,0.,1.}
    assert [label.get_text() for label in axis.get_xticklabels()]==list(g.RECEIVER_NAMES)
    assert [label.get_text() for label in axis.get_yticklabels()]==list(g.SENDER_NAMES)
    assert [label.get_text() for label in fig.axes[2].get_yticklabels()]==['invalid edge','valid absent','retained']
    plt.close(fig)


def test_v2_mutate_then_raise_keeps_primary_and_identity_failure(source_namespace):
    import torch as t
    p,sender=dev_prepared_v2(t,source_namespace,t.float64)
    def bad(*args):
        p.heldout.clean_tokens[0,0]+=1
        raise RuntimeError('literal DEV primary failure')
    wired=source_namespace['bind_own_transfer'](circuit_logits=bad,sender_outputs=sender,prepared=p,
        kl_divergence=None,acdc=None,receiver_order=None,receiver_owner=None,random_circuit=None)
    with pytest.raises(source_namespace['TransferFailure']) as caught: wired.heldout_metric(p.edges)
    attempt=caught.value.attempt
    assert attempt['primary_failure']['message']=='literal DEV primary failure'
    assert 'post-failure identity validation' in attempt['secondary_failures'][0]
    assert attempt['completed']==0 and attempt['attempted']==1


def test_v2_nonfinite_numerator_rejected(dev):
    t,_,_,_,_,_,summarize=dev
    def row(value):return dict(row_ids=('DEV-numerator-overflow',),values=t.tensor([value],dtype=t.float64))
    records={'full':row(-1e308+1e292),'empty':row(-1e308),'selected':row(1e308)}
    records.update({f'random_{i}':row(-1e308) for i in range(20)})
    with pytest.raises(ValueError,match='numerator'):summarize(records)


def test_v2_completed_rows_are_snapshots_not_late_aliases(source_namespace):
    import torch as t
    p,sender=dev_prepared_v2(t,source_namespace,t.float32);source=t.tensor([.5,.25]*32,dtype=t.float32);count=[0]
    def metric(*args):
        count[0]+=1
        if count[0]==4:
            source.fill_(99.)
            raise RuntimeError('literal DEV aliases changed before late failure')
        return source
    p.heldout.metric=metric
    wired=source_namespace['bind_own_transfer'](circuit_logits=lambda *args:t.tensor([[.25,-.5,.75]]*64,dtype=t.float32),
        sender_outputs=sender,prepared=p,kl_divergence=None,acdc=None,
        receiver_order=None,receiver_owner=None,random_circuit=lambda edges,*args:edges.clone())
    with pytest.raises(source_namespace['TransferFailure']) as caught:wired.evaluate(p.edges)
    assert caught.value.attempt['completed']==3
    for row in caught.value.attempt['rows'].values():
        assert row['values'].tolist()==[.5,.25]*32 and row['values'] is not source


# V3 adverse regressions: fixed CPU literals and reported metadata only.
@pytest.mark.parametrize('cohort_name', ['discovery', 'heldout'])
@pytest.mark.parametrize('entry', [0, 1, 2, 3])
def test_v3_caller_list_attribution_mutation_rejected_before_callback(source_namespace, cohort_name, entry):
    import torch as t
    p, sender = dev_prepared_v2(t, source_namespace, t.float64)
    task = getattr(p, cohort_name)
    caller_binding = list(task.cache_binding)
    task.cache_binding = caller_binding
    calls = []
    wired = source_namespace['bind_own_transfer'](
        circuit_logits=lambda *args: calls.append(args) or t.tensor([[.25,-.5,.75]]*64, dtype=t.float64),
        sender_outputs=sender, prepared=p,
        kl_divergence=lambda *args: t.tensor(.125, dtype=t.float64),
        acdc=None, receiver_order=None, receiver_owner=None, random_circuit=None)
    assert calls == []
    caller_binding[entry] = 'f'*64 if entry == 3 else object()
    with pytest.raises(ValueError, match='attribution'):
        wired.discovery_kl(p.edges)
    assert calls == []


@pytest.mark.parametrize('binding_type', [tuple, list])
def test_v3_valid_attribution_sequence_contract_preserved(source_namespace, binding_type):
    import torch as t
    p, sender = dev_prepared_v2(t, source_namespace, t.float32)
    for task in (p.discovery, p.heldout):
        task.cache_binding = binding_type(task.cache_binding)
    wired = source_namespace['bind_own_transfer'](
        circuit_logits=lambda *args: t.tensor([[.25,-.5,.75]]*64, dtype=t.float32),
        sender_outputs=sender, prepared=p,
        kl_divergence=lambda *args: t.tensor(.125, dtype=t.float32),
        acdc=None, receiver_order=None, receiver_owner=None, random_circuit=None)
    assert wired.discovery_kl(p.edges) == .125


@pytest.mark.parametrize('exceptional', [False, True])
def test_v3_list_attribution_mutation_inside_callback_rejected_and_retained(source_namespace, exceptional):
    import torch as t
    p, sender = dev_prepared_v2(t, source_namespace, t.float64)
    p.heldout.cache_binding = list(p.heldout.cache_binding)
    value = t.tensor([[.25,-.5,.75]]*64, dtype=t.float64)
    def callback(*args):
        p.heldout.cache_binding[2] = object()
        p.heldout.cache_binding[3] = 'f'*64
        if exceptional:
            raise RuntimeError('DEV primary survives attribution drift')
        return value
    wired = source_namespace['bind_own_transfer'](
        circuit_logits=callback, sender_outputs=sender, prepared=p,
        kl_divergence=None, acdc=None, receiver_order=None, receiver_owner=None, random_circuit=None)
    with pytest.raises(source_namespace['TransferFailure']) as caught:
        wired.heldout_metric(p.edges)
    attempt = caught.value.attempt
    assert attempt['status'] == 'failed' and attempt['completed'] == 0
    if exceptional:
        assert attempt['primary_failure']['message'] == 'DEV primary survives attribution drift'
        assert 'attribution' in attempt['secondary_failures'][0]
    else:
        assert 'attribution' in attempt['primary_failure']['message']
        assert attempt['captures'][-1]['returned'] is value


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_v3_cpu_masks_independent_of_reported_activation_device(source_namespace, dtype_name, monkeypatch):
    import torch as t
    dtype = getattr(t, dtype_name)
    p, sender = dev_prepared_v2(t, source_namespace, dtype)
    class ReportedMetaTensor(t.Tensor):
        @property
        def device(self):
            return t.device('meta')
    # Storage and every numerical operation stay on CPU. Only device metadata
    # differs, allowing source wiring to be checked without a CUDA/MPS runtime.
    p.model.W_O = p.model.W_O.as_subclass(ReportedMetaTensor)
    p.model.cfg.device = 'meta'
    for task in (p.discovery, p.heldout):
        for field in ('clean_tokens', 'corrupt_tokens', 'corrupt_out', 'answers', 'full_clean_logprobs'):
            setattr(task, field, getattr(task, field).as_subclass(ReportedMetaTensor))
        task.cache_binding = [p.model, task.corrupt_tokens, sender, '2'*64]
    logits = t.tensor([[.25,-.5,.75]]*64, dtype=dtype).as_subclass(ReportedMetaTensor)
    kl = t.tensor(.125, dtype=dtype).as_subclass(ReportedMetaTensor)
    calls, compared_on = [], []
    tensor = t.tensor
    def tracked_tensor(data, *args, **kwargs):
        if data is p.graph.valid:
            compared_on.append(kwargs.get('device'))
        return tensor(data, *args, **kwargs)
    monkeypatch.setattr(t, 'tensor', tracked_tensor)
    wired = source_namespace['bind_own_transfer'](
        circuit_logits=lambda *args: calls.append(args) or logits,
        sender_outputs=sender, prepared=p, kl_divergence=lambda *args: kl,
        acdc=None, receiver_order=None, receiver_owner=None, random_circuit=None)
    assert calls == [] and compared_on == [t.device('cpu')]
    assert wired.discovery_kl(p.edges) == .125
    assert calls[0][3] is p.edges and calls[0][4] is p.edges
    assert p.edges.device == t.device('cpu')
    assert p.model.W_O.as_subclass(t.Tensor).device == t.device('cpu')
    assert p.discovery.corrupt_out.as_subclass(t.Tensor).device == t.device('cpu')
    with pytest.raises(ValueError, match='device'):
        wired.discovery_kl(p.edges.as_subclass(ReportedMetaTensor))
    assert len(calls) == 1
