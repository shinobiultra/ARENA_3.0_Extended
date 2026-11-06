"""Blank graph prediction companion. Import/freeze/reveal/replay never evaluate a DAG.

Local seals detect accidental changes, not hostile code or prior human exposure.
The notebook owns every taught graph method; this module owns record plumbing.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import struct
import time

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / 'docs/mastery/graph_prediction_protocol.json'
TESTS = ROOT / 'tests/mastery/test_graph_prediction.py'
SCHEMA = 'arena-blind-attribution-graph-prediction'
VERSION = 1
CASE = {
    'id': '8.7-predetermined-weight-variant-v1',
    'node_order': ['feature:country=France', 'head:subject-router', 'mlp:relation-lookup',
                   'feature:capital=Paris', 'head:syntax-router', 'mlp:style-cleanup',
                   'logit:Paris-vs-Rome'],
    'edge_weights': [[0,1,.92], [1,2,1.08], [2,3,.96], [3,6,1.17], [2,6,.27],
                     [0,4,.41], [4,5,.33], [5,6,.24], [1,5,.16]],
    'clean_inputs': [1,0,0,0,0,0,0], 'corrupt_inputs': [0,0,0,0,0,0,0],
    'source_index': 0, 'target_index': 6, 'max_depth': 4,
    'threshold': .25, 'ig_steps': 16, 'random_seed': 0,
    'path_fraction': .8, 'control_margin': .5,
    'dtype': 'torch.float32', 'device': 'cpu', 'threads': 1, 'budget_seconds': 120,
    'metric': 'scalar final node activation: logit:Paris-vs-Rome',
    'roles': {'TRAIN':'none', 'DEV':'published historical lesson and distinct practice fixtures; exposed',
              'TEST':'none', 'target':'fixed synthetic prereveal diagnostic; no distributional claim'},
}
METHOD_NAMES = ('make_graph', 'activations', 'scalar_metric', 'exact_edge_effects',
                'integrated_scores', 'threshold_graph', 'top_path', 'path_report',
                'graph_controls', 'assemble_result')
ASSESSMENT = 'local_self_report_no_unseen_certification'
_ATTEMPTED = set()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seal(payload):
    out = deepcopy(payload)
    out['seal'] = digest(out)
    return json.dumps(out, sort_keys=True, allow_nan=False)


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate JSON key')
        out[key] = value
    return out


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('missing or unknown fields')


def _finite(value, lower=None, upper=None):
    if (type(value) not in (float, int) or not math.isfinite(value)
            or (lower is not None and value < lower) or (upper is not None and value > upper)):
        raise ValueError('expected finite number in declared range')


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('expected lowercase SHA256')


def _raw_inputs(case):
    """Pack literal values only: no constructor, tensor, activation, score or search."""
    n = len(case['node_order'])
    weights = [0.] * (n*n)
    for u,v,w in case['edge_weights']:
        weights[u*n+v] = w
    values = {'weights': weights, 'clean': case['clean_inputs'], 'corrupt': case['corrupt_inputs']}
    return {key: {'shape': [n,n] if key == 'weights' else [n], 'dtype':'torch.float32',
                  'sha256': hashlib.sha256(struct.pack('<'+'f'*len(value), *value)).hexdigest()}
            for key,value in values.items()}


def environment_identity():
    # Runtime configuration only; no target tensors or student definitions.
    import torch as t
    return {'python': platform.python_version(), 'torch': version('torch'),
            'threads':t.get_num_threads(),'default_dtype':str(t.get_default_dtype()),
            'default_device':str(t.get_default_device()),
            'deterministic_algorithms':t.are_deterministic_algorithms_enabled(),
            'platform': platform.platform(), 'byteorder':'little',
            'execution_contract': {'device':'cpu','dtype':'torch.float32','threads':1}}


def validate_source(source, *, complete=True):
    """Definitions only: never execute student source while freezing or committing."""
    if type(source) is not str or len(source.encode()) > 200_000:
        raise ValueError('expected bounded method source')
    tree = ast.parse(source)
    names = []
    for node in tree.body:
        if (not isinstance(node, ast.FunctionDef) or node.decorator_list
                or getattr(node,'type_params',[])):
            raise ValueError('only undecorated function definitions are allowed at module scope')
        # Defaults/annotations execute at definition time. Keep signatures plain.
        if (node.returns or node.args.defaults or any(v is not None for v in node.args.kw_defaults)
                or any(a.annotation for a in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs])
                or (node.args.vararg and node.args.vararg.annotation)
                or (node.args.kwarg and node.args.kwarg.annotation)):
            raise ValueError('method signatures must have no defaults or annotations')
        names.append(node.name)
    if len(set(names)) != len(names) or any(n not in METHOD_NAMES for n in names):
        raise ValueError('unknown or duplicate core function')
    if complete and set(names) != set(METHOD_NAMES):
        raise ValueError('define every named core method')
    compile(tree, '<graph learner methods>', 'exec')
    return source


def freeze_inputs(method_source):
    validate_source(method_source)
    return {'case_sha256': digest(CASE), 'method_sha256': sha_text(method_source),
            'helper_sha256': _file(__file__), 'tests_sha256': _file(TESTS),
            'protocol_sha256': _file(PROTOCOL), 'raw_inputs': _raw_inputs(CASE),
            'node_order': deepcopy(CASE['node_order']), 'environment': environment_identity()}


def _validate_identity(identity, source):
    _keys(identity, {'case_sha256','method_sha256','helper_sha256','tests_sha256','protocol_sha256',
                     'raw_inputs','node_order','environment'})
    for key in ('case_sha256','method_sha256','helper_sha256','tests_sha256','protocol_sha256'):
        _sha(identity[key])
    if (identity['case_sha256'] != digest(CASE) or identity['method_sha256'] != sha_text(source)
            or digest(identity['raw_inputs']) != digest(_raw_inputs(CASE)) or identity['node_order'] != CASE['node_order']):
        raise ValueError('frozen case / method / raw input identity mismatch')
    env = identity['environment']
    _keys(env, {'python','torch','platform','byteorder','execution_contract','threads','default_dtype','default_device','deterministic_algorithms'})
    if (any(type(env[k]) is not str or not env[k] for k in ('python','torch','platform'))
            or env['byteorder'] != 'little'
            or type(env['threads']) is not int or env['threads'] != 1
            or env['default_dtype'] != 'torch.float32' or env['default_device'] != 'cpu'
            or type(env['deterministic_algorithms']) is not bool
            or digest(env['execution_contract']) != digest({'device':'cpu','dtype':'torch.float32','threads':1})):
        raise ValueError('invalid runtime identity')


def commit_prediction(identity, method_source, prediction, confidence, reasoning, prior_exposure='unknown'):
    """Return a new immutable sealed JSON text. No graph function is called."""
    validate_source(method_source)
    if identity != freeze_inputs(method_source):
        raise ValueError('stale identity; freeze again before committing')
    text = _seal({'schema':SCHEMA,'schema_version':VERSION,'case':deepcopy(CASE),
                  'identity':deepcopy(identity),'method_source':method_source,
                  'prediction':prediction,'confidence':confidence,'reasoning':reasoning,
                  'prior_exposure':prior_exposure,'assessment':ASSESSMENT,'status':'committed'})
    read_record(text)
    return text


def read_record(text):
    """Strict local format. Replay checks stored evidence, without evaluating methods."""
    if type(text) is not str or len(text.encode()) > 2_000_000:
        raise ValueError('expected bounded JSON text')
    record = json.loads(text, object_pairs_hook=_pairs,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if type(record) is not dict or 'seal' not in record:
        raise ValueError('missing sealed record')
    seal = record.pop('seal')
    if seal != digest(record):
        raise ValueError('record seal mismatch')
    if text != _seal(record):
        raise ValueError('immutable record text must retain its canonical JSON bytes')
    status = record.get('status')
    keys = {'schema','schema_version','case','identity','method_source','prediction','confidence',
            'reasoning','prior_exposure','assessment','status'}
    if status in ('observed','failed'):
        keys |= {'commitment_sha256', 'observation' if status == 'observed' else 'failure'}
    _keys(record, keys)
    if (record['schema'] != SCHEMA or type(record['schema_version']) is not int
            or record['schema_version'] != VERSION or digest(record['case']) != digest(CASE)
            or status not in ('committed','observed','failed') or record['assessment'] != ASSESSMENT):
        raise ValueError('unsupported graph case / status / schema')
    if type(record['prediction']) is not bool:
        raise ValueError('enter a boolean forecast')
    _finite(record['confidence'],0,1)
    if type(record['reasoning']) is not str or not record['reasoning'].strip():
        raise ValueError('enter causal reasoning')
    if record['prior_exposure'] not in ('unknown','seen','self_reported_not_seen'):
        raise ValueError('invalid exposure')
    validate_source(record['method_source'])
    _validate_identity(record['identity'], record['method_source'])
    if status in ('observed','failed'):
        parent = {k:v for k,v in record.items() if k not in ('commitment_sha256','observation','failure')}
        parent['status'] = 'committed'
        if record['commitment_sha256'] != sha_text(_seal(parent)):
            raise ValueError('attempt does not link to immutable commitment')
    if status == 'observed':
        _validate_measurements(record['observation'],CASE)
    elif status == 'failed':
        f = record['failure']
        _keys(f, {'stage','error_type','message','elapsed_seconds'})
        if any(type(f[k]) is not str or not f[k] for k in ('stage','error_type','message')):
            raise ValueError('malformed failure')
        _finite(f['elapsed_seconds'],0)
    return record


def assert_current(record_text, method_source):
    record = read_record(record_text)
    if record['status'] != 'committed':
        raise ValueError('attempt already observed or failed; preserve it')
    if method_source != record['method_source'] or record['identity'] != freeze_inputs(method_source):
        raise ValueError('stale inputs/source/helper/tests/protocol/runtime; new commitment required')
    return record


def namespace(method_source):
    validate_source(method_source)
    import torch as t
    ns = {'t':t, 'random':random, 'math':math}
    exec(compile(method_source, '<graph learner methods>', 'exec'), ns)
    return ns


def reveal_methods(record_text):
    record = read_record(record_text)
    assert_current(record_text, record['method_source'])
    return REFERENCE_METHODS


def _matrix(value, n):
    if type(value) is not list or len(value) != n:
        raise ValueError('wrong score shape')
    for row in value:
        if type(row) is not list or len(row) != n:
            raise ValueError('wrong score shape')
        for item in row:
            _finite(item)


def _selected_from_scores(scores, threshold):
    # Selection from STORED scores only. No DAG or student method is evaluated.
    candidates = [(abs(s),u,v,s) for u,row in enumerate(scores) for v,s in enumerate(row)
                  if abs(s) >= threshold]
    candidates.sort(key=lambda x:(-x[0],x[1],x[2]))
    return [[u,v,round(s,6)] for _,u,v,s in candidates if s != 0]


def _path_from_edges(edges, source, target, depth):
    adjacency = {}
    for u,v,s in edges:
        adjacency.setdefault(u,[]).append((v,s))
    for group in adjacency.values():
        group.sort(key=lambda e:abs(e[1]), reverse=True)
    best, best_score = [], -math.inf
    def visit(u, path, score):
        nonlocal best,best_score
        if u == target and len(path)>1:
            if score>best_score:
                best,best_score = path,score
            return
        if len(path)-1 >= depth:
            return
        for v,s in adjacency.get(u,[]):
            if v not in path:
                visit(v,path+[v],score*abs(s))
    visit(source,[source],1.)
    if not best:
        raise ValueError('no source-target path; preserve failed attempt')
    return best


def _validate_measurements(obs, case, *, check_path_choice=False):
    _keys(obs, {'kind','elapsed_seconds','scores','selected_edges','path','path_drops','metrics',
                'ratios','control_edges','margins','gates','predicate'})
    if obs['kind'] != 'fresh_cpu_graph_interventions':
        raise ValueError('historical/guided/foreign observations are not target evidence')
    _finite(obs['elapsed_seconds'],0,case['budget_seconds'])
    n = len(case['node_order'])
    active = {(u,v) for u,v,_ in case['edge_weights']}
    _keys(obs['scores'], {'exact','IG16'})
    for scores in obs['scores'].values():
        _matrix(scores,n)
        if any(s != 0 for u,row in enumerate(scores) for v,s in enumerate(row) if (u,v) not in active):
            raise ValueError('scores outside active support')
    def edges(value):
        if type(value) is not list:
            raise ValueError('malformed graph edge list')
        for edge in value:
            if (type(edge) is not list or len(edge)!=3 or any(type(v) is not int or not 0<=v<n for v in edge[:2])):
                raise ValueError('malformed edge indices')
            _finite(edge[2])
    edges(obs['selected_edges'])
    selected = _selected_from_scores(obs['scores']['IG16'], case['threshold'])
    if obs['selected_edges'] != selected or any((u,v) not in active for u,v,_ in selected):
        raise ValueError('selected graph differs from frozen threshold rule')
    if (type(obs['path']) is not list or any(type(v) is not int or not 0<=v<n for v in obs['path'])):
        raise ValueError('malformed path indices')
    if (len(obs['path'])<2 or len(obs['path'])-1>case['max_depth']
            or len(set(obs['path']))!=len(obs['path'])
            or obs['path'][0]!=case['source_index'] or obs['path'][-1]!=case['target_index']
            or any((u,v) not in {(a,b) for a,b,_ in selected} for u,v in zip(obs['path'],obs['path'][1:]))):
        raise ValueError('missing or invalid recovered selected-graph path')
    # Search is an explicit postcommit run check only; stored replay does not search.
    if check_path_choice and obs['path'] != _path_from_edges(selected,case['source_index'],case['target_index'],case['max_depth']):
        raise ValueError('path differs from stored score search')
    path_edges = list(zip(obs['path'],obs['path'][1:]))
    if (type(obs['path_drops']) is not list or len(obs['path_drops']) != len(path_edges)
            or any(obs['scores']['exact'][u][v] != drop for (u,v),drop in zip(path_edges,obs['path_drops']))):
        raise ValueError('missing or inconsistent per-edge interventions')
    _keys(obs['metrics'], {'full','corrupt','path_only','path_removed','selected','random','reversed'})
    for value in obs['metrics'].values():
        _finite(value)
    m = obs['metrics']; denominator = m['full']-m['corrupt']
    if denominator == 0:
        raise ValueError('zero clean-corrupt denominator')
    ratios = {'faithfulness':(m['path_only']-m['corrupt'])/denominator,
              'completeness':(m['full']-m['path_removed'])/denominator,
              'minimality':min(obs['path_drops'])/denominator}
    margins = {'random':m['selected']-m['random'],'reversed':m['selected']-m['reversed']}
    if obs['ratios'] != ratios or obs['margins'] != margins:
        raise ValueError('ratio/margin denominator or unrounded arithmetic mismatch')
    _keys(obs['ratios'], {'faithfulness','completeness','minimality'})
    _keys(obs['margins'], {'random','reversed'})
    for value in [*obs['ratios'].values(),*obs['margins'].values(),*obs['path_drops']]:
        _finite(value)
    _keys(obs['control_edges'], {'random','reversed'})
    edges(obs['control_edges']['random']); edges(obs['control_edges']['reversed'])
    excluded = {(u,v) for u,v,_ in selected}
    # Match original torch.where row-major order; sample actual nonzero weights.
    pool = sorted((u,v,w) for u,v,w in case['edge_weights'] if (u,v) not in excluded)
    if len(pool) < len(selected):
        raise ValueError('insufficient nonselected random pool')
    # Float32 weight storage precedes six-decimal graph-score display.
    sampled = random.Random(case['random_seed']).sample(pool,k=len(selected))
    expected_random = [[u,v,round(struct.unpack('<f',struct.pack('<f',w))[0],6)] for u,v,w in sampled]
    if (obs['control_edges']['random'] != expected_random
            or obs['control_edges']['reversed'] != [[v,u,s] for u,v,s in selected]):
        raise ValueError('missing/mutated equal-size or reversed controls')
    gates = {'path':all(v>=case['path_fraction'] for v in ratios.values()),
             'random':margins['random']>=case['control_margin'],
             'reversed':margins['reversed']>=case['control_margin']}
    _keys(obs['gates'], {'path','random','reversed'})
    if (obs['gates'] != gates or any(type(v) is not bool for v in obs['gates'].values())
            or type(obs['predicate']) is not bool or obs['predicate'] != all(gates.values())):
        raise ValueError('inclusive gates differ from stored unrounded values')
    digest(obs)


def _graph_snapshot(graph):
    _keys(graph, {'nodes','source','target','max_depth','weights','clean','corrupt'})
    if any(type(graph[k]) is not int for k in ('source','target','max_depth')):
        raise ValueError('graph indices/depth must be exact integers')
    return {'nodes':list(graph['nodes']), 'source':graph['source'], 'target':graph['target'],
            'max_depth':graph['max_depth'],
            'tensors':{key:{'shape':list(graph[key].shape),'dtype':str(graph[key].dtype),
                           'device':str(graph[key].device),
                           'sha256':hashlib.sha256(graph[key].detach().contiguous().numpy().tobytes()).hexdigest()}
                       for key in ('weights','clean','corrupt')}}


def _bind_assembly(assembled, captured, elapsed_seconds):
    """Strict untrusted student payload before adding recorder-owned metadata."""
    _keys(assembled, {'scores','selected_edges','path','path_drops','metrics',
                      'ratios','control_edges','margins','gates','predicate'})
    for key,value in captured.items():
        if assembled[key] != value:
            raise ValueError('assembly changed captured measurements')
    _finite(elapsed_seconds,0)
    return {**deepcopy(assembled),'kind':'fresh_cpu_graph_interventions',
            'elapsed_seconds':elapsed_seconds}


def run_experiment(record_text, method_source, *, authorized=False):
    """One explicit future attempt. Exceptions become immutable linked failure receipts.

    This local opt-in is not external permission or a hard timeout. Use a separate
    authorized process with an external 120-second timeout for a target attempt.
    """
    record = read_record(record_text)
    if record['status'] != 'committed':
        raise ValueError('preserve the completed/failed attempt; do not retry it')
    if authorized is not True:
        raise ValueError('a separately authorized target attempt is required')
    attempt_key=sha_text(record_text)
    if attempt_key in _ATTEMPTED:
        raise ValueError('this process already attempted this commitment; preserve the first receipt')
    _ATTEMPTED.add(attempt_key)
    started = time.monotonic(); stage = 'identity'
    def deadline():
        if time.monotonic()-started > CASE['budget_seconds']:
            raise TimeoutError('120-second aggregate ceiling exceeded')
    try:
        assert_current(record_text,method_source)
        import torch as t
        if (t.get_num_threads()!=1 or t.get_default_dtype()!=t.float32
                or str(t.get_default_device())!='cpu'):
            raise ValueError('set one CPU thread, float32 and CPU default device')
        ns = namespace(method_source)
        stage = 'construct'
        graph = ns['make_graph'](deepcopy(CASE))
        frozen = _graph_snapshot(graph)
        expected = record['identity']['raw_inputs']
        if (frozen['nodes'] != CASE['node_order'] or frozen['source'] != CASE['source_index']
                or frozen['target'] != CASE['target_index'] or frozen['max_depth'] != CASE['max_depth']
                or {k:{a:b for a,b in v.items() if a!='device'} for k,v in frozen['tensors'].items()} != expected
                or any(v['device']!='cpu' for v in frozen['tensors'].values())):
            raise ValueError('constructor changed frozen raw inputs or graph metadata')
        def call(name,*args):
            nonlocal stage
            stage = name; deadline()
            if _graph_snapshot(graph) != frozen or environment_identity()!=record['identity']['environment']:
                raise ValueError('method mutated frozen graph or runtime')
            out = ns[name](*args)
            deadline()
            if _graph_snapshot(graph) != frozen or environment_identity()!=record['identity']['environment']:
                raise ValueError('method mutated frozen graph or runtime')
            return out
        def metric(mask,inputs):
            value = call('scalar_metric',graph,mask.clone(),inputs.clone())
            value = float(value.detach().item()) if hasattr(value,'detach') else value
            _finite(value)
            return value
        active = (graph['weights']!=0).float()
        exact = call('exact_edge_effects',graph).detach().clone()
        ig = call('integrated_scores',graph,CASE['ig_steps']).detach().clone()
        for scores in (exact,ig):
            if (scores.shape != graph['weights'].shape or scores.dtype!=t.float32
                    or scores.device.type!='cpu' or not t.isfinite(scores).all()):
                raise ValueError('invalid score matrix')
        selected = deepcopy(call('threshold_graph',ig.clone(),CASE['threshold']))
        path = deepcopy(call('top_path',selected,graph['source'],graph['target'],graph['max_depth']))
        if not path:
            raise ValueError('no recovered source-target path')
        path_result = deepcopy(call('path_report',graph,list(path),exact.clone()))
        control_result = deepcopy(call('graph_controls',graph,deepcopy(selected),CASE['random_seed']))
        # Bind student assembly to independently captured calls of the taught scalar method.
        m = {'full':metric(active,graph['clean']), 'corrupt':metric(active,graph['corrupt'])}
        def mask(edges):
            out = t.zeros_like(active)
            for u,v,*_ in edges:
                out[u,v] = 1
            return out
        path_pairs = list(zip(path,path[1:]))
        removed = active.clone()
        for u,v in path_pairs:
            removed[u,v] = 0
        m.update(path_only=metric(mask(path_pairs),graph['clean']),
                 path_removed=metric(removed,graph['clean']),
                 selected=metric(mask(selected),graph['clean']),
                 random=metric(mask(control_result['random_edges']),graph['clean']),
                 reversed=metric(mask(control_result['reversed_edges']),graph['clean']))
        for u,v in zip(*t.where(active!=0)):
            ablated = active.clone(); ablated[u,v] = 0
            # Preserve original float32 exact-score storage.
            measured = t.tensor(m['full']-metric(ablated,graph['clean']),dtype=t.float32).item()
            if exact[u,v].item() != measured:
                raise ValueError('exact score differs from captured single-edge intervention')
        expected_path = {'metrics':{k:m[k] for k in ('full','corrupt','path_only','path_removed')},
                         'drops':[exact[u,v].item() for u,v in path_pairs]}
        if path_result != expected_path:
            raise ValueError('path report differs from captured intervention measurements')
        if control_result['metrics'] != {k:m[k] for k in ('selected','random','reversed')}:
            raise ValueError('control report differs from captured control measurements')
        assembled = call('assemble_result',exact.clone(),ig.clone(),deepcopy(selected),list(path),
                         deepcopy(path_result),deepcopy(control_result))
        expected_assembly = {'scores':{'exact':exact.tolist(),'IG16':ig.tolist()},
            'selected_edges':selected,'path':path,'path_drops':expected_path['drops'],'metrics':m,
            'control_edges':{'random':control_result['random_edges'],'reversed':control_result['reversed_edges']}}
        stage = 'assembly_binding'
        obs = _bind_assembly(assembled,expected_assembly,time.monotonic()-started)
        stage = 'receipt_validation'; _validate_measurements(obs,CASE,check_path_choice=True); deadline()
        record.update(status='observed',observation=obs,commitment_sha256=sha_text(record_text))
    except Exception as error:
        record.update(status='failed',commitment_sha256=sha_text(record_text),
                      failure={'stage':stage,'error_type':type(error).__name__,
                               'message':str(error)[:1500] or type(error).__name__,
                               'elapsed_seconds':time.monotonic()-started})
    result = _seal(record)
    read_record(result)
    return result


def replay(receipt_text):
    record = read_record(receipt_text)
    if record['status'] != 'observed':
        raise ValueError('no completed measured observation to replay')
    return deepcopy(record['observation'])


def export_record(record_text, destination):
    """Opt-in, exclusive, owner-readable file; never overwrite an existing attempt."""
    read_record(record_text)
    path = Path(destination)
    fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream:
        stream.write(record_text)
    return path


def plot_receipt(receipt_text):
    """Only stored values: selected graph/path, signed heatmaps, ratios and controls."""
    import matplotlib.pyplot as plt
    import numpy as np
    obs = replay(receipt_text)
    fig,axes = plt.subplots(2,3,figsize=(17,9),layout='constrained')
    xy = {i:(i%4,-(i//4)) for i in range(len(CASE['node_order']))}
    path_edges = set(zip(obs['path'],obs['path'][1:]))
    ax = axes[0,0]
    for u,v,s in obs['selected_edges']:
        ax.annotate('',xy=xy[v],xytext=xy[u],arrowprops={'arrowstyle':'->',
                    'color':'crimson' if (u,v) in path_edges else 'gray'})
        ax.text((xy[u][0]+xy[v][0])/2,(xy[u][1]+xy[v][1])/2,f'{s:.3f}',fontsize=7)
    for i,(x,y) in xy.items():
        ax.text(x,y,str(i),bbox={'boxstyle':'circle','fc':'white'},ha='center',va='center')
    ax.set(xlim=(-.5,3.5),ylim=(-1.5,.5),title='Stored selected graph; red = recovered path'); ax.axis('off')
    for ax,name in zip(axes[0,1:],('exact','IG16')):
        values=np.array(obs['scores'][name]); limit=max(float(abs(values).max()),1e-12)
        im=ax.imshow(values,cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set(title=name+' signed scores',xlabel='Target index',ylabel='Source index')
        fig.colorbar(im,ax=ax)
    for ax,key,threshold in [(axes[1,0],'ratios',CASE['path_fraction']),
                              (axes[1,1],'margins',CASE['control_margin'])]:
        names=list(obs[key]); values=[obs[key][n] for n in names]
        ax.bar(names,values); ax.axhline(threshold,color='black',ls='--')
        ax.set(title=f'Stored {key}; inclusive threshold {threshold}',ylabel='Unrounded values determine gates')
    axes[1,2].axis('off')
    axes[1,2].text(0,1,'Node key\n'+'\n'.join(f'{i}: {n}' for i,n in enumerate(CASE['node_order']))+
                   '\n\nCombined predicate: '+str(obs['predicate'])+
                   '\nSynthetic diagnostic; no mastery badge.\nLabels rounded only for display.',va='top',fontsize=8)
    return fig


# Generic method guidance: no target construction, scores, path, metrics or output.
REFERENCE_METHODS = '''def make_graph(case):
    n = len(case['node_order'])
    weights = t.zeros((n,n),dtype=t.float32,device='cpu')
    for u,v,w in case['edge_weights']:
        if not 0 <= u < v < n or not math.isfinite(w):
            raise ValueError('require finite topologically ordered forward weights')
        weights[u,v] = w
    return {'nodes':list(case['node_order']), 'weights':weights,
            'clean':t.tensor(case['clean_inputs'],dtype=t.float32,device='cpu'),
            'corrupt':t.tensor(case['corrupt_inputs'],dtype=t.float32,device='cpu'),
            'source':case['source_index'],'target':case['target_index'],
            'max_depth':case['max_depth']}

def activations(graph, edge_mask, inputs):
    if edge_mask.shape != graph['weights'].shape or inputs.shape != graph['clean'].shape:
        raise ValueError('input / edge mask shape mismatch')
    if not t.isfinite(edge_mask).all() or not t.isfinite(inputs).all():
        raise ValueError('nonfinite input / edge mask')
    values = [inputs[i] for i in range(len(graph['nodes']))]
    weights = graph['weights'] * edge_mask
    for v in range(1,len(values)):
        values[v] = values[v] + sum(values[u]*weights[u,v] for u in range(v))
    return t.stack(values)

def scalar_metric(graph, edge_mask, inputs):
    return activations(graph,edge_mask,inputs)[graph['target']]

def exact_edge_effects(graph):
    active = (graph['weights']!=0).float()
    full = float(scalar_metric(graph,active,graph['clean']))
    effects = t.zeros_like(active)
    for u,v in zip(*t.where(active!=0)):
        keep = active.clone()
        keep[u,v] = 0
        effects[u,v] = full-float(scalar_metric(graph,keep,graph['clean']))
    return effects

def integrated_scores(graph, steps):
    if type(steps) is not int or steps <= 0:
        raise ValueError('positive integer midpoint count required')
    active = (graph['weights']!=0).float()
    scores = t.zeros_like(active)
    for k in range(steps):
        keep = (active*((k+.5)/steps)).detach().requires_grad_(True)
        output = scalar_metric(graph,keep,graph['clean'])
        gradient, = t.autograd.grad(output,keep)
        scores = scores + gradient.detach()*active
    return scores/steps

def threshold_graph(scores, threshold):
    if threshold <= 0 or not math.isfinite(threshold) or not t.isfinite(scores).all():
        raise ValueError('finite scores and positive threshold required')
    candidates = []
    for u in range(scores.shape[0]):
        for v in range(scores.shape[1]):
            s = float(scores[u,v])
            if abs(s) >= threshold:
                candidates.append((abs(s),u,v,s))
    candidates.sort(key=lambda item:(-item[0],item[1],item[2]))
    return [[u,v,round(s,6)] for _,u,v,s in candidates if s != 0]

def top_path(edges, source, target, max_depth):
    adjacency = {}
    for u,v,s in edges:
        adjacency.setdefault(u,[]).append((v,s))
    for group in adjacency.values():
        group.sort(key=lambda e:abs(e[1]),reverse=True)
    best = []
    best_score = -math.inf
    def search(u, path, score):
        nonlocal best,best_score
        if u == target and len(path)>1:
            if score > best_score:
                best,best_score = path,score
            return
        if len(path)-1 >= max_depth:
            return
        for v,s in adjacency.get(u,[]):
            if v not in path:
                search(v,path+[v],score*abs(s))
    search(source,[source],1.)
    if not best:
        raise ValueError('no selected source-target path')
    return best

def path_report(graph, path, exact):
    if len(path)<2:
        raise ValueError('nonempty recovered path required')
    active = (graph['weights']!=0).float()
    only = t.zeros_like(active)
    removed = active.clone()
    pairs = list(zip(path,path[1:]))
    for u,v in pairs:
        only[u,v] = 1
        removed[u,v] = 0
    metrics = {'full':float(scalar_metric(graph,active,graph['clean'])),
               'corrupt':float(scalar_metric(graph,active,graph['corrupt'])),
               'path_only':float(scalar_metric(graph,only,graph['clean'])),
               'path_removed':float(scalar_metric(graph,removed,graph['clean']))}
    if metrics['full'] == metrics['corrupt']:
        raise ValueError('zero clean-corrupt denominator')
    return {'metrics':metrics,'drops':[float(exact[u,v]) for u,v in pairs]}

def graph_controls(graph, selected, seed):
    if not selected:
        raise ValueError('empty selected graph')
    excluded = {(u,v) for u,v,_ in selected}
    pool = [(int(u),int(v),float(graph['weights'][u,v]))
            for u,v in zip(*t.where(graph['weights']!=0)) if (int(u),int(v)) not in excluded]
    if len(pool)<len(selected):
        raise ValueError('insufficient nonselected edges for equal-size control')
    random_edges = [[u,v,round(w,6)] for u,v,w in random.Random(seed).sample(pool,k=len(selected))]
    reversed_edges = [[v,u,s] for u,v,s in selected]
    metrics = {}
    for name,edges in [('selected',selected),('random',random_edges),('reversed',reversed_edges)]:
        mask = t.zeros_like(graph['weights'])
        for u,v,_ in edges:
            mask[u,v] = 1
        metrics[name] = float(scalar_metric(graph,mask,graph['clean']))
    return {'metrics':metrics,'random_edges':random_edges,'reversed_edges':reversed_edges}

def assemble_result(exact, ig, selected, path, path_result, controls):
    metrics = {**path_result['metrics'],**controls['metrics']}
    denominator = metrics['full']-metrics['corrupt']
    if denominator == 0:
        raise ValueError('zero clean-corrupt denominator')
    ratios = {'faithfulness':(metrics['path_only']-metrics['corrupt'])/denominator,
              'completeness':(metrics['full']-metrics['path_removed'])/denominator,
              'minimality':min(path_result['drops'])/denominator}
    margins = {'random':metrics['selected']-metrics['random'],
               'reversed':metrics['selected']-metrics['reversed']}
    gates = {'path':all(v>=.8 for v in ratios.values()),
             'random':margins['random']>=.5,'reversed':margins['reversed']>=.5}
    return {'scores':{'exact':exact.tolist(),'IG16':ig.tolist()},'selected_edges':selected,
            'path':path,'path_drops':path_result['drops'],'metrics':metrics,
            'ratios':ratios,'control_edges':{'random':controls['random_edges'],
                                           'reversed':controls['reversed_edges']},
            'margins':margins,'gates':gates,'predicate':all(gates.values())}
'''


def practice_namespace(source):
    validate_source(source,complete=False)
    import torch as t
    ns = {'t':t,'random':random,'math':math}
    exec(compile(source,'<distinct DEV fixture methods>','exec'),ns)
    return ns


def practice_test(ns, stage):
    """Tiny DISTINCT DEV organism, never CASE or the seven-node target.

    Four anonymous nodes only. Distinct 2-edge and signed 3-edge chains, plus a
    parallel-route nonzero-corrupt fixture expose integration and normalization
    mistakes. Analytic expected values are independent of the student methods.
    """
    import torch as t
    case = {'node_order':['practice-input','practice-middle','practice-output','practice-tail'],
            'edge_weights':[[0,1,2.],[1,2,.5],[0,3,.125],[2,3,.25]],
            'clean_inputs':[1.,0.,0.,0.], 'corrupt_inputs':[0.,0.,0.,0.],
            'source_index':0,'target_index':2,'max_depth':2}
    chain = {'node_order':['practice-input','practice-chain-a','practice-chain-b','practice-output'],
             'edge_weights':[[0,1,2.],[1,2,.5],[2,3,3.]],
             'clean_inputs':[1.,0.,0.,0.], 'corrupt_inputs':[.25,0.,0.,0.],
             'source_index':0,'target_index':3,'max_depth':3}
    parallel = {**chain,'edge_weights':[*chain['edge_weights'],[1,3,.5],[0,3,1.]]}
    # Route products are 3, 1 and 1: full=5, quarter-input corrupt=1.25.
    # Keeping only the long chain gives 3; removing it leaves direct route 1.
    # Full-graph single-edge drops on that chain are 4,3,3.
    parallel_expected = {'metrics':{'full':5.,'corrupt':1.25,'path_only':3.,'path_removed':1.},
                         'drops':[4.,3.,3.]}
    graph = ns['make_graph'](deepcopy(case))
    active = (graph['weights']!=0).float()
    if stage == 'graph':
        assert t.equal(ns['activations'](graph,active,graph['clean']),t.tensor([1.,2.,1.,.375]))
        assert float(ns['scalar_metric'](graph,active,graph['corrupt'])) == 0.
    elif stage == 'scores':
        exact = ns['exact_edge_effects'](graph)
        ig = ns['integrated_scores'](graph,4)
        expected = t.zeros((4,4)); expected[0,1]=expected[1,2]=1.
        assert t.equal(exact,expected)
        assert t.equal(ig,expected*.5)
        negative = deepcopy(case); negative['edge_weights'][1][2] = -.5
        other = ns['make_graph'](negative)
        assert ns['exact_edge_effects'](other)[0,1].item() == -1.
        assert ns['integrated_scores'](other,4)[1,2].item() == -.5
        # Analytic independent oracle: M(alpha)=sign*3*alpha**3. Each active
        # gate derivative is sign*3*alpha**2. Midpoint mean is
        # sign*(1-1/(4*steps**2)); this depends on BOTH depth and steps.
        for sign in (1.,-1.):
            signed = deepcopy(chain); signed['edge_weights'][2][2] *= sign
            deep_graph = ns['make_graph'](signed)
            for steps in (2,4):
                expected = t.zeros((4,4),dtype=t.float32)
                value = sign*(1-1/(4*steps**2))
                for u,v in ((0,1),(1,2),(2,3)):
                    expected[u,v] = value
                measured = ns['integrated_scores'](deep_graph,steps)
                assert t.equal(measured,expected), 'signed 3-edge midpoint oracle: depth/steps/unsupported scores'
    elif stage == 'selection':
        scores = t.zeros((4,4)); scores[0,1]=-.5; scores[1,2]=.5
        edges = ns['threshold_graph'](scores,.5)
        assert edges == [[0,1,-.5],[1,2,.5]]
        assert ns['top_path'](edges,0,2,2) == [0,1,2]
        try:
            ns['top_path']([],0,2,2)
        except ValueError:
            pass
        else:
            raise AssertionError('missing path must fail')
    elif stage == 'interventions':
        exact = ns['exact_edge_effects'](graph)
        report = ns['path_report'](graph,[0,1,2],exact)
        controls = ns['graph_controls'](graph,[[0,1,.5],[1,2,.5]],0)
        assert report == {'metrics':{'full':1.,'corrupt':0.,'path_only':1.,'path_removed':0.},'drops':[1.,1.]}
        assert controls['metrics'] == {'selected':1.,'random':0.,'reversed':0.}
        assert len(controls['random_edges']) == 2
        try:
            ns['graph_controls'](graph,[[0,1,.5],[1,2,.5],[0,3,.1]],0)
        except ValueError:
            pass
        else:
            raise AssertionError('insufficient pool must fail')
        parallel_graph = ns['make_graph'](deepcopy(parallel))
        effects = ns['exact_edge_effects'](parallel_graph)
        observed = ns['path_report'](parallel_graph,[0,1,2,3],effects)
        assert observed == parallel_expected, 'nonzero corrupt anchors and unequal full-graph path drops'
    elif stage == 'assembly':
        exact=ns['exact_edge_effects'](graph); ig=ns['integrated_scores'](graph,4)
        edges=ns['threshold_graph'](ig,.5); path=ns['top_path'](edges,0,2,2)
        report=ns['path_report'](graph,path,exact); controls=ns['graph_controls'](graph,edges,0)
        out=ns['assemble_result'](exact,ig,edges,path,report,controls)
        assert out['ratios']=={'faithfulness':1.,'completeness':1.,'minimality':1.}
        assert out['margins']=={'random':1.,'reversed':1.} and out['predicate'] is True
        # Independent fabricated boundary fixture, not a target observation.
        report={'metrics':{'full':1.,'corrupt':0.,'path_only':.8,'path_removed':.2},'drops':[.8]}
        controls={'metrics':{'selected':1.,'random':.5,'reversed':.5},'random_edges':[],'reversed_edges':[]}
        boundary=ns['assemble_result'](exact,ig,edges,path,report,controls)
        assert boundary['predicate'] is True
        controls['metrics']['random']=.500001
        assert ns['assemble_result'](exact,ig,edges,path,report,controls)['predicate'] is False
        parallel_graph = ns['make_graph'](deepcopy(parallel))
        effects = ns['exact_edge_effects'](parallel_graph)
        observed = ns['path_report'](parallel_graph,[0,1,2,3],effects)
        assert observed == parallel_expected
        # Scalar assembly-only control fixture. These are not complete graph
        # measurements and are never sealed/exported as an empirical receipt.
        assembly_controls = {'metrics':{'selected':3.,'random':1.,'reversed':0.},
                             'random_edges':[],'reversed_edges':[]}
        assembled = ns['assemble_result'](effects,t.zeros((4,4)),[],[0,1,2,3],observed,assembly_controls)
        expected_ratios = {'faithfulness':7/15,'completeness':16/15,'minimality':4/5}
        assert assembled['ratios'] == expected_ratios, 'one nonzero full-corrupt denominator and MINIMUM path drop required'
        assert assembled['predicate'] is False
        # A distinct actual DEV corruption larger than clean makes D negative.
        # Ratios keep their signs; they are not probabilities to clamp or abs.
        higher_corrupt = deepcopy(parallel); higher_corrupt['corrupt_inputs'][0] = 2.
        negative_graph = ns['make_graph'](higher_corrupt)
        negative_effects = ns['exact_edge_effects'](negative_graph)
        negative_path = ns['path_report'](negative_graph,[0,1,2,3],negative_effects)
        assert negative_path == {'metrics':{'full':5.,'corrupt':10.,'path_only':3.,'path_removed':1.},
                                 'drops':[4.,3.,3.]}
        negative = ns['assemble_result'](negative_effects,t.zeros((4,4)),[],[0,1,2,3],negative_path,assembly_controls)
        assert negative['ratios'] == {'faithfulness':7/5,'completeness':-4/5,'minimality':-3/5}
        assert negative['predicate'] is False
        zero_anchor = deepcopy(parallel_expected)
        zero_anchor['metrics']['corrupt'] = zero_anchor['metrics']['full']
        try:
            ns['assemble_result'](effects,t.zeros((4,4)),[],[0,1,2,3],zero_anchor,assembly_controls)
        except ValueError:
            pass
        else:
            raise AssertionError('zero full-corrupt denominator must fail')
    else:
        raise ValueError('unknown practice stage')
    return 'Distinct exposed DEV '+stage+' fixture passed'


def _stub(name):
    tree=ast.parse(REFERENCE_METHODS)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    args=', '.join(a.arg for a in node.args.args)
    return f'def {name}({args}):\n    raise NotImplementedError("Implement {name}")\n'


def build_cells():
    cells=[]
    def add(kind,name,text,stage):
        item={'cell_type':kind,'id':'graph-'+name,'source':text.strip()+'\n','metadata':{'stage':stage}}
        if kind=='code':
            item.update(outputs=[],execution_count=None)
        cells.append(item)
    add('markdown','intro', '''# [8.7] Predict a graph, then test its interventions
For one predetermined seven-node DAG, **will the recovered path satisfy all three exact
intervention criteria AND will the selected graph beat both graph controls?**
Write a boolean forecast and causal reasoning before any target run or solution reveal.
Confidence means P(event=True): the event is ALL THREE path ratios ≥.8 AND BOTH margins ≥.5.
It is the probability of that full event, even when your boolean forecast is False; no probability
score or calibration claim is computed by this companion. The target remains dormant, requiring separate authority.

Cold open: a strong-looking edge may belong to a redundant route. A graph picture alone
cannot tell you how much the selected path preserves, removes, or individually requires.
You will write the propagation, signed exact/IG scores, directed path search and interventions,
then compare a committed prediction with a stored graph, signed score heatmaps and gate plots.
No target answer, graph, selected path or success narrative is supplied.

Objectives: distinguish edge score from exact causal necessity; recover a path from scores;
keep the clean-corrupt denominator fixed; inspect three path criteria and TWO separate controls.
This is a public-derived fixed synthetic diagnostic. TRAIN: none. DEV: exposed historical
lesson and distinct independent practice fixtures. TEST: none. New weights do not establish
human unseen status or generalization. Exposure options: `unknown`, `seen`,
`self_reported_not_seen`. Equivalent known outputs require `seen` and guided practice.
Human walkthrough and scientific acceptance remain deferred.''','forecast')
    add('code','setup', '''from pathlib import Path
import sys
root = next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'mastery_hub').is_dir())
if str(root) not in sys.path:
    sys.path.insert(0,str(root))
from mastery_hub.units.graph_prediction import lesson
from IPython.display import Code, Markdown, display
import torch as t
t.set_num_threads(1)
t.set_default_dtype(t.float32)
t.set_default_device('cpu')
CASE = dict(lesson.CASE)
print(CASE)  # Raw inputs/protocol only; no target constructor or propagation.
# Existing installed environment only. No install, model, training, network or GPU.''','setup')
    add('markdown','protocol', '''## Freeze the question before solving it
Raw nodes/weights and inputs appear above. Unspecified weights are zero. Process nodes in
the given topological order. The scalar metric is the final `Paris-vs-Rome` activation.
Use 16 midpoint **edge-gate** IG steps: all active masks scale together by `(k+.5)/16`;
average gradients times active support. Select absolute score **≥ .25**, no edge cap.
Sort by descending magnitude, source then target; retain signed scores rounded to six
decimals *before* path search, matching the historical graph representation.
Recover the highest product of absolute selected-edge scores, from source 0 to target 6,
maximum 4 edges. Stable adjacency order breaks ties. No predefined path is supplied.

Let D=M_full−M_corrupt. D=0 is a failed attempt. Compare unrounded values:
faithfulness=(M_path_only−M_corrupt)/D; completeness=(M_full−M_path_removed)/D;
minimality=min(full-graph single-path-edge ablation drops)/D. Each must be **≥ .8**.
The selected graph must separately beat a size-matched random graph and its reversed-edge
graph by **≥ .5** each. Random.Random(0) samples actual nonzero edges, excluding ALL
selected graph edges; this matches edge count only. Reversed edges keep signed scores but
backward edges do not propagate in the fixed topological order. Keep all six control classes:
full/corrupt anchors, path only/removed, every path edge, random, reversed, exact versus IG.
No path or insufficient random pool is a failure to preserve; never shrink or repair controls.
A valid negative predicate is a measured rejection, not a failed implementation.

Expected outputs below are ONLY exposed four-node DEV fixtures. They do not contain the
seven-node target. Core method source is visible and editable; hash plumbing is shared.''','protocol')
    groups=[('graph',('make_graph','activations','scalar_metric')),
            ('scores',('exact_edge_effects','integrated_scores')),
            ('selection',('threshold_graph','top_path')),
            ('interventions',('path_report','graph_controls')),
            ('assembly',('assemble_result',))]
    prose={
      'graph':('1: propagate a linear DAG','Create CPU float32 weights from raw triples and clean/corrupt inputs. Return nodes, weights, clean, corrupt, source, target, max_depth. activations(graph,edge_mask,inputs) adds each weighted earlier source into each target; scalar_metric selects graph["target"]. Do not instantiate CASE.','DEV activations [1,2,1,.375]; corrupt metric 0.','Build a list of scalar activations rather than in-place tensor assignments, preserving autograd. A reversed mask cannot create a backwards physical weight.'),
      'scores':('2: distinguish exact drops from IG','Exact effects are full metric minus each full-graph single-edge ablation. Store float32 signed scores, zero off active support. IG averages fresh edge-mask gradients at midpoint alpha, times active support; retain signs.','DEV two-edge chain exact 1/1 and IG .5/.5; separate signed THREE-edge chain gives IG ±.9375 at 2 steps and ±.984375 at 4 steps, derived from a cubic gate polynomial.','Use autograd.grad on a fresh leaf mask, detach each gradient. This integral follows gates, not node activations.'),
      'selection':('3: recover a directed path','Threshold on absolute IG, inclusive comparison; deterministic sort then six-decimal signed graph scores. Depth-first search maximizes the product of absolute rounded scores, no cycles, stable per-source magnitude order and strict > replacement. Raise on no path.','DEV -.5 and +.5 edges both selected at .5; recovered anonymous path [0,1,2]; empty graph raises.','Build adjacency from selected edges. Search from source; never use a planted or historical path object.'),
      'interventions':('4: confront the selected graph with controls','path_report returns metrics full/corrupt/path_only/path_removed and full-graph single-edge drops for the recovered path. graph_controls returns selected/random/reversed metrics and exact random_edges/reversed_edges. Sample row-major actual nonzero candidates excluding the selected graph. Fail if too few.','DEV baseline path metrics 1,0,1,0 and controls 1,0,0; separate parallel-route fixture has full=5, corrupt=1.25, path-only=3, path-removed=1 and unequal drops [4,3,3]. Oversized random control raises.','Masks switch actual weights on/off; graph score values are not replacement weights. Path removal switches off every recovered path edge together.'),
      'assembly':('5: assemble evidence without rewriting it','Return scores exact/IG16, selected_edges, path, path_drops, metrics, ratios, control_edges, margins, gates and predicate. Compute the three normalized ratios using the one D, and two margins, then inclusive comparisons .8/.5. Never round measurements before gating.','DEV baseline ratios all 1; separate nonzero-corrupt, unequal-drop fixture requires ratios 7/15, 16/15, 4/5. This distinguishes full−corrupt from full and min from max. Boundary .8/.5 is inclusive; zero denominator raises.','Full-graph edge drops measure necessity in the original background, not in a path-only graph. No silent epsilon or dropped control is allowed.')}
    prior=[]
    for stage,names in groups:
        title,task,expected,help_text=prose[stage]
        add('markdown',stage+'-instructions',f'## Exercise {title}\n{task}\n\nExpected independent DEV output: {expected}\n\n<details><summary>Help / interpretation</summary>{help_text}</details>\n\n<details><summary>Solution</summary>Reference definitions are available only after a valid commitment. Use the explicit reveal cell below; adopting edited methods requires a new honest commitment.</details>','exercise')
        source='\n'.join(_stub(name) for name in names)
        prior.append(stage.upper()+'_SOURCE')
        add('code',stage,f"{stage.upper()}_SOURCE = '''\n{source}'''\nmethod_source = '\\n'.join([{', '.join(prior)}])\npractice = lesson.practice_namespace(method_source)\nprint(lesson.practice_test(practice, {stage!r}))",'practice')
    add('markdown', 'dev-forecast', '''## Exposed DEV ladder — a path and a bypass (10–15 minutes)
**Implementation preparation + complete exposed DEV cycle. The seven-node target remains dormant.**
Hand-trace the separate four-node DAG: 0→1 weight2, 1→2 weight.5, 2→3 weight3,
plus 1→3 weight.5. Clean input at node0 is1; corrupt input is.25. The long path contributes3,
the shortcut1, so full=4 and corrupt=1. Hand-predict path-only, path-removed and
faithfulness using the one full−corrupt denominator. Then add ONLY direct 0→3 weight1.
Choose **A:** unchanged path weights imply unchanged path faithfulness; **B:** a new bypass
changes the anchors and normalized preservation even if the path itself is unchanged.
What paired observation rules A out? Keep raw metrics beside ratios.
This is public DEV, not the target predicate or an authorized target attempt.''', 'dev')
    add('code', 'dev-forecast-card', '''dev_forecast_faithfulness=2/3  # Original four-node DEV, not target.
dev_explanation='B'
dev_disconfirming_observation='Path-only stays 3 but its normalized preservation changes.' ''', 'dev')
    add('code', 'dev-paired', r'''from copy import deepcopy
import matplotlib.pyplot as plt
dev_source='\n'.join([GRAPH_SOURCE,SCORES_SOURCE,SELECTION_SOURCE,INTERVENTIONS_SOURCE,ASSEMBLY_SOURCE])
dev_ns=lesson.practice_namespace(dev_source)
dev_case={'node_order':['DEV input','DEV a','DEV b','DEV output'],
          'edge_weights':[[0,1,2.],[1,2,.5],[2,3,3.],[1,3,.5]],
          'clean_inputs':[1.,0.,0.,0.],'corrupt_inputs':[.25,0.,0.,0.],
          'source_index':0,'target_index':3,'max_depth':3}
dev_bypass=deepcopy(dev_case);dev_bypass['edge_weights'].append([0,3,1.])
dev_reports={}
for name,case in [('Original',dev_case),('Direct bypass',dev_bypass)]:
    graph=dev_ns['make_graph'](case)
    effects=dev_ns['exact_edge_effects'](graph)
    dev_reports[name]=dev_ns['path_report'](graph,[0,1,2,3],effects)
def dev_ratios(report):
    m=report['metrics'];denominator=m['full']-m['corrupt']
    return [(m['path_only']-m['corrupt'])/denominator,
            (m['full']-m['path_removed'])/denominator,min(report['drops'])/denominator]
fig,axes=plt.subplots(1,2,figsize=(12,3),layout='constrained')
for name,report in dev_reports.items():
    axes[0].plot(['full','corrupt','path_only','path_removed'],[report['metrics'][k] for k in ('full','corrupt','path_only','path_removed')],'o-',label=name)
    axes[1].plot(['faithfulness','completeness','minimality'],dev_ratios(report),'o-',label=name)
axes[0].set(ylabel='Scalar output',title='DEV anchors and path-removal control')
axes[1].set(ylabel='Effect / (full−corrupt)',title='DEV path criteria')
axes[1].axhline(.8,color='gray',linestyle='--',label='Target-style path gate, reference only')
for ax in axes:ax.legend()
display(fig)
print('DEV ratios:',{name:dev_ratios(report) for name,report in dev_reports.items()})
print('DEV faithfulness absolute forecast error:',abs(dev_ratios(dev_reports['Original'])[0]-dev_forecast_faithfulness))
assert dev_reports['Original']['metrics']['path_only']==dev_reports['Direct bypass']['metrics']['path_only']==3.''', 'dev')
    add('markdown', 'dev-reflection', '''### Worked DEV reflection / self-check
Original full/corrupt/path-only/path-removed are 4,1,3,0; ratios are 2/3,4/3,1.
With the direct bypass they become 5,1.25,3,1; ratios are 7/15,16/15,4/5.
A forecast of unchanged faithfulness 2/3 errs by .2. Unchanged path-only but changed
faithfulness rules A out. Completeness can exceed1 with these corrupt anchors: it is a ratio,
not a probability. A good explanation cites both anchors, the bypass that survives removal,
and the minimum single-edge drop; it does not infer the five-part target predicate from this example.

### Try It Yourself — changed-condition DEV transfer
Keep the bypass graph fixed, change ONLY corrupt node0 input .25→0. Predict all three ratios
before running. Revised explanation B predicts [3/5,4/5,3/5]. Observe whether the raw clean/path
metrics stay fixed while the denominator changes. No target graph or target API is used.''', 'dev')
    add('code', 'dev-transfer', '''dev_transfer_case=deepcopy(dev_bypass);dev_transfer_case['corrupt_inputs'][0]=0.
dev_transfer_forecast=[3/5,4/5,3/5]
dev_transfer_graph=dev_ns['make_graph'](dev_transfer_case)
dev_transfer_report=dev_ns['path_report'](dev_transfer_graph,[0,1,2,3],dev_ns['exact_edge_effects'](dev_transfer_graph))
dev_transfer_observed=dev_ratios(dev_transfer_report)
print('DEV transfer absolute forecast errors:',[abs(actual-predicted) for actual,predicted in zip(dev_transfer_observed,dev_transfer_forecast)])
assert dev_transfer_observed==[3/5,4/5,3/5]  # Method check, independent of forecast.
fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
ax.plot(['faithfulness','completeness','minimality'],dev_ratios(dev_reports['Direct bypass']),'o-',label='Corrupt input .25')
ax.plot(['faithfulness','completeness','minimality'],dev_transfer_observed,'o-',label='Corrupt input 0')
ax.axhline(.8,color='gray',linestyle='--',label='Path gate, reference only')
ax.set(ylabel='Ratio',title='Observed DEV anchor transfer');ax.legend();display(fig)
print('DEV changed-anchor ratios:',dev_transfer_observed)
dev_revision=''  # Cite the changed denominator and unchanged clean/path measurements.''', 'dev')
    add('markdown','freeze-help','''## Commit before target execution or reference reveal
Freezing reads raw literal float32 bytes, exact student source, helper/tests/protocol files,
node order and installed runtime versions. It never creates or evaluates the target graph.
Fill your own forecast and reasoning. A missing forecast cannot create a valid commitment.
This JSON text is immutable: preserve it along with all failed or completed attempts.
Editing methods, inputs, protocol, tests or runtime invalidates that commitment.
A local hash is not an exam-security mechanism or a certificate of human prior exposure.''','freeze')
    add('code','freeze', '''# Reassemble after editing any exercise cell; no method is executed here.
method_source = '\\n'.join([GRAPH_SOURCE,SCORES_SOURCE,SELECTION_SOURCE,INTERVENTIONS_SOURCE,ASSEMBLY_SOURCE])
identity = lesson.freeze_inputs(method_source)''','freeze')
    add('code','commit', '''prediction = None  # Your bool: all THREE path gates AND BOTH control margins?
confidence = None  # P(the full five-part conjunction is TRUE), in [0,1].
reasoning = ''  # Explain the denominator, alternate routes, and both controls.
prior_exposure = 'unknown'
commitment = lesson.commit_prediction(identity,method_source,prediction,confidence,reasoning,prior_exposure)
print('Commitment created. Preserve its exact JSON text before continuing.')''','commit')
    add('code','reveal', '''# OPTIONAL postcommit reveal. This displays generic definitions, no target result.
# Opening solutions is exposure; edited methods require a NEW commitment.
# reference = lesson.reveal_methods(commitment)
# display(Markdown('<details><summary>Postcommit solution definitions</summary>\\n\\n```python\\n'+reference+'\\n```\\n</details>'))''','reveal')
    add('markdown','run-help','''## Optional future target attempt — separate authorization required
The blank artifact does not authorize or run a target experiment. If later authorized,
allow one CPU float32 thread, one attempt with aggregate 120-second ceiling and an external
timeout. No search, weight/seed/threshold changes, repair or retry after seeing an outcome.
Save the returned receipt even when status is `failed`; inspect its error stage and preserve it.
A hard external timeout may prevent a return: preserve the original commitment and external
failure evidence. This notebook cannot enforce a hostile process or certify execution authority.''','run-instructions')
    add('code','run', '''receipt = None
# Only after separate target authorization, explicitly uncomment ONE call:
# receipt = lesson.run_experiment(commitment,method_source,authorized=True)
# status = lesson.read_record(receipt)['status']
# print(status)
# if status == 'observed':
#     display(lesson.plot_receipt(receipt))
# else:
#     print(lesson.read_record(receipt)['failure'])''','run')
    add('markdown','interpret','''## Read the stored result
The selected graph and recovered path show *what was selected*. Signed exact-versus-IG
heatmaps show how full-graph necessity and midpoint gate attribution differ. The three
ratio bars test the same denominator; two margin bars retain distinct controls.
Read full precision stored values for gates; plotted labels are rounded for display.
A plotted graph is not a causal proof. Controls may pass or fail. A measured rejection
is useful; do not adjust the organism to make the intended story succeed.''','reflect')
    add('code','reflection', '''observation = None  # Your observed predicate, only after inspecting a valid receipt.
reflection = ''  # Explain every ratio/margin and any surprise; cite stored measurements.
transfer_question = ''  # Name a changed condition and fixed denominator; no execution here.
transfer_prediction = None
transfer_confidence = None
transfer_reasoning = ''
# Try It Yourself: inspect stored values for a chosen pair without rerunning any graph.
# u,v = 0,1
# obs = lesson.replay(receipt)
# print({name:scores[u][v] for name,scores in obs['scores'].items()})
# print(obs['ratios'],obs['margins'])''','play')
    add('markdown','export-help','''## Keep the attempt private and replay without execution
Export is opt-in, private and exclusive. A stored observation can be plotted in a new
session without torch, source-method execution, current runtime or a new target run.
Keep the original commitment, every returned failure and measured receipt; never overwrite.
Reflection stays learner-authored in this notebook. A changed-condition forecast remains
unexecuted until a new frozen case and separately authorized attempt are agreed.''','export')
    add('code','export', '''# Explicit opt-in private exports; choose NEW paths, never overwrite:
# lesson.export_record(commitment,Path.home()/'graph-prediction-commitment-1.json')
# lesson.export_record(receipt,Path.home()/'graph-prediction-attempt-1.json')
# Read-only stored replay in a later session:
# saved = Path.home().joinpath('graph-prediction-attempt-1.json').read_text()
# display(lesson.plot_receipt(saved))''','export')
    add('markdown','limits','''## Limits, model connection and anomaly hunting
This is one hand-constructed linear component DAG, with fixed positive weights and symbolic
feature/head/MLP labels. It is not a pretrained transformer or learned-transcoder result.
A size-only random graph ignores degree/semantic matching; reversal is structurally harsh
because only topologically forward weights propagate. D can be small or negative in other
organisms; never reinterpret a ratio without its anchors. Midpoint IG is an approximation.
The frozen single candidate was not chosen using its outcome. No independent TEST,
calibration, human-unseen guarantee, mastery badge or course-ready claim follows.

Attribution graphs in real circuit tracing require learned features, replacement-model
fidelity, frozen attention/error handling and prompt generalization. These are open here.
After commitment, connect the method to the existing published 8.7 lesson and
[Anthropic's circuit tracing methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).
Future anomaly work could ask whether redundancy or a different corruption changes the
three ratios; propose the exact new case and forecast first. The minimal play cell only
inspects stored evidence. Human learner walkthrough and scientific adoption remain pending.''','limitations')
    return cells
