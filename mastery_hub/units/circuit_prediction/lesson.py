"""Blind compiled-circuit prediction: local identity, commitments, and receipt replay.

The notebook owns compiler, scoring, AUC, and result assembly. No model is loaded
or run at import, commitment, solution reveal, export, or replay. Local seals detect
accidental edits; they do not constitute exam security or certify prior exposure.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / 'chapter8_automated_circuits/exercises/part8_circuit_benchmarks/utils.py'
SCHEMA = 'arena-blind-circuit-prediction'
VERSION = 2
CASE = {
    'id': '8.8-unembedding-gain-v1', 'gains': [10.0, 1.0], 'sharpness': 20.0,
    'sigma': 0.03, 'noise_seed': 0, 'data_seed': 0, 'n': 64, 'vocab': 16,
    'half_len': 8, 'n_layers': 2, 'n_heads': 3, 'd_head': 17, 'd_model': 81,
    'dtype': 'torch.float32', 'token_dtype': 'torch.int64', 'device': 'cpu',
    'threads': 1, 'ig_steps': 10,
    'metric': 'mean clean-next-token logprob, positions 9:16, target positions 10:17',
    'ranking': 'absolute edge score; positive-negative ties count 0.5',
    'forecast': 'EAP planted-edge ROC AUC(gain=1) minus ROC AUC(gain=10)',
    'control': 'gain=1, sigma=0; otherwise same inputs',
    'data_role': 'fixed synthetic diagnostic; no independent TEST claim',
    'evidence_class': 'compiled transformer exploratory intervention',
    'budget_seconds': 120,
}
PLANTED = [('input', '0.0<v>'), ('input', '0.1<v>'), ('0.0', '1.0<k>'),
           ('0.1', '1.0<k>'), ('input', '1.0<q>'), ('input', '1.0<v>'), ('1.0', 'logits')]
METHOD_NAMES = ('compile_select_aggregate', 'compile_induction', 'add_weight_noise',
                'patching_scores', 'eap_scores', 'eap_ig_scores', 'roc_auc', 'assemble_result')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seal(payload):
    out = deepcopy(payload)
    out['seal'] = digest(payload)
    return json.dumps(out, sort_keys=True, allow_nan=False)


def read_record(text):
    if type(text) is not str or len(text.encode()) > 5_000_000:
        raise ValueError('expected bounded JSON text')
    out = json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if type(out) is not dict or 'seal' not in out:
        raise ValueError('missing sealed record')
    seal = out.pop('seal')
    if (digest(out) != seal or out.get('schema') != SCHEMA
            or type(out.get('schema_version')) is not int or out.get('schema_version') not in (1, VERSION)):
        raise ValueError('invalid seal or unsupported schema')
    required = {'schema','schema_version','case','identity','method_source','prediction',
                'confidence','reasoning','prior_exposure','assessment','status'}
    if out.get('status') == 'observed':
        required |= {'observation','commitment_sha256'}
    if set(out) != required:
        raise ValueError('missing or unknown record fields')
    if digest(out.get('case')) != digest(CASE) or out.get('status') not in ('committed', 'observed'):
        raise ValueError('unsupported case or status')
    if out.get('prior_exposure') not in ('unknown', 'seen', 'self_reported_not_seen'):
        raise ValueError('invalid exposure')
    if out.get('assessment') != 'local_self_report_no_unseen_certification':
        raise ValueError('unsupported assessment claim')
    if not isinstance(out.get('identity'), dict) or not isinstance(out.get('method_source'), str):
        raise ValueError('missing method identity')
    p, c = out.get('prediction'), out.get('confidence')
    if (type(p) not in (int, float) or not math.isfinite(p) or not -1 <= p <= 1
            or type(c) not in (int, float) or not math.isfinite(c) or not 0 <= c <= 1
            or not isinstance(out.get('reasoning'), str) or not out['reasoning'].strip()):
        raise ValueError('invalid forecast')
    if out['identity'].get('case_sha256') != digest(CASE):
        raise ValueError('case identity mismatch')
    if out['identity'].get('method_sha256') != hashlib.sha256(out['method_source'].encode()).hexdigest():
        raise ValueError('method identity mismatch')
    _validate_identity(out['identity'],out['schema_version'])
    if out['status'] == 'observed':
        _validate_observation(out)
    return out



def _unique_pairs(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _sha(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('expected lowercase SHA256')


def _finite(value, lower=None, upper=None):
    if (type(value) not in (int,float) or not math.isfinite(value)
            or (lower is not None and value < lower) or (upper is not None and value > upper)):
        raise ValueError('invalid finite numeric observation')


def _validate_identity(identity, version=VERSION):
    keys = {'case_sha256','method_sha256','helper_sha256','plumbing_sha256','environment',
            'tensors','ordered_model_states','sender_order','receiver_order'}
    if version >= 2:
        keys.add('model_metadata')
    if type(identity) is not dict or set(identity) != keys:
        raise ValueError('missing or unknown identity fields')
    for key in ('case_sha256','method_sha256','helper_sha256','plumbing_sha256'):
        _sha(identity[key])
    if type(identity['environment']) is not dict or not identity['environment']:
        raise ValueError('missing environment identity')
    senders = ['input']+[f'{l}.{h}' for l in range(2) for h in range(3)]
    receivers = [f'{l}.{h}<{c}>' for l in range(2) for c in 'qkv' for h in range(3)]+['logits']
    if identity['sender_order'] != senders or identity['receiver_order'] != receivers:
        raise ValueError('sender / receiver order mismatch')
    def tensor(item):
        if type(item) is not dict or set(item) != {'shape','dtype','sha256'}:
            raise ValueError('malformed tensor identity')
        if (type(item['shape']) is not list or any(type(v) is not int or v < 0 for v in item['shape'])
                or type(item['dtype']) is not str or not item['dtype'].startswith('torch.')):
            raise ValueError('malformed tensor dtype or shape')
        _sha(item['sha256'])
    expected = {'clean':([64,17],'torch.int64'), 'corrupt':([64,17],'torch.int64'),
                'valid_edges':([7,19],'torch.bool'), 'planted_edges':([7,19],'torch.bool')}
    if type(identity['tensors']) is not dict or set(identity['tensors']) != set(expected):
        raise ValueError('missing input tensor')
    for key,(shape,dtype) in expected.items():
        item=identity['tensors'][key]
        tensor(item)
        if item['shape'] != shape or item['dtype'] != dtype:
            raise ValueError('input tensor shape / dtype mismatch')
    if type(identity['ordered_model_states']) is not dict or set(identity['ordered_model_states']) != {'gain10','gain1','gain1_no_noise'}:
        raise ValueError('missing model state')
    if version >= 2:
        metadata=identity['model_metadata']
        if type(metadata) is not dict or set(metadata) != set(identity['ordered_model_states']):
            raise ValueError('missing ordered model configuration metadata')
        for modules in metadata.values():
            if type(modules) is not list or not modules:
                raise ValueError('missing ordered module metadata')
            names=[]
            for module in modules:
                if (type(module) is not dict or set(module) != {'name','training','config'}
                        or type(module['name']) is not str or type(module['training']) is not bool
                        or (module['config'] is not None and type(module['config']) is not dict)):
                    raise ValueError('invalid module configuration metadata')
                names.append(module['name'])
            if modules[0]['name'] != '' or not modules[0]['config'] or len(set(names))!=len(names):
                raise ValueError('missing root configuration or duplicate module name')
    for state in identity['ordered_model_states'].values():
        if type(state) is not list or not state:
            raise ValueError('empty ordered model state')
        names=[]
        for item in state:
            if type(item) is not dict or set(item) != {'name','shape','dtype','sha256'} or type(item['name']) is not str:
                raise ValueError('malformed state entry')
            tensor({k:v for k,v in item.items() if k != 'name'})
            names.append(item['name'])
        layout=[(item['name'],item['shape'],item['dtype']) for item in state]
        if layout != state_layout():
            raise ValueError('model state shape / dtype / order mismatch')
        if len(set(names)) != len(names):
            raise ValueError('duplicate model state name')



def state_layout():
    """Versioned layout for the fixed local TransformerLens attention-only model."""
    out=[('embed.W_E',[16,81],'torch.float32'),('pos_embed.W_pos',[17,81],'torch.float32')]
    for layer in range(2):
        for name,shape,dtype in [('W_Q',[3,81,17],'torch.float32'),('W_O',[3,17,81],'torch.float32'),
            ('b_Q',[3,17],'torch.float32'),('b_O',[81],'torch.float32'),('W_K',[3,81,17],'torch.float32'),
            ('W_V',[3,81,17],'torch.float32'),('b_K',[3,17],'torch.float32'),('b_V',[3,17],'torch.float32'),
            ('mask',[17,17],'torch.bool'),('IGNORE',[],'torch.float32')]:
            out.append((f'blocks.{layer}.attn.{name}',shape,dtype))
    return out+[('unembed.W_U',[81,16],'torch.float32'),('unembed.b_U',[16],'torch.float32')]

def _validate_observation(record):
    obs=record['observation']
    expected={'kind','results','delta_eap_auc','elapsed_seconds','sender_order','receiver_order',
              'valid_mask','planted_mask','absolute_forecast_error'}
    if type(obs) is not dict or set(obs) != expected or obs['kind'] != 'fresh_cpu_inference':
        raise ValueError('malformed observation')
    _sha(record['commitment_sha256'])
    committed = {k:v for k,v in record.items() if k not in ('observation','commitment_sha256')}
    committed['status'] = 'committed'
    if hashlib.sha256(_seal(committed).encode()).hexdigest() != record['commitment_sha256']:
        raise ValueError('receipt does not link to committed parent')
    for key in ('sender_order','receiver_order'):
        if obs[key] != record['identity'][key]:
            raise ValueError('observation edge order mismatch')
    for key in ('valid_mask','planted_mask'):
        mask=obs[key]
        if (type(mask) is not list or len(mask)!=7 or any(type(row) is not list or len(row)!=19
                or any(type(v) is not bool for v in row) for row in mask)):
            raise ValueError('invalid edge mask')
    for mask_key,identity_key in [('valid_mask','valid_edges'),('planted_mask','planted_edges')]:
        mask_hash=hashlib.sha256(bytes(int(v) for row in obs[mask_key] for v in row)).hexdigest()
        if mask_hash != record['identity']['tensors'][identity_key]['sha256']:
            raise ValueError('observation mask differs from frozen identity')
    if any(p and not v for pr,vr in zip(obs['planted_mask'],obs['valid_mask']) for p,v in zip(pr,vr)):
        raise ValueError('planted edge must be valid')
    if sum(sum(row) for row in obs['planted_mask']) != 7:
        raise ValueError('planted mask changed')
    if type(obs['results']) is not dict or set(obs['results']) != {'gain10','gain1','gain1_no_noise'}:
        raise ValueError('missing paired condition or control')
    for result in obs['results'].values():
        if type(result) is not dict or set(result) != {'auc','scores','task_accuracy','answer_logprob'}:
            raise ValueError('malformed condition result')
        if (type(result['auc']) is not dict or type(result['scores']) is not dict
                or set(result['auc']) != {'exact','EAP','IG10'} or set(result['scores']) != {'exact','EAP','IG10'}):
            raise ValueError('missing scoring method')
        _finite(result['task_accuracy'],0,1)
        _finite(result['answer_logprob'],upper=0)
        for auc in result['auc'].values():
            _finite(auc,0,1)
        for method,scores in result['scores'].items():
            if type(scores) is not list or len(scores)!=7 or any(type(row) is not list or len(row)!=19 for row in scores):
                raise ValueError('invalid edge-score matrix')
            for row in scores:
                for value in row:
                    _finite(value)
            positive=[abs(scores[u][v]) for u in range(7) for v in range(19) if obs['valid_mask'][u][v] and obs['planted_mask'][u][v]]
            negative=[abs(scores[u][v]) for u in range(7) for v in range(19) if obs['valid_mask'][u][v] and not obs['planted_mask'][u][v]]
            if not positive or not negative:
                raise ValueError('AUC requires both classes')
            auc=sum((p>n)+.5*(p==n) for p in positive for n in negative)/(len(positive)*len(negative))
            if abs(auc-result['auc'][method])>1e-6:
                raise ValueError('reported AUC inconsistent with stored edge scores')
    _finite(obs['elapsed_seconds'],0,CASE['budget_seconds'])
    delta=obs['results']['gain1']['auc']['EAP']-obs['results']['gain10']['auc']['EAP']
    _finite(obs['delta_eap_auc'],-1,1)
    _finite(obs['absolute_forecast_error'],0,2)
    if abs(delta-obs['delta_eap_auc'])>1e-12 or abs(abs(delta-record['prediction'])-obs['absolute_forecast_error'])>1e-12:
        raise ValueError('inconsistent forecast result')

def commit_prediction(identity, method_source, prediction, confidence, reasoning, prior_exposure='unknown'):
    """Return immutable JSON text; never mutate a previous attempt or run a model."""
    text = _seal({'schema': SCHEMA, 'schema_version': VERSION, 'case': deepcopy(CASE),
        'identity': deepcopy(identity), 'method_source': method_source,
        'prediction': prediction, 'confidence': confidence, 'reasoning': reasoning,
        'prior_exposure': prior_exposure,
        'assessment': 'local_self_report_no_unseen_certification', 'status': 'committed'})
    read_record(text)
    return text


def _plumbing():
    spec = importlib.util.spec_from_file_location('blind_circuit_local_plumbing', UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def namespace(method_source):
    """Execute learner-authored definitions in a fresh namespace, with local plumbing."""
    import torch as t
    ns = {'t': t, 'utils': _plumbing(), 'CASE': deepcopy(CASE)}
    exec(compile(method_source, '<learner methods>', 'exec'), ns)
    if not all(callable(ns.get(name)) for name in METHOD_NAMES):
        raise ValueError('implement every named core method')
    return ns


def tensor_identity(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}


def environment_identity():
    import torch
    import numpy
    import transformer_lens
    from importlib.metadata import version
    return {'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': numpy.__version__, 'transformer_lens': version('transformer-lens'),
        'platform': platform.platform(), 'threads': torch.get_num_threads(),
        'default_dtype': str(torch.get_default_dtype()),
        'default_device': str(torch.get_default_device()),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def prepare(method_source):
    """Compile weights and hash inputs without a forward pass or attribution run.

    This deliberately recomputes a fresh state before each expensive run. Frozen
    values cannot silently change with edited cells, helper code, dtype, or runtime.
    """
    import torch as t
    if (t.get_num_threads() != CASE['threads'] or str(t.get_default_dtype()) != CASE['dtype']
            or str(t.get_default_device()) != 'cpu'):
        raise ValueError('set one CPU thread, float32 and CPU default device before freezing')
    ns = namespace(method_source)
    utils = ns['utils']
    clean, corrupt = utils.make_induction_data(CASE['n'], CASE['vocab'], CASE['half_len'],
                                               t.Generator().manual_seed(CASE['data_seed']), 'cpu')
    models, graphs = {}, {}
    states = {}
    for label, gain, sigma in [('gain10', 10., .03), ('gain1', 1., .03), ('gain1_no_noise', 1., 0.)]:
        model = ns['add_weight_noise'](ns['compile_induction'](gain), sigma, CASE['noise_seed'])
        if any(p.device.type != 'cpu' or p.dtype != t.float32 for p in model.parameters()):
            raise ValueError('case requires float32 CPU weights')
        graph = utils.EdgeGraph(model)
        models[label], graphs[label] = model, graph
        states[label] = [{'name': name, **tensor_identity(value)} for name, value in model.state_dict().items()]
    graph = graphs['gain10']
    identity = {'case_sha256': digest(CASE), 'method_sha256': hashlib.sha256(method_source.encode()).hexdigest(),
        'helper_sha256': _file(__file__), 'plumbing_sha256': _file(UTILS),
        'environment': environment_identity(),
        'tensors': {'clean': tensor_identity(clean), 'corrupt': tensor_identity(corrupt),
                    'valid_edges': tensor_identity(graph.edge_mask),
                    'planted_edges': tensor_identity(graph.mask_from_edges(PLANTED))},
        'ordered_model_states': states,
        'model_metadata': {label:_model_metadata(model) for label,model in models.items()},
        'sender_order': graph.sender_names, 'receiver_order': graph.receiver_names}
    return identity, ns, models, graphs, clean, corrupt


def freeze_inputs(method_source):
    return prepare(method_source)[0]


def assert_current(record_text, method_source):
    record = read_record(record_text)
    if record['schema_version'] != VERSION:
        raise ValueError('historical commitment is replay-only; freeze a new current-version forecast')
    if record['status'] != 'committed':
        raise ValueError('this attempt already has an observation; replay it or make a new forecast')
    # Cheap staleness checks happen before any model compilation.
    identity = record['identity']
    if (method_source != record['method_source'] or identity.get('helper_sha256') != _file(__file__)
            or identity.get('plumbing_sha256') != _file(UTILS)
            or identity.get('environment') != environment_identity()):
        raise ValueError('stale method, helper, or environment; make a new commitment')
    prepared = prepare(method_source)
    if prepared[0] != identity:
        raise ValueError('stale tensors, masks, or model state; make a new commitment')
    return record, prepared




def _configuration_value(value):
    """Canonical JSON for local forward configuration, including dtype/device."""
    import torch as t
    import numpy as np
    if isinstance(value,np.generic):
        return {'python_type':type(value).__module__+'.'+type(value).__qualname__,
                'value':_configuration_value(value.item())}
    if value is None or type(value) in (str,bool,int,float):
        digest(value)
        return value
    if isinstance(value,(t.dtype,t.device)):
        return {'python_type':type(value).__name__,'value':str(value)}
    if type(value) is dict and all(type(key) is str for key in value):
        return {key:_configuration_value(item) for key,item in value.items()}
    if type(value) in (list,tuple):
        return [_configuration_value(item) for item in value]
    raise ValueError('unsupported forward configuration value')


def _model_metadata(model):
    metadata=[]
    for name,module in model.named_modules():
        cfg=getattr(module,'cfg',None)
        metadata.append({'name':name,'training':bool(module.training),
            'config':None if cfg is None else _configuration_value(cfg.to_dict())})
    return metadata

def _runtime_snapshot(models, graphs, clean, corrupt, corrupt_outputs):
    """Capture mutable scientific inputs; this is binding, not an execution sandbox."""
    snapshot = {'environment':environment_identity(),
                'clean': tensor_identity(clean), 'corrupt': tensor_identity(corrupt),
                'models': {}, 'graphs': {}, 'corrupt_outputs': {}}
    if set(models) != set(graphs):
        raise ValueError('runtime model / graph conditions changed')
    for label, model in models.items():
        graph = graphs[label]
        if graph.model is not model:
            raise ValueError('runtime graph model changed')
        snapshot['models'][label] = {
            'object': id(model), 'metadata':_model_metadata(model),
            'state': [{'name': name, **tensor_identity(value)} for name,value in model.state_dict().items()]}
        snapshot['graphs'][label] = {
            'object': id(graph), 'sender_order': list(graph.sender_names),
            'receiver_order': list(graph.receiver_names),
            'valid_edges': tensor_identity(graph.edge_mask),
            'planted_edges': tensor_identity(graph.mask_from_edges(PLANTED)),
            'n_layers': graph.n_layers, 'n_heads': graph.n_heads, 'logits': graph.logits}
    for label,value in corrupt_outputs.items():
        snapshot['corrupt_outputs'][label] = tensor_identity(value)
    return snapshot


def _capture_scores(value, valid):
    import torch as t
    if (not isinstance(value,t.Tensor) or value.shape != valid.shape
            or not value.is_floating_point() or not t.isfinite(value).all()
            or (value.detach().cpu()[~valid.detach().cpu()] != 0).any()):
        raise ValueError('scorer must return finite floating edge scores with invalid edges zero')
    return value.detach().cpu().clone()


def _bind_assembly(result, scores, logits, clean):
    """Check the visible student's assembly against actual captured outputs."""
    if type(result) is not dict or set(result) != {'auc','scores','task_accuracy','answer_logprob'}:
        raise ValueError('malformed student result assembly')
    digest(result)  # finite JSON-native output only
    expected_scores = {name:value.tolist() for name,value in scores.items()}
    if result['scores'] != expected_scores:
        raise ValueError('assembled scores differ from captured scorer outputs')
    targets = clean[:,10:17,None]
    selected = logits[:,9:16]
    accuracy = (selected.argmax(-1) == targets[...,0]).float().mean().item()
    logprob = selected.log_softmax(-1).gather(-1,targets).mean().item()
    for key,expected in [('task_accuracy',accuracy),('answer_logprob',logprob)]:
        _finite(result[key])
        if abs(result[key]-expected)>1e-7:
            raise ValueError('assembled task behavior differs from captured logits')
    # AUC is checked against these exact score arrays by the receipt validator.
    return deepcopy(result)

def run_experiment(record_text, method_source):
    """Explicit fresh CPU run only; no reveal/replay path calls this function."""
    started = time.monotonic()
    def deadline():
        if time.monotonic()-started > CASE['budget_seconds']:
            raise TimeoutError('cooperative CPU budget exceeded; no successful receipt')
    record, prepared = assert_current(record_text, method_source)
    deadline()
    _, ns, models, graphs, clean, corrupt = prepared
    results = {}
    import torch as t
    corrupt_outputs = {}
    frozen = _runtime_snapshot(models, graphs, clean, corrupt, corrupt_outputs)
    def unchanged():
        if _runtime_snapshot(models, graphs, clean, corrupt, corrupt_outputs) != frozen:
            raise ValueError('student call mutated frozen runtime inputs, graph, cache, or model state')
    for label, model in models.items():
        graph = graphs[label]
        unchanged()
        corrupt_out = graph.sender_outputs(corrupt)
        unchanged()
        corrupt_outputs[label] = corrupt_out
        # Adding a newly computed stock sender cache is expected; all later
        # student calls must preserve its exact bytes along with frozen inputs.
        frozen = _runtime_snapshot(models, graphs, clean, corrupt, corrupt_outputs)
        targets = clean[:, 10:17, None]
        def metric(logits):
            return logits[:, 9:16].log_softmax(-1).gather(-1, targets).mean()
        scores = {}
        for name, fn in [('exact', ns['patching_scores']), ('EAP', ns['eap_scores']), ('IG10', ns['eap_ig_scores'])]:
            deadline()
            unchanged()
            value = fn(graph, clean, corrupt_out, metric)
            deadline()
            unchanged()
            scores[name] = _capture_scores(value, graph.edge_mask)
        unchanged()
        with t.no_grad():
            logits = model(clean).detach().cpu().clone()
        deadline()
        unchanged()
        captured_clean = clean.detach().cpu().clone()
        # Assembly gets disposable score/logit copies. Captured measurements and
        # earlier condition tables cannot be changed through returned aliases.
        assembled = ns['assemble_result'](
            {name:value.clone() for name,value in scores.items()}, graph,
            graph.mask_from_edges(PLANTED).clone(), logits.clone(), clean)
        deadline()
        unchanged()
        results[label] = _bind_assembly(assembled, scores, logits, captured_clean)
    unchanged()
    observation = {'kind': 'fresh_cpu_inference', 'results': results,
        'delta_eap_auc': results['gain1']['auc']['EAP'] - results['gain10']['auc']['EAP'],
        'elapsed_seconds': time.monotonic() - started,
        'sender_order': graphs['gain10'].sender_names, 'receiver_order': graphs['gain10'].receiver_names,
        'valid_mask': graphs['gain10'].edge_mask.tolist(),
        'planted_mask': graphs['gain10'].mask_from_edges(PLANTED).tolist()}
    observation['absolute_forecast_error'] = abs(observation['delta_eap_auc'] - record['prediction'])
    digest(observation)  # rejects NaN / nonportable output before recording
    record['status'] = 'observed'
    record['observation'] = observation
    committed = {k:v for k,v in record.items() if k not in ('observation','commitment_sha256')}
    committed['status'] = 'committed'
    record['commitment_sha256'] = hashlib.sha256(_seal(committed).encode()).hexdigest()
    result = _seal(record)
    read_record(result)
    deadline()
    return result


def replay(receipt_text):
    """Read only stored values. No model, method evaluation, or environment import."""
    record = read_record(receipt_text)
    if record['status'] != 'observed':
        raise ValueError('no observation to replay')
    return deepcopy(record['observation'])


def plot_receipt(receipt_text):
    """Recreate the paired AUC plot and planted-edge effect heatmap from a receipt."""
    import matplotlib.pyplot as plt
    import numpy as np
    obs = replay(receipt_text)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout='constrained')
    for method in ['exact', 'EAP', 'IG10']:
        axes[0].plot([10, 1], [obs['results'][k]['auc'][method] for k in ['gain10', 'gain1']], 'o-', label=method)
        axes[1].scatter(method, obs['results']['gain1_no_noise']['auc'][method])
    for ax in axes[:2]:
        ax.axhline(.5, color='gray', ls='--', label='random ranking expectation')
        ax.set_ylim(-.02, 1.02)
        ax.set_ylabel('Absolute-score ROC AUC vs planted structure')
    axes[0].set(xlabel='Unembedding gain', title='Paired noisy conditions', xticks=[10, 1])
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=7)
    axes[1].set_title('No-noise gain1 control')
    planted = np.asarray(obs['planted_mask'], dtype=bool)
    locations = list(zip(*np.where(planted)))
    values = [[obs['results'][label]['scores'][method][u][v]
               for u, v in locations] for label in ['gain10', 'gain1'] for method in ['exact', 'EAP', 'IG10']]
    vmax = max(np.abs(values).max(), 1e-12)
    im = axes[2].imshow(values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[2].set_xticks(range(len(locations)), [f'{obs["sender_order"][u]}→{obs["receiver_order"][v]}' for u,v in locations], rotation=90, fontsize=7)
    axes[2].set_yticks(range(6), [f'{label}/{method}' for label in ['gain10', 'gain1'] for method in ['exact', 'EAP', 'IG10']], fontsize=7)
    axes[2].set_title('Signed effects on planted edges')
    fig.colorbar(im, ax=axes[2], label='Change / derivative of mean answer logprob')
    return fig


def export_record(record_text, destination):
    """Explicit opt-in private export; exclusive creation prevents overwrites."""
    read_record(record_text)
    path = Path(destination)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(record_text)
    return path


ATTEMPT_SCHEMA = SCHEMA + '-reflection'


def reflect(receipt_text, explanation, transfer_question, transfer_prediction, confidence, reasoning):
    """Append a portable reflection envelope; preserve the original sealed receipt."""
    replay(receipt_text)
    result = _seal({'schema':ATTEMPT_SCHEMA,'schema_version':1,'receipt':receipt_text,
                  'receipt_sha256':hashlib.sha256(receipt_text.encode()).hexdigest(),
                  'explanation':explanation,'transfer_question':transfer_question,
                  'transfer_prediction':transfer_prediction,'confidence':confidence,'reasoning':reasoning})
    read_attempt(result)
    return result


def read_attempt(text):
    if type(text) is not str or len(text.encode())>6_000_000:
        raise ValueError('expected bounded attempt text')
    out=json.loads(text,object_pairs_hook=_unique_pairs)
    keys={'schema','schema_version','receipt','receipt_sha256','explanation','transfer_question',
          'transfer_prediction','confidence','reasoning','seal'}
    if type(out) is not dict or set(out)!=keys:
        raise ValueError('malformed reflection envelope')
    seal=out.pop('seal')
    if (digest(out)!=seal or out['schema']!=ATTEMPT_SCHEMA
            or type(out['schema_version']) is not int or out['schema_version']!=1):
        raise ValueError('unsupported or damaged reflection envelope')
    if type(out['receipt']) is not str or hashlib.sha256(out['receipt'].encode()).hexdigest()!=out['receipt_sha256']:
        raise ValueError('receipt link mismatch')
    replay(out['receipt'])
    for key in ('explanation','transfer_question','transfer_prediction','reasoning'):
        if type(out[key]) is not str or not out[key].strip():
            raise ValueError('fill reflection and transfer forecast before archiving')
    _finite(out['confidence'],0,1)
    return out


def export_attempt(attempt_text, destination):
    read_attempt(attempt_text)
    path=Path(destination)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        f.write(attempt_text)
    return path

# Reference code is method guidance, without measured case outcomes. The notebook
# only displays it after commitment; adopting edited code requires a new forecast.
REFERENCE_METHODS = '''def compile_select_aggregate(model, layer, head, query_space, key_space,
                             predicate, value_space, out_space, sharpness=20., amplitude=1.):
    attn = model.blocks[layer].attn
    n_key = predicate.shape[1]
    n_value = value_space.stop - value_space.start
    with t.no_grad():
        attn.W_Q[head][query_space, :n_key] = predicate * sharpness * model.cfg.d_head**.5
        attn.W_K[head][key_space, :n_key] = t.eye(n_key)
        attn.W_V[head][value_space, :n_value] = t.eye(n_value)
        attn.W_O[head][:n_value, out_space] = t.eye(n_value) * amplitude


def compile_induction(gain):
    model = utils.empty_attn_only_model(2, 3, 81, 17, 16, 17, 'cpu')
    tok, pos, prev, out, junk = slice(0,16), slice(16,33), slice(33,49), slice(49,65), slice(65,81)
    with t.no_grad():
        model.embed.W_E[:, tok] = t.eye(16)
        model.pos_embed.W_pos[:, pos] = t.eye(17)
        model.unembed.W_U[out, :] = gain * t.eye(16)
    previous_position = t.zeros(17,17)
    for i in range(1,17):
        previous_position[i,i-1] = 1
    for head in [0,1]:
        compile_select_aggregate(model,0,head,pos,pos,previous_position,tok,prev)
    compile_select_aggregate(model,0,2,pos,pos,t.zeros(17,17),tok,junk)
    compile_select_aggregate(model,1,0,tok,prev,t.eye(16),tok,out)
    compile_select_aggregate(model,1,1,junk,junk,t.eye(16),junk,junk)
    compile_select_aggregate(model,1,2,pos,pos,t.eye(17),tok,junk)
    return model


def add_weight_noise(model, sigma, seed):
    generator = t.Generator().manual_seed(seed)
    with t.no_grad():
        for name, param in model.named_parameters():
            if name.split('.')[-1] in ('W_Q', 'W_K', 'W_V', 'W_O'):
                param.add_(t.randn(param.shape, generator=generator) * sigma)
    return model


def patching_scores(graph, tokens, corrupt_out, metric):
    scores = t.zeros_like(graph.edge_mask, dtype=t.float32)
    with t.no_grad():
        full = metric(graph.run(tokens, corrupt_out, graph.edge_mask)).item()
        for u, v in graph.edge_mask.nonzero().tolist():
            keep = graph.edge_mask.clone()
            keep[u,v] = False
            scores[u,v] = full - metric(graph.run(tokens, corrupt_out, keep)).item()
    return scores


def eap_scores(graph, tokens, corrupt_out, metric):
    keep = graph.edge_mask.float().requires_grad_(True)
    value = metric(graph.run(tokens, corrupt_out, keep))
    return t.autograd.grad(value, keep)[0].detach() * graph.edge_mask


def eap_ig_scores(graph, tokens, corrupt_out, metric):
    total = t.zeros_like(graph.edge_mask, dtype=t.float32)
    for k in range(CASE['ig_steps']):
        keep = (graph.edge_mask.float() * (k+.5) / CASE['ig_steps']).requires_grad_(True)
        value = metric(graph.run(tokens, corrupt_out, keep))
        total += t.autograd.grad(value, keep)[0].detach()
    return total / CASE['ig_steps'] * graph.edge_mask


def roc_auc(scores, planted, valid):
    values, labels = scores[valid].abs(), planted[valid]
    positive, negative = values[labels], values[~labels]
    if not len(positive) or not len(negative):
        raise ValueError('AUC requires both classes')
    return ((positive[:,None] > negative[None,:]).float()
            + .5 * (positive[:,None] == negative[None,:]).float()).mean().item()


def assemble_result(scores, graph, planted, logits, clean):
    targets = clean[:,10:17,None]
    selected = logits[:,9:16]
    return {'auc': {name: roc_auc(value, planted, graph.edge_mask) for name,value in scores.items()},
            'scores': {name: value.detach().cpu().tolist() for name,value in scores.items()},
            'task_accuracy': (selected.argmax(-1) == targets[...,0]).float().mean().item(),
            'answer_logprob': selected.log_softmax(-1).gather(-1,targets).mean().item()}
'''


def reveal_methods(record_text):
    """Postcommit text only. This never runs a model or overwrites student cells."""
    read_record(record_text)
    return REFERENCE_METHODS


def practice_namespace(method_source):
    """Only independent fixture data; avoid importing TransformerLens for unit math."""
    import torch as t
    ns = {'t': t, 'CASE': deepcopy(CASE)}
    exec(compile(method_source, '<practice methods>', 'exec'), ns)
    return ns


def practice_test(ns, stage):
    """Independent small fixtures have known answers, never target condition outcomes."""
    import torch as t
    from types import SimpleNamespace
    if stage in ('select', 'induction', 'compiler'):
        attn = SimpleNamespace(W_Q=t.zeros(1,8,3), W_K=t.zeros(1,8,3),
                               W_V=t.zeros(1,8,3), W_O=t.zeros(1,3,8))
        model = SimpleNamespace(blocks=[SimpleNamespace(attn=attn)], cfg=SimpleNamespace(d_head=3))
        p = t.tensor([[1.,0.],[0.,1.]])
        ns['compile_select_aggregate'](model,0,0,slice(0,2),slice(2,4),p,slice(4,6),slice(6,8),2.,.7)
        t.testing.assert_close(attn.W_Q[0,:2,:2], p * 2 * 3**.5)
        t.testing.assert_close(attn.W_K[0,2:4,:2], t.eye(2))
        t.testing.assert_close(attn.W_V[0,4:6,:2], t.eye(2))
        t.testing.assert_close(attn.W_O[0,:2,6:8], t.eye(2)*.7)
        assert attn.W_Q[:,2:].count_nonzero() == 0
        if stage == 'select':
            return 'select: independent practice fixture passed'
        # Exercise the complete program and noise code on weights-only mock
        # modules. These fixtures have no forward method and never evaluate a
        # target condition. A non-target gain is used for layout checks.
        def empty_model(layers, heads, d_model, d_head, vocab, seq, device):
            assert (layers,heads,d_model,d_head,vocab,seq,device)==(2,3,81,17,16,17,'cpu')
            def weights(**shapes):
                module=t.nn.Module()
                for name,shape in shapes.items():
                    module.register_parameter(name,t.nn.Parameter(t.zeros(shape),requires_grad=False))
                return module
            model=t.nn.Module()
            model.cfg=SimpleNamespace(d_head=d_head)
            model.embed=weights(W_E=(vocab,d_model))
            model.pos_embed=weights(W_pos=(seq,d_model))
            model.unembed=weights(W_U=(d_model,vocab))
            model.blocks=t.nn.ModuleList()
            for _ in range(layers):
                block=t.nn.Module()
                block.attn=weights(W_Q=(heads,d_model,d_head),W_K=(heads,d_model,d_head),
                                   W_V=(heads,d_model,d_head),W_O=(heads,d_head,d_model))
                model.blocks.append(block)
            return model
        saved=ns.get('utils')
        ns['utils']=SimpleNamespace(empty_attn_only_model=empty_model)
        try:
            model=ns['compile_induction'](2.5)
            t.testing.assert_close(model.embed.W_E[:,:16],t.eye(16))
            t.testing.assert_close(model.pos_embed.W_pos[:,16:33],t.eye(17))
            t.testing.assert_close(model.unembed.W_U[49:65],2.5*t.eye(16))
            assert model.embed.W_E[:,16:].count_nonzero()==0
            assert model.unembed.W_U[:49].count_nonzero()==model.unembed.W_U[65:].count_nonzero()==0
            previous=t.zeros(17,17)
            previous[1:,:-1]=t.eye(16)
            specs=[(0,0,slice(16,33),slice(16,33),previous,slice(0,16),slice(33,49)),
                   (0,1,slice(16,33),slice(16,33),previous,slice(0,16),slice(33,49)),
                   (0,2,slice(16,33),slice(16,33),t.zeros(17,17),slice(0,16),slice(65,81)),
                   (1,0,slice(0,16),slice(33,49),t.eye(16),slice(0,16),slice(49,65)),
                   (1,1,slice(65,81),slice(65,81),t.eye(16),slice(65,81),slice(65,81)),
                   (1,2,slice(16,33),slice(16,33),t.eye(17),slice(0,16),slice(65,81))]
            for l,h,qs,ks,p,vs,outs in specs:
                attn=model.blocks[l].attn
                q,k,v,o=(t.zeros_like(getattr(attn,n)[h]) for n in ('W_Q','W_K','W_V','W_O'))
                n=p.shape[1];nv=vs.stop-vs.start
                q[qs,:n]=p*20*17**.5;k[ks,:n]=t.eye(n)
                v[vs,:nv]=t.eye(nv);o[:nv,outs]=t.eye(nv)
                for name,expected in zip(('W_Q','W_K','W_V','W_O'),(q,k,v,o)):
                    t.testing.assert_close(getattr(attn,name)[h],expected)
            if stage == 'induction':
                return 'induction: independent practice fixture passed'
            before={n:p.clone() for n,p in model.named_parameters()}
            rng=t.random.get_rng_state().clone()
            first=ns['add_weight_noise'](model,.07,19)
            second=ns['add_weight_noise'](ns['compile_induction'](2.5),.07,19)
            generator=t.Generator().manual_seed(19)
            for (name,p),(name2,p2) in zip(first.named_parameters(),second.named_parameters()):
                assert name==name2
                t.testing.assert_close(p,p2)
                expected=before[name]
                if name.split('.')[-1] in ('W_Q','W_K','W_V','W_O'):
                    expected=expected+t.randn(p.shape,generator=generator)*.07
                t.testing.assert_close(p,expected)
            assert t.equal(rng,t.random.get_rng_state()), 'noise must use a local generator'
            zero=ns['add_weight_noise'](ns['compile_induction'](2.5),0.,19)
            for name,p in zero.named_parameters():
                t.testing.assert_close(p,before[name])
        finally:
            ns['utils']=saved
    elif stage == 'scores':
        class PracticeGraph:
            edge_mask = t.tensor([[True, True, False]])
            def run(self, tokens, corrupt_out, keep):
                return (3 * keep[0,0]**2 - 2 * keep[0,1]).reshape(1)
        graph = PracticeGraph()
        args = (graph, None, None, lambda output: output.sum())
        for name, expected in [('patching_scores', [3.,-2.,0.]),
                               ('eap_scores', [6.,-2.,0.]),
                               ('eap_ig_scores', [3.,-2.,0.])]:
            t.testing.assert_close(ns[name](*args), t.tensor([expected],dtype=t.float32))
    elif stage == 'auc':
        valid = t.tensor([[True,True,True,True,False]])
        planted = t.tensor([[True,False,True,False,True]])
        for values, expected in [([-4.,1.,3.,0.,99.],1.), ([1.,4.,0.,3.,99.],0.),
                                 ([0.,0.,0.,0.,99.],.5), ([2.,2.,3.,1.,99.],.875)]:
            assert abs(ns['roc_auc'](t.tensor([values]),planted,valid)-expected) < 1e-7
    elif stage == 'assembly':
        graph = SimpleNamespace(edge_mask=t.tensor([[True,True]]))
        planted = t.tensor([[True,False]])
        clean = t.ones(2,17,dtype=t.long)
        logits = t.zeros(2,17,3)
        logits[:,:,1] = 2
        result = ns['assemble_result']({'fixture':t.tensor([[2.,1.]])},graph,planted,logits,clean)
        assert result['auc'] == {'fixture':1.} and result['task_accuracy'] == 1.
        assert abs(result['answer_logprob'] - t.log_softmax(t.tensor([0.,2.,0.]),0)[1].item()) < 1e-7
        assert result['scores'] == {'fixture': [[2.,1.]]}
    else:
        raise ValueError('unknown fixture stage')
    return f'{stage}: independent practice fixture passed'


def _stub(source, name):
    """Keep only function signature and a student-visible stub."""
    import ast
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines()
    signature = '\n'.join(lines[node.lineno-1:node.body[0].lineno-1])
    return signature + '\n    raise NotImplementedError("Implement ' + name + '")\n'


def build_cells():
    cells = []
    def add(kind, name, text, stage):
        cell = {'cell_type':kind, 'id':'circuit-'+name, 'source':text.strip()+'\n',
                'metadata': {'stage':stage}}
        if kind == 'code':
            cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown','intro', '''# [8.8] Blind prediction: changing output gain
For a compiled induction transformer with fixed attention sharpness and weight noise,
what is **Δ planted-edge ROC AUC for EAP = AUC(gain 1) − AUC(gain 10)**?
Commit a number in [−1, 1], confidence, and a causal explanation before either condition runs.

Given `BOS 7 4 2 7`, predict `4`: copy the token after the earlier `7`.
You will compile this computation, score edge interventions, and compare a controlled change.
The only experimental change is unembedding gain 10 → 1. Attention sharpness stays 20;
attention-weight noise σ=0.03, architecture, data, corruption, seeds and mean-answer-logprob
metric stay fixed. The postcommit control removes weight noise at gain 1.

Objectives: write a select/aggregate compiler; distinguish single-edge differences from
edge-mask derivatives and midpoint integrated gradients; implement tie-aware absolute-score
ROC AUC; compare a forecast with paired measurements and task behavior.

**Exposure:** opening equivalent results or the reference lesson makes this practice.
Use `unknown`, `seen`, or `self_reported_not_seen`; no automatic unseen score or mastery badge.
This is an exploratory compiled-model diagnostic, not an independent TEST evaluation.
Expected outputs before commitment below refer only to independent practice fixtures.
No target result or target figure is prefilled.''','forecast')
    add('code','setup', '''from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'mastery_hub').is_dir())
if str(root) not in sys.path:
    sys.path.insert(0,str(root))
from mastery_hub.units.circuit_prediction import lesson
import torch as t
from IPython.display import display, Code
t.set_num_threads(1)
t.set_default_dtype(t.float32)
t.set_default_device('cpu')
CASE = dict(lesson.CASE)
print(CASE)
# Existing installed environment only. No install, pretrained loading, training, or GPU calls.''','setup')
    groups = [('compiler',('compile_select_aggregate', 'compile_induction', 'add_weight_noise')),
              ('scores',('patching_scores', 'eap_scores', 'eap_ig_scores')),
              ('auc',('roc_auc',)), ('assembly',('assemble_result',))]
    prose = {
        'compiler': '''## Exercise 1: build the compiler in three small steps (20–30 minutes)

### 1a. Selection and copying: `compile_select_aggregate`
Arguments `model, layer, head` identify one head. `query_space`, `key_space`, `value_space`,
`out_space` are half-open residual slices, with widths nq, nk, nv, nv respectively.
`predicate` has shape `(nq,nk)`; nk and nv must fit inside `d_head`. The head's W_Q/W_K/W_V
have shape `(d_model,d_head)` and W_O has shape `(d_head,d_model)`.
Return nothing: write only this head's designated blocks into initially zero weights.
Q embeds predicate × sharpness × √d_head, K and V identity blocks, O amplitude × identity.
Then q Q (k K)^T/√d_head = sharpness × q predicate k^T; V/O copies selected values.
In the 8D DEV fixture, nq=nk=nv=2 and d_head=3: query [1,0] gives scores [2,0].
Implement this helper first. The cell's `select` test checks its four weight blocks without a forward call.
<details><summary>Hint 1a: which axes?</summary>Write W_Q[query_space,:nk],
W_K[key_space,:nk], W_V[value_space,:nv], and W_O[:nv,out_space], all under no_grad.
An identity predicate is only a special case: query and key widths may differ.</details>

### 1b. Compose two layers: `compile_induction`
Start with `utils.empty_attn_only_model(2,3,81,17,16,17,'cpu')`. This utility only allocates
an empty architecture; the program's weights are yours. Residual coordinates are
`tok=0:16`, `pos=16:33`, `prev=33:49`, `out=49:65`, `junk=65:81`.
Write identity token/position embeddings and unembed only out, multiplied by `gain`.
Heads 0.0 and 0.1 select previous position and copy tok→prev. Head 0.2 selects uniformly
and copies tok→junk. Head 1.0 matches tok→prev and copies tok→out. Head 1.1 matches
junk→junk and copies junk→junk; 1.2 matches positions and copies tok→junk. All other
weights stay zero. Use sharpness20, amplitude1. Return the configured model; do not call it.

Hand trace `BOS 7 4 2 7`: at position2, layer0's prev slice contains the earlier token7;
at position4 the query token7 matches that position2 key and copies position2's current token4.
The two previous-token heads add into the same prev slice. Finite softmax is a soft match,
not an exact symbolic selector. The separate `induction` test checks every composed weight block
on a weights-only mock with gain2.5, without running either target gain condition.
<details><summary>Hint 1b: previous-position matrix</summary>Make a 17×17 zero predicate,
then write identity16 into rows1:, columns:-1. Query position q selects key position q−1.
The induction predicate is identity16, between token and previous-token slices.</details>

### 1c. Add controlled noise: `add_weight_noise`
Only W_Q/W_K/W_V/W_O receive Gaussian noise. Use a local generator seeded by the argument,
iterate named parameters in existing order, preserve embedding/unembedding weights and the global RNG.
Return the model. The final `compiler` check includes zero-noise and repeated-seed controls.
<details><summary>Hint 1c: reproducibility</summary>Sample with a local torch.Generator;
noise scale is sigma, independent of gain. A zero sigma must preserve all weights exactly.</details>

Run the cell after each substep: it prints successful local checks before reaching an unfinished
later function. A later NotImplementedError means continue to the next substep, not erase the earlier work.''',
        'scores': '''## Exercise 2: score each valid edge
`graph.run(tokens, corrupt_out, keep)` interpolates each edge between its clean sender
and the corresponding sender on corrupted input. Invalid edges must stay zero.
Exact: m(all edges) − m(all except e). EAP: ∂m/∂keep_e at all valid keep=1.
IG10: average these derivatives at ten midpoint masks `(k+0.5)/10`, k=0…9.
This is the common straight-line **edge-mask** path, not activation-space IG.
<details><summary>Help</summary>Create a fresh leaf mask each time. Use `autograd.grad`
on the scalar metric, detach scores, and retain signs. A negative score can still rank
highly by absolute importance. No model outcome is needed to implement the definitions.</details>''',
        'auc': '''## Exercise 3: rank against planted structure
Restrict scores and labels to valid edges, then take absolute scores. AUC is the mean over
all positive/negative pairs of `1` for positive>negative and `0.5` for ties.
Reject inputs with an empty class. The random ranking expectation is 0.5.
<details><summary>Help</summary>Use broadcasting to compare every positive with every
negative. Ignore invalid entries even if their score is huge.</details>''',
        'assembly': '''## Exercise 4: assemble the scientific result visibly
Return JSON-native `auc` and full signed `scores` per method, plus `task_accuracy` and
`answer_logprob`. For clean prompts use logits at positions `9:16`, targets `clean[:,10:17]`;
mean over both prompts and predictable positions. Accuracy is argmax agreement;
answer logprob is gathered log-softmax. Do not change the denominator by condition.
<details><summary>Help</summary>Detach tensors before converting to lists. The fixture is
an invented three-token classifier, unrelated to either benchmark outcome.</details>'''}
    for stage, names in groups:
        add('markdown',stage+'-instructions',prose[stage], 'practice')
        source = '\n\n'.join(_stub(REFERENCE_METHODS, name) for name in names)
        checks = ("\nprint(lesson.practice_test(ns, 'select'))\nprint(lesson.practice_test(ns, 'induction'))" if stage == 'compiler' else '')
        add('code',stage, f"{stage.upper()}_SOURCE = '''\n{source}'''\nns = lesson.practice_namespace({stage.upper()}_SOURCE" + ("" if stage != 'assembly' else ' + "\\n" + AUC_SOURCE') + ')' + checks + f'\nprint(lesson.practice_test(ns, {stage!r}))', 'practice')
        add('markdown',stage+'-expected',f'Expected: `{stage}: independent practice fixture passed`. This does not evaluate the target case.','practice')
    add('markdown', 'dev-forecast', '''## Exposed DEV ladder — select, then copy (8–12 minutes)
Before freezing the exact target, use only this one-head, two-key arithmetic fixture.
It has residual width 8 and head width 3: query/key/value/output each occupy two coordinates.
With query [1,0], identity predicate, sharpness 2 and amplitude .7, hand-predict both
output coordinates. A hard selector would give [.7,0]; finite softmax need not.

Write your forecast, then choose an explanation: **A:** amplitude controls what gets selected;
**B:** Q/K and sharpness control selection, while V/O amplitude scales the copy.
Which paired observation would rule A out? Compare .7 copy, zero copy and uniform selection.
This is public DEV, not either gain condition, not an EAP/AUC target estimate.
Numeric confidence is an **unscored self-rating** of your reasoning, not P(a number is exactly right).''', 'dev')
    add('code', 'dev-forecast-card', '''dev_forecast = [.7, 0.]  # Worked hard-selection approximation; replace before running.
dev_self_rating = .6  # unscored self-rating, not a calibrated probability
dev_explanation = 'B'
dev_disconfirming_observation = 'Amplitude changes output but leaves attention unchanged.' ''', 'dev')
    add('code', 'dev-paired', '''from types import SimpleNamespace
import matplotlib.pyplot as plt
dev_ns = lesson.practice_namespace(COMPILER_SOURCE)
def dev_select_copy(query, amplitude, uniform=False):
    # No transformer constructor or target call: only small public weight arrays.
    attn = SimpleNamespace(**{name: t.zeros(shape) for name,shape in
        [('W_Q',(1,8,3)),('W_K',(1,8,3)),('W_V',(1,8,3)),('W_O',(1,3,8))]})
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=attn)], cfg=SimpleNamespace(d_head=3))
    predicate = t.zeros(2,2) if uniform else t.eye(2)
    dev_ns['compile_select_aggregate'](model,0,0,slice(0,2),slice(2,4),predicate,
                                     slice(4,6),slice(6,8),2.,amplitude)
    residual = t.zeros(2,8)
    residual[:,2:4] = t.eye(2); residual[:,4:6] = t.eye(2)
    q = t.zeros(8); q[:2] = t.tensor(query)
    scores = (q @ attn.W_Q[0]) @ (residual @ attn.W_K[0]).T / 3**.5
    attention = scores.softmax(-1)
    output = (attention @ (residual @ attn.W_V[0]) @ attn.W_O[0])[6:8]
    return attention, output
dev_pairs = {name: dev_select_copy([1.,0.], amplitude, uniform) for name,amplitude,uniform in
             [('copy .7',.7,False),('zero copy',0.,False),('uniform selection',.7,True)]}
fig,axes = plt.subplots(1,2,figsize=(10,3),layout='constrained')
for name,(attention,output) in dev_pairs.items():
    axes[0].plot([0,1],attention.tolist(), 'x--' if name=='zero copy' else 'o-',label=name)
    axes[1].plot([0,1],output.tolist(),'o-',label=name)
axes[0].set(title='DEV selection',xlabel='Key',ylabel='Attention probability',xticks=[0,1])
axes[1].set(title='DEV copied coordinates',xlabel='Output coordinate',ylabel='Activation',xticks=[0,1])
for ax in axes: ax.legend()
display(fig)
print('DEV copy:',dev_pairs['copy .7'][1].tolist())
print('DEV coordinate absolute forecast errors:',(dev_pairs['copy .7'][1]-t.tensor(dev_forecast)).abs().tolist())
assert t.equal(dev_pairs['copy .7'][0], dev_pairs['zero copy'][0])
assert dev_pairs['zero copy'][1].count_nonzero() == 0''', 'dev')
    add('markdown', 'dev-reflection', '''### Worked DEV reflection / self-check
Expected attention is softmax([2,0]) ≈ [.880797,.119203]; copied output is
[.616558,.083442], while uniform selection copies [.35,.35]. A hard-copy forecast
[.7,0] has absolute coordinate errors about [.083442,.083442]. Finite selection,
not failed copying, explains that error. Zero amplitude preserves attention and erases output:
that observation rules out A for this fixture. Self-check: cite *both* the unchanged attention
and changed output, distinguish the uniform-selection control, and say what remains untested.
This does not predict target AUC or learned-model behavior.

### Try It Yourself — changed-condition DEV transfer
Freeze the next change in words: query [1,0] → [0,1], same keys, values, predicate,
sharpness and amplitude. Predict the two output coordinates before the next cell.
Explanation B predicts reversal; an explanation tied to absolute key position does not.''', 'dev')
    add('code', 'dev-transfer', '''dev_transfer_forecast = [.083442, .616558]  # Worked exposed forecast; edit first.
dev_transfer_attention, dev_transfer_output = dev_select_copy([0.,1.], .7)
t.testing.assert_close(dev_transfer_output, dev_pairs['copy .7'][1].flip(0))
fig,ax = plt.subplots(figsize=(6,3),layout='constrained')
ax.plot([0,1],dev_pairs['copy .7'][1].tolist(),'o-',label='Original query')
ax.plot([0,1],dev_transfer_output.tolist(),'o-',label='Swapped query')
ax.plot([0,1],dev_pairs['zero copy'][1].tolist(),'--',label='Zero-copy control')
ax.set(xlabel='Output coordinate',ylabel='Activation',title='Observed DEV query transfer',xticks=[0,1]);ax.legend()
display(fig)
print('DEV transfer observed:',dev_transfer_output.tolist())
print('DEV transfer absolute forecast errors:',(dev_transfer_output-t.tensor(dev_transfer_forecast)).abs().tolist())
dev_revision = ''  # Cite the reversal; explain why this tests content selection, not target gain.''', 'dev')
    add('markdown','commit-instructions','''## Freeze, then commit
Finish all functions and pass each independent fixture first. Freezing constructs weights,
data and masks for fingerprinting, but makes **no model forward calls**. Read the complete
case printed above, then enter your own numerical forecast, confidence, reasoning and exposure.
The record is immutable JSON text. Rerunning commit creates a separate attempt; preserve the
old variable or export it if desired. Editing methods or runtime afterwards requires a new forecast.
Do not inspect the original answer-rich 8.8 notebook before this step if you want blind practice.''','commit')
    add('code','freeze', r'''METHOD_SOURCE = '\n\n'.join([COMPILER_SOURCE, SCORES_SOURCE, AUC_SOURCE, ASSEMBLY_SOURCE])
# The code is student-authored; inspect it before executing.
identity = lesson.freeze_inputs(METHOD_SOURCE)
print('Frozen case, method, helper, environment, tensor and ordered-weight identities.')''','freeze')
    add('code','commit', '''prediction = None  # Your numeric Δ EAP AUC, in [-1,1]
confidence = None  # Unscored reasoning self-rating in [0,1]; not a probability
reasoning = ''  # Explain a causal pathway, and a reason your estimate could fail.
prior_exposure = 'unknown'  # unknown / seen / self_reported_not_seen
commitment = lesson.commit_prediction(identity, METHOD_SOURCE, prediction, confidence, reasoning, prior_exposure)
print('Commitment created. This cell runs no conditions.')''','commit')
    add('markdown','postcommit','''## Observe only after commitment
The next cell explicitly runs three fixed CPU conditions: noisy gain 10, noisy gain 1,
and no-noise gain 1. It scores exact patching, EAP, and IG10 on all valid edges.
One local run has a cooperative 120-second budget checked between stages, one CPU thread,
no downloads/training. A single learner function cannot be forcibly interrupted by this helper;
an over-budget stage is rejected before any successful receipt is returned.
The plotting and replay cells only read stored measurements.
<details><summary>Solution access after commitment</summary>The optional cell below
returns reference method text after validating a commitment. It never modifies your functions.
If you adopt a changed method, freeze again and make a new forecast before running.</details>''','postcommit')
    add('code','solutions', '''# Optional postcommit reference, no benchmark runs:
# display(Code(lesson.reveal_methods(commitment), language='python'))''','solutions')
    add('code','run', '''receipt = lesson.run_experiment(commitment, METHOD_SOURCE)
observation = lesson.replay(receipt)
display(lesson.plot_receipt(receipt))
for condition, result in observation['results'].items():
    print(condition, 'accuracy:', result['task_accuracy'], 'mean answer logprob:', result['answer_logprob'])
print('Observed Δ EAP AUC:', observation['delta_eap_auc'])
print('Absolute forecast error:', observation['absolute_forecast_error'])''','run')
    add('markdown','interpretation','''## Interpret the paired result
Compare the three methods across gains, then the no-noise control. Did changing the output
scale change the task accuracy, its confidence, the ordering of edge scores, or several of
these? Distinguish the derivative at the endpoint from finite single-edge interventions and
the integrated path. Explain both evidence for your causal story and a remaining alternative.
The heatmap shows signed effects on the seven **planted structural edges**. Weight noise can
introduce additional paths: these labels are not complete causal ground truth after noise.
AUC ranks structure; it does not measure circuit completeness or prove scientific correctness.
The horizontal 0.5 line is the expected random-ranking AUC, not a fitted chance result.
The control AUC and task behavior are diagnostics, not an independent replication.''','reflect')
    add('code','reflection', '''reflection = ''  # What matched? What changed your causal model? Cite the control.
transfer_question = ''  # Name a changed condition, metric, and fixed denominator.
next_prediction = ''  # Predict it BEFORE testing in separate authorized work.
transfer_confidence = None
transfer_reasoning = ''
# After filling every reflection field, explicitly archive without changing the receipt:
# attempt = lesson.reflect(receipt, reflection, transfer_question, next_prediction, transfer_confidence, transfer_reasoning)
# Try It Yourself: inspect one stored edge without rerunning any condition.
edge = ('input', '1.0<q>')
obs = lesson.replay(receipt)
u, v = obs['sender_order'].index(edge[0]), obs['receiver_order'].index(edge[1])
for condition, result in obs['results'].items():
    print(condition, {name: score[u][v] for name, score in result['scores'].items()})''','play')
    add('markdown','replay-instructions','''## Save and replay privately
Export is opt-in and creates a new private file without overwriting. A fresh Python session
can import this helper, read that JSON text, and call `plot_receipt`; no torch, model,
TransformerLens, source methods, environment match, or benchmark execution is required for
receipt replay. Keep your forecast, causal reflection, and experiment receipt together.''','replay')
    add('code','export', '''# Optional explicit private export:
# lesson.export_attempt(attempt, Path.home()/'circuit-prediction-attempt-1.json')
# In another session:
# saved = Path.home().joinpath('circuit-prediction-attempt-1.json').read_text()
# archived = lesson.read_attempt(saved)
# print(archived['explanation'], archived['transfer_prediction'])
# display(lesson.plot_receipt(archived['receipt']))''','export')
    add('markdown','limitations','''## Limits and connection to real models
This intervention is a small finite-softmax compiled induction transformer. Weights are
written rather than learned; there is no pretrained-model result here. One fixed synthetic
cohort and seed supports a paired diagnostic, not a distributional or real-model claim.
IG10 approximates a path integral; exact single-edge effects depend on the corruption and
background circuit. Redundant paths can have small single-edge effects. Local hashes detect
accidental edits, not a dishonest self-report or modified software. Confidence is recorded;
only absolute numeric forecast error is computed, with no calibration/mastery badge.

RASP and Tracr connect program structure to transformer weights. Read their primary papers
after commitment, alongside the original 8.8 reference. Further investigation could vary
corruption or path discretization under a separate forecast and explicit compute budget:
[RASP](https://arxiv.org/abs/2106.06981), [Tracr](https://arxiv.org/abs/2301.05062).
Human learner walkthrough and scientific acceptance remain deferred.''','limitations')
    return cells
