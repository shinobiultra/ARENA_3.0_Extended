"""Blind eight-point Data Shapley denominator challenge.

The notebook owns every numerical method. Import, freeze, commitment and replay
perform no training. Local sealed JSON detects accidental drift, not cheating or
prior exposure; observations are separate from learner-reported guided records.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
TESTS = ROOT / 'tests/mastery/test_shapley_prediction.py'
SCHEMA = 'arena-blind-shapley-prediction'
VERSION = 1
CASE = {
    'id': '16.9-coalition-denominator-v1',
    'train_x': [[2.,1.],[2.,1.],[1.5,1.5],[2.,-.5],[-2.,-1.],[-1.5,-1.5],[2.,1.],[.2,2.5]],
    'train_y': [1.,1.,1.,1.,0.,0.,0.,1.],
    'val_x': [[2.,1.],[1.5,1.2],[2.,-.5],[1.,2.],[-2.,-1.],[-1.5,-1.2],[-2.,.5],[-1.,-2.]],
    'val_y': [1.,1.,1.,1.,0.,0.,0.,0.],
    'initial_state': [0.,0.,0.], 'state_order': ['w0','w1','bias'],
    'point_roles': ['duplicate A','duplicate A copy','positive diagonal','positive low',
                    'negative','negative diagonal','opposite-labelled copy of A','rare positive'],
    'steps': 40, 'lr': .2, 'dtype': 'torch.float64', 'device': 'cpu', 'threads': 1,
    'coalition_order': 'ascending bitmask, bit i means training row i',
    'policies': ['fixed_n','subset_mean'], 'empty_policy': 'zero initialization; denominator clamped to 1',
    'utility': 'mean validation BCE at zero minus mean validation BCE after training, nats',
    'weighting': '1/(n*comb(n-1,k)) for every predecessor subset of size k',
    'forecast': 'phi[7](subset_mean) minus phi[7](fixed_n)',
    'controls': ['empty','full_same','zero_lr','duplicate_0_1','efficiency','label_0_6_relocation'],
    'budget_seconds': 120, 'evidence_class': 'analytic_toy',
    'data_role': 'fixed synthetic diagnostic; no independent TEST claim',
}
METHOD_NAMES = ('train_coalitions','coalition_utilities','exact_shapley','marginal_by_size','assemble_result')
CONDITIONS = ('fixed_n','subset_mean','swap_fixed_n','swap_subset_mean','zero_fixed_n','zero_subset_mean')
ASSESSMENT = 'local_self_report_no_unseen_certification'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seal(payload):
    out = deepcopy(payload)
    out['seal'] = digest(payload)
    return json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False)


def _pairs(pairs):
    result = {}
    for k,v in pairs:
        _require(k not in result, 'duplicate JSON key')
        result[k] = v
    return result


def _parse(text, limit=2_000_000):
    _require(type(text) is str and len(text.encode()) <= limit, 'expected bounded JSON text')
    try:
        out = json.loads(text,object_pairs_hook=_pairs,
                         parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (RecursionError,OverflowError) as exc:
        raise ValueError('invalid JSON nesting') from exc
    _require(type(out) is dict and 'seal' in out,'missing sealed object')
    seal = out.pop('seal')
    _require(seal == digest(out),'invalid record seal')
    return out


def _keys(value, expected):
    _require(type(value) is dict and set(value)==set(expected), 'missing or unknown fields')


def _number(value, low=None, high=None):
    _require(type(value) in (int,float) and math.isfinite(value)
             and (low is None or value>=low) and (high is None or value<=high), 'invalid finite number')


def _text(value):
    _require(type(value) is str and 0<len(value.strip())<=20_000,'fill explanatory text')


def _sha(value):
    _require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value), 'invalid SHA256')


def _tensor_hash(rows):
    def flatten(value):
        return [v for part in value for v in flatten(part)] if type(value) is list else [value]
    values=flatten(rows)
    return hashlib.sha256(struct.pack('<'+'d'*len(values),*values)).hexdigest()


def _input_layout():
    return {'train_x':([8,2],CASE['train_x']), 'train_y':([8],CASE['train_y']),
            'val_x':([8,2],CASE['val_x']), 'val_y':([8],CASE['val_y']),
            'initial_state':([3],CASE['initial_state']),
            'coalition_masks':([256,8],[[float((m>>i)&1) for i in range(8)] for m in range(256)])}


def environment_identity():
    import torch
    return {'python':platform.python_version(),'torch':torch.__version__,
            'platform':platform.platform(),'byteorder':sys.byteorder,
            'threads':torch.get_num_threads(),'default_dtype':str(torch.get_default_dtype()),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'default_device':str(torch.get_default_device())}


def _method_tree(source, complete=True):
    _require(type(source) is str and 0<len(source)<=100_000,'expected bounded method source')
    try:
        tree=ast.parse(source)
    except SyntaxError as exc:
        raise ValueError('invalid method source') from exc
    _require(tree.body and all(isinstance(n,ast.FunctionDef) for n in tree.body),
             'method source must contain only function definitions')
    names=tuple(n.name for n in tree.body)
    _require(names==METHOD_NAMES if complete else len(set(names))==len(names) and all(n in METHOD_NAMES for n in names),
             'implement the named core methods in order')
    # Defining functions must not run learner expressions in decorators/defaults.
    for node in tree.body:
        arguments=node.args.posonlyargs+node.args.args+node.args.kwonlyargs+[a for a in (node.args.vararg,node.args.kwarg) if a]
        _require(not node.decorator_list and not node.returns and not node.type_params
                 and not any(a.annotation for a in arguments),
                 'use plain unannotated function definitions')
        _require(all(isinstance(v,ast.Constant) for v in node.args.defaults+list(filter(None,node.args.kw_defaults))),
                 'defaults must be literal constants')
    return tree


def method_namespace(source, complete=True):
    """Evaluate only definitions; learner bodies run exclusively at explicit call sites."""
    import torch as t
    _method_tree(source,complete)
    ns={'t':t,'math':math}
    exec(compile(source,'<student Shapley methods>','exec'),ns)
    return ns


def freeze_inputs(method_source):
    """Hash exact data/init/source/runtime without evaluating a method or fitting a model."""
    _method_tree(method_source)
    env=environment_identity()
    _require(env['threads']==1 and env['default_device']=='cpu','set one CPU thread and CPU default device before freezing')
    return {'case_sha256':digest(CASE),'method_sha256':hashlib.sha256(method_source.encode()).hexdigest(),
            'helper_sha256':_file(__file__),'tests_sha256':_file(TESTS),'environment':env,
            'inputs':{name:{'shape':shape,'dtype':CASE['dtype'],'sha256':_tensor_hash(rows)}
                      for name,(shape,rows) in _input_layout().items()}}


def _validate_identity(identity, source):
    _keys(identity,('case_sha256','method_sha256','helper_sha256','tests_sha256','environment','inputs'))
    for k in ('case_sha256','method_sha256','helper_sha256','tests_sha256'):
        _sha(identity[k])
    _require(identity['case_sha256']==digest(CASE) and identity['method_sha256']==hashlib.sha256(source.encode()).hexdigest(),
             'case or method identity mismatch')
    env=identity['environment']
    _keys(env,('python','torch','platform','byteorder','threads','default_dtype','deterministic_algorithms','default_device'))
    for k in ('python','torch','platform','default_dtype'):
        _text(env[k])
    _require(env['byteorder'] in ('little','big') and type(env['threads']) is int and env['threads']==1
             and type(env['deterministic_algorithms']) is bool and env['default_device']=='cpu','invalid environment identity')
    _keys(identity['inputs'],_input_layout())
    for name,(shape,rows) in _input_layout().items():
        _require(identity['inputs'][name]=={'shape':shape,'dtype':CASE['dtype'],'sha256':_tensor_hash(rows)},
                 'ordered input shape/dtype/bytes mismatch')


def commit_prediction(identity,method_source,prediction,confidence,reasoning,prior_exposure='unknown'):
    text=_seal({'schema':SCHEMA,'schema_version':VERSION,'status':'committed','case':deepcopy(CASE),
                'identity':deepcopy(identity),'method_source':method_source,
                'prediction':prediction,'confidence':confidence,'reasoning':reasoning,
                'prior_exposure':prior_exposure,'assessment':ASSESSMENT})
    read_record(text)
    return text


def _close(a,b):
    return math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12)


def _vector(value,n):
    _require(type(value) is list and len(value)==n,'invalid vector shape')
    for x in value:
        _number(x)


def _reference_credits(values,n=8):
    return [math.fsum((values[m|(1<<i)]-values[m])/(n*math.comb(n-1,m.bit_count()))
                     for m in range(1<<n) if not m&(1<<i)) for i in range(n)]


def _validate_result(result):
    _keys(result,('states','utilities','shapley','marginals7','loo','states_sha256'))
    states=result['states']
    _require(type(states) is list and len(states)==256,'incomplete ordered coalition states')
    for row in states:
        _vector(row,3)
    _vector(result['utilities'],256)
    for key,n in [('shapley',8),('marginals7',8),('loo',8)]:
        _vector(result[key],n)
    _require(result['states_sha256']==_tensor_hash(states),'fitted-state hash mismatch')
    # Replay needs no model library: verify BCE from stored fitted states directly.
    for state,utility in zip(states,result['utilities'],strict=True):
        losses=[]
        for x,y in zip(CASE['val_x'],CASE['val_y'],strict=True):
            z=state[0]*x[0]+state[1]*x[1]+state[2]
            losses.append(max(z,0)+math.log1p(math.exp(-abs(z)))-y*z)
        _require(_close(utility,math.log(2)-math.fsum(losses)/8),'utility differs from fitted state')
    values=result['utilities']
    expected=_reference_credits(values)
    _require(all(_close(a,b) for a,b in zip(result['shapley'],expected,strict=True)),'Shapley allocation mismatch')
    strata=[math.fsum(values[m|128]-values[m] for m in range(128) if m.bit_count()==k)/math.comb(7,k) for k in range(8)]
    _require(all(_close(a,b) for a,b in zip(result['marginals7'],strata,strict=True)),'marginal strata mismatch')
    _require(all(_close(result['loo'][i],values[255]-values[255^(1<<i)]) for i in range(8)),'LOO mismatch')
    _require(_close(math.fsum(result['shapley']),values[255]-values[0]),'efficiency failed')
    _require(_close(math.fsum(strata)/8,result['shapley'][7]),'strata mean mismatch')
    _require(states[0]==[0.,0.,0.] and _close(values[0],0),'empty coalition failed')


def _control_checks(results):
    checks={}
    for policy in ('fixed_n','subset_mean'):
        base,swapped,zero=(results[prefix+policy] for prefix in ('','swap_','zero_'))
        phi=base['shapley']
        checks[policy+'_efficiency']=_close(math.fsum(phi),base['utilities'][-1]-base['utilities'][0])
        checks[policy+'_duplicate_0_1']=_close(phi[0],phi[1])
        order=[6,1,2,3,4,5,0,7]
        checks[policy+'_label_0_6_relocation']=all(_close(swapped['shapley'][i],phi[order[i]]) for i in range(8))
        checks[policy+'_zero_lr']=all(v==0 for state in zero['states'] for v in state) and all(_close(v,0) for v in zero['utilities']+zero['shapley'])
        checks[policy+'_empty']=base['states'][0]==[0.,0.,0.] and _close(base['utilities'][0],0)
    checks['full_same']=all(_close(a,b) for a,b in zip(results['fixed_n']['states'][-1],results['subset_mean']['states'][-1],strict=True)) and _close(results['fixed_n']['utilities'][-1],results['subset_mean']['utilities'][-1])
    return checks


def _validate_failure(failure):
    _keys(failure,('kind','stage','error_type','message','elapsed_seconds','measured','results','controls'))
    _require(failure['kind']=='failed_cpu_training','invalid failure classification')
    for key in ('stage','error_type','message'):
        _text(failure[key])
    _number(failure['elapsed_seconds'],0)
    _require(type(failure['measured']) is dict and set(failure['measured'])<=set(CONDITIONS),'unknown measured condition')
    for measured in failure['measured'].values():
        _require(type(measured) is dict and set(measured) in ({'states','states_sha256'}, {'states','states_sha256','utilities'}),
                 'malformed partial measurement')
        _require(type(measured['states']) is list and len(measured['states'])==256,'invalid measured states')
        for row in measured['states']:
            _vector(row,3)
        _require(measured['states_sha256']==_tensor_hash(measured['states']),'measured state hash mismatch')
        if 'utilities' in measured:
            _vector(measured['utilities'],256)
    _require(type(failure['results']) is dict and set(failure['results'])<=set(failure['measured']),'unknown completed condition')
    for label,result in failure['results'].items():
        _validate_result(result)
        for key in ('states','utilities','states_sha256'):
            _require(result[key]==failure['measured'][label][key],'failure lost actual measurements')
    if failure['controls'] is not None:
        _require(set(failure['results'])==set(CONDITIONS),'controls without complete measurements')
        _require(failure['controls']==_control_checks(failure['results']),'invalid failed controls')


def read_record(text):
    """Strict, bounded, model-free parsing; observed values are machine receipts."""
    out=_parse(text)
    _require(out.get('schema')==SCHEMA and type(out.get('schema_version')) is int and out.get('schema_version')==VERSION,
             'unsupported schema')
    if out.get('status')=='committed':
        _keys(out,('schema','schema_version','status','case','identity','method_source','prediction','confidence',
                   'reasoning','prior_exposure','assessment'))
        _require(digest(out['case'])==digest(CASE),'unsupported case')
        _method_tree(out['method_source'])
        _validate_identity(out['identity'],out['method_source'])
        _number(out['prediction']);_number(out['confidence'],0,1);_text(out['reasoning'])
        _require(out['prior_exposure'] in ('unknown','seen','self_reported_not_seen') and out['assessment']==ASSESSMENT,
                 'invalid exposure or assessment')
    elif out.get('status') in ('observed','failed'):
        _keys(out,('schema','schema_version','status','commitment','commitment_sha256',
                   'observation' if out['status']=='observed' else 'failure'))
        _require(type(out['commitment']) is str and hashlib.sha256(out['commitment'].encode()).hexdigest()==out['commitment_sha256'],
                 'commitment link mismatch')
        # Disallow nested observed records before recursing.
        _require(_parse(out['commitment']).get('status')=='committed','expected original commitment')
        commitment=read_record(out['commitment'])
        if out['status']=='failed':
            _validate_failure(out['failure'])
            return out
        obs=out['observation']
        _keys(obs,('kind','results','controls','delta_phi7','absolute_forecast_error','elapsed_seconds','nonzero_lr_fits','zero_lr_checks'))
        _require(obs['kind']=='fresh_cpu_training' and type(obs['nonzero_lr_fits']) is int and obs['nonzero_lr_fits']==1024
                 and type(obs['zero_lr_checks']) is int and obs['zero_lr_checks']==512,'invalid execution classification/inventory')
        _number(obs['elapsed_seconds'],0,120)
        _require(obs['elapsed_seconds']>0,'invalid elapsed time')
        _keys(obs['results'],CONDITIONS)
        for result in obs['results'].values():
            _validate_result(result)
        controls=_control_checks(obs['results'])
        _require(obs['controls']==controls and all(type(v) is bool and v for v in obs['controls'].values()),'control failure')
        expected=obs['results']['subset_mean']['shapley'][7]-obs['results']['fixed_n']['shapley'][7]
        _number(obs['delta_phi7']);_number(obs['absolute_forecast_error'],0)
        _require(_close(obs['delta_phi7'],expected) and _close(obs['absolute_forecast_error'],abs(expected-commitment['prediction'])),
                 'forecast scoring mismatch')
    else:
        raise ValueError('unsupported record status')
    return out


def assert_current(record_text,method_source):
    record=read_record(record_text)
    _require(record['status']=='committed','already observed: replay or create a separate commitment')
    _require(method_source==record['method_source'],'stale method text; freeze and commit again')
    # No numerical student method is evaluated before all input identities match.
    _require(freeze_inputs(method_source)==record['identity'],'stale inputs, helper, tests or environment')
    return record


class ExperimentFailure(ValueError):
    """An unscored failure with exportable measured evidence; never retry implicitly."""
    def __init__(self, record_text):
        self.record_text=record_text
        failure=read_record(record_text)['failure']
        super().__init__(failure['stage']+': '+failure['message'])


def run_experiment(commitment_text,method_source):
    """Run explicitly; retain partial actual measurements in any structured failure."""
    started=time.monotonic()
    assert_current(commitment_text,method_source)
    capture={'stage':'setup','measured':{},'results':{},'controls':None}
    try:
        return _execute_experiment(commitment_text,method_source,started,capture)
    except Exception as error:
        failure={'kind':'failed_cpu_training','error_type':type(error).__name__,
                 'message':(str(error) or type(error).__name__)[:20_000],
                 'elapsed_seconds':time.monotonic()-started,**capture}
        text=_seal({'schema':SCHEMA,'schema_version':VERSION,'status':'failed',
                    'commitment':commitment_text,'commitment_sha256':hashlib.sha256(commitment_text.encode()).hexdigest(),
                    'failure':failure})
        read_record(text)
        raise ExperimentFailure(text) from error


def _execute_experiment(commitment_text,method_source,started,capture):
    """Explicit six-condition tiny CPU training, under a cooperative overall deadline."""
    def deadline():
        if time.monotonic()-started>CASE['budget_seconds']:
            raise TimeoutError('CPU budget exceeded; no successful receipt')
    deadline()
    import torch as t
    ns=method_namespace(method_source)
    capture['stage']='practice fixtures'
    for stage in ('training','utility','shapley','strata','assembly'):
        practice_test(ns,stage)
        deadline()
    frozen=read_record(commitment_text)['identity']
    _require(environment_identity()==frozen['environment'],'practice changed frozen environment')
    x=t.tensor(CASE['train_x'],dtype=t.float64,device='cpu')
    y=t.tensor(CASE['train_y'],dtype=t.float64,device='cpu')
    vx=t.tensor(CASE['val_x'],dtype=t.float64,device='cpu')
    vy=t.tensor(CASE['val_y'],dtype=t.float64,device='cpu')
    masks=((t.arange(256,device='cpu')[:,None] >> t.arange(8,device='cpu'))&1).to(t.float64)
    results=capture['results']
    for label in CONDITIONS:
        deadline()
        policy='subset_mean' if label.endswith('subset_mean') else 'fixed_n'
        labels=y[t.tensor([6,1,2,3,4,5,0,7],device='cpu')] if label.startswith('swap_') else y
        lr=0. if label.startswith('zero_') else CASE['lr']
        labels_before=_tensor_hash(labels.tolist())
        capture['stage']=label+': training'
        states=ns['train_coalitions'](x,labels,masks,CASE['steps'],lr,policy)
        _require(isinstance(states,t.Tensor) and states.shape==(256,3) and states.dtype==t.float64
                 and states.device.type=='cpu' and bool(t.isfinite(states).all()),'invalid fitted CPU float64 states')
        capture['measured'][label]={'states':states.tolist(),'states_sha256':_tensor_hash(states.tolist())}
        deadline()
        _require(_tensor_hash(labels.tolist())==labels_before,'method changed condition labels')
        for name,tensor in [('train_x',x),('train_y',y),('val_x',vx),('val_y',vy),('coalition_masks',masks)]:
            _require(_tensor_hash(tensor.tolist())==frozen['inputs'][name]['sha256'],'method changed frozen inputs')
        _require(environment_identity()==frozen['environment'],'method changed frozen environment')
        capture['stage']=label+': utility'
        utilities=ns['coalition_utilities'](states,vx,vy)
        _require(isinstance(utilities,t.Tensor) and utilities.shape==(256,) and utilities.dtype==t.float64
                 and utilities.device.type=='cpu' and bool(t.isfinite(utilities).all()),'invalid actual utility tensor')
        capture['measured'][label]['utilities']=utilities.tolist()
        for name,tensor in [('val_x',vx),('val_y',vy)]:
            _require(_tensor_hash(tensor.tolist())==frozen['inputs'][name]['sha256'],'utility changed frozen inputs')
        _require(states.tolist()==capture['measured'][label]['states'],'utility changed fitted states')
        _require(environment_identity()==frozen['environment'],'utility changed frozen environment')
        deadline()
        capture['stage']=label+': assembly'
        result=ns['assemble_result'](states,utilities,8,7)
        _require(type(result) is dict and 'states_sha256' not in result,'invalid student assembly')
        measured=capture['measured'][label]
        _require(result.get('states')==states.tolist()==measured['states']
                 and result.get('utilities')==utilities.tolist()==measured['utilities'],
                 'assembly changed actual fitted states or utilities')
        result['states_sha256']=measured['states_sha256']
        _require(environment_identity()==frozen['environment'],'assembly changed frozen environment')
        _validate_result(result)
        results[label]=result
        deadline()
    capture['stage']='paired controls'
    controls=_control_checks(results)
    capture['controls']=controls
    _require(all(controls.values()),'control failure: '+str(controls))
    delta=results['subset_mean']['shapley'][7]-results['fixed_n']['shapley'][7]
    committed=read_record(commitment_text)
    result=_seal({'schema':SCHEMA,'schema_version':VERSION,'status':'observed','commitment':commitment_text,
                  'commitment_sha256':hashlib.sha256(commitment_text.encode()).hexdigest(),
                  'observation':{'kind':'fresh_cpu_training','results':results,'controls':controls,
                                 'delta_phi7':delta,'absolute_forecast_error':abs(delta-committed['prediction']),
                                 'elapsed_seconds':time.monotonic()-started,'nonzero_lr_fits':1024,'zero_lr_checks':512}})
    read_record(result)
    deadline()
    return result


def replay(receipt_text):
    record=read_record(receipt_text)
    _require(record['status']=='observed','no observation to replay')
    return deepcopy(record['observation'])


def replay_failure(failure_text):
    record=read_record(failure_text)
    _require(record['status']=='failed','expected an unscored failure')
    return deepcopy(record['failure'])


def plot_receipt(receipt_text):
    """Plot stored credits and marginal strata only; no torch or student evaluation."""
    import matplotlib.pyplot as plt
    obs=replay(receipt_text)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for offset,policy in [(-.2,'fixed_n'),(.2,'subset_mean')]:
        result=obs['results'][policy]
        axes[0].bar([i+offset for i in range(8)],result['shapley'],width=.38,label=policy)
        axes[1].plot(range(8),result['marginals7'],'o-',label=policy)
    axes[0].set(xticks=range(8),xlabel='Training player (0/1 duplicates, 6 opposite label, 7 rare)',ylabel='Exact Data Shapley (mean validation BCE nats)',title='Same total utility; different learning games')
    axes[1].set(xlabel='Predecessor coalition size',ylabel='Mean marginal utility of player 7 (nats)',title='Eight size strata average to φ7')
    for ax in axes:
        ax.axhline(0,color='gray',lw=.7);ax.legend()
    return fig


def _exclusive(text,destination):
    path=Path(destination)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as handle:
        handle.write(text)
    return path


def export_record(text,destination):
    read_record(text)
    return _exclusive(text,destination)


def reflect(receipt_text,explanation,transfer_question,transfer_prediction,confidence,reasoning):
    replay(receipt_text)
    text=_seal({'schema':SCHEMA+'-reflection','schema_version':VERSION,'receipt':receipt_text,
                'receipt_sha256':hashlib.sha256(receipt_text.encode()).hexdigest(),
                'explanation':explanation,'transfer_question':transfer_question,
                'transfer_prediction':transfer_prediction,'confidence':confidence,'reasoning':reasoning})
    read_attempt(text)
    return text


def read_attempt(text):
    out=_parse(text,3_000_000)
    _keys(out,('schema','schema_version','receipt','receipt_sha256','explanation','transfer_question','transfer_prediction','confidence','reasoning'))
    _require(out['schema']==SCHEMA+'-reflection' and type(out['schema_version']) is int and out['schema_version']==VERSION,
             'unsupported reflection schema')
    _require(type(out['receipt']) is str and hashlib.sha256(out['receipt'].encode()).hexdigest()==out['receipt_sha256'],
             'receipt link mismatch')
    replay(out['receipt'])
    for key in ('explanation','transfer_question','transfer_prediction','reasoning'):
        _text(out[key])
    _number(out['confidence'],0,1)
    return out


def export_attempt(text,destination):
    read_attempt(text)
    return _exclusive(text,destination)


REFERENCE_METHODS = '''def train_coalitions(x, y, masks, steps, lr, policy):
    n, d = x.shape
    if policy not in ('fixed_n', 'subset_mean'):
        raise ValueError('unknown denominator policy')
    weights = x.new_zeros((len(masks), d))
    bias = x.new_zeros(len(masks))
    denominator = x.new_full((len(masks),), n) if policy == 'fixed_n' else masks.sum(1).clamp_min(1)
    for _ in range(steps):
        errors = (t.sigmoid(weights @ x.T + bias[:, None]) - y) * masks
        weights = weights - lr * (errors @ x) / denominator[:, None]
        bias = bias - lr * errors.sum(1) / denominator
    return t.cat([weights, bias[:, None]], 1)


def coalition_utilities(states, val_x, val_y):
    logits = states[:, :-1] @ val_x.T + states[:, -1:]
    loss = t.nn.functional.binary_cross_entropy_with_logits(logits, val_y.expand_as(logits), reduction='none').mean(1)
    return states.new_full((len(states),), math.log(2)) - loss


def exact_shapley(values, n):
    if values.shape != (2**n,):
        raise ValueError('incomplete coalition table')
    credits = values.new_zeros(n)
    for i in range(n):
        for mask in range(2**n):
            if not mask & (1 << i):
                credits[i] += (values[mask | (1 << i)] - values[mask]) / (n * math.comb(n-1, mask.bit_count()))
    return credits


def marginal_by_size(values, n, player):
    means = values.new_zeros(n)
    for mask in range(2**n):
        if not mask & (1 << player):
            k = mask.bit_count()
            means[k] += (values[mask | (1 << player)] - values[mask]) / math.comb(n-1, k)
    return means


def assemble_result(states, values, n, player):
    full = 2**n - 1
    return {'states': states.tolist(), 'utilities': values.tolist(),
            'shapley': exact_shapley(values, n).tolist(),
            'marginals7': marginal_by_size(values, n, player).tolist(),
            'loo': [float(values[full]-values[full ^ (1 << i)]) for i in range(n)]}
'''


def reveal_methods(commitment_text):
    record=read_record(commitment_text)
    _require(record['status']=='committed','expected a committed forecast')
    return REFERENCE_METHODS


def practice_test(ns,stage):
    """Independent non-target fixtures. Never accesses CASE tensors or target utilities."""
    import torch as t
    if stage=='training':
        x=t.tensor([[.4,-1.],[1.2,.3],[-.5,.8]],dtype=t.float64)
        y=t.tensor([0.,1.,1.],dtype=t.float64)
        masks=t.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,1.],[1.,0.,1.]],dtype=t.float64)
        for policy in ('fixed_n','subset_mean'):
            for steps,lr in [(4,.13),(0,.13),(4,0.)]:
                before=[v.clone() for v in (x,y,masks)]
                actual=ns['train_coalitions'](x,y,masks,steps,lr,policy)
                for a,b in zip((x,y,masks),before,strict=True):
                    t.testing.assert_close(a,b,atol=0,rtol=0)
                expected=[]
                for mask in masks:
                    w=t.zeros(2,dtype=t.float64,requires_grad=True)
                    b=t.zeros((),dtype=t.float64,requires_grad=True)
                    denominator=3 if policy=='fixed_n' else max(float(mask.sum()),1)
                    for _ in range(steps):
                        loss=(mask*t.nn.functional.binary_cross_entropy_with_logits(x@w+b,y,reduction='none')).sum()/denominator
                        gw,gb=t.autograd.grad(loss,(w,b))
                        w=(w-lr*gw).detach().requires_grad_(True)
                        b=(b-lr*gb).detach().requires_grad_(True)
                    expected.append(t.cat([w.detach(),b.detach()[None]]))
                t.testing.assert_close(actual,t.stack(expected),atol=1e-12,rtol=1e-12)
    elif stage=='utility':
        states=t.tensor([[0.,0.,0.],[.3,-.4,.2]],dtype=t.float64)
        vx=t.tensor([[.7,-.2],[-.9,.5],[.3,.8]],dtype=t.float64)
        vy=t.tensor([1.,0.,1.],dtype=t.float64)
        expected=[]
        for w0,w1,b in states.tolist():
            losses=[]
            for (x0,x1),y in zip(vx.tolist(),vy.tolist(),strict=True):
                z=w0*x0+w1*x1+b
                losses.append(max(z,0)+math.log1p(math.exp(-abs(z)))-y*z)
            expected.append(math.log(2)-math.fsum(losses)/3)
        before=[v.clone() for v in (states,vx,vy)]
        t.testing.assert_close(ns['coalition_utilities'](states,vx,vy),t.tensor(expected,dtype=t.float64),atol=1e-12,rtol=1e-12)
        for a,b in zip((states,vx,vy),before,strict=True):
            t.testing.assert_close(a,b,atol=0,rtol=0)
    elif stage in ('shapley','strata','assembly'):
        values=t.tensor([.2,-.7,.4,1.2,-.3,.8,-.9,1.5],dtype=t.float64)
        expected=t.zeros(3,dtype=t.float64)
        strata=[[] for _ in range(3)]
        for order in itertools.permutations(range(3)):
            mask=0
            for k,i in enumerate(order):
                following=mask|(1<<i)
                expected[i]+=(values[following]-values[mask])/6
                if i==2:strata[k].append(float(values[following]-values[mask]))
                mask=following
        target=t.tensor([sum(s)/len(s) for s in strata],dtype=t.float64)
        if stage=='shapley':
            t.testing.assert_close(ns['exact_shapley'](values,3),expected,atol=1e-12,rtol=1e-12)
            with_null=t.tensor([float(m&1)+2*float(bool(m&2)) for m in range(16)],dtype=t.float64)
            t.testing.assert_close(ns['exact_shapley'](with_null,4),t.tensor([1.,2.,0.,0.],dtype=t.float64),atol=1e-12,rtol=1e-12)
        elif stage=='strata':
            t.testing.assert_close(ns['marginal_by_size'](values,3,2),target,atol=1e-12,rtol=1e-12)
        else:
            state=t.zeros(8,3,dtype=t.float64)
            actual=ns['assemble_result'](state,values,3,2)
            _keys(actual,('states','utilities','shapley','marginals7','loo'))
            _require(actual['states']==state.tolist() and actual['utilities']==values.tolist(),'wrong cached result')
            t.testing.assert_close(t.tensor(actual['shapley'],dtype=t.float64),expected,atol=1e-12,rtol=1e-12)
            t.testing.assert_close(t.tensor(actual['marginals7'],dtype=t.float64),target,atol=1e-12,rtol=1e-12)
            _require(actual['loo']==[float(values[7]-values[7^(1<<i)]) for i in range(3)],'wrong LOO baseline')
    else:
        raise ValueError('unknown fixture stage')
    return stage+': independent practice fixture passed'


def _stub(name):
    node=next(n for n in ast.parse(REFERENCE_METHODS).body if n.name==name)
    signature=REFERENCE_METHODS.splitlines()[node.lineno-1]
    return signature+'\n    raise NotImplementedError("Implement '+name+'")\n'


def build_cells():
    cells=[]
    def add(kind,name,text,stage):
        cell={'cell_type':kind,'id':'shapley-'+name,'source':text.strip()+'\n','metadata':{'stage':stage}}
        if kind=='code':cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown','intro','''# [16.9] Blind prediction: change the learning game
**How much does exact Data Shapley credit for the rare positive example change when coalition training divides by coalition size?**
Predict **Δφ7 = φ7(subset mean) − φ7(fixed n)** in mean-validation BCE nats.
A positive prediction means more credit under subset-mean training; a negative one means less.
The estimate may be any finite number. There is no imposed [-1, 1] range.

Both games start logistic regression from zero and take 40 gradient-descent steps at learning rate 0.2.
For a nonempty coalition S, the only change is dividing summed training gradients by 8 versus |S|.
Empty S leaves the model at zero. Validation loss always averages the same eight validation rows.
These are **different learning algorithms/games**, not competing estimators of one fixed game.

You will implement batched coalition training, utility, exact Shapley and coalition-size marginal curves;
then forecast, run paired conditions, see controls and compare the evidence with your explanation.
The shared full-coalition endpoint does not determine how its total utility is distributed.
No target result or figure is prefilled. Practice tests use a separate small invented fixture.

Exposure is `unknown`, `seen`, or `self_reported_not_seen`. Opening equivalent outcomes makes this practice;
a new fingerprint never proves you are unseen. This is an exploratory analytic toy with fresh tiny-model
training after commitment, not independent TEST evaluation or a mastery certificate.''','forecast')
    add('code','setup', '''from pathlib import Path
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'mastery_hub').is_dir())
if str(root) not in sys.path: sys.path.insert(0,str(root))
from mastery_hub.units.shapley_prediction import lesson
import torch as t
from IPython.display import display, Code
t.set_num_threads(1)
CASE = dict(lesson.CASE)
print(CASE)
# Existing installed environment; explicit CPU float64, no downloads or GPU calls.''','setup')
    add('markdown','data','''## Know the players before forecasting
The printed case contains every ordered training/validation row and label. Players 0 and 1 are exact
duplicates. Player 6 has the same features but the opposite label. Player 7 is the rare positive at (0.2, 2.5).
All rows remain fixed. No outcome is needed to inspect the inputs.

A coalition is a bitmask m in 0…255: bit i selects training row i. `masks[m,i]` is its 0/1 membership.
For each coalition fit a separate two-weight plus scalar-bias logistic regressor. Vectorizing all 256
models avoids Python loops over training runs without changing the game.''','practice')
    descriptions={
      'training':'''## Exercise 1: train all coalition models visibly
Implement `train_coalitions(x, y, masks, steps, lr, policy)` returning `(K,d+1)` final states,
weights first, bias last. Allocate zeros with `x.new_zeros`. Logits have shape `(K,n)`.
The per-example derivative is `sigmoid(logits)-y`, multiplied by `masks`. Sum weighted errors
for bias and multiply by x for weights. Divide by n under `fixed_n`; under `subset_mean`,
divide by each mask's sum clamped to at least one. Do not mutate x/y/masks.
<details><summary>Help</summary>The empty mask gives zero summed gradients in every step.
Use one old logits/errors tensor to update both weights and bias. `errors @ x` computes all weight gradients.
The independent fixture includes both denominator policies, empty/singleton/full coalitions, zero steps and zero LR.</details>''',
      'utility':'''## Exercise 2: validation utility with a fixed denominator
Implement `coalition_utilities(states, val_x, val_y)`, returning one utility per fitted state.
Compute mean validation BCE across the same validation rows for every coalition, then subtract from log(2),
the zero model's loss. Never divide validation loss by training-coalition size.
<details><summary>Help</summary>`states[:, :-1] @ val_x.T + states[:, -1:]` gives logits.
Unreduced stable BCE, averaged on the validation axis, avoids overflow and accidental cross-model averaging.</details>''',
      'shapley':'''## Exercise 3: exact Shapley allocation
Implement `exact_shapley(values, n)`. For every player i and predecessor mask S not containing i,
add `[v(S∪{i})−v(S)] / [n × comb(n−1, |S|)]`. Require the complete bitmask table of length 2**n.
Retain `v(empty)`; it need not be zero for an arbitrary practice game.
<details><summary>Help</summary>`mask.bit_count()` is coalition size; `mask | (1 << i)` inserts i.
The fixture averages all six orderings of a three-player interacting game independently.
An additional four-player fixture contains null players.</details>''',
      'strata':'''## Exercise 4: mean marginal by predecessor size
Implement `marginal_by_size(values, n, player)`, an n-vector. At each k, average that player's marginal
contribution over all `comb(n−1,k)` predecessor coalitions of size k. The unweighted mean of these n strata
must equal that player's exact Shapley value. This diagnoses where an algorithm change affects the allocation.
<details><summary>Help</summary>Accumulate each predecessor marginal into index `mask.bit_count()`
after dividing by the number of predecessors at that size. Do not weight sizes by their number of coalitions again.</details>''',
      'assembly':'''## Exercise 5: assemble the result from cached values
Implement `assemble_result(states, values, n, player)`. Return JSON-native fields `states`, `utilities`,
`shapley`, `marginals7`, and `loo`. Use your exact/strata functions; `marginals7` names the selected player's
curve (player 7 in the target; player 2 in the practice fixture). LOO_i is `v(full)−v(full without i)`.
LOO is a baseline from the same cached table and need not allocate total utility correctly.
<details><summary>Help</summary>Use `.tolist()` for tensors. `full ^ (1 << i)` removes i from a full mask.
Every model state and coalition utility must remain in ascending bitmask order for reproducible evidence.</details>'''}
    for stage,name in zip(('training','utility','shapley','strata','assembly'),METHOD_NAMES,strict=True):
        add('markdown',stage+'-instructions',descriptions[stage],'practice')
        extra=' + "\\n\\n" + SHAPLEY_SOURCE + "\\n\\n" + STRATA_SOURCE' if stage=='assembly' else ''
        add('code',stage,f'{stage.upper()}_SOURCE = \'\'\'\n{_stub(name)}\'\'\'\nns = lesson.method_namespace({stage.upper()}_SOURCE{extra}, complete=False)\nprint(lesson.practice_test(ns, {stage!r}))','practice')
        add('markdown',stage+'-expected',f'Expected: `{stage}: independent practice fixture passed`. These tests never evaluate the eight-player target.','practice')
    add('markdown', 'dev-forecast', '''## Exposed DEV ladder — a denominator changes a step (8–12 minutes)
Use the separate three-row training fixture below, one gradient step at lr=.2 from zero.
For singleton {0}, sigmoid(0)−y0=.5 and x0=[.4,−1]. Hand-predict the two weights and bias
under denominator 3 versus denominator 1, then predict their state-norm ratio.

Choose **A:** changing the denominator scales every coalition identically, including full;
or **B:** smaller coalitions receive a larger step, while the full coalition is unchanged.
What observation rules one explanation out? Plot all three players' allocations as well:
a scaled parameter step need not imply equally scaled validation utility or Shapley credit.
These are public three-player DEV observations, not the eight-player target.
Confidence below is an **unscored self-rating**, not a probability of an exact numeric value.''', 'dev')
    add('code', 'dev-forecast-card', '''dev_forecast_ratio = 3.  # Hand prediction: singleton state norm, subset_mean / fixed_n
dev_self_rating = .7  # unscored self-rating
dev_explanation = 'B'
dev_disconfirming_observation = 'Full models agree while singleton states differ.' ''', 'dev')
    add('code', 'dev-paired', r'''import matplotlib.pyplot as plt
dev_source = '\n\n'.join([TRAINING_SOURCE,UTILITY_SOURCE,SHAPLEY_SOURCE,STRATA_SOURCE,ASSEMBLY_SOURCE])
dev_ns = lesson.method_namespace(dev_source)
dev_x = t.tensor([[.4,-1.],[1.2,.3],[-.5,.8]],dtype=t.float64)
dev_y = t.tensor([0.,1.,1.],dtype=t.float64)
dev_masks = t.tensor([[(m>>i)&1 for i in range(3)] for m in range(8)],dtype=t.float64)
dev_results = {}
for policy in ('fixed_n','subset_mean'):
    states = dev_ns['train_coalitions'](dev_x,dev_y,dev_masks,1,.2,policy)
    values = dev_ns['coalition_utilities'](states,dev_x,dev_y)
    credits = dev_ns['exact_shapley'](values,3)
    dev_results[policy] = (states,values,credits)
dev_zero_states = dev_ns['train_coalitions'](dev_x,dev_y,dev_masks,1,0.,'subset_mean')
dev_zero_credit = dev_ns['exact_shapley'](dev_ns['coalition_utilities'](dev_zero_states,dev_x,dev_y),3)
fig,axes = plt.subplots(1,2,figsize=(11,3),layout='constrained')
for policy,(states,values,credits) in dev_results.items():
    axes[0].plot(range(8),states.norm(dim=1).tolist(),'o-',label=policy)
    axes[1].plot(range(3),credits.tolist(),'o-',label=policy)
axes[0].plot(range(8),dev_zero_states.norm(dim=1).tolist(),'--',label='Zero learning rate')
axes[1].plot(range(3),dev_zero_credit.tolist(),'--',label='Zero learning rate')
axes[0].set(xlabel='Coalition bitmask',ylabel='State norm',title='DEV one-step learning',xticks=range(8))
axes[1].set(xlabel='DEV player',ylabel='Validation BCE improvement (nats)',title='Exact DEV allocation',xticks=range(3))
for ax in axes: ax.legend()
display(fig)
dev_observed_ratio = float(dev_results['subset_mean'][0][1].norm()/dev_results['fixed_n'][0][1].norm())
print('DEV singleton norm ratio:',dev_observed_ratio)
print('DEV ratio absolute forecast error:',abs(dev_observed_ratio-dev_forecast_ratio))
t.testing.assert_close(dev_results['subset_mean'][0][7],dev_results['fixed_n'][0][7])
assert dev_zero_states.count_nonzero() == 0''', 'dev')
    add('markdown', 'dev-reflection', '''### Worked DEV reflection / self-check
The singleton states are [−.013333,.033333,−.033333] and [−.04,.1,−.1]; their norm ratio
is 3. A mistaken forecast of 1 has absolute ratio error 2. The full-coalition state is identical:
that observation rules A out. A good reflection cites singleton AND full controls, then explains
why nonlinear BCE and allocations over all predecessor sizes prevent multiplying every credit by 3.
The zero-LR allocation stays zero. These checks do not establish that either learning game is better.

### Try It Yourself — changed-condition DEV transfer
Change ONLY the number of training steps from one to two. Freeze data, masks, zero
initialization, lr=.2, validation denominator and policies. Predict the singleton state-norm
ratio before running. Does the one-step explanation imply exactly3 after a second nonlinear
step? Distinguish local gradient scaling from final model scaling.
The next cell trains a new DEV condition; it does not just inspect an already stored player.''', 'dev')
    add('code', 'dev-transfer', '''dev_transfer_steps = 2
dev_transfer_forecast = 3.  # Deliberate one-step extrapolation; replace before running.
dev_transfer_results = {}
for policy in ('fixed_n','subset_mean'):
    states=dev_ns['train_coalitions'](dev_x,dev_y,dev_masks,dev_transfer_steps,.2,policy)
    values=dev_ns['coalition_utilities'](states,dev_x,dev_y)
    dev_transfer_results[policy]=(states,values,dev_ns['exact_shapley'](values,3))
dev_transfer_observed=float(dev_transfer_results['subset_mean'][0][1].norm()/dev_transfer_results['fixed_n'][0][1].norm())
print('DEV two-step singleton ratio:',dev_transfer_observed,'absolute forecast error:',abs(dev_transfer_observed-dev_transfer_forecast))
assert abs(dev_transfer_observed-3.) > 1e-8
t.testing.assert_close(dev_transfer_results['subset_mean'][0][7],dev_transfer_results['fixed_n'][0][7])
fig,axes=plt.subplots(1,2,figsize=(11,3),layout='constrained')
axes[0].plot([1,2],[dev_observed_ratio,dev_transfer_observed],'o-',label='Observed singleton norm ratio')
axes[0].axhline(3,color='gray',linestyle='--',label='Exact step-1 extrapolation')
axes[0].set(xlabel='Training steps',ylabel='subset_mean / fixed_n state norm',title='Observed DEV duration transfer',xticks=[1,2])
for policy in ('fixed_n','subset_mean'):
    axes[1].plot(range(3),dev_transfer_results[policy][2].tolist(),'o-',label=policy+' two steps')
axes[1].plot(range(3),dev_zero_credit.tolist(),'--',label='Zero-LR control')
axes[1].set(xlabel='DEV player',ylabel='Shapley credit (BCE nats)',title='New DEV allocations',xticks=range(3))
for ax in axes:ax.legend()
display(fig)
dev_revision = ''  # Why does a second gradient see different states under the two policies?''', 'dev')
    add('markdown','dev-transfer-check','''<details><summary>DEV transfer self-check</summary>
The ratio is no longer exactly3. At step2 the policies evaluate sigmoid at different
parameters: n/|S| scales a gradient at the same state, not every final state after nonlinear
training. Cite the measured discrepancy, the matching full-coalition control, and whether
allocation changed. No universal claim about rare-point credit follows.</details>''','dev')
    add('markdown','freeze-instructions','''## Freeze exact inputs, then forecast
Finish each implementation and its separate fixture first. Freezing hashes the exact ordered inputs,
initial state, coalition masks, case rules, method text, helper/test sources and environment; it performs **no training or utility evaluation**.
Record your own numerical estimate, confidence and causal reason. Think about how a player's marginal
changes in small coalitions versus large ones. Name an observation that would challenge your explanation.
Editing source/runtime after freezing requires a new commitment. Local hashes detect accidental drift;
they are not exam security or proof of exposure history.''','commit')
    add('code','freeze',r'''METHOD_SOURCE = '\n\n'.join([TRAINING_SOURCE, UTILITY_SOURCE, SHAPLEY_SOURCE, STRATA_SOURCE, ASSEMBLY_SOURCE])
identity = lesson.freeze_inputs(METHOD_SOURCE)
print('Frozen exact inputs and source; no target model has been trained.')''','freeze')
    add('code','commit', '''prediction = None  # Your finite Δφ7 in mean-validation BCE nats
confidence = None  # Unscored reasoning self-rating in [0,1]; not a probability
reasoning = ''  # Causal explanation and what could contradict it
prior_exposure = 'unknown'  # unknown / seen / self_reported_not_seen
commitment = lesson.commit_prediction(identity, METHOD_SOURCE, prediction, confidence, reasoning, prior_exposure)
print('Committed. No target training or utility evaluation has run.')''','commit')
    add('markdown','postcommit','''## Run the fixed paired experiment after commitment
The explicit run repeats the independent practice checks, then fits four 256-coalition games:
both denominator policies and their label-0↔6 relocation controls. Two additional zero-learning-rate
batches check inert training. Full-coalition states/utility must match, empty models stay zero,
duplicates share credit, exact allocations satisfy efficiency and relocation swaps only the corresponding credits.
There is one cooperative 120-second CPU budget, checked between stages; slow learner functions are
rejected on return rather than forcibly interrupted. No GPU/download is used. Preserve exceptions as failed attempts.
Reference implementation text is optional after commitment and contains no measured outcome.
Adopting changed code requires freezing again and creating a separate forecast.''','postcommit')
    add('code','solutions', '''# Optional postcommit implementation help, no training:
# display(Code(lesson.reveal_methods(commitment), language='python'))''','solutions')
    add('code','run', '''try:
    receipt = lesson.run_experiment(commitment, METHOD_SOURCE)
except lesson.ExperimentFailure as failed:
    failed_record = failed.record_text  # Preserves actual measurements and original commitment.
    print(lesson.replay_failure(failed_record))
    # Optional: lesson.export_record(failed_record, Path.home()/'shapley-failed-attempt-1.json')
    raise  # Stop here; replay/export failure without rerunning training.
observation = lesson.replay(receipt)
display(lesson.plot_receipt(receipt))
print('Observed Δφ7:', observation['delta_phi7'], 'nats')
print('Absolute forecast error:', observation['absolute_forecast_error'])
print('Controls:', observation['controls'])''','run')
    add('markdown','interpretation','''## Explain the allocation, not just the winner
Read the paired bars alongside the marginal-by-size curves. Which predecessor sizes changed most?
Does that support your explanation? Both games share a full-coalition model/utility, but their allocation
across players can differ. Compare LOO without treating it as an efficiency-preserving allocation.
Check the empty/zero-LR, duplicate, relocation and efficiency controls before interpreting the forecast error.
A passing control does not prove the game is a good definition of real-world data value.

This is an eight-point unregularized logistic toy, 40 fixed steps, one learning rate and one validation set.
No random cohort, confidence interval, independent TEST performance, universal pruning conclusion or human
mastery claim follows from this exact calculation. Different training schedules define further games.
The [canonical 16.9 reference](../../../chapter16_shapley_attribution_baselines/exercises/part9_data_shapley/16.9_Data_Shapley_in_One_Training_Run_solutions.ipynb)
offers retraining and in-run accounting as distinct concepts; this challenge isolates the retraining game's denominator only.
After recording your forecast, read Ghorbani & Zou's [Data Shapley: Equitable Valuation of Data for Machine Learning](https://arxiv.org/abs/1904.02868).
Its retraining-based data valuation is distinct from the In-Run attribution later in the canonical section.''','reflect')
    add('code','play', '''# Try It Yourself: inspect another stored player without fitting anything.
player = 0
obs = lesson.replay(receipt)
for policy in ('fixed_n','subset_mean'):
    result = obs['results'][policy]
    print(policy, 'Shapley:', result['shapley'][player], 'LOO:', result['loo'][player])
reflection = ''  # What matched, what failed, and which control informed the explanation?
transfer_question = ''  # Change ONE condition and retain metric/units/denominator definitions.
next_prediction = ''  # New forecast BEFORE a separately authorized experiment.
transfer_confidence = None
transfer_reasoning = ''
# attempt = lesson.reflect(receipt, reflection, transfer_question, next_prediction, transfer_confidence, transfer_reasoning)''','play')
    add('markdown','export-instructions','''## Keep forecast, result and transfer together
Reflection wraps the exact original receipt, preserving the original forecast and machine observation.
Save only by explicit opt-in. Export creates a new private file (0600) and refuses to overwrite.
Replay/plotting read stored values; they evaluate no student code, training loop, torch model or current environment.
Seals validate structure and accidental drift, not an adversarially edited local record.''','replay')
    add('code','export', '''# Optional exports to explicit new filenames:
# lesson.export_record(receipt, Path.home()/'shapley-prediction-receipt-1.json')
# lesson.export_attempt(attempt, Path.home()/'shapley-prediction-reflection-1.json')
# Later, model-free replay:
# saved = Path.home().joinpath('shapley-prediction-receipt-1.json').read_text()
# display(lesson.plot_receipt(saved))''','replay')
    return cells
