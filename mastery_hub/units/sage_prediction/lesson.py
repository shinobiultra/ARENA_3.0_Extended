"""Blind SAGE Gaussian restriction challenge; import/freeze never solve the target.

Learners own every numerical method. Stored receipts bind actual arrays and retain
failures. Seals detect accidental drift, not adversarial editing or prior exposure.
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
TESTS = ROOT / 'tests/mastery/test_sage_prediction.py'
SCHEMA = 'arena-blind-sage-prediction'
VERSION = 1
ASSESSMENT = 'local_self_report_no_unseen_certification'
CASE = {
    'id': '16.4-gaussian-restriction-v1',
    'covariance': [[1.,.6,-.3],[.6,1.,.14],[-.3,.14,1.]],
    'truth': [1.,-.5,.25], 'model': [.75,.25,-.5], 'noise_variance': .25,
    'features': ['A','B','C'], 'player': 1, 'mean': [0.,0.,0.],
    'dtype': 'torch.float64', 'device': 'cpu', 'threads': 1,
    'coalition_order': 'ascending bitmask; A bit 0, B bit 1, C bit 2',
    'rules': ['conditional','marginal'],
    'risk': '(truth-restricted_model)^T covariance (truth-restricted_model) + independent noise variance',
    'game': 'risk(empty) minus risk(S), squared-target units',
    'allocation': 'exact factorial-weight Shapley over all eight coalitions',
    'forecast': 'phi_B(conditional) minus phi_B(marginal), signed squared-target units',
    'controls': ['empty_full_agreement','efficiency','identity_covariance','zero_predictor',
                 'noise_shift','feature_permutation','six_ordering_oracle'],
    'permutation': [2,0,1], 'noise_control_variance': .75,
    'transfer': 'covariance only becomes (covariance + identity)/2; separate commitment',
    'budget_seconds': 120, 'evidence_class': 'analytic_toy',
    'data_role': 'fixed synthetic diagnostic; no independent TEST claim',
}
METHOD_NAMES = ('restriction_coefficients','risk_game','exact_shapley','marginal_by_size','assemble_result')
STAGES = ('restriction','risk','shapley','strata','assembly')
CONDITIONS = ('conditional','marginal','identity_conditional','identity_marginal',
              'zero_conditional','zero_marginal','noise_conditional','noise_marginal',
              'permuted_conditional','permuted_marginal')
_CASE_DEFINITION = json.dumps(CASE, sort_keys=True, allow_nan=False)
RECORD_LIMIT = 4_000_000
ATTEMPT_LIMIT = 6_000_000
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



def _case(variant='primary', *, frozen=False):
    _require(variant in ('primary','transfer'),'unknown case variant')
    case=json.loads(_CASE_DEFINITION) if frozen else deepcopy(CASE)
    case['variant']=variant
    if variant=='transfer':
        case['covariance']=[[(v+float(i==j))/2 for j,v in enumerate(row)]
                            for i,row in enumerate(case['covariance'])]
    return case


def _layout(case):
    return {'covariance':([3,3],case['covariance']), 'truth':([3],case['truth']),
            'model':([3],case['model']), 'mean':([3],case['mean']),
            'noise_variance':([1],[case['noise_variance']]),
            'coalition_masks':([8,3],[[float((m>>i)&1) for i in range(3)] for m in range(8)])}


def environment_identity():
    import torch
    return {'python':platform.python_version(),'torch':torch.__version__,
            'platform':platform.platform(),'byteorder':sys.byteorder,
            'threads':torch.get_num_threads(),'default_dtype':str(torch.get_default_dtype()),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'default_device':str(torch.get_default_device())}


def _method_tree(source,complete=True):
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
    for node in tree.body:
        arguments=node.args.posonlyargs+node.args.args+node.args.kwonlyargs+[a for a in (node.args.vararg,node.args.kwarg) if a]
        _require(not node.decorator_list and not node.returns and not node.type_params
                 and not any(a.annotation for a in arguments),'use plain unannotated function definitions')
        _require(all(isinstance(v,ast.Constant) for v in node.args.defaults+list(filter(None,node.args.kw_defaults))),
                 'defaults must be literal constants')
    return tree


def method_namespace(source,complete=True):
    """Define plain functions only; no learner body executes here."""
    import torch as t
    _method_tree(source,complete)
    namespace={'t':t,'math':math}
    exec(compile(source,'<student SAGE methods>','exec'),namespace)
    return namespace


def freeze_inputs(method_source,variant='primary'):
    """Hash public inputs, source and runtime; no solving, risk or allocation work."""
    _method_tree(method_source)
    env=environment_identity()
    _require(env['threads']==1 and env['default_device']=='cpu','set one CPU thread and CPU default device before freezing')
    case=_case(variant)
    return {'case_sha256':digest(case),'method_sha256':hashlib.sha256(method_source.encode()).hexdigest(),
            'helper_sha256':_file(__file__),'tests_sha256':_file(TESTS),'environment':env,
            'inputs':{name:{'shape':shape,'dtype':case['dtype'],'sha256':_tensor_hash(rows)}
                      for name,(shape,rows) in _layout(case).items()}}


def _validate_identity(identity,source,case):
    _keys(identity,('case_sha256','method_sha256','helper_sha256','tests_sha256','environment','inputs'))
    for key in ('case_sha256','method_sha256','helper_sha256','tests_sha256'):
        _sha(identity[key])
    _require(identity['case_sha256']==digest(case) and identity['method_sha256']==hashlib.sha256(source.encode()).hexdigest(),
             'case or method identity mismatch')
    env=identity['environment']
    _keys(env,('python','torch','platform','byteorder','threads','default_dtype','deterministic_algorithms','default_device'))
    for key in ('python','torch','platform','default_dtype'):
        _text(env[key])
    _require(env['byteorder'] in ('little','big') and type(env['threads']) is int and env['threads']==1
             and type(env['deterministic_algorithms']) is bool and env['default_device']=='cpu','invalid environment identity')
    _keys(identity['inputs'],_layout(case))
    for name,(shape,rows) in _layout(case).items():
        _require(identity['inputs'][name]=={'shape':shape,'dtype':case['dtype'],'sha256':_tensor_hash(rows)},
                 'ordered input shape/dtype/bytes mismatch')


def commit_prediction(identity,method_source,prediction,confidence,reasoning,prior_exposure='unknown',*,variant='primary',parent_attempt=None):
    payload={'schema':SCHEMA,'schema_version':VERSION,'status':'committed','case':_case(variant),
             'identity':deepcopy(identity),'method_source':method_source,'prediction':prediction,
             'confidence':confidence,'reasoning':reasoning,'prior_exposure':prior_exposure,
             'assessment':ASSESSMENT,'parent_attempt':parent_attempt}
    text=_seal(payload)
    read_record(text)
    return text


def _close(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11)


def _vector(value,n):
    _require(type(value) is list and len(value)==n,'invalid vector shape')
    for v in value:
        _number(v)


def _matrix(value,n,d):
    _require(type(value) is list and len(value)==n,'invalid matrix shape')
    for row in value:
        _vector(row,d)


def _condition(case,label):
    _require(label in CONDITIONS,'unknown measured condition')
    out=deepcopy(case)
    out['rule']='conditional' if label.endswith('conditional') else 'marginal'
    if label.startswith('identity_'):
        out['covariance']=[[float(i==j) for j in range(3)] for i in range(3)]
    elif label.startswith('zero_'):
        out['model']=[0.,0.,0.]
    elif label.startswith('noise_'):
        out['noise_variance']=case['noise_control_variance']
    elif label.startswith('permuted_'):
        order=case['permutation']
        out['covariance']=[[case['covariance'][i][j] for j in order] for i in order]
        for name in ('truth','model','features'):
            out[name]=[case[name][i] for i in order]
        out['player']=order.index(case['player'])
    return out


def _validate_coefficients(condition,mask,coefficients):
    """Check the defining linear equations against saved coefficients; never solve."""
    selected=[i for i in range(3) if mask&(1<<i)]
    for i in range(3):
        if i not in selected:
            _require(_close(coefficients[i],0),'nonselected coefficient is nonzero')
    if len(selected)==3 or condition['rule']=='marginal':
        _require(all(_close(coefficients[i],condition['model'][i]) for i in selected),
                 'restricted coefficients differ from predictor')
    else:
        covariance=condition['covariance']
        for i in selected:
            left=math.fsum(covariance[i][j]*coefficients[j] for j in selected)
            right=math.fsum(covariance[i][j]*condition['model'][j] for j in range(3))
            _require(_close(left,right),'conditional coefficients violate independent linear equations')


def _reference_credits(game):
    credits=[0.,0.,0.]
    for order in itertools.permutations(range(3)):
        mask=0
        for i in order:
            added=mask|(1<<i)
            credits[i]+=(game[added]-game[mask])/6
            mask=added
    return credits


def _strata(game,player):
    return [math.fsum(game[m|(1<<player)]-game[m] for m in range(8)
                      if not m&(1<<player) and m.bit_count()==k)/math.comb(2,k) for k in range(3)]


def _validate_partial(measured):
    _require(type(measured) is dict and set(measured) in (
        {'coefficients','coefficients_sha256'},
        {'coefficients','coefficients_sha256','risks','game','risks_sha256','game_sha256'}),'malformed partial measurement')
    _matrix(measured['coefficients'],8,3)
    _require(measured['coefficients_sha256']==_tensor_hash(measured['coefficients']),'coefficient hash mismatch')
    if 'risks' in measured:
        for key in ('risks','game'):
            _vector(measured[key],8)
            _require(measured[key+'_sha256']==_tensor_hash(measured[key]),key+' hash mismatch')


def _validate_result(result,condition):
    _keys(result,('coefficients','risks','game','shapley','marginals_b',
                  'coefficients_sha256','risks_sha256','game_sha256'))
    _validate_partial({k:result[k] for k in ('coefficients','risks','game','coefficients_sha256','risks_sha256','game_sha256')})
    _vector(result['shapley'],3);_vector(result['marginals_b'],3)
    for mask,(coeff,risk,value) in enumerate(zip(result['coefficients'],result['risks'],result['game'],strict=True)):
        _validate_coefficients(condition,mask,coeff)
        residual=[a-b for a,b in zip(condition['truth'],coeff,strict=True)]
        expected_risk=math.fsum(residual[i]*condition['covariance'][i][j]*residual[j]
                                for i in range(3) for j in range(3))+condition['noise_variance']
        _require(_close(risk,expected_risk),'risk differs from stored coefficients')
        _require(_close(value,result['risks'][0]-risk),'game differs from actual risks')
    _require(all(_close(a,b) for a,b in zip(result['shapley'],_reference_credits(result['game']),strict=True)),
             'Shapley differs from independent six-ordering oracle')
    _require(all(_close(a,b) for a,b in zip(result['marginals_b'],_strata(result['game'],condition['player']),strict=True)),
             'marginal strata mismatch')
    _require(_close(math.fsum(result['shapley']),result['game'][-1]-result['game'][0]),'efficiency failed')
    _require(_close(math.fsum(result['marginals_b'])/3,result['shapley'][condition['player']]),'strata mean mismatch')


def _control_checks(results,case):
    checks={}
    for rule in ('conditional','marginal'):
        base=results[rule]
        identity=results['identity_'+rule]
        zero=results['zero_'+rule]
        noisy=results['noise_'+rule]
        permuted=results['permuted_'+rule]
        checks[rule+'_efficiency']=_close(math.fsum(base['shapley']),base['game'][-1]-base['game'][0])
        checks[rule+'_zero_predictor']=all(_close(v,0) for v in zero['game']+zero['shapley'])
        checks[rule+'_noise_shift']=all(_close(b-a,case['noise_control_variance']-case['noise_variance']) for a,b in zip(base['risks'],noisy['risks'],strict=True)) and all(_close(a,b) for key in ('game','shapley') for a,b in zip(base[key],noisy[key],strict=True))
        checks[rule+'_feature_permutation']=all(_close(permuted['shapley'][i],base['shapley'][case['permutation'][i]]) for i in range(3))
        checks[rule+'_six_ordering_oracle']=all(_close(a,b) for a,b in zip(base['shapley'],_reference_credits(base['game']),strict=True))
    a,b=results['conditional'],results['marginal']
    checks['empty_full_agreement']=all(_close(x,y) for m in (0,7) for x,y in zip(a['coefficients'][m],b['coefficients'][m],strict=True)) and all(_close(a['risks'][m],b['risks'][m]) for m in (0,7))
    a,b=results['identity_conditional'],results['identity_marginal']
    checks['identity_covariance']=all(_close(x,y) for key in ('risks','game','shapley') for x,y in zip(a[key],b[key],strict=True))
    return checks


def _validate_failure(failure,case):
    _keys(failure,('kind','stage','error_type','message','elapsed_seconds','measured','results','controls','allocations'))
    _require(failure['kind']=='failed_cpu_analytic','invalid failure classification')
    for key in ('stage','error_type','message'):
        _text(failure[key])
    _number(failure['elapsed_seconds'],0)
    _require(type(failure['measured']) is dict and set(failure['measured'])<=set(CONDITIONS),'unknown partial condition')
    for measured in failure['measured'].values():
        _validate_partial(measured)
    _require(type(failure['results']) is dict and set(failure['results'])<=set(failure['measured']),'unknown completed condition')
    for label,result in failure['results'].items():
        _validate_result(result,_condition(case,label))
        _require(all(result[k]==v for k,v in failure['measured'][label].items()),'failure lost actual measurements')
    _require(type(failure['allocations']) is dict and set(failure['allocations'])<=set(failure['measured']),
             'unknown returned allocation condition')
    for returned in failure['allocations'].values():
        _keys(returned,('shapley','marginals_b'))
        _vector(returned['shapley'],3);_vector(returned['marginals_b'],3)
    if failure['controls'] is not None:
        _keys(failure['results'],CONDITIONS)
        _require(failure['controls']==_control_checks(failure['results'],case),'invalid failed controls')


def read_record(text):
    """Strict portable parser; validates stored arrays, executes no student methods."""
    out=_parse(text,RECORD_LIMIT)
    _require(out.get('schema')==SCHEMA and type(out.get('schema_version')) is int and out.get('schema_version')==VERSION,
             'unsupported schema')
    if out.get('status')=='committed':
        _keys(out,('schema','schema_version','status','case','identity','method_source','prediction','confidence',
                   'reasoning','prior_exposure','assessment','parent_attempt'))
        variant=out['case'].get('variant') if type(out['case']) is dict else None
        _require(digest(out['case'])==digest(_case(variant,frozen=True)),'unsupported case')
        _method_tree(out['method_source']);_validate_identity(out['identity'],out['method_source'],out['case'])
        _number(out['prediction']);_number(out['confidence'],0,1);_text(out['reasoning'])
        _require(out['prior_exposure'] in ('unknown','seen','self_reported_not_seen') and out['assessment']==ASSESSMENT,
                 'invalid exposure or assessment')
        if variant=='primary':
            _require(out['parent_attempt'] is None,'primary commitment cannot have a parent')
        else:
            attempt=read_attempt(out['parent_attempt'])
            _require(attempt['transfer_prediction']==out['prediction'] and attempt['confidence']==out['confidence']
                     and attempt['reasoning']==out['reasoning'],'transfer forecast changed from reflection')
    elif out.get('status') in ('observed','failed'):
        _keys(out,('schema','schema_version','status','commitment','commitment_sha256',
                   'observation' if out['status']=='observed' else 'failure'))
        _require(type(out['commitment']) is str and hashlib.sha256(out['commitment'].encode()).hexdigest()==out['commitment_sha256'],
                 'commitment link mismatch')
        _require(_parse(out['commitment'],RECORD_LIMIT).get('status')=='committed','expected original commitment')
        committed=read_record(out['commitment']);case=committed['case']
        if out['status']=='failed':
            _validate_failure(out['failure'],case)
            return out
        obs=out['observation']
        _keys(obs,('kind','results','controls','delta_phi_b','absolute_forecast_error','elapsed_seconds','condition_count','coalition_count'))
        _require(obs['kind']=='fresh_cpu_analytic' and type(obs['condition_count']) is int and obs['condition_count']==10
                 and type(obs['coalition_count']) is int and obs['coalition_count']==80,'invalid execution inventory')
        _number(obs['elapsed_seconds'],0,case['budget_seconds']);_require(obs['elapsed_seconds']>0,'invalid elapsed time')
        _keys(obs['results'],CONDITIONS)
        for label,result in obs['results'].items():
            _validate_result(result,_condition(case,label))
        controls=_control_checks(obs['results'],case)
        _require(obs['controls']==controls and all(type(v) is bool and v for v in obs['controls'].values()),'control failure')
        expected=obs['results']['conditional']['shapley'][1]-obs['results']['marginal']['shapley'][1]
        _number(obs['delta_phi_b']);_number(obs['absolute_forecast_error'],0)
        _require(_close(obs['delta_phi_b'],expected) and _close(obs['absolute_forecast_error'],abs(expected-committed['prediction'])),
                 'forecast scoring mismatch')
    else:
        raise ValueError('unsupported record status')
    return out


def assert_current(record_text,method_source):
    record=read_record(record_text)
    _require(record['status']=='committed','already observed: replay or create a separate commitment')
    _require(method_source==record['method_source'],'stale method text; freeze and commit again')
    _require(freeze_inputs(method_source,record['case']['variant'])==record['identity'],
             'stale inputs, helper, tests or environment')
    return record


class ExperimentFailure(ValueError):
    """Unscored failure with retained actual arrays and original commitment."""
    def __init__(self,record_text):
        self.record_text=record_text
        failure=read_record(record_text)['failure']
        super().__init__(failure['stage']+': '+failure['message'])


def run_experiment(commitment_text,method_source):
    started=time.monotonic()
    committed=assert_current(commitment_text,method_source)
    capture={'stage':'setup','measured':{},'results':{},'controls':None,'allocations':{}}
    try:
        return _execute_experiment(commitment_text,method_source,started,capture,committed)
    except Exception as error:
        failure={'kind':'failed_cpu_analytic','error_type':type(error).__name__,
                 'message':(str(error) or type(error).__name__)[:20_000],
                 'elapsed_seconds':time.monotonic()-started,**capture}
        text=_seal({'schema':SCHEMA,'schema_version':VERSION,'status':'failed','commitment':commitment_text,
                    'commitment_sha256':hashlib.sha256(commitment_text.encode()).hexdigest(),'failure':failure})
        read_record(text)
        raise ExperimentFailure(text) from error


def _execute_experiment(commitment_text,source,started,capture,committed):
    import torch as t
    case=committed['case'];frozen=committed['identity']
    def deadline():
        if time.monotonic()-started>case['budget_seconds']:
            raise TimeoutError('CPU budget exceeded; no successful receipt')
    def current():
        _require(freeze_inputs(source,case['variant'])==frozen,'method changed frozen inputs, helper, tests or environment')
    def snapshot(tensors):
        return [(str(v.dtype),str(v.device),list(v.shape),_tensor_hash(v.detach().cpu().tolist())) for v in tensors]
    def call(stage,function,args):
        deadline();current();capture['stage']=stage
        before=snapshot(args)
        result=function(*args)
        return result,before
    def unchanged(args,before):
        _require(snapshot(args)==before,'method mutated actual input arrays')
        current();deadline()
    def tensor(value,shape):
        _require(isinstance(value,t.Tensor) and list(value.shape)==shape and value.dtype==t.float64
                 and value.device.type=='cpu' and bool(t.isfinite(value).all()),'expected finite CPU float64 returned tensor')
    deadline();namespace=method_namespace(source)
    for stage in STAGES:
        capture['stage']='practice '+stage
        current();practice_test(namespace,stage);current();deadline()
    for label in CONDITIONS:
        condition=_condition(case,label)
        covariance=t.tensor(condition['covariance'],dtype=t.float64,device='cpu')
        model=t.tensor(condition['model'],dtype=t.float64,device='cpu')
        truth=t.tensor(condition['truth'],dtype=t.float64,device='cpu')
        # Literal rule/scalar args do not expose mutable source objects.
        args=(covariance,model)
        output,before=call(label+': restriction',lambda c,b:namespace['restriction_coefficients'](c,b,condition['rule']),args)
        tensor(output,[8,3])
        measured={'coefficients':output.detach().tolist()}
        measured['coefficients_sha256']=_tensor_hash(measured['coefficients'])
        capture['measured'][label]=measured
        unchanged(args,before)
        coefficients=output.detach().clone()
        args=(coefficients,covariance,truth)
        pair,before=call(label+': risk',lambda c,s,a:namespace['risk_game'](c,s,a,condition['noise_variance']),args)
        _require(type(pair) in (tuple,list) and len(pair)==2,'risk_game must return risks and game')
        risks,game=pair;tensor(risks,[8]);tensor(game,[8])
        measured.update(risks=risks.detach().tolist(),game=game.detach().tolist())
        measured.update(risks_sha256=_tensor_hash(measured['risks']),game_sha256=_tensor_hash(measured['game']))
        unchanged(args,before)
        _require(coefficients.tolist()==measured['coefficients'],'risk changed actual coefficients')
        # Assembly receives disposable clones; neither aliases nor fabrication can rewrite measurements.
        args=(coefficients.clone(),risks.detach().clone(),game.detach().clone())
        result,before=call(label+': assembly',lambda c,r,g:namespace['assemble_result'](c,r,g,3,condition['player']),args)
        # Preserve finite, correctly shaped returned allocations as unscored
        # debugging evidence even if their mathematical validation later fails.
        if type(result) is dict and 'shapley' in result and 'marginals_b' in result:
            try:
                _vector(result['shapley'],3);_vector(result['marginals_b'],3)
            except ValueError:
                pass
            else:
                capture['allocations'][label]=deepcopy({k:result[k] for k in ('shapley','marginals_b')})
        unchanged(args,before)
        _require(type(result) is dict and set(result)=={'coefficients','risks','game','shapley','marginals_b'},'invalid student assembly')
        for key in ('coefficients','risks','game'):
            _require(result[key]==measured[key],'assembly differs from actual measured arrays')
        result=deepcopy(result)
        for key in ('coefficients_sha256','risks_sha256','game_sha256'):
            result[key]=measured[key]
        _validate_result(result,condition)
        capture['results'][label]=result
        current();deadline()
    capture['stage']='paired controls'
    controls=_control_checks(capture['results'],case);capture['controls']=controls
    _require(all(controls.values()),'control failure: '+str(controls))
    delta=capture['results']['conditional']['shapley'][1]-capture['results']['marginal']['shapley'][1]
    text=_seal({'schema':SCHEMA,'schema_version':VERSION,'status':'observed','commitment':commitment_text,
                'commitment_sha256':hashlib.sha256(commitment_text.encode()).hexdigest(),
                'observation':{'kind':'fresh_cpu_analytic','results':capture['results'],'controls':controls,
                               'delta_phi_b':delta,'absolute_forecast_error':abs(delta-committed['prediction']),
                               'elapsed_seconds':time.monotonic()-started,'condition_count':10,'coalition_count':80}})
    read_record(text);current();deadline()
    return text


def replay(receipt_text):
    record=read_record(receipt_text)
    _require(record['status']=='observed','no observation to replay')
    return deepcopy(record['observation'])


def replay_failure(failure_text):
    record=read_record(failure_text)
    _require(record['status']=='failed','expected an unscored failure')
    return deepcopy(record['failure'])


def plot_receipt(receipt_text):
    """Plot already stored arrays; no student evaluation or new experiment."""
    import matplotlib.pyplot as plt
    observation=replay(receipt_text)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for offset,rule in ((-.2,'conditional'),(.2,'marginal')):
        result=observation['results'][rule]
        axes[0].bar([i+offset for i in range(3)],result['shapley'],width=.38,label=rule)
        axes[1].plot(range(3),result['marginals_b'],'o-',label=rule)
    axes[0].set(xticks=range(3),xticklabels=['A','B','C'],xlabel='Feature',ylabel='Exact SAGE (squared-target units)',title='One predictor, two missing-input rules')
    axes[1].set(xticks=range(3),xlabel='Predecessor coalition size',ylabel='Mean B marginal (squared-target units)',title='Three strata average to B SAGE')
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


def reflect(receipt_text,explanation,transfer_prediction,confidence,reasoning):
    observed=read_record(receipt_text)
    _require(observed['status']=='observed' and read_record(observed['commitment'])['case']['variant']=='primary',
             'transfer reflection requires the primary observation')
    text=_seal({'schema':SCHEMA+'-reflection','schema_version':VERSION,'receipt':receipt_text,
                'receipt_sha256':hashlib.sha256(receipt_text.encode()).hexdigest(),
                'explanation':explanation,'transfer_case':_case('transfer',frozen=True),
                'transfer_prediction':transfer_prediction,'confidence':confidence,'reasoning':reasoning})
    read_attempt(text)
    return text


def read_attempt(text):
    out=_parse(text,ATTEMPT_LIMIT)
    _keys(out,('schema','schema_version','receipt','receipt_sha256','explanation','transfer_case',
               'transfer_prediction','confidence','reasoning'))
    _require(out['schema']==SCHEMA+'-reflection' and type(out['schema_version']) is int and out['schema_version']==VERSION,
             'unsupported reflection schema')
    _require(type(out['receipt']) is str and hashlib.sha256(out['receipt'].encode()).hexdigest()==out['receipt_sha256'],
             'receipt link mismatch')
    parent=_parse(out['receipt'],RECORD_LIMIT)
    _require(parent.get('status')=='observed' and type(parent.get('commitment')) is str,'expected primary observation')
    _require(_parse(parent['commitment'],RECORD_LIMIT).get('case',{}).get('variant')=='primary','reflection cannot recurse through transfer')
    replay(out['receipt'])
    _require(digest(out['transfer_case'])==digest(_case('transfer',frozen=True)),'unsupported transfer condition')
    _number(out['transfer_prediction']);_number(out['confidence'],0,1)
    _text(out['explanation']);_text(out['reasoning'])
    return out


def commit_transfer(attempt_text,identity,method_source,prior_exposure='unknown'):
    attempt=read_attempt(attempt_text)
    return commit_prediction(identity,method_source,attempt['transfer_prediction'],attempt['confidence'],
                             attempt['reasoning'],prior_exposure,variant='transfer',parent_attempt=attempt_text)


def export_attempt(text,destination):
    read_attempt(text)
    return _exclusive(text,destination)


REFERENCE_METHODS = '''def restriction_coefficients(covariance, model, rule):
    d = len(model)
    if rule not in ('conditional', 'marginal'):
        raise ValueError('unknown missing-input rule')
    if not 1 <= d <= 8 or covariance.shape != (d, d) or model.shape != (d,):
        raise ValueError('incompatible matrix/vector shapes')
    if not bool(t.isfinite(covariance).all()) or not bool(t.isfinite(model).all()):
        raise ValueError('finite Gaussian inputs required')
    if not t.allclose(covariance, covariance.T) or bool((t.linalg.eigvalsh(covariance) <= 0).any()):
        raise ValueError('symmetric positive definite covariance required')
    coefficients = covariance.new_zeros((2**d, d))
    for mask in range(2**d):
        selected = [i for i in range(d) if mask & (1 << i)]
        if len(selected) == d:
            coefficients[mask] = model
        elif selected:
            if rule == 'conditional':
                block = covariance[selected][:, selected]
                coefficients[mask, selected] = t.linalg.solve(block, covariance[selected] @ model)
            else:
                coefficients[mask, selected] = model[selected]
    return coefficients


def risk_game(coefficients, covariance, truth, noise_variance):
    d = len(truth)
    if coefficients.shape != (2**d, d) or covariance.shape != (d, d) or truth.shape != (d,):
        raise ValueError('incompatible risk input shapes')
    if not math.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError('nonnegative finite noise variance required')
    residual = truth[None, :] - coefficients
    risks = t.einsum('md,df,mf->m', residual, covariance, residual) + noise_variance
    return risks, risks[0] - risks


def exact_shapley(values, n):
    if values.shape != (2**n,) or not 1 <= n <= 8:
        raise ValueError('complete ascending-bitmask coalition table required')
    credits = values.new_zeros(n)
    for i in range(n):
        for mask in range(2**n):
            if not mask & (1 << i):
                k = mask.bit_count()
                weight = math.factorial(k) * math.factorial(n-k-1) / math.factorial(n)
                credits[i] += weight * (values[mask | (1 << i)] - values[mask])
    return credits


def marginal_by_size(values, n, player):
    if values.shape != (2**n,) or not 0 <= player < n:
        raise ValueError('complete game and valid player required')
    means = values.new_zeros(n)
    for mask in range(2**n):
        if not mask & (1 << player):
            k = mask.bit_count()
            means[k] += (values[mask | (1 << player)] - values[mask]) / math.comb(n-1, k)
    return means


def assemble_result(coefficients, risks, values, n, player):
    return {'coefficients': coefficients.tolist(), 'risks': risks.tolist(), 'game': values.tolist(),
            'shapley': exact_shapley(values, n).tolist(),
            'marginals_b': marginal_by_size(values, n, player).tolist()}
'''


def reveal_methods(commitment_text):
    record=read_record(commitment_text)
    _require(record['status']=='committed','expected a committed forecast')
    return REFERENCE_METHODS


def practice_test(namespace,stage):
    """Immediate independent fixtures; none use CASE or its target coefficients."""
    import torch as t
    def unchanged(tensors,before):
        for actual,expected in zip(tensors,before,strict=True):
            t.testing.assert_close(actual,expected,atol=0,rtol=0)
    if stage=='restriction':
        covariance=t.tensor([[4.,1.],[1.,1.]],dtype=t.float64)
        model=t.tensor([2.,-1.],dtype=t.float64)
        for rule,expected in [('conditional',[[0.,0.],[1.75,0.],[0.,1.],[2.,-1.]]),
                              ('marginal',[[0.,0.],[2.,0.],[0.,-1.],[2.,-1.]])]:
            before=[v.clone() for v in (covariance,model)]
            actual=namespace['restriction_coefficients'](covariance,model,rule)
            t.testing.assert_close(actual,t.tensor(expected,dtype=t.float64),atol=1e-12,rtol=1e-12)
            unchanged((covariance,model),before)
        one=namespace['restriction_coefficients'](t.tensor([[2.]],dtype=t.float64),t.tensor([3.],dtype=t.float64),'conditional')
        t.testing.assert_close(one,t.tensor([[0.],[3.]],dtype=t.float64))
        for bad in (t.tensor([[1.,2.],[2.,1.]],dtype=t.float64),t.tensor([[1.,.3],[.2,1.]],dtype=t.float64)):
            try:
                namespace['restriction_coefficients'](bad,model,'conditional')
            except ValueError:
                pass
            else:
                raise AssertionError('reject non-positive-definite or asymmetric covariance')
    elif stage=='risk':
        coefficients=t.tensor([[0.,0.],[1.,0.],[0.,-2.],[1.,-2.]],dtype=t.float64)
        covariance=t.tensor([[2.,.5],[.5,1.]],dtype=t.float64)
        truth=t.tensor([.5,1.],dtype=t.float64)
        before=[v.clone() for v in (coefficients,covariance,truth)]
        risks,game=namespace['risk_game'](coefficients,covariance,truth,.3)
        # Hand-expanded 2*r0² + r0*r1 + r1² + .3, not the student's einsum.
        expected=[]
        for c0,c1 in coefficients.tolist():
            r0,r1=.5-c0,1-c1
            expected.append(2*r0*r0+r0*r1+r1*r1+.3)
        expected=t.tensor(expected,dtype=t.float64)
        t.testing.assert_close(risks,expected,atol=1e-12,rtol=1e-12)
        t.testing.assert_close(game,expected[0]-expected,atol=1e-12,rtol=1e-12)
        unchanged((coefficients,covariance,truth),before)
    elif stage in ('shapley','strata'):
        values=t.tensor([.4,1.4,-.6,1.9,.4,3.4,-.6,3.9],dtype=t.float64)
        before=values.clone()
        if stage=='shapley':
            actual=namespace['exact_shapley'](values,3)
            expected=_reference_credits(values.tolist())
            t.testing.assert_close(actual,t.tensor(expected,dtype=t.float64),atol=1e-12,rtol=1e-12)
            t.testing.assert_close(namespace['exact_shapley'](t.tensor([3.,5.],dtype=t.float64),1),t.tensor([2.],dtype=t.float64))
        else:
            actual=namespace['marginal_by_size'](values,3,2)
            expected=_strata(values.tolist(),2)
            t.testing.assert_close(actual,t.tensor(expected,dtype=t.float64),atol=1e-12,rtol=1e-12)
            assert _close(float(actual.mean()),_reference_credits(values.tolist())[2])
        unchanged((values,),(before,))
    elif stage=='assembly':
        coefficients=t.arange(24,dtype=t.float64).reshape(8,3)/10
        risks=t.tensor([7.,6.,8.,5.,4.,3.,6.,2.],dtype=t.float64)
        values=risks[0]-risks
        before=[v.clone() for v in (coefficients,risks,values)]
        result=namespace['assemble_result'](coefficients,risks,values,3,2)
        assert type(result) is dict and set(result)=={'coefficients','risks','game','shapley','marginals_b'}
        assert result['coefficients']==coefficients.tolist() and result['risks']==risks.tolist() and result['game']==values.tolist()
        for key,expected in [('shapley',_reference_credits(values.tolist())),('marginals_b',_strata(values.tolist(),2))]:
            assert all(_close(a,b) for a,b in zip(result[key],expected,strict=True))
        unchanged((coefficients,risks,values),before)
    else:
        raise ValueError('unknown practice stage')
    return stage+': independent practice fixture passed'


def _stub(name):
    tree=ast.parse(REFERENCE_METHODS)
    node=next(n for n in tree.body if n.name==name)
    args=ast.unparse(node.args)
    return f'def {name}({args}):\n    raise NotImplementedError("Implement this method before its practice test.")\n'


def build_cells():
    """Deterministic blank learner notebook, with no target numerical execution."""
    cells=[]
    def add(kind,key,source,phase):
        cell={'cell_type':kind,'id':'sage-'+key,'metadata':{'sage_prediction_phase':phase},'source':source}
        if kind=='code':
            cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown','title', '''# One predictor, two missing-input rules

A fixed model makes predictions from three correlated measurements. You can make an input unavailable by
averaging over its conditional distribution given retained inputs, or by marginally replacing it.
How much does that choice change feature B's allocated improvement in squared prediction loss?

**Forecast:** conditional SAGE for B minus marginal SAGE for B, in squared-target units. The signed answer
can be any finite real number; it is not a probability or correlation and is not capped to [-1,1].
No target result or figure is prefilled. The public case was fixed before its outcome was calculated.
Build each method, pass separate practice fixtures, commit a numerical forecast and then explicitly run.

This is an exploratory analytic Gaussian toy, not independent TEST evaluation or a mastery certificate.
Exposure is `unknown`, `seen`, or `self_reported_not_seen`; a new hash never proves you are unseen.''','forecast')
    add('code','setup', '''from pathlib import Path
from copy import deepcopy
import sys
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'mastery_hub').is_dir())
if str(root) not in sys.path: sys.path.insert(0,str(root))
from mastery_hub.units.sage_prediction import lesson
import torch as t
from IPython.display import display, Code
t.set_num_threads(1)
CASE = deepcopy(lesson.CASE)
print(CASE)
# Existing installed environment. CPU float64, one thread; no sampling, training, download or GPU.''','setup')
    add('markdown','inputs', r'''## Know the public inputs

Let $X=(A,B,C)$ be zero-mean Gaussian with covariance
$\Sigma=\begin{pmatrix}1&.6&-.3\\.6&1&.14\\-.3&.14&1\end{pmatrix}$.
The target is $Y=a^\top X+\epsilon$, with $a=(1,-.5,.25)$ and independent noise variance $.25$.
The fixed predictor is $f(X)=b^\top X$, with $b=(.75,.25,-.5)$. No model is fitted.
Feature names A/B/C do not imply a ranking or a sign for any contribution.

Every coalition has an integer bitmask: A is bit 0, B bit 1, C bit 2; rows 0…7 include empty and full.
For a retained set S, the restricted predictor is $c_S^\top X$. Conditional restriction uses
$c_S[S]=\Sigma_{SS}^{-1}\Sigma_{S,:}b$, zero elsewhere; marginal restriction uses $c_S[S]=b[S]$,
zero elsewhere. Empty coefficients are zero, full coefficients equal b. This covariance is positive definite,
so use `solve`, not an estimated imputer or a pseudoinverse.

The risk is $R(S)=(a-c_S)^\top\Sigma(a-c_S)+\sigma^2$ and the game is $v(S)=R(\emptyset)-R(S)$.
No sampling is needed. These are two different restriction games for the same predictor/population/loss.''','practice')
    descriptions={
        'restriction':'''## Exercise 1: restricted coefficients
Implement `restriction_coefficients(covariance, model, rule)` returning `(2**d,d)`.
For each ascending mask, select the retained indices. Conditional rule solves the covariance subproblem;
marginal rule copies the selected predictor weights. Explicitly handle empty/full. Validate shapes, finite
inputs and symmetric positive definite covariance. Never mutate inputs.
<details><summary>Help</summary>`covariance[selected][:, selected]` selects the square block;
`covariance[selected] @ model` is its right-hand side. Leave nonselected coefficients zero.
The independent practice case uses different two-dimensional inputs plus a one-dimensional check.</details>''',
        'risk':'''## Exercise 2: risk before allocation
Implement `risk_game(coefficients, covariance, truth, noise_variance)` returning `(risks, game)`,
both shape `(2**d,)`. Compute the residual coefficient for each mask, its covariance quadratic form and
add independent noise variance. Subtract every risk from the empty risk. Do not average coefficients or games.
<details><summary>Help</summary>The quadratic contraction is over the two feature axes, preserving the
coalition axis. The practice oracle expands a separate two-dimensional quadratic by hand.</details>''',
        'shapley':'''## Exercise 3: exact factorial Shapley
Implement `exact_shapley(values,n)`, returning n signed allocations from all `2**n` masks.
For i and S not containing i, weight its marginal by `factorial(k)*factorial(n-k-1)/factorial(n)`.
Keep the empty value in arbitrary practice games. The independent oracle averages all six orderings
of a different three-feature interacting game.
<details><summary>Help</summary>`mask.bit_count()` gives k. Efficiency compares the sum of allocations
with full minus empty; the formula allocates interactions without assuming the game is additive.</details>''',
        'strata':'''## Exercise 4: B's marginal by coalition size
Implement `marginal_by_size(values,n,player)`. Return n means, one per predecessor size k.
Average over `comb(n-1,k)` predecessor masks not containing the player. The unweighted mean of these
n strata equals that player's Shapley value; do not weight sizes by their counts again.
<details><summary>Help</summary>Insert the player with `mask | (1 << player)` and accumulate into
index `mask.bit_count()`. The practice fixture selects player 2, rather than B (index 1).</details>''',
        'assembly':'''## Exercise 5: preserve actual arrays in a JSON-native result
Implement `assemble_result(coefficients,risks,values,n,player)`. Return exactly `coefficients`, `risks`,
`game`, `shapley`, `marginals_b`. Use `.tolist()` and your own allocation/strata methods. The last field
holds the selected player's curve (B in the target, another player in practice). Do not mutate any input.
<details><summary>Help</summary>The harness keeps detached actual measurements separately, checks them
independently, and rejects an assembly that fabricates or changes coefficients, risks or game values.</details>''',
    }
    for stage,name in zip(STAGES,METHOD_NAMES,strict=True):
        add('markdown',stage+'-instructions',descriptions[stage],'practice')
        extra=' + "\\n\\n" + SHAPLEY_SOURCE + "\\n\\n" + STRATA_SOURCE' if stage=='assembly' else ''
        add('code',stage,f'{stage.upper()}_SOURCE = \'\'\'\n{_stub(name)}\'\'\'\nns = lesson.method_namespace({stage.upper()}_SOURCE{extra}, complete=False)\nprint(lesson.practice_test(ns, {stage!r}))','practice')
        add('markdown',stage+'-expected',f'Expected: `{stage}: independent practice fixture passed`. No public target coefficients, risks or allocations are evaluated by this check.','practice')
    add('markdown', 'dev-forecast', '''## Exposed DEV ladder — can a proxy earn credit? (8–12 minutes)
Take two standard Gaussian inputs with correlation rho, truth Y=X0 and predictor f=X0.
Hand-predict B-only coefficients and B's allocation at rho=0. Then change ONLY rho to .5.
For conditional restriction, E[X0|X1]=rho X1; marginal restriction still drops X0.
Choose **A:** an unused predictor input can never earn credit; **B:** conditional access to a
correlated proxy can reduce risk. Name a B-only risk or allocation observation that rules A out.
This concerns information under restriction, not a causal intervention on the population.
All arrays below are exposed two-feature DEV, separate from the three-feature exact target.
Numeric confidence is an **unscored self-rating**, not a calibrated probability.''', 'dev')
    add('code', 'dev-forecast-card', '''dev_forecast_b = .5  # Deliberately rough worked forecast at rho=.5; edit before running.
dev_self_rating = .6  # unscored self-rating
dev_explanation = 'B'
dev_disconfirming_observation = 'B receives conditional credit while marginal credit stays zero.' ''', 'dev')
    add('code', 'dev-paired', r'''import matplotlib.pyplot as plt
dev_source = '\n\n'.join([RESTRICTION_SOURCE,RISK_SOURCE,SHAPLEY_SOURCE,STRATA_SOURCE,ASSEMBLY_SOURCE])
dev_ns = lesson.method_namespace(dev_source)
dev_model=t.tensor([1.,0.],dtype=t.float64);dev_truth=dev_model.clone()
def dev_proxy_game(rho, model):
    covariance=t.tensor([[1.,rho],[rho,1.]],dtype=t.float64)
    out={}
    for rule in ('conditional','marginal'):
        coeff=dev_ns['restriction_coefficients'](covariance,model,rule)
        risks,values=dev_ns['risk_game'](coeff,covariance,dev_truth,0.)
        out[rule]=(coeff,risks,dev_ns['exact_shapley'](values,2))
    return out
dev_independent=dev_proxy_game(0.,dev_model)
dev_correlated=dev_proxy_game(.5,dev_model)
dev_zero=dev_proxy_game(.5,t.zeros_like(dev_model))
fig,axes=plt.subplots(1,2,figsize=(10,3),layout='constrained')
for rule in ('conditional','marginal'):
    axes[0].plot([0,.5],[float(dev_independent[rule][2][1]),float(dev_correlated[rule][2][1])],'o-',label=rule)
    axes[1].plot(range(4),dev_correlated[rule][1].tolist(),'o-',label=rule)
axes[0].plot([0,.5],[0.,float(dev_zero['conditional'][2][1])],'--',label='Zero predictor')
axes[1].plot(range(4),dev_zero['conditional'][1].tolist(),'--',label='Zero predictor')
axes[0].set(xlabel='Correlation rho',ylabel='B SAGE (squared-target units)',title='DEV proxy allocation')
axes[1].set(xlabel='Coalition mask',ylabel='Squared-error risk',title='DEV paired restriction rules',xticks=range(4))
for ax in axes:ax.legend()
display(fig)
print('DEV B conditional/marginal credit:',float(dev_correlated['conditional'][2][1]),float(dev_correlated['marginal'][2][1]))
print('DEV B credit absolute forecast error:',abs(float(dev_correlated['conditional'][2][1])-dev_forecast_b))
assert abs(float(dev_correlated['conditional'][2][1])-.125)<1e-12
assert dev_zero['conditional'][2].count_nonzero()==0''', 'dev')
    add('markdown', 'dev-reflection', '''### Worked DEV reflection / self-check
B-only conditional coefficient is .5 and risk is .75, so its empty-to-singleton gain is .25.
B adds nothing after A; averaging the two orders gives B credit .125. The rough forecast .5
has absolute error .375 squared-target units: it confused a coefficient with allocated risk reduction.
B's marginal credit stays 0; independent covariance and a zero predictor also give B zero credit.
A good self-check names the proxy route, squares rho in the risk, averages predecessor orders,
and avoids a physical-causality claim. That observed nonzero conditional credit rules A out.

### Try It Yourself — changed-condition DEV transfer
Change ONLY rho from +.5 to −.5. Predict B's coefficient AND its credit before the next cell.
Does the sign-sensitive explanation “negative correlation means negative credit” survive?
Keep truth, predictor, unit variances and restriction/loss definitions fixed.''', 'dev')
    add('code', 'dev-transfer', '''dev_transfer_forecast = {'coefficient':-.5,'credit':.125}
dev_negative=dev_proxy_game(-.5,dev_model)
dev_transfer_observed={'coefficient':float(dev_negative['conditional'][0][2,1]),
                       'credit':float(dev_negative['conditional'][2][1])}
print('DEV sign-transfer observation:',dev_transfer_observed)
print('DEV transfer absolute forecast errors:',{key:abs(dev_transfer_observed[key]-dev_transfer_forecast[key]) for key in dev_transfer_observed})
assert dev_transfer_observed == {'coefficient':-.5,'credit':.125}  # Method check, independent of forecast.
fig,ax=plt.subplots(figsize=(6,3),layout='constrained')
ax.plot([-.5,0,.5],[dev_transfer_observed['credit'],float(dev_independent['conditional'][2][1]),float(dev_correlated['conditional'][2][1])],'o-',label='Conditional B')
ax.axhline(0,color='gray',linestyle='--',label='Marginal / zero-predictor control')
ax.set(xlabel='Correlation rho',ylabel='B SAGE (squared-target units)',title='Observed DEV sign transfer');ax.legend();display(fig)
dev_revision=''  # Cite the coefficient sign change AND unchanged positive credit.''', 'dev')
    add('markdown','freeze-instructions','''## Freeze, forecast, commit
Finish every method and its independent fixture. Freezing records exact covariance/vectors/noise, dtype,
ordered masks, method/helper/test source hashes and runtime without evaluating a target method.
Forecast the signed B contrast numerically, give confidence and a causal explanation, and name an observation
that would contradict your reasoning. Preserve prior exposure honestly. Source/runtime edits require a new
identity and commitment. Local seals are accidental-drift checks, not exam security.''','commit')
    add('code','freeze',r'''METHOD_SOURCE = '\n\n'.join([RESTRICTION_SOURCE, RISK_SOURCE, SHAPLEY_SOURCE, STRATA_SOURCE, ASSEMBLY_SOURCE])
identity = lesson.freeze_inputs(METHOD_SOURCE)
print('Frozen exact inputs and source; no target restriction or risk has been evaluated.')''','freeze')
    add('code','commit', '''prediction = None  # Your signed conditional-minus-marginal B SAGE, squared-target units
confidence = None  # Unscored reasoning self-rating in [0,1]; not a probability
reasoning = ''  # Causal explanation and a possible disconfirming observation
prior_exposure = 'unknown'  # unknown / seen / self_reported_not_seen
commitment = lesson.commit_prediction(identity, METHOD_SOURCE, prediction, confidence, reasoning, prior_exposure)
print('Committed. No target numerical methods have run.')''','commit')
    add('markdown','postcommit','''## Explicit experiment, then inspect controls
Only the explicit run evaluates the public target. It repeats the separate practice fixtures and runs ten
8-coalition conditions: two restriction rules for the frozen case, identity covariance, zero predictor,
noise variance .75 and feature permutation [C,A,B]. It checks empty/full agreement, efficiency, identity-rule
equality, zero-predictor utility, the noise-risk shift with invariant credit, permutation relabeling and an
independent six-ordering allocation oracle. The covariance transfer is **not** included.

Inputs, returned arrays, assembly and runtime are checked around each student call. Failures retain actual
returned measurements and the original forecast without assigning a successful score. One cooperative
120-second CPU budget is checked between stages; a slow student body is rejected on return, not forcibly
interrupted. No model, sampling, download or GPU runs. Reference method text is optional after commitment;
adopting changed source requires freezing and committing again.''','postcommit')
    add('code','solutions', '''# Optional implementation help after commitment; this evaluates no target:
# display(Code(lesson.reveal_methods(commitment), language='python'))''','solutions')
    add('code','run', '''try:
    receipt = lesson.run_experiment(commitment, METHOD_SOURCE)
except lesson.ExperimentFailure as failed:
    failed_record = failed.record_text
    print(lesson.replay_failure(failed_record))
    # Optional: lesson.export_record(failed_record, Path.home()/'sage-failed-attempt-1.json')
    raise  # Preserve/replay/export the failed record; no implicit retry.
observation = lesson.replay(receipt)
display(lesson.plot_receipt(receipt))
print('Observed conditional-minus-marginal B SAGE:', observation['delta_phi_b'], 'squared-target units')
print('Absolute forecast error:', observation['absolute_forecast_error'])
print('Controls:', observation['controls'])''','run')
    add('markdown','interpretation','''## Explain the paired games
Read all three bars alongside B's marginal-by-size curves. Which predecessor sizes account for the contrast?
Do shared empty/full risks determine individual allocations? Which controls constrain your explanation,
and which claims do they leave open? Give a discrepancy explanation without rewriting your forecast.

The risks and allocations are exact for a fixed linear Gaussian model with known covariance. They do not
establish importance for an arbitrary nonlinear predictor, learned conditional imputer, real recipe population
or physical intervention. There is no confidence interval, independent TEST score, unseen certification or
universal feature ranking here. The original [SAGE paper](https://papers.neurips.cc/paper_files/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html)
connects this performance game to global feature attribution. After committing, compare the reference-rich
[canonical 16.4 lesson](../../../chapter16_shapley_attribution_baselines/exercises/part4_sage/16.4_SAGE_Global_Importance_exercises.ipynb),
which also separates empirical marginal replacement from conditional information.''','reflect')
    add('code','play', '''# Try It Yourself: inspect stored coalitions and another feature; no recomputation.
mask = 3  # A+B; change to any 0..7
feature = 0  # A; change to 0/1/2
for rule in ('conditional','marginal'):
    stored = lesson.replay(receipt)['results'][rule]
    print(rule, 'coefficients:', stored['coefficients'][mask], 'risk:', stored['risks'][mask],
          'game:', stored['game'][mask], 'feature SAGE:', stored['shapley'][feature])''','play')
    add('markdown','transfer-instructions', '''## Forecast one changed scientific condition
Change **only** covariance to `(Sigma + I)/2`; keep truth, predictor, independent noise, loss, mask order,
allocation and numeric B contrast fixed. This changes input dependence, not a random seed. The primary run
has not evaluated this transfer. Give a signed numeric forecast, confidence and causal explanation before
creating a separate transfer commitment. A primary reflection preserves the exact primary receipt.
No second experiment is started by reflection or transfer commitment.''','transfer')
    add('code','reflection', '''explanation = ''  # What matched, what failed, which control mattered, and why?
transfer_prediction = None  # Numeric forecast for the covariance-only transfer
transfer_confidence = None
transfer_reasoning = ''
attempt = lesson.reflect(receipt, explanation, transfer_prediction, transfer_confidence, transfer_reasoning)
# A separate explicit commitment is required before any transfer numerical evaluation:
# transfer_identity = lesson.freeze_inputs(METHOD_SOURCE, variant='transfer')
# transfer_commitment = lesson.commit_transfer(attempt, transfer_identity, METHOD_SOURCE, prior_exposure)
# transfer_receipt = lesson.run_experiment(transfer_commitment, METHOD_SOURCE)
# display(lesson.plot_receipt(transfer_receipt))''','transfer')
    add('markdown','export-instructions','''## Keep portable records without rerunning
Exports are opt-in, create private files with mode 0600, and refuse overwrites. Replay and plots validate
stored arrays and immutable links without executing student code, solving a new target or consulting the
current runtime. Replay validation uses independent arithmetic on already stored measurements; it cannot
certify an adversarial local editor or a learner's exposure history. Failed attempts remain unscored.''','replay')
    add('code','export', '''# Optional explicit new filenames:
# lesson.export_record(receipt, Path.home()/'sage-prediction-receipt-1.json')
# lesson.export_attempt(attempt, Path.home()/'sage-prediction-reflection-1.json')
# Later, without a new experiment:
# saved = Path.home().joinpath('sage-prediction-receipt-1.json').read_text()
# display(lesson.plot_receipt(saved))''','replay')
    return cells
