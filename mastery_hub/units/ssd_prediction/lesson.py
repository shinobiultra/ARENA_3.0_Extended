"""Blind fixed-filter SSD state-carry challenge; import/freeze never evaluate it.

Local seals detect accidental drift, not adversarial edits or unseen exposure.
Every numerical method remains student-visible. Replay validates stored arrays.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
TESTS=ROOT/'tests/mastery/test_ssd_prediction.py'
SCHEMA='arena-blind-ssd-prediction'
VERSION=1
ASSESSMENT='local_self_report_no_unseen_certification'
CASE={
 'id':'5.4-fixed-ssd-state-carry-v1',
 'X':[[1.,-.25],[-.5,.8],[0.,-.6],[.75,0.],[-1.,.5],[.25,-1.],
      [0.,.3],[-.25,0.],[.5,-.4],[0.,.7],[-.75,0.],[.4,-.2]],
 'B':[[1.,0.],[0.,1.],[1.,-1.],[.5,1.],[-1.,.5],[1.,1.],
      [0.,-1.],[-.5,1.],[1.,.25],[.25,-1.],[-1.,-.5],[.5,.75]],
 'C':[[.5,1.],[1.,-.5],[-.5,.25],[1.,1.],[.25,-1.],[-1.,.5],
      [1.,-.25],[.5,.5],[-.75,1.],[1.,.25],[.25,-.5],[.8,-.6]],
 'transition_probabilities':[.82,.94], 'log_decay_scale':1.,
 'initial_state':[[0.,0.],[0.,0.]], 'readout_weights':[.7,-.4],
 'gammas':[0.,.25,.5,.65,.85,1.], 'intervention_index':3,'baseline_index':5,
 'chunk_size':5,'control_chunk_sizes':[1,3,12,17],
 'dtype':'torch.float64','device':'cpu','threads':1,
 'layout':'X[time,head], B/C[time,state] shared across heads; state[head,state]; read after write',
 'forecast':'final readout at carry .65 minus final readout at carry1; signed filter-output units',
 'transfer':'only log_decay_scale becomes .5, i.e. A -> A/2; separate commitment',
 'budget_seconds':120,'evidence_class':'analytic_toy',
 'data_role':'fixed synthetic filter; no independent TEST or released-model claim',
}
METHOD_NAMES=('ssd_recurrent','ssd_quadratic','ssd_chunked','readout','assemble_result')
STAGES=('recurrent','quadratic','chunked','readout','assembly')
METHODS=('recurrent','quadratic','chunked')
RESULT_METHODS=METHODS+('scalar_oracle',)
CONDITIONS=tuple(['curve_'+str(i) for i in range(6)]+['chunk_'+str(i) for i in range(4)]+['zero','future','scaled','head_swap'])
_CASE_DEFINITION=json.dumps(CASE,sort_keys=True,allow_nan=False)
RECORD_LIMIT=4_000_000
ATTEMPT_LIMIT=6_000_000

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



def _case(variant='primary',*,frozen=False):
    _require(variant in ('primary','transfer'),'unknown case variant')
    case=json.loads(_CASE_DEFINITION) if frozen else deepcopy(CASE)
    case['variant']=variant
    if variant=='transfer':case['log_decay_scale']=.5
    return case


def _layout(case):
    length=len(case['X']);heads=len(case['transition_probabilities']);states=len(case['B'][0])
    return {'X':([length,heads],case['X']),'B':([length,states],case['B']),
            'C':([length,states],case['C']),
            'transition_probabilities':([heads],case['transition_probabilities']),
            'initial_state':([heads,states],case['initial_state']),
            'readout_weights':([heads],case['readout_weights']),
            'gammas':([len(case['gammas'])],case['gammas']),
            'log_decay_scale':([1],[case['log_decay_scale']])}


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
    namespace={'t':t,'math':math,'validate_inputs':validate_inputs}
    exec(compile(source,'<student SSD methods>','exec'),namespace)
    return namespace


def freeze_inputs(method_source,variant='primary'):
    """Hash public inputs, source and runtime; no numerical SSD or readout evaluation."""
    _method_tree(method_source)
    env=environment_identity()
    _require(env['threads']==1 and env['default_device']=='cpu','set one CPU thread and CPU default device before freezing')
    case=_case(variant)
    return {'case_sha256':digest(case),'method_sha256':hashlib.sha256(method_source.encode()).hexdigest(),
            'helper_sha256':_file(__file__),'tests_sha256':_file(TESTS),'environment':env,
            'inputs':{name:{'shape':shape,'dtype':case['dtype'],'order':'C','encoding':'float64 little-endian','sha256':_tensor_hash(rows)}
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
        _require(identity['inputs'][name]=={'shape':shape,'dtype':case['dtype'],'order':'C','encoding':'float64 little-endian','sha256':_tensor_hash(rows)},
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


def _array(value,shape):
    if not shape:
        _number(value)
    else:
        _require(type(value) is list and len(value)==shape[0],'incorrect array shape')
        for item in value:_array(item,shape[1:])


def _flat(value):
    return [x for row in value for x in _flat(row)] if type(value) is list else [value]


def _equal(a,b):
    return len(_flat(a))==len(_flat(b)) and all(_close(x,y) for x,y in zip(_flat(a),_flat(b),strict=True))


def _condition(case,label):
    _require(label in CONDITIONS,'unknown condition')
    c=deepcopy(case);c['gamma']=case['gammas'][case['intervention_index']]
    if label.startswith('curve_'):c['gamma']=case['gammas'][int(label.split('_')[1])]
    elif label.startswith('chunk_'):
        c['gamma']=1.;c['chunk_size']=case['control_chunk_sizes'][int(label.split('_')[1])]
    elif label=='zero':c['X']=[[0.]*len(row) for row in c['X']]
    elif label=='future':c['X'][-1]=[v+2*(i+1) for i,v in enumerate(c['X'][-1])]
    elif label=='scaled':c['X']=[[2*v for v in row] for row in c['X']]
    elif label=='head_swap':
        for key in ('transition_probabilities','initial_state','readout_weights'):c[key]=list(reversed(c[key]))
        c['X']=[list(reversed(row)) for row in c['X']]
    return c


def _shapes(case):
    length=len(case['X']);heads=len(case['transition_probabilities']);states=len(case['B'][0])
    return {'outputs':[length,heads],'states':[length,heads,states],'readouts':[length]}


def _validate_partial(measured,case):
    _require(type(measured) is dict and set(measured)<=set(CONDITIONS),'unknown partial condition')
    for methods in measured.values():
        _require(type(methods) is dict and set(methods)<=set(RESULT_METHODS),'unknown partial method')
        for result in methods.values():
            names=set(_shapes(case))
            _require(type(result) is dict and set(result)<=names|{k+'_sha256' for k in names},'incorrect measurement fields')
            _require(all((k in result)==(k+'_sha256' in result) for k in names),'incomplete measured array/hash pair')
            for name,shape in _shapes(case).items():
                if name not in result:continue
                _array(result[name],shape)
                _require(result[name+'_sha256']==_tensor_hash(result[name]),'actual array hash mismatch')


def _validate_result(result,condition):
    """Check stored recurrence/readout residuals; never rerun a student method."""
    _keys(result,RESULT_METHODS)
    shapes=_shapes(condition)
    for method,arrays in result.items():
        _keys(arrays,tuple(shapes)+tuple(k+'_sha256' for k in shapes))
        for key,shape in shapes.items():
            _array(arrays[key],shape)
            _require(arrays[key+'_sha256']==_tensor_hash(arrays[key]),'result array hash mismatch')
        previous=condition['initial_state']
        for pos,state in enumerate(arrays['states']):
            for head,row in enumerate(state):
                transition=condition['gamma']*condition['transition_probabilities'][head]**condition['log_decay_scale']
                for n,value in enumerate(row):
                    _require(_close(value,transition*previous[head][n]+condition['X'][pos][head]*condition['B'][pos][n]),
                             'stored state violates recurrence')
                _require(_close(arrays['outputs'][pos][head],math.fsum(condition['C'][pos][n]*v for n,v in enumerate(row))),
                         'stored output differs from actual state')
            _require(_close(arrays['readouts'][pos],math.fsum(w*y for w,y in zip(condition['readout_weights'],arrays['outputs'][pos],strict=True))),
                     'stored readout differs from actual output')
            previous=state
    base=result['scalar_oracle']
    _require(all(_equal(base[key],result[m][key]) for m in METHODS for key in shapes),'three forms disagree with independent scalar oracle')


def _control_checks(results,case):
    base=results['curve_'+str(case['intervention_index'])]['recurrent']
    full=results['curve_'+str(case['baseline_index'])]['recurrent']
    zero=results['curve_0']['recurrent']
    return {
      'three_forms':all(_equal(result['recurrent'][k],result[m][k]) for result in results.values() for m in METHODS for k in _shapes(case)),
      'chunk_padding':all(_equal(results['chunk_'+str(i)][m][k],full[k]) for i in range(4) for m in METHODS for k in _shapes(case)),
      'zero_writes':all(_close(v,0.) for m in METHODS for k in _shapes(case) for v in _flat(results['zero'][m][k])),
      'current_write_only':all(_close(zero['states'][pos][h][n],case['X'][pos][h]*case['B'][pos][n]) for pos in range(len(case['X'])) for h in range(len(case['transition_probabilities'])) for n in range(len(case['B'][0]))),
      'future_causality':all(_equal(base[k][:-1],results['future']['recurrent'][k][:-1]) for k in _shapes(case)),
      'linearity':all(_equal([2*v for v in _flat(base[k])],_flat(results['scaled']['recurrent'][k])) for k in _shapes(case)),
      'coherent_head_swap':_equal(base['readouts'],results['head_swap']['recurrent']['readouts']) and _equal([list(reversed(row)) for row in base['states']],results['head_swap']['recurrent']['states']),
      'independent_scalar_oracle':all(_equal(result['scalar_oracle'][k],result[m][k]) for result in results.values() for m in METHODS for k in _shapes(case)),
    }


def _validate_failure(failure,case):
    _keys(failure,('kind','stage','error_type','message','elapsed_seconds','measured','results','controls','returned_assemblies'))
    _require(failure['kind']=='failed_cpu_analytic','invalid failure classification')
    for key in ('stage','error_type','message'):_text(failure[key])
    _number(failure['elapsed_seconds'],0)
    _validate_partial(failure['measured'],case)
    _require(type(failure['results']) is dict and set(failure['results'])<=set(failure['measured']),'unknown completed condition')
    for label,result in failure['results'].items():
        _validate_result(result,_condition(case,label))
        _require(result==failure['measured'][label],'failure lost actual measurements')
    _require(type(failure['returned_assemblies']) is dict and set(failure['returned_assemblies'])<=set(failure['measured']),'unknown returned assembly')
    for result in failure['returned_assemblies'].values():
        _require(type(result) is dict and set(result)<=set(METHODS),'malformed raw assembly')
        for values in result.values():
            _require(type(values) is dict and set(values)<=set(_shapes(case)),'malformed raw returned arrays')
            for k,v in values.items():_array(v,_shapes(case)[k])
    if failure['controls'] is not None:
        _keys(failure['results'],CONDITIONS)
        _require(failure['controls']==_control_checks(failure['results'],case),'invalid failed controls')


def read_record(text):
    """Validate stored arrays/links; no student methods or fresh experiment."""
    out=_parse(text,RECORD_LIMIT)
    _require(out.get('schema')==SCHEMA and type(out.get('schema_version')) is int and out.get('schema_version')==VERSION,'unsupported schema')
    if out.get('status')=='committed':
        _keys(out,('schema','schema_version','status','case','identity','method_source','prediction','confidence','reasoning','prior_exposure','assessment','parent_attempt'))
        variant=out['case'].get('variant') if type(out['case']) is dict else None
        _require(digest(out['case'])==digest(_case(variant,frozen=True)),'unsupported case')
        _method_tree(out['method_source']);_validate_identity(out['identity'],out['method_source'],out['case'])
        _number(out['prediction']);_number(out['confidence'],0,1);_text(out['reasoning'])
        _require(out['prior_exposure'] in ('unknown','seen','self_reported_not_seen') and out['assessment']==ASSESSMENT,'invalid exposure')
        if variant=='primary':_require(out['parent_attempt'] is None,'primary cannot have parent')
        else:
            attempt=read_attempt(out['parent_attempt'])
            _require(attempt['transfer_prediction']==out['prediction'] and attempt['confidence']==out['confidence'] and attempt['reasoning']==out['reasoning'],'transfer forecast changed')
            primary=read_record(read_record(attempt['receipt'])['commitment'])
            _require(out['method_source']==primary['method_source'],'single-factor transfer must preserve primary method source')
            _require(all(out['identity'][k]==primary['identity'][k] for k in ('method_sha256','helper_sha256','tests_sha256','environment')),
                     'single-factor transfer must preserve primary source and runtime identity')
    elif out.get('status') in ('observed','failed'):
        _keys(out,('schema','schema_version','status','commitment','commitment_sha256','observation' if out['status']=='observed' else 'failure'))
        _require(type(out['commitment']) is str and hashlib.sha256(out['commitment'].encode()).hexdigest()==out['commitment_sha256'],'commitment link mismatch')
        _require(_parse(out['commitment'],RECORD_LIMIT).get('status')=='committed','expected original commitment')
        committed=read_record(out['commitment']);case=committed['case']
        if out['status']=='failed':
            _validate_failure(out['failure'],case);return out
        obs=out['observation'];_keys(obs,('kind','results','controls','delta_readout','absolute_forecast_error','elapsed_seconds','condition_count'))
        _require(obs['kind']=='fresh_cpu_analytic' and type(obs['condition_count']) is int and obs['condition_count']==len(CONDITIONS),'invalid execution inventory')
        _number(obs['elapsed_seconds'],0,case['budget_seconds']);_require(obs['elapsed_seconds']>0,'invalid elapsed')
        _keys(obs['results'],CONDITIONS)
        for label,result in obs['results'].items():_validate_result(result,_condition(case,label))
        controls=_control_checks(obs['results'],case)
        _require(obs['controls']==controls and all(type(v) is bool and v for v in controls.values()),'control failure')
        expected=obs['results']['curve_'+str(case['intervention_index'])]['recurrent']['readouts'][-1]-obs['results']['curve_'+str(case['baseline_index'])]['recurrent']['readouts'][-1]
        _number(obs['delta_readout']);_number(obs['absolute_forecast_error'],0)
        _require(_close(obs['delta_readout'],expected) and _close(obs['absolute_forecast_error'],abs(expected-committed['prediction'])),'scoring mismatch')
    else:raise ValueError('unsupported record status')
    return out


def assert_current(record_text,method_source):
    record=read_record(record_text)
    _require(record['status']=='committed','already observed: replay or create a separate commitment')
    _require(method_source==record['method_source'],'stale method text; freeze and commit again')
    _require(freeze_inputs(method_source,record['case']['variant'])==record['identity'],'stale inputs, helper, tests or environment')
    return record


class ExperimentFailure(ValueError):
    def __init__(self,record_text):
        self.record_text=record_text
        failure=read_record(record_text)['failure']
        super().__init__(failure['stage']+': '+failure['message'])


def _scalar_oracle(X,log_decay,B,C,initial,gamma):
    """Independent Python scalar loops; called only inside committed execution/practice."""
    state=deepcopy(initial);outputs=[];history=[]
    for pos,row in enumerate(X):
        for h,x in enumerate(row):
            for n,b in enumerate(B[pos]):state[h][n]=gamma*math.exp(log_decay[pos][h])*state[h][n]+x*b
        history.append(deepcopy(state))
        outputs.append([math.fsum(c*s for c,s in zip(C[pos],state[h],strict=True)) for h in range(len(row))])
    return outputs,history


def run_experiment(commitment_text,method_source):
    started=time.monotonic();committed=assert_current(commitment_text,method_source)
    capture={'stage':'setup','measured':{},'results':{},'controls':None,'returned_assemblies':{}}
    try:return _execute_experiment(commitment_text,method_source,started,capture,committed)
    except Exception as error:
        failure={'kind':'failed_cpu_analytic','error_type':type(error).__name__,'message':(str(error) or type(error).__name__)[:20000],'elapsed_seconds':time.monotonic()-started,**capture}
        text=_seal({'schema':SCHEMA,'schema_version':VERSION,'status':'failed','commitment':commitment_text,'commitment_sha256':hashlib.sha256(commitment_text.encode()).hexdigest(),'failure':failure})
        read_record(text)
        raise ExperimentFailure(text) from error


def _execute_experiment(commitment_text,source,started,capture,committed):
    import torch as t
    case=committed['case'];identity=committed['identity']
    def current():
        _require(freeze_inputs(source,case['variant'])==identity,'method changed frozen input/source/runtime identity')
        if time.monotonic()-started>case['budget_seconds']:raise TimeoutError('CPU budget exceeded')
    def snapshot(args):return [(str(v.dtype),str(v.device),list(v.shape),_tensor_hash(v.detach().cpu().tolist())) for v in args]
    def checked(stage,function,args):
        current();capture['stage']=stage;before=snapshot(args)
        return function(*args),before
    def unchanged(args,before):
        _require(snapshot(args)==before,'method mutated actual input arrays');current()
    def tensor(value,shape):
        _require(isinstance(value,t.Tensor) and list(value.shape)==shape and value.dtype==t.float64 and value.device.type=='cpu' and bool(t.isfinite(value).all()),'expected finite CPU float64 tensor')
    def remember(values):return {**values,**{k+'_sha256':_tensor_hash(v) for k,v in values.items()}}
    namespace=method_namespace(source);current()
    for stage in STAGES:
        capture['stage']='practice '+stage;practice_test(namespace,stage);current()
    for label in CONDITIONS:
        c=_condition(case,label);shapes=_shapes(c)
        X=t.tensor(c['X'],dtype=t.float64);B=t.tensor(c['B'],dtype=t.float64);C=t.tensor(c['C'],dtype=t.float64)
        initial=t.tensor(c['initial_state'],dtype=t.float64);weights=t.tensor(c['readout_weights'],dtype=t.float64)
        A=t.tensor(c['transition_probabilities'],dtype=t.float64).log()[None,:].expand(len(X),-1).clone()*c['log_decay_scale']
        arrays={};capture['measured'][label]=arrays
        oracle=_scalar_oracle(c['X'],A.tolist(),c['B'],c['C'],c['initial_state'],c['gamma'])
        oracle_readouts=[math.fsum(w*y for w,y in zip(c['readout_weights'],row,strict=True)) for row in oracle[0]]
        arrays['scalar_oracle']=remember({'outputs':oracle[0],'states':oracle[1],'readouts':oracle_readouts})
        for method in METHODS:
            args=tuple(v.clone() for v in (X,A,B,C,initial))
            fn=namespace['ssd_'+method]
            result,before=checked(label+': '+method,lambda *a:fn(*a,c['gamma'],c['chunk_size']) if method=='chunked' else fn(*a,c['gamma']),args)
            _require(type(result) in (tuple,list) and len(result)==2,'SSD must return outputs and state history')
            output,states=result
            arrays[method]={}
            # Retain each valid returned array even if its partner is malformed.
            for key,value in (('outputs',output),('states',states)):
                try:tensor(value,shapes[key])
                except ValueError:continue
                arrays[method].update(remember({key:value.detach().tolist()}))
            tensor(output,shapes['outputs']);tensor(states,shapes['states'])
            unchanged(args,before)
            _require(_equal(arrays[method]['outputs'],oracle[0]) and _equal(arrays[method]['states'],oracle[1]),'independent scalar oracle mismatch')
            args=(output.detach().clone(),weights.clone())
            readouts,before=checked(label+': readout',namespace['readout'],args);tensor(readouts,shapes['readouts'])
            arrays[method].update(remember({'readouts':readouts.detach().tolist()}));unchanged(args,before)
        # Assembly sees disposable tensors; measured lists are detached and immutable.
        args=tuple(t.tensor(arrays[m][k],dtype=t.float64) for m in METHODS for k in ('outputs','states','readouts'))
        result,before=checked(label+': assembly',namespace['assemble_result'],args)
        if type(result) is dict:
            raw={}
            for m in METHODS:
                if type(result.get(m)) is not dict:continue
                raw[m]={}
                for k,shape in shapes.items():
                    if k not in result[m]:continue
                    try:_array(result[m][k],shape)
                    except ValueError:continue
                    raw[m][k]=deepcopy(result[m][k])
            capture['returned_assemblies'][label]=raw
        unchanged(args,before);_keys(result,METHODS)
        for m in METHODS:
            _keys(result[m],shapes)
            _require(all(result[m][k]==arrays[m][k] for k in shapes),'assembly fabricated actual arrays')
        validated=deepcopy(arrays);_validate_result(validated,c);capture['results'][label]=validated;current()
    capture['stage']='paired controls';checks=_control_checks(capture['results'],case);capture['controls']=checks
    _require(all(checks.values()),'control failure: '+str(checks))
    delta=capture['results']['curve_'+str(case['intervention_index'])]['recurrent']['readouts'][-1]-capture['results']['curve_'+str(case['baseline_index'])]['recurrent']['readouts'][-1]
    text=_seal({'schema':SCHEMA,'schema_version':VERSION,'status':'observed','commitment':commitment_text,'commitment_sha256':hashlib.sha256(commitment_text.encode()).hexdigest(),
       'observation':{'kind':'fresh_cpu_analytic','results':capture['results'],'controls':checks,'delta_readout':delta,'absolute_forecast_error':abs(delta-committed['prediction']),'elapsed_seconds':time.monotonic()-started,'condition_count':len(CONDITIONS)}})
    read_record(text);current();return text


def replay(receipt_text):
    record=read_record(receipt_text);_require(record['status']=='observed','no observation to replay')
    return deepcopy(record['observation'])


def replay_failure(text):
    record=read_record(text);_require(record['status']=='failed','expected unscored failure')
    return deepcopy(record['failure'])


def plot_receipt(text):
    import matplotlib.pyplot as plt
    obs=replay(text);case=read_record(read_record(text)['commitment'])['case']
    baseline=obs['results']['curve_'+str(case['baseline_index'])]['recurrent']['readouts'][-1]
    values=[obs['results']['curve_'+str(i)]['recurrent']['readouts'][-1]-baseline for i in range(6)]
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    axes[0].plot(case['gammas'],values,'o-');axes[0].axhline(0,color='gray',lw=.7)
    axes[0].set(xlabel='Past-state carry multiplier',ylabel='Final readout minus full carry',title='Fixed filter carry intervention')
    extent=max(abs(v) for index in (case['baseline_index'],case['intervention_index'])
               for v in _flat(obs['results']['curve_'+str(index)]['recurrent']['states'])) or 1
    for ax,index,title in ((axes[1],case['baseline_index'],'Full carry state'),(axes[2],case['intervention_index'],'Intervened state')):
        states=obs['results']['curve_'+str(index)]['recurrent']['states']
        matrix=[[v for h in row for v in h] for row in states]
        im=ax.imshow(list(zip(*matrix)),aspect='auto',cmap='RdBu_r',vmin=-extent,vmax=extent)
        ax.set(xlabel='Real token position',ylabel='Head/state channel',title=title);fig.colorbar(im,ax=ax)
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


def validate_inputs(X,A,B,C,initial,gamma):
    """Shape/finite validation only, no SSD computation."""
    import torch as t
    _require(all(isinstance(v,t.Tensor) and v.dtype==t.float64 and v.device.type=='cpu' and bool(t.isfinite(v).all()) for v in (X,A,B,C,initial)),'finite CPU float64 arrays required')
    _require(X.ndim==2 and min(X.shape)>0 and A.shape==X.shape,'X/A must be nonempty [time,head]')
    _require(B.ndim==2 and B.shape==C.shape and B.shape[0]==X.shape[0] and B.shape[1]>0,'B/C must be shared [time,state]')
    _require(initial.shape==(X.shape[1],B.shape[1]),'initial state shape mismatch')
    _require(bool((A<=0).all()),'log decay must be nonpositive')
    _number(gamma,0,1)


REFERENCE_METHODS='''def ssd_recurrent(X, A, B, C, initial, gamma):
    validate_inputs(X,A,B,C,initial,gamma)
    state=initial.clone()
    states=[]
    outputs=[]
    for pos in range(len(X)):
        state=gamma*A[pos].exp()[:,None]*state+X[pos,:,None]*B[pos,None,:]
        states.append(state)
        outputs.append((state*C[pos,None,:]).sum(-1))
    return t.stack(outputs),t.stack(states)


def ssd_quadratic(X, A, B, C, initial, gamma):
    validate_inputs(X,A,B,C,initial,gamma)
    length=len(X)
    writes=X[:,:,None]*B[:,None,:]
    if gamma==0:
        states=writes
    else:
        effective=A+math.log(gamma)
        prefix=effective.cumsum(0)
        index=t.arange(length,device=X.device)
        causal=index[:,None]>=index[None,:]
        log_weights=(prefix[:,None,:]-prefix[None,:,:]).masked_fill(~causal[:,:,None],-math.inf)
        weights=log_weights.exp()
        states=t.einsum('tsh,shn->thn',weights,writes)+prefix.exp()[:,:,None]*initial[None,:,:]
    return (states*C[:,None,:]).sum(-1),states


def ssd_chunked(X, A, B, C, initial, gamma, chunk_size):
    validate_inputs(X,A,B,C,initial,gamma)
    if type(chunk_size) is not int or chunk_size<=0:
        raise ValueError('chunk_size must be a positive integer')
    length,heads=X.shape
    n=B.shape[1]
    writes=X[:,:,None]*B[:,None,:]
    if gamma==0:
        return (writes*C[:,None,:]).sum(-1),writes
    # Fold gamma into REAL-token transitions before identity padding.
    effective=A+math.log(gamma)
    pad=(-length)%chunk_size
    effective=t.cat((effective,effective.new_zeros((pad,heads))),0)
    writes=t.cat((writes,writes.new_zeros((pad,heads,n))),0)
    reads=t.cat((C,C.new_zeros((pad,n))),0)
    chunks=(length+pad)//chunk_size
    Ac=effective.reshape(chunks,chunk_size,heads)
    Wc=writes.reshape(chunks,chunk_size,heads,n)
    Cc=reads.reshape(chunks,chunk_size,n)
    prefixes=Ac.cumsum(1)
    indices=t.arange(chunk_size,device=X.device)
    causal=indices[:,None]>=indices[None,:]
    local_log_weights=(prefixes[:,:,None,:]-prefixes[:,None,:,:]).masked_fill(~causal[None,:,:,None],-math.inf)
    local_weights=local_log_weights.exp()
    local_states=t.einsum('ctsh,cshn->cthn',local_weights,Wc)
    boundary=initial.clone()
    histories=[]
    outputs=[]
    for chunk in range(chunks):
        states=local_states[chunk]+prefixes[chunk].exp()[:,:,None]*boundary[None,:,:]
        outputs.append((states*Cc[chunk,:,None,:]).sum(-1))
        histories.append(states)
        boundary=states[-1]
    return t.cat(outputs,0)[:length],t.cat(histories,0)[:length]


def readout(outputs, weights):
    if outputs.ndim!=2 or weights.shape!=(outputs.shape[1],):
        raise ValueError('readout shape mismatch')
    return (outputs*weights[None,:]).sum(-1)


def assemble_result(ry, rs, rr, qy, qs, qr, cy, cs, cr):
    return {'recurrent':{'outputs':ry.tolist(),'states':rs.tolist(),'readouts':rr.tolist()},
            'quadratic':{'outputs':qy.tolist(),'states':qs.tolist(),'readouts':qr.tolist()},
            'chunked':{'outputs':cy.tolist(),'states':cs.tolist(),'readouts':cr.tolist()}}
'''


def reveal_methods(commitment_text):
    _require(read_record(commitment_text)['status']=='committed','reveal requires a commitment')
    return REFERENCE_METHODS


def practice_fixture():
    """Different T3 inputs; never returns any public target array."""
    import torch as t
    return (t.tensor([[1.,2.],[-1.,.5],[.25,-.75]],dtype=t.float64),
            t.tensor([[.5,.75]]*3,dtype=t.float64).log(),
            t.tensor([[1.,0.],[0.,1.],[1.,-1.]],dtype=t.float64),
            t.tensor([[1.,1.],[1.,-1.],[.5,.25]],dtype=t.float64),
            t.tensor([[.5,-.25],[0.,.75]],dtype=t.float64))


def practice_test(namespace,stage):
    import torch as t
    _require(stage in STAGES,'unknown practice stage')
    args=practice_fixture();gamma=.4
    expected=_scalar_oracle(*(v.tolist() for v in args),gamma)
    if stage in METHODS:
        fn=namespace['ssd_'+stage]
        def call(values):
            clones=tuple(v.clone() for v in values)
            result=fn(*clones,gamma,5) if stage=='chunked' else fn(*clones,gamma)
            _require(all(t.equal(a,b) for a,b in zip(clones,values,strict=True)),'practice method mutated inputs')
            return result
        output,states=call(args)
        _require(_equal(output.tolist(),expected[0]) and _equal(states.tolist(),expected[1]),'independent practice oracle mismatch')
        # Exact carry composition, including nonzero initial state.
        first=tuple(v[:2] for v in args[:4])+(args[4],)
        left_y,left_s=call(first)
        right=tuple(v[2:] for v in args[:4])+(left_s[-1],)
        right_y,right_s=call(right)
        _require(t.allclose(t.cat((left_y,right_y)),output,atol=1e-11,rtol=1e-10) and t.allclose(t.cat((left_s,right_s)),states,atol=1e-11,rtol=1e-10),'split-state composition failed')
        # Orthogonal Q=[[0,1],[-1,0]]: B,C,initial all transform by Q.
        def basis(v):return t.stack((v[...,1],-v[...,0]),-1)
        transformed=(args[0],args[1],basis(args[2]),basis(args[3]),basis(args[4]))
        by,bs=call(transformed)
        _require(t.allclose(by,output,atol=1e-11,rtol=1e-10) and t.allclose(bs,basis(states),atol=1e-11,rtol=1e-10),'orthogonal state-basis control failed')
    elif stage=='readout':
        outputs=t.tensor([[2.,-1.],[0.,4.]],dtype=t.float64);weights=t.tensor([.25,-.5],dtype=t.float64)
        _require(_equal(namespace['readout'](outputs,weights).tolist(),[1.,-2.]),'practice readout mismatch')
    else:
        y=t.tensor(expected[0],dtype=t.float64);states=t.tensor(expected[1],dtype=t.float64);r=t.tensor([1.,-2.,.5],dtype=t.float64)
        result=namespace['assemble_result'](*(v.clone() for _ in METHODS for v in (y,states,r)))
        _keys(result,METHODS)
        _require(all(result[m]=={'outputs':y.tolist(),'states':states.tolist(),'readouts':r.tolist()} for m in METHODS),'practice assembly must retain actual arrays')
    return 'Practice '+stage+' passed (different inputs).'


def _stub(name):
    tree=ast.parse(REFERENCE_METHODS)
    node=next(n for n in tree.body if n.name==name)
    first=REFERENCE_METHODS.splitlines()[node.lineno-1]
    return first+'\n    raise NotImplementedError("Implement '+name+' before freezing")\n'


def build_cells():
    cells=[]
    def md(text,id):cells.append({'cell_type':'markdown','id':id,'metadata':{},'source':text})
    def code(text,id,**metadata):cells.append({'cell_type':'code','id':id,'metadata':metadata,'source':text,'outputs':[],'execution_count':None})
    md('''# One sequence, three equivalent computations

How much does weakening past-state carry change a fixed filter's final signed readout?

Implement recurrent, quadratic and chunked SSD, forecast a number, then inspect an effect curve and state heatmaps. This exact small filter connects to [5.4 Mamba-2 and State-Space Duality](../../../chapter5_modern_architectures/exercises/part4_mamba2_ssd/5.4_Mamba-2_and_State-Space_Duality_exercises.ipynb). It tests state transport; it does not reproduce the historical trained recall, 130M checkpoint or CUDA results.

The public case was fixed before measurement. No outcome sign or monotonic trend is promised. Viewing any case output before your commitment counts as prior exposure; local records cannot certify that you have never seen it.''','intro')
    code('''from pathlib import Path
import sys
import torch as t
root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'mastery_hub').is_dir())
if str(root) not in sys.path:sys.path.insert(0,str(root))
from mastery_hub.units.ssd_prediction import lesson
from mastery_hub.units.ssd_prediction.lesson import validate_inputs
from IPython.display import display
import math
t.set_num_threads(1)
# CPU float64 is explicit in every method input; no target runs on import.
''','setup')
    md('''## Public inputs and prediction

X is [time,head], shared B/C are [time,state], and initial state is [head,state]. Read **after** writing:

`state[t,h,n] = gamma*exp(A[h])*state[t-1,h,n]+X[t,h]*B[t,n]`

`Y[t,h] = sum_n C[t,n]*state[t,h,n]`; `r[t]=.7*Y[t,0]-.4*Y[t,1]`.

T=12, H=2, N=2, zero initial state, CPU float64, one thread. A=log([.82,.94]). Primary target: **r[11](gamma=.65) minus r[11](gamma=1)**, signed filter-output units; there is no [-1,1] cap. Chunk size5 is an equivalent factorization, not a changed model.

The curve uses gamma [0,.25,.5,.65,.85,1]. Transfer changes only A→A/2, requires a separate forecast and commitment, and is never a primary control.''','public-definition')
    code("# Inspect raw public inputs only; this performs no SSD or readout evaluation.\npublic_inputs=lesson._case()\npublic_inputs",'raw-inputs')
    helptexts={
      'recurrent':'At each real token decay the previous state, add the outer-product write, then read. Return every output and every real-token state.',
      'quadratic':'For t>=s the write weight is exp(sum of effective log transitions over s+1..t). The diagonal is one. Handle gamma0 explicitly: no log(0) or0*(-inf). Include initial-state decay separately.',
      'chunked':'Use a causal quadratic diagonal block per chunk plus the decayed incoming boundary state. Loop over chunks, not tokens. Fold gamma into real transitions BEFORE identity padding: padding has decay1 and zero writes/reads. Crop every returned array to real tokens.',
      'readout':'Contract the head output with the frozen readout weights. Preserve signed values.',
      'assembly':'Convert the nine actual output/state/readout tensors to JSON-native lists under recurrent/quadratic/chunked. Do not recompute or fabricate measurements.'}
    for name,stage in zip(METHOD_NAMES,STAGES,strict=True):
        md('## Implement '+name+'\n\n'+helptexts[stage]+'\n\n<details><summary>Help</summary>Use the different T3 practice fixture. Expected feedback is “Practice '+stage+' passed”. Reference methods become available only after a commitment; edits afterwards require a new commitment.</details>',stage+'-help')
        code(stage.upper()+"_SOURCE = '''\n"+_stub(name)+"'''\nnamespace=lesson.method_namespace("+stage.upper()+"_SOURCE,complete=False)\nprint(lesson.practice_test(namespace,'"+stage+"'))",name,ssd_method=True)
        md('Expected: Practice '+stage+' passed (different inputs). No public-case method is called here.',stage+'-expected')
    md('''## Exposed DEV ladder — trace the carry before forecasting (8–12 minutes)
Use the existing T3 fixture below, not the T12 target. Hand-trace token 0:
head-0 initial [.5,−.25], decay .5*gamma, write [1,0], read [1,1], so Y[0,0]=1+.125*gamma.
For head 1, Y[0,1]=2+.5625*gamma. Predict both at gamma=.4, then compare gamma=0.
Choose **A:** stronger carry must increase every signed output; **B:** carry transports state,
but signed writes and read directions determine the output. Name a paired trajectory that
could rule A out. No target outcome is used here. Numeric confidence is an **unscored self-rating**.''', 'dev-forecast')
    code('''dev_forecast_first=[1.05,2.225]
dev_self_rating=.7  # unscored self-rating
dev_explanation='B'
dev_disconfirming_observation='More carry can lower a later signed output.' ''', 'dev-forecast-card')
    code(r'''import matplotlib.pyplot as plt
# Public T3 practice literals; no public-target runner or _case call.
practice_X,practice_A,practice_B,practice_C,practice_initial=lesson.practice_fixture()
dev_source='\n\n'.join([RECURRENT_SOURCE,QUADRATIC_SOURCE,CHUNKED_SOURCE,READOUT_SOURCE,ASSEMBLY_SOURCE])
dev_ns=lesson.method_namespace(dev_source)
dev_outputs={};dev_states={}
for gamma in (0.,.4):
    y,state=dev_ns['ssd_recurrent'](practice_X,practice_A,practice_B,practice_C,practice_initial,gamma)
    dev_outputs[gamma]=y;dev_states[gamma]=state
    cy,cs=dev_ns['ssd_chunked'](practice_X,practice_A,practice_B,practice_C,practice_initial,gamma,2)
    t.testing.assert_close(cy,y);t.testing.assert_close(cs,state)
zero_y,zero_s=dev_ns['ssd_recurrent'](t.zeros_like(practice_X),practice_A,practice_B,practice_C,t.zeros_like(practice_initial),.4)
fig,axes=plt.subplots(1,2,figsize=(11,3),layout='constrained')
for head,ax in enumerate(axes):
    for gamma,y in dev_outputs.items(): ax.plot(range(3),y[:,head].tolist(),'o-',label='gamma='+str(gamma))
    ax.plot(range(3),zero_y[:,head].tolist(),'--',label='Zero write + initial control')
    ax.set(xlabel='DEV token',ylabel='Signed output',title='DEV head '+str(head),xticks=range(3));ax.legend()
display(fig)
t.testing.assert_close(dev_outputs[.4][0],t.tensor([1.05,2.225],dtype=t.float64))
print('DEV first-token absolute forecast error:',(dev_outputs[.4][0]-t.tensor(dev_forecast_first,dtype=t.float64)).abs().tolist())
assert zero_s.count_nonzero()==0
print('DEV output changes:',(dev_outputs[.4]-dev_outputs[0.]).tolist())''', 'dev-paired')
    md('''### Worked DEV reflection / self-check
The first-token prediction [1.05,2.225] follows directly from read-after-write. Later head-0
output falls from .0625 to .034 at token 2: the read [.5,.25] mixes transported
state with a new signed write. That observed decrease rules A out even though
initial carry raised token 0. A good reflection cites one increase, one decrease, the unchanged
quadratic/chunked factorization check, and the inert zero-write/zero-initial control.
A wrong monotonic forecast is a reason to revise the readout model, not tune the target.

### Try It Yourself — changed-condition DEV transfer
Freeze one change: negate ONLY the last-token read direction C[2]. Predict the last output
and the states before running. Explanation B predicts opposite last outputs with identical
states; a story that output sign directly measures stored-memory sign predicts otherwise.''', 'dev-reflection')
    code('''dev_transfer_C=practice_C.clone();dev_transfer_C[-1]*=-1
dev_transfer_forecast=(-dev_outputs[.4][-1]).clone()
dev_transfer_y,dev_transfer_s=dev_ns['ssd_recurrent'](practice_X,practice_A,practice_B,dev_transfer_C,practice_initial,.4)
t.testing.assert_close(dev_transfer_y[-1],-dev_outputs[.4][-1])  # Method invariant, not a forecast gate.
print('DEV last-token absolute forecast error:',(dev_transfer_y[-1]-dev_transfer_forecast).abs().tolist())
t.testing.assert_close(dev_transfer_s,dev_states[.4])
t.testing.assert_close(dev_transfer_y[:-1],dev_outputs[.4][:-1])
fig,ax=plt.subplots(figsize=(6,3),layout='constrained')
ax.plot(range(3),dev_outputs[.4][:,0].tolist(),'o-',label='Original read')
ax.plot(range(3),dev_transfer_y[:,0].tolist(),'o-',label='Last read negated')
ax.plot(range(3),zero_y[:,0].tolist(),'--',label='Inert control')
ax.set(xlabel='DEV token',ylabel='Head-0 signed output',title='Observed DEV read-direction transfer',xticks=range(3));ax.legend();display(fig)
print('DEV transfer last output:',dev_transfer_y[-1].tolist())
dev_revision=''  # Explain how identical states can give opposite last outputs.''', 'dev-transfer')
    md('''## Commit before any public-case result

Write a signed numerical forecast, an unscored confidence self-rating in [0,1], and causal reasoning. Only absolute point-forecast error is assessed; no probabilistic calibration follows. State exposure honestly: unknown, seen, or self_reported_not_seen. Freeze binds ordered inputs/layout/dtype, exact method/helper/test source and CPU runtime. Freeze/commit do not run the target. Change code or runtime? Freeze and commit again.''','commit-help')
    code("prediction=None\nconfidence=None  # unscored self-rating in [0,1], not a probability\nreasoning=''\nprior_exposure='unknown'",'forecast')
    code("method_source=\"\\n\\n\".join([RECURRENT_SOURCE,QUADRATIC_SOURCE,CHUNKED_SOURCE,READOUT_SOURCE,ASSEMBLY_SOURCE])\nidentity=lesson.freeze_inputs(method_source)\n# Fill the forecast cell, then explicitly commit:\ncommitment=lesson.commit_prediction(identity,method_source,prediction,confidence,reasoning,prior_exposure)",'commit')
    md('''## Reveal, then measure once

After commitment you may inspect reference code with `lesson.reveal_methods(commitment)`. It is diagnostic help; changing the committed implementation requires a new forecast/commitment. Only then run the public case and its controls. Keep any `ExperimentFailure.record_text`; it is an unscored failure, never silently retried.''','measure-help')
    code("try:\n    receipt=lesson.run_experiment(commitment,method_source)\nexcept lesson.ExperimentFailure as error:\n    failure_record=error.record_text\n    print('Keep failure_record and export it; diagnose before any retry.')\n    raise\nlesson.plot_receipt(receipt)\n# lesson.export_record(commitment, '/tmp/ssd-commitment.json')\n# lesson.export_record(receipt, '/tmp/ssd-receipt.json')",'measure')
    md('''## Reflect and forecast the single-factor transfer

Explain your forecast error using actual state/readout trajectories and the controls. No monotonicity follows merely from stronger carry because signed writes and read directions can cancel. The transfer changes only A→A/2; keep all other inputs and the same signed metric. Reflection and replay compute no new experiment.''','reflection-help')
    code("# attempt=lesson.reflect(receipt, explanation='',transfer_prediction=None,confidence=None,reasoning='')\n# transfer_identity=lesson.freeze_inputs(method_source,variant='transfer')\n# transfer_commitment=lesson.commit_transfer(attempt,transfer_identity,method_source,prior_exposure)\n# transfer_receipt=lesson.run_experiment(transfer_commitment,method_source)\n# lesson.plot_receipt(transfer_receipt)\n# lesson.export_attempt(attempt, '/tmp/ssd-reflection.json')",'transfer')
    md('''## Try It Yourself — a different practice filter

Change gamma, chunk size or an input in the T3 fixture and compare equivalent methods. This fixture is different from the public case, so its values do not reveal your primary or transfer answer. The orthogonal state-basis and split-cache tests also use this fixture.''','play-help')
    code("# This practice-only play never calls the public-case runner.\npractice_X,practice_A,practice_B,practice_C,practice_initial=lesson.practice_fixture()\npractice_gamma=.4\npractice_chunk_size=2\npractice_namespace=lesson.method_namespace(method_source)\npy,ps=practice_namespace['ssd_chunked'](practice_X,practice_A,practice_B,practice_C,practice_initial,practice_gamma,practice_chunk_size)\nimport matplotlib.pyplot as plt\nplt.plot(py.numpy());plt.xlabel('Practice token');plt.ylabel('Practice head output')",'practice-play')
    md('''## What the controls establish

After commitment, all three computations are checked against a scalar nested-loop oracle. Different chunk sizes, nondivisible padding, future perturbations, zero writes, gamma0, linearity and coherent head/readout swaps test algorithm correctness. Split-state/nonzero-initial-state and orthogonal basis controls use distinct practice inputs. Their success does not establish language-model memory or speed. This companion has no sampling, training, model loading, download, GPU or released-model claim. Historical 5.4 evidence stays in the full lesson.

Exports use private exclusive files; replay plots stored arrays without rerunning methods. Local hashes catch accidental drift, not hostile local editing or false exposure reports. Keep failed records and diagnose them before any retry.''','limits')
    return cells
