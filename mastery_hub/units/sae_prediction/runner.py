"""One explicit opt-in CPU attempt. No numerical work at import/freeze/commit.

The fixed-loop oracle is an independent scalar implementation, disclosed after
commit. Replay validates saved evidence, never calls this oracle or student code.
"""
from copy import deepcopy
import math
import os
from pathlib import Path
import time
from . import records as r

METHODS=('relu','topk','gated','jumprelu')
CONDITIONS=('primary','coherent','zero','mismatch')
ATOL=RTOL=1e-12

def close(a,b):
    return type(a) in (float,int) and type(b) in (float,int) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=ATOL+RTOL*abs(b)

def equal(a,b):
    if type(a) is list and type(b) is list:return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return close(a,b)

def scalar_oracle(case):
    """Explicit Python-float i,j,k loops; no tensor operations or fsum/reordering."""
    n,d,f=r.stored_recipe(case)
    def affine(X,W,b):
        out=[]
        for i in range(n):
            row=[]
            for j in range(f):
                value=0.0
                for k in range(d):value+=float(X[i][k])*float(W[k][j])
                row.append(value+float(b[j]))
            out.append(row)
        return out
    def condition(c,ids,zero=False,mismatch=False):
        M=affine(c['X'],c['W_enc'],c['b_enc']);G=affine(c['X'],c['W_gate'],c['b_gate'])
        codes={m:[] for m in METHODS};recons={m:[] for m in METHODS}
        errors={m:[] for m in METHODS};counts={m:[] for m in METHODS};residuals={m:[] for m in METHODS}
        D=c['D'] if not mismatch else [c['D'][j] for j in c['decoder_permutation']]
        for i in range(n):
            positives=[max(float(v),0.0) for v in M[i]]
            selected=sorted(range(f),key=lambda j:(-positives[j],ids[j]))[:c['k']]
            for method in METHODS:
                z=[]
                for j in range(f):
                    active=(method=='relu' or method=='topk' and j in selected or method=='gated' and G[i][j]>0 or method=='jumprelu' and positives[j]>c['jump_threshold'][j])
                    z.append(positives[j] if active and not zero else 0.0)
                reconstruction=[]
                for k in range(d):
                    value=0.0
                    for j in range(f):value+=z[j]*float(D[j][k])
                    reconstruction.append(value+float(c['b_dec'][k]))
                residual=[reconstruction[k]-c['X'][i][k] for k in range(d)];e=0.0
                for value in residual:e+=value*value
                codes[method].append(z);recons[method].append(reconstruction);residuals[method].append(residual)
                errors[method].append(e/d);counts[method].append(sum(v>0 for v in z))
        return {'M':r.array(M),'G':r.array(G),**{key:{m:r.array(value[m],'int64' if key=='counts' else 'float64') for m in METHODS}
                    for key,value in [('codes',codes),('reconstructions',recons),('residuals',residuals),('errors',errors),('counts',counts)]}}
    p=case['coherent_feature_permutation'];coherent=deepcopy(case)
    for name in ('W_enc','W_gate'):coherent[name]=[[row[j] for j in p] for row in case[name]]
    for name in ('b_enc','b_gate','jump_threshold','D'):coherent[name]=[deepcopy(case[name][j]) for j in p]
    return {'primary':condition(case,list(range(f))), 'coherent':condition(coherent,p),
            'zero':condition(case,list(range(f)),zero=True),'mismatch':condition(case,list(range(f)),mismatch=True)}

def _tensors(case):
    import torch as t
    return {name:t.tensor(case[name],dtype=t.float64,device='cpu') for name in r.ARRAY_NAMES}

def _tensor(value,dims,dtype='float64'):
    import torch as t
    r.require(isinstance(value,t.Tensor) and value.device.type=='cpu' and
              value.dtype==(t.float64 if dtype=='float64' else t.int64) and list(value.shape)==dims and
              bool(t.isfinite(value).all()),'expected finite CPU '+dtype+' '+str(dims))
    return value

def _snapshot_args(args):
    import torch as t
    result=[]
    for value in args:
        if isinstance(value,t.Tensor):
            result.append((str(value.dtype),str(value.device),list(value.shape),r.array_digest(value.detach().tolist(),'int64' if value.dtype==t.int64 else 'float64')))
        else:result.append(deepcopy(value))
    return result

class AttemptFailure(ValueError):
    def __init__(self,record_text):
        self.record_text=record_text
        super().__init__(r.read(record_text)['payload']['error'])

def _private_directory(path):return r.private_directory(path)

def run(commitment_text,method_source,attempt_directory,*,authorized=False):
    """Future target dispatch requires separate explicit authorization. Ships dormant.

    Failure: capture -> finally mutation/live checks -> semantic checks. Persist
    the exact exception-carried failure text. A write failure retains the start
    ledger, any partial journal, and a failure text in the raised exception.
    """
    r.require(authorized is True,'explicit later authorization required; target dormant')
    identity=r.assert_current(commitment_text,method_source)
    import torch as t
    r.require(t.get_num_threads()==1,'one CPU thread required')
    case=identity['recipe'];session_key=r.digest({'recipe':identity['recipe_sha256'],'session':identity['session']})
    ledger_root=_private_directory(Path(identity['attempt_registry']))
    ledger=ledger_root/(session_key+'.start.json')
    start={'recipe_sha256':identity['recipe_sha256'],'session':identity['session'],'commitment_sha256':r.sha(commitment_text.encode()),'status':'started_or_unknown'}
    try:r.private_write(ledger,r.canonical(start))
    except FileExistsError as exc:raise ValueError('recipe/session attempt already consumed') from exc
    started=time.monotonic();attempt_id=r.digest(start);captured={};secondary=[];stage=['setup'];out=None;sequence=[0]
    ns=None;function_identity=None
    def current():
        if ns is not None and function_identity is not None:
            r.require({name:(id(ns[name]),id(ns[name].__code__),ns[name].__defaults__,ns[name].__kwdefaults__) for name in r.METHOD_NAMES}==function_identity,'student changed function source')
        r.assert_current(commitment_text,method_source)
        r.require(time.monotonic()-started<=60,'60 second wall budget exceeded')
        r.require(time.process_time()-cpu_start<=60,'60 second CPU budget exceeded')
    def journal():
        if out is not None:
            sequence[0]+=1;r.private_write(out/f'partial-{sequence[0]:04d}.json',r.canonical({'stage':stage[0],'captured':captured}))
    def call(label,fn,args,returns):
        stage[0]=label;current();before=_snapshot_args(args);original=None;returned=None;post=[]
        try:
            returned=fn(*args)
            values=returned if len(returns)>1 and type(returned) in (tuple,list) else (returned,)
            for index,(dims,dtype) in enumerate(returns):
                if index>=len(values):continue
                try:_tensor(values[index],dims,dtype)
                except ValueError:continue
                # Capture finite values before post-call checks, including wrong values.
                captured[label+'/'+str(index)]=r.array(values[index].detach().clone().tolist(),dtype)
            journal()
        except Exception as exc:original=exc
        finally:
            try:r.require(_snapshot_args(args)==before,'student mutated actual input/config')
            except Exception as exc:post.append(exc)
            try:current()
            except Exception as exc:post.append(exc)
            secondary.extend(post)
        if original is not None:raise original
        r.require(not post,'; '.join(str(exc) for exc in post))
        values=returned if len(returns)>1 and type(returned) in (tuple,list) else (returned,)
        r.require(len(values)==len(returns),'wrong return arity')
        for value,(dims,dtype) in zip(values,returns):_tensor(value,dims,dtype)
        # Downstream functions receive independent detached storage even on aliases.
        return tuple(v.detach().clone() for v in values) if len(returns)>1 else values[0].detach().clone()
    cpu_start=time.process_time()
    try:
        requested=Path(attempt_directory);r.require(not requested.exists(),'attempt export directory must be new')
        out=_private_directory(requested)
        ns=r.namespace(method_source);original_functions={k:ns[k] for k in r.METHOD_NAMES}
        function_identity={name:(id(ns[name]),id(ns[name].__code__),ns[name].__defaults__,ns[name].__kwdefaults__) for name in r.METHOD_NAMES}
        def checked(label,name,args,returns):
            r.require(all(ns[k] is original_functions[k] for k in r.METHOD_NAMES),'student replaced function source')
            value=call(label,ns[name],args,returns)
            r.require(all(ns[k] is original_functions[k] for k in r.METHOD_NAMES),'student replaced function source')
            return value
        observation=_evaluate(case,checked,current)
        observation.update(attempt_id=attempt_id,elapsed_seconds=time.monotonic()-started)
        parent=r.read(commitment_text)['payload'];observation['forecast_error']=abs(parent['prediction']-observation['metric'])
        text=r._record('observation',observation,commitment_text);r.read(text);current()
        r.export(text,out/'observation.json');return text
    except Exception as error:
        details={'primary':r.error_detail(error),
                 'secondary':[r.error_detail(exc) for exc in secondary],'storage':[]}
        def retain_original(detail,exc,role):
            if not detail['message_truncated']:return
            if detail['original_utf8_bytes']>r.ERROR_ARTIFACT_BYTES:
                details['storage'].append(r.error_detail(ValueError('Original error exceeds bounded recovery artifact limit; full message unavailable in serialized evidence')))
                return
            path=(out if out is not None else ledger_root)/(session_key+'.'+role+'.message.txt')
            try:
                r.private_write(path,str(exc))
                detail['recovery_artifact']={'path':str(path.absolute()),'bytes':detail['original_utf8_bytes'],'sha256':detail['original_sha256']}
            except Exception as artifact_error:details['storage'].append(r.error_detail(artifact_error))
        retain_original(details['primary'],error,'primary')
        for index,(detail,exc) in enumerate(zip(details['secondary'],secondary)):
            retain_original(detail,exc,'secondary-'+str(index))
        def failure_text():
            payload={'attempt_id':attempt_id,'stage':stage[0],
                     'error':details['primary']['summary'],
                     'secondary':[v['summary'] for v in details['secondary']],
                     'captured':captured,'elapsed_seconds':time.monotonic()-started,
                     'write_error':details['storage'][0]['summary'] if details['storage'] else None,
                     'error_details':details}
            text=r._record('failure',payload,commitment_text)
            r.read(text)  # Validate the exact writer/reader byte contract BEFORE emitting.
            return text
        text=failure_text()
        try:
            if out is None:raise OSError('attempt directory unavailable; stable start retained')
            r.export(text,out/'failure.json')
        except Exception as write_error:
            details['storage'].append(r.error_detail(write_error));text=failure_text()
            try:r.private_write(ledger_root/(session_key+'.failure.json'),text)
            except Exception as fallback_error:
                details['storage'].append(r.error_detail(fallback_error));text=failure_text()
        # Every emitted text has passed read; the primary and all storage errors
        # survive even when no destination can retain the complete final text.
        raise AttemptFailure(text) from error

def _evaluate(case,call,current):
    import torch as t
    n,d,f=r.stored_recipe(case);oracle=scalar_oracle(case);current()
    def condition(c,ids,zero=False,mismatch=False):
        a=_tensors(c);label='zero' if zero else ('mismatch' if mismatch else ('primary' if ids==tuple(range(f)) else 'coherent'))
        def one(name,args,dims,dtype='float64',tag=None):
            return call(label+'/'+(tag or name),name,tuple(v.clone() if isinstance(v,t.Tensor) else v for v in args),[(dims,dtype)])
        M=one('affine',(a['X'],a['W_enc'],a['b_enc']),[n,f],tag='M/affine');G=one('affine',(a['X'],a['W_gate'],a['b_gate']),[n,f],tag='G/affine')
        values={'M':r.array(M.tolist()),'G':r.array(G.tolist()),**{k:{} for k in ('codes','reconstructions','residuals','errors','counts')}}
        for method in METHODS:
            if zero:Z=t.zeros((n,f),dtype=t.float64,device='cpu')
            elif method=='relu':Z=one('relu_encode',(M,),[n,f])
            elif method=='topk':Z=one('topk_encode',(M,c['k'],ids),[n,f])
            elif method=='gated':Z=one('gated_encode',(M,G),[n,f])
            else:Z=one('jump_encode',(M,a['jump_threshold']),[n,f])
            D=a['D'][c['decoder_permutation']].clone() if mismatch else a['D'].clone()
            R=one('decode',(Z,D,a['b_dec']),[n,d],tag=method+'/decode');E,L=call(label+'/'+method+'/measure_rows','measure_rows',(a['X'].clone(),Z.clone(),R.clone()),[([n],'float64'),([n],'int64')])
            for key,value,dtype in [('codes',Z,'float64'),('reconstructions',R,'float64'),('residuals',R-a['X'],'float64'),('errors',E,'float64'),('counts',L,'int64')]:values[key][method]=r.array(value.tolist(),dtype)
        # Oracle comparisons happen only after valid returned arrays are retained.
        _compare_condition(values,oracle[label]);return values
    primary=condition(case,tuple(range(f)))
    p=case['coherent_feature_permutation'];coherent=deepcopy(case)
    for name in ('W_enc','W_gate'):coherent[name]=[[row[j] for j in p] for row in case[name]]
    for name in ('b_enc','b_gate','jump_threshold','D'):coherent[name]=[deepcopy(case[name][j]) for j in p]
    conditions={'primary':primary,'coherent':condition(coherent,tuple(p)),
                'zero':condition(case,tuple(range(f)),zero=True),'mismatch':condition(case,tuple(range(f)),mismatch=True)}
    means={m:sum(primary['errors'][m]['values'])/n for m in METHODS}
    return {'conditions':conditions,'oracle':oracle,'means':means,'metric':means['topk']-means['jumprelu'],
            'diagnostics':{'scalar_oracle':True,'coherent_invariance':True}}

def _compare_condition(actual,expected):
    r.keys(actual,expected)
    for name in ('M','G'):r.require(equal(actual[name]['values'],expected[name]['values']),'scalar oracle affine mismatch')
    for group in ('codes','reconstructions','residuals','errors','counts'):
        for method in METHODS:
            a=actual[group][method]['values'];b=expected[group][method]['values']
            r.require(a==b if group=='counts' else equal(a,b),'scalar oracle '+group+' mismatch')
            if group=='codes':r.require([v>0 for v in r.flat(a)]==[v>0 for v in r.flat(b)],'exact support mismatch')

def validate_saved(p,case,forecast):
    """Stored-only support/residual/reduction checks; no constructor/oracle/runner.

    Oracle arrays are retained evidence. Compare them directly; do not regenerate
    expected forward arrays. Replay does not certify hostile hand-forged records.
    """
    r.keys(p,('conditions','oracle','means','metric','diagnostics','attempt_id','elapsed_seconds','forecast_error'))
    n,d,f=r.stored_recipe(case);r.keys(p['conditions'],CONDITIONS);r.keys(p['oracle'],CONDITIONS)
    for label in CONDITIONS:
        actual=p['conditions'][label];saved_oracle=p['oracle'][label]
        for c in (actual,saved_oracle):
            r.keys(c,('M','G','codes','reconstructions','residuals','errors','counts'))
            r.validate_array(c['M'],[n,f]);r.validate_array(c['G'],[n,f])
            for group,dims,dtype in [('codes',[n,f],'float64'),('reconstructions',[n,d],'float64'),('residuals',[n,d],'float64'),('errors',[n],'float64'),('counts',[n],'int64')]:
                r.keys(c[group],METHODS)
                for m in METHODS:r.validate_array(c[group][m],dims,dtype)
        _compare_condition(actual,saved_oracle)
        for m in METHODS:
            Z=actual['codes'][m]['values'];R=actual['reconstructions'][m]['values'];E=actual['errors'][m]['values'];L=actual['counts'][m]['values'];res=actual['residuals'][m]['values']
            r.require(all(v>=0 for v in r.flat(Z)),'negative code')
            for i in range(n):
                r.require(L[i]==sum(v>0 for v in Z[i]),'support count mismatch')
                r.require(equal(res[i],[R[i][j]-case['X'][i][j] for j in range(d)]),'residual mismatch')
                r.require(close(E[i],sum(v*v for v in res[i])/d),'per-row MSE reduction mismatch')
                if label=='zero':r.require(all(v==0 for v in Z[i]),'zero code')
                if m=='topk':r.require(L[i]<=case['k'],'TopK bound')
                positive=[max(v,0.0) for v in actual['M']['values'][i]]
                ids=case['coherent_feature_permutation'] if label=='coherent' else list(range(f))
                theta=[case['jump_threshold'][j] for j in ids]
                selected=sorted(range(f),key=lambda j:(-positive[j],ids[j]))[:case['k']]
                for j in range(f):
                    active=(label!='zero' and positive[j]>0 and
                            (m=='relu' or m=='topk' and j in selected or
                             m=='gated' and actual['G']['values'][i][j]>0 or
                             m=='jumprelu' and positive[j]>theta[j]))
                    r.require((Z[i][j]>0)==active,'stored exact support condition')
                    r.require(close(Z[i][j],positive[j] if active else 0.0),'stored amplitude condition')
    pfeat=case['coherent_feature_permutation'];primary=p['conditions']['primary'];coherent=p['conditions']['coherent']
    for m in METHODS:
        for i in range(n):
            r.require(equal(coherent['codes'][m]['values'][i],[primary['codes'][m]['values'][i][j] for j in pfeat]),'coherent code invariance')
        r.require(equal(coherent['reconstructions'][m]['values'],primary['reconstructions'][m]['values']),'coherent reconstruction invariance')
    r.keys(p['means'],METHODS)
    for m in METHODS:r.require(close(p['means'][m],sum(primary['errors'][m]['values'])/n),'mean includes all rows')
    r.require(close(p['metric'],p['means']['topk']-p['means']['jumprelu']),'signed metric')
    r.require(close(p['forecast_error'],abs(forecast['prediction']-p['metric'])),'forecast error')
    r.keys(p['diagnostics'],('scalar_oracle','coherent_invariance'))
    r.require(all(value is True for value in p['diagnostics'].values()),'diagnostics')
    r.number(p['elapsed_seconds']);r.require(p['elapsed_seconds']>=0,'elapsed')
    r.require(type(p['attempt_id']) is str and len(p['attempt_id'])==64,'attempt identity')
