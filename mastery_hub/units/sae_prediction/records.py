"""Immutable local honesty receipts. Imports and freeze never evaluate a recipe.

Seals detect accidental drift, not hostile editing or false exposure declarations.
Replay uses recoverable stored snapshots and never compares today's installation.
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

ROOT = Path(__file__).resolve().parents[3]
UNIT = Path(__file__).resolve().parent
SCHEMA = 'arena-fixed-forward-sae'
VERSION = 2
SOURCE_LIMIT = 65_536
RECORD_LIMIT = 16_000_000
ERROR_SUMMARY_BYTES = 20_000
ERROR_MESSAGE_BYTES = 1_000_000
ERROR_ARTIFACT_BYTES = 8_000_000
# Authoritative anchor is deliberately independent of XDG_DATA_HOME and forecasts.
# DEV/review harnesses explicitly root-bind this constant to their private sandbox.
SESSION_ANCHOR = None  # None resolves the OS account home; never HOME/XDG.
METHOD_NAMES = ('affine','relu_encode','topk_encode','gated_encode','jump_encode','decode','measure_rows')
SIGNATURES = (('X','W','b'),('M',),('M','k','original_feature_ids'),('M','G'),('M','theta'),('Z','D','b_dec'),('X','Z','reconstruction'))
SOURCE_PATHS = ('mastery_hub/units/sae_prediction/protocol.json',
 'mastery_hub/units/sae_prediction/records.py','mastery_hub/units/sae_prediction/runner.py',
 'mastery_hub/units/sae_prediction/lesson.py','mastery_hub/units/sae_prediction/exercises.ipynb',
 'scripts/build_sae_prediction_notebook.py','tests/mastery/test_sae_prediction.py',
 'tests/mastery/test_sae_prediction_v2.py','tests/mastery/conftest.py')
ARRAY_NAMES = ('X','D','W_enc','b_enc','W_gate','b_gate','b_dec','jump_threshold')

def require(condition, message):
    if not condition: raise ValueError(message)

def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value): return sha(canonical(value).encode('utf-8'))
def keys(value, expected): require(type(value) is dict and set(value)==set(expected),'missing/unknown fields')
def number(value): require(type(value) in (int,float) and math.isfinite(value),'expected finite number excluding bool')
def explanation(value): require(type(value) is str and 0<len(value.strip()) and len(value.encode('utf-8'))<=20_000,'fill bounded explanatory text')
def _pairs(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,'duplicate JSON key');out[k]=v
    return out

def parse(text, limit=RECORD_LIMIT):
    try:
        require(type(text) is str and 0<len(text.encode('utf-8'))<=limit,'invalid/oversize UTF-8 JSON')
        return json.loads(text,object_pairs_hook=_pairs,parse_constant=lambda x:require(False,'nonfinite JSON'))
    except (UnicodeError,RecursionError,OverflowError,json.JSONDecodeError) as exc: raise ValueError('invalid bounded JSON') from exc

def _seal(value):
    value=deepcopy(value);value.pop('seal',None);value['seal']=digest(value)
    text=canonical(value);require(len(text.encode('utf-8'))<=RECORD_LIMIT,'record exceeds limit');return text

def _record(kind,payload,parent=None):return _seal({'schema':SCHEMA,'version':VERSION,'kind':kind,'parent':parent,'payload':payload})

def flat(values):
    if type(values) is list:
        for value in values:yield from flat(value)
    else:yield values

def shape(values):
    if type(values) is not list:return []
    require(len(values)>0,'empty array');dims=shape(values[0])
    require(all(shape(v)==dims for v in values),'ragged array');return [len(values),*dims]

def array_digest(values,dtype='float64'):
    require(dtype in ('float64','int64'),'unsupported array dtype')
    vals=list(flat(values));code='d' if dtype=='float64' else 'q'
    for value in vals:
        if dtype=='float64':number(value)
        else:require(type(value) is int and -(2**63)<=value<2**63,'expected int64 count')
    return sha(struct.pack('<'+code*len(vals),*vals))

def array(values,dtype='float64'):
    return {'values':deepcopy(values),'shape':shape(values),'dtype':dtype,'sha256':array_digest(values,dtype)}

def validate_array(value,dims,dtype='float64'):
    keys(value,('values','shape','dtype','sha256'))
    require(type(dims) is list and all(type(v) is int and v>0 for v in dims),'exact integer expected shape')
    require(type(value['shape']) is list and all(type(v) is int and v>0 for v in value['shape']),'exact integer array shape')
    require(dtype in ('float64','int64') and type(value['dtype']) is str,'array dtype')
    require(value['shape']==dims and shape(value['values'])==dims and value['dtype']==dtype,'array shape/dtype mismatch')
    require(value['sha256']==array_digest(value['values'],dtype),'array byte hash mismatch')
    return value['values']

def raw_inputs(case):return {name:array(case[name]) for name in ARRAY_NAMES}

# Versioned replay metadata: no current protocol file/runtime/source is consulted.
_RECIPE_KEYS = (
    'id','example_order','feature_order','method_order','X','D','W_enc','b_enc',
    'W_gate','b_gate','b_dec','jump_threshold','k','decoder_permutation',
    'coherent_feature_permutation','dtype','device','threads','tie_policy',
    'gate_boundary','jump_boundary','activation_count_boundary','metric',
)
_POLICY = {'dtype':'float64','device':'cpu','threads':1,
 'tie_policy':'descending positive value then ascending original feature index',
 'gate_boundary':'strictly_greater_than_zero','jump_boundary':'strictly_greater_than_threshold',
 'activation_count_boundary':'strictly_greater_than_zero',
 'metric':'mean_squared_error(topk)-mean_squared_error(jumprelu)'}
RUNTIME_KEYS = ('python','torch','platform','byteorder','torch_config','device','dtype','threads',
 'interop_threads','default_dtype','default_device','deterministic','matmul_precision','mkldnn',
 'cudnn_deterministic','cudnn_benchmark','tf32')

def stored_recipe(c):
    keys(c,_RECIPE_KEYS)
    require(type(c['X']) is list and c['X'] and type(c['X'][0]) is list and c['X'][0] and type(c['D']) is list and c['D'],'matrix collections')
    n=len(c['X']);d=len(c['X'][0]);f=len(c['D'])
    require(n>0 and d>0 and f>0,'empty recipe')
    for k,dims in {'X':[n,d],'D':[f,d],'W_enc':[d,f],'W_gate':[d,f],'b_enc':[f],
                   'b_gate':[f],'b_dec':[d],'jump_threshold':[f]}.items():
        require(shape(c[k])==dims,'recipe shape '+k);array_digest(c[k])
    require(type(c['k']) is int and 0<=c['k']<=f,'invalid k')
    require(all(v>=0 for v in c['jump_threshold']),'negative threshold')
    for k,v in _POLICY.items():require(type(c[k]) is type(v) and c[k]==v,'recipe policy')
    require(type(c['id']) is str and bool(c['id'].strip()),'recipe ID')
    for name,length in (('example_order',n),('feature_order',f),('method_order',4)):
        require(type(c[name]) is list and len(c[name])==length and all(type(v) is str and v.strip() for v in c[name]),'typed label collection')
        require(len(set(c[name]))==length,'unique labels')
    require(c['method_order']==['relu','topk','gated','jumprelu'],'method identities')
    for k in ('decoder_permutation','coherent_feature_permutation'):
        require(type(c[k]) is list and all(type(v) is int for v in c[k]) and sorted(c[k])==list(range(f)),'permutation')
    return n,d,f

def validate_source(source,complete=True):
    try:require(type(source) is str and 0<len(source.encode('utf-8'))<=SOURCE_LIMIT,'bounded source bytes');tree=ast.parse(source)
    except (UnicodeError,SyntaxError,RecursionError) as exc:raise ValueError('invalid method source') from exc
    require(tree.body and all(type(n) is ast.FunctionDef for n in tree.body),'plain definitions only')
    names=tuple(n.name for n in tree.body)
    require(names==METHOD_NAMES if complete else names==METHOD_NAMES[:len(names)],'seven ordered core methods')
    for node,signature in zip(tree.body,SIGNATURES):
        a=node.args
        require(not node.decorator_list and not node.returns and not node.type_params and
                not a.posonlyargs and not a.kwonlyargs and not a.vararg and not a.kwarg and
                not a.defaults and not a.kw_defaults and tuple(v.arg for v in a.args)==signature and
                all(v.annotation is None for v in a.args),'exact plain method signature')
    return tree

def namespace(source,complete=True):
    import torch as t
    validate_source(source,complete);out={'t':t,'require':require}
    exec(compile(source,'<inline SAE student methods>','exec'),out);return out

def runtime_identity():
    import torch as t
    return {'python':sys.version,'torch':str(t.__version__),'platform':platform.platform(),
      'byteorder':sys.byteorder,'torch_config':t.__config__.show(), 'device':'cpu','dtype':'float64',
      'threads':t.get_num_threads(),'interop_threads':t.get_num_interop_threads(),
      'default_dtype':str(t.get_default_dtype()),'default_device':str(t.get_default_device()),
      'deterministic':t.are_deterministic_algorithms_enabled(),
      'matmul_precision':t.get_float32_matmul_precision(),
      'mkldnn':t.backends.mkldnn.enabled,'cudnn_deterministic':t.backends.cudnn.deterministic,
      'cudnn_benchmark':t.backends.cudnn.benchmark,'tf32':t.backends.cuda.matmul.allow_tf32}

def snapshot_sources():
    result={}
    for relative in SOURCE_PATHS:
        if relative=='mastery_hub/units/sae_prediction/exercises.ipynb':
            # Hash a recoverable pristine template, never the evolving notebook
            # containing this very receipt. Generator baseline excludes itself.
            from scripts.build_sae_prediction_notebook import make_notebook
            raw=(json.dumps(make_notebook(),indent=1,ensure_ascii=False)+'\n').encode('utf-8')
        else:raw=(ROOT/relative).read_bytes()
        result[relative]={'utf8':raw.decode('utf-8'),'sha256':sha(raw)}
    return result

def _freeze(source,case,session):
    validate_source(source);stored_recipe(case);explanation(session)
    require(len(session.encode())<=200,'session too long')
    binding=bind_session(digest(case),session)
    return _record('identity',{'recipe':deepcopy(case),'recipe_sha256':digest(case),
      'raw_inputs':raw_inputs(case),'original_feature_ids':list(range(len(case['D']))),
      'method_source':source,'method_sha256':sha(source.encode()),'sources':snapshot_sources(),
      'runtime':runtime_identity(),'session':session,
      'attempt_registry':binding['attempt_registry'],'session_binding':binding})

def freeze(method_source,session='learner-primary-v1'):
    return _freeze(method_source,parse((UNIT/'protocol.json').read_text()),session)

def commit(identity_text,prediction,confidence,reasoning,prior_exposure='unknown'):
    identity=read(identity_text);require(identity['kind']=='identity','identity required')
    number(prediction);number(confidence);require(0<=confidence<=1,'confidence range')
    explanation(reasoning);explanation(prior_exposure)
    return _record('commitment',{'prediction':prediction,'confidence':confidence,'reasoning':reasoning,
                                'prior_exposure':prior_exposure},identity_text)

def _validate_identity(p):
    keys(p,('recipe','recipe_sha256','raw_inputs','original_feature_ids','method_source','method_sha256','sources','runtime','session','attempt_registry','session_binding'))
    stored_recipe(p['recipe']);require(p['recipe_sha256']==digest(p['recipe']),'recipe hash')
    keys(p['raw_inputs'],ARRAY_NAMES)
    for name,expected in raw_inputs(p['recipe']).items():
        actual=p['raw_inputs'][name];validate_array(actual,expected['shape'])
        require(actual['sha256']==expected['sha256'],'raw input byte identity')
    ids=p['original_feature_ids']
    require(type(ids) is list and all(type(v) is int for v in ids) and len(set(ids))==len(ids) and ids==list(range(len(p['recipe']['D']))),'typed unique original IDs')
    validate_source(p['method_source']);require(p['method_sha256']==sha(p['method_source'].encode()),'method source hash')
    keys(p['sources'],SOURCE_PATHS)
    for value in p['sources'].values():
        keys(value,('utf8','sha256'));require(type(value['utf8']) is str and value['sha256']==sha(value['utf8'].encode()),'source snapshot hash')
    keys(p['runtime'],RUNTIME_KEYS)
    for name in ('threads','interop_threads'):require(type(p['runtime'][name]) is int and p['runtime'][name]>0,'runtime thread count')
    for name in ('deterministic','mkldnn','cudnn_deterministic','cudnn_benchmark','tf32'):require(type(p['runtime'][name]) is bool,'runtime boolean flag')
    for name in set(RUNTIME_KEYS)-{'threads','interop_threads','deterministic','mkldnn','cudnn_deterministic','cudnn_benchmark','tf32'}:require(type(p['runtime'][name]) is str,'runtime string')
    explanation(p['session']);explanation(p['attempt_registry'])
    require(Path(p['attempt_registry']).is_absolute() and '\x00' not in p['attempt_registry'],'absolute frozen attempt registry')
    validate_binding(p['session_binding'])
    require(p['session_binding']['recipe_sha256']==p['recipe_sha256'] and p['session_binding']['session']==p['session'] and p['session_binding']['attempt_registry']==p['attempt_registry'],'bound authorized session identity')

def read(text,_depth=0):
    require(_depth<=3,'parent nesting limit');d=parse(text)
    keys(d,('schema','version','kind','parent','payload','seal'))
    require(d['schema']==SCHEMA and type(d['version']) is int and d['version']==VERSION,'unsupported schema/version')
    body={k:v for k,v in d.items() if k!='seal'};require(d['seal']==digest(body),'stale seal')
    p=d['payload'];kind=d['kind']
    if kind=='identity':require(d['parent'] is None,'identity parent');_validate_identity(p)
    else:
        require(type(d['parent']) is str,'parent text required');parent=read(d['parent'],_depth+1)
        if kind=='commitment':
            require(parent['kind']=='identity','commitment parent');keys(p,('prediction','confidence','reasoning','prior_exposure'))
            number(p['prediction']);number(p['confidence']);require(0<=p['confidence']<=1,'confidence range')
            explanation(p['reasoning']);explanation(p['prior_exposure'])
        elif kind in ('observation','failure'):
            require(parent['kind']=='commitment','result parent');identity=read(parent['parent'],_depth+1)['payload']
            if kind=='observation':
                from .runner import validate_saved
                validate_saved(p,identity['recipe'],parent['payload'])
            else:
                keys(p,('attempt_id','stage','error','secondary','captured','elapsed_seconds','write_error','error_details'))
                explanation(p['stage']);explanation(p['error']);require(type(p['secondary']) is list,'secondary findings')
                for v in p['secondary']:explanation(v)
                require(type(p['captured']) is dict,'partial data')
                for v in p['captured'].values():validate_array(v,v['shape'],v['dtype'])
                number(p['elapsed_seconds']);require(p['elapsed_seconds']>=0,'elapsed time')
                require(p['write_error'] is None or type(p['write_error']) is str,'write error')
                keys(p['error_details'],('primary','secondary','storage'))
                validate_error(p['error_details']['primary'])
                for group in ('secondary','storage'):
                    require(type(p['error_details'][group]) is list,'error collection')
                    for value in p['error_details'][group]:validate_error(value)
                require(p['error']==p['error_details']['primary']['summary'],'primary summary link')
                require(p['secondary']==[v['summary'] for v in p['error_details']['secondary']],'secondary summary link')
                storage=p['error_details']['storage']
                require(p['write_error']==(storage[0]['summary'] if storage else None),'storage summary link')
        elif kind=='reflection':
            require(parent['kind'] in ('observation','failure'),'reflection parent');keys(p,('explanation',));explanation(p['explanation'])
        else:raise ValueError('unknown record kind')
    return deepcopy(d)

def assert_current(commitment_text,source):
    d=read(commitment_text);require(d['kind']=='commitment','commitment required');identity=read(d['parent'])['payload']
    require(source==identity['method_source'],'actual source changed')
    require(snapshot_sources()==identity['sources'],'live source snapshot changed')
    require(canonical(runtime_identity())==canonical(identity['runtime']),'live runtime changed')
    require(load_binding(identity['recipe_sha256'],identity['session'])==identity['session_binding'],'live session binding changed')
    return identity

def reflect(parent_text,explanation_text):
    parent=read(parent_text);require(parent['kind'] in ('observation','failure'),'completed/failure record required')
    explanation(explanation_text);return _record('reflection',{'explanation':explanation_text},parent_text)

def private_write(path,text):
    path=Path(path)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as handle:handle.write(text);handle.flush();os.fsync(handle.fileno())
    return path

def export(record_text,new_path):
    read(record_text);return private_write(new_path,record_text)


def private_directory(path):
    path=Path(path)
    if path.exists():
        require(path.is_dir() and not path.is_symlink() and path.stat().st_mode&0o077==0,'private directory required')
    else:
        missing=[];ancestor=path
        while not ancestor.exists():missing.append(ancestor);ancestor=ancestor.parent
        require(not ancestor.is_symlink(),'symlink directory parent')
        for directory in reversed(missing):
            try:directory.mkdir(mode=0o700)
            except FileExistsError:require(directory.is_dir() and not directory.is_symlink() and directory.stat().st_mode&0o077==0,'private concurrent directory required')
    return path


def binding_path(recipe_sha256,session):
    anchor=SESSION_ANCHOR
    if anchor is None:
        try:
            import pwd
            anchor=Path(pwd.getpwuid(os.getuid()).pw_dir)/'.local/share/arena-mastery/sae_prediction/sessions'
        except (ImportError,AttributeError,KeyError) as error:
            raise ValueError('host must configure an absolute root-bound SESSION_ANCHOR') from error
    require(Path(anchor).is_absolute(),'root-bound session anchor must be absolute')
    return Path(anchor)/(digest({'recipe':recipe_sha256,'session':session})+'.binding.json')


def validate_binding(binding):
    keys(binding,('schema','version','recipe_sha256','session','anchor_path','attempt_registry','seal'))
    require(binding['schema']=='arena-sae-authorized-session' and type(binding['version']) is int and binding['version']==1,'session binding schema')
    body={k:v for k,v in binding.items() if k!='seal'};require(binding['seal']==digest(body),'session binding seal')
    require(type(binding['recipe_sha256']) is str and len(binding['recipe_sha256'])==64,'session recipe digest')
    explanation(binding['session'])
    for name in ('anchor_path','attempt_registry'):
        explanation(binding[name]);require(Path(binding[name]).is_absolute() and '\x00' not in binding[name],'absolute binding location')


def load_binding(recipe_sha256,session):
    path=binding_path(recipe_sha256,session)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'r',encoding='utf-8') as handle:
        require(os.fstat(handle.fileno()).st_mode&0o077==0,'private binding file required')
        binding=parse(handle.read(20_001),20_000)
    validate_binding(binding)
    require(binding['anchor_path']==str(path) and binding['recipe_sha256']==recipe_sha256 and binding['session']==session,'authoritative session binding')
    return binding


def bind_session(recipe_sha256,session):
    path=binding_path(recipe_sha256,session);private_directory(path.parent)
    proposed={'schema':'arena-sae-authorized-session','version':1,'recipe_sha256':recipe_sha256,
              'session':session,'anchor_path':str(path),
              'attempt_registry':str(Path(os.environ.get('XDG_DATA_HOME',str(Path.home()/'.local/share'))).expanduser().absolute()/'arena-mastery'/'sae_prediction'/'attempts')}
    text=_seal(proposed)
    try:private_write(path,text)
    except FileExistsError:pass
    # Resolve the FIRST exclusive binding. Re-freeze never chooses a new ledger.
    return load_binding(recipe_sha256,session)


def byte_prefix(text,limit):
    return text.encode('utf-8')[:limit].decode('utf-8','ignore')


def error_summary(exception_type,message,original_utf8_bytes):
    display=exception_type+': '+message
    truncated=len((exception_type+': ').encode('utf-8'))+original_utf8_bytes>ERROR_SUMMARY_BYTES
    marker=' [truncated; see error_details]'
    return byte_prefix(display,ERROR_SUMMARY_BYTES-len(marker.encode()))+marker if truncated else display


def error_detail(error):
    """Byte-aware display plus a separately bounded recoverable exact message.

    Inline recovery is exact up to ERROR_MESSAGE_BYTES. Larger original messages
    may be stored as a private artifact by the runner, up to ERROR_ARTIFACT_BYTES.
    Both levels disclose truncation; never claim a prefix is the complete error.
    """
    message=str(error);raw=message.encode('utf-8');display=type(error).__name__+': '+message
    return {'exception_type':type(error).__name__,'summary':error_summary(type(error).__name__,message,len(raw)),
            'summary_truncated':len(display.encode('utf-8'))>ERROR_SUMMARY_BYTES,
            'message':byte_prefix(message,ERROR_MESSAGE_BYTES),'message_truncated':len(raw)>ERROR_MESSAGE_BYTES,
            'original_utf8_bytes':len(raw),'original_sha256':sha(raw),'recovery_artifact':None}


def validate_error(value):
    keys(value,('exception_type','summary','summary_truncated','message','message_truncated','original_utf8_bytes','original_sha256','recovery_artifact'))
    explanation(value['exception_type']);explanation(value['summary'])
    require(type(value['message']) is str and len(value['message'].encode('utf-8'))<=ERROR_MESSAGE_BYTES,'bounded recoverable error message')
    require(type(value['summary_truncated']) is bool and type(value['message_truncated']) is bool,'error truncation indicators')
    require(type(value['original_utf8_bytes']) is int and value['original_utf8_bytes']>=0,'original error byte size')
    require(type(value['original_sha256']) is str and len(value['original_sha256'])==64,'original error digest')
    raw=value['message'].encode('utf-8')
    require(value['message_truncated']==(value['original_utf8_bytes']>ERROR_MESSAGE_BYTES),'message truncation disclosure')
    require(value['summary']==error_summary(value['exception_type'],value['message'],value['original_utf8_bytes']),'error summary prefix')
    require(value['summary_truncated']==(len((value['exception_type']+': ').encode())+value['original_utf8_bytes']>ERROR_SUMMARY_BYTES),'summary truncation disclosure')
    if not value['message_truncated']:
        require(len(raw)==value['original_utf8_bytes'] and sha(raw)==value['original_sha256'],'exact original error recovery')
    else:require(len(raw)<=value['original_utf8_bytes'],'partial original error recovery')
    artifact=value['recovery_artifact']
    if artifact is not None:
        keys(artifact,('path','bytes','sha256'));explanation(artifact['path'])
        require(
            Path(artifact['path']).is_absolute() and type(artifact['bytes']) is int
            and artifact['bytes']==value['original_utf8_bytes']
            and artifact['bytes']<=ERROR_ARTIFACT_BYTES
            and artifact['sha256']==value['original_sha256'],
            'bounded recovery artifact identity',
        )
