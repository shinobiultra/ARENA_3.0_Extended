"""Closed-scope saved-evidence verifier. Never imports or executes lesson code.

This is separate from release audits. A pass means technical saved execution
identity only, never human acceptance, a new run, or empirical quality admission.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any
import yaml
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parents[1]
ANCHORS_PATH = ROOT / 'scripts/fresh_lesson_evidence_anchors.json'
ANCHORS_SHA256 = 'ed4e319ff5073859828c26e19d97f2d50e0b13c66b210c6bab3e1f060de57694'
METHODS = {
    '7.1': ('7_1_logit_lens_tuned_lens_and_patchscopes', 'chapter7_activation_to_language/exercises/part1_lenses_patchscopes'),
    '8.4': ('8_4_acdc_and_circuit_metrics', 'chapter8_automated_circuits/exercises/part4_acdc_circuit_metrics'),
    '12.3': ('12_3_mini_vlm_from_scratch', 'chapter12_vlm_interpretability/exercises/part3_mini_vlm_from_scratch'),
}
class EvidenceError(ValueError):
    """Evidence is missing, changed, malformed or inconsistent."""
def require(condition: bool, message: str) -> None:
    if not condition: raise EvidenceError(message)
def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def safe_path(root: Path, relative: str) -> Path:
    require(type(relative) is str and bool(relative), 'path must be a nonempty string')
    p = PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts and '.' not in relative.split('/') and '\\' not in relative, 'unsafe path')
    current = root
    for part in p.parts:
        current = current / part
        require(not current.is_symlink(), 'symlink evidence path rejected')
    require(current.resolve().is_relative_to(root.resolve()), 'path escapes root')
    return current

def read(root: Path, name: str) -> bytes:
    try: return safe_path(root, name).read_bytes()
    except OSError as exc: raise EvidenceError(f'missing evidence: {name}') from exc

def require_hash(root: Path, name: str, expected: str, size: int | None = None) -> bytes:
    b = read(root, name)
    require(sha(b) == expected and (size is None or len(b) == size), f'changed evidence: {name}')
    return b

def source(cell: dict) -> str:
    return ''.join(cell['source'])
def cell_ast(cell: dict) -> ast.AST:
    # Parse IPython syntax without executing it. Transformed source also exposes
    # shell/magic calls in the AST rather than deleting their lines.
    return ast.parse(TransformerManager().transform_cell(source(cell)))
def code_cells(nb: dict) -> list[dict]:
    return [c for c in nb['cells'] if c['cell_type'] == 'code']
def output_text(c: dict) -> str:
    return '\n'.join(''.join(o.get('text', [])) if o.get('output_type') == 'stream' else ''.join(o.get('data', {}).get('text/plain', [])) for o in c.get('outputs', []))
def notebook_assertions(nb: dict) -> list[dict]:
    return [{'code_ordinal': i, 'source_sha256': sha(source(c).encode()), 'assertion': ast.get_source_segment(source(c), n), 'line': n.lineno}
            for i,c in enumerate(code_cells(nb)) for n in ast.walk(cell_ast(c)) if isinstance(n, ast.Assert)]
def test_calls(cell: dict) -> list[str]:
    return [n.func.attr for n in ast.walk(cell_ast(cell)) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
            and n.func.value.id == 'tests' and n.func.attr.startswith('test_')]

def verify_sequence(raw: dict, result: dict, expected_count: int, cell_limit: float) -> None:
    indexed = [(i,c) for i,c in enumerate(raw['cells']) if c['cell_type'] == 'code']
    require(type(result['completed_code_cells']) is int and type(result['expected_code_cells']) is int, 'integer cell counts required')
    require(result['completed_code_cells'] == result['expected_code_cells'] == len(indexed) == expected_count, 'incomplete code sequence')
    require(len(result['cells']) == expected_count, 'missing or duplicated cell receipts')
    hashes = [sha(source(c).encode()) for _,c in indexed]
    require(hashes == result['original_code_sha256s'], 'source sequence mismatch')
    for ordinal, ((index,c), receipt) in enumerate(zip(indexed,result['cells'])):
        require(set(receipt) == {'index','source_sha256','duration_seconds','execution_count','has_error'}, 'unknown cell receipt fields')
        require(type(receipt['index']) is int and receipt['index'] == index, 'wrong ordered cell index')
        require(receipt['source_sha256'] == hashes[ordinal], 'receipt source mismatch')
        require(type(receipt['execution_count']) is int and type(c['execution_count']) is int and receipt['execution_count'] == c['execution_count'] == ordinal+1, 'wrong execution count')
        require(receipt['has_error'] is False and not any(o.get('output_type') == 'error' for o in c['outputs']), 'code error present')
        t = receipt['duration_seconds']
        require(type(t) in (int,float) and math.isfinite(t) and 0 <= t <= cell_limit, 'cell duration exceeds frozen budget')

def verify_recovery(raw: dict, recovered: dict) -> None:
    expected = copy.deepcopy(raw)
    require(expected['cells'][176].get('id') == '24eb8d04', 'wrong frozen recovery target')
    del expected['cells'][176]['id']
    require(expected == recovered, 'recovery changes more than the sole approved id')

def observations(cells: list[dict]) -> list[dict]:
    rows=[]
    for i,c in enumerate(cells):
        text=output_text(c)
        if text.strip() and not re.fullmatch(r'\s*All tests in `[^`]+` passed!\s*',text):
            rows.append({
                'lesson_code_ordinal':i,
                'cell_source_sha256':sha(source(c).encode()),
                'plain_output':text,
                'output_sha256':sha(json.dumps(c['outputs'],sort_keys=True).encode()),
                'role':'descriptive saved observation; no newly fitted pass threshold'
            })
    return rows

def verify_trace_arithmetic(trace: dict) -> dict:
    """Recompute saved-logit decisions; 1e-12 matches the retained science review."""
    from collections import defaultdict
    accuracies={}
    for key,expected in [('image_only_traces',10000),('qwen_control_traces',500)]:
        rows=trace[key];require(len(rows)==expected,'trace denominator changed')
        groups=defaultdict(list);seen=set()
        for row in rows:
            identity=(row['condition'],row['kind'],row['scene_index'])
            require(identity not in seen,'duplicate scientific trace row');seen.add(identity)
            logits=row['candidate_logits'];words=row['candidate_words']
            require(len(logits)==len(words)>0 and all(type(x) in (int,float) and math.isfinite(x) for x in logits),'nonfinite candidate logits')
            prediction=words[max(range(len(logits)),key=logits.__getitem__)]
            require(prediction==row['prediction'] and row['correct'] is (prediction==row['answer']),'saved argmax/correct mismatch')
            groups[(row['condition'],row['kind'])].append(row['correct'])
        for (condition,kind),values in groups.items():
            actual=sum(values)/len(values)
            expected_value=trace['image_only_accuracy'][condition][kind] if key=='image_only_traces' else trace['qwen_accuracy'][condition]
            require(actual==expected_value,'saved accuracy aggregation mismatch')
            accuracies[condition+'/'+kind]={'correct':sum(values),'denominator':len(values),'accuracy':actual}
    pairs=trace['qwen_pair_traces'];require(len(pairs)==1000,'patch row denominator changed')
    grouped=defaultdict(list);seen=set()
    for row in pairs:
        identity=(row['pair_index'],row['layer'],row['token_set'])
        require(identity not in seen,'duplicate patch row');seen.add(identity)
        cl=row['clean_logit_difference'];cr=row['corrupt_logit_difference'];pa=row['patched_logit_difference']
        require(all(math.isfinite(x) for x in (cl,cr,pa)) and cl-cr>0,'undefined patch denominator')
        value=(pa-cr)/(cl-cr)
        require(abs(row['denominator']-(cl-cr))<1e-12 and abs(value-row['recovery'])<1e-12,'patch recovery arithmetic mismatch')
        grouped[(row['token_set'],row['layer'])].append(value)
    for (name,layer),values in grouped.items():
        require(len(values)==20,'patch group denominator changed')
        require(abs(sum(values)/20-trace['qwen_recovery'][name][trace['qwen_layers'].index(layer)])<1e-12,'patch mean mismatch')
    return accuracies

def verify(section: str, root: Path = ROOT, *, cache_digests: dict | None = None) -> dict:
    """Verify one anchored method; cache_digests is process-local for shared assets."""
    require(section in METHODS, 'unreviewed section')
    anchors = json.loads(require_hash(root,'scripts/fresh_lesson_evidence_anchors.json',ANCHORS_SHA256))
    require(set(anchors) == set(METHODS), 'unknown anchor method')
    a=anchors[section];method,directory=METHODS[section]
    require(a['method'] == method and a['directory'] == directory, 'wrong method/path identity')
    data={}
    for name,entry in a['files'].items():
        require(set(entry)=={'sha256','size_bytes'}, 'unknown anchor fields')
        b=require_hash(root,name,entry['sha256'],entry['size_bytes'])
        if name.endswith('.json'): data[name]=json.loads(b)
        elif name.endswith('.yml'): data[name]=yaml.safe_load(b)
    base=a['base'];run=base+'/'+a['run'];mini=section=='12.3'
    get=lambda name: data[name]
    protocol=get(base+'/protocol.json');result=get(run+'/result.json');runtime=get(run+'/kernel_runtime.json')
    manifest_name=base+('/source_manifest.json' if mini else '/before_fresh_manifest.json');manifest=get(manifest_name)
    require(result['protocol_sha256']==sha(read(root,base+'/protocol.json')), 'result/protocol mismatch')
    for key,filename in [('runner_sha256','runner.py'),('kernel_guard_sha256','cuda_kernel.py'),('cache_manifest_sha256','cache_manifest.json'),('source_manifest_sha256' if mini else 'before_manifest_sha256',manifest_name.split('/')[-1])]:
        require_hash(root,base+'/'+filename,protocol[key])
    kernel_name=next(n for n in a['files'] if n.endswith('/kernel.json'));kernel=get(kernel_name)
    require(set(kernel)=={'argv','display_name','language'} and kernel['language']=='python','invalid kernel spec')
    require(kernel['argv']==[str(ROOT/'.venv/bin/python'),str(ROOT/base/'cuda_kernel.py'),'-f','{connection_file}'],'wrong frozen kernel argv')
    if mini: require_hash(root,kernel_name,protocol['kernel_spec_sha256'])
    require(runtime['python']==kernel['argv'][0] and runtime['torch']==protocol['versions']['torch'],'runtime interpreter/torch mismatch')
    require(set(runtime['versions'])=={'transformer-lens','transformers','huggingface-hub','ipykernel'},'runtime version coverage changed')
    for package,v in runtime['versions'].items(): require(protocol['versions'][package]==v,'runtime package version mismatch')
    require(runtime['allocation_fraction']==protocol['allocation_fraction']==0.9 and runtime['cuda_visible_devices']==protocol['environment']['CUDA_VISIBLE_DEVICES']=='0','runtime device policy mismatch')
    require(
        runtime['cuda_build']=='13.2' and runtime['default_dtype']=='torch.float32'
        and runtime['matmul_allow_tf32'] is False and runtime['cudnn_allow_tf32'] is True
        and runtime['deterministic'] is False,
        'runtime numerical policy mismatch'
    )
    require('5090' in runtime['gpu'] and (not mini or runtime['phase']=='cuda_ready'),'runtime GPU not ready')
    require(result['peak_whole_gpu_mib']<=protocol['whole_gpu_ceiling_mib'] and result['kernel_cleanup_complete'] is True,'GPU budget/cleanup mismatch')
    historical=a['historical_source_archive']
    archive_bytes=require_hash(root,historical['path'],historical['sha256'],historical['size_bytes'])
    if not mini: require(manifest['archive_sha256']==historical['sha256'],'historical archive receipt mismatch')
    import io
    with tarfile.open(fileobj=io.BytesIO(archive_bytes),mode='r:gz') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        require(len(names)==len(set(names)),'duplicate historical archive member')
        for m in members:
            p=PurePosixPath(m.name)
            require(not p.is_absolute() and '..' not in p.parts and m.isfile(),'unsafe historical archive member')
        for e in manifest['files']:
            require(e['path'] in names,'missing historical source input')
            content=archive.extractfile(e['path']).read()
            require(len(content)==e['bytes'] and sha(content)==e['sha256'],'historical source input changed')
    draft='docs/mastery/evidence/provenance_inventory_20261019/draft_contracts/'+section.replace('.','_')
    contract=get(draft+'.artifacts.lock.draft.yml');report=get(draft+'.verification_report.draft.json')
    require(contract['notebook_id']==report['notebook_id']==method and contract['canonical_directory']==directory,'wrong method contract')
    require(report['accepted'] is False and report['human_accepted'] is False and report['new_execution'] is False,'draft acceptance boundary changed')
    # Exact document anchors enforce all nested fields, including unknown ones.
    # Independently recompute their meaning below; stored observed_pass is ignored.
    for entry in contract['publication_files_separate_from_execution_inputs']:
        require(set(entry)=={'path','sha256','size_bytes'},'unknown publication entry')
        require_hash(root,entry['path'],entry['sha256'],entry['size_bytes'])
    prefixes=[directory+'/']+(['chapter8_automated_circuits/exercises/part2_attribution_patching_eap/'] if section=='8.4' else [])
    deps=[e for e in manifest['files'] if any(e['path'].startswith(p) for p in prefixes) and e['path'].endswith('.py')]
    require(deps==contract['execution_evidence']['execution_dependencies'],'dependency closure omitted/changed')
    for e in deps:require_hash(root,e['path'],e['sha256'],e['bytes'])
    raw_name=run+('/FAILED.ipynb' if mini else '/executed.ipynb')
    raw=json.loads(read(root,raw_name));recovered=raw
    expected=protocol['expected_lesson_code_cells']+protocol['expected_evidence_probe_cells'] if mini else protocol['sections'][section]['expected_code_cells']
    limits=protocol if mini else protocol['sections'][section]
    verify_sequence(raw,result,expected,limits['cell_timeout_seconds'])
    require(type(result['walltime_seconds']) in (int,float) and 0<=result['walltime_seconds']<=limits['walltime_seconds'],'run exceeded budget')
    if mini:
        recovery=get(run+'/schema_recovery.json')
        verify_trace_arithmetic(get(run+'/scientific_trace.json'))
        require(result['status']=='failed' and result['error']=='<NotebookValidationError: "Additional properties are not allowed (\'id\' was unexpected)">','raw postflight failure changed')
        require_hash(root,raw_name,recovery['raw_failed_notebook_sha256'])
        recovered_name=run+'/recovered_completed.ipynb';recovered=json.loads(require_hash(root,recovered_name,recovery['recovered_notebook_sha256']))
        verify_recovery(raw,recovered)
        for key,name in [
            ('original_result_sha256',run+'/result.json'),
            ('execution_input_sha256',base+'/execution_input.ipynb'),
            ('scientific_trace_sha256',run+'/scientific_trace.json'),
            ('protocol_sha256',base+'/protocol.json')
        ]:
            require_hash(root,name,recovery[key])
        require_hash(root,base+'/execution_input.ipynb',protocol['execution_input_sha256']);require_hash(root,base+'/candidate.ipynb',protocol['candidate_sha256'])
        require(code_cells(json.loads(read(root,base+'/execution_input.ipynb')))[-1]['source']==code_cells(raw)[-1]['source'],'probe boundary changed')
        require(expected==62 and protocol['expected_lesson_code_cells']==61 and protocol['expected_evidence_probe_cells']==1,'wrong lesson/probe boundary')
        prior=protocol['prior_attempts'];total=result['walltime_seconds']
        for p in prior:
            old=json.loads(require_hash(root,base+'/'+p['result_path'],p['result_sha256']));require(old['walltime_seconds']==p['walltime_seconds'],'prior budget mismatch');total+=old['walltime_seconds']
        require(total<=protocol['aggregate_initial_walltime_seconds'] and total==recovery['aggregate_attempt_seconds'],'aggregate budget mismatch')
        cells=code_cells(recovered)[:61]
    else:
        require(result['status']=='success','raw controller failed');require_hash(root,raw_name,result['executed_notebook_sha256']);cells=code_cells(raw)
    current_name=next(e['path'] for e in contract['publication_files_separate_from_execution_inputs'] if e['path'].endswith('_solutions.ipynb'))
    current=json.loads(read(root,current_name))
    fields=('source','outputs','execution_count','metadata')
    require([{k:c.get(k) for k in fields} for c in cells]==[{k:c.get(k) for k in fields} for c in code_cells(current)],'published lesson changed')
    require(current.get('metadata',{}).get('widgets')==recovered.get('metadata',{}).get('widgets'),'widgets changed')
    require(notebook_assertions(raw)==contract['notebook_assertions']==report['notebook_assertions'],'notebook assertions incomplete')
    test_source=read(root,directory+'/tests.py').decode();functions={n.name:n for n in ast.parse(test_source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    declared=iter(report['tests']);seen=0
    for ci,c in enumerate(cells):
        for name in test_calls(c):
            t=next(declared,None);require(t is not None and t['name']==name and t['lesson_code_ordinal']==ci,'AST test call inventory changed')
            fn=functions[name];require(sha(ast.get_source_segment(test_source,fn).encode())==t['function_sha256'],'test function changed')
            assertions=[ast.get_source_segment(test_source,n) for n in ast.walk(fn) if isinstance(n,ast.Assert)]
            require(assertions==t['assertions'],'test assertions omitted')
            require(f'All tests in `{name}` passed!' in output_text(c),'missing same-cell success');seen+=1
    require(next(declared,None) is None,'extra declared test')
    require(observations(cells)==report['observations'],'observations not derived from raw outputs')
    # The records list cached blobs outside the repository. Only these pinned
    # cache identities may resolve symlinks; never use general evidence paths.
    cache=get(base+'/cache_manifest.json');memo={} if cache_digests is None else cache_digests
    revisions={}
    for item in cache:
        require(set(item)=={'repo','revision','path','target','bytes','sha256'},'unknown cache record fields')
        require(re.fullmatch(r'[0-9a-f]{40}',item['revision']) is not None,'invalid revision pin')
        require(item['repo'] not in revisions or revisions[item['repo']]==item['revision'],'multiple revisions for same repo')
        revisions[item['repo']]=item['revision']
        relative=PurePosixPath(item['path']);require(not relative.is_absolute() and '..' not in relative.parts,'unsafe cache path')
        target=Path(item['target']);cache_root=Path('/home/bitwise/.cache/huggingface/hub')
        require(target.is_relative_to(cache_root/item['repo']/'blobs') and not target.is_symlink(),'unexpected external blob target')
        staged=root/base/'hf-hub'/item['repo']/'snapshots'/item['revision']/item['path']
        require(staged.is_file() and staged.resolve()==target.resolve(),'staged cache link/revision changed')
        st=target.stat();key=(str(target),st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        if key not in memo:
            with target.open('rb') as f:memo[key]=hashlib.file_digest(f,'sha256').hexdigest()
        require(st.st_size==item['bytes'] and memo[key]==item['sha256'],'cached bytes changed')
    if mini:require(revisions==protocol['pins'],'model revisions mismatch')
    return {
        'section':section,
        'method':method,
        'technical_saved_execution_identity':'pass',
        'human_accepted':False,
        'course_accepted':False,
        'new_execution':False,
        'ast_test_calls':seen,
        'notebook_assertions':len(notebook_assertions(raw)),
        'declared_cache_assets':len(cache),
        'runtime_coverage':'Recorded kernel versions plus frozen controller protocol versions; not independent reconstruction of unrecorded packages.'
    }

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('sections',nargs='*',default=list(METHODS));args=parser.parse_args();memo={}
    print(json.dumps([verify(s,cache_digests=memo) for s in args.sections],indent=2))
