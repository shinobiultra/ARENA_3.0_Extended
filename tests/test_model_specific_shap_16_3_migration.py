"""Static/DEV engineering checks; no attribution, data, model or tensor execution."""
from __future__ import annotations

import ast
import base64
import gzip
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import symtable
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import nbformat
from nbformat.validator import get_validator
import pytest

C=Path(__file__).resolve().parents[1]
W=C.parent
R=Path('/home/bitwise/Documents/ARENA_3.0_Extended')
CH='chapter16_shapley_attribution_baselines'
P=C/CH/'exercises/part3_model_specific_shap'
E=C/'docs/mastery/evidence/model_specific_shap_16_3_migration_20261021'
STEM='16.3_TreeSHAP_DeepSHAP_and_GradientSHAP'
D=[('tree','part13_treeshap','16.13_TreeSHAP_from_First_Principles',87),('deep','part14_deepshap','16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP',114),('gradient','part15_gradientshap','16.15_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate',126)]


def raw(path):
    if path.exists(): return path.read_bytes()
    if path==E/'cell_inverse_map.json':return gzip.decompress((E/'cell_inverse_map.json.gz').read_bytes())
    if path.is_relative_to(W/'preimages'):
        with tarfile.open(E/'before.tar.gz') as archive:
            member=str(path.relative_to(W/'preimages'))
            if member in archive.getnames():return archive.extractfile(member).read()
        contract=json.loads((E/'before.tar.gz.review-bundle.json').read_text())
        record=contract['external_payloads'][member];payload=Path(record['absolute_path']).read_bytes()
        assert len(payload)==record['size_bytes'] and hashlib.sha256(payload).hexdigest()==record['sha256']
        return payload
    if path.is_relative_to(W/'raw_generated'):return (E/'raw_generated'/path.name).read_bytes()
    raise FileNotFoundError(path)


def read(path):
    def unique(pairs):
        out={}
        for k,v in pairs:
            assert k not in out
            out[k]=v
        return out
    return json.loads(raw(path),object_pairs_hook=unique,parse_constant=lambda x:pytest.fail(x))


def source(c):return ''.join(c['source'])


def load_inert():
    source_text=(P/'solutions.py').read_text()
    code=compile(source_text,str(P/'solutions.py'),'exec')
    namespace={'__name__':'model_specific_DEV_inert','__file__':str(P/'solutions.py')}
    original_import=__import__
    def blocked(*args,**kwargs):raise AssertionError('inert import attempted runtime work')
    def imports(name,*args,**kwargs):
        if name=='__future__':return original_import(name,*args,**kwargs)
        raise AssertionError('inert import attempted external import: '+name)
    before=list(sys.path)
    with patch('builtins.__import__',imports),patch('builtins.open',blocked),patch.object(Path,'cwd',blocked),patch.object(Path,'home',blocked),patch.object(socket,'socket',blocked):exec(code,namespace)
    assert sys.path==before
    return namespace


def test_strict_saved_schema_without_normalization():
    for path in [P/f'{STEM}_{kind}.ipynb' for kind in ['exercises','solutions']]+[C/'infrastructure/chapters'/CH/'master_16_3.ipynb']:
        book=read(path)
        assert (book['nbformat'],book['nbformat_minor'])==(4,5)
        assert not list(get_validator(4,5).iter_errors(book))
        identifiers=[c['id'] for c in book['cells']]
        assert len(identifiers)==len(set(identifiers))
        for cell in book['cells']:
            if cell['cell_type']=='markdown':assert 'outputs' not in cell and 'execution_count' not in cell


def test_all_654_original_cell_objects_and_six_books_have_exact_inverses():
    mapping=read(E/'cell_inverse_map.json');metadata=read(E/'legacy_book_metadata.json')
    proof=read(E/'cell_map_encoding_proof.json');decoded=gzip.decompress((E/'cell_inverse_map.json.gz').read_bytes())
    assert proof['exact_byte_equality'] is True
    assert len(decoded)==proof['decoded_size_bytes']
    assert hashlib.sha256(decoded).hexdigest()==proof['decoded_sha256']==proof['plaintext']['sha256']
    plain=Path(proof['plaintext']['absolute_path'])
    if plain.exists():assert plain.read_bytes()==decoded
    for kind in ['exercises','solutions']:
        book=read(P/f'{STEM}_{kind}.ipynb')
        assert len(mapping[kind])==327
        for alias,part,stem,count in D:
            rows=[r for r in mapping[kind] if r['legacy']==alias]
            assert len(rows)==count
            reconstructed=[]
            for i,row in enumerate(rows):
                assert row['original_index']==i
                cell=book['cells'][row['candidate_index']]
                assert cell==row['transformed_cell']
                old=row['original_cell'];assert cell['metadata']==old['metadata']
                if 'id' in old:assert cell['id']==old['id']
                if old['cell_type']=='code':
                    assert cell['outputs']==old['outputs']
                    assert cell['execution_count']==old['execution_count']
                assert set(cell)-{'id','source','outputs','execution_count'}==set(old)-{'id','source','outputs','execution_count'}
                reconstructed.append(old)
            original=read(W/'preimages'/CH/'exercises'/part/f'{stem}_{kind}.ipynb')
            assert {**metadata[alias][kind],'cells':reconstructed}==original


def test_original_sources_pages_assets_and_archived_members_unchanged():
    manifest=read(E/'source_manifest.json')
    assert len(manifest)==306
    contract=read(E/'before.tar.gz.review-bundle.json')
    external=contract['external_payloads']
    assert len(external)==49
    with tarfile.open(E/'before.tar.gz') as archive:
        members=set(archive.getnames())
        assert len(members)==257
        assert members.isdisjoint(external)
        assert members|set(external)=={row['path'] for row in manifest}
        for row in manifest:
            if row['path'] in members:archived=archive.extractfile(row['path']).read()
            else:
                record=external[row['path']]
                assert record['absolute_path']==str(R/row['path'])
                assert record['size_bytes']==row['size'] and record['sha256']==row['sha256']
                archived=Path(record['absolute_path']).read_bytes()
            assert hashlib.sha256(archived).hexdigest()==row['sha256']
            assert len(archived)==row['size']
            assert (R/row['path']).read_bytes()==archived


def test_all_original_stubs_solutions_dropdowns_tests_play_and_outputs_survive():
    maps=read(E/'cell_inverse_map.json')
    for kind,outputs in [('exercises',46),('solutions',73)]:
        book=read(P/f'{STEM}_{kind}.ipynb')
        assert sum(bool(c.get('outputs')) for c in book['cells'])==outputs
        assert sum('image/png' in o.get('data',{}) for c in book['cells'] for o in c.get('outputs',[]))==54
        assert sum(c['cell_type']=='code' for c in book['cells'])==422
        for row in maps[kind]:
            old=source(row['original_cell']);new=source(row['transformed_cell'])
            if row['original_cell']['cell_type']=='markdown':
                for needle in ['<details>','</details>','<summary>','</summary>','NotImplementedError']:
                    assert new.count(needle)==old.count(needle)
            else:assert new.count('raise NotImplementedError')==old.count('raise NotImplementedError')
    for alias,_,_,_ in D:
        originals={k:[r for r in maps[k] if r['legacy']==alias] for k in maps}
        for oldex,oldsol in zip(originals['exercises'],originals['solutions'],strict=True):
            if oldex['original_cell']['source']!=oldsol['original_cell']['source']:
                assert oldex['transformed_cell']['source']!=oldsol['transformed_cell']['source']


def test_canonical_python_equals_normal_no_execute_generation_bytes():
    assert (P/'solutions.py').read_bytes()==raw(W/'raw_generated'/'solutions.py')
    for kind in ['exercises','solutions']:
        final=read(P/f'{STEM}_{kind}.ipynb');generated=read(W/'raw_generated'/f'{STEM}_{kind}.ipynb')
        left=[ast.dump(ast.parse(source(c))) for c in final['cells'] if c['cell_type']=='code']
        right=[ast.dump(ast.parse(source(c))) for c in generated['cells'] if c['cell_type']=='code']
        assert left==right


class Undo(ast.NodeTransformer):
    def __init__(self,mapping,namespace=None,original=None):
        self.mapping={v:k for k,v in mapping.items()};self.namespace=namespace;self.imports=[]
        if original:self.imports=[n for n in ast.walk(original) if isinstance(n,ast.ImportFrom)]
    def visit_Name(self,node):node.id=self.mapping.get(node.id,node.id);return node
    def visit_FunctionDef(self,node):node.name=self.mapping.get(node.name,node.name);return self.generic_visit(node)
    visit_AsyncFunctionDef=visit_FunctionDef
    def visit_Global(self,node):node.names=[self.mapping.get(n,n) for n in node.names];return node
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id==self.namespace:node.func.id='globals'
        return self.generic_visit(node)
    def visit_Assign(self,node):
        # Explicit namespace construction replaces only a corresponding original module import.
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='SimpleNamespace' and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            target=self.mapping.get(node.targets[0].id,node.targets[0].id)
            matches=[old for old in self.imports if any((a.asname or a.name)==target for a in old.names)]
            if matches:return deepcopy(matches[0])
        return self.generic_visit(node)


def test_every_original_top_level_function_and_stub_scientific_ast_exact_inverse():
    maps=read(E/'alpha_maps.json');cells=read(E/'cell_inverse_map.json');checked=0
    for kind in cells:
        for row in cells[kind]:
            if row['original_cell']['cell_type']!='code':continue
            oldtree=ast.parse(source(row['original_cell']));newtree=ast.parse(source(row['transformed_cell']))
            for original in oldtree.body:
                if not isinstance(original,ast.FunctionDef):continue
                name=maps[row['legacy']].get(original.name,original.name)
                matching=[n for n in ast.walk(newtree) if isinstance(n,ast.FunctionDef) and n.name==name]
                assert len(matching)==1,(row['legacy'],row['original_index'],name)
                recovered=Undo(maps[row['legacy']],row['legacy']+'_curriculum_namespace',original).visit(deepcopy(matching[0]))
                assert ast.dump(recovered)==ast.dump(original),(kind,row['legacy'],row['original_index'],original.name)
                checked+=1
    assert checked>=300


def test_all_228_expanded_helper_sources_have_exact_scientific_ast_inverses():
    records=read(E/'helper_source_map.json');assert len(records)==228
    module=ast.parse((P/'solutions.py').read_text())
    for row in records:
        original=ast.parse(row['original_source']).body[0]
        if not isinstance(original,ast.FunctionDef):continue
        matches=[n for n in ast.walk(module) if isinstance(n,ast.FunctionDef) and n.name==row['target']]
        assert len(matches)==1,row['target']
        recovered=Undo(row['alpha_map'],original=original).visit(deepcopy(matches[0]))
        assert ast.dump(recovered)==ast.dump(original),row['key']


def test_actual_import_traps_every_external_import_runtime_path_and_network():
    namespace=load_inert()
    assert callable(namespace[read(E/'alpha_maps.json')['tree']['tree_shap']])
    assert callable(namespace[read(E/'alpha_maps.json')['deep']['rescale_slope']])
    assert callable(namespace[read(E/'alpha_maps.json')['gradient']['expected_gradients']])
    assert read(E/'alpha_maps.json')['deep']['rescale_multipliers'] not in namespace
    assert 'gradient_helper_gradientshap_methods__eg_preserve_execution_state' not in namespace
    for name in ['t','torch','np','canonical_dir','tree_root_dir','tree__DEFAULT_METHODS']:assert name not in namespace


def test_decorator_initializers_bind_original_definitions_with_dev_identity_only():
    ns=load_inert();events=[]
    def identity(fn):events.append(('identity',fn.__name__));return fn
    def no_grad():events.append(('factory',));return identity
    ns['t']=SimpleNamespace(no_grad=no_grad)
    ns['contextlib']=SimpleNamespace(contextmanager=identity)
    ns['gradient_helper_gradientshap_reference__state_restoring_adapter']=identity
    for record in read(E/'decorator_initializers.json'):
        original=ast.parse(record['source']).body[0]
        ns[record['initializer']]()
        fn=ns[record['function']]
        assert callable(fn)
        assert fn.__qualname__==fn.__name__==record['function']
        assert fn.__closure__ is None
        assert tuple(fn.__defaults__ or ())==tuple(ast.literal_eval(x) for x in original.args.defaults)
    assert sum(event[0]=='factory' for event in events)==3
    assert sum(event[0]=='identity' for event in events)==8


def test_dynamic_curriculum_namespaces_preserve_every_current_key_and_rebinding():
    ns=load_inert();maps=read(E/'alpha_maps.json')
    for alias,names in maps.items():
        for name,target in names.items():ns[target]=object()
        adapter=ns[alias+'_curriculum_namespace'];first=adapter()
        for name,target in names.items():assert first[name] is ns[target]
        for name,target in names.items():
            replacement=object();ns[target]=replacement
            assert adapter()[name] is replacement
        name,target=next(iter(names.items()));ns.pop(target)
        assert name not in adapter()
        assert name in first


def test_tree_omitted_fallback_is_explicit_current_visible_binding_only_dev_marker():
    ns=load_inert()
    def first_marker():raise AssertionError('DEV callback must never execute')
    def second_marker():raise AssertionError('DEV callback must never execute')
    ns[read(E/'alpha_maps.json')['tree']['tree_shap']]=first_marker
    ns['refresh_tree_default_methods']()
    assert ns['tree__DEFAULT_METHODS'].tree_shap is first_marker
    # This resolver only returns a DEV marker; it does not invoke scientific code.
    assert ns['tree__method'](None,'tree_shap') is first_marker
    ns[read(E/'alpha_maps.json')['tree']['tree_shap']]=second_marker
    ns['refresh_tree_default_methods']()
    assert ns['tree__DEFAULT_METHODS'].tree_shap is second_marker
    assert ns['tree__method'](None,'tree_shap') is second_marker
    with pytest.raises(AttributeError):ns['tree__method'](None,'DEV_missing_callback')


def test_local_plumbing_inverses_keep_original_scientific_test_fallbacks_and_data_inputs():
    records=read(E/'local_plumbing_map.json');assert len(records)==9
    for row in records:
        assert (C/row['target_path']).read_text()==row['transformed_source']
        assert raw(W/'preimages'/row['source_path']).decode()==row['original_source']
        assert hashlib.sha256(row['original_source'].encode()).hexdigest()==row['original_sha256']
        if row['source_path'].endswith('_plots.py'):assert row['original_source']==row['transformed_source']
        if row['source_path'].endswith('_data.py'):assert 'parents[1]' in row['transformed_source'] and '/exercises/part3_' not in row['transformed_source']
        if row['source_path'].endswith('_methods_tests.py'):
            assert 'as methods' in row['transformed_source']
            assert '.exercises.part3_model_specific_shap' not in row['transformed_source']


def test_all_162_media_records_byte_exact_and_every_sidebar_panel_uses_standard_renderer():
    sys.path.insert(0,str(R))
    try:from infrastructure.extension.render_media import inline_page_media
    finally:sys.path.pop(0)
    page=C/CH/'instructions/pages/03_[16.3]_TreeSHAP_DeepSHAP_and_GradientSHAP_Practice.md'
    rendered=inline_page_media(page.read_text(),page)
    assert '../../exercises/part3_model_specific_shap/media/' not in rendered
    media=read(E/'media_map.json');assert len(media)==162
    assert rendered.count('src="data:image/')>=162
    for row in media:
        data=(C/row['target']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['sha256']
        assert base64.b64encode(data).decode() in rendered
    shell=page.with_suffix('.py').read_text()
    assert 'inline_page_media(markdown_content_file.read_text' in shell
    assert all(name in shell for name in ['TreeSHAP','DeepSHAP','Expected Gradients','Historical figure shelf'])


def test_method_specific_blank_forecasts_precede_every_original_result_bearing_cell():
    for kind in ['exercises','solutions']:
        book=read(P/f'{STEM}_{kind}.ipynb');mapping=read(E/'cell_inverse_map.json')[kind]
        text='\n'.join(source(c) for c in book['cells'])
        assert text.index('Your forecast')<text.index('Historical [16.13]')
        for row in mapping:
            old=row['original_cell'];s=source(old)
            if old['cell_type']=='code' and (old.get('outputs') or 'show_figure(' in s):
                previous=source(book['cells'][row['candidate_index']-1])
                assert 'Your commitment' in previous or previous.startswith('### Visible supporting scientific function'),(row['legacy'],row['original_index'])
            if old['cell_type']=='markdown' and 'Try It Yourself' in s:assert 'Try It Yourself' in source(row['transformed_cell'])
        for needle in ['seen / unknown / unexposed','No answer is prefilled','Monte Carlo standard error','intact-donor','population-conditional']:
            assert needle in text


def test_helper_constants_keep_literal_values_and_all_assignment_names_scoped():
    module=ast.parse((P/'solutions.py').read_text())
    assignments={name:n for n in ast.walk(module) if isinstance(n,(ast.Assign,ast.AnnAssign)) for target in (n.targets if isinstance(n,ast.Assign) else [n.target]) if isinstance(target,ast.Name) for name in [target.id]}
    for row in read(E/'helper_source_map.json'):
        original=ast.parse(row['original_source']).body[0]
        if not isinstance(original,(ast.Assign,ast.AnnAssign)):continue
        candidate=assignments[row['target']]
        if row['key'][2]=='HERE':
            assert '/ "exercises" /' in row['transformed_source']
            continue  # Declared input-path compatibility keeps the original legacy source directory.
        recovered=Undo(row['alpha_map']).visit(deepcopy(candidate))
        assert ast.dump(recovered)==ast.dump(original),row['key']
        for target in (candidate.targets if isinstance(candidate,ast.Assign) else [candidate.target]):
            if isinstance(target,ast.Name):assert target.id.startswith(row['key'][0]+'_helper_')


def test_no_unresolved_scientific_function_globals_after_explicit_binding():
    text=(P/'solutions.py').read_text();tree=ast.parse(text)
    table=symtable.symtable(text,'canonical','exec')
    bound={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
    bound.update(name for n in ast.walk(tree) if isinstance(n,ast.Global) for name in n.names)
    import builtins
    allowed=set(vars(builtins))|{'__name__','__file__'}
    missing=[]
    def visit(scope):
        for symbol in scope.get_symbols():
            if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in bound|allowed:missing.append((scope.get_name(),symbol.get_name()))
        for child in scope.get_children():visit(child)
    visit(table)
    assert missing==[]


def test_complete_namespace_maps_cover_required_keys_at_original_cell_boundaries():
    maps=read(E/'alpha_maps.json');checked=0
    for alias,part,stem,_ in D:
        original=read(W/'preimages'/CH/'exercises'/part/f'{stem}_solutions.ipynb')
        available=set();functions={}
        def requirements(name,seen=None):
            seen=set() if seen is None else seen
            if name in seen or name not in functions:return set()
            seen.add(name);required=set()
            for n in ast.walk(functions[name]):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
                    if n.func.id=='_method' and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and isinstance(n.args[1].value,str):required.add(n.args[1].value)
                    required.update(requirements(n.func.id,seen))
            return required
        for index,cell in enumerate(original['cells']):
            if cell['cell_type']!='code':continue
            nodes=ast.parse(source(cell)).body
            for n in nodes:
                if isinstance(n,ast.FunctionDef):functions[n.name]=n;available.add(n.name)
                elif isinstance(n,(ast.Import,ast.ImportFrom)):available.update(a.asname or a.name.split('.')[0] for a in n.names)
                elif isinstance(n,ast.Assign):available.update(z.id for target in n.targets for z in ast.walk(target) if isinstance(z,ast.Name))
            for n in nodes:
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)):continue
                for call in ast.walk(n):
                    if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='globals' for x in ast.walk(call)):
                        required=requirements(call.func.id)
                        assert required<=available,(alias,index,call.func.id,required-available)
                        assert required<=set(maps[alias]),(alias,index,required-set(maps[alias]))
                        checked+=1
    assert checked>=10


def test_proposed_guided_case_passes_existing_validator_and_all_27_callbacks_exist():
    import importlib.util
    spec=importlib.util.spec_from_file_location('model_specific_existing_case_validator',R/'mastery_hub/units/extension_predictions/lesson.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=read(E/'proposed_case.json')
    assert module.validate_case(case)==case
    assert len(case['method_functions'])==len(set(case['method_functions']))==27
    names={n.name for n in ast.walk(ast.parse((P/'solutions.py').read_text())) if isinstance(n,ast.FunctionDef)}
    assert set(case['method_functions'])<=names
    for key in ['master_path','exercise_path','solutions_path']:assert (C/case[key]).is_file()
    assert case['data_role']=='historical' and case['status']=='guided_practice'
    assert case['transfer_prediction_type']=='binary'
    assert not any(key in case for key in ['answer','prediction','observation','confidence','learner_response'])


def test_no_scientific_runtime_import_shortcuts_remain_in_generated_module():
    module=ast.parse((P/'solutions.py').read_text())
    for n in ast.walk(module):
        if not isinstance(n,ast.ImportFrom):continue
        assert n.level==0
        target=n.module or ''
        assert not any(target.endswith('_'+stem) for stem in ['methods','study','diagnostics','reference']),target
        assert not target.endswith('.run_native'),target
    for n in module.body:
        if isinstance(n,(ast.FunctionDef,ast.ImportFrom)) or isinstance(n,ast.If):continue
        if isinstance(n,ast.Assign):
            if any(isinstance(t,ast.Name) and t.id=='MAIN' for t in n.targets):continue
            ast.literal_eval(n.value)


def test_explicit_namespace_maps_include_all_original_assigned_bindings_including_loops():
    maps=read(E/'alpha_maps.json')
    values=[]
    for alias,part,stem,_ in D:
        book=read(W/'preimages'/CH/'exercises'/part/f'{stem}_solutions.ipynb')
        text='\n\n'.join(source(c) for c in book['cells'] if c['cell_type']=='code')
        table=symtable.symtable(text,'original-book','exec')
        assigned={symbol.get_name() for symbol in table.get_symbols() if symbol.is_assigned()}
        assert assigned<=maps[alias].keys(),(alias,assigned-maps[alias].keys())
        values.extend(maps[alias].values())
    assert len(values)==len(set(values))
