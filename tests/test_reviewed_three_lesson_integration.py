"""Exact three-lesson custody and fail-closed regeneration; no taught imports."""
from pathlib import Path
import ast,copy,hashlib,json,sys,tarfile
import pytest
from nbformat.validator import get_validator

ROOT=Path(__file__).resolve().parents[1]
EV=ROOT/'docs/mastery/evidence/lesson_integration_20261021_first3'
sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/core')]
from infrastructure.extension.build import ExtensionMasterFile,normalize_generated_notebook
from infrastructure.extension.source_preservation import restore_notebooks
from scripts.evidence_bundles import validate_review_bundle

SECTIONS=['6.1','9.6','12.5']
load=lambda p:json.loads(p.read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
src=lambda c:''.join(c['source'])

def payloads(section):
    recipe=load(ROOT/'infrastructure/extension/source_preservation_contracts.json')['sections'][section]
    payload={}
    for kind,binding in recipe['archives'].items():
        validate_review_bundle(ROOT/binding['path'])
        with tarfile.open(ROOT/binding['path'])as archive:payload[kind]={name:archive.extractfile(name).read()for name in archive.getnames()}
    return recipe,payload

def arguments(section):
    recipe,payload=payloads(section)
    master=ROOT/'infrastructure/chapters'/recipe['identity']['chapter']/('master_'+section.replace('.','_')+'.py')
    return {'master_path':master,'identity':recipe['identity'],'root':ROOT,
        'raw_notebooks':{kind:payload['raw'][pin['member']].decode()for kind,pin in recipe['books'].items()},
        'raw_modules':{name:payload['raw'][pin['member']].decode()for name,pin in recipe['modules'].items()}},recipe,payload

def test_existing50_cases_and8_recipes_exact_order_and_schema():
    before=load(EV/'input_state.json');registry=load(ROOT/'mastery_hub/extension_checkpoints.json')
    assert registry['checkpoints'][:50]==before['original50cases']['checkpoints']
    assert [c['section']for c in registry['checkpoints'][50:53]]==SECTIONS
    assert json.dumps(registry['checkpoints'][:50],sort_keys=True)==json.dumps(before['original50cases']['checkpoints'],sort_keys=True)
    contracts=load(ROOT/'infrastructure/extension/source_preservation_contracts.json')
    # This receipt owns the initial cohort; later appends have separate custody tests.
    original_recipes=before['original8recipes']['sections']
    protected_sections=set(original_recipes)|set(SECTIONS)
    assert protected_sections<=contracts['sections'].keys()
    assert {k:v for k,v in contracts['sections'].items()if k in original_recipes}==original_recipes
    assert json.dumps({k:contracts['sections'][k]for k in original_recipes},sort_keys=True)==json.dumps(original_recipes,sort_keys=True)
    assert len(protected_sections)==11

def test_additive_authoritative_headers_remove_to_every_exact_prior_cell_object():
    from scripts.rome_runtime_metadata_checkpoints import recover_intro_checkpoint_bytes
    rows=load(EV/'v2/additive_part_header_inverse.json')
    assert len(rows)==4
    archive=EV/'v2/before_additive_headers.tar.gz';validate_review_bundle(archive)
    with tarfile.open(archive)as handle:
        for row in rows:
            old=json.loads(handle.extractfile(row['path']).read());new=load(ROOT/row['path'])
            if row['path'] in ('chapter9_alignment_interpretability/exercises/part6_rome_memit/9.6_Model_Editing_with_ROME_and_MEMIT_exercises.ipynb', 'chapter9_alignment_interpretability/exercises/part6_rome_memit/9.6_Model_Editing_with_ROME_and_MEMIT_solutions.ipynb'):
                new=json.loads(recover_intro_checkpoint_bytes(ROOT,row['path'],(ROOT/row['path']).read_bytes()))
            ids={op['id']for op in row['operations']}
            assert len(ids)==len(row['operations'])
            assert len({c['id']for c in new['cells']})==len(new['cells'])
            for op in row['operations']:
                header=new['cells'][op['new_header_index']]
                assert header=={'cell_type':'markdown','metadata':{},'id':op['id'],'source':op['source']}
                assert new['cells'][op['anchor_after_index']]['id']==op['anchor_id']
            inverse=copy.deepcopy(new);inverse['cells']=[c for c in inverse['cells']if c['id']not in ids]
            assert inverse==old

def test_first_master_adoption_is_explicit_id_only_and_strict():
    from scripts.rome_runtime_metadata_checkpoints import recover_intro_checkpoint_bytes
    rows=load(EV/'v2/first_master_adoption_inverse.json');assert len(rows)==3
    for row in rows:
        before=(ROOT/row['before_exact_path']).read_bytes();after=(ROOT/row['path']).read_bytes()
        if row['path']=='infrastructure/chapters/chapter9_alignment_interpretability/master_9_6.ipynb':
            after=recover_intro_checkpoint_bytes(ROOT,row['path'],after)
        assert sha(before)==row['before_sha256']and sha(after)==row['after_sha256']
        old=json.loads(before);new=json.loads(after)
        assert not list(get_validator(4,5).iter_errors(new))
        inverse=copy.deepcopy(new)
        for op in row['id_operations']:
            assert inverse['cells'][op['cell']]['id']==op['after']
            inverse['cells'][op['cell']]['id']=op['before']
        assert inverse==old

def test_case_locator_clarification_changes_only_locator_fields_and_resolves_exact_sources():
    changes=load(EV/'v2/case_locator_clarification_inverse.json')
    assert [row['section']for row in changes]==SECTIONS
    archive=EV/'v2/original_sources.tar.gz';validate_review_bundle(archive)
    manifest=load(EV/'v2/original_sources_manifest.json')
    with tarfile.open(archive)as handle:
        assert len(handle.getmembers())==len(manifest['files'])==46
        for name,pin in manifest['files'].items():
            data=handle.extractfile(name).read()
            assert sha(data)==pin['sha256']and len(data)==pin['size_bytes']
    current=load(ROOT/'mastery_hub/extension_checkpoints.json')['checkpoints'][50:53]
    for change,case in zip(changes,current,strict=True):
        assert case==change['after']
        assert json.dumps(case,sort_keys=True)==json.dumps(change['after'],sort_keys=True)
        before=change['before']
        assert {k:v for k,v in before.items()if k not in {'config_fields','evidence_detail'}}=={
            k:v for k,v in case.items()if k not in {'config_fields','evidence_detail'}}
        assert {k:v for k,v in before['config_fields'].items()if k!='source_identity'}=={
            k:v for k,v in case['config_fields'].items()if k!='source_identity'}
        assert str(archive.relative_to(ROOT))in case['config_fields']['source_identity']
        assert sha(archive.read_bytes())in case['evidence_detail']
        assert 'zero-based'in case['evidence_detail']and 'cell ID'in case['evidence_detail']

@pytest.mark.parametrize('section',SECTIONS)
def test_reviewed_recipe_preserves_every_complete_object_except_navigation(section):
    args,recipe,payload=arguments(section);result=restore_notebooks(**args)
    for kind,text in result.items():
        pin=recipe['books'][kind];old=json.loads(payload['before'][pin['member']]);new=json.loads(text)
        assert not list(get_validator(4,5).iter_errors(new))
        oldnav=[i for i,c in enumerate(old['cells'])if c['cell_type']=='markdown'and '> **Colab:'in src(c)]
        assert len(oldnav)==1;i=oldnav[0]
        assert 'start the guided workbook'in src(new['cells'][i])
        assert '#section-'+section.replace('.','-')in src(new['cells'][i])
        restored=copy.deepcopy(new);restored['cells'][i]['source']=old['cells'][i]['source'];assert restored==old
        assert new==load(ROOT/pin['member'])
    for name,pin in recipe['modules'].items():
        actual=(ROOT/pin['member']).read_bytes()
        assert actual==payload['raw'][pin['member']]==payload['before'][pin['member']]
        assert sha(actual)==pin['sha256']
        assert ast.dump(ast.parse(actual))==ast.dump(ast.parse(payload['raw'][pin['member']]))

@pytest.mark.parametrize('section',SECTIONS)
def test_actual_registered_noexecute_regeneration_matches_published_bytes(tmp_path,section):
    args,recipe,payload=arguments(section);identity=recipe['identity'];chapter=tmp_path/identity['chapter']
    for d in [chapter/'exercises'/identity['exercise_dir'],chapter/'instructions/pages']:d.mkdir(parents=True)
    master=args['master_path'].with_suffix('.ipynb')
    saved=master.read_bytes()
    m=ExtensionMasterFile(master_path=master,chapter_dir=chapter,chapter_name_long='Source-only test',
        exercise_dir_name=identity['exercise_dir'],streamlit_page_name=identity['page'])
    m.generate_files(overwrite=True)
    # Match the actual generate() pipeline's existing final JSON serializer;
    # this only sorts/serializes the already preserved valid object.
    for pin in recipe['books'].values():normalize_generated_notebook(tmp_path/pin['member'])
    assert m.source_preservation_applied
    assert master.read_bytes()==saved
    for pin in recipe['books'].values():assert (tmp_path/pin['member']).read_bytes()==(ROOT/pin['member']).read_bytes()
    for pin in recipe['modules'].values():assert (tmp_path/pin['member']).read_bytes()==(ROOT/pin['member']).read_bytes()

@pytest.mark.parametrize('section',SECTIONS)
@pytest.mark.parametrize('mutation',['source','output','count','top_metadata','cell_metadata'])
def test_actual_generation_refuses_mutated_published_book_before_any_write(tmp_path,section,mutation):
    args,recipe,payload=arguments(section);identity=recipe['identity'];chapter=tmp_path/identity['chapter']
    for d in [chapter/'exercises'/identity['exercise_dir'],chapter/'instructions/pages']:d.mkdir(parents=True)
    for pin in recipe['books'].values():
        p=tmp_path/pin['member'];p.write_bytes((ROOT/pin['member']).read_bytes())
    path=tmp_path/recipe['books']['solutions']['member'];book=load(path)
    code=next(c for c in book['cells']if c['cell_type']=='code')
    if mutation=='source':code['source']=['unapproved_source = 0\n']
    elif mutation=='output':code['outputs']=[{'output_type':'stream','name':'stdout','text':['unreviewed result\n']}]
    elif mutation=='count':code['execution_count']=999
    elif mutation=='top_metadata':book['metadata']['unapproved_learner_work']=True
    else:code['metadata']['unapproved_learner_work']=True
    path.write_text(json.dumps(book))
    sentinel=chapter/'instructions/pages/unrelated.md';sentinel.write_text('keep this existing source')
    before={str(p.relative_to(tmp_path)):p.read_bytes()for p in tmp_path.rglob('*')if p.is_file()}
    m=ExtensionMasterFile(master_path=args['master_path'].with_suffix('.ipynb'),chapter_dir=chapter,
        chapter_name_long='Source-only refusal fixture',exercise_dir_name=identity['exercise_dir'],streamlit_page_name=identity['page'])
    with pytest.raises(ValueError,match='published notebook has unreviewed changes'):m.generate_files(overwrite=True)
    assert before=={str(p.relative_to(tmp_path)):p.read_bytes()for p in tmp_path.rglob('*')if p.is_file()}

@pytest.mark.parametrize('section',SECTIONS)
@pytest.mark.parametrize('mutation',['raw_whitespace','module_comment','identity'])
def test_exact_new_source_bindings_refuse_semantically_inert_drift(section,mutation):
    args,_,_=arguments(section)
    if mutation=='raw_whitespace':
        book=json.loads(args['raw_notebooks']['exercises']);book['cells'][0]['source'][0]+=' '
        args['raw_notebooks']['exercises']=json.dumps(book)
    elif mutation=='module_comment':args['raw_modules']['solutions']+='\n# changed source bytes\n'
    else:args['identity']={**args['identity'],'exercise_dir':'unreviewed_alias'}
    with pytest.raises(ValueError):restore_notebooks(**args)
