"""Human-authorized source views, blank practice and intact originals; static/DEV only."""
from pathlib import Path
import ast,copy,hashlib,json,tarfile
from nbformat.validator import get_validator
import pytest
from scripts.practice06_execution_checkpoint import load_practice06_checkpoint_json
C=Path(__file__).resolve().parents[1];W=C.parent;R=Path('/home/bitwise/Documents/ARENA_3.0_Extended');E=C/'docs/mastery/evidence/lesson_integration_20261021_q008';load=lambda p:load_practice06_checkpoint_json(C,p);src=lambda c:''.join(c['source']);sha=lambda b:hashlib.sha256(b).hexdigest();M=load(E/'practice_mapping.json');PROJECTIONS=load(E/'practice_book_projection_inverse.json');NAV=load(E/'navigation_inverse.json')
@pytest.mark.parametrize('section',['0.6','1.6','16.3'])
def test_active_practice_code_is_blank_and_directly_named(section):
 i=M[section];p=C/i['directory']/(i['stem']+'_exercises.ipynb');b=load(p);assert '_Practice_exercises.ipynb'in p.name;assert not list(get_validator(4,5).iter_errors(b));assert all(c.get('execution_count')is None and c.get('outputs')==[]for c in b['cells']if c['cell_type']=='code')
@pytest.mark.parametrize('section',['0.6','1.6','16.3'])
@pytest.mark.parametrize('kind',['exercises','solutions'])
def test_full_typed_projection_and_navigation_inverse_recovers_intact_reference(section,kind):
 row=next(x for x in PROJECTIONS if x['section']==section and x['kind']==kind);book=load(C/row['practice_path']);reference=load(C/row['reference_path']);nav=next(x for x in NAV if x['section']==section and x['kind']==kind);book['cells'][nav['cell']]['source']=nav['before_source'];assert len(book['cells'])==len(reference['cells']);assert book['metadata']==reference['metadata']
 for op in row['operations']:
  assert sha(json.dumps(book['cells'][op['cell']],sort_keys=True,ensure_ascii=False).encode())==op['after_sha256'];book['cells'][op['cell']]=op['before_whole_object']
 assert book==reference
@pytest.mark.parametrize('section',['0.6','1.6'])
def test_only_duplicated_foundation_appendix_is_readonly_full_exact_source(section):
 row=next(x for x in PROJECTIONS if x['section']==section and x['kind']=='exercises');book=load(C/row['practice_path']);converted=[op for op in row['operations']if op['kind_projection']];assert len(converted)==8
 for op in converted:
  before=op['before_whole_object'];after=book['cells'][op['cell']];assert before['cell_type']=='code'and after['cell_type']=='markdown';assert after['metadata']==before['metadata']and after['id']==before['id'];assert 'outputs'not in after and'execution_count'not in after;fence=src(after).split('```python\n',1)[1];original=src(before);assert fence[:len(original)]==original;assert ast.dump(ast.parse(original))==ast.dump(ast.parse(fence[:len(original)]));assert 'read-only original source'in src(after)
@pytest.mark.parametrize('section',['0.6','1.6','16.3'])
def test_solutions_code_wholeobjects_counts_outputs_and_sources_are_intact(section):
 row=next(x for x in PROJECTIONS if x['section']==section and x['kind']=='solutions');before=load(C/row['reference_path']);after=load(C/row['practice_path']);assert[(n,c)for n,c in enumerate(before['cells'])if c['cell_type']=='code']==[(n,c)for n,c in enumerate(after['cells'])if c['cell_type']=='code']
def test_all_originals_and_helper_modules_keep_original_paths_and_bytes():
 for row in load(E/'original_source_preservation_baseline.json'):
  assert sha((R/row['path']).read_bytes())==row['sha256']
  if(C/row['path']).is_file():assert(C/row['path']).read_bytes()==(R/row['path']).read_bytes()
 proof=load(E/'original_replay_path_proof.json');assert proof['original_source_books']==10 and proof['DEV_distinct_original_vs_practice_binding_origins'];assert proof['old_original_modules_not_replaced_by_inert_API']and proof['scientific_or_setup_calls']==0
@pytest.mark.parametrize('section',['0.6','1.6','16.3'])
def test_direct_original_links_resolve_from_both_active_book_and_actual_page(section):
 i=M[section];book=C/i['directory']/(i['stem']+'_exercises.ipynb');page=C/i['chapter']/'instructions/pages'/(i['page']+'.md');group=next(x for x in load(E/'practice_direct_original_links.json')if x['section']==section)
 for origin in group['originals']:
  for parent in [book.parent,page.parent]:
   p=(parent/origin['href']).resolve();assert p==C/origin['path'];assert sha(p.read_bytes())==origin['sha256']
  assert ']('+origin['href']+')'in page.read_text();intro=[c for c in load(book)['cells']if c['cell_type']=='markdown'and'### Practice copies and intact originals'in src(c)];assert len(intro)==1;assert origin['path']in src(intro[0])
@pytest.mark.parametrize('section',['0.6','1.6','16.3'])
def test_actual_normal_raw_modules_and_all_scientific_source_are_byte_exact(section):
 i=M[section];rel=Path(i['directory'])/'solutions.py';rec=load(C/'infrastructure/extension/source_preservation_contracts.json')['sections'][section]
 with tarfile.open(C/rec['archives']['raw']['path'])as archive:raw=archive.extractfile(str(rel)).read()
 if section=='16.3':frozen=(C/'docs/mastery/evidence/model_specific_shap_16_3_migration_20261021/raw_generated/solutions.py').read_bytes()
 else:
  with tarfile.open(C/'docs/mastery/evidence/lesson_integration_20261021_final3/foundation_v3/custody/complete_review.tar.gz')as archive:frozen=archive.extractfile('candidate/'+i['chapter']+'/exercises/'+i['old_dir']+'/solutions.py').read()
 assert(C/rel).read_bytes()==raw==frozen
def test_all_active_foundation_code_sources_and_local_reveals_stay_exact():
 for section in ['0.6','1.6']:
  i=M[section];row=next(x for x in PROJECTIONS if x['section']==section and x['kind']=='exercises');old=load(C/row['reference_path']);new=load(C/row['practice_path']);converted={x['cell']for x in row['operations']if x['kind_projection']}
  for n,c in enumerate(old['cells']):
   if c['cell_type']=='code'and n not in converted:assert new['cells'][n]['cell_type']=='code'and new['cells'][n]['source']==c['source']and new['cells'][n]['id']==c['id']and new['cells'][n]['metadata']==c['metadata']
  old_reveals=[c for c in old['cells']if c['cell_type']=='markdown'and'<summary>Solution</summary>'in src(c)];new_reveals=[c for c in new['cells']if c['cell_type']=='markdown'and'<summary>Solution</summary>'in src(c)];assert old_reveals==new_reveals
 defcount=[]
 for section in ['0.6','1.6']:
  i=M[section];p=C/i['directory']/'solutions.py';defcount.extend(n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)))
 assert len(defcount)>=49

def test_old57_case_objects_and_old15_recipe_objects_unchanged():
 from scripts.rome_runtime_metadata_checkpoints import recover_recipe_checkpoint_contracts
 state=load(E/'input_state.json');cases=load(C/'mastery_hub/extension_checkpoints.json');contracts=load(C/'infrastructure/extension/source_preservation_contracts.json');contracts=recover_recipe_checkpoint_contracts(C,(C/'infrastructure/extension/source_preservation_contracts.json').read_bytes());assert cases['checkpoints'][:57]==state['original57cases']['checkpoints'];assert{k:contracts['sections'][k]for k in state['original15recipes']['sections']}==state['original15recipes']['sections'];assert list(contracts['sections'])[:15]==list(state['original15recipes']['sections'])
