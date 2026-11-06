"""Exact source-only follow-on custody; no taught imports or numerical callbacks."""
from pathlib import Path
import ast,copy,hashlib,json,sys,tarfile
import pytest
from nbformat.validator import get_validator
ROOT=Path(__file__).resolve().parents[1];EV=ROOT/'docs/mastery/evidence/lesson_integration_20261021_followon4';sys.path[:0]=[str(ROOT),str(ROOT/'infrastructure/core')]
from infrastructure.extension.build import ExtensionMasterFile,normalize_generated_notebook
from infrastructure.extension.source_preservation import restore_notebooks
from scripts.evidence_bundles import validate_review_bundle
SECTIONS=['6.2','14.2','16.7','16.8'];load=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();src=lambda c:''.join(c['source'])
def arguments(section):
 recipe=load(ROOT/'infrastructure/extension/source_preservation_contracts.json')['sections'][section];payload={}
 for label,binding in recipe['archives'].items():
  validate_review_bundle(ROOT/binding['path'])
  with tarfile.open(ROOT/binding['path'])as h:payload[label]={m.name:h.extractfile(m).read()for m in h.getmembers()}
 master=ROOT/'infrastructure/chapters'/recipe['identity']['chapter']/('master_'+section.replace('.','_')+'.py')
 return {'master_path':master,'identity':recipe['identity'],'root':ROOT,'raw_notebooks':{k:payload['raw'][p['member']].decode()for k,p in recipe['books'].items()},'raw_modules':{k:payload['raw'][p['member']].decode()for k,p in recipe['modules'].items()}},recipe,payload

def test_original53_cases_ordered_prefix_and11_recipe_objects_exact():
 from scripts.rome_runtime_metadata_checkpoints import recover_recipe_checkpoint_contracts
 state=load(EV/'input_state.json');reg=load(ROOT/'mastery_hub/extension_checkpoints.json');contracts=load(ROOT/'infrastructure/extension/source_preservation_contracts.json')
 contracts=recover_recipe_checkpoint_contracts(ROOT,(ROOT/'infrastructure/extension/source_preservation_contracts.json').read_bytes())
 assert len(reg['checkpoints'][:57])==57 and reg['checkpoints'][:53]==state['original53cases']['checkpoints']
 assert json.dumps(reg['checkpoints'][:53],sort_keys=True)==json.dumps(state['original53cases']['checkpoints'],sort_keys=True)
 assert [c['section']for c in reg['checkpoints'][53:57]]==SECTIONS
 # This receipt owns its fixed cohort; later appends require separate validation.
 original_recipes=state['original11recipes']['sections']
 protected_sections=set(original_recipes)|set(SECTIONS)
 assert protected_sections<=contracts['sections'].keys()
 assert len(protected_sections)==15 and {k:v for k,v in contracts['sections'].items()if k in original_recipes}==original_recipes
 assert json.dumps({k:contracts['sections'][k]for k in original_recipes},sort_keys=True)==json.dumps(original_recipes,sort_keys=True)

def test_master_adoption_has_complete_preimages_and_explicit_ID_or_serialization_inverse():
 rows=load(EV/'first_master_adoption_inverse.json');assert len(rows)==4
 for row in rows:
  before=(ROOT/row['before_exact_path']).read_bytes();after=(ROOT/row['path']).read_bytes();assert sha(before)==row['before_sha256'] and sha(after)==row['after_sha256']
  a=json.loads(before);b=json.loads(after);assert not list(get_validator(4,5).iter_errors(b));inverse=copy.deepcopy(b)
  for op in row['id_operations']:
   assert inverse['cells'][op['cell']]['id']==op['after']
   if op['before']is None:inverse['cells'][op['cell']].pop('id')
   else:inverse['cells'][op['cell']]['id']=op['before']
  assert inverse==a
  if row['section']=='14.2':assert a==b and row['serialization_only'] and before!=after

def test_original_source_archive_and108_historical_output_locators_exact():
 archive=EV/'original_sources.tar.gz';validate_review_bundle(archive);manifest=load(EV/'original_sources_manifest.json')
 with tarfile.open(archive)as h:
  payload={m.name:h.extractfile(m).read()for m in h.getmembers()};assert len(payload)==len(manifest['files'])==135
  for member,pin in manifest['files'].items():assert sha(payload[member])==pin['sha256'] and len(payload[member])==pin['size_bytes']
 rows=load(EV/'historical_output_locators.json');assert len(rows)==108
 for row in rows:
  assert row['source_archive_sha256']==sha(archive.read_bytes()) and sha(payload[row['member']])==row['member_sha256']
  original=json.loads(payload[row['member']]);old=original['cells'][row['original_index']];assert old['cell_type']=='code'and old['outputs']
  current=load(ROOT/row['current_solutions_path'])
  for item in row['current_cells']:
   cell=current['cells'][item['index']];assert cell['id']==item['cell_id']and sha(src(cell).encode())==item['source_sha256']

def test_locator_clarifications_preserve_every_scientific_case_field():
 changes=load(EV/'case_locator_clarification_inverse.json');cases=load(ROOT/'mastery_hub/extension_checkpoints.json')['checkpoints'][53:57];assert [r['section']for r in changes]==SECTIONS
 for row,case in zip(changes,cases,strict=True):
  assert row['after']==case
  assert json.dumps(case,sort_keys=True)==json.dumps(row['after'],sort_keys=True)
  before=row['before'];assert {k:v for k,v in case.items()if k not in['config_fields','evidence_detail']}=={k:v for k,v in before.items()if k not in['config_fields','evidence_detail']}
  assert {k:v for k,v in case['config_fields'].items()if k!='source_identity'}=={k:v for k,v in before['config_fields'].items()if k!='source_identity'}
  assert 'zero-based' in case['config_fields']['source_identity']and 'ID 'in case['evidence_detail']

def test_annotation_disclosure_is_append_only_README_and_no_runtime_patch():
 row=load(EV/'token_region_annotation_disclosure.json');before=(ROOT/row['before_path']).read_bytes();after=(ROOT/row['after_path']).read_bytes();assert sha(before)==row['before_sha256']and sha(after)==row['after_sha256'];assert after.startswith(before);assert b'23 annotated VLM' in after

@pytest.mark.parametrize('section',SECTIONS)
def test_complete_before_objects_restored_except_guided_navigation(section):
 args,recipe,payload=arguments(section);result=restore_notebooks(**args)
 for kind,text in result.items():
  pin=recipe['books'][kind];old=json.loads(payload['before'][pin['member']]);new=json.loads(text);assert not list(get_validator(4,5).iter_errors(new));indices=[i for i,c in enumerate(old['cells'])if c['cell_type']=='markdown'and'> **Colab:'in src(c)];assert len(indices)==1;i=indices[0];assert '#section-'+section.replace('.','-')in src(new['cells'][i]);undo=copy.deepcopy(new);undo['cells'][i]['source']=old['cells'][i]['source'];assert undo==old;assert new==load(ROOT/pin['member'])
 for pin in recipe['modules'].values():assert (ROOT/pin['member']).read_bytes()==payload['raw'][pin['member']]==payload['before'][pin['member']]

@pytest.mark.parametrize('section',SECTIONS)
def test_actual_registered_generation_matches_whole_published_bytes(tmp_path,section):
 args,recipe,payload=arguments(section);identity=recipe['identity'];chapter=tmp_path/identity['chapter']
 for d in [chapter/'exercises'/identity['exercise_dir'],chapter/'instructions/pages']:d.mkdir(parents=True)
 master=args['master_path'].with_suffix('.ipynb');before=master.read_bytes();m=ExtensionMasterFile(master_path=master,chapter_dir=chapter,chapter_name_long='Source-only test',exercise_dir_name=identity['exercise_dir'],streamlit_page_name=identity['page']);m.generate_files(overwrite=True)
 for pin in recipe['books'].values():normalize_generated_notebook(tmp_path/pin['member'])
 assert m.source_preservation_applied and master.read_bytes()==before
 for pin in [*recipe['books'].values(),*recipe['modules'].values()]:assert (tmp_path/pin['member']).read_bytes()==(ROOT/pin['member']).read_bytes()

@pytest.mark.parametrize('section',SECTIONS)
@pytest.mark.parametrize('mutation',['source','output','count','top_metadata','cell_metadata'])
def test_actual_generation_refuses_unreviewed_published_objects_before_any_write(tmp_path,section,mutation):
 args,recipe,_=arguments(section);identity=recipe['identity'];chapter=tmp_path/identity['chapter']
 for d in [chapter/'exercises'/identity['exercise_dir'],chapter/'instructions/pages']:d.mkdir(parents=True)
 for pin in recipe['books'].values():(tmp_path/pin['member']).write_bytes((ROOT/pin['member']).read_bytes())
 path=tmp_path/recipe['books']['solutions']['member'];book=load(path);code=next(c for c in book['cells']if c['cell_type']=='code')
 if mutation=='source':code['source']=['unapproved_source=0\n']
 elif mutation=='output':code['outputs']=[{'output_type':'stream','name':'stdout','text':['unreviewed output\n']}]
 elif mutation=='count':code['execution_count']=987
 elif mutation=='top_metadata':book['metadata']['unapproved_learner_work']=True
 else:code['metadata']['unapproved_learner_work']=True
 path.write_text(json.dumps(book));before={str(p.relative_to(tmp_path)):p.read_bytes()for p in tmp_path.rglob('*')if p.is_file()}
 m=ExtensionMasterFile(master_path=args['master_path'].with_suffix('.ipynb'),chapter_dir=chapter,chapter_name_long='Refusal fixture',exercise_dir_name=identity['exercise_dir'],streamlit_page_name=identity['page'])
 with pytest.raises(ValueError,match='published notebook has unreviewed changes'):m.generate_files(overwrite=True)
 assert before=={str(p.relative_to(tmp_path)):p.read_bytes()for p in tmp_path.rglob('*')if p.is_file()}

@pytest.mark.parametrize('section',SECTIONS)
@pytest.mark.parametrize('mutation',['raw_whitespace','module_comment','identity'])
def test_exact_source_bindings_refuse_even_semantically_inert_drift(section,mutation):
 args,_,_=arguments(section)
 if mutation=='raw_whitespace':book=json.loads(args['raw_notebooks']['exercises']);book['cells'][0]['source'][0]+=' ';args['raw_notebooks']['exercises']=json.dumps(book)
 elif mutation=='module_comment':args['raw_modules']['solutions']+='\n# unreviewed bytes\n'
 else:args['identity']={**args['identity'],'exercise_dir':'unreviewed_alias'}
 with pytest.raises(ValueError):restore_notebooks(**args)
