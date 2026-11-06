"""Static source-custody and DEV import checks; no taught method is executed."""
from __future__ import annotations
import ast,base64,builtins,copy,hashlib,importlib.util,json,re,socket,sys,tarfile,textwrap
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch
import nbformat
import pytest

C=Path(__file__).resolve().parents[1];W=C.parent;R=C;CH='chapter16_shapley_attribution_baselines';PART='part7_token_region_shap';P=C/CH/'exercises'/PART;E=C/'docs/mastery/evidence/token_region_16_7_migration_20261021';STEM='16.7_TokenSHAP_and_Region_SHAP_for_LLMs_and_VLMs'

def filebytes(path):
 if path.exists():return path.read_bytes()
 if path.is_relative_to(W/'preimages'):
  with tarfile.open(E/'before.tar.gz') as t:return t.extractfile(str(path.relative_to(W/'preimages'))).read()
 if path.is_relative_to(W/'raw_generated'):return (E/'raw_generated'/path.name).read_bytes()
 raise FileNotFoundError(path)
def read(path):
 def unique(rows):
  result={}
  for k,v in rows:assert k not in result;result[k]=v
  return result
 return json.loads(filebytes(path),object_pairs_hook=unique,parse_constant=lambda x:pytest.fail(x))
def src(c):return ''.join(c['source'])
def rename(text,mapping):
 # Load only the source-transform definitions: no preparation side effects or scientific calls.
 path=W/'build_candidate.py'
 if not path.exists():path=E/'preparation_scripts/build_candidate.py'
 tree=ast.parse(filebytes(path));body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name in ['bindings','rename']]
 ns={};exec(compile(ast.Module(body=body,type_ignores=[]),'<DEV source alpha renamer>','exec'),ns)
 return ns['rename'](text,mapping)

def test_strict_saved_books_and_master():
 for p in [P/f'{STEM}_{k}.ipynb' for k in ['exercises','solutions']]+[C/'infrastructure/chapters'/CH/'master_16_7.ipynb']:
  b=read(p);assert (b['nbformat'],b['nbformat_minor'])==(4,5);nbformat.validate(b);assert len({c['id'] for c in b['cells']})==len(b['cells'])
  for c in b['cells']:
   if c['cell_type']=='markdown':assert 'outputs' not in c and 'execution_count' not in c

def test_every_original_cell_object_and_metadata_has_an_exact_inverse():
 m=read(E/'cell_inverse_map.json');metadata=read(E/'legacy_book_metadata.json')
 for kind,rows in m.items():
  b=read(P/f'{STEM}_{kind}.ipynb');assert len(rows)==69;recovered={}
  for row in rows:
   cell=b['cells'][row['candidate_index']];old=row['original_cell'];assert cell==row['transformed_cell'];assert cell['metadata']==old['metadata']
   if cell['cell_type']=='code':assert cell['outputs']==old['outputs'] and cell['execution_count']==old['execution_count']
   recovered.setdefault((row['legacy'],row['source_path']),[]).append((row['original_index'],old))
  for (alias,path),pairs in recovered.items():
   rebuilt=dict(metadata[alias+'_'+kind],cells=[v for i,v in sorted(pairs)]);assert rebuilt==read(W/'preimages'/path)

def test_all_student_stubs_and_historical_output_objects_survive():
 m=read(E/'cell_inverse_map.json');b=read(P/f'{STEM}_solutions.ipynb');assert sum(bool(c.get('outputs')) for c in b['cells'])==25
 for a,n in [('token',8),('region',7)]:
  x=[r for r in m['exercises'] if r['legacy']==a];y=[r for r in m['solutions'] if r['legacy']==a];different=0
  for ex,sol in zip(x,y,strict=True):
   if ex['original_cell']['source']!=sol['original_cell']['source']:
    different+=1;assert ex['transformed_cell']['source']!=sol['transformed_cell']['source']
  assert different==n

def test_raw_converter_has_exact_all_code_asts():
 for kind in ['exercises','solutions']:
  final=read(P/f'{STEM}_{kind}.ipynb');raw=read(W/'raw_generated'/f'{STEM}_{kind}.ipynb')
  f=[ast.dump(ast.parse(src(c))) for c in final['cells'] if c['cell_type']=='code'];r=[ast.dump(ast.parse(src(c))) for c in raw['cells'] if c['cell_type']=='code'];assert f==r;assert len(f)==34

def test_all_notebook_scientific_definitions_and_defaults_match_original_asts():
 maps=read(E/'alpha_maps.json')['notebooks'];m=read(E/'cell_inverse_map.json');count=0
 for kind,rows in m.items():
  for row in rows:
   if row['original_cell']['cell_type']!='code':continue
   original=ast.parse(src(row['original_cell']));restored=ast.parse(rename(src(row['transformed_cell']),{v:k for k,v in maps[row['legacy']].items()}));new={n.name:n for n in restored.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
   for old in original.body:
    if not isinstance(old,(ast.FunctionDef,ast.ClassDef)):continue
    if row['legacy']=='token' and old.name in ['run_gpu_test','run_full_experiment']:
     old=copy.deepcopy(old);old.body[0]=ast.parse('run = token_reference.'+old.name).body[0]
    assert ast.dump(old)==ast.dump(new[old.name]),(kind,row['legacy'],row['original_index'],old.name)
    count+=1
 assert count==76

def test_full_reference_function_class_default_asts_and_literals():
 maps=read(E/'alpha_maps.json')['references'];records=read(E/'reference_inverse_map.json');count=0
 for row in records:
  original=filebytes(W/'preimages'/row['source_path']).decode();assert row['original_source']==original
  candidate=next(c for c in read(P/f'{STEM}_solutions.ipynb')['cells'] if c['id']==row['candidate_cell_id']);assert src(candidate)==row['transformed_source']
  restored=ast.parse(rename(src(candidate),{v:k for k,v in maps[row['legacy']].items()}));new={n.name:n for n in restored.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  for old in ast.parse(original).body:
   if isinstance(old,(ast.FunctionDef,ast.ClassDef)):assert ast.dump(old)==ast.dump(new[old.name]);count+=1
  original_literals=[n for n in ast.parse(original).body if isinstance(n,(ast.Assign,ast.AnnAssign)) and not any(isinstance(x,ast.Name) and x.id=='MAIN' for x in ast.walk(n))]
  for old in original_literals:
   found=[n for n in ast.walk(restored) if isinstance(n,type(old)) and ast.dump(n)==ast.dump(old)]
   assert found,ast.unparse(old)
  # Both library flags are explicit false; the original notebook MAIN remains separate.
  assert row['legacy']+'_reference_MAIN = False' in src(candidate)
 assert count==56

def test_scoped_alpha_renaming_keeps_parameters_keywords_attributes_and_comprehensions():
 text='''thing = 11
other = 17
def callback(thing, *, other=other):
    local = [thing for thing in range(3)]
    nested = [other for other in range(2)]
    return api.thing(thing=thing, other=other), local, nested
def uses_global():
    return thing + other
'''
 actual=rename(text,{'thing':'DEV_thing','other':'DEV_other'});a=ast.parse(actual);f=a.body[2]
 assert ast.unparse(f.args)=='thing, *, other=DEV_other'
 assert 'api.thing(thing=thing, other=other)' in actual
 assert '[thing for thing in range(3)]' in actual
 assert '[other for other in range(2)]' in actual
 assert 'return DEV_thing + DEV_other' in actual
 assert ast.dump(ast.parse(rename(actual,{'DEV_thing':'thing','DEV_other':'other'})))==ast.dump(ast.parse(text))

def test_all_original_input_bytes_and_legacy_helpers_stay_exact():
 rows=read(E/'source_manifest.json');assert len(rows)==34
 with tarfile.open(E/'before.tar.gz') as t:
  for row in rows:
   raw=t.extractfile(row['path']).read();assert hashlib.sha256(raw).hexdigest()==row['sha256'];assert len(raw)==row['size'];assert (R/row['path']).read_bytes()==raw
 for a,part in [('token','part4_tokenshap_token_shapley'),('region','part5_vlm_modality_region_shap')]:
  for name in ['tests.py','utils.py']:assert (P/f'{a}_{name}').read_bytes()==filebytes(W/'preimages'/CH/'exercises'/part/name)

def test_all_solution_fences_parse_after_only_declared_first_def_indent_repair():
 maps=read(E/'alpha_maps.json')['notebooks'];m=read(E/'cell_inverse_map.json');repaired=set();count=0
 for kind,rows in m.items():
  for row in rows:
   if row['original_cell']['cell_type']!='markdown':continue
   oldf=re.findall(r'```python\n(.*?)```',src(row['original_cell']),re.S);newf=re.findall(r'```python\n(.*?)```',src(row['transformed_cell']),re.S);assert len(oldf)==len(newf)
   for old,new in zip(oldf,newf,strict=True):
    try:ast.parse(old);normalized=old
    except IndentationError:
     assert row['legacy']=='region';assert old.startswith('        def ');normalized=old[8:];repaired.add(row['original_index'])
     assert normalized.splitlines()[1:]==old.splitlines()[1:]
    restored=rename(new,{v:k for k,v in maps[row['legacy']].items()});assert ast.dump(ast.parse(restored))==ast.dump(ast.parse(normalized));count+=1
 assert repaired=={6,8,11,13,16,18,20};assert count>=30

def test_actual_generated_module_import_is_inert():
 import torch,matplotlib,matplotlib.pyplot,pandas,numpy,IPython.display
 sys.dont_write_bytecode=True
 baseline_path=list(sys.path);sys.path[:0]=[str(C),str(R)]
 # All dependency imports occur outside the observed raw generated module boundary.
 __import__(CH+'.exercises.part4_tokenshap_token_shapley.tests');__import__(CH+'.exercises.part5_vlm_modality_region_shap.tests')
 rng=torch.random.get_rng_state().clone();dtype=torch.get_default_dtype();python_rng=__import__('random').getstate();numpy_rng=numpy.random.get_state()
 expected_path=list(sys.path);events=[]
 def forbidden(*args,**kwargs):events.append((args,kwargs));raise AssertionError('Unexpected lesson runtime work on import')
 spec=importlib.util.spec_from_file_location('token_region_DEV_import',P/'solutions.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
 try:
  with __import__('contextlib').ExitStack() as stack:
   for obj,names in [(torch,['tensor','as_tensor','zeros','ones','empty','arange','linspace','rand','randn','manual_seed','set_default_dtype','set_default_device','compile']),(torch.nn.Module,['__init__']),(Path,['cwd','read_text','read_bytes','write_text','write_bytes','mkdir','open']),(socket,['socket','create_connection']),(matplotlib.pyplot,['show','subplots','figure']),(IPython.display,['display'])]:
    for name in names:stack.enter_context(patch.object(obj,name,forbidden))
   stack.enter_context(patch.object(matplotlib.pyplot.style,'use',forbidden));stack.enter_context(patch.object(__import__('os'),'chdir',forbidden));stack.enter_context(patch.object(__import__('random'),'seed',forbidden))
   spec.loader.exec_module(mod)
  assert not events;assert torch.equal(rng,torch.random.get_rng_state());assert dtype==torch.get_default_dtype();assert python_rng==__import__('random').getstate();after=numpy.random.get_state();assert numpy_rng[0]==after[0] and numpy.array_equal(numpy_rng[1],after[1]) and numpy_rng[2:]==after[2:];assert sys.path==expected_path
  for name in ['token_all_coalitions','region_all_coalitions','token_exact_shapley_values','region_exact_shapley_values','token_reference_run_neural_token_shapley_preflight','region_reference_run_real_clip_vlm_shap_signature_result']:assert callable(getattr(mod,name))
  assert mod.token_reference_MAIN is False and mod.region_reference_MAIN is False
  assert Path(mod.token_tests.__file__).resolve()==R/CH/'exercises/part4_tokenshap_token_shapley/tests.py'
  assert Path(mod.region_tests.__file__).resolve()==R/CH/'exercises/part5_vlm_modality_region_shap/tests.py'
  assert mod.region_reference.run_real_clip_vlm_shap_signature_result is mod.region_reference_run_real_clip_vlm_shap_signature_result
  assert mod.token_reference.run_full_experiment is mod.token_reference_run_full_experiment
  for name in ['token_root_dir','region_root_dir','root_dir','section_dir','figure_path','assets_dir','signature_result','payload']:assert not hasattr(mod,name)
  for name in ['token_reference_exact_shapley_values','region_reference_exact_shapley_values']:
   f=getattr(mod,name);assert f.__closure__ is None and f.__qualname__==name
  # Initialize plumbing with path-only DEV fixtures; no real paths or scientific callbacks are evaluated.
  calls=[]
  class DEVPath(PurePosixPath):
   def exists(self):calls.append(('exists',str(self)));return str(self)=='/DEV_ROOT/'+CH
  mod.Path=SimpleNamespace(cwd=lambda:DEVPath('/DEV_ROOT'))
  with patch.object(torch,'tensor',forbidden),patch.object(torch,'zeros',forbidden):mod.initialize_lesson_plumbing()
  assert str(mod.section_dir)=='/DEV_ROOT/'+CH+'/exercises/part4_tokenshap_token_shapley';assert str(mod.figure_path)=='/DEV_ROOT/'+CH+'/exercises/'+PART+'/media/tokenshap_exact_signature.png';assert str(mod.assets_dir)=='/DEV_ROOT/'+CH+'/exercises/'+PART+'/media'
 finally:sys.path[:]=baseline_path;sys.modules.pop(spec.name,None)

def test_standard_page_media_inlines_all_historical_bytes():
 sys.path.insert(0,str(R))
 try:
  from infrastructure.extension.render_media import inline_page_media
 finally:sys.path.pop(0)
 page=C/CH/'instructions/pages/07_[16.7]_TokenSHAP_and_Region_SHAP_for_LLMs_and_VLMs.md';shell=page.with_suffix('.py').read_text();assert 'inline_page_media(markdown_content_file.read_text' in shell
 text=page.read_text();inlined=inline_page_media(text,page);rows=read(E/'media_map.json');assert len(rows)==8
 decoded={hashlib.sha256(base64.b64decode(data)).hexdigest() for data in re.findall(r'data:image/(?:png|svg\+xml);base64,([^"\s>]+)',inlined)}
 for row in rows:assert hashlib.sha256((C/row['target']).read_bytes()).hexdigest()==row['sha256'];assert row['sha256'] in decoded
 assert f'../../exercises/{PART}/media/' not in inlined

def test_forecasts_reflections_independent_estimands_and_no_report_primary():
 b=read(P/f'{STEM}_exercises.ipynb');text='\n'.join(src(c) for c in b['cells'] if c['cell_type']=='markdown');assert text.index('Your commitment')<text.index('Visible complete token reference')
 assert text.count('### Predict before inspection')==15;assert '______' in text;assert 'Reflect after testing' in text
 for s in ['position-preserving','empty-coalition baseline','not aligned or pooled','five-token analytic learner game','historical guided practice','Try It Yourself','Limitations']:assert s in text


def test_legacy_report_input_identity_uses_only_a_DEV_path_resolver():
 old=R/CH/'exercises/part5_vlm_modality_region_shap/tests.py'
 tree=ast.parse(old.read_text());resolver=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_section_dir')
 class DEVPath(PurePosixPath):
  def resolve(self):return self
 ns={'Path':DEVPath,'__file__':str(old)}
 exec(compile(ast.Module(body=[resolver],type_ignores=[]),'<DEV path-only resolver>','exec'),ns)
 report=ns['_section_dir']()/'verification_report.json'
 assert str(report)==str(R/CH/'exercises/part5_vlm_modality_region_shap/verification_report.json')
 # Inspect the report literal, never call the legacy reader or GPU/report-dependent test.
 reader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gpu_report')
 assert 'verification_report.json' in [n.value for n in ast.walk(reader) if isinstance(n,ast.Constant)]
