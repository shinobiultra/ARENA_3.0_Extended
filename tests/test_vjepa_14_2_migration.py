"""Source and DEV engineering checks only. Never execute target numerical methods."""
from pathlib import Path
import ast,base64,copy,hashlib,importlib.util,json,re,sys,types
import pytest
ROOT=Path(__file__).resolve().parents[1];C=ROOT;W=ROOT/'docs/mastery/evidence/lesson_integration_20261021_followon4/vjepa_source_fixture';CH='chapter14_jepa_world_models';PART='part2_vjepa';P=C/CH/'exercises'/PART;E=C/'docs/mastery/evidence/vjepa_14_2_migration_20261021';STEM='14.2_V-JEPA_and_Video_World_Models'
load=lambda p:json.loads(p.read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
src=lambda c:''.join(c['source'])

def test_all_frozen_original_inputs_unchanged():
 rows=load(E/'source_manifest.json');assert len(rows)==35
 for row in rows:
  assert sha((ROOT/row['path']).read_bytes())==row['sha256']
  assert (W/'preimages'/row['path']).read_bytes()==(ROOT/row['path']).read_bytes()

def test_complete_cell_objects_and_top_metadata_inverse():
 m=load(E/'cell_inverse_map.json');meta=load(E/'legacy_book_metadata.json')
 for kind,rows in m.items():
  b=load(P/f'{STEM}_{kind}.ipynb');assert len(rows)==85
  for alias in ['scratch','controls']:
   selected=[r for r in rows if r['legacy']==alias];old=load(W/'preimages'/selected[0]['source_path'])
   assert {k:v for k,v in old.items() if k!='cells'}==meta[alias+'_'+kind]
   restored=[]
   for r in selected:
    assert old['cells'][r['original_index']]==r['original_cell']
    assert b['cells'][r['candidate_index']]==r['transformed_cell']
    recovered=copy.deepcopy(b['cells'][r['candidate_index']]);recovered['source']=copy.deepcopy(r['original_cell']['source'])
    if 'id' in r['original_cell']:recovered['id']=r['original_cell']['id']
    else:recovered.pop('id',None)
    recovered.update(r['compatibility_removals']);assert recovered==r['original_cell'];restored.append(recovered)
   assert restored==old['cells']

def module_defs(text):
 return [n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]

def test_all_original_method_and_stub_asts_invert_exactly():
 # Scope-aware token mapper is preparation code, never a taught numeric implementation.
 spec=importlib.util.spec_from_file_location('vj_prepare',W/'build_candidate.py')
 # Read just helper definitions; do not rerun preparation as a side effect.
 text=(W/'build_candidate.py').read_text();tree=ast.parse(text)
 rename=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rename')
 import io,tokenize
 ns={'ast':ast,'io':io,'tokenize':tokenize};exec(compile(ast.Module(body=[rename],type_ignores=[]),'DEV_source_mapper','exec'),ns)
 maps=load(E/'alpha_maps.json');count=0
 for kind,rows in load(E/'cell_inverse_map.json').items():
  for row in rows:
   old=row['original_cell'];new=row['transformed_cell']
   if old['cell_type']!='code':continue
   before=module_defs(src(old));after=module_defs(ns['rename'](src(new),{v:k for k,v in maps[row['legacy']].items()}))
   assert [ast.dump(x,include_attributes=False) for x in before]==[ast.dump(x,include_attributes=False) for x in after],row['id'];count+=len(before)
 assert count>100

def test_normal_converter_module_and_all_code_cells_match():
 # Generated module must be the ordinary converter product, byte for byte.
 assert (P/'solutions.py').read_bytes()==(W/'cli_probe'/CH/'exercises'/PART/'solutions.py').read_bytes()
 for kind in ['exercises','solutions']:
  raw=load(W/'raw_generated'/f'{STEM}_{kind}.ipynb');final=load(P/f'{STEM}_{kind}.ipynb')
  old=[ast.dump(ast.parse(src(c)),include_attributes=False) for c in raw['cells'] if c['cell_type']=='code']
  new=[ast.dump(ast.parse(src(c)),include_attributes=False) for c in final['cells'] if c['cell_type']=='code']
  assert old==new

def test_saved_books_and_master_strict_schema():
 import nbformat
 for path in [*P.glob('*.ipynb'),C/'infrastructure/chapters'/CH/'master_14_2.ipynb']:
  b=load(path);assert b['nbformat']==4 and b['nbformat_minor']==5;assert len({c['id'] for c in b['cells']})==len(b['cells']);nbformat.validate(b)

def test_media_bytes_authoritative_in_master_and_all_sidebar_panels():
 sys.path.insert(0,str(ROOT));from infrastructure.extension.render_media import inline_page_media
 page=C/CH/'instructions/pages/02_[14.2]_V-JEPA_and_Video_World_Models.md';shell=page.with_suffix('.py').read_text();text=page.read_text();master=(C/'infrastructure/chapters'/CH/'master_14_2.py').read_text()
 assert 'inline_page_media' in shell
 panels=text.split('=== NEW CHAPTER ===');assert len(panels)==4
 rendered=[inline_page_media(s,page) for s in panels];full='\n'.join(rendered)
 for row in load(E/'media_map.json'):
  p=C/row['target'];assert sha(p.read_bytes())==row['sha256'];assert '../../exercises/part2_vjepa/media/'+p.name in master
  assert '../../exercises/part2_vjepa/media/'+p.name in text
  if p.suffix=='.png':assert 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() in full
  else:assert 'data:image/svg+xml;base64,'+base64.b64encode(p.read_bytes()).decode() in full
 assert len(load(E/'media_map.json'))==16

class DEVPath:
 def __init__(self,value='DEV_fixture_root'):self.value=str(value)
 def __truediv__(self,v):return DEVPath(self.value+'/'+str(v))
 def __str__(self):return self.value
 @classmethod
 def cwd(cls):return cls('DEV_fixture_root')
 @property
 def parent(self):return DEVPath(self.value.rsplit('/',1)[0])
 @property
 def parents(self):return [self.parent]
 def exists(self):return True

def imported_with_DEV_stubs(monkeypatch,module_name='DEV_vjepa_candidate'):
 calls=[]
 def forbidden(*a,**kw):raise AssertionError('Forbidden target/runtime operation during plain import')
 torch=types.ModuleType('torch');torch.Tensor=object;torch.long='DEV-long';torch.float32='DEV-float32';torch.float64='DEV-float64'
 torch.nn=types.ModuleType('torch.nn');torch.nn.Module=type('DEVModule',(),{});torch.nn.functional=types.ModuleType('torch.nn.functional')
 torch.tensor=forbidden;torch.no_grad=forbidden;torch.set_num_threads=forbidden;torch.device=forbidden;torch.__version__='DEV-not-a-runtime'
 mods={'torch':torch,'torch.nn':torch.nn,'torch.nn.functional':torch.nn.functional}
 for name in ['numpy','pandas','matplotlib','matplotlib.pyplot','IPython','IPython.display','matplotlib.patches']:
  mods[name]=types.ModuleType(name)
 mods['matplotlib.pyplot'].style=types.SimpleNamespace(use=forbidden)
 mods['matplotlib.patches'].Rectangle=type('DEVRectangle',(),{})
 mods['IPython.display'].display=forbidden;mods['IPython.display'].Markdown=forbidden
 for name,module in mods.items():monkeypatch.setitem(sys.modules,name,module)
 module=types.ModuleType(module_name);module.__file__=str(P/'solutions.py');monkeypatch.setitem(sys.modules,module.__name__,module)
 exec(compile((P/'solutions.py').read_text(),str(P/'solutions.py'),'exec'),module.__dict__)
 assert not calls
 return module,torch,mods,calls

def test_plain_module_import_inert_without_paths_tensors_decorators_or_legacy_imports(monkeypatch):
 before=list(sys.path);module,torch,mods,calls=imported_with_DEV_stubs(monkeypatch)
 assert sys.path==before
 for name in ['DIRECTIONS','REVERSED_DIRECTIONS','scratch_root_dir','controls_root_dir','scratch_reference','controls_reference','scratch_tests','controls_tests','ema_update','evaluate_small_vjepa']:assert not hasattr(module,name),name
 assert callable(module.make_moving_object_batch) and callable(module.controls_reference_run_full_experiment)

def test_explicit_decorator_binding_preserves_original_signatures_and_bodies(monkeypatch):
 module,torch,mods,calls=imported_with_DEV_stubs(monkeypatch)
 def no_grad():
  calls.append('decorator-factory')
  def bind(fn):calls.append(('decorator-bind',fn.__name__));return fn
  return bind
 torch.no_grad=no_grad;module.initialize_vjepa_decorators()
 rows=load(E/'decorator_bindings.json');assert len(rows)==13
 assert [v[1] for v in calls if isinstance(v,tuple)]==[r['name'] for r in rows]
 for row in rows:
  fn=getattr(module,row['name']);assert fn.__qualname__==row['name'] and fn.__closure__ is None
  # The nested global def is the exact original decorated AST, including defaults and return annotations.
  expected=ast.parse(row['source']).body[0];tree=ast.parse((P/'solutions.py').read_text());init=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initialize_vjepa_decorators');actual=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name==row['name']);assert ast.dump(actual,include_attributes=False)==ast.dump(expected,include_attributes=False)
 previous=list(calls);module.initialize_vjepa_decorators();assert calls==previous

def test_explicit_runtime_DEV_recorder_no_target_bodies_and_no_wrapper_recursion(monkeypatch):
 module,torch,mods,calls=imported_with_DEV_stubs(monkeypatch)
 monkeypatch.setattr(module,'Path',DEVPath)
 torch.tensor=lambda values,**kwargs: calls.append(('DEV-tensor',values,kwargs)) or ('DEV-record',len(calls))
 torch.set_num_threads=lambda n:calls.append(('DEV-threads',n));torch.device=lambda s:('DEV-device',s)
 torch.no_grad=lambda:lambda fn:fn
 mods['matplotlib.pyplot'].style.use=lambda style:calls.append(('DEV-style',style))
 for name in ['part1_jepa_world_model_controls','part1_jepa_world_model_controls.tests']:
  fake=types.ModuleType(name);fake.__file__='DEV_engineering_fixture/'+name+'.py';monkeypatch.setitem(sys.modules,name,fake)
 # The real legacy test module's global fallback is originally an unqualified import.
 package='chapter14_jepa_world_models.exercises.part3_vjepa_from_scratch'
 for name in ['chapter14_jepa_world_models','chapter14_jepa_world_models.exercises',package]:
  fake=types.ModuleType(name);fake.__path__=[];monkeypatch.setitem(sys.modules,name,fake)
 original_reference=types.ModuleType(package+'.solutions');original_reference.__file__=str(ROOT/CH/'exercises/part3_vjepa_from_scratch/solutions.py')
 original_tests=types.ModuleType(package+'.tests');original_tests.__file__=str(ROOT/CH/'exercises/part3_vjepa_from_scratch/tests.py');original_tests.solutions=module
 sys.modules[package].solutions=original_reference;sys.modules[package].tests=original_tests
 monkeypatch.setitem(sys.modules,package+'.solutions',original_reference);monkeypatch.setitem(sys.modules,package+'.tests',original_tests)
 monkeypatch.setattr(sys,'path',list(sys.path))
 module.initialize_vjepa_runtime()
 assert [c[1] for c in calls if c[0]=='DEV-tensor']==[[[1,0],[-1,0],[0,1],[0,-1]],[1,0,3,2]]
 assert str(module.scratch_assets_dir).endswith('chapter14_jepa_world_models/exercises/part2_vjepa/media')
 assert str(module.controls_assets_dir).endswith('chapter14_jepa_world_models/exercises/part2_vjepa/media')
 assert str(module.controls_section_dir).endswith('part1_jepa_world_model_controls')
 for n in ['run_smoke_test','run_gpu_test','run_full_experiment']:
  assert getattr(module.controls_reference,n) is getattr(module,'controls_reference_'+n)
  assert getattr(module.controls_reference,n) is not getattr(module,'controls_'+n)
 assert module.ema_update.__qualname__=='ema_update'
 assert module.scratch_tests is original_tests and module.scratch_tests.solutions is original_reference
 assert module.scratch_reference is original_reference
 assert module.scratch_tests.__file__==str(ROOT/CH/'exercises/part3_vjepa_from_scratch/tests.py')
 previous=list(calls);previous_decorated=module.ema_update;module.initialize_vjepa_runtime()
 assert calls==previous and module.ema_update is previous_decorated
 assert module.scratch_tests.solutions is original_reference

def test_original_modules_and_tests_remain_full_byte_copies():
 for alias,part in [('scratch','part3_vjepa_from_scratch'),('controls','part1_jepa_world_model_controls')]:
  for name in ['solutions.py','tests.py','utils.py']:
   original=W/'preimages'/CH/'exercises'/part/name
   if original.exists():assert original.read_bytes()==(P/f'{alias}_legacy_{name}').read_bytes()

def test_blank_forecasts_precede_both_original_claims_and_all_exercises():
 b=load(P/f'{STEM}_exercises.ipynb');ids=[c['id'] for c in b['cells']]
 assert ids.index('vj-predict-intro')<ids.index('vj-scratch-legacy-000')
 assert ids.index('vj-predict-controls-0')<ids.index('vj-controls-legacy-000')
 for c in b['cells']:
  if c['id'].startswith('vj-predict-'):assert '______' in src(c)
 assert sum('### Exercise ' in src(c) for c in b['cells'] if c['cell_type']=='markdown')==23
 assert 'play_mode = "tube"' in src(b['cells'][ids.index('vj-scratch-legacy-043')])
 assert 'play_case = "blue_circle_right"' in src(b['cells'][ids.index('vj-controls-legacy-035')])

def test_complete_frozen_reference_and_shared_metric_methods_are_visible_exact_AST():
 text=(W/'build_candidate.py').read_text();rename=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='rename')
 import io,tokenize
 ns={'ast':ast,'io':io,'tokenize':tokenize};exec(compile(ast.Module(body=[rename],type_ignores=[]),'DEV_source_mapper','exec'),ns)
 b=load(P/f'{STEM}_solutions.ipynb');by_id={c['id']:c for c in b['cells']}
 for old_path,ident in [('arena_ext/jepa_world_models.py','vj-visible-shared-controls'),(f'{CH}/exercises/part1_jepa_world_model_controls/solutions.py','vj-visible-controls-reference')]:
  original=(W/'preimages'/old_path).read_text();rewritten=src(by_id[ident]);reverse={v:k for k,v in load(E/'alpha_maps.json')['controls'].items()}
  reverse.update({'controls_reference_'+n:n for n in ['run_smoke_test','run_gpu_test','run_full_experiment']})
  restored=ns['rename'](rewritten,reverse)
  before=[ast.dump(n,include_attributes=False) for n in module_defs(original)];after=[ast.dump(n,include_attributes=False) for n in module_defs(restored)];assert before==after


def test_unqualified_solutions_library_cannot_be_its_legacy_fallback(monkeypatch):
 module,torch,mods,calls=imported_with_DEV_stubs(monkeypatch,module_name='solutions')
 assert sys.modules['solutions'] is module
 monkeypatch.setattr(module,'Path',DEVPath)
 torch.tensor=lambda values,**kwargs:('DEV-record',repr(values))
 torch.set_num_threads=lambda n:None;torch.device=lambda name:('DEV-device',name);torch.no_grad=lambda:lambda fn:fn
 mods['matplotlib.pyplot'].style.use=lambda name:None
 for name in ['part1_jepa_world_model_controls','part1_jepa_world_model_controls.tests']:
  fake=types.ModuleType(name);fake.__file__='DEV_engineering_fixture/'+name+'.py';monkeypatch.setitem(sys.modules,name,fake)
 package='chapter14_jepa_world_models.exercises.part3_vjepa_from_scratch'
 for name in ['chapter14_jepa_world_models','chapter14_jepa_world_models.exercises',package]:
  fake=types.ModuleType(name);fake.__path__=[];monkeypatch.setitem(sys.modules,name,fake)
 legacy=types.ModuleType(package+'.solutions');legacy.__file__=str(ROOT/CH/'exercises/part3_vjepa_from_scratch/solutions.py')
 legacy.DEV_legacy_identity_marker=object()
 tests=types.ModuleType(package+'.tests');tests.__file__=str(ROOT/CH/'exercises/part3_vjepa_from_scratch/tests.py');tests.solutions=module
 # DEV marker reads the imported module binding; no taught callback or assertion runs.
 exec('def DEV_read_reference_identity():\n    return solutions.DEV_legacy_identity_marker\n',tests.__dict__)
 sys.modules[package].solutions=legacy;sys.modules[package].tests=tests
 monkeypatch.setitem(sys.modules,package+'.solutions',legacy);monkeypatch.setitem(sys.modules,package+'.tests',tests)
 before_file=tests.__file__;module.initialize_vjepa_runtime()
 assert sys.modules['solutions'] is module
 assert module.scratch_reference is legacy and module.scratch_reference is not module
 assert tests.solutions is legacy and tests.__file__==before_file
 assert tests.DEV_read_reference_identity.__globals__['solutions'] is legacy
 assert tests.DEV_read_reference_identity() is legacy.DEV_legacy_identity_marker
 module.initialize_vjepa_runtime();assert tests.solutions is legacy

def test_plain_import_cannot_mutate_already_loaded_original_test_module(monkeypatch):
 name='chapter14_jepa_world_models.exercises.part3_vjepa_from_scratch.tests'
 legacy=types.ModuleType(name);legacy.solutions=object();legacy.__file__=str(ROOT/CH/'exercises/part3_vjepa_from_scratch/tests.py')
 before=dict(legacy.__dict__);monkeypatch.setitem(sys.modules,name,legacy)
 imported_with_DEV_stubs(monkeypatch,module_name='solutions')
 assert legacy.__dict__==before
