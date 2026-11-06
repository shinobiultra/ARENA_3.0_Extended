"""Exact-current source checkpoint inverses; only file/static DEV fault injection."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os
import pytest
ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'scripts/rome_runtime_metadata_checkpoints.py'
E=ROOT/'docs/mastery/evidence/rome_runtime_metadata_20261021_reconstructed_v1'
@pytest.fixture
def helper():
 spec=importlib.util.spec_from_file_location('DEV_rome_checkpoint_bridge',HELPER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
@pytest.fixture
def contract():return json.loads((E/'checkpoint_inverse_contract.json').read_text())
def inject_read(monkeypatch,helper,relative,data):
 original=helper._read
 def fault(root,path):return data if path==relative else original(root,path)
 monkeypatch.setattr(helper,'_read',fault)
@pytest.mark.parametrize('index',[0,1,2])
def test_exact_original_bytes_return_and_complete_inverse(helper,contract,index):
 import tarfile
 path=helper.JSON_TARGETS[index];actual=(ROOT/path).read_bytes();recovered=helper.recover_intro_checkpoint_bytes(ROOT,path,actual)
 with tarfile.open(ROOT/contract['artifacts']['actual60_preimages']['archive_path'])as h:expected=h.extractfile(path).read()
 assert recovered==expected and recovered!=actual
 assert hashlib.sha256(recovered).hexdigest()==contract['artifacts']['actual60_preimages']['files'][path]['sha256']
def test_complete18_recipe_inverse_keeps17_and60_cases_exact(helper,contract):
 after=json.loads((ROOT/helper.CONTRACTS).read_bytes());before=helper.recover_recipe_checkpoint_contracts(ROOT,(ROOT/helper.CONTRACTS).read_bytes())
 assert len(before['sections'])==len(after['sections'])==18
 assert list(before['sections'])==list(after['sections'])
 assert helper._typed({k:v for k,v in before['sections'].items()if k!='9.6'})==helper._typed({k:v for k,v in after['sections'].items()if k!='9.6'})
 assert helper._typed(before['sections']['9.6'])==helper._typed(contract['contracts']['before_recipe'])
@pytest.mark.parametrize('index',list(range(7)))
def test_any_current_product_byte_drift_rejected(helper,contract,monkeypatch,index):
 path=(*helper.PRODUCTS,*helper.UNCHANGED)[index];inject_read(monkeypatch,helper,path,(ROOT/path).read_bytes()+b'\n')
 with pytest.raises(ValueError,match='Current approved product changed'):helper.recover_recipe_checkpoint_contracts(ROOT,(ROOT/helper.CONTRACTS).read_bytes())
@pytest.mark.parametrize('mutation',['code','output','count','metadata','top_metadata','id','cell_count','source_representation'])
def test_whole_notebook_field_mutations_rejected(helper,monkeypatch,mutation):
 path=helper.BOOKS[0];obj=json.loads((ROOT/path).read_bytes());cell=next(c for c in obj['cells']if c['cell_type']=='code')
 if mutation=='code':cell['source']=['DEV unapproved source\n']
 elif mutation=='output':cell['outputs']=[{'output_type':'stream','name':'stdout','text':['DEV unapproved output']}]
 elif mutation=='count':cell['execution_count']=991
 elif mutation=='metadata':cell['metadata']['DEV']=True
 elif mutation=='top_metadata':obj['metadata']['DEV']=True
 elif mutation=='id':cell['id']='DEV-unapproved-id'
 elif mutation=='cell_count':obj['cells'].pop()
 else:obj['cells'][1]['source']=''.join(obj['cells'][1]['source'])
 inject_read(monkeypatch,helper,path,json.dumps(obj).encode())
 with pytest.raises(ValueError):helper.recover_intro_checkpoint_bytes(ROOT,path,(ROOT/path).read_bytes())
@pytest.mark.parametrize('index',[0,1,2])
def test_supplied_old_payload_cannot_substitute_current_file(helper,contract,index):
 import tarfile
 path=helper.JSON_TARGETS[index]
 with tarfile.open(ROOT/contract['artifacts']['actual60_preimages']['archive_path'])as h:old=h.extractfile(path).read()
 with pytest.raises(ValueError,match='Supplied bytes differ'):helper.recover_intro_checkpoint_bytes(ROOT,path,old)
@pytest.mark.parametrize('mutation',['source','id','cell','path','extra_operation','remove_field','bool_index','float_index','duplicate_key','nonfinite'])
def test_inverse_contract_cannot_authorize_itself(helper,contract,monkeypatch,mutation):
 obj=copy.deepcopy(contract)
 if mutation=='source':obj['intro_inverses'][0]['fields']['source']['before']=['DEV changed source']
 elif mutation=='id':obj['intro_inverses'][0]['fields']['id']['before']='DEV changed ID'
 elif mutation=='cell':obj['intro_inverses'][0]['cell']=4
 elif mutation=='path':obj['intro_inverses'][0]['path']='../escape'
 elif mutation=='extra_operation':obj['intro_inverses'][0]['fields']['metadata']={'before':{},'after':{}}
 elif mutation=='remove_field':obj.pop('section')
 elif mutation=='bool_index':obj['intro_inverses'][0]['cell']=True
 elif mutation=='float_index':obj['intro_inverses'][0]['cell']=3.0
 if mutation=='duplicate_key':data=b'{"schema":"x","schema":"x"}'
 elif mutation=='nonfinite':data=b'{"value":NaN}'
 else:data=json.dumps(obj).encode()
 inject_read(monkeypatch,helper,helper.CONTRACT_PATH,data)
 with pytest.raises(ValueError,match='Inverse contract changed'):helper.recover_recipe_checkpoint_contracts(ROOT,(ROOT/helper.CONTRACTS).read_bytes())
@pytest.mark.parametrize('role',['actual60_preimages','before','raw'])
@pytest.mark.parametrize('field',['archive_path','sidecar_path','manifest_path'])
def test_archive_sidecar_manifest_drift_rejected(helper,contract,monkeypatch,role,field):
 path=contract['artifacts'][role][field];inject_read(monkeypatch,helper,path,(ROOT/path).read_bytes()+b'DEV drift')
 with pytest.raises(ValueError,match='Artifact identity changed'):helper.recover_recipe_checkpoint_contracts(ROOT,(ROOT/helper.CONTRACTS).read_bytes())
@pytest.mark.parametrize('section',list(json.loads((ROOT/'infrastructure/extension/source_preservation_contracts.json').read_text())['sections']))
def test_any_of18_recipe_changes_rejected(helper,monkeypatch,section):
 obj=json.loads((ROOT/helper.CONTRACTS).read_bytes());obj['sections'][section]['DEV_unapproved']=True;inject_read(monkeypatch,helper,helper.CONTRACTS,json.dumps(obj).encode())
 with pytest.raises(ValueError,match='Current complete18 contract bytes changed'):helper.recover_recipe_checkpoint_contracts(ROOT,(ROOT/helper.CONTRACTS).read_bytes())
@pytest.mark.parametrize('mutation',['drop_recipe','add_recipe','rename_recipe','top_key','schema','duplicate_key','nonfinite','subset_input'])
def test_complete_contract_shape_and_caller_projection_rejected(helper,monkeypatch,mutation):
 original=(ROOT/helper.CONTRACTS).read_bytes();obj=json.loads(original)
 if mutation=='drop_recipe':obj['sections'].pop('9.6')
 elif mutation=='add_recipe':obj['sections']['DEV99']={}
 elif mutation=='rename_recipe':obj['sections']['DEV9.6']=obj['sections'].pop('9.6')
 elif mutation=='top_key':obj['DEV']=True
 elif mutation=='schema':obj['schema']='DEV'
 if mutation=='duplicate_key':data=b'{"sections":{},"sections":{}}'
 elif mutation=='nonfinite':data=b'{"sections":{},"DEV":Infinity}'
 elif mutation=='subset_input':
  with pytest.raises(ValueError,match='Supplied contracts differ'):helper.recover_recipe_checkpoint_contracts(ROOT,json.dumps({'sections':{'9.6':obj['sections']['9.6']}}).encode())
  return
 else:data=json.dumps(obj).encode()
 inject_read(monkeypatch,helper,helper.CONTRACTS,data)
 with pytest.raises(ValueError):helper.recover_recipe_checkpoint_contracts(ROOT,original)
@pytest.mark.parametrize('mutation',['append','drop','bool_scalar','float_scalar','question'])
def test_any_current60_case_change_rejected(helper,monkeypatch,mutation):
 obj=json.loads((ROOT/helper.CASES).read_bytes())
 if mutation=='append':obj['checkpoints'].append(copy.deepcopy(obj['checkpoints'][0]))
 elif mutation=='drop':obj['checkpoints'].pop()
 elif mutation=='bool_scalar':obj['schema_version']=True
 elif mutation=='float_scalar':obj['schema_version']=1.0
 else:obj['checkpoints'][0]['question']='DEV changed question'
 inject_read(monkeypatch,helper,helper.CASES,json.dumps(obj).encode())
 with pytest.raises(ValueError,match='Current complete60 case bytes changed'):helper.recover_recipe_checkpoint_contracts(ROOT,(ROOT/helper.CONTRACTS).read_bytes())
@pytest.mark.parametrize('path',['../escape','/absolute','other.ipynb','infrastructure/chapters/chapter9_alignment_interpretability/master_9_6.py'])
def test_unknown_checkpoint_path_rejected_without_fallback(helper,path):
 with pytest.raises(ValueError,match='Unknown historical checkpoint path'):helper.recover_intro_checkpoint_bytes(ROOT,path,b'DEV')
@pytest.mark.parametrize('data',[b'{"a":1,"a":1}',b'{"a":NaN}',b'{"a":Infinity}',b'{"a":-Infinity}'])
def test_strict_json_rejects_duplicate_keys_and_nonfinite(helper,data):
 with pytest.raises(ValueError):helper._json(data)
def test_scalar_types_remain_distinct(helper):
 assert len({helper._typed({'x':v})for v in [True,1,1.0,'1']})==4
@pytest.mark.parametrize('kind',['target_symlink','directory_symlink','root_symlink','fifo','unreadable','locked_parent','non_directory'])
def test_actual_filesystem_inputs_do_not_follow_links_or_hide_permission_errors(helper,tmp_path,kind):
 file=tmp_path/'original';file.write_bytes(b'DEV')
 if kind=='target_symlink':(tmp_path/'link').symlink_to(file);root=tmp_path;path='link'
 elif kind=='directory_symlink':(tmp_path/'directory').symlink_to(tmp_path,target_is_directory=True);root=tmp_path;path='directory/original'
 elif kind=='root_symlink':(tmp_path/'rootlink').symlink_to(tmp_path,target_is_directory=True);root=tmp_path/'rootlink';path='original'
 elif kind=='fifo':os.mkfifo(tmp_path/'fifo');root=tmp_path;path='fifo'
 elif kind=='unreadable':file.chmod(0);root=tmp_path;path='original'
 elif kind=='locked_parent':
  parent=tmp_path/'locked';parent.mkdir();(parent/'file').write_bytes(b'DEV');parent.chmod(0);root=tmp_path;path='locked/file'
 else:root=tmp_path;path='original/child'
 try:
  with pytest.raises((ValueError,OSError)):helper._read(root,path)
 finally:
  if kind=='unreadable':file.chmod(0o600)
  if kind=='locked_parent':parent.chmod(0o700)
def test_helper_imports_only_file_metadata_plumbing():
 tree=ast.parse(HELPER.read_text());imports=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):imports.extend(v.name for v in node.names)
  elif isinstance(node,ast.ImportFrom):imports.append(node.module)
 assert set(imports)<={'pathlib','copy','hashlib','json','os','stat','tarfile','scripts.evidence_bundles'}

@pytest.mark.parametrize('index',[0,1,2])
def test_old_supplied_bytes_cannot_hide_mutated_current_file(helper,contract,monkeypatch,index):
 import tarfile
 path=helper.JSON_TARGETS[index]
 with tarfile.open(ROOT/contract['artifacts']['actual60_preimages']['archive_path'])as h:old=h.extractfile(path).read()
 inject_read(monkeypatch,helper,path,(ROOT/path).read_bytes()+b'DEV changed actual file')
 with pytest.raises(ValueError,match='Current approved product changed'):helper.recover_intro_checkpoint_bytes(ROOT,path,old)
