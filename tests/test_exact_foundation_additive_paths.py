"""Q008 finite permission guard; no scientific execution."""
from pathlib import Path
import ast,importlib.util,json
import pytest
C=Path(__file__).resolve().parents[1]
R=Path('/home/bitwise/Documents/ARENA_3.0_Extended')
PROOF=json.loads((C/'docs/mastery/evidence/lesson_integration_20261021_q008/practice_audit_inverse.json').read_text())
@pytest.fixture
def audit():
 spec=importlib.util.spec_from_file_location('Q008_exact_practice_audit',C/'scripts/audit_original_arena_preservation.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
@pytest.mark.parametrize('path',PROOF['exact_files'])
def test_exact_eight_files_allowed(audit,path):
 assert audit.is_allowed_preservation_change(path)
 assert audit.is_allowed_preservation_change('./'+path)
@pytest.mark.parametrize('directory',PROOF['exact_directory_prefixes'])
def test_exact_two_practice_directories_allowed(audit,directory):
 assert audit.is_allowed_preservation_change(directory.rstrip('/'))
 assert audit.is_allowed_preservation_change(directory+'solutions.py')
@pytest.mark.parametrize('suffix',['.bak','_extra','/child.py','/../master_0_5.py'])
def test_file_entries_do_not_become_prefix_permissions(audit,suffix):
 for path in PROOF['exact_files']:assert not audit.is_allowed_preservation_change(path+suffix)
@pytest.mark.parametrize('suffix',['_evil/solutions.py','/../part0_prereqs/solutions.py','/nested/../../part0_prereqs/solutions.py'])
def test_directory_neighbors_and_traversal_rejected(audit,suffix):
 for directory in PROOF['exact_directory_prefixes']:assert not audit.is_allowed_preservation_change(directory.rstrip('/')+suffix)
def test_original_upstream_files_still_rejected(audit):
 for chapter in range(5):
  for path in [f'chapter{chapter}_original/exercises/solutions.py',f'infrastructure/chapters/chapter{chapter}_original/master_0_5.py']:assert not audit.is_allowed_preservation_change(path)
 assert not audit.is_allowed_preservation_change('chapter0_fundamentals/exercises/part0_prereqs/solutions.py')
 assert not audit.is_allowed_preservation_change('chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/solutions.py')
def test_old_asserts_and_constants_wholebytes_unchanged(audit):
 assert (C/'tests/test_original_arena_preservation_audit.py').read_bytes()==(R/'tests/test_original_arena_preservation_audit.py').read_bytes()
 tree=ast.parse(PROOF['root_source_before']);before={x.targets[0].id:ast.literal_eval(x.value)for x in tree.body if isinstance(x,ast.Assign)and len(x.targets)==1 and isinstance(x.targets[0],ast.Name)and x.targets[0].id in {'ADDITIVE_EXTENSION_PREFIXES','ADDITIVE_ORIGINAL_CHAPTER_PATHS','ORIGINAL_CHAPTER_PREFIXES','COMPATIBILITY_PATCH_RATIONALES'}}
 for name,value in before.items():assert getattr(audit,name)==value
 assert len(audit.ADDITIVE_ORIGINAL_CHAPTER_EXACT_FILES)==8
 assert len(audit.ADDITIVE_ORIGINAL_CHAPTER_PRACTICE_DIRS)==2
def test_every_original_source_byte_unchanged():
 for row in json.loads((C/'docs/mastery/evidence/lesson_integration_20261021_q008/original_source_preservation_baseline.json').read_text()):
  import hashlib
  assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256']
  if (C/row['path']).is_file():assert (C/row['path']).read_bytes()==(R/row['path']).read_bytes()
