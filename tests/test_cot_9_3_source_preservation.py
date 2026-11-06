"""Pure source contract: no lesson imports, fixtures, tensors or science execution."""
from pathlib import Path
import ast,base64,hashlib,json,re
import yaml
ROOT=Path(__file__).resolve().parents[1]
SECTION=ROOT/'chapter9_alignment_interpretability/exercises/part3_cot_faithfulness'
EVIDENCE=ROOT/'docs/mastery/evidence/cot_faithfulness_master_migration'
def source(c):return ''.join(c['source'])
def digest(b):return hashlib.sha256(b).hexdigest()
def normalize(s):return re.sub(r'\s+',' ',s).strip()
def assert_preserved_prose(text,rows):
 for r in rows:assert normalize(r['adapted_text']) in normalize(text),r['legacy_cell']
CANONICAL_STEM='9.3_Chain-of-Thought_Faithfulness'
CANONICAL_PAGE='03_[9.3]_Chain-of-Thought_Faithfulness'
def test_actual_configuration_and_objectives():
 for filename in ['config.yaml','config_extension.yaml']:
  config=yaml.safe_load((ROOT/'infrastructure/core'/filename).read_text())
  mapping=config['conversion_map']['9.3']
  assert mapping['exercise_dir']==SECTION.name
  assert mapping['streamlit_page']==CANONICAL_PAGE
  # Exact MasterFileData.colab_name derivation, checked against real config.
  assert mapping['streamlit_page'].split('_[')[-1].replace(']','')==CANONICAL_STEM
 master=(ROOT/'infrastructure/chapters/chapter9_alignment_interpretability/master_9_3.py').read_text()
 assert master.count('## Content & Learning Objectives')==1
 assert master.count('## Original study objectives')==1
 assert '> ##### Learning Objectives\n>\n> 1. Build an exact toy organism' in master
 case=json.loads((EVIDENCE/'proposed_case_canonical_paths.json').read_text())
 for kind,key in [('exercises','exercise_path'),('solutions','solutions_path')]:
  assert ROOT/case[key]==SECTION/f'{CANONICAL_STEM}_{kind}.ipynb'

def test_parity():
 test_actual_configuration_and_objectives()
 ledger=json.loads((EVIDENCE/'parity_ledger.json').read_text())
 for kind in ['solutions','exercises']:
  book=json.loads((SECTION/f'{CANONICAL_STEM}_{kind}.ipynb').read_text());code=[c for c in book['cells'] if c['cell_type']=='code'];assert len(code)==15
  prose='\n'.join(source(c) for c in book['cells'] if c['cell_type']=='markdown');assert_preserved_prose(prose,ledger['markdown_cells'])
  assert len(book['metadata']['migration']['original_all_cell_metadata'])==47
  if kind=='solutions':
   assert sum(len(c['outputs']) for c in code)==14
   for c,r in zip(code,ledger['code_cells'],strict=True):
    assert digest(ast.dump(ast.parse(source(c))).encode())==r['routed_ast_sha256']
    assert digest(json.dumps(c['outputs'],sort_keys=True).encode())==r['outputs_sha256']
    assert c['execution_count']==r['execution_count']
  else:
   assert all(c['execution_count'] is None for c in code)
   assert sum(source(c).count('raise NotImplementedError') for c in code)==10
   for c in code:
    for o in c['outputs']:
     assert o['output_type']=='display_data'
     assert ''.join(o['data']['text/plain'])=='<IPython.core.display.HTML object>'
     assert '<img ' in ''.join(o['data']['text/html']) and 'data:image/png;base64,' in ''.join(o['data']['text/html'])
 for r in ledger['figures']:assert digest((SECTION/'media'/r['name']).read_bytes())==r['sha256']
 assert len({r['sha256'] for r in ledger['figures']})==3
 # A dropped prose paragraph and altered taught source must fail the same comparisons.
 first=ledger['markdown_cells'][0]['adapted_text'];mutated=first.replace('By the end','Removed forecast',1)
 assert normalize(mutated)!=normalize(first)
 try:assert_preserved_prose('',ledger['markdown_cells'])
 except AssertionError:pass
 else:raise AssertionError('dropped-prose negative control failed')
 original=json.loads((SECTION/f'{CANONICAL_STEM}_solutions.ipynb').read_text());c=next(c for c in original['cells'] if c['cell_type']=='code');changed=ast.parse(source(c)+'\nSCIENCE_DEFAULT_CHANGED = True\n')
 assert digest(ast.dump(changed).encode())!=ledger['code_cells'][0]['routed_ast_sha256']
 # All student functions are own notebook implementations, not proxy/hidden helper bindings.
 module=ast.parse((SECTION/'solutions.py').read_text());defs={n.name:n for n in module.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for c in original['cells']:
  if c['cell_type']=='code':
   for n in ast.parse(source(c)).body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert ast.dump(n)==ast.dump(defs[n.name]),n.name
 wrapper=defs['run_gpu_test'];assert 'reference_solutions.run_gpu_test' in ast.unparse(wrapper)
 assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and 'part2_cot_faithfulness' in ast.unparse(n) for n in module.body)
 page=(ROOT/f'chapter9_alignment_interpretability/instructions/pages/{CANONICAL_PAGE}.md').read_text()
 for r in ledger['markdown_cells']:
  # Page media links are canonically rewritten, so normalize only that explicit route.
  text=r['adapted_text'].replace('media/historical_saved_toy_panel.png','../../exercises/part3_cot_faithfulness/media/historical_saved_toy_panel.png')
  assert_preserved_prose(page,[{**r,'adapted_text':text}])
if __name__=='__main__':test_parity();print('PASS: source parity, prose, metadata, own functions, output/counts, 3 visuals, expected-only HTML, lazy routing and mutation negatives')
def test_only_configured_cot_page_filename_is_published():
    pages = ROOT / 'chapter9_alignment_interpretability/instructions/pages'
    for suffix in ('.md', '.py'):
        assert (pages / ('03_[9.3]_Chain-of-Thought_Faithfulness' + suffix)).is_file()
        assert not (pages / ('03_[9.3]_Chain_of_Thought_Faithfulness' + suffix)).exists()

