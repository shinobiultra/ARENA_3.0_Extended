"""Pure structural custody tests; never import or execute a taught method/model."""
from pathlib import Path
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
import re
import tarfile
import unittest

ROOT=Path(os.environ.get('ARENA_MIGRATION_ROOT',Path(__file__).resolve().parents[1]))
LEGACY=Path(os.environ.get('ARENA_LEGACY_ROOT',ROOT))
CH='chapter9_alignment_interpretability'
OLD=LEGACY/CH/'exercises/part5_rome_model_editing'
NEW=ROOT/CH/'exercises/part6_rome_memit'
E=ROOT/'docs/mastery/evidence/rome_9_6_migration_20261021/v2'
MAPPING=ROOT/'docs/mastery/evidence/rome_9_6_migration_20261021/post_header_mapping_20261021/original_cell_mapping.json'
STEM='9.6_Model_Editing_with_ROME_and_MEMIT'
PAGE=ROOT/CH/'instructions/pages/06_[9.6]_Model_Editing_with_ROME_and_MEMIT.md'
MASTER=ROOT/'infrastructure/chapters'/CH/'master_9_6.py'
load=lambda p:json.loads(p.read_text())
source=lambda c:''.join(c['source'])
sha=lambda b:hashlib.sha256(b).hexdigest()


def pair(kind):
    return load(OLD/f'9.5_ROME_Model_Editing_{kind}.ipynb'),load(NEW/f'{STEM}_{kind}.ipynb')


def inverse(kind,new):
    result=deepcopy(new)
    cells={c['id']:c for c in result['cells']}
    for change in load(E/'declared_source_changes.json')[kind]:
        cell=cells[change['id']]
        if source(cell)!=change['after_source']:raise ValueError('canonical source changed')
        # Restore the exact original string/list representation, not normalized whitespace.
        old=pair(kind)[0]['cells'][change['original_index']]
        cell['source']=deepcopy(old['source'])
    result['cells']=[cells[row['id']] for row in load(MAPPING)[kind]]
    return result


class RomeMigration(unittest.TestCase):
    def test_exact_inverse_every_cell_object_and_top_metadata(self):
        for kind in ['exercises','solutions']:
            old,new=pair(kind)
            self.assertEqual(len(old['cells']),69)
            self.assertEqual(inverse(kind,new),old)
            for row in load(MAPPING)[kind]:
                self.assertEqual(new['cells'][row['canonical_index']]['id'],row['id'])
                self.assertEqual(sha(json.dumps(old['cells'][row['original_index']],sort_keys=True,ensure_ascii=False).encode()),row['original_cell_sha256'])

    def test_all_source_inputs_and_archive_recovery(self):
        freeze=load(E/'source_freeze.json')
        with tarfile.open(E/'original_preimages.tar.gz') as archive:
            self.assertEqual(set(archive.getnames()),set(freeze))
            for relative,pin in freeze.items():
                actual=(LEGACY/relative).read_bytes()
                self.assertEqual(sha(actual),pin['sha256']);self.assertEqual(len(actual),pin['bytes'])
                self.assertEqual(archive.extractfile(relative).read(),actual)

    def test_all_original_outputs_counts_execution_and_metadata(self):
        for kind in ['exercises','solutions']:
            old,new=pair(kind)
            cs=[c for c in new['cells'] if c['cell_type']=='code']
            self.assertEqual(len(cs),32)
            self.assertEqual(sum(bool(c['outputs']) for c in cs),19 if kind=='solutions' else 0)
            self.assertEqual(sum('image/png' in o.get('data',{}) for c in cs for o in c['outputs']),3 if kind=='solutions' else 0)
            before={c['id']:c for c in old['cells']}
            for cell in cs:
                self.assertEqual({k:v for k,v in cell.items() if k!='source'},{k:v for k,v in before[cell['id']].items() if k!='source'})
            self.assertEqual({k:v for k,v in new.items() if k!='cells'},{k:v for k,v in old.items() if k!='cells'})

    def test_unchanged_code_science_stubs_tests_and_all_inline_definitions(self):
        for kind in ['exercises','solutions']:
            old,new=pair(kind);a=[c for c in old['cells'] if c['cell_type']=='code'];b=[c for c in new['cells'] if c['cell_type']=='code']
            for i,(x,y) in enumerate(zip(a,b,strict=True)):
                if i:self.assertEqual(ast.dump(ast.parse(source(x))),ast.dump(ast.parse(source(y))),(kind,i))
            self.assertEqual(sum(source(c).count('raise NotImplementedError') for c in a),sum(source(c).count('raise NotImplementedError') for c in b))
        old,new=pair('solutions')
        inline={n.name:ast.dump(n) for c in old['cells'] if c['cell_type']=='code' for n in ast.parse(source(c)).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        module={n.name:ast.dump(n) for n in ast.parse((NEW/'solutions.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='initialize_notebook_plumbing'}
        self.assertEqual(module,inline)
        self.assertEqual(len(inline),28)

    def test_import_initialization_guard_and_explicit_visible_binding(self):
        tree=ast.parse((NEW/'solutions.py').read_text())
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef)):continue
            if isinstance(n,ast.Assign):
                if any(isinstance(t,ast.Name) and t.id=='MAIN' for t in n.targets):continue
                ast.literal_eval(n.value);continue
            self.assertIsInstance(n,ast.If)
            self.assertIsInstance(n.test,ast.Name)
            self.assertEqual(n.test.id,'MAIN')
        init=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initialize_notebook_plumbing')
        assigned={t.id for n in ast.walk(init) if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
        declared=set(next(n for n in init.body if isinstance(n,ast.Global)).names)
        self.assertLessEqual(assigned,declared)
        self.assertIn('t.manual_seed(0)',ast.unparse(init));self.assertIn('device',declared)
        for p in [NEW/'solutions.py',NEW/'tests.py',PAGE.with_suffix('.py')]:compile(p.read_text(),str(p),'exec')
        self.assertNotIn('import_module(', '\n'.join(ast.unparse(n) for n in ast.parse((NEW/'tests.py').read_text()).body if not isinstance(n,ast.FunctionDef)))

    def test_forecasts_precede_every_reveal_and_thirteen_real_exercises(self):
        for kind in ['exercises','solutions']:
            _,b=pair(kind);cells=b['cells'];byid={c['id']:i for i,c in enumerate(cells)}
            self.assertLess(byid['rome96-forecast-open'],byid['rome-000'])
            for number,original_index in enumerate([5,9,13,19,23,28,32,37,41,45,50,54,58],1):
                self.assertLess(byid[f'rome96-forecast-e{number}'],byid[f'rome-{original_index:03d}'])
                c=cells[byid[f'rome-{original_index:03d}']];text=source(c)
                self.assertIn(f'### Exercise {number} -',text)
                self.assertRegex(text,r'```\nDifficulty:.*\nImportance:.*\nSuggested time:.*\n```')
                self.assertIn('Confidence (0–100%): ______',source(cells[byid[f'rome96-forecast-e{number}']]))
            self.assertEqual(sum('## Forecast before inspection —' in source(c) for c in cells),16)
            self.assertEqual(sum('## Reflect after inspection —' in source(c) for c in cells),15)
            self.assertLess(byid['rome96-forecast-signature'],byid['rome-062'])
            self.assertLess(byid['rome96-forecast-play'],byid['rome-065'])

    def test_all_historical_helpers_assets_and_local_media_route(self):
        for p in OLD.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.ipynb']:
                self.assertEqual(p.read_bytes(),(NEW/('historical_'+str(p.relative_to(OLD)))).read_bytes())
        for name in ['rome_model_editing_signature.png','rome_rank_one_toy_oracle.png']:
            self.assertEqual((LEGACY/CH/'instructions/assets'/name).read_bytes(),(NEW/'media'/name).read_bytes())
        paths=[PAGE]+[NEW/f'{STEM}_{k}.ipynb' for k in ['exercises','solutions']]
        for path in paths:
            text=path.read_text() if path.suffix=='.md' else '\n'.join(source(c) for c in load(path)['cells'] if c['cell_type']=='markdown')
            links=[link for link in re.findall(r'<img src="([^"]+)"',text) if link.endswith('/rome_model_editing_signature.png')]
            self.assertEqual(len(links),1)
            self.assertEqual((path.parent/links[0]).resolve(),(NEW/'media/rome_model_editing_signature.png').resolve())

    def test_case_schema_and_scope_limits_no_new_science(self):
        spec=importlib.util.spec_from_file_location('rome_guided_schema',LEGACY/'mastery_hub/units/extension_predictions/lesson.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        case=load(E/'proposed_case.json');self.assertEqual(module.validate_case(case),case)
        self.assertEqual(case['evidence_class'],'recorded_real_run');self.assertEqual(case['data_role'],'historical')
        self.assertFalse(any(k in case for k in ['answer','prediction','observation','observed_result']))
        page=PAGE.read_text()
        for phrase in ['does not implement MEMIT','official ROME protocol parity','Human learner walkthrough','scoring-only','negative localization']:
            self.assertIn(phrase,page)
        self.assertFalse((NEW/'verification_report.json').exists());self.assertFalse((NEW/'artifacts.lock.yml').exists())
        for kind in ['exercises','solutions']:
            _,b=pair(kind)
            self.assertTrue(all('id' in c for c in b['cells']))
            self.assertEqual(len({c['id'] for c in b['cells']}),len(b['cells']))

    def test_exact_inverse_rejects_mutated_source_and_preserves_failures(self):
        _,b=pair('solutions');q=deepcopy(b)
        c=next(c for c in q['cells'] if c['id']=='rome-001');c['source'][0]='import changed\n'
        with self.assertRaisesRegex(ValueError,'canonical source changed'):inverse('solutions',q)
        self.assertNotEqual(inverse('solutions',b),inverse('exercises',pair('exercises')[1]))


if __name__=='__main__':unittest.main()
