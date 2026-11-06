"""Static existing5.9 preservation checks. No taught module or torch import."""
from pathlib import Path
from html.parser import HTMLParser
import ast
import builtins
import hashlib
import importlib.util
import json
import os
import re
import symtable
import sys
import unittest

ROOT=Path(os.environ.get('ARENA_MIGRATION_ROOT',Path(__file__).resolve().parents[1]))
LEGACY_ROOT=Path(os.environ.get('ARENA_LEGACY_ROOT',ROOT))
CH='chapter5_modern_architectures'
OLD=LEGACY_ROOT/CH/'exercises/part6_multimodal_embedding_function_models'
NEW=ROOT/CH/'exercises/part9_specialist_models'
STEM='5.9_Embedding_Function-Calling_and_Specialist_Models'
EV=ROOT/'docs/mastery/evidence/embedding_5_9_migration_20261020'
PAGE=ROOT/CH/'instructions/pages/09_[5.9]_Embedding_Function-Calling_and_Specialist_Models.md'
MASTER=ROOT/'infrastructure/chapters'/CH/'master_5_9.py'
source=lambda c:''.join(c['source'])
read=lambda p:json.loads(p.read_text())
code=lambda b:[c for c in b['cells'] if c['cell_type']=='code']


def books(kind):
    return (read(OLD/f'5.6_Multimodal_Embedding_and_Function_Calling_Models_{kind}.ipynb'),
            read(NEW/f'{STEM}_{kind}.ipynb'))


def inverse_setup(text):
    return (text.removeprefix('MAIN = __name__ == "__main__"\n')
        .replace('section = "part9_specialist_models"\nlegacy_section = "part6_multimodal_embedding_function_models"',
                 'section = "part6_multimodal_embedding_function_models"')
        .replace('legacy_section_dir = exercises_dir / legacy_section\nsection_dir = legacy_section_dir\ncanonical_section_dir = exercises_dir / section',
                 'section_dir = exercises_dir / section')
        .replace('assets_dir = canonical_section_dir / "media"','assets_dir = root_dir / chapter / "instructions" / "assets"')
        .replace('import part9_specialist_models.tests as tests','import part6_multimodal_embedding_function_models.tests as tests'))


def md_delta(text,first=False):
    if first:text=text.split('\n',1)[1].lstrip('\n')
    text=text.replace('## Learning objectives','## What you will implement')
    text=text.replace('<summary>Solution</summary>','<summary>Historical worked solution</summary>')
    text=text.replace('![Embedding router signature result](../../instructions/assets/embedding_tool_router_signature.png)',
        '<img src="../../../chapter5_modern_architectures/exercises/part9_specialist_models/media/embedding_tool_router_signature.png" alt="Embedding router signature result" width="1100">')
    text=re.sub(r'Difficulty: ([1-5])/5',lambda m:'Difficulty: '+'🔴'*int(m[1])+'⚪'*(5-int(m[1])),text)
    return re.sub(r'Importance: ([1-5])/5',lambda m:'Importance: '+'🔵'*int(m[1])+'⚪'*(5-int(m[1])),text)


class EmbeddingMigrationTests(unittest.TestCase):
    def test_all_eleven_code_cells_exact_inverse_and_all_inline_definitions(self):
        for kind in ['exercises','solutions']:
            old,new=books(kind)
            self.assertEqual(len(old['cells']),29)
            self.assertEqual(len(code(new)),11)
            for i,(a,b) in enumerate(zip(code(old),code(new),strict=True)):
                before,after=source(a),source(b)
                if i==0:after=inverse_setup(after)
                after=after.replace('from chapter5_modern_architectures.exercises.part6_multimodal_embedding_function_models.solutions import run_gpu_test as run',
                                    'from part6_multimodal_embedding_function_models.solutions import run_gpu_test as run')
                self.assertEqual(ast.dump(ast.parse(before)),ast.dump(ast.parse(after)),(kind,i))
        old,_=books('solutions')
        definitions={n.name:ast.dump(n) for n in ast.parse('\n'.join(source(c) for c in code(old))).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        actual={n.name:ast.dump(n) for n in ast.parse((NEW/'solutions.py').read_text().replace(
            'from chapter5_modern_architectures.exercises.part6_multimodal_embedding_function_models.solutions import run_gpu_test as run',
            'from part6_multimodal_embedding_function_models.solutions import run_gpu_test as run')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(definitions,actual)
        self.assertEqual(sum(k.startswith('FunctionDef') for k in definitions.values()),11)

    def test_complete_original_cell_objects_and_top_metadata(self):
        for kind in ['exercises','solutions']:
            old,new=books(kind)
            self.assertEqual({k:v for k,v in old.items() if k!='cells'},{k:v for k,v in new.items() if k!='cells'})
            by_id={c['id']:c for c in new['cells']}
            self.assertEqual(len(by_id),len(new['cells']))
            for c in old['cells']:
                self.assertEqual({k:v for k,v in c.items() if k!='source'},
                                 {k:v for k,v in by_id[c['id']].items() if k!='source'})
            self.assertEqual(sum(bool(c.get('outputs')) for c in code(new)),0 if kind=='exercises' else 9)
            self.assertEqual(sum('image/png' in o.get('data',{}) for c in code(new) for o in c.get('outputs',[])),0)

    def test_every_original_prose_body_and_student_stubs_and_own_function_tests(self):
        for kind in ['exercises','solutions']:
            old,new=books(kind);norm=lambda x:' '.join(x.split())
            for i,c in enumerate(old['cells']):
                if c['cell_type']=='markdown':
                    matches=[n for n in new['cells'] if n['cell_type']=='markdown' and norm(source(n))==norm(md_delta(source(c),i==0))]
                    self.assertEqual(len(matches),1,(kind,i))
            calls=lambda b:[ast.dump(n) for c in code(b) for n in ast.walk(ast.parse(source(c))) if isinstance(n,ast.Call)
                             and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='tests']
            self.assertEqual(calls(old),calls(new));self.assertEqual(len(calls(new)),4)
        old,new=books('exercises')
        self.assertEqual(sum(source(c).count('raise NotImplementedError') for c in code(old)),7)
        self.assertEqual(sum(source(c).count('raise NotImplementedError') for c in code(new)),7)
        play=source(code(new)[7]);self.assertIn('PLAY_MIN_MARGIN = 0.15',play);self.assertIn('PLAY_ALLOWED_TOOLS = t.tensor([True, True, False])',play)

    def test_import_guard_optional_imports_and_function_global_dependencies(self):
        text=(NEW/'solutions.py').read_text();tree=ast.parse(text)
        available=set(dir(builtins))|{'__name__','__file__','__package__'}
        def imports(nodes):
            for n in nodes:
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)):available.add(n.name)
                elif isinstance(n,(ast.Import,ast.ImportFrom)):available.update(a.asname or a.name.split('.')[0] for a in n.names)
                elif isinstance(n,ast.Assign):
                    ast.literal_eval(n.value) if not any(isinstance(t,ast.Name) and t.id=='MAIN' for t in n.targets) else None
                    available.update(t.id for t in n.targets if isinstance(t,ast.Name))
                elif isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='MAIN':imports(n.orelse)
                else:self.fail(ast.unparse(n))
        imports(tree.body)
        def walk_table(table):
            if table.get_type()=='function':
                missing={s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced() and s.get_name() not in available}
                self.assertFalse(missing,(table.get_name(),missing))
            for child in table.get_children():walk_table(child)
        walk_table(symtable.symtable(text,'<static imported lesson>','exec'))
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):self.assertFalse((n.module or '').startswith('arena_ext'))
        top_imports=[ast.unparse(n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('sentence_transformers' in x or 'transformers' in x for x in top_imports))
        bge=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_real_bge_on_cpu')
        self.assertTrue(any(isinstance(n,ast.ImportFrom) and n.module=='sentence_transformers' for n in bge.body))
        self.assertIn('import_module("chapter5_modern_architectures.exercises.part6_multimodal_embedding_function_models.tests")',(NEW/'tests.py').read_text())
        for n in tree.body:
            if isinstance(n,ast.FunctionDef):
                for value in [*n.decorator_list,*n.args.defaults,*(x for x in n.args.kw_defaults if x is not None)]:
                    self.assertFalse(any(isinstance(x,ast.Call) for x in ast.walk(value)))
        for path in [NEW/'solutions.py',NEW/'tests.py',NEW/'utils.py']:
            compile(path.read_text(),str(path),'exec')

    def test_legacy_reports_assets_and_native_report_identity(self):
        for path in OLD.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.ipynb':
                self.assertEqual(path.read_bytes(),(NEW/('historical_'+str(path.relative_to(OLD)))).read_bytes())
        for name in ['embedding_tool_router_signature.png','embedding_function_calling_ladder.svg']:
            self.assertEqual((LEGACY_ROOT/CH/'instructions/assets'/name).read_bytes(),(NEW/'media'/name).read_bytes())
        self.assertFalse((NEW/'verification_report.json').exists());self.assertFalse((NEW/'artifacts.lock.yml').exists())
        setup=source(code(books('solutions')[1])[0]);last=source(code(books('solutions')[1])[-1])
        self.assertIn('section_dir = legacy_section_dir',setup)
        self.assertIn('verification_report_path = section_dir / "verification_report.json"',last)
        self.assertEqual((OLD/'utils.py').read_bytes(),(NEW/'utils.py').read_bytes())

    def test_media_paths_exact_canonical_and_pure_case_schema(self):
        class Images(HTMLParser):
            def __init__(self):super().__init__();self.sources=[]
            def handle_starttag(self,tag,attrs):
                if tag=='img':self.sources.extend(v for k,v in attrs if k=='src')
        for path in [PAGE,NEW/f'{STEM}_exercises.ipynb',NEW/f'{STEM}_solutions.ipynb']:
            text=path.read_text() if path.suffix=='.md' else '\n'.join(source(c) for c in read(path)['cells'] if c['cell_type']=='markdown')
            images=Images();images.feed(text)
            local=[x for x in images.sources if not x.startswith(('http:','https:','data:'))]
            self.assertEqual(len(local),1)
            self.assertEqual((path.parent/local[0]).resolve(),(NEW/'media/embedding_tool_router_signature.png').resolve())
            self.assertTrue((path.parent/local[0]).is_file())
        helper=LEGACY_ROOT/'mastery_hub/units/extension_predictions/lesson.py'
        spec=importlib.util.spec_from_file_location('embedding_pure_case_schema',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        case=read(EV/'proposed_case.json');self.assertEqual(module.validate_case(case),case)
        self.assertFalse(any(k in case for k in ['answer','prediction','observation','observed_result']))
        self.assertIn('12 tool-required',case['config_fields']['metric_and_selection'])

    def test_frozen_original_sources_and_no_scope_overclaim(self):
        for path,row in read(EV/'source_freeze.json').items():
            data=(LEGACY_ROOT/path).read_bytes();self.assertEqual(hashlib.sha256(data).hexdigest(),row['sha256']);self.assertEqual(len(data),row['bytes'])
        text=PAGE.read_text()
        for phrase in ['EmbeddingGemma internal geometry','FunctionGemma internal representation/probing/steering','MedGemma-style domain-specialist evaluation','no embedded PNG']:
            self.assertIn(phrase,text)
        self.assertIn('## Content & Learning Objectives',text)
        self.assertEqual(text.count('## Forecast before running — EMB-'),8)
        self.assertEqual(text.count('## Reflect after inspection — EMB-'),8)


if __name__=='__main__':unittest.main()
