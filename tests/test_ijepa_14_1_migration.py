"""Source-only canonical 14.1 regression tests; never import the taught JEPA module or torch."""
from __future__ import annotations

import ast
import base64
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import re
import symtable
import sys
import unittest

ROOT = Path(os.environ.get('ARENA_MIGRATION_ROOT', Path(__file__).resolve().parents[1]))
LEGACY_ROOT = Path(os.environ.get('ARENA_LEGACY_ROOT', ROOT))
CHAPTER = 'chapter14_jepa_world_models'
OLD = LEGACY_ROOT / CHAPTER / 'exercises/part2_ijepa_from_scratch'
NEW = ROOT / CHAPTER / 'exercises/part1_ijepa_from_scratch'
STEM = '14.1_I-JEPA_from_Scratch'
EVIDENCE = ROOT / 'docs/mastery/evidence/ijepa_14_1_migration_20261020'
MASTER = ROOT / 'infrastructure/chapters' / CHAPTER / 'master_14_1'
PAGE = ROOT / CHAPTER / 'instructions/pages/01_[14.1]_I-JEPA_from_Scratch'
CORE = {
    '_draw_scene', 'build_scene_dataset', 'patchify', 'sample_block_masks',
    'object_target_masks', 'gather_masked_tokens', 'PatchTransformerEncoder',
    'LatentPredictor', 'normalized_prediction_loss', 'update_target_encoder',
    'IJEPAModel', 'collapse_diagnostics', 'fit_ridge_classifier', '_probe_accuracy',
    'evaluate_probes', 'evaluate_transfer_probes', 'cyclic_object_mismatch_indices',
    'replace_scene_keys', 'erase_scene_keys', 'evaluate_prediction', '_train_model',
    '_extract_features', 'run_experiment', 'validate_run', 'run_seed_suite',
}
RECORDS = {
    'SceneBatch', 'CollapseDiagnostics', 'ProbeMetrics', 'PredictionMetrics',
    'TrainingTrace', 'IJEPARun', '_report_metrics',
}


def read(path):
    return json.loads(path.read_text())


def source(cell):
    return ''.join(cell['source'])


def codes(nb):
    return [c for c in nb['cells'] if c['cell_type'] == 'code']


def old_book(kind):
    return read(OLD / f'14.2_I-JEPA_from_Scratch_{kind}.ipynb')


def new_book(kind):
    return read(NEW / f'{STEM}_{kind}.ipynb')


def ast_equal(a, b):
    return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def exact_spans(text, names):
    lines = text.splitlines(keepends=True)
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            result[node.name] = ''.join(lines[start - 1:node.end_lineno])
    return result


def inverse_setup(text):
    text = text.removeprefix('MAIN = __name__ == "__main__"\n\n')
    begin = text.index('root_dir = next(')
    end = text.index('from part1_ijepa_from_scratch import ijepa_plumbing as reference')
    original = '''SECTION_DIR = Path.cwd()
if not (SECTION_DIR / "solutions.py").exists():
    SECTION_DIR = Path("chapter14_jepa_world_models/exercises/part2_ijepa_from_scratch").resolve()
if str(SECTION_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SECTION_DIR.parent))

'''
    text = text[:begin] + original + text[end:]
    return (text.replace('from part1_ijepa_from_scratch import ijepa_plumbing as reference',
                         'from part2_ijepa_from_scratch import solutions as reference')
            .replace('from part1_ijepa_from_scratch import tests',
                     'from part2_ijepa_from_scratch import tests')
            .replace('from part1_ijepa_from_scratch.ijepa_plumbing import (',
                     'from part2_ijepa_from_scratch.solutions import ('))


def markdown_delta(text, first=False):
    if first:
        text = text.split('\n', 1)[1].lstrip('\n')
    text = text.replace('## Learning Objectives', '## What you will implement')
    text = text.replace('<summary>Solution</summary>', '<summary>Historical worked solution</summary>')
    text = text.replace('![I-JEPA signature result](../../instructions/assets/ijepa_signature_result.png)', '<img src="media/ijepa_signature_result.png" alt="I-JEPA signature result" width="1100">')
    text = re.sub(
        r'> \*\*Difficulty:\*\* ([1-5])/5 &nbsp; \| &nbsp; \*\*Importance:\*\* ([1-5])/5 &nbsp; \| &nbsp; \*\*Time:\*\* ([^\n]+)',
        lambda m: '> ```yaml\n> Difficulty: ' + '🔴' * int(m[1]) + '⚪' * (5-int(m[1]))
        + '\n> Importance: ' + '🔵' * int(m[2]) + '⚪' * (5-int(m[2]))
        + '\n> Time: ' + m[3] + '\n> ```', text,
    )
    return re.sub(r'### Exercise (\d+):', r'### Exercise - \1:', text)


class LocalMediaImages(HTMLParser):
    """Read actual HTML image attributes without executing rendered content."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sources=[]

    def handle_starttag(self,tag,attrs):
        if tag=='img':
            values=[value for name,value in attrs if name=='src']
            assert len(values)==1 and isinstance(values[0],str) and values[0], 'Image requires one nonempty src'
            self.sources.append(values[0])

    handle_startendtag=handle_starttag


def resolved_local_media(document,document_path,media_dir,expected_hashes):
    """Require every local image to resolve to one exact, preserved canonical asset."""
    parser=LocalMediaImages()
    parser.feed(document)
    parser.close()
    expected={path.resolve():checksum for path,checksum in expected_hashes.items()}
    resolved=[]
    for src in parser.sources:
        url=urlsplit(src)
        if url.scheme in {'https','http','data'}:
            continue
        assert not url.scheme and not url.netloc and not url.query and not url.fragment, src
        assert '\\' not in src and '\x00' not in src, src
        path=(document_path.parent/unquote(url.path)).resolve()
        assert path.is_relative_to(media_dir.resolve()), ('Image outside canonical media',src)
        assert path in expected, ('Unknown/wrong canonical image',src)
        assert path.is_file(), ('Missing canonical image',src)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[path], ('Image bytes changed',src)
        resolved.append(path)
    assert len(resolved)==len(expected) and set(resolved)==set(expected), 'Missing/duplicate canonical images'
    return resolved


class IJEPAMigrationTests(unittest.TestCase):
    def test_all_original_code_and_exact_declared_inverse(self):
        for kind in ('exercises', 'solutions'):
            old = codes(old_book(kind))
            new = codes(new_book(kind))
            self.assertEqual(len(old), 28)
            self.assertEqual(len(new), 28)
            original_indices = [i for i,c in enumerate(old_book(kind)['cells']) if c['cell_type']=='code']
            for index, before, after in zip(original_indices, old, new, strict=True):
                a, b = source(before), source(after)
                if index == 1:
                    b = inverse_setup(b)
                elif index == 53:
                    tree = ast.parse(b)
                    plot = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='plot_signature_result']
                    self.assertEqual(len(plot), 1)
                    lines = b.splitlines(keepends=True)
                    b = ''.join(lines[:plot[0].lineno-1])
                elif index == 54:
                    b = b.replace('plot_signature_result(run)', 'reference.plot_signature_result(run)')
                self.assertEqual(a.strip(), b.strip(), (kind,index))
                self.assertTrue(ast_equal(a,b),(kind,index))

    def test_complete_metadata_ids_outputs_and_counts(self):
        for kind in ('exercises','solutions'):
            old,new=old_book(kind),new_book(kind)
            self.assertEqual(old['metadata'],new['metadata'])
            self.assertEqual({k:v for k,v in old.items() if k!='cells'},
                             {k:v for k,v in new.items() if k!='cells'})
            for a,b in zip(codes(old),codes(new),strict=True):
                self.assertEqual({k:v for k,v in a.items() if k!='source'},
                                 {k:v for k,v in b.items() if k!='source'})
            self.assertEqual(sum(bool(c.get('outputs')) for c in codes(new)),0 if kind=='exercises' else 14)
            self.assertEqual(sum('image/png' in o.get('data',{}) for c in codes(new) for o in c['outputs']),
                             0 if kind=='exercises' else 1)
            ids=[c['id'] for c in new['cells']]
            self.assertEqual(len(ids),len(set(ids)))

    def test_every_original_markdown_body_and_metadata(self):
        norm=lambda s: re.sub(r'\s+',' ',s).strip()
        for kind in ('exercises','solutions'):
            old,new=old_book(kind),new_book(kind)
            for index,a in enumerate(old['cells']):
                if a['cell_type']!='markdown':
                    continue
                expected=norm(markdown_delta(source(a),first=index==0))
                matches=[b for b in new['cells'] if b['cell_type']=='markdown' and norm(source(b))==expected]
                self.assertEqual(len(matches),1,(kind,index))
                self.assertEqual({k:v for k,v in a.items() if k!='source'},
                                 {k:v for k,v in matches[0].items() if k!='source'})
            self.assertTrue(any(source(c).splitlines()[0]=='## Content & Learning Objectives' for c in new['cells']))
            self.assertEqual(sum('## Forecast before running — IJ-' in source(c) for c in new['cells']),14)
            self.assertEqual(sum('## Reflect after inspection — IJ-' in source(c) for c in new['cells']),14)

    def test_student_nested_class_stubs_and_test_wiring(self):
        old,new=codes(old_book('exercises')),codes(new_book('exercises'))
        count=0
        for a,b in zip(old,new,strict=True):
            if 'NotImplementedError' in source(a):
                self.assertTrue(ast_equal(source(a),source(b)))
                count+=1
        self.assertEqual(count,12)
        calls=lambda text: [ast.dump(n) for n in ast.walk(ast.parse(text))
                           if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                           and isinstance(n.func.value,ast.Name) and n.func.value.id=='tests']
        original_calls=[x for c in old for x in calls(source(c))]
        canonical_calls=[x for c in new for x in calls(source(c))]
        self.assertEqual(original_calls,canonical_calls)
        self.assertEqual(len(original_calls),22)
        self.assertIn('MISMATCH_SHIFT = 2',source(new[-1]))
        self.assertIn('PLAY_SEED = 17',source(new[-1]))

    def test_plumbing_exact_spans_and_inline_plot_and_methods(self):
        original=(OLD/'solutions.py').read_text()
        local=(NEW/'ijepa_plumbing.py').read_text()
        self.assertEqual(exact_spans(original,RECORDS),exact_spans(local,RECORDS))
        definitions={n.name for n in ast.parse(local).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(definitions,RECORDS)
        solution=(NEW/'solutions.py').read_text()
        inline={n.name for n in ast.parse(solution).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertTrue(CORE<=inline,CORE-inline)
        self.assertEqual(exact_spans(original,{'plot_signature_result'}),
                         exact_spans(solution,{'plot_signature_result'}))
        for node in ast.walk(ast.parse(solution+'\n'+local)):
            if isinstance(node,ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('arena_ext'))

    def test_generated_python_has_no_unguarded_scientific_effects(self):
        tree=ast.parse((NEW/'solutions.py').read_text())
        def check(node):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                for decoration in node.decorator_list:
                    self.assertEqual(ast.unparse(decoration),'t.no_grad()')
                return
            if isinstance(node,ast.ClassDef):
                for statement in node.body:
                    if isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        check(statement)
                return
            if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='MAIN':
                self.assertTrue(all(isinstance(n,(ast.Import,ast.ImportFrom)) for n in node.orelse))
                return
            self.assertIsInstance(node,(ast.Import,ast.ImportFrom,ast.Assign))
            for call in ast.walk(node):
                if isinstance(call,ast.Call):
                    self.assertIsInstance(node,ast.Assign)
                    self.assertEqual(len(node.targets),1)
                    self.assertIsInstance(node.targets[0],ast.Name)
                    self.assertIn(node.targets[0].id,{'KEY_ROW_SLICE','KEY_COLUMN_SLICE'})
                    self.assertIsInstance(call.func,ast.Name)
                    self.assertEqual(call.func.id,'slice')
                    self.assertFalse(call.keywords)
                    self.assertTrue(all(isinstance(x,(ast.Constant,ast.Name,ast.BinOp,ast.Mult))
                                        for arg in call.args for x in ast.walk(arg)
                                        if not isinstance(x,ast.expr_context)))
        for node in tree.body:
            check(node)
        bridge=ast.parse((NEW/'tests.py').read_text())
        self.assertTrue(all(isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))
                            or isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) for n in bridge.body))
        self.assertIn('import_module("part2_ijepa_from_scratch.tests")',(NEW/'tests.py').read_text())
        for path in (NEW/'solutions.py',NEW/'ijepa_plumbing.py',NEW/'tests.py'):
            compile(path.read_text(),str(path),'exec')
        for kind in ('exercises','solutions'):
            for c in codes(new_book(kind)):
                compile(source(c),'<unexecuted I-JEPA cell>','exec')

    def test_imported_functions_have_module_global_dependencies(self):
        text=(NEW/'solutions.py').read_text()
        tree=ast.parse(text)
        available=set(dir(builtins)) | {'__name__','__file__','__package__'}

        def collect(nodes):
            for node in nodes:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                    available.add(node.name)
                elif isinstance(node,(ast.Import,ast.ImportFrom)):
                    available.update(a.asname or a.name.split('.')[0] for a in node.names)
                elif isinstance(node,ast.Assign):
                    available.update(n.id for target in node.targets for n in ast.walk(target)
                                     if isinstance(n,ast.Name))
                elif isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='MAIN':
                    collect(node.orelse)  # Imported module observes only this branch.
        collect(tree.body)
        self.assertTrue({'KEY_ROW_SLICE','KEY_COLUMN_SLICE'} <= available)
        for name in ('KEY_ROW_SLICE','KEY_COLUMN_SLICE'):
            assignments=[n for n in tree.body if isinstance(n,ast.Assign)
                         and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
            self.assertEqual(len(assignments),1,name)
            value=assignments[0].value
            self.assertIsInstance(value,ast.Call)
            self.assertEqual(ast.unparse(value.func),'slice')
        tables=symtable.symtable(text,'<source-only imported I-JEPA>','exec')

        def check(table):
            if table.get_type()=='function':
                missing={symbol.get_name() for symbol in table.get_symbols()
                         if symbol.is_global() and symbol.is_referenced()
                         and symbol.get_name() not in available}
                self.assertFalse(missing,(table.get_name(),missing))
            for child in table.get_children():
                check(child)
        check(tables)
        # Definition-time defaults/decorators must not depend on MAIN-only globals either.
        definition_nodes=[*tree.body, *(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body)]
        for node in definition_nodes:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                expressions=[*node.decorator_list,*node.args.defaults,
                             *(x for x in node.args.kw_defaults if x is not None)]
                for expression in expressions:
                    missing={n.id for n in ast.walk(expression) if isinstance(n,ast.Name)
                             and isinstance(n.ctx,ast.Load) and n.id not in available}
                    self.assertFalse(missing,(node.name,missing))

    def test_historical_files_and_both_png_identities(self):
        for p in OLD.iterdir():
            if p.is_file() and p.suffix!='.ipynb' and p.name!='build_notebooks.py':
                self.assertEqual(p.read_bytes(),(NEW/('historical_'+p.name)).read_bytes())
        for p in (OLD/'expected_outputs').rglob('*'):
            if p.is_file():
                self.assertEqual(p.read_bytes(),(NEW/'historical_expected_outputs'/p.relative_to(OLD/'expected_outputs')).read_bytes())
        for name in ('artifacts.lock.yml','verification_report.json','verification_report.schema.json'):
            self.assertFalse((NEW/name).exists())
        self.assertEqual((LEGACY_ROOT/CHAPTER/'instructions/assets/ijepa_signature_result.png').read_bytes(),
                         (NEW/'media/ijepa_signature_result.png').read_bytes())
        embedded=old_book('solutions')['cells'][54]['outputs'][4]['data']['image/png']
        self.assertEqual(base64.b64decode(''.join(embedded)),(NEW/'media/ijepa_historical_cell54_output4.png').read_bytes())

    def test_master_and_page_preserve_surface_and_core(self):
        master=MASTER.with_suffix('.py').read_text()
        page=Path(str(PAGE)+'.md').read_text()
        self.assertIn('### 1️⃣ Specify the image world',master)
        self.assertIn('> ##### Learning Objectives\n>\n> -',master)
        self.assertTrue(MASTER.with_suffix('.ipynb').exists())
        self.assertTrue(Path(str(PAGE)+'.py').exists())
        for name in CORE:
            self.assertRegex(page,rf'(?:def|class) {re.escape(name)}\b')
        self.assertIn('Historical worked solution',page)
        signature=(NEW/'media/ijepa_signature_result.png').resolve()
        legacy_signature=LEGACY_ROOT/CHAPTER/'instructions/assets/ijepa_signature_result.png'
        expected={signature:hashlib.sha256(legacy_signature.read_bytes()).hexdigest()}
        self.assertEqual(resolved_local_media(page,Path(str(PAGE)+'.md'),NEW/'media',expected),[signature])
        for kind in ('exercises','solutions'):
            notebook_path=NEW/f'{STEM}_{kind}.ipynb'
            markdown='\n'.join(source(c) for c in new_book(kind)['cells'] if c['cell_type']=='markdown')
            self.assertEqual(resolved_local_media(markdown,notebook_path,NEW/'media',expected),[signature])
        valid_src=str(signature.relative_to(ROOT))
        valid_src='../../../'+valid_src
        self.assertEqual(resolved_local_media(f'<img src="{valid_src}">',Path(str(PAGE)+'.md'),NEW/'media',expected),[signature])
        for bad_src in ('../../../'+str(legacy_signature.relative_to(ROOT)),
                        '../../../'+str((NEW/'media/unknown.png').relative_to(ROOT)),
                        '../../../'+str((NEW/'media/ijepa_historical_cell54_output4.png').relative_to(ROOT))):
            with self.assertRaises(AssertionError):
                resolved_local_media(f'<img src="{bad_src}">',Path(str(PAGE)+'.md'),NEW/'media',expected)
        with self.assertRaises(AssertionError):
            resolved_local_media('<img>',Path(str(PAGE)+'.md'),NEW/'media',expected)
        with self.assertRaises(AssertionError):
            resolved_local_media(f'<img src="{valid_src}">',Path(str(PAGE)+'.md'),NEW/'media',{signature:'0'*64})
        self.assertTrue(any('media/ijepa_signature_result.png' in source(c) for c in new_book('solutions')['cells']))

    def test_proposed_case_uses_durable_canonical_paths_and_fixed_protocol(self):
        case=read(EVIDENCE/'proposed_case.json')
        self.assertEqual(case['section'],'14.1')
        self.assertEqual(case['prediction_type'],'binary')
        for field in ('master_path','exercise_path','solutions_path'):
            self.assertTrue((ROOT/case[field]).exists(),field)
            self.assertNotIn('/tmp/',case[field])
        binding=read(EVIDENCE/'protocol_binding.json')
        self.assertFalse(binding['answer_computed'])
        self.assertEqual(binding['source_defaults'],
                         {'examples_per_combination':10,'epochs':30,'max_vram_gb':24.0})
        for name,row in binding['source_bindings'].items():
            text=(LEGACY_ROOT/row['source']).read_text()
            segment=''.join(text.splitlines(keepends=True)[row['start']-1:row['end']])
            self.assertEqual(hashlib.sha256(segment.encode()).hexdigest(),row['sha256'],name)
        self.assertFalse(any(k in case for k in ('answer','observed_value','correct_answer')))
        self.assertIn('position/ordinal-matched',case['config_fields']['metric_and_selection'])
        self.assertIn('120 evaluation scenes',case['config_fields']['metric_and_selection'])

    def test_actual_pure_case_schema_and_historical_figure_context(self):
        helper=LEGACY_ROOT/'mastery_hub/units/extension_predictions/lesson.py'
        text=helper.read_text()
        tree=ast.parse(text)
        # The shared worksheet helper is stdlib-only; this is not the taught JEPA module.
        for node in tree.body:
            if isinstance(node,ast.ImportFrom):
                self.assertIn(node.module,{'__future__','copy','pathlib','uuid'})
            elif isinstance(node,ast.Import):
                self.assertTrue({n.name for n in node.names} <= {'hashlib','json','math','os','re'})
        sys.dont_write_bytecode=True
        spec=importlib.util.spec_from_file_location('ijepa_actual_pure_case_schema',helper)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        case=read(EVIDENCE/'proposed_case.json')
        self.assertEqual(module.validate_case(case),case)
        self.assertIsInstance(case['config_fields']['durable_source_identity'],str)
        identity=read(EVIDENCE/'durable_source_identity.json')
        for path,checksum in identity.items():
            self.assertIn(path,case['config_fields']['durable_source_identity'])
            self.assertIn(checksum,case['config_fields']['durable_source_identity'])
        for kind in ('exercises','solutions'):
            nb=new_book(kind)
            image=[i for i,c in enumerate(nb['cells']) if '<img src="media/ijepa_signature_result.png"' in source(c)]
            self.assertEqual(len(image),1)
            context=nb['cells'][image[0]+1]
            self.assertEqual(context['cell_type'],'markdown')
            self.assertIn('Historical asset context.',source(context))
            self.assertIn('Both retained figures',source(context))
            self.assertIn("notebook's embedded PNG",source(context))
            self.assertIn('does not establish different numerical results',source(context))
            self.assertIn('key-swap bars disagree',source(context))
            self.assertIn('Use the exact retained per-seed table',source(context))
        page=Path(str(PAGE)+'.md').read_text()
        self.assertIn('Historical asset context.',page)

    def test_protected_source_snapshot_hashes(self):
        freeze=read(EVIDENCE/'source_freeze.json')
        for rel,record in freeze.items():
            data=(LEGACY_ROOT/rel).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),record['sha256'],rel)
            self.assertEqual(len(data),record['bytes'],rel)
        self.assertFalse(any(rel.startswith(('chapter0_','chapter1_','chapter2_','chapter3_','chapter4_'))
                             for rel in freeze))


if __name__=='__main__':
    unittest.main()
