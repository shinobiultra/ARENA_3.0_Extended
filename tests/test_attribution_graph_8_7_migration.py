"""Static canonical-8.7 preservation checks; no taught imports or computation."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import unittest

ROOT = Path(os.environ.get('ARENA_MIGRATION_ROOT', Path(__file__).resolve().parents[1]))
LEGACY_ROOT = Path(os.environ.get('ARENA_LEGACY_ROOT', ROOT))
OLD = LEGACY_ROOT / 'chapter8_automated_circuits/exercises/part4_circuit_tracing_attribution_graphs'
NEW = ROOT / 'chapter8_automated_circuits/exercises/part7_attribution_graphs'
STEM = '8.7_Circuit_Tracing_with_Attribution_Graphs'
MASTER = ROOT / 'infrastructure/chapters/chapter8_automated_circuits/master_8_7.py'
PAGE = ROOT / 'chapter8_automated_circuits/instructions/pages/07_[8.7]_Circuit_Tracing_with_Attribution_Graphs.md'


def read(path):
    return json.loads(path.read_text())


def source(cell):
    return ''.join(cell['source'])


def code(nb):
    return [c for c in nb['cells'] if c['cell_type'] == 'code']


def setup(text):
    return (text.replace('section = "part4_circuit_tracing_attribution_graphs"',
                         'section = "part7_attribution_graphs"\nlegacy_section = "part4_circuit_tracing_attribution_graphs"')
            .replace('section_dir = exercises_dir / section',
                     'section_dir = exercises_dir / legacy_section\ncanonical_section_dir = exercises_dir / section')
            .replace('assets_dir = root_dir / chapter / "instructions" / "assets"',
                     'assets_dir = canonical_section_dir / "media"'))


def notebook_links(text):
    return (text.replace('./8.4_Circuit_Tracing_with_Attribution_Graphs_exercises.ipynb',
                         '../../../chapter8_automated_circuits/exercises/part7_attribution_graphs/8.7_Circuit_Tracing_with_Attribution_Graphs_exercises.ipynb')
            .replace('./8.4_Circuit_Tracing_with_Attribution_Graphs_solutions.ipynb',
                     '../../../chapter8_automated_circuits/exercises/part7_attribution_graphs/8.7_Circuit_Tracing_with_Attribution_Graphs_solutions.ipynb')
            .replace('8.7_Circuit_Tracing_with_Attribution_Graphs_exercises.ipynb) | [solutions]',
                     '8.7_Circuit_Tracing_with_Attribution_Graphs_exercises.ipynb) |\n> [solutions]'))


class AttributionGraphMigrationTests(unittest.TestCase):
    def test_full_original_code_objects_outputs_counts_metadata_and_ids(self):
        for kind in ['exercises', 'solutions']:
            old = read(next(OLD.glob('*_' + kind + '.ipynb')))
            new = read(NEW / (STEM + '_' + kind + '.ipynb'))
            self.assertEqual(len(old['cells']), 34)
            self.assertEqual(len(code(old)), 11)
            self.assertEqual(len(code(new)), 11)
            self.assertEqual(old['metadata'], new['metadata'])
            self.assertEqual(old['nbformat'], new['nbformat'])
            self.assertEqual(old['nbformat_minor'], new['nbformat_minor'])
            for i, (a, b) in enumerate(zip(code(old), code(new), strict=True)):
                expected = setup(source(a)) if i == 0 else source(a)
                self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(source(b))), (kind, i))
                inverse = copy.deepcopy(b)
                inverse['source'] = copy.deepcopy(a['source'])
                self.assertEqual(inverse, a, (kind, i))
                if i != 0:
                    self.assertEqual(source(a), source(b), (kind, i))
            expected_outputs = 9 if kind == 'solutions' else 0
            expected_png = 3 if kind == 'solutions' else 0
            self.assertEqual(sum(bool(c.get('outputs')) for c in code(new)), expected_outputs)
            self.assertEqual(sum('image/png' in o.get('data', {}) for c in code(new) for o in c.get('outputs', [])), expected_png)

    def test_full_student_and_solution_prose_and_dropdown_bodies(self):
        norm = lambda x: re.sub(r'\s+', ' ', x).strip()
        for kind in ['exercises', 'solutions']:
            old = read(next(OLD.glob('*_' + kind + '.ipynb')))
            new = read(NEW / (STEM + '_' + kind + '.ipynb'))
            joined = norm('\n'.join(source(c) for c in new['cells'] if c['cell_type'] == 'markdown'))
            for i, cell in enumerate(old['cells']):
                if cell['cell_type'] != 'markdown':
                    continue
                text = source(cell).replace('<summary>Solution</summary>', '<summary>Historical worked solution</summary>')
                text = text.replace('## Learning Objectives', '## What you will implement')
                text = notebook_links(text)
                if i == 0:
                    text = text.split('\n', 1)[1]
                if text.startswith('## Exercise -'):
                    title, body = text.split('\n', 1)
                    self.assertIn(norm(title.replace('## Exercise', '### Exercise', 1)), joined)
                    text = body.split('minutes on this exercise.', 1)[1].strip()
                    self.assertIn('Difficulty: 🔴🔴🔴⚪⚪', joined)
                    self.assertIn('Importance: 🔵🔵🔵🔵⚪', joined)
                self.assertIn(norm(text), joined, (kind, i))
                matching = [c for c in new['cells'] if c.get('id') == cell.get('id')]
                self.assertEqual(len(matching), 1)
                self.assertEqual(matching[0].get('metadata'), cell.get('metadata'))
            self.assertIn('## Content & Learning Objectives', joined)
            self.assertEqual(joined.count('Forecast: ______'), 9)
            self.assertEqual(joined.count('Observed result: ______'), 9)
            self.assertIn('released-transcoder', joined.lower())
            self.assertIn('frozen-attention', joined.lower())
            self.assertIn('supernode', joined.lower())
            self.assertIn('error nodes', joined.lower())

    def test_all_original_test_calls_and_student_arguments_exact(self):
        def calls(nb):
            return [ast.dump(n) for c in code(nb) for n in ast.walk(ast.parse(source(c)))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == 'tests']
        for kind in ['exercises', 'solutions']:
            old = read(next(OLD.glob('*_' + kind + '.ipynb')))
            new = read(NEW / (STEM + '_' + kind + '.ipynb'))
            self.assertEqual(calls(old), calls(new))
            self.assertEqual(len(calls(new)), 7)
        student = code(read(NEW / (STEM + '_exercises.ipynb')))
        self.assertIn('threshold = 0.10\nig_steps = 4\nrandom_seed = 3', source(student[-2]))

    def test_inline_methods_and_guarded_generated_module(self):
        original = ast.parse('\n'.join(source(c) for c in code(read(next(OLD.glob('*solutions.ipynb'))))))
        text = (NEW / 'solutions.py').read_text()
        module = ast.parse(text)
        definitions = {n.name: n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for node in original.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                self.assertEqual(ast.dump(node), ast.dump(definitions[node.name]), node.name)
        forbidden = {'mkdir', 'append', 'read_text', 'savefig', 'show', 'manual_seed', 'seed', 'set_grad_enabled'}
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'MAIN':
                continue
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    if isinstance(call.func, ast.Name):
                        self.assertNotIn(call.func.id, definitions)
                    if isinstance(call.func, ast.Attribute):
                        self.assertNotIn(call.func.attr, forbidden)
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('arena_ext'))
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name.startswith('arena_ext') for n in node.names))
        self.assertIn('root_dir = Path(__file__).resolve().parents[3]', text)
        self.assertIn('part4_circuit_tracing_attribution_graphs.solutions', source(code(read(NEW / (STEM + '_solutions.ipynb')))[-1]))
        compile(text, '<canonical module>', 'exec')
        compile(MASTER.read_text(), '<canonical master>', 'exec')
        self.assertTrue(PAGE.is_file())
        for kind in ['exercises', 'solutions']:
            for c in code(read(NEW / (STEM + '_' + kind + '.ipynb'))):
                compile(source(c), '<canonical notebook cell>', 'exec')

    def test_historical_artifacts_and_full_asset_bytes(self):
        for name in ['artifacts.lock.yml', 'verification_report.json', 'verification_report.schema.json']:
            self.assertFalse((NEW / name).exists())
            self.assertEqual((NEW / ('historical_' + name)).read_bytes(), (OLD / name).read_bytes())
        for name in ['tests.py', 'utils.py', 'expected_outputs/README.md', 'expected_outputs/reference_metrics.json', 'expected_outputs/smoke_test.json']:
            self.assertEqual((NEW / name).read_bytes(), (OLD / name).read_bytes())
        assets = LEGACY_ROOT / 'chapter8_automated_circuits/instructions/assets'
        for path in assets.glob('circuit_tracing_attribution_graphs*'):
            self.assertEqual((NEW / 'media' / path.name).read_bytes(), path.read_bytes())
        self.assertEqual(len(list((NEW / 'media').glob('attribution_graph_legacy_cell_26_output_*.png'))), 3)


    def test_canonical_local_notebook_routes_resolve_from_every_surface(self):
        paths = [MASTER, MASTER.with_suffix('.ipynb'), PAGE,
                 NEW / (STEM + '_exercises.ipynb'), NEW / (STEM + '_solutions.ipynb')]
        total = 0
        for path in paths:
            text = ('\n'.join(source(c) for c in read(path)['cells'] if c['cell_type'] == 'markdown')
                    if path.suffix == '.ipynb' else path.read_text())
            links = re.findall(r'\[[^\]]*\]\(([^)]+\.ipynb(?:#[^)]*)?)\)', text)
            workbook_url = '../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-8-7'
            local_urls = [url for url in links if '://' not in url]
            workbook_links = [url for url in local_urls if url == workbook_url]
            expected_workbook_count = 1 if path.parent == NEW and path.suffix == '.ipynb' else 0
            self.assertEqual(len(workbook_links), expected_workbook_count, path)
            for url in workbook_links:
                target = (path.parent / url.split('#', 1)[0]).resolve()
                self.assertEqual(target, (ROOT / 'mastery_hub/units/extension_predictions/exercises.ipynb').resolve())
                self.assertEqual(url.split('#', 1)[1], 'section-8-7')
                self.assertTrue(target.is_file(), (path, url))
            expected_lesson_urls = {
                f'../../../{NEW.relative_to(ROOT).as_posix()}/{STEM}_{kind}.ipynb'
                for kind in ['exercises', 'solutions']
            }
            lesson_links = [url for url in local_urls if url != workbook_url]
            for url in lesson_links:
                self.assertIn(url, expected_lesson_urls, (path, url))
            local = [url.split('#', 1)[0] for url in lesson_links]
            self.assertEqual(len(local), 2, path)
            for url in local:
                target = (path.parent / url).resolve()
                self.assertTrue(target.is_file(), (path, url))
                self.assertEqual(target.parent, NEW.resolve())
                self.assertIn(target.name, {STEM + '_exercises.ipynb', STEM + '_solutions.ipynb'})
                self.assertNotIn('8.4_Circuit_Tracing', url)
                total += 1
        self.assertEqual(total, 10)


if __name__ == '__main__':
    unittest.main()
