"""Static preservation checks; deliberately never import/execute taught methods."""
import ast
import json
import os
import re
from pathlib import Path
import unittest

ROOT = Path(os.environ.get('ARENA_MIGRATION_ROOT', Path(__file__).resolve().parents[1]))
LEGACY_ROOT = Path(os.environ.get('ARENA_LEGACY_ROOT', ROOT))
OLD = LEGACY_ROOT / 'chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs'
NEW = ROOT / 'chapter6_sparse_feature_methods/exercises/part4_transcoders'
STEM = '6.4_Transcoders_and_Skip_Transcoders'


def read(path):
    return json.loads(path.read_text())


def code_cells(notebook):
    return [c for c in notebook['cells'] if c['cell_type'] == 'code']


def source(cell):
    return ''.join(cell['source'])


class TranscoderMigrationTests(unittest.TestCase):
    def test_all_original_markdown_survives(self):
        old = read(next(OLD.glob('*solutions.ipynb')))
        new = read(NEW / (STEM + '_solutions.ipynb'))
        normalize = lambda text: re.sub(r'\s+', ' ', text).strip()
        joined = normalize('\n'.join(source(c) for c in new['cells'] if c['cell_type'] == 'markdown'))
        for i, cell in enumerate(old['cells']):
            if cell['cell_type'] != 'markdown':
                continue
            text = source(cell).replace('<summary>Solution</summary>', '<summary>Historical worked solution</summary>')
            if i == 0:
                text = text.split('\n', 1)[1].replace('## Learning Objectives', '## What you will implement')
            text = re.sub(r'Difficulty: (easy|medium|hard)', lambda m: 'Difficulty: ' +
                          {'easy': '🔴🔴⚪⚪⚪', 'medium': '🔴🔴🔴⚪⚪', 'hard': '🔴🔴🔴🔴⚪'}[m[1]], text)
            text = re.sub(r'Importance: (very high|high)', lambda m: 'Importance: ' +
                          ('🔵🔵🔵🔵🔵' if m[1] == 'very high' else '🔵🔵🔵🔵⚪'), text)
            self.assertIn(normalize(text), joined, f'Legacy markdown cell {i} missing')

    def test_both_notebook_metadata_preserved(self):
        for kind in ['exercises', 'solutions']:
            old = read(next(OLD.glob('*_' + kind + '.ipynb')))
            new = read(NEW / (STEM + '_' + kind + '.ipynb'))
            for key, value in old['metadata'].items():
                self.assertEqual(new['metadata'][key], value)
            for before, after in zip(code_cells(old), code_cells(new), strict=True):
                self.assertEqual(before.get('metadata', {}), after.get('metadata', {}))

    def test_setup_inverse_and_exact_learning_objectives_heading(self):
        original = source(code_cells(read(next(OLD.glob('*solutions.ipynb'))))[0])
        current = source(code_cells(read(NEW / (STEM + '_solutions.ipynb')))[0])
        expected = original.replace('section = "part3_transcoders_attribution_graphs"',
                                    'section = "part4_transcoders"\nlegacy_section = "part3_transcoders_attribution_graphs"')
        expected = expected.replace('section_dir = exercises_dir / section',
                                    'section_dir = exercises_dir / legacy_section\ncanonical_section_dir = exercises_dir / section')
        expected = expected.replace('assets_dir = root_dir / chapter / "instructions" / "assets"',
                                    'assets_dir = canonical_section_dir / "media"')
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(current)))
        student = read(NEW / (STEM + '_exercises.ipynb'))
        self.assertTrue(any(source(c).splitlines()[0] == '## Content & Learning Objectives'
                            for c in student['cells'] if source(c).strip()))
        self.assertFalse((NEW / 'verification_report.json').exists())
        self.assertFalse((NEW / 'artifacts.lock.yml').exists())

    def test_historical_execution_objects_and_metadata(self):
        old = read(next(OLD.glob('*solutions.ipynb')))
        new = read(NEW / (STEM + '_solutions.ipynb'))
        a, b = code_cells(old), code_cells(new)
        self.assertEqual(len(a), 13)
        self.assertEqual(len(b), 13)
        for before, after in zip(a, b, strict=True):
            self.assertEqual(before.get('outputs', []), after.get('outputs', []))
            self.assertEqual(before.get('execution_count'), after.get('execution_count'))
            self.assertEqual(before.get('metadata', {}), after.get('metadata', {}))
        for key, value in old['metadata'].items():
            self.assertEqual(new['metadata'][key], value)
        self.assertEqual(sum(bool(c.get('outputs')) for c in b), 12)
        self.assertEqual(sum('image/png' in o.get('data', {}) for c in b for o in c.get('outputs', [])), 3)

    def test_all_nonsetup_code_is_identical_ast(self):
        before = code_cells(read(next(OLD.glob('*solutions.ipynb'))))
        after = code_cells(read(NEW / (STEM + '_solutions.ipynb')))
        for old, new in zip(before[1:], after[1:], strict=True):
            self.assertEqual(ast.dump(ast.parse(source(old))), ast.dump(ast.parse(source(new))))

    def test_explicit_legacy_bridge_and_gpu_opt_in(self):
        notebook = read(NEW / (STEM + '_solutions.ipynb'))
        cells = code_cells(notebook)
        setup = source(cells[0])
        self.assertIn('legacy_section = "part3_transcoders_attribution_graphs"', setup)
        self.assertIn('section_dir = exercises_dir / legacy_section', setup)
        self.assertIn('EXERCISE_ID = "6_3_transcoders_and_attribution_graphs"', setup)
        last = ast.parse(source(cells[-1]))
        self.assertTrue(all(isinstance(n, ast.FunctionDef) for n in last.body))
        self.assertIn('part3_transcoders_attribution_graphs.solutions', source(cells[-1]))
        self.assertIn('learner-trained SAE comparison', ''.join(source(c) for c in notebook['cells']))

    def test_student_stubs_and_explicit_student_test_arguments(self):
        cells = code_cells(read(NEW / (STEM + '_exercises.ipynb')))
        self.assertEqual(sum('NotImplementedError' in source(c) for c in cells), 6)
        calls = [n for c in cells for n in ast.walk(ast.parse(source(c)))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'tests']
        self.assertEqual(len(calls), 7)
        self.assertTrue(all(c.args for c in calls))
        for c in cells:
            compile(source(c), '<student cell>', 'exec')

    def test_reference_bytes_and_import_guards(self):
        for name in ['tests.py', 'utils.py', 'artifacts.lock.yml', 'verification_report.json',
                     'verification_report.schema.json', 'expected_outputs/README.md',
                     'expected_outputs/reference_metrics.json', 'expected_outputs/smoke_test.json']:
            destination = ('historical_' + name) if name in {'verification_report.json', 'artifacts.lock.yml', 'verification_report.schema.json'} else name
            self.assertEqual((OLD / name).read_bytes(), (NEW / destination).read_bytes())
        module = (NEW / 'solutions.py').read_text()
        tree = ast.parse(module)
        method_calls = {'make_toy_organism', 'make_toy_input', 'plot_attribution_graph',
                        'plot_signature_result', 'run_gpu_test', 'run_full_experiment'}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'MAIN':
                continue
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                    self.assertNotIn(call.func.id, method_calls)
        compile(module, str(NEW / 'solutions.py'), 'exec')


if __name__ == '__main__':
    unittest.main()
