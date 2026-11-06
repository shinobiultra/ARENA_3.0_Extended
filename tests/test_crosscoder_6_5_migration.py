"""Static source preservation only; never imports a taught module or runs training."""
import ast
import json
import os
from pathlib import Path
import re
import unittest

ROOT = Path(os.environ.get('ARENA_MIGRATION_ROOT', Path(__file__).resolve().parents[1]))
LEGACY_ROOT = Path(os.environ.get('ARENA_LEGACY_ROOT', ROOT))
OLD = LEGACY_ROOT / 'chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing'
NEW = ROOT / 'chapter6_sparse_feature_methods/exercises/part5_crosscoders'
STEM = '6.5_Crosscoders_and_Model_Diffing'
STUDENT_CODE_MAP = {**{i: i for i in range(8)}, 8: 13, 9: 14}


def read(path):
    return json.loads(path.read_text())


def code(nb):
    return [c for c in nb['cells'] if c['cell_type'] == 'code']


def source(cell):
    return ''.join(cell['source'])


def changed_setup(text):
    return (text.replace('section = "part4_crosscoders_model_diffing"',
                         'section = "part5_crosscoders"\nlegacy_section = "part4_crosscoders_model_diffing"')
            .replace('section_dir = exercises_dir / section',
                     'section_dir = exercises_dir / legacy_section\ncanonical_section_dir = exercises_dir / section')
            .replace('asset_dir = root_dir / chapter / "instructions" / "assets"',
                     'asset_dir = canonical_section_dir / "media"')
            .replace('from arena_ext.crosscoders import (',
                     'from part5_crosscoders.crosscoder_plumbing import (')
            .replace('from arena_ext.features import roc_auc_binary',
                     'from part5_crosscoders.crosscoder_plumbing import roc_auc_binary'))


class CrosscoderMigrationTests(unittest.TestCase):
    def test_solution_code_exact_ast_and_setup_inverse(self):
        before = code(read(next(OLD.glob('*solutions.ipynb'))))
        after = code(read(NEW / (STEM + '_solutions.ipynb')))
        self.assertEqual(len(before), 15)
        self.assertEqual(len(after), 15)
        for i, (a, b) in enumerate(zip(before, after, strict=True)):
            expected = changed_setup(source(a)) if i == 0 else source(a)
            self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(source(b))), i)

    def test_local_plumbing_exact_source_spans_and_inline_methods(self):
        names = {
            'CrosscoderInterventionReport', 'PlantedCrosscoderData',
            'PlantedFeatureMatchReport', 'SparseCrosscoderForward',
            'SparseCrosscoderLoss', 'SparseCrosscoderReconstructionReport',
            'TrainedSparseCrosscoder', 'FeatureOwner', 'roc_auc_binary'}

        def spans(text):
            lines = text.splitlines(keepends=True)
            found = {}
            for node in ast.parse(text).body:
                name = getattr(node, 'name', None)
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                if name in names:
                    start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])]) - 1
                    found[name] = ''.join(lines[start:node.end_lineno])
            return found

        original = {}
        for path in ['arena_ext/crosscoders.py', 'arena_ext/features.py']:
            original.update(spans((LEGACY_ROOT / path).read_text()))
        text = (NEW / 'crosscoder_plumbing.py').read_text()
        local = spans(text)
        self.assertEqual(set(local), names)
        self.assertEqual(local, original)  # Exact bytes include decorators and bodies.
        definitions = [n.name for n in ast.parse(text).body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        self.assertEqual(set(definitions), names - {'FeatureOwner'})
        self.assertEqual(len(definitions), 8)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, {'__future__', 'dataclasses', 'typing'})
            if isinstance(node, ast.Import):
                self.assertEqual([(n.name, n.asname) for n in node.names], [('torch', 't')])
        module = ast.parse((NEW / 'solutions.py').read_text())
        inline = {n.name for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertTrue({'SparseCrosscoder', 'train_sparse_crosscoder',
                         'make_planted_crosscoder_data'} <= inline)
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('arena_ext'))
                if node.module == 'part5_crosscoders.crosscoder_plumbing':
                    self.assertTrue({n.name for n in node.names} <= names)
        setup = source(code(read(NEW / (STEM + '_solutions.ipynb')))[0])
        original_setup = source(code(read(next(OLD.glob('*solutions.ipynb'))))[0])
        old_imports = [n.name for node in ast.walk(ast.parse(original_setup))
                       if isinstance(node, ast.ImportFrom) and node.module == 'arena_ext.crosscoders'
                       for n in node.names]
        local_imports = [n.name for node in ast.walk(ast.parse(setup))
                         if isinstance(node, ast.ImportFrom)
                         and node.module == 'part5_crosscoders.crosscoder_plumbing'
                         for n in node.names]
        self.assertEqual(local_imports, old_imports + ['roc_auc_binary'])

    def test_all_original_prose_in_both_notebooks(self):
        norm = lambda x: re.sub(r'\s+', ' ', x).strip()
        for kind in ['exercises', 'solutions']:
            old = read(next(OLD.glob('*_' + kind + '.ipynb')))
            new = read(NEW / (STEM + '_' + kind + '.ipynb'))
            joined = norm('\n'.join(source(c) for c in new['cells'] if c['cell_type'] == 'markdown'))
            for i, cell in enumerate(old['cells']):
                if cell['cell_type'] != 'markdown':
                    continue
                text = source(cell)
                if i == 0:
                    text = text.split('\n', 1)[1]
                text = text.replace('## Learning Objectives', '## What you will implement')
                text = text.replace('<summary>Solution</summary>', '<summary>Historical worked solution</summary>')
                text = text.replace('### Exercise - Compose the Evidence Ladder',
                                    '### Composition exercise - Compose the Evidence Ladder')
                text = re.sub(r'Difficulty: (easy|medium|hard)', lambda m: 'Difficulty: ' +
                              {'easy': '🔴🔴⚪⚪⚪', 'medium': '🔴🔴🔴⚪⚪', 'hard': '🔴🔴🔴🔴⚪'}[m[1]], text)
                text = re.sub(r'Importance: (very high|high)', lambda m: 'Importance: ' +
                              ('🔵🔵🔵🔵🔵' if m[1] == 'very high' else '🔵🔵🔵🔵⚪'), text)
                self.assertIn(norm(text), joined, (kind, i))

    def test_exact_output_objects_counts_and_metadata(self):
        old = read(next(OLD.glob('*solutions.ipynb')))
        new = read(NEW / (STEM + '_solutions.ipynb'))
        self.assertEqual(len(old['cells']), 32)
        for a, b in zip(code(old), code(new), strict=True):
            for field in ['outputs', 'execution_count', 'metadata']:
                self.assertEqual(a.get(field), b.get(field))
        self.assertEqual(sum(bool(c.get('outputs')) for c in code(new)), 13)
        self.assertEqual(sum('image/png' in o.get('data', {}) for c in code(new) for o in c.get('outputs', [])), 4)
        for kind in ['exercises', 'solutions']:
            a = read(next(OLD.glob('*_' + kind + '.ipynb')))
            b = read(NEW / (STEM + '_' + kind + '.ipynb'))
            for k, v in a['metadata'].items():
                self.assertEqual(b['metadata'][k], v)
        eold = code(read(next(OLD.glob('*exercises.ipynb'))))
        enew = code(read(NEW / (STEM + '_exercises.ipynb')))
        for i, j in STUDENT_CODE_MAP.items():
            self.assertEqual(eold[i].get('metadata', {}), enew[j].get('metadata', {}))

    def test_exact_nested_student_class_and_function_stubs(self):
        old = code(read(next(OLD.glob('*exercises.ipynb'))))
        new = code(read(NEW / (STEM + '_exercises.ipynb')))
        for i, j in STUDENT_CODE_MAP.items():
            before = ast.parse(source(old[i]))
            after = {n.name: n for n in ast.parse(source(new[j])).body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for node in before.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    self.assertEqual(ast.dump(node), ast.dump(after[node.name]), node.name)
        text = '\n'.join(source(c) for c in new)
        self.assertIn('# Play cell. Run this only after your implementation works.',
                      '\n'.join(source(c) for c in read(NEW / (STEM + '_exercises.ipynb'))['cells']))
        calls = [n for c in new for n in ast.walk(ast.parse(source(c)))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'tests']
        self.assertEqual(len(calls), 7)
        self.assertTrue(all(n.args for n in calls))

    def test_historical_names_and_optional_gpu_bridge(self):
        for name in ['artifacts.lock.yml', 'verification_report.json', 'verification_report.schema.json']:
            self.assertFalse((NEW / name).exists())
            self.assertEqual((NEW / ('historical_' + name)).read_bytes(), (OLD / name).read_bytes())
        for name in ['tests.py', 'utils.py', 'expected_outputs/README.md',
                     'expected_outputs/reference_metrics.json', 'expected_outputs/smoke_test.json']:
            self.assertEqual((NEW / name).read_bytes(), (OLD / name).read_bytes())
        last = source(code(read(NEW / (STEM + '_solutions.ipynb')))[-1])
        self.assertIn('part4_crosscoders_model_diffing.solutions', last)
        self.assertTrue(all(isinstance(n, ast.FunctionDef) for n in ast.parse(last).body))

    def test_import_guard_source_and_exact_heading(self):
        text = (NEW / 'solutions.py').read_text()
        for node in ast.parse(text).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'MAIN':
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                if isinstance(call.func, ast.Name):
                    self.assertNotIn(call.func.id, {'make_planted_crosscoder_data', 'train_sparse_crosscoder',
                                                  'run_crosscoder_signature_result', 'run_smoke_test', 'run_gpu_test'})
                if isinstance(call.func, ast.Attribute):
                    self.assertNotIn(call.func.attr, {'mkdir', 'read_text', 'savefig', 'show'})
        for kind in ['exercises', 'solutions']:
            nb = read(NEW / (STEM + '_' + kind + '.ipynb'))
            self.assertTrue(any(source(c).splitlines()[0] == '## Content & Learning Objectives'
                                for c in nb['cells'] if source(c).strip()))
            for c in code(nb):
                compile(source(c), '<cell>', 'exec')
        compile(text, '<solutions module>', 'exec')


if __name__ == '__main__':
    unittest.main()
