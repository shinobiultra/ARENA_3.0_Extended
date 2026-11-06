"""Source-only complete historical 13.1 migration checks; no taught computation."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('ARENA_MIGRATION_ROOT', Path(__file__).resolve().parents[1]))
LEGACY_ROOT = Path(os.environ.get('ARENA_LEGACY_ROOT', ROOT))
CHAPTER = 'chapter13_image_generation_interpretability'
OLD = LEGACY_ROOT / CHAPTER / 'exercises/part1_diffusion_image_controls'
NEW = ROOT / CHAPTER / 'exercises/part1_diffusion_cross_attention'
STEM = '13.1_Diffusion_Models_and_Cross-Attention_Attribution'
MASTER = ROOT / 'infrastructure/chapters' / CHAPTER / 'master_13_1.py'
PAGE = ROOT / CHAPTER / 'instructions/pages/01_[13.1]_Diffusion_Models_and_Cross-Attention_Attribution.md'
DOCS = ROOT / 'docs/mastery/migrations'


def read(path):
    return json.loads(path.read_text())


def source(cell):
    return ''.join(cell['source'])


def original(kind):
    return read(OLD / ('13.1_Diffusion_and_Image_Generation_Controls_' + kind + '.ipynb'))


def canonical(kind):
    return read(NEW / (STEM + '_' + kind + '.ipynb'))


def setup(text):
    text = (text.replace('section = "part1_diffusion_image_controls"',
                         'section = "part1_diffusion_cross_attention"\nlegacy_section = "part1_diffusion_image_controls"')
            .replace('section_dir = exercises_dir / section',
                     'section_dir = exercises_dir / legacy_section\ncanonical_section_dir = exercises_dir / section')
            .replace('assets_dir = root_dir / chapter / "instructions" / "assets"',
                     'assets_dir = canonical_section_dir / "media"')
            .replace('import part1_diffusion_image_controls.solutions as reference',
                     'import part1_diffusion_cross_attention.lesson_reference as reference'))
    return '\n'.join(line for line in text.splitlines()
                     if not line.startswith('from arena_ext.vlm_interpretability import contrastive_alignment_report')) + '\n'


def repaired_constructor_source(text):
    """The sole reviewed scientific-source edit; historical markdown is excluded."""
    return text.replace('class CrossAttentionCaptureStore:',
                        '@dataclass\nclass CrossAttentionCaptureStore:', 1)


def constructor_inverse(node):
    node = copy.deepcopy(node)
    if isinstance(node, ast.ClassDef) and node.name == 'CrossAttentionCaptureStore':
        assert len(node.decorator_list) == 1
        decorator = node.decorator_list[0]
        assert isinstance(decorator, ast.Name) and decorator.id == 'dataclass'
        node.decorator_list = []
    return node


def adapted_markdown(text):
    return (text.replace('<summary>Solution</summary>', '<summary>Historical worked solution</summary>')
            .replace('13.1_Diffusion_and_Image_Generation_Controls_', STEM + '_')
            .replace('../../instructions/assets/', 'media/'))


def calls(nb):
    return [ast.dump(n) for c in nb['cells'] if c['cell_type'] == 'code'
            for n in ast.walk(ast.parse(source(c)))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == 'tests']


def original_nodes(nb):
    return [n for c in nb['cells'] if c['cell_type'] == 'code'
            for n in ast.parse(source(c)).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))]


class DiffusionMigrationTests(unittest.TestCase):
    def test_full_original_cells_metadata_ids_counts_and_outputs(self):
        rows = read(DOCS / 'diffusion_13_1_cells.json')
        for kind in ['exercises', 'solutions']:
            a, b = original(kind), canonical(kind)
            self.assertEqual(len(a['cells']), 49)
            self.assertEqual(sum(c['cell_type'] == 'code' for c in a['cells']), 20)
            self.assertEqual(sum(c['cell_type'] == 'code' for c in b['cells']), 21)
            for name in ['metadata', 'nbformat', 'nbformat_minor']:
                self.assertEqual(a[name], b[name], (kind, name))
            mapping = [row for row in rows if row['kind'] == kind]
            self.assertEqual(len(mapping), 49)
            for row in mapping:
                index = row['legacy_cell']
                before, after = a['cells'][index], b['cells'][row['canonical_cell']]
                expected = (setup(source(before)) if index == 1 else source(before))
                if index == 29:
                    expected = repaired_constructor_source(expected)
                if before['cell_type'] == 'markdown':
                    expected = adapted_markdown(source(before))
                self.assertEqual(source(after), expected, (kind, index))
                inverse = copy.deepcopy(after)
                inverse['source'] = copy.deepcopy(before['source'])
                self.assertEqual(inverse, before, (kind, index))
            self.assertEqual(sum(bool(c.get('outputs')) for c in b['cells']), 16 if kind == 'solutions' else 0)
            self.assertEqual(sum('image/png' in o.get('data', {}) for c in b['cells']
                                 for o in c.get('outputs', [])), 3 if kind == 'solutions' else 0)

    def test_every_original_test_call_and_nested_student_stub_preserved(self):
        for kind in ['exercises', 'solutions']:
            self.assertEqual(calls(original(kind)), calls(canonical(kind)))
            self.assertEqual(len(calls(canonical(kind))), 12)
        a = original_nodes(original('exercises'))
        b = {n.name: n for n in original_nodes(canonical('exercises'))}
        for node in a:
            self.assertEqual(ast.dump(node), ast.dump(constructor_inverse(b[node.name])), node.name)
        student = '\n'.join(source(c) for c in canonical('exercises')['cells'] if c['cell_type'] == 'code')
        self.assertIn('PLAY_MASK_SHIFT = (0, 0)', student)
        self.assertIn('PLAY_MIN_REGION_MASS = 0.20', student)
        # An explicit variant retains the original student definition order in
        # historical cell29, while the solution order differs.
        master = MASTER.read_text()
        self.assertIn('# ! FILTERS: [ex]', master)
        self.assertIn('# ! FILTERS: [soln,st,python]', master)
        self.assertIn('# EXERCISE', master)
        self.assertIn('# SOLUTION', master)

    def test_scientific_definitions_exact_and_no_hidden_method_import(self):
        text = (NEW / 'solutions.py').read_text()
        module = ast.parse(text)
        definitions = {n.name: n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for node in original_nodes(original('solutions')):
            self.assertEqual(ast.dump(node), ast.dump(constructor_inverse(definitions[node.name])), node.name)
        for path in [NEW / 'solutions.py', NEW / 'lesson_reference.py', MASTER]:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('arena_ext'))
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(n.name.startswith('arena_ext') for n in node.names))
            compile(path.read_text(), str(path), 'exec')
        for kind in ['exercises', 'solutions']:
            for cell in canonical(kind)['cells']:
                if cell['cell_type'] == 'code':
                    compile(source(cell), '<notebook cell>', 'exec')
        # Root adopted the independently reviewed single-decorator compatibility
        # repair. Exact inverse checks preserve all other code and old outputs.
        store = definitions['CrossAttentionCaptureStore']
        self.assertEqual(len(store.decorator_list), 1)
        self.assertEqual(ast.dump(store.decorator_list[0]), ast.dump(ast.Name(id='dataclass', ctx=ast.Load())))
        self.assertIn('Constructor compatibility repair', '\n'.join(source(c) for c in canonical('solutions')['cells']))

    def test_repaired_constructor_fields_and_explicit_keyword_call_static(self):
        texts = [((NEW / 'solutions.py').read_text(), 1), (MASTER.read_text(), 2)]
        for kind in ['exercises', 'solutions']:
            texts.extend((source(c), 1) for c in canonical(kind)['cells']
                         if c['cell_type'] == 'code' and 'class CrossAttentionCaptureStore:' in source(c))
        self.assertEqual(len(texts), 4)
        for text, expected_count in texts:
            stores = [n for n in ast.parse(text).body
                      if isinstance(n, ast.ClassDef) and n.name == 'CrossAttentionCaptureStore']
            self.assertEqual(len(stores), expected_count)
            for other in stores[1:]:
                self.assertEqual(ast.dump(other), ast.dump(stores[0]))
            store = stores[0]
            self.assertEqual([ast.dump(n) for n in store.decorator_list],
                             [ast.dump(ast.Name(id='dataclass', ctx=ast.Load()))])
            fields = [n for n in store.body if isinstance(n, ast.AnnAssign)]
            names = [n.target.id for n in fields]
            self.assertEqual(names, ['token_groups', 'maps', 'resolutions'])
            self.assertTrue(all(n.value is None for n in fields))
            self.assertFalse(any(isinstance(n, ast.Assign) for n in store.body))
            methods = [n for n in store.body if isinstance(n, ast.FunctionDef)]
            self.assertEqual([n.name for n in methods], ['for_groups'])
            method = methods[0]
            self.assertEqual([ast.dump(n) for n in method.decorator_list],
                             [ast.dump(ast.Name(id='classmethod', ctx=ast.Load()))])
            calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == 'cls']
            self.assertEqual(len(calls), 1)
            call = calls[0]
            self.assertEqual(call.args, [])
            self.assertEqual([n.arg for n in call.keywords], names)
            expected = ast.parse('cls(token_groups=token_groups, maps={key: [] for key in token_groups}, resolutions=[])', mode='eval').body
            self.assertEqual(ast.dump(call), ast.dump(expected))

    def test_exact_extracted_source_spans_and_ast(self):
        code = '\n'.join(source(c) for c in canonical('solutions')['cells'] if c['cell_type'] == 'code')
        nodes = {n.name: n for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        helper_nodes = {}
        for node in ast.parse((NEW / 'lesson_reference.py').read_text()).body:
            name = getattr(node, 'name', None)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            if name:
                helper_nodes[name] = node
        for row in read(DOCS / 'diffusion_13_1_extracted_spans.json'):
            path = LEGACY_ROOT / row['source_path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['source_file_sha256'])
            span = ''.join(path.read_text().splitlines(True)[row['start_line'] - 1:row['end_line']])
            self.assertEqual(hashlib.sha256(span.encode()).hexdigest(), row['source_sha256'])
            node = (nodes if row['symbol'] in nodes else helper_nodes)[row['symbol']]
            self.assertEqual(hashlib.sha256(ast.dump(node).encode()).hexdigest(), row['ast_sha256'])
        self.assertIn('part1_diffusion_image_controls.solutions import', (NEW / 'lesson_reference.py').read_text())

    def test_all_legacy_assets_helpers_and_historical_identity(self):
        for name in ['artifacts.lock.yml', 'verification_report.json', 'verification_report.schema.json']:
            self.assertFalse((NEW / name).exists())
            self.assertEqual((NEW / ('historical_' + name)).read_bytes(), (OLD / name).read_bytes())
        for name in ['tests.py', 'utils.py', 'expected_outputs/README.md',
                     'expected_outputs/reference_metrics.json', 'expected_outputs/smoke_test.json']:
            self.assertEqual((NEW / name).read_bytes(), (OLD / name).read_bytes())
        for path in (LEGACY_ROOT / CHAPTER / 'instructions/assets').glob('diffusion_image*'):
            self.assertEqual((NEW / 'media' / path.name).read_bytes(), path.read_bytes())
        for i, cell in enumerate(original('solutions')['cells']):
            for j, out in enumerate(cell.get('outputs', [])):
                if 'image/png' in out.get('data', {}):
                    import base64
                    value = out['data']['image/png']
                    value = ''.join(value) if isinstance(value, list) else value
                    self.assertEqual((NEW / 'media' / f'diffusion_13_1_legacy_cell_{i}_output_{j}.png').read_bytes(),
                                     base64.b64decode(value))

    def test_source_only_main_guards_and_dependency_sentinel_import(self):
        text = (NEW / 'solutions.py').read_text()
        module = ast.parse(text)
        definitions = {n.name for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        prohibited = {'mkdir', 'append', 'read_text', 'savefig', 'show', 'manual_seed', 'seed',
                      'set_grad_enabled', 'tensor', 'ones', 'zeros', 'randn', 'arange', 'meshgrid',
                      'device', 'default_rng', 'set_default_dtype', 'cuda', 'from_pretrained'}
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
                        self.assertNotIn(call.func.attr, prohibited)
        # Execute inert import scaffolding with dependency sentinels, never actual
        # torch/numpy/matplotlib/IPython, taught tests, model code or lesson calls.
        class NoTensor:
            def __new__(cls, *args, **kwargs):
                raise AssertionError('Import attempted tensor/device construction')
        dependencies = {}
        for name in ['torch', 'numpy', 'pandas', 'matplotlib', 'matplotlib.pyplot',
                     'IPython', 'IPython.display']:
            dependencies[name] = types.ModuleType(name)
        dependencies['torch'].Tensor = NoTensor
        dependencies['torch'].device = NoTensor
        dependencies['IPython.display'].display = lambda *a, **k: self.fail('display called during import')
        dependencies['matplotlib'].pyplot = dependencies['matplotlib.pyplot']
        dependencies['IPython'].display = dependencies['IPython.display']
        inert = types.ModuleType('arena_diffusion_migration_import_probe')
        inert.__file__ = str(NEW / 'solutions.py')
        dependencies[inert.__name__] = inert
        before = list(sys.path)
        with patch.dict(sys.modules, dependencies), patch.object(Path, 'write_text', side_effect=AssertionError), \
                patch.object(Path, 'mkdir', side_effect=AssertionError), patch.object(Path, 'read_text', side_effect=AssertionError):
            exec(compile(text, str(NEW / 'solutions.py'), 'exec'), inert.__dict__)
        self.assertFalse(inert.MAIN)
        self.assertEqual(sys.path, before)
        self.assertNotIn('signature_result', inert.__dict__)
        self.assertNotIn('verification_report', inert.__dict__)
        self.assertNotIn('reference', inert.__dict__)
        self.assertNotIn('tests', inert.__dict__)

    def test_proposed_case_actual_pure_worksheet_validator(self):
        import importlib.util
        helper = Path(os.environ.get('ARENA_WORKSHEET_HELPER',
                                     ROOT / 'mastery_hub/units/extension_predictions/lesson.py'))
        case_path = Path(os.environ.get('ARENA_PROPOSED_DIFFUSION_CASE',
                                       ROOT / 'docs/mastery/migrations/diffusion_13_1_inspection.json'))
        spec = importlib.util.spec_from_file_location('diffusion_actual_pure_worksheet', helper)
        worksheet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worksheet)
        payload = read(case_path)
        case = payload.get('proposed_guided_case', payload)
        self.assertEqual(worksheet.validate_case(case), case)
        self.assertEqual(case['transfer_prediction_type'], 'binary')
        self.assertIn('PLAY_MASK_SHIFT from(0,0) to(16,0)', case['transfer'])
        self.assertIn('remain>=.20 AND control attention mass', case['transfer'])
        self.assertIn('remain<.20', case['transfer'])
        self.assertIn('red_square case only', case['transfer'])
        self.assertIn('No answer is supplied or computed', case['transfer'])
        invalid = copy.deepcopy(case)
        invalid['transfer_prediction_type'] = 'scalar'
        with self.assertRaisesRegex(ValueError, 'this worksheet uses binary transfer forecasts'):
            worksheet.validate_case(invalid)

    def test_actual_config_identity_objectives_and_blank_exposed_practice(self):
        config = (ROOT / 'infrastructure/core/config.yaml').read_text()
        self.assertIn('01_[13.1]_Diffusion_Models_and_Cross-Attention_Attribution', config)
        self.assertIn('part1_diffusion_cross_attention', config)
        self.assertTrue(PAGE.is_file())
        self.assertTrue(PAGE.with_suffix('.py').is_file())
        self.assertTrue((ROOT / 'infrastructure/chapters' / CHAPTER / 'master_13_1.ipynb').is_file())
        for kind in ['exercises', 'solutions']:
            joined = '\n'.join(source(c) for c in canonical(kind)['cells'] if c['cell_type'] == 'markdown')
            self.assertIn('## Content & Learning Objectives', joined)
            self.assertIn('> ##### Learning Objectives', joined)
            self.assertEqual(joined.count('Forecast: ______'), 12)
            self.assertEqual(joined.count('Observed result: ______'), 12)
            self.assertIn('Historical worked solution', joined)
            self.assertIn('patterned disc', joined)
            self.assertIn('not establish clean solid-shape prompt fidelity', joined)
            self.assertIn('13.2 is a separate lesson', joined)


if __name__ == '__main__':
    unittest.main()
