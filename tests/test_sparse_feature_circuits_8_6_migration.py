"""Static source-only preservation, globals and routing checks; never imports lesson code."""
import ast
import builtins
import copy
import json
import os
from pathlib import Path
import re
import symtable
import unittest

ROOT = Path(os.environ.get('ARENA_MIGRATION_ROOT', Path(__file__).resolve().parents[1]))
LEGACY_ROOT = Path(os.environ.get('ARENA_LEGACY_ROOT', ROOT))
OLD = LEGACY_ROOT / 'chapter8_automated_circuits/exercises/part5_sparse_feature_circuits'
NEW = ROOT / 'chapter8_automated_circuits/exercises/part6_sparse_feature_circuits'
STEM = '8.6_Sparse_Feature_Circuits_and_SHIFT'
MASTER = ROOT / 'infrastructure/chapters/chapter8_automated_circuits/master_8_6.py'
PAGE = ROOT / 'chapter8_automated_circuits/instructions/pages/06_[8.6]_Sparse_Feature_Circuits_and_SHIFT.md'


def read(path):
    return json.loads(path.read_text())


def source(cell):
    return ''.join(cell['source'])


def code(nb):
    return [c for c in nb['cells'] if c['cell_type'] == 'code']


def setup(text):
    return (text.replace('section = "part5_sparse_feature_circuits"',
                         'section = "part6_sparse_feature_circuits"\nlegacy_section = "part5_sparse_feature_circuits"')
            .replace('section_dir = exercises_dir / section',
                     'section_dir = exercises_dir / legacy_section\ncanonical_section_dir = exercises_dir / section')
            .replace('assets_dir = section_dir / "assets"', 'assets_dir = canonical_section_dir / "assets"'))


def prose(text, index):
    text = text.replace('<summary>Solution</summary>', '<summary>Historical worked solution</summary>')
    text = text.replace('## Learning Objectives', '## What you will implement')
    if index == 0:
        title, body = text.split('\n', 1)
        text = '**Historical lesson title:** ' + title.removeprefix('# ') + '\n' + body
    for before, after in [('Difficulty: medium', 'Difficulty: 🔴🔴🔴⚪⚪'),
                          ('Difficulty: hard', 'Difficulty: 🔴🔴🔴🔴⚪'),
                          ('Importance: high', 'Importance: 🔵🔵🔵🔵⚪'),
                          ('Importance: medium', 'Importance: 🔵🔵🔵⚪⚪')]:
        text = text.replace(before, after)
    return text


class SparseFeatureCircuitMigrationTests(unittest.TestCase):
    def test_full_code_objects_outputs_counts_metadata_ids_and_student_stubs(self):
        for kind in ['exercises', 'solutions']:
            old = read(next(OLD.glob('*_' + kind + '.ipynb')))
            new = read(NEW / (STEM + '_' + kind + '.ipynb'))
            self.assertEqual(len(old['cells']), 34)
            self.assertEqual(len(code(old)), 18)
            self.assertEqual(len(code(new)), 18)
            for key in ['metadata', 'nbformat', 'nbformat_minor']:
                self.assertEqual(old[key], new[key])
            for i, (a, b) in enumerate(zip(code(old), code(new), strict=True)):
                expected = setup(source(a)) if i == 0 else source(a)
                self.assertEqual(expected, source(b), (kind, i))
                self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(source(b))), (kind, i))
                inverse = copy.deepcopy(b)
                inverse['source'] = copy.deepcopy(a['source'])
                self.assertEqual(inverse, a, (kind, i))
            self.assertEqual(sum(bool(c.get('outputs')) for c in code(new)), 14 if kind == 'solutions' else 0)
            self.assertEqual(sum('image/png' in o.get('data', {}) for c in code(new)
                                 for o in c.get('outputs', [])), 2 if kind == 'solutions' else 0)

    def test_all_original_prose_variant_introductions_dropdowns_and_blank_forecasts(self):
        norm = lambda s: re.sub(r'\s+', ' ', s).strip()
        introductions = []
        for kind in ['exercises', 'solutions']:
            old = read(next(OLD.glob('*_' + kind + '.ipynb')))
            new = read(NEW / (STEM + '_' + kind + '.ipynb'))
            joined = norm('\n'.join(source(c) for c in new['cells'] if c['cell_type'] == 'markdown'))
            for i, cell in enumerate(old['cells']):
                if cell['cell_type'] != 'markdown':
                    continue
                self.assertIn(norm(prose(source(cell), i)), joined, (kind, i))
                mapped = [c for c in new['cells'] if c.get('id') == cell.get('id')]
                self.assertEqual(len(mapped), 1)
                self.assertEqual(mapped[0].get('metadata'), cell.get('metadata'))
                if i == 0:
                    introductions.append(source(mapped[0]))
            self.assertEqual(joined.count('Forecast: ______'), 12)
            self.assertEqual(joined.count('Observed result: ______'), 12)
            self.assertIn('## Content & Learning Objectives', joined)
            self.assertIn('does not implement or validate the complete SHIFT method', joined)
            self.assertIn('released-SAE feature-circuit replication', joined)
            self.assertIn('Source-convention correction', joined)
        self.assertNotEqual(*introductions)

    def test_all_fourteen_original_test_calls_and_play_cells_exact(self):
        def calls(nb):
            return [ast.dump(n) for c in code(nb) for n in ast.walk(ast.parse(source(c)))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == 'tests']
        for kind in ['exercises', 'solutions']:
            old = read(next(OLD.glob('*_' + kind + '.ipynb')))
            new = read(NEW / (STEM + '_' + kind + '.ipynb'))
            self.assertEqual(calls(old), calls(new))
            self.assertEqual(len(calls(new)), 14)
            old_code = code(old)
            new_code = code(new)
            self.assertEqual(source(old_code[14]), source(new_code[14]))
            self.assertEqual(source(old_code[15]), source(new_code[15]))

    def test_all_33_core_functions_and_seven_record_classes_exact_inline(self):
        old = ast.parse('\n'.join(source(c) for c in code(read(next(OLD.glob('*solutions.ipynb'))))))
        new = ast.parse((NEW / 'solutions.py').read_text())
        definitions = {n.name: n for n in new.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(sum(isinstance(n, ast.FunctionDef) for n in new.body), 33)
        self.assertEqual(sum(isinstance(n, ast.ClassDef) for n in new.body), 7)
        for node in old.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                self.assertEqual(ast.dump(node), ast.dump(definitions[node.name]), node.name)
        for node in ast.walk(new):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith('arena_ext'))
            if isinstance(node, ast.Import):
                self.assertFalse(any(n.name.startswith('arena_ext') for n in node.names))

    def test_import_allocation_effect_guards_and_global_default_decorator_availability(self):
        text = (NEW / 'solutions.py').read_text()
        module = ast.parse(text)
        available = set(dir(builtins)) | {'__file__', '__name__'}
        definitions = {n.name for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for node in module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                available.update(n.asname or n.name.split('.')[0] for n in node.names)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                available.add(node.name)
            if isinstance(node, ast.Assign):
                available.update(n.id for n in node.targets if isinstance(n, ast.Name))
        for node in module.body:
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'MAIN':
                continue
            if isinstance(node, ast.FunctionDef):
                for default in [*node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]]:
                    self.assertFalse(any(isinstance(n, ast.Call) for n in ast.walk(default)), node.name)
                self.assertFalse(node.decorator_list, node.name)
                continue
            if isinstance(node, ast.ClassDef):
                self.assertEqual(len(node.decorator_list), 1)
                self.assertEqual(ast.unparse(node.decorator_list[0]), 'dataclass(frozen=True)')
                self.assertTrue(all(isinstance(n, ast.AnnAssign) for n in node.body))
                continue
            for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
                allowed = ((isinstance(call.func, ast.Name) and call.func.id == 'Path')
                           or (isinstance(call.func, ast.Attribute) and call.func.attr == 'resolve'))
                self.assertTrue(allowed, ast.unparse(call))
        table = symtable.symtable(text, '<canonical source>', 'exec')
        def check(table):
            for symbol in table.get_symbols():
                if symbol.is_global() and symbol.is_referenced():
                    self.assertIn(symbol.get_name(), available, (table.get_name(), symbol.get_name()))
            for child in table.get_children():
                check(child)
        for child in table.get_children():
            if child.get_name() in definitions:
                check(child)
        self.assertIn('root_dir = Path(__file__).resolve().parents[3]', text)
        self.assertIn('part5_sparse_feature_circuits.solutions', text)
        compile(text, '<canonical module>', 'exec')
        compile(MASTER.read_text(), '<canonical master>', 'exec')
        for kind in ['exercises', 'solutions']:
            for cell in code(read(NEW / (STEM + '_' + kind + '.ipynb'))):
                compile(source(cell), '<canonical notebook cell>', 'exec')

    def test_canonical_local_notebook_routes_resolve_on_all_five_surfaces(self):
        paths = [MASTER, MASTER.with_suffix('.ipynb'), PAGE,
                 NEW / (STEM + '_exercises.ipynb'), NEW / (STEM + '_solutions.ipynb')]
        total = 0
        for path in paths:
            text = ('\n'.join(source(c) for c in read(path)['cells'] if c['cell_type'] == 'markdown')
                    if path.suffix == '.ipynb' else path.read_text())
            links = re.findall(r'\[[^\]]*\]\(([^)]+\.ipynb(?:#[^)]*)?)\)', text)
            workbook_url = '../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-8-6'
            local_urls = [url for url in links if '://' not in url]
            workbook_links = [url for url in local_urls if url == workbook_url]
            expected_workbook_count = 1 if path.parent == NEW and path.suffix == '.ipynb' else 0
            self.assertEqual(len(workbook_links), expected_workbook_count, path)
            for url in workbook_links:
                target = (path.parent / url.split('#', 1)[0]).resolve()
                self.assertEqual(target, (ROOT / 'mastery_hub/units/extension_predictions/exercises.ipynb').resolve())
                self.assertEqual(url.split('#', 1)[1], 'section-8-6')
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
                total += 1
        self.assertEqual(total, 10)

    def test_historical_helpers_reports_protocols_and_assets_exact(self):
        for name in ['artifacts.lock.yml', 'verification_report.json', 'verification_report.schema.json']:
            self.assertFalse((NEW / name).exists())
            self.assertEqual((NEW / ('historical_' + name)).read_bytes(), (OLD / name).read_bytes())
        for name in ['tests.py', 'utils.py', 'expected_outputs/README.md',
                     'expected_outputs/reference_metrics.json', 'expected_outputs/smoke_test.json']:
            self.assertEqual((NEW / name).read_bytes(), (OLD / name).read_bytes())
        for path in (OLD / 'assets').iterdir():
            self.assertEqual(path.read_bytes(), (NEW / 'assets' / path.name).read_bytes())
        self.assertEqual(len(list((NEW / 'media').glob('sparse_feature_legacy_cell_25_output_*.png'))), 2)


    def test_all_local_png_and_svg_image_routes_resolve_from_actual_surfaces(self):
        paths = [MASTER, MASTER.with_suffix('.ipynb'), PAGE,
                 NEW / (STEM + '_exercises.ipynb'), NEW / (STEM + '_solutions.ipynb')]
        self.assertEqual(PAGE.name, '06_[8.6]_Sparse_Feature_Circuits_and_SHIFT.md')
        total = 0
        for path in paths:
            text = ('\n'.join(source(c) for c in read(path)['cells'] if c['cell_type'] == 'markdown')
                    if path.suffix == '.ipynb' else path.read_text())
            html = re.findall(r'src=["\']([^"\']+)["\']', text)
            markdown = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
            images = [url for url in html + markdown
                      if re.search(r'\.(?:png|jpe?g|gif|webp|svg)(?:[?#].*)?$', url, re.IGNORECASE)
                      and '://' not in url and not url.startswith('data:')]
            expected = 0 if path.parent == NEW else 2
            self.assertEqual(len(images), expected, path)
            for url in images:
                target = (path.parent / url.split('#', 1)[0].split('?', 1)[0]).resolve()
                self.assertTrue(target.is_file(), (path, url))
                self.assertEqual(target.parent, (NEW / 'media').resolve())
                self.assertIn(target.name, {'sparse_feature_legacy_cell_25_output_1.png',
                                           'sparse_feature_legacy_cell_25_output_2.png'})
                self.assertTrue(target.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                total += 1
        self.assertEqual(total, 6)


if __name__ == '__main__':
    unittest.main()
