"""Source-only preservation checks; never imports or runs taught code."""
from pathlib import Path
import ast
import builtins
import copy
import gzip
import hashlib
import importlib.util
import json
import re
import symtable
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "docs/mastery/evidence/maze_migration_20261020"
N = ROOT / "chapter14_jepa_world_models/exercises/part5_maze_transformers"
STEM = "14.5_Maze_Transformers_and_World_Models"
text = lambda c: "".join(c["source"])
parse = lambda s: ast.dump(ast.parse(s))


def routes():
    env = {"records": (E / "record_support.py").read_text(),
           "old_import": (E / "old_import.txt").read_text()}
    exec(compile((E / "source_routes.py").read_text(), "pure routes", "exec"), env)
    return env


class Preservation(unittest.TestCase):
    def test_full_original_objects(self):
        mapping = json.loads((E / "cell_mapping.json").read_text())
        inverse = routes()["inverse"]
        for kind in ["exercises", "solutions"]:
            old = json.loads(gzip.decompress((E / f"legacy_{kind}.ipynb.gz").read_bytes()))
            new = json.loads((N / f"{STEM}_{kind}.ipynb").read_text())
            self.assertEqual(old["metadata"], new["metadata"])
            rows = [r for r in mapping if r["kind"] == kind]
            self.assertEqual(len(rows), 48)
            for row in rows:
                before = old["cells"][row["legacy_cell"]]
                after = copy.deepcopy(new["cells"][row["canonical_cell"]])
                after["source"] = inverse(text(after), row["legacy_cell"]).splitlines(keepends=True)
                self.assertEqual(before, after)
            self.assertEqual(sum(c["cell_type"] == "code" for c in new["cells"]), 29)
            if kind == "solutions":
                self.assertEqual(sum(bool(c.get("outputs")) for c in new["cells"]), 16)
                self.assertEqual(sum("image/png" in o.get("data", {}) for c in new["cells"]
                                     for o in c.get("outputs", [])), 3)

    def test_methods_stubs_tests_and_assertions(self):
        env = routes()
        for kind in ["exercises", "solutions"]:
            old = json.loads(gzip.decompress((E / f"legacy_{kind}.ipynb.gz").read_bytes()))
            new = json.loads((N / f"{STEM}_{kind}.ipynb").read_text())
            before = [text(c) for c in old["cells"] if c["cell_type"] == "code"]
            after = [text(c) for c in new["cells"] if c["cell_type"] == "code"]
            self.assertEqual(before[1:], after[1:])
            self.assertEqual(sum(s.count("NotImplementedError") for s in before),
                             sum(s.count("NotImplementedError") for s in after))
            for s in after:
                ast.parse(s)
        original = gzip.decompress((E / "legacy_tests.py.gz").read_bytes()).decode()
        expected = original.replace("<summary>Solution</summary>",
                                    "<summary>Historical worked solution</summary>").replace(
            "section / 'expected_outputs/reference_metrics.json'",
            "section / 'expected_outputs/historical_reference_metrics.json'")
        self.assertEqual(expected, (N / "tests.py").read_text())
        self.assertEqual([ast.dump(n) for n in ast.walk(ast.parse(expected)) if isinstance(n, ast.Assert)],
                         [ast.dump(n) for n in ast.walk(ast.parse((N / "tests.py").read_text()))
                          if isinstance(n, ast.Assert)])

    def test_generated_python_is_inert_and_closed(self):
        source = (N / "solutions.py").read_text()
        module = ast.parse(source)
        self.assertNotIn("arena_ext", source)
        eager = []
        for n in module.body:
            if isinstance(n, ast.If):
                self.assertEqual(ast.unparse(n.test), "MAIN")
                self.assertEqual(n.orelse, [])
                continue
            self.assertIsInstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.ClassDef))
            eager.append(n)
            if isinstance(n, ast.Assign):
                calls = [ast.unparse(x) for x in ast.walk(n.value) if isinstance(x, ast.Call)]
                self.assertTrue(not calls or ast.unparse(n.targets[0]) == "root_dir", calls)
                if calls:
                    self.assertEqual(ast.unparse(n.value), "Path(__file__).resolve().parents[3]")
        bound = set(dir(builtins)) | {"__file__", "__name__"}
        for n in eager:
            if isinstance(n, ast.Import):
                bound.update(a.asname or a.name.split(".")[0] for a in n.names)
            if isinstance(n, ast.ImportFrom):
                self.assertFalse(n.module and "part5_maze_transformers" in n.module)
                bound.update(a.asname or a.name for a in n.names)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                bound.add(n.name)
            if isinstance(n, ast.Assign):
                bound.update(x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name))
        for n in ast.walk(module):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in [*n.args.defaults, *n.args.kw_defaults]:
                    if default is not None:
                        self.assertFalse(any(isinstance(x, ast.Call) for x in ast.walk(default)))
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                for dec in n.decorator_list:
                    self.assertIn(ast.unparse(dec), {"dataclass(frozen=True)", "property", "t.no_grad()"})
                if isinstance(n, ast.ClassDef):
                    for item in n.body:
                        self.assertIsInstance(item, (ast.FunctionDef, ast.AnnAssign, ast.Expr))
                        if isinstance(item, ast.AnnAssign):
                            self.assertIsNone(item.value)
                        if isinstance(item, ast.Expr):
                            self.assertIsInstance(item.value, ast.Constant)
                            self.assertIsInstance(item.value.value, str)
        top = symtable.symtable(source, "canonical solutions.py", "exec")
        missing = {}
        def visit(table):
            for symbol in table.get_symbols():
                if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in bound:
                    missing.setdefault(table.get_name(), []).append(symbol.get_name())
            for child in table.get_children():
                visit(child)
        for child in top.get_children():
            visit(child)
        self.assertEqual(missing, {})
        compile(source, str(N / "solutions.py"), "exec")

    def test_exact_support_and_all_method_globals(self):
        original = gzip.decompress((E / "legacy_solutions.py.gz").read_bytes()).decode()
        evidence = json.loads((E / "support_extraction.json").read_text())
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), evidence["source_sha256"])
        lines = original.splitlines(keepends=True)
        generated = ast.parse((N / "solutions.py").read_text())
        defs = {n.name: n for n in generated.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        r = evidence["record"]
        records = "".join(lines[r["start"] - 1:r["end"]]).strip() + "\n"
        self.assertEqual(records, (E / "record_support.py").read_text())
        for row in evidence["reporting_wrappers"]:
            s = "".join(lines[row["start"] - 1:row["end"]])
            self.assertEqual(hashlib.sha256(s.encode()).hexdigest(), row["sha256"])
            self.assertEqual(ast.dump(ast.parse(s).body[0]), ast.dump(defs[row["name"]]))
        old = json.loads(gzip.decompress((E / "legacy_solutions.ipynb.gz").read_bytes()))
        for c in old["cells"][2:]:
            if c["cell_type"] == "code":
                for n in ast.parse(text(c)).body:
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                        self.assertEqual(ast.dump(n), ast.dump(defs[n.name]))

    def test_history_images_and_local_navigation(self):
        images = json.loads((E / "historical_images.json").read_text())
        self.assertEqual(len(images), 3)
        for row in images:
            self.assertEqual(hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest(), row["sha256"])
        report = json.loads((N / "expected_outputs/historical_reference_metrics.json").read_text())
        self.assertEqual(report["protocol_version"], "maze-14.4-policy-controls-v2")
        self.assertFalse(report["accepted"])
        self.assertFalse(report["cuda"]["contract_passed"])
        self.assertIn("targeted patch >= 0.15", report["cuda"]["contract_failure"])
        self.assertFalse((N / "verification_report.json").exists())
        targets = [N / f"{STEM}_{k}.ipynb" for k in ["exercises", "solutions"]]
        master = ROOT / "infrastructure/chapters/chapter14_jepa_world_models/master_14_5.py"
        page = ROOT / "chapter14_jepa_world_models/instructions/pages/05_[14.5]_Maze_Transformers_and_World_Models.md"
        surfaces = [*targets, master, master.with_suffix(".ipynb"), page]
        self.assertEqual(len(surfaces), 5)
        expected = [(2, 0), (2, 0), (4, 3), (4, 3), (2, 3)]
        workbook_url = '../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-14-5'
        expected_lesson_urls = {
            f'../../../{N.relative_to(ROOT).as_posix()}/{STEM}_{kind}.ipynb'
            for kind in ['exercises', 'solutions']
        }
        for p, (book_count, image_count) in zip(surfaces, expected, strict=True):
            content = p.read_text() if p.suffix != ".ipynb" else "\n".join(
                text(c) for c in json.loads(p.read_text())["cells"] if c["cell_type"] == "markdown")
            links = re.findall(r"\]\(([^)]+)\)", content)
            links += re.findall(r"(?:src|href)=[\"']([^\"']+)[\"']", content)
            seen = []
            workbook_links = []
            for link in links:
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc:
                    continue
                relative = unquote(parsed.path)
                if not relative.lower().endswith((".ipynb", ".png", ".svg")):
                    continue
                resolved = (p.parent / relative).resolve()
                self.assertTrue(resolved.is_relative_to(ROOT), (p, link))
                self.assertTrue(resolved.is_file(), (p, link))
                if relative.lower().endswith('.ipynb'):
                    if link == workbook_url:
                        self.assertEqual(resolved, (ROOT / 'mastery_hub/units/extension_predictions/exercises.ipynb').resolve())
                        self.assertEqual(parsed.fragment, 'section-14-5')
                        workbook_links.append(link)
                        continue
                    self.assertIn(link, expected_lesson_urls, (p, link))
                    self.assertEqual(resolved.parent, N.resolve())
                    self.assertIn(resolved.name, {STEM + '_exercises.ipynb', STEM + '_solutions.ipynb'})
                seen.append(relative)
            self.assertEqual(len(workbook_links), 1 if p in targets else 0, p)
            self.assertEqual(sum(x.endswith(".ipynb") for x in seen), book_count, p)
            self.assertEqual(sum(x.lower().endswith((".png", ".svg")) for x in seen), image_count, p)
        self.assertEqual(page.name, "05_[14.5]_Maze_Transformers_and_World_Models.md")

    def test_strict_case_schema_and_no_answers(self):
        case = json.loads((E / "proposed_case.json").read_text())
        helper = ROOT / "mastery_hub/units/extension_predictions/lesson.py"
        if not helper.exists():
            helper = Path("/home/bitwise/Documents/ARENA_3.0_Extended") / helper.relative_to(ROOT)
        spec = importlib.util.spec_from_file_location("pure_schema_only", helper)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.validate_case(case), case)
        self.assertTrue(all(type(v) is str and v.strip() for v in case["config_fields"].values()))
        self.assertEqual(case["transfer_prediction_type"], "binary")
        self.assertFalse(set(case) & {"answer", "observation", "forecast", "prediction"})
        binding = json.loads((E / "case_source_inspection.json").read_text())["raw_precision_source"]
        raw = ROOT / binding["path"]
        self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), binding["sha256"])
        self.assertEqual(binding["object"], "cpu")
        self.assertEqual(binding["constrained_rate_key"], "goal_reach_rate")
        self.assertIn(binding["path"], case["config_fields"]["raw_precision_source"])
        self.assertIn(binding["sha256"], case["config_fields"]["raw_precision_source"])
        self.assertNotIn("constrained_goal_reach_rate", case["config_fields"]["predicate"])


if __name__ == "__main__":
    unittest.main()
