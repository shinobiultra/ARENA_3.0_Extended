"""Pure static monitor migration regressions. Never imports or executes taught modules."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import ast
import builtins
import copy
import gzip
import hashlib
import json
import re
import runpy
import unittest
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SECTION = ROOT / "chapter9_alignment_interpretability/exercises/part5_white_box_monitors"
EVIDENCE = ROOT / "docs/mastery/evidence/white_box_monitors_migration_20261020"
STEM = "9.5_White-Box_Monitors_and_Red-Teaming_with_Internals"
MASTER = ROOT / "infrastructure/chapters/chapter9_alignment_interpretability/master_9_5.py"
PAGE = ROOT / "chapter9_alignment_interpretability/instructions/pages/05_[9.5]_White-Box_Monitors_and_Red-Teaming_with_Internals.md"
source = lambda cell: "".join(cell["source"])
digest = lambda value: hashlib.sha256(value).hexdigest()
routes = runpy.run_path(str(EVIDENCE / "source_routes.py"))
route, inverse = routes["route"], routes["inverse"]


def book(kind):
    old = json.loads(gzip.decompress((EVIDENCE / f"legacy_{kind}.ipynb.gz").read_bytes()))
    new = json.loads((SECTION / f"{STEM}_{kind}.ipynb").read_text())
    return old, new


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


class WhiteBoxMigrationTests(unittest.TestCase):
    def test_every_original_cell_object_ids_metadata_outputs_and_inverse(self):
        mapping = json.loads((EVIDENCE / "cell_mapping.json").read_text())
        for kind, old_cells, codes, outputs, pngs in [("exercises", 38, 21, 0, 0), ("solutions", 40, 22, 13, 2)]:
            old, new = book(kind)
            rows = [row for row in mapping if row["kind"] == kind]
            assert len(old["cells"]) == old_cells == len(rows)
            assert new["metadata"] == old["metadata"]
            assert new["nbformat"] == old["nbformat"] and new["nbformat_minor"] == old["nbformat_minor"]
            for row in rows:
                before = old["cells"][row["legacy_cell"]]
                after = new["cells"][row["canonical_cell"]]
                assert source(after) == route(source(before), row["legacy_cell"])
                restored = copy.deepcopy(after)
                restored["source"] = inverse(source(after), row["legacy_cell"]).splitlines(keepends=True)
                assert restored == before, (kind, row["legacy_cell"])
            assert sum(c["cell_type"] == "code" for c in new["cells"]) == codes
            assert sum(bool(c.get("outputs")) for c in new["cells"]) == outputs
            assert sum("image/png" in o.get("data", {}) for c in new["cells"] for o in c.get("outputs", [])) == pngs
        # The two solution-only original cells must have no invented student counterpart.
        exercise_rows = [r for r in mapping if r["kind"] == "exercises"]
        assert max(r["legacy_cell"] for r in exercise_rows) == 37
        solution_rows = [r for r in mapping if r["kind"] == "solutions"]
        assert {r["legacy_cell"] for r in solution_rows} == set(range(40))

    def test_all_inline_functions_classes_stubs_and_argument_wiring_preserved(self):
        old, new = book("solutions")
        original_tree = ast.parse("\n".join(source(c) for c in old["cells"] if c["cell_type"] == "code"))
        generated = ast.parse((SECTION / "solutions.py").read_text())
        original_defs = {n.name: n for n in original_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        generated_defs = {n.name: n for n in generated.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert sum(isinstance(n, ast.FunctionDef) for n in original_defs.values()) == 21
        assert sum(isinstance(n, ast.ClassDef) for n in original_defs.values()) == 3
        for name, node in original_defs.items():
            expected = copy.deepcopy(node)
            if name == "run_gpu_test":
                expected.body[0].module = "chapter9_alignment_interpretability.exercises.part4_white_box_evals_monitors.solutions"
            assert ast.dump(expected) == ast.dump(generated_defs[name]), name
        for kind in ["exercises", "solutions"]:
            old, new = book(kind)
            oldcode = [c for c in old["cells"] if c["cell_type"] == "code"]
            newcode = [c for c in new["cells"] if c["cell_type"] == "code"]
            for before, after in zip(oldcode, newcode, strict=True):
                if "tests.test_" in source(before):
                    getcalls = lambda cell: [ast.dump(n) for n in ast.walk(ast.parse(source(cell))) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "tests"]
                    assert getcalls(before) == getcalls(after)
        old, new = book("exercises")
        count = lambda n: sum(source(c).count("raise NotImplementedError") for c in n["cells"] if c["cell_type"] == "code")
        assert count(old) == count(new) == 15

    def test_generated_python_import_safety_and_exact_adapter_dependency_closure(self):
        text = (SECTION / "solutions.py").read_text()
        tree = ast.parse(text)
        assert 'root_dir = Path(__file__).resolve().parents[3]' in text
        eager_calls = []
        def calls(expr):
            return [ast.unparse(n.func) for n in ast.walk(expr) if isinstance(n, ast.Call)]
        for node in tree.body:
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "MAIN":
                continue
            if isinstance(node, ast.FunctionDef):
                annotations = [a.annotation for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                               if a.annotation is not None]
                if node.args.vararg and node.args.vararg.annotation:
                    annotations.append(node.args.vararg.annotation)
                if node.args.kwarg and node.args.kwarg.annotation:
                    annotations.append(node.args.kwarg.annotation)
                if node.returns is not None:
                    annotations.append(node.returns)
                expressions = [*node.decorator_list, *node.args.defaults,
                               *[x for x in node.args.kw_defaults if x is not None], *annotations]
                for expr in expressions:
                    eager_calls.extend(calls(expr))
                continue
            if isinstance(node, ast.ClassDef):
                for expr in [*node.decorator_list, *node.bases, *node.body]:
                    eager_calls.extend(calls(expr))
                continue
            eager_calls.extend(calls(node))
        assert sorted(eager_calls) == sorted(["Path", "Path(__file__).resolve", "tuple"] + ["dataclass"] * 7), eager_calls
        assert not any(x.startswith(("t.", "torch.", "np.", "numpy.")) for x in eager_calls)
        allimports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert not any("arena_ext" in ast.unparse(n) for n in allimports)
        imported_names = {a.asname or a.name for n in allimports for a in n.names}
        rows = json.loads((EVIDENCE / "reference_extraction.json").read_text())
        reference = (SECTION / "validation_reference.py").read_text()
        refdefs = {n.name:n for n in ast.parse(reference).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        defs = {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for row in rows:
            span = "\n".join(reference.splitlines()[row["start"] - 1:row["end"]])
            assert digest(span.encode()) == row["sha256"]
            assert ast.dump(defs[row["name"]]) == ast.dump(refdefs[row["name"]])
            assert span in PAGE.read_text() or " ".join(span.split()) in " ".join(PAGE.read_text().split())
            assert all(row["name"] not in imported_names for _ in [0])
        for name in [r["name"] for r in rows]:
            for call in [n for n in ast.walk(defs[name]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]:
                assert call.func.id in defs or hasattr(builtins,call.func.id) or call.func.id == "dataclass", (name,call.func.id)
        # The adapter's shared calls bind the canonical inline notebook functions, never a reference import.
        required = {"build_monitor_records", "activation_matrix", "fit_white_box_monitor", "binary_auroc", "score_white_box_monitor"}
        used = {n.func.id for n in ast.walk(defs["run_toy_monitor_experiment"]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        assert required - {"binary_auroc"} <= used
        assert required <= set(defs) and not required.intersection(imported_names)

    def test_all_original_assertions_and_reference_bytes_retained(self):
        original = gzip.decompress((EVIDENCE / "legacy_tests.py.gz").read_bytes()).decode()
        expected = original.replace("from arena_ext import white_box_monitors as reference", "from . import validation_reference as reference")
        expected = expected.replace("exercises.part4_white_box_evals_monitors import (", "exercises.part5_white_box_monitors import (")
        expected = expected.replace("9.4_White-box_Evals_and_Monitors", STEM)
        expected = expected.replace('(_section_dir() / "verification_report.json")', '(_section_dir() / "historical_verification_report.json")')
        expected = expected.replace('"<summary>Solution</summary>"', '"<summary>Historical worked solution</summary>"')
        expected = expected.replace('"## Learning Objectives"', '"## Original learning objectives"')
        expected = expected.replace(
            'node.module == "part4_white_box_evals_monitors.solutions"',
            'node.module in {"part4_white_box_evals_monitors.solutions", "chapter9_alignment_interpretability.exercises.part4_white_box_evals_monitors.solutions"}'
        )
        actual = (SECTION / "tests.py").read_text()
        assert actual == expected
        before_asserts = [n for n in ast.walk(ast.parse(original)) if isinstance(n, ast.Assert)]
        after_asserts = [n for n in ast.walk(ast.parse(actual)) if isinstance(n, ast.Assert)]
        assert len(before_asserts) == len(after_asserts) >= 50
        identities = json.loads((EVIDENCE / "source_identity.json").read_text())
        ref = next(r for r in identities if r["path"] == "arena_ext/white_box_monitors.py")
        assert digest((SECTION / "validation_reference.py").read_bytes()) == ref["sha256"]

    def test_historical_artifacts_figures_and_truthful_boundaries(self):
        identities = json.loads((EVIDENCE / "source_identity.json").read_text())
        for row in identities:
            path = row["path"]
            marker = "exercises/part4_white_box_evals_monitors/"
            if marker in path and not path.endswith((".py", ".ipynb")):
                rel = Path(path.split(marker)[1]); target = SECTION / rel.with_name("historical_" + rel.name)
                assert digest(target.read_bytes()) == row["sha256"], target
            if "instructions/assets/white_box_evals_monitors" in path:
                assert digest((SECTION / "media" / Path(path).name).read_bytes()) == row["sha256"]
        for row in json.loads((EVIDENCE / "historical_figures.json").read_text()):
            assert digest((ROOT / row["path"]).read_bytes()) == row["sha256"]
        assert not (SECTION / "verification_report.json").exists()
        assert not (SECTION / "artifacts.lock.yml").exists()
        report = json.loads((SECTION / "historical_verification_report.json").read_text())
        assert report["notebook_id"] == "9_4_white_box_evals_and_monitors"
        for kind in ["exercises", "solutions"]:
            _, nb = book(kind)
            text = "\n".join(source(c) for c in nb["cells"])
            assert text.count("## Content & Learning Objectives") == 1
            assert text.count("### Prediction checkpoint ") == text.count("### Reflection checkpoint ") == 10
            assert text.count("<summary>Historical worked solution</summary>") == 8
            assert "Adversarial input optimization" in text
            assert "Human learner walkthrough" in text
            assert 'FEATURE_TO_PERTURB = "cot_unfaithful"' in text and "DELTA = 1.60" in text
            assert "## Test-only aggregation: inspect the complete assembly" in text
        text = PAGE.read_text()
        assert text.count("## Content & Learning Objectives") == 1
        assert "white_box_legacy_cell_30_output_0.png" in text
        assert "white_box_legacy_cell_33_output_1.png" in text
        assert f"part5_white_box_monitors/{STEM}_exercises.ipynb" in text
        assert f"part5_white_box_monitors/{STEM}_solutions.ipynb" in text

    def test_case_provenance_and_control_operation_bound_to_exact_legacy_cells(self):
        case = json.loads((EVIDENCE / "proposed_case.json").read_text())
        proof = json.loads((EVIDENCE / "case_source_inspection.json").read_text())
        raw = gzip.decompress((EVIDENCE / "legacy_solutions.ipynb.gz").read_bytes())
        old = json.loads(raw)
        path = ("chapter9_alignment_interpretability/exercises/part4_white_box_evals_monitors/"
                "9.4_White-box_Evals_and_Monitors_solutions.ipynb")
        row = proof["sources"]["original_solution_notebook"]
        assert row["path"] == path and row["sha256"] == digest(raw)
        assert path in case["config_fields"]["source_identity"]
        assert digest(raw) in case["config_fields"]["source_identity"]
        assert "chapter1_transformer_interp" not in case["config_fields"]["source_identity"]
        assert proof["case_sha256"] == digest((EVIDENCE / "proposed_case.json").read_bytes())
        bindings = proof["sources"]["original_cell_bindings"]
        assert [b["legacy_cell"] for b in bindings] == [4,6,9,12,15,18,21,24,27,30,33]
        for binding in bindings:
            cell = old["cells"][binding["legacy_cell"]]
            text = source(cell)
            assert binding["cell_id"] == cell.get("id")
            assert binding["source_sha256"] == digest(text.encode())
            assert binding["ast_sha256"] == digest(ast.dump(ast.parse(text)).encode())
            assert binding["first_line"] == text.splitlines()[0]
            expected = f'{binding["legacy_cell"]} / {binding["cell_id"]} / {binding["source_sha256"]}'
            assert expected in case["config_fields"]["source_cell_bindings"]
        control = next(n for n in ast.parse(source(old["cells"][27])).body
                       if isinstance(n,ast.FunctionDef) and n.name == "shuffled_label_control_scores")
        calls = [n for n in ast.walk(control) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Attribute) and n.func.attr == "roll"]
        assert len(calls) == 1 and ast.unparse(calls[0].func.value) == "train_labels"
        shifts = next(k.value.value for k in calls[0].keywords if k.arg == "shifts")
        assert shifts == 7
        assert f"train_labels.roll(shifts={shifts})" in case["config_fields"]["controls"]
        assert proof["sources"]["control_roll"]["shifts"] == shifts
        assert proof["sources"]["control_roll"]["exact_call"] == ast.unparse(calls[0])
        assert proof["sources"]["control_roll"]["cell_id"] == old["cells"][27]["id"]
        assert proof["sources"]["control_roll"]["source_sha256"] == digest(source(old["cells"][27]).encode())
        generated = ast.parse((SECTION / "solutions.py").read_text())
        actual = next(n for n in generated.body if isinstance(n,ast.FunctionDef)
                      and n.name == "shuffled_label_control_scores")
        assert ast.dump(actual) == ast.dump(control)
        play = source(old["cells"][33])
        assert 'before.item():+.3f' in play and 'after.item():+.3f' in play
        assert 'predicted_change.item():+.3f' in play
        assert "1e-6" not in case["config_fields"]["predicate"]
        assert "three-decimal precision" in case["config_fields"]["predicate"]
        assert "source-derived linear relationship" in case["config_fields"]["intervention"]
        assert "Exact numeric-linearity validation is unavailable" in case["config_fields"]["intervention"]
        precision = proof["sources"]["intervention_evidence"]
        assert precision["raw_before_after_tensors_saved"] is False
        assert precision["numeric_linearity_tolerance_outcome_saved"] is False
        assert precision["legacy_cell"] == 33 and precision["cell_id"] == old["cells"][33]["id"]
        assert not any(o.get("output_type") == "error" for o in old["cells"][33]["outputs"])

    def test_page_media_resolves_and_v1_lost_draft_predecessor_is_explicit(self):
        page = PAGE.read_text()
        identities=json.loads((EVIDENCE/'source_identity.json').read_text())
        original_assets={Path(row['path']).name:row['sha256'] for row in identities
                         if 'instructions/assets/white_box_evals_monitors' in row['path']}
        assert set(original_assets)=={'white_box_evals_monitors_validation_loop.svg',
                                     'white_box_evals_monitors_signature_result.png'}
        expected={(SECTION/'media'/name).resolve():checksum for name,checksum in original_assets.items()}
        for row in json.loads((EVIDENCE/'historical_figures.json').read_text()):
            path=(ROOT/row['path']).resolve()
            assert path.parent==(SECTION/'media').resolve()
            expected[path]=row['sha256']
        assert {path.name for path in expected}=={
            'white_box_evals_monitors_validation_loop.svg','white_box_evals_monitors_signature_result.png',
            'white_box_legacy_cell_30_output_0.png','white_box_legacy_cell_33_output_1.png'}
        assert len(resolved_local_media(page,PAGE,SECTION/'media',expected))==4
        notebook_assets={(SECTION/'media'/name).resolve():checksum for name,checksum in original_assets.items()}
        for kind in ['exercises','solutions']:
            _,nb=book(kind)
            markdown='\n'.join(source(c) for c in nb['cells'] if c['cell_type']=='markdown')
            notebook_path=SECTION/f'{STEM}_{kind}.ipynb'
            assert len(resolved_local_media(markdown,notebook_path,SECTION/'media',notebook_assets))==2
        signature=(SECTION/'media/white_box_evals_monitors_signature_result.png').resolve()
        single={signature:original_assets[signature.name]}
        valid_src='../../../'+str(signature.relative_to(ROOT))
        assert resolved_local_media(f'<img src="{valid_src}">',PAGE,SECTION/'media',single)==[signature]
        for bad_src in ('../../../chapter9_alignment_interpretability/instructions/assets/'+signature.name,
                        '../../../'+str((SECTION/'media/unknown.png').relative_to(ROOT)),
                        '../../../'+str((SECTION/'media/white_box_legacy_cell_30_output_0.png').relative_to(ROOT))):
            with self.assertRaises(AssertionError):
                resolved_local_media(f'<img src="{bad_src}">',PAGE,SECTION/'media',single)
        with self.assertRaises(AssertionError):
            resolved_local_media('<img>',PAGE,SECTION/'media',single)
        with self.assertRaises(AssertionError):
            resolved_local_media(f'<img src="{valid_src}">',PAGE,SECTION/'media',{signature:'0'*64})
        history = json.loads((EVIDENCE / "freeze_history.json").read_text())
        assert history["raw_custody_status"] == "INCOMPLETE_V1_RAW_CUSTODY_GAP_OPEN"
        assert len(history["surviving_v1_candidate_files"]) == 33
        missing = history["missing_v1_candidate_files"]
        assert len(missing) == 1
        assert missing[0]["status"] == "LOST_UNADOPTED_NEW_DRAFT_README_PREDECESSOR"
        assert missing[0]["not_reconstructed"] is True
        assert missing[0]["not_original_or_legacy_teaching"] is True
        assert missing[0]["sha256"] == "fb0a044479bcad1a1b472e48a65d7b1e92ac47e2d674e2d94ff557ca5fd2c58f"
        assert history["all_original_legacy_sources_preserved"] is True
        # The old route-repair hash still identifies its exact historical page.
        # Current canonical routing is checked above by resolved target and bytes.
        navigation=ROOT/'docs/mastery/evidence/lesson_integration_20261020/navigation_before.tar.gz'
        member=history['page_media_route_repair']['canonical_page']
        assert member==str(PAGE.relative_to(ROOT))
        with tarfile.open(navigation,'r:gz') as archive:
            matches=[item for item in archive if item.name==member]
            assert len(matches)==1 and matches[0].isfile() and matches[0].size<1_000_000
            historical=archive.extractfile(matches[0]).read()
        assert history['page_media_route_repair']['after_sha256']==digest(historical)

    def test_proposed_case_strict_schema_compile_and_no_prefill(self):
        case = json.loads((EVIDENCE / "proposed_case.json").read_text())
        expected = {'section','title','master_path','question','prediction_type','forecast_items','config_fields','controls',
                    'method_functions','exercise_path','solutions_path','evidence_class','evidence_detail','data_role',
                    'known_limitations','status','experiment','observation_prompt','transfer','transfer_prediction_type'}
        assert set(case) == expected and case['section'] == '9.5'
        assert case['prediction_type'] == case['transfer_prediction_type'] == 'binary' and case['forecast_items'] == []
        assert case['status'] == 'guided_practice' and case['data_role'] == 'historical'
        assert case['evidence_class'] == 'recorded_real_run'
        assert all(type(v) is str and v.strip() for v in case['config_fields'].values())
        assert all((ROOT / case[k]).is_file() for k in ['master_path','exercise_path','solutions_path'])
        assert not {'answer','observation','prediction','confidence','observed_result'}.intersection(case)
        text = json.dumps(case)
        assert '80 pairs' in text and '24 held-out rows' in text and 'strictly score>0' in text
        for path in [MASTER, SECTION/'solutions.py', SECTION/'tests.py', SECTION/'validation_reference.py']:
            compile(path.read_text(), str(path), 'exec')
        for kind in ['exercises','solutions']:
            _, nb = book(kind)
            for cell in nb['cells']:
                if cell['cell_type'] == 'code': compile(source(cell), '<notebook cell>', 'exec')


if __name__ == "__main__":
    unittest.main(verbosity=2)
