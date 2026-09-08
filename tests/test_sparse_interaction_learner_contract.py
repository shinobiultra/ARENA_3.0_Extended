"""Synthetic published-tree tests; build source only, never execute a notebook."""

import ast
import base64
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from scripts import evidence_bundles
from scripts import sparse_interaction_learner_contract as contract


ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _refresh_report(root, here):
    path = here / "verification_report.json"
    report = json.loads(path.read_text())
    report["report_inputs"]["files"] = [{"path": item["path"], **evidence_bundles.file_record(root / item["path"])}
                                         for item in report["report_inputs"]["files"]]
    _write(path, report)


@pytest.fixture(params=["part18_spex", "part19_proxyspex"])
def published(tmp_path, request):
    # These builders only read source/lesson text. No execute_* function is called.
    from chapter16_shapley_attribution_baselines.exercises.part18_spex import build_notebooks as spex_builder
    from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import build_notebooks as proxy_builder

    name = request.param
    config = contract.CONFIG[name]
    builder = spex_builder if name == "part18_spex" else proxy_builder
    student, solved = builder.build_notebooks()
    source_here = ROOT / contract.CHAPTER / "exercises" / name
    root = tmp_path / "portable_repository"
    here = root / contract.CHAPTER / "exercises" / name
    here.mkdir(parents=True)
    source_names = {*config["core"], "lesson_cells.py", "build_notebooks.py", "lesson.md",
                    *(name + ".py" for name in config["test_modules"].values())}
    for filename in source_names:
        shutil.copyfile(source_here / filename, here / filename)
    shared = root / contract.CHAPTER / "exercises/part18_spex"
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / contract.CHAPTER / "exercises/part18_spex/native_freeze.json", shared / "native_freeze.json")
    shutil.copyfile(ROOT / contract.CHAPTER / "exercises/part18_spex/native_learner.py", shared / "native_learner.py")
    code, figure_names, choices = contract._visible_contract(student, solved, here, config)
    image = io.BytesIO()
    Image.new("RGB", (3, 3), color="green").save(image, format="PNG")
    png = image.getvalue()
    count = 0
    for cell in solved["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        cell["execution_count"] = count
        cell["outputs"] = []
        tree = contract._tree(cell["source"])
        figures = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "show_figure"]
        for _ in figures:
            cell["outputs"].append({"output_type": "display_data", "metadata": {},
                                    "data": {"image/png": base64.b64encode(png).decode()}})
    exercises = here / (config["stem"] + "_exercises.ipynb")
    solutions = here / (config["stem"] + "_solutions.ipynb")
    _write(exercises, student)
    _write(solutions, solved)
    assets = root / contract.CHAPTER / "instructions/assets"
    assets.mkdir(parents=True)
    for name in figure_names:
        (assets / (config["prefix"] + name)).write_bytes(png)
    page = root / contract.CHAPTER / "instructions/pages" / config["page"]
    page.parent.mkdir(parents=True)
    page.write_text("Synthetic published-tree test; no native execution or scientific result claimed.\n")
    expected = here / "expected_outputs"
    expected.mkdir()
    native = {"status": "COMPLETE", "execution_kind": "cold_source_native", "returncode": 0,
              "child_received_learner_payloads": False, "failures": [], "resource_violations": [], "changed_sources": [],
              "output_directory": "/unavailable/historical/native", "synthetic_test_only": True}
    _write(expected / "native_execution.json", native)
    _write(expected / "fresh_native_execution.json", dict(native, output_directory="/unavailable/historical/fresh"))
    fresh = {"status": "PASS", "failures": [], "native_child_received_learner_payloads": False,
             "cached_learner_models_or_values_used_for_native_execution": False,
             "learner_directory": "/unavailable/historical/native", "fresh_directory": "/unavailable/historical/fresh",
             "learner_execution_sha256": contract._digest(expected / "native_execution.json"),
             "fresh_execution_sha256": contract._digest(expected / "fresh_native_execution.json")}
    _write(expected / "fresh_reproduction.json", fresh)
    fixture = root / contract.FIXTURE
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(gzip.compress(json.dumps({"schema": "synthetic_contract_test_only", "scores": [1.5, -0.25]}).encode(), mtime=0))
    evidence_bundles.seal_json_data(fixture, provenance={
        "kind": "native_execution_packaging", "measurement_claim": "Synthetic contract test; no actual native measurement.",
        "source_execution": {"absolute_path": str(expected / "native_execution.json"), **evidence_bundles.file_record(expected / "native_execution.json")},
    })
    # The portable validator must never follow this unavailable historical locator.
    sidecar = Path(str(fixture) + ".data-contract.json")
    data_contract = json.loads(sidecar.read_text())
    data_contract["provenance"]["source_execution"]["absolute_path"] = "/unavailable/historical/native/execution.json"
    _write(sidecar, data_contract)
    sources = {str((here / name).relative_to(root)): contract._digest(here / name) for name in source_names}
    sources[(shared / "native_learner.py").relative_to(root).as_posix()] = contract._digest(shared / "native_learner.py")
    learner = {"status": "passed", "sources_unchanged": True, "source_inputs": sources, "changed_sources": [],
               "executed_notebook": "/unavailable/historical/own_answer.ipynb", "notebook_sha256": contract._digest(solutions),
               "executed_code_cells": count, "total_code_cells": count, "figures": {name: hashlib_png(png) for name in figure_names},
               "native_execution_sha256": contract._digest(expected / "native_execution.json"),
               "fresh_reproduction_sha256": contract._digest(expected / "fresh_reproduction.json"),
               "model_queries": 0,
               "portable_replay": {"status": "PASS", "fixture_sha256": contract._digest(fixture),
                                   "no_full_study_cache_required": True, "new_model_queries": 0,
                                   "replay_records": ["/unavailable/default", "/unavailable/play"]}}
    if config["section"] == "16.18":
        learner["source_sha256"] = {str((root / name).relative_to(here)): value for name, value in sources.items()}
        learner["source_sha256_after"] = deepcopy(learner["source_sha256"])
        del learner["source_inputs"]
    else:
        learner["completed_from_own_adjacent_answers"] = True
        learner["replay_receipts"] = {"native_default_replay.json": "1" * 64, "native_play_replay.json": "2" * 64}
        learner["portable_replay"].pop("replay_records")
        learner["portable_replay"].pop("new_model_queries")
    _write(expected / "learner_execution.json", learner)
    bindings = {"learner_execution_sha256": contract._digest(expected / "learner_execution.json"),
                "fresh_reproduction_sha256": contract._digest(expected / "fresh_reproduction.json"),
                "native_execution_sha256": contract._digest(expected / "native_execution.json"),
                "fresh_execution_sha256": contract._digest(expected / "fresh_native_execution.json"), "fixture_sha256": contract._digest(fixture)}
    replay_paths = learner["portable_replay"].get("replay_records", ["native_default_replay.json", "native_play_replay.json"])
    replay_outcomes = []
    stages = {"final", "constant", "additive"} if config["section"] == "16.18" else {"proxy", "extracted", "truncated", "refined", "constant", "additive"}
    for locator, choice in zip(replay_paths, choices):
        hashes = {locator: learner["replay_receipts"][locator]} if locator.endswith(".json") else {"record.json": "1" * 64, "arrays.npz": "2" * 64}
        replay_outcomes.append({"record_path": locator, "hashes": hashes, "replay": {
            **choice, "status": "COMPLETE", "new_model_queries": 0, "new_physical_model_rows": 0,
            "audit_labels_received_by_fit": False, "estimator_refits": 2, "baseline_fits": 4,
            "comparisons": [{"condition": condition, "prediction_allclose": dict.fromkeys(stages, True),
                             "prediction_tolerance": {"atol": 1e-8, "rtol": 1e-7}} for condition in ("real", "shuffled")],
        }})
    _write(expected / "native_summary.json", {"schema": "arena.sparse-interaction.native-summary.v1", "section": config["section"],
                                              "status": "COMPLETE_VERIFIED_STUDY",
                                              "bindings": bindings, "inventory": {"prompt_count": 32, "condition_count": 1152, "reference_count": 9},
                                              "replay_outcomes": replay_outcomes})
    all_files = [path for path in root.rglob("*") if path.is_file()]
    report = {"metrics": {"gpu_test": {"section": config["section"], "actual_learner_passed": True,
                                        "native_execution_complete": True, "fresh_reproduction_passed": True,
                                        "learner_evidence_sha256": bindings["learner_execution_sha256"],
                                        "fresh_evidence_sha256": bindings["fresh_reproduction_sha256"],
                                        "native_summary_sha256": contract._digest(expected / "native_summary.json"),
                                        **{key: bindings[key] for key in ("native_execution_sha256", "fresh_execution_sha256", "fixture_sha256")},
                                        "recorded_study_reexecuted_in_notebook": False}},
              "report_inputs": {"files": [{"path": path.relative_to(root).as_posix(), **evidence_bundles.file_record(path)} for path in all_files]}}
    _write(here / "verification_report.json", report)
    return {"root": root, "here": here, "exercises": exercises, "solutions": solutions, "config": config}


def hashlib_png(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def test_published_contract_works_without_historical_cache_or_notebook_model_queries(published):
    blockers = contract.learner_contract_blockers(published["exercises"])
    assert blockers == [], blockers
    assert contract.has_learner_contract(published["exercises"])


@pytest.mark.parametrize("mutation", ["missing_answer", "test_in_dead_branch", "namespace_bypass", "late_namespace_bypass",
                                      "missing_native_play", "hidden_native_namespace", "changed_solution", "stub_signature"])
def test_visible_method_and_replay_tampering_is_rejected_even_with_refreshed_report(published, mutation):
    path = published["exercises"]
    notebook = json.loads(path.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    exercise = next(cell for cell in code if cell.get("metadata", {}).get("arena", {}).get("exercise_functions"))
    if mutation == "missing_answer":
        index = next(i for i, cell in enumerate(notebook["cells"]) if cell.get("metadata", {}).get("arena", {}).get("solution_for"))
        notebook["cells"].pop(index)
    elif mutation == "test_in_dead_branch":
        tree = contract._tree(exercise["source"])
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        statements = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]
        tree.body = definitions + [ast.If(test=ast.Constant(False), body=statements, orelse=[])]
        exercise["source"] = ast.unparse(ast.fix_missing_locations(tree))
    elif mutation == "namespace_bypass":
        cell = next(cell for cell in code if cell.get("metadata", {}).get("arena", {}).get("calling_cell")
                    in {"methods_namespace", "student_namespace"})
        method = "recover_spectrum" if published["config"]["section"] == "16.18" else "fit_proxy"
        cell["source"] = cell["source"].replace(f"{method}={method}", f"{method}=hidden.{method}")
    elif mutation == "late_namespace_bypass":
        cell = next(cell for cell in code if cell.get("metadata", {}).get("arena", {}).get("calling_cell") == "native_study")
        method = next(iter(published["config"]["methods"]))
        cell["source"] = f"if True:\n    methods.{method} = hidden.{method}\n" + cell["source"]
    elif mutation == "missing_native_play":
        cell = next(cell for cell in code if cell.get("metadata", {}).get("arena", {}).get("calling_cell") == "native_play")
        cell["source"] = "print('Skipped actual replay')"
    elif mutation == "hidden_native_namespace":
        cell = next(cell for cell in code if cell.get("metadata", {}).get("arena", {}).get("calling_cell") == "native_study")
        cell["source"] = cell["source"].replace("methods=methods", "methods=hidden").replace("proxy_namespace=methods", "proxy_namespace=hidden")
    elif mutation == "changed_solution":
        solved = json.loads(published["solutions"].read_text())
        next(cell for cell in solved["cells"] if cell["cell_type"] == "code")["source"] += "\nprint('Changed actual solution')"
        _write(published["solutions"], solved)
    else:
        tree = contract._tree(exercise["source"])
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        function.args.args.append(ast.arg(arg="unrelated_extra_parameter"))
        exercise["source"] = ast.unparse(ast.fix_missing_locations(tree))
    _write(path, notebook)
    _refresh_report(published["root"], published["here"])
    assert contract.learner_contract_blockers(path)


@pytest.mark.parametrize("mutation", ["source", "fixture", "receipt", "output_error", "output_missing", "manifest_omission"])
def test_evidence_and_output_tampering_rejects(published, mutation):
    here, root = published["here"], published["root"]
    if mutation == "source":
        source = here / published["config"]["core"][0]
        source.write_text(source.read_text() + "\nSOURCE_DRIFT = True\n")
    elif mutation == "fixture":
        with (root / contract.FIXTURE).open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "receipt":
        target = here / "expected_outputs/fresh_reproduction.json"
        value = json.loads(target.read_text())
        value["status"] = "FAIL"
        _write(target, value)
    elif mutation.startswith("output_"):
        target = published["solutions"]
        value = json.loads(target.read_text())
        cell = next(cell for cell in value["cells"] if cell["cell_type"] == "code" and cell["outputs"])
        if mutation == "output_error":
            cell["outputs"].append({"output_type": "error", "ename": "RuntimeError", "evalue": "synthetic failure"})
        else:
            cell["outputs"] = []
        _write(target, value)
    else:
        target = here / "verification_report.json"
        value = json.loads(target.read_text())
        value["report_inputs"]["files"] = [item for item in value["report_inputs"]["files"] if item["path"] != contract.FIXTURE]
        _write(target, value)
    assert not contract.has_learner_contract(published["exercises"])


def test_unrelated_and_unpublished_drafts_are_not_certified(tmp_path):
    assert not contract.has_learner_contract(tmp_path / "other_exercises.ipynb")
    for directory, config in contract.CONFIG.items():
        path = ROOT / contract.CHAPTER / "exercises" / directory / (config["stem"] + "_exercises.ipynb")
        if not path.exists():
            assert not contract.has_learner_contract(path)


@pytest.mark.parametrize("mutation", ["audit_leak", "wrong_visible_prompt", "missing_shuffle", "prediction_drift", "false_full_notebook_run"])
def test_copied_replay_and_scope_tampering_rejects_without_opening_caches(published, mutation):
    here, root = published["here"], published["root"]
    summary_path = here / "expected_outputs/native_summary.json"
    summary = json.loads(summary_path.read_text())
    replay = summary["replay_outcomes"][0]["replay"]
    if mutation == "audit_leak":
        replay["audit_labels_received_by_fit"] = True
    elif mutation == "wrong_visible_prompt":
        replay["prompt_index"] = 31
    elif mutation == "missing_shuffle":
        replay["comparisons"] = replay["comparisons"][:1]
    elif mutation == "prediction_drift":
        stages = replay["comparisons"][0]["prediction_allclose"]
        stages[next(iter(stages))] = False
    _write(summary_path, summary)
    report_path = here / "verification_report.json"
    report = json.loads(report_path.read_text())
    report["metrics"]["gpu_test"]["native_summary_sha256"] = contract._digest(summary_path)
    if mutation == "false_full_notebook_run":
        report["metrics"]["gpu_test"]["recorded_study_reexecuted_in_notebook"] = True
    _write(report_path, report)
    _refresh_report(root, here)
    assert not contract.has_learner_contract(published["exercises"])
