"""Read-only visible learner contract for the published SPEX/ProxySPEX pair.

This validates source structure and portable, report-bound receipts. It never
executes notebook code, imports model code, fits an estimator, or follows a
historical absolute cache locator. Scientific outcome validation is a separate
native-report contract; zero-query selected notebook replay is intentional.
"""

from __future__ import annotations

import ast
import base64
from collections import Counter
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import re

from scripts import evidence_bundles


CHAPTER = "chapter16_shapley_attribution_baselines"
SPEX = {
    "boolean_character", "predict_spectrum", "normalized_fwht", "shifted_subsample_queries", "bin_index",
    "bch_generator_polynomial", "make_bch_code", "bch_encode", "make_query_design", "transform_query_values",
    "bch_syndromes", "berlekamp_massey", "chien_search", "decode_bch_parity", "chase_patterns", "soft_decode_parity",
    "signed_bin_check", "decode_bin", "recover_spectrum", "fourier_to_sii", "fourier_to_shapley", "masked_word_inputs",
}
PROXY = {
    "character_matrix", "predict_spectrum", "path_indicator_fourier", "tree_to_fourier", "fit_boosted_trees",
    "predict_boosted_trees", "ensemble_to_fourier", "select_energy_support", "grouped_mask_folds", "ridge_path",
    "refine_coefficients", "fit_proxy", "stage_errors",
}
CONFIG = {
    "part18_spex": {"section": "16.18", "stem": "16.18_Find_the_Interactions_Without_Trying_Every_Subset",
                    "page": "18_[16.18]_Find_the_Interactions_Without_Trying_Every_Subset.md", "prefix": "1618_",
                    "methods": SPEX, "core": ("spex_methods.py", "native_data.py"), "cells": 31, "figures": 12,
                    "test_modules": {"tests": "spex_methods_tests", "lesson_tests": "lesson_tests", "native_tests": "native_data_tests"}},
    "part19_proxyspex": {"section": "16.19", "stem": "16.19_Explain_the_Proxy_Check_the_Model",
                         "page": "19_[16.19]_Explain_the_Proxy_Check_the_Model.md", "prefix": "1619_",
                         "methods": PROXY, "core": ("proxy_methods.py",), "cells": 27, "figures": 13,
                         "test_modules": {"tests": "proxy_methods_tests", "learner_tests": "proxy_learner_tests"}},
}
FIXTURE = f"{CHAPTER}/exercises/part18_spex/expected_outputs/native_query_fixture.json.gz"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 * 1024**2, "Missing or oversized published JSON: " + str(path))
    return json.loads(path.read_text())


def _text(value):
    return "".join(value) if isinstance(value, list) else value


def _tree(source):
    # The current Proxy setup has this display-only magic. Do not silently
    # discard arbitrary magics such as %run that could replace visible methods.
    lines = []
    for line in _text(source).splitlines():
        if line.strip() == "%matplotlib inline":
            lines.append("")
        else:
            lines.append(line)
    return ast.parse("\n".join(lines))


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _relative(name):
    _require(isinstance(name, str) and name, "Missing report path")
    value = PurePosixPath(name)
    _require(not value.is_absolute() and "\\" not in name and ":" not in name
             and all(part not in {"", ".", ".."} for part in name.split("/")), "Unsafe report path")
    return Path(*value.parts)


def _core_definitions(here, config):
    definitions = {}
    for filename in config["core"]:
        for node in _tree((here / filename).read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name in config["methods"]:
                _require(node.name not in definitions, "Duplicate taught core function")
                definitions[node.name] = node
    _require(set(definitions) == config["methods"], "Taught core function inventory changed")
    return definitions


def _answer(source, section):
    source = _text(source)
    _require("<summary>Solution</summary>" in source, "Own answer must be inside a visible Solution dropdown")
    pattern = r"<code>(.*?)</code>" if section == "16.18" else r"```python\s*\n(.*?)\n```"
    match = re.search(pattern, source, re.S)
    _require(match is not None, "Missing own-answer code")
    tree = _tree(html.unescape(match[1]) if section == "16.18" else match[1])
    _require(tree.body and all(isinstance(node, ast.FunctionDef) for node in tree.body), "Unexpected statements inside own-answer function group")
    return {node.name: node for node in tree.body}


def _test_coverage(tree, definitions, aliases):
    covered = set()
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                and call.func.value.id in aliases and call.func.attr.startswith("test_")):
            covered.update(value.id for value in [*call.args, *(kw.value for kw in call.keywords)]
                           if isinstance(value, ast.Name) and value.id in definitions)
    # Construction tests receive make_bch_code, which calls the student's
    # generator through its own globals. This is the one explicit indirect test.
    if "make_bch_code" in covered and "bch_generator_polynomial" in definitions:
        calls = [node for node in ast.walk(definitions["make_bch_code"]) if isinstance(node, ast.Call)]
        if any(isinstance(node.func, ast.Name) and node.func.id == "bch_generator_polynomial" for node in calls):
            covered.add("bch_generator_polynomial")
    return covered


def _namespace_update(statement, methods, namespace):
    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in methods:
                raise ValueError("A taught function was replaced by an assignment")
            if isinstance(target, ast.Name) and target.id == "methods":
                value = statement.value
                _require(isinstance(statement, ast.Assign) and namespace is None and isinstance(value, ast.Call)
                         and isinstance(value.func, ast.Name) and value.func.id == "SimpleNamespace" and not value.args,
                         "Student method namespace was replaced or constructed from hidden code")
                namespace = {}
                for keyword in value.keywords:
                    _require(keyword.arg is not None and isinstance(keyword.value, ast.Name), "Hidden method namespace value")
                    _require(keyword.arg == keyword.value.id, "Method namespace does not use the student's function")
                    namespace[keyword.arg] = keyword.value.id
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "methods":
                _require(namespace is not None and isinstance(statement.value, ast.Name)
                         and target.attr == statement.value.id, "Late method assignment bypasses the student function")
                namespace[target.attr] = statement.value.id
    if isinstance(statement, ast.Delete):
        _require(not any(isinstance(node, ast.Name) and node.id in methods | {"methods"}
                         for node in ast.walk(statement)), "Student functions were deleted")
    return namespace


def _visible_contract(student, solved, here, config):
    definitions = _core_definitions(here, config)
    groups, answers, pending, reconstructed = {}, {}, None, []
    namespace, seen_stubs, imported_tests = None, set(), {}
    constants, replay_conditions, plumbing_imports = {}, [], set()
    figure_names, replays, native_calls = [], {}, {}
    bridge = None
    for cell in student["cells"]:
        info = cell.get("metadata", {}).get("arena", {})
        if cell["cell_type"] == "markdown":
            if info.get("solution_for"):
                _require(pending is not None, "Own answer is detached from its exercise")
                found = _answer(cell["source"], config["section"])
                _require(info["solution_for"] == pending["answer_key"], "Own-answer metadata belongs to another exercise")
                _require(set(found) == set(pending["names"]), "Own answer does not match adjacent exercise functions")
                _require(not (set(found) & set(answers)), "Duplicate own answer")
                for name, node in found.items():
                    _require(_dump(node) == _dump(definitions[name]), "Own answer differs from frozen core: " + name)
                answers.update(found)
                pending["tree"].body = [deepcopy(found[node.name]) if isinstance(node, ast.FunctionDef) and node.name in found else node
                                        for node in pending["tree"].body]
                pending = None
            continue
        _require(cell["cell_type"] == "code", "Unsupported learner cell type")
        _require(pending is None, "Exercise lacks an adjacent own answer before the next code cell")
        tree = _tree(cell["source"])
        reconstructed.append(tree)
        names = info.get("exercise_functions", [])
        if names:
            _require(len(names) == len(set(names)) and set(names) <= config["methods"], "Unknown or duplicate exercise function")
            functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
            _require(set(functions) == set(names) and not (set(names) & seen_stubs), "Exercise function inventory differs from its metadata")
            for node in functions.values():
                signature, expected_signature = deepcopy(node), deepcopy(definitions[node.name])
                signature.body = expected_signature.body = []
                _require(_dump(signature) == _dump(expected_signature), "Student stub signature differs from its own answer")
                body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                        and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)) else node.body
                _require(len(body) == 1 and isinstance(body[0], ast.Raise)
                         and isinstance(body[0].exc, ast.Call) and isinstance(body[0].exc.func, ast.Name)
                         and body[0].exc.func.id == "NotImplementedError", "Exercise must contain an actual student implementation stub")
            _require(set(names) <= _test_coverage(tree, definitions, config["test_modules"]), "Immediate function tests do not cover the exercise")
            seen_stubs.update(names)
            pending = {"names": names, "tree": tree, "answer_key": info.get("exercise_key", names)}
            groups[tuple(names)] = cell.get("id")
        else:
            _require(not any(isinstance(node, ast.FunctionDef) and node.name in config["methods"] for node in tree.body),
                     "Taught implementation appears outside its exercise")
        calling = info.get("calling_cell")
        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    _require(alias.name not in config["methods"], "Taught method imported from hidden implementation")
                    target = alias.asname or alias.name
                    if target in {"native_learner", "SimpleNamespace"}:
                        expected_module = f"{CHAPTER}.exercises.part18_spex" if target == "native_learner" else "types"
                        _require(alias.name == target and statement.module == expected_module, "Native replay plumbing import changed")
                        plumbing_imports.add(target)
                    if target in config["test_modules"]:
                        _require(alias.name == config["test_modules"][target]
                                 and statement.module == f"{CHAPTER}.exercises.{here.name}", "Immediate test alias points to another module")
                        imported_tests[target] = alias.name
            if isinstance(statement, ast.FunctionDef):
                if statement.name == "run_native_bridge":
                    bridge = statement
                continue
            if isinstance(statement, ast.Assign):
                _require(not any(isinstance(target, ast.Name) and target.id in set(config["test_modules"]) | {"native_learner", "SimpleNamespace"}
                                 for target in statement.targets),
                         "Immediate test module was overwritten")
                if isinstance(statement.value, ast.Constant):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            constants[target.id] = statement.value.value
            if isinstance(statement, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                for nested in ast.walk(statement):
                    if isinstance(nested, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        targets = nested.targets if isinstance(nested, ast.Assign) else [nested.target]
                        _require(not any((isinstance(target, ast.Name) and target.id in config["methods"] | {"methods"})
                                 or (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "methods")
                                 for target in targets), "Conditional code can replace the student's methods")
            namespace = _namespace_update(statement, config["methods"], namespace)
            direct = statement.value if isinstance(statement, (ast.Assign, ast.Expr, ast.AnnAssign)) else None
            if isinstance(direct, ast.Call):
                name = ast.unparse(direct.func)
                if name in {"native_learner.replay_condition", "run_native_bridge"}:
                    _require(calling in {"native_study", "native_play"} and calling not in replays, "Unexpected or duplicate native replay")
                    _require(namespace is not None and all(namespace.get(name) == name for name in config["methods"]),
                             "Native replay does not receive all student methods")
                    kwargs = {kw.arg: ast.unparse(kw.value) for kw in direct.keywords}
                    expected = ("methods", "'spex'") if config["section"] == "16.18" else ("proxy_namespace", None)
                    _require(direct.args and ast.unparse(direct.args[0]) == "native_evidence"
                             and kwargs.get(expected[0]) == "methods", "Native replay bypasses the student's namespace")
                    _require((name == "native_learner.replay_condition" and kwargs.get("method") == expected[1])
                             if expected[1] else name == "run_native_bridge", "Wrong native replay method")
                    _require({"prompt_index", "seed", "budget"} <= kwargs.keys(), "Replay conditions must be explicit")
                    choices = {}
                    for keyword in direct.keywords:
                        if keyword.arg in {"prompt_index", "seed", "budget"}:
                            value = keyword.value.value if isinstance(keyword.value, ast.Constant) else constants.get(ast.unparse(keyword.value))
                            _require(type(value) is int, "Default replay choices are not visibly declared integer inputs")
                            choices[keyword.arg] = value
                    replay_conditions.append({"method": "spex" if config["section"] == "16.18" else "proxyspex", **choices})
                    replays[calling] = ast.unparse(statement.targets[0]) if isinstance(statement, ast.Assign) else None
            for call in (node for node in ast.walk(statement) if isinstance(node, ast.Call)):
                name = ast.unparse(call.func)
                if name == "show_figure":
                    _require(len(call.args) >= 2 and isinstance(call.args[1], ast.Constant)
                             and isinstance(call.args[1].value, str), "Visible figure needs a fixed output name")
                    figure_names.append(call.args[1].value + ".png")
                if calling in {"native_study", "native_play"}:
                    native_calls.setdefault(calling, []).append(call)
    _require(pending is None and seen_stubs == set(answers) == config["methods"], "Incomplete visible exercise and own-answer inventory")
    _require(imported_tests == config["test_modules"], "Immediate test imports are incomplete")
    _require(plumbing_imports == {"native_learner", "SimpleNamespace"}, "Native learner plumbing imports are incomplete")
    _require(set(replays) == {"native_study", "native_play"}, "Both actual selected native replay calls are required")
    for key, variable in replays.items():
        _require(variable is not None and any(ast.unparse(call.func) == "show_figure"
                 and any(isinstance(node, ast.Name) and node.id == variable for node in ast.walk(call)) for call in native_calls[key]),
                 "Native replay result is not used in a visible figure")
    if config["section"] == "16.19":
        source_bridge = next(node for node in _tree((here / "lesson_cells.py").read_text()).body
                             if isinstance(node, ast.FunctionDef) and node.name == "run_native_bridge")
        _require(bridge is not None and _dump(bridge) == _dump(source_bridge), "Visible native bridge differs from the source-bound namespace adapter")
        forwarded = [node for node in ast.walk(bridge) if isinstance(node, ast.Call) and ast.unparse(node.func) == "native_learner.replay_condition"]
        _require(len(forwarded) == 1 and any(kw.arg == "methods" and ast.unparse(kw.value) == "proxy_namespace" for kw in forwarded[0].keywords),
                 "Visible native bridge does not forward student functions")
    solution_cells = [cell for cell in solved["cells"] if cell["cell_type"] == "code"]
    _require(len(reconstructed) == len(solution_cells) == config["cells"], "Canonical learner code-cell count changed")
    _require(all(_dump(tree) == _dump(_tree(cell["source"])) for tree, cell in zip(reconstructed, solution_cells)),
             "Canonical solutions differ from filling the adjacent own answers")
    _require(len(figure_names) == len(set(figure_names)) == config["figures"], "Visible figure-call inventory changed")
    return solution_cells, figure_names, replay_conditions


def _replay_evidence(learner, summary, choices):
    """Validate copied replay receipts, never their historical physical locators."""
    paths = learner["portable_replay"].get("replay_records")
    if paths is None:
        receipts = learner.get("replay_receipts", {})
        _require(set(receipts) == {"native_default_replay.json", "native_play_replay.json"}, "Missing both actual ProxySPEX replay receipts")
        paths = ["native_default_replay.json", "native_play_replay.json"]
        _require(learner.get("model_queries") == 0, "Portable replay should not launch model queries")
    else:
        _require(learner["portable_replay"].get("new_model_queries") == 0, "Portable replay should not launch model queries")
    _require(isinstance(paths, list) and len(paths) == len(set(paths)) == 2, "Exactly two actual replay records are required")
    outcomes = summary.get("replay_outcomes", [])
    _require([row["record_path"] for row in outcomes] == paths, "Summary replay records differ from actual learner receipts")
    for outcome, declared in zip(outcomes, choices):
        path, hashes, replay = outcome["record_path"], outcome["hashes"], outcome["replay"]
        expected_files = {path} if path.endswith(".json") else {"record.json", "arrays.npz"}
        _require(set(hashes) == expected_files and all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) for value in hashes.values()),
                 "Replay physical identity is incomplete")
        if path.endswith(".json"):
            _require(hashes[path] == learner["replay_receipts"][path], "Copied replay identity changed")
        _require(all(replay.get(key) == value for key, value in declared.items()), "Actual replay used different visible prompt/seed/budget/method choices")
        _require(replay.get("status") == "COMPLETE" and replay.get("new_model_queries") == replay.get("new_physical_model_rows") == 0
                 and replay.get("audit_labels_received_by_fit") is False
                 and replay.get("estimator_refits") == 2 and replay.get("baseline_fits") == 4, "Actual replay fit or audit boundary is incomplete")
        comparisons = replay.get("comparisons", [])
        _require(len(comparisons) == 2 and {row.get("condition") for row in comparisons} == {"real", "shuffled"}, "Replay omits its actual shuffled control")
        stages = {"final", "constant", "additive"} if declared["method"] == "spex" else {"proxy", "extracted", "truncated", "refined", "constant", "additive"}
        for comparison in comparisons:
            _require(set(comparison.get("prediction_allclose", {})) == stages
                     and all(value is True for value in comparison["prediction_allclose"].values())
                     and comparison.get("prediction_tolerance") == {"atol": 1e-8, "rtol": 1e-7}, "Pointwise replay comparisons are missing or failed")


def _evidence(here, root, config, student_path, solution_path, cells, figure_names, choices):
    expected = here / "expected_outputs"
    learner = _json(expected / "learner_execution.json")
    fresh = _json(expected / "fresh_reproduction.json")
    summary = _json(expected / "native_summary.json")
    native = _json(expected / "native_execution.json")
    fresh_execution = _json(expected / "fresh_native_execution.json")
    report = _json(here / "verification_report.json")
    files = {}
    for item in report["report_inputs"]["files"]:
        name = item["path"]
        path = root / _relative(name)
        _require(name not in files and path.is_file() and not path.is_symlink(), "Duplicate or missing report-bound file")
        _require(_digest(path) == item["sha256"] and path.stat().st_size == item["size_bytes"], "Report-bound file changed: " + name)
        files[name] = item
    required = {str(path.relative_to(root)) for path in (student_path, solution_path, here / "lesson_cells.py", here / "build_notebooks.py",
                *(here / name for name in config["core"]), *(expected / name for name in (
                    "learner_execution.json", "fresh_reproduction.json", "native_summary.json", "native_execution.json", "fresh_native_execution.json")))}
    required.update({FIXTURE, FIXTURE + ".data-contract.json", f"{CHAPTER}/instructions/pages/{config['page']}"})
    required.add(f"{CHAPTER}/exercises/part18_spex/native_learner.py")
    required.update((here / (name + ".py")).relative_to(root).as_posix() for name in config["test_modules"].values())
    assets = root / CHAPTER / "instructions/assets"
    for name in figure_names:
        required.add((assets / (config["prefix"] + name)).relative_to(root).as_posix())
    _require(required <= files.keys(), "Report omits a canonical learner, source, receipt, portable fixture or figure")
    _require(learner.get("status") in {"passed", "FULL_LEARNER_PASS"} and learner.get("sources_unchanged") is True
             and not learner.get("changed_sources") and not learner.get("failure"), "Actual own-answer execution did not pass unchanged")
    _require(learner.get("notebook_sha256") == _digest(solution_path), "Published solutions differ from the actual executed own-answer notebook")
    _require(learner.get("executed_code_cells") == learner.get("total_code_cells") == len(cells), "Actual learner execution skipped cells")
    counts = [cell.get("execution_count") for cell in cells]
    _require(all(type(value) is int and value > 0 for value in counts) and counts == sorted(set(counts)), "Canonical solution cells lack actual ordered execution")
    images = 0
    for cell in cells:
        _require(not any(output.get("output_type") == "error" for output in cell.get("outputs", [])), "Canonical solution retains an execution error")
        for output in cell.get("outputs", []):
            if "image/png" in output.get("data", {}):
                data = base64.b64decode("".join(_text(output["data"]["image/png"]).split()), validate=True)
                evidence_bundles.inspect_member("embedded.png", data)
                images += 1
    _require(images == len(learner.get("figures", {})) == config["figures"], "Actual embedded PNG inventory changed")
    _require(set(learner["figures"]) == set(figure_names), "Learner figure names differ from visible calls")
    for name, checksum in learner["figures"].items():
        _require(_digest(assets / (config["prefix"] + name)) == checksum, "Published figure differs from actual learner bytes")
    sources = learner.get("source_inputs", learner.get("source_sha256", {}))
    _require(isinstance(sources, dict) and sources, "Missing actual learner source hashes")
    if "source_sha256_after" in learner:
        _require(sources == learner["source_sha256_after"], "Learner source before/after hashes differ")
    source_paths = set()
    for name, checksum in sources.items():
        relative = _relative(name)
        path = root / relative if relative.parts[0] == CHAPTER else here / relative
        _require(path.is_file() and _digest(path) == checksum, "Actual learner source changed: " + name)
        source_paths.add(path.resolve())
        _require(path.relative_to(root).as_posix() in files, "Report does not bind an actual learner source")
    _require({(here / name).resolve() for name in (*config["core"], "lesson_cells.py", "build_notebooks.py")} <= source_paths,
             "Learner source inventory omits visible methods or builders")
    freeze_path = root / CHAPTER / "exercises/part18_spex/native_freeze.json"
    freeze = _json(freeze_path)
    _require(freeze_path.relative_to(root).as_posix() in files, "Report does not bind the original scientific freeze")
    for name in config["core"]:
        path = here / name
        _require(freeze["source_inputs"].get(path.relative_to(root).as_posix()) == _digest(path), "Taught core differs from the frozen scientific method")
    fixture = root / FIXTURE
    evidence_bundles.validate_json_data(fixture)
    bindings = summary["bindings"]
    for key, path in (("learner_execution_sha256", expected / "learner_execution.json"),
                      ("fresh_reproduction_sha256", expected / "fresh_reproduction.json"),
                      ("native_execution_sha256", expected / "native_execution.json"),
                      ("fresh_execution_sha256", expected / "fresh_native_execution.json"), ("fixture_sha256", fixture)):
        _require(bindings.get(key) == _digest(path), "Native summary identity differs from published bytes: " + key)
    _require(summary.get("schema") == "arena.sparse-interaction.native-summary.v1" and summary.get("section") == config["section"]
             and summary.get("status") == "COMPLETE_VERIFIED_STUDY",
             "Wrong native summary identity")
    _require(all(summary["inventory"].get(key) == value for key, value in {"prompt_count": 32, "condition_count": 1152, "reference_count": 9}.items()),
             "Recorded complete study inventory was narrowed")
    replay = learner.get("portable_replay", {})
    _require(replay.get("status") == "PASS" and replay.get("no_full_study_cache_required") is True
             and replay.get("fixture_sha256") == bindings["fixture_sha256"], "Portable student replay is incomplete or needs the historical full cache")
    _replay_evidence(learner, summary, choices)
    _require(learner.get("native_execution_sha256") == bindings["native_execution_sha256"]
             and learner.get("fresh_reproduction_sha256") == bindings["fresh_reproduction_sha256"], "Learner replay uses different native evidence")
    _require(fresh.get("status") == "PASS" and not fresh.get("failures")
             and fresh.get("native_child_received_learner_payloads") is False
             and fresh.get("cached_learner_models_or_values_used_for_native_execution") is False, "Cold source-only reproduction is incomplete")
    _require(fresh.get("learner_execution_sha256") == bindings["native_execution_sha256"]
             and fresh.get("fresh_execution_sha256") == bindings["fresh_execution_sha256"], "Cold reproduction identities differ")
    for record in (native, fresh_execution):
        _require(record.get("status") == "COMPLETE" and record.get("execution_kind") == "cold_source_native"
                 and record.get("returncode") == 0 and record.get("child_received_learner_payloads") is False
                 and not any(record.get(key) for key in ("failures", "resource_violations", "changed_sources")), "External native execution did not complete")
    values = report["metrics"]["gpu_test"]
    _require(values.get("section") == config["section"] and values.get("recorded_study_reexecuted_in_notebook") is False
             and values.get("actual_learner_passed") is True and values.get("native_execution_complete") is True
             and values.get("fresh_reproduction_passed") is True, "Report confuses portable replay with a full notebook model run")
    for field, binding in (("learner_evidence_sha256", "learner_execution_sha256"), ("fresh_evidence_sha256", "fresh_reproduction_sha256"),
                           ("native_execution_sha256", "native_execution_sha256"), ("fresh_execution_sha256", "fresh_execution_sha256"),
                           ("fixture_sha256", "fixture_sha256")):
        _require(values.get(field) == bindings[binding], "Report metric identity differs from actual published evidence")
    _require(values.get("native_summary_sha256") == _digest(expected / "native_summary.json"), "Report metric binds a different native summary")


def learner_contract_blockers(path):
    """Return explicit blockers; filesystem checks use only the published repository."""
    path = Path(path)
    config = CONFIG.get(path.parent.name)
    if config is None or path.name != config["stem"] + "_exercises.ipynb":
        return ["Not a canonical 16.18/16.19 exercise notebook"]
    try:
        here, root = path.parent, path.parent.parents[2]
        solution_path = here / (config["stem"] + "_solutions.ipynb")
        student, solved = _json(path), _json(solution_path)
        cells, figures, choices = _visible_contract(student, solved, here, config)
        _evidence(here, root, config, path, solution_path, cells, figures, choices)
        return []
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, AttributeError, StopIteration) as error:
        return [str(error) or type(error).__name__]


def has_learner_contract(path):
    return not learner_contract_blockers(path)
