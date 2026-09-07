"""Static, analytic, and synthetic contracts for the pre-freeze 16.16 harness."""

from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import (
    run_native,
)


HERE = Path(__file__).resolve().parent
METHOD_NAMES = (
    "mobius_transform",
    "shapley_taylor_interactions",
    "faith_design_matrix",
    "faith_kernel_weights",
    "faith_shap_interactions",
    "faith_shap_closed_form",
    "reconstruct_game",
    "faithfulness_error",
    "masked_model_game",
    "contextual_pair_synergy",
)
METHOD_HELPERS = {
    "_interaction_counts",
    "_interaction_float_input",
    "_interaction_game_batches",
    "_interaction_model",
    "_interaction_model_output",
    "_interaction_n_players",
    "_interaction_order",
    "_interaction_preserve_model_state",
    "_interaction_tensor",
}
FIGURE_STEMS = (
    "01_exact_allocation_ledger",
    "02_why_faith_moves_singletons",
    "10_dividend_play",
    "03_native_game_and_controls",
    "04_fixed_row_interaction_graph",
    "05_allrow_pair_heatmaps",
    "06_heldout_pair_order_capture",
    "07_order_two_reconstruction",
    "08_background_changes_the_game",
    "09_contextual_pair_interventions",
    "11_donor_context_play",
)


def _module_source(name: str) -> str:
    return (HERE / name).read_text()


def _function(source: str, name: str) -> ast.FunctionDef:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing visible function: {name}")


def _calls(node: ast.AST) -> list[str]:
    result = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        if isinstance(item.func, ast.Name):
            result.append(item.func.id)
        elif isinstance(item.func, ast.Attribute):
            result.append(item.func.attr)
    return result


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": run_native.digest(path),
        "size_bytes": path.stat().st_size,
    }


def test_zero_run_freeze_guard_is_live_before_and_after_freeze():
    freeze = HERE / "protocol_freeze.json"
    if not freeze.is_file():
        with pytest.raises(FileNotFoundError, match="cannot fit or evaluate"):
            run_native.validate_freeze()
        return
    record = run_native.validate_freeze()
    assert record["native_fits_before_freeze"] == 0
    assert record["native_game_runs_before_freeze"] == 0
    assert record["native_attribution_runs_before_freeze"] == 0
    assert record["audit_loads_before_freeze"] == 0
    if record["independent_approval"] is not True:
        assert record["independent_approval"] is False
        assert record["execution_scope"] == "development_only"
        resolution = json.loads((HERE / "pre_run_resolution.json").read_text())
        assert resolution["independent_recheck"] == "pending"
        assert resolution["publication_authorized"] is False


def test_all_21_known_model_hashes_are_predeclared_and_required():
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    states = manifest["known_16_14_evidence"]["model_state_hashes"]
    assert tuple(states) == run_native.MODEL_KEYS
    assert sum(len(record) for record in states.values()) == 21
    passing = {
        "status": "PASS",
        "expected_model_state_hashes": states,
        "model_state_hashes": copy.deepcopy(states),
        "mismatches": (),
    }
    assert run_native._state_hash_gate(passing) == 21
    failing = copy.deepcopy(passing)
    failing["model_state_hashes"]["clean_0"]["raw_folded_state"] = "0" * 64
    with pytest.raises(ValueError, match="differ"):
        run_native._state_hash_gate(failing)


def test_manifest_declares_exact_ten_physical_shards_and_aliases():
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    contract = manifest["physical_hybrid_shards"]
    assert contract["count"] == 10
    assert [item["artifact"] for item in contract["shards"]] == list(
        run_native.EXTERNAL_DETERMINISTIC_ARTIFACTS
    )
    expected_paths = [
        ".".join(run_native._main_hybrid_path(split, background))
        for split in ("explain", "audit")
        for background in run_native.BACKGROUND_SIZES
    ]
    assert [item["payload_path"] for item in contract["shards"]] == expected_paths
    for item in contract["shards"]:
        split = "explain" if item["stage"] == "prepared_before_audit" else "audit"
        background = (
            "primary64"
            if item["key"].startswith("primary_")
            else next(name for name in tuple(run_native.BACKGROUND_SIZES)[1:] if name in item["key"])
        )
        assert item["shape"] == list(run_native._expected_hybrid_shape(split, background))
        assert item["dtype"] == "float64"
        assert item["elements"] == int(np.prod(item["shape"]))
    assert tuple(contract["descriptor_schema"]) == (
        "kind",
        "artifact",
        "manifest",
        "payload_path",
        "array_kind",
        "shape",
        "dtype",
        "elements",
        "value_sha256",
        "artifact_sha256",
        "size_bytes",
    )
    assert contract["descriptor_kind"] == "external_tensor"
    assert contract["shallow_replacement_only"] is True
    assert len(contract["aliases"]) == 2
    assert contract["view_descriptor_kind"] == "external_tensor_view"
    assert contract["logical_split_mapping_count"] == 6
    assert len(contract["logical_split_mappings"]) == 6
    assert all(item["no_tensor_concatenation"] is True for item in contract["logical_split_mappings"])


def test_public_method_api_is_exact_and_visible():
    methods = __import__(
        "chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.interaction_methods",
        fromlist=["*"],
    )
    public = tuple(name for name in METHOD_NAMES if callable(getattr(methods, name, None)))
    assert public == METHOD_NAMES
    signatures = {
        "mobius_transform": ("game_values", "n_players"),
        "shapley_taylor_interactions": ("mobius", "n_players", "order"),
        "faith_design_matrix": ("n_players", "max_order"),
        "faith_kernel_weights": ("n_players",),
        "faith_shap_interactions": ("game_values", "n_players", "max_order"),
        "faith_shap_closed_form": ("mobius", "n_players", "max_order"),
        "reconstruct_game": ("coefficients", "n_players"),
        "faithfulness_error": ("game_values", "coefficients", "n_players"),
        "masked_model_game": ("model", "x", "reference", "coalition_batch_size"),
        "contextual_pair_synergy": ("game_values", "pair", "n_players", "context_masks"),
    }
    for name, expected in signatures.items():
        assert tuple(inspect.signature(getattr(methods, name)).parameters) == expected


def test_recursive_manifest_hashes_every_synthetic_leaf(tmp_path: Path):
    payload = {
        "tensor": torch.arange(12, dtype=torch.float64).reshape(3, 4),
        "numpy": np.arange(5, dtype=np.int64),
        "nested": {"flag": False, "label": "complete"},
    }
    artifact, manifest = run_native.save_payload(tmp_path, "payload.pt", payload)
    assert manifest["artifact_sha256"] == run_native.digest(artifact)
    assert manifest["counts"] == {"arrays": 2, "array_elements": 17, "records": 6}
    records = {record["path"]: record for record in manifest["records"]}
    assert records["/tensor"]["shape"] == [3, 4]
    assert records["/tensor"]["dtype"] == "float64"
    assert records["/numpy"]["dtype"] == "int64"
    changed = copy.deepcopy(payload)
    changed["tensor"][0, 0] = 99
    assert list(run_native._manifest_records(changed)) != list(
        run_native._manifest_records(payload)
    )
    with pytest.raises(ValueError, match="timing must live only"):
        run_native.save_payload(tmp_path, "invalid.pt", {"timings": {"fit_seconds": 1.0}})
    array = torch.arange(2)
    clean, timings = run_native.split_measured_timings(
        {"science": array, "reference": {"timings": {"package_seconds": 1.25}}}
    )
    assert set(clean) == {"science", "reference"} and clean["reference"] == {}
    assert timings == {"/reference/timings": {"package_seconds": 1.25}}
    assert clean["science"] is array


def test_external_tensor_and_view_restore_rehash_bytes_and_values(tmp_path: Path):
    value = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    _, manifest = run_native.save_payload(tmp_path, "primary_explain_hybrid.pt", value)
    direct = run_native._external_reference(value, manifest, ("primary64", "hybrid_prediction"))
    alias = run_native._external_view_reference(
        value[0], manifest, ("background_variants", "primary64", "hybrid_prediction")
    )
    assert direct["kind"] == "external_tensor"
    assert alias["kind"] == "external_tensor_view"
    assert alias["selector"] == {"axis": 0, "index": 0}
    assert alias["target_payload_path"] == "/primary64/hybrid_prediction"
    assert torch.equal(run_native.restore_external_tensor(tmp_path, direct), value)
    assert torch.equal(run_native.restore_external_tensor(tmp_path, alias), value[0])
    corrupted = dict(direct, value_sha256="0" * 64)
    with pytest.raises(ValueError, match="value identity"):
        run_native.restore_external_tensor(tmp_path, corrupted)


def test_shallow_replacement_preserves_unrelated_tensor_identity():
    hybrid = torch.ones(2, 3)
    small = torch.arange(4)
    payload = {"primary64": {"hybrid_prediction": hybrid, "coalition_values": small}}
    descriptor = {"kind": "external_tensor", "artifact": "x.pt"}
    replaced = run_native._replace_paths(
        payload, {("primary64", "hybrid_prediction"): descriptor}
    )
    assert replaced["primary64"]["hybrid_prediction"] is descriptor
    assert replaced["primary64"]["coalition_values"] is small
    assert payload["primary64"]["hybrid_prediction"] is hybrid


def test_logical_all_rows_points_to_existing_split_descriptors():
    def descriptor(name: str) -> dict[str, str]:
        return {"kind": "external_tensor", "artifact": name}

    explain_main = descriptor("primary_explain_hybrid.pt")
    explain_view = {"kind": "external_tensor_view", "artifact": "primary_explain_hybrid.pt"}
    audit_main = descriptor("primary_audit_hybrid.pt")
    audit_view = {"kind": "external_tensor_view", "artifact": "primary_audit_hybrid.pt"}
    prepared = {
        "primary64": {"hybrid_prediction": explain_main},
        "background_variants": {
            "primary64": {"hybrid_prediction": explain_view},
            **{
                name: {"hybrid_prediction": descriptor(f"{name}_explain_hybrid.pt")}
                for name in tuple(run_native.BACKGROUND_SIZES)[1:]
            },
        },
    }
    logical = lambda: {
        "storage": "split_external_shards",
        "explain": torch.empty(0),
        "audit": torch.empty(0),
    }
    evaluated = {
        "primary64_audit": {"hybrid_prediction": audit_main},
        "background_variants_audit": {
            "primary64": {"hybrid_prediction": audit_view},
            **{
                name: {"hybrid_prediction": descriptor(f"{name}_audit_hybrid.pt")}
                for name in tuple(run_native.BACKGROUND_SIZES)[1:]
            },
        },
        "all_rows": {
            "primary64": {"hybrid_prediction": logical()},
            "background_variants": {
                name: {"hybrid_prediction": logical()} for name in run_native.BACKGROUND_SIZES
            },
        },
    }
    result = run_native._replace_logical_all_rows(evaluated, prepared)
    primary = result["all_rows"]["primary64"]["hybrid_prediction"]
    assert primary["explain"] is explain_main and primary["audit"] is audit_main
    clean = result["all_rows"]["background_variants"]["primary64"]["hybrid_prediction"]
    assert clean["explain"] is explain_view and clean["audit"] is audit_view
    for name in tuple(run_native.BACKGROUND_SIZES)[1:]:
        item = result["all_rows"]["background_variants"][name]["hybrid_prediction"]
        assert item["explain"] is prepared["background_variants"][name]["hybrid_prediction"]
        assert item["audit"] is evaluated["background_variants_audit"][name]["hybrid_prediction"]


def test_physical_seal_covers_fitted_prepared_and_all_explain_shards(tmp_path: Path):
    states = json.loads((HERE / "data_manifest.json").read_text())["known_16_14_evidence"][
        "model_state_hashes"
    ]
    reproduction = {
        "status": "PASS",
        "expected_model_state_hashes": states,
        "model_state_hashes": copy.deepcopy(states),
        "mismatches": (),
    }
    fitted = {"reproduction": reproduction}
    prepared = {
        "phase": "prepared_before_audit",
        "audit_used_for_selection": False,
        "reproduction": reproduction,
        "explain_source_ids": torch.arange(3, dtype=torch.int64),
    }
    run_native.save_payload(tmp_path, "fitted.pt", fitted)
    run_native.save_payload(tmp_path, "prepared.pt", prepared)
    for background in run_native.BACKGROUND_SIZES:
        run_native.save_payload(
            tmp_path,
            run_native.hybrid_artifact_name("explain", background),
            torch.zeros(1, dtype=torch.float64),
        )
    boundary = run_native.write_before_audit(tmp_path, fitted, prepared)
    assert boundary["state_hash_count"] == 21
    assert len(boundary["artifacts"]) == 7
    run_native._verify_boundary(tmp_path)
    with (tmp_path / "prepared.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="Pre-audit artifact changed"):
        run_native._verify_boundary(tmp_path)


def test_complete_tiny_ten_shard_directory_roundtrips_and_detects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import (
        reproduce_native,
    )

    tiny_rows = {"explain": 2, "audit": 1}
    tiny_backgrounds = {
        "primary64": 2,
        "nested16": 1,
        "nested128": 3,
        "independent64_a": 2,
        "independent64_b": 2,
    }
    monkeypatch.setattr(run_native, "SPLIT_ROWS", tiny_rows)
    monkeypatch.setattr(run_native, "BACKGROUND_SIZES", tiny_backgrounds)
    monkeypatch.setattr(run_native, "COALITIONS", 4)
    monkeypatch.setattr(reproduce_native, "BACKGROUND_SIZES", tiny_backgrounds)
    states = json.loads((HERE / "data_manifest.json").read_text())["known_16_14_evidence"][
        "model_state_hashes"
    ]
    reproduction = {
        "status": "PASS",
        "expected_model_state_hashes": states,
        "model_state_hashes": copy.deepcopy(states),
        "mismatches": (),
    }

    def phase_payload(split: str) -> dict:
        rows = tiny_rows[split]
        primary = torch.arange(
            7 * rows * 4 * 2, dtype=torch.float64
        ).reshape(7, rows, 4, 2)
        variants = {"primary64": {"hybrid_prediction": primary[0]}}
        for background, donors in tuple(tiny_backgrounds.items())[1:]:
            variants[background] = {
                "hybrid_prediction": torch.arange(
                    rows * 4 * donors, dtype=torch.float64
                ).reshape(rows, 4, donors)
            }
        primary_key = "primary64" if split == "explain" else "primary64_audit"
        variants_key = (
            "background_variants" if split == "explain" else "background_variants_audit"
        )
        return {
            primary_key: {"hybrid_prediction": primary},
            variants_key: variants,
        }

    fitted = {
        "models": {key: {"kind": "synthetic"} for key in run_native.MODEL_KEYS},
        "train_ids": torch.arange(3, dtype=torch.int64),
        "reproduction": reproduction,
    }
    prepared = {
        "phase": "prepared_before_audit",
        "audit_used_for_selection": False,
        "reproduction": reproduction,
        "explain_source_ids": torch.tensor([3, 4], dtype=torch.int64),
        **phase_payload("explain"),
    }
    run_native.persist_prepared(tmp_path, fitted, prepared)
    evaluated = {
        "phase": "evaluated_after_audit",
        "audit_used_for_selection": False,
        "audit_source_ids": torch.tensor([5], dtype=torch.int64),
        **phase_payload("audit"),
    }
    evaluated["all_rows"] = {
        "primary64": {
            "hybrid_prediction": {
                "storage": "split_external_shards",
                "logical_shape": (7, 3, 4, 2),
                "dtype": "float64",
                "explain": prepared["primary64"]["hybrid_prediction"],
                "audit": evaluated["primary64_audit"]["hybrid_prediction"],
            }
        },
        "background_variants": {},
    }
    for background, donors in tiny_backgrounds.items():
        evaluated["all_rows"]["background_variants"][background] = {
            "hybrid_prediction": {
                "storage": "split_external_shards",
                "logical_shape": (3, 4, donors),
                "dtype": "float64",
                "explain": prepared["background_variants"][background]["hybrid_prediction"],
                "audit": evaluated["background_variants_audit"][background]["hybrid_prediction"],
            }
        }
    summary = run_native.persist_completed(
        tmp_path,
        fitted,
        prepared,
        evaluated,
        {"status": "PASS"},
        {"status": "complete"},
        {"phase": "play"},
        timings={"total_seconds": 1.0},
    )
    assert summary["full_hybrid_coordinates"] == sum(
        int(np.prod(run_native._expected_hybrid_shape(split, background)))
        for split in ("explain", "audit")
        for background in tiny_backgrounds
    )
    manifests = reproduce_native.validate_artifact_directory(tmp_path)
    assert set(manifests) == set(run_native.DETERMINISTIC_ARTIFACTS) | {
        run_native.TIMING_ARTIFACT
    }
    shard = tmp_path / "nested16_audit_hybrid.pt"
    with shard.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="size changed"):
        reproduce_native.validate_artifact_directory(tmp_path)


def test_runner_has_literal_seal_then_audit_load_and_no_cache_input():
    source = _module_source("run_native.py")
    node = _function(source, "run")
    segment = ast.get_source_segment(source, node) or ""
    seal = segment.index("persist_prepared(directory, fitted, prepared)")
    audit = segment.index('data_module.load_split("audit")')
    evaluate = segment.index("study.evaluate_study(")
    complete = segment.index("persist_completed(")
    assert seal < audit < evaluate < complete
    assert "cached_model_inputs_used" in segment
    assert "load_payload" not in segment[:seal]
    assert "fit_known_models(train)" in segment
    assert "verify_known_reproduction" in segment


def test_reproduction_uses_distinct_source_only_subprocess():
    source = _module_source("reproduce_native.py")
    node = _function(source, "reproduce")
    assert "subprocess.run" in ast.unparse(node)
    segment = ast.get_source_segment(source, node) or ""
    assert 'str(HERE / "run_native.py")' in segment
    assert "fitted.pt" not in segment
    assert '"CUDA_VISIBLE_DEVICES": ""' in segment
    assert '"OMP_NUM_THREADS": "4"' in segment
    assert '"child_received_learner_payloads": False' in segment


def test_checkpoint_own_answers_define_and_use_every_taught_method():
    build = __import__(
        "chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.build_notebooks",
        fromlist=["*"],
    )
    learner, solved, blockers = build.build_checkpoint()
    learner_ids = [cell["id"] for cell in learner["cells"]]
    solved_ids = [cell["id"] for cell in solved["cells"]]
    assert len(learner_ids) - len(solved_ids) == 8
    assert set(solved_ids) < set(learner_ids)
    assert all(
        "<summary>Solution</summary>" in "".join(cell["source"])
        for cell in learner["cells"]
        if cell["id"] not in set(solved_ids)
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in solved["cells"] if cell["cell_type"] == "code"
    )
    tree = ast.parse(code)
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    loaded_names = [
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    ]
    assert set(METHOD_NAMES) <= defined
    assert all(loaded_names.count(name) >= 1 for name in METHOD_NAMES)
    assert len(build.EXERCISE_GROUPS) == 8
    assert tuple(build.FIGURE_STEMS) == FIGURE_STEMS
    if not (HERE / "protocol_freeze.json").is_file():
        assert blockers, "The pre-freeze checkpoint should remain explicitly noncanonical."
    else:
        assert not blockers, "A frozen learner cannot retain checkpoint blockers."


def test_learner_imports_only_named_method_plumbing():
    source = (HERE / "lesson.md").read_text()
    cells = __import__(
        "chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap.build_notebooks",
        fromlist=["lesson_cells"],
    ).lesson_cells(allow_unresolved=True)
    code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.ImportFrom):
            continue
        if (node.module or "").endswith("interaction_methods"):
            imported = {alias.name for alias in node.names}
            assert imported <= METHOD_HELPERS
            assert "*" not in imported
    # Draft status is enforced by the executable freeze check, not a prose marker.
    assert "freeze = validate_freeze()" in code
    assert "CHECKPOINT BLOCKER" not in source


def test_visual_review_gate_rehashes_every_file(tmp_path: Path):
    from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import (
        publish_study,
    )

    learner, fresh = tmp_path / "learner", tmp_path / "fresh"
    (learner / "figures").mkdir(parents=True)
    fresh.mkdir()
    completed = learner / "completed_learner.ipynb"
    completed.write_text("{}")
    reproduction = fresh / "reproduction.json"
    reproduction.write_text("{}")
    browser = learner / "browser_report.json"
    stdout = learner / "browser_stdout.txt"
    independent = learner / "figure_review.md"
    for path, value in ((browser, "report"), (stdout, "stdout"), (independent, "review")):
        path.write_text(value)
    screenshots = {}
    contacts = {}
    for label, target in (("screen", screenshots), ("contact", contacts)):
        path = learner / f"{label}.png"
        path.write_bytes(label.encode())
        target[label] = _record(path)
    figures = {}
    for name in publish_study.EXPECTED_FIGURES:
        path = learner / "figures" / name
        path.write_bytes(name.encode())
        figures[name] = run_native.digest(path)
    review = {
        "status": "PASS",
        "browser_math_and_dropdowns_checked": True,
        "lesson_sha256": run_native.digest(HERE / "lesson.md"),
        "completed_learner_sha256": run_native.digest(completed),
        "figures": figures,
        "evidence_files": {
            "completed_notebook": _record(completed),
            "browser_report": _record(browser),
            "browser_stdout": _record(stdout),
            "independent_figure_review": _record(independent),
            "fresh_reproduction": _record(reproduction),
            "screenshots": screenshots,
            "inspection_contacts": contacts,
        },
    }
    (learner / "visual_review.json").write_text(json.dumps(review))
    execution = {"figures": figures}
    publish_study.validate_visual_review(learner, fresh, execution)
    (learner / "screen.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size changed|bytes changed"):
        publish_study.validate_visual_review(learner, fresh, execution)


def test_publisher_rehashes_review_sets_instead_of_accepting_nonempty_maps():
    source = _module_source("publish_study.py")
    node = _function(source, "validate_visual_review")
    calls = _calls(node)
    assert "_resolve_review_file" in calls
    segment = ast.get_source_segment(source, node) or ""
    for key in (
        "completed_notebook",
        "browser_report",
        "browser_stdout",
        "screenshots",
        "inspection_contacts",
        "independent_figure_review",
        "fresh_reproduction",
    ):
        assert key in segment
    assert "figure_hashes" in segment and "digest" in calls
    publish = _function(source, "publish")
    publish_segment = ast.get_source_segment(source, publish) or ""
    assert "student_by_id" in publish_segment and "solved_by_id" in publish_segment
    assert "validate_page_surface(page)" in publish_segment
    assert "zip(" not in publish_segment


def test_owned_sources_compile_without_scientific_execution():
    for name in (
        "run_native.py",
        "execute_learner.py",
        "reproduce_native.py",
        "publish_study.py",
        "interaction_contracts.py",
    ):
        compile(_module_source(name), str(HERE / name), "exec")


def test_exercise_help_cannot_move_to_another_exercise_or_below_its_answer():
    from . import build_notebooks as build

    source = (HERE / "lesson.md").read_text()
    build.validate_exercise_surface(source)
    changed = source.replace("<summary>Help - construct one intact hybrid</summary>",
                             "<summary>Implementation note</summary>").replace(
        "<summary>Help - turn donor predictions into a complete game</summary>",
        "<summary>Batching note</summary>")
    with pytest.raises(ValueError, match="exercise 7 needs at least one Help"):
        build.validate_exercise_surface(changed)
    expected = "<summary>Expected output</summary>"
    solution = "<summary>Solution</summary>"
    changed = source.replace(expected, "<summary>Swap target</summary>", 1)
    changed = changed.replace(solution, expected, 1).replace("<summary>Swap target</summary>", solution, 1)
    with pytest.raises(ValueError, match="expected output, help, interpretation, then solution"):
        build.validate_exercise_surface(changed)


def test_page_keeps_expandable_provided_source_and_every_student_operation():
    from . import build_notebooks as build

    learner, solved, _ = build.build_checkpoint()
    code_cells = [cell for cell in solved["cells"] if cell["cell_type"] == "code"]
    for cell in code_cells:
        definitions = {node.name for node in ast.parse("".join(cell["source"])).body
                       if isinstance(node, ast.FunctionDef)}
        if definitions.intersection(METHOD_NAMES):
            assert not cell["metadata"].get("jupyter", {}).get("source_hidden", False)
    pieces = [build.render_instruction_code(cell) if cell["cell_type"] == "code"
              else "".join(cell["source"]) for cell in learner["cells"]]
    page = "\n\n".join(pieces)
    build.validate_page_surface(page)
    provided = next(cell for cell in learner["cells"]
                    if cell["metadata"].get("arena", {}).get("provided_code") ==
                    "Provided explanation preparation and pair ranking")
    block = build.render_instruction_code(provided)
    assert block.startswith("<details>\n<summary>")
    with pytest.raises(ValueError, match="lost or reordered learner code cell"):
        build.validate_page_surface(page.replace(block, ""))


def test_development_freeze_requires_six_bound_resolutions_and_cannot_publish(tmp_path, monkeypatch):
    from . import publish_study

    names = ("protocol.md", "data_manifest.json", "interaction_data.py")
    for name in names:
        (tmp_path / name).write_text("synthetic gate fixture\n")
    implementation = tmp_path / "implementation.py"
    implementation.write_text("original fixture implementation\n")
    resolution = {
        "reviewer": "root", "independent_recheck": "pending", "publication_authorized": False,
        "findings": [{"id": i, "status": "RESOLVED_ROOT_CHECK"} for i in range(1, 7)],
        "source_sha256": {name: run_native.digest(tmp_path / name) for name in names},
    }
    resolution["source_sha256"][implementation.name] = run_native.digest(implementation)
    resolution_path = tmp_path / "pre_run_resolution.json"
    resolution_path.write_text(json.dumps(resolution))
    freeze = {
        "sha256": {name: resolution["source_sha256"][name] for name in names},
        "native_fits_before_freeze": 0, "native_game_runs_before_freeze": 0,
        "native_attribution_runs_before_freeze": 0, "audit_loads_before_freeze": 0,
        "frozen_at_utc": "synthetic-test-only", "independent_approval": False,
        "execution_scope": "development_only",
        "pre_run_resolution_sha256": run_native.digest(resolution_path),
    }
    freeze_path = tmp_path / "protocol_freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    monkeypatch.setattr(run_native, "HERE", tmp_path)
    monkeypatch.setattr(run_native, "source_paths", lambda: [tmp_path / name for name in names])
    monkeypatch.setattr(publish_study, "HERE", tmp_path)
    assert run_native.validate_freeze()["independent_approval"] is False
    with pytest.raises(ValueError, match="independent publication review is still pending"):
        publish_study.validate_publication_review({"source_inputs": {}})
    implementation.write_text("corrected fixture implementation\n")
    with pytest.raises(ValueError, match="Resolved pre-run source changed"):
        run_native.validate_freeze()
    evidence_path = tmp_path / "diagnostic.json"
    evidence_path.write_text('{"synthetic_test": true}\n')
    amendments = {
        "protocol_freeze_sha256": run_native.digest(freeze_path),
        "independent_approval": False, "publication_authorized": False,
        "entries": [{
            "classification": "implementation_correction", "reason": "synthetic regression",
            "recorded_at_utc": "synthetic-test-only", "scientific_design_changed": False,
            "evidence_files": {str(evidence_path): {"sha256": run_native.digest(evidence_path),
                                                    "size_bytes": evidence_path.stat().st_size}},
            "source_changes": {implementation.name: {"before": resolution["source_sha256"][implementation.name],
                                                       "after": run_native.digest(implementation)}},
        }],
    }
    (tmp_path / "implementation_amendments.json").write_text(json.dumps(amendments))
    assert run_native.validate_freeze()["independent_approval"] is False
    evidence_path.write_text("tampered evidence\n")
    with pytest.raises(ValueError, match="correction evidence changed"):
        run_native.validate_freeze()
    resolution["findings"].pop()
    resolution_path.write_text(json.dumps(resolution))
    freeze["pre_run_resolution_sha256"] = run_native.digest(resolution_path)
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="all six findings"):
        run_native.validate_freeze()
