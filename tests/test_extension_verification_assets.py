import ast
import csv
import json
from pathlib import Path

import yaml

from scripts.run_extension_verification_reports import expected_metric_failures


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_REPORT_FIELDS = {
    "notebook_id",
    "date_run",
    "git_commit",
    "report_inputs",
    "gt_tier",
    "evidence_level",
    "claim_scope",
    "gpu_name",
    "peak_vram_gb",
    "wall_clock_seconds",
    "models",
    "datasets",
    "tests_passed",
    "metrics",
    "baselines",
    "negative_controls",
    "ood_tests",
    "known_failures",
    "safety_notes",
    "accepted",
}
REQUIRED_LOCK_FIELDS = {
    "notebook_id",
    "gt_tier",
    "exercise_metadata",
    "required_gpu_gb",
    "max_allowed_gpu_gb",
    "evidence_level",
    "claim_scope",
    "models",
    "datasets",
    "expected_metrics",
    "controls",
}
EXERCISE_METADATA_FIELDS = {
    "EXERCISE_ID",
    "GT_TIER",
    "DIFFICULTY",
    "IMPORTANCE",
    "EXPECTED_RUNTIME",
    "REQUIRES_GPU",
}
REGISTRY_COLUMNS = {
    "name",
    "type",
    "provider",
    "repo_or_source_id",
    "license",
    "gated",
    "revision",
    "local_status",
    "max_vram_gb",
    "used_in_notebooks",
    "gt_tier",
    "notes",
}
METHOD_REGISTRY_COLUMNS = {
    "method_name",
    "paper",
    "year",
    "category",
    "model_family",
    "has_code",
    "has_weights",
    "local_24gb_status",
    "implementation_status",
    "verification_status",
    "baseline_status",
    "notes",
}
LADDER_REGISTRY_COLUMNS = {
    "section",
    "notebook_id",
    "title",
    "gt_tier",
    "difficulty",
    "importance",
    "requires_gpu",
    "metadata_source",
    "fixture_provenance_source",
    "visible_tests_source",
    "report_source",
    "gpu_evidence_requirement",
    "toy_oracle_requirement",
    "slow_fast_oracle_requirement",
    "property_test_requirement",
    "debug_mode_requirement",
    "release_status",
    "remaining_release_evidence",
}


def test_expected_metric_failures_accepts_exact_threshold_and_vram_metrics():
    metrics = {
        "gpu_test": {
            "peak_vram_gb": 1.25,
            "preflight_passed": True,
            "accuracy": 0.95,
            "loss": 0.1,
            "shape": [2, 3],
        }
    }
    expected = {
        "max_allowed_gpu_gb": 2.0,
        "preflight_passed": True,
        "accuracy_min": 0.9,
        "loss_max": 0.2,
        "shape": [2, 3],
    }
    assert expected_metric_failures(expected, metrics) == []


def test_expected_metric_failures_reports_threshold_and_missing_metric_failures():
    metrics = {"gpu_test": {"peak_vram_gb": 3.0, "accuracy": 0.7}}
    expected = {
        "max_allowed_gpu_gb": 2.0,
        "accuracy_min": 0.9,
        "missing_preflight_passed": True,
    }
    failures = expected_metric_failures(expected, metrics)
    assert any("peak_vram_gb <= 2.0" in failure for failure in failures)
    assert any("accuracy" in failure and ">= 0.9" in failure for failure in failures)
    assert any("missing_preflight_passed" in failure for failure in failures)


def test_expected_metric_failures_prefers_exact_metric_names_over_suffix_syntax():
    metrics = {"gpu_test": {"peak_vram_gb": 1.0, "score_min": -4.0, "score_max": 4.0}}
    expected = {
        "max_allowed_gpu_gb": 2.0,
        "score_min": -4.0,
        "score_max": 4.0,
    }
    assert expected_metric_failures(expected, metrics) == []

    failures = expected_metric_failures({"score_max": 3.0}, metrics)
    assert failures == ["expected gpu_test.score_max == 3.0, got 4.0"]
METHOD_IMPLEMENTATION_STATUSES = {
    "REQUIRED_IMPLEMENT",
    "REQUIRED_LOAD_WEIGHTS",
    "TOY_REPRO_ONLY",
    "READ_ONLY_TOO_EXPENSIVE",
    "WAIT_FOR_WEIGHTS",
    "DEPRECATED_BY_NEWER_METHOD",
}


def is_extension_section(number: str) -> bool:
    if number in {"0.6", "1.6"}:
        return True
    try:
        return int(number.split(".", maxsplit=1)[0]) >= 5
    except ValueError:
        return False


def extension_exercise_dirs() -> list[Path]:
    config = yaml.safe_load((ROOT / "infrastructure/core/config.yaml").read_text())
    exercise_dirs = []
    for chapter_name, chapter in config["chapters"].items():
        for section in chapter.get("sections", []):
            number = str(section.get("number", ""))
            if is_extension_section(number):
                exercise_dirs.append(
                    ROOT / chapter_name / "exercises" / section["exercise_dir"]
                )
    return exercise_dirs


def test_every_extension_notebook_has_verification_assets():
    missing = []
    for exercise_dir in extension_exercise_dirs():
        expected_paths = [
            exercise_dir / "README.md",
            exercise_dir / "artifacts.lock.yml",
            exercise_dir / "verification_report.json",
            exercise_dir / "verification_report.schema.json",
            exercise_dir / "expected_outputs/README.md",
            exercise_dir / "expected_outputs/smoke_test.json",
            exercise_dir / "expected_outputs/reference_metrics.json",
        ]
        missing.extend(path for path in expected_paths if not path.exists())

    assert missing == []


def test_shapley_mechinterp_agreement_artifacts_match_roadmap_contract():
    exercise_dir = (
        ROOT
        / "chapter16_shapley_attribution_baselines/exercises/part8_shapley_mechinterp_agreement"
    )
    artifact_dir = exercise_dir / "artifacts"
    expected_files = [
        artifact_dir / "agreement_matrix.csv",
        artifact_dir / "deletion_curves.png",
        artifact_dir / "insertion_curves.png",
        artifact_dir / "topk_overlap_heatmap.png",
        artifact_dir / "method_disagreement_examples.md",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected_files)

    rows = list(csv.DictReader((artifact_dir / "agreement_matrix.csv").open()))
    assert len(rows) >= 7
    assert set(rows[0]) == {
        "task",
        "method_a",
        "method_b",
        "player_type_a",
        "player_type_b",
        "metric",
        "value",
        "interpretation",
    }
    assert any(row["task"] == "xor_control" for row in rows)
    assert any(row["task"] == "shuffled_label_control" for row in rows)
    assert all(float(row["value"]) == float(row["value"]) for row in rows)

    for image_name in [
        "deletion_curves.png",
        "insertion_curves.png",
        "topk_overlap_heatmap.png",
    ]:
        assert (artifact_dir / image_name).stat().st_size > 10_000

    disagreement_examples = (artifact_dir / "method_disagreement_examples.md").read_text()
    assert "XOR interaction" in disagreement_examples
    assert "shuffled-label control" in disagreement_examples


def test_artifact_locks_and_report_schemas_have_required_fields():
    for exercise_dir in extension_exercise_dirs():
        lock = yaml.safe_load((exercise_dir / "artifacts.lock.yml").read_text())
        report = json.loads((exercise_dir / "verification_report.json").read_text())
        schema = json.loads((exercise_dir / "verification_report.schema.json").read_text())
        smoke = json.loads((exercise_dir / "expected_outputs/smoke_test.json").read_text())

        assert REQUIRED_LOCK_FIELDS.issubset(lock)
        assert lock["gt_tier"] in {"GT-0", "GT-1", "GT-2", "GT-3", "GT-4"}
        assert EXERCISE_METADATA_FIELDS.issubset(lock["exercise_metadata"])
        assert lock["exercise_metadata"]["EXERCISE_ID"] == lock["notebook_id"]
        assert lock["exercise_metadata"]["GT_TIER"] == lock["gt_tier"]
        assert 1 <= lock["exercise_metadata"]["DIFFICULTY"] <= 5
        assert 1 <= lock["exercise_metadata"]["IMPORTANCE"] <= 5
        assert isinstance(lock["exercise_metadata"]["EXPECTED_RUNTIME"], str)
        assert lock["exercise_metadata"]["EXPECTED_RUNTIME"]
        assert isinstance(lock["exercise_metadata"]["REQUIRES_GPU"], bool)
        assert lock["max_allowed_gpu_gb"] <= 24
        assert set(schema["required"]) == REQUIRED_REPORT_FIELDS
        assert REQUIRED_REPORT_FIELDS.issubset(report)
        assert report["report_inputs"]["algorithm"] == "sha256"
        assert report["report_inputs"]["combined_sha256"]
        input_paths = {item["path"] for item in report["report_inputs"]["files"]}
        assert input_paths
        assert (exercise_dir / "solutions.py").relative_to(ROOT).as_posix() in input_paths
        assert (exercise_dir / "artifacts.lock.yml").relative_to(ROOT).as_posix() in input_paths
        exercise_prefix = exercise_dir.relative_to(ROOT).as_posix()
        assert any(
            path.startswith(f"{exercise_prefix}/") and path.endswith("_exercises.ipynb")
            for path in input_paths
        )
        assert any(
            path.startswith(f"{exercise_prefix}/") and path.endswith("_solutions.ipynb")
            for path in input_paths
        )
        chapter_dir = exercise_dir.parents[1].relative_to(ROOT).as_posix()
        assert any(
            path.startswith(f"{chapter_dir}/instructions/pages/") and path.endswith(".md")
            for path in input_paths
        )
        assert report["notebook_id"] == lock["notebook_id"]
        assert report["gt_tier"] == lock["gt_tier"]
        assert report["evidence_level"] == lock["evidence_level"]
        assert report["claim_scope"] == lock["claim_scope"]
        assert "gpu_evidence" in report["metrics"]
        assert report["metrics"]["gpu_evidence"]["category"] in {
            "cuda_section_metric",
            "cuda_environment_or_budget",
            "cpu_or_budget_metric",
            "placeholder_only",
            "missing",
        }
        section_specific_expected_metrics = set(lock["expected_metrics"]) - {
            "tests_passed",
            "accepted",
            "max_allowed_gpu_gb",
        }
        assert section_specific_expected_metrics
        assert report["tests_passed"] is True
        assert report["accepted"] is True
        assert report["peak_vram_gb"] <= lock["max_allowed_gpu_gb"]
        assert smoke["notebook_id"] == lock["notebook_id"]
        assert smoke["accepted"] is True


def test_expected_output_fixture_readmes_record_provenance_and_tolerances():
    required_phrases = [
        "How this fixture was produced",
        "Trusted implementation",
        "Random seed",
        "Allowed tolerances",
        "When to regenerate",
        "scripts/generate_extension_verification_assets.py",
        "scripts/run_extension_verification_reports.py",
        "rtol=1e-5",
        "atol=1e-6",
        "Do not regenerate merely to hide a failing check",
    ]
    for exercise_dir in extension_exercise_dirs():
        readme = (exercise_dir / "expected_outputs/README.md").read_text()
        missing_phrases = [phrase for phrase in required_phrases if phrase not in readme]
        assert missing_phrases == []


def test_global_artifact_registry_exists_and_uses_required_columns():
    csv_path = ROOT / "docs/artifact_registry.csv"
    md_path = ROOT / "docs/artifact_registry.md"
    lock_path = ROOT / "docs/artifact_registry.lock.yml"

    assert csv_path.exists()
    assert md_path.exists()
    assert lock_path.exists()

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert set(rows[0]) == REGISTRY_COLUMNS
    statuses = {row["local_status"] for row in rows}
    source_ids = {row["repo_or_source_id"] for row in rows}
    assert "REQUIRED" in statuses
    assert "GENERATED_BY_COURSE" in statuses
    assert "course_generated_refusal_proxy_prompts_v1" in source_ids
    assert "course_generated_modular_addition_checkpoints_v1" in source_ids


def test_method_registry_exists_and_tracks_sota_statuses():
    csv_path = ROOT / "docs/method_registry.csv"
    md_path = ROOT / "docs/method_registry.md"
    lock_path = ROOT / "docs/method_registry.lock.yml"

    assert csv_path.exists()
    assert md_path.exists()
    assert lock_path.exists()

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert set(rows[0]) == METHOD_REGISTRY_COLUMNS
    method_names = {row["method_name"] for row in rows}
    implementation_statuses = {row["implementation_status"] for row in rows}

    assert "Refusal directions" in method_names
    assert "Sparse Feature Circuits" in method_names
    assert "TokenSHAP and TokenShapley" in method_names
    assert "V-JEPA 2.1 dense features" in method_names
    assert "Checkpoint archaeology" in method_names
    assert METHOD_IMPLEMENTATION_STATUSES.issubset(implementation_statuses)


def test_hard_exercise_ladder_registry_tracks_all_hard_sections():
    csv_path = ROOT / "docs/hard_exercise_ladder_registry.csv"
    md_path = ROOT / "docs/hard_exercise_ladder_registry.md"
    lock_path = ROOT / "docs/hard_exercise_ladder_registry.lock.yml"

    assert csv_path.exists()
    assert md_path.exists()
    assert lock_path.exists()

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert set(rows[0]) == LADDER_REGISTRY_COLUMNS

    hard_notebook_ids = set()
    for exercise_dir in extension_exercise_dirs():
        lock = yaml.safe_load((exercise_dir / "artifacts.lock.yml").read_text())
        if lock["exercise_metadata"]["DIFFICULTY"] >= 3:
            hard_notebook_ids.add(lock["notebook_id"])

    assert {row["notebook_id"] for row in rows} == hard_notebook_ids
    for row in rows:
        assert row["release_status"] == "released_with_cuda_section_metric"
        assert row["visible_tests_source"] != "MISSING"
        assert row["report_source"] != "PENDING"
        for source_key in (
            "metadata_source",
            "fixture_provenance_source",
            "visible_tests_source",
            "report_source",
        ):
            source_path = row[source_key].split(":", maxsplit=1)[0]
            assert (ROOT / source_path).exists()
        assert row["remaining_release_evidence"] == "", (
            "Accepted hard-exercise rows should not keep stale unresolved-evidence "
            f"tokens for {row['notebook_id']}."
        )


def test_extension_full_experiment_hooks_are_callable_not_placeholders():
    placeholder_hooks = []
    for exercise_dir in extension_exercise_dirs():
        solution_path = exercise_dir / "solutions.py"
        tree = ast.parse(solution_path.read_text(), filename=str(solution_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "run_full_experiment":
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call):
                    func = child.exc.func
                    if isinstance(func, ast.Name) and func.id == "NotImplementedError":
                        placeholder_hooks.append(solution_path)
                elif isinstance(child, ast.Raise) and isinstance(child.exc, ast.Name):
                    if child.exc.id == "NotImplementedError":
                        placeholder_hooks.append(solution_path)

    assert placeholder_hooks == []


def test_hard_exercise_verification_ladder_policy_tracks_roadmap_appendix():
    policy_path = ROOT / "docs/hard_exercise_verification_ladders.md"
    assert policy_path.exists()

    policy = policy_path.read_text()
    required_terms = [
        "shape tests",
        "dtype and device tests",
        "small hand-computed examples",
        "sub-function tests",
        "brute-force or reference comparison",
        "randomized property tests",
        "real-model smoke test",
        "full verification report",
        "EXERCISE_ID",
        "GT_TIER",
        "DIFFICULTY",
        "IMPORTANCE",
        "EXPECTED_RUNTIME",
        "REQUIRES_GPU",
        "slow_correct_version",
        "fast_vectorized_version",
        "assets/expected_outputs/",
        "rtol=",
        "atol=",
        "looks_reasonable",
        "debug=True",
        "@pytest.mark.gpu",
        "@pytest.mark.requires_gated_model",
    ]
    missing_terms = [term for term in required_terms if term not in policy]
    assert missing_terms == []
