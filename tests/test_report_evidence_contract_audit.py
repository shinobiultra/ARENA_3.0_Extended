from scripts.audit_report_evidence_contracts import (
    report_freshness_blockers,
    report_contract_blockers,
    report_evidence_blockers,
    verification_report_records,
)
from arena_ext.verification import build_report_input_manifest, verify_report_input_manifest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _valid_lock_report():
    lock = {
        "notebook_id": "section_x",
        "gt_tier": "GT-1",
        "evidence_level": "real_cuda_preflight",
        "claim_scope": "A concrete CUDA preflight claim with baselines, controls, models, datasets, and narrow scope.",
        "controls": ["baseline_metric", "random_direction_control"],
        "expected_metrics": {"accuracy_min": 0.9},
    }
    report = {
        "notebook_id": "section_x",
        "gt_tier": "GT-1",
        "evidence_level": "real_cuda_preflight",
        "claim_scope": lock["claim_scope"],
        "tests_passed": True,
        "accepted": True,
        "known_failures": [],
        "report_inputs": build_report_input_manifest(ROOT, [ROOT / "pyproject.toml"]),
        "metrics": {
            "gpu_evidence": {
                "category": "cuda_section_metric",
                "uses_cuda": True,
                "placeholder_only": False,
                "section_specific_metric_keys": ["accuracy"],
            },
            "gpu_test": {"accuracy": 0.95},
        },
        "baselines": {
            "declared_controls": ["baseline_metric", "random_direction_control"],
            "expected_metrics": {"accuracy_min": 0.9},
        },
        "negative_controls": {"declared_controls": ["random_direction_control"]},
        "ood_tests": {"declared": "required_where_applicable"},
        "models": [{"id": "tiny-model"}],
        "datasets": [{"id": "generated"}],
        "safety_notes": ["No unsafe prompts or raw gated weights."],
    }
    return lock, report


def test_report_contract_accepts_complete_report():
    lock, report = _valid_lock_report()

    assert report_contract_blockers(lock, report) == []


def test_report_contract_flags_placeholder_report():
    lock, report = _valid_lock_report()
    lock["evidence_level"] = "notebook_contract"
    report["evidence_level"] = "notebook_contract"
    report["metrics"]["gpu_evidence"] = {
        "category": "placeholder_only",
        "uses_cuda": False,
        "placeholder_only": True,
        "section_specific_metric_keys": [],
    }
    report["negative_controls"] = {}
    report["known_failures"] = ["metric failed"]

    blockers = report_contract_blockers(lock, report)

    assert "section_x: placeholder evidence_level=notebook_contract" in blockers
    assert "section_x: gpu_evidence.category='placeholder_only'" in blockers
    assert "section_x: gpu_evidence.uses_cuda is not true" in blockers
    assert "section_x: gpu_evidence is placeholder_only" in blockers
    assert "section_x: negative_controls are empty" in blockers
    assert "section_x: accepted report has known_failures" in blockers


def test_report_contract_flags_missing_report_inputs():
    lock, report = _valid_lock_report()
    report.pop("report_inputs")

    blockers = report_contract_blockers(lock, report)

    assert "section_x: report_inputs.files must be a nonempty list" in blockers


def test_report_input_manifest_detects_changed_file(tmp_path):
    source = tmp_path / "input.py"
    source.write_text("value = 1\n")
    manifest = build_report_input_manifest(tmp_path, [source])
    source.write_text("value = 2\n")

    blockers = verify_report_input_manifest(tmp_path, manifest)

    assert "report input hash changed: input.py" in blockers
    assert "report_inputs.combined_sha256 does not match current inputs" in blockers


def test_report_freshness_blockers_prefixes_notebook_id():
    blockers = report_freshness_blockers({"notebook_id": "section_x"})

    assert "section_x: report_inputs.files must be a nonempty list" in blockers


def test_report_contract_flags_expected_metric_regression():
    lock, report = _valid_lock_report()
    report["metrics"]["gpu_test"]["accuracy"] = 0.5

    blockers = report_contract_blockers(lock, report)

    assert any("expected gpu_test.accuracy >= 0.9" in blocker for blocker in blockers)


def test_current_report_evidence_contracts_pass():
    assert report_evidence_blockers() == []


def test_report_record_discovery_includes_docs_evidence_reports():
    report_paths = {record["report_path"].as_posix() for record in verification_report_records()}

    assert any(path.endswith("docs/evidence/ioi_path_patching/verification_report.json") for path in report_paths)
    assert any(path.endswith("docs/evidence/othello_gpt/verification_report.json") for path in report_paths)
