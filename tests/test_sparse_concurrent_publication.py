"""Synthetic scheduling evidence; no model evaluation or native outcome claim."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from chapter16_shapley_attribution_baselines.exercises.part18_spex import publish_study as publish
from chapter16_shapley_attribution_baselines.exercises.part18_spex import run_native


@pytest.fixture
def concurrent_case(tmp_path):
    outer, original = tmp_path / "outer", tmp_path / "original"
    fresh = outer / "native"
    fresh.mkdir(parents=True)
    coordinator = tmp_path / "synthetic_coordinator.py"
    coordinator.write_text("# Synthetic test source; never executed.\n")
    supplement = tmp_path / "supplement.json"
    publish.write_json(supplement, {"primary_run": str(original), "reproduction_root": str(outer),
                                    "coordinator_sha256": publish.bundles.sha256(coordinator)})
    approval = tmp_path / "approval.json"
    publish.write_json(approval, {"status": "APPROVE_CONCURRENT_COLD_REPRODUCTION", "independent_review": True,
                                 "supplement_sha256": publish.bundles.sha256(supplement), "synthetic_only": True})
    scheduling = {"mode": "concurrent_cold_native_then_frozen_comparison",
                  "unchanged_sequential_reproduce_function_used": False,
                  "outer_supervision_receipt": str(outer / "execution.json"),
                  "supplement_path": str(supplement), "supplement_sha256": publish.bundles.sha256(supplement),
                  "independent_approval_path": str(approval), "independent_approval_sha256": publish.bundles.sha256(approval),
                  "coordinator_sha256": publish.bundles.sha256(coordinator)}
    reproduction = {"scheduling": scheduling, "elapsed_seconds": 1.0, "learner_directory": str(original)}
    publish.write_json(fresh / "reproduction.json", reproduction)
    sources = {str(path): publish.bundles.file_record(path) for path in (coordinator, supplement, approval)}
    receipt = {"status": "COMPLETE", "returncode": 0, "execution_kind": "concurrent_cold_reproduction_coordinator",
               "failures": [], "resource_violations": [], "changed_sources": [],
               "source_inputs": sources, "source_inputs_after": deepcopy(sources),
               "limits": {"wall_seconds": 10800, "rss_bytes": 8 * 1024**3, "output_bytes": 8 * 1024**3},
               "elapsed_seconds": 1.1, "sampled_peak_rss_bytes": 4096,
               "supplement_sha256": scheduling["supplement_sha256"], "approval_sha256": scheduling["independent_approval_sha256"],
               "command": [sys.executable, str(coordinator), "--child"]}
    run_native._finalize_receipt(outer, receipt, 8 * 1024**3)
    return reproduction, fresh, receipt


def test_concurrent_receipts_bind_outer_resources_source_and_approval(concurrent_case):
    reproduction, fresh, _ = concurrent_case
    evidence = publish.validate_concurrent_schedule(reproduction, fresh)
    assert set(evidence) == {"supervision", "supplement", "approval", "coordinator"}
    assert all(publish.verify_record(record).is_file() for record in evidence.values())


@pytest.mark.parametrize("change", ["failed_outer", "resource_breach", "source_drift", "too_long", "rss", "output_limit", "wrong_kind"])
def test_inner_success_cannot_bypass_outer_failure(concurrent_case, change):
    reproduction, fresh, receipt = concurrent_case
    if change == "failed_outer":
        receipt["status"] = "FAIL"
    elif change == "resource_breach":
        receipt["resource_violations"] = ["wall_seconds"]
    elif change == "source_drift":
        receipt["source_inputs_after"] = {}
    elif change == "too_long":
        receipt["elapsed_seconds"] = 10801
    elif change == "rss":
        receipt["sampled_peak_rss_bytes"] = 8 * 1024**3 + 1
    elif change == "output_limit":
        receipt["limits"]["output_bytes"] *= 2
    elif change == "wrong_kind":
        receipt["execution_kind"] = "cold_source_native"
    run_native._finalize_receipt(fresh.parent, receipt, 8 * 1024**3)
    with pytest.raises(ValueError):
        publish.validate_concurrent_schedule(reproduction, fresh)


@pytest.mark.parametrize("change", ["new_file", "approval", "supplement", "coordinator", "foreign_outer", "sequential_claim", "reproduction_time"])
def test_scheduling_tampering_is_rejected(concurrent_case, change):
    reproduction, fresh, receipt = concurrent_case
    scheduling = reproduction["scheduling"]
    if change == "new_file":
        (fresh.parent / "unrecorded.log").write_text("extra bytes")
    elif change in {"approval", "supplement"}:
        key = "independent_approval_path" if change == "approval" else "supplement_path"
        path = Path(scheduling[key])
        path.write_text(path.read_text() + "\n")
    elif change == "coordinator":
        Path(receipt["command"][1]).write_text("# Changed source.\n")
    elif change == "foreign_outer":
        scheduling["outer_supervision_receipt"] = str(fresh / "execution.json")
    elif change == "sequential_claim":
        scheduling["unchanged_sequential_reproduce_function_used"] = True
    elif change == "reproduction_time":
        reproduction["elapsed_seconds"] = 10801
    with pytest.raises(ValueError):
        publish.validate_concurrent_schedule(reproduction, fresh)


def test_existing_sequential_reproduction_keeps_its_contract(tmp_path):
    assert publish.validate_concurrent_schedule({"status": "PASS"}, tmp_path) == {}


def test_concurrent_inner_receipt_cannot_omit_schedule(concurrent_case):
    reproduction, fresh, _ = concurrent_case
    del reproduction["scheduling"]
    with pytest.raises(ValueError, match="omitted"):
        publish.validate_concurrent_schedule(reproduction, fresh)


def test_stager_copies_the_authoritative_scheduling_evidence_for_both_lessons(tmp_path, monkeypatch):
    from chapter16_shapley_attribution_baselines.exercises.part18_spex.publication_tests import _native_pair, _section_inputs

    original, fresh = _native_pair(tmp_path)
    manifests = _section_inputs(tmp_path, original, fresh)
    outer = tmp_path / "synthetic_outer"
    outer.mkdir()
    evidence = {}
    for role in ("supervision", "supplement", "approval", "coordinator"):
        path = outer / (role + (".py" if role == "coordinator" else ".json"))
        if role == "coordinator":
            path.write_text("# Synthetic packaging source; never executed.\n")
        else:
            publish.write_json(path, {"synthetic_only": True, "role": role})
        evidence[role] = publish.physical_record(path)
    monkeypatch.setattr(publish, "validate_concurrent_schedule", lambda *args: evidence)
    output = tmp_path / "candidate"
    candidate = publish.stage_candidate(original, fresh, manifests, output)
    for section in publish.SECTIONS:
        for role, source in evidence.items():
            name = "scheduling_" + role + (".py" if role == "coordinator" else ".json")
            target = publish.artifact_target(section, "supporting_evidence", name)
            assert candidate["artifacts"][target]["sha256"] == source["sha256"]
            assert (output / "tree" / target).read_bytes() == Path(source["absolute_path"]).read_bytes()
