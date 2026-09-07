from copy import deepcopy
from pathlib import Path

import yaml

import scripts.audit_roadmap_final_completeness as audit
from scripts.audit_roadmap_final_completeness import (
    COURSE_STATUS_PATH,
    METHOD_REGISTRY_PATH,
    ROADMAP_PATH,
    load_roadmap_requirement_registry,
    roadmap_requirement_registry_blockers,
    roadmap_requirement_status_blockers,
)


def _write_registry(tmp_path: Path, registry: dict) -> Path:
    path = tmp_path / "roadmap_requirement_registry.yml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False))
    return path


def _structure_blockers(path: Path) -> list[str]:
    return roadmap_requirement_registry_blockers(
        registry_path=path,
        roadmap_path=ROADMAP_PATH,
        method_registry_path=METHOD_REGISTRY_PATH,
        course_status_path=COURSE_STATUS_PATH,
    )


def test_current_roadmap_requirement_registry_is_structurally_valid():
    assert roadmap_requirement_registry_blockers() == []


def test_current_registry_is_exhaustive_over_selected_families():
    registry = load_roadmap_requirement_registry()
    by_id = {row["id"]: row for row in registry["requirements"]}
    required_missing = {
        "modern.rwkv",
        "modern.retnet",
        "modern.recurrent_gemma_griffin",
        "modern.hyena",
        "modern.xlstm",
        "data_attribution.influence_functions",
        "data_attribution.tracin",
        "data_attribution.datamodels",
        "data_attribution.dataset_cartography",
        "alignment.rome_model_editing",
        "alignment.memit_model_editing",
        "alignment.unlearning",
        "intrinsic.concept_bottleneck_models",
        "activation.maia",
        "activation.openmaia",
        "vlm.real_object_hallucination_arbitration",
        "vlm.multimodal_saes",
        "vlm.multimodal_crosscoders",
        "image.denoising_time_causal_patching",
        "image.autoregressive_image_tokens",
        "jepa.ijepa_from_scratch",
        "jepa.vljepa",
        "world.maze_transformers",
        "world.sudoku_transformers",
        "world.dreamer_lite",
        "world.tdmpc2_lite",
        "world.iris_lite",
        "shapley.treeshap",
        "shapley.deepshap",
        "shapley.sage",
        "shapley.spex_proxyspex",
        "training.induction_head_emergence",
        "training.cross_architecture_dynamics",
    }

    assert required_missing <= set(by_id)
    partial_implementations = {
        "image.denoising_time_causal_patching": "PREFLIGHT_ONLY",
        "jepa.ijepa_from_scratch": "TOY_ONLY",
        "data_attribution.influence_functions": "TAUGHT",
        "data_attribution.tracin": "TAUGHT",
        "data_attribution.datamodels": "TAUGHT",
        "data_attribution.dataset_cartography": "TAUGHT",
        "shapley.sage": "TAUGHT",
        "shapley.treeshap": "TAUGHT",
        "shapley.deepshap": "TAUGHT",
    }
    for requirement_id in required_missing:
        row = by_id[requirement_id]
        assert row["status"] == partial_implementations.get(requirement_id, "UNIMPLEMENTED")
        if row["status"] == "TAUGHT":
            assert not row["blocker"]
            expected_section = {
                "data_attribution.datamodels": "16.10",
                "data_attribution.dataset_cartography": "16.11",
                "shapley.sage": "16.12",
                "shapley.treeshap": "16.13",
                "shapley.deepshap": "16.14",
            }.get(requirement_id, "16.9")
            assert row["section"] == expected_section
            assert len(row["evidence"]) >= 5
        else:
            assert row["blocker"], "Partial or missing implementations must retain an explicit blocker."
        if requirement_id in partial_implementations:
            assert row["evidence"], "A partial implementation needs traceable existing evidence."


def test_unimplemented_requirements_block_course_completion():
    blockers = roadmap_requirement_status_blockers()

    assert any("modern.rwkv: UNIMPLEMENTED" in blocker for blocker in blockers)
    assert not any(blocker.startswith("circuits.acdc:") for blocker in blockers)
    assert len(blockers) == sum(
        row["status"] == "UNIMPLEMENTED"
        for row in load_roadmap_requirement_registry()["requirements"]
    )


def test_registry_rejects_stale_roadmap_hash(tmp_path):
    registry = deepcopy(load_roadmap_requirement_registry())
    registry["roadmap"]["sha256"] = "0" * 64

    blockers = _structure_blockers(_write_registry(tmp_path, registry))

    assert any("roadmap sha256 is stale" in blocker for blocker in blockers)


def test_registry_rejects_unexplained_deferred_requirement(tmp_path):
    registry = deepcopy(load_roadmap_requirement_registry())
    deferred = next(row for row in registry["requirements"] if row["status"] == "DEFERRED")
    deferred["blocker"] = ""
    deferred.pop("deferred_until")

    blockers = _structure_blockers(_write_registry(tmp_path, registry))

    assert any(f"{deferred['id']}: DEFERRED requirement needs an explicit blocker" in blocker for blocker in blockers)
    assert any(f"{deferred['id']}: DEFERRED requirement needs deferred_until" in blocker for blocker in blockers)


def test_registry_rejects_missing_method_registry_mapping(tmp_path):
    registry = deepcopy(load_roadmap_requirement_registry())
    row = next(
        row
        for row in registry["requirements"]
        if "Fake-result diagnostics" in row.get("method_registry_names", [])
    )
    row["method_registry_names"].remove("Fake-result diagnostics")

    blockers = _structure_blockers(_write_registry(tmp_path, registry))

    assert any("method_registry.csv rows not represented" in blocker for blocker in blockers)
    assert any("Fake-result diagnostics" in blocker for blocker in blockers)


def test_registry_rejects_missing_tracked_section_mapping(tmp_path):
    registry = deepcopy(load_roadmap_requirement_registry())
    for row in registry["requirements"]:
        if "17.1" in row.get("course_sections", []):
            row["course_sections"].remove("17.1")

    blockers = _structure_blockers(_write_registry(tmp_path, registry))

    assert any("tracked extension sections not represented" in blocker for blocker in blockers)
    assert any("17.1" in blocker for blocker in blockers)


def test_registry_rejects_missing_partial_evidence_and_blocker(tmp_path):
    registry = deepcopy(load_roadmap_requirement_registry())
    partial = next(row for row in registry["requirements"] if row["status"] == "TOY_ONLY")
    partial["evidence"] = []
    partial["blocker"] = None

    blockers = _structure_blockers(_write_registry(tmp_path, registry))

    assert any(f"{partial['id']}: TOY_ONLY requirement needs evidence paths" in blocker for blocker in blockers)
    assert any(f"{partial['id']}: TOY_ONLY requirement needs an explicit blocker" in blocker for blocker in blockers)


def test_final_completeness_gate_includes_registry_status_blockers(monkeypatch):
    monkeypatch.setattr(audit, "roadmap_requirement_registry_blockers", lambda: [])
    monkeypatch.setattr(
        audit,
        "roadmap_requirement_status_blockers",
        lambda: ["missing.topic: UNIMPLEMENTED - no learner surface"],
    )
    monkeypatch.setattr(audit, "original_preservation_blockers", lambda: [])
    monkeypatch.setattr(audit, "course_surface_blockers", lambda: [])
    monkeypatch.setattr(audit, "report_evidence_blockers", lambda: [])
    monkeypatch.setattr(audit, "style_depth_blockers", lambda: [])
    monkeypatch.setattr(audit, "REPORT_REQUIREMENTS", ())
    monkeypatch.setattr(audit, "LEGACY_REQUIREMENTS", ())

    assert audit.roadmap_final_completeness_blockers() == [
        "roadmap status: missing.topic: UNIMPLEMENTED - no learner surface"
    ]


def test_refusal_requirement_rejects_missing_regeneration_and_full_replication_claim(monkeypatch):
    import json
    requirement = next(row for row in audit.REPORT_REQUIREMENTS if row.requirement_id == 'gt2_refusal_direction')
    real_report = json.loads((audit.ROOT / requirement.report_path).read_text())
    assert audit.report_requirement_blockers(requirement) == []
    altered = deepcopy(real_report)
    altered['metrics']['gpu_test']['fresh_generation_examples'] = 104
    altered['metrics']['gpu_test']['paper_equivalent_full_replication'] = True
    monkeypatch.setattr(audit, '_load_json', lambda path: altered)
    blockers = audit.report_requirement_blockers(requirement)
    assert any('fresh_generation_examples' in item for item in blockers)
    assert any('paper_equivalent_full_replication' in item for item in blockers)


def test_ccs_requirement_keeps_vocabulary_counts_refits_and_selection_separate(monkeypatch):
    import json
    requirement = next(row for row in audit.REPORT_REQUIREMENTS if row.requirement_id == 'gt0_contrast_consistent_search')
    real = json.loads((audit.ROOT / requirement.report_path).read_text())
    assert audit.report_requirement_blockers(requirement) == []
    altered = deepcopy(real)
    measured = altered['metrics']['gpu_test']
    # Reporter scores must not be passed off as separate vocabulary forwards.
    measured['fresh_compared_examples'] = measured['retained_final_outcomes']
    measured['all_reporters_refitted'] = False
    measured['final_used_for_selection'] = True
    measured['whole_paper_replication'] = True
    monkeypatch.setattr(audit, '_load_json', lambda path: altered)
    blockers = audit.report_requirement_blockers(requirement)
    for key in ['fresh_compared_examples', 'all_reporters_refitted', 'final_used_for_selection', 'whole_paper_replication']:
        assert any(key in item for item in blockers), key
