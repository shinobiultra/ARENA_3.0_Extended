"""Audit the roadmap's final completeness matrix against current evidence.

This gate is intentionally about roadmap coverage, not about whether a single
`gt_tier` field can describe a hybrid notebook. Several extension notebooks
contain both exact/toy contracts and real-model preflights, so this audit checks
named report metrics, controls, and legacy notebook evidence requirement by
requirement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_course_surface import course_surface_blockers
from scripts.audit_original_arena_preservation import original_preservation_blockers
from scripts.audit_report_evidence_contracts import report_evidence_blockers
from scripts.audit_arena_style_depth import style_depth_blockers


ROADMAP_PATH = ROOT / "Extension-Roadmap.md"
ROADMAP_REQUIREMENT_REGISTRY_PATH = ROOT / "docs/roadmap_requirement_registry.yml"
METHOD_REGISTRY_PATH = ROOT / "docs/method_registry.csv"
COURSE_STATUS_PATH = ROOT / "docs/arena_style_rewrite_status.yml"

ROADMAP_STATUSES = frozenset(
    {"TAUGHT", "TOY_ONLY", "PREFLIGHT_ONLY", "DEFERRED", "UNIMPLEMENTED"}
)
IMPLEMENTED_ROADMAP_STATUSES = frozenset({"TAUGHT", "TOY_ONLY", "PREFLIGHT_ONLY"})
PARTIAL_ROADMAP_STATUSES = frozenset({"TOY_ONLY", "PREFLIGHT_ONLY"})
ROADMAP_REF_RE = re.compile(r"^Extension-Roadmap\.md:(\d+)(?:-(\d+))?$")
REQUIRED_REGISTRY_FIELDS = frozenset(
    {
        "id",
        "topic",
        "chapter",
        "section",
        "status",
        "roadmap_refs",
        "evidence",
        "reason",
        "blocker",
    }
)


@dataclass(frozen=True)
class ReportRequirement:
    requirement_id: str
    group: str
    description: str
    report_path: str
    lock_path: str
    metric_expectations: dict[str, Any]
    required_controls: tuple[str, ...] = ()
    require_primary_gt_tier: str | None = None


@dataclass(frozen=True)
class LegacyNotebookRequirement:
    requirement_id: str
    group: str
    description: str
    required_paths: tuple[str, ...]
    modern_report_path: str | None = None


def _get_path(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, float | int):
            return False
        if "min" in expected and float(actual) < float(expected["min"]):
            return False
        if "max" in expected and float(actual) > float(expected["max"]):
            return False
        return True
    if isinstance(expected, float | int) and isinstance(actual, float | int):
        return abs(float(actual) - float(expected)) <= 1e-6
    return actual == expected


def _load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text())


def _load_yaml(relative_path: str) -> dict[str, Any]:
    return yaml.safe_load((ROOT / relative_path).read_text())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_roadmap_requirement_registry(
    registry_path: Path = ROADMAP_REQUIREMENT_REGISTRY_PATH,
) -> dict[str, Any]:
    loaded = yaml.safe_load(registry_path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _method_registry_names(method_registry_path: Path) -> list[str]:
    with method_registry_path.open(newline="") as handle:
        return [row["method_name"] for row in csv.DictReader(handle)]


def _course_status_sections(course_status_path: Path) -> list[str]:
    status = yaml.safe_load(course_status_path.read_text())
    if not isinstance(status, dict) or not isinstance(status.get("sections"), dict):
        return []
    return [str(section) for section in status["sections"]]


def roadmap_requirement_registry_blockers(
    *,
    registry_path: Path = ROADMAP_REQUIREMENT_REGISTRY_PATH,
    roadmap_path: Path = ROADMAP_PATH,
    method_registry_path: Path = METHOD_REGISTRY_PATH,
    course_status_path: Path = COURSE_STATUS_PATH,
) -> list[str]:
    """Validate that the normalized registry is exhaustive and internally honest.

    Exhaustiveness is protected in three ways: the registry is pinned to the
    complete roadmap file hash, every row has an inspectable roadmap line anchor,
    and the registry must cover every existing method-registry row and every
    tracked extension section. A roadmap edit therefore cannot silently bypass
    reconciliation of the normalized requirement inventory.
    """

    blockers: list[str] = []
    if not registry_path.exists():
        return [f"roadmap registry: missing {registry_path.relative_to(ROOT)}"]
    if not roadmap_path.exists():
        return [f"roadmap registry: missing {roadmap_path.relative_to(ROOT)}"]

    try:
        registry = load_roadmap_requirement_registry(registry_path)
    except (OSError, yaml.YAMLError) as exc:
        return [f"roadmap registry: cannot parse {registry_path.name}: {exc}"]

    if registry.get("schema_version") != 1:
        blockers.append(
            f"roadmap registry: schema_version must be 1, got {registry.get('schema_version')!r}"
        )

    roadmap_text = roadmap_path.read_text()
    roadmap_lines = roadmap_text.splitlines()
    roadmap_meta = registry.get("roadmap")
    if not isinstance(roadmap_meta, dict):
        blockers.append("roadmap registry: roadmap metadata must be a mapping")
        roadmap_meta = {}
    expected_hash = _sha256_text(roadmap_text)
    if roadmap_meta.get("path") != "Extension-Roadmap.md":
        blockers.append("roadmap registry: roadmap.path must be Extension-Roadmap.md")
    if roadmap_meta.get("sha256") != expected_hash:
        blockers.append(
            "roadmap registry: roadmap sha256 is stale; reconcile every requirement "
            "against the latest Extension-Roadmap.md"
        )
    if roadmap_meta.get("line_count") != len(roadmap_lines):
        blockers.append(
            f"roadmap registry: roadmap line_count must be {len(roadmap_lines)}, "
            f"got {roadmap_meta.get('line_count')!r}"
        )

    declared_statuses = registry.get("allowed_statuses")
    if set(declared_statuses or []) != ROADMAP_STATUSES:
        blockers.append(
            "roadmap registry: allowed_statuses must contain exactly "
            f"{sorted(ROADMAP_STATUSES)}"
        )

    requirements = registry.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        return blockers + ["roadmap registry: requirements must be a nonempty list"]

    ids: list[str] = []
    mapped_method_names: list[str] = []
    covered_course_sections: set[str] = set()
    for index, requirement in enumerate(requirements):
        label = f"roadmap registry row {index + 1}"
        if not isinstance(requirement, dict):
            blockers.append(f"{label}: requirement must be a mapping")
            continue
        missing_fields = sorted(REQUIRED_REGISTRY_FIELDS - set(requirement))
        if missing_fields:
            blockers.append(f"{label}: missing fields {missing_fields}")
            continue

        requirement_id = requirement.get("id")
        if not isinstance(requirement_id, str) or not requirement_id.strip():
            blockers.append(f"{label}: id must be a nonempty string")
            requirement_id = label
        else:
            ids.append(requirement_id)
            label = requirement_id

        for field in ("topic", "chapter", "section", "reason"):
            value = requirement.get(field)
            if not isinstance(value, str) or not value.strip():
                blockers.append(f"{label}: {field} must be a nonempty string")

        status = requirement.get("status")
        if status not in ROADMAP_STATUSES:
            blockers.append(
                f"{label}: invalid status {status!r}; expected one of {sorted(ROADMAP_STATUSES)}"
            )

        blocker = requirement.get("blocker")
        if status == "TAUGHT" and blocker not in (None, ""):
            blockers.append(f"{label}: TAUGHT requirement must not declare a blocker")
        if status in PARTIAL_ROADMAP_STATUSES | {"DEFERRED", "UNIMPLEMENTED"}:
            if not isinstance(blocker, str) or not blocker.strip():
                blockers.append(f"{label}: {status} requirement needs an explicit blocker")
        if status == "DEFERRED":
            deferred_until = requirement.get("deferred_until")
            if not isinstance(deferred_until, str) or not deferred_until.strip():
                blockers.append(f"{label}: DEFERRED requirement needs deferred_until")

        refs = requirement.get("roadmap_refs")
        if not isinstance(refs, list) or not refs:
            blockers.append(f"{label}: roadmap_refs must be a nonempty list")
        else:
            for ref in refs:
                match = ROADMAP_REF_RE.fullmatch(str(ref))
                if match is None:
                    blockers.append(f"{label}: malformed roadmap ref {ref!r}")
                    continue
                start = int(match.group(1))
                end = int(match.group(2) or start)
                if start < 1 or end < start or end > len(roadmap_lines):
                    blockers.append(
                        f"{label}: roadmap ref {ref!r} is outside 1-{len(roadmap_lines)}"
                    )

        evidence = requirement.get("evidence")
        if not isinstance(evidence, list):
            blockers.append(f"{label}: evidence must be a list")
        else:
            if status in IMPLEMENTED_ROADMAP_STATUSES and not evidence:
                blockers.append(f"{label}: {status} requirement needs evidence paths")
            for relative_path in evidence:
                if not isinstance(relative_path, str) or not relative_path.strip():
                    blockers.append(f"{label}: evidence paths must be nonempty strings")
                elif not (ROOT / relative_path).exists():
                    blockers.append(f"{label}: missing evidence path {relative_path}")

        course_sections = requirement.get("course_sections", [])
        if not isinstance(course_sections, list):
            blockers.append(f"{label}: course_sections must be a list when present")
        else:
            covered_course_sections.update(str(section) for section in course_sections)

        method_names = requirement.get("method_registry_names", [])
        if not isinstance(method_names, list):
            blockers.append(f"{label}: method_registry_names must be a list when present")
        else:
            mapped_method_names.extend(str(name) for name in method_names)

    duplicate_ids = sorted(name for name, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        blockers.append(f"roadmap registry: duplicate requirement ids {duplicate_ids}")

    if method_registry_path.exists():
        current_method_names = _method_registry_names(method_registry_path)
        missing_methods = sorted(set(current_method_names) - set(mapped_method_names))
        duplicate_methods = sorted(
            name for name, count in Counter(mapped_method_names).items() if count > 1
        )
        unknown_methods = sorted(set(mapped_method_names) - set(current_method_names))
        if missing_methods:
            blockers.append(
                f"roadmap registry: method_registry.csv rows not represented {missing_methods}"
            )
        if duplicate_methods:
            blockers.append(
                f"roadmap registry: method_registry.csv rows represented more than once {duplicate_methods}"
            )
        if unknown_methods:
            blockers.append(
                f"roadmap registry: unknown method_registry_names {unknown_methods}"
            )
    else:
        blockers.append("roadmap registry: docs/method_registry.csv is missing")

    if course_status_path.exists():
        current_sections = set(_course_status_sections(course_status_path))
        missing_sections = sorted(current_sections - covered_course_sections)
        unknown_sections = sorted(covered_course_sections - current_sections)
        if missing_sections:
            blockers.append(
                f"roadmap registry: tracked extension sections not represented {missing_sections}"
            )
        if unknown_sections:
            blockers.append(
                f"roadmap registry: unknown course_sections {unknown_sections}"
            )
    else:
        blockers.append("roadmap registry: docs/arena_style_rewrite_status.yml is missing")

    return blockers


def roadmap_requirement_status_blockers(
    registry_path: Path = ROADMAP_REQUIREMENT_REGISTRY_PATH,
) -> list[str]:
    """Return release blockers implied by honest roadmap statuses."""

    try:
        registry = load_roadmap_requirement_registry(registry_path)
    except (OSError, yaml.YAMLError):
        return []
    blockers: list[str] = []
    for requirement in registry.get("requirements", []):
        if not isinstance(requirement, dict):
            continue
        if requirement.get("status") == "UNIMPLEMENTED":
            blockers.append(
                f"{requirement.get('id', '<missing id>')}: UNIMPLEMENTED - "
                f"{requirement.get('blocker') or requirement.get('reason') or 'no reason supplied'}"
            )
    return blockers


REPORT_REQUIREMENTS: tuple[ReportRequirement, ...] = (
    # GT-0 exact / controlled mathematical coverage.
    ReportRequirement(
        "gt0_circuit_discovery",
        "GT-0",
        "Circuit discovery has an exact/toy sparse-feature circuit contract.",
        "chapter8_automated_circuits/exercises/part5_sparse_feature_circuits/verification_report.json",
        "chapter8_automated_circuits/exercises/part5_sparse_feature_circuits/artifacts.lock.yml",
        {
            "gt_tier": "GT-0",
            "metrics.gpu_test.official_sae_feature_attribution_passed": True,
            "metrics.gpu_test.official_sae_feature_attribution_random_fraction": 0.0,
        },
        ("one_layer_official_resid5_sae_feature_attribution_random_control",),
        "GT-0",
    ),
    ReportRequirement(
        "gt0_representation_geometry",
        "GT-0",
        "Geometry has exact mathematical controls and learner-owned Mistral calendar interventions.",
        "chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/verification_report.json",
        "chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/artifacts.lock.yml",
        {
            "gt_tier": "GT-0",
            "metrics.gpu_test.actual_learner_passed": True,
            "metrics.gpu_test.own_answer_code_cells": 35,
            "metrics.gpu_test.embedded_figures": 13,
            "metrics.gpu_test.final_prompts": 579,
            "metrics.gpu_test.all_donor_pairs": 5634,
            "metrics.gpu_test.retained_outcomes": 120649,
            "metrics.gpu_test.null_draws_per_kind": 32,
            "metrics.gpu_test.projection_fits": 120,
            "metrics.gpu_test.fresh_compared_examples": 6656,
            "metrics.gpu_test.fresh_max_logit_error": {"max": 1e-6},
            "metrics.gpu_test.fresh_top_token_mismatches": 0,
            "metrics.gpu_test.final_used_for_selection": False,
            "metrics.gpu_test.tsne_is_transductive": True,
        },
        ("same_rank_random_donor_plane_32_draws", "equal_norm_random_edit_32_draws",
         "class_order_permuted_circular_probe_32_draws", "independent_noise_reducer_control",
         "permuted_training_labels", "no_op", "mean_only", "leave_one_calendar_class_out"),
        "GT-0",
    ),
    ReportRequirement(
        "gt0_representation_alignment",
        "GT-0",
        "Own-answer CCA/SVCCA/Procrustes oracles and native GPT-2 donor transport, independently reproduced on CPU.",
        "chapter11_representation_geometry/exercises/part2_cca_svcca_alignment/verification_report.json",
        "chapter11_representation_geometry/exercises/part2_cca_svcca_alignment/artifacts.lock.yml",
        {
            "gt_tier": "GT-0",
            "metrics.gpu_test.actual_learner_passed": True,
            "metrics.gpu_test.native_model_reloaded": True,
            "metrics.gpu_test.fresh_reproduction_passed": True,
            "metrics.gpu_test.own_answer_code_cells": 20,
            "metrics.gpu_test.embedded_figures": 9,
            "metrics.gpu_test.retained_outcomes": 12816,
            "metrics.gpu_test.main_forward_rows": 12888,
            "metrics.gpu_test.fresh_compared_examples": 5184,
            "metrics.gpu_test.fresh_condition_batches": 108,
            "metrics.gpu_test.fresh_max_logit_error": {"max": 1e-5},
            "metrics.gpu_test.fresh_top_token_mismatches": 0,
            "metrics.gpu_test.final_used_for_selection": False,
            "metrics.gpu_test.whole_paper_replication": False,
        },
        ("independent_views", "shuffled_training_correspondence", "five_seed_width_sample_noise_sweep",
         "low_variance_causal_feature", "native_no_op", "actual_target_layer_donor", "equal_rank_pca_procrustes",
         "ridge_transport", "unseen_name_split", "same_norm_random_edits_16_draws", "fresh_native_cpu_model_all_conditions"),
        "GT-0",
    ),
    ReportRequirement(
        "gt0_distributed_alignment_search",
        "GT-0",
        "Own-answer structural and gradient oracles, frozen GPT-2 interchanges, and fresh CPU reproduction of every final condition.",
        "chapter11_representation_geometry/exercises/part3_causal_abstraction_das/verification_report.json",
        "chapter11_representation_geometry/exercises/part3_causal_abstraction_das/artifacts.lock.yml",
        {
            "gt_tier": "GT-0",
            "metrics.gpu_test.actual_learner_passed": True,
            "metrics.gpu_test.native_model_reloaded": True,
            "metrics.gpu_test.fresh_reproduction_passed": True,
            "metrics.gpu_test.own_answer_code_cells": 32,
            "metrics.gpu_test.embedded_figures": 12,
            "metrics.gpu_test.final_prompt_count": 96,
            "metrics.gpu_test.retained_final_outcomes": 16512,
            "metrics.gpu_test.fresh_compared_examples": 16512,
            "metrics.gpu_test.fresh_condition_batches": 43,
            "metrics.gpu_test.fresh_native_forward_rows": 17232,
            "metrics.gpu_test.fresh_max_logit_error": {"max": 1e-5},
            "metrics.gpu_test.fresh_top_token_mismatches": 0,
            "metrics.gpu_test.primary_metric_matches_actual_tokens": True,
            "metrics.gpu_test.weights_unchanged": True,
            "metrics.gpu_test.model_gradients_absent": True,
            "metrics.gpu_test.final_used_for_selection": False,
            "metrics.gpu_test.whole_paper_replication": False,
        },
        ("exact_rotated_causal_organism", "perfectly_predictive_unused_copy",
         "independent_projector_finite_difference", "unchanged_side_variable",
         "three_alignment_seeds", "three_shuffled_target_fits", "32_equal_rank_random_subspaces",
         "whole_position_donor", "observational_ridge_role_probe", "explicit_counterfactual_prompt",
         "conditional_and_unconditional_denominators", "fresh_cpu_all_full_vocabulary_outputs"),
        "GT-0",
    ),
    ReportRequirement(
        "gt0_concept_erasure",
        "GT-0",
        "Own-answer affine erasure, independent constrained and author oracles, and freshly refitted native GPT-2 controls.",
        "chapter11_representation_geometry/exercises/part4_concept_erasure/verification_report.json",
        "chapter11_representation_geometry/exercises/part4_concept_erasure/artifacts.lock.yml",
        {
            "gt_tier": "GT-0",
            "metrics.gpu_test.actual_learner_passed": True,
            "metrics.gpu_test.native_model_reloaded": True,
            "metrics.gpu_test.fresh_reproduction_passed": True,
            "metrics.gpu_test.own_answer_code_cells": 24,
            "metrics.gpu_test.embedded_figures": 12,
            "metrics.gpu_test.final_prompt_count": 672,
            "metrics.gpu_test.retained_final_outcomes": 71232,
            "metrics.gpu_test.fresh_compared_examples": 71232,
            "metrics.gpu_test.fresh_condition_batches": 106,
            "metrics.gpu_test.fresh_max_logit_error": {"max": 1e-5},
            "metrics.gpu_test.fresh_top_token_mismatches": 0,
            "metrics.gpu_test.primary_metric_matches_actual_tokens": True,
            "metrics.gpu_test.all_erasers_and_probes_refitted": True,
            "metrics.gpu_test.weights_unchanged": True,
            "metrics.gpu_test.model_gradients_absent": True,
            "metrics.gpu_test.final_used_for_selection": False,
            "metrics.gpu_test.whole_paper_replication": False,
        },
        ("four_point_ground_truth", "independent_numpy_constrained_optimum", "pinned_author_reference_parity",
         "singular_and_translated_covariance", "old_vs_refitted_probe", "nonlinear_equal_mean_counterexample",
         "independent_probe_fitting_split", "two_prescribed_sites", "16_random_rank_one_draws_per_site",
         "16_per_example_norm_matched_draws_per_eraser_per_site", "unrelated_object_task",
         "all_native_errors_retained", "full_vocabulary_metrics", "fresh_cpu_all_conditions_and_refits"),
        "GT-0",
    ),
    ReportRequirement(
        "gt0_contrast_consistent_search",
        "GT-0",
        "Original CCS loss, exact competing-feature oracles, fresh native capture and all independently refitted reporters.",
        'chapter11_representation_geometry/exercises/part5_contrast_consistent_search/verification_report.json',
        'chapter11_representation_geometry/exercises/part5_contrast_consistent_search/artifacts.lock.yml',
        {
            'gt_tier': 'GT-0',
            'metrics.gpu_test.native_model_reloaded': True,
            'metrics.gpu_test.actual_learner_passed': True,
            'metrics.gpu_test.fresh_reproduction_passed': True,
            'metrics.gpu_test.own_answer_code_cells': 32,
            'metrics.gpu_test.embedded_figures': 15,
            'metrics.gpu_test.final_prompt_count': 512,
            'metrics.gpu_test.distinct_final_base_reviews': 64,
            'metrics.gpu_test.retained_final_outcomes': 98304,
            'metrics.gpu_test.fresh_compared_examples': 3586,
            'metrics.gpu_test.fresh_condition_batches': 18,
            'metrics.gpu_test.refitted_reporters': 1152,
            'metrics.gpu_test.fresh_max_logit_error': {'max': 1e-05},
            'metrics.gpu_test.fresh_top_token_mismatches': 0,
            'metrics.gpu_test.primary_metric_matches_actual_tokens': True,
            'metrics.gpu_test.all_reporters_refitted': True,
            'metrics.gpu_test.weights_unchanged': True,
            'metrics.gpu_test.model_gradients_absent': True,
            'metrics.gpu_test.final_used_for_selection': False,
            'metrics.gpu_test.whole_paper_replication': False,
        },
        (
            'exact_competing_feature_counterexample',
            'pinned_author_loss_centering_parity',
            'asymmetric_loss_complement_check',
            'single_vs_batched_restart_parity',
            'train_only_branch_centering',
            'fixed_calibration_orientation',
            'ten_all_restarts_per_mode',
            'raw_no_centering_control',
            'shuffled_pair_control',
            'paired_difference_pca',
            'supervised_labelled_probe',
            '32_matched_random_directions',
            'neutral_reader_context_control',
            'independent_opinion_crossed_with_target',
            'all_three_sites_and_two_wrappers',
            'all_native_errors_retained',
            'restricted_vs_full_vocabulary_outputs',
            'native_padding_and_full_model_parity',
            'fixed_reporter_native_play',
            'fresh_cpu_all_captures_and_refits',
        ),
        "GT-0",
    ),
    ReportRequirement(
        "gt0_shapley",
        "GT-0",
        "SHAP/Shapley has exact ground-truth games.",
        "chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley_ground_truth_games/verification_report.json",
        "chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley_ground_truth_games/artifacts.lock.yml",
        {
            "gt_tier": "GT-0",
            "metrics.gpu_test.satisfies_efficiency": True,
            "metrics.gpu_test.shuffled_control_rejected": True,
        },
        ("efficiency_axiom_check_on_trained_model", "shuffled_label_trained_model_negative_control"),
        "GT-0",
    ),
    ReportRequirement(
        "gt0_world_models",
        "GT-0",
        "World-model claims include exact toy target/transition controls.",
        "chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls/verification_report.json",
        "chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls/artifacts.lock.yml",
        {
            "metrics.gpu_test.jepa_predicts_target": True,
            "metrics.gpu_test.transition_consistent": True,
        },
        ("toy_world_state_probe", "toy_action_transition_consistency"),
    ),
    ReportRequirement(
        "gt0_vlms",
        "GT-0",
        "VLM attribution has exact modality/region SHAP controls.",
        "chapter16_shapley_attribution_baselines/exercises/part5_vlm_modality_region_shap/verification_report.json",
        "chapter16_shapley_attribution_baselines/exercises/part5_vlm_modality_region_shap/artifacts.lock.yml",
        {
            "gt_tier": "GT-0",
            "metrics.gpu_test.modality_satisfies_efficiency": True,
            "metrics.gpu_test.region_satisfies_efficiency": True,
        },
        ("structured_object_background_ocr_region_coalition_table",),
        "GT-0",
    ),
    ReportRequirement(
        "gt0_diffusion_image_generation",
        "GT-0",
        "Diffusion / image generation has toy noising and sampler ground truth.",
        "chapter5_modern_architectures/exercises/part5_diffusion_language_models/verification_report.json",
        "chapter5_modern_architectures/exercises/part5_diffusion_language_models/artifacts.lock.yml",
        {
            "gt_tier": "GT-0",
            "metrics.gpu_test.activation_trajectory_shape_ok": True,
            "metrics.gpu_test.shuffled_control_fails": True,
            "metrics.gpu_test.text_own_answer_code_cells": 43,
            "metrics.gpu_test.text_training_runs": 12,
            "metrics.gpu_test.text_generation_conditions": 126,
            "metrics.gpu_test.text_generated_examples": 64512,
            "metrics.gpu_test.text_figures": 12,
            "metrics.gpu_test.text_fresh_generated_examples": 21504,
            "metrics.gpu_test.text_all_fresh_outcomes_match_learner": True,
            "metrics.gpu_test.text_final_used_for_selection": False,
            "metrics.gpu_test.text_natural_language_quality_established": False,
            "metrics.gpu_test.diffusiongemma_config_supported": True,
            "metrics.gpu_test.diffusiongemma_processor_supported": True,
            "metrics.gpu_test.diffusiongemma_model_class_supported": True,
            "metrics.gpu_test.diffusiongemma_generation_ready": True,
        },
        (
            "forward_noising_extreme_timestep_contract",
            "independently_trained_shuffled_label_denoiser",
        ),
        "GT-0",
    ),
    ReportRequirement(
        "gt0_lora_dora",
        "GT-0",
        "LoRA/DoRA has exact rank, merge, and decomposition controls.",
        "chapter15_peft_misalignment/exercises/part1_lora_dora_adapter_controls/verification_report.json",
        "chapter15_peft_misalignment/exercises/part1_lora_dora_adapter_controls/artifacts.lock.yml",
        {
            "gt_tier": "GT-0",
            "metrics.gpu_test.dora_norm_preserved": True,
            "metrics.gpu_test.trained_lora_adapter_rank": 1,
        },
        ("exact_lora_delta_contract", "exact_dora_row_magnitude_contract"),
        "GT-0",
    ),
    # GT-1 reference / released-weight coverage.
    ReportRequirement(
        "gt1_gemma",
        "GT-1",
        "Gemma-style block has reference implementation parity.",
        "chapter5_modern_architectures/exercises/part1_gemma_from_scratch/verification_report.json",
        "chapter5_modern_architectures/exercises/part1_gemma_from_scratch/artifacts.lock.yml",
        {
            "metrics.gpu_test.reference_parity_passed": True,
            "metrics.gpu_test.reference_model_family": "transformers.GemmaForCausalLM",
        },
        ("huggingface_gemma_reference_architecture_logits_parity",),
    ),
    ReportRequirement(
        "gt1_mamba",
        "GT-1",
        "Mamba has official checkpoint and scan/cache parity evidence.",
        "chapter5_modern_architectures/exercises/part3_mamba_from_scratch/verification_report.json",
        "chapter5_modern_architectures/exercises/part3_mamba_from_scratch/artifacts.lock.yml",
        {
            "gt_tier": "GT-1",
            "metrics.gpu_test.official_mamba_logits_generation_preflight_passed": True,
            "metrics.gpu_test.scan_passed": True,
        },
        ("pinned_official_mamba_130m_hf_logits_generation",),
        "GT-1",
    ),
    ReportRequirement(
        "gt1_clip_siglip",
        "GT-1",
        "CLIP/SigLIP real local extraction and retrieval runs.",
        "chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/verification_report.json",
        "chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/artifacts.lock.yml",
        {
            "metrics.gpu_test.real_clip_rendered_shape_preflight_passed": True,
            "metrics.gpu_test.real_siglip_rendered_shape_preflight_passed": True,
            "metrics.gpu_test.real_clip_visual_token_activation_patching_preflight_passed": True,
            "metrics.gpu_test.real_siglip_visual_token_activation_patching_preflight_passed": True,
            "metrics.gpu_test.real_clip_activation_patch_full_sequence_matches_corrupt": True,
            "metrics.gpu_test.real_siglip_activation_patch_full_sequence_matches_corrupt": True,
        },
        (
            "pinned_real_clip_rendered_shape_retrieval",
            "pinned_real_clip_hidden_visual_token_activation_patch",
            "real_clip_full_visual_sequence_activation_patch_control",
            "pinned_real_siglip_safetensors_rendered_shape_retrieval",
            "pinned_real_siglip_hidden_visual_token_activation_patch",
            "real_siglip_full_visual_sequence_activation_patch_control",
        ),
    ),
    ReportRequirement(
        "gt1_vlm_loading",
        "GT-1",
        "Real local VLM loading/generation is verified.",
        "chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/verification_report.json",
        "chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/artifacts.lock.yml",
        {
            "metrics.gpu_test.real_qwen25_vl_generation_preflight_passed": True,
            "metrics.gpu_test.real_qwen25_vl_accuracy": 1.0,
        },
        ("pinned_real_qwen25_vl_rendered_shape_generation",),
    ),
    ReportRequirement(
        "gt1_lora_dora",
        "GT-1",
        "LoRA/DoRA trainable adapter behavior is compared to references/baselines.",
        "chapter15_peft_misalignment/exercises/part1_lora_dora_adapter_controls/verification_report.json",
        "chapter15_peft_misalignment/exercises/part1_lora_dora_adapter_controls/artifacts.lock.yml",
        {
            "metrics.gpu_test.trained_lora_merge_max_abs_diff": 0.0,
            "metrics.gpu_test.matched_peft_comparison_passed": True,
        },
        ("merge_unmerge_logit_parity", "matched_rank1_lora_vs_rank1_dora_vs_full_finetune_comparison"),
    ),
    ReportRequirement(
        "gt1_diffusion_model_loading",
        "GT-1",
        "Image-generation model loading and SD1.5 interpretability controls are verified on CUDA.",
        "chapter13_image_generation_interpretability/exercises/part1_diffusion_image_controls/verification_report.json",
        "chapter13_image_generation_interpretability/exercises/part1_diffusion_image_controls/artifacts.lock.yml",
        {
            "gt_tier": "GT-1",
            "metrics.gpu_test.sd_turbo_preflight_passed": True,
            "metrics.gpu_test.sd_turbo_cross_attention_localized": True,
            "metrics.gpu_test.sd15_strict_experiment_passed": True,
            "metrics.gpu_test.sd15_model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "metrics.gpu_test.sd15_revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
            "metrics.gpu_test.sd15_fixed_seed_generation_passed": True,
            "metrics.gpu_test.sd15_daam_baseline_included": True,
            "metrics.gpu_test.sd15_cross_attention_maps_captured": True,
            "metrics.gpu_test.sd15_token_ablation_passed": True,
            "metrics.gpu_test.sd15_random_token_ablation_weaker": True,
            "metrics.gpu_test.sd15_image_quality_preserved": True,
            "metrics.gpu_test.sd15_white_noise_rejected": True,
            "metrics.gpu_test.sd15_min_target_control_attention_gap": {"min": 0.005},
            "metrics.gpu_test.sd15_min_target_lift_over_mask_fraction": {"min": 0.01},
            "metrics.gpu_test.sd15_min_captured_cross_attention_map_count": {"min": 32},
            "metrics.gpu_test.sd15_min_target_ablation_drop": {"min": 0.05},
            "metrics.gpu_test.sd15_max_random_control_drop": {"max": 0.0},
            "metrics.gpu_test.sd15_min_target_region_fraction": {"min": 0.02},
            "metrics.gpu_test.sd15_max_high_frequency_energy": {"max": 0.12},
            "metrics.gpu_test.sd15_min_white_noise_high_frequency_gap": {"min": 0.12},
            "metrics.gpu_test.sd15_clip_image_to_text_accuracy": 1.0,
            "metrics.gpu_test.sd15_clip_text_to_image_accuracy": 1.0,
            "metrics.gpu_test.sd15_clip_mean_positive_margin": {"min": 2.0},
        },
        (
            "pinned_sd_turbo_safetensors_generation",
            "pinned_sd_turbo_cross_attention_capture",
            "pinned_sd15_safetensors_generation",
            "sd15_daam_cross_attention_capture",
            "sd15_target_token_ablation_control",
            "sd15_random_token_ablation_control",
            "sd15_image_quality_metric",
            "sd15_white_noise_control",
        ),
        "GT-1",
    ),
    ReportRequirement(
        "gt1_jepa_loading",
        "GT-1",
        "V-JEPA 2 feature extraction loads and runs locally.",
        "chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls/verification_report.json",
        "chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls/artifacts.lock.yml",
        {
            "gt_tier": "GT-1",
            "metrics.gpu_test.vjepa2_preflight_passed": True,
            "metrics.gpu_test.vjepa2_synthetic_occlusion_permanence_passed": True,
            "metrics.gpu_test.vjepa2_world_model_controls_passed": True,
            "metrics.gpu_test.masked_prediction_passed": True,
            "metrics.gpu_test.latent_rollout_passed": True,
        },
        (
            "pinned_vjepa2_vitl_safetensors_feature_extraction",
            "frozen_vjepa2_masked_latent_prediction_head",
            "action_conditioned_latent_rollout_head",
        ),
        "GT-1",
    ),
    # GT-2 published-replication style coverage.
    ReportRequirement(
        "gt2_sparse_feature_circuits",
        "GT-2",
        "Sparse Feature Circuits has official-code/artifact replication evidence.",
        "chapter8_automated_circuits/exercises/part5_sparse_feature_circuits/verification_report.json",
        "chapter8_automated_circuits/exercises/part5_sparse_feature_circuits/artifacts.lock.yml",
        {
            "metrics.gpu_test.official_sparse_feature_circuit_replication_passed": True,
            "metrics.gpu_test.official_sparse_feature_circuit_examples": 100,
            "metrics.gpu_test.official_sparse_feature_circuit_faithfulness_passed": True,
            "metrics.gpu_test.shift_editing_passed": True,
            "metrics.gpu_test.shift_editing_random_edit_control_fails": True,
            "metrics.gpu_test.shift_editing_ood_improvement": {"min": 0.5},
        },
        (
            "official_code_100_example_sparse_feature_graph_artifact",
            "official_heldout_simple_test_faithfulness_evaluation",
            "generated_shift_style_spurious_feature_editing",
            "same_size_random_feature_edit_control",
        ),
    ),
    ReportRequirement(
        "gt2_refusal_direction",
        "GT-2",
        "Own-answer refusal-direction study with train-only fitting, development-frozen final generation and independent CUDA regeneration; bounded model/dataset scope.",
        "chapter9_alignment_interpretability/exercises/part1_refusal_directions_safe_steering/verification_report.json",
        "chapter9_alignment_interpretability/exercises/part1_refusal_directions_safe_steering/artifacts.lock.yml",
        {
            "gt_tier": "GT-2",
            "metrics.gpu_test.accepted": True,
            "metrics.gpu_test.dataset_id": "josephmayo/refusal-compliance-pairs",
            "metrics.gpu_test.completed_learner_code_cells": 26,
            "metrics.gpu_test.stored_learner_audit.raw_metrics_recomputed": True,
            "metrics.gpu_test.stored_learner_audit.selection_passed": True,
            "metrics.gpu_test.stored_learner_audit.development_candidates": 15,
            "metrics.gpu_test.stored_learner_audit.final_conditions": 85,
            "metrics.gpu_test.stored_learner_audit.final_separation.accuracy": {"min": 0.9},
            "metrics.gpu_test.stored_learner_audit.addition_on_benign.delta": {"min": 0.125},
            "metrics.gpu_test.stored_learner_audit.ablation_on_refusal.delta": {"max": -0.25},
            "metrics.gpu_test.fresh_generation_examples": 3640,
            "metrics.gpu_test.fresh_cuda_conditions": 34,
            "metrics.gpu_test.fresh_separation_directions": 65,
            "metrics.gpu_test.all_fresh_outcomes_match_learner": True,
            "metrics.gpu_test.fresh_score_max_error": {"max": 1e-4},
            "metrics.gpu_test.fresh_training_vector_max_error": {"max": 1e-4},
            "metrics.gpu_test.marker_absence_is_safety_or_correctness": False,
            "metrics.gpu_test.paper_equivalent_full_replication": False,
        },
        (
            "training_only_direction_and_threshold",
            "shifted_evaluation_threshold_failure",
            "genuine_training_label_permutations_32_draws",
            "same_norm_random_directions_32_draws",
            "all_24_layers_and_four_positions_including_undefined",
            "centered_training_pca_and_null_alignment",
            "all_15_development_candidates_including_capability_failures",
            "frozen_selection_before_final_prompts",
            "all_85_final_amplitude_and_null_conditions",
            "paired_marker_transition_denominators",
            "independent_32_question_exact_benign_task_baseline",
            "persistent_cached_and_uncached_generation_parity",
            "independent_weight_projection_oracle_on_disposable_tiny_qwen",
            "explicit_marker_false_positive_and_false_negative_examples",
            "fresh_cuda_regeneration_of_all_34_unit_strength_conditions",
        ),
        "GT-2",
    ),
    ReportRequirement(
        "gt2_gemma_scope_feature_analysis",
        "GT-2",
        "Gemma Scope semantic feature analysis runs on real Gemma activations.",
        "chapter6_sparse_feature_methods/exercises/part2_gemma_scope_deep_dive/verification_report.json",
        "chapter6_sparse_feature_methods/exercises/part2_gemma_scope_deep_dive/artifacts.lock.yml",
        {
            "metrics.gpu_test.gemma_scope_real_activation_preflight_passed": True,
            "metrics.gpu_test.gemma_scope_real_activation_model_id": "google/gemma-3-1b-it",
            "metrics.gpu_test.gemma_scope_real_activation_feature_auc": {"min": 0.95},
            "metrics.gpu_test.gemma_scope_real_activation_auc_margin": {"min": 0.25},
            "metrics.gpu_test.gemma_scope_real_activation_label_shuffle_auc": {"max": 0.05},
            "metrics.gpu_test.gemma_scope_semantic_feature_claimed": True,
            "metrics.gpu_test.gemma3_base_ready_for_real_activations": True,
        },
        (
            "real_gemma3_layer13_activation_capture",
            "heldout_real_gemma_activation_semantic_feature_auc",
            "real_gemma_activation_random_feature_control",
            "real_gemma_activation_label_shuffle_control",
        ),
    ),
    # GT-3 controlled-proxy coverage.
    ReportRequirement(
        "gt3_vlm_hallucination_grounding",
        "GT-3",
        "VLM grounding/modality controls include image/text baselines and object patching.",
        "chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/verification_report.json",
        "chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/artifacts.lock.yml",
        {
            "metrics.gpu_test.object_patch_flips_answer": True,
            "metrics.gpu_test.text_only_fails_image_questions": True,
            "metrics.gpu_test.joint_beats_text_only": True,
            "metrics.gpu_test.visual_sequence_patch_passed": True,
            "metrics.gpu_test.real_clip_activation_patch_random_control_same_size": True,
            "metrics.gpu_test.real_siglip_activation_patch_random_control_same_size": True,
        },
        (
            "text_only_prior_failure_baseline",
            "object_region_patch",
            "real_clip_hidden_same_size_random_token_activation_patch_control",
            "real_siglip_hidden_same_size_random_token_activation_patch_control",
        ),
    ),
    ReportRequirement(
        "gt3_vlm_clothing_geometry",
        "GT-3",
        "VLM clothing/object/color/style geometry uses controlled labels and confound controls.",
        "chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/verification_report.json",
        "chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/artifacts.lock.yml",
        {
            "metrics.gpu_test.clothing_predicts_factors": True,
            "metrics.gpu_test.clothing_rejects_text_prior": True,
            "metrics.gpu_test.clothing_rejects_random_labels": True,
        },
        ("synthetic_clothing_garment_color_style_schema", "clothing_text_prior_corruption_control"),
    ),
    ReportRequirement(
        "gt3_image_generation_concept_directions",
        "GT-3",
        "Image-generation concept directions have controlled SD1.5 prompt/region tests.",
        "chapter13_image_generation_interpretability/exercises/part1_diffusion_image_controls/verification_report.json",
        "chapter13_image_generation_interpretability/exercises/part1_diffusion_image_controls/artifacts.lock.yml",
        {
            "metrics.gpu_test.latent_direction_effect": True,
            "metrics.gpu_test.region_selective": True,
            "metrics.gpu_test.sd_turbo_cross_attention_localized": True,
            "metrics.gpu_test.sd15_strict_experiment_passed": True,
            "metrics.gpu_test.sd15_daam_baseline_included": True,
            "metrics.gpu_test.sd15_token_ablation_passed": True,
            "metrics.gpu_test.sd15_random_token_ablation_weaker": True,
            "metrics.gpu_test.sd15_image_quality_preserved": True,
            "metrics.gpu_test.sd15_white_noise_rejected": True,
            "metrics.gpu_test.sd15_min_target_control_attention_gap": {"min": 0.005},
            "metrics.gpu_test.sd15_min_target_ablation_drop": {"min": 0.05},
        },
        (
            "latent_direction_random_control",
            "target_token_vs_background_token_attention_control",
            "sd15_daam_cross_attention_capture",
            "sd15_target_token_ablation_control",
            "sd15_random_token_ablation_control",
            "sd15_image_quality_metric",
            "sd15_white_noise_control",
        ),
    ),
    ReportRequirement(
        "gt3_safe_lora_misalignment_proxy",
        "GT-3",
        "Safe LoRA proxy drift uses generated labels and random controls.",
        "chapter15_peft_misalignment/exercises/part1_lora_dora_adapter_controls/verification_report.json",
        "chapter15_peft_misalignment/exercises/part1_lora_dora_adapter_controls/artifacts.lock.yml",
        {
            "metrics.gpu_test.trained_lora_preflight_passed": True,
            "metrics.gpu_test.trained_lora_random_label_control_fails": True,
            "metrics.gpu_test.trained_lora_random_adapter_control_fails": True,
        },
        ("trained_rank1_lora_safe_proxy_adapter", "random_label_training_control", "same_norm_random_adapter_control"),
    ),
    ReportRequirement(
        "gt3_jepa_object_permanence",
        "GT-3",
        "JEPA object permanence uses controlled occlusion and negative controls.",
        "chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls/verification_report.json",
        "chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls/artifacts.lock.yml",
        {
            "metrics.gpu_test.real_latent_object_permanence_passed": True,
            "metrics.gpu_test.causal_latent_patching_passed": True,
            "metrics.gpu_test.causal_latent_patch_random_gap": {"min": 0.4},
        },
        (
            "real_latent_occluded_vs_absent_object_permanence",
            "causal_latent_token_patching_control",
            "same_size_random_token_patch_negative_control",
        ),
    ),
)


LEGACY_REQUIREMENTS: tuple[LegacyNotebookRequirement, ...] = (
    LegacyNotebookRequirement(
        "gt2_othello_gpt",
        "GT-2",
        "Othello-GPT published world-model replication has original ARENA material.",
        (
            "chapter1_transformer_interp/exercises/part53_othellogpt/solutions.py",
            "chapter1_transformer_interp/instructions/pages/33_[1.5.3]_OthelloGPT.md",
        ),
        "docs/evidence/othello_gpt/verification_report.json",
    ),
    LegacyNotebookRequirement(
        "gt2_ioi_path_patching",
        "GT-2",
        "IOI/path-patching published replication has original ARENA material.",
        (
            "chapter1_transformer_interp/exercises/part41_indirect_object_identification/ioi_circuit_extraction.py",
            "chapter1_transformer_interp/instructions/pages/21_[1.4.1]_Indirect_Object_Identification.md",
            "chapter1_transformer_interp/instructions/pages/22_[1.4.2]_SAE_Circuits.md",
        ),
        "docs/evidence/ioi_path_patching/verification_report.json",
    ),
)


def report_requirement_blockers(requirement: ReportRequirement) -> list[str]:
    blockers: list[str] = []
    report_file = ROOT / requirement.report_path
    lock_file = ROOT / requirement.lock_path
    if not report_file.exists():
        return [f"{requirement.requirement_id}: missing report {requirement.report_path}"]
    if not lock_file.exists():
        return [f"{requirement.requirement_id}: missing lock {requirement.lock_path}"]

    report = _load_json(requirement.report_path)
    lock = _load_yaml(requirement.lock_path)
    if report.get("accepted") is not True or report.get("tests_passed") is not True:
        blockers.append(f"{requirement.requirement_id}: report is not accepted")
    evidence = report.get("metrics", {}).get("gpu_evidence", {})
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence
    if (evidence.get("category") != "cuda_section_metric" or evidence.get("uses_cuda") is not True) and not has_native_cpu_model_evidence(lock, report):
        blockers.append(f"{requirement.requirement_id}: lacks section-specific CUDA evidence")
    if requirement.require_primary_gt_tier is not None:
        actual_tier = report.get("gt_tier")
        if actual_tier != requirement.require_primary_gt_tier:
            blockers.append(
                f"{requirement.requirement_id}: expected primary gt_tier "
                f"{requirement.require_primary_gt_tier}, got {actual_tier}"
            )

    for metric_path, expected in requirement.metric_expectations.items():
        actual = _get_path(report, metric_path)
        if not _matches(actual, expected):
            blockers.append(
                f"{requirement.requirement_id}: expected {metric_path} == "
                f"{expected!r}, got {actual!r}"
            )

    controls = set(lock.get("controls", []))
    missing_controls = sorted(set(requirement.required_controls) - controls)
    if missing_controls:
        blockers.append(
            f"{requirement.requirement_id}: missing controls {missing_controls}"
        )
    return blockers


def legacy_requirement_blockers(requirement: LegacyNotebookRequirement) -> list[str]:
    blockers: list[str] = []
    for relative_path in requirement.required_paths:
        if not (ROOT / relative_path).exists():
            blockers.append(f"{requirement.requirement_id}: missing {relative_path}")
    if requirement.modern_report_path is None:
        blockers.append(
            f"{requirement.requirement_id}: original ARENA material exists, but no "
            "extension verification_report.json currently proves the published "
            "replication under the new CUDA/evidence contract"
        )
    elif not (ROOT / requirement.modern_report_path).exists():
        blockers.append(
            f"{requirement.requirement_id}: missing modern report "
            f"{requirement.modern_report_path}"
        )
    else:
        report = _load_json(requirement.modern_report_path)
        claim_text = json.dumps(report, sort_keys=True).lower()
        evidence = report.get("metrics", {}).get("gpu_evidence", {})
        gpu_test = report.get("metrics", {}).get("gpu_test", {})
        if report.get("gt_tier") != "GT-2":
            blockers.append(
                f"{requirement.requirement_id}: expected GT-2 report, "
                f"got {report.get('gt_tier')!r}"
            )
        if report.get("accepted") is not True or report.get("tests_passed") is not True:
            blockers.append(f"{requirement.requirement_id}: report is not accepted")
        if evidence.get("category") != "cuda_section_metric" or evidence.get("uses_cuda") is not True:
            blockers.append(
                f"{requirement.requirement_id}: lacks section-specific CUDA evidence"
            )
        if requirement.requirement_id == "gt2_othello_gpt":
            expected_metrics = {
                "model_loaded": True,
                "sample_first_top3_matches_expected": True,
                "within_vram_budget": True,
            }
            for key, expected in expected_metrics.items():
                if gpu_test.get(key) != expected:
                    blockers.append(
                        f"{requirement.requirement_id}: expected gpu_test.{key} "
                        f"== {expected!r}, got {gpu_test.get(key)!r}"
                    )
            numeric_expectations = {
                "legal_top1_accuracy": ("min", 0.99),
                "board_probe_accuracy": ("min", 0.95),
                "swapped_parity_probe_accuracy": ("max", 0.60),
                "peak_vram_gb": ("max", 8.0),
            }
            for key, (direction, threshold) in numeric_expectations.items():
                actual = gpu_test.get(key)
                if not isinstance(actual, float | int):
                    blockers.append(
                        f"{requirement.requirement_id}: missing numeric gpu_test.{key}"
                    )
                elif direction == "min" and float(actual) < threshold:
                    blockers.append(
                        f"{requirement.requirement_id}: expected gpu_test.{key} "
                        f">= {threshold}, got {actual!r}"
                    )
                elif direction == "max" and float(actual) > threshold:
                    blockers.append(
                        f"{requirement.requirement_id}: expected gpu_test.{key} "
                        f"<= {threshold}, got {actual!r}"
                    )
            if "othello" not in claim_text:
                blockers.append(
                    f"{requirement.requirement_id}: modern report does not contain "
                    "Othello-GPT verification evidence"
                )
        if requirement.requirement_id == "gt2_ioi_path_patching":
            expected_metrics = {
                "preflight_passed": True,
                "published_heads_beat_random_controls": True,
                "random_controls_passed": True,
                "negative_name_mover_sign_control_passed": True,
            }
            for key, expected in expected_metrics.items():
                if gpu_test.get(key) != expected:
                    blockers.append(
                        f"{requirement.requirement_id}: expected gpu_test.{key} "
                        f"== {expected!r}, got {gpu_test.get(key)!r}"
                    )
            numeric_expectations = {
                "clean_corrupt_gap": ("min", 1.0),
                "name_mover_mean_damage": ("min", 0.15),
                "random_control_abs_mean_damage": ("max", 0.02),
                "known_to_random_control_ratio": ("min", 10.0),
            }
            for key, (direction, threshold) in numeric_expectations.items():
                actual = gpu_test.get(key)
                if not isinstance(actual, float | int):
                    blockers.append(
                        f"{requirement.requirement_id}: missing numeric gpu_test.{key}"
                    )
                elif direction == "min" and float(actual) < threshold:
                    blockers.append(
                        f"{requirement.requirement_id}: expected gpu_test.{key} "
                        f">= {threshold}, got {actual!r}"
                    )
                elif direction == "max" and float(actual) > threshold:
                    blockers.append(
                        f"{requirement.requirement_id}: expected gpu_test.{key} "
                        f"<= {threshold}, got {actual!r}"
                    )
            if "ioi" not in claim_text or "path-patching" not in claim_text:
                blockers.append(
                    f"{requirement.requirement_id}: modern report does not contain "
                    "IOI/path-patching verification evidence"
                )
    return blockers


def roadmap_final_completeness_blockers() -> list[str]:
    blockers: list[str] = []
    registry_structure = roadmap_requirement_registry_blockers()
    blockers.extend(registry_structure)
    if not registry_structure:
        blockers.extend(
            f"roadmap status: {blocker}"
            for blocker in roadmap_requirement_status_blockers()
        )
    blockers.extend(
        f"original preservation: {blocker}" for blocker in original_preservation_blockers()
    )
    blockers.extend(f"course surface: {blocker}" for blocker in course_surface_blockers())
    blockers.extend(
        f"report evidence contract: {blocker}" for blocker in report_evidence_blockers()
    )
    blockers.extend(f"ARENA style depth: {blocker}" for blocker in style_depth_blockers())
    for requirement in REPORT_REQUIREMENTS:
        blockers.extend(report_requirement_blockers(requirement))
    for requirement in LEGACY_REQUIREMENTS:
        blockers.extend(legacy_requirement_blockers(requirement))
    return blockers


def main() -> None:
    blockers = roadmap_final_completeness_blockers()
    try:
        registry = load_roadmap_requirement_registry()
    except (OSError, yaml.YAMLError):
        registry = {}
    requirements = registry.get("requirements", [])
    status_counts = Counter(
        requirement.get("status")
        for requirement in requirements
        if isinstance(requirement, dict)
    )
    print(f"roadmap_requirements_checked={len(requirements)}")
    for status in sorted(ROADMAP_STATUSES):
        print(f"roadmap_status_{status.lower()}={status_counts.get(status, 0)}")
    print(f"report_requirements_checked={len(REPORT_REQUIREMENTS)}")
    print(f"legacy_requirements_checked={len(LEGACY_REQUIREMENTS)}")
    print(f"roadmap_final_completeness_blockers={len(blockers)}")
    if blockers:
        print("ROADMAP_FINAL_COMPLETENESS=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("ROADMAP_FINAL_COMPLETENESS=PASS")


if __name__ == "__main__":
    main()
