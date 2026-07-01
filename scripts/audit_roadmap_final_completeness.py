"""Audit the roadmap's final completeness matrix against current evidence.

This gate is intentionally about roadmap coverage, not about whether a single
`gt_tier` field can describe a hybrid notebook. Several extension notebooks
contain both exact/toy contracts and real-model preflights, so this audit checks
named report metrics, controls, and legacy notebook evidence requirement by
requirement.
"""

from __future__ import annotations

import json
import sys
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
        "Representation geometry has PCA/SVD and random-label controls.",
        "chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/verification_report.json",
        "chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/artifacts.lock.yml",
        {
            "gt_tier": "GT-0",
            "metrics.gpu_test.predicts_heldout_labels": True,
            "metrics.gpu_test.pythia_weekday_permuted_label_accuracy": 0.0,
            "metrics.gpu_test.pythia_visualization_seed_count": {"min": 5},
            "metrics.gpu_test.pythia_visualization_setting_count": {"min": 3},
            "metrics.gpu_test.pythia_weekday_visualization_passed": True,
            "metrics.gpu_test.pythia_month_visualization_passed": True,
            "metrics.gpu_test.pythia_weekday_umap_min_trustworthiness": {"min": 0.9},
            "metrics.gpu_test.pythia_month_umap_min_trustworthiness": {"min": 0.9},
        },
        (
            "permuted_label_control",
            "white_noise_geometry_control",
            "five_seed_umap_visualization_sweep",
            "three_reducer_hyperparameter_settings",
            "visualization_random_token_negative_control",
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
            "metrics.gpu_test.diffusiongemma_config_supported": True,
            "metrics.gpu_test.diffusiongemma_processor_supported": True,
            "metrics.gpu_test.diffusiongemma_model_class_supported": True,
            "metrics.gpu_test.diffusiongemma_generation_ready": True,
        },
        ("forward_noising_extreme_timestep_contract", "shuffled_label_negative_control"),
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
        "Refusal-direction replication uses a public refusal/compliance dataset with aggregate behavioral evidence.",
        "chapter9_alignment_interpretability/exercises/part1_refusal_directions_safe_steering/verification_report.json",
        "chapter9_alignment_interpretability/exercises/part1_refusal_directions_safe_steering/artifacts.lock.yml",
        {
            "gt_tier": "GT-2",
            "metrics.gpu_test.gt2_refusal_direction_preflight_passed": True,
            "metrics.gpu_test.gt2_refusal_direction_gt2_ready": True,
            "metrics.gpu_test.gt2_refusal_direction_dataset_id": "josephmayo/refusal-compliance-pairs",
            "metrics.gpu_test.gt2_refusal_direction_raw_prompt_text_saved": False,
            "metrics.gpu_test.gt2_refusal_direction_completion_text_saved": False,
            "metrics.gpu_test.gt2_refusal_direction_heldout_accuracy": {"min": 0.9},
            "metrics.gpu_test.gt2_refusal_direction_label_shuffle_fails": True,
            "metrics.gpu_test.gt2_refusal_direction_random_direction_fails": True,
            "metrics.gpu_test.gt2_refusal_direction_pc1_variance_fraction": {"min": 0.2},
            "metrics.gpu_test.gt2_refusal_direction_baseline_behavioral_accuracy": {"min": 0.75},
            "metrics.gpu_test.gt2_refusal_direction_projection_delta": {"max": -0.5},
            "metrics.gpu_test.gt2_refusal_direction_target_beats_random_projection": True,
            "metrics.gpu_test.gt2_refusal_direction_paper_equivalent_behavioral_completion_evidence": True,
        },
        (
            "public_refusal_compliance_pairs_dataset",
            "gt2_mean_difference_refusal_direction",
            "gt2_layer_sweep_control",
            "gt2_position_sweep_control",
            "gt2_pca_svd_pc1_control",
            "gt2_label_shuffle_control",
            "gt2_random_direction_control",
            "gt2_behavioral_addition_projection_aggregate_completion_effect",
            "gt2_no_raw_prompt_or_completion_text_saved",
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


BLOCKED_REQUIREMENTS: tuple[tuple[str, str, str], ...] = ()


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
    if evidence.get("category") != "cuda_section_metric" or evidence.get("uses_cuda") is not True:
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
    for requirement_id, group, reason in BLOCKED_REQUIREMENTS:
        blockers.append(f"{requirement_id} ({group}): {reason}")
    return blockers


def main() -> None:
    blockers = roadmap_final_completeness_blockers()
    print(f"report_requirements_checked={len(REPORT_REQUIREMENTS)}")
    print(f"legacy_requirements_checked={len(LEGACY_REQUIREMENTS)}")
    print(f"blocked_requirements_declared={len(BLOCKED_REQUIREMENTS)}")
    print(f"roadmap_final_completeness_blockers={len(blockers)}")
    if blockers:
        print("ROADMAP_FINAL_COMPLETENESS=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("ROADMAP_FINAL_COMPLETENESS=PASS")


if __name__ == "__main__":
    main()
