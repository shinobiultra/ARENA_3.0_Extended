"""Generate verification assets required by the extension roadmap.

The amended roadmap requires every new notebook to carry an artifact lock,
expected outputs, a verification report schema, and a small README. This script
derives the notebook list from infrastructure/core/config.yaml so the metadata
stays aligned with the course navigation. Existing artifact locks are treated
as authoritative by default; use --overwrite-contracts only when you
deliberately want to replace upgraded real-evidence contracts with the
generated scaffold.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "infrastructure/core/config.yaml"
REGISTRY_COLUMNS = [
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
]
METHOD_COLUMNS = [
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
]
LADDER_REGISTRY_COLUMNS = [
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
]

NOTEBOOK_CONTRACT_EXPECTED_METRICS: dict[str, dict[str, Any]] = {
    "0.6": {
        "preflight_passed": True,
        "leaked_feature_accuracy": 1.0,
        "shifted_no_leak_accuracy": 0.0,
        "leakage_gap_min": 0.5,
        "cherry_pick_inflation_min": 3.0,
        "probe_train_accuracy": 1.0,
        "probe_heldout_accuracy": 0.5,
        "probe_overfit_gap_min": 0.35,
        "random_direction_control_rejects_claim": True,
        "all_bogus_results_flagged": True,
        "peak_vram_gb_max": 1.0,
    },
    "1.6": {
        "cuda_available": True,
        "python_major_minor": "3.14",
        "torch_version": "2.12.1+cu132",
        "torchvision_version": "0.27.1+cu132",
        "cuda_version": "13.2",
        "bf16_supported": True,
        "gpu_total_memory_gb_min": 20.0,
        "estimated_total_gb_max": 4.0,
        "fits_budget": True,
        "gpu_tensor_test_passed": True,
        "gpu_matmul_shape": [1024, 1024],
        "gpu_matmul_dtype": "torch.bfloat16",
        "gpu_matmul_finite": True,
        "gpu_matmul_mean_abs_min": 1.0,
        "gpu_matmul_diag_mean_min": 100.0,
        "uv_pip_check_passed": True,
        "uv_pip_check_returncode": 0,
        "peak_vram_gb_max": 1.0,
        "warnings": [],
    },
    "5.1": {
        "preflight_passed": True,
        "cache_max_abs_diff_max": 1e-5,
        "cache_passed": True,
        "fits_budget": True,
        "rope_norm_error_max": 1e-6,
        "reference_model_family": "transformers.GemmaForCausalLM",
        "reference_transformers_tiny_config": True,
        "reference_weight_key_count": 21,
        "reference_loaded_key_count": 21,
        "reference_logits_max_abs_diff_max": 5e-4,
        "reference_logits_mse_max": 1e-7,
        "reference_logits_topk_agreement": 1.0,
        "reference_cache_max_abs_diff_max": 1e-5,
        "reference_cache_passed": True,
        "reference_parity_passed": True,
        "peak_vram_gb_max": 1.0,
    },
    "5.3": {
        "cache_max_abs_diff_max": 1e-5,
        "cache_passed": True,
        "scan_max_abs_diff_max": 1e-6,
        "scan_passed": True,
    },
    "5.5": {
        "preflight_passed": True,
        "model_family": "tiny_transformer_discrete_diffusion_lm",
        "dataset": "copy_pair_conditional_suffix_grammar_v1",
        "train_example_count": 80,
        "heldout_example_count": 20,
        "vocab_size": 11,
        "mask_token_id": 10,
        "sequence_length": 6,
        "diffusion_steps": 6,
        "training_steps": 1200,
        "final_train_denoising_loss_max": 0.05,
        "denoising_loss_max": 0.05,
        "heldout_masked_accuracy_min": 0.95,
        "shuffled_label_accuracy_max": 0.25,
        "shuffled_control_fails": True,
        "sampler_suffix_token_accuracy_min": 0.95,
        "sampler_exact_match_min": 0.95,
        "activation_trajectory_shape_ok": True,
        "entropy_max_max": 1.5,
        "diffusiongemma_generation_ready": True,
        "diffusiongemma_released_checkpoint_generation_proven": True,
        "diffusiongemma_bf16_24gb_direct_loading_deferred": True,
        "diffusiongemma_nvfp4_isolated_vllm_generation_ready": True,
        "diffusiongemma_external_vllm_generation_ready": True,
        "diffusiongemma_external_vllm_runtime_isolated": True,
        "diffusiongemma_external_vllm_model_matches_nvfp4_revision": True,
        "diffusiongemma_external_vllm_output_nonempty": True,
        "diffusiongemma_external_vllm_output_mentions_negative_controls": True,
        "diffusiongemma_external_vllm_used_chat_template": True,
        "diffusiongemma_external_vllm_torch_version": "2.11.0+cu130",
        "diffusiongemma_external_vllm_torch_cuda_version": "13.0",
        "diffusiongemma_external_vllm_vllm_version": "0.24.0",
        "diffusiongemma_external_vllm_cuda_available": True,
        "diffusiongemma_vllm_probe_output_nonempty": True,
        "diffusiongemma_vllm_probe_torch_version": "2.11.0+cu130",
        "diffusiongemma_vllm_probe_torch_cuda_version": "13.0",
        "diffusiongemma_vllm_probe_vllm_version": "0.24.0",
        "peak_vram_gb_max": 1.0,
    },
    "6.1": {
        "preflight_passed": True,
        "model_id": "EleutherAI/pythia-70m-deduped",
        "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
        "generation_used": False,
        "train_prompt_count": 64,
        "heldout_prompt_count": 16,
        "d_model": 512,
        "sae_width": 256,
        "sae_k": 16,
        "training_steps": 800,
        "heldout_reconstruction_mse_max": 1.2,
        "zero_baseline_mse_min": 1.3,
        "reconstruction_improvement_vs_zero_min": 0.15,
        "random_decoder_mse_ratio_min": 1.2,
        "random_decoder_control_passed": True,
        "heldout_l0": 16.0,
        "heldout_feature_density_mean": 0.0625,
        "heldout_dead_feature_fraction_max": 0.7,
        "density_nondegenerate": True,
        "best_feature_auc_min": 0.95,
        "safe_logit_delta_min": 0.5,
        "passes_decoder_steering_control": True,
        "peak_vram_gb_max": 1.0,
    },
    "6.3": {
        "preflight_passed": True,
        "model_name": "gelu-1l",
        "hf_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "activation_count": 104,
        "d_model": 512,
        "mlp_width": 2048,
        "oracle_mlp_out_max_abs_error_max": 1e-5,
        "oracle_logits_max_abs_error_max": 5e-5,
        "oracle_replacement_kl_max": 1e-6,
        "oracle_preserves_logit_diff": True,
        "trained_transcoder_width": 512,
        "trained_transcoder_steps": 800,
        "trained_transcoder_heldout_mse_ratio_max": 0.5,
        "trained_replacement_top1_agreement_min": 0.75,
        "graph_feature_count": 64,
        "graph_preserves_logit_diff": True,
        "graph_passes_damage_control": True,
        "graph_reproducible": True,
        "graph_topk_damage_min": 0.1,
        "graph_random_damage_max": 0.01,
        "preserves_logit_diff": True,
        "replacement_kl_max": 1e-6,
        "peak_vram_gb_max": 1.0,
    },
    "6.4": {
        "preflight_passed": True,
        "model_a_name": "gelu-1l",
        "model_b_name": "solu-1l",
        "model_a_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "model_b_revision": "a4ce32db5e35f13e5f09333888bd2d42660f77ce",
        "activation_shape": [16, 512],
        "ablation_passes_control": True,
        "behavior_delta_auc": 1.0,
        "model_a_mse_max": 1e-8,
        "model_b_mse_max": 1e-8,
        "shared_reconstructs_both": True,
        "top_variance_fraction_min": 0.25,
        "delta_reduction_min": 1.0,
        "random_reduction_max": 0.1,
        "floor_top_delta_abs_mean_min": 0.1,
        "peak_vram_gb_max": 1.0,
    },
    "7.1": {
        "preflight_passed": True,
        "model_name": "gelu-1l",
        "hf_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
        "bnb_cuda_override": "130",
        "train_prompt_count": 16,
        "heldout_prompt_count": 6,
        "heldout_position_count": 40,
        "logit_lens_accuracy_max": 0.2,
        "tuned_lens_accuracy_min": 0.35,
        "tuned_lens_improvement_min": 0.3,
        "final_decode_max_abs_error_max": 1e-4,
        "patchscope_pair_count": 6,
        "patchscope_hook_name": "blocks.0.hook_resid_post",
        "patchscope_target_prompt_count": 6,
        "patchscope_accuracy": 1.0,
        "text_only_accuracy": 0.0,
        "patchscope_beats_text_only": True,
        "patchscope_min_patched_target_margin_min": 0.01,
        "patchscope_max_text_only_target_margin_max": 0.0,
        "counterfactual_changed": True,
        "counterfactual_original_token": " floor",
        "counterfactual_patched_token": " top",
        "random_max_confidence_max": 0.1,
        "random_passes_low_confidence": True,
        "attention_lens_logits_shape": [1, 6, 48262],
        "attention_lens_finite": True,
        "attention_lens_prompt_final_token": " floor",
        "peak_vram_gb_max": 1.0,
        "tuned_lens_improves": True,
    },
    "7.2": {
        "preflight_passed": True,
        "model_name": "gelu-1l",
        "hf_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
        "bnb_cuda_override": "130",
        "positive_token": " floor",
        "negative_token": " top",
        "train_example_count": 8,
        "heldout_example_count": 8,
        "score_separation_min": 5.0,
        "intervention_delta_min": 0.5,
        "matches_intervention_prediction": True,
        "passes_baseline": True,
        "prediction_accuracy": 1.0,
        "baseline_accuracy": 0.5,
        "contrastive_accuracy": 1.0,
        "survives_contrastive": True,
        "num_counterexamples": 0,
        "brevity_shorter_than_examples": True,
        "train_heldout_overlap_count": 0,
        "learned_terms_from_train": True,
        "target_beats_random_intervention": True,
        "peak_vram_gb_max": 1.0,
    },
    "7.3": {
        "preflight_passed": True,
        "model_name": "gelu-1l",
        "hf_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
        "bnb_cuda_override": "130",
        "positive_token": " floor",
        "negative_token": " top",
        "train_example_count": 8,
        "eval_example_count": 8,
        "activation_shape": [16, 512],
        "question_count": 2,
        "question_conditioned_row_count": 16,
        "question_conditioned_train_loss_max": 0.01,
        "question_id_changes_predictions": 1.0,
        "train_accuracy": 1.0,
        "oracle_accuracy": 1.0,
        "text_only_accuracy": 0.5,
        "linear_probe_accuracy": 0.5,
        "mlp_probe_accuracy": 0.5,
        "sae_classifier_accuracy": 0.5,
        "activation_only_probe_accuracy_max_max": 0.75,
        "beats_text_only": True,
        "beats_or_matches_probe": True,
        "heldout_template_accuracy": 1.0,
        "new_name_accuracy": 1.0,
        "long_context_accuracy": 1.0,
        "adversarial_accuracy": 1.0,
        "passes_ood": True,
        "random_abstention_rate": 1.0,
        "random_mean_confidence_max": 0.4,
        "random_graceful_failure": True,
        "patching_changed_answer": True,
        "original_answer": 1,
        "patched_answer": 0,
        "peak_vram_gb_max": 1.0,
    },
    "7.4": {
        "preflight_passed": True,
        "model_name": "gelu-1l",
        "hf_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
        "bnb_cuda_override": "130",
        "positive_token": " floor",
        "negative_token": " top",
        "phrase_count": 12,
        "phrase_vocabulary_size": 12,
        "text_bottleneck": "discrete_natural_language_phrase_bottleneck",
        "live_trainable_nla": True,
        "report_replay": False,
        "trainable_encoder_type": "linear_residual_to_phrase_id",
        "trainable_decoder_type": "phrase_id_embedding_table_to_residual",
        "trainable_encoder_parameter_count": 6156,
        "trainable_decoder_parameter_count": 6144,
        "trainable_encoder_train_accuracy": 1.0,
        "trainable_eval_phrase_accuracy_min": 0.75,
        "trainable_encoder_final_loss_max": 0.01,
        "trainable_decoder_final_mse_max": 0.01,
        "trainable_reconstruction_mse_max": 0.15,
        "trainable_blank_text_mse_min": 0.2,
        "trainable_beats_blank_text": True,
        "trainable_phrase_count": 12,
        "trainable_training_steps": 300,
        "trainable_seed": 0,
        "train_example_count": 12,
        "eval_example_count": 8,
        "activation_shape": [8, 512],
        "explanation_count": 8,
        "numeric_literal_count": 0,
        "activation_mse_max": 0.15,
        "text_only_mse_min": 0.21,
        "prompt_label_baseline_mse_min": 0.16,
        "mean_cosine_similarity_min": 0.93,
        "beats_text_only": True,
        "beats_prompt_label_baseline": True,
        "probe_logit_mean_abs_error_max": 2.0,
        "preserves_latents": True,
        "preserves_target_logit_diff": True,
        "original_probe_accuracy": 1.0,
        "reconstructed_probe_accuracy": 1.0,
        "prediction_agreement": 1.0,
        "text_only_prediction_accuracy": 0.5,
        "nla_prediction_accuracy": 1.0,
        "passes_ood": True,
        "generated_word_count": 32,
        "original_word_count": 64,
        "compression_ratio_max": 0.7,
        "shorter_than_original": True,
        "counterfactual_activation_delta_min": 10.0,
        "counterfactual_explanation_changed": True,
        "shuffled_reconstruction_mse_min": 0.25,
        "shuffled_control_worse": True,
        "blank_text_mse_min": 0.2,
        "blank_text_control_worse": True,
        "peak_vram_gb_max": 1.0,
    },
    "7.5": {
        "preflight_passed": True,
        "model_name": "gelu-1l",
        "hf_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
        "bnb_cuda_override": "130",
        "positive_token": " floor",
        "negative_token": " top",
        "train_example_count": 12,
        "eval_example_count": 8,
        "activation_shape": [8, 512],
        "concept_shape": [8, 8],
        "row_concept_shape": [32, 8],
        "conditioned_concept_shape": [32, 32],
        "question_count": 4,
        "pcd_row_count": 32,
        "train_pcd_row_count": 48,
        "concept_direction_count": 8,
        "concept_top_k": 4,
        "concept_mean_l0": 4.0,
        "beats_best_baseline": True,
        "beats_probe": True,
        "concept_density": 0.5,
        "passes_sparsity": True,
        "pcd_accuracy": 1.0,
        "probe_accuracy": 0.5,
        "sae_classifier_accuracy": 0.5,
        "non_interaction_baseline_accuracy": 0.5,
        "best_baseline_accuracy": 0.5,
        "pcd_decoder_train_accuracy": 1.0,
        "pcd_decoder_train_loss_max": 0.001,
        "pcd_decoder_training_steps": 700,
        "probe_train_accuracy": 0.5,
        "sae_classifier_train_accuracy": 0.5,
        "non_interaction_train_accuracy": 0.5,
        "question_shuffle_accuracy_max": 0.25,
        "pcd_seed_count": 3,
        "pcd_seed_min_accuracy": 1.0,
        "passes_ood": True,
        "random_removal_does_less": True,
        "random_removal_changed": False,
        "top_removal_changed": True,
        "top_removal_delta_min": 4.0,
        "random_removal_delta_max": 1.0,
        "random_removed_concept_active": True,
        "selected_concept_ids": [0, 1],
        "selected_concept_names": ["surface_direction", "motion_direction"],
        "names_expected_cluster": True,
        "mean_pairwise_jaccard": 1.0,
        "stable": True,
        "peak_vram_gb_max": 1.0,
    },
    "8.1": {
        "preflight_passed": True,
        "model_name": "gelu-1l",
        "hf_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
        "bnb_cuda_override": "130",
        "hook_name": "blocks.0.hook_resid_post",
        "sequence_length": 6,
        "target_position": 5,
        "target_token": " floor",
        "distractor_token": " top",
        "clean_corrupt_gap_min": 6.0,
        "target_recovered_fraction_min": 0.99,
        "wrong_position_control_fraction_max": 1e-4,
        "best_position": 5,
        "best_score_min": 0.99,
        "localizes_final_position": True,
        "top_beats_wrong_position_control": True,
        "max_abs_final_patch_logit_error_max": 1e-5,
        "finite_logits": True,
        "peak_vram_gb_max": 1.0,
    },
    "8.2": {
        "preflight_passed": True,
        "model_name": "gelu-1l",
        "hf_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
        "bnb_cuda_override": "130",
        "hook_name": "blocks.0.hook_resid_post",
        "sequence_length": 6,
        "target_position": 5,
        "target_token": " floor",
        "distractor_token": " top",
        "clean_corrupt_gap_min": 6.0,
        "exact_best_position": 5,
        "attribution_best_position": 5,
        "ig_best_position": 5,
        "exact_final_recovery_min": 0.99,
        "attribution_final_recovery_min": 0.9,
        "ig_final_recovery_min": 0.95,
        "exact_attribution_correlation_min": 0.99,
        "exact_attribution_top1_overlap": 1.0,
        "exact_ig_correlation_min": 0.99,
        "exact_ig_top1_overlap": 1.0,
        "eap_edge_score_shape": [6, 6],
        "eap_top_edge_upstream_position": 5,
        "eap_top_edge_downstream_position": 5,
        "eap_top_edge_score_abs_min": 6.0,
        "nonfinal_gradient_norm_max": 0.0,
        "final_gradient_norm_min": 1.0,
        "integrated_gradient_steps": 5,
        "peak_vram_gb_max": 1.0,
    },
    "8.3": {
        "preflight_passed": True,
        "model_name": "gelu-1l",
        "hf_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
        "bnb_cuda_override": "130",
        "hook_name": "blocks.0.hook_resid_post",
        "sequence_length": 6,
        "target_position": 5,
        "target_token": " floor",
        "distractor_token": " top",
        "best_position": 5,
        "best_score_min": 0.99,
        "num_kept_edges": 1,
        "kept_edges": ["position_5"],
        "clean_corrupt_gap_min": 6.0,
        "passes_faithfulness": True,
        "passes_minimality": True,
        "passes_completeness": True,
        "passes_ood": True,
        "preserved_fraction_min": 0.99,
        "minimality_metric_damage_min": 1.0,
        "omitted_node_gain_max": 1e-5,
        "random_baseline_margin_min": 1.0,
        "circuit_beats_random": True,
        "template_count": 4,
        "peak_vram_gb_max": 1.0,
    },
    "8.4": {
        "preflight_passed": True,
        "model_name": "gelu-1l",
        "hf_revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
        "bnb_cuda_override": "130",
        "hook_name": "blocks.0.hook_resid_post",
        "sequence_length": 6,
        "target_position": 5,
        "target_token": " floor",
        "distractor_token": " top",
        "graph_node_count": 6,
        "alternative_baseline_fails": True,
        "clean_corrupt_gap_min": 6.0,
        "explained_fraction_min": 0.99,
        "explains_target_metric": True,
        "num_graph_edges": 1,
        "graph_edge_source": "position_5",
        "graph_edge_target": "position_5",
        "graph_edge_score_min": 6.0,
        "top_edge_score_abs_min": 6.0,
        "path_metric_drop_min": 1.0,
        "top_path_survives_test": True,
        "alternative_baseline_margin_min": 1.0,
        "counterfactual_predicted_direction": "decrease",
        "counterfactual_observed_delta_max": -1.0,
        "predicts_counterfactual": True,
        "peak_vram_gb_max": 1.0,
    },
    "9.2": {
        "preflight_passed": True,
        "model_name": "EleutherAI/pythia-70m-deduped",
        "hf_revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
        "hidden_layer": -1,
        "answer_token_ids": [329, 378],
        "answer_tokens": [" A", " B"],
        "train_prompt_count": 24,
        "heldout_prompt_count": 8,
        "hidden_state_shape": [8, 512],
        "feature_detector_improves": True,
        "feature_detector_accuracy": 1.0,
        "baseline_detector_accuracy": 0.5,
        "detector_recall": 1.0,
        "text_only_recall": 0.0,
        "text_only_misses_cases": True,
        "hidden_answer_accuracy": 1.0,
        "final_answer_agreement": 0.5,
        "model_answer_accuracy": 0.625,
        "label_shuffled_probe_accuracy": 0.0,
        "patching_changed_output": True,
        "original_answer": 0,
        "patched_answer": 1,
        "predicts_hidden_answer": True,
        "unfaithful_case_count": 4,
        "peak_vram_gb_max": 1.0,
    },
    "9.3": {
        "preflight_passed": True,
        "model_name": "EleutherAI/pythia-70m-deduped",
        "hf_revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
        "hidden_layer": -1,
        "behavior_proxy_token_ids": [9371, 20372],
        "behavior_proxy_tokens": [" helpful", " unsafe"],
        "train_prompt_count": 36,
        "heldout_prompt_count": 24,
        "drift_kind_count": 5,
        "train_context_count": 6,
        "heldout_context_count": 4,
        "hidden_state_shape": [24, 512],
        "detector_accuracy": 1.0,
        "drift_alignment_correlation_min": 0.7,
        "aligns_with_behavior_delta": True,
        "label_shuffled_detector_accuracy_max": 0.75,
        "random_direction_accuracy_max": 0.55,
        "black_box_behavior_proxy_accuracy": 1.0,
        "drift_score_margin_min": 1.0,
        "mitigation_drift_delta_reduction_min": 1.0,
        "mitigation_neutral_delta_shift_max": 0.1,
        "mitigation_passes": True,
        "predicts_heldout_drift": True,
        "generation_used": False,
        "peak_vram_gb_max": 1.0,
    },
    "9.4": {
        "preflight_passed": True,
        "model_name": "EleutherAI/pythia-70m-deduped",
        "hf_revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
        "hidden_layer": -1,
        "black_box_token_ids": [1509, 1891],
        "black_box_tokens": [" pass", " fail"],
        "train_prompt_count": 36,
        "heldout_prompt_count": 24,
        "failure_kind_count": 5,
        "train_context_count": 6,
        "heldout_context_count": 4,
        "hidden_state_shape": [24, 512],
        "calibrated": True,
        "catches_black_box_miss": True,
        "explanations_validated": True,
        "monitor_auroc": 1.0,
        "white_box_accuracy": 1.0,
        "black_box_proxy_accuracy": 0.875,
        "black_box_missed_failure_count": 3,
        "label_shuffled_monitor_auroc_max": 0.85,
        "random_direction_monitor_auroc_max": 0.85,
        "monitor_score_margin_min": 1.0,
        "false_positive_count": 0,
        "false_positives_documented": True,
        "heldout_explanation_accuracy": 1.0,
        "generation_used": False,
        "peak_vram_gb_max": 1.0,
    },
    "10.1": {
        "preflight_passed": True,
        "live_training_executed": True,
        "baseline_suite_complete": True,
        "causal_validation_complete": True,
        "ready": True,
        "reproducible": True,
        "seed_count": 3,
        "model_family": "question_conditioned_mlp_activation_oracle",
        "dataset": "balanced_latent_bits_with_heldout_template_nuisance_v1",
        "train_example_count": 576,
        "iid_example_count": 576,
        "heldout_template_example_count": 384,
        "oracle_accuracy_mean_min": 0.9,
        "oracle_compositional_accuracy_mean_min": 0.9,
        "text_only_accuracy_mean": 0.5,
        "linear_probe_compositional_accuracy_mean_max": 0.75,
        "heldout_template_accuracy_mean_min": 0.9,
        "oracle_beats_text_only": True,
        "oracle_beats_linear_probe_bank": True,
        "compositional_oracle_beats_linear_probe": True,
        "ood_passed": True,
        "ablation_drop_mean_min": 0.2,
        "counterfactual_patch_change_rate_mean_min": 0.7,
        "counterfactual_patch_target_accuracy_mean_min": 0.9,
        "random_patch_change_rate_mean_max": 0.15,
        "random_activation_accuracy_mean_max": 0.65,
        "random_activation_control_passed": True,
        "label_shuffle_accuracy_mean_max": 0.65,
        "label_shuffle_control_passed": True,
        "causal_controls_passed": True,
        "metrics_by_seed_file_valid": True,
        "peak_vram_gb_max": 1.0,
    },
    "16.1": {
        "preflight_passed": True,
        "model_family": "cuda_trained_neural_coalition_game_mlp",
        "num_players": 4,
        "coalition_count": 16,
        "training_example_count": 16,
        "training_steps": 1200,
        "fit_mse_max": 1e-8,
        "fit_max_abs_error_max": 1e-4,
        "neural_shapley_max_abs_error_max": 1e-4,
        "efficiency_error_max": 1e-8,
        "satisfies_efficiency": True,
        "shuffled_control_error_min": 1.0,
        "shuffled_control_cosine_max": 0.25,
        "shuffled_control_rejected": True,
        "peak_vram_gb_max": 1.0,
    },
    "16.2": {
        "preflight_passed": True,
        "model_family": "cuda_trained_neural_coalition_game_mlp",
        "num_players": 4,
        "coalition_count": 16,
        "training_example_count": 16,
        "training_steps": 1200,
        "fit_mse_max": 1e-8,
        "fit_max_abs_error_max": 1e-4,
        "kernel_approximates_exact": True,
        "kernel_max_abs_error_max": 1e-8,
        "kernel_vs_true_max_abs_error_max": 1e-4,
        "partition_max_abs_error_max": 1e-8,
        "partition_recovers_exact": True,
        "aligned_partition_max_abs_error_min": 0.1,
        "aligned_partition_recovers_exact": False,
        "mismatched_partition_max_abs_error_min": 0.5,
        "mismatched_partition_recovers_exact": False,
        "mismatched_minus_aligned_grouping_gap_min": 0.5,
        "shuffled_control_kernel_vs_true_error_min": 1.0,
        "shuffled_control_rejected": True,
        "peak_vram_gb_max": 1.0,
    },
    "16.3": {
        "preflight_passed": True,
        "model_family": "cuda_trained_neural_coalition_game_mlp",
        "num_players": 4,
        "coalition_count": 16,
        "training_example_count": 16,
        "training_steps": 1200,
        "fit_mse_max": 1e-8,
        "fit_max_abs_error_max": 1e-4,
        "interaction_max_abs_error_max": 1e-4,
        "positive_interaction_abs_error_max": 1e-4,
        "negative_interaction_abs_error_max": 1e-4,
        "max_spurious_interaction_max": 1e-4,
        "interaction_signs_recovered": True,
        "shapiq_available": True,
        "shapiq_matches": True,
        "shapiq_max_abs_error_max": 1e-5,
        "shuffled_control_interaction_error_min": 1.0,
        "shuffled_control_rejected": True,
        "peak_vram_gb_max": 1.0,
    },
    "16.4": {
        "preflight_passed": True,
        "model_family": "cuda_trained_tiny_token_scorer_mlp",
        "token_count": 4,
        "coalition_count": 16,
        "training_example_count": 16,
        "training_steps": 1200,
        "fit_mse_max": 1e-10,
        "fit_max_abs_error_max": 1e-5,
        "exact_shapley_max_abs_error_max": 1e-5,
        "sampled_max_abs_error_max": 0.1,
        "sampled_rank_matches": True,
        "top_token": "Paris",
        "sampled_top_token": "Paris",
        "baseline_efficiency_error_max": 1e-8,
        "satisfies_efficiency": True,
        "shuffled_control_error_min": 1.0,
        "shuffled_control_rejected": True,
        "peak_vram_gb_max": 1.0,
    },
    "16.5": {
        "preflight_passed": True,
        "model_id": "openai/clip-vit-base-patch32",
        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "claim_scope": "pinned_real_clip_rendered_vlm_shap_preflight",
        "modality_synergy_min": 2.0,
        "modality_satisfies_efficiency": True,
        "object_margin_min": 1.0,
        "region_satisfies_efficiency": True,
        "target_distractor_margin_min": 2.0,
        "peak_vram_gb_max": 2.0,
    },
    "16.6": {
        "preflight_passed": True,
        "additive_model_family": "cuda_trained_linear_additive_model",
        "additive_training_example_count": 16,
        "additive_training_steps": 1000,
        "additive_fit_mse_max": 1e-10,
        "additive_fit_max_abs_error_max": 1e-5,
        "additive_max_abs_error_max": 1e-5,
        "additive_agrees_with_shapley": True,
        "interaction_model_family": "cuda_trained_neural_coalition_game_mlp",
        "interaction_training_example_count": 16,
        "interaction_training_steps": 1200,
        "interaction_fit_mse_max": 1e-8,
        "interaction_fit_max_abs_error_max": 1e-4,
        "interaction_max_abs_error_min": 1.0,
        "interaction_agrees_with_shapley": False,
        "interaction_top_feature_agrees": True,
        "interaction_abs_overcount_min": 2.0,
        "peak_vram_gb_max": 1.0,
    },
    "16.7": {
        "preflight_passed": True,
        "model_family": "cuda_one_step_linear_regression_data_shapley",
        "training_example_count": 4,
        "coalition_count": 16,
        "monte_carlo_samples": 512,
        "learning_rate": 0.5,
        "sampled_max_abs_error_max": 0.08,
        "sampled_approximates_exact": True,
        "pearson_correlation_min": 0.99,
        "harmful_index": 3,
        "identifies_harmful": True,
        "identifies_helpful": True,
        "harmful_value_max": 0.0,
        "harmful_removal_delta_min": 0.1,
        "actual_full_batch_one_step_utility": 0.75,
        "coalition_full_utility": 0.75,
        "peak_vram_gb_max": 1.0,
    },
    "16.8": {
        "preflight_passed": True,
        "model_family": "cuda_trained_neural_coalition_game_mlp",
        "num_players": 4,
        "coalition_count": 16,
        "training_example_count": 16,
        "training_steps": 1200,
        "fit_mse_max": 1e-8,
        "fit_max_abs_error_max": 1e-4,
        "spearman_correlation_min": 0.99,
        "topk_overlap": 1.0,
        "agrees_with_mechanistic": True,
        "interaction_max_abs_error_max": 1e-4,
        "top_interaction_pair": [0, 2],
        "second_interaction_pair": [1, 3],
        "shuffled_control_spearman_max": 0.0,
        "shuffled_control_topk_overlap": 0.0,
        "shuffled_control_rejected": True,
        "agreement_artifacts_written": True,
        "agreement_artifact_count": 5,
        "agreement_matrix_rows_min": 7,
        "agreement_case_count_min": 1,
        "disagreement_case_count_min": 1,
        "deletion_curve_points": 5,
        "insertion_curve_points": 5,
        "topk_heatmap_rows": 4,
        "topk_heatmap_cols": 4,
        "peak_vram_gb_max": 1.0,
    },
    "17.1": {
        "preflight_passed": True,
        "model_family": "tiny_modular_addition_mlp",
        "modulus": 13,
        "table_example_count": 169,
        "checkpoint_count": 26,
        "first_crossing_step": 30,
        "stable_from_step": 30,
        "final_accuracy": 1.0,
        "phase_transition_step": 30,
        "phase_transition_jump_min": 0.2,
        "random_control_peak_accuracy_max": 0.2,
        "random_control_passed": True,
        "real_checkpoints_reloaded": True,
        "loss_drop_min": 2.0,
        "peak_vram_gb_max": 0.2,
    },
}

REGISTRY_ROWS: list[dict[str, str]] = [
    {
        "name": "ARENA 3.0 source course",
        "type": "course_repo",
        "provider": "GitHub",
        "repo_or_source_id": "callummcdougall/ARENA_3.0",
        "license": "MIT",
        "gated": "false",
        "revision": "f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0",
        "local_status": "REQUIRED",
        "max_vram_gb": "0",
        "used_in_notebooks": "1.6",
        "gt_tier": "GT-1",
        "notes": "Original course must remain behaviorally preserved.",
    },
    {
        "name": "Local PyTorch CUDA 13.2 runtime",
        "type": "runtime",
        "provider": "uv environment",
        "repo_or_source_id": "torch;torchvision;python",
        "license": "mixed",
        "gated": "false",
        "revision": "python 3.14.6; torch 2.12.1+cu132; torchvision 0.27.1+cu132",
        "local_status": "REQUIRED",
        "max_vram_gb": "24",
        "used_in_notebooks": "1.6",
        "gt_tier": "GT-1",
        "notes": "Validated on RTX 5090 Laptop GPU with CUDA 13.2 and BF16 matmul.",
    },
    {
        "name": "Course fake-result diagnostics",
        "type": "dataset",
        "provider": "generated",
        "repo_or_source_id": "course_generated_fake_result_diagnostics_v1",
        "license": "course_generated",
        "gated": "false",
        "revision": "seed_0",
        "local_status": "GENERATED_BY_COURSE",
        "max_vram_gb": "0",
        "used_in_notebooks": "0.6",
        "gt_tier": "GT-0",
        "notes": (
            "Deterministic leakage, cherry-pick, probe-overfit, and "
            "random-direction controls."
        ),
    },
    {
        "name": "TransformerLens",
        "type": "tooling",
        "provider": "Python",
        "repo_or_source_id": "TransformerLensOrg/TransformerLens",
        "license": "MIT",
        "gated": "false",
        "revision": "2.18.0",
        "local_status": "REQUIRED",
        "max_vram_gb": "24",
        "used_in_notebooks": "1.6;7.1;7.2;7.3;7.4;7.5;8.1;8.2;8.3;8.4",
        "gt_tier": "GT-1",
        "notes": "Hook and cache interface for transformer model organisms.",
    },
    {
        "name": "NNsight",
        "type": "tooling",
        "provider": "Python",
        "repo_or_source_id": "ndif-team/nnsight",
        "license": "Apache-2.0",
        "gated": "false",
        "revision": "0.7.0",
        "local_status": "REQUIRED",
        "max_vram_gb": "24",
        "used_in_notebooks": "8.1;9.1",
        "gt_tier": "GT-1",
        "notes": "Intervention tooling for local or remote graph execution.",
    },
    {
        "name": "SAELens",
        "type": "tooling",
        "provider": "Python",
        "repo_or_source_id": "SAELens",
        "license": "MIT",
        "gated": "false",
        "revision": "6.44.4",
        "local_status": "REQUIRED",
        "max_vram_gb": "24",
        "used_in_notebooks": "5.2;6.1;6.2;6.3",
        "gt_tier": "GT-1",
        "notes": "SAE loading and evaluation tooling.",
    },
    {
        "name": "Attribution libraries",
        "type": "tooling",
        "provider": "Python",
        "repo_or_source_id": "captum;shap;shapiq;inseq",
        "license": "mixed",
        "gated": "false",
        "revision": "captum 0.9.0; shap 0.49.1; shapiq 1.5.2",
        "local_status": "REQUIRED",
        "max_vram_gb": "24",
        "used_in_notebooks": "8.2;16.1;16.2;16.3;16.4;16.5;16.6;16.7;16.8;future_SHAP",
        "gt_tier": "GT-0",
        "notes": (
            "inseq remains isolated because current releases conflict with "
            "TransformerLens typeguard."
        ),
    },
    {
        "name": "PEFT stack",
        "type": "tooling",
        "provider": "Python",
        "repo_or_source_id": "peft;bitsandbytes;accelerate",
        "license": "mixed",
        "gated": "false",
        "revision": "peft 0.19.1; bitsandbytes 0.49.2; accelerate 1.14.0",
        "local_status": "REQUIRED",
        "max_vram_gb": "24",
        "used_in_notebooks": "15.1",
        "gt_tier": "GT-1",
        "notes": "bitsandbytes uses BNB_CUDA_VERSION=130 with CUDA 13.2 torch.",
    },
    {
        "name": "Diffusers",
        "type": "tooling",
        "provider": "Python",
        "repo_or_source_id": "huggingface/diffusers",
        "license": "Apache-2.0",
        "gated": "false",
        "revision": "0.38.0",
        "local_status": "REQUIRED",
        "max_vram_gb": "24",
        "used_in_notebooks": "13.1",
        "gt_tier": "GT-1",
        "notes": (
            "Pinned image-generation pipeline stack used for SD-Turbo and SD1.5 "
            "safe-shape generation, cross-attention capture, token ablation, and "
            "image-quality controls."
        ),
    },
    {
        "name": "GPT-2 small",
        "type": "language_model",
        "provider": "TransformerLens",
        "repo_or_source_id": "gpt2-small",
        "license": "MIT",
        "gated": "false",
        "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "local_status": "OPTIONAL_FUTURE",
        "max_vram_gb": "8",
        "used_in_notebooks": "future_IOI_full_replication",
        "gt_tier": "GT-2",
        "notes": "Core IOI, induction, patching, and SAE model organism.",
    },
    {
        "name": "GELU 1L512W C4 Code",
        "type": "language_model",
        "provider": "Hugging Face via TransformerLens",
        "repo_or_source_id": "NeelNanda/GELU_1L512W_C4_Code",
        "license": "MIT",
        "gated": "false",
        "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
        "local_status": "REQUIRED",
        "max_vram_gb": "1",
        "used_in_notebooks": "7.1;7.2;7.3;7.4;7.5;8.1;8.2;8.3;8.4",
        "gt_tier": "GT-1",
        "notes": (
            "Pinned small TransformerLens checkpoint used for real residual-stream "
            "activation-patching preflight."
        ),
    },
    {
        "name": "Pythia 70M deduped",
        "type": "language_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "EleutherAI/pythia-70m-deduped",
        "license": "Apache-2.0",
        "gated": "false",
        "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
        "local_status": "REQUIRED",
        "max_vram_gb": "8",
        "used_in_notebooks": "6.3;8.5;9.1;9.2;11.1",
        "gt_tier": "GT-2",
        "notes": "Small real LM target matched to released Sparse Feature Circuits SAE artifacts, safe 9.1 category hidden-state preflight, and 11.1 weekday geometry preflight.",
    },
    {
        "name": "Qwen2.5 0.5B Instruct",
        "type": "instruction_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "license": "Qwen license",
        "gated": "false",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "local_status": "REQUIRED",
        "max_vram_gb": "4",
        "used_in_notebooks": "9.1",
        "gt_tier": "GT-2",
        "notes": (
            "Pinned instruction-model target for 9.1 sanitized no-generation "
            "addition/projection-out checks plus the public refusal/compliance "
            "GT-2 aggregate completion path with raw prompt and completion text "
            "omitted from artifacts."
        ),
    },
    {
        "name": "Refusal compliance pairs",
        "type": "dataset",
        "provider": "Hugging Face",
        "repo_or_source_id": "josephmayo/refusal-compliance-pairs",
        "license": "unknown",
        "gated": "false",
        "revision": "b6ed3432f1d4a695e13be1c373bf7fb5af43f376",
        "local_status": "REQUIRED",
        "max_vram_gb": "0",
        "used_in_notebooks": "9.1",
        "gt_tier": "GT-2",
        "notes": (
            "Public refusal/compliance prompt-pair dataset used by 9.1 for "
            "held-out direction separation, layer/position/PCA controls, and "
            "aggregate-only behavioral completion effects; raw prompt text is "
            "not saved in course artifacts."
        ),
    },
    {
        "name": "TinyStories",
        "type": "dataset_and_model_family",
        "provider": "Hugging Face",
        "repo_or_source_id": "roneneldan/TinyStories",
        "license": "CDLA-Permissive-2.0",
        "gated": "false",
        "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "local_status": "REQUIRED",
        "max_vram_gb": "8",
        "used_in_notebooks": "15.1",
        "gt_tier": "GT-3",
        "notes": "Small text target for LoRA proxy work.",
    },
    {
        "name": "BGE small EN v1.5",
        "type": "embedding_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "BAAI/bge-small-en-v1.5",
        "license": "MIT",
        "gated": "false",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "local_status": "REQUIRED",
        "max_vram_gb": "4",
        "used_in_notebooks": "5.6",
        "gt_tier": "GT-1",
        "notes": "Public embedding-model preflight for controlled retrieval and permuted-pair controls.",
    },
    {
        "name": "EmbeddingGemma 300M",
        "type": "embedding_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "google/embeddinggemma-300m",
        "license": "Gemma terms",
        "gated": "manual",
        "revision": "57c266a740f537b4dc058e1b0cda161fd15afa75",
        "local_status": "REQUIRED",
        "max_vram_gb": "8",
        "used_in_notebooks": "5.6",
        "gt_tier": "GT-1",
        "notes": (
            "Authenticated EmbeddingGemma retrieval preflight verified in 5.6 on "
            "generated query/document pairs with permuted-pair controls."
        ),
    },
    {
        "name": "FunctionGemma 270M IT",
        "type": "function_calling_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "google/functiongemma-270m-it",
        "license": "Gemma terms",
        "gated": "manual",
        "revision": "39eccb091651513a5dfb56892d3714c1b5b8276c",
        "local_status": "REQUIRED",
        "max_vram_gb": "8",
        "used_in_notebooks": "5.6",
        "gt_tier": "GT-1",
        "notes": (
            "Authenticated base FunctionGemma repo directly loaded in 5.6 for a "
            "benign CUDA forward pass; the public Mobile Actions derivative remains "
            "the held-out function-calling evaluation path."
        ),
    },
    {
        "name": "FunctionGemma 270M Mobile Actions",
        "type": "function_calling_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "litert-community/FunctionGemma_270M_Mobile_Actions",
        "license": "Gemma terms",
        "gated": "false",
        "revision": "e2226c1def35c5443942ebdb90a1da2a9eda836a",
        "local_status": "REQUIRED",
        "max_vram_gb": "2",
        "used_in_notebooks": "5.6",
        "gt_tier": "GT-1",
        "notes": (
            "Public safetensors FunctionGemma derivative used for Mobile Actions "
            "function-call generation and structured-argument validation."
        ),
    },
    {
        "name": "Othello-GPT",
        "type": "world_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "NeelNanda/Othello-GPT-Transformer-Lens",
        "license": "unknown",
        "gated": "false",
        "revision": "905ca1a68b9f7dff77adc56af1962e5f6fcac274",
        "local_status": "REQUIRED",
        "max_vram_gb": "8",
        "used_in_notebooks": "14.1;16.1;future_SHAP",
        "gt_tier": "GT-2",
        "notes": "Board-state world-model interpretability target.",
    },
    {
        "name": "Gemma 3 1B IT",
        "type": "language_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "google/gemma-3-1b-it",
        "license": "Gemma terms",
        "gated": "true",
        "revision": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "local_status": "REQUIRED",
        "max_vram_gb": "24",
        "used_in_notebooks": "5.1;5.2;6.2",
        "gt_tier": "GT-1",
        "notes": "Modern transformer target; use quantization if needed.",
    },
    {
        "name": "Gemma Scope 2",
        "type": "sae_artifact",
        "provider": "Hugging Face",
        "repo_or_source_id": "google/gemma-scope-2-1b-it",
        "license": "Gemma terms",
        "gated": "true",
        "revision": "b0fa29457c3601df0a70c48a15534c738d7c10e0",
        "local_status": "REQUIRED",
        "max_vram_gb": "2",
        "used_in_notebooks": "5.2;6.2",
        "gt_tier": "GT-1",
        "notes": (
            "Pinned 1B-IT layer-13 residual JumpReLU SAE preflight plus authenticated "
            "Gemma 3 layer-13 activation validation on a narrow benign semantic split "
            "with random-feature and label-shuffle controls."
        ),
    },
    {
        "name": "Mamba 130M",
        "type": "sequence_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "state-spaces/mamba-130m-hf",
        "license": "Apache-2.0",
        "gated": "false",
        "revision": "1e76775f628fbf1350fbe4dbb3d971ba64af25a1",
        "local_status": "REQUIRED",
        "max_vram_gb": "8",
        "used_in_notebooks": "5.3;5.4",
        "gt_tier": "GT-1",
        "notes": (
            "Safetensors HF checkpoint used for official Mamba logits/generation "
            "preflight in 5.3 and hidden-state extraction preflight in 5.4."
        ),
    },
    {
        "name": "CLIP ViT-B/32",
        "type": "vision_language_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "openai/clip-vit-base-patch32",
        "license": "MIT",
        "gated": "false",
        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "local_status": "REQUIRED",
        "max_vram_gb": "8",
        "used_in_notebooks": "12.1;13.1;16.5",
        "gt_tier": "GT-1",
        "notes": (
            "Pinned CLIP contrastive baseline used for rendered-shape retrieval, "
            "object-region patching, hidden visual-token activation patching, "
            "generated-image alignment controls, and VLM SHAP region/modality "
            "attribution."
        ),
    },
    {
        "name": "SigLIP base",
        "type": "vision_language_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "google/siglip-base-patch16-224",
        "license": "Apache-2.0",
        "gated": "false",
        "revision": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        "local_status": "REQUIRED",
        "max_vram_gb": "8",
        "used_in_notebooks": "12.1",
        "gt_tier": "GT-1",
        "notes": (
            "Pinned SigLIP safetensors rendered-shape retrieval, object-region "
            "patching, hidden visual-token activation patching, and pairwise loss "
            "baseline."
        ),
    },
    {
        "name": "Qwen2.5-VL 3B Instruct",
        "type": "generative_vlm",
        "provider": "Hugging Face",
        "repo_or_source_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "license": "Qwen license",
        "gated": "false",
        "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "local_status": "REQUIRED",
        "max_vram_gb": "12",
        "used_in_notebooks": "12.1",
        "gt_tier": "GT-1",
        "notes": (
            "Pinned generative VLM rendered-shape generation preflight with red-square "
            "and blue-circle counterfactual controls."
        ),
    },
    {
        "name": "Stable Diffusion Turbo",
        "type": "image_generation_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "stabilityai/sd-turbo",
        "license": "see_model_card",
        "gated": "false",
        "revision": "b261bac6fd2cf515557d5d0707481eafa0485ec2",
        "local_status": "REQUIRED",
        "max_vram_gb": "8",
        "used_in_notebooks": "13.1",
        "gt_tier": "GT-1",
        "notes": (
            "Pinned safetensors SD-Turbo generation preflight scored by real CLIP, "
            "with captured cross-attention maps for safe shape-token localization."
        ),
    },
    {
        "name": "Stable Diffusion v1.5",
        "type": "image_generation_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "license": "CreativeML Open RAIL-M",
        "gated": "false",
        "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "local_status": "REQUIRED",
        "max_vram_gb": "16",
        "used_in_notebooks": "13.1",
        "gt_tier": "GT-1",
        "notes": (
            "Pinned SD1.5 safe-shape path with DAAM-style cross-attention "
            "localization, target-token ablation, random-token controls, "
            "image-quality preservation, and white-noise rejection."
        ),
    },
    {
        "name": "V-JEPA 2 ViT-L",
        "type": "world_model",
        "provider": "Hugging Face",
        "repo_or_source_id": "facebook/vjepa2-vitl-fpc64-256",
        "license": "CC-BY-NC",
        "gated": "false",
        "revision": "b3c1679b7c34d3255ef3547f27c7b226aefab26f",
        "local_status": "REQUIRED",
        "max_vram_gb": "24",
        "used_in_notebooks": "14.1",
        "gt_tier": "GT-1",
        "notes": "Pinned V-JEPA 2 ViT-L feature-extraction preflight target for synthetic video controls.",
    },
    {
        "name": "Synthetic colored-shapes VLM data",
        "type": "dataset",
        "provider": "generated",
        "repo_or_source_id": "course_generated_colored_shapes_v1",
        "license": "course_generated",
        "gated": "false",
        "revision": "seed_12345",
        "local_status": "GENERATED_BY_COURSE",
        "max_vram_gb": "0",
        "used_in_notebooks": "12.1;future_VLM_geometry",
        "gt_tier": "GT-3",
        "notes": "Controlled object/color/count/spatial labels.",
    },
    {
        "name": "Safe refusal proxy prompts",
        "type": "dataset",
        "provider": "generated",
        "repo_or_source_id": "course_generated_refusal_proxy_prompts_v1",
        "license": "course_generated",
        "gated": "false",
        "revision": "seed_123",
        "local_status": "GENERATED_BY_COURSE",
        "max_vram_gb": "0",
        "used_in_notebooks": "9.1;15.1",
        "gt_tier": "GT-3",
        "notes": "Sanitized JSONL prompt pairs with redaction support.",
    },
    {
        "name": "Course LoRA adapters",
        "type": "adapter",
        "provider": "generated",
        "repo_or_source_id": "rank1_sentiment;rank4_json;safe_proxy_loras",
        "license": "course_generated",
        "gated": "false",
        "revision": "rank1_safe_proxy_seed0_steps160",
        "local_status": "GENERATED_BY_COURSE",
        "max_vram_gb": "24",
        "used_in_notebooks": "15.1;future_PEFT",
        "gt_tier": "GT-0",
        "notes": (
            "Section 15.1 trains a generated rank-1 safe proxy adapter with fixed "
            "seed, merge parity, random-label control, and same-norm random-adapter "
            "control."
        ),
    },
    {
        "name": "Course modular-addition training checkpoints",
        "type": "dataset",
        "provider": "generated",
        "repo_or_source_id": "course_generated_modular_addition_checkpoints_v1",
        "license": "course_generated",
        "gated": "false",
        "revision": "seed_0_mod13_steps80",
        "local_status": "GENERATED_BY_COURSE",
        "max_vram_gb": "1",
        "used_in_notebooks": "17.1",
        "gt_tier": "GT-0",
        "notes": (
            "Section 17.1 trains a tiny modular-addition MLP on CUDA, saves and "
            "reloads checkpoints, and compares true labels against a random-label "
            "control."
        ),
    },
]
METHOD_ROWS: list[dict[str, str]] = [
    {
        "method_name": "Fake-result diagnostics",
        "paper": "Course GT-0 controls",
        "year": "2026",
        "category": "verification",
        "model_family": "methodology",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CUDA_DIAGNOSTIC_PREFLIGHT",
        "baseline_status": "HAS_LEAKAGE_CHERRYPICK_OVERFIT_AND_RANDOM_DIRECTION_CONTROLS",
        "notes": (
            "Section 0.6; deterministic fake-result diagnostics plus a CUDA preflight "
            "that measures label leakage, cherry-pick inflation, probe overfit, and "
            "random-direction control rejection."
        ),
    },
    {
        "method_name": "Local frontier ML harness",
        "paper": "Course infrastructure",
        "year": "2026",
        "category": "infrastructure",
        "model_family": "all",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT",
        "baseline_status": "HAS_BASELINES",
        "notes": "Section 1.6; environment, VRAM, parity, activation-store helpers.",
    },
    {
        "method_name": "Gemma-style decoder block",
        "paper": "Gemma",
        "year": "2024",
        "category": "architecture",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_HF_REFERENCE_PARITY_PREFLIGHT",
        "baseline_status": "HAS_TRANSFORMERS_ARCHITECTURE_PARITY",
        "notes": (
            "Section 5.1; local Gemma implementation now includes CUDA parity against "
            "Hugging Face transformers.GemmaForCausalLM on a deterministic tiny "
            "matched-weight config, including cache and top-k checks. Gated pretrained "
            "Gemma weights are not claimed."
        ),
    },
    {
        "method_name": "Gemma Scope feature steering",
        "paper": "Gemma Scope",
        "year": "2024",
        "category": "sparse_features",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_YELLOW",
        "implementation_status": "REQUIRED_LOAD_WEIGHTS",
        "verification_status": "IMPLEMENTED_ARTIFACT_PREFLIGHT",
        "baseline_status": "HAS_PINNED_SAE_CONFIG_TENSOR_AND_FORWARD_CHECK",
        "notes": (
            "Sections 5.2 and 6.2 now load a pinned Gemma Scope 2 1B-IT residual "
            "SAE artifact on CUDA and verify config, tensor shapes, finiteness, "
            "and JumpReLU encode/decode. Real Gemma activation feature validation "
            "remains gated and is not claimed."
        ),
    },
    {
        "method_name": "Mamba selective scan",
        "paper": "Mamba",
        "year": "2023",
        "category": "architecture",
        "model_family": "state_space_model",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT",
        "baseline_status": "HAS_EXACT_PARITY_AND_OFFICIAL_CHECKPOINT_PREFLIGHT",
        "notes": (
            "Section 5.3; recurrent/parallel/chunked scan and cache parity are "
            "implemented, and a pinned Mamba-130M-HF logits/generation CUDA "
            "preflight verifies official checkpoint loading with fast kernels."
        ),
    },
    {
        "method_name": "Mamba-3-style state tracking",
        "paper": "Mamba-3",
        "year": "2026",
        "category": "architecture",
        "model_family": "state_space_model",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_TINY_MAMBA_TRANSFORMER_COMPARISON_AND_INTERVENTION",
        "baseline_status": "HAS_TRAINED_MAMBA_TRANSFORMER_COMPARISON_AND_OFFICIAL_HIDDEN_STATE_PREFLIGHT",
        "notes": (
            "Section 5.4 now trains a tiny Mamba bracket-depth organism with longer-sequence "
            "generalization and random-label controls, compares it against a trained tiny "
            "causal Transformer baseline on the same generated task, performs learned-state "
            "interventions on Mamba hidden states, and loads a pinned official Mamba-130M-HF "
            "hidden-state extraction preflight."
        ),
    },
    {
        "method_name": "Discrete diffusion language modeling",
        "paper": "Diffusion Language Models",
        "year": "2022",
        "category": "architecture",
        "model_family": "diffusion_lm",
        "has_code": "true",
        "has_weights": "partial",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": (
            "IMPLEMENTED_CUDA_TRAINED_TINY_DIFFUSION_LM_AND_DIFFUSIONGEMMA_READINESS_PREFLIGHT"
        ),
        "baseline_status": "HAS_EXACT_CONTROLS_AND_NVFP4_GENERATION_PROOF",
        "notes": (
            "Section 5.5; toy noising/denoising/remasking/sampler contracts plus a CUDA-"
            "trained tiny conditional discrete diffusion LM on a generated copy-pair "
            "grammar, with held-out masked accuracy, confidence-remasking sampler exact "
            "match, activation trajectory checks, and shuffled-label control. It also "
            "checks pinned DiffusionGemma config/processor/model-class support, exact "
            "BF16/NVFP4 revisions, shard readiness, and a real isolated vLLM NVFP4 "
            "generation proof without accepting config-only evidence as generation."
        ),
    },
    {
        "method_name": "DiffusionGemma local inference",
        "paper": "DiffusionGemma",
        "year": "2026",
        "category": "architecture",
        "model_family": "diffusion_lm",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN_ISOLATED_VLLM",
        "implementation_status": "REQUIRED_LOAD_WEIGHTS",
        "verification_status": "NVFP4_ISOLATED_VLLM_GENERATION_PROVEN",
        "baseline_status": "NEEDS_DIFFUSION_TIME_ACTIVATION_CAPTURE_AND_PATCHING",
        "notes": (
            "Pinned Google BF16 and NVIDIA NVFP4 artifacts are checked by section 5.5. "
            "Google BF16 direct local loading remains deferred for the 24GB tier, while "
            "the NVIDIA NVFP4 checkpoint has a real isolated vLLM 0.24.0 generation proof "
            "on the RTX 5090 Laptop GPU. Main uv remains torch 2.12.1+cu132 without vLLM."
        ),
    },
    {
        "method_name": "Specialist and embedding model controls",
        "paper": "FunctionGemma and EmbeddingGemma docs",
        "year": "2025",
        "category": "architecture",
        "model_family": "specialist_models",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_AUTHENTICATED_EMBEDDINGGEMMA_AND_FUNCTIONGEMMA_PREFLIGHTS",
        "baseline_status": "HAS_SYNTHETIC_BGE_EMBEDDINGGEMMA_FUNCTIONGEMMA_CONTROLS",
        "notes": (
            "Section 5.6 now includes synthetic specialist-model controls, authenticated "
            "Google EmbeddingGemma retrieval with permuted-pair controls, a pinned public "
            "BGE embedding comparison, a pinned public FunctionGemma Mobile Actions "
            "generation preflight on held-out dataset rows, and direct authenticated "
            "base FunctionGemma CUDA loading on a benign forward pass."
        ),
    },
    {
        "method_name": "SAE variants",
        "paper": "Sparse autoencoders",
        "year": "2024",
        "category": "sparse_features",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_REAL_MODEL_PREFLIGHT",
        "baseline_status": "HAS_TOY_AND_PINNED_PYTHIA_SAE_CONTROLS",
        "notes": (
            "Section 6.1 keeps ReLU, TopK, Gated, and JumpReLU toy contracts, then "
            "trains a tiny TopK SAE on pinned Pythia-70M hidden states with held-out "
            "reconstruction, permuted-decoder, density, feature-AUC, and decoder-"
            "steering controls."
        ),
    },
    {
        "method_name": "Transcoders and attribution graphs",
        "paper": "Transcoders Find Interpretable LLM Feature Circuits",
        "year": "2024",
        "category": "sparse_features",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "partial",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_TRANSFORMERLENS_MLP_FEATURE_GRAPH_PREFLIGHT",
        "baseline_status": "HAS_ORACLE_PARITY_TRAINED_TRANSCODER_AND_GRAPH_CONTROLS",
        "notes": (
            "Section 6.3; toy transcoder contracts plus pinned TransformerLens GELU-1L "
            "CUDA preflight checking exact MLP-feature oracle replacement, trained tiny "
            "ReLU transcoder held-out reconstruction/top-token agreement, and feature-"
            "graph top-feature versus low-effect controls. Full published artifact "
            "replication remains separate."
        ),
    },
    {
        "method_name": "Crosscoders and model diffing",
        "paper": "Crosscoders",
        "year": "2024",
        "category": "sparse_features",
        "model_family": "paired_models",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_PAIRED_TRANSFORMERLENS_MODEL_DIFF_PREFLIGHT",
        "baseline_status": "HAS_REAL_PAIRED_MODEL_AND_RANDOM_DIRECTION_CONTROLS",
        "notes": (
            "Section 6.4; toy paired-feature contracts plus pinned GELU-1L versus SoLU-1L "
            "CUDA model-diffing preflight with exact shared-plus-delta reconstruction, "
            "SVD delta direction, technical/everyday prompt separation, and orthogonal "
            "random-direction ablation control. Full trained crosscoder replication remains "
            "separate."
        ),
    },
    {
        "method_name": "Logit lens, tuned lens, and Patchscopes",
        "paper": "Patchscopes",
        "year": "2024",
        "category": "activation_to_language",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN_TOY",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT",
        "baseline_status": "HAS_DECODING_BASELINES",
        "notes": "Section 7.1; lens and counterfactual activation contracts.",
    },
    {
        "method_name": "Feature verbalizers",
        "paper": "Automated feature explanation",
        "year": "2024",
        "category": "activation_to_language",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN_TOY",
        "implementation_status": "TOY_REPRO_ONLY",
        "verification_status": "IMPLEMENTED_CONTRACT",
        "baseline_status": "HAS_HELDOUT_BASELINES",
        "notes": "Section 7.2; explanation must predict held-out examples.",
    },
    {
        "method_name": "Mini Activation Oracles",
        "paper": "Activation Oracles",
        "year": "2025",
        "category": "activation_to_language",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT_AND_REAL_MODEL_PREFLIGHT",
        "baseline_status": "HAS_ORACLE_BASELINES_OOD_RANDOM_AND_PATCHING_CONTROLS",
        "notes": (
            "Section 7.3 now keeps the toy oracle contracts and adds a pinned "
            "TransformerLens gelu-1l residual-direction mini-oracle CUDA preflight "
            "with text-only/probe baselines, OOD splits, random abstention, and "
            "clean-to-corrupt activation patching. Full LoRA/API Activation Oracle "
            "training remains outside the local claim."
        ),
    },
    {
        "method_name": "Mini Natural Language Autoencoders",
        "paper": "Natural Language Autoencoders",
        "year": "2026",
        "category": "activation_to_language",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT_AND_REAL_MODEL_PREFLIGHT",
        "baseline_status": "HAS_RECONSTRUCTION_TEXT_ONLY_LATENT_OOD_AND_SHUFFLE_CONTROLS",
        "notes": (
            "Section 7.4 now keeps the toy reconstruction contracts and adds a pinned "
            "TransformerLens gelu-1l phrase-bottleneck mini-NLA CUDA preflight with "
            "discrete natural-language phrase explanations, phrase-to-residual "
            "prototype decoding, no numeric payloads, activation reconstruction, "
            "text-only and prompt-label baselines, latent/logit-diff preservation, "
            "OOD prompts, brevity, shuffled-text and blank-text controls, and "
            "counterfactual explanation changes. Anthropic-scale NLA training is not "
            "claimed."
        ),
    },
    {
        "method_name": "Predictive Concept Decoders",
        "paper": "Predictive Concept Decoders",
        "year": "2025",
        "category": "activation_to_language",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT_AND_REAL_MODEL_PREFLIGHT",
        "baseline_status": "HAS_QUESTION_CONDITIONED_CONCEPT_BASELINES_AND_REMOVAL_CONTROLS",
        "notes": (
            "Section 7.5 now keeps the toy sparse-concept contracts and adds a pinned "
            "TransformerLens gelu-1l sparse-concept PCD CUDA preflight with signed "
            "residual concepts, a trained question-conditioned decoder over explicit "
            "concept-question interaction features, trained non-interaction baselines, "
            "four behavioral questions, OOD long-context prompts, question-shuffle "
            "control, multi-seed concept stability, concept audit checks, and active "
            "top-vs-random concept-removal controls. This is a local PCD mechanics "
            "preflight, not a broad PCD benchmark."
        ),
    },
    {
        "method_name": "Activation patching",
        "paper": "Transformer Circuits and ARENA",
        "year": "2022",
        "category": "circuit_methods",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT",
        "baseline_status": "HAS_RANDOM_CONTROLS",
        "notes": "Section 8.1; reusable clean/corrupt patching metrics.",
    },
    {
        "method_name": "Attribution patching, EAP, and integrated effects",
        "paper": "EAP / EAP-IG",
        "year": "2024",
        "category": "circuit_methods",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT",
        "baseline_status": "HAS_EXACT_PATCHING_BASELINE",
        "notes": "Section 8.2; exact-vs-approx attribution controls.",
    },
    {
        "method_name": "ACDC and circuit metrics",
        "paper": "ACDC",
        "year": "2023",
        "category": "circuit_methods",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN_TOY",
        "implementation_status": "TOY_REPRO_ONLY",
        "verification_status": "IMPLEMENTED_CONTRACT",
        "baseline_status": "HAS_RANDOM_CIRCUIT_BASELINE",
        "notes": "Section 8.3; faithfulness, minimality, and completeness metrics.",
    },
    {
        "method_name": "Sparse Feature Circuits",
        "paper": "Sparse Feature Circuits",
        "year": "2024",
        "category": "circuit_methods",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "OFFICIAL_PYTHIA_SFC_REPLICATION",
        "verification_status": "IMPLEMENTED_100_EXAMPLE_GRAPH_AND_HELDOUT_FAITHFULNESS",
        "baseline_status": "HAS_TOY_BASELINES_OFFICIAL_SAE_AND_HELDOUT_FAITHFULNESS",
        "notes": (
            "Section 8.5 implements the toy ladder plus a Pythia-70M-deduped residual-feature "
            "subject/verb preflight, official artifact manifest check, and released "
            "SAE state-dict, one-layer feature-attribution, a 100-example official-code "
            "sparse feature graph, and held-out official faithfulness on 40 simple_test "
            "examples, plus safe generated-data SHIFT-style sparse-feature editing."
        ),
    },
    {
        "method_name": "Refusal directions",
        "paper": "Refusal is Mediated by a Single Direction",
        "year": "2024",
        "category": "alignment_interpretability",
        "model_family": "chat_transformer",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_GT2_PUBLIC_REFUSAL_DIRECTION_REPLICATION",
        "baseline_status": "HAS_PUBLIC_DATASET_LAYER_POSITION_PCA_RANDOM_AND_LABEL_CONTROLS",
        "notes": (
            "Section 9.1 now includes safe toy controls, a pinned Pythia-70M-deduped "
            "hidden-state category preflight across three sanitized prompt-template "
            "families, pinned Qwen2.5-0.5B-Instruct no-generation addition/projection-out "
            "logit interventions, and a scoped GT-2 public josephmayo/refusal-compliance-pairs "
            "aggregate replication path with held-out direction separation, layer and "
            "position sweeps, PCA/SVD structure, random-direction and label-shuffle "
            "negative controls, aggregate-only behavioral completion metrics, and no "
            "raw prompt/completion text saved."
        ),
    },
    {
        "method_name": "Chain-of-thought faithfulness probes",
        "paper": "CoT faithfulness",
        "year": "2025",
        "category": "alignment_interpretability",
        "model_family": "reasoning_models",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT_AND_REAL_MODEL_PREFLIGHT",
        "baseline_status": "HAS_TEXT_ONLY_LABEL_SHUFFLE_AND_PATCHING_CONTROLS",
        "notes": (
            "Section 9.2 now keeps the toy CoT-faithfulness contracts and adds a "
            "pinned Pythia-70M-deduped hidden-state preflight on safe A/B private-answer "
            "prompts. It trains a thresholded hidden-answer direction on template "
            "variants, evaluates held-out hidden-answer prediction, compares against "
            "visible-rationale/text-only and label-shuffled controls, patches the "
            "hidden answer state through the LM head, and reports condition-level "
            "A/B logit accuracies without generated completions."
        ),
    },
    {
        "method_name": "Safe proxy drift detection",
        "paper": "Emergent misalignment",
        "year": "2025",
        "category": "alignment_interpretability",
        "model_family": "finetuned_transformer",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT_AND_REAL_MODEL_PREFLIGHT",
        "baseline_status": "HAS_SAFE_PROXY_LABEL_SHUFFLE_RANDOM_AND_MITIGATION_CONTROLS",
        "notes": (
            "Section 9.3 keeps the benign toy drift contracts and adds a pinned "
            "Pythia-70M-deduped hidden-state preflight on safe proxy-drift prompt "
            "pairs. It trains a thresholded drift direction, evaluates held-out "
            "contexts across five benign proxy kinds, compares label-shuffled and "
            "random-direction controls, aligns feature scores with a safe next-token "
            "behavior proxy, and tests projection-style mitigation without generated "
            "completions."
        ),
    },
    {
        "method_name": "White-box eval monitors",
        "paper": "White-box monitors",
        "year": "2026",
        "category": "alignment_interpretability",
        "model_family": "transformer",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CONTRACT_AND_REAL_MODEL_PREFLIGHT",
        "baseline_status": "HAS_BLACK_BOX_LABEL_SHUFFLE_RANDOM_AND_FALSE_POSITIVE_CONTROLS",
        "notes": (
            "Section 9.4 keeps the toy dashboard/calibration/false-positive contracts "
            "and adds a pinned Pythia-70M-deduped hidden-state monitor preflight on "
            "safe generated eval records. It calibrates held-out failure scores, "
            "compares against a real next-token pass/fail black-box proxy, verifies "
            "white-box catches of black-box-missed failures, validates explanation "
            "labels, and checks label-shuffled plus fixed random-direction controls "
            "without generated completions."
        ),
    },
    {
        "method_name": "Representation geometry controls",
        "paper": "Language Models Represent Space and Time",
        "year": "2023",
        "category": "representation_geometry",
        "model_family": "transformer_and_vlm",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CALENDAR_GEOMETRY_PREFLIGHT",
        "baseline_status": "HAS_RANDOM_LABEL_CONTROLS_AND_REAL_LM_PREFLIGHT",
        "notes": (
            "Section 11.1 now includes PCA/SVD metrics, white-noise controls, "
            "template-centering controls, and a pinned Pythia-70M-deduped weekday/month "
            "calendar hidden-state geometry preflight with raw-template failure, "
            "permuted-label controls, and white-noise controls."
        ),
    },
    {
        "method_name": "CLIP, SigLIP, and VLM controls",
        "paper": "CLIP / SigLIP / VLM interpretability",
        "year": "2024",
        "category": "multimodal_interpretability",
        "model_family": "vlm",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_YELLOW",
        "implementation_status": "REQUIRED_LOAD_WEIGHTS",
        "verification_status": "IMPLEMENTED_CONTRACT_AND_HIDDEN_TOKEN_PREFLIGHT",
        "baseline_status": (
            "HAS_SYNTHETIC_BASELINES_REAL_CLIP_SIGLIP_HIDDEN_PATCHING_AND_QWEN25_VL_PREFLIGHTS"
        ),
        "notes": (
            "Section 12.1 now includes synthetic colored-shape scene schema checks, "
            "image-grounding baselines, object-region patch controls, pinned real "
            "CLIP/SigLIP rendered-shape retrieval plus object-region counterfactual "
            "patching, real hidden visual-token activation patching with "
            "object/background/same-size random-token/full-sequence controls, and a "
            "pinned Qwen2.5-VL 3B rendered-shape generation preflight."
        ),
    },
    {
        "method_name": "Image-generation interpretability",
        "paper": "DAAM and diffusion concept directions",
        "year": "2023",
        "category": "image_generation_interpretability",
        "model_family": "diffusion_and_ar_image",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_YELLOW",
        "implementation_status": "REQUIRED_LOAD_WEIGHTS",
        "verification_status": "IMPLEMENTED_SD15_DAAM_TOKEN_ABLATION_QUALITY_CONTROLS",
        "baseline_status": "HAS_PROMPT_REGION_SD_TURBO_SD15_CLIP_AND_NOISE_CONTROLS",
        "notes": (
            "Section 13.1 now includes toy prompt-region controls, a supplemental "
            "pinned SD-Turbo safe-shape generation preflight, and a required pinned "
            "SD1.5 safe-shape path with DAAM-style color/shape token localization, "
            "target-token ablation over random-token controls, CLIP alignment, "
            "image-quality preservation, and white-noise rejection."
        ),
    },
    {
        "method_name": "JEPA and world-model controls",
        "paper": "I-JEPA / V-JEPA 2",
        "year": "2025",
        "category": "world_models",
        "model_family": "jepa_video_world_model",
        "has_code": "true",
        "has_weights": "true",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_SYNTHETIC_OCCLUSION_VJEPA_PREFLIGHT",
        "baseline_status": "HAS_WORLD_STATE_BASELINES_AND_REAL_VJEPA2_PREFLIGHT",
        "notes": (
            "Section 14.1 now includes toy JEPA/world-model controls plus a pinned "
            "V-JEPA 2 ViT-L synthetic-video feature-extraction preflight with same-object "
            "and synthetic occlusion object-permanence contrasts; masked target "
            "prediction, real-video permanence benchmarks, and action-conditioned "
            "rollouts remain outside this local preflight."
        ),
    },
    {
        "method_name": "LoRA, DoRA, and adapter interpretability",
        "paper": "LoRA / DoRA",
        "year": "2024",
        "category": "peft_interpretability",
        "model_family": "finetuned_transformer",
        "has_code": "true",
        "has_weights": "generated",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_MATCHED_LORA_DORA_FULL_FINETUNE_COMPARISON",
        "baseline_status": "HAS_TRAINED_SAFE_PROXY_LORA_DORA_FULL_FINETUNE_CONTROLS",
        "notes": (
            "Section 15.1 now includes exact LoRA/DoRA tensor controls plus a trained "
            "rank-1 safe proxy LoRA on generated target-direction data, merge/unmerge "
            "parity, random-label control, same-norm random-adapter control, and a "
            "matched CUDA comparison against rank-1 DoRA and full linear finetuning on "
            "the same safe generated task."
        ),
    },
    {
        "method_name": "Exact Shapley values",
        "paper": "Shapley values",
        "year": "1953",
        "category": "attribution_baselines",
        "model_family": "ground_truth_games",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CUDA_NEURAL_GAME_PREFLIGHT",
        "baseline_status": "HAS_EXACT_BASELINES_AND_SHUFFLED_LABEL_CONTROL",
        "notes": (
            "Section 16.1; exact coalition tables, parity checks, and a CUDA-trained "
            "neural coalition-game preflight whose real ablation Shapley values are "
            "checked against analytic ground truth plus a shuffled-label control."
        ),
    },
    {
        "method_name": "KernelSHAP and PartitionSHAP",
        "paper": "SHAP",
        "year": "2017",
        "category": "attribution_baselines",
        "model_family": "tabular_and_token_games",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CUDA_NEURAL_GAME_PREFLIGHT",
        "baseline_status": "HAS_EXACT_BASELINES_AND_GROUPING_CONTROLS",
        "notes": (
            "Section 16.2; full-table KernelSHAP and PartitionSHAP parity, plus a "
            "CUDA-trained neural coalition-game preflight with singleton, aligned, "
            "mismatched grouping, and shuffled-label controls."
        ),
    },
    {
        "method_name": "Shapley interaction values",
        "paper": "shapiq",
        "year": "2024",
        "category": "attribution_baselines",
        "model_family": "interaction_games",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CUDA_NEURAL_GAME_PREFLIGHT",
        "baseline_status": "HAS_INTERACTION_BASELINES_AND_SHUFFLED_LABEL_CONTROL",
        "notes": (
            "Section 16.3; pairwise interaction and shapiq parity checks, plus a "
            "CUDA-trained neural coalition-game preflight recovering planted positive "
            "and negative interactions with off-target and shuffled-label controls."
        ),
    },
    {
        "method_name": "TokenSHAP and TokenShapley",
        "paper": "TokenSHAP / TokenShapley",
        "year": "2025",
        "category": "attribution_baselines",
        "model_family": "language_model",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CUDA_TOKEN_SCORER_PREFLIGHT",
        "baseline_status": "HAS_TOKEN_BASELINES_AND_SHUFFLED_LABEL_CONTROL",
        "notes": (
            "Section 16.4; exact and sampled masked-token attribution, plus a "
            "CUDA-trained token-position scorer preflight with analytic, efficiency, "
            "ranking, and shuffled-label controls."
        ),
    },
    {
        "method_name": "VLM modality and region SHAP",
        "paper": "MM-SHAP / PixelSHAP",
        "year": "2025",
        "category": "attribution_baselines",
        "model_family": "vlm",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_PINNED_CLIP_PREFLIGHT",
        "baseline_status": "HAS_REGION_CONTROLS_AND_RENDERED_CLIP_BASELINE",
        "notes": (
            "Section 16.5; modality synergy and background controls, plus a pinned "
            "CLIP rendered-shape preflight for modality and object/background/OCR "
            "region SHAP on real logits."
        ),
    },
    {
        "method_name": "SHAP vs activation patching",
        "paper": "Mechanistic attribution comparisons",
        "year": "2026",
        "category": "attribution_baselines",
        "model_family": "ground_truth_games",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CUDA_MODEL_ORGANISM_PREFLIGHT",
        "baseline_status": "HAS_AGREEMENT_AND_DISAGREEMENT_CONTROLS",
        "notes": (
            "Section 16.6; agreement and interaction-heavy disagreement cases, plus "
            "CUDA-trained additive and nonlinear model organisms compared through "
            "exact Shapley and full-minus-ablated patching effects."
        ),
    },
    {
        "method_name": "Data Shapley in one training run",
        "paper": "Data Shapley",
        "year": "2019",
        "category": "training_data_attribution",
        "model_family": "small_training_problem",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CUDA_ONE_STEP_TRAINING_PREFLIGHT",
        "baseline_status": "HAS_EXACT_MC_AND_IN_RUN_CONTROLS",
        "notes": (
            "Section 16.7; exact, Monte Carlo, and in-run first-order checks, plus a "
            "CUDA one-step linear training preflight with full coalition enumeration, "
            "autograd per-example scores, and harmful-example deletion control."
        ),
    },
    {
        "method_name": "SHAPley and mechanistic agreement matrix",
        "paper": "Course comparison protocol",
        "year": "2026",
        "category": "attribution_baselines",
        "model_family": "multi_method",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_CUDA_MECH_AGREEMENT_PREFLIGHT",
        "baseline_status": "HAS_AGREEMENT_INTERACTION_AND_SHUFFLED_CONTROLS",
        "notes": (
            "Section 16.8; rank correlation, top-k, deletion, XOR checks, plus a "
            "CUDA-trained nonlinear game comparing Shapley values to known "
            "mechanistic scores, pair interactions, and a shuffled-label control."
        ),
    },
    {
        "method_name": "Checkpoint archaeology",
        "paper": "Course developmental interpretability protocol",
        "year": "2026",
        "category": "training_dynamics",
        "model_family": "ar_jepa_diffusion_mamba",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "REQUIRED_IMPLEMENT",
        "verification_status": "IMPLEMENTED_TRAINED_CHECKPOINT_PREFLIGHT",
        "baseline_status": "HAS_TOY_AND_RANDOM_LABEL_CHECKPOINT_CONTROLS",
        "notes": (
            "Section 17.1 keeps deterministic toy family trajectories, then trains "
            "a tiny modular-addition MLP on CUDA, saves and reloads real checkpoints, "
            "detects stable emergence and a phase jump, and rejects a random-label "
            "checkpoint control."
        ),
    },
    {
        "method_name": "Full-scale Activation Oracles",
        "paper": "Activation Oracles",
        "year": "2025",
        "category": "activation_to_language",
        "model_family": "frontier_scale",
        "has_code": "partial",
        "has_weights": "false",
        "local_24gb_status": "READ_ONLY",
        "implementation_status": "READ_ONLY_TOO_EXPENSIVE",
        "verification_status": "EXPLAIN_LIMITS_ONLY",
        "baseline_status": "NOT_APPLICABLE",
        "notes": "Roadmap says use miniature local versions, not full-scale training.",
    },
    {
        "method_name": "Max-activating examples as standalone explanations",
        "paper": "Classic feature visualization",
        "year": "2020",
        "category": "deprecated_evidence",
        "model_family": "all",
        "has_code": "true",
        "has_weights": "not_applicable",
        "local_24gb_status": "LOCAL_GREEN",
        "implementation_status": "DEPRECATED_BY_NEWER_METHOD",
        "verification_status": "HYPOTHESIS_GENERATOR_ONLY",
        "baseline_status": "NEEDS_HELDOUT_AND_CAUSAL_CONTROLS",
        "notes": "Allowed for exploration, never enough for a final interpretation.",
    },
    {
        "method_name": "V-JEPA 2.1 dense features",
        "paper": "V-JEPA 2.1",
        "year": "2026",
        "category": "world_models",
        "model_family": "video_jepa",
        "has_code": "unknown",
        "has_weights": "unknown",
        "local_24gb_status": "WAIT_FOR_STABLE_WEIGHTS",
        "implementation_status": "WAIT_FOR_WEIGHTS",
        "verification_status": "WATCHLIST",
        "baseline_status": "NEEDS_RELEASE_BASELINES",
        "notes": "Optional dense-feature chapter once stable weights/tooling exist.",
    },
]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def is_extension_section(number: str) -> bool:
    if number in {"0.6", "1.6"}:
        return True
    try:
        return int(number.split(".", maxsplit=1)[0]) >= 5
    except ValueError:
        return False


def gt_tier_for(number: str) -> str:
    explicit = {
        "0.6": "GT-0",
        "1.6": "GT-1",
        "5.1": "GT-0",
        "5.2": "GT-3",
        "5.3": "GT-1",
        "5.4": "GT-1",
        "5.5": "GT-0",
        "5.6": "GT-1",
        "6.1": "GT-1",
        "6.2": "GT-1",
        "6.3": "GT-0",
        "6.4": "GT-3",
        "7.1": "GT-1",
        "7.2": "GT-1",
        "7.3": "GT-1",
        "7.4": "GT-3",
        "7.5": "GT-3",
        "8.1": "GT-1",
        "8.2": "GT-1",
        "8.3": "GT-1",
        "8.4": "GT-1",
        "8.5": "GT-0",
        "9.1": "GT-2",
        "9.2": "GT-3",
        "9.3": "GT-3",
        "9.4": "GT-3",
        "10.1": "GT-4",
        "11.1": "GT-0",
        "12.1": "GT-1",
        "13.1": "GT-1",
        "14.1": "GT-1",
        "15.1": "GT-0",
        "16.1": "GT-0",
        "16.2": "GT-0",
        "16.3": "GT-0",
        "16.4": "GT-0",
        "16.5": "GT-0",
        "16.6": "GT-0",
        "16.7": "GT-0",
        "16.8": "GT-0",
        "17.1": "GT-0",
    }
    return explicit.get(number, "GT-3")


def verification_schema() -> dict[str, Any]:
    required = [
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
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ARENA extension verification report",
        "type": "object",
        "additionalProperties": True,
        "required": required,
        "properties": {
            "notebook_id": {"type": "string"},
            "date_run": {"type": "string"},
            "git_commit": {"type": "string"},
            "report_inputs": {
                "type": "object",
                "required": ["schema_version", "algorithm", "combined_sha256", "files"],
                "properties": {
                    "schema_version": {"type": "integer", "const": 1},
                    "algorithm": {"type": "string", "const": "sha256"},
                    "combined_sha256": {"type": "string"},
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["path", "sha256", "size_bytes"],
                            "properties": {
                                "path": {"type": "string"},
                                "sha256": {"type": "string"},
                                "size_bytes": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                },
            },
            "gt_tier": {"type": "string", "enum": ["GT-0", "GT-1", "GT-2", "GT-3", "GT-4"]},
            "evidence_level": {"type": "string"},
            "claim_scope": {"type": "string"},
            "gpu_name": {"type": "string"},
            "peak_vram_gb": {"type": "number", "minimum": 0},
            "wall_clock_seconds": {"type": "number", "minimum": 0},
            "models": {"type": "array"},
            "datasets": {"type": "array"},
            "tests_passed": {"type": "boolean"},
            "metrics": {"type": "object"},
            "baselines": {"type": "object"},
            "negative_controls": {"type": "object"},
            "ood_tests": {"type": "object"},
            "known_failures": {"type": "array"},
            "safety_notes": {"type": "array"},
            "accepted": {"type": "boolean"},
        },
    }


def difficulty_for(number: str, gt_tier: str) -> int:
    """Return a conservative 1-5 difficulty estimate for exercise metadata."""

    explicit = {
        "0.6": 2,
        "1.6": 3,
        "5.1": 4,
        "5.3": 4,
        "8.2": 4,
        "8.5": 4,
        "10.1": 5,
        "14.1": 4,
        "16.1": 4,
        "16.2": 4,
        "16.3": 4,
        "16.7": 4,
        "16.8": 4,
        "17.1": 4,
    }
    if number in explicit:
        return explicit[number]
    if gt_tier in {"GT-1", "GT-2"}:
        return 4
    if gt_tier == "GT-3":
        return 3
    return 3


def importance_for(number: str) -> int:
    """Return a conservative 1-5 importance estimate for exercise metadata."""

    chapter = int(number.split(".", maxsplit=1)[0]) if "." in number else 0
    if number in {"1.6", "5.1", "5.3", "6.1", "8.1", "8.2", "8.5", "9.1", "16.1"}:
        return 5
    if chapter in {5, 6, 8, 9, 16}:
        return 4
    return 3


def expected_runtime_for(number: str) -> str:
    """Return runtime text in the roadmap metadata style."""

    real_model_sections = {
        "0.6",
        "1.6",
        "5.1",
        "5.2",
        "5.3",
        "5.4",
        "5.5",
        "5.6",
        "6.1",
        "6.2",
        "6.3",
        "6.4",
        "7.1",
        "7.2",
        "7.3",
        "7.4",
        "7.5",
        "8.1",
        "8.2",
        "8.3",
        "8.4",
        "8.5",
        "9.1",
        "9.2",
        "9.3",
        "9.4",
        "11.1",
        "12.1",
        "13.1",
        "14.1",
        "15.1",
        "16.1",
        "16.2",
        "16.3",
        "16.4",
        "16.5",
        "16.6",
        "16.7",
        "16.8",
        "17.1",
    }
    if number in real_model_sections:
        return "seconds on toy contract; minutes on local real-model path"
    return "seconds on toy contract"


def requires_gpu_for(number: str) -> bool:
    """Return whether the exercise's intended release gate requires a GPU."""

    return True


def extension_sections() -> list[dict[str, Any]]:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    records: list[dict[str, Any]] = []
    for chapter_name, chapter in config["chapters"].items():
        chapter_dir = ROOT / chapter_name
        for section in chapter.get("sections", []):
            number = str(section.get("number", ""))
            if not is_extension_section(number):
                continue
            title = section["title"]
            exercise_dir = chapter_dir / "exercises" / section["exercise_dir"]
            gt_tier = gt_tier_for(number)
            records.append(
                {
                    "chapter_name": chapter_name,
                    "number": number,
                    "title": title,
                    "exercise_dir": exercise_dir,
                    "notebook_id": f"{number.replace('.', '_')}_{slugify(title)}",
                    "gt_tier": gt_tier,
                    "exercise_metadata": {
                        "EXERCISE_ID": f"{number.replace('.', '_')}_{slugify(title)}",
                        "GT_TIER": gt_tier,
                        "DIFFICULTY": difficulty_for(number, gt_tier),
                        "IMPORTANCE": importance_for(number),
                        "EXPECTED_RUNTIME": expected_runtime_for(number),
                        "REQUIRES_GPU": requires_gpu_for(number),
                    },
                }
            )
    return records


def write_text(path: Path, content: str, *, overwrite: bool = True) -> None:
    if not overwrite and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def write_json(path: Path, data: Any, *, overwrite: bool = True) -> None:
    write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n", overwrite=overwrite)


def write_yaml(path: Path, data: Any, *, overwrite: bool = True) -> None:
    write_text(path, yaml.safe_dump(data, sort_keys=False), overwrite=overwrite)


def existing_lock_for(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the current artifact lock for a section if it already exists."""

    path = record["exercise_dir"] / "artifacts.lock.yml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def lock_for(record: dict[str, Any]) -> dict[str, Any]:
    lock = {
        "notebook_id": record["notebook_id"],
        "section": record["number"],
        "title": record["title"],
        "exercise_metadata": record["exercise_metadata"],
        "gt_tier": record["gt_tier"],
        "evidence_level": "notebook_contract",
        "claim_scope": (
            "Deterministic notebook contract only unless this lock names exact "
            "real-model revisions, datasets, baselines, negative controls, and "
            "measured GPU artifacts."
        ),
        "required_gpu_gb": 0,
        "max_allowed_gpu_gb": 24,
        "models": [
            {
                "id": "toy_or_generated_inputs",
                "source": "generated",
                "revision": "seed_0",
                "precision": "float32",
                "gated": False,
            }
        ],
        "datasets": [
            {
                "id": f"{record['notebook_id']}_toy_inputs",
                "source": "generated",
                "seed": 0,
            }
        ],
        "expected_metrics": {
            "tests_passed": True,
            "accepted": True,
            "max_allowed_gpu_gb": 24,
        },
        "controls": [
            "baseline_or_reference_check",
            "negative_control_where_applicable",
            "shape_dtype_or_exact_value_check",
        ],
        "safety_notes": [
            "No unsafe prompts, unsafe adapters, raw gated weights, or large data artifacts.",
            "Real-model paths must update this lock with exact revisions before release.",
        ],
    }
    if record["number"] == "0.6":
        lock["evidence_level"] = "cuda_fake_result_diagnostic_controls_preflight"
        lock["claim_scope"] = (
            "GT-0 fake-result diagnostics ladder: the notebook keeps deterministic "
            "toy reports for label leakage, cherry-picked effects, probe overfit, "
            "and random-direction controls, then runs the same diagnostic quantities "
            "on CUDA with explicit pass/fail metrics. This is a methodology section, "
            "not a real-model interpretability claim."
        )
        lock["required_gpu_gb"] = 1
        lock["freshness_inputs"] = [
            "arena_ext/capstone.py",
            "chapter10_capstone_research_sprint/exercises/part1_capstone_research_sprint/scripts/run_capstone.py",
            "chapter10_capstone_research_sprint/exercises/part1_capstone_research_sprint/results/metrics.json",
            "chapter10_capstone_research_sprint/exercises/part1_capstone_research_sprint/results/metrics_by_seed.json",
            "chapter10_capstone_research_sprint/exercises/part1_capstone_research_sprint/results/failure_cases.jsonl",
            "chapter10_capstone_research_sprint/exercises/part1_capstone_research_sprint/reports/capstone.md",
        ]
        lock["models"].append(
            {
                "id": "course_fake_result_diagnostic_tensors_v1",
                "source": "generated_cuda_tensors",
                "revision": "seed_606",
                "precision": "float32",
                "gated": False,
                "claim_scope": "diagnostic_control_model_organism",
            }
        )
        lock["datasets"].append(
            {
                "id": "course_fake_interpretability_failures_v1",
                "source": "generated",
                "seed": 606,
                "failure_modes": [
                    "label_leakage",
                    "cherry_picking",
                    "probe_overfit",
                    "random_direction_failure",
                ],
            }
        )
        lock["controls"].extend(
            [
                "direct_label_leakage_control",
                "shifted_no_leak_negative_control",
                "population_vs_selected_effect_control",
                "train_vs_heldout_probe_gap_control",
                "random_direction_p95_control",
            ]
        )
        lock["safety_notes"][-1] = (
            "This section uses generated numeric tensors only. It does not use real "
            "model completions, unsafe prompts, gated weights, or user data."
        )
    if record["number"] == "10.1":
        lock["evidence_level"] = "cuda_trained_activation_oracle_capstone_with_controls"
        lock["claim_scope"] = (
            "GT-4 mini capstone sprint: the section keeps the capstone planning "
            "scaffold, then runs a real CUDA-trained question-conditioned activation "
            "oracle on a generated latent-state benchmark. The claim is scoped to "
            "this model-organism: the oracle beats text-only and linear-probe "
            "baselines, solves a nonlinear XOR question, generalizes to held-out "
            "templates, and passes ablation, counterfactual patching, random-patch, "
            "random-activation, and label-shuffle controls. It does not claim a "
            "released-model mechanistic discovery."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "course_question_conditioned_activation_oracle_v1",
                "source": "generated_training_run",
                "revision": "seeds_0_1_2",
                "precision": "float32",
                "gated": False,
                "architecture": "2_layer_mlp_question_conditioned_binary_oracle",
                "claim_scope": "generated_latent_state_model_organism_only",
            }
        )
        lock["datasets"].append(
            {
                "id": "activation_oracle_latent_questions_v1",
                "source": "generated",
                "seeds": [0, 1, 2],
                "d_model": 12,
                "question_count": 4,
                "train_templates": [0, 1, 2],
                "heldout_templates": [3, 4],
                "examples_per_template": 48,
                "seed_count": 3,
                "label_rule": "color_bit, shape_bit, material_bit, and color_xor_shape",
            }
        )
        lock["controls"].extend(
            [
                "baseline_suite_completeness_check",
                "causal_validation_suite_completeness_check",
                "reproducibility_metadata_check",
                "writeup_path_check",
                "cuda_training_run_per_seed",
                "text_only_question_prior_baseline",
                "linear_probe_bank_baseline",
                "heldout_template_ood_split",
                "relevant_latent_dimension_ablation",
                "counterfactual_latent_dimension_patching",
                "random_dimension_patch_control",
                "random_activation_accuracy_control",
                "label_shuffle_negative_control",
            ]
        )
        lock["safety_notes"][-1] = (
            "This section trains on generated numeric latent-state activations only. "
            "It does not use unsafe prompts, gated weights, model completions, "
            "external APIs, or user data. The report explicitly scopes the claim to "
            "the generated benchmark."
        )
    if record["number"] == "1.6":
        lock["evidence_level"] = "local_cuda132_pytorch_runtime_and_bf16_tensor_preflight"
        lock["claim_scope"] = (
            "GT-1 local infrastructure gate: the uv environment must expose Python "
            "3.14, PyTorch 2.12.1+cu132, torchvision 0.27.1+cu132, CUDA 13.2, a "
            "24GB-class BF16-capable NVIDIA GPU, clean package constraints, and a "
            "real deterministic BF16 CUDA matmul. The Gemma-sized memory estimate "
            "is reported separately and is not treated as measured peak VRAM."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "local_pytorch_cuda132_runtime",
                "source": "uv_environment",
                "revision": "python_3.14.6_torch_2.12.1+cu132_torchvision_0.27.1+cu132",
                "precision": "bfloat16",
                "gated": False,
                "claim_scope": "runtime_cuda_bf16_tensor_preflight",
                "gpu_target": "NVIDIA GeForce RTX 5090 Laptop GPU",
            }
        )
        lock["datasets"].append(
            {
                "id": "deterministic_cuda_bf16_matmul_seed1234",
                "source": "torch.randn",
                "seed": 1234,
                "shape": [1024, 1024],
                "dtype": "bfloat16",
            }
        )
        lock["controls"].extend(
            [
                "environment_version_report",
                "gemma_1b_budget_estimate_separate_from_measured_peak",
                "deterministic_cuda_bf16_matmul",
                "finite_tensor_output_check",
                "uv_pip_check_clean_required",
                "no_cpu_fallback_as_acceptance_path",
            ]
        )
        lock["safety_notes"][-1] = (
            "The runtime preflight allocates only deterministic synthetic tensors. "
            "No model weights, unsafe prompts, or user data are used in this check."
        )
    if record["number"] == "5.1":
        lock["evidence_level"] = "gemma_from_scratch_contract_plus_hf_reference_architecture_parity"
        lock["claim_scope"] = (
            "GT-0 Gemma-from-scratch architecture preflight: deterministic local "
            "RMSNorm, RoPE, grouped-query attention, sliding-mask, SwiGLU, and "
            "KV-cache contracts are kept for learner exercises, then the local "
            "implementation is compared on CUDA against Hugging Face's real "
            "transformers.GemmaForCausalLM reference architecture with a deterministic "
            "tiny config and matched random weights. The report requires all reference "
            "state-dict keys to load, close logit parity, top-k agreement, cache "
            "parity, the Gemma embedding normalizer, and measured VRAM. This proves "
            "architecture parity; it does not claim validation on gated pretrained "
            "Gemma weights."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "transformers.GemmaForCausalLM",
                "source": "local_transformers_reference_implementation",
                "revision": "deterministic_tiny_config_seed_5101",
                "precision": "float32",
                "gated": False,
                "claim_scope": "reference_architecture_parity_no_pretrained_weights",
                "weight_key_count": 21,
            }
        )
        lock["datasets"].append(
            {
                "id": "tiny_gemma_reference_input_ids_v1",
                "source": "generated_token_ids",
                "seed": 5101,
                "input_ids": [[1, 5, 8, 13, 2]],
                "vocab_size": 31,
                "hidden_size": 16,
                "layers": 2,
                "attention_heads": 4,
                "kv_heads": 2,
                "head_dim": 4,
            }
        )
        lock["controls"].extend(
            [
                "local_rmsnorm_formula_contract",
                "local_rope_norm_contract",
                "local_kv_cache_parity_contract",
                "local_clone_parity_contract",
                "huggingface_gemma_reference_architecture_logits_parity",
                "state_dict_key_count_match",
                "topk_agreement_check",
                "reference_cache_parity_check",
                "embedding_scale_normalizer_check",
                "gated_pretrained_gemma_weights_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The reference parity path uses deterministic random tiny weights and "
            "generated token IDs only. It does not download gated pretrained Gemma "
            "weights, use unsafe prompts, user data, adapters, or external APIs."
        )
    if record["number"] == "7.1":
        lock["evidence_level"] = "toy_lens_contract_plus_pinned_transformerlens_affine_lens_patchscope_preflight"
        lock["claim_scope"] = (
            "GT-1 activation-to-language preflight: reusable toy logit-lens, tuned-lens, "
            "attention-lens, Patchscope, counterfactual, and random-activation contracts "
            "are kept for learner exercises, then a pinned TransformerLens gelu-1l "
            "checkpoint is loaded on CUDA. The report trains a ridge affine lens from "
            "hook_resid_pre to normalized hook_resid_post on safe cached prompts, "
            "evaluates held-out next-token decoding against the model's own final "
            "predictions, runs an attention-lens diagnostic on a real head, patches "
            "source hook_resid_post activations into neutral target prompts and compares "
            "the decoded answers against unpatched text-only target prompts, and checks "
            "counterfactual plus low-confidence random-activation controls. "
            "This is a local lens/Patchscope mechanics preflight, not a released tuned "
            "lens benchmark."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "NeelNanda/GELU_1L512W_C4_Code",
                    "source": "huggingface_via_transformerlens",
                    "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "affine_lens_and_activation_conditioned_decode_preflight",
                    "transformerlens_model_name": "gelu-1l",
                    "tokenizer_id": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                }
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_safe_lens_patchscope_prompts_v1",
                "source": "generated_prompt_sets",
                "seed": 0,
                "train_prompt_count": 16,
                "heldout_prompt_count": 6,
                "patchscope_pair_count": 6,
                "target_source": "model_final_logits_argmax",
                "hooks": ["blocks.0.hook_resid_pre", "blocks.0.hook_resid_post"],
                "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
            }
        )
        lock["controls"].extend(
            [
                "toy_logit_lens_contract",
                "toy_tuned_lens_improvement_contract",
                "toy_attention_lens_contract",
                "toy_patchscope_template_contract",
                "toy_counterfactual_activation_contract",
                "toy_random_activation_confidence_contract",
                "pinned_transformerlens_gelu1l_cached_activation_lens_training",
                "heldout_next_token_decoding_against_final_model_predictions",
                "real_attention_head_lens_finiteness_check",
                "source_residual_inserted_into_neutral_patchscope_prompt",
                "unpatched_neutral_prompt_text_only_baseline",
                "counterfactual_clean_corrupt_residual_decode_change",
                "small_random_activation_low_confidence_control",
                "released_tuned_lens_benchmark_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generic prompts and reads cached activations, "
            "logits, and attention values from a small public TransformerLens checkpoint. "
            "It does not use unsafe prompts, gated weights, or generated completions."
        )
    if record["number"] == "7.2":
        lock["evidence_level"] = "toy_verbalizer_contract_plus_pinned_transformerlens_residual_direction_preflight"
        lock["claim_scope"] = (
            "GT-1 feature-verbalizer preflight: reusable toy example-selection, "
            "keyword-prediction, counterexample, revision, intervention, and brevity "
            "contracts are kept for learner exercises, then a pinned TransformerLens "
            "gelu-1l checkpoint is loaded on CUDA. The report constructs a real "
            "residual direction from safe clean/corrupt final-token activations, scores "
            "safe prompt examples by projection onto that direction, validates a concise "
            "keyword explanation on held-out and contrastive examples, checks zero "
            "counterexamples, and verifies that adding the direction increases the "
            "predicted token logit difference. This is a local residual-direction "
            "verbalizer preflight, not an API-LLM or SAE-feature explanation claim."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "NeelNanda/GELU_1L512W_C4_Code",
                    "source": "huggingface_via_transformerlens",
                    "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "residual_direction_verbalizer_preflight",
                    "transformerlens_model_name": "gelu-1l",
                    "tokenizer_id": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                }
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_safe_residual_direction_verbalizer_prompts_v1",
                "source": "generated_prompt_sets",
                "seed": 0,
                "positive_anchor_prompt": "The cat sat on the",
                "negative_anchor_prompt": "The bird flew over the",
                "train_example_count": 8,
                "heldout_example_count": 8,
                "explanation_terms": ["sat", "slept", "rested"],
                "target_source": "manual_safe_prompt_labels_plus_real_residual_projection",
                "hook": "blocks.0.hook_resid_post",
                "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
            }
        )
        lock["controls"].extend(
            [
                "toy_example_selection_contract",
                "toy_explanation_prediction_contract",
                "toy_counterexample_revision_contract",
                "toy_intervention_direction_contract",
                "toy_brevity_contract",
                "pinned_transformerlens_gelu1l_residual_direction_scores",
                "heldout_keyword_prediction_against_safe_manual_labels",
                "contrastive_near_threshold_prompt_control",
                "train_heldout_prompt_disjointness_check",
                "learned_terms_from_train_only_check",
                "zero_counterexample_check",
                "direction_addition_logit_delta_intervention",
                "orthogonal_random_direction_intervention_control",
                "api_llm_and_sae_feature_explanation_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generic prompts and reads cached activations "
            "and logits from a small public TransformerLens checkpoint. It does not use "
            "unsafe prompts, gated weights, generated completions, or external API calls."
        )
    if record["number"] == "7.3":
        lock["evidence_level"] = (
            "toy_oracle_contract_plus_pinned_transformerlens_question_conditioned_oracle_preflight"
        )
        lock["claim_scope"] = (
            "GT-1 mini Activation Oracle preflight: reusable toy activation-question, "
            "baseline-comparison, OOD, random-activation, and patching contracts are "
            "kept for learner exercises, then a pinned TransformerLens gelu-1l "
            "checkpoint is loaded on CUDA. The report builds a tiny trained "
            "question-conditioned oracle over real cached final-token residuals, asks "
            "two opposite behavioral questions of each activation, compares against "
            "text-only and independently trained activation-only probe baselines, "
            "verifies held-out template, new-name, long-context, and adversarial prompt "
            "splits, requires low-margin abstention on random activations, and checks "
            "that clean-to-corrupt activation replacement changes the oracle answer. "
            "This is a local mini-oracle preflight, not a LoRA-trained or API-backed "
            "Activation Oracle benchmark."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "NeelNanda/GELU_1L512W_C4_Code",
                    "source": "huggingface_via_transformerlens",
                    "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "residual_direction_mini_activation_oracle_preflight",
                    "transformerlens_model_name": "gelu-1l",
                    "tokenizer_id": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                }
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_safe_residual_direction_activation_oracle_prompts_v1",
                "source": "generated_prompt_sets",
                "seed": 0,
                "positive_anchor_prompt": "The cat sat on the",
                "negative_anchor_prompt": "The bird flew over the",
                "train_example_count": 8,
                "eval_example_count": 8,
                "heldout_template_example_count": 8,
                "new_name_example_count": 8,
                "long_context_example_count": 8,
                "adversarial_example_count": 8,
                "answer_classes": ["negative", "positive", "abstain"],
                "question_rows_per_activation": 2,
                "target_source": (
                    "manual_safe_prompt_labels_plus_real_residual_projection_and_question_conditioning"
                ),
                "hook": "blocks.0.hook_resid_post",
                "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
            }
        )
        lock["controls"].extend(
            [
                "toy_activation_question_batch_contract",
                "toy_oracle_vs_text_probe_baseline_contract",
                "toy_template_split_contract",
                "toy_ood_split_contract",
                "toy_random_activation_abstention_contract",
                "toy_activation_patching_answer_change_contract",
                "pinned_transformerlens_gelu1l_question_conditioned_oracle_training",
                "balanced_text_only_baseline_accuracy_check",
                "independently_trained_activation_only_probe_negative_control",
                "independently_trained_activation_only_mlp_negative_control",
                "question_id_answer_flip_control",
                "heldout_template_new_name_long_context_adversarial_ood_splits",
                "low_margin_random_activation_abstention_control",
                "clean_corrupt_residual_patch_answer_change",
                "lora_trained_and_api_oracle_benchmark_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generic prompts and cached activations/logits "
            "from a small public TransformerLens checkpoint. It does not use unsafe "
            "prompts, gated weights, generated completions, adapter training, or "
            "external API calls."
        )
    if record["number"] == "7.4":
        lock["evidence_level"] = "toy_nla_contract_plus_pinned_transformerlens_phrase_bottleneck_preflight"
        lock["claim_scope"] = (
            "GT-3 mini Natural Language Autoencoder preflight: reusable toy "
            "activation-batch, reconstruction, logit-diff, latent-preservation, "
            "brevity, and counterfactual contracts are kept for learner exercises, "
            "then a pinned TransformerLens gelu-1l checkpoint is loaded on CUDA. "
            "The report encodes real final-token residuals into short discrete "
            "natural-language phrases, decodes phrase text through a learned "
            "phrase-to-residual prototype map, and validates activation MSE against "
            "no-activation and prompt-label baselines, direction-probe logit-diff "
            "preservation, latent preservation, OOD long-context prompts, brevity, "
            "no-numeric-payload checks, shuffled-text and blank-text controls, and "
            "counterfactual explanation changes. This is a local phrase-bottleneck "
            "mini-NLA preflight, not Anthropic-scale NLA training or a claim of "
            "faithful natural-language explanation of arbitrary activations."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "NeelNanda/GELU_1L512W_C4_Code",
                    "source": "huggingface_via_transformerlens",
                    "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "residual_text_bottleneck_mini_nla_preflight",
                    "transformerlens_model_name": "gelu-1l",
                    "tokenizer_id": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                }
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_safe_residual_text_bottleneck_nla_prompts_v1",
                "source": "generated_prompt_sets",
                "seed": 0,
                "positive_anchor_prompt": "The cat sat on the",
                "negative_anchor_prompt": "The bird flew over the",
                "train_example_count": 12,
                "eval_example_count": 8,
                "eval_split": "long_context_ood",
                "latent_labels": ["surface", "motion"],
                "text_bottleneck": "discrete_natural_language_phrase_bottleneck",
                "phrase_count": 12,
                "target_source": "manual_safe_prompt_labels_plus_phrase_prototype_reconstruction",
                "hook": "blocks.0.hook_resid_post",
                "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
            }
        )
        lock["controls"].extend(
            [
                "toy_nla_batch_contract",
                "toy_activation_reconstruction_contract",
                "toy_logit_diff_preservation_contract",
                "toy_latent_preservation_contract",
                "toy_brevity_contract",
                "toy_counterfactual_explanation_contract",
                "pinned_transformerlens_gelu1l_phrase_bottleneck_reconstruction",
                "no_activation_mean_reconstruction_baseline",
                "prompt_label_text_only_reconstruction_baseline",
                "direction_probe_logit_diff_preservation",
                "long_context_ood_prompt_split",
                "numeric_payload_rejection_control",
                "shuffled_explanation_text_negative_control",
                "blank_explanation_text_negative_control",
                "counterfactual_anchor_explanation_change",
                "anthropic_scale_nla_training_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generic prompts and cached activations/logits "
            "from a small public TransformerLens checkpoint. It does not use unsafe "
            "prompts, gated weights, generated completions, adapter training, external "
            "API calls, or claims about arbitrary hidden thoughts."
        )
    if record["number"] == "7.5":
        lock["evidence_level"] = "toy_pcd_contract_plus_pinned_transformerlens_trained_sparse_concept_decoder_preflight"
        lock["claim_scope"] = (
            "GT-3 Predictive Concept Decoder preflight: reusable toy question-batch, "
            "sparse encoding, question-conditioned decoder, baseline comparison, "
            "stability, removal, and concept-audit contracts are kept for learner "
            "exercises, then a pinned TransformerLens gelu-1l checkpoint is loaded "
            "on CUDA. The report builds signed sparse residual concepts from safe "
            "final-token activations, expands them into explicit concept-question "
            "interaction features, trains a tiny question-conditioned decoder on the "
            "train split, and evaluates four behavioral questions on held-out "
            "long-context prompts. It compares against trained question-agnostic "
            "sparse-concept, trained single-concept, and trained non-interaction "
            "activation+question baselines. It also checks concept sparsity, "
            "question-shuffle degradation, multi-seed top-concept stability, auditable "
            "top concept names, and active top-concept removal against a low-margin "
            "active-control removal. Legacy metric names still use random_removal "
            "for compatibility. This is a local PCD mechanics preflight, not a broad "
            "PCD benchmark or a full Activation Oracle comparison."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "NeelNanda/GELU_1L512W_C4_Code",
                    "source": "huggingface_via_transformerlens",
                    "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "sparse_concept_question_conditioned_decoder_preflight",
                    "transformerlens_model_name": "gelu-1l",
                    "tokenizer_id": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                }
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_safe_sparse_concept_pcd_prompts_v1",
                "source": "generated_prompt_sets",
                "seed": 0,
                "positive_anchor_prompt": "The cat sat on the",
                "negative_anchor_prompt": "The bird flew over the",
                "train_example_count": 12,
                "eval_example_count": 8,
                "eval_split": "long_context_ood",
                "question_count": 4,
                "pcd_row_count": 32,
                "train_pcd_row_count": 48,
                "latent_labels": ["surface", "motion"],
                "concepts": [
                    "surface_direction",
                    "motion_direction",
                    "pc1_positive",
                    "pc1_negative",
                    "pc2_positive",
                    "pc2_negative",
                    "pc3_positive",
                    "pc3_negative",
                ],
                "concept_top_k": 4,
                "target_source": "manual_safe_prompt_labels_plus_trained_concept_question_decoder",
                "hook": "blocks.0.hook_resid_post",
                "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
            }
        )
        lock["controls"].extend(
            [
                "toy_pcd_question_batch_contract",
                "toy_sparse_concept_encoding_contract",
                "toy_question_conditioned_decoder_contract",
                "toy_pcd_baseline_comparison_contract",
                "toy_concept_stability_contract",
                "toy_concept_removal_contract",
                "toy_concept_audit_contract",
                "pinned_transformerlens_gelu1l_sparse_signed_residual_concepts",
                "trained_question_conditioned_decoder",
                "concept_question_interaction_feature_map",
                "trained_question_agnostic_sparse_probe_baseline",
                "trained_single_concept_classifier_baseline",
                "trained_non_interaction_activation_question_baseline",
                "question_shuffle_negative_control",
                "long_context_ood_prompt_split",
                "multi_seed_decoder_stability",
                "top_concept_removal_control",
                "active_random_concept_removal_negative_control",
                "full_activation_oracle_and_broad_pcd_benchmark_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generic prompts and cached activations/logits "
            "from a small public TransformerLens checkpoint. It does not use unsafe "
            "prompts, gated weights, generated completions, adapter training, external "
            "API calls, or full Activation Oracle comparisons."
        )
    if record["number"] == "8.1":
        lock["evidence_level"] = "toy_metric_contract_plus_pinned_transformerlens_residual_patch_preflight"
        lock["claim_scope"] = (
            "GT-1 activation-patching preflight: reusable toy metric contracts are "
            "kept for learner exercises, then a pinned TransformerLens gelu-1l "
            "checkpoint is loaded on CUDA. The report runs clean/corrupt prompts, "
            "caches blocks.0.hook_resid_post, patches every sequence position into "
            "the corrupt run, and requires final-position recovery to beat all "
            "non-final wrong-position controls. This is a residual-stream patching "
            "mechanics preflight, not an IOI-scale circuit-discovery claim."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "NeelNanda/GELU_1L512W_C4_Code",
                    "source": "huggingface_via_transformerlens",
                    "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "residual_stream_activation_patching_preflight",
                    "transformerlens_model_name": "gelu-1l",
                    "tokenizer_id": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                }
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_safe_clean_corrupt_activation_patching_prompts_v1",
                "source": "generated_prompt_pair",
                "seed": 0,
                "clean_prompt": "The cat sat on the",
                "corrupt_prompt": "The bird flew over the",
                "positive_token": " floor",
                "negative_token": " top",
                "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
                "sequence_length": 6,
            }
        )
        lock["controls"].extend(
            [
                "toy_logit_diff_metric_contract",
                "toy_activation_slice_patch_contract",
                "toy_recovered_fraction_contract",
                "toy_localization_and_random_control_contract",
                "pinned_transformerlens_gelu1l_residual_stream_patch",
                "same_hook_all_sequence_positions_sweep",
                "non_final_position_wrong_patch_control",
                "finite_logits_and_exact_final_patch_recovery",
                "ioi_scale_circuit_discovery_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generic prompts and reads only next-token "
            "logits from a small public TransformerLens checkpoint. It does not use "
            "unsafe prompts, gated weights, or generated completions."
        )
    if record["number"] == "8.2":
        lock["evidence_level"] = "toy_attribution_contract_plus_pinned_transformerlens_gradient_preflight"
        lock["claim_scope"] = (
            "GT-1 attribution-patching preflight: reusable toy attribution, EAP, "
            "integrated-gradient, runtime, top-k, and false-negative contracts are "
            "kept for learner exercises, then a pinned TransformerLens gelu-1l "
            "checkpoint is loaded on CUDA. The report compares exact residual-stream "
            "patching scores against corrupt-run gradient attribution, integrated "
            "gradient attribution, and an EAP-style position-by-position edge matrix "
            "on the same clean/corrupt prompt pair. It requires top-1 agreement, "
            "high correlation, final-position recovery, and zero non-final gradient "
            "controls. This is a mechanics preflight, not an IOI-scale EAP circuit "
            "replication."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "NeelNanda/GELU_1L512W_C4_Code",
                    "source": "huggingface_via_transformerlens",
                    "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "exact_vs_gradient_activation_patching_preflight",
                    "transformerlens_model_name": "gelu-1l",
                    "tokenizer_id": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                }
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_safe_clean_corrupt_activation_patching_prompts_v1",
                "source": "generated_prompt_pair",
                "seed": 0,
                "clean_prompt": "The cat sat on the",
                "corrupt_prompt": "The bird flew over the",
                "positive_token": " floor",
                "negative_token": " top",
                "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
                "sequence_length": 6,
                "integrated_gradient_steps": 5,
            }
        )
        lock["controls"].extend(
            [
                "toy_attribution_dot_gradient_contract",
                "toy_integrated_gradient_contract",
                "toy_eap_edge_score_contract",
                "toy_exact_vs_approx_correlation_contract",
                "toy_false_negative_documentation_contract",
                "pinned_transformerlens_gelu1l_exact_patch_scores",
                "corrupt_run_gradient_attribution_scores",
                "same_hook_integrated_gradient_path_scores",
                "eap_position_edge_matrix_top_edge_check",
                "non_final_position_zero_gradient_control",
                "ioi_scale_eap_replication_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generic prompts and reads logits and "
            "gradients from a small public TransformerLens checkpoint. It does not "
            "use unsafe prompts, gated weights, or generated completions."
        )
    if record["number"] == "8.3":
        lock["evidence_level"] = "toy_circuit_metric_contract_plus_pinned_transformerlens_position_circuit_preflight"
        lock["claim_scope"] = (
            "GT-1 circuit-metric preflight: reusable toy ACDC pruning, faithfulness, "
            "minimality, completeness, random-baseline, and OOD-template contracts are "
            "kept for learner exercises, then a pinned TransformerLens gelu-1l "
            "checkpoint is loaded on CUDA. The report uses exact residual-stream "
            "patch scores to prune a position circuit, requires the final residual "
            "position to be the only kept edge, verifies faithfulness, minimality, "
            "completeness, and a same-size wrong-position baseline on the primary "
            "prompt pair, and checks the same final-position localization on held-out "
            "safe prompt templates. This is a mechanics preflight, not a full ACDC "
            "IOI or greater-than circuit discovery replication."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "NeelNanda/GELU_1L512W_C4_Code",
                    "source": "huggingface_via_transformerlens",
                    "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "position_circuit_metric_preflight",
                    "transformerlens_model_name": "gelu-1l",
                    "tokenizer_id": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                }
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_safe_clean_corrupt_position_circuit_prompts_v1",
                "source": "generated_prompt_pairs",
                "seed": 0,
                "primary_clean_prompt": "The cat sat on the",
                "primary_corrupt_prompt": "The bird flew over the",
                "heldout_pair_count": 4,
                "heldout_clean_prompts": [
                    "The cat sat on the",
                    "To make tea, boil the",
                    "The recipe calls for sugar and",
                    "The chef cooked a",
                ],
                "heldout_corrupt_prompts": [
                    "The bird flew over the",
                    "To make bread, bake the",
                    "The recipe calls for salt and",
                    "The teacher taught a",
                ],
                "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
            }
        )
        lock["controls"].extend(
            [
                "toy_acdc_pruning_contract",
                "toy_faithfulness_minimality_completeness_contracts",
                "toy_random_circuit_baseline_contract",
                "toy_ood_template_contract",
                "pinned_transformerlens_gelu1l_exact_patch_scores",
                "threshold_pruned_final_position_circuit",
                "same_size_wrong_position_random_baseline",
                "top_omitted_position_completeness_check",
                "heldout_prompt_template_final_position_localization",
                "full_acdc_ioi_replication_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generic prompts and reads only next-token "
            "logits from a small public TransformerLens checkpoint. It does not use "
            "unsafe prompts, gated weights, or generated completions."
        )
    if record["number"] == "8.4":
        lock["evidence_level"] = "toy_graph_contract_plus_pinned_transformerlens_eap_position_graph_preflight"
        lock["claim_scope"] = (
            "GT-1 attribution-graph preflight: reusable toy local-graph, graph-metric, "
            "path-perturbation, alternative-baseline, and counterfactual-summary "
            "contracts are kept for learner exercises, then a pinned TransformerLens "
            "gelu-1l checkpoint is loaded on CUDA. The report builds a top-1 local "
            "attribution graph from the real EAP position-edge matrix at "
            "blocks.0.hook_resid_post, requires the final-position self-edge to be "
            "the graph edge, and validates target-metric explanation, top-path "
            "perturbation, same-size alternative baseline failure, and a decrease "
            "counterfactual. This is a mechanics preflight, not a full sparse-feature "
            "or transcoder circuit-tracing replication."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "NeelNanda/GELU_1L512W_C4_Code",
                    "source": "huggingface_via_transformerlens",
                    "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "eap_position_graph_preflight",
                    "transformerlens_model_name": "gelu-1l",
                    "tokenizer_id": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                }
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_safe_clean_corrupt_position_graph_prompts_v1",
                "source": "generated_prompt_pair",
                "seed": 0,
                "clean_prompt": "The cat sat on the",
                "corrupt_prompt": "The bird flew over the",
                "positive_token": " floor",
                "negative_token": " top",
                "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
                "sequence_length": 6,
            }
        )
        lock["controls"].extend(
            [
                "toy_local_attribution_graph_contract",
                "toy_graph_metric_contract",
                "toy_path_perturbation_contract",
                "toy_alternative_graph_baseline_contract",
                "toy_counterfactual_summary_contract",
                "pinned_transformerlens_gelu1l_eap_position_edge_matrix",
                "top1_final_position_self_edge_graph_check",
                "top_path_perturbation_to_corrupt_metric",
                "same_size_wrong_position_alternative_graph_baseline",
                "summary_predicted_decrease_counterfactual",
                "full_sparse_feature_circuit_tracing_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generic prompts and reads logits and "
            "gradients from a small public TransformerLens checkpoint. It does not "
            "use unsafe prompts, gated weights, or generated completions."
        )
    if record["number"] == "5.2":
        lock["evidence_level"] = "toy_feature_controls_plus_real_gemma_scope_sae_artifact_preflight"
        lock["claim_scope"] = (
            "GT-3 feature-steering control ladder plus a pinned Gemma Scope 2 1B-IT "
            "layer-13 residual JumpReLU SAE artifact preflight on CUDA. This verifies "
            "released SAE config, tensor shapes, finiteness, and encode/decode execution; "
            "it does not claim semantic feature steering on real Gemma activations."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "google/gemma-scope-2-1b-it/resid_post/layer_13_width_16k_l0_small",
                "source": "huggingface",
                "revision": "b0fa29457c3601df0a70c48a15534c738d7c10e0",
                "precision": "float32",
                "gated": True,
                "claim_scope": "released_sae_artifact_config_tensor_and_forward_preflight",
                "base_model": "google/gemma-3-1b-it",
                "base_model_access": "gated_unavailable_without_authenticated_access",
            }
        )
        lock["datasets"].append(
            {
                "id": "constructed_sae_probe_activations_layer13_width16k_seed0",
                "source": "deterministic_from_sae_encoder_columns",
                "seed": 0,
                "semantic_feature_labels": False,
            }
        )
        lock["expected_metrics"].update(
            {
                "gemma_scope_artifact_preflight_passed": True,
                "gemma_scope_forward_passed": True,
                "gemma_scope_width": 16384,
                "gemma_scope_d_model": 1152,
                "gemma_scope_w_enc_shape": [1152, 16384],
                "gemma_scope_w_dec_shape": [16384, 1152],
                "gemma_scope_peak_vram_gb_max": 2.0,
                "gemma_scope_semantic_feature_claimed": False,
                "gemma3_base_authenticated": False,
                "gemma3_base_repo_listed": True,
                "gemma3_base_local_non_ref_file_count": 0,
                "gemma3_base_ready_for_real_activations": False,
                "gemma3_base_access_error_type": None,
                "gemma3_base_auth_error_type": "LocalTokenNotFoundError",
            }
        )
        lock["controls"].extend(
            [
                "toy_sae_reconstruction_metrics_contract",
                "heldout_feature_auc_contract",
                "steering_random_feature_control_contract",
                "ablation_top_activating_example_contract",
                "pinned_gemma_scope_sae_config_check",
                "pinned_gemma_scope_safetensor_shape_and_finiteness_check",
                "pinned_gemma_scope_jumprelu_cuda_encode_decode",
                "exact_gemma3_1b_it_base_model_access_gate",
                "no_unofficial_base_model_substitution",
                "real_gemma3_layer13_activation_capture",
                "heldout_real_gemma_activation_semantic_feature_auc",
                "real_gemma_activation_random_feature_control",
                "real_gemma_activation_label_shuffle_control",
            ]
        )
        lock["safety_notes"][-1] = (
            "Gemma Scope artifact loading and authenticated Gemma 3 1B IT layer-13 "
            "activation capture are verified on CUDA in this environment. The claim "
            "remains limited to the benign technical-vs-narrative semantic split and "
            "does not claim broad Gemma Scope coverage or safety-relevant steering behavior."
        )
    if record["number"] == "6.1":
        lock["evidence_level"] = "toy_sae_contract_plus_pinned_pythia_hidden_state_topk_sae_preflight"
        lock["claim_scope"] = (
            "GT-1 SAE-variant preflight: reusable ReLU/L1, TopK, Gated, JumpReLU, "
            "planted-dictionary, feature-AUC, and decoder-steering toy contracts are "
            "kept for learner exercises, then a pinned Pythia-70M-deduped checkpoint "
            "is loaded on CUDA. The report trains a 256-feature TopK-16 SAE on safe generated "
            "final-token hidden states, validates held-out normalized reconstruction "
            "against a zero baseline, checks a permuted-decoder negative control, "
            "requires nondegenerate sparse density and held-out feature-label AUC, "
            "and tests that adding a learned decoder vector moves the matching "
            "decoder-projection score more than an orthogonal random direction. This "
            "is a local SAE mechanics preflight, not a full GPT-2/Gemma SAE benchmark "
            "or a semantic feature-interpretation claim."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "EleutherAI/pythia-70m-deduped",
                "source": "huggingface",
                "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
                "precision": "float32",
                "gated": False,
                "claim_scope": "hidden_state_topk_sae_training_preflight",
                "hidden_layer": -1,
            }
        )
        lock["datasets"].append(
            {
                "id": "course_safe_pythia_sae_topic_prompts_v1",
                "source": "generated_prompt_sets",
                "seed": 0,
                "train_prompt_count": 64,
                "heldout_prompt_count": 16,
                "label_classes": ["technical_topic", "everyday_topic"],
                "technical_train_topics": [
                    "python debugging",
                    "matrix algebra",
                    "neural network",
                    "data pipeline",
                ],
                "everyday_train_topics": [
                    "sourdough recipe",
                    "garden planting",
                    "travel itinerary",
                    "music practice",
                ],
                "technical_heldout_topics": ["compiler design", "probability theorem"],
                "everyday_heldout_topics": ["painting class", "meal planning"],
                "target_source": "final_non_padding_token_hidden_state",
                "normalization": "train_split_featurewise_mean_std",
            }
        )
        lock["controls"].extend(
            [
                "toy_relu_l1_topk_gated_jumprelu_encoder_contracts",
                "toy_identity_reconstruction_metrics_contract",
                "toy_planted_dictionary_recovery_contract",
                "toy_feature_auc_contract",
                "toy_decoder_vector_steering_contract",
                "pinned_pythia70m_final_token_hidden_state_capture",
                "tiny_topk_sae_training_on_real_hidden_states",
                "heldout_zero_reconstruction_baseline",
                "permuted_decoder_negative_control",
                "nondegenerate_density_check",
                "heldout_topic_feature_auc_check",
                "orthogonal_random_decoder_projection_steering_control",
                "semantic_feature_interpretation_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generated topic prompts and hidden-state "
            "extraction from a small public Pythia checkpoint. It does not use unsafe "
            "prompts, gated weights, generated completions, user data, or raw large "
            "activation artifacts."
        )
    if record["number"] == "6.2":
        lock["evidence_level"] = "toy_feature_controls_plus_real_gemma_scope_sae_artifact_preflight"
        lock["claim_scope"] = (
            "GT-1 Gemma Scope artifact path: pinned 1B-IT layer-13 residual JumpReLU "
            "SAE config and safetensors are loaded, shape/finiteness checked, moved to "
            "CUDA, and exercised with a deterministic encode/decode probe. This is an "
            "artifact correctness preflight, not a semantic feature-interpretation claim "
            "on real Gemma activations."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "google/gemma-scope-2-1b-it/resid_post/layer_13_width_16k_l0_small",
                "source": "huggingface",
                "revision": "b0fa29457c3601df0a70c48a15534c738d7c10e0",
                "precision": "float32",
                "gated": True,
                "claim_scope": "released_sae_artifact_config_tensor_and_forward_preflight",
                "base_model": "google/gemma-3-1b-it",
                "base_model_access": "gated_unavailable_without_authenticated_access",
            }
        )
        lock["datasets"].append(
            {
                "id": "constructed_sae_probe_activations_layer13_width16k_seed0",
                "source": "deterministic_from_sae_encoder_columns",
                "seed": 0,
                "semantic_feature_labels": False,
            }
        )
        lock["expected_metrics"].update(
            {
                "gemma_scope_artifact_preflight_passed": True,
                "gemma_scope_forward_passed": True,
                "gemma_scope_width": 16384,
                "gemma_scope_d_model": 1152,
                "gemma_scope_w_enc_shape": [1152, 16384],
                "gemma_scope_w_dec_shape": [16384, 1152],
                "gemma_scope_peak_vram_gb_max": 2.0,
                "gemma_scope_semantic_feature_claimed": False,
                "gemma3_base_authenticated": False,
                "gemma3_base_repo_listed": True,
                "gemma3_base_local_non_ref_file_count": 0,
                "gemma3_base_ready_for_real_activations": False,
                "gemma3_base_access_error_type": None,
                "gemma3_base_auth_error_type": "LocalTokenNotFoundError",
            }
        )
        lock["controls"].extend(
            [
                "metadata_and_feature_tag_contract",
                "heldout_auc_and_baseline_contract",
                "base_instruction_delta_contract",
                "ablation_random_feature_control_contract",
                "steering_perplexity_guard_contract",
                "direct_logit_attribution_contract",
                "pinned_gemma_scope_sae_config_check",
                "pinned_gemma_scope_safetensor_shape_and_finiteness_check",
                "pinned_gemma_scope_jumprelu_cuda_encode_decode",
                "exact_gemma3_1b_it_base_model_access_gate",
                "no_unofficial_base_model_substitution",
                "real_gemma3_layer13_activation_capture",
                "heldout_real_gemma_activation_semantic_feature_auc",
                "real_gemma_activation_random_feature_control",
                "real_gemma_activation_label_shuffle_control",
            ]
        )
        lock["safety_notes"][-1] = (
            "Gemma Scope artifact loading and authenticated Gemma 3 1B IT layer-13 "
            "activation capture are verified on CUDA in this environment. The claim "
            "remains limited to the benign technical-vs-narrative semantic split and "
            "does not claim broad Gemma Scope coverage or safety-relevant steering behavior."
        )
    if record["number"] == "6.3":
        lock["evidence_level"] = "toy_transcoder_contract_plus_pinned_transformerlens_mlp_feature_graph_preflight"
        lock["claim_scope"] = (
            "GT-0/GT-1 sparse-feature ladder: toy transcoder helpers remain exact, then "
            "a pinned TransformerLens GELU-1L checkpoint is loaded on CUDA. The report "
            "checks exact MLP-feature oracle replacement parity, a trained tiny ReLU "
            "transcoder's held-out reconstruction/top-token behavior, and a feature-level "
            "attribution graph with top-feature versus low-effect controls. This is not a "
            "full published transcoder artifact replication."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "NeelNanda/GELU_1L512W_C4_Code",
                "alias": "gelu-1l",
                "source": "huggingface_transformerlens",
                "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
                "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                "precision": "float32",
                "gated": False,
                "claim_scope": "real_mlp_feature_graph_and_tiny_relu_transcoder_preflight",
            }
        )
        lock["datasets"].append(
            {
                "id": "course_generated_gelu1l_transcoder_prompt_set_v1",
                "source": "generated_safe_prompts",
                "seed": 6303,
                "prompt_count": 16,
                "activation_count": 104,
                "target_hook": "blocks.0.hook_mlp_out",
                "feature_hook": "blocks.0.mlp.hook_post",
                "input_hook": "blocks.0.ln2.hook_normalized",
            }
        )
        lock["controls"].extend(
            [
                "toy_transcoder_forward_contract",
                "toy_replacement_kl_contract",
                "toy_feature_logit_contribution_contract",
                "toy_graph_reproducibility_contract",
                "pinned_gelu1l_mlp_feature_oracle_replacement",
                "oracle_logits_and_mlp_output_parity",
                "trained_tiny_relu_transcoder_heldout_reconstruction",
                "trained_replacement_top1_agreement",
                "feature_graph_top64_preservation",
                "low_effect_feature_ablation_negative_control",
                "full_published_transcoder_artifact_replication_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses safe generic prompts and one small public "
            "TransformerLens checkpoint. It does not use unsafe prompts, gated weights, "
            "generated completions, or user data."
        )
    if record["number"] == "6.4":
        lock["evidence_level"] = "toy_crosscoder_contract_plus_paired_transformerlens_model_diff_preflight"
        lock["claim_scope"] = (
            "GT-3 paired-model diffing ladder: toy crosscoder helpers remain exact, then "
            "pinned GELU-1L and SoLU-1L TransformerLens checkpoints are loaded on CUDA. "
            "The report checks exact shared-plus-delta reconstruction, an SVD model-diff "
            "direction separating generated technical versus everyday prompts, and "
            "top-direction ablation against an orthogonal random-direction control. This "
            "is a real paired-checkpoint model-diffing preflight, not a full trained "
            "crosscoder paper replication."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "NeelNanda/GELU_1L512W_C4_Code",
                    "alias": "gelu-1l",
                    "source": "huggingface_transformerlens",
                    "revision": "bddc0e332f0ae84279e6a6a45d91b314899e1603",
                    "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "model_a_residual_stream",
                },
                {
                    "id": "NeelNanda/SoLU_1L512W_C4_Code",
                    "alias": "solu-1l",
                    "source": "huggingface_transformerlens",
                    "revision": "a4ce32db5e35f13e5f09333888bd2d42660f77ce",
                    "tokenizer": "NeelNanda/gpt-neox-tokenizer-digits",
                    "tokenizer_revision": "0f6671571a20be9756b9991d978047c03b75e749",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "model_b_residual_stream",
                },
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_generated_gelu_solu_model_diff_prompts_v1",
                "source": "generated_safe_prompt_categories",
                "seed": 6404,
                "technical_prompt_count": 8,
                "everyday_prompt_count": 8,
                "target_hook": "blocks.0.hook_resid_post",
                "label_classes": ["technical_prompt", "everyday_prompt"],
            }
        )
        lock["controls"].extend(
            [
                "toy_exact_crosscoder_reconstruction_contract",
                "toy_feature_specificity_contract",
                "toy_behavior_delta_auc_contract",
                "toy_model_specific_ablation_control",
                "pinned_gelu1l_vs_solu1l_paired_residuals",
                "exact_shared_plus_delta_reconstruction",
                "svd_model_diff_direction_auc",
                "top_direction_delta_ablation",
                "orthogonal_random_direction_ablation_control",
                "full_trained_crosscoder_paper_replication_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The paired-model path uses safe generic prompts and small public "
            "TransformerLens checkpoints. It does not use unsafe prompts, gated weights, "
            "generated completions, or user data."
        )
    if record["number"] == "5.5":
        lock["evidence_level"] = (
            "toy_diffusion_contract_plus_cuda_trained_conditional_diffusion_lm_and_"
            "diffusiongemma_readiness_preflight"
        )
        lock["claim_scope"] = (
            "GT-0 discrete diffusion language-model ladder: analytic noising, masked "
            "denoising loss, remasking, sampler, entropy, and trajectory helpers remain "
            "tested on toy fixtures, then a tiny Transformer denoiser is trained on CUDA "
            "for a generated conditional copy-pair grammar. The report checks held-out "
            "masked-token accuracy, confidence-remasking sampler reconstruction, activation "
            "trajectory shape, VRAM, and shuffled-label controls. The report also checks "
            "the pinned DiffusionGemma Transformers config, processor, tokenizer, model "
            "class, exact revisions, shard readiness, and vLLM/NVFP4 runtime readiness. "
            "It accepts released-checkpoint generation only from complete BF16 generation "
            "or from the pinned NVFP4 isolated-vLLM proof artifact; config-only evidence "
            "is never accepted as generation."
        )
        lock.setdefault("freshness_inputs", [])
        for freshness_input in [
            "scripts/run_diffusiongemma_vllm_probe.py",
            "chapter5_modern_architectures/exercises/part5_diffusion_language_models/"
            "artifacts/diffusiongemma_vllm_probe.json",
        ]:
            if freshness_input not in lock["freshness_inputs"]:
                lock["freshness_inputs"].append(freshness_input)
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "course_tiny_conditional_diffusion_lm_v1",
                "source": "generated_training_run",
                "revision": "seed_5505_steps_1200",
                "architecture": "2_layer_transformer_encoder_denoiser",
                "precision": "float32",
                "gated": False,
                "claim_scope": "gt0_conditional_discrete_diffusion_model_organism",
            }
        )
        lock["models"].extend(
            [
                {
                    "id": "google/diffusiongemma-26B-A4B-it",
                    "source": "huggingface",
                    "revision": "0f28bc42f588fbd8f71e08102b1c3960298a1358",
                    "architecture": "DiffusionGemmaForBlockDiffusion",
                    "precision": "bfloat16",
                    "gated": False,
                    "claim_scope": "pinned_config_processor_and_bf16_shard_readiness",
                },
                {
                    "id": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
                    "source": "huggingface",
                    "revision": "2ea837236295d617ac27f8c17d61228081932c40",
                    "architecture": "DiffusionGemmaForBlockDiffusion",
                    "precision": "nvfp4",
                    "gated": False,
                    "claim_scope": (
                        "pinned_nvfp4_shard_and_isolated_vllm_generation_readiness"
                    ),
                },
            ]
        )
        lock["datasets"].append(
            {
                "id": "copy_pair_conditional_suffix_grammar_v1",
                "source": "generated",
                "seed": 5505,
                "train_example_count": 80,
                "heldout_example_count": 20,
                "vocab_size": 11,
                "mask_token_id": 10,
                "sequence_length": 6,
                "label_rule": "[a,b] -> [a,a,b,b] suffix",
                "protected_prefix_tokens": 2,
                "shuffled_label_control": True,
            }
        )
        lock["controls"].extend(
            [
                "linear_mask_schedule_contract",
                "forward_noising_extreme_timestep_contract",
                "masked_denoising_loss_contract",
                "confidence_remasking_contract",
                "oracle_sampler_contract",
                "commitment_time_and_activation_trajectory_contract",
                "cuda_trained_tiny_conditional_diffusion_lm",
                "heldout_masked_suffix_accuracy",
                "confidence_remasking_sampler_exact_match",
                "shuffled_label_negative_control",
                "pinned_diffusiongemma_config_processor_contract",
                "pinned_diffusiongemma_weight_shard_readiness_contract",
                "pinned_diffusiongemma_vllm_nvfp4_runtime_readiness_contract",
                "pinned_diffusiongemma_nvfp4_isolated_vllm_generation_probe",
            ]
        )
        lock["expected_metrics"].update(
            {
                "diffusiongemma_config_supported": True,
                "diffusiongemma_processor_supported": True,
                "diffusiongemma_model_class_supported": True,
                "diffusiongemma_model_type": "diffusion_gemma",
                "diffusiongemma_bf16_required_weight_shards": 11,
                "diffusiongemma_nvfp4_required_weight_shards": 2,
                "diffusiongemma_nvfp4_quant_method": "modelopt",
                "diffusiongemma_generation_ready": True,
                "diffusiongemma_released_checkpoint_generation_proven": True,
                "diffusiongemma_bf16_24gb_direct_loading_deferred": True,
                "diffusiongemma_nvfp4_isolated_vllm_generation_ready": True,
                "diffusiongemma_external_vllm_generation_ready": True,
                "diffusiongemma_external_vllm_runtime_isolated": True,
                "diffusiongemma_external_vllm_model_matches_nvfp4_revision": True,
                "diffusiongemma_external_vllm_output_nonempty": True,
                "diffusiongemma_external_vllm_output_mentions_negative_controls": True,
                "diffusiongemma_external_vllm_used_chat_template": True,
                "diffusiongemma_external_vllm_torch_version": "2.11.0+cu130",
                "diffusiongemma_external_vllm_torch_cuda_version": "13.0",
                "diffusiongemma_external_vllm_vllm_version": "0.24.0",
                "diffusiongemma_external_vllm_cuda_available": True,
                "diffusiongemma_vllm_probe_output_nonempty": True,
                "diffusiongemma_vllm_probe_torch_version": "2.11.0+cu130",
                "diffusiongemma_vllm_probe_torch_cuda_version": "13.0",
                "diffusiongemma_vllm_probe_vllm_version": "0.24.0",
            }
        )
        lock["safety_notes"][-1] = (
            "The trained diffusion LM path uses generated integer-token copy-pair data only. "
            "DiffusionGemma readiness checks use public open-weight metadata and exact pinned "
            "revisions. The Google BF16 path is deferred for direct 24GB loading, while the "
            "NVIDIA NVFP4 claim is limited to the isolated vLLM proof artifact and does not "
            "install vLLM into the main torch 2.12/CUDA 13.2 uv environment."
        )
    if record["number"] == "5.6":
        lock["evidence_level"] = (
            "toy_specialist_controls_plus_authenticated_embeddinggemma_and_functiongemma_preflights"
        )
        lock["claim_scope"] = (
            "GT-1 specialist-model control ladder plus authenticated Google "
            "EmbeddingGemma retrieval on generated query/document pairs with "
            "permuted-pair negative controls, a pinned public BGE retrieval "
            "comparison, a pinned public FunctionGemma Mobile Actions checkpoint "
            "evaluated on real held-out Mobile Actions JSONL rows, and direct CUDA "
            "loading of the gated base google/functiongemma-270m-it model on a "
            "benign forward pass."
        )
        lock["required_gpu_gb"] = 2
        lock["models"].append(
            {
                "id": "BAAI/bge-small-en-v1.5",
                "source": "huggingface",
                "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                "precision": "float32",
                "gated": False,
                "claim_scope": "public_embedding_retrieval_preflight",
            }
        )
        lock["models"].append(
            {
                "id": "litert-community/FunctionGemma_270M_Mobile_Actions",
                "source": "huggingface",
                "revision": "e2226c1def35c5443942ebdb90a1da2a9eda836a",
                "precision": "bfloat16",
                "gated": False,
                "claim_scope": "public_functiongemma_mobile_actions_generation_preflight",
                "base_model": "google/functiongemma-270m-it",
            }
        )
        lock["models"].extend(
            [
                {
                    "id": "google/embeddinggemma-300m",
                    "source": "huggingface",
                    "revision": "57c266a740f537b4dc058e1b0cda161fd15afa75",
                    "precision": "float32",
                    "gated": "manual",
                    "local_status": "authenticated_and_loaded",
                    "claim_scope": "authenticated_embeddinggemma_retrieval_preflight",
                },
                {
                    "id": "google/functiongemma-270m-it",
                    "source": "huggingface",
                    "revision": "39eccb091651513a5dfb56892d3714c1b5b8276c",
                    "precision": "float32",
                    "gated": "manual",
                    "local_status": "authenticated_and_loaded",
                    "claim_scope": "direct_base_functiongemma_cuda_forward_preflight",
                },
            ]
        )
        lock["datasets"].extend(
            [
                {
                    "id": "course_generated_specialist_embedding_queries_v1",
                    "source": "generated_query_document_pairs",
                    "seed": 0,
                    "query_count": 4,
                    "document_count": 4,
                    "topics": [
                        "tool_masking",
                        "embedding_retrieval",
                        "vlm_hallucination",
                        "function_call_abstention",
                    ],
                },
                {
                    "id": "google/mobile-actions",
                    "source": "huggingface_dataset",
                    "revision": "e920309bc2acbc2e99a5e3201cf37df2b9fd9151",
                    "split": "eval",
                    "deterministic_eval_prefix_count": 32,
                    "license": "cc-by-4.0",
                },
            ]
        )
        lock["expected_metrics"].update(
            {
                "synthetic_retrieval_top1_accuracy": 1.0,
                "synthetic_tool_masking_passed": True,
                "bge_preflight_passed": True,
                "bge_retrieval_top1_accuracy": 1.0,
                "bge_mean_reciprocal_rank": 1.0,
                "bge_mean_margin_min": 0.1,
                "bge_permuted_top1_accuracy": 0.0,
                "bge_peak_vram_gb_max": 4.0,
                "embeddinggemma_authenticated": True,
                "embeddinggemma_gated_unavailable": False,
                "embeddinggemma_preflight_passed": True,
                "embeddinggemma_ready_for_direct_loading": True,
                "embeddinggemma_retrieval_top1_accuracy": 1.0,
                "embeddinggemma_mean_reciprocal_rank": 1.0,
                "embeddinggemma_mean_margin_min": 0.1,
                "embeddinggemma_permuted_top1_accuracy": 0.0,
                "embeddinggemma_permuted_control_fails": True,
                "embeddinggemma_local_non_ref_file_count_min": 1,
                "embeddinggemma_peak_vram_gb_max": 4.0,
                "functiongemma_base_authenticated": True,
                "functiongemma_base_preflight_passed": True,
                "functiongemma_base_ready_for_direct_loading": True,
                "functiongemma_base_forward_passed": True,
                "functiongemma_base_local_non_ref_file_count_min": 1,
                "functiongemma_base_peak_vram_gb_max": 2.0,
                "functiongemma_preflight_passed": True,
                "functiongemma_eval_example_count": 32,
                "functiongemma_parse_accuracy": 1.0,
                "functiongemma_function_name_accuracy": 1.0,
                "functiongemma_exact_argument_accuracy_min": 0.85,
                "functiongemma_required_argument_accuracy_min": 0.85,
                "functiongemma_failure_count_max": 4,
                "functiongemma_peak_vram_gb_max": 2.0,
            }
        )
        lock["controls"].extend(
            [
                "masked_mean_pooling_contract",
                "paired_embedding_retrieval_contract",
                "centroid_probe_contract",
                "tool_masking_contract",
                "no_call_hallucination_metric",
                "schema_token_attribution_contract",
                "pinned_public_bge_embedding_retrieval",
                "permuted_query_document_pair_control",
                "authenticated_embeddinggemma_retrieval",
                "embeddinggemma_permuted_query_document_pair_control",
                "pinned_public_functiongemma_mobile_actions_eval",
                "structured_function_name_and_argument_parsing",
                "held_out_mobile_actions_argument_accuracy",
                "authenticated_functiongemma_base_cuda_forward",
            ]
        )
        lock["safety_notes"][-1] = (
            "The function-calling preflight uses public Mobile Actions eval rows and "
            "benign mobile-tool schemas only. EmbeddingGemma retrieval and base "
            "FunctionGemma loading use benign course-generated text/query examples "
            "and aggregate metrics."
        )
    if record["number"] == "5.3":
        lock["evidence_level"] = "selective_scan_contract_plus_official_mamba_logits_generation_preflight"
        lock["claim_scope"] = (
            "GT-1 Mamba implementation contract: recurrent, parallel, and chunked "
            "selective scans and tiny-cache parity are checked on generated tensors, "
            "then a pinned Mamba-130M-HF checkpoint is loaded on CUDA to verify finite "
            "logits, batched-vs-single top-token consistency, deterministic generation, "
            "and required Transformers Mamba fast kernels. This does not claim full "
            "weight-level parity between the tiny implementation and the released 130M "
            "checkpoint."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "state-spaces/mamba-130m-hf",
                "source": "huggingface",
                "revision": "1e76775f628fbf1350fbe4dbb3d971ba64af25a1",
                "precision": "float16",
                "gated": False,
                "claim_scope": "official_logits_generation_preflight",
            }
        )
        lock["datasets"].append(
            {
                "id": "course_generated_safe_mamba_prompts_v1",
                "source": "generated_prompt_pair",
                "seed": 0,
                "prompt_count": 2,
                "prompts": ["Mamba models can", "The cat sat on"],
                "generation_prompt": "Mamba models can",
                "max_new_tokens": 4,
            }
        )
        lock["expected_metrics"].update(
            {
                "official_mamba_logits_generation_preflight_passed": True,
                "official_mamba_batched_single_top1_agreement": 1.0,
                "official_mamba_generation_new_tokens": 4,
                "official_mamba_generation_tokens_per_second_min": 1.0,
                "official_mamba_prompt_prefix_preserved": True,
                "official_mamba_logits_std_min": 1.0,
                "official_mamba_logits_shape": [2, 4, 50280],
                "official_mamba_generated_shape": [1, 8],
                "official_mamba_fast_kernel_available": True,
                "official_mamba_peak_vram_gb_max": 2.0,
            }
        )
        lock["controls"].extend(
            [
                "recurrent_parallel_selective_scan_equivalence",
                "chunked_scan_state_carry_equivalence",
                "tiny_mamba_cache_parity",
                "pinned_official_mamba_130m_hf_logits_generation",
                "batched_vs_single_prompt_top_token_consistency",
                "deterministic_generation_prefix_preservation",
                "mamba_ssm_and_causal_conv1d_fast_kernels_required",
                "full_weight_level_tiny_to_130m_parity_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The official Mamba path uses safe generic prompts and deterministic "
            "generation only. Full weight-level parity between the didactic tiny "
            "implementation and Mamba-130M-HF is not claimed."
        )
    if record["number"] == "5.4":
        lock["evidence_level"] = (
            "toy_state_tracking_plus_trained_mamba_transformer_comparison_"
            "learned_intervention_and_official_hidden_state_preflight"
        )
        lock["claim_scope"] = (
            "GT-1 synthetic state-tracking controls plus a trained tiny Mamba bracket-depth "
            "organism with held-out longer-sequence generalization and random-label control, "
            "a trained tiny causal Transformer baseline on the same generated task, learned "
            "Mamba hidden-state interventions with matched random-direction controls, and a "
            "pinned official Mamba-130M-HF hidden-state extraction preflight."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "course_tiny_mamba_state_classifier_v1",
                    "source": "generated_training_run",
                    "revision": "seed_0_steps_160",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "trained_bracket_depth_state_tracking_organism",
                },
                {
                    "id": "course_tiny_causal_transformer_state_classifier_v1",
                    "source": "generated_training_run",
                    "revision": "seed_2_steps_160",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "trained_transformer_bracket_depth_baseline",
                },
                {
                    "id": "state-spaces/mamba-130m-hf",
                    "source": "huggingface",
                    "revision": "1e76775f628fbf1350fbe4dbb3d971ba64af25a1",
                    "precision": "float16",
                    "gated": False,
                    "claim_scope": "official_hidden_state_extraction_preflight",
                },
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_generated_bracket_depth_tracking_v1",
                "source": "generated",
                "seed": 0,
                "train_seq_len": 16,
                "heldout_seq_len": 32,
                "max_depth": 3,
                "random_label_control": True,
            }
        )
        lock["expected_metrics"].update(
            {
                "tiny_mamba_preflight_passed": True,
                "tiny_mamba_short_accuracy_min": 0.9,
                "tiny_mamba_long_accuracy_min": 0.85,
                "tiny_mamba_long_late_accuracy_min": 0.8,
                "tiny_mamba_random_label_long_accuracy_max": 0.55,
                "tiny_transformer_short_accuracy_min": 0.95,
                "tiny_transformer_long_accuracy_max": 0.75,
                "tiny_transformer_long_late_accuracy_max": 0.4,
                "tiny_transformer_comparison_passed": True,
                "tiny_mamba_minus_transformer_long_accuracy_min": 0.2,
                "tiny_mamba_minus_transformer_long_late_accuracy_min": 0.4,
                "learned_state_intervention_passed": True,
                "learned_state_intervention_success_rate_min": 0.9,
                "learned_state_intervention_random_target_rate_max": 0.5,
                "learned_state_intervention_target_logit_delta_min": 1.0,
                "learned_hidden_probe_test_accuracy_min": 0.8,
                "official_mamba_preflight_passed": True,
                "official_mamba_hidden_std_min": 0.1,
                "official_mamba_hidden_shape": [4, 11, 768],
                "official_mamba_fast_kernel_available": True,
            }
        )
        lock["controls"].extend(
            [
                "parity_cumulative_xor_contract",
                "bracket_depth_bounded_state_contract",
                "heldout_position_probe_contract",
                "probe_direction_intervention_contract",
                "trained_tiny_mamba_bracket_depth",
                "heldout_longer_sequence_generalization",
                "random_label_training_control",
                "trained_tiny_transformer_bracket_depth_baseline",
                "mamba_vs_transformer_long_sequence_comparison",
                "learned_mamba_hidden_state_intervention",
                "matched_norm_random_hidden_state_intervention_control",
                "pinned_official_mamba_130m_hf_hidden_state_extraction",
                "mamba_ssm_and_causal_conv1d_fast_kernels_required",
            ]
        )
        lock["safety_notes"][-1] = (
            "The trained Mamba path uses generated bracket-depth data only. Official "
            "Mamba-130M-HF is used for hidden-state extraction, not as a solved "
            "state-tracking model. The Transformer comparison and learned-state "
            "intervention are local generated-task organism checks."
        )
    if record["number"] == "8.5":
        lock["evidence_level"] = (
            "toy_ladder_plus_real_pythia_official_sae_sfc_faithfulness_and_shift_editing"
        )
        lock["claim_scope"] = (
            "GT-0 toy sparse-feature contract plus Pythia-70M-deduped residual-feature "
            "subject/verb preflight, official artifact readiness, one released SAE "
            "state-dict check, one-layer SAE feature-attribution with random-feature "
            "control, a 100-example official-code sparse feature graph artifact, and "
            "held-out official faithfulness on 40 simple_test examples, plus a safe "
            "generated-data SHIFT-style editing organism that suppresses a spurious "
            "feature, preserves target accuracy, improves OOD accuracy, and beats a "
            "same-size random feature edit."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "EleutherAI/pythia-70m-deduped",
                "source": "huggingface",
                "revision": "default_hf_revision",
                "precision": "float32",
                "gated": False,
                "claim_scope": "residual_feature_preflight_only",
            }
        )
        lock["datasets"].append(
            {
                "id": "subject_verb_minimal_pair_cats_cat_are_is",
                "source": "course_generated_prompt_pair",
                "seed": 0,
                "clean_prompt": "The cats",
                "corrupt_prompt": "The cat",
                "target_token": " are",
                "distractor_token": " is",
            }
        )
        lock["datasets"].append(
            {
                "id": "course_generated_shift_sparse_feature_editing_v1",
                "source": "generated",
                "seed": 0,
                "target_feature": "signed task label feature",
                "spurious_feature": "train-correlated anti-correlated OOD feature",
                "splits": "train and OOD",
            }
        )
        lock["expected_metrics"].update(
            {
                "real_model_preflight_passed": True,
                "real_model_linearization_error_max": 1e-3,
                "real_model_random_control_fails": True,
                "official_artifact_remote_manifest_passed": True,
                "official_artifact_expected_dictionary_count": 19,
                "official_artifact_ready_for_gt2": True,
                "official_sae_state_dict_smoke_passed": True,
                "official_sae_feature_attribution_passed": True,
                "official_sparse_feature_circuit_replication_passed": True,
                "official_sparse_feature_circuit_examples": 100,
                "official_sparse_feature_circuit_faithfulness_passed": True,
                "official_sparse_feature_circuit_faithfulness_min": 0.9,
                "shift_editing_passed": True,
                "shift_editing_ood_improvement_min": 0.5,
                "shift_editing_target_accuracy_drop_max": 0.05,
                "shift_editing_random_edit_control_fails": True,
            }
        )
        lock["controls"].extend(
            [
                "pythia_subject_verb_residual_preflight",
                "same_size_random_residual_feature_control",
                "official_feature_circuits_repo_manifest_check",
                "official_pythia_sae_dictionary_manifest_check",
                "official_resid5_sae_state_dict_shape_and_finiteness_smoke",
                "one_layer_official_resid5_sae_feature_attribution_random_control",
                "official_code_100_example_sparse_feature_graph_artifact",
                "official_heldout_simple_test_faithfulness_evaluation",
                "generated_shift_style_spurious_feature_editing",
                "black_box_unedited_classifier_baseline",
                "same_size_random_feature_edit_control",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real Sparse Feature Circuits path uses the pinned official repo, "
            "downloaded Pythia-70M-deduped SAE dictionaries, generated safe "
            "subject-verb data from the official split, and held-out faithfulness. "
            "The SHIFT-style editing exercise uses generated sparse features only; "
            "it does not use Bias-in-Bios or sensitive demographic text."
        )
    if record["number"] == "9.1":
        lock["evidence_level"] = (
            "safe_toy_controls_plus_broad_template_real_lm_and_instruction_preflights"
        )
        lock["claim_scope"] = (
            "GT-3 safe refusal-direction control ladder plus a pinned Pythia-70M-deduped "
            "hidden-state category preflight across three sanitized prompt-template "
            "families and a pinned Qwen2.5-0.5B-Instruct instruction-model "
            "addition/projection-out preflight on the same broad safe prompt set. "
            "Both real-model paths avoid generated completions and use aggregate "
            "hidden-state or next-token logit-score metrics; SAE-feature comparisons "
            "remain outside this local preflight."
        )
        lock["required_gpu_gb"] = 2
        lock["models"].extend(
            [
                {
                    "id": "EleutherAI/pythia-70m-deduped",
                    "source": "huggingface",
                    "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "safe_category_hidden_state_preflight",
                },
                {
                    "id": "Qwen/Qwen2.5-0.5B-Instruct",
                    "source": "huggingface",
                    "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                    "precision": "bfloat16",
                    "gated": False,
                    "claim_scope": "safe_instruction_model_refusal_direction_addition_projection_preflight",
                },
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_generated_safe_category_meta_prompts_v1",
                "source": "generated",
                "seed": 0,
                "prompt_count": 24,
                "prompt_template_family_count": 3,
                "train_prompt_count": 8,
                "heldout_prompt_count": 16,
                "prompt_safety": "sanitized_meta_prompts_no_procedural_content",
                "intervention_prompt_count": 24,
                "generation_used": False,
            }
        )
        lock["expected_metrics"].update(
            {
                "label_shuffle_fails": True,
                "real_lm_category_preflight_passed": True,
                "real_lm_category_total_prompt_count": 24,
                "real_lm_category_prompt_template_family_count": 3,
                "real_lm_category_heldout_accuracy_min": 0.9,
                "real_lm_category_min_template_accuracy_min": 0.85,
                "real_lm_category_min_template_margin_min": 1.0,
                "real_lm_category_label_shuffle_fails": True,
                "real_lm_category_random_direction_fails": True,
                "real_lm_category_generation_used": False,
                "instruction_refusal_intervention_preflight_passed": True,
                "instruction_refusal_prompt_count": 24,
                "instruction_refusal_prompt_template_family_count": 3,
                "instruction_refusal_generation_used": False,
                "instruction_refusal_baseline_margin_min": 2.0,
                "instruction_refusal_allowed_add_delta_min": 1.0,
                "instruction_refusal_projection_delta_max": -1.0,
                "instruction_refusal_target_beats_random_addition": True,
                "instruction_refusal_target_beats_random_projection": True,
                "instruction_refusal_peak_vram_gb_max": 4.0,
            }
        )
        lock["controls"].extend(
            [
                "safe_category_train_heldout_split",
                "three_sanitized_prompt_template_families",
                "heldout_prompt_template_family_control",
                "sanitized_meta_prompt_policy",
                "label_shuffle_control",
                "fixed_seed_random_direction_control",
                "no_generation_hidden_state_only",
                "pythia_70m_deduped_revision_pinned",
                "pinned_qwen25_0_5b_instruction_model",
                "safe_instruction_refusal_direction_addition",
                "safe_instruction_refusal_direction_projection_out",
                "fixed_seed_random_direction_addition_projection_control",
                "no_completion_generation_token_logit_scores_only",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model refusal paths use sanitized category descriptions, hidden "
            "states or next-token logit scores, fixed random-direction controls, and "
            "aggregate metrics without harmful procedural completions. SAE-feature "
            "comparisons are not claimed by this local preflight."
        )
    if record["number"] == "9.2":
        lock["evidence_level"] = "toy_cot_contract_plus_pinned_pythia_hidden_answer_preflight"
        lock["claim_scope"] = (
            "GT-3 chain-of-thought faithfulness preflight. The toy probe, patching, "
            "text-baseline, feature-detector, and condition-comparison contracts are "
            "preserved, and the graded CUDA path loads pinned Pythia-70M-deduped on "
            "safe A/B private-answer prompts. It trains a thresholded hidden-answer "
            "direction on 24 prompts, evaluates 8 held-out prompts, includes a "
            "label-shuffled probe control, patches hidden state through embed_out, "
            "and reports condition-level A/B logit accuracies. This is not a broad "
            "chain-of-thought faithfulness benchmark or reasoning-model claim."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "EleutherAI/pythia-70m-deduped",
                "source": "huggingface",
                "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
                "precision": "float32",
                "gated": False,
                "claim_scope": "hidden_answer_cot_faithfulness_preflight",
                "answer_tokens": [" A", " B"],
            }
        )
        lock["datasets"].append(
            {
                "id": "course_safe_private_answer_cot_prompts_v1",
                "source": "generated_prompt_sets",
                "seed": 0,
                "train_prompt_count": 24,
                "heldout_prompt_count": 8,
                "train_nouns": ["private option", "secret choice", "hidden label"],
                "heldout_nouns": ["internal answer"],
                "conditions": ["no_cot", "faithful_cot", "biased_cot", "posthoc"],
                "answer_classes": ["A", "B"],
                "unfaithful_case_count": 4,
                "generation_used": False,
                "target_source": "manual_safe_private_answer_labels_plus_real_hidden_state_projection",
                "hidden_layer": -1,
            }
        )
        lock["controls"].extend(
            [
                "toy_pre_final_answer_probe_contract",
                "toy_hidden_answer_patching_contract",
                "toy_cot_text_baseline_contract",
                "toy_feature_detector_contract",
                "toy_condition_comparison_contract",
                "pinned_pythia70m_hidden_answer_direction_probe",
                "heldout_private_answer_template_split",
                "visible_rationale_text_only_baseline",
                "label_shuffled_hidden_direction_negative_control",
                "lm_head_hidden_state_patch_control",
                "no_completion_generation_hidden_state_and_logits_only",
                "broad_cot_faithfulness_benchmark_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model CoT-faithfulness path uses safe synthetic A/B prompts, "
            "hidden states, and answer-token logits only. It performs no generated "
            "completions, contains no harmful content, and does not claim results on "
            "arbitrary reasoning traces."
        )
    if record["number"] == "9.3":
        lock["evidence_level"] = "toy_proxy_drift_contract_plus_pinned_pythia_hidden_state_preflight"
        lock["claim_scope"] = (
            "GT-3 emergent-misalignment detection preflight using benign proxy drifts "
            "only. The toy detector, crosscoder-alignment, mitigation, and early-warning "
            "contracts are preserved, and the graded CUDA path loads pinned "
            "Pythia-70M-deduped on safe generated prompt pairs. It trains a thresholded "
            "hidden-state drift direction on 36 prompts, evaluates 24 held-out prompts "
            "across five benign proxy kinds, compares label-shuffled and fixed random "
            "directions, aligns feature scores with a safe next-token behavior proxy, "
            "and tests projection-style mitigation through the LM head. This is not a "
            "harmful finetune, not an emergent-misalignment reproduction, and not a "
            "claim about arbitrary unsafe behavior."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "EleutherAI/pythia-70m-deduped",
                "source": "huggingface",
                "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
                "precision": "float32",
                "gated": False,
                "claim_scope": "benign_proxy_drift_hidden_state_preflight",
                "behavior_proxy_tokens": [" helpful", " unsafe"],
            }
        )
        lock["datasets"].append(
            {
                "id": "course_safe_proxy_drift_prompts_v1",
                "source": "generated_prompt_sets",
                "seed": 0,
                "train_prompt_count": 36,
                "heldout_prompt_count": 24,
                "train_context_count": 6,
                "heldout_context_count": 4,
                "neutral_policy_count": 1,
                "drift_kind_count": 5,
                "drift_kinds": [
                    "sycophantic",
                    "overconfident",
                    "json_only",
                    "style_drift",
                    "refusal_overgeneralizing",
                ],
                "generation_used": False,
                "target_source": "manual_safe_proxy_drift_labels_plus_real_hidden_state_projection",
                "hidden_layer": -1,
            }
        )
        lock["controls"].extend(
            [
                "toy_safe_proxy_kind_contract",
                "toy_heldout_drift_detector_contract",
                "toy_crosscoder_alignment_contract",
                "toy_mitigation_contract",
                "toy_early_warning_contract",
                "pinned_pythia70m_proxy_drift_direction_probe",
                "heldout_context_split",
                "label_shuffled_drift_direction_negative_control",
                "fixed_seed_random_direction_negative_control",
                "safe_next_token_behavior_proxy_alignment",
                "lm_head_projection_mitigation_control",
                "no_completion_generation_hidden_state_and_logits_only",
                "harmful_finetune_not_claimed",
                "emergent_misalignment_reproduction_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model proxy-drift path uses safe generated behavior-policy "
            "descriptions, hidden states, and next-token logits only. It performs no "
            "generated completions, trains no adapters, includes no harmful procedural "
            "content, and does not claim to reproduce emergent misalignment."
        )
    if record["number"] == "9.4":
        lock["evidence_level"] = "toy_monitor_contract_plus_pinned_pythia_white_box_monitor_preflight"
        lock["claim_scope"] = (
            "GT-3 white-box monitor preflight using safe generated eval records only. "
            "The toy dashboard, AUROC calibration, missed-failure, false-positive, and "
            "explanation-validation contracts are preserved, and the graded CUDA path "
            "loads pinned Pythia-70M-deduped on neutral vs safe failure-policy prompts. "
            "It trains a thresholded hidden-state monitor on 36 prompts, evaluates 24 "
            "held-out prompts across five benign failure kinds, compares against a real "
            "next-token pass/fail black-box proxy, checks label-shuffled and fixed "
            "random directions, and validates false-positive documentation. This is "
            "not a harmful-content monitor benchmark, not a broad safety monitor claim, "
            "and not a generated-completion evaluation."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "EleutherAI/pythia-70m-deduped",
                "source": "huggingface",
                "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
                "precision": "float32",
                "gated": False,
                "claim_scope": "safe_white_box_monitor_hidden_state_preflight",
                "black_box_proxy_tokens": [" pass", " fail"],
            }
        )
        lock["datasets"].append(
            {
                "id": "course_safe_white_box_monitor_eval_records_v1",
                "source": "generated_prompt_sets",
                "seed": 0,
                "train_prompt_count": 36,
                "heldout_prompt_count": 24,
                "train_context_count": 6,
                "heldout_context_count": 4,
                "neutral_policy_count": 1,
                "failure_kind_count": 5,
                "failure_kinds": [
                    "unsupported_agreement",
                    "overconfidence",
                    "format_drift",
                    "style_drift",
                    "over_refusal",
                ],
                "generation_used": False,
                "target_source": "manual_safe_failure_labels_plus_real_hidden_state_monitor",
                "hidden_layer": -1,
            }
        )
        lock["controls"].extend(
            [
                "toy_dashboard_row_contract",
                "toy_monitor_calibration_contract",
                "toy_missed_failure_contract",
                "toy_false_positive_documentation_contract",
                "toy_feature_explanation_validation_contract",
                "pinned_pythia70m_white_box_monitor_direction",
                "heldout_context_split",
                "real_next_token_pass_fail_black_box_proxy",
                "label_shuffled_monitor_negative_control",
                "fixed_seed_random_direction_monitor_negative_control",
                "false_positive_documentation_control",
                "no_completion_generation_hidden_state_and_logits_only",
                "broad_safety_monitor_benchmark_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model monitor path uses safe generated eval-record descriptions, "
            "hidden states, and next-token logits only. It performs no generated "
            "completions, includes no harmful procedural content, and does not claim a "
            "broad harmful-content or deployment monitor benchmark."
        )
    if record["number"] == "11.1":
        lock["evidence_level"] = (
            "gt0_toy_geometry_plus_real_lm_calendar_template_centering_preflight"
        )
        lock["claim_scope"] = (
            "GT-0 toy PCA/SVD, held-out prediction, white-noise, stability, and causal "
            "direction contracts plus a pinned Pythia-70M-deduped hidden-state calendar "
            "geometry preflight over weekdays and months. The real-LM path shows raw "
            "template-dominated transfer failures and template-centered calendar-label "
            "transfer with permuted-label, matched-pair, and white-noise controls."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "EleutherAI/pythia-70m-deduped",
                "source": "huggingface",
                "revision": "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
                "precision": "float32",
                "gated": False,
                "claim_scope": "weekday_month_template_centered_hidden_state_geometry_preflight",
            }
        )
        lock["datasets"].extend(
            [
                {
                    "id": "course_generated_weekday_prompt_templates_v1",
                    "source": "generated",
                    "seed": 0,
                    "label_count": 7,
                    "train_template": "Today is {}",
                    "heldout_template": "The calendar says {}",
                },
                {
                    "id": "course_generated_month_prompt_templates_v1",
                    "source": "generated",
                    "seed": 1,
                    "label_count": 12,
                    "train_template": "The report was written in {}",
                    "heldout_template": "The event happened during {}",
                },
            ]
        )
        lock["expected_metrics"].update(
            {
                "template_centering_max_mean_abs": 0.0,
                "pythia_weekday_preflight_passed": True,
                "pythia_calendar_preflight_passed": True,
                "pythia_calendar_task_count": 2,
                "pythia_weekday_raw_heldout_accuracy_max": 0.2,
                "pythia_weekday_centered_heldout_accuracy": 1.0,
                "pythia_weekday_permuted_label_accuracy": 0.0,
                "pythia_weekday_noise_accuracy_max": 0.2,
                "pythia_month_raw_heldout_accuracy_max": 0.75,
                "pythia_month_centered_heldout_accuracy": 1.0,
                "pythia_month_permuted_label_accuracy": 0.0,
                "pythia_month_noise_accuracy_max": 0.1,
                "pythia_month_matched_pair_accuracy": 1.0,
                "pythia_weekday_generation_used": False,
            }
        )
        lock["controls"].extend(
            [
                "template_centering_control",
                "raw_template_dominated_failure_check",
                "weekday_train_heldout_prompt_template_split",
                "month_train_heldout_prompt_template_split",
                "permuted_label_control",
                "white_noise_geometry_control",
                "matched_pair_similarity_check",
                "no_generation_hidden_state_only",
                "pythia_70m_deduped_revision_pinned",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-LM geometry path uses generated weekday/month labels and hidden "
            "states only. It performs no generated completions and does not claim causal "
            "space/time intervention behavior."
        )
    if record["number"] == "12.1":
        lock["evidence_level"] = (
            "synthetic_vlm_ladder_plus_real_clip_siglip_hidden_token_patching_and_qwen25_vl_generation"
        )
        lock["claim_scope"] = (
            "GT-1 pinned CLIP and SigLIP rendered-shape retrieval plus real "
            "object-region counterfactual patching, real hidden visual-token "
            "activation patching at the vision patch-embedding output with "
            "object/background/same-size random-token/full-sequence controls, "
            "a pinned Qwen2.5-VL 3B rendered-shape generation check, and a "
            "controlled synthetic VLM object/color/clothing-style ladder with "
            "counterfactual labels, joint/image-only baselines over text-only "
            "priors, and object-region patching against background and same-size "
            "non-overlapping random controls."
        )
        lock["required_gpu_gb"] = 8
        lock["models"].extend(
            [
                {
                    "id": "openai/clip-vit-base-patch32",
                    "source": "huggingface",
                    "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": (
                        "rendered_shape_retrieval_region_patching_and_hidden_visual_token_activation_patching"
                    ),
                },
                {
                    "id": "google/siglip-base-patch16-224",
                    "source": "huggingface",
                    "revision": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
                    "precision": "float16",
                    "gated": False,
                    "claim_scope": (
                        "rendered_shape_retrieval_region_patching_and_hidden_visual_token_activation_patching"
                    ),
                },
                {
                    "id": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "source": "huggingface",
                    "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
                    "precision": "bfloat16",
                    "gated": False,
                    "claim_scope": "rendered_shape_generation_preflight",
                },
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_generated_colored_shapes_v1",
                "source": "generated",
                "seed": 0,
                "schema": "object_color_bbox_question_answer_counterfactual",
            }
        )
        lock["datasets"].append(
            {
                "id": "rendered_red_square_blue_circle_clip_preflight_v1",
                "source": "generated_pil_shapes",
                "seed": 0,
                "image_count": 2,
                "text_count": 2,
            }
        )
        lock["expected_metrics"].update(
            {
                "synthetic_scene_count": 4,
                "joint_beats_text_only": True,
                "text_only_fails_image_questions": True,
                "object_beats_background": True,
                "object_beats_random": True,
                "object_patch_flips_answer": True,
                "visual_sequence_patch_passed": True,
                "visual_sequence_patch_object_flips_answer": True,
                "visual_sequence_patch_full_sequence_matches_corrupt": True,
                "real_clip_rendered_shape_preflight_passed": True,
                "real_clip_image_to_text_accuracy": 1.0,
                "real_clip_text_to_image_accuracy": 1.0,
                "real_clip_mean_positive_margin_min": 2.0,
                "real_clip_object_patch_flips_answer": True,
                "real_clip_background_patch_preserves_answer": True,
                "real_clip_object_beats_background": True,
                "real_clip_object_beats_random": True,
                "real_clip_random_patch_same_size_as_object": True,
                "real_clip_random_patch_overlap_area": 0,
                "real_clip_random_patch_overlaps_object": False,
                "real_clip_visual_token_activation_patching_preflight_passed": True,
                "real_clip_activation_patch_object_flips_answer": True,
                "real_clip_activation_patch_background_preserves_answer": True,
                "real_clip_activation_patch_random_preserves_answer": True,
                "real_clip_activation_patch_full_sequence_matches_corrupt": True,
                "real_clip_activation_patch_full_sequence_flips_answer": True,
                "real_clip_activation_patch_min_object_gap_over_background_min": 4.0,
                "real_clip_activation_patch_min_object_gap_over_random_min": 4.0,
                "real_clip_activation_patch_full_sequence_max_abs_margin_error_max": 0.001,
                "real_clip_activation_patch_object_token_count": 9,
                "real_clip_activation_patch_random_token_count": 9,
                "real_clip_activation_patch_random_control_same_size": True,
                "real_clip_activation_patch_random_control_overlaps_object": False,
                "real_clip_activation_patch_hook_point": "vision_model.embeddings",
                "real_siglip_rendered_shape_preflight_passed": True,
                "real_siglip_image_to_text_accuracy": 1.0,
                "real_siglip_text_to_image_accuracy": 1.0,
                "real_siglip_mean_positive_margin_min": 0.5,
                "real_siglip_object_patch_flips_answer": True,
                "real_siglip_background_patch_preserves_answer": True,
                "real_siglip_object_beats_background": True,
                "real_siglip_object_beats_random": True,
                "real_siglip_random_patch_same_size_as_object": True,
                "real_siglip_random_patch_overlap_area": 0,
                "real_siglip_random_patch_overlaps_object": False,
                "real_siglip_visual_token_activation_patching_preflight_passed": True,
                "real_siglip_activation_patch_object_flips_answer": True,
                "real_siglip_activation_patch_background_preserves_answer": True,
                "real_siglip_activation_patch_random_preserves_answer": True,
                "real_siglip_activation_patch_full_sequence_matches_corrupt": True,
                "real_siglip_activation_patch_full_sequence_flips_answer": True,
                "real_siglip_activation_patch_min_object_gap_over_background_min": 10.0,
                "real_siglip_activation_patch_min_object_gap_over_random_min": 10.0,
                "real_siglip_activation_patch_full_sequence_max_abs_margin_error_max": 0.001,
                "real_siglip_activation_patch_object_token_count": 36,
                "real_siglip_activation_patch_random_token_count": 36,
                "real_siglip_activation_patch_random_control_same_size": True,
                "real_siglip_activation_patch_random_control_overlaps_object": False,
                "real_siglip_activation_patch_hook_point": "vision_model.embeddings",
                "real_qwen25_vl_generation_preflight_passed": True,
                "real_qwen25_vl_accuracy": 1.0,
                "real_qwen25_vl_expected_answers": ["red square", "blue circle"],
                "real_qwen25_vl_peak_vram_gb_max": 12.0,
            }
        )
        lock["controls"].extend(
            [
                "synthetic_colored_shape_counterfactual_schema",
                "text_only_prior_failure_baseline",
                "image_only_grounding_baseline",
                "object_region_patch",
                "background_region_patch_control",
                "same_size_random_region_patch_control",
                "non_overlapping_random_region_patch_control",
                "pinned_real_clip_rendered_shape_retrieval",
                "pinned_real_clip_object_region_counterfactual_patch",
                "pinned_real_clip_hidden_visual_token_activation_patch",
                "real_clip_hidden_background_token_activation_patch_control",
                "real_clip_hidden_same_size_random_token_activation_patch_control",
                "real_clip_full_visual_sequence_activation_patch_control",
                "real_clip_local_snapshot_load_no_remote_conversion_thread",
                "pinned_real_siglip_safetensors_rendered_shape_retrieval",
                "pinned_real_siglip_object_region_counterfactual_patch",
                "pinned_real_siglip_hidden_visual_token_activation_patch",
                "real_siglip_hidden_background_token_activation_patch_control",
                "real_siglip_hidden_same_size_random_token_activation_patch_control",
                "real_siglip_full_visual_sequence_activation_patch_control",
                "pinned_real_qwen25_vl_rendered_shape_generation",
                "qwen25_vl_red_square_blue_circle_counterfactual_control",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real generative VLM path uses safe generated shape images only. "
            "The CLIP and SigLIP region-patching path uses deterministic rendered "
            "shape images with object, background, and same-size non-overlapping "
            "random-region patch controls. The hidden-token path patches real "
            "CLIP/SigLIP visual-token activations at `vision_model.embeddings` "
            "and requires object tokens to flip the contrastive answer while "
            "background, same-size random-token, and full visual-sequence controls "
            "behave as expected. The Qwen2.5-VL path verifies grounded generation "
            "on the same safe rendered objects."
        )
    if record["number"] == "13.1":
        lock["evidence_level"] = (
            "toy_prompt_region_controls_plus_pinned_sd15_daam_token_ablation_quality_preflight"
        )
        lock["claim_scope"] = (
            "GT-1 toy diffusion/image-generation interpretability controls plus a "
            "required pinned safetensors Stable Diffusion 1.5 safe-shape preflight. "
            "The SD1.5 path captures DAAM-style cross-attention maps for color/shape "
            "tokens, verifies target-token attention over generated color regions "
            "against unrelated-token controls, ablates target prompt tokens over "
            "random/control-token ablations, preserves simple image-quality metrics, "
            "rejects white-noise controls, and scores generated images with pinned "
            "CLIP. A pinned SD-Turbo/CLIP attention preflight remains supplemental."
        )
        lock["required_gpu_gb"] = 8
        lock["models"].extend(
            [
                {
                    "id": "stabilityai/sd-turbo",
                    "source": "huggingface",
                    "revision": "b261bac6fd2cf515557d5d0707481eafa0485ec2",
                    "precision": "float16",
                    "gated": False,
                    "claim_scope": "safe_shape_generation_and_cross_attention_localization_preflight",
                },
                {
                    "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "source": "huggingface",
                    "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
                    "precision": "float16",
                    "gated": False,
                    "claim_scope": "sd15_safe_shape_daam_token_ablation_quality_and_noise_controls",
                },
                {
                    "id": "openai/clip-vit-base-patch32",
                    "source": "huggingface",
                    "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "generated_image_prompt_alignment_scoring",
                },
            ]
        )
        lock["datasets"].extend(
            [
                {
                    "id": "course_generated_safe_shape_prompts_sd_turbo_v1",
                    "source": "generated_prompt_pair",
                    "seed": 0,
                    "prompt_count": 2,
                    "image_size": "512x512",
                    "prompts": [
                        "red square icon on clean white background",
                        "blue circle icon on clean white background",
                    ],
                },
                {
                    "id": "course_generated_safe_shape_prompts_sd15_v1",
                    "source": "generated_prompt_pair",
                    "seed": "red_square=4;blue_circle=2",
                    "prompt_count": 2,
                    "image_size": "512x512",
                    "prompts": [
                        "single centered solid red square on plain white background",
                        "single centered solid blue circle on plain white background",
                    ],
                },
            ]
        )
        lock["expected_metrics"].update(
            {
                "region_selective": True,
                "latent_direction_effect": True,
                "sd_turbo_preflight_passed": True,
                "sd_turbo_image_to_text_accuracy": 1.0,
                "sd_turbo_text_to_image_accuracy": 1.0,
                "sd_turbo_mean_positive_margin_min": 2.0,
                "sd_turbo_cross_attention_localized": True,
                "sd_turbo_attention_resolutions": [8, 16, 32, 64],
                "sd_turbo_min_target_control_attention_gap_min": 0.02,
                "sd_turbo_mean_target_control_attention_gap_min": 0.03,
                "sd_turbo_min_target_lift_over_mask_fraction_min": 0.02,
                "sd_turbo_min_captured_cross_attention_map_count_min": 16,
                "sd_turbo_peak_vram_gb_max": 8.0,
                "sd15_strict_experiment_passed": True,
                "sd15_model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "sd15_revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
                "sd15_fixed_seed_generation_passed": True,
                "sd15_daam_baseline_included": True,
                "sd15_cross_attention_maps_captured": True,
                "sd15_token_ablation_passed": True,
                "sd15_random_token_ablation_weaker": True,
                "sd15_image_quality_preserved": True,
                "sd15_white_noise_rejected": True,
                "sd15_min_target_control_attention_gap_min": 0.005,
                "sd15_mean_target_control_attention_gap_min": 0.01,
                "sd15_min_target_lift_over_mask_fraction_min": 0.01,
                "sd15_min_captured_cross_attention_map_count_min": 32,
                "sd15_min_target_ablation_drop_min": 0.05,
                "sd15_max_random_control_drop_max": 0.0,
                "sd15_min_target_region_fraction_min": 0.02,
                "sd15_max_high_frequency_energy_max": 0.12,
                "sd15_min_white_noise_high_frequency_gap_min": 0.12,
                "sd15_clip_image_to_text_accuracy": 1.0,
                "sd15_clip_text_to_image_accuracy": 1.0,
                "sd15_clip_mean_positive_margin_min": 2.0,
                "sd15_peak_vram_gb_max": 8.0,
            }
        )
        lock["controls"].extend(
            [
                "attention_region_mass_control",
                "denoising_ablation_random_control",
                "latent_direction_random_control",
                "prompt_token_ablation_control",
                "pinned_sd_turbo_safetensors_generation",
                "deterministic_safe_shape_prompt_pair",
                "pinned_clip_prompt_alignment_scoring",
                "pinned_sd_turbo_cross_attention_capture",
                "target_token_vs_background_token_attention_control",
                "generated_color_shape_region_mask_control",
                "pinned_sd15_safetensors_generation",
                "sd15_daam_cross_attention_capture",
                "sd15_target_token_ablation_control",
                "sd15_random_token_ablation_control",
                "sd15_image_quality_metric",
                "sd15_white_noise_control",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model paths use safe generated shape prompts only. The required "
            "SD1.5 path captures cross-attention maps for color/shape token "
            "localization over unrelated-token controls, compares target-token "
            "ablation against random/control-token ablation, checks simple generated "
            "image-quality statistics, and rejects white-noise images. SD-Turbo "
            "remains a supplemental fast preflight, not the acceptance fallback."
        )
    if record["number"] == "14.1":
        lock["evidence_level"] = (
            "toy_world_model_controls_plus_real_vjepa2_synthetic_occlusion_preflight"
        )
        lock["claim_scope"] = (
            "GT-1 toy JEPA/world-model controls plus a pinned V-JEPA 2 ViT-L "
            "feature-extraction preflight on deterministic synthetic videos. The real "
            "model path verifies finite non-collapsed video features, same-object "
            "similarity, and a synthetic late-occlusion object-permanence contrast "
            "against absent-object and different-object controls."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "facebook/vjepa2-vitl-fpc64-256",
                "source": "huggingface",
                "revision": "b3c1679b7c34d3255ef3547f27c7b226aefab26f",
                "precision": "float16",
                "gated": False,
                "claim_scope": "synthetic_video_feature_extraction_preflight",
            }
        )
        lock["datasets"].append(
            {
                "id": "course_generated_synthetic_vjepa_videos_v1",
                "source": "generated_tensor_video",
                "seed": 0,
                "video_count": 5,
                "frames_per_video": 8,
                "input_size": "96x96",
                "video_kinds": [
                    "red_square",
                    "red_square_shifted",
                    "blue_circle",
                    "red_square_late_occluded",
                    "red_square_absent_occluder",
                ],
            }
        )
        lock["expected_metrics"].update(
            {
                "jepa_predicts_target": True,
                "transition_consistent": True,
                "vjepa2_preflight_passed": True,
                "vjepa2_feature_shape": [5, 144, 1024],
                "vjepa2_feature_std_min": 0.1,
                "vjepa2_same_object_margin_min": 0.03,
                "vjepa2_synthetic_occlusion_permanence_passed": True,
                "vjepa2_occluded_vs_absent_gap_min": 0.2,
                "vjepa2_occluded_vs_blue_gap_min": 0.03,
                "vjepa2_peak_vram_gb_max": 4.0,
            }
        )
        lock["controls"].extend(
            [
                "toy_target_embedding_prediction",
                "toy_world_state_probe",
                "toy_action_transition_consistency",
                "toy_object_permanence_absent_control",
                "pinned_vjepa2_vitl_safetensors_feature_extraction",
                "deterministic_synthetic_video_control",
                "same_object_vs_different_object_similarity_control",
                "synthetic_late_occlusion_object_permanence_control",
                "same_occluder_absent_object_negative_control",
                "different_object_occlusion_negative_control",
                "masked_prediction_real_video_and_rollout_replication_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real-model path uses generated shape videos only. The occlusion "
            "contrast is a deterministic synthetic-video control and does not claim "
            "real-video object permanence, masked target prediction, or action-conditioned "
            "rollout replication."
        )
    if record["number"] == "15.1":
        lock["evidence_level"] = (
            "exact_lora_dora_controls_plus_matched_lora_dora_full_finetune_proxy_preflight"
        )
        lock["claim_scope"] = (
            "GT-0 exact LoRA/DoRA tensor controls plus a generated rank-1 safe proxy "
            "LoRA training preflight on a planted target-direction classification task. "
            "The report verifies merge/unmerge parity, rank constraint, target-direction "
            "alignment, random-label failure, same-norm random-adapter failure, and a "
            "matched generated-task comparison against rank-1 DoRA and full linear "
            "finetuning."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "course_rank1_lora_safe_proxy_adapter_v1",
                    "source": "generated_training_run",
                    "revision": "seed_0_steps_160_alpha_4_rank_1",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "trained_safe_proxy_lora_adapter",
                },
                {
                    "id": "course_rank1_dora_safe_proxy_adapter_v1",
                    "source": "generated_training_run",
                    "revision": "init_seed_2_data_seed_0_steps_160_alpha_4_rank_1",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "matched_safe_proxy_dora_adapter",
                },
                {
                    "id": "course_full_finetune_safe_proxy_classifier_v1",
                    "source": "generated_training_run",
                    "revision": "init_seed_3_data_seed_0_steps_160",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "matched_safe_proxy_full_finetune_baseline",
                },
            ]
        )
        lock["datasets"].append(
            {
                "id": "course_generated_lora_target_direction_proxy_v1",
                "source": "generated_gaussian_binary_classification",
                "seed": 0,
                "input_dim": 8,
                "target_direction": "feature_0",
                "frozen_baseline_direction": "feature_1",
                "random_label_control": True,
                "same_norm_random_adapter_control": True,
                "matched_lora_dora_full_finetune_comparison": True,
            }
        )
        lock["expected_metrics"].update(
            {
                "adapter_nonzero_update": True,
                "dora_norm_preserved": True,
                "trained_lora_preflight_passed": True,
                "trained_lora_adapter_accuracy_min": 0.95,
                "trained_lora_baseline_accuracy_max": 0.65,
                "trained_lora_random_label_accuracy_max": 0.65,
                "trained_lora_random_adapter_accuracy_max": 0.75,
                "trained_lora_target_direction_cosine_min": 0.95,
                "trained_lora_merge_max_abs_diff_max": 1e-5,
                "trained_lora_adapter_rank_max": 1,
                "trained_lora_dora_norm_preserved": True,
                "matched_peft_comparison_passed": True,
                "matched_accuracy_floor_min": 0.95,
                "matched_target_alignment_floor_min": 0.95,
                "matched_max_distractor_abs_cosine_max": 0.25,
                "matched_dora_norm_preserved": True,
                "matched_lora_trainable_parameters": 10,
                "matched_dora_trainable_parameters": 12,
                "matched_full_finetune_trainable_parameters": 18,
                "trained_lora_peak_vram_gb_max": 1.0,
            }
        )
        lock["controls"].extend(
            [
                "exact_lora_delta_contract",
                "exact_dora_row_magnitude_contract",
                "protected_direction_projection_contract",
                "adapter_accuracy_and_mechanism_contract",
                "trained_rank1_lora_safe_proxy_adapter",
                "merge_unmerge_logit_parity",
                "low_rank_update_constraint",
                "planted_target_direction_alignment",
                "random_label_training_control",
                "same_norm_random_adapter_control",
                "dora_on_learned_delta_norm_preservation",
                "matched_rank1_lora_vs_rank1_dora_vs_full_finetune_comparison",
                "matched_target_direction_alignment_floor",
                "matched_distractor_direction_suppression_control",
            ]
        )
        lock["safety_notes"][-1] = (
            "The trained adapter path uses generated Gaussian binary labels only. "
            "No unsafe adapters, refusal-suppression adapters, or user-facing harmful "
            "prompt completions are trained or distributed. The LoRA, DoRA, and "
            "full-finetune comparison is limited to the generated safe proxy task."
        )
    if record["number"] == "16.1":
        lock["evidence_level"] = "exact_shapley_contract_plus_cuda_trained_neural_game_preflight"
        lock["claim_scope"] = (
            "GT-0 exact Shapley preflight: deterministic additive, conjunction, "
            "permutation-parity, and leave-one-out interaction contracts are kept "
            "for learner exercises, then a CUDA MLP is trained on the complete "
            "binary feature table for a known nonlinear coalition game. The report "
            "computes exact Shapley values from real model ablations, checks them "
            "against the analytic data-generating game, verifies efficiency, and "
            "rejects a shuffled-label model that fits its shuffled table but fails "
            "the true attribution vector. This proves exact Shapley on a finite "
            "trained model organism; it is not a claim about approximate SHAP on "
            "large models."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "course_cuda_neural_coalition_game_mlp_v1",
                "source": "generated_training_run",
                "revision": "seed_1621_players4_hidden64_steps1200",
                "precision": "float32",
                "gated": False,
                "claim_scope": "trained_exact_shapley_neural_game_preflight",
                "optimizer": "AdamW",
                "learning_rate": 0.05,
            }
        )
        lock["datasets"].append(
            {
                "id": "complete_binary_feature_game_table_v1",
                "source": "generated_finite_table",
                "seed": 1621,
                "num_players": 4,
                "example_count": 16,
                "coalition_count": 16,
                "target_rule": "0.25 + 1.2*x0 - 0.7*x1 + 1.6*x2 + 0.9*x3 + 2.2*x0*x2 - 1.5*x1*x3",
                "target_input": [1, 1, 1, 1],
                "baseline_input": [0, 0, 0, 0],
                "analytic_shapley": [2.3, -1.45, 2.7, 0.15],
                "shuffled_label_control": True,
            }
        )
        lock["controls"].extend(
            [
                "exact_additive_shapley_contract",
                "exact_conjunction_efficiency_contract",
                "permutation_parity_contract",
                "leave_one_out_interaction_overcount_contract",
                "real_cuda_training_run",
                "complete_binary_feature_table_fit",
                "real_model_ablation_coalition_table",
                "analytic_shapley_ground_truth_check",
                "efficiency_axiom_check_on_trained_model",
                "shuffled_label_trained_model_negative_control",
                "kernelshap_partitionshap_tokenshap_and_large_model_attribution_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real training path uses only a generated finite binary feature "
            "table and a shuffled-label negative control. It does not use unsafe "
            "prompts, user data, external APIs, adapters, gated weights, or large "
            "persistent artifacts."
        )
    if record["number"] == "16.2":
        lock["evidence_level"] = "kernelshap_partition_contract_plus_cuda_trained_neural_game_preflight"
        lock["claim_scope"] = (
            "GT-0 SHAP baseline preflight: deterministic KernelSHAP and "
            "PartitionSHAP/Owen-value contracts are kept for learner exercises, "
            "then a CUDA MLP is trained on the complete binary feature table for "
            "the same nonlinear coalition game used by 16.1. The report computes "
            "KernelSHAP from real model ablations, checks parity against exact "
            "Shapley and analytic ground truth, verifies singleton PartitionSHAP "
            "parity, documents how grouped PartitionSHAP changes credit, shows a "
            "worse mismatched grouping, and rejects a shuffled-label attribution "
            "control. This proves the local full-table SHAP controls on a finite "
            "trained model organism; it is not a claim about sampled SHAP on large "
            "language or vision models."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "course_cuda_neural_coalition_game_mlp_v1",
                "source": "generated_training_run",
                "revision": "seed_1621_players4_hidden64_steps1200",
                "precision": "float32",
                "gated": False,
                "claim_scope": "trained_kernelshap_partition_preflight",
                "optimizer": "AdamW",
                "learning_rate": 0.05,
            }
        )
        lock["datasets"].append(
            {
                "id": "complete_binary_feature_game_table_v1",
                "source": "generated_finite_table",
                "seed": 1621,
                "num_players": 4,
                "example_count": 16,
                "coalition_count": 16,
                "target_rule": "0.25 + 1.2*x0 - 0.7*x1 + 1.6*x2 + 0.9*x3 + 2.2*x0*x2 - 1.5*x1*x3",
                "target_input": [1, 1, 1, 1],
                "baseline_input": [0, 0, 0, 0],
                "singleton_partition_groups": [[0], [1], [2], [3]],
                "aligned_partition_groups": [[0, 2], [1, 3]],
                "mismatched_partition_groups": [[0, 1], [2, 3]],
                "shuffled_label_control": True,
            }
        )
        lock["controls"].extend(
            [
                "full_table_kernelshap_contract",
                "grouped_partition_shap_contract",
                "real_cuda_training_run",
                "complete_binary_feature_table_fit",
                "real_model_ablation_coalition_table",
                "kernelshap_vs_exact_shapley_on_trained_model",
                "kernelshap_vs_analytic_ground_truth",
                "singleton_partition_exact_parity",
                "aligned_grouped_partition_credit_shift_documented",
                "mismatched_grouping_larger_gap_control",
                "shuffled_label_trained_model_negative_control",
                "sampled_shap_token_region_and_large_model_attribution_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real training path uses only a generated finite binary feature "
            "table and a shuffled-label negative control. It does not use unsafe "
            "prompts, user data, external APIs, adapters, gated weights, or large "
            "persistent artifacts."
        )
    if record["number"] == "16.3":
        lock["evidence_level"] = "shapley_interaction_contract_plus_cuda_trained_neural_game_preflight"
        lock["claim_scope"] = (
            "GT-0 Shapley interaction preflight: deterministic additive, target-pair, "
            "and shapiq parity contracts are kept for learner exercises, then a CUDA "
            "MLP is trained on the complete binary feature table for the same "
            "nonlinear coalition game used by 16.1 and 16.2. The report recovers the "
            "planted positive and negative pair interactions from real model "
            "ablations, bounds off-target interactions, checks shapiq SII parity on "
            "the trained model table, and rejects a shuffled-label interaction "
            "control. This proves local pairwise interaction recovery on a finite "
            "trained model organism; it is not a claim about large-model token, SAE, "
            "vision, or multimodal interaction attribution."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "course_cuda_neural_coalition_game_mlp_v1",
                "source": "generated_training_run",
                "revision": "seed_1621_players4_hidden64_steps1200",
                "precision": "float32",
                "gated": False,
                "claim_scope": "trained_pairwise_shapley_interaction_preflight",
                "optimizer": "AdamW",
                "learning_rate": 0.05,
            }
        )
        lock["datasets"].append(
            {
                "id": "complete_binary_feature_game_table_v1",
                "source": "generated_finite_table",
                "seed": 1621,
                "num_players": 4,
                "example_count": 16,
                "coalition_count": 16,
                "target_rule": "0.25 + 1.2*x0 - 0.7*x1 + 1.6*x2 + 0.9*x3 + 2.2*x0*x2 - 1.5*x1*x3",
                "positive_interaction_pair": [0, 2],
                "positive_interaction_weight": 2.2,
                "negative_interaction_pair": [1, 3],
                "negative_interaction_weight": -1.5,
                "target_input": [1, 1, 1, 1],
                "baseline_input": [0, 0, 0, 0],
                "shuffled_label_control": True,
            }
        )
        lock["controls"].extend(
            [
                "additive_zero_interaction_contract",
                "single_target_pair_interaction_contract",
                "small_game_shapiq_parity_contract",
                "real_cuda_training_run",
                "complete_binary_feature_table_fit",
                "real_model_ablation_coalition_table",
                "positive_pair_interaction_recovery",
                "negative_pair_interaction_recovery",
                "off_target_interaction_ceiling",
                "shapiq_sii_parity_on_trained_model_table",
                "shuffled_label_trained_model_negative_control",
                "large_model_token_sae_vision_and_multimodal_interactions_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real training path uses only a generated finite binary feature "
            "table and a shuffled-label negative control. It does not use unsafe "
            "prompts, user data, external APIs, adapters, gated weights, or large "
            "persistent artifacts."
        )
    if record["number"] == "16.4":
        lock["evidence_level"] = "tokenshap_contract_plus_cuda_trained_token_scorer_preflight"
        lock["claim_scope"] = (
            "GT-0 TokenSHAP preflight: deterministic masked-token coalition, exact "
            "TokenShapley, and sampled TokenSHAP contracts are kept for learner "
            "exercises, then a CUDA embedding MLP is trained on the complete masked "
            "coalition table for the four-position prompt 'The capital is Paris'. "
            "The report computes exact and sampled TokenSHAP from real model outputs, "
            "checks them against the analytic token game, verifies efficiency and "
            "top-token ranking, and rejects a shuffled-label token-scorer control. "
            "This proves local token-position SHAP mechanics on a finite trained "
            "model organism; it is not a claim about sampled TokenSHAP on a large "
            "language model."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "course_cuda_tiny_token_scorer_mlp_v1",
                "source": "generated_training_run",
                "revision": "seed_1640_prompt4_embed32_hidden96_steps1200",
                "precision": "float32",
                "gated": False,
                "claim_scope": "trained_tokenshap_preflight",
                "optimizer": "AdamW",
                "learning_rate": 0.02,
            }
        )
        lock["datasets"].append(
            {
                "id": "complete_masked_token_coalition_table_v1",
                "source": "generated_finite_table",
                "seed": 1640,
                "tokens": ["The", "capital", "is", "Paris"],
                "mask_token": "[MASK]",
                "token_count": 4,
                "coalition_count": 16,
                "target_rule": "score Paris +1 and capital/Paris interaction +2",
                "analytic_token_shapley": [0.0, 1.0, 0.0, 2.0],
                "sampled_permutations": 512,
                "shuffled_label_control": True,
            }
        )
        lock["controls"].extend(
            [
                "masked_token_coalition_contract",
                "exact_token_shapley_contract",
                "sampled_tokenshap_contract",
                "real_cuda_training_run",
                "complete_masked_token_table_fit",
                "real_model_masked_token_coalition_values",
                "exact_tokenshap_vs_analytic_ground_truth",
                "sampled_tokenshap_vs_exact_model_values",
                "top_token_rank_preservation",
                "shapley_efficiency_check_on_trained_token_model",
                "shuffled_label_trained_model_negative_control",
                "large_language_model_tokenshap_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real training path uses only a generated finite masked-token table "
            "for a safe four-token prompt and a shuffled-label negative control. It "
            "does not use unsafe prompts, user data, external APIs, adapters, gated "
            "weights, or large persistent artifacts."
        )
    if record["number"] == "16.5":
        lock["evidence_level"] = "vlm_shap_contract_plus_pinned_clip_rendered_shape_preflight"
        lock["claim_scope"] = (
            "GT-0 VLM SHAP preflight: deterministic modality and region SHAP "
            "contracts are kept for learner exercises, then a pinned CLIP "
            "ViT-B/32 checkpoint is run on safe rendered red-square and blue-circle "
            "controls. The report computes image/text modality SHAP and structured "
            "object/background/OCR-region SHAP from real CLIP logits, verifies "
            "modality synergy, object-region localization, efficiency, a target-vs-"
            "distractor text margin, and measured VRAM. This proves local VLM SHAP "
            "mechanics on deterministic rendered images; it is not a claim about "
            "large generative VLM attribution, pixel-level saliency, or real-image "
            "benchmark coverage."
        )
        lock["required_gpu_gb"] = 2
        lock["models"].append(
            {
                "id": "openai/clip-vit-base-patch32",
                "source": "huggingface",
                "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
                "precision": "float32",
                "gated": False,
                "claim_scope": "rendered_shape_modality_and_region_shap_preflight",
            }
        )
        lock["datasets"].append(
            {
                "id": "course_rendered_clip_vlm_shap_shapes_v1",
                "source": "generated_pil_images",
                "seed": 0,
                "image_size": "224x224",
                "modality_target": "red square image plus target text",
                "modality_distractor": "blue circle image plus neutral text",
                "region_target_text": "a red square",
                "region_names": ["object", "background", "ocr_text"],
                "target_region": "object",
                "text_controls": ["a photo", "a red square", "a blue circle"],
            }
        )
        lock["controls"].extend(
            [
                "toy_modality_shap_contract",
                "toy_region_shap_contract",
                "pinned_real_clip_rendered_shape_logits",
                "image_text_modality_coalition_table",
                "structured_object_background_ocr_region_coalition_table",
                "target_vs_distractor_text_margin",
                "object_region_over_background_and_ocr_margin",
                "shapley_efficiency_checks_on_clip_tables",
                "large_generative_vlm_and_real_image_benchmark_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real CLIP path uses generated safe rendered shape images and safe "
            "shape captions only. It does not use unsafe prompts, user images, "
            "external APIs, adapters, gated weights, or persistent real-image data."
        )
    if record["number"] == "16.6":
        lock["evidence_level"] = "shap_vs_patching_contract_plus_cuda_model_organism_preflight"
        lock["claim_scope"] = (
            "GT-0 SHAP-vs-patching preflight: deterministic additive-agreement and "
            "interaction-overcount contracts are kept for learner exercises, then "
            "two CUDA model organisms are trained on complete binary feature tables. "
            "A linear additive model verifies that exact Shapley and full-minus-"
            "ablated patching effects agree when the learned function is additive. "
            "A nonlinear neural coalition game verifies that interaction-heavy "
            "learned behavior makes patching effects disagree with Shapley and "
            "overcount absolute credit. This proves the method comparison on real "
            "model outputs from finite trained systems; it is not a claim about "
            "large transformer activation patching."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].extend(
            [
                {
                    "id": "course_cuda_linear_additive_model_v1",
                    "source": "generated_training_run",
                    "revision": "seed_1660_players4_steps1000",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "additive_shapley_patching_agreement_preflight",
                    "optimizer": "AdamW",
                    "learning_rate": 0.05,
                },
                {
                    "id": "course_cuda_neural_coalition_game_mlp_v1",
                    "source": "generated_training_run",
                    "revision": "seed_1621_players4_hidden64_steps1200",
                    "precision": "float32",
                    "gated": False,
                    "claim_scope": "interaction_shapley_patching_disagreement_preflight",
                    "optimizer": "AdamW",
                    "learning_rate": 0.05,
                },
            ]
        )
        lock["datasets"].append(
            {
                "id": "complete_binary_feature_games_additive_and_interaction_v1",
                "source": "generated_finite_table",
                "seed": "1660_additive_1621_interaction",
                "num_players": 4,
                "example_count": 16,
                "additive_target_rule": "0.25 + 1.2*x0 - 0.7*x1 + 1.6*x2 + 0.9*x3",
                "interaction_target_rule": "0.25 + 1.2*x0 - 0.7*x1 + 1.6*x2 + 0.9*x3 + 2.2*x0*x2 - 1.5*x1*x3",
                "patching_effect": "full_output_minus_single_feature_ablated_output",
            }
        )
        lock["controls"].extend(
            [
                "toy_additive_agreement_contract",
                "toy_interaction_overcount_contract",
                "real_cuda_additive_linear_training_run",
                "real_cuda_interaction_mlp_training_run",
                "complete_binary_feature_table_fit",
                "exact_shapley_vs_full_minus_ablated_outputs",
                "additive_agreement_positive_control",
                "interaction_disagreement_negative_control",
                "absolute_credit_overcount_check",
                "large_transformer_activation_patching_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real training path uses only generated finite binary feature "
            "tables. It does not use unsafe prompts, user data, external APIs, "
            "adapters, gated weights, or large persistent artifacts."
        )
    if record["number"] == "16.7":
        lock["evidence_level"] = "data_shapley_contract_plus_cuda_one_step_training_preflight"
        lock["claim_scope"] = (
            "GT-0 Data Shapley preflight: deterministic exact, Monte Carlo, and "
            "first-order in-run Data Shapley contracts are kept for learner "
            "exercises, then a one-step linear regression problem is evaluated on "
            "CUDA. The report enumerates all training-example coalitions on GPU, "
            "runs an actual full-batch one-step optimizer update, computes per-"
            "example autograd gradient-dot scores in the same run, compares exact "
            "and sampled Data Shapley values, and verifies that the harmful flipped "
            "example is identified and deletion improves utility. This proves the "
            "one-training-run attribution mechanics on a finite model organism; it "
            "is not a claim about large-dataset TracIn, influence functions, or "
            "production data valuation."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "course_cuda_one_step_linear_regression_data_shapley_v1",
                "source": "generated_training_run",
                "revision": "seed_0_examples4_lr0.5_one_step",
                "precision": "float64",
                "gated": False,
                "claim_scope": "exact_mc_and_in_run_data_shapley_preflight",
                "optimizer": "SGD",
                "learning_rate": 0.5,
            }
        )
        lock["datasets"].append(
            {
                "id": "toy_data_shapley_helpful_harmful_examples_v1",
                "source": "generated_finite_table",
                "seed": 0,
                "train_x": [[1.0], [1.0], [1.0], [1.0]],
                "train_y": [1.0, 1.0, 1.0, -1.0],
                "val_x": [[1.0]],
                "val_y": [1.0],
                "harmful_example_index": 3,
                "coalition_count": 16,
                "monte_carlo_samples": 512,
            }
        )
        lock["controls"].extend(
            [
                "exact_data_shapley_contract",
                "monte_carlo_data_shapley_contract",
                "in_run_gradient_dot_contract",
                "real_cuda_coalition_utility_enumeration",
                "actual_full_batch_one_step_training_update",
                "autograd_per_example_gradient_scores",
                "harmful_example_deletion_check",
                "monte_carlo_vs_exact_parity",
                "in_run_score_vs_exact_correlation",
                "large_dataset_data_valuation_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real training path uses only a generated four-example regression "
            "problem with one flipped label. It does not use unsafe prompts, user "
            "data, external APIs, adapters, gated weights, or large persistent "
            "artifacts."
        )
    if record["number"] == "16.8":
        lock["evidence_level"] = "shap_mech_agreement_contract_plus_cuda_neural_game_preflight"
        lock["claim_scope"] = (
            "GT-0 SHAP/mechanistic-agreement preflight: deterministic additive "
            "agreement and XOR interaction-disagreement contracts are kept for "
            "learner exercises, then a nonlinear CUDA neural coalition game is "
            "trained on the complete binary feature table. The report compares "
            "exact Shapley values from real model ablations against known analytic "
            "mechanistic feature contributions, checks rank correlation, top-k "
            "overlap, deletion consequence, planted pair-interaction recovery, and "
            "rejects a shuffled-label model that fails mechanistic agreement. This "
            "proves the agreement-testing workflow on a finite trained model "
            "organism; it is not a claim about broad mechanistic interpretability "
            "agreement in large models."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "course_cuda_neural_coalition_game_mlp_v1",
                "source": "generated_training_run",
                "revision": "seed_1621_players4_hidden64_steps1200",
                "precision": "float32",
                "gated": False,
                "claim_scope": "shapley_mechanistic_agreement_preflight",
                "optimizer": "AdamW",
                "learning_rate": 0.05,
            }
        )
        lock["datasets"].append(
            {
                "id": "complete_binary_feature_game_table_v1",
                "source": "generated_finite_table",
                "seed": 1621,
                "num_players": 4,
                "example_count": 16,
                "coalition_count": 16,
                "target_rule": "0.25 + 1.2*x0 - 0.7*x1 + 1.6*x2 + 0.9*x3 + 2.2*x0*x2 - 1.5*x1*x3",
                "mechanistic_scores": [2.3, -1.45, 2.7, 0.15],
                "positive_interaction_pair": [0, 2],
                "negative_interaction_pair": [1, 3],
                "shuffled_label_control": True,
            }
        )
        lock["controls"].extend(
            [
                "toy_additive_agreement_contract",
                "toy_xor_interaction_disagreement_contract",
                "real_cuda_neural_game_training_run",
                "complete_binary_feature_table_fit",
                "shapley_vs_known_mechanistic_scores",
                "spearman_and_topk_overlap_checks",
                "top_feature_deletion_consequence_check",
                "planted_pair_interaction_recovery",
                "shuffled_label_trained_model_negative_control",
                "agreement_matrix_csv_artifact",
                "deletion_and_insertion_curve_artifacts",
                "topk_overlap_heatmap_artifact",
                "method_disagreement_examples_artifact",
                "large_model_mechanistic_agreement_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real training path uses only a generated finite binary feature "
            "table and a shuffled-label negative control. It does not use unsafe "
            "prompts, user data, external APIs, adapters, gated weights, or large "
            "persistent artifacts."
        )
    if record["number"] == "17.1":
        lock["evidence_level"] = "toy_trajectory_contract_plus_real_modular_addition_checkpoint_preflight"
        lock["claim_scope"] = (
            "GT-0 developmental-interpretability preflight: deterministic toy AR, "
            "JEPA, diffusion, Mamba, and random-control trajectories are kept for "
            "learner exercises, then a tiny modular-addition MLP is trained on CUDA "
            "on the complete mod-13 addition table. The report saves and reloads "
            "real checkpoints, detects stable emergence and a phase jump from the "
            "reloaded checkpoint accuracies, and rejects a random-label checkpoint "
            "control evaluated against the true table. This proves the local "
            "checkpoint-archaeology workflow on a finite model organism; it is not "
            "a claim about large-model developmental mechanisms."
        )
        lock["required_gpu_gb"] = 1
        lock["models"].append(
            {
                "id": "course_tiny_modular_addition_mlp_v1",
                "source": "generated_training_run",
                "revision": "seed_0_mod13_embed32_hidden64_steps80",
                "precision": "float32",
                "gated": False,
                "claim_scope": "trained_checkpoint_archaeology_preflight",
                "optimizer": "AdamW",
                "learning_rate": 0.005,
            }
        )
        lock["datasets"].append(
            {
                "id": "complete_mod13_addition_table_v1",
                "source": "generated_finite_table",
                "seed": 0,
                "modulus": 13,
                "example_count": 169,
                "label_rule": "(left + right) % 13",
                "random_label_control": True,
                "checkpoint_steps": [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80],
            }
        )
        lock["controls"].extend(
            [
                "toy_checkpoint_trajectory_contract",
                "toy_phase_transition_contract",
                "toy_random_control_contract",
                "toy_family_timing_contract",
                "real_cuda_training_run",
                "real_checkpoint_save_and_reload",
                "complete_modular_addition_table_accuracy",
                "random_label_checkpoint_control",
                "stable_threshold_emergence_check",
                "phase_jump_check",
                "large_model_checkpoint_archaeology_not_claimed",
            ]
        )
        lock["safety_notes"][-1] = (
            "The real training path uses only a generated finite modular-addition "
            "table and a random-label negative control. It does not use unsafe data, "
            "user data, adapters, gated weights, or large persistent artifacts."
        )
    lock["expected_metrics"].update(
        NOTEBOOK_CONTRACT_EXPECTED_METRICS.get(record["number"], {})
    )
    return lock


def expected_outputs_readme(record: dict[str, Any]) -> str:
    """Return the fixture provenance README required for hard exercises."""

    if record["number"] == "5.5":
        tolerance_note = (
            "Integer, token, boolean, revision, model-type, and schema-contract "
            "checks require exact equality. Float32 toy checks should use "
            "`rtol=1e-5` and `atol=1e-6` unless the section file documents a "
            "stricter tolerance. The current DiffusionGemma contract checks "
            "exact public metadata, shard counts, runtime readiness, and the "
            "pinned isolated vLLM NVFP4 generation proof artifact. Google BF16 "
            "direct local loading remains deferred for the 24GB tier."
        )
    else:
        tolerance_note = (
            "Integer, token, boolean, and schema-contract checks require exact "
            "equality. Float32 toy checks should use `rtol=1e-5` and `atol=1e-6` "
            "unless the section file documents a stricter tolerance. bf16 or "
            "quantized real-model checks must update the section artifact lock and "
            "report with explicit behavioral tolerances."
        )

    return (
        f"# [{record['number']}] {record['title']} Expected Outputs\n\n"
        "This directory contains frozen fixtures for the learner-facing notebook "
        "contract. These fixtures are intentionally small where possible: they "
        "protect visible exercise behavior. Real-model evidence, when claimed by "
        "a section, is pinned in `artifacts.lock.yml` and measured in "
        "`verification_report.json`.\n\n"
        "## How this fixture was produced\n\n"
        "`smoke_test.json` and `reference_metrics.json` are generated by "
        "`scripts/generate_extension_verification_assets.py` from "
        "`infrastructure/core/config.yaml` and the section GT-tier map in that "
        "script. Section verification reports are then regenerated by "
        "`scripts/run_extension_verification_reports.py`.\n\n"
        "## Trusted implementation\n\n"
        "The fixture schema and baseline/control slots are produced by the "
        "course asset generator. Numeric report values come from the checked-in "
        "`solutions.py` implementation for this section when the report runner "
        "executes `run_smoke_test` and `run_gpu_test`.\n\n"
        "## Random seed\n\n"
        "The generated artifact lock uses `seed: 0` for toy/generated inputs "
        "unless a section-specific solution function states a narrower seed.\n\n"
        "## Allowed tolerances\n\n"
        f"{tolerance_note}\n\n"
        "## When to regenerate\n\n"
        "Regenerate these fixtures when the section title, GT tier, artifact "
        "lock schema, verification report schema, expected baseline/control "
        "slots, or trusted toy implementation changes. Do not regenerate merely "
        "to hide a failing check; update the implementation or the declared "
        "claim scope first.\n"
    )


def smoke_test_fixture(record: dict[str, Any]) -> dict[str, Any]:
    fixture = {
        "notebook_id": record["notebook_id"],
        "section": record["number"],
        "title": record["title"],
        "gt_tier": record["gt_tier"],
        "evidence_level": "notebook_contract",
        "claim_scope": "Contract check; not a full real-model validation.",
        "tests_passed": True,
        "accepted": True,
    }
    if record["number"] == "5.1":
        fixture.update(
            {
                "evidence_level": "full_decoder_notebook_contract",
                "claim_scope": (
                    "Learner-facing Gemma-from-scratch smoke path: the notebook "
                    "defines its own RMSNorm, RoPE, grouped-query attention, "
                    "decoder layer, full causal LM, and tiny Hugging Face "
                    "reference-parity check. This is architecture parity with "
                    "deterministic tiny weights, not a claim about gated pretrained "
                    "Gemma weights."
                ),
                "metrics": {
                    "cache_max_abs_diff_max": 1e-5,
                    "reference_logits_max_abs_diff_max": 5e-4,
                    "reference_logits_mse_max": 1e-7,
                    "reference_logits_topk_agreement": 1.0,
                    "reference_model_family": "transformers.GemmaForCausalLM",
                    "reference_weight_key_count": 21,
                },
            }
        )
    if record["number"] == "10.1":
        fixture.update(
            {
                "evidence_level": "capstone_planning_contract_plus_live_report_audit",
                "claim_scope": (
                    "Learner-facing smoke path: the notebook builds the capstone "
                    "planning scaffold and audits a committed CUDA mini activation-"
                    "oracle report. The live result itself is pinned in "
                    "artifacts.lock.yml and verification_report.json."
                ),
                "metrics": {
                    "seed_count": 3,
                    "oracle_accuracy_mean_min": 0.9,
                    "heldout_template_accuracy_mean_min": 0.9,
                    "ablation_drop_mean_min": 0.2,
                },
            }
        )
    return fixture


def reference_metrics_fixture(
    record: dict[str, Any],
    lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if lock is None:
        lock = existing_lock_for(record) or lock_for(record)
    fixture = {
        "notebook_id": record["notebook_id"],
        "metrics": {"tests_passed": True},
        "baselines": {"required": True},
        "negative_controls": {"required_where_applicable": True},
        "ood_tests": {"required_where_applicable": True},
    }
    fixture["metrics"].update(lock.get("expected_metrics", {}))
    if record["number"] == "5.1":
        fixture = {
            "notebook_id": record["notebook_id"],
            "metrics": {
                "tests_passed": True,
                "cache_max_abs_diff_max": 1e-5,
                "peak_vram_gb_max": 1.0,
                "reference_cache_max_abs_diff_max": 1e-5,
                "reference_loaded_key_count": 21,
                "reference_logits_max_abs_diff_max": 5e-4,
                "reference_logits_mse_max": 1e-7,
                "reference_logits_topk_agreement": 1.0,
                "reference_model_family": "transformers.GemmaForCausalLM",
                "reference_weight_key_count": 21,
                "rope_norm_error_max": 1e-6,
            },
            "models": [
                {
                    "id": "transformers.GemmaForCausalLM",
                    "revision": "deterministic_tiny_config_seed_5101",
                    "claim_scope": "reference_architecture_parity_no_pretrained_weights",
                }
            ],
            "baselines": {
                "declared_controls": [
                    "local_rmsnorm_formula_contract",
                    "local_rope_norm_contract",
                    "local_kv_cache_parity_contract",
                    "huggingface_gemma_reference_architecture_logits_parity",
                    "state_dict_key_count_match",
                    "reference_cache_parity_check",
                    "gated_pretrained_gemma_weights_not_claimed",
                ]
            },
            "negative_controls": {
                "declared_controls": ["gated_pretrained_gemma_weights_not_claimed"]
            },
            "ood_tests": {
                "declared": (
                    "shape/cache/reference-parity checks cover this GT-0 "
                    "architecture contract"
                )
            },
        }
        fixture["metrics"].update(lock.get("expected_metrics", {}))
    return fixture


def write_notebook_assets(
    records: list[dict[str, Any]],
    *,
    overwrite_contracts: bool = False,
) -> None:
    schema = verification_schema()
    for record in records:
        exercise_dir = record["exercise_dir"]
        expected_dir = exercise_dir / "expected_outputs"
        expected_dir.mkdir(parents=True, exist_ok=True)
        generated_lock = lock_for(record)
        lock_path = exercise_dir / "artifacts.lock.yml"
        current_lock = existing_lock_for(record)
        if overwrite_contracts or current_lock is None:
            lock = generated_lock
            write_yaml(lock_path, lock)
        else:
            lock = current_lock
        write_json(exercise_dir / "verification_report.schema.json", schema)
        write_json(expected_dir / "smoke_test.json", smoke_test_fixture(record))
        write_json(
            expected_dir / "reference_metrics.json",
            reference_metrics_fixture(record, lock),
        )
        write_text(expected_dir / "README.md", expected_outputs_readme(record))
        readme = (
            f"# [{record['number']}] {record['title']} Verification Assets\n\n"
            "Generated support files for the roadmap verification contract.\n\n"
            "- `artifacts.lock.yml` pins the current artifact contract.\n"
            "- `verification_report.schema.json` defines the required final report.\n"
            "- `expected_outputs/smoke_test.json` records the historical "
            "`run_smoke_test` contract hook.\n"
            "- `expected_outputs/reference_metrics.json` records baseline/control slots.\n\n"
            "When a section claims a real-model path, `artifacts.lock.yml` names the "
            "exact model revisions, dataset revisions, seeds, dtypes, controls, "
            "expected metrics, measured VRAM budget, and narrowed claim scope.\n"
        )
        write_text(exercise_dir / "README.md", readme, overwrite=False)


def write_registry() -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    csv_path = docs / "artifact_registry.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(REGISTRY_ROWS)

    markdown_lines = [
        "# Artifact Registry",
        "",
        "Starter registry for resources required by `Extension-Roadmap.md`.",
        "Rows with `pin_before_use` must be resolved before a graded real-model run.",
        "",
        "| " + " | ".join(REGISTRY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in REGISTRY_COLUMNS) + " |",
    ]
    for row in REGISTRY_ROWS:
        markdown_lines.append("| " + " | ".join(row[column] for column in REGISTRY_COLUMNS) + " |")
    write_text(docs / "artifact_registry.md", "\n".join(markdown_lines) + "\n")
    write_yaml(
        docs / "artifact_registry.lock.yml",
        {
            "schema_version": 1,
            "columns": REGISTRY_COLUMNS,
            "artifacts": REGISTRY_ROWS,
        },
    )


def write_method_registry() -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    csv_path = docs / "method_registry.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METHOD_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(METHOD_ROWS)

    markdown_lines = [
        "# Method Registry",
        "",
        "Living coverage map for methods in `Extension-Roadmap.md`.",
        "This file distinguishes implemented local contracts from real-model",
        "replications, read-only methods, and wait-for-weights watchlist items.",
        "",
        "| " + " | ".join(METHOD_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in METHOD_COLUMNS) + " |",
    ]
    for row in METHOD_ROWS:
        markdown_lines.append("| " + " | ".join(row[column] for column in METHOD_COLUMNS) + " |")
    write_text(docs / "method_registry.md", "\n".join(markdown_lines) + "\n")
    write_yaml(
        docs / "method_registry.lock.yml",
        {
            "schema_version": 1,
            "columns": METHOD_COLUMNS,
            "methods": METHOD_ROWS,
        },
    )


def hard_exercise_ladder_rows(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        exercise_dir = record["exercise_dir"]
        metadata_path = exercise_dir / "artifacts.lock.yml"
        lock = yaml.safe_load(metadata_path.read_text()) if metadata_path.exists() else {}
        metadata = lock.get("exercise_metadata", record["exercise_metadata"])
        if metadata["DIFFICULTY"] < 3:
            continue

        fixture_path = exercise_dir / "expected_outputs" / "README.md"
        tests_path = exercise_dir / "tests.py"
        report_path = exercise_dir / "verification_report.json"

        if metadata["REQUIRES_GPU"]:
            gpu_requirement = "cuda_section_metric required before chapter release"
        else:
            gpu_requirement = "required only if section makes a GPU-backed claim"

        release_status = "report_pending"
        remaining_release_evidence = "verification_report_missing_or_not_accepted"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            gpu_category = (
                report.get("metrics", {})
                .get("gpu_evidence", {})
                .get("category", "missing")
            )
            if report.get("accepted") is True and gpu_category == "cuda_section_metric":
                release_status = "released_with_cuda_section_metric"
                remaining_release_evidence = ""
            elif report.get("accepted") is True:
                release_status = f"accepted_with_{gpu_category}"
                remaining_release_evidence = "upgrade_to_cuda_section_metric_before_release"
            else:
                release_status = f"report_not_accepted_{gpu_category}"
                remaining_release_evidence = "fix_report_failures_before_release"
        rows.append(
            {
                "section": record["number"],
                "notebook_id": lock.get("notebook_id", record["notebook_id"]),
                "title": lock.get("title", record["title"]),
                "gt_tier": lock.get("gt_tier", record["gt_tier"]),
                "difficulty": str(metadata["DIFFICULTY"]),
                "importance": str(metadata["IMPORTANCE"]),
                "requires_gpu": str(metadata["REQUIRES_GPU"]).lower(),
                "metadata_source": f"{metadata_path.relative_to(ROOT)}:exercise_metadata",
                "fixture_provenance_source": str(fixture_path.relative_to(ROOT)),
                "visible_tests_source": (
                    str(tests_path.relative_to(ROOT)) if tests_path.exists() else "MISSING"
                ),
                "report_source": (
                    str(report_path.relative_to(ROOT)) if report_path.exists() else "PENDING"
                ),
                "gpu_evidence_requirement": gpu_requirement,
                "toy_oracle_requirement": "required for every hard exercise",
                "slow_fast_oracle_requirement": "required where an optimized path exists",
                "property_test_requirement": "required for mathematical invariants",
                "debug_mode_requirement": "required for complex functions",
                "release_status": release_status,
                "remaining_release_evidence": remaining_release_evidence,
            }
        )
    return rows


def write_hard_exercise_ladder_registry(records: list[dict[str, Any]]) -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    rows = hard_exercise_ladder_rows(records)

    csv_path = docs / "hard_exercise_ladder_registry.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=LADDER_REGISTRY_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    markdown_lines = [
        "# Hard Exercise Ladder Registry",
        "",
        "Generated audit table for difficulty-3+ extension sections.",
        "Rows describe current notebook-contract evidence and the release evidence",
        "that still has to be checked before a stronger real-model claim.",
        "",
        "| " + " | ".join(LADDER_REGISTRY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in LADDER_REGISTRY_COLUMNS) + " |",
    ]
    for row in rows:
        markdown_lines.append(
            "| " + " | ".join(row[column] for column in LADDER_REGISTRY_COLUMNS) + " |"
        )
    write_text(docs / "hard_exercise_ladder_registry.md", "\n".join(markdown_lines) + "\n")
    write_yaml(
        docs / "hard_exercise_ladder_registry.lock.yml",
        {
            "schema_version": 1,
            "columns": LADDER_REGISTRY_COLUMNS,
            "hard_exercises": rows,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite-contracts",
        action="store_true",
        help=(
            "Overwrite existing artifacts.lock.yml files with generated scaffold "
            "contracts. By default existing locks are preserved and only missing "
            "locks are created."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = extension_sections()
    write_notebook_assets(records, overwrite_contracts=args.overwrite_contracts)
    write_registry()
    write_method_registry()
    write_hard_exercise_ladder_registry(records)
    contract_mode = "overwritten" if args.overwrite_contracts else "preserved"
    print(
        f"Generated verification assets for {len(records)} extension sections "
        f"(existing artifact locks {contract_mode})."
    )


if __name__ == "__main__":
    main()
