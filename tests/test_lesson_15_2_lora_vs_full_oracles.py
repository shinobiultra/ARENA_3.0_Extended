"""15.2 LoRA vs full fine-tuning: tiny CPU oracles and the exact additive matrix play, no model loads."""
from __future__ import annotations

import importlib

import pytest

MODULE = 'chapter15_peft_misalignment.exercises.part2_lora_vs_full_finetuning'


ORACLES = [
    ('test_toy_lora_delta_uses_scaled_b_times_a', ['toy_lora_delta']),
    ('test_lora_linear_starts_as_exact_base_and_trains_only_factors', ['LoRALinear']),
    ('test_merge_unmerge_preserves_logits_and_base_weight', ['merge_unmerge_max_diff', 'LoRALinear']),
    ('test_inject_lora_replaces_real_named_linear_modules', ['inject_lora', 'LoRALinear']),
    ('test_freeze_except_lora_exposes_exact_parameter_budget', ['freeze_except_lora', 'LoRALinear']),
    ('test_codebook_split_has_disjoint_templates_and_twenty_heldout_examples', ['make_codebook_examples']),
    ('test_random_label_control_preserves_prompts_and_balance', ['make_codebook_examples', 'randomize_training_labels']),
    ('test_matched_training_loop_learns_exact_next_token_organism', ['train_on_codebook']),
    ('test_spectrum_distinguishes_rank_one_from_full_rank', ['summarize_spectrum']),
    ('test_dominant_activation_drift_recovers_planted_direction', ['dominant_activation_drift']),
    ('test_direction_ablation_removes_exact_adaptation_component', ['evaluate_direction_ablation']),
    ('test_same_norm_random_lora_matches_learned_norm', ['make_same_norm_random_lora']),
]


@pytest.mark.parametrize('oracle,functions', ORACLES)
def test_generated_visible_methods_against_independent_cpu_oracles(oracle, functions):
    torch = importlib.import_module('torch')
    module = importlib.import_module(MODULE + '.solutions')
    tests = importlib.import_module(MODULE + '.tests')
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        getattr(tests, oracle)(*[getattr(module, function) for function in functions])
    finally:
        torch.set_default_dtype(previous)


def test_additive_cpu_play_knobs_match_independent_diagonal_oracles(monkeypatch):
    torch = importlib.import_module('torch')
    module = importlib.import_module(MODULE + '.solutions')
    def forbidden(*args, **kwargs):
        raise AssertionError('Matrix play attempted model loading or training')
    monkeypatch.setattr(module, 'load_pinned_model', forbidden)
    monkeypatch.setattr(module, 'train_on_codebook', forbidden)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        for layer, diagonal in enumerate(((3., 1., .3, .1), (1.5, 1.2, .8, .4))):
            for rank in (1, 2, 3, 4):
                for replay in (0., .2, 1.):
                    result = module.lora_matrix_play(rank=rank, layer=layer, protected_weight=replay)
                    expected = torch.diag(torch.tensor([value / (1 + replay) if i < rank else 0 for i, value in enumerate(diagonal)], dtype=torch.float64))
                    torch.testing.assert_close(result['update'], expected, atol=1e-12, rtol=1e-12)
                    module.check_lora_matrix_play(result)
                    assert int(torch.linalg.matrix_rank(result['update'])) == rank
        weak = module.lora_matrix_play(rank=2, layer=0, protected_weight=0.)
        strong = module.lora_matrix_play(rank=2, layer=0, protected_weight=1.)
        assert strong['metrics']['Rank-constrained fit']['replay_displacement'] < weak['metrics']['Rank-constrained fit']['replay_displacement']
        assert strong['metrics']['Rank-constrained fit']['task_error'] > weak['metrics']['Rank-constrained fit']['task_error']
        for invalid in ({'rank': 0, 'layer': 0, 'protected_weight': 0.}, {'rank': 1, 'layer': 2, 'protected_weight': 0.}, {'rank': 1, 'layer': 0, 'protected_weight': -1.}):
            with pytest.raises(ValueError):
                module.lora_matrix_play(**invalid)
    finally:
        torch.set_default_dtype(previous_dtype)
