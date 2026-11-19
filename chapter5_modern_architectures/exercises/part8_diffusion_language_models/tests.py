from collections.abc import Callable

import torch as t

from arena_ext import diffusion_lm as reference


def _solutions():
    from chapter5_modern_architectures.exercises.part5_diffusion_language_models import (
        solutions,
    )

    return solutions


def test_linear_mask_schedule_and_expected_fraction(
    linear_mask_schedule: Callable | None = None,
    expected_mask_fraction: Callable | None = None,
):
    solutions = _solutions()
    linear_mask_schedule = linear_mask_schedule or solutions.linear_mask_schedule
    expected_mask_fraction = expected_mask_fraction or solutions.expected_mask_fraction
    schedule = linear_mask_schedule(5, mask_token_id=99)
    timesteps = t.tensor([0, 2, 4])
    t.testing.assert_close(
        schedule.mask_probs,
        t.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
        msg="Linear mask schedule should interpolate from 0 to 1 over num_steps.",
    )
    assert schedule.num_steps == 5, "Schedule num_steps should equal len(mask_probs)."
    assert expected_mask_fraction(schedule, timesteps) == 0.5, (
        "Expected mask fraction should average the selected timestep probabilities."
    )
    reference_schedule = reference.linear_mask_schedule(5, mask_token_id=99)
    t.testing.assert_close(
        schedule.mask_probs,
        reference_schedule.mask_probs,
        msg="Local schedule should match the independent reference implementation.",
    )
    print("All tests in `test_linear_mask_schedule_and_expected_fraction` passed!")


def test_forward_noising_extremes_and_seeded_masks(
    apply_forward_noising: Callable | None = None,
    linear_mask_schedule: Callable | None = None,
):
    solutions = _solutions()
    apply_forward_noising = apply_forward_noising or solutions.apply_forward_noising
    linear_mask_schedule = linear_mask_schedule or solutions.linear_mask_schedule
    input_ids = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    schedule = linear_mask_schedule(3, mask_token_id=99)
    generator = t.Generator().manual_seed(123)
    result = apply_forward_noising(input_ids, t.tensor([0, 1, 2]), schedule, generator=generator)
    reference_result = reference.apply_forward_noising(
        input_ids,
        t.tensor([0, 1, 2]),
        schedule,
        generator=t.Generator().manual_seed(123),
    )
    t.testing.assert_close(
        result.noisy_tokens[0],
        input_ids[0],
        msg="Timestep 0 should leave every token unmasked when min_mask_prob is 0.",
    )
    assert bool(result.noisy_tokens[2].eq(99).all().item()), (
        "Final timestep should mask every token when max_mask_prob is 1."
    )
    t.testing.assert_close(
        result.noisy_tokens,
        reference_result.noisy_tokens,
        msg="Seeded forward noising should match the independent reference.",
    )
    t.testing.assert_close(
        result.mask,
        reference_result.mask,
        msg="Seeded noising masks should match the independent reference.",
    )
    print("All tests in `test_forward_noising_extremes_and_seeded_masks` passed!")


def test_masked_denoising_loss_uses_only_masked_positions(
    masked_denoising_loss: Callable | None = None,
):
    solutions = _solutions()
    masked_denoising_loss = masked_denoising_loss or solutions.masked_denoising_loss
    logits = t.zeros(1, 3, 4)
    target = t.tensor([[0, 1, 2]])
    mask = t.tensor([[True, False, True]])
    logits[0, 0, 0] = 5.0
    logits[0, 1, 3] = 20.0
    logits[0, 2, 2] = 5.0
    actual = masked_denoising_loss(logits, target, mask)
    expected = reference.masked_denoising_loss(logits, target, mask)
    t.testing.assert_close(
        actual,
        expected,
        msg="Masked denoising loss should ignore unmasked positions entirely.",
    )
    assert actual.item() < 0.03, (
        "Confident correct logits at masked positions should produce low loss."
    )
    print("All tests in `test_masked_denoising_loss_uses_only_masked_positions` passed!")


def test_confidence_remask_entropy_and_uniform_control(
    confidence_remask: Callable | None = None,
    token_entropy: Callable | None = None,
    uniform_remask: Callable | None = None,
):
    solutions = _solutions()
    confidence_remask = confidence_remask or solutions.confidence_remask
    token_entropy = token_entropy or solutions.token_entropy
    uniform_remask = uniform_remask or solutions.uniform_remask
    logits = t.tensor([[[5.0, 0.0], [0.0, 0.0], [6.0, 0.0], [0.1, 0.0]]])
    current = t.zeros(1, 4, dtype=t.long)
    remasked = confidence_remask(logits, current, mask_token_id=99, next_mask_fraction=0.5)
    entropy = token_entropy(logits)
    assert int(remasked.eq(99).sum().item()) == 2, (
        "Confidence remasking should mask the requested fraction of positions."
    )
    assert remasked[0, 0] != 99 and remasked[0, 2] != 99, (
        "High-confidence positions should remain committed."
    )
    t.testing.assert_close(
        entropy,
        reference.token_entropy(logits),
        msg="Token entropy should match categorical entropy from logits.",
    )
    uniform = uniform_remask(
        t.tensor([[1, 2, 3, 4]]),
        mask_token_id=99,
        next_mask_fraction=0.5,
        generator=t.Generator().manual_seed(0),
    )
    assert int(uniform.eq(99).sum().item()) == 2, (
        "Uniform remasking should mask the requested number of positions."
    )
    print("All tests in `test_confidence_remask_entropy_and_uniform_control` passed!")


def test_oracle_diffusion_sampler_recovers_target(
    diffusion_sampler: Callable | None = None,
    linear_mask_schedule: Callable | None = None,
):
    solutions = _solutions()
    diffusion_sampler = diffusion_sampler or solutions.diffusion_sampler
    linear_mask_schedule = linear_mask_schedule or solutions.linear_mask_schedule
    target = t.tensor([[1, 2, 3, 4]])
    schedule = linear_mask_schedule(4, mask_token_id=0)

    def oracle_model(tokens, step):
        _ = tokens, step
        logits = t.zeros(1, 4, 6)
        logits.scatter_(2, target.unsqueeze(-1), 10.0)
        return logits

    output, stats = diffusion_sampler(shape=(1, 4), schedule=schedule, model_fn=oracle_model)
    assert bool(t.equal(output, target)), (
        "Oracle denoiser should exactly reconstruct the target sequence."
    )
    assert len(stats) == schedule.num_steps, (
        "Sampler should report one stats object per denoising step."
    )
    assert stats[0].step == 3 and stats[-1].step == 0, (
        "Sampler should iterate from highest noise to lowest noise."
    )
    assert stats[-1].mask_fraction == 0.0, "Final sample should contain no mask tokens."
    print("All tests in `test_oracle_diffusion_sampler_recovers_target` passed!")


def test_commitment_edit_distance_and_activation_trajectory(
    commitment_times: Callable | None = None,
    stable_commitment_times: Callable | None = None,
    edit_distance: Callable | None = None,
    validate_activation_trajectory: Callable | None = None,
):
    solutions = _solutions()
    commitment_times = commitment_times or solutions.commitment_times
    stable_commitment_times = stable_commitment_times or solutions.stable_commitment_times
    edit_distance = edit_distance or solutions.edit_distance
    validate_activation_trajectory = (
        validate_activation_trajectory or solutions.validate_activation_trajectory
    )
    trajectory = t.tensor(
        [
            [[0, 0, 0]],
            [[1, 0, 0]],
            [[1, 2, 0]],
            [[1, 2, 3]],
        ]
    )
    activations = [t.zeros(1, 3, 4) for _ in range(4)]
    t.testing.assert_close(
        commitment_times(trajectory, mask_token_id=0),
        t.tensor([[1, 2, 3]]),
        msg="Commitment time should be the first trajectory index where a token is unmasked.",
    )
    unstable_trajectory = t.tensor(
        [
            [[1, 0, 0]],
            [[2, 2, 0]],
            [[1, 2, 0]],
            [[1, 2, 3]],
        ]
    )
    t.testing.assert_close(
        stable_commitment_times(
            unstable_trajectory,
            target_tokens=t.tensor([[1, 2, 3]]),
            mask_token_id=0,
        ),
        t.tensor([[2, 1, 3]]),
        msg="Stable commitment should ignore early guesses that are later changed or remasked.",
    )
    assert edit_distance([1, 2, 3], [1, 4, 3, 5]) == 2, (
        "Edit distance should count one substitution and one insertion in this example."
    )
    assert validate_activation_trajectory(
        activations,
        expected_steps=4,
        batch=1,
        seq_len=3,
    ), "Activation trajectory should validate when every step has matching leading dims."
    assert not validate_activation_trajectory(
        activations[:-1],
        expected_steps=4,
        batch=1,
        seq_len=3,
    ), "Activation trajectory should fail when the number of steps is wrong."
    print("All tests in `test_commitment_edit_distance_and_activation_trajectory` passed!")


def test_copy_pair_dataset_and_conditional_suffix_noising(
    copy_pair_dataset: Callable | None = None,
    conditional_suffix_noising: Callable | None = None,
    linear_mask_schedule: Callable | None = None,
):
    solutions = _solutions()
    copy_pair_dataset = copy_pair_dataset or solutions.copy_pair_dataset
    conditional_suffix_noising = (
        conditional_suffix_noising or solutions.conditional_suffix_noising
    )
    linear_mask_schedule = linear_mask_schedule or solutions.linear_mask_schedule
    clean = copy_pair_dataset(t.device("cpu"))
    assert clean.shape == (100, 6), "The exact grammar should enumerate all 100 prefixes."
    t.testing.assert_close(clean[:, 2], clean[:, 0])
    t.testing.assert_close(clean[:, 3], clean[:, 0])
    t.testing.assert_close(clean[:, 4], clean[:, 1])
    t.testing.assert_close(clean[:, 5], clean[:, 1])

    schedule = linear_mask_schedule(
        2,
        mask_token_id=10,
        min_mask_prob=0.0,
        max_mask_prob=1.0,
    )
    noisy, mask, timesteps = conditional_suffix_noising(clean[:8], schedule)
    t.testing.assert_close(noisy[:, :2], clean[:8, :2])
    assert not mask[:, :2].any(), "Condition tokens must never contribute to denoising loss."
    assert noisy[:, 2:].eq(10).all(), "At maximal noise every suffix token should be masked."
    assert mask[:, 2:].all(), "The suffix loss mask should match the maximal corruption."
    assert timesteps.eq(1).all(), "A two-step schedule has one nonzero training timestep."
    print("All tests in `test_copy_pair_dataset_and_conditional_suffix_noising` passed!")


def test_tiny_conditional_diffusion_lm_forward_shape(TinyConditionalDiffusionLM: type | None = None):
    solutions = _solutions()
    TinyConditionalDiffusionLM = TinyConditionalDiffusionLM or solutions.TinyConditionalDiffusionLM
    model = TinyConditionalDiffusionLM(d_model=32).eval()
    input_ids = t.tensor([[1, 2, 10, 10, 10, 10], [3, 4, 10, 10, 10, 10]])
    timesteps = t.tensor([0, 5])
    with t.inference_mode():
        logits = model(input_ids, timesteps)
    assert logits.shape == (2, 6, 11), (
        f"Diffusion LM logits should have shape (batch, seq, vocab), got {tuple(logits.shape)}."
    )
    print("All tests in `test_tiny_conditional_diffusion_lm_forward_shape` passed!")


def test_tiny_training_loop_learns_copy_pair_batch(
    train_tiny_diffusion_model: Callable | None = None,
    TinyConditionalDiffusionLM: type | None = None,
    copy_pair_dataset: Callable | None = None,
    linear_mask_schedule: Callable | None = None,
):
    solutions = _solutions()
    train_tiny_diffusion_model = (
        train_tiny_diffusion_model or solutions.train_tiny_diffusion_model
    )
    TinyConditionalDiffusionLM = TinyConditionalDiffusionLM or solutions.TinyConditionalDiffusionLM
    copy_pair_dataset = copy_pair_dataset or solutions.copy_pair_dataset
    linear_mask_schedule = linear_mask_schedule or solutions.linear_mask_schedule
    assert t.cuda.is_available(), "The section training exercise requires CUDA."
    device = t.device("cuda")
    rows = copy_pair_dataset(device)[:32]
    schedule = linear_mask_schedule(
        4,
        mask_token_id=10,
        min_mask_prob=0.25,
        max_mask_prob=1.0,
    )
    t.manual_seed(55)
    model = TinyConditionalDiffusionLM(num_steps=4, d_model=32).to(device)
    history = train_tiny_diffusion_model(
        model,
        rows,
        schedule,
        steps=120,
        record_every=20,
        seed=55,
    )
    assert history["steps"] == [1, 20, 40, 60, 80, 100, 120], (
        "The training helper should record the requested checkpoints, including the first step."
    )
    assert history["losses"][-1] <= 0.10, (
        "The tiny denoiser should fit the deterministic 32-example grammar batch."
    )
    assert history["losses"][-1] <= 0.10 * history["losses"][0], (
        "Training loss should fall by at least one order of magnitude."
    )
    print("All tests in `test_tiny_training_loop_learns_copy_pair_batch` passed!")


def test_conditional_diffusion_sample_preserves_prefix_and_records_steps(
    conditional_diffusion_sample: Callable | None = None,
    TinyConditionalDiffusionLM: type | None = None,
    copy_pair_dataset: Callable | None = None,
    linear_mask_schedule: Callable | None = None,
):
    solutions = _solutions()
    conditional_diffusion_sample = (
        conditional_diffusion_sample or solutions.conditional_diffusion_sample
    )
    TinyConditionalDiffusionLM = TinyConditionalDiffusionLM or solutions.TinyConditionalDiffusionLM
    copy_pair_dataset = copy_pair_dataset or solutions.copy_pair_dataset
    linear_mask_schedule = linear_mask_schedule or solutions.linear_mask_schedule
    rows = copy_pair_dataset(t.device("cpu"))[:2]
    schedule = linear_mask_schedule(
        4,
        mask_token_id=10,
        min_mask_prob=0.25,
        max_mask_prob=1.0,
    )
    model = TinyConditionalDiffusionLM(num_steps=4, d_model=32).eval()
    with t.inference_mode():
        sampled, trajectory, entropy, activations = conditional_diffusion_sample(
            model,
            rows,
            schedule,
        )
    assert sampled.shape == rows.shape, "Sampling must preserve the batch and sequence dimensions."
    assert trajectory.shape == (4, 2, 6), (
        "The trajectory should contain one complete token state per reverse step."
    )
    t.testing.assert_close(trajectory[:, :, :2], rows[None, :, :2].expand(4, -1, -1))
    assert len(entropy) == 4, "The sampler should record one entropy value per reverse step."
    assert len(activations) == 4, (
        "The sampler should retain one hidden-state tensor per reverse step."
    )
    print(
        "All tests in `test_conditional_diffusion_sample_preserves_prefix_and_records_steps` passed!"
    )
