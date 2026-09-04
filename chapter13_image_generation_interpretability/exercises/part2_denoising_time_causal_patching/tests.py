from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch as t


def _solutions():
    from chapter13_image_generation_interpretability.exercises.part2_denoising_time_causal_patching import (
        solutions,
    )

    return solutions


def _value(record, name: str):
    return record[name] if isinstance(record, dict) else getattr(record, name)


def test_diffusion_schedule_has_exact_endpoint_and_valid_noise(
    make_diffusion_schedule: Callable | None = None,
    q_sample: Callable | None = None,
):
    solutions = _solutions()
    make_diffusion_schedule = make_diffusion_schedule or solutions.make_diffusion_schedule
    q_sample = q_sample or solutions.q_sample
    schedule = make_diffusion_schedule(n_steps=4, beta_start=0.1, beta_end=0.4)
    assert schedule.betas.shape == (5,), (
        "A four-step schedule needs the exact clean endpoint plus four noisy endpoints."
    )
    t.testing.assert_close(schedule.alpha_bars[0], t.tensor(1.0))
    assert bool((schedule.alpha_bars[1:] < schedule.alpha_bars[:-1]).all()), (
        "Retained signal must decrease monotonically as forward-noising time increases."
    )
    clean = t.arange(24, dtype=t.float32).reshape(2, 3, 2, 2)
    noise = -t.ones_like(clean)
    t.testing.assert_close(q_sample(clean, t.zeros(2, dtype=t.long), noise, schedule), clean)
    print("All tests in `test_diffusion_schedule_has_exact_endpoint_and_valid_noise` passed!")


def test_rendered_world_has_exact_labels_and_masks(
    render_object_world: Callable | None = None,
):
    solutions = _solutions()
    render_object_world = render_object_world or solutions.render_object_world
    labels = t.tensor([[0, 0, 0, 0], [1, 2, 1, 1]])
    images, masks = render_object_world(labels, size=16)
    assert images.shape == (2, 3, 16, 16), (
        "Each label row should render one 16x16 RGB training image."
    )
    assert masks.shape == (2, 16, 16), (
        "Each rendered image needs an aligned exact object-region mask."
    )
    assert int(masks[0].sum()) == 25, (
        "The square fixture should occupy exactly its preregistered 5x5 support."
    )
    assert int(masks[1].sum()) == 21, (
        "The circle fixture should contain the 21 lattice points inside its radius."
    )
    t.testing.assert_close(images[0, :, masks[0]][:, 0], t.tensor([1.0, -1.0, -1.0]))
    t.testing.assert_close(images[1, :, masks[1]][:, 0], t.tensor([-1.0, -1.0, 1.0]))
    print("All tests in `test_rendered_world_has_exact_labels_and_masks` passed!")


def test_timestep_embedding_shape_and_zero_phase(
    sinusoidal_timestep_embedding: Callable | None = None,
):
    solutions = _solutions()
    function = sinusoidal_timestep_embedding or solutions.sinusoidal_timestep_embedding
    embeddings = function(t.tensor([0, 1, 7]), width=8)
    assert embeddings.shape == (3, 8), (
        "Every scalar timestep should map to one fixed-width conditioning vector."
    )
    t.testing.assert_close(embeddings[0, :4], t.zeros(4))
    t.testing.assert_close(embeddings[0, 4:], t.ones(4))
    with pytest.raises(ValueError, match="even"):
        function(t.tensor([1]), width=7)
    print("All tests in `test_timestep_embedding_shape_and_zero_phase` passed!")


def test_hookable_denoiser_shape_and_gradients(
    model_class=None,
    render_object_world: Callable | None = None,
):
    solutions = _solutions()
    model_class = model_class or solutions.HookableObjectDenoiser
    render_object_world = render_object_world or solutions.render_object_world
    model = model_class(width=12, concept_channels=4)
    labels = t.tensor([[0, 0, 0, 0], [1, 2, 1, 1]])
    images, _ = render_object_world(labels)
    prediction = model(images, t.tensor([1, 7]), labels)
    assert prediction.shape == images.shape, (
        "An x0-predicting denoiser must return one RGB tensor matching each input image."
    )
    prediction.square().mean().backward()
    assert model.shape_embedding.weight.grad is not None, (
        "The denoising objective must train the shape-conditioning pathway."
    )
    assert model.color_embedding.weight.grad is not None, (
        "The denoising objective must train the color-conditioning pathway."
    )
    print("All tests in `test_hookable_denoiser_shape_and_gradients` passed!")


def test_activation_cache_records_every_requested_time_and_layer(
    register_activation_cache: Callable | None = None,
):
    solutions = _solutions()
    register_activation_cache = register_activation_cache or solutions.register_activation_cache
    model = solutions.HookableObjectDenoiser(width=12, concept_channels=4).eval()
    cache = solutions.ActivationCache(values={}, timestep=3)
    handles = register_activation_cache(model, cache, layers=("concept", "late"))
    labels = t.tensor([[0, 0, 0, 0]])
    images, _ = solutions.render_object_world(labels)
    try:
        model(images, t.tensor([3]), labels)
    finally:
        for handle in handles:
            handle.remove()
    assert set(cache.values) == {(3, "concept"), (3, "late")}, (
        "The cache should contain exactly the requested named layers at the active time."
    )
    assert cache.values[(3, "concept")].shape == (1, 12, 16, 16), (
        "Cached activations must retain batch, channel, and spatial axes for patching."
    )
    print("All tests in `test_activation_cache_records_every_requested_time_and_layer` passed!")


def test_activation_replacement_changes_only_selected_coordinates(
    make_activation_replacement_hook: Callable | None = None,
):
    solutions = _solutions()
    function = make_activation_replacement_hook or solutions.make_activation_replacement_hook
    donor = t.arange(2 * 4 * 4 * 4, dtype=t.float32).reshape(2, 4, 4, 4)
    recipient = t.zeros_like(donor)
    mask = t.zeros(2, 4, 4, dtype=t.bool)
    mask[:, 1:3, 1:3] = True
    patched = function(donor, t.tensor([1, 3]), mask)(None, None, recipient)
    expected = recipient.clone()
    for batch in range(2):
        for channel in (1, 3):
            expected[batch, channel][mask[batch]] = donor[batch, channel][mask[batch]]
    t.testing.assert_close(patched, expected)
    t.testing.assert_close(recipient, t.zeros_like(recipient))
    print("All tests in `test_activation_replacement_changes_only_selected_coordinates` passed!")


def test_denoise_trajectory_is_deterministic_and_caches_steps(
    denoise_trajectory: Callable | None = None,
):
    solutions = _solutions()
    denoise_trajectory = denoise_trajectory or solutions.denoise_trajectory
    t.manual_seed(7)
    model = solutions.HookableObjectDenoiser(width=12, concept_channels=4).eval()
    schedule = solutions.make_diffusion_schedule()
    labels = t.tensor([[0, 0, 0, 0]])
    noise = t.randn(1, 3, 16, 16, generator=t.Generator().manual_seed(11))
    cache = solutions.ActivationCache(values={})
    first, states = denoise_trajectory(model, labels, noise, schedule, cache=cache)
    second, _ = denoise_trajectory(model, labels, noise, schedule)
    t.testing.assert_close(first, second)
    assert set(states) == set(range(solutions.N_DIFFUSION_STEPS + 1)), (
        "The saved trajectory should include x_T through the final x_0 state."
    )
    assert len(cache.values) == solutions.N_DIFFUSION_STEPS * len(solutions.PATCH_LAYERS), (
        "A complete causal sweep needs every layer-by-timestep activation cell."
    )
    print("All tests in `test_denoise_trajectory_is_deterministic_and_caches_steps` passed!")


def test_regional_causal_metric_has_exact_oracles(
    regional_causal_metric: Callable | None = None,
):
    solutions = _solutions()
    function = regional_causal_metric or solutions.regional_causal_metric
    clean = t.ones(1, 3, 4, 4)
    corrupt = t.zeros_like(clean)
    mask = t.zeros(1, 4, 4, dtype=t.bool)
    mask[:, :2, :2] = True
    patched = corrupt.clone()
    patched[:, :, :2, :2] = 1.0
    report = function(clean, corrupt, patched, mask)
    assert _value(report, "recovery") == pytest.approx(1.0), (
        "Exact target-region copying should define one unit of regional recovery."
    )
    assert _value(report, "outside_change") == pytest.approx(0.0), (
        "The exact regional oracle must preserve every off-target pixel."
    )
    assert _value(report, "selectivity") == pytest.approx(1.0), (
        "Perfect recovery with no spillover should have unit causal selectivity."
    )
    print("All tests in `test_regional_causal_metric_has_exact_oracles` passed!")


def test_center_mask_is_preregistered_and_centered(
    make_center_mask: Callable | None = None,
):
    solutions = _solutions()
    function = make_center_mask or solutions.make_center_mask
    mask = function(8, 10, fraction=0.5)
    assert mask.shape == (8, 10), (
        "The preregistered control mask must preserve the requested feature-map shape."
    )
    assert int(mask.sum()) == 20, (
        "A half-height by half-width mask should select exactly one quarter of this grid."
    )
    assert bool(mask[2:6, 2:7].all()), (
        "The selected control region should be centered, not chosen after seeing outputs."
    )
    assert not bool(mask[:2].any()), (
        "Rows outside the centered region must remain unselected."
    )
    with pytest.raises(ValueError, match="fraction"):
        function(8, 8, fraction=1.0)
    print("All tests in `test_center_mask_is_preregistered_and_centered` passed!")


def test_latent_patch_preserves_the_complement_exactly(
    apply_latent_patch: Callable | None = None,
):
    solutions = _solutions()
    function = apply_latent_patch or solutions.apply_latent_patch
    recipient = t.zeros(1, 4, 6, 6)
    donor = t.ones_like(recipient)
    mask = solutions.make_center_mask(6, 6, fraction=0.5)
    patched = function(recipient, donor, mask, mix=0.25)
    selector = mask[None, None].expand_as(patched)
    t.testing.assert_close(patched[selector], t.full_like(patched[selector], 0.25))
    t.testing.assert_close(patched[~selector], recipient[~selector])
    print("All tests in `test_latent_patch_preserves_the_complement_exactly` passed!")


def test_random_latent_patch_is_seeded_and_same_size(
    apply_random_latent_patch: Callable | None = None,
):
    solutions = _solutions()
    function = apply_random_latent_patch or solutions.apply_random_latent_patch
    recipient = t.zeros(1, 4, 8, 8)
    donor = t.randn_like(recipient)
    mask = solutions.make_center_mask(8, 8, fraction=0.5)
    first = function(recipient, donor, mask, seed=19)
    second = function(recipient, donor, mask, seed=19)
    selector = mask[None, None].expand_as(first)
    t.testing.assert_close(first, second)
    t.testing.assert_close(first[~selector], recipient[~selector])
    assert float(first[selector].std()) > 0.1, (
        "The same-size random control should contain non-degenerate donor-scale variation."
    )
    print("All tests in `test_random_latent_patch_is_seeded_and_same_size` passed!")


def test_calibrated_recovery_anchors_recipient_and_donor(
    calibrated_recovery: Callable | None = None,
):
    solutions = _solutions()
    function = calibrated_recovery or solutions.calibrated_recovery
    assert function(-2.0, 3.0, -2.0) == pytest.approx(0.0), (
        "The recipient score must anchor calibrated recovery at zero."
    )
    assert function(-2.0, 3.0, 3.0) == pytest.approx(1.0), (
        "The donor score must anchor calibrated recovery at one."
    )
    assert function(-2.0, 3.0, 0.5) == pytest.approx(0.5), (
        "The midpoint between recipient and donor must calibrate to one half."
    )
    with pytest.raises(ValueError, match="nonzero"):
        function(1.0, 1.0, 1.0)
    print("All tests in `test_calibrated_recovery_anchors_recipient_and_donor` passed!")


def test_latent_transfer_metric_rewards_regional_transfer(
    latent_transfer_metric: Callable | None = None,
):
    solutions = _solutions()
    function = latent_transfer_metric or solutions.latent_transfer_metric
    donor = t.ones(1, 3, 8, 8)
    recipient = t.zeros_like(donor)
    mask = solutions.make_center_mask(8, 8, fraction=0.5)[None]
    patched = recipient.clone()
    patched[:, :, mask[0]] = 1.0
    report = function(donor, recipient, patched, mask)
    assert _value(report, "recovery") == pytest.approx(1.0), (
        "Copying the donor region exactly should achieve full latent-transfer recovery."
    )
    assert _value(report, "outside_change") == pytest.approx(0.0), (
        "The latent patch oracle should preserve the recipient outside the region."
    )
    assert _value(report, "selectivity") == pytest.approx(1.0), (
        "Perfect donor transfer without spillover should have unit selectivity."
    )
    print("All tests in `test_latent_transfer_metric_rewards_regional_transfer` passed!")


def test_real_sd15_case_contract_is_pinned_and_counterfactual():
    solutions = _solutions()
    assert solutions.REAL_SD15_MODEL_ID == "stable-diffusion-v1-5/stable-diffusion-v1-5", (
        "The required real-model gate must use the preregistered SD1.5 repository."
    )
    assert len(solutions.REAL_SD15_REVISION) == 40, (
        "The SD1.5 extension must pin an immutable 40-character commit revision."
    )
    assert len(solutions.REAL_SD15_CASES) == 2, (
        "Both red-to-blue and blue-to-red counterfactual cases are required."
    )
    for case in solutions.REAL_SD15_CASES:
        assert case["donor_prompt"] != case["recipient_prompt"], (
            "A causal donor/recipient pair must differ in the target concept."
        )
        assert case["donor_text"] != case["recipient_text"], (
            "CLIP scoring labels must preserve the same donor/recipient contrast."
        )
        assert case["target_color"] in {"red", "blue"}, (
            "Each case must preregister which color is expected to transfer."
        )
        assert isinstance(case["seed"], int), (
            "Every real-model case needs an explicit matched latent-noise seed."
        )


def test_sd15_trajectory_rejects_invalid_step_count(
    run_sd15_latent_trajectory: Callable | None = None,
):
    solutions = _solutions()
    function = run_sd15_latent_trajectory or solutions.run_sd15_latent_trajectory
    with pytest.raises(ValueError, match="at least two"):
        function(
            None,
            "donor",
            "negative",
            t.zeros(1, 4, 8, 8),
            num_inference_steps=1,
        )
    print("All tests in `test_sd15_trajectory_rejects_invalid_step_count` passed!")


def validate_real_sd15_signature(result: dict):
    assert result["accepted"], "The real causal result must pass every preregistered gate."
    assert result["model_id"] == _solutions().REAL_SD15_MODEL_ID, (
        "The reported model must match the pinned lockfile identifier."
    )
    assert result["revision"] == _solutions().REAL_SD15_REVISION, (
        "The reported revision must match the preregistered immutable commit."
    )
    assert result["within_vram_budget"], (
        "The real experiment must remain within the declared 24 GiB budget."
    )
    assert len(result["cases"]) == 2, (
        "The real signature requires both counterfactual color-transfer cases."
    )
    for case in result["cases"]:
        report = case["report"]
        assert _value(report, "target_beats_controls"), (
            "The selected donor patch must beat every matched real-model control."
        )
        assert _value(report, "best_clip_recovery") >= 0.10, (
            "The donor intervention must recover at least 10% of the CLIP concept contrast."
        )
        assert _value(report, "best_regional_selectivity") >= 0.05, (
            "The decoded image change must be measurably stronger inside the target region."
        )
        assert len(case["clip_recoveries"]) == result["num_inference_steps"] - 1, (
            "The real sweep must score every non-initial denoising intervention time."
        )
    print("All tests in `validate_real_sd15_signature` passed!")


@pytest.mark.slow
def test_trained_toy_causal_sweep_beats_preregistered_controls():
    solutions = _solutions()
    result = solutions.run_toy_experiment(training_steps=350, seed=0)
    metrics = solutions.serializable_metrics(result)
    assert metrics["accepted"], (
        "The executed model-organism signature must pass every preregistered gate."
    )
    assert metrics["training_loss_ratio"] < 0.10, (
        "End-to-end training should reduce weighted denoising loss by at least 90%."
    )
    assert metrics["best_layer"] == "concept", (
        "The preregistered feature should localize at the planted concept boundary."
    )
    assert metrics["best_timestep"] == 1, (
        "The pinned seed should localize strongest causal transfer at the final cleanup step."
    )
    assert metrics["best_selectivity"] > 0.85, (
        "The target activation patch should strongly recover the object without broad damage."
    )
    controls = metrics["controls"]
    assert controls["matched_seed_unpatched"]["selectivity"] == pytest.approx(0.0), (
        "The matched-seed unpatched trajectory must define zero recovery."
    )
    assert controls["same_size_random_channels"]["selectivity"] < 0.10, (
        "Equal-count non-concept channels should not reproduce target transfer."
    )
    assert controls["same_size_random_location"]["selectivity"] < 0.0, (
        "Moving the donor patch to the opposite region should be actively nonselective."
    )
    assert controls["wrong_timestep"]["selectivity"] < 0.02, (
        "The opposite-end denoising time should have negligible target selectivity."
    )
    assert controls["shuffled_labels"]["selectivity"] < 0.02, (
        "An unrelated donor label should not transfer the preregistered target."
    )
    assert controls["untrained_model"]["selectivity"] < 0.0, (
        "The architecture alone should not create the learned causal effect."
    )


def test_notebook_contract(run_smoke_test: Callable | None = None):
    solutions = _solutions()
    run_smoke_test = run_smoke_test or solutions.run_smoke_test
    smoke = run_smoke_test(cpu=True)
    assert smoke["tests_passed"], (
        "The smoke contract must verify the exact clean endpoint."
    )
    assert smoke["dataset_size"] == 24, (
        "The generated world should enumerate all 2x3x2x2 labeled states."
    )
    assert smoke["image_shape"] == [2, 3, 16, 16], (
        "The smoke pair should contain two RGB 16x16 images."
    )
    assert smoke["alpha_bar_t0"] == pytest.approx(1.0), (
        "The analytic diffusion schedule must retain all signal at timestep zero."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_section_file_contract():
    section = Path(__file__).resolve().parent
    required = {
        "13.2_Denoising-Time_Causal_Patching_exercises.ipynb",
        "13.2_Denoising-Time_Causal_Patching_solutions.ipynb",
        "README.md",
        "artifacts.lock.yml",
        "solutions.py",
        "tests.py",
        "expected_outputs/README.md",
        "expected_outputs/reference_metrics.json",
        "expected_outputs/smoke_test.json",
    }
    missing = sorted(path for path in required if not (section / path).exists())
    assert not missing, f"13.2 is missing required learner files: {missing}"


# These named entrypoints are called immediately after the corresponding learner
# exercises. They preserve the compact pytest tests above while adding exact,
# explanatory feedback at the notebook boundary.
def test_render_object_world_has_exact_state_semantics(
    render_object_world: Callable | None = None,
):
    test_rendered_world_has_exact_labels_and_masks(render_object_world)


def test_q_sample_matches_closed_form_endpoints(q_sample: Callable | None = None):
    solutions = _solutions()
    q_sample = q_sample or solutions.q_sample
    test_diffusion_schedule_has_exact_endpoint_and_valid_noise(q_sample=q_sample)
    schedule = solutions.make_diffusion_schedule()
    clean = t.full((2, 3, 4, 4), 0.25)
    noise = t.full_like(clean, -0.5)
    timesteps = t.tensor([3, 7])
    alpha_bar = schedule.alpha_bars[timesteps].view(-1, 1, 1, 1)
    expected = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
    assert t.allclose(q_sample(clean, timesteps, noise, schedule), expected, atol=1e-7), (
        "Forward noising must match the analytic marginal at arbitrary timesteps."
    )


def test_sinusoidal_timestep_embedding_has_paired_coordinates(
    sinusoidal_timestep_embedding: Callable | None = None,
):
    test_timestep_embedding_shape_and_zero_phase(sinusoidal_timestep_embedding)
    function = sinusoidal_timestep_embedding or _solutions().sinusoidal_timestep_embedding
    embedding = function(t.tensor([0, 1, 8]), width=12)
    assert not t.allclose(embedding[1], embedding[2]), (
        "Different denoising times must receive distinguishable conditioning vectors."
    )


def test_hookable_denoiser_exposes_the_planted_concept_location(
    model_class=None,
    register_activation_cache: Callable | None = None,
):
    solutions = _solutions()
    model_class = model_class or solutions.HookableObjectDenoiser
    register_activation_cache = (
        register_activation_cache or solutions.register_activation_cache
    )
    t.manual_seed(0)
    model = model_class(width=12, concept_channels=4).eval()
    noisy = t.randn(2, 3, 16, 16)
    noisy[1] = noisy[0]
    labels = t.tensor([[0, 0, 0, 0], [0, 2, 0, 0]])
    cache = solutions.ActivationCache(values={}, timestep=4)
    handles = register_activation_cache(model, cache)
    try:
        prediction = model(noisy, t.tensor([4, 4]), labels)
    finally:
        for handle in handles:
            handle.remove()
    assert prediction.shape == noisy.shape, (
        "The denoiser must predict one clean RGB tensor for every noisy input."
    )
    early_difference = cache.values[(4, "early")][0] - cache.values[(4, "early")][1]
    assert t.equal(early_difference, t.zeros_like(early_difference)), (
        "Equal noisy inputs must remain equal before label-condition injection."
    )
    concept_difference = cache.values[(4, "concept")][0] - cache.values[(4, "concept")][1]
    mask = solutions.render_object_masks(labels[:1])[0]
    assert concept_difference[:4, mask].abs().sum() > 0, (
        "Changing color must alter the planted concept channels inside the object."
    )
    assert t.equal(concept_difference[4:], t.zeros_like(concept_difference[4:])), (
        "The exact bottleneck reserves only the first concept channels."
    )
    assert t.equal(
        concept_difference[:4, ~mask], t.zeros_like(concept_difference[:4, ~mask])
    ), "The planted object feature must not leak outside its exact spatial support."
    print(
        "All tests in `test_hookable_denoiser_exposes_the_planted_concept_location` passed!"
    )


def test_activation_replacement_changes_only_requested_entries(
    make_activation_replacement_hook: Callable | None = None,
):
    test_activation_replacement_changes_only_selected_coordinates(
        make_activation_replacement_hook
    )


def test_denoise_trajectory_is_matched_seed_and_cache_complete(
    denoise_trajectory: Callable | None = None,
):
    test_denoise_trajectory_is_deterministic_and_caches_steps(denoise_trajectory)


def test_regional_causal_metric_has_exact_anchors(
    regional_causal_metric: Callable | None = None,
):
    solutions = _solutions()
    function = regional_causal_metric or solutions.regional_causal_metric
    clean = t.ones(1, 3, 4, 4)
    corrupt = t.zeros_like(clean)
    mask = t.zeros(1, 4, 4, dtype=t.bool)
    mask[:, 1:3, 1:3] = True
    half_recovered = corrupt.clone()
    half_recovered[:, :, 1:3, 1:3] = 0.5
    report = function(clean, corrupt, half_recovered, mask)
    assert _value(report, "recovery") == pytest.approx(0.75), (
        "Moving target pixels halfway to clean should remove exactly 75% of squared error."
    )
    assert _value(report, "outside_change") == pytest.approx(0.0), (
        "A perfectly localized intervention should have zero off-target change."
    )
    baseline = function(clean, corrupt, corrupt, mask)
    assert _value(baseline, "recovery") == pytest.approx(0.0), (
        "The matched-seed unpatched recipient defines the zero-recovery anchor."
    )
    print("All tests in `test_regional_causal_metric_has_exact_anchors` passed!")


def test_train_object_denoiser_reduces_real_image_loss(
    train_object_denoiser: Callable | None = None,
):
    train_object_denoiser = (
        train_object_denoiser or _solutions().train_object_denoiser
    )
    model, trace = train_object_denoiser(
        device="cpu", steps=140, batch_size=32, seed=0
    )
    assert sum(parameter.numel() for parameter in model.parameters()) > 1_000, (
        "The exercise must optimize a real multi-layer denoiser, not a scalar oracle."
    )
    assert trace.final_loss < trace.first_loss * 0.2, (
        "End-to-end training should reduce weighted image loss by at least 80%."
    )
    assert len(trace.losses) == 140, (
        "Every optimization step should remain visible for the learner's loss curve."
    )
    print("All tests in `test_train_object_denoiser_reduces_real_image_loss` passed!")


def test_end_to_end_training_and_signature_are_meaningful(signature_result=None):
    result = signature_result or _solutions().run_toy_experiment(
        device="cpu", training_steps=350, seed=0
    )
    trace = result["training_trace"]
    assert trace.final_loss < trace.first_loss * 0.05, (
        "End-to-end denoising training should reduce weighted image loss by at least 95%."
    )
    assert result["accepted"], (
        "The signature result must clear its preregistered loss and control margins."
    )
    assert result["best_layer"] == "concept", (
        "The target sweep is preregistered at the exact object-feature bottleneck."
    )
    assert result["best_metric"].recovery > 0.85, (
        "The activation patch should visibly recover most of the clean target region."
    )
    assert result["best_metric"].selectivity > 0.8, (
        "Target recovery must remain localized rather than rewrite the whole image."
    )
    print("All tests in `test_end_to_end_training_and_signature_are_meaningful` passed!")


def test_causal_controls_fail_for_the_right_reasons(signature_result=None):
    result = signature_result or _solutions().run_toy_experiment(
        device="cpu", training_steps=350, seed=0
    )
    target = result["best_metric"].selectivity
    controls = result["controls"]
    assert controls["matched_seed_unpatched"].selectivity == pytest.approx(0.0), (
        "The unpatched matched-seed trajectory must define the zero-effect baseline."
    )
    for name in (
        "same_size_random_channels",
        "same_size_random_location",
        "wrong_timestep",
        "shuffled_labels",
        "untrained_model",
    ):
        assert target > controls[name].selectivity + 0.7, (
            f"The target patch must beat the preregistered {name} control by 0.7."
        )
    assert controls["pixel_patch_upper_bound"].recovery == pytest.approx(1.0), (
        "Direct target-pixel copying must remain the exact recovery upper bound."
    )
    print("All tests in `test_causal_controls_fail_for_the_right_reasons` passed!")


def test_cuda_entrypoints_have_no_cpu_fallback():
    source = inspect.getsource(_solutions()._cuda_report)
    assert "if not t.cuda.is_available()" in source and "raise RuntimeError" in source, (
        "The CUDA verification path must fail explicitly when CUDA is unavailable."
    )
    assert 'run_toy_experiment(device="cuda"' in source, (
        "The CUDA path must train and evaluate the actual denoiser on the GPU."
    )
    assert 'device="cpu"' not in source, (
        "A hidden CPU fallback would make the CUDA evidence ambiguous."
    )
    assert "max_memory_allocated" in source, (
        "The CUDA path must measure peak allocation against the 24 GiB contract."
    )
    print("All tests in `test_cuda_entrypoints_have_no_cpu_fallback` passed!")


def test_gpu_result_packages_real_numeric_acceptance_metrics():
    solutions = _solutions()
    reports = [
        solutions.SD15PatchControlReport(
            best_step_index=2,
            best_scheduler_timestep=851,
            best_clip_recovery=recovery,
            best_regional_selectivity=selectivity,
            wrong_timestep_recovery=wrong_time,
            wrong_region_recovery=-0.2,
            random_latent_recovery=random,
            unpatched_recovery=0.0,
            target_beats_controls=True,
        )
        for recovery, selectivity, wrong_time, random in (
            (0.65, 0.22, 0.46, 0.21),
            (0.50, 0.24, 0.29, 0.15),
        )
    ]
    toy = {
        "accepted": True,
        "peak_vram_gb": 0.1,
        "max_vram_gb": 24.0,
        "controls": {
            "same_size_random_channels": {"selectivity": 0.01},
            "wrong_timestep": {"selectivity": 0.0},
            "shuffled_labels": {"selectivity": 0.0},
            "untrained_model": {"selectivity": -0.1},
            "pixel_patch_upper_bound": {"recovery": 1.0},
        },
    }
    real = {
        "model_id": solutions.REAL_SD15_MODEL_ID,
        "revision": solutions.REAL_SD15_REVISION,
        "clip_model_id": solutions.REAL_CLIP_MODEL_ID,
        "clip_revision": solutions.REAL_CLIP_REVISION,
        "torch_version": "test",
        "cuda_version": "test",
        "gpu_name": "test",
        "num_inference_steps": 20,
        "peak_vram_gb": 3.2,
        "runtime_seconds": 1.0,
        "within_vram_budget": True,
        "accepted": True,
        "cases": [
            {
                "case": {"case_id": f"case_{index}", "seed": index},
                "clip_recoveries": [report.best_clip_recovery],
                "regional_metrics": [
                    solutions.LatentTransferMetric(1.0, 0.5, 0.5, 0.1, 0.4)
                ],
                "report": report,
            }
            for index, report in enumerate(reports)
        ],
    }
    packaged = solutions._package_gpu_result(toy, real)
    assert packaged["cuda_available"] is True
    assert packaged["peak_vram_gb"] == pytest.approx(3.2)
    assert packaged["dataset_size"] == 24
    assert packaged["layer_count"] == 4
    assert packaged["diffusion_steps"] == 8
    assert packaged["heatmap_cell_count"] == 32
    assert packaged["random_channel_selectivity"] == pytest.approx(0.01)
    assert packaged["wrong_timestep_selectivity"] == pytest.approx(0.0)
    assert packaged["shuffled_label_selectivity"] == pytest.approx(0.0)
    assert packaged["untrained_model_selectivity"] == pytest.approx(-0.1)
    assert packaged["pixel_patch_recovery"] == pytest.approx(1.0)
    assert packaged["sd15_case_count"] == 2
    assert packaged["sd15_num_inference_steps"] == 20
    assert packaged["sd15_min_best_clip_recovery"] == pytest.approx(0.50)
    assert packaged["sd15_min_best_regional_selectivity"] == pytest.approx(0.22)
    assert packaged["sd15_min_target_control_margin"] == pytest.approx(0.19)
    assert packaged["sd15_accepted"] is True and packaged["accepted"] is True
    assert "visual_payload" not in packaged, (
        "Verification reports must keep image and latent payloads in the figure artifact, "
        "not serialize them into the numeric evidence report."
    )


def test_notebook_contract_is_learner_visible():
    path = Path(__file__).with_name(
        "13.2_Denoising-Time_Causal_Patching_exercises.ipynb"
    )
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    required = (
        "def q_sample",
        "def sinusoidal_timestep_embedding",
        "class HookableObjectDenoiser",
        "def register_activation_cache",
        "def make_activation_replacement_hook",
        "def denoise_trajectory",
        "def regional_causal_metric",
        "def causal_patch_sweep",
        "def run_gpu_test(max_vram_gb: float = 24.0)",
        "def run_full_experiment(max_vram_gb: float = 24.0)",
        "## Try It Yourself",
        "## Anomaly hunting",
        "Common bug",
        "<summary>Expected output</summary>",
        "<summary>Interpretation</summary>",
        "<summary>Solution</summary>",
    )
    missing = [needle for needle in required if needle not in source]
    assert not missing, (
        "The learner notebook is missing visible pedagogy or method code: "
        + ", ".join(missing)
    )
    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    assert "verification_report.json" not in code_source, (
        "The learner result must be generated in notebook code, not loaded from a report."
    )
    assert "run_toy_experiment(" not in source, (
        "The notebook must visibly assemble training and patching, not call one hidden runner."
    )
    assert source.count("tests.test_") >= 9, (
        "Hard exercises need immediate semantic subfunction tests."
    )
    print("All tests in `test_notebook_contract_is_learner_visible` passed!")
