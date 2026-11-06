"""13.2 Denoising-time causal patching: tiny CPU tensor oracles, a stub DDIM loop and cache-hook checks."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

MODULE = "chapter13_image_generation_interpretability.exercises.part2_denoising_time_causal_patching"


@pytest.fixture(scope="module")
def methods():
    # COLOR_VALUES is created at import by the byte-preserved legacy module.
    # Reproduce its historical float32 process default, then restore ambient state.
    import torch
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        return importlib.import_module(MODULE + ".solutions")
    finally:
        torch.set_default_dtype(previous)


@pytest.fixture
def cpu_state():
    import torch
    previous_dtype, previous_threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        yield torch
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


TENSOR_ORACLES = [
    ("test_diffusion_schedule_has_exact_endpoint_and_valid_noise", ("make_diffusion_schedule", "q_sample")),
    ("test_q_sample_matches_closed_form_endpoints", ("q_sample",)),
    ("test_rendered_world_has_exact_labels_and_masks", ("render_object_world",)),
    ("test_sinusoidal_timestep_embedding_has_paired_coordinates", ("sinusoidal_timestep_embedding",)),
    ("test_activation_replacement_changes_only_selected_coordinates", ("make_activation_replacement_hook",)),
    ("test_regional_causal_metric_has_exact_oracles", ("regional_causal_metric",)),
    ("test_regional_causal_metric_has_exact_anchors", ("regional_causal_metric",)),
    ("test_center_mask_is_preregistered_and_centered", ("make_center_mask",)),
    ("test_latent_patch_preserves_the_complement_exactly", ("apply_latent_patch",)),
    ("test_random_latent_patch_is_seeded_and_same_size", ("apply_random_latent_patch",)),
    ("test_calibrated_recovery_anchors_recipient_and_donor", ("calibrated_recovery",)),
    ("test_latent_transfer_metric_rewards_regional_transfer", ("latent_transfer_metric",)),
    ("test_sd15_trajectory_rejects_invalid_step_count", ("run_sd15_latent_trajectory",)),
    ("test_real_sd15_case_contract_is_pinned_and_counterfactual", ()),
]


@pytest.mark.parametrize("oracle,names", TENSOR_ORACLES)
def test_generated_methods_against_existing_independent_tensor_oracles(methods, cpu_state, oracle, names):
    checks = importlib.import_module(MODULE + ".tests")
    getattr(checks, oracle)(*[getattr(methods, name) for name in names])


def test_native_ddim_loop_with_independent_two_step_cfg_and_patch_oracle(methods, cpu_state, monkeypatch):
    t = cpu_state
    class Scheduler:
        def set_timesteps(self, count, device):
            self.timesteps = t.arange(count - 1, -1, -1, device=device)
        def scale_model_input(self, value, timestep):
            return value
        def step(self, noise, timestep, current, eta, return_dict):
            assert eta == 0.0 and return_dict is False
            return (current - noise,)
    def encode_prompt(**kwargs):
        return t.tensor([[2.0]]), t.tensor([[1.0]])
    def unet(value, timestep, encoder_hidden_states, return_dict):
        assert encoder_hidden_states.tolist() == [[1.0], [2.0]]
        return (t.cat([t.ones_like(value[:1]), 2 * t.ones_like(value[:1])]),)
    pipe = SimpleNamespace(scheduler=Scheduler(), encode_prompt=encode_prompt, unet=unet)
    monkeypatch.setattr(methods, "decode_sd15_latents", lambda _, x: (None, x[:, :3].clone()))
    initial = t.zeros(1, 4, 2, 2)
    baseline = methods.run_sd15_latent_trajectory(pipe, "positive", "negative", initial, num_inference_steps=2, guidance_scale=3)
    # 1 + 3*(2-1) = 4; two deterministic scheduler updates yield -8.
    assert t.equal(baseline.latent_states[-1], t.full_like(initial, -8))
    assert len(baseline.latent_states) == 3 and baseline.scheduler_timesteps == (1, 0)
    mask = t.tensor([[True, False], [False, False]])
    donor = (initial.clone(), t.full_like(initial, 10), t.full_like(initial, 20))
    patched = methods.run_sd15_latent_trajectory(pipe, "positive", "negative", initial, num_inference_steps=2, guidance_scale=3, patch_step_index=1, donor_states=donor, spatial_mask=mask)
    assert t.equal(patched.latent_states[-1][:, :, 0, 0], t.full((1, 4), 6.0))
    assert t.equal(patched.latent_states[-1][:, :, 1, 1], t.full((1, 4), -8.0))
    assert t.equal(initial, t.zeros_like(initial)) and t.equal(donor[1], t.full_like(initial, 10))


def test_cache_hooks_preserve_time_layer_identity_and_detach_copies(methods, cpu_state):
    t = cpu_state
    layers = {name: t.nn.Identity() for name in methods.PATCH_LAYERS}
    fixture = SimpleNamespace(patch_layer=layers.__getitem__)
    cache = methods.ActivationCache(values={}, timestep=3)
    handles = methods.register_activation_cache(fixture, cache)
    try:
        x = t.ones(1, 2, 2, 2, requires_grad=True)
        for name, layer in layers.items():
            layer(x)
        assert set(cache.values) == {(3, name) for name in layers}
        x.detach().fill_(5)
        assert all(t.equal(value, t.ones_like(value)) and not value.requires_grad for value in cache.values.values())
    finally:
        for handle in handles:
            handle.remove()
    assert all(not layer._forward_hooks for layer in layers.values())
