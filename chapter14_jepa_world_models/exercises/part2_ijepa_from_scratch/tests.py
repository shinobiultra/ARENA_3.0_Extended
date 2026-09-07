"""Visible tests for [14.2] I-JEPA from Scratch.

Each test accepts the learner's implementation as an optional argument, which
lets notebook cells test one sub-function immediately after it is written.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from dataclasses import replace
from pathlib import Path

import torch as t
from torch import nn

from . import solutions


def test_build_scene_dataset(fn=solutions.build_scene_dataset) -> None:
    scenes = fn(examples_per_combination=2, seed=3)
    assert scenes.images.shape == (24, 3, 32, 32), (
        "Expected 3 objects x 4 positions x 2 examples. Check the loop order and CHW layout."
    )
    assert t.equal(t.bincount(scenes.object_labels), t.tensor([8, 8, 8])), (
        "Each object class should appear eight times; check that every object-position pair is generated."
    )
    assert t.equal(t.bincount(scenes.position_labels), t.tensor([6, 6, 6, 6])), (
        "Each position class should appear six times; check that the generated world is balanced."
    )
    assert scenes.images.min() >= 0 and scenes.images.max() <= 1, (
        "Generated RGB values must stay in [0, 1]. Clamp only after drawing the scene."
    )
    cue_patches = scenes.images[:, :, :4, 12:16].mean(dim=(-1, -2))
    t.testing.assert_close(cue_patches[0], t.tensor([0.95, 0.10, 0.10]))
    t.testing.assert_close(cue_patches[-1], t.tensor([0.10, 0.90, 0.32]))
    pixel_masks = solutions.object_target_masks(scenes.position_labels).view(-1, 8, 8)
    pixel_masks = pixel_masks.repeat_interleave(4, 1).repeat_interleave(4, 2)
    cue_mask = t.zeros(32, 32, dtype=t.bool)
    cue_mask[:4, 12:16] = True
    for image, colour, target_mask in zip(scenes.images, cue_patches, pixel_masks):
        class_pixels = image.sub(colour[:, None, None]).abs().amax(dim=0) < 1e-6
        assert not class_pixels[~(target_mask | cue_mask)].any(), (
            "Every class-coloured object pixel must be hidden by the target mask; "
            "otherwise the predictor can read object identity outside the context key."
        )
    print("All tests in `test_build_scene_dataset` passed!")


def test_replace_scene_keys(fn=solutions.replace_scene_keys) -> None:
    scenes = solutions.build_scene_dataset(2, seed=11)
    source_indices = solutions.cyclic_object_mismatch_indices(
        scenes.object_labels, scenes.position_labels, shift=1
    )
    swapped = fn(scenes, source_indices)
    key_mask = t.zeros_like(scenes.images, dtype=t.bool)
    key_mask[:, :, :4, 12:16] = True
    t.testing.assert_close(swapped.images[~key_mask], scenes.images[~key_mask], rtol=0, atol=0)
    t.testing.assert_close(
        swapped.images[:, :, :4, 12:16],
        scenes.images[source_indices, :, :4, 12:16],
        rtol=0,
        atol=0,
    )
    assert t.equal(swapped.object_labels, scenes.object_labels), (
        "A key intervention must preserve the target object labels."
    )
    assert t.equal(swapped.position_labels, scenes.position_labels), (
        "A key intervention must preserve every target position."
    )
    print("All tests in `test_replace_scene_keys` passed!")


def test_patchify_hand_computed(fn=solutions.patchify) -> None:
    image = t.arange(16.0).view(1, 1, 4, 4)
    actual = fn(image, patch_size=2)
    expected = t.tensor([[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]]).float()
    t.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("All tests in `test_patchify_hand_computed` passed!")


def test_erase_scene_keys(fn=solutions.erase_scene_keys) -> None:
    scenes = solutions.build_scene_dataset(2, seed=19)
    original = scenes.images.clone()
    erased = fn(scenes)
    keep = t.ones_like(original, dtype=t.bool)
    keep[:, :, :4, 12:16] = False
    t.testing.assert_close(erased.images[keep], original[keep], rtol=0, atol=0)
    background = original[:, :, :4, :4].mean((1, 2, 3), keepdim=True)
    t.testing.assert_close(
        erased.images[:, :, :4, 12:16], background.expand(-1, 3, 4, 4), rtol=0, atol=0
    )
    t.testing.assert_close(scenes.images, original, rtol=0, atol=0)
    assert t.equal(erased.object_labels, scenes.object_labels), "Erasure must preserve the original target class."
    assert t.equal(erased.position_labels, scenes.position_labels), "Erasure must preserve the original target position."
    print("All tests in `test_erase_scene_keys` passed!")


def test_patchify_rejects_bad_shape(fn=solutions.patchify) -> None:
    try:
        fn(t.zeros(2, 3, 31, 32), patch_size=4)
    except ValueError as error:
        assert "divisible" in str(error), (
            "The ValueError should explain that image dimensions must be divisible by patch_size."
        )
    else:
        raise AssertionError("Non-divisible image sizes must raise a useful ValueError.")
    print("All tests in `test_patchify_rejects_bad_shape` passed!")


def test_sample_block_masks(fn=solutions.sample_block_masks) -> None:
    first = fn(12, grid_size=8, block_size=3, generator=t.Generator().manual_seed(11))
    second = fn(12, grid_size=8, block_size=3, generator=t.Generator().manual_seed(11))
    assert t.equal(first, second), "A fixed generator should make masks reproducible."
    assert first.shape == (12, 64), (
        "A mask batch should have shape [batch, grid_size**2]."
    )
    assert t.equal(first.sum(1), t.full((12,), 9)), "A 3x3 block must select exactly 9 patches."
    for row in first.view(-1, 8, 8):
        coordinates = row.nonzero()
        height = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
        width = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
        assert (height, width) == (3, 3), "Selected patches must form one contiguous square."
    print("All tests in `test_sample_block_masks` passed!")


def test_object_target_masks(fn=solutions.object_target_masks) -> None:
    labels = t.tensor([0, 1, 2, 3])
    masks = fn(labels).view(4, 8, 8)
    assert t.equal(masks.sum(dim=(1, 2)), t.full((4,), 4)), (
        "Every object target must contain exactly one contiguous 2x2 block."
    )
    expected_starts = ((1, 1), (1, 5), (5, 1), (5, 5))
    for row, (top, left) in enumerate(expected_starts):
        assert masks[row, top : top + 2, left : left + 2].all(), (
            "The target block must overlap the exact object location."
        )
        assert not masks[row, 0, 3], "The context-key patch must remain visible."
    print("All tests in `test_object_target_masks` passed!")


def test_cyclic_object_mismatch_preserves_position(
    fn=solutions.cyclic_object_mismatch_indices,
) -> None:
    scenes = solutions.build_scene_dataset(3, seed=0)
    for shift in (1, 2):
        indices = fn(scenes.object_labels, scenes.position_labels, shift=shift)
        assert t.equal(scenes.position_labels[indices], scenes.position_labels), (
            "The intervention should change object context without changing position."
        )
        assert t.all(scenes.object_labels[indices] != scenes.object_labels), (
            "Every mismatched context must come from a different object class."
        )
        assert t.equal(indices.sort().values, t.arange(len(indices))), (
            "The matched intervention should be a permutation, not resampling."
        )
    try:
        fn(scenes.object_labels, scenes.position_labels, shift=3)
    except ValueError as error:
        assert "different object" in str(error), 'A complete class cycle must be rejected as an identity intervention.'
    else:
        raise AssertionError("A full-cycle shift must be rejected because it is not a control.")
    print("All tests in `test_cyclic_object_mismatch_preserves_position` passed!")


def test_gather_masked_tokens(fn=solutions.gather_masked_tokens) -> None:
    tokens = t.arange(2 * 4 * 3).view(2, 4, 3)
    mask = t.tensor([[True, False, True, False], [False, True, False, True]])
    expected = t.stack((tokens[0, [0, 2]], tokens[1, [1, 3]]))
    t.testing.assert_close(fn(tokens, mask), expected, rtol=0, atol=0)
    unequal_mask = t.tensor([[True, False, True, False], [False, True, False, False]])
    try:
        fn(tokens, unequal_mask)
    except ValueError as error:
        assert "same number" in str(error), (
            "The unequal-count error should explain the fixed-cardinality requirement."
        )
    else:
        raise AssertionError(
            "Rows selecting different token counts must be rejected instead of mixing batches."
        )
    print("All tests in `test_gather_masked_tokens` passed!")


def test_normalized_prediction_loss(fn=solutions.normalized_prediction_loss) -> None:
    target = t.tensor([[[3.0, 4.0], [0.0, 2.0]]])
    assert fn(target, target).item() < 1e-12, (
        "Identical prediction and target directions should have zero normalized loss."
    )
    assert fn(7 * target, target).item() < 1e-12, (
        "Latent loss should be invariant to positive prediction scale after normalization."
    )
    opposite = -target
    assert fn(opposite, target).item() > 3.9, (
        "Opposite unit directions should incur the maximum squared distance of four."
    )
    print("All tests in `test_normalized_prediction_loss` passed!")


def test_update_target_encoder(fn=solutions.update_target_encoder) -> None:
    online = nn.Linear(2, 2, bias=False)
    target = nn.Linear(2, 2, bias=False)
    with t.no_grad():
        online.weight.fill_(4.0)
        target.weight.fill_(2.0)
    fn(online, target, ema_decay=0.75)
    t.testing.assert_close(target.weight, t.full_like(target.weight, 2.5), rtol=0, atol=0)
    print("All tests in `test_update_target_encoder` passed!")


def test_encoder_and_predictor_semantics(
    encoder_cls=solutions.PatchTransformerEncoder,
    predictor_cls=solutions.LatentPredictor,
) -> None:
    # Remove nonlinear mixing so projection, position addition, and visible-token
    # gathering can be checked against a four-patch calculation by hand.
    patches = t.tensor([[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]]])
    visible_mask = t.tensor([[True, False, True, False]])
    encoder = encoder_cls(patch_dim=2, d_model=4)
    for attribute in ("patch_projection", "position", "transformer", "norm"):
        assert hasattr(encoder, attribute), (
            f"The encoder needs `{attribute}` so projection and spatial semantics are explicit."
        )
    projection_weight = t.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
    )
    projection_bias = t.tensor([0.5, -0.5, 1.0, 2.0])
    positions = t.tensor(
        [
            [10.0, 20.0, 30.0, 40.0],
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
            [13.0, 23.0, 33.0, 43.0],
        ]
    )
    with t.no_grad():
        encoder.patch_projection.weight.copy_(projection_weight)
        encoder.patch_projection.bias.copy_(projection_bias)
        encoder.position.zero_()
        encoder.position[:, :4].copy_(positions)
    encoder.transformer = nn.Identity()
    encoder.norm = nn.Identity()
    context = encoder(patches, visible_mask)
    full_projected = patches @ projection_weight.T + projection_bias + positions
    expected_context = full_projected[:, [0, 2]]
    t.testing.assert_close(context, expected_context, rtol=0, atol=0)

    class WeightedGlobalMix(nn.Module):
        """Deterministic mixer whose output reveals where context was scattered."""

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, values: t.Tensor) -> t.Tensor:
            self.calls += 1
            weights = t.arange(
                1, values.shape[1] + 1, dtype=values.dtype, device=values.device
            ).view(1, -1, 1)
            return values + (weights * values).sum(dim=1, keepdim=True)

    target_mask = t.tensor([[False, True, False, True]])
    predictor = predictor_cls(d_model=4)
    for attribute in ("mask_token", "position", "transformer", "norm"):
        assert hasattr(predictor, attribute), (
            f"The predictor needs `{attribute}` so mask and scatter semantics are explicit."
        )
    mask_token = t.tensor([[[-1.0, -2.0, -3.0, -4.0]]])
    predictor_positions = t.tensor(
        [
            [0.0, 0.1, 0.2, 0.3],
            [1.0, 1.1, 1.2, 1.3],
            [2.0, 2.1, 2.2, 2.3],
            [3.0, 3.1, 3.2, 3.3],
        ]
    )
    with t.no_grad():
        predictor.mask_token.copy_(mask_token)
        predictor.position.zero_()
        predictor.position[:, :4].copy_(predictor_positions)
    mixer = WeightedGlobalMix()
    predictor.transformer = mixer
    predictor.norm = nn.Identity()
    predicted = predictor(context, target_mask)

    full = mask_token.expand(1, 4, -1).clone()
    full[:, [0, 2]] = context
    positioned = full + predictor_positions
    weights = t.arange(1, 5, dtype=positioned.dtype).view(1, -1, 1)
    mixed = positioned + (weights * positioned).sum(dim=1, keepdim=True)
    expected_prediction = mixed[:, [1, 3]]
    t.testing.assert_close(predicted, expected_prediction, rtol=0, atol=0)
    assert mixer.calls == 1, "The predictor must pass the scattered full grid through its mixer."
    print("All tests in `test_encoder_and_predictor_semantics` passed!")


def test_ijepa_gradient_boundary(model_cls=solutions.IJEPAModel) -> None:
    t.manual_seed(0)
    model = model_cls(d_model=16)
    patches = t.randn(2, 64, 48)
    mask = solutions.sample_block_masks(2, generator=t.Generator().manual_seed(0))
    prediction, target = model(patches, mask)
    loss = solutions.normalized_prediction_loss(prediction, target)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.online_encoder.parameters()), (
        "The online context encoder must receive gradients from the latent prediction loss."
    )
    assert any(parameter.grad is not None for parameter in model.predictor.parameters()), (
        "The predictor must receive gradients from the latent prediction loss."
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters()), (
        "The target encoder must be updated by EMA, not backpropagation."
    )
    print("All tests in `test_ijepa_gradient_boundary` passed!")


def test_one_real_optimization_step(model_cls=solutions.IJEPAModel) -> None:
    t.manual_seed(1)
    scenes = solutions.build_scene_dataset(2, seed=1)
    patches = solutions.patchify(scenes.images[:8])
    mask = solutions.sample_block_masks(8, generator=t.Generator().manual_seed(1))
    model = model_cls(d_model=16)
    optimizer = t.optim.AdamW(
        list(model.online_encoder.parameters()) + list(model.predictor.parameters()), lr=1e-3
    )
    online_before = next(model.online_encoder.parameters()).detach().clone()
    target_before = next(model.target_encoder.parameters()).detach().clone()
    prediction, target = model(patches, mask)
    loss = solutions.normalized_prediction_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    solutions.update_target_encoder(model.online_encoder, model.target_encoder, 0.9)
    online_after = next(model.online_encoder.parameters()).detach()
    target_after = next(model.target_encoder.parameters()).detach()
    assert t.isfinite(loss), "A real optimization step produced a non-finite latent loss."
    assert not t.equal(online_before, online_after), (
        "The optimizer step should update the online encoder through the prediction loss."
    )
    assert not t.equal(target_before, target_after), (
        "The target should move after a real optimizer+EMA step."
    )
    print("All tests in `test_one_real_optimization_step` passed!")


def test_collapse_diagnostics(fn=solutions.collapse_diagnostics) -> None:
    collapsed = fn(t.ones(16, 8))
    structured = fn(t.eye(8).repeat(2, 1))
    assert collapsed.feature_std == 0.0, (
        "A constant representation must have exactly zero feature spread."
    )
    assert collapsed.effective_rank <= 1.01, (
        "A constant representation should have effective rank at most one."
    )
    assert structured.feature_std > 0.2, (
        "Repeated one-hot features should retain substantial per-dimension variation."
    )
    assert structured.effective_rank > 6.5, (
        "The eight independent one-hot directions should produce high effective rank."
    )
    print("All tests in `test_collapse_diagnostics` passed!")


def test_ridge_probe_toy_oracle(fn=solutions.evaluate_probes) -> None:
    labels = t.arange(4).repeat_interleave(10)
    features = t.nn.functional.one_hot(labels, 4).float()
    object_labels = labels % 2
    metrics = fn(features, object_labels, labels, seed=0)
    assert metrics.object_accuracy == 1.0, (
        "The ridge probe should decode the deterministic binary label from one-hot features."
    )
    assert metrics.position_accuracy == 1.0, (
        "The ridge probe should decode all four one-hot classes on the held-out split."
    )
    print("All tests in `test_ridge_probe_toy_oracle` passed!")


def test_transfer_probe_toy_oracle(fn=solutions.evaluate_transfer_probes) -> None:
    object_labels = t.arange(3).repeat_interleave(4)
    position_labels = t.arange(4).repeat(3)
    features = t.cat(
        (
            t.nn.functional.one_hot(object_labels, 3),
            t.nn.functional.one_hot(position_labels, 4),
        ),
        dim=1,
    ).float()
    metrics = fn(
        features,
        object_labels,
        position_labels,
        2.0 * features,
        object_labels,
        position_labels,
    )
    assert metrics.object_accuracy == 1.0, (
        "A transfer probe should recover object labels from the independent scaled feature table."
    )
    assert metrics.position_accuracy == 1.0, (
        "A transfer probe should recover position labels from the independent scaled feature table."
    )
    print("All tests in `test_transfer_probe_toy_oracle` passed!")


def test_evaluate_prediction_exact_oracle(fn=solutions.evaluate_prediction) -> None:
    class KeyReadingEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_labels: list[t.Tensor] = []

        def forward(self, patches: t.Tensor, visible_mask: t.Tensor | None = None) -> t.Tensor:
            key_rgb = patches[:, 3].reshape(len(patches), 3, -1).mean(dim=-1)
            labels = key_rgb[:, [0, 2, 1]].argmax(dim=-1)
            self.seen_labels.append(labels.detach().cpu())
            latents = t.nn.functional.one_hot(labels, 3).float()
            token_count = patches.shape[1] if visible_mask is None else int(visible_mask[0].sum())
            return latents[:, None].expand(-1, token_count, -1)

    class ContextCopyingPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, context: t.Tensor, target_mask: t.Tensor) -> t.Tensor:
            self.calls += 1
            target_count = int(target_mask[0].sum())
            return context[:, :1].expand(-1, target_count, -1)

    class ExactModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.online_encoder = KeyReadingEncoder()
            self.target_encoder = KeyReadingEncoder()
            self.predictor = ContextCopyingPredictor()

    model = ExactModel()
    centroid_scenes = solutions.build_scene_dataset(2, seed=5)
    evaluation_scenes = solutions.build_scene_dataset(2, seed=50_005)
    metrics = fn(
        model,
        centroid_scenes,
        evaluation_scenes,
        device=t.device("cpu"),
        mismatch_shift=1,
    )
    expected_scalars = {
        "latent_loss": 0.0,
        "mismatched_context_loss": 2.0,
        "zero_context_loss": 1.0,
        "key_erased_context_loss": 4 / 3,
        "key_erased_context_object_accuracy": 1 / 3,
        "key_erased_non_key_pixel_max_abs_diff": 0.0,
        "target_object_accuracy": 1.0,
        "predicted_object_accuracy": 1.0,
        "mismatched_context_object_accuracy": 0.0,
        "alternate_key_object_accuracy": 1.0,
        "non_key_pixel_max_abs_diff": 0.0,
    }
    for field, expected in expected_scalars.items():
        actual = getattr(metrics, field)
        assert abs(actual - expected) < 1e-7, (
            f"Exact key-reading oracle expected {field}={expected}, got {actual}."
        )
    assert len(model.online_encoder.seen_labels) == 6, (
        "Evaluation must encode original, key-swapped, and key-erased context for centroid and held-out scenes."
    )
    assert len(model.target_encoder.seen_labels) == 2, (
        "Evaluation must construct target latents for centroid and held-out scenes."
    )
    assert model.predictor.calls == 8, (
        "Evaluation must score correct, mismatched, key-erased, and zero context on both scene batches."
    )
    for original, swapped in zip(
        model.online_encoder.seen_labels[::3], model.online_encoder.seen_labels[1::3]
    ):
        assert t.equal(swapped, (original + 1) % 3), (
            "The exact oracle did not observe the requested cyclic key intervention."
        )
    print("All tests in `test_evaluate_prediction_exact_oracle` passed!")


def test_training_loop_one_epoch(train_fn=solutions._train_model) -> None:
    scenes = solutions.build_scene_dataset(2, seed=4)
    model, trace, target_change = train_fn(
        scenes,
        device=t.device("cpu"),
        seed=4,
        epochs=1,
        batch_size=24,
        learning_rate=1e-3,
        ema_decay=0.9,
        joint_target_ablation=False,
    )
    assert len(trace.losses) == len(trace.feature_std) == len(trace.effective_rank) == 1, (
        "One epoch should append exactly one loss, spread, and effective-rank diagnostic."
    )
    assert t.isfinite(t.tensor(trace.losses[0])), (
        "The one-epoch training loss must remain finite."
    )
    assert trace.feature_std[0] > 0, (
        "The target representation should not become exactly constant after one epoch."
    )
    assert target_change > 0, (
        "The frozen target encoder should move through EMA even though it receives no gradients."
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters()), (
        "The target encoder must remain outside the optimizer and autograd graph."
    )
    print("All tests in `test_training_loop_one_epoch` passed!")


def test_matched_joint_target_first_shared_update(train_fn=solutions._train_model) -> None:
    scenes = solutions.build_scene_dataset(2, seed=13)
    ema_model, _, _ = train_fn(
        scenes,
        device=t.device("cpu"),
        seed=13,
        epochs=1,
        batch_size=len(scenes.images),
        learning_rate=1e-3,
        ema_decay=0.9,
        joint_target_ablation=False,
    )
    no_ema_model, _, _ = train_fn(
        scenes,
        device=t.device("cpu"),
        seed=13,
        epochs=1,
        batch_size=len(scenes.images),
        learning_rate=1e-3,
        ema_decay=0.0,
        joint_target_ablation=True,
    )
    for ema_parameter, no_ema_parameter in zip(
        list(ema_model.online_encoder.parameters()) + list(ema_model.predictor.parameters()),
        list(no_ema_model.online_encoder.parameters()) + list(no_ema_model.predictor.parameters()),
    ):
        t.testing.assert_close(ema_parameter, no_ema_parameter, rtol=0, atol=0)
    print("All tests in `test_matched_joint_target_first_shared_update` passed!")


def test_run_experiment_structure(fn=solutions.run_experiment) -> None:
    run = fn(device="cpu", seed=7, examples_per_combination=2, epochs=1)
    assert len(run.trace.losses) == len(run.joint_target_trace.losses) == 1, (
        "A one-epoch experiment must produce one diagnostic row for each matched system."
    )
    assert run.train_eval_exact_overlap == 0, (
        "Even the small contract must generate evaluation tensors absent from training."
    )
    assert run.prediction.non_key_pixel_max_abs_diff == 0.0, (
        "The end-to-end experiment must preserve non-key pixels during intervention."
    )
    print("All tests in `test_run_experiment_structure` passed!")


def _valid_run_fixture() -> solutions.IJEPARun:
    prediction = solutions.PredictionMetrics(
        latent_loss=0.10,
        mismatched_context_loss=0.20,
        zero_context_loss=0.60,
        key_erased_context_loss=0.20,
        key_erased_context_object_accuracy=1 / 3,
        key_erased_non_key_pixel_max_abs_diff=0.0,
        target_object_accuracy=1.0,
        predicted_object_accuracy=1.0,
        mismatched_context_object_accuracy=0.0,
        alternate_key_object_accuracy=1.0,
        non_key_pixel_max_abs_diff=0.0,
    )
    joint_target_prediction = replace(
        prediction,
        latent_loss=0.30,
        predicted_object_accuracy=0.50,
    )
    random_prediction = replace(
        prediction,
        latent_loss=3.0,
        predicted_object_accuracy=1 / 3,
    )
    probe = solutions.ProbeMetrics(object_accuracy=1.0, position_accuracy=1.0)
    diagnostics = solutions.CollapseDiagnostics(
        feature_std=0.10,
        effective_rank=2.0,
        mean_pairwise_cosine=0.0,
    )
    joint_target_diagnostics = solutions.CollapseDiagnostics(
        feature_std=0.01,
        effective_rank=1.0,
        mean_pairwise_cosine=0.0,
    )
    collapsed = solutions.CollapseDiagnostics(
        feature_std=0.0,
        effective_rank=1.0,
        mean_pairwise_cosine=0.0,
    )
    trace = solutions.TrainingTrace(
        losses=[1.0, 0.1],
        feature_std=[0.10, 0.10],
        effective_rank=[2.0, 2.0],
    )
    return solutions.IJEPARun(
        device="cpu",
        seed=0,
        initial_loss=1.0,
        final_loss=0.1,
        loss_ratio=0.1,
        probes=probe,
        shuffled_label_probes=probe,
        collapsed_control_probes=probe,
        joint_target_probes=probe,
        random_encoder_probes=probe,
        prediction=prediction,
        joint_target_prediction=joint_target_prediction,
        random_prediction=random_prediction,
        diagnostics=diagnostics,
        joint_target_diagnostics=joint_target_diagnostics,
        collapsed_control_diagnostics=collapsed,
        target_parameter_change=1.0,
        target_has_gradients=False,
        peak_vram_gb=0.0,
        trace=trace,
        joint_target_trace=trace,
        train_eval_exact_overlap=0,
        train_semantic_template_count=12,
        evaluation_semantic_template_count=12,
        shared_semantic_template_count=12,
    )


def test_validate_run_contract(fn=solutions.validate_run) -> None:
    valid = _valid_run_fixture()
    fn(valid, require_cuda=False, max_vram_gb=24.0)
    invalid_runs = {
        "insufficient loss reduction": replace(valid, loss_ratio=0.30),
        "collapsed EMA features": replace(
            valid,
            diagnostics=replace(valid.diagnostics, feature_std=0.001),
        ),
        "failed prediction": replace(
            valid,
            prediction=replace(valid.prediction, predicted_object_accuracy=0.50),
        ),
        "failed alternate-key control": replace(
            valid,
            prediction=replace(valid.prediction, alternate_key_object_accuracy=0.50),
        ),
        "non-key intervention leakage": replace(
            valid,
            prediction=replace(valid.prediction, non_key_pixel_max_abs_diff=1e-3),
        ),
        "erased-key shortcut": replace(valid, prediction=replace(valid.prediction, key_erased_context_object_accuracy=1.0)),
        "erased-key leakage": replace(valid, prediction=replace(valid.prediction, key_erased_non_key_pixel_max_abs_diff=1e-3)),
        "missing semantic template": replace(valid, shared_semantic_template_count=11),
        "untrained shortcut": replace(
            valid,
            random_prediction=replace(
                valid.random_prediction,
                predicted_object_accuracy=0.90,
            ),
        ),
        "target gradients": replace(valid, target_has_gradients=True),
        "train/evaluation overlap": replace(valid, train_eval_exact_overlap=1),
        "memory-budget breach": replace(valid, peak_vram_gb=25.0),
    }
    for failure_name, invalid in invalid_runs.items():
        try:
            fn(invalid, require_cuda=False, max_vram_gb=24.0)
        except AssertionError:
            continue
        raise AssertionError(
            f"validate_run accepted a fixture with {failure_name}; the final contract must fail closed."
        )
    print("All tests in `test_validate_run_contract` passed!")


def test_hard_oracles_reject_shortcuts() -> None:
    class ZeroEncoder(nn.Module):
        def __init__(self, patch_dim: int = 2, d_model: int = 4) -> None:
            super().__init__()
            self.patch_projection = nn.Linear(patch_dim, d_model)
            self.position = nn.Parameter(t.zeros(1, 64, d_model))
            self.transformer = nn.Identity()
            self.norm = nn.Identity()

        def forward(self, patches: t.Tensor, visible_mask: t.Tensor | None = None) -> t.Tensor:
            count = patches.shape[1] if visible_mask is None else int(visible_mask[0].sum())
            return t.zeros(len(patches), count, self.position.shape[-1])

    class ZeroPredictor(nn.Module):
        def __init__(self, d_model: int = 4) -> None:
            super().__init__()
            self.mask_token = nn.Parameter(t.zeros(1, 1, d_model))
            self.position = nn.Parameter(t.zeros(1, 64, d_model))
            self.transformer = nn.Identity()
            self.norm = nn.Identity()

        def forward(self, context: t.Tensor, target_mask: t.Tensor) -> t.Tensor:
            return t.zeros(len(context), int(target_mask[0].sum()), context.shape[-1])

    try:
        test_encoder_and_predictor_semantics(ZeroEncoder, ZeroPredictor)
    except AssertionError:
        pass
    else:
        raise AssertionError("The semantic architecture oracle accepted shape-correct zero modules.")

    def fabricated_prediction(*_args, **_kwargs) -> solutions.PredictionMetrics:
        return solutions.PredictionMetrics(
            latent_loss=0.0,
            mismatched_context_loss=2.0,
            zero_context_loss=1.0,
            key_erased_context_loss=4 / 3,
            key_erased_context_object_accuracy=1 / 3,
            key_erased_non_key_pixel_max_abs_diff=0.0,
            target_object_accuracy=1.0,
            predicted_object_accuracy=1.0,
            mismatched_context_object_accuracy=0.0,
            alternate_key_object_accuracy=1.0,
            non_key_pixel_max_abs_diff=0.0,
        )

    try:
        test_evaluate_prediction_exact_oracle(fabricated_prediction)
    except AssertionError:
        pass
    else:
        raise AssertionError("The prediction oracle accepted a fabricated metrics dataclass.")

    def no_op_validator(*_args, **_kwargs) -> None:
        return None

    try:
        test_validate_run_contract(no_op_validator)
    except AssertionError:
        pass
    else:
        raise AssertionError("The validator contract accepted a no-op learner implementation.")
    print("All tests in `test_hard_oracles_reject_shortcuts` passed!")


def test_cpu_smoke_signature_result() -> None:
    report = solutions.run_smoke_test(cpu=True)
    assert report["heldout_prediction_accuracy"] >= 0.90, (
        "The trained predictor must recover the masked object on disjoint scenes."
    )
    assert report["mismatched_context_accuracy"] <= 0.10, (
        "A key-only object intervention should stop predicting the original target."
    )
    assert report["alternate_key_accuracy"] >= 0.90, (
        "A key-only object intervention should predict the alternate key's class."
    )
    assert report["non_key_pixel_max_abs_diff"] == 0.0, (
        "The causal control must preserve every pixel outside the key patch."
    )
    assert report["random_prediction_accuracy"] <= 0.50, (
        "The same untrained architecture must remain near chance."
    )
    assert report["random_to_trained_loss_ratio"] > 20.0, (
        "The held-out latent objective must improve substantially over initialization."
    )
    assert report["train_eval_exact_overlap"] == 0, (
        "The signature result must use image tensors absent from training."
    )
    assert report["random_encoder_object_probe_accuracy"] >= 0.90, (
        "The easy random-feature probe should remain visible as a negative result."
    )
    print("All tests in `test_cpu_smoke_signature_result` passed!")


def test_solution_dropdowns_parse_as_python() -> None:
    section_dir = Path(__file__).resolve().parent
    paths = (
        section_dir / "14.2_I-JEPA_from_Scratch_exercises.ipynb",
        section_dir / "14.2_I-JEPA_from_Scratch_solutions.ipynb",
        section_dir.parents[1] / "instructions/pages/02_[14.2]_I-JEPA_from_Scratch.md",
    )
    pattern = re.compile(
        r"<summary>Solution</summary>\s*```python\n(.*?)\n```",
        flags=re.DOTALL,
    )
    for path in paths:
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            text = "\n".join(
                "".join(cell["source"])
                if isinstance(cell["source"], list)
                else cell["source"]
                for cell in notebook["cells"]
                if cell["cell_type"] == "markdown"
            )
        else:
            text = path.read_text(encoding="utf-8")
        blocks = pattern.findall(text)
        assert len(blocks) == 12, (
            f"Expected one Python solution dropdown per exercise in {path.name}; found {len(blocks)}."
        )
        for exercise_number, block in enumerate(blocks, start=1):
            try:
                ast.parse(textwrap.dedent(block))
            except SyntaxError as error:
                raise AssertionError(
                    f"Exercise {exercise_number} solution dropdown in {path.name} is invalid Python: {error}"
                ) from error
    print("All tests in `test_solution_dropdowns_parse_as_python` passed!")


def test_learner_surface_contract() -> None:
    section_dir = Path(__file__).resolve().parent
    exercise_path = section_dir / "14.2_I-JEPA_from_Scratch_exercises.ipynb"
    solution_path = section_dir / "14.2_I-JEPA_from_Scratch_solutions.ipynb"
    page_path = section_dir.parents[1] / "instructions/pages/02_[14.2]_I-JEPA_from_Scratch.md"
    exercise = json.loads(exercise_path.read_text(encoding="utf-8"))
    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    def source_text(cell: dict) -> str:
        source = cell["source"]
        return "".join(source) if isinstance(source, list) else source

    exercise_markdown = "\n".join(
        source_text(cell) for cell in exercise["cells"] if cell["cell_type"] == "markdown"
    )
    exercise_code = "\n".join(
        source_text(cell) for cell in exercise["cells"] if cell["cell_type"] == "code"
    )
    solution_code = "\n".join(
        source_text(cell) for cell in solution["cells"] if cell["cell_type"] == "code"
    )
    page = page_path.read_text(encoding="utf-8")

    assert exercise_markdown.count("### Exercise ") == 12, 'Keep all twelve learner implementation exercises.'
    for label in ("Expected output", "Solution", "Interpretation"):
        assert exercise_markdown.count(f"<summary>{label}</summary>") >= 12, 'Each exercise needs its own expected output, answer, and interpretation.'
    assert exercise_markdown.count("<summary>Help - reveal a hint</summary>") >= 12, 'Every exercise needs a usable hint.'
    for name in (
        "object_target_masks",
        "normalized_prediction_loss",
        "update_target_encoder",
        "evaluate_transfer_probes",
        "cyclic_object_mismatch_indices",
        "replace_scene_keys",
        "evaluate_prediction",
        "_train_model",
        "run_experiment",
    ):
        assert f"def {name}" in exercise_code or f"class {name}" in exercise_code, 'The learner must implement this core function locally.'
        assert f"def {name}" in solution_code or f"class {name}" in solution_code, 'The solved notebook must expose this core function.'
    assert 'run_seed_suite(device="cuda"' in exercise_code, 'The visible result must execute all declared seeds on CUDA.'
    assert "MISMATCH_SHIFT" in exercise_code and "mismatch_shift=MISMATCH_SHIFT" in exercise_code, 'The play parameter must reach the actual intervention.'
    assert "reference.run_experiment" not in exercise_code + solution_code, 'Execute learner definitions rather than the reference experiment.'
    assert "verification_report.json" not in exercise_code + solution_code, 'Generate the result live rather than reading release evidence.'
    assert "random encoder features already" in exercise_markdown.lower(), 'Explain why the successful random probe disqualifies the probe claim.'
    assert "masked object" in page.lower(), 'The page must state the concrete masked-object question.'
    assert "no_ema_epochs" not in page + exercise_code + solution_code, 'Match the training budget of both target conditions.'
    print("All tests in `test_learner_surface_contract` passed!")


def test_answers_complete_the_learner_and_match_reference() -> None:
    section = Path(__file__).resolve().parent
    learner = json.loads((section / '14.2_I-JEPA_from_Scratch_exercises.ipynb').read_text())
    solved = json.loads((section / '14.2_I-JEPA_from_Scratch_solutions.ipynb').read_text())
    reference = {node.name: node for node in ast.parse((section / 'solutions.py').read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    count = 0
    for index, (exercise, solution) in enumerate(zip(learner['cells'], solved['cells'], strict=True)):
        if exercise['cell_type'] != 'code' or 'exercise' not in exercise['metadata'].get('tags', []):
            continue
        answer_cell = learner['cells'][index + 2]
        answer = re.search(r'<summary>Solution</summary>\s*```python\n(.*?)\n```',
                           ''.join(answer_cell['source']), re.S)[1]
        answer_tree = ast.parse(answer)
        supplied = {node.name for node in answer_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        required = {node.name for node in ast.parse(''.join(exercise['source'])).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        assert required == supplied, f'Exercise {count + 1} scaffolding and answer disagree: {required ^ supplied}'
        assert ast.dump(answer_tree) == ast.dump(ast.parse(''.join(solution['source']))), 'The answer must reproduce the solved cell.'
        for node in answer_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert ast.dump(node) == ast.dump(reference[node.name]), f'{node.name} is stale relative to the implementation.'
        count += 1
    assert count == 12, 'All twelve exercises must be completable from their own answers.'
    print('All tests in `test_answers_complete_the_learner_and_match_reference` passed!')


def test_executed_notebook_has_live_result() -> None:
    path = Path(__file__).resolve().parent / '14.2_I-JEPA_from_Scratch_solutions.ipynb'
    notebook = json.loads(path.read_text())
    code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert all(cell['execution_count'] is not None for cell in code), 'Execute every solution cell in a fresh kernel, including CUDA and play.'
    outputs = [output for cell in code for output in cell['outputs']]
    assert not any(output['output_type'] == 'error' for output in outputs), 'The stored notebook contains an execution error.'
    assert any('image/png' in output.get('data', {}) for output in outputs), 'The signature must be a live visible figure.'
    assert 'SIGNATURE PASSED for all three declared seeds.' in json.dumps(outputs), 'The three-seed result was not executed.'
    print('All tests in `test_executed_notebook_has_live_result` passed!')
