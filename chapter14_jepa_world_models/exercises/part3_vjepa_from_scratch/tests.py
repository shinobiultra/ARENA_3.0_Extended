"""Semantic tests for [14.3] V-JEPA from Scratch."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
import yaml

SECTION_DIR = Path(__file__).resolve().parent
if str(SECTION_DIR) not in sys.path:
    sys.path.insert(0, str(SECTION_DIR))

import solutions


def test_make_moving_object_batch_exact(make_batch: Callable | None = None) -> None:
    make_batch = make_batch or solutions.make_moving_object_batch
    world = make_batch(16, seed=3)
    assert world.videos.shape == (16, 6, 3, 16, 16)
    assert sorted(world.state_labels.unique().tolist()) == list(range(8))
    assert t.equal(world.state_labels, world.color_labels * 4 + world.direction_labels)
    assert world.clip_keys.shape == (16, 4)
    assert world.clip_keys.unique(dim=0).shape[0] == 16
    t.testing.assert_close(world.clip_keys[:, :2], t.stack([world.color_labels, world.direction_labels], dim=1))
    t.testing.assert_close(world.clip_keys[:, 2:], world.start_xy)
    for index in range(16):
        dx, dy = world.velocity_xy[index].tolist()
        for frame in range(1, 6):
            expected = t.roll(world.videos[index, frame - 1], shifts=(dy, dx), dims=(-2, -1))
            t.testing.assert_close(world.videos[index, frame], expected)
    print("All tests in `test_make_moving_object_batch_exact` passed!")


def test_make_moving_object_batch_rejects_invalid_world(make_batch: Callable | None = None) -> None:
    make_batch = make_batch or solutions.make_moving_object_batch
    with pytest.raises(ValueError, match="at least 8"):
        make_batch(7)
    with pytest.raises(ValueError, match="too small"):
        make_batch(8, frames=8, image_size=4, object_size=3)
    with pytest.raises(ValueError, match="shape"):
        make_batch(8, excluded_clip_keys=t.zeros(2, 3, dtype=t.long))
    print("All tests in `test_make_moving_object_batch_rejects_invalid_world` passed!")


def test_disjoint_world_splits_and_duplicate_audit(
    split_fn: Callable | None = None,
    audit_fn: Callable | None = None,
) -> None:
    split_fn = split_fn or solutions.make_disjoint_worlds
    audit_fn = audit_fn or solutions.audit_clip_splits
    train_world, test_world = split_fn(64, 32, seed=11)
    audit = audit_fn(train_world, test_world)
    assert audit.train_unique == 64 and audit.test_unique == 32
    assert audit.train_duplicates == audit.test_duplicates == audit.train_test_overlap == 0
    assert audit.is_disjoint
    train_keys = {tuple(row) for row in train_world.clip_keys.tolist()}
    test_keys = {tuple(row) for row in test_world.clip_keys.tolist()}
    assert train_keys.isdisjoint(test_keys)
    reversed_test_keys = {tuple(row) for row in solutions.physical_reverse_keys(test_world).tolist()}
    assert reversed_test_keys.isdisjoint(train_keys)
    assert reversed_test_keys.isdisjoint(test_keys)
    print("All tests in `test_disjoint_world_splits_and_duplicate_audit` passed!")


def test_physical_reverse_keys_and_collision_audit(
    reverse_keys_fn: Callable | None = None,
    physical_keys_fn: Callable | None = None,
    audit_fn: Callable | None = None,
) -> None:
    reverse_keys_fn = reverse_keys_fn or solutions.reverse_canonical_keys
    physical_keys_fn = physical_keys_fn or solutions.physical_reverse_keys
    audit_fn = audit_fn or solutions.audit_reversal_collisions
    train_world, test_world = solutions.make_disjoint_worlds(64, 32, seed=13)
    expected = reverse_keys_fn(test_world.clip_keys, frames=6, image_size=16, speed=1)
    t.testing.assert_close(physical_keys_fn(test_world), expected)
    t.testing.assert_close(
        reverse_keys_fn(expected, frames=6, image_size=16, speed=1),
        test_world.clip_keys,
    )
    fit_keys = train_world.clip_keys[:16]
    audit = audit_fn(train_world, fit_keys, test_world)
    assert audit.collision_free
    assert audit.reversed_unique == len(test_world.videos)
    assert audit.reversed_duplicates == 0
    assert audit.reversed_pretraining_overlap == 0
    assert audit.reversed_probe_fit_overlap == 0
    assert audit.reversed_test_overlap == 0

    source_keys = train_world.clip_keys[: len(test_world.videos)]
    collision_forward = reverse_keys_fn(source_keys, frames=6, image_size=16, speed=1)
    collision_direction = collision_forward[:, 1]
    colliding_test = replace(
        test_world,
        clip_keys=collision_forward,
        color_labels=collision_forward[:, 0],
        direction_labels=collision_direction,
        state_labels=collision_forward[:, 0] * 4 + collision_direction,
        start_xy=collision_forward[:, 2:],
        velocity_xy=solutions.DIRECTIONS[collision_direction],
    )
    collision = audit_fn(train_world, fit_keys, colliding_test)
    assert not collision.collision_free
    assert collision.reversed_pretraining_overlap > 0
    print("All tests in `test_physical_reverse_keys_and_collision_audit` passed!")


def test_tubelet_grid_and_embedding(
    grid_fn: Callable | None = None,
    tubelet_cls: type[nn.Module] | None = None,
) -> None:
    grid_fn = grid_fn or solutions.tubelet_grid
    tubelet_cls = tubelet_cls or solutions.TubeletEmbedding
    assert grid_fn((6, 16, 16), (2, 4, 4)) == (3, 4, 4)
    layer = tubelet_cls(in_channels=3, d_model=2, tubelet_size=(2, 4, 4))
    with t.no_grad():
        layer.proj.weight.fill_(1.0)
        layer.proj.bias.zero_()
    video = t.arange(2 * 6 * 3 * 16 * 16, dtype=t.float32).reshape(2, 6, 3, 16, 16)
    tokens, grid = layer(video)
    expected = F.conv3d(video.permute(0, 2, 1, 3, 4), layer.proj.weight, stride=(2, 4, 4))
    assert grid == (3, 4, 4)
    assert tokens.shape == (2, 48, 2)
    t.testing.assert_close(tokens, expected.flatten(2).transpose(1, 2))
    print("All tests in `test_tubelet_grid_and_embedding` passed!")


def test_tubelet_embedding_rejects_wrong_rank(
    grid_fn: Callable | None = None,
    tubelet_cls: type[nn.Module] | None = None,
) -> None:
    grid_fn = grid_fn or solutions.tubelet_grid
    tubelet_cls = tubelet_cls or solutions.TubeletEmbedding
    with pytest.raises(ValueError, match="batch, frames"):
        tubelet_cls()(t.zeros(6, 3, 16, 16))
    with pytest.raises(ValueError, match="divisible"):
        grid_fn((5, 16, 16), (2, 4, 4))
    for shape in ((2, 5, 3, 16, 16), (2, 6, 3, 15, 16), (2, 6, 3, 16, 15)):
        with pytest.raises(ValueError, match="divisible"):
            tubelet_cls()(t.zeros(shape))
    print("All tests in `test_tubelet_embedding_rejects_wrong_rank` passed!")


def test_spatiotemporal_sincos_is_factorized(position_fn: Callable | None = None) -> None:
    position_fn = position_fn or solutions.spatiotemporal_sincos
    position = position_fn((3, 4, 4), 24, device="cpu")
    assert position.shape == (1, 48, 24)
    assert t.isfinite(position).all()
    assert not t.equal(position[:, 0], position[:, 1])
    assert not t.equal(position[:, 0], position[:, 16])
    print("All tests in `test_spatiotemporal_sincos_is_factorized` passed!")


def test_spatial_tube_masks_repeat_through_time(mask_fn: Callable | None = None) -> None:
    mask_fn = mask_fn or solutions.sample_spatial_tube_mask
    generator = t.Generator().manual_seed(0)
    mask = mask_fn(8, (3, 4, 4), block_shape=(2, 2), generator=generator).reshape(8, 3, 4, 4)
    assert (mask.sum(dim=(1, 2, 3)) == 12).all()
    for sample in mask:
        t.testing.assert_close(sample[0], sample[1])
        t.testing.assert_close(sample[1], sample[2])
        coordinates = sample[0].nonzero()
        assert coordinates[:, 0].max() - coordinates[:, 0].min() == 1
        assert coordinates[:, 1].max() - coordinates[:, 1].min() == 1
    with pytest.raises(ValueError, match="spatial"):
        mask_fn(1, (3, 4, 4), block_shape=(5, 2))
    print("All tests in `test_spatial_tube_masks_repeat_through_time` passed!")


def test_future_prediction_mask_is_separate_diagnostic(mask_fn: Callable | None = None) -> None:
    mask_fn = mask_fn or solutions.future_prediction_mask
    mask = mask_fn(2, (3, 4, 4)).reshape(2, 3, 4, 4)
    assert not mask[:, :-1].any()
    assert mask[:, -1].all()
    with pytest.raises(ValueError, match="visible"):
        mask_fn(1, (3, 4, 4), future_steps=3)
    print("All tests in `test_future_prediction_mask_is_separate_diagnostic` passed!")


def test_expand_tubelet_mask_matches_pixels(expand_fn: Callable | None = None) -> None:
    expand_fn = expand_fn or solutions.expand_tubelet_mask
    mask = t.zeros(1, 8, dtype=t.bool)
    mask[0, 5] = True
    expanded = expand_fn(mask, (2, 2, 2), (2, 3, 4))
    assert expanded.shape == (1, 4, 6, 8)
    assert expanded.sum() == 2 * 3 * 4
    assert expanded[0, 2:4, 0:3, 4:8].all()
    assert not expanded[0, :2].any()
    with pytest.raises(ValueError, match="flattened"):
        expand_fn(t.zeros(1, 7, dtype=t.bool), (2, 2, 2), (2, 3, 4))
    print("All tests in `test_expand_tubelet_mask_matches_pixels` passed!")


def test_pack_visible_tokens_preserves_order(pack_fn: Callable | None = None) -> None:
    pack_fn = pack_fn or solutions.pack_visible_tokens
    tokens = t.arange(2 * 6 * 3).reshape(2, 6, 3).float()
    mask = t.tensor([[False, True, False, True, False, False]]).repeat(2, 1)
    visible = pack_fn(tokens, mask)
    assert visible.shape == (2, 4, 3)
    t.testing.assert_close(visible[0], tokens[0, [0, 2, 4, 5]])
    with pytest.raises(ValueError, match="same number"):
        pack_fn(tokens, t.tensor([[False] * 6, [True, False, False, False, False, False]]))
    print("All tests in `test_pack_visible_tokens_preserves_order` passed!")


def test_video_encoder_visible_shape(encoder_cls: type[nn.Module] | None = None) -> None:
    encoder_cls = encoder_cls or solutions.VideoEncoder
    encoder = encoder_cls(d_model=12, n_heads=3)
    video = solutions.make_moving_object_batch(8, seed=2).videos
    mask = solutions.sample_spatial_tube_mask(8, (3, 4, 4), block_shape=(2, 2))
    visible, positions, grid = encoder.encode_visible(video, mask)
    assert grid == (3, 4, 4)
    assert positions.shape == (1, 48, 12)
    assert visible.shape == (8, 36, 12)
    assert t.isfinite(visible).all()
    print("All tests in `test_video_encoder_visible_shape` passed!")


def test_masked_pixels_cannot_change_visible_context(encoder_cls: type[nn.Module] | None = None) -> None:
    encoder_cls = encoder_cls or solutions.VideoEncoder
    solutions.set_seed(13)
    encoder = encoder_cls(d_model=12, n_heads=3).eval()
    video = solutions.make_moving_object_batch(8, seed=13).videos
    mask = solutions.future_prediction_mask(8, (3, 4, 4))
    visible, _, _ = encoder.encode_visible(video, mask)
    changed = video.clone()
    changed[:, 4:] = t.rand_like(changed[:, 4:])
    changed_visible, _, _ = encoder.encode_visible(changed, mask)
    t.testing.assert_close(visible, changed_visible, rtol=0.0, atol=0.0)
    print("All tests in `test_masked_pixels_cannot_change_visible_context` passed!")


def test_predictor_places_visible_and_mask_tokens(predictor_cls: type[nn.Module] | None = None) -> None:
    predictor_cls = predictor_cls or solutions.VideoPredictor
    predictor = predictor_cls(d_model=12, n_heads=3, depth=1)
    mask = t.tensor([[False, True, False, True], [False, True, False, True]])
    visible = t.randn(2, 2, 12)
    positions = t.zeros(1, 4, 12)
    predicted = predictor(visible, mask, positions)
    assert predicted.shape == (2, 2, 12)
    assert t.isfinite(predicted).all()
    print("All tests in `test_predictor_places_visible_and_mask_tokens` passed!")


def test_small_vjepa_forward_stops_target_grad(model_cls: type[nn.Module] | None = None) -> None:
    model_cls = model_cls or solutions.SmallVJEPA
    model = model_cls(d_model=12, n_heads=3)
    video = solutions.make_moving_object_batch(8).videos
    mask = solutions.future_prediction_mask(8, (3, 4, 4))
    predicted, target, visible = model(video, mask)
    assert predicted.shape == target.shape == (8, 16, 12)
    assert visible.shape == (8, 32, 12)
    assert predicted.requires_grad and not target.requires_grad
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    t.testing.assert_close(target.mean(dim=-1), t.zeros_like(target.mean(dim=-1)), atol=1e-5, rtol=0.0)
    print("All tests in `test_small_vjepa_forward_stops_target_grad` passed!")


def test_ema_update_exact(ema_fn: Callable | None = None) -> None:
    ema_fn = ema_fn or solutions.ema_update
    context = nn.Linear(3, 2)
    target = nn.Linear(3, 2)
    with t.no_grad():
        context.weight.fill_(2.0)
        context.bias.fill_(4.0)
        target.weight.zero_()
        target.bias.zero_()
    ema_fn(target, context, 0.75)
    t.testing.assert_close(target.weight, t.full_like(target.weight, 0.5))
    t.testing.assert_close(target.bias, t.full_like(target.bias, 1.0))
    with pytest.raises(ValueError, match="momentum"):
        ema_fn(target, context, 1.1)
    print("All tests in `test_ema_update_exact` passed!")


def test_vjepa_l1_loss_is_exact_and_stops_target_grad(loss_fn: Callable | None = None) -> None:
    loss_fn = loss_fn or solutions.vjepa_l1_loss
    target = t.tensor([[[1.0, -1.0], [-2.0, 2.0]]])
    assert float(loss_fn(target, target)) == 0.0
    assert float(loss_fn(target.flip(-1), target)) == 3.0
    predicted = target.clone().requires_grad_(True)
    teacher = target.clone().requires_grad_(True)
    loss_fn(predicted, teacher).backward()
    assert predicted.grad is not None and teacher.grad is None
    with pytest.raises(ValueError, match="same shape"):
        loss_fn(t.zeros(1, 2, 3), t.zeros(1, 3, 3))
    print("All tests in `test_vjepa_l1_loss_is_exact_and_stops_target_grad` passed!")


def test_collapse_diagnostics_exact(collapse_fn: Callable | None = None) -> None:
    collapse_fn = collapse_fn or solutions.collapse_diagnostics
    structured = collapse_fn(t.eye(8), min_std=0.05, min_rank=2.0)
    collapsed = collapse_fn(t.ones(8, 8), min_std=0.05, min_rank=2.0)
    assert structured.non_collapsed and structured.effective_rank > 6.0
    assert not collapsed.non_collapsed
    assert collapsed.feature_std == 0.0
    assert collapsed.effective_rank == 0.0, "Constant features contain no nonzero centered direction."
    print("All tests in `test_collapse_diagnostics_exact` passed!")


def test_temporal_state_features_keep_order(
    feature_fn: Callable | None = None,
    rank_bound_fn: Callable | None = None,
) -> None:
    feature_fn = feature_fn or solutions.temporal_state_features
    rank_bound_fn = rank_bound_fn or solutions.temporal_state_feature_rank_bound
    tokens = t.arange(2 * 3 * 2 * 2 * 4).reshape(2, 12, 4).float()
    features = feature_fn(tokens, (3, 2, 2))
    reversed_features = feature_fn(tokens.reshape(2, 3, 4, 4).flip(1).reshape(2, 12, 4), (3, 2, 2))
    assert features.shape == (2, 16)
    assert not t.equal(features, reversed_features)
    t.testing.assert_close(features[:, -4:], -reversed_features[:, -4:])
    assert rank_bound_fn((3, 2, 2), 4) == 12
    random_features = feature_fn(t.randn(40, 12, 4), (3, 2, 2))
    centered = random_features - random_features.mean(dim=0, keepdim=True)
    assert int(t.linalg.matrix_rank(centered)) <= 12
    print("All tests in `test_temporal_state_features_keep_order` passed!")


def test_ridge_probe_and_random_labels(
    fit_fn: Callable | None = None,
    accuracy_fn: Callable | None = None,
) -> None:
    fit_fn = fit_fn or solutions.fit_ridge_probe
    accuracy_fn = accuracy_fn or solutions.probe_accuracy
    labels = t.arange(4).repeat_interleave(4)
    features = F.one_hot(labels, 4).float()
    weights = fit_fn(features, labels, n_classes=4, ridge=1e-4)
    assert accuracy_fn(features, labels, weights) == 1.0
    wrong = labels.roll(4)
    assert accuracy_fn(features, wrong, weights) == 0.0
    print("All tests in `test_ridge_probe_and_random_labels` passed!")


def test_probe_suite_progression(
    index_fn: Callable | None = None,
    fit_suite_fn: Callable | None = None,
    score_suite_fn: Callable | None = None,
) -> None:
    index_fn = index_fn or solutions.balanced_probe_fit_indices
    fit_suite_fn = fit_suite_fn or solutions.fit_probe_suite
    score_suite_fn = score_suite_fn or solutions.score_probe_suite
    world = solutions.make_moving_object_batch(64, seed=9)
    fit_indices = index_fn(world, 32)
    assert t.bincount(world.state_labels[fit_indices], minlength=8).tolist() == [4] * 8
    features = F.one_hot(world.state_labels, 8).float()
    probes = fit_suite_fn(features, world, fit_indices)
    scores = score_suite_fn(features, world, probes)
    assert scores == {"color": 1.0, "direction": 1.0, "state": 1.0}
    with pytest.raises(ValueError, match="multiple of 8"):
        index_fn(world, 30)
    print("All tests in `test_probe_suite_progression` passed!")


def test_retrieval_is_key_matched_and_tie_aware(
    details_fn: Callable | None = None,
    accuracy_fn: Callable | None = None,
) -> None:
    details_fn = details_fn or solutions.retrieval_details
    accuracy_fn = accuracy_fn or solutions.retrieval_accuracy
    gallery = t.eye(4)
    keys = t.arange(4)
    assert accuracy_fn(gallery, gallery, keys, keys) == 1.0
    with pytest.raises(ValueError, match="exactly once"):
        accuracy_fn(gallery, gallery, keys, keys.roll(1).where(keys != 3, t.tensor(0)))

    tied_gallery = t.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tied = details_fn(tied_gallery[:1], tied_gallery, t.tensor([0]), t.tensor([0, 1, 2]))
    assert tied.accuracy == 0.5
    assert tied.tie_counts.tolist() == [2]
    assert tied.true_ranks.tolist() == [1]
    print("All tests in `test_retrieval_is_key_matched_and_tie_aware` passed!")


def test_encode_and_future_helpers_match_direct_model_calls(
    encode_fn: Callable | None = None,
    teacher_fn: Callable | None = None,
    predict_fn: Callable | None = None,
) -> None:
    encode_fn = encode_fn or solutions.encode_state_features
    teacher_fn = teacher_fn or solutions.future_teacher_latents
    predict_fn = predict_fn or solutions.predict_future_features
    solutions.set_seed(17)
    model = solutions.SmallVJEPA(d_model=12, n_heads=3).eval()
    world = solutions.make_moving_object_batch(8, seed=17)
    encoded = encode_fn(model.target_encoder, world.videos, batch_size=3)
    tokens, grid = model.target_encoder.encode_all(world.videos)
    t.testing.assert_close(encoded, solutions.temporal_state_features(tokens, grid))
    teacher = teacher_fn(model.target_encoder, world.videos, batch_size=3)
    predicted, target = predict_fn(model, world.videos, batch_size=3)
    assert predicted.shape == teacher.shape == (8, 16 * 12)
    t.testing.assert_close(target, teacher)
    print("All tests in `test_encode_and_future_helpers_match_direct_model_calls` passed!")


def test_future_video_baselines_are_semantic(
    copy_fn: Callable | None = None,
    dynamics_video_fn: Callable | None = None,
    dynamics_frame_fn: Callable | None = None,
) -> None:
    copy_fn = copy_fn or solutions.copy_last_frame_video
    dynamics_video_fn = dynamics_video_fn or solutions.exact_dynamics_video
    dynamics_frame_fn = dynamics_frame_fn or solutions.exact_dynamics_prediction
    world = solutions.make_moving_object_batch(16, seed=4)
    copy_video = copy_fn(world)
    exact_video = dynamics_video_fn(world)
    t.testing.assert_close(copy_video[:, :4], world.videos[:, :4])
    t.testing.assert_close(copy_video[:, 4], world.videos[:, 3])
    t.testing.assert_close(copy_video[:, 5], world.videos[:, 3])
    t.testing.assert_close(exact_video, world.videos)
    t.testing.assert_close(dynamics_frame_fn(world), world.videos[:, 5])
    assert float(F.mse_loss(copy_video[:, 4:], world.videos[:, 4:])) > 0.005
    print("All tests in `test_future_video_baselines_are_semantic` passed!")


def test_blank_future_removes_future_without_mutation(blank_fn: Callable | None = None) -> None:
    blank_fn = blank_fn or solutions.blank_future_video
    world = solutions.make_moving_object_batch(16, seed=4)
    original = world.videos.clone()
    blank = blank_fn(world)
    t.testing.assert_close(blank[:, :4], original[:, :4], rtol=0, atol=0)
    assert t.count_nonzero(blank[:, 4:]) == 0, "Every future pixel must be erased."
    t.testing.assert_close(world.videos, original, rtol=0, atol=0)
    changed_future = original.clone()
    changed_future[:, 4:] = 7.0
    t.testing.assert_close(blank_fn(replace(world, videos=changed_future)), blank, rtol=0, atol=0)
    assert blank.data_ptr() != world.videos.data_ptr(), "Return a separate tensor."
    for context_frames in (0, 6):
        with pytest.raises(ValueError, match="context_frames"):
            blank_fn(world, context_frames=context_frames)
    print("All tests in `test_blank_future_removes_future_without_mutation` passed!")


def test_future_teacher_features_can_retain_context(teacher_fn: Callable | None = None) -> None:
    teacher_fn = teacher_fn or solutions.future_teacher_latents

    class GlobalMeanEncoder(nn.Module):
        def encode_all(self, video):
            # Analytic attention surrogate: each token receives the full clip mean.
            full_clip_mean = video.mean(dim=(1, 3, 4))
            return full_clip_mean[:, None, :].expand(-1, 3, -1), (3, 1, 1)

    video = t.zeros(2, 6, 3, 1, 1)
    video[0, :4, 0] = 1.0
    video[1, :4, 1] = 1.0
    actual = teacher_fn(GlobalMeanEncoder(), video)
    expected = F.layer_norm(t.tensor([[2 / 3, 0., 0.], [0., 2 / 3, 0.]]), (3,))
    t.testing.assert_close(actual, expected)
    assert not t.equal(actual[0], actual[1]), "Identical blank futures can have distinct target features."
    print("All tests in `test_future_teacher_features_can_retain_context` passed!")


def test_reverse_direction_labels_are_physical(reverse_fn: Callable | None = None) -> None:
    reverse_fn = reverse_fn or solutions.reverse_direction_labels
    labels = t.tensor([0, 1, 2, 3])
    assert reverse_fn(labels).tolist() == [1, 0, 3, 2]
    t.testing.assert_close(reverse_fn(reverse_fn(labels)), labels)
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        reverse_fn(t.tensor([4]))
    print("All tests in `test_reverse_direction_labels_are_physical` passed!")


def test_random_predictor_and_shared_teacher_scoring(
    random_fn: Callable | None = None,
    score_fn: Callable | None = None,
) -> None:
    random_fn = random_fn or solutions.random_predictor_features
    score_fn = score_fn or solutions.score_future_predictors
    solutions.set_seed(23)
    model = solutions.SmallVJEPA(d_model=12, n_heads=3).eval()
    world = solutions.make_moving_object_batch(16, seed=23)
    random_a = random_fn(model, world.videos, seed=5, batch_size=7)
    random_b = random_fn(model, world.videos, seed=5, batch_size=7)
    t.testing.assert_close(random_a, random_b)
    scores, details = score_fn(model, world, seed=23)
    assert set(scores) == {"vjepa", "blank_future", "copy_last_frame", "exact_dynamics", "random_predictor"}
    blank_video = t.zeros_like(world.videos)
    blank_video[:, :4] = world.videos[:, :4]
    blank_latents = solutions.future_teacher_latents(model.target_encoder, blank_video)
    targets = solutions.future_teacher_latents(model.target_encoder, world.videos)
    expected_cosine = float(F.cosine_similarity(blank_latents, targets, dim=-1).mean())
    expected_retrieval = solutions.retrieval_details(blank_latents, targets, world.clip_keys, world.clip_keys)
    assert scores["blank_future"].teacher_latent_cosine == pytest.approx(expected_cosine)
    assert scores["blank_future"].retrieval_accuracy == pytest.approx(expected_retrieval.accuracy)
    assert scores["exact_dynamics"].teacher_latent_cosine >= 0.999999
    assert scores["exact_dynamics"].retrieval_accuracy == 1.0
    assert details.top1_indices.shape == (16,)
    print("All tests in `test_random_predictor_and_shared_teacher_scoring` passed!")


def test_training_updates_online_not_target_directly(
    train_fn: Callable | None = None,
    step_fn: Callable | None = None,
) -> None:
    train_fn = train_fn or solutions.train_small_vjepa
    step_fn = step_fn or solutions.vjepa_train_step
    world = solutions.make_moving_object_batch(32, seed=5)
    solutions.set_seed(5)
    model = solutions.SmallVJEPA(d_model=12, n_heads=3)
    initial_context = next(model.context_encoder.parameters()).detach().clone()
    initial_target = next(model.target_encoder.parameters()).detach().clone()
    trained, history = train_fn(
        world,
        model=model,
        steps=6,
        batch_size=8,
        learning_rate=3e-3,
        seed=5,
    )
    assert set(history) == {"vjepa_l1"}
    assert len(history["vjepa_l1"]) == 6
    assert all(t.isfinite(t.tensor(history[key])).all() for key in history)
    assert not t.equal(initial_context, next(trained.context_encoder.parameters()))
    assert not t.equal(initial_target, next(trained.target_encoder.parameters()))
    assert all(parameter.grad is None for parameter in trained.target_encoder.parameters())

    step_globals = step_fn.__globals__
    original_mask_fn = step_globals["sample_spatial_tube_mask"]
    seen_masks = []

    def recording_mask(*args, **kwargs):
        mask = original_mask_fn(*args, **kwargs)
        seen_masks.append(mask.detach().cpu())
        return mask

    step_globals["sample_spatial_tube_mask"] = recording_mask
    try:
        train_fn(world, model=solutions.SmallVJEPA(d_model=12, n_heads=3), steps=1, batch_size=8, seed=6)
    finally:
        step_globals["sample_spatial_tube_mask"] = original_mask_fn
    assert len(seen_masks) == 1
    tube_mask = seen_masks[0].reshape(8, 3, 4, 4)
    t.testing.assert_close(tube_mask[:, 0], tube_mask[:, 1])
    t.testing.assert_close(tube_mask[:, 1], tube_mask[:, 2])
    print("All tests in `test_training_updates_online_not_target_directly` passed!")


def test_training_contract_is_l1_only(
    train_fn: Callable | None = None,
    step_fn: Callable | None = None,
) -> None:
    train_fn = train_fn or solutions.train_small_vjepa
    step_fn = step_fn or solutions.vjepa_train_step
    parameter_names = set(inspect.signature(train_fn).parameters)
    assert "variance_weight" not in parameter_names
    assert "covariance_weight" not in parameter_names
    step_names = set(inspect.unwrap(step_fn).__code__.co_names)
    assert "vjepa_l1_loss" in step_names
    assert "sample_spatial_tube_mask" in step_names
    assert "ema_update" in step_names
    assert not {"variance_hinge_loss", "covariance_loss", "smooth_l1_loss"} & step_names
    print("All tests in `test_training_contract_is_l1_only` passed!")


def test_short_training_improves_latent_objective(
    train_fn: Callable | None = None,
    model_cls: type[nn.Module] | None = None,
) -> None:
    train_fn = train_fn or solutions.train_small_vjepa
    model_cls = model_cls or solutions.SmallVJEPA
    world = solutions.make_moving_object_batch(64, seed=7)
    model, history = train_fn(
        world,
        model=model_cls(d_model=12, n_heads=3),
        steps=36,
        batch_size=16,
        learning_rate=3e-3,
        seed=7,
    )
    first = sum(history["vjepa_l1"][:6]) / 6
    last = sum(history["vjepa_l1"][-6:]) / 6
    assert last < first * 0.8, (
        f"The short real training run should lower latent loss by at least 20%; got {first:.4f} -> {last:.4f}."
    )
    assert isinstance(model, model_cls)
    print("All tests in `test_short_training_improves_latent_objective` passed!")


def test_complete_evaluator_intervals_and_fingerprint(
    evaluate_fn: Callable | None = None,
    summary_fn: Callable | None = None,
    fingerprint_fn: Callable | None = None,
) -> None:
    evaluate_fn = evaluate_fn or solutions.evaluate_small_vjepa
    summary_fn = summary_fn or solutions.summarize_multiseed
    fingerprint_fn = fingerprint_fn or solutions.evaluation_fingerprint
    train_world, test_world = solutions.make_disjoint_worlds(32, 16, seed=29)
    solutions.set_seed(29)
    model = solutions.SmallVJEPA(d_model=12, n_heads=3).eval()
    metrics = evaluate_fn(model, train_world, test_world, {"vjepa_l1": [1.0]}, seed=29, probe_train_examples=16)
    assert metrics.duplicate_audit.is_disjoint
    assert metrics.reversal_audit.collision_free
    assert metrics.effective_rank_max == 36
    assert set(metrics.predictors) == {"vjepa", "blank_future", "copy_last_frame", "exact_dynamics", "random_predictor"}
    assert metrics.predictors["exact_dynamics"].teacher_latent_cosine >= 0.999999
    assert metrics.predictors["exact_dynamics"].retrieval_accuracy == 1.0
    changed = replace(metrics, state_probe_accuracy=metrics.state_probe_accuracy + 0.25)
    summary = summary_fn({29: metrics, 30: changed})
    interval = summary["state_probe_accuracy"]
    assert interval.minimum == metrics.state_probe_accuracy
    assert interval.maximum == changed.state_probe_accuracy
    assert interval.mean == (metrics.state_probe_accuracy + changed.state_probe_accuracy) / 2
    assert fingerprint_fn(metrics) == fingerprint_fn(solutions.metrics_dict(metrics))
    assert fingerprint_fn(metrics) != fingerprint_fn(changed)
    print("All tests in `test_complete_evaluator_intervals_and_fingerprint` passed!")


def test_signature_figure_uses_honest_axes(plot_fn: Callable | None = None) -> None:
    plot_fn = plot_fn or solutions.make_signature_figure
    train_world, test_world = solutions.make_disjoint_worlds(16, 8, seed=31)
    solutions.set_seed(31)
    model = solutions.SmallVJEPA(d_model=12, n_heads=3).eval()
    history = {"vjepa_l1": [1.0, 0.8]}
    metrics = solutions.evaluate_small_vjepa(
        model, train_world, test_world, history, seed=31, probe_train_examples=16
    )
    predicted, target = solutions.predict_future_features(model, test_world.videos)
    retrieval = solutions.retrieval_details(predicted, target, test_world.clip_keys, test_world.clip_keys)
    fig = plot_fn(history, metrics, test_world, retrieval)
    titled_axes = {axis.get_title(): axis for axis in fig.axes if axis.get_title()}
    assert "C. Cosine, same target" in titled_axes
    assert "D. Retrieval, same gallery" in titled_axes
    assert "E. Collapse check" in titled_axes
    assert titled_axes["C. Cosine, same target"].get_ylabel() == "matched teacher-latent cosine"
    assert titled_axes["D. Retrieval, same gallery"].get_ylabel() == "tie-aware top-1 accuracy"
    assert titled_axes["E. Collapse check"].get_ylim()[1] == 36
    assert "max 36" in titled_axes["E. Collapse check"].get_ylabel()
    assert all("ties=" in label.get_text() for label in titled_axes["D. Retrieval, same gallery"].get_xticklabels())
    plt.close(fig)
    print("All tests in `test_signature_figure_uses_honest_axes` passed!")


def test_notebook_final_results_use_learner_functions() -> None:
    exercise_path = SECTION_DIR / "14.3_V_JEPA_from_Scratch_exercises.ipynb"
    solution_path = SECTION_DIR / "14.3_V_JEPA_from_Scratch_solutions.ipynb"
    notebook_defs = {}
    for path in (exercise_path, solution_path):
        notebook = json.loads(path.read_text())
        code_sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        sources = ["".join(cell["source"]) for cell in notebook["cells"]]
        final_run = next(source for source in sources if source.startswith("EVAL_SEEDS ="))
        play_cell = next(source for source in sources if source.startswith("play_mode ="))
        trees = [ast.parse(source) for source in code_sources]
        definitions = [
            node.name
            for tree in trees
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        notebook_defs[path.name] = definitions
        assert "reference." not in final_run
        for learner_name in (
            "make_disjoint_worlds",
            "train_small_vjepa",
            "evaluate_small_vjepa",
            "summarize_multiseed",
            "predict_future_features",
            "retrieval_details",
            "make_signature_figure",
        ):
            assert sum(name == learner_name for name in definitions) == 1
        final_tree = ast.parse(final_run)
        final_calls = {
            node.func.id
            for node in ast.walk(final_tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert {
            "validate_learner_pipeline_dependencies",
            "make_disjoint_worlds",
            "train_small_vjepa",
            "evaluate_small_vjepa",
            "summarize_multiseed",
            "predict_future_features",
            "retrieval_details",
            "make_signature_figure",
        } <= final_calls
        assert "expand_tubelet_mask" in play_cell
        assert "gallery_target" in play_cell and "top1_index" in play_cell
        assert any("test_tubelet_grid_and_embedding(tubelet_grid, TubeletEmbedding)" in source for source in sources)
        assert any("test_masked_pixels_cannot_change_visible_context(VideoEncoder)" in source for source in sources)
        train_cell = next(source for source in code_sources if "def vjepa_train_step(" in source)
        if path == exercise_path:
            assert "def train_small_vjepa(" in train_cell and "YOUR CODE HERE" in train_cell
        else:
            assert "def train_small_vjepa(" in train_cell and "YOUR CODE HERE" not in train_cell
            assert "vjepa_train_step(" in train_cell
    assert sorted(notebook_defs[exercise_path.name]) == sorted(notebook_defs[solution_path.name])
    print("All tests in `test_notebook_final_results_use_learner_functions` passed!")


def test_dependency_validator_rejects_reference_fallback() -> None:
    namespace = {name: getattr(solutions, name) for name in {
        *solutions.LEARNER_DEPENDENCY_GRAPH,
        *(dependency for values in solutions.LEARNER_DEPENDENCY_GRAPH.values() for dependency in values),
        "make_disjoint_worlds",
        "summarize_multiseed",
        "make_signature_figure",
    }}
    with pytest.raises(RuntimeError, match="Reference implementations"):
        solutions.validate_learner_pipeline_dependencies(namespace, solutions)
    print("All tests in `test_dependency_validator_rejects_reference_fallback` passed!")


def test_solution_dropdowns_complete_the_actual_learner() -> None:
    learner = json.loads((SECTION_DIR / "14.3_V_JEPA_from_Scratch_exercises.ipynb").read_text())
    solved = json.loads((SECTION_DIR / "14.3_V_JEPA_from_Scratch_solutions.ipynb").read_text())
    answer = ""
    completed = 0
    for exercise_cell, solved_cell in zip(learner["cells"], solved["cells"], strict=True):
        source = "".join(exercise_cell["source"])
        if exercise_cell["cell_type"] == "markdown":
            if "<summary>Solution</summary>" in source:
                answer = re.search(r"<summary>Solution</summary>\s*```python\n(.*?)\n```", source, re.S)[1]
                ast.parse(answer)  # A copied dropdown must be valid Python.
            continue
        lines = source.splitlines(keepends=True)
        replacements = []
        for node in ast.parse(source).body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
            if "YOUR CODE HERE" not in "".join(lines[start:node.end_lineno]):
                continue
            answer_node = next(item for item in ast.parse(answer).body if getattr(item, "name", None) == node.name)
            answer_start = min([answer_node.lineno, *(item.lineno for item in answer_node.decorator_list)]) - 1
            replacement = "\n".join(answer.splitlines()[answer_start:answer_node.end_lineno]) + "\n"
            replacements.append((start, node.end_lineno, replacement))
        for start, end, replacement in reversed(replacements):
            lines[start:end] = [replacement]
        completed += bool(replacements)
        actual = ast.dump(ast.parse("".join(lines)))
        expected = ast.dump(ast.parse("".join(solved_cell["source"])))
        assert actual == expected, "Solved cells must preserve provided helpers and immediate learner tests."
    assert completed == 15
    print("All tests in `test_solution_dropdowns_complete_the_actual_learner` passed!")


def test_reference_metrics_meet_preregistered_thresholds() -> None:
    payload = json.loads((SECTION_DIR / "expected_outputs" / "reference_metrics.json").read_text())
    assert payload["protocol_version"] == solutions.EVALUATION_PROTOCOL_VERSION
    assert payload["run"]["device"] == "cpu"
    assert payload["run"]["seeds"] == [0, 1, 2]
    thresholds = payload["acceptance_thresholds"]
    for metrics in payload["per_seed"].values():
        assert metrics["train_loss_ratio"] <= thresholds["train_loss_ratio_max"]
        assert metrics["random_label_state_accuracy"] <= thresholds["random_label_state_accuracy_max"]
        assert abs(metrics["predictors"]["random_predictor"]["teacher_latent_cosine"]) <= thresholds[
            "random_predictor_teacher_latent_cosine_max"
        ]
        assert metrics["predictors"]["random_predictor"]["retrieval_accuracy"] <= thresholds[
            "random_predictor_retrieval_accuracy_max"
        ]
        assert metrics["predictors"]["exact_dynamics"]["teacher_latent_cosine"] >= thresholds[
            "exact_dynamics_teacher_latent_cosine_min"
        ]
        assert metrics["predictors"]["exact_dynamics"]["retrieval_accuracy"] >= thresholds[
            "exact_dynamics_retrieval_accuracy_min"
        ]
        assert metrics["feature_std"] >= thresholds["feature_std_min"]
        assert metrics["effective_rank"] >= thresholds["effective_rank_min"]
        assert metrics["effective_rank_max"] == 72
        assert metrics["duplicate_audit"]["is_disjoint"]
        assert metrics["duplicate_audit"]["train_duplicates"] == 0
        assert metrics["duplicate_audit"]["test_duplicates"] == 0
        assert metrics["duplicate_audit"]["train_test_overlap"] == 0
        assert metrics["reversal_audit"]["collision_free"]
        assert metrics["reversal_audit"]["reversed_pretraining_overlap"] == 0
        assert metrics["reversal_audit"]["reversed_probe_fit_overlap"] == 0
        assert metrics["reversal_audit"]["reversed_test_overlap"] == 0
        assert all(score["max_tie_count"] >= 1 for score in metrics["predictors"].values())
    print("All tests in `test_reference_metrics_meet_preregistered_thresholds` passed!")


def test_section_metadata_matches_reference_protocol() -> None:
    reference = json.loads((SECTION_DIR / "expected_outputs" / "reference_metrics.json").read_text())
    smoke = json.loads((SECTION_DIR / "expected_outputs" / "smoke_test.json").read_text())
    artifact_lock = yaml.safe_load((SECTION_DIR / "artifacts.lock.yml").read_text())

    protocol = reference["protocol_version"]
    assert protocol == solutions.EVALUATION_PROTOCOL_VERSION
    assert smoke["protocol_version"] == protocol
    assert artifact_lock["evaluation_protocol"] == protocol
    assert "Persistence-Shortcut Lab" in artifact_lock["title"]
    assert smoke["primary_training_history_keys"] == ["vjepa_l1"]
    assert smoke["active_auxiliary_regularizers"] == []
    assert smoke["spatial_tube_training_target_tokens"] == 12
    assert smoke["future_diagnostic_target_tokens"] == 16
    assert smoke["state_readout_columns"] == 96
    assert smoke["state_readout_effective_rank_max"] == 72
    assert artifact_lock["expected_metrics"]["effective_rank_attainable_max"] == 72
    assert artifact_lock["expected_metrics"]["tied_queries_all_methods"] == 0
    assert "no_auxiliary_variance_or_covariance_penalty" in artifact_lock["simplifications"]
    assert "full_future_mask_used_only_as_out_of_distribution_diagnostic" in artifact_lock["simplifications"]
    assert all(value == 0 for value in smoke["required_reversal_audit"].values())
    assert all(
        artifact_lock["datasets"][0][field] == 0
        for field in (
            "reversed_pretraining_key_overlap",
            "reversed_probe_fit_key_overlap",
            "reversed_test_key_overlap",
        )
    )
    print("All tests in `test_section_metadata_matches_reference_protocol` passed!")


def test_reference_assets_and_metrics_are_fresh() -> None:
    metrics_path = SECTION_DIR / "expected_outputs" / "reference_metrics.json"
    payload = json.loads(metrics_path.read_text())
    freshness = payload["freshness"]
    signature_path = SECTION_DIR.parents[1] / "instructions" / "assets" / "vjepa_from_scratch_signature.png"
    world_path = SECTION_DIR.parents[1] / "instructions" / "assets" / "vjepa_from_scratch_world.png"

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert sha256(SECTION_DIR / "solutions.py") == freshness["solutions_sha256"]
    assert sha256(SECTION_DIR / "regenerate_reference.py") == freshness["generator_sha256"]
    assert sha256(signature_path) == freshness["signature_asset_sha256"]
    assert sha256(world_path) == freshness["world_asset_sha256"]

    interval_payload = payload["intervals"]
    expected_fingerprint = solutions.evaluation_fingerprint(
        {"seed_metrics": payload["per_seed"]["0"], "intervals": interval_payload}
    )
    assert expected_fingerprint == freshness["signature_evaluation_fingerprint"]
    with Image.open(signature_path) as image:
        assert image.info["EvaluationProtocol"] == payload["protocol_version"]
        assert image.info["EvaluationFingerprint"] == expected_fingerprint
    with Image.open(world_path) as image:
        assert image.info["EvaluationProtocol"] == payload["protocol_version"]
    print("All tests in `test_reference_assets_and_metrics_are_fresh` passed!")
