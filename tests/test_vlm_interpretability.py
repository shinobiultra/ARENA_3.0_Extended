import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.vlm_interpretability import (
        bbox_to_patch_indices,
        clip_contrastive_logits,
        clothing_geometry_report,
        contrastive_alignment_report,
        controlled_vlm_baseline_report,
        generate_synthetic_clothing_scenes,
        generate_synthetic_colored_shape_scenes,
        modality_arbitration_report,
        object_hallucination_report,
        patch_visual_token_activations,
        same_size_non_overlapping_token_control,
        siglip_pairwise_loss,
        visual_region_patch_report,
        visual_sequence_patch_report,
        visual_token_attribution_report,
    )


def test_clip_contrastive_logits_aligns_paired_embeddings():
    image_embeddings = t.eye(3)
    text_embeddings = t.eye(3)

    logits = clip_contrastive_logits(
        image_embeddings,
        text_embeddings,
        logit_scale=5.0,
    )
    report = contrastive_alignment_report(
        logits,
        min_accuracy=1.0,
        min_positive_margin=4.0,
    )

    assert logits.diag().tolist() == pytest.approx([5.0, 5.0, 5.0])
    assert report.image_to_text_accuracy == 1.0
    assert report.text_to_image_accuracy == 1.0
    assert report.mean_positive_margin == pytest.approx(5.0)
    assert report.aligned


def test_siglip_pairwise_loss_rewards_positive_and_negative_pairs():
    logits = t.tensor([[4.0, -4.0], [-3.0, 3.0]])
    labels = t.eye(2)

    loss = siglip_pairwise_loss(logits, labels)

    assert loss.item() < 0.05


def test_visual_token_attribution_report_requires_localized_mass():
    token_activations = t.tensor(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
        ]
    )
    text_direction = t.tensor([1.0, 0.0])

    report = visual_token_attribution_report(
        token_activations,
        text_direction,
        top_k=2,
        min_top_token_mass=0.8,
    )

    assert report.top_token_indices.tolist() == [1, 2]
    assert report.top_token_mass == pytest.approx(1.0)
    assert report.localized


def test_bbox_to_patch_indices_handles_cls_and_patch_overlap():
    indices = bbox_to_patch_indices(
        (32, 32, 96, 96),
        image_size=(224, 224),
        patch_size=32,
        has_cls_token=True,
    )

    assert indices == (9, 10, 16, 17)
    assert all(index > 0 for index in indices)


def test_same_size_non_overlapping_token_control_is_disjoint_and_deterministic():
    object_indices = (2, 3, 4)
    first = same_size_non_overlapping_token_control(
        object_indices,
        num_tokens=12,
        protected_indices=(0, 1),
        seed=7,
    )
    second = same_size_non_overlapping_token_control(
        object_indices,
        num_tokens=12,
        protected_indices=(0, 1),
        seed=7,
    )

    assert first == second
    assert len(first) == len(object_indices)
    assert not (set(first) & set(object_indices))
    assert not (set(first) & {0, 1})


def test_patch_visual_token_activations_clones_and_replaces_only_selected_tokens():
    clean = t.zeros(2, 4, 3)
    corrupt = t.arange(24, dtype=t.float32).reshape(2, 4, 3)

    patched = patch_visual_token_activations(clean, corrupt, [1, 3])

    assert patched is not clean
    t.testing.assert_close(patched[:, 1], corrupt[:, 1])
    t.testing.assert_close(patched[:, 3], corrupt[:, 3])
    t.testing.assert_close(patched[:, 0], clean[:, 0])
    t.testing.assert_close(patched[:, 2], clean[:, 2])
    t.testing.assert_close(clean, t.zeros_like(clean))


def test_object_hallucination_report_flags_text_prior_without_visual_evidence():
    report = object_hallucination_report(
        object_score=0.9,
        visual_evidence_score=0.2,
        text_prior_score=0.8,
        min_object_score=0.7,
        min_visual_evidence=0.5,
        min_text_prior_gap=0.4,
    )

    assert report.text_prior_gap == pytest.approx(0.6)
    assert report.flags_hallucination


def test_modality_arbitration_report_prefers_visual_answer_under_conflict():
    candidate_scores = t.tensor([0.1, 0.8, 0.2])

    report = modality_arbitration_report(
        candidate_scores,
        visual_index=1,
        text_prior_index=2,
        min_visual_margin=0.5,
    )

    assert report.visual_margin == pytest.approx(0.6)
    assert report.trusts_visual_evidence


def test_generate_synthetic_colored_shape_scenes_has_counterfactual_schema():
    scenes = generate_synthetic_colored_shape_scenes(
        colors=("red", "blue"),
        shapes=("cube", "sphere"),
        split="test",
    )

    assert len(scenes) == 4
    assert scenes[0].image_id == "test_cube_red"
    assert scenes[0].question == "What color is the cube?"
    assert scenes[0].answer == "red"
    assert scenes[0].counterfactual_answer == "blue"
    assert scenes[0].spurious_text == "blue"
    assert scenes[0].bbox[0] < scenes[0].bbox[2]
    assert scenes[0].bbox[1] < scenes[0].bbox[3]


def test_synthetic_clothing_geometry_rejects_text_prior_labels():
    scenes = generate_synthetic_clothing_scenes(split="heldout")
    assert len(scenes) == 8
    assert scenes[0].question == "What color is the formal shirt?"
    assert scenes[0].spurious_text == "blue"

    train_embeddings = t.tensor(
        [
            [3.0, 0.0, 2.0, 0.0, 1.5, 0.0],
            [3.0, 0.0, 0.0, 2.0, 0.0, 1.5],
            [0.0, 3.0, 2.0, 0.0, 0.0, 1.5],
            [0.0, 3.0, 0.0, 2.0, 1.5, 0.0],
        ]
    )
    heldout_embeddings = train_embeddings + 0.05
    garment_labels = t.tensor([0, 0, 1, 1])
    color_labels = t.tensor([0, 1, 0, 1])
    style_labels = t.tensor([0, 1, 1, 0])
    flipped_color_labels = 1 - color_labels
    random_color_labels = t.tensor([0, 1, 1, 0])

    report = clothing_geometry_report(
        train_embeddings,
        heldout_embeddings,
        garment_labels,
        garment_labels,
        color_labels,
        color_labels,
        style_labels,
        style_labels,
        flipped_color_labels,
        random_color_labels,
    )

    assert report.garment_accuracy == pytest.approx(1.0)
    assert report.color_accuracy == pytest.approx(1.0)
    assert report.style_accuracy == pytest.approx(1.0)
    assert report.text_prior_color_agreement == pytest.approx(0.0)
    assert report.random_color_agreement == pytest.approx(0.5)
    assert report.predicts_clothing_factors
    assert report.rejects_text_prior
    assert report.rejects_random_labels


def test_synthetic_clothing_geometry_rejects_reused_random_label_control():
    embeddings = t.eye(4)
    garment_labels = t.tensor([0, 0, 1, 1])
    color_labels = t.tensor([0, 1, 0, 1])
    style_labels = t.tensor([0, 1, 1, 0])
    flipped_color_labels = 1 - color_labels

    with pytest.raises(ValueError, match="random_color_labels"):
        clothing_geometry_report(
            embeddings,
            embeddings,
            garment_labels,
            garment_labels,
            color_labels,
            color_labels,
            style_labels,
            style_labels,
            flipped_color_labels,
            flipped_color_labels,
        )


def test_controlled_vlm_baseline_report_requires_image_grounding():
    labels = t.tensor([0, 1, 0, 1])
    joint_logits = t.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]])
    image_only_logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5]])
    text_only_logits = t.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    report = controlled_vlm_baseline_report(
        joint_logits,
        image_only_logits,
        text_only_logits,
        labels,
        max_text_only_accuracy=0.5,
    )

    assert report.joint_accuracy == pytest.approx(1.0)
    assert report.image_only_accuracy == pytest.approx(1.0)
    assert report.text_only_accuracy == pytest.approx(0.5)
    assert report.joint_beats_text_only
    assert report.text_only_fails_image_questions


def test_visual_region_patch_report_localizes_object_tokens():
    clean_contributions = t.tensor(
        [
            [2.0, -1.0],
            [1.0, -0.5],
            [0.1, 0.0],
            [0.0, 0.1],
        ]
    )
    corrupt_contributions = t.tensor(
        [
            [-1.0, 2.0],
            [-0.5, 1.0],
            [0.1, 0.0],
            [0.0, 0.1],
        ]
    )

    report = visual_region_patch_report(
        clean_contributions,
        corrupt_contributions,
        object_token_indices=[0, 1],
        background_token_indices=[2],
        random_token_indices=[2, 3],
        target_index=0,
        counterfactual_index=1,
        min_object_gap=1.0,
    )

    assert report.clean_margin == pytest.approx(4.5)
    assert report.object_patch_margin == pytest.approx(-4.5)
    assert report.object_patch_effect == pytest.approx(9.0)
    assert report.background_patch_effect == pytest.approx(0.0)
    assert report.random_patch_effect == pytest.approx(0.0)
    assert report.object_beats_background
    assert report.object_beats_random
    assert report.flips_answer


def test_visual_region_patch_report_rejects_smaller_random_control():
    clean_contributions = t.zeros(4, 2)
    corrupt_contributions = t.zeros(4, 2)

    with pytest.raises(ValueError, match="same number of tokens"):
        visual_region_patch_report(
            clean_contributions,
            corrupt_contributions,
            object_token_indices=[0, 1],
            background_token_indices=[2],
            random_token_indices=[3],
            target_index=0,
            counterfactual_index=1,
        )


def test_visual_sequence_patch_report_requires_object_and_full_sequence_controls():
    clean_logits = t.tensor([[4.0, 0.0], [0.0, 4.0]])
    corrupt_logits = t.tensor([[0.0, 4.0], [4.0, 0.0]])
    background_logits = t.tensor([[3.8, 0.0], [0.0, 3.8]])
    random_logits = t.tensor([[3.7, 0.0], [0.0, 3.7]])

    report = visual_sequence_patch_report(
        clean_logits,
        corrupt_logits,
        corrupt_logits,
        background_logits,
        random_logits,
        corrupt_logits,
        target_indices=t.tensor([0, 1]),
        counterfactual_indices=t.tensor([1, 0]),
        min_object_gap=1.0,
    )

    assert report.object_patch_flips_answer
    assert report.background_patch_preserves_answer
    assert report.random_patch_preserves_answer
    assert report.full_sequence_patch_flips_answer
    assert report.full_sequence_patch_matches_corrupt
    assert report.object_beats_background
    assert report.object_beats_random
    assert report.passes_activation_patching_controls
    assert report.full_sequence_patch_max_abs_margin_error == pytest.approx(0.0)


def test_visual_sequence_patch_report_rejects_missing_full_sequence_match():
    clean_logits = t.tensor([[4.0, 0.0], [0.0, 4.0]])
    corrupt_logits = t.tensor([[0.0, 4.0], [4.0, 0.0]])
    weak_full_patch_logits = t.tensor([[1.0, 2.0], [2.0, 1.0]])

    report = visual_sequence_patch_report(
        clean_logits,
        corrupt_logits,
        corrupt_logits,
        clean_logits,
        clean_logits,
        weak_full_patch_logits,
        target_indices=t.tensor([0, 1]),
        counterfactual_indices=t.tensor([1, 0]),
        min_object_gap=1.0,
    )

    assert not report.full_sequence_patch_matches_corrupt
    assert not report.passes_activation_patching_controls
