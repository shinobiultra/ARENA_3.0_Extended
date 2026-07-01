"""Vision-language interpretability utilities for local VLM notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t
import torch.nn.functional as F


@dataclass(frozen=True)
class ContrastiveAlignmentReport:
    image_to_text_accuracy: float
    text_to_image_accuracy: float
    mean_positive_margin: float
    aligned: bool


@dataclass(frozen=True)
class VisualTokenAttributionReport:
    token_scores: t.Tensor
    top_token_indices: t.Tensor
    top_token_mass: float
    localized: bool


@dataclass(frozen=True)
class ObjectHallucinationReport:
    object_score: float
    visual_evidence_score: float
    text_prior_score: float
    text_prior_gap: float
    flags_hallucination: bool


@dataclass(frozen=True)
class ModalityArbitrationReport:
    visual_choice_score: float
    text_prior_choice_score: float
    visual_margin: float
    trusts_visual_evidence: bool


@dataclass(frozen=True)
class SyntheticVLMScene:
    image_id: str
    shape: str
    color: str
    bbox: tuple[float, float, float, float]
    question: str
    answer: str
    counterfactual_answer: str
    spurious_text: str | None
    split: str


@dataclass(frozen=True)
class SyntheticClothingScene:
    image_id: str
    garment_type: str
    color: str
    style: str
    bbox: tuple[float, float, float, float]
    question: str
    answer: str
    counterfactual_answer: str
    spurious_text: str | None
    split: str


@dataclass(frozen=True)
class ClothingGeometryReport:
    garment_accuracy: float
    color_accuracy: float
    style_accuracy: float
    text_prior_color_agreement: float
    random_color_agreement: float
    predicts_clothing_factors: bool
    rejects_text_prior: bool
    rejects_random_labels: bool


@dataclass(frozen=True)
class ControlledVLMBaselineReport:
    joint_accuracy: float
    image_only_accuracy: float
    text_only_accuracy: float
    joint_beats_text_only: bool
    text_only_fails_image_questions: bool


@dataclass(frozen=True)
class VisualRegionPatchReport:
    clean_margin: float
    object_patch_margin: float
    background_patch_margin: float
    random_patch_margin: float
    object_patch_effect: float
    background_patch_effect: float
    random_patch_effect: float
    object_beats_background: bool
    object_beats_random: bool
    flips_answer: bool


@dataclass(frozen=True)
class VisualSequencePatchReport:
    clean_margins: list[float]
    corrupt_margins: list[float]
    object_patch_margins: list[float]
    background_patch_margins: list[float]
    random_patch_margins: list[float]
    full_sequence_patch_margins: list[float]
    object_patch_effects: list[float]
    background_patch_effects: list[float]
    random_patch_effects: list[float]
    full_sequence_patch_effects: list[float]
    min_object_gap_over_background: float
    min_object_gap_over_random: float
    full_sequence_patch_max_abs_margin_error: float
    object_patch_flips_answer: bool
    background_patch_preserves_answer: bool
    random_patch_preserves_answer: bool
    full_sequence_patch_flips_answer: bool
    full_sequence_patch_matches_corrupt: bool
    object_beats_background: bool
    object_beats_random: bool
    passes_activation_patching_controls: bool


def _l2_normalize(values: t.Tensor, *, eps: float = 1e-8) -> t.Tensor:
    return values.float() / values.float().norm(dim=-1, keepdim=True).clamp_min(eps)


def bbox_to_patch_indices(
    bbox: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
    patch_size: int,
    has_cls_token: bool,
) -> tuple[int, ...]:
    """Map a pixel-space bounding box to flattened ViT patch-token indices."""

    image_width, image_height = image_size
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if image_width % patch_size != 0 or image_height % patch_size != 0:
        raise ValueError("image dimensions must be divisible by patch_size.")
    if not (0 <= bbox[0] < bbox[2] <= image_width and 0 <= bbox[1] < bbox[3] <= image_height):
        raise ValueError("bbox must lie inside the image and have positive area.")

    grid_width = image_width // patch_size
    grid_height = image_height // patch_size
    offset = 1 if has_cls_token else 0
    indices: list[int] = []
    for row in range(grid_height):
        for col in range(grid_width):
            patch_bbox = (
                col * patch_size,
                row * patch_size,
                (col + 1) * patch_size,
                (row + 1) * patch_size,
            )
            x1 = max(bbox[0], patch_bbox[0])
            y1 = max(bbox[1], patch_bbox[1])
            x2 = min(bbox[2], patch_bbox[2])
            y2 = min(bbox[3], patch_bbox[3])
            if max(0, x2 - x1) * max(0, y2 - y1) > 0:
                indices.append(offset + row * grid_width + col)
    if not indices:
        raise ValueError("bbox does not overlap any patch tokens.")
    return tuple(indices)


def same_size_non_overlapping_token_control(
    object_indices: tuple[int, ...] | list[int],
    *,
    num_tokens: int,
    seed: int = 0,
    protected_indices: tuple[int, ...] | list[int] = (),
) -> tuple[int, ...]:
    """Sample a same-size token control disjoint from object and protected tokens."""

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive.")
    object_set = set(int(index) for index in object_indices)
    protected_set = set(int(index) for index in protected_indices)
    if not object_set:
        raise ValueError("object_indices must be nonempty.")
    if min(object_set | protected_set) < 0 or max(object_set | protected_set) >= num_tokens:
        raise ValueError("token index is out of range.")

    available = [
        index
        for index in range(num_tokens)
        if index not in object_set and index not in protected_set
    ]
    if len(available) < len(object_set):
        raise ValueError("not enough non-object tokens for a same-size control.")
    generator = t.Generator(device="cpu").manual_seed(seed)
    permutation = t.randperm(len(available), generator=generator).tolist()
    return tuple(available[index] for index in permutation[: len(object_set)])


def patch_visual_token_activations(
    clean_activations: t.Tensor,
    corrupt_activations: t.Tensor,
    token_indices: tuple[int, ...] | list[int],
) -> t.Tensor:
    """Clone clean visual-token activations and replace selected tokens from corrupt."""

    if clean_activations.shape != corrupt_activations.shape:
        raise ValueError("clean and corrupt activations must have matching shape.")
    if clean_activations.ndim != 3:
        raise ValueError("visual activations must have shape (batch, tokens, d_model).")
    if not token_indices:
        raise ValueError("token_indices must be nonempty.")
    index_tensor = t.tensor(
        list(token_indices),
        dtype=t.long,
        device=clean_activations.device,
    )
    if index_tensor.unique().numel() != index_tensor.numel():
        raise ValueError("token_indices must be unique.")
    if index_tensor.min().item() < 0 or index_tensor.max().item() >= clean_activations.shape[1]:
        raise ValueError("token index is out of range.")
    patched = clean_activations.clone()
    patched[:, index_tensor] = corrupt_activations[:, index_tensor].to(
        dtype=patched.dtype,
        device=patched.device,
    )
    return patched


def generate_synthetic_colored_shape_scenes(
    *,
    colors: tuple[str, ...] = ("red", "blue"),
    shapes: tuple[str, ...] = ("cube", "sphere"),
    split: str = "train",
    include_spurious_text: bool = True,
) -> tuple[SyntheticVLMScene, ...]:
    """Generate controlled object/color VLM scenes with counterfactual labels."""

    if len(colors) < 2:
        raise ValueError("at least two colors are required for counterfactual labels.")
    if len(shapes) == 0:
        raise ValueError("at least one shape is required.")

    scenes = []
    for shape_index, shape in enumerate(shapes):
        for color_index, color in enumerate(colors):
            counterfactual = colors[(color_index + 1) % len(colors)]
            x1 = 0.1 + 0.15 * color_index
            y1 = 0.2 + 0.12 * shape_index
            x2 = min(x1 + 0.25, 0.95)
            y2 = min(y1 + 0.25, 0.95)
            scenes.append(
                SyntheticVLMScene(
                    image_id=f"{split}_{shape}_{color}",
                    shape=shape,
                    color=color,
                    bbox=(round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)),
                    question=f"What color is the {shape}?",
                    answer=color,
                    counterfactual_answer=counterfactual,
                    spurious_text=counterfactual if include_spurious_text else None,
                    split=split,
                )
            )
    return tuple(scenes)


def generate_synthetic_clothing_scenes(
    *,
    garment_types: tuple[str, ...] = ("shirt", "coat"),
    colors: tuple[str, ...] = ("red", "blue"),
    styles: tuple[str, ...] = ("formal", "athletic"),
    split: str = "train",
    include_spurious_text: bool = True,
) -> tuple[SyntheticClothingScene, ...]:
    """Generate controlled clothing scenes with garment/color/style factors."""

    if len(garment_types) < 2:
        raise ValueError("at least two garment types are required.")
    if len(colors) < 2:
        raise ValueError("at least two colors are required.")
    if len(styles) < 2:
        raise ValueError("at least two styles are required.")

    scenes = []
    for garment_index, garment_type in enumerate(garment_types):
        for color_index, color in enumerate(colors):
            for style_index, style in enumerate(styles):
                counterfactual_color = colors[(color_index + 1) % len(colors)]
                x1 = 0.08 + 0.08 * garment_index
                y1 = 0.12 + 0.07 * style_index
                x2 = min(x1 + 0.30, 0.95)
                y2 = min(y1 + 0.38, 0.95)
                scenes.append(
                    SyntheticClothingScene(
                        image_id=f"{split}_{style}_{color}_{garment_type}",
                        garment_type=garment_type,
                        color=color,
                        style=style,
                        bbox=(round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)),
                        question=f"What color is the {style} {garment_type}?",
                        answer=color,
                        counterfactual_answer=counterfactual_color,
                        spurious_text=counterfactual_color if include_spurious_text else None,
                        split=split,
                    )
                )
    return tuple(scenes)


def _nearest_centroid_predictions(
    train_points: t.Tensor,
    train_labels: t.Tensor,
    heldout_points: t.Tensor,
) -> t.Tensor:
    if train_points.ndim != 2 or heldout_points.ndim != 2:
        raise ValueError("points must have shape (examples, dimensions).")
    if train_points.shape[-1] != heldout_points.shape[-1]:
        raise ValueError("train and heldout point dimensions must match.")
    labels = train_labels.flatten().long()
    if labels.numel() != train_points.shape[0]:
        raise ValueError("train_labels must have one value per train point.")

    unique_labels = labels.unique(sorted=True)
    centroids = t.stack(
        [train_points[labels.eq(label)].float().mean(dim=0) for label in unique_labels]
    )
    distances = t.cdist(heldout_points.float(), centroids)
    return unique_labels[distances.argmin(dim=-1)]


def _accuracy(predictions: t.Tensor, labels: t.Tensor) -> float:
    flattened = labels.flatten().long()
    if predictions.shape != flattened.shape:
        raise ValueError("labels must have one value per prediction.")
    return predictions.eq(flattened).float().mean().item()


def clothing_geometry_report(
    train_embeddings: t.Tensor,
    heldout_embeddings: t.Tensor,
    train_garment_labels: t.Tensor,
    heldout_garment_labels: t.Tensor,
    train_color_labels: t.Tensor,
    heldout_color_labels: t.Tensor,
    train_style_labels: t.Tensor,
    heldout_style_labels: t.Tensor,
    text_prior_color_labels: t.Tensor,
    random_color_labels: t.Tensor,
    *,
    min_factor_accuracy: float = 0.9,
    max_text_prior_agreement: float = 0.5,
    max_random_agreement: float = 0.5,
) -> ClothingGeometryReport:
    """Check whether clothing geometry predicts image factors over text priors."""

    heldout_color_flat = heldout_color_labels.flatten().long()
    text_prior_flat = text_prior_color_labels.flatten().long().to(heldout_color_flat.device)
    random_flat = random_color_labels.flatten().long().to(heldout_color_flat.device)
    if text_prior_flat.shape != heldout_color_flat.shape:
        raise ValueError("text_prior_color_labels must have one value per held-out point.")
    if random_flat.shape != heldout_color_flat.shape:
        raise ValueError("random_color_labels must have one value per held-out point.")
    if t.equal(random_flat, text_prior_flat):
        raise ValueError("random_color_labels must be distinct from text_prior_color_labels.")
    if t.equal(random_flat, heldout_color_flat):
        raise ValueError("random_color_labels must be distinct from heldout_color_labels.")

    garment_predictions = _nearest_centroid_predictions(
        train_embeddings,
        train_garment_labels,
        heldout_embeddings,
    )
    color_predictions = _nearest_centroid_predictions(
        train_embeddings,
        train_color_labels,
        heldout_embeddings,
    )
    style_predictions = _nearest_centroid_predictions(
        train_embeddings,
        train_style_labels,
        heldout_embeddings,
    )
    garment_accuracy = _accuracy(garment_predictions, heldout_garment_labels)
    color_accuracy = _accuracy(color_predictions, heldout_color_labels)
    style_accuracy = _accuracy(style_predictions, heldout_style_labels)
    text_prior_agreement = _accuracy(color_predictions, text_prior_color_labels)
    random_agreement = _accuracy(color_predictions, random_color_labels)
    predicts = (
        garment_accuracy >= min_factor_accuracy
        and color_accuracy >= min_factor_accuracy
        and style_accuracy >= min_factor_accuracy
    )
    rejects_text_prior = text_prior_agreement <= max_text_prior_agreement
    rejects_random = random_agreement <= max_random_agreement
    return ClothingGeometryReport(
        garment_accuracy=garment_accuracy,
        color_accuracy=color_accuracy,
        style_accuracy=style_accuracy,
        text_prior_color_agreement=text_prior_agreement,
        random_color_agreement=random_agreement,
        predicts_clothing_factors=predicts,
        rejects_text_prior=rejects_text_prior,
        rejects_random_labels=rejects_random,
    )


def controlled_vlm_baseline_report(
    joint_logits: t.Tensor,
    image_only_logits: t.Tensor,
    text_only_logits: t.Tensor,
    labels: t.Tensor,
    *,
    min_joint_accuracy: float = 0.9,
    min_image_only_accuracy: float = 0.9,
    max_text_only_accuracy: float = 0.5,
) -> ControlledVLMBaselineReport:
    """Check synthetic VLM baselines before trusting real-model patching."""

    if joint_logits.shape != image_only_logits.shape or joint_logits.shape != text_only_logits.shape:
        raise ValueError("all logit tensors must have matching shape.")
    if joint_logits.ndim != 2:
        raise ValueError("logits must have shape (examples, candidates).")
    flattened_labels = labels.flatten().long()
    if flattened_labels.numel() != joint_logits.shape[0]:
        raise ValueError("labels must have one value per example.")

    joint_accuracy = joint_logits.argmax(dim=-1).eq(flattened_labels).float().mean().item()
    image_only_accuracy = (
        image_only_logits.argmax(dim=-1).eq(flattened_labels).float().mean().item()
    )
    text_only_accuracy = text_only_logits.argmax(dim=-1).eq(flattened_labels).float().mean().item()
    text_only_fails = text_only_accuracy <= max_text_only_accuracy
    return ControlledVLMBaselineReport(
        joint_accuracy=joint_accuracy,
        image_only_accuracy=image_only_accuracy,
        text_only_accuracy=text_only_accuracy,
        joint_beats_text_only=(
            joint_accuracy >= min_joint_accuracy
            and image_only_accuracy >= min_image_only_accuracy
            and joint_accuracy > text_only_accuracy
        ),
        text_only_fails_image_questions=text_only_fails,
    )


def visual_region_patch_report(
    clean_token_contributions: t.Tensor,
    corrupt_token_contributions: t.Tensor,
    *,
    object_token_indices: tuple[int, ...] | list[int],
    background_token_indices: tuple[int, ...] | list[int],
    target_index: int,
    counterfactual_index: int,
    random_token_indices: tuple[int, ...] | list[int] | None = None,
    min_object_gap: float = 0.5,
) -> VisualRegionPatchReport:
    """Patch visual-token contribution rows and compare object vs background effects."""

    if clean_token_contributions.shape != corrupt_token_contributions.shape:
        raise ValueError("clean and corrupt token contributions must match.")
    if clean_token_contributions.ndim != 2:
        raise ValueError("token contributions must have shape (tokens, candidates).")
    num_tokens, num_candidates = clean_token_contributions.shape
    if not 0 <= target_index < num_candidates:
        raise ValueError("target_index is out of range.")
    if not 0 <= counterfactual_index < num_candidates:
        raise ValueError("counterfactual_index is out of range.")

    def normalize_indices(indices: tuple[int, ...] | list[int]) -> t.Tensor:
        if len(indices) == 0:
            raise ValueError("patch index set must be nonempty.")
        index_tensor = t.tensor(indices, dtype=t.long, device=clean_token_contributions.device)
        if index_tensor.unique().numel() != index_tensor.numel():
            raise ValueError("patch token indices must be unique.")
        if index_tensor.min().item() < 0 or index_tensor.max().item() >= num_tokens:
            raise ValueError("patch token index is out of range.")
        return index_tensor

    object_indices = normalize_indices(object_token_indices)
    background_indices = normalize_indices(background_token_indices)
    if random_token_indices is None:
        object_index_set = set(object_indices.tolist())
        available_indices = [
            index for index in range(num_tokens) if index not in object_index_set
        ]
        if len(available_indices) < object_indices.numel():
            raise ValueError(
                "not enough non-object tokens for a same-size random-token control."
            )
        generator = t.Generator(device="cpu").manual_seed(0)
        permutation = t.randperm(len(available_indices), generator=generator).tolist()
        random_token_indices = tuple(
            available_indices[index]
            for index in permutation[: object_indices.numel()]
        )
    random_indices = normalize_indices(random_token_indices)
    if random_indices.numel() != object_indices.numel():
        raise ValueError(
            "random_token_indices must contain the same number of tokens as object_token_indices."
        )
    if set(random_indices.tolist()) & set(object_indices.tolist()):
        raise ValueError("random_token_indices must not overlap object_token_indices.")

    def margin_for(contributions: t.Tensor) -> float:
        logits = contributions.float().sum(dim=0)
        return (logits[target_index] - logits[counterfactual_index]).item()

    def patched_margin(indices: t.Tensor) -> float:
        patched = clean_token_contributions.clone()
        patched[indices] = corrupt_token_contributions[indices]
        return margin_for(patched)

    clean_margin = margin_for(clean_token_contributions)
    object_margin = patched_margin(object_indices)
    background_margin = patched_margin(background_indices)
    random_margin = patched_margin(random_indices)
    object_effect = clean_margin - object_margin
    background_effect = clean_margin - background_margin
    random_effect = clean_margin - random_margin
    return VisualRegionPatchReport(
        clean_margin=clean_margin,
        object_patch_margin=object_margin,
        background_patch_margin=background_margin,
        random_patch_margin=random_margin,
        object_patch_effect=object_effect,
        background_patch_effect=background_effect,
        random_patch_effect=random_effect,
        object_beats_background=object_effect - background_effect >= min_object_gap,
        object_beats_random=object_effect - random_effect >= min_object_gap,
        flips_answer=object_margin < 0,
    )


def visual_sequence_patch_report(
    clean_logits: t.Tensor,
    corrupt_logits: t.Tensor,
    object_patch_logits: t.Tensor,
    background_patch_logits: t.Tensor,
    random_patch_logits: t.Tensor,
    full_sequence_patch_logits: t.Tensor,
    *,
    target_indices: t.Tensor,
    counterfactual_indices: t.Tensor,
    min_object_gap: float = 1.0,
    max_full_sequence_margin_error: float = 1e-3,
) -> VisualSequencePatchReport:
    """Evaluate hidden visual-token patching against background and random controls."""

    logit_tensors = (
        clean_logits,
        corrupt_logits,
        object_patch_logits,
        background_patch_logits,
        random_patch_logits,
        full_sequence_patch_logits,
    )
    if len({tuple(logits.shape) for logits in logit_tensors}) != 1:
        raise ValueError("all logit tensors must have the same shape.")
    if clean_logits.ndim != 2:
        raise ValueError("logits must have shape (examples, candidates).")

    targets = target_indices.flatten().long().to(clean_logits.device)
    counterfactuals = counterfactual_indices.flatten().long().to(clean_logits.device)
    if targets.shape != counterfactuals.shape:
        raise ValueError("target_indices and counterfactual_indices must have the same shape.")
    if targets.numel() != clean_logits.shape[0]:
        raise ValueError("there must be one target index per example.")
    if targets.numel() == 0:
        raise ValueError("at least one example is required.")
    if targets.min().item() < 0 or targets.max().item() >= clean_logits.shape[1]:
        raise ValueError("target index is out of range.")
    if (
        counterfactuals.min().item() < 0
        or counterfactuals.max().item() >= clean_logits.shape[1]
    ):
        raise ValueError("counterfactual index is out of range.")

    row_indices = t.arange(clean_logits.shape[0], device=clean_logits.device)

    def margins(logits: t.Tensor) -> list[float]:
        values = (
            logits.float()[row_indices, targets]
            - logits.float()[row_indices, counterfactuals]
        )
        return [float(value) for value in values.detach().cpu()]

    clean_margins = margins(clean_logits)
    corrupt_margins = margins(corrupt_logits)
    object_margins = margins(object_patch_logits)
    background_margins = margins(background_patch_logits)
    random_margins = margins(random_patch_logits)
    full_sequence_margins = margins(full_sequence_patch_logits)

    def effects(patched_margins: list[float]) -> list[float]:
        return [
            clean_margin - patched_margin
            for clean_margin, patched_margin in zip(clean_margins, patched_margins)
        ]

    object_effects = effects(object_margins)
    background_effects = effects(background_margins)
    random_effects = effects(random_margins)
    full_sequence_effects = effects(full_sequence_margins)
    object_gap_over_background = [
        object_effect - background_effect
        for object_effect, background_effect in zip(object_effects, background_effects)
    ]
    object_gap_over_random = [
        object_effect - random_effect
        for object_effect, random_effect in zip(object_effects, random_effects)
    ]
    full_sequence_errors = [
        abs(full_margin - corrupt_margin)
        for full_margin, corrupt_margin in zip(full_sequence_margins, corrupt_margins)
    ]
    max_full_sequence_error = max(full_sequence_errors)

    object_flips = all(margin < 0 for margin in object_margins)
    background_preserves = all(margin > 0 for margin in background_margins)
    random_preserves = all(margin > 0 for margin in random_margins)
    full_sequence_flips = all(margin < 0 for margin in full_sequence_margins)
    full_sequence_matches_corrupt = (
        full_sequence_flips and max_full_sequence_error <= max_full_sequence_margin_error
    )
    object_beats_background = min(object_gap_over_background) >= min_object_gap
    object_beats_random = min(object_gap_over_random) >= min_object_gap
    passes = (
        object_flips
        and background_preserves
        and random_preserves
        and full_sequence_matches_corrupt
        and object_beats_background
        and object_beats_random
    )

    return VisualSequencePatchReport(
        clean_margins=clean_margins,
        corrupt_margins=corrupt_margins,
        object_patch_margins=object_margins,
        background_patch_margins=background_margins,
        random_patch_margins=random_margins,
        full_sequence_patch_margins=full_sequence_margins,
        object_patch_effects=object_effects,
        background_patch_effects=background_effects,
        random_patch_effects=random_effects,
        full_sequence_patch_effects=full_sequence_effects,
        min_object_gap_over_background=min(object_gap_over_background),
        min_object_gap_over_random=min(object_gap_over_random),
        full_sequence_patch_max_abs_margin_error=max_full_sequence_error,
        object_patch_flips_answer=object_flips,
        background_patch_preserves_answer=background_preserves,
        random_patch_preserves_answer=random_preserves,
        full_sequence_patch_flips_answer=full_sequence_flips,
        full_sequence_patch_matches_corrupt=full_sequence_matches_corrupt,
        object_beats_background=object_beats_background,
        object_beats_random=object_beats_random,
        passes_activation_patching_controls=passes,
    )


def clip_contrastive_logits(
    image_embeddings: t.Tensor,
    text_embeddings: t.Tensor,
    *,
    logit_scale: float = 10.0,
) -> t.Tensor:
    """Return CLIP-style image-text cosine logits."""

    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (batch, d_model).")
    if image_embeddings.shape[-1] != text_embeddings.shape[-1]:
        raise ValueError("image and text embedding dimensions must match.")
    if logit_scale <= 0:
        raise ValueError("logit_scale must be positive.")

    image_normalized = _l2_normalize(image_embeddings)
    text_normalized = _l2_normalize(text_embeddings)
    return logit_scale * image_normalized @ text_normalized.T


def siglip_pairwise_loss(logits: t.Tensor, labels: t.Tensor) -> t.Tensor:
    """Return the pairwise logistic loss used by SigLIP-style objectives."""

    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same shape.")
    signed_labels = t.where(labels.float() > 0, 1.0, -1.0)
    return F.softplus(-signed_labels * logits.float()).mean()


def contrastive_alignment_report(
    logits: t.Tensor,
    *,
    min_accuracy: float = 1.0,
    min_positive_margin: float = 1.0,
) -> ContrastiveAlignmentReport:
    """Check paired image-text retrieval accuracy and positive-pair margin."""

    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must be a square (batch, batch) matrix.")
    batch = logits.shape[0]
    if batch == 0:
        raise ValueError("logits must be nonempty.")

    targets = t.arange(batch, device=logits.device)
    image_predictions = logits.argmax(dim=-1)
    text_predictions = logits.argmax(dim=0)
    image_accuracy = image_predictions.eq(targets).float().mean().item()
    text_accuracy = text_predictions.eq(targets).float().mean().item()

    positives = logits.diag()
    if batch == 1:
        mean_margin = float("inf")
    else:
        negative_mask = t.eye(batch, dtype=t.bool, device=logits.device)
        negatives = logits.masked_fill(negative_mask, -float("inf"))
        image_margins = positives - negatives.max(dim=-1).values
        text_margins = positives - negatives.max(dim=0).values
        mean_margin = ((image_margins + text_margins) / 2).mean().item()

    aligned = (
        image_accuracy >= min_accuracy
        and text_accuracy >= min_accuracy
        and mean_margin >= min_positive_margin
    )
    return ContrastiveAlignmentReport(
        image_to_text_accuracy=image_accuracy,
        text_to_image_accuracy=text_accuracy,
        mean_positive_margin=mean_margin,
        aligned=aligned,
    )


def visual_token_attribution_report(
    token_activations: t.Tensor,
    text_direction: t.Tensor,
    *,
    top_k: int = 2,
    min_top_token_mass: float = 0.6,
) -> VisualTokenAttributionReport:
    """Score visual tokens against a text direction and check locality."""

    if token_activations.ndim != 2:
        raise ValueError("token_activations must have shape (tokens, d_model).")
    if text_direction.ndim != 1:
        raise ValueError("text_direction must have shape (d_model,).")
    if token_activations.shape[-1] != text_direction.shape[0]:
        raise ValueError("token and direction dimensions must match.")
    if top_k <= 0 or top_k > token_activations.shape[0]:
        raise ValueError("top_k must be in [1, num_tokens].")

    direction = _l2_normalize(text_direction.unsqueeze(0)).squeeze(0)
    token_scores = token_activations.float() @ direction
    _, top_indices = token_scores.topk(top_k)
    positive_scores = token_scores.clamp_min(0)
    total_positive_mass = positive_scores.sum()
    if total_positive_mass.item() == 0:
        top_mass = 0.0
    else:
        top_mass = (positive_scores[top_indices].sum() / total_positive_mass).item()

    return VisualTokenAttributionReport(
        token_scores=token_scores,
        top_token_indices=top_indices,
        top_token_mass=top_mass,
        localized=top_mass >= min_top_token_mass,
    )


def object_hallucination_report(
    *,
    object_score: float,
    visual_evidence_score: float,
    text_prior_score: float,
    min_object_score: float = 0.7,
    min_visual_evidence: float = 0.5,
    min_text_prior_gap: float = 0.2,
) -> ObjectHallucinationReport:
    """Flag object claims that appear text-prior-driven without visual evidence."""

    text_prior_gap = text_prior_score - visual_evidence_score
    flags_hallucination = (
        object_score >= min_object_score
        and visual_evidence_score < min_visual_evidence
        and text_prior_gap >= min_text_prior_gap
    )
    return ObjectHallucinationReport(
        object_score=object_score,
        visual_evidence_score=visual_evidence_score,
        text_prior_score=text_prior_score,
        text_prior_gap=text_prior_gap,
        flags_hallucination=flags_hallucination,
    )


def modality_arbitration_report(
    candidate_scores: t.Tensor,
    *,
    visual_index: int,
    text_prior_index: int,
    min_visual_margin: float = 0.1,
) -> ModalityArbitrationReport:
    """Check whether a conflicting VLM answer follows visual evidence over text prior."""

    if candidate_scores.ndim != 1:
        raise ValueError("candidate_scores must have shape (num_candidates,).")
    num_candidates = candidate_scores.shape[0]
    if not 0 <= visual_index < num_candidates:
        raise ValueError("visual_index is out of range.")
    if not 0 <= text_prior_index < num_candidates:
        raise ValueError("text_prior_index is out of range.")

    scores = candidate_scores.float()
    visual_choice_score = scores[visual_index].item()
    text_prior_choice_score = scores[text_prior_index].item()
    visual_margin = visual_choice_score - text_prior_choice_score
    return ModalityArbitrationReport(
        visual_choice_score=visual_choice_score,
        text_prior_choice_score=text_prior_choice_score,
        visual_margin=visual_margin,
        trusts_visual_evidence=visual_margin >= min_visual_margin,
    )
