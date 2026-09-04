# %%
"""Reference solutions for [12.1] CLIP, SigLIP, and VLM Controls."""

import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import torch as t

chapter = "chapter12_vlm_interpretability"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.vlm_interpretability import (
    build_toy_clip_batch,
    bbox_to_patch_indices,
    clip_contrastive_logits,
    clothing_geometry_report,
    contrastive_alignment_report,
    controlled_vlm_baseline_report,
    deterministic_derangement,
    generate_synthetic_clothing_scenes,
    generate_synthetic_colored_shape_scenes,
    modality_arbitration_report,
    object_hallucination_report,
    patch_visual_token_activations,
    retrieval_accuracy,
    retrieval_table,
    same_size_non_overlapping_token_control,
    siglip_pairwise_loss,
    toy_caption_features,
    train_toy_clip,
    visual_region_patch_report,
    visual_sequence_patch_report,
    visual_token_attribution_report,
)

MAIN = __name__ == "__main__"

REAL_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
REAL_CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

REAL_CLIP_TEXTS = (
    "a simple red square on a white background",
    "a simple blue circle on a white background",
)

REAL_SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
REAL_SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"

REAL_QWEN25_VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REAL_QWEN25_VL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
REAL_QWEN25_VL_ITEMS = (
    ("red", "square"),
    ("blue", "circle"),
)

SHAPE_OBJECT_BBOX = (76, 76, 148, 148)
SHAPE_DRAW_BBOX = SHAPE_OBJECT_BBOX
SHAPE_BACKGROUND_BBOX = (0, 0, 40, 40)
SHAPE_RANDOM_CONTROL_SEED = 0


def _bbox_overlap_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _same_bbox_size(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return (
        first[2] - first[0] == second[2] - second[0]
        and first[3] - first[1] == second[3] - second[1]
    )


def _same_size_random_control_bbox(
    object_bbox: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int] = (224, 224),
    seed: int = 0,
) -> tuple[int, int, int, int]:
    object_width = object_bbox[2] - object_bbox[0]
    object_height = object_bbox[3] - object_bbox[1]
    image_width, image_height = image_size
    if object_width > image_width or object_height > image_height:
        raise ValueError("object_bbox must fit inside the image.")
    candidate_bboxes = (
        (0, 0, object_width, object_height),
        (image_width - object_width, 0, image_width, object_height),
        (0, image_height - object_height, object_width, image_height),
        (
            image_width - object_width,
            image_height - object_height,
            image_width,
            image_height,
        ),
    )
    valid_bboxes = [
        bbox
        for bbox in candidate_bboxes
        if bbox != object_bbox and _bbox_overlap_area(bbox, object_bbox) == 0
    ]
    if not valid_bboxes:
        raise ValueError(
            "Could not construct a same-size non-overlapping random control bbox. "
            "Use a smaller object patch or a larger image."
        )
    generator = t.Generator(device="cpu").manual_seed(seed)
    candidate_index = int(t.randperm(len(valid_bboxes), generator=generator)[0].item())
    return valid_bboxes[candidate_index]


SHAPE_RANDOM_CONTROL_BBOX = _same_size_random_control_bbox(
    SHAPE_OBJECT_BBOX,
    seed=SHAPE_RANDOM_CONTROL_SEED,
)


def extract_contrastive_embeddings(
    model,
    batch: Mapping[str, t.Tensor],
) -> tuple[t.Tensor, t.Tensor]:
    """Extract normalized image and text embeddings from a CLIP-like model."""

    if "pixel_values" not in batch or "input_ids" not in batch:
        raise KeyError("batch must contain pixel_values and input_ids.")
    image_output = model.get_image_features(pixel_values=batch["pixel_values"])
    text_kwargs = {"input_ids": batch["input_ids"]}
    if "attention_mask" in batch:
        text_kwargs["attention_mask"] = batch["attention_mask"]
    text_output = model.get_text_features(**text_kwargs)
    image_embeddings = (
        image_output if isinstance(image_output, t.Tensor) else image_output.pooler_output
    )
    text_embeddings = (
        text_output if isinstance(text_output, t.Tensor) else text_output.pooler_output
    )
    image_embeddings = image_embeddings.float()
    text_embeddings = text_embeddings.float()
    return (
        image_embeddings / image_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8),
        text_embeddings / text_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-8),
    )


def bidirectional_retrieval_metrics(
    image_embeddings: t.Tensor,
    text_embeddings: t.Tensor,
    *,
    logit_scale: float = 1.0,
    logit_bias: float | t.Tensor = 0.0,
) -> dict[str, object]:
    """Return the full retrieval matrix, both accuracies, and the positive margin."""

    logits = clip_contrastive_logits(
        image_embeddings,
        text_embeddings,
        logit_scale=logit_scale,
    )
    logits = logits + t.as_tensor(logit_bias, device=logits.device, dtype=logits.dtype)
    report = contrastive_alignment_report(
        logits,
        min_accuracy=0.0,
        min_positive_margin=float("-inf"),
    )
    return {
        "logits": logits,
        "image_to_text_accuracy": report.image_to_text_accuracy,
        "text_to_image_accuracy": report.text_to_image_accuracy,
        "mean_positive_margin": report.mean_positive_margin,
    }


def capture_module_output(
    module: t.nn.Module,
    forward_fn: Callable[[], object],
) -> tuple[object, t.Tensor]:
    """Run ``forward_fn`` once and return its result plus one hooked activation."""

    captured: list[t.Tensor] = []

    def save_output(_module, _args, output):
        if not isinstance(output, t.Tensor):
            raise TypeError("the selected hook point must return a tensor.")
        captured.append(output.detach())

    handle = module.register_forward_hook(save_output)
    try:
        result = forward_fn()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one hooked activation, captured {len(captured)}.")
    return result, captured[0]


def patch_hidden_token_rows(
    clean_activations: t.Tensor,
    corrupt_activations: t.Tensor,
    token_indices: tuple[int, ...] | list[int],
) -> t.Tensor:
    """Replace selected clean sequence rows with counterfactual rows."""

    if clean_activations.shape != corrupt_activations.shape:
        raise ValueError("clean and corrupt activations must have the same shape.")
    if clean_activations.ndim != 3:
        raise ValueError("activations must have shape (batch, tokens, hidden).")
    index = t.tensor(tuple(int(i) for i in token_indices), device=clean_activations.device)
    if index.numel() == 0:
        raise ValueError("token_indices must be nonempty.")
    if index.min() < 0 or index.max() >= clean_activations.shape[1]:
        raise ValueError("token index out of range.")
    if index.unique().numel() != index.numel():
        raise ValueError("token_indices must be unique.")
    patched = clean_activations.clone()
    patched[:, index] = corrupt_activations[:, index].to(patched)
    return patched


def run_with_activation_patch(
    model,
    embedding_module: t.nn.Module,
    clean_inputs: Mapping[str, t.Tensor],
    corrupt_pixel_values: t.Tensor,
    token_indices: tuple[int, ...] | list[int],
) -> t.Tensor:
    """Patch hidden visual rows during a clean CLIP-like forward pass."""

    with t.inference_mode():
        corrupt_activations = embedding_module(corrupt_pixel_values)

    def patch_hook(_module, _args, clean_activations):
        return patch_hidden_token_rows(clean_activations, corrupt_activations, token_indices)

    handle = embedding_module.register_forward_hook(patch_hook)
    try:
        with t.inference_mode():
            return model(**dict(clean_inputs)).logits_per_image.detach().float().cpu()
    finally:
        handle.remove()


def causal_patch_metrics(
    clean_logits: t.Tensor,
    corrupt_logits: t.Tensor,
    patched_logits: Mapping[str, t.Tensor],
    *,
    target_indices: t.Tensor,
    counterfactual_indices: t.Tensor,
) -> dict[str, object]:
    """Score target-minus-counterfactual margins for patch and control conditions."""

    def margins(logits: t.Tensor) -> t.Tensor:
        rows = t.arange(logits.shape[0], device=logits.device)
        return logits[rows, target_indices.to(logits.device)] - logits[
            rows, counterfactual_indices.to(logits.device)
        ]

    clean = margins(clean_logits)
    corrupt = margins(corrupt_logits)
    rows = []
    for condition, logits in patched_logits.items():
        patched = margins(logits)
        rows.append(
            {
                "condition": condition,
                "mean_margin": float(patched.mean().item()),
                "mean_effect": float((clean - patched).mean().item()),
                "all_flip": bool((patched < 0).all().item()),
            }
        )
    return {
        "clean_margins": clean.tolist(),
        "corrupt_margins": corrupt.tolist(),
        "rows": rows,
    }


def trim_generated_tokens(
    input_ids: t.Tensor,
    generated_ids: t.Tensor,
) -> list[t.Tensor]:
    """Remove the padded prompt prefix from each generated token sequence."""

    if input_ids.ndim != 2 or generated_ids.ndim != 2:
        raise ValueError("input_ids and generated_ids must both be rank-2 tensors.")
    if input_ids.shape[0] != generated_ids.shape[0]:
        raise ValueError("input_ids and generated_ids must have the same batch size.")
    return [
        output[len(prompt) :]
        for prompt, output in zip(input_ids, generated_ids)
    ]


def decode_qwen_answers(
    processor,
    input_ids: t.Tensor,
    generated_ids: t.Tensor,
) -> list[str]:
    """Remove each prompt prefix and decode only Qwen's generated answer tokens."""

    trimmed = trim_generated_tokens(input_ids, generated_ids)
    return [
        answer.strip().lower()
        for answer in processor.batch_decode(trimmed, skip_special_tokens=True)
    ]


# %%
def contrastive_smoke_test() -> dict:
    image_embeddings = t.eye(3)
    text_embeddings = t.eye(3)
    logits = clip_contrastive_logits(
        image_embeddings,
        text_embeddings,
        logit_scale=5.0,
    )
    return contrastive_alignment_report(
        logits,
        min_accuracy=1.0,
        min_positive_margin=4.0,
    ).__dict__


def siglip_smoke_test() -> dict:
    logits = t.tensor([[4.0, -4.0], [-3.0, 3.0]])
    labels = t.eye(2)
    return {"loss": siglip_pairwise_loss(logits, labels).item()}


def token_attribution_smoke_test() -> dict:
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
    result = report.__dict__.copy()
    result["token_scores"] = report.token_scores.tolist()
    result["top_token_indices"] = report.top_token_indices.tolist()
    return result


def hallucination_smoke_test() -> dict:
    return object_hallucination_report(
        object_score=0.9,
        visual_evidence_score=0.2,
        text_prior_score=0.8,
        min_object_score=0.7,
        min_visual_evidence=0.5,
        min_text_prior_gap=0.4,
    ).__dict__


def arbitration_smoke_test() -> dict:
    candidate_scores = t.tensor([0.1, 0.8, 0.2])
    return modality_arbitration_report(
        candidate_scores,
        visual_index=1,
        text_prior_index=2,
        min_visual_margin=0.5,
    ).__dict__


def synthetic_scene_schema_smoke_test() -> dict:
    scenes = generate_synthetic_colored_shape_scenes(
        colors=("red", "blue"),
        shapes=("cube", "sphere"),
        split="train",
    )
    return {
        "num_scenes": len(scenes),
        "first_scene": scenes[0].__dict__,
        "has_spurious_text_control": all(scene.spurious_text is not None for scene in scenes),
        "has_counterfactual_answers": all(
            scene.answer != scene.counterfactual_answer for scene in scenes
        ),
    }


def toy_clip_signature_result(
    *,
    device: str | t.device = "cpu",
    steps: int = 250,
    seed: int = 0,
) -> dict:
    """Train a tiny CLIP on rendered colored shapes and return visible metrics."""

    batch = build_toy_clip_batch(
        colors=("red", "blue", "green", "yellow"),
        shapes=("square", "circle", "triangle"),
        image_size=48,
    )
    trained = train_toy_clip(
        batch.image_features,
        batch.text_features,
        steps=steps,
        seed=seed,
        device=device,
    )
    image_ids = tuple(scene.image_id for scene in batch.scenes)
    random_permutation = deterministic_derangement(len(batch.captions), seed=seed)
    random_logits = trained.retrieval_logits[:, random_permutation]
    random_report = contrastive_alignment_report(
        random_logits,
        min_accuracy=1.0,
        min_positive_margin=1.0,
    )

    conflict_captions = tuple(
        f"a {scene.counterfactual_answer} {scene.shape}" for scene in batch.scenes
    )
    conflict_features = toy_caption_features(
        conflict_captions,
        colors=batch.colors,
        shapes=batch.shapes,
    )
    conflict_embeddings = conflict_features @ trained.text_projection
    conflict_logits = clip_contrastive_logits(
        trained.image_embeddings,
        conflict_embeddings,
    )
    conflict_report = contrastive_alignment_report(
        conflict_logits,
        min_accuracy=1.0,
        min_positive_margin=1.0,
    )

    random_caption_list = [batch.captions[index] for index in random_permutation.tolist()]
    return {
        "scene_count": len(batch.scenes),
        "image_grid_shape": list(batch.image_tensors.shape),
        "captions": list(batch.captions),
        "image_ids": list(image_ids),
        "loss_start": trained.train_losses[0],
        "loss_end": trained.train_losses[-1],
        "loss_drop": trained.train_losses[0] - trained.train_losses[-1],
        "loss_curve": list(trained.train_losses),
        "retrieval_logits": trained.retrieval_logits.tolist(),
        "retrieval_rows": retrieval_table(
            trained.retrieval_logits,
            batch.captions,
            image_ids=image_ids,
        ),
        "image_to_text_accuracy": trained.report.image_to_text_accuracy,
        "text_to_image_accuracy": trained.report.text_to_image_accuracy,
        "mean_positive_margin": trained.report.mean_positive_margin,
        "aligned": trained.report.aligned,
        "random_caption_permutation": random_permutation.tolist(),
        "random_captions": random_caption_list,
        "random_caption_logits": random_logits.tolist(),
        "random_caption_accuracy": retrieval_accuracy(random_logits),
        "random_caption_aligned": random_report.aligned,
        "random_caption_rows": retrieval_table(
            random_logits,
            tuple(random_caption_list),
            image_ids=image_ids,
        ),
        "conflict_captions": list(conflict_captions),
        "conflict_caption_logits": conflict_logits.tolist(),
        "conflict_caption_accuracy": retrieval_accuracy(conflict_logits),
        "conflict_caption_aligned": conflict_report.aligned,
        "conflict_caption_rows": retrieval_table(
            conflict_logits,
            conflict_captions,
            image_ids=image_ids,
        ),
        "control_claim_passed": (
            trained.report.aligned
            and not random_report.aligned
            and not conflict_report.aligned
            and retrieval_accuracy(random_logits) <= 0.25
            and retrieval_accuracy(conflict_logits) <= 0.25
        ),
    }


def _deterministic_permuted_labels(
    labels: t.Tensor,
    *,
    forbidden: tuple[t.Tensor, ...] = (),
    seed: int = 0,
) -> t.Tensor:
    """Return a seeded permutation distinct from true labels and named controls."""

    flat_labels = labels.flatten().long()
    if flat_labels.numel() < 2:
        raise ValueError("at least two labels are required for a permutation control.")
    forbidden_flat = tuple(
        forbidden_labels.flatten().long().to(flat_labels.device)
        for forbidden_labels in forbidden
    )
    for offset in range(32):
        generator = t.Generator(device="cpu").manual_seed(seed + offset)
        permutation = t.randperm(flat_labels.numel(), generator=generator).to(
            flat_labels.device
        )
        permuted = flat_labels[permutation]
        if t.equal(permuted, flat_labels):
            continue
        if any(t.equal(permuted, forbidden_labels) for forbidden_labels in forbidden_flat):
            continue
        return permuted.reshape_as(labels)
    raise ValueError("could not build a distinct deterministic label permutation.")


def clothing_geometry_smoke_test(device: str | t.device = "cpu") -> dict:
    device = t.device(device)
    scenes = generate_synthetic_clothing_scenes(split="heldout")
    train_embeddings = t.tensor(
        [
            [3.0, 0.0, 2.0, 0.0, 1.5, 0.0],
            [3.0, 0.0, 0.0, 2.0, 0.0, 1.5],
            [0.0, 3.0, 2.0, 0.0, 0.0, 1.5],
            [0.0, 3.0, 0.0, 2.0, 1.5, 0.0],
        ],
        device=device,
    )
    heldout_embeddings = train_embeddings + 0.05
    garment_labels = t.tensor([0, 0, 1, 1], device=device)
    color_labels = t.tensor([0, 1, 0, 1], device=device)
    style_labels = t.tensor([0, 1, 1, 0], device=device)
    text_prior_color_labels = 1 - color_labels
    random_color_labels = _deterministic_permuted_labels(
        color_labels,
        forbidden=(text_prior_color_labels,),
        seed=0,
    )
    report = clothing_geometry_report(
        train_embeddings,
        heldout_embeddings,
        garment_labels,
        garment_labels,
        color_labels,
        color_labels,
        style_labels,
        style_labels,
        text_prior_color_labels,
        random_color_labels,
    )
    result = report.__dict__.copy()
    result["scene_count"] = len(scenes)
    result["first_scene"] = scenes[0].__dict__
    result["has_spurious_text_control"] = all(scene.spurious_text for scene in scenes)
    result["text_prior_color_labels"] = text_prior_color_labels.detach().cpu().tolist()
    result["random_color_labels"] = random_color_labels.detach().cpu().tolist()
    result["random_labels_distinct_from_text_prior"] = not t.equal(
        random_color_labels,
        text_prior_color_labels,
    )
    result["random_labels_distinct_from_true_labels"] = not t.equal(
        random_color_labels,
        color_labels,
    )
    result["random_label_seed"] = 0
    return result


def controlled_baselines_smoke_test() -> dict:
    labels = t.tensor([0, 1, 0, 1])
    joint_logits = t.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]])
    image_only_logits = t.tensor([[2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5]])
    text_only_logits = t.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    return controlled_vlm_baseline_report(
        joint_logits,
        image_only_logits,
        text_only_logits,
        labels,
        min_joint_accuracy=1.0,
        min_image_only_accuracy=1.0,
        max_text_only_accuracy=0.5,
    ).__dict__


def visual_region_patch_smoke_test(device: str | t.device = "cpu") -> dict:
    device = t.device(device)
    clean_contributions = t.tensor(
        [
            [2.0, -1.0],
            [1.0, -0.5],
            [0.1, 0.0],
            [0.0, 0.1],
        ],
        device=device,
    )
    corrupt_contributions = t.tensor(
        [
            [-1.0, 2.0],
            [-0.5, 1.0],
            [0.1, 0.0],
            [0.0, 0.1],
        ],
        device=device,
    )
    object_token_indices = [0, 1]
    background_token_indices = [2]
    random_token_indices = [2, 3]
    result = visual_region_patch_report(
        clean_contributions,
        corrupt_contributions,
        object_token_indices=object_token_indices,
        background_token_indices=background_token_indices,
        random_token_indices=random_token_indices,
        target_index=0,
        counterfactual_index=1,
        min_object_gap=1.0,
    ).__dict__
    result["object_token_count"] = len(object_token_indices)
    result["background_token_count"] = len(background_token_indices)
    result["random_token_count"] = len(random_token_indices)
    result["random_control_same_size"] = len(random_token_indices) == len(
        object_token_indices
    )
    return result


def visual_sequence_patch_smoke_test(device: str | t.device = "cpu") -> dict:
    device = t.device(device)
    clean_logits = t.tensor([[4.0, 0.0], [0.0, 4.0]], device=device)
    corrupt_logits = t.tensor([[0.0, 4.0], [4.0, 0.0]], device=device)
    background_logits = t.tensor([[3.8, 0.0], [0.0, 3.8]], device=device)
    random_logits = t.tensor([[3.7, 0.0], [0.0, 3.7]], device=device)
    report = visual_sequence_patch_report(
        clean_logits,
        corrupt_logits,
        corrupt_logits,
        background_logits,
        random_logits,
        corrupt_logits,
        target_indices=t.tensor([0, 1], device=device),
        counterfactual_indices=t.tensor([1, 0], device=device),
        min_object_gap=1.0,
    )
    return report.__dict__.copy()


def _render_shape_image(color: str, shape: str):
    """Render a deterministic 224x224 colored-shape image for CLIP preflights."""

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    bbox = list(SHAPE_DRAW_BBOX)
    if shape == "square":
        draw.rectangle(bbox, fill=color)
    elif shape == "circle":
        draw.ellipse(bbox, fill=color)
    else:
        raise ValueError("shape must be 'square' or 'circle'.")
    return image


def _patch_image_region(clean_image, corrupt_image, bbox: tuple[int, int, int, int]):
    patched = clean_image.copy()
    patched.paste(corrupt_image.crop(bbox), bbox)
    return patched


def _contrastive_region_patch_report(logits: t.Tensor, *, min_object_gap: float) -> dict:
    """Score real contrastive-model margins under object and background patches."""

    if logits.shape != (8, 2):
        raise ValueError(
            "expected logits for 2 clean, 2 object-patched, 2 background, and 2 random images."
        )

    def margins(start: int) -> list[float]:
        return [
            float((logits[start + index, index] - logits[start + index, 1 - index]).item())
            for index in range(2)
        ]

    clean_margins = margins(0)
    object_patch_margins = margins(2)
    background_patch_margins = margins(4)
    random_patch_margins = margins(6)
    object_effects = [
        clean_margin - object_margin
        for clean_margin, object_margin in zip(clean_margins, object_patch_margins)
    ]
    background_effects = [
        clean_margin - background_margin
        for clean_margin, background_margin in zip(clean_margins, background_patch_margins)
    ]
    random_effects = [
        clean_margin - random_margin
        for clean_margin, random_margin in zip(clean_margins, random_patch_margins)
    ]
    object_gap_over_background = [
        object_effect - background_effect
        for object_effect, background_effect in zip(object_effects, background_effects)
    ]
    object_gap_over_random = [
        object_effect - random_effect
        for object_effect, random_effect in zip(object_effects, random_effects)
    ]
    return {
        "clean_margins": clean_margins,
        "object_patch_margins": object_patch_margins,
        "background_patch_margins": background_patch_margins,
        "random_patch_margins": random_patch_margins,
        "object_patch_effects": object_effects,
        "background_patch_effects": background_effects,
        "random_patch_effects": random_effects,
        "min_object_gap_over_background": min(object_gap_over_background),
        "min_object_gap_over_random": min(object_gap_over_random),
        "object_patch_flips_answer": all(margin < 0 for margin in object_patch_margins),
        "background_patch_preserves_answer": all(
            margin > 0 for margin in background_patch_margins
        ),
        "random_patch_preserves_answer": all(
            margin > 0 for margin in random_patch_margins
        ),
        "object_beats_background": min(object_gap_over_background) >= min_object_gap,
        "object_beats_random": min(object_gap_over_random) >= min_object_gap,
    }


def _run_logits_with_visual_token_patch(
    model,
    inputs,
    corrupt_pixel_values: t.Tensor,
    embedding_module,
    token_indices: tuple[int, ...],
) -> t.Tensor:
    with t.inference_mode():
        corrupt_activations = embedding_module(corrupt_pixel_values)

    def patch_hook(_module, _args, output):
        return patch_visual_token_activations(output, corrupt_activations, token_indices)

    hook = embedding_module.register_forward_hook(patch_hook)
    try:
        with t.inference_mode():
            return model(**inputs).logits_per_image.detach().float().cpu()
    finally:
        hook.remove()


def _real_contrastive_visual_token_activation_patch_report(
    *,
    model,
    inputs,
    corrupt_inputs,
    embedding_module,
    model_id: str,
    revision: str,
    local_snapshot: str,
    patch_size: int,
    has_cls_token: bool,
    min_object_gap: float,
    max_full_sequence_margin_error: float = 1e-3,
) -> dict:
    """Patch hidden visual-token activations in a real contrastive VLM."""

    with t.inference_mode():
        clean_logits = model(**inputs).logits_per_image.detach().float().cpu()
        corrupt_logits = model(
            **{**inputs, "pixel_values": corrupt_inputs.pixel_values}
        ).logits_per_image.detach().float().cpu()
        sample_activations = embedding_module(inputs.pixel_values)

    num_tokens = int(sample_activations.shape[1])
    image_size = (224, 224)
    object_indices = bbox_to_patch_indices(
        SHAPE_OBJECT_BBOX,
        image_size=image_size,
        patch_size=patch_size,
        has_cls_token=has_cls_token,
    )
    background_indices = bbox_to_patch_indices(
        SHAPE_BACKGROUND_BBOX,
        image_size=image_size,
        patch_size=patch_size,
        has_cls_token=has_cls_token,
    )
    protected_indices = tuple([0] if has_cls_token else []) + background_indices
    random_indices = same_size_non_overlapping_token_control(
        object_indices,
        num_tokens=num_tokens,
        protected_indices=protected_indices,
        seed=SHAPE_RANDOM_CONTROL_SEED,
    )
    full_sequence_indices = tuple(range(1 if has_cls_token else 0, num_tokens))

    object_patch_logits = _run_logits_with_visual_token_patch(
        model,
        inputs,
        corrupt_inputs.pixel_values,
        embedding_module,
        object_indices,
    )
    background_patch_logits = _run_logits_with_visual_token_patch(
        model,
        inputs,
        corrupt_inputs.pixel_values,
        embedding_module,
        background_indices,
    )
    random_patch_logits = _run_logits_with_visual_token_patch(
        model,
        inputs,
        corrupt_inputs.pixel_values,
        embedding_module,
        random_indices,
    )
    full_sequence_patch_logits = _run_logits_with_visual_token_patch(
        model,
        inputs,
        corrupt_inputs.pixel_values,
        embedding_module,
        full_sequence_indices,
    )

    report = visual_sequence_patch_report(
        clean_logits,
        corrupt_logits,
        object_patch_logits,
        background_patch_logits,
        random_patch_logits,
        full_sequence_patch_logits,
        target_indices=t.tensor([0, 1]),
        counterfactual_indices=t.tensor([1, 0]),
        min_object_gap=min_object_gap,
        max_full_sequence_margin_error=max_full_sequence_margin_error,
    )
    result = report.__dict__.copy()
    result.update(
        {
            "cuda_available": True,
            "model_id": model_id,
            "revision": revision,
            "local_snapshot": str(local_snapshot),
            "claim_scope": (
                "pinned_real_contrastive_vlm_visual_token_activation_patching"
            ),
            "hook_point": "vision_model.embeddings",
            "patch_size": patch_size,
            "has_cls_token": has_cls_token,
            "num_visual_tokens": num_tokens,
            "object_token_indices": list(object_indices),
            "background_token_indices": list(background_indices),
            "random_token_indices": list(random_indices),
            "full_sequence_token_count": len(full_sequence_indices),
            "object_token_count": len(object_indices),
            "background_token_count": len(background_indices),
            "random_token_count": len(random_indices),
            "random_control_same_size": len(random_indices) == len(object_indices),
            "random_control_overlaps_object": bool(
                set(random_indices) & set(object_indices)
            ),
            "clean_logits": clean_logits.tolist(),
            "corrupt_logits": corrupt_logits.tolist(),
            "object_patch_logits": object_patch_logits.tolist(),
            "background_patch_logits": background_patch_logits.tolist(),
            "random_patch_logits": random_patch_logits.tolist(),
            "full_sequence_patch_logits": full_sequence_patch_logits.tolist(),
            "preflight_passed": report.passes_activation_patching_controls,
        }
    )
    return result


def real_clip_visual_token_activation_patching_preflight(
    max_vram_gb: float = 24.0,
) -> dict:
    """Run hidden visual-token activation patching on a pinned CLIP checkpoint."""

    if not t.cuda.is_available():
        raise RuntimeError("12.1 real CLIP activation patching preflight requires CUDA.")

    from huggingface_hub import snapshot_download
    from transformers import CLIPModel, CLIPProcessor

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    local_snapshot = snapshot_download(
        REAL_CLIP_MODEL_ID,
        revision=REAL_CLIP_REVISION,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "pytorch_model.bin",
        ],
    )
    processor = CLIPProcessor.from_pretrained(local_snapshot, use_fast=False)
    model = CLIPModel.from_pretrained(
        local_snapshot,
        use_safetensors=False,
    ).to(device)
    model.eval()
    images = (
        _render_shape_image("red", "square"),
        _render_shape_image("blue", "circle"),
    )
    corrupt_images = (images[1], images[0])
    inputs = processor(
        text=list(REAL_CLIP_TEXTS),
        images=list(images),
        return_tensors="pt",
        padding=True,
    ).to(device)
    corrupt_inputs = processor(
        text=list(REAL_CLIP_TEXTS),
        images=list(corrupt_images),
        return_tensors="pt",
        padding=True,
    ).to(device)
    report = _real_contrastive_visual_token_activation_patch_report(
        model=model,
        inputs=inputs,
        corrupt_inputs=corrupt_inputs,
        embedding_module=model.vision_model.embeddings,
        model_id=REAL_CLIP_MODEL_ID,
        revision=REAL_CLIP_REVISION,
        local_snapshot=str(local_snapshot),
        patch_size=int(model.vision_model.config.patch_size),
        has_cls_token=True,
        min_object_gap=4.0,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    report["peak_vram_gb"] = peak_vram_gb
    report["within_vram_budget"] = peak_vram_gb <= max_vram_gb
    report["preflight_passed"] = (
        report["preflight_passed"] and report["within_vram_budget"]
    )

    del inputs, corrupt_inputs, model, processor
    t.cuda.empty_cache()
    return report


def real_siglip_visual_token_activation_patching_preflight(
    max_vram_gb: float = 24.0,
) -> dict:
    """Run hidden visual-token activation patching on a pinned SigLIP checkpoint."""

    if not t.cuda.is_available():
        raise RuntimeError("12.1 real SigLIP activation patching preflight requires CUDA.")

    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, SiglipModel

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    local_snapshot = snapshot_download(
        REAL_SIGLIP_MODEL_ID,
        revision=REAL_SIGLIP_REVISION,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "model.safetensors",
        ],
    )
    processor = AutoProcessor.from_pretrained(local_snapshot, use_fast=False)
    model = SiglipModel.from_pretrained(
        local_snapshot,
        use_safetensors=True,
        dtype=t.float16,
    ).to(device)
    model.eval()
    images = (
        _render_shape_image("red", "square"),
        _render_shape_image("blue", "circle"),
    )
    corrupt_images = (images[1], images[0])
    inputs = processor(
        text=list(REAL_CLIP_TEXTS),
        images=list(images),
        return_tensors="pt",
        padding="max_length",
    ).to(device)
    corrupt_inputs = processor(
        text=list(REAL_CLIP_TEXTS),
        images=list(corrupt_images),
        return_tensors="pt",
        padding="max_length",
    ).to(device)
    report = _real_contrastive_visual_token_activation_patch_report(
        model=model,
        inputs=inputs,
        corrupt_inputs=corrupt_inputs,
        embedding_module=model.vision_model.embeddings,
        model_id=REAL_SIGLIP_MODEL_ID,
        revision=REAL_SIGLIP_REVISION,
        local_snapshot=str(local_snapshot),
        patch_size=int(model.vision_model.config.patch_size),
        has_cls_token=False,
        min_object_gap=10.0,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    report["peak_vram_gb"] = peak_vram_gb
    report["within_vram_budget"] = peak_vram_gb <= max_vram_gb
    report["preflight_passed"] = (
        report["preflight_passed"] and report["within_vram_budget"]
    )

    del inputs, corrupt_inputs, model, processor
    t.cuda.empty_cache()
    return report


def real_clip_rendered_shape_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run a pinned CLIP checkpoint on deterministic rendered shape images."""

    if not t.cuda.is_available():
        raise RuntimeError("12.1 real CLIP rendered-shape preflight requires CUDA.")

    from huggingface_hub import snapshot_download
    from transformers import CLIPModel, CLIPProcessor

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    images = (
        _render_shape_image("red", "square"),
        _render_shape_image("blue", "circle"),
    )
    object_patched_images = (
        _patch_image_region(images[0], images[1], SHAPE_OBJECT_BBOX),
        _patch_image_region(images[1], images[0], SHAPE_OBJECT_BBOX),
    )
    background_patched_images = (
        _patch_image_region(images[0], images[1], SHAPE_BACKGROUND_BBOX),
        _patch_image_region(images[1], images[0], SHAPE_BACKGROUND_BBOX),
    )
    random_patched_images = (
        _patch_image_region(images[0], images[1], SHAPE_RANDOM_CONTROL_BBOX),
        _patch_image_region(images[1], images[0], SHAPE_RANDOM_CONTROL_BBOX),
    )
    local_snapshot = snapshot_download(
        REAL_CLIP_MODEL_ID,
        revision=REAL_CLIP_REVISION,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "pytorch_model.bin",
        ],
    )
    processor = CLIPProcessor.from_pretrained(local_snapshot, use_fast=False)
    model = CLIPModel.from_pretrained(
        local_snapshot,
        use_safetensors=False,
    ).to(device)
    model.eval()
    inputs = processor(
        text=list(REAL_CLIP_TEXTS),
        images=[
            *images,
            *object_patched_images,
            *background_patched_images,
            *random_patched_images,
        ],
        return_tensors="pt",
        padding=True,
    ).to(device)
    with t.inference_mode():
        output = model(**inputs)
    all_logits = output.logits_per_image.detach().float().cpu()
    logits = all_logits[:2]
    report = contrastive_alignment_report(
        logits,
        min_accuracy=1.0,
        min_positive_margin=2.0,
    )
    region_patch = _contrastive_region_patch_report(all_logits, min_object_gap=4.0)
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3

    del inputs, output, model, processor
    t.cuda.empty_cache()

    return {
        "cuda_available": True,
        "model_id": REAL_CLIP_MODEL_ID,
        "revision": REAL_CLIP_REVISION,
        "local_snapshot": str(local_snapshot),
        "claim_scope": "pinned_real_clip_rendered_shape_retrieval_and_region_patching",
        "image_count": len(images),
        "patched_image_count": (
            len(object_patched_images)
            + len(background_patched_images)
            + len(random_patched_images)
        ),
        "text_count": len(REAL_CLIP_TEXTS),
        "object_patch_bbox": SHAPE_OBJECT_BBOX,
        "background_patch_bbox": SHAPE_BACKGROUND_BBOX,
        "random_patch_bbox": SHAPE_RANDOM_CONTROL_BBOX,
        "random_patch_seed": SHAPE_RANDOM_CONTROL_SEED,
        "random_patch_same_size_as_object": _same_bbox_size(
            SHAPE_RANDOM_CONTROL_BBOX,
            SHAPE_OBJECT_BBOX,
        ),
        "random_patch_overlap_area": _bbox_overlap_area(
            SHAPE_RANDOM_CONTROL_BBOX,
            SHAPE_OBJECT_BBOX,
        ),
        "random_patch_overlaps_object": (
            _bbox_overlap_area(SHAPE_RANDOM_CONTROL_BBOX, SHAPE_OBJECT_BBOX) > 0
        ),
        "image_to_text_accuracy": report.image_to_text_accuracy,
        "text_to_image_accuracy": report.text_to_image_accuracy,
        "mean_positive_margin": report.mean_positive_margin,
        "aligned": report.aligned,
        "logits": logits.tolist(),
        "region_patch": region_patch,
        "object_patch_flips_answer": region_patch["object_patch_flips_answer"],
        "background_patch_preserves_answer": region_patch[
            "background_patch_preserves_answer"
        ],
        "random_patch_preserves_answer": region_patch["random_patch_preserves_answer"],
        "object_beats_background": region_patch["object_beats_background"],
        "object_beats_random": region_patch["object_beats_random"],
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": (
            report.aligned
            and region_patch["object_patch_flips_answer"]
            and region_patch["background_patch_preserves_answer"]
            and region_patch["random_patch_preserves_answer"]
            and region_patch["object_beats_background"]
            and region_patch["object_beats_random"]
            and peak_vram_gb <= max_vram_gb
        ),
    }


def real_qwen25_vl_rendered_shape_generation_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run a pinned generative Qwen2.5-VL checkpoint on rendered shape questions."""

    if not t.cuda.is_available():
        raise RuntimeError("12.1 real Qwen2.5-VL generation preflight requires CUDA.")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    images = [_render_shape_image(color, shape) for color, shape in REAL_QWEN25_VL_ITEMS]
    processor = AutoProcessor.from_pretrained(
        REAL_QWEN25_VL_MODEL_ID,
        revision=REAL_QWEN25_VL_REVISION,
        min_pixels=224 * 224,
        max_pixels=224 * 224,
        use_fast=False,
    )
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Answer with two lowercase words: color shape."},
                ],
            }
        ]
        for image in images
    ]
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        REAL_QWEN25_VL_MODEL_ID,
        revision=REAL_QWEN25_VL_REVISION,
        dtype=t.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    model.generation_config.temperature = None
    with t.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )
    generated_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated)
    ]
    answers = [
        answer.strip().lower()
        for answer in processor.batch_decode(generated_trimmed, skip_special_tokens=True)
    ]
    expected_answers = [f"{color} {shape}" for color, shape in REAL_QWEN25_VL_ITEMS]
    per_example_correct = [
        color in answer and shape in answer
        for answer, (color, shape) in zip(answers, REAL_QWEN25_VL_ITEMS)
    ]
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3

    del inputs, generated, model, processor
    t.cuda.empty_cache()

    return {
        "cuda_available": True,
        "model_id": REAL_QWEN25_VL_MODEL_ID,
        "revision": REAL_QWEN25_VL_REVISION,
        "claim_scope": "pinned_real_qwen25_vl_rendered_shape_generation_preflight",
        "image_count": len(images),
        "prompt_count": len(texts),
        "expected_answers": expected_answers,
        "answers": answers,
        "per_example_correct": per_example_correct,
        "accuracy": sum(per_example_correct) / len(per_example_correct),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": all(per_example_correct) and peak_vram_gb <= max_vram_gb,
    }


def real_siglip_rendered_shape_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run a pinned SigLIP checkpoint on deterministic rendered shape images."""

    if not t.cuda.is_available():
        raise RuntimeError("12.1 real SigLIP rendered-shape preflight requires CUDA.")

    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, SiglipModel

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    images = (
        _render_shape_image("red", "square"),
        _render_shape_image("blue", "circle"),
    )
    object_patched_images = (
        _patch_image_region(images[0], images[1], SHAPE_OBJECT_BBOX),
        _patch_image_region(images[1], images[0], SHAPE_OBJECT_BBOX),
    )
    background_patched_images = (
        _patch_image_region(images[0], images[1], SHAPE_BACKGROUND_BBOX),
        _patch_image_region(images[1], images[0], SHAPE_BACKGROUND_BBOX),
    )
    random_patched_images = (
        _patch_image_region(images[0], images[1], SHAPE_RANDOM_CONTROL_BBOX),
        _patch_image_region(images[1], images[0], SHAPE_RANDOM_CONTROL_BBOX),
    )
    local_snapshot = snapshot_download(
        REAL_SIGLIP_MODEL_ID,
        revision=REAL_SIGLIP_REVISION,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "model.safetensors",
        ],
    )
    processor = AutoProcessor.from_pretrained(local_snapshot, use_fast=False)
    model = SiglipModel.from_pretrained(
        local_snapshot,
        use_safetensors=True,
        dtype=t.float16,
    ).to(device)
    model.eval()
    inputs = processor(
        text=list(REAL_CLIP_TEXTS),
        images=[
            *images,
            *object_patched_images,
            *background_patched_images,
            *random_patched_images,
        ],
        return_tensors="pt",
        padding="max_length",
    ).to(device)
    with t.inference_mode():
        output = model(**inputs)
    all_logits = output.logits_per_image.detach().float().cpu()
    logits = all_logits[:2]
    report = contrastive_alignment_report(
        logits,
        min_accuracy=1.0,
        min_positive_margin=0.5,
    )
    region_patch = _contrastive_region_patch_report(all_logits, min_object_gap=10.0)
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3

    del inputs, output, model, processor
    t.cuda.empty_cache()

    return {
        "cuda_available": True,
        "model_id": REAL_SIGLIP_MODEL_ID,
        "revision": REAL_SIGLIP_REVISION,
        "local_snapshot": str(local_snapshot),
        "claim_scope": "pinned_real_siglip_rendered_shape_retrieval_and_region_patching",
        "image_count": len(images),
        "patched_image_count": (
            len(object_patched_images)
            + len(background_patched_images)
            + len(random_patched_images)
        ),
        "text_count": len(REAL_CLIP_TEXTS),
        "object_patch_bbox": SHAPE_OBJECT_BBOX,
        "background_patch_bbox": SHAPE_BACKGROUND_BBOX,
        "random_patch_bbox": SHAPE_RANDOM_CONTROL_BBOX,
        "random_patch_seed": SHAPE_RANDOM_CONTROL_SEED,
        "random_patch_same_size_as_object": _same_bbox_size(
            SHAPE_RANDOM_CONTROL_BBOX,
            SHAPE_OBJECT_BBOX,
        ),
        "random_patch_overlap_area": _bbox_overlap_area(
            SHAPE_RANDOM_CONTROL_BBOX,
            SHAPE_OBJECT_BBOX,
        ),
        "random_patch_overlaps_object": (
            _bbox_overlap_area(SHAPE_RANDOM_CONTROL_BBOX, SHAPE_OBJECT_BBOX) > 0
        ),
        "image_to_text_accuracy": report.image_to_text_accuracy,
        "text_to_image_accuracy": report.text_to_image_accuracy,
        "mean_positive_margin": report.mean_positive_margin,
        "aligned": report.aligned,
        "logits": logits.tolist(),
        "region_patch": region_patch,
        "object_patch_flips_answer": region_patch["object_patch_flips_answer"],
        "background_patch_preserves_answer": region_patch[
            "background_patch_preserves_answer"
        ],
        "random_patch_preserves_answer": region_patch["random_patch_preserves_answer"],
        "object_beats_background": region_patch["object_beats_background"],
        "object_beats_random": region_patch["object_beats_random"],
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": (
            report.aligned
            and region_patch["object_patch_flips_answer"]
            and region_patch["background_patch_preserves_answer"]
            and region_patch["random_patch_preserves_answer"]
            and region_patch["object_beats_background"]
            and region_patch["object_beats_random"]
            and peak_vram_gb <= max_vram_gb
        ),
    }


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "toy_clip_signature": toy_clip_signature_result(device="cpu"),
        "contrastive": contrastive_smoke_test(),
        "siglip": siglip_smoke_test(),
        "token_attribution": token_attribution_smoke_test(),
        "hallucination": hallucination_smoke_test(),
        "arbitration": arbitration_smoke_test(),
        "synthetic_scene_schema": synthetic_scene_schema_smoke_test(),
        "clothing_geometry": clothing_geometry_smoke_test(),
        "controlled_baselines": controlled_baselines_smoke_test(),
        "visual_region_patch": visual_region_patch_smoke_test(),
        "visual_sequence_patch": visual_sequence_patch_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("12.1 GPU verification requires CUDA.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    toy_clip = toy_clip_signature_result(device=device)
    image_embeddings = t.eye(3, device=device)
    text_embeddings = t.eye(3, device=device)
    logits = clip_contrastive_logits(
        image_embeddings,
        text_embeddings,
        logit_scale=5.0,
    )
    alignment = contrastive_alignment_report(
        logits,
        min_accuracy=1.0,
        min_positive_margin=4.0,
    )
    token_activations = t.tensor(
        [[0.0, 0.0], [3.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
        device=device,
    )
    token_report = visual_token_attribution_report(
        token_activations,
        t.tensor([1.0, 0.0], device=device),
        top_k=2,
        min_top_token_mass=0.8,
    )
    labels = t.tensor([0, 1, 0, 1], device=device)
    baselines = controlled_vlm_baseline_report(
        t.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0]], device=device),
        t.tensor([[2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5]], device=device),
        t.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], device=device),
        labels,
        min_joint_accuracy=1.0,
        min_image_only_accuracy=1.0,
        max_text_only_accuracy=0.5,
    )
    clothing_geometry = clothing_geometry_smoke_test(device)
    region_patch = visual_region_patch_smoke_test(device)
    sequence_patch = visual_sequence_patch_smoke_test(device)
    t.cuda.synchronize()
    synthetic_peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    real_clip = real_clip_rendered_shape_preflight(max_vram_gb=max_vram_gb)
    real_clip_activation_patch = real_clip_visual_token_activation_patching_preflight(
        max_vram_gb=max_vram_gb
    )
    real_siglip = real_siglip_rendered_shape_preflight(max_vram_gb=max_vram_gb)
    real_siglip_activation_patch = real_siglip_visual_token_activation_patching_preflight(
        max_vram_gb=max_vram_gb
    )
    real_qwen25_vl = real_qwen25_vl_rendered_shape_generation_preflight(
        max_vram_gb=max_vram_gb
    )
    peak_vram_gb = max(
        synthetic_peak_vram_gb,
        real_clip["peak_vram_gb"],
        real_clip_activation_patch["peak_vram_gb"],
        real_siglip["peak_vram_gb"],
        real_siglip_activation_patch["peak_vram_gb"],
        real_qwen25_vl["peak_vram_gb"],
    )
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "toy_clip_scene_count": toy_clip["scene_count"],
        "toy_clip_loss_start": toy_clip["loss_start"],
        "toy_clip_loss_end": toy_clip["loss_end"],
        "toy_clip_loss_drop": toy_clip["loss_drop"],
        "toy_clip_image_to_text_accuracy": toy_clip["image_to_text_accuracy"],
        "toy_clip_text_to_image_accuracy": toy_clip["text_to_image_accuracy"],
        "toy_clip_mean_positive_margin": toy_clip["mean_positive_margin"],
        "toy_clip_aligned": toy_clip["aligned"],
        "toy_clip_random_caption_accuracy": toy_clip["random_caption_accuracy"],
        "toy_clip_random_caption_aligned": toy_clip["random_caption_aligned"],
        "toy_clip_conflict_caption_accuracy": toy_clip["conflict_caption_accuracy"],
        "toy_clip_conflict_caption_aligned": toy_clip["conflict_caption_aligned"],
        "toy_clip_control_claim_passed": toy_clip["control_claim_passed"],
        "toy_clip_signature": toy_clip,
        "image_to_text_accuracy": alignment.image_to_text_accuracy,
        "text_to_image_accuracy": alignment.text_to_image_accuracy,
        "mean_positive_margin": alignment.mean_positive_margin,
        "aligned": alignment.aligned,
        "top_token_mass": token_report.top_token_mass,
        "localized": token_report.localized,
        "synthetic_scene_count": synthetic_scene_schema_smoke_test()["num_scenes"],
        "clothing_scene_count": clothing_geometry["scene_count"],
        "clothing_garment_accuracy": clothing_geometry["garment_accuracy"],
        "clothing_color_accuracy": clothing_geometry["color_accuracy"],
        "clothing_style_accuracy": clothing_geometry["style_accuracy"],
        "clothing_text_prior_color_agreement": clothing_geometry[
            "text_prior_color_agreement"
        ],
        "clothing_random_color_agreement": clothing_geometry["random_color_agreement"],
        "clothing_random_labels_distinct_from_text_prior": clothing_geometry[
            "random_labels_distinct_from_text_prior"
        ],
        "clothing_random_labels_distinct_from_true_labels": clothing_geometry[
            "random_labels_distinct_from_true_labels"
        ],
        "clothing_predicts_factors": clothing_geometry["predicts_clothing_factors"],
        "clothing_rejects_text_prior": clothing_geometry["rejects_text_prior"],
        "clothing_rejects_random_labels": clothing_geometry["rejects_random_labels"],
        "joint_accuracy": baselines.joint_accuracy,
        "image_only_accuracy": baselines.image_only_accuracy,
        "text_only_accuracy": baselines.text_only_accuracy,
        "joint_beats_text_only": baselines.joint_beats_text_only,
        "text_only_fails_image_questions": baselines.text_only_fails_image_questions,
        "object_patch_effect": region_patch["object_patch_effect"],
        "background_patch_effect": region_patch["background_patch_effect"],
        "random_patch_effect": region_patch["random_patch_effect"],
        "random_patch_same_size": region_patch["random_control_same_size"],
        "object_beats_background": region_patch["object_beats_background"],
        "object_beats_random": region_patch["object_beats_random"],
        "object_patch_flips_answer": region_patch["flips_answer"],
        "visual_sequence_patch_passed": sequence_patch[
            "passes_activation_patching_controls"
        ],
        "visual_sequence_patch_object_flips_answer": sequence_patch[
            "object_patch_flips_answer"
        ],
        "visual_sequence_patch_full_sequence_matches_corrupt": sequence_patch[
            "full_sequence_patch_matches_corrupt"
        ],
        "real_clip_rendered_shape_preflight_passed": real_clip["preflight_passed"],
        "real_clip_image_to_text_accuracy": real_clip["image_to_text_accuracy"],
        "real_clip_text_to_image_accuracy": real_clip["text_to_image_accuracy"],
        "real_clip_mean_positive_margin": real_clip["mean_positive_margin"],
        "real_clip_object_patch_flips_answer": real_clip["object_patch_flips_answer"],
        "real_clip_background_patch_preserves_answer": real_clip[
            "background_patch_preserves_answer"
        ],
        "real_clip_random_patch_preserves_answer": real_clip[
            "random_patch_preserves_answer"
        ],
        "real_clip_object_beats_background": real_clip["object_beats_background"],
        "real_clip_object_beats_random": real_clip["object_beats_random"],
        "real_clip_random_patch_same_size_as_object": real_clip[
            "random_patch_same_size_as_object"
        ],
        "real_clip_random_patch_overlap_area": real_clip["random_patch_overlap_area"],
        "real_clip_random_patch_overlaps_object": real_clip[
            "random_patch_overlaps_object"
        ],
        "real_clip_peak_vram_gb": real_clip["peak_vram_gb"],
        "real_clip_preflight": real_clip,
        "real_clip_visual_token_activation_patching_preflight_passed": (
            real_clip_activation_patch["preflight_passed"]
        ),
        "real_clip_activation_patch_object_flips_answer": real_clip_activation_patch[
            "object_patch_flips_answer"
        ],
        "real_clip_activation_patch_background_preserves_answer": (
            real_clip_activation_patch["background_patch_preserves_answer"]
        ),
        "real_clip_activation_patch_random_preserves_answer": real_clip_activation_patch[
            "random_patch_preserves_answer"
        ],
        "real_clip_activation_patch_full_sequence_matches_corrupt": (
            real_clip_activation_patch["full_sequence_patch_matches_corrupt"]
        ),
        "real_clip_activation_patch_full_sequence_flips_answer": (
            real_clip_activation_patch["full_sequence_patch_flips_answer"]
        ),
        "real_clip_activation_patch_min_object_gap_over_background": (
            real_clip_activation_patch["min_object_gap_over_background"]
        ),
        "real_clip_activation_patch_min_object_gap_over_random": (
            real_clip_activation_patch["min_object_gap_over_random"]
        ),
        "real_clip_activation_patch_full_sequence_max_abs_margin_error": (
            real_clip_activation_patch["full_sequence_patch_max_abs_margin_error"]
        ),
        "real_clip_activation_patch_object_token_count": real_clip_activation_patch[
            "object_token_count"
        ],
        "real_clip_activation_patch_random_token_count": real_clip_activation_patch[
            "random_token_count"
        ],
        "real_clip_activation_patch_random_control_same_size": real_clip_activation_patch[
            "random_control_same_size"
        ],
        "real_clip_activation_patch_random_control_overlaps_object": (
            real_clip_activation_patch["random_control_overlaps_object"]
        ),
        "real_clip_activation_patch_hook_point": real_clip_activation_patch[
            "hook_point"
        ],
        "real_clip_activation_patch_peak_vram_gb": real_clip_activation_patch[
            "peak_vram_gb"
        ],
        "real_clip_activation_patch_preflight": real_clip_activation_patch,
        "real_siglip_rendered_shape_preflight_passed": real_siglip["preflight_passed"],
        "real_siglip_image_to_text_accuracy": real_siglip["image_to_text_accuracy"],
        "real_siglip_text_to_image_accuracy": real_siglip["text_to_image_accuracy"],
        "real_siglip_mean_positive_margin": real_siglip["mean_positive_margin"],
        "real_siglip_object_patch_flips_answer": real_siglip["object_patch_flips_answer"],
        "real_siglip_background_patch_preserves_answer": real_siglip[
            "background_patch_preserves_answer"
        ],
        "real_siglip_random_patch_preserves_answer": real_siglip[
            "random_patch_preserves_answer"
        ],
        "real_siglip_object_beats_background": real_siglip["object_beats_background"],
        "real_siglip_object_beats_random": real_siglip["object_beats_random"],
        "real_siglip_random_patch_same_size_as_object": real_siglip[
            "random_patch_same_size_as_object"
        ],
        "real_siglip_random_patch_overlap_area": real_siglip[
            "random_patch_overlap_area"
        ],
        "real_siglip_random_patch_overlaps_object": real_siglip[
            "random_patch_overlaps_object"
        ],
        "real_siglip_peak_vram_gb": real_siglip["peak_vram_gb"],
        "real_siglip_preflight": real_siglip,
        "real_siglip_visual_token_activation_patching_preflight_passed": (
            real_siglip_activation_patch["preflight_passed"]
        ),
        "real_siglip_activation_patch_object_flips_answer": real_siglip_activation_patch[
            "object_patch_flips_answer"
        ],
        "real_siglip_activation_patch_background_preserves_answer": (
            real_siglip_activation_patch["background_patch_preserves_answer"]
        ),
        "real_siglip_activation_patch_random_preserves_answer": (
            real_siglip_activation_patch["random_patch_preserves_answer"]
        ),
        "real_siglip_activation_patch_full_sequence_matches_corrupt": (
            real_siglip_activation_patch["full_sequence_patch_matches_corrupt"]
        ),
        "real_siglip_activation_patch_full_sequence_flips_answer": (
            real_siglip_activation_patch["full_sequence_patch_flips_answer"]
        ),
        "real_siglip_activation_patch_min_object_gap_over_background": (
            real_siglip_activation_patch["min_object_gap_over_background"]
        ),
        "real_siglip_activation_patch_min_object_gap_over_random": (
            real_siglip_activation_patch["min_object_gap_over_random"]
        ),
        "real_siglip_activation_patch_full_sequence_max_abs_margin_error": (
            real_siglip_activation_patch["full_sequence_patch_max_abs_margin_error"]
        ),
        "real_siglip_activation_patch_object_token_count": real_siglip_activation_patch[
            "object_token_count"
        ],
        "real_siglip_activation_patch_random_token_count": real_siglip_activation_patch[
            "random_token_count"
        ],
        "real_siglip_activation_patch_random_control_same_size": (
            real_siglip_activation_patch["random_control_same_size"]
        ),
        "real_siglip_activation_patch_random_control_overlaps_object": (
            real_siglip_activation_patch["random_control_overlaps_object"]
        ),
        "real_siglip_activation_patch_hook_point": real_siglip_activation_patch[
            "hook_point"
        ],
        "real_siglip_activation_patch_peak_vram_gb": real_siglip_activation_patch[
            "peak_vram_gb"
        ],
        "real_siglip_activation_patch_preflight": real_siglip_activation_patch,
        "real_qwen25_vl_generation_preflight_passed": real_qwen25_vl["preflight_passed"],
        "real_qwen25_vl_accuracy": real_qwen25_vl["accuracy"],
        "real_qwen25_vl_answers": real_qwen25_vl["answers"],
        "real_qwen25_vl_expected_answers": real_qwen25_vl["expected_answers"],
        "real_qwen25_vl_peak_vram_gb": real_qwen25_vl["peak_vram_gb"],
        "real_qwen25_vl_preflight": real_qwen25_vl,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": (
            peak_vram_gb <= max_vram_gb
            and real_clip["within_vram_budget"]
            and real_clip_activation_patch["within_vram_budget"]
            and real_siglip["within_vram_budget"]
            and real_siglip_activation_patch["within_vram_budget"]
            and real_qwen25_vl["within_vram_budget"]
        ),
        "full_path": (
            "A tiny trained CLIP retrieves rendered colored-shape captions while "
            "random-caption and conflict controls fail; pinned CLIP, SigLIP, "
            "and Qwen2.5-VL controls also pass on rendered "
            "shape retrieval, object-region patching, hidden visual-token and "
            "full visual-sequence activation patching against background and "
            "same-size random-token controls, grounding, and generation evidence, "
            "plus controlled clothing factor geometry."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
