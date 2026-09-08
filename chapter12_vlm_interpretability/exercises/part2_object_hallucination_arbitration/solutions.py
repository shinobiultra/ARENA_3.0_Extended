from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch as t
from torch import Tensor, nn


SECTION_NUMBER = "12.5"
NOTEBOOK_ID = "12_5_object_hallucination_arbitration"
IMAGE_SIZE = 96
PATCH_GRID = 6
PATCH_SIZE = IMAGE_SIZE // PATCH_GRID
N_VISUAL_TOKENS = PATCH_GRID**2
PROMPT_TOKEN = N_VISUAL_TOKENS
ANSWER_TOKEN = N_VISUAL_TOKENS + 1
SEQ_LEN = N_VISUAL_TOKENS + 2
D_MODEL = 16
N_LAYERS = 4
VISUAL_STRENGTH = 4.0
PRIOR_STRENGTH = 6.0

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"

COLORS = ("red", "blue", "green", "yellow")
SHAPES = ("square", "circle")
TARGETS = tuple((color, shape) for color in COLORS for shape in SHAPES)
COLOR_RGB = {
    "red": (224, 54, 49),
    "blue": (47, 91, 214),
    "green": (40, 153, 82),
    "yellow": (224, 181, 38),
}
OBJECT_ANCHORS = ((0, 0), (64, 0), (0, 64), (64, 64))
CONDITIONS = (
    "neutral_present",
    "neutral_absent",
    "aligned_present",
    "aligned_absent",
    "conflict_present",
    "conflict_absent",
    "distractor_absent",
    "counterfactual_present",
)
CONFLICT_CONDITIONS = ("conflict_present", "conflict_absent")
SPLITS = ("development", "validation", "test")

PRIOR_NEUTRAL = 0
PRIOR_PRESENT = 1
PRIOR_ABSENT = 2

REAL_LAYER_CANDIDATE_FRACTIONS = (0.15, 0.35, 0.55, 0.75)
REAL_ACCEPTANCE_THRESHOLDS = {
    "test_example_count_min": 32,
    "validation_pair_count_min": 8,
    "test_pair_count_min": 8,
    "joint_accuracy_min": 0.55,
    "conflict_accuracy_min": 0.25,
    "hallucination_rate_max": 0.60,
    "image_question_accuracy_min": 0.60,
    "text_only_accuracy_max": 0.75,
    "object_directional_effect_min": 0.10,
    "object_flip_fraction_min": 0.25,
    "wall_clock_seconds_max": 2400.0,
    "peak_vram_gb_max": 24.0,
}


@dataclass(frozen=True)
class ArbitrationExample:
    """One task row; only ``image`` and ``prompt`` are model inputs."""

    example_id: str
    image: Tensor
    target_color: str
    target_shape: str
    object_present: bool
    prior: str
    condition: str
    split: str
    variant: int
    target_bbox: tuple[int, int, int, int]
    distractor_bbox: tuple[int, int, int, int] | None
    distractor_color: str | None
    distractor_shape: str | None
    prompt: str

    @property
    def label(self) -> int:
        return int(self.object_present)

    @property
    def answer(self) -> str:
        return "yes" if self.object_present else "no"


@dataclass(frozen=True)
class RenderedScene:
    image: Tensor
    target_bbox: tuple[int, int, int, int]
    distractor_bbox: tuple[int, int, int, int] | None
    distractor_color: str | None
    distractor_shape: str | None


@dataclass(frozen=True)
class ControlRegions:
    object: tuple[int, ...]
    background: tuple[int, ...]
    random: tuple[int, ...]

    def validate(self) -> None:
        regions = (self.object, self.background, self.random)
        if not regions[0] or len({len(region) for region in regions}) != 1:
            raise ValueError("Object, background, and random regions must be nonempty and equal size")
        if any(len(region) != len(set(region)) for region in regions):
            raise ValueError("Every control region must contain unique indices")
        for left, right in ((0, 1), (0, 2), (1, 2)):
            if not set(regions[left]).isdisjoint(regions[right]):
                raise ValueError("Object, background, and random regions must be pairwise disjoint")


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    shape: str,
    fill: tuple[int, int, int],
) -> None:
    if shape == "square":
        draw.rectangle((bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=fill)
    elif shape == "circle":
        draw.ellipse((bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=fill)
    else:
        raise ValueError(f"Unsupported shape: {shape}")


def _to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return t.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_pil(image: Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).round().astype(np.uint8))


def render_scene(
    target_color: str,
    target_shape: str,
    *,
    object_present: bool,
    variant: int,
    target_index: int,
    include_distractor: bool,
) -> RenderedScene:
    """Render a factorial scene without embedding its presence label in metadata pixels."""

    if target_color not in COLORS or target_shape not in SHAPES:
        raise ValueError("target_color and target_shape must be in the factorial vocabulary")
    background = 238 - 3 * (variant % 3)
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (background,) * 3)
    draw = ImageDraw.Draw(image)

    position_index = (target_index + variant) % len(OBJECT_ANCHORS)
    x, y = OBJECT_ANCHORS[position_index]
    target_bbox = (x, y, x + 32, y + 32)
    if object_present:
        _draw_shape(draw, target_bbox, target_shape, COLOR_RGB[target_color])

    distractor_bbox = None
    distractor_color = None
    distractor_shape = None
    if include_distractor:
        dx, dy = OBJECT_ANCHORS[(position_index + 3) % len(OBJECT_ANCHORS)]
        distractor_bbox = (dx, dy, dx + 32, dy + 32)
        color_index = COLORS.index(target_color)
        shape_index = SHAPES.index(target_shape)
        distractor_color = COLORS[(color_index + 1 + variant) % len(COLORS)]
        if distractor_color == target_color:
            distractor_color = COLORS[(color_index + 1) % len(COLORS)]
        distractor_shape = SHAPES[(shape_index + 1 + variant) % len(SHAPES)]
        _draw_shape(draw, distractor_bbox, distractor_shape, COLOR_RGB[distractor_color])
    return RenderedScene(
        image=_to_tensor(image),
        target_bbox=target_bbox,
        distractor_bbox=distractor_bbox,
        distractor_color=distractor_color,
        distractor_shape=distractor_shape,
    )


def build_prompt(target_color: str, target_shape: str, prior: str) -> str:
    question = f"Is there a {target_color} {target_shape} in the image?"
    if prior == "neutral":
        context = "Ignore prior captions and use the image."
    elif prior == "claims_present":
        context = f"A previous caption claims a {target_color} {target_shape} is present."
    elif prior == "claims_absent":
        context = f"A previous caption claims no {target_color} {target_shape} is present."
    else:
        raise ValueError(f"Unknown prior: {prior}")
    return f"{context} {question} Answer with exactly one word: yes or no."


def encode_prompt(prompt: str) -> Tensor:
    """Encode the query and stated prior from text; no answer label is encoded."""

    colors = [index for index, color in enumerate(COLORS) if f" {color} " in prompt]
    shapes = [index for index, shape in enumerate(SHAPES) if f" {shape} " in prompt]
    if len(colors) != 1 or len(shapes) != 1:
        raise ValueError(f"Prompt must name exactly one color and shape: {prompt!r}")
    if prompt.startswith("Ignore prior captions"):
        prior = PRIOR_NEUTRAL
    elif prompt.startswith("A previous caption claims no"):
        prior = PRIOR_ABSENT
    elif prompt.startswith("A previous caption claims a"):
        prior = PRIOR_PRESENT
    else:
        raise ValueError(f"Prompt has no recognized prior clause: {prompt!r}")
    return t.tensor((colors[0], shapes[0], prior), dtype=t.long)


def build_arbitration_benchmark(variants_per_target: int = 3) -> list[ArbitrationExample]:
    """Build a color-shape factorial benchmark with disjoint validation and test renders."""

    if variants_per_target != 3:
        raise ValueError("The audited benchmark requires exactly three variants per target")
    specs = (
        ("neutral_present", True, "neutral", False),
        ("neutral_absent", False, "neutral", False),
        ("aligned_present", True, "claims_present", False),
        ("aligned_absent", False, "claims_absent", False),
        ("conflict_present", True, "claims_absent", True),
        ("conflict_absent", False, "claims_present", True),
        ("distractor_absent", False, "neutral", True),
        ("counterfactual_present", True, "neutral", True),
    )
    examples: list[ArbitrationExample] = []
    for target_index, (target_color, target_shape) in enumerate(TARGETS):
        for variant, split in enumerate(SPLITS):
            for condition, present, prior, distractor in specs:
                scene = render_scene(
                    target_color,
                    target_shape,
                    object_present=present,
                    variant=variant,
                    target_index=target_index,
                    include_distractor=distractor,
                )
                examples.append(
                    ArbitrationExample(
                        example_id=f"{split[:3]}_{target_color}_{target_shape}_{condition}",
                        image=scene.image,
                        target_color=target_color,
                        target_shape=target_shape,
                        object_present=present,
                        prior=prior,
                        condition=condition,
                        split=split,
                        variant=variant,
                        target_bbox=scene.target_bbox,
                        distractor_bbox=scene.distractor_bbox,
                        distractor_color=scene.distractor_color,
                        distractor_shape=scene.distractor_shape,
                        prompt=build_prompt(target_color, target_shape, prior),
                    )
                )
    return examples


def make_counterfactual(example: ArbitrationExample) -> ArbitrationExample:
    target_index = TARGETS.index((example.target_color, example.target_shape))
    scene = render_scene(
        example.target_color,
        example.target_shape,
        object_present=not example.object_present,
        variant=example.variant,
        target_index=target_index,
        include_distractor=example.distractor_bbox is not None,
    )
    return replace(
        example,
        example_id=f"{example.example_id}_pixel_counterfactual",
        image=scene.image,
        object_present=not example.object_present,
        target_bbox=scene.target_bbox,
        distractor_bbox=scene.distractor_bbox,
        distractor_color=scene.distractor_color,
        distractor_shape=scene.distractor_shape,
    )


def batch_inputs(examples: Sequence[ArbitrationExample]) -> tuple[Tensor, Tensor, Tensor]:
    if not examples:
        raise ValueError("examples must be nonempty")
    images = t.stack([example.image for example in examples])
    prompts = t.stack([encode_prompt(example.prompt) for example in examples])
    labels = t.tensor([example.label for example in examples], dtype=t.long)
    return images, prompts, labels


def bbox_to_patch_indices(
    bbox: tuple[int, int, int, int],
    *,
    image_size: int = IMAGE_SIZE,
    patch_grid: int = PATCH_GRID,
) -> tuple[int, ...]:
    """Return row-major patches with positive-area overlap with ``bbox``."""

    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= image_size and 0 <= y1 < y2 <= image_size):
        raise ValueError(f"bbox must lie inside the image, got {bbox}")
    patch_size = image_size / patch_grid
    cols = range(int(x1 // patch_size), min(patch_grid, int(math.ceil(x2 / patch_size))))
    rows = range(int(y1 // patch_size), min(patch_grid, int(math.ceil(y2 / patch_size))))
    return tuple(row * patch_grid + col for row in rows for col in cols)


def geometry_background_indices(example: ArbitrationExample) -> tuple[int, ...]:
    """Return patches outside target and distractor geometry, independent of pixel color."""

    excluded = set(bbox_to_patch_indices(example.target_bbox))
    if example.distractor_bbox is not None:
        excluded.update(bbox_to_patch_indices(example.distractor_bbox))
    return tuple(index for index in range(N_VISUAL_TOKENS) if index not in excluded)


def _translated_region_candidates(
    object_indices: Sequence[int], available: set[int]
) -> list[tuple[int, ...]]:
    rows = [index // PATCH_GRID for index in object_indices]
    cols = [index % PATCH_GRID for index in object_indices]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    if len(object_indices) != height * width:
        return []
    candidates = []
    for top in range(PATCH_GRID - height + 1):
        for left in range(PATCH_GRID - width + 1):
            region = tuple(
                row * PATCH_GRID + col
                for row in range(top, top + height)
                for col in range(left, left + width)
            )
            if set(region) <= available:
                candidates.append(region)
    return candidates


def matched_control_regions(example: ArbitrationExample, *, seed: int = 0) -> ControlRegions:
    """Choose two equal-size, pairwise-disjoint regions from true background geometry."""

    object_region = bbox_to_patch_indices(example.target_bbox)
    background_pool = set(geometry_background_indices(example))
    candidates = _translated_region_candidates(object_region, background_pool)
    if not candidates:
        raise RuntimeError("No translated background region matches the object geometry")
    target_center = np.mean(
        [(index // PATCH_GRID, index % PATCH_GRID) for index in object_region], axis=0
    )
    candidates.sort(
        key=lambda region: (
            -float(
                np.linalg.norm(
                    np.mean(
                        [(index // PATCH_GRID, index % PATCH_GRID) for index in region], axis=0
                    )
                    - target_center
                )
            ),
            region,
        )
    )
    background = candidates[0]
    remaining = [region for region in candidates[1:] if set(region).isdisjoint(background)]
    if not remaining:
        raise RuntimeError("No second nonoverlapping matched background region is available")
    generator = t.Generator().manual_seed(seed)
    random_region = remaining[int(t.randint(len(remaining), (1,), generator=generator).item())]
    controls = ControlRegions(object_region, background, random_region)
    controls.validate()
    return controls


class PixelPromptGroundTruthVLM(nn.Module):
    """Transparent analytic VLM whose only inputs are pixels and prompt tokens.

    Layer 0 recognizes target-color 2x2 square/circle geometry, layer 1 routes the
    prompt's stated prior, and layer 2 integrates both into a yes/no margin.
    """

    def __init__(self) -> None:
        super().__init__()
        colors = t.tensor([COLOR_RGB[color] for color in COLORS], dtype=t.float32) / 255.0
        self.register_buffer("color_prototypes", colors)

    def embed(
        self,
        images: Tensor,
        prompt_tokens: Tensor,
        *,
        visual_scale: float = 1.0,
        text_scale: float = 1.0,
    ) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"images must have shape [batch, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]")
        if prompt_tokens.shape != (len(images), 3) or prompt_tokens.dtype != t.long:
            raise ValueError("prompt_tokens must be a long tensor with shape [batch, 3]")
        if not ((0 <= prompt_tokens[:, 0]) & (prompt_tokens[:, 0] < len(COLORS))).all():
            raise ValueError("prompt color token is out of range")
        if not ((0 <= prompt_tokens[:, 1]) & (prompt_tokens[:, 1] < len(SHAPES))).all():
            raise ValueError("prompt shape token is out of range")

        pixels = images.permute(0, 2, 3, 1)
        targets = self.color_prototypes[prompt_tokens[:, 0]].to(images)
        color_distance = (pixels - targets[:, None, None]).abs().amax(dim=-1)
        target_pixels = (color_distance < 0.025).float()
        occupancy = target_pixels.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(
            2, PATCH_SIZE, PATCH_SIZE
        ).mean(dim=(-1, -2))

        hidden = t.zeros(len(images), SEQ_LEN, D_MODEL, device=images.device)
        hidden[:, :N_VISUAL_TOKENS, 0] = occupancy.reshape(len(images), -1) * visual_scale
        hidden[:, :N_VISUAL_TOKENS, 2] = visual_scale
        hidden[:, PROMPT_TOKEN, 4] = prompt_tokens[:, 1].float()
        prior_lookup = t.tensor((0.0, PRIOR_STRENGTH, -PRIOR_STRENGTH), device=images.device)
        hidden[:, PROMPT_TOKEN, 5] = prior_lookup[prompt_tokens[:, 2]] * text_scale
        return hidden

    @staticmethod
    def _shape_match(occupancy: Tensor, shape_token: Tensor) -> Tensor:
        grids = occupancy.reshape(len(occupancy), PATCH_GRID, PATCH_GRID)
        windows = grids.unfold(1, 2, 1).unfold(2, 2, 1).mean(dim=(-1, -2))
        square_score = t.exp(-((windows - 1.0) / 0.08).pow(2))
        circle_score = t.exp(-((windows - 0.79) / 0.08).pow(2))
        scores = t.where(shape_token[:, None, None] < 0.5, square_score, circle_score)
        return scores.amax(dim=(1, 2))

    def step(self, hidden: Tensor, layer: int) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[1:] != (SEQ_LEN, D_MODEL):
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        out = hidden.clone()
        if layer == 0:
            match = self._shape_match(
                hidden[:, :N_VISUAL_TOKENS, 0], hidden[:, PROMPT_TOKEN, 4]
            )
            present = match >= 0.60
            visual_enabled = hidden[:, :N_VISUAL_TOKENS, 2].amax(dim=1) > 0
            visual_margin = t.where(present, VISUAL_STRENGTH, -VISUAL_STRENGTH)
            out[:, ANSWER_TOKEN, 0] = visual_margin * visual_enabled.float()
        elif layer == 1:
            out[:, ANSWER_TOKEN, 1] = hidden[:, PROMPT_TOKEN, 5]
        elif layer == 2:
            out[:, ANSWER_TOKEN, 3] = hidden[:, ANSWER_TOKEN, 0] + hidden[:, ANSWER_TOKEN, 1]
        elif layer == 3:
            out[:, ANSWER_TOKEN, 8] = t.tanh(hidden[:, ANSWER_TOKEN, 3])
        else:
            raise ValueError(f"layer must be in [0, {N_LAYERS - 1}], got {layer}")
        return out

    @staticmethod
    def logits_from_hidden(hidden: Tensor) -> Tensor:
        margin = hidden[:, ANSWER_TOKEN, 3]
        return t.stack((-0.5 * margin, 0.5 * margin), dim=-1)

    def continue_from(self, hidden: Tensor, start_layer: int) -> tuple[Tensor, list[Tensor]]:
        if not 0 <= start_layer <= N_LAYERS:
            raise ValueError(f"start_layer must be in [0, {N_LAYERS}]")
        states = [hidden]
        for layer in range(start_layer, N_LAYERS):
            hidden = self.step(hidden, layer)
            states.append(hidden)
        return self.logits_from_hidden(hidden), states

    def forward(
        self,
        images: Tensor,
        prompt_tokens: Tensor,
        *,
        visual_scale: float = 1.0,
        text_scale: float = 1.0,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        hidden = self.embed(
            images, prompt_tokens, visual_scale=visual_scale, text_scale=text_scale
        )
        logits, states = self.continue_from(hidden, 0)
        return (logits, states) if return_cache else logits


def capture_arbitration_cache(
    model: PixelPromptGroundTruthVLM,
    images: Tensor,
    prompt_tokens: Tensor,
    *,
    visual_scale: float = 1.0,
    text_scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    logits, states = model(
        images,
        prompt_tokens,
        visual_scale=visual_scale,
        text_scale=text_scale,
        return_cache=True,
    )
    return logits, t.stack(states)


def answer_margin(logits: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError(f"Expected [batch, no/yes] logits, got {tuple(logits.shape)}")
    return logits[:, 1] - logits[:, 0]


def binary_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    labels = labels.to(predictions.device)
    if labels.shape != predictions.shape or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must be binary and match the logits batch")
    absent = labels == 0
    present = labels == 1
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "hallucination_rate": float((predictions[absent] == 1).float().mean()),
        "miss_rate": float((predictions[present] == 0).float().mean()),
    }


def benchmark_metrics(
    examples: Sequence[ArbitrationExample], logits: Tensor
) -> dict[str, float]:
    labels = t.tensor([example.label for example in examples], device=logits.device)
    result = binary_metrics(logits, labels)
    conflict = t.tensor(
        [example.condition in CONFLICT_CONDITIONS for example in examples], device=logits.device
    )
    neutral = t.tensor([example.prior == "neutral" for example in examples], device=logits.device)
    predictions = logits.argmax(dim=-1)
    result.update(
        {
            "joint_accuracy": result.pop("accuracy"),
            "conflict_accuracy": float((predictions[conflict] == labels[conflict]).float().mean()),
            "neutral_accuracy": float((predictions[neutral] == labels[neutral]).float().mean()),
        }
    )
    return result


def condition_metrics(
    examples: Sequence[ArbitrationExample], logits: Tensor
) -> pd.DataFrame:
    predictions = logits.argmax(dim=-1).cpu()
    rows = []
    for condition in CONDITIONS:
        indices = [index for index, example in enumerate(examples) if example.condition == condition]
        labels = t.tensor([examples[index].label for index in indices])
        predicted = predictions[indices]
        absent = labels == 0
        rows.append(
            {
                "condition": condition,
                "n": len(indices),
                "accuracy": float((predicted == labels).float().mean()),
                "hallucination_rate": (
                    float((predicted[absent] == 1).float().mean()) if bool(absent.any()) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def modality_baseline_report(
    model: PixelPromptGroundTruthVLM,
    examples: Sequence[ArbitrationExample],
    *,
    seed: int = 0,
) -> pd.DataFrame:
    images, prompts, labels = batch_inputs(examples)
    generator = t.Generator().manual_seed(seed)
    permutation = t.randperm(len(images), generator=generator)
    conditions = {
        "joint": model(images, prompts),
        "image_question_only": model(images, prompts, text_scale=0.0),
        "text_only": model(images, prompts, visual_scale=0.0),
        "shuffled_images": model(images[permutation], prompts),
    }
    rows = []
    for condition, logits in conditions.items():
        rows.append({"condition": condition, **binary_metrics(logits, labels)})
    return pd.DataFrame(rows)


def replace_token_activations(
    recipient: Tensor,
    donor: Tensor,
    token_indices: Sequence[int],
    *,
    strength: float = 1.0,
) -> Tensor:
    """Interpolate donor activations into selected recipient token rows."""

    if recipient.shape != donor.shape or recipient.ndim != 3:
        raise ValueError("recipient and donor must have identical [batch, sequence, width] shapes")
    indices = tuple(int(index) for index in token_indices)
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("token_indices must be nonempty and unique")
    if min(indices) < 0 or max(indices) >= recipient.shape[1]:
        raise IndexError("token index is out of bounds")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    patched = recipient.clone()
    patched[:, indices] = recipient[:, indices] + strength * (
        donor[:, indices] - recipient[:, indices]
    )
    return patched


def counterfactual_patch_sweep(
    model: PixelPromptGroundTruthVLM,
    recipient: ArbitrationExample,
    donor: ArbitrationExample,
    *,
    strength: float = 1.0,
) -> pd.DataFrame:
    """Run layerwise object, background, random, and full-visual patches."""

    if recipient.prompt != donor.prompt:
        raise ValueError("A causal image counterfactual must preserve the prompt exactly")
    if recipient.label == donor.label:
        raise ValueError("Recipient and donor must have different pixel-ground-truth labels")
    recipient_images, recipient_prompts, _ = batch_inputs([recipient])
    donor_images, donor_prompts, _ = batch_inputs([donor])
    recipient_logits, recipient_cache = capture_arbitration_cache(
        model, recipient_images, recipient_prompts
    )
    donor_logits, donor_cache = capture_arbitration_cache(model, donor_images, donor_prompts)
    recipient_margin = float(answer_margin(recipient_logits).item())
    donor_margin = float(answer_margin(donor_logits).item())
    direction = 1.0 if donor_margin > recipient_margin else -1.0
    controls = matched_control_regions(recipient, seed=recipient.variant + 17)
    regions: Mapping[str, tuple[int, ...]] = {
        "object_patch": controls.object,
        "background_patch": controls.background,
        "same_size_random": controls.random,
        "full_visual_patch": tuple(range(N_VISUAL_TOKENS)),
    }
    rows: list[dict[str, object]] = []
    recipient_prediction = int(recipient_logits.argmax(dim=-1).item())
    donor_prediction = int(donor_logits.argmax(dim=-1).item())
    for layer in range(N_LAYERS + 1):
        rows.extend(
            (
                {
                    "example_id": recipient.example_id,
                    "split": recipient.split,
                    "layer": layer,
                    "condition": "recipient",
                    "margin": recipient_margin,
                    "prediction": recipient_prediction,
                    "token_count": 0,
                    "directional_effect": 0.0,
                    "answer_flip": False,
                },
                {
                    "example_id": recipient.example_id,
                    "split": recipient.split,
                    "layer": layer,
                    "condition": "donor",
                    "margin": donor_margin,
                    "prediction": donor_prediction,
                    "token_count": 0,
                    "directional_effect": abs(donor_margin - recipient_margin),
                    "answer_flip": donor_prediction != recipient_prediction,
                },
            )
        )
        for name, indices in regions.items():
            patched = replace_token_activations(
                recipient_cache[layer], donor_cache[layer], indices, strength=strength
            )
            logits, _ = model.continue_from(patched, layer)
            margin = float(answer_margin(logits).item())
            prediction = int(logits.argmax(dim=-1).item())
            rows.append(
                {
                    "example_id": recipient.example_id,
                    "split": recipient.split,
                    "layer": layer,
                    "condition": name,
                    "margin": margin,
                    "prediction": prediction,
                    "token_count": len(indices),
                    "directional_effect": direction * (margin - recipient_margin),
                    "answer_flip": prediction != recipient_prediction,
                }
            )
    return pd.DataFrame(rows)


def summarize_patch_rows(rows: pd.DataFrame, *, layer: int) -> dict[str, float]:
    selected = rows[rows.layer == layer]
    required = {"object_patch", "background_patch", "same_size_random", "full_visual_patch"}
    if not required <= set(selected.condition):
        raise ValueError(f"Patch rows are missing conditions: {required - set(selected.condition)}")
    grouped = selected.groupby("condition", sort=False)
    summary: dict[str, float] = {}
    for condition in required:
        group = grouped.get_group(condition)
        summary[f"{condition}_directional_effect_mean"] = float(group.directional_effect.mean())
        summary[f"{condition}_absolute_effect_mean"] = float(group.directional_effect.abs().mean())
        summary[f"{condition}_flip_fraction"] = float(group.answer_flip.astype(float).mean())
    return summary


def select_layer_on_validation(
    model: PixelPromptGroundTruthVLM,
    validation_pairs: Sequence[tuple[ArbitrationExample, ArbitrationExample]],
    *,
    candidate_layers: Sequence[int] = tuple(range(N_LAYERS + 1)),
) -> tuple[int, pd.DataFrame]:
    """Select one layer using validation pairs only; test rows are never inspected."""

    if not validation_pairs or any(pair[0].split != "validation" for pair in validation_pairs):
        raise ValueError("Layer selection requires nonempty validation recipient pairs")
    if len(set(candidate_layers)) != len(candidate_layers):
        raise ValueError("candidate_layers must be unique")
    all_rows = pd.concat(
        [counterfactual_patch_sweep(model, recipient, donor) for recipient, donor in validation_pairs],
        ignore_index=True,
    )
    summaries = []
    for layer in candidate_layers:
        summary = summarize_patch_rows(all_rows, layer=layer)
        control = max(
            summary["background_patch_absolute_effect_mean"],
            summary["same_size_random_absolute_effect_mean"],
        )
        summaries.append(
            {
                "layer": int(layer),
                "object_directional_effect": summary["object_patch_directional_effect_mean"],
                "background_absolute_effect": summary["background_patch_absolute_effect_mean"],
                "random_absolute_effect": summary["same_size_random_absolute_effect_mean"],
                "selection_score": summary["object_patch_directional_effect_mean"] - control,
            }
        )
    frame = pd.DataFrame(summaries).sort_values("layer").reset_index(drop=True)
    best_score = frame.selection_score.max()
    selected_layer = int(frame.loc[frame.selection_score == best_score, "layer"].min())
    return selected_layer, frame


def intervention_playground(
    model: PixelPromptGroundTruthVLM,
    recipient: ArbitrationExample,
    *,
    layer: int,
    region: str,
    strength: float,
) -> dict[str, object]:
    """Parameterized learner intervention used by the Try It Yourself cell."""

    donor = make_counterfactual(recipient)
    sweep = counterfactual_patch_sweep(model, recipient, donor, strength=strength)
    allowed = {"object_patch", "background_patch", "same_size_random", "full_visual_patch"}
    if region not in allowed:
        raise ValueError(f"region must be one of {sorted(allowed)}")
    row = sweep[(sweep.layer == layer) & (sweep.condition == region)]
    if len(row) != 1:
        raise ValueError(f"layer must be in [0, {N_LAYERS}]")
    record = row.iloc[0]
    return {
        "layer": int(layer),
        "region": region,
        "strength": float(strength),
        "margin": float(record.margin),
        "prediction": "yes" if int(record.prediction) else "no",
        "answer_flip": bool(record.answer_flip),
    }


def _counterfactual_pairs(
    examples: Sequence[ArbitrationExample], split: str
) -> list[tuple[ArbitrationExample, ArbitrationExample]]:
    recipients = [
        example
        for example in examples
        if example.split == split and example.condition == "counterfactual_present"
    ]
    return [(recipient, make_counterfactual(recipient)) for recipient in recipients]


def toy_signature_result() -> dict[str, object]:
    examples = build_arbitration_benchmark()
    model = PixelPromptGroundTruthVLM()
    test_examples = [example for example in examples if example.split == "test"]
    images, prompts, _ = batch_inputs(test_examples)
    logits = model(images, prompts)
    metrics = benchmark_metrics(test_examples, logits)
    baselines = modality_baseline_report(model, test_examples)
    selected_layer, validation_curve = select_layer_on_validation(
        model, _counterfactual_pairs(examples, "validation")
    )
    test_patch_rows = pd.concat(
        [
            counterfactual_patch_sweep(model, recipient, donor)
            for recipient, donor in _counterfactual_pairs(examples, "test")
        ],
        ignore_index=True,
    )
    test_patch = summarize_patch_rows(test_patch_rows, layer=selected_layer)
    accepted = bool(
        len(examples) == 192
        and metrics["joint_accuracy"] == 0.75
        and metrics["conflict_accuracy"] == 0.0
        and metrics["hallucination_rate"] == 0.25
        and selected_layer == 0
        and test_patch["object_patch_flip_fraction"] == 1.0
        and test_patch["object_patch_directional_effect_mean"] > 0
        and test_patch["background_patch_absolute_effect_mean"] == 0
        and test_patch["same_size_random_absolute_effect_mean"] == 0
    )
    return {
        "accepted": accepted,
        "examples": examples,
        "test_examples": test_examples,
        "test_logits": logits,
        "metrics": metrics,
        "conditions": condition_metrics(test_examples, logits),
        "baselines": baselines,
        "selected_layer": selected_layer,
        "validation_curve": validation_curve,
        "test_patch_rows": test_patch_rows,
        "test_patch_summary": test_patch,
    }


def make_cold_open_figure(output_path: str | Path | None = None):
    import matplotlib.pyplot as plt

    examples = build_arbitration_benchmark()
    example = next(
        item
        for item in examples
        if item.split == "test"
        and item.target_color == "red"
        and item.target_shape == "square"
        and item.condition == "conflict_absent"
    )
    model = PixelPromptGroundTruthVLM()
    images, prompts, _ = batch_inputs([example])
    joint = float(answer_margin(model(images, prompts)).item())
    image_question = float(answer_margin(model(images, prompts, text_scale=0.0)).item())

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), constrained_layout=True)
    axes[0].imshow(tensor_to_pil(example.image))
    axes[0].set_title("Pixels: no red square")
    axes[0].axis("off")
    labels = ("joint\nanswer", "image +\nquestion")
    margins = (joint, image_question)
    axes[1].bar(labels, margins, color=("#C43C39", "#27866F"), width=0.62)
    axes[1].axhline(0, color="#222222", linewidth=1)
    axes[1].set_ylabel("yes minus no margin")
    axes[1].set_title("The stated prior creates a hallucination")
    for index, margin in enumerate(margins):
        answer = "yes" if margin >= 0 else "no"
        axes[1].text(
            index,
            margin * 0.55,
            answer,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def make_signature_figure(
    result: Mapping[str, object], output_path: str | Path | None = None
):
    import matplotlib.pyplot as plt

    test_examples: Sequence[ArbitrationExample] = result["test_examples"]  # type: ignore[assignment]
    logits: Tensor = result["test_logits"]  # type: ignore[assignment]
    chosen = [
        index
        for index, example in enumerate(test_examples)
        if example.condition in {"neutral_present", "conflict_absent", "distractor_absent"}
    ][:4]
    metrics: Mapping[str, float] = result["metrics"]  # type: ignore[assignment]
    baselines: pd.DataFrame = result["baselines"]  # type: ignore[assignment]
    validation: pd.DataFrame = result["validation_curve"]  # type: ignore[assignment]
    patch: Mapping[str, float] = result["test_patch_summary"]  # type: ignore[assignment]

    fig = plt.figure(figsize=(15, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(1.0, 0.9, 1.0))
    for column, index in enumerate(chosen):
        example = test_examples[index]
        prediction = "yes" if int(logits[index].argmax()) else "no"
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(tensor_to_pil(example.image))
        ax.set_title(
            f"{example.target_color} {example.target_shape}\n"
            f"{example.condition.replace('_', ' ')}\ntruth={example.answer}, model={prediction}",
            fontsize=9,
        )
        ax.axis("off")

    ax = fig.add_subplot(grid[1, :2])
    names = ("joint", "conflict", "hallucination")
    values = (
        metrics["joint_accuracy"],
        metrics["conflict_accuracy"],
        metrics["hallucination_rate"],
    )
    ax.bar(names, values, color=("#27866F", "#C43C39", "#D9852B"))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate on untouched test split")
    ax.set_title("Conflict exposes prior-driven errors")

    ax = fig.add_subplot(grid[1, 2:])
    ax.bar(
        baselines.condition,
        baselines.accuracy,
        color=("#27866F", "#3977A8", "#C43C39", "#7A6A9E"),
    )
    ax.axhline(0.5, color="#222222", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Question-only and shuffled-image controls fail")
    ax.tick_params(axis="x", rotation=18)

    ax = fig.add_subplot(grid[2, :2])
    ax.plot(
        validation.layer,
        validation.object_directional_effect,
        marker="o",
        linewidth=2,
        label="object patch",
    )
    ax.plot(
        validation.layer,
        validation.background_absolute_effect,
        marker="o",
        label="background control",
    )
    ax.plot(
        validation.layer,
        validation.random_absolute_effect,
        marker="o",
        label="random control",
    )
    ax.axvline(int(result["selected_layer"]), color="#222222", linestyle="--")
    ax.set_xlabel("cached layer (selection uses validation only)")
    ax.set_ylabel("directional margin effect")
    ax.set_title("Validation selects the pre-routing visual-token layer")
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[2, 2:])
    control_names = ("object", "background", "random", "full visual")
    control_values = (
        patch["object_patch_directional_effect_mean"],
        patch["background_patch_absolute_effect_mean"],
        patch["same_size_random_absolute_effect_mean"],
        patch["full_visual_patch_directional_effect_mean"],
    )
    ax.bar(control_names, control_values, color=("#27866F", "#3977A8", "#7A6A9E", "#D9852B"))
    ax.set_ylabel("effect on untouched test pairs")
    ax.set_title("Object patches flip answers; disjoint controls do not")
    ax.tick_params(axis="x", rotation=16)
    fig.suptitle("12.5 Object Hallucination and Modality Arbitration", fontsize=17)
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def make_layer_intervention_figure(
    result: Mapping[str, object], output_path: str | Path | None = None
):
    import matplotlib.pyplot as plt

    rows: pd.DataFrame = result["test_patch_rows"]  # type: ignore[assignment]
    conditions = ("object_patch", "background_patch", "same_size_random", "full_visual_patch")
    table = (
        rows[rows.condition.isin(conditions)]
        .groupby(["condition", "layer"])
        .directional_effect.mean()
        .unstack("layer")
        .loc[list(conditions)]
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    limit = max(1.0, float(table.abs().to_numpy().max()))
    image = ax.imshow(table, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(table.index)), [name.replace("_", " ") for name in table.index])
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_xlabel("intervention layer")
    ax.set_title("Donor-directed effects vanish after evidence leaves patch rows")
    fig.colorbar(image, ax=ax, label="directional margin effect")
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def visual_token_indices(input_ids: Tensor, image_token_id: int) -> list[Tensor]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    return [t.where(row == image_token_id)[0] for row in input_ids]


def bbox_to_vlm_token_indices(
    bbox: tuple[int, int, int, int],
    image_token_positions: Tensor,
    grid_thw: Tensor,
    *,
    image_size: int = IMAGE_SIZE,
    merge_size: int = 2,
) -> Tensor:
    temporal, grid_h, grid_w = (int(value) for value in grid_thw.tolist())
    if temporal != 1 or grid_h % merge_size or grid_w % merge_size:
        raise ValueError(f"Unsupported visual grid {tuple(grid_thw.tolist())}")
    token_h, token_w = grid_h // merge_size, grid_w // merge_size
    if image_token_positions.numel() != token_h * token_w:
        raise ValueError("Image-token positions do not match the merged visual grid")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= image_size and 0 <= y1 < y2 <= image_size):
        raise ValueError(f"bbox must lie inside the image, got {bbox}")
    cols = range(int(x1 / image_size * token_w), min(token_w, math.ceil(x2 / image_size * token_w)))
    rows = range(int(y1 / image_size * token_h), min(token_h, math.ceil(y2 / image_size * token_h)))
    local = t.tensor(
        [row * token_w + col for row in rows for col in cols],
        device=image_token_positions.device,
    )
    return image_token_positions[local]


def geometry_matched_vlm_controls(
    example: ArbitrationExample,
    image_token_positions: Tensor,
    grid_thw: Tensor,
    *,
    merge_size: int,
    seed: int,
) -> ControlRegions:
    object_positions = bbox_to_vlm_token_indices(
        example.target_bbox, image_token_positions, grid_thw, merge_size=merge_size
    )
    excluded = set(object_positions.tolist())
    if example.distractor_bbox is not None:
        distractor = bbox_to_vlm_token_indices(
            example.distractor_bbox, image_token_positions, grid_thw, merge_size=merge_size
        )
        excluded.update(distractor.tolist())
    available = [position for position in image_token_positions.tolist() if position not in excluded]
    count = int(object_positions.numel())
    if len(available) < 2 * count:
        raise RuntimeError("Not enough true-background visual tokens for two disjoint controls")
    generator = t.Generator().manual_seed(seed)
    order = t.randperm(len(available), generator=generator).tolist()
    background = tuple(available[index] for index in order[:count])
    random_region = tuple(available[index] for index in order[count : 2 * count])
    controls = ControlRegions(tuple(object_positions.tolist()), background, random_region)
    controls.validate()
    return controls


def capture_module_output(
    module: nn.Module, forward: Callable[[], object]
) -> tuple[object, Tensor]:
    captured: list[Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object):
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, Tensor):
            raise TypeError("The hooked module must return a tensor or tensor-first tuple")
        captured.append(tensor.detach())

    handle = module.register_forward_hook(hook)
    try:
        result = forward()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one hook call, got {len(captured)}")
    return result, captured[0]


def _require_cuda(max_vram_gb: float) -> None:
    if not t.cuda.is_available():
        raise RuntimeError("The pinned real-VLM experiment requires CUDA; CPU fallback is forbidden.")
    if not 0 < max_vram_gb <= REAL_ACCEPTANCE_THRESHOLDS["peak_vram_gb_max"]:
        raise ValueError("max_vram_gb must be positive and no greater than the audited budget")


def load_pinned_real_vlm(max_vram_gb: float = 24.0):
    _require_cuda(max_vram_gb)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    processor.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=t.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
    ).eval()
    return model, processor


def _qwen_messages(
    example: ArbitrationExample, *, include_image: bool, neutral_prior: bool
) -> list[dict[str, object]]:
    prompt = (
        build_prompt(example.target_color, example.target_shape, "neutral")
        if neutral_prior
        else example.prompt
    )
    content: list[dict[str, object]] = []
    if include_image:
        content.append({"type": "image", "image": tensor_to_pil(example.image)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def prepare_real_vlm_batch(
    processor,
    examples: Sequence[ArbitrationExample],
    *,
    include_image: bool = True,
    neutral_prior: bool = False,
) -> dict[str, Tensor]:
    messages = [
        _qwen_messages(example, include_image=include_image, neutral_prior=neutral_prior)
        for example in examples
    ]
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    images = [tensor_to_pil(example.image) for example in examples] if include_image else None
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    return {
        key: value.to("cuda:0") if isinstance(value, Tensor) else value
        for key, value in inputs.items()
    }


def answer_token_ids(processor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    tokenizer = processor.tokenizer

    def variants(word: str) -> tuple[int, ...]:
        ids = set()
        for text in (word, word.capitalize(), f" {word}", f" {word.capitalize()}"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
        if not ids:
            raise RuntimeError(f"No single-token variant found for {word!r}")
        return tuple(sorted(ids))

    return variants("yes"), variants("no")


def _real_margin_from_outputs(
    outputs, inputs: Mapping[str, Tensor], processor, *, batch_index: int = 0
) -> float:
    yes_ids, no_ids = answer_token_ids(processor)
    last = int(inputs["attention_mask"][batch_index].sum().item() - 1)
    logits = outputs.logits[batch_index, last].float()
    yes_score = t.logsumexp(logits[list(yes_ids)], dim=0)
    no_score = t.logsumexp(logits[list(no_ids)], dim=0)
    return float((yes_score - no_score).item())


@t.inference_mode()
def collect_real_vlm_evidence(
    model,
    processor,
    examples: Sequence[ArbitrationExample],
    *,
    batch_size: int = 8,
    include_image: bool = True,
    neutral_prior: bool = False,
) -> tuple[Tensor, Tensor]:
    margins: list[float] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        inputs = prepare_real_vlm_batch(
            processor, batch, include_image=include_image, neutral_prior=neutral_prior
        )
        outputs = model(**inputs, use_cache=False)
        margins.extend(
            _real_margin_from_outputs(outputs, inputs, processor, batch_index=index)
            for index in range(len(batch))
        )
    margin_tensor = t.tensor(margins)
    return margin_tensor, (margin_tensor >= 0).long()


def patch_real_vlm_layer(
    model,
    recipient_inputs: Mapping[str, Tensor],
    donor_hidden: Tensor,
    *,
    layer: int,
    token_indices: Tensor,
    strength: float = 1.0,
):
    block = model.model.language_model.layers[layer]

    def patch_hook(_module: nn.Module, _inputs: tuple[object, ...], output: object):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, Tensor):
            raise TypeError("Language block output must begin with a tensor")
        patched = hidden.clone()
        selected = token_indices.to(hidden.device)
        patched[:, selected] = hidden[:, selected] + strength * (
            donor_hidden[:, selected].to(hidden) - hidden[:, selected]
        )
        return (patched, *output[1:]) if isinstance(output, tuple) else patched

    handle = block.register_forward_hook(patch_hook)
    try:
        return model(**recipient_inputs, use_cache=False)
    finally:
        handle.remove()


@t.inference_mode()
def real_activation_patch_controls(
    model,
    processor,
    recipient: ArbitrationExample,
    donor: ArbitrationExample,
    *,
    layer: int,
    include_full_visual: bool = True,
) -> list[dict[str, object]]:
    if recipient.prompt != donor.prompt or recipient.label == donor.label:
        raise ValueError("Real patching requires a prompt-identical, label-changing image pair")
    recipient_inputs = prepare_real_vlm_batch(processor, [recipient])
    donor_inputs = prepare_real_vlm_batch(processor, [donor])
    if recipient_inputs["input_ids"].shape != donor_inputs["input_ids"].shape:
        raise RuntimeError("Counterfactual inputs must have identical tokenized lengths")
    block = model.model.language_model.layers[layer]
    donor_outputs, donor_hidden = capture_module_output(
        block, lambda: model(**donor_inputs, use_cache=False)
    )
    recipient_outputs = model(**recipient_inputs, use_cache=False)
    recipient_margin = _real_margin_from_outputs(recipient_outputs, recipient_inputs, processor)
    donor_margin = _real_margin_from_outputs(donor_outputs, donor_inputs, processor)
    direction = 1.0 if donor_margin > recipient_margin else -1.0

    image_token_id = int(model.config.image_token_id)
    visual_positions = visual_token_indices(recipient_inputs["input_ids"], image_token_id)[0]
    controls = geometry_matched_vlm_controls(
        recipient,
        visual_positions,
        recipient_inputs["image_grid_thw"][0],
        merge_size=int(model.config.vision_config.spatial_merge_size),
        seed=recipient.variant + 313 * TARGETS.index(
            (recipient.target_color, recipient.target_shape)
        ),
    )
    patch_regions: dict[str, tuple[int, ...]] = {
        "object_patch": controls.object,
        "background_patch": controls.background,
        "same_size_random": controls.random,
    }
    if include_full_visual:
        patch_regions["full_visual_patch"] = tuple(visual_positions.tolist())

    recipient_prediction = recipient_margin >= 0
    rows: list[dict[str, object]] = [
        {
            "example_id": recipient.example_id,
            "split": recipient.split,
            "layer": layer,
            "condition": "recipient",
            "margin": recipient_margin,
            "prediction": "yes" if recipient_prediction else "no",
            "token_count": 0,
            "directional_effect": 0.0,
            "answer_flip": False,
        },
        {
            "example_id": recipient.example_id,
            "split": recipient.split,
            "layer": layer,
            "condition": "donor",
            "margin": donor_margin,
            "prediction": "yes" if donor_margin >= 0 else "no",
            "token_count": 0,
            "directional_effect": abs(donor_margin - recipient_margin),
            "answer_flip": (donor_margin >= 0) != recipient_prediction,
        },
    ]
    for condition, indices in patch_regions.items():
        outputs = patch_real_vlm_layer(
            model,
            recipient_inputs,
            donor_hidden,
            layer=layer,
            token_indices=t.tensor(indices, device="cuda:0"),
        )
        margin = _real_margin_from_outputs(outputs, recipient_inputs, processor)
        rows.append(
            {
                "example_id": recipient.example_id,
                "split": recipient.split,
                "layer": layer,
                "condition": condition,
                "margin": margin,
                "prediction": "yes" if margin >= 0 else "no",
                "token_count": len(indices),
                "directional_effect": direction * (margin - recipient_margin),
                "answer_flip": (margin >= 0) != recipient_prediction,
            }
        )
    return rows


def _candidate_real_layers(model) -> tuple[int, ...]:
    count = len(model.model.language_model.layers)
    return tuple(
        sorted(
            {
                min(count - 1, max(0, int(round(fraction * (count - 1)))))
                for fraction in REAL_LAYER_CANDIDATE_FRACTIONS
            }
        )
    )


def _summarize_real_selection(rows: pd.DataFrame, layer: int) -> dict[str, float]:
    selected = rows[rows.layer == layer]
    result = {}
    for condition in ("object_patch", "background_patch", "same_size_random"):
        group = selected[selected.condition == condition]
        result[f"{condition}_directional"] = float(group.directional_effect.mean())
        result[f"{condition}_absolute"] = float(group.directional_effect.abs().mean())
    return result


def select_real_layer_on_validation(
    model,
    processor,
    validation_pairs: Sequence[tuple[ArbitrationExample, ArbitrationExample]],
    candidate_layers: Sequence[int],
) -> tuple[int, list[dict[str, object]]]:
    if not validation_pairs or any(pair[0].split != "validation" for pair in validation_pairs):
        raise ValueError("Real layer selection may use validation pairs only")
    rows = [
        row
        for layer in candidate_layers
        for recipient, donor in validation_pairs
        for row in real_activation_patch_controls(
            model,
            processor,
            recipient,
            donor,
            layer=layer,
            include_full_visual=False,
        )
    ]
    frame = pd.DataFrame(rows)
    summaries = []
    for layer in candidate_layers:
        summary = _summarize_real_selection(frame, layer)
        control = max(
            summary["background_patch_absolute"], summary["same_size_random_absolute"]
        )
        summaries.append(
            {
                "layer": int(layer),
                "object_directional_effect": summary["object_patch_directional"],
                "control_effect": control,
                "selection_score": summary["object_patch_directional"] - control,
            }
        )
    summary_frame = pd.DataFrame(summaries)
    best = summary_frame.selection_score.max()
    selected = int(summary_frame.loc[summary_frame.selection_score == best, "layer"].min())
    return selected, summary_frame.to_dict(orient="records")


def evaluate_real_acceptance(result: Mapping[str, object]) -> tuple[bool, list[str]]:
    required = {
        "model_id",
        "revision",
        "test_example_count",
        "validation_pair_count",
        "test_pair_count",
        "selected_layer",
        "layer_selection_split",
        "reporting_split",
        "joint_accuracy",
        "conflict_accuracy",
        "hallucination_rate",
        "image_question_accuracy",
        "text_only_accuracy",
        "test_patch_summary",
        "peak_vram_gb",
        "wall_clock_seconds",
        "joint_margins", "image_question_margins", "text_only_margins",
        "test_patch_rows", "candidate_layers",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"Real result is missing required fields: {sorted(missing)}")
    failures = []
    if result["model_id"] != MODEL_ID or result["revision"] != MODEL_REVISION:
        failures.append("model identity does not match the pinned checkpoint")
    rate_keys = (
        "joint_accuracy", "conflict_accuracy", "hallucination_rate",
        "image_question_accuracy", "text_only_accuracy",
    )
    numeric_keys = (*rate_keys, "test_example_count", "validation_pair_count",
                    "test_pair_count", "selected_layer", "peak_vram_gb", "wall_clock_seconds")
    for key in numeric_keys:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{key} must be finite numeric evidence")
        elif key in rate_keys and not 0 <= value <= 1:
            failures.append(f"{key} must be a probability in [0, 1]")
        elif key not in rate_keys and value < 0:
            failures.append(f"{key} must be nonnegative")
    for key in ("test_example_count", "validation_pair_count", "test_pair_count", "selected_layer"):
        if not isinstance(result[key], int) or isinstance(result[key], bool):
            failures.append(f"{key} must be an integer")
    if result["selected_layer"] not in result["candidate_layers"]:
        failures.append("selected_layer must be one of the evaluated candidates")
    if failures:
        return False, failures

    # Recompute behavior from ordered example margins rather than trusting five
    # caller-supplied aggregate scores. These are preflight metrics, not proof of
    # correcting hallucinations: the intervention cohort has neutral priors.
    examples = build_arbitration_benchmark()
    test_examples = [example for example in examples if example.split == "test"]
    expected_pairs = _counterfactual_pairs(examples, "test")
    if result["test_example_count"] != len(test_examples):
        failures.append("test_example_count disagrees with the fixed held-out suite")
    if result["test_pair_count"] != len(expected_pairs):
        failures.append("test_pair_count disagrees with the fixed counterfactual cohort")
    if result["validation_pair_count"] != len(_counterfactual_pairs(examples, "validation")):
        failures.append("validation_pair_count disagrees with the fixed validation cohort")
    vectors = {}
    for key in ("joint_margins", "image_question_margins", "text_only_margins"):
        values = result[key]
        if (not isinstance(values, list) or len(values) != len(test_examples)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
            failures.append(f"{key} must contain one finite margin per held-out example")
        else:
            vectors[key] = t.tensor(values, dtype=t.float64) >= 0
    if failures:
        return False, failures
    labels = t.tensor([example.label for example in test_examples])
    conflict = t.tensor([example.condition in CONFLICT_CONDITIONS for example in test_examples])
    joint = vectors["joint_margins"]
    derived = {
        "joint_accuracy": float((joint == labels).float().mean()),
        "conflict_accuracy": float((joint[conflict] == labels[conflict]).float().mean()),
        "hallucination_rate": float(joint[labels == 0].float().mean()),
        "image_question_accuracy": float((vectors["image_question_margins"] == labels).float().mean()),
        "text_only_accuracy": float((vectors["text_only_margins"] == labels).float().mean()),
    }
    for key, value in derived.items():
        if not math.isclose(float(result[key]), value, rel_tol=0, abs_tol=1e-6):
            failures.append(f"{key} disagrees with raw held-out margins")
    try:
        rows = pd.DataFrame(result["test_patch_rows"])
        expected_ids = {recipient.example_id for recipient, _ in expected_pairs}
        required_conditions = {"recipient", "donor", "object_patch", "background_patch", "same_size_random", "full_visual_patch"}
        if (set(rows.example_id) != expected_ids or set(rows.split) != {"test"}
                or set(rows.layer) != {result["selected_layer"]}
                or len(rows) != len(expected_ids) * len(required_conditions)):
            raise ValueError("patch cohort, split, layer, or row count mismatch")
        for _, group in rows.groupby("example_id"):
            if set(group.condition) != required_conditions or group.condition.duplicated().any():
                raise ValueError("each example needs exactly one row for every control")
            recipient_margin = float(group.loc[group.condition == "recipient", "margin"].iloc[0])
            donor_margin = float(group.loc[group.condition == "donor", "margin"].iloc[0])
            direction = 1.0 if donor_margin > recipient_margin else -1.0
            for row in group.itertuples():
                expected_effect = direction * (float(row.margin) - recipient_margin)
                expected_flip = (float(row.margin) >= 0) != (recipient_margin >= 0)
                if (not math.isclose(float(row.directional_effect), expected_effect, rel_tol=0, abs_tol=1e-6)
                        or row.answer_flip != expected_flip
                        or row.prediction != ("yes" if row.margin >= 0 else "no")):
                    raise ValueError("patch effect, prediction, or flip disagrees with raw margin")
        if not np.isfinite(rows[["margin", "directional_effect"]].to_numpy(dtype=float)).all():
            raise ValueError("patch effects must be finite")
        if any(not isinstance(value, bool) for value in rows.answer_flip):
            raise ValueError("answer_flip must contain booleans")
        recomputed = summarize_patch_rows(rows, layer=int(result["selected_layer"]))
        supplied = result["test_patch_summary"]
        for key, value in recomputed.items():
            if key not in supplied or not math.isclose(float(supplied[key]), value, rel_tol=0, abs_tol=1e-6):
                failures.append(f"test_patch_summary {key} disagrees with raw patch rows")
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        failures.append(f"invalid test_patch_rows: {exc}")
    if failures:
        return False, failures

    def minimum(key: str, threshold_key: str) -> None:
        if float(result[key]) < REAL_ACCEPTANCE_THRESHOLDS[threshold_key]:
            failures.append(f"{key} below {REAL_ACCEPTANCE_THRESHOLDS[threshold_key]}")

    def maximum(key: str, threshold_key: str) -> None:
        if float(result[key]) > REAL_ACCEPTANCE_THRESHOLDS[threshold_key]:
            failures.append(f"{key} above {REAL_ACCEPTANCE_THRESHOLDS[threshold_key]}")

    minimum("test_example_count", "test_example_count_min")
    minimum("validation_pair_count", "validation_pair_count_min")
    minimum("test_pair_count", "test_pair_count_min")
    minimum("joint_accuracy", "joint_accuracy_min")
    minimum("conflict_accuracy", "conflict_accuracy_min")
    maximum("hallucination_rate", "hallucination_rate_max")
    minimum("image_question_accuracy", "image_question_accuracy_min")
    maximum("text_only_accuracy", "text_only_accuracy_max")
    maximum("peak_vram_gb", "peak_vram_gb_max")
    maximum("wall_clock_seconds", "wall_clock_seconds_max")
    if result["layer_selection_split"] != "validation" or result["reporting_split"] != "test":
        failures.append("layer selection and reporting splits are not validation/test")

    patch: Mapping[str, float] = result["test_patch_summary"]  # type: ignore[assignment]
    object_effect = float(patch["object_patch_directional_effect_mean"])
    background = float(patch["background_patch_absolute_effect_mean"])
    random_control = float(patch["same_size_random_absolute_effect_mean"])
    full_effect = float(patch["full_visual_patch_directional_effect_mean"])
    flip_fraction = float(patch["object_patch_flip_fraction"])
    if object_effect <= max(background, random_control):
        failures.append("object patch does not beat both matched controls")
    if full_effect <= max(background, random_control):
        failures.append("full-visual positive control does not beat matched controls")
    if not (
        object_effect >= REAL_ACCEPTANCE_THRESHOLDS["object_directional_effect_min"]
        or flip_fraction >= REAL_ACCEPTANCE_THRESHOLDS["object_flip_fraction_min"]
    ):
        failures.append("object patch has neither a directional effect nor enough answer flips")
    return not failures, failures


def run_real_vlm_benchmark(max_vram_gb: float = 24.0) -> dict[str, object]:
    _require_cuda(max_vram_gb)
    examples = build_arbitration_benchmark()
    test_examples = [example for example in examples if example.split == "test"]
    validation_pairs = _counterfactual_pairs(examples, "validation")
    test_pairs = _counterfactual_pairs(examples, "test")
    t.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, processor = load_pinned_real_vlm(max_vram_gb=max_vram_gb)

    joint_margins, joint_predictions = collect_real_vlm_evidence(model, processor, test_examples)
    image_margins, image_predictions = collect_real_vlm_evidence(
        model, processor, test_examples, neutral_prior=True
    )
    text_margins, text_predictions = collect_real_vlm_evidence(
        model, processor, test_examples, include_image=False
    )
    labels = t.tensor([example.label for example in test_examples])
    conflict = t.tensor([example.condition in CONFLICT_CONDITIONS for example in test_examples])
    absent = labels == 0

    candidate_layers = _candidate_real_layers(model)
    selected_layer, validation_summary = select_real_layer_on_validation(
        model, processor, validation_pairs, candidate_layers
    )
    patch_rows = [
        row
        for recipient, donor in test_pairs
        for row in real_activation_patch_controls(
            model,
            processor,
            recipient,
            donor,
            layer=selected_layer,
            include_full_visual=True,
        )
    ]
    patch_summary = summarize_patch_rows(pd.DataFrame(patch_rows), layer=selected_layer)
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "test_example_count": len(test_examples),
        "validation_pair_count": len(validation_pairs),
        "test_pair_count": len(test_pairs),
        "candidate_layers": list(candidate_layers),
        "selected_layer": selected_layer,
        "layer_selection_split": "validation",
        "reporting_split": "test",
        "joint_accuracy": float((joint_predictions == labels).float().mean()),
        "conflict_accuracy": float((joint_predictions[conflict] == labels[conflict]).float().mean()),
        "hallucination_rate": float((joint_predictions[absent] == 1).float().mean()),
        "image_question_accuracy": float((image_predictions == labels).float().mean()),
        "text_only_accuracy": float((text_predictions == labels).float().mean()),
        "joint_margins": joint_margins.tolist(),
        "image_question_margins": image_margins.tolist(),
        "text_only_margins": text_margins.tolist(),
        "validation_layer_summary": validation_summary,
        "test_patch_rows": patch_rows,
        "test_patch_summary": patch_summary,
        "peak_vram_gb": t.cuda.max_memory_allocated() / 1024**3,
        "wall_clock_seconds": elapsed,
    }
    accepted, failures = evaluate_real_acceptance(result)
    result["acceptance_failures"] = failures
    result["accepted"] = accepted
    return result


def run_smoke_test(cpu: bool = True) -> dict[str, object]:
    if not cpu:
        raise ValueError("The deterministic smoke fixture is CPU-only; use run_gpu_test for Qwen")
    result = toy_signature_result()
    metrics: Mapping[str, float] = result["metrics"]  # type: ignore[assignment]
    patch: Mapping[str, float] = result["test_patch_summary"]  # type: ignore[assignment]
    baselines: pd.DataFrame = result["baselines"]  # type: ignore[assignment]
    baseline_accuracy = dict(zip(baselines.condition, baselines.accuracy))
    return {
        "notebook_id": NOTEBOOK_ID,
        "organism": "pixel_prompt_ground_truth_vlm_v2",
        "example_count": len(result["examples"]),  # type: ignore[arg-type]
        "validation_pair_count": 8,
        "test_pair_count": 8,
        "selected_layer": int(result["selected_layer"]),
        "joint_accuracy": metrics["joint_accuracy"],
        "conflict_accuracy": metrics["conflict_accuracy"],
        "hallucination_rate": metrics["hallucination_rate"],
        "image_question_accuracy": float(baseline_accuracy["image_question_only"]),
        "text_only_accuracy": float(baseline_accuracy["text_only"]),
        "shuffled_images_accuracy": float(baseline_accuracy["shuffled_images"]),
        "object_patch_directional_effect": patch["object_patch_directional_effect_mean"],
        "object_patch_flip_fraction": patch["object_patch_flip_fraction"],
        "background_patch_absolute_effect": patch["background_patch_absolute_effect_mean"],
        "random_patch_absolute_effect": patch["same_size_random_absolute_effect_mean"],
        "full_visual_directional_effect": patch["full_visual_patch_directional_effect_mean"],
        "accepted": bool(result["accepted"]),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, object]:
    return run_real_vlm_benchmark(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, object]:
    return run_gpu_test(max_vram_gb=max_vram_gb)


def write_real_result(result: Mapping[str, object], output_path: str | Path) -> None:
    accepted, failures = evaluate_real_acceptance(result)
    if bool(result.get("accepted")) != accepted or list(result.get("acceptance_failures", [])) != failures:
        raise ValueError("Result acceptance fields do not match the strict acceptance contract")
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
