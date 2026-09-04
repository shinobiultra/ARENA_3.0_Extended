"""Reference solutions for [12.3] Mini VLM from Scratch.

The section intentionally stays small enough to run on CPU. It trains a real
PyTorch VQA model on rendered colored-shape scenes, then evaluates causal
visual-token patching with object/background/random controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch as t
import torch.nn as nn
import torch.nn.functional as F


COLORS = ("red", "blue", "green", "yellow")
SHAPES = ("square", "circle", "triangle")
ANSWER_VOCAB = COLORS + SHAPES
QUESTION_TYPES = ("color", "shape")
QUESTION_TO_ID = {question: index for index, question in enumerate(QUESTION_TYPES)}
ANSWER_TO_ID = {answer: index for index, answer in enumerate(ANSWER_VOCAB)}
UNKNOWN_QUESTION_ID = len(QUESTION_TYPES)

IMAGE_SIZE = 48
PATCH_GRID = 6
OBJECT_SIZE = 24
VISION_FEATURE_DIM = 4
DEFAULT_D_MODEL = 48
DEFAULT_SEED = 123

POSITION_ANCHORS: dict[str, tuple[int, int]] = {
    "center": (12, 12),
    "left": (0, 12),
    "right": (24, 12),
    "top": (12, 0),
    "bottom": (12, 24),
}
POSITIONS = tuple(POSITION_ANCHORS)
STYLES = ("solid", "muted")
TRAIN_STYLE = "solid"
HELDOUT_STYLE = "muted"

COLOR_RGB: dict[str, tuple[float, float, float]] = {
    "red": (0.92, 0.08, 0.07),
    "blue": (0.08, 0.20, 0.95),
    "green": (0.06, 0.66, 0.22),
    "yellow": (0.98, 0.80, 0.06),
}


@dataclass(frozen=True)
class VQAExample:
    image_id: str
    color: str
    shape: str
    style: str
    position: str
    bbox: tuple[int, int, int, int]
    question_type: str
    question: str
    answer: str
    counterfactual_answer: str
    split: str


@dataclass(frozen=True)
class VQABatch:
    examples: tuple[VQAExample, ...]
    images: t.Tensor
    question_ids: t.Tensor
    labels: t.Tensor
    counterfactual_labels: t.Tensor


@dataclass
class MiniVLMTrainingResult:
    model: "MiniVLM"
    encoder: "FrozenPatchEncoder"
    train_batch: VQABatch
    heldout_batch: VQABatch
    train_cache: t.Tensor
    heldout_cache: t.Tensor
    losses: tuple[float, ...]
    train_accuracy: float
    heldout_accuracy: float


def _next_value(value: str, values: tuple[str, ...]) -> str:
    return values[(values.index(value) + 1) % len(values)]


def render_controlled_scene(
    color: str,
    shape: str,
    position: str,
    *,
    style: str = TRAIN_STYLE,
    image_size: int = IMAGE_SIZE,
    object_size: int = OBJECT_SIZE,
) -> tuple[t.Tensor, tuple[int, int, int, int]]:
    """Render one controlled VQA image and return ``(image, bbox)``.

    The image is a real tensor, not a symbolic fixture. Shape evidence is carried
    by which frozen visual patch tokens contain non-white pixels.
    """

    if color not in COLOR_RGB:
        raise ValueError(f"unknown color: {color!r}")
    if shape not in SHAPES:
        raise ValueError(f"unknown shape: {shape!r}")
    if position not in POSITION_ANCHORS:
        raise ValueError(f"unknown position: {position!r}")
    if style not in STYLES:
        raise ValueError(f"unknown style: {style!r}")
    if object_size <= 0 or object_size > image_size:
        raise ValueError("object_size must be positive and fit inside image_size.")

    x1, y1 = POSITION_ANCHORS[position]
    x2, y2 = x1 + object_size, y1 + object_size
    if x2 > image_size or y2 > image_size:
        raise ValueError("position anchor and object_size must fit in the image.")

    coords = t.arange(image_size, dtype=t.float32) + 0.5
    yy, xx = t.meshgrid(coords, coords, indexing="ij")
    in_box = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    if shape == "square":
        mask = in_box
    elif shape == "circle":
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        radius = object_size / 2
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2) & in_box
    elif shape == "triangle":
        cx = (x1 + x2) / 2
        local_y = ((yy - y1) / object_size).clamp(0, 1)
        half_width = (object_size / 2) * local_y
        mask = in_box & ((xx - cx).abs() <= half_width)
    else:  # pragma: no cover - guarded above, retained for type checkers
        raise ValueError(f"unsupported shape: {shape!r}")

    image = t.ones(3, image_size, image_size, dtype=t.float32)
    rgb = t.tensor(COLOR_RGB[color], dtype=t.float32)
    if style == HELDOUT_STYLE:
        rgb = 0.82 * rgb + 0.18
    image[:, mask] = rgb[:, None]
    return image, (x1, y1, x2, y2)


def build_vqa_batch(
    split: str,
    *,
    colors: tuple[str, ...] = COLORS,
    shapes: tuple[str, ...] = SHAPES,
) -> VQABatch:
    """Build the disjoint solid-train or muted-style-heldout split."""

    if split not in {"train", "heldout"}:
        raise ValueError("split must be 'train' or 'heldout'.")

    examples: list[VQAExample] = []
    images: list[t.Tensor] = []
    question_ids: list[int] = []
    labels: list[int] = []
    counterfactual_labels: list[int] = []

    style = TRAIN_STYLE if split == "train" else HELDOUT_STYLE
    for position in POSITIONS:
        for color in colors:
            for shape in shapes:
                image, bbox = render_controlled_scene(
                    color,
                    shape,
                    position,
                    style=style,
                )
                for question_type in QUESTION_TYPES:
                    answer = color if question_type == "color" else shape
                    counterfactual = (
                        _next_value(color, colors)
                        if question_type == "color"
                        else _next_value(shape, shapes)
                    )
                    examples.append(
                        VQAExample(
                            image_id=f"{split}_{style}_{position}_{color}_{shape}_{question_type}",
                            color=color,
                            shape=shape,
                            style=style,
                            position=position,
                            bbox=bbox,
                            question_type=question_type,
                            question=(
                                "What color is the object?"
                                if question_type == "color"
                                else "What shape is the object?"
                            ),
                            answer=answer,
                            counterfactual_answer=counterfactual,
                            split=split,
                        )
                    )
                    images.append(image)
                    question_ids.append(QUESTION_TO_ID[question_type])
                    labels.append(ANSWER_TO_ID[answer])
                    counterfactual_labels.append(ANSWER_TO_ID[counterfactual])

    return VQABatch(
        examples=tuple(examples),
        images=t.stack(images),
        question_ids=t.tensor(question_ids, dtype=t.long),
        labels=t.tensor(labels, dtype=t.long),
        counterfactual_labels=t.tensor(counterfactual_labels, dtype=t.long),
    )


class FrozenPatchEncoder(nn.Module):
    """Frozen pixel-patch encoder used as the mini VLM vision tower."""

    def __init__(self, *, grid_size: int = PATCH_GRID):
        super().__init__()
        self.grid_size = grid_size

    @property
    def num_tokens(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def feature_dim(self) -> int:
        return VISION_FEATURE_DIM

    def forward(self, images: t.Tensor) -> t.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width).")
        batch, channels, height, width = images.shape
        if height != width or height % self.grid_size != 0:
            raise ValueError("image height/width must be square and divisible by grid_size.")
        patch = height // self.grid_size
        patches = images.float().reshape(
            batch,
            channels,
            self.grid_size,
            patch,
            self.grid_size,
            patch,
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        mean_rgb = patches.mean(dim=(-1, -2))
        nonwhite = (patches < 0.98).any(dim=3).float().mean(dim=(-1, -2))
        tokens = t.cat(
            [
                mean_rgb,
                nonwhite.unsqueeze(-1),
            ],
            dim=-1,
        )
        return tokens.reshape(batch, self.num_tokens, self.feature_dim)


def encode_visual_token_cache(
    encoder: FrozenPatchEncoder,
    images: t.Tensor,
    *,
    device: str | t.device = "cpu",
) -> t.Tensor:
    """Encode images once and detach the visual tokens for fast VLM training."""

    target_device = t.device(device)
    encoder = encoder.to(target_device)
    with t.no_grad():
        cache = encoder(images.to(target_device))
    return cache.detach()


def patch_indices_from_bbox(
    bbox: tuple[int, int, int, int],
    *,
    image_size: int = IMAGE_SIZE,
    grid_size: int = PATCH_GRID,
) -> tuple[int, ...]:
    """Return patch-token indices whose image patches overlap ``bbox``."""

    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= image_size and 0 <= y1 < y2 <= image_size):
        raise ValueError("bbox must lie inside the image with positive area.")
    if image_size % grid_size != 0:
        raise ValueError("image_size must be divisible by grid_size.")
    patch = image_size // grid_size
    indices: list[int] = []
    for row in range(grid_size):
        patch_y1 = row * patch
        patch_y2 = patch_y1 + patch
        for col in range(grid_size):
            patch_x1 = col * patch
            patch_x2 = patch_x1 + patch
            overlaps = max(x1, patch_x1) < min(x2, patch_x2) and max(y1, patch_y1) < min(
                y2, patch_y2
            )
            if overlaps:
                indices.append(row * grid_size + col)
    if not indices:
        raise ValueError("bbox does not overlap any patch token.")
    return tuple(indices)


def same_size_random_region_indices(
    object_indices: tuple[int, ...] | list[int],
    *,
    num_tokens: int,
    seed: int = DEFAULT_SEED,
) -> tuple[int, ...]:
    """Choose a same-size non-object visual-token control region."""

    object_set = set(int(index) for index in object_indices)
    if not object_set:
        raise ValueError("object_indices must be nonempty.")
    if min(object_set) < 0 or max(object_set) >= num_tokens:
        raise ValueError("object index out of range.")
    available = [index for index in range(num_tokens) if index not in object_set]
    if len(available) < len(object_set):
        raise ValueError("not enough non-object tokens for a same-size control.")
    generator = t.Generator(device="cpu").manual_seed(seed)
    order = t.randperm(len(available), generator=generator).tolist()
    chosen = sorted(available[index] for index in order[: len(object_set)])
    return tuple(chosen)


def patch_visual_tokens(
    clean_cache: t.Tensor,
    corrupt_cache: t.Tensor,
    patch_indices: tuple[int, ...] | list[int],
) -> t.Tensor:
    """Return ``clean_cache`` with selected token rows replaced by ``corrupt_cache``."""

    if clean_cache.shape != corrupt_cache.shape:
        raise ValueError("clean_cache and corrupt_cache must have the same shape.")
    if clean_cache.ndim != 3:
        raise ValueError("visual caches must have shape (batch, tokens, features).")
    if len(patch_indices) == 0:
        raise ValueError("patch_indices must be nonempty.")
    index = t.tensor(tuple(int(i) for i in patch_indices), device=clean_cache.device)
    if index.min() < 0 or index.max() >= clean_cache.shape[1]:
        raise ValueError("patch index out of range.")
    if len(set(index.detach().cpu().tolist())) != index.numel():
        raise ValueError("patch_indices must be unique.")
    patched = clean_cache.clone()
    patched[:, index] = corrupt_cache[:, index].to(
        device=patched.device,
        dtype=patched.dtype,
    )
    return patched


def _replace_token_rows(
    values: t.Tensor,
    patch_indices: tuple[int, ...] | list[int],
    replacement_rows: t.Tensor,
) -> t.Tensor:
    index = t.tensor(tuple(int(i) for i in patch_indices), device=values.device)
    if replacement_rows.ndim != 3:
        raise ValueError("replacement_rows must have shape (batch, patched_tokens, dim).")
    if replacement_rows.shape[1] != index.numel():
        raise ValueError("replacement row count must match patch_indices length.")
    if replacement_rows.shape[0] == 1 and values.shape[0] != 1:
        replacement_rows = replacement_rows.expand(values.shape[0], -1, -1)
    if replacement_rows.shape[0] != values.shape[0]:
        raise ValueError("replacement batch size must be 1 or match values batch size.")
    patched = values.clone()
    patched[:, index] = replacement_rows.to(device=values.device, dtype=values.dtype)
    return patched


class VisualConnector(nn.Module):
    """Map frozen vision features into the decoder residual-stream width."""

    def __init__(self, vision_feature_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(vision_feature_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visual_cache: t.Tensor) -> t.Tensor:
        if visual_cache.ndim != 3:
            raise ValueError("visual_cache must have shape (batch, tokens, features).")
        return self.norm(self.proj(visual_cache.float()))


class CausalDecoderBlock(nn.Module):
    """One pre-norm causal self-attention block with an inspectable value stream."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        sequence: t.Tensor,
        *,
        replacement_values: t.Tensor | None = None,
        patch_indices: tuple[int, ...] = (),
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        batch, seq_len, _ = sequence.shape
        normalized = self.ln1(sequence)

        def split_heads(values: t.Tensor) -> t.Tensor:
            return values.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        queries = split_heads(self.q_proj(normalized))
        keys = split_heads(self.k_proj(normalized))
        values = self.v_proj(normalized)
        cached_values = values.detach()
        if replacement_values is not None:
            values = _replace_token_rows(values, patch_indices, replacement_values)
        values_by_head = split_heads(values)
        scores = queries @ keys.transpose(-1, -2) / self.head_dim**0.5
        causal_mask = t.triu(
            t.ones(seq_len, seq_len, dtype=t.bool, device=sequence.device),
            diagonal=1,
        )
        attention = scores.masked_fill(causal_mask, float("-inf")).softmax(dim=-1)
        context = (attention @ values_by_head).transpose(1, 2).reshape(batch, seq_len, self.d_model)
        sequence = sequence + self.out_proj(context)
        sequence = sequence + self.mlp(self.ln2(sequence))
        return sequence, cached_values, attention.detach()


class MiniVLM(nn.Module):
    """Tiny visual-prefix causal decoder which predicts from the final question token."""

    def __init__(
        self,
        *,
        vision_feature_dim: int = VISION_FEATURE_DIM,
        d_model: int = DEFAULT_D_MODEL,
        num_visual_tokens: int = PATCH_GRID * PATCH_GRID,
        num_layers: int = 2,
        num_heads: int = 4,
        num_answers: int = len(ANSWER_VOCAB),
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_visual_tokens = num_visual_tokens
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.visual_connector = VisualConnector(vision_feature_dim, d_model)
        self.question_embed = nn.Embedding(len(QUESTION_TYPES) + 1, d_model)
        self.decoder_blocks = nn.ModuleList(
            [CausalDecoderBlock(d_model, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.answer_head = nn.Linear(d_model, num_answers)
        with t.no_grad():
            self.question_embed.weight[UNKNOWN_QUESTION_ID].zero_()

    def forward(
        self,
        visual_cache: t.Tensor,
        question_ids: t.Tensor,
        *,
        ablate_question: bool = False,
        patch_spec: dict[str, object] | None = None,
        return_cache: bool = False,
    ) -> t.Tensor | tuple[t.Tensor, dict[str, t.Tensor]]:
        if visual_cache.ndim != 3:
            raise ValueError("visual_cache must have shape (batch, visual_tokens, features).")
        if visual_cache.shape[1] != self.num_visual_tokens:
            raise ValueError("visual_cache has the wrong number of visual tokens.")
        if question_ids.ndim != 1 or question_ids.shape[0] != visual_cache.shape[0]:
            raise ValueError("question_ids must have shape (batch,).")

        stage = None if patch_spec is None else str(patch_spec["stage"])
        patch_indices = () if patch_spec is None else tuple(patch_spec["indices"])  # type: ignore[arg-type]
        replacement = None if patch_spec is None else patch_spec["values"]  # type: ignore[index]

        if stage == "cache":
            visual_cache = _replace_token_rows(visual_cache, patch_indices, replacement)  # type: ignore[arg-type]

        visual_tokens = self.visual_connector(visual_cache)
        cache: dict[str, t.Tensor] = {}
        if return_cache:
            cache["cache"] = visual_cache.detach()
            cache["projected"] = visual_tokens.detach()
        if stage == "projected":
            visual_tokens = _replace_token_rows(visual_tokens, patch_indices, replacement)  # type: ignore[arg-type]

        question_tokens = self.question_embed(question_ids.to(visual_cache.device))
        if ablate_question:
            question_tokens = t.zeros_like(question_tokens)
        sequence = t.cat([visual_tokens, question_tokens.unsqueeze(1)], dim=1)
        for layer_index, block in enumerate(self.decoder_blocks):
            layer_stage = f"value_{layer_index}"
            sequence, values, attention = block(
                sequence,
                replacement_values=replacement if stage == layer_stage else None,  # type: ignore[arg-type]
                patch_indices=patch_indices,
            )
            if return_cache:
                cache[layer_stage] = values[:, : self.num_visual_tokens]
                cache[f"attention_{layer_index}"] = attention[:, :, -1, :].detach()

        answer_state = self.final_norm(sequence[:, -1])
        logits = self.answer_head(answer_state)
        if return_cache:
            return logits, cache
        return logits


def build_multimodal_sequence(
    model: MiniVLM,
    visual_cache: t.Tensor,
    question_ids: t.Tensor,
) -> t.Tensor:
    """Project visual tokens and append the question token."""

    visual_tokens = model.visual_connector(visual_cache)
    question_tokens = model.question_embed(question_ids.to(visual_tokens.device)).unsqueeze(1)
    return t.cat([visual_tokens, question_tokens], dim=1)


def vqa_accuracy(logits: t.Tensor, labels: t.Tensor) -> float:
    """Return exact-match VQA accuracy."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, answers).")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape (batch,).")
    return float((logits.argmax(dim=-1) == labels.to(logits.device)).float().mean().item())


def answer_margin(
    logits: t.Tensor,
    target_labels: t.Tensor,
    counterfactual_labels: t.Tensor,
) -> t.Tensor:
    """Return target-answer logit minus counterfactual-answer logit."""

    if target_labels.shape != counterfactual_labels.shape:
        raise ValueError("target and counterfactual labels must have matching shapes.")
    target = logits.gather(1, target_labels.to(logits.device).reshape(-1, 1)).squeeze(1)
    counter = logits.gather(
        1,
        counterfactual_labels.to(logits.device).reshape(-1, 1),
    ).squeeze(1)
    return target - counter


def mini_vlm_loss(
    model: MiniVLM,
    visual_cache: t.Tensor,
    question_ids: t.Tensor,
    labels: t.Tensor,
) -> t.Tensor:
    """Return next-answer cross entropy from the final causal question position."""

    logits = model(visual_cache, question_ids)
    if not isinstance(logits, t.Tensor):
        raise TypeError("MiniVLM must return logits when return_cache=False.")
    return F.cross_entropy(logits, labels.to(logits.device))


def toy_ground_truth_patch_report(
    *,
    grid_size: int = PATCH_GRID,
    target_answer: str = "red",
    counterfactual_answer: str = "blue",
) -> dict[str, object]:
    """Exact oracle showing why object patching should flip this toy game."""

    num_tokens = grid_size * grid_size
    object_indices = patch_indices_from_bbox(
        POSITION_ANCHORS["center"]
        + (
            POSITION_ANCHORS["center"][0] + OBJECT_SIZE,
            POSITION_ANCHORS["center"][1] + OBJECT_SIZE,
        ),
        grid_size=grid_size,
    )
    background_indices = same_size_random_region_indices(
        object_indices,
        num_tokens=num_tokens,
        seed=0,
    )
    random_indices = same_size_random_region_indices(
        object_indices,
        num_tokens=num_tokens,
        seed=1,
    )
    target = ANSWER_TO_ID[target_answer]
    counter = ANSWER_TO_ID[counterfactual_answer]
    clean = t.zeros(1, num_tokens, len(ANSWER_VOCAB))
    corrupt = t.zeros_like(clean)
    clean[:, object_indices, target] = 1.0
    clean[:, object_indices, counter] = -0.25
    corrupt[:, object_indices, counter] = 1.0
    corrupt[:, object_indices, target] = -0.25

    def margin(cache: t.Tensor) -> float:
        logits = cache.sum(dim=1)
        return float((logits[:, target] - logits[:, counter]).item())

    object_patched = patch_visual_tokens(clean, corrupt, object_indices)
    background_patched = patch_visual_tokens(clean, corrupt, background_indices)
    random_patched = patch_visual_tokens(clean, corrupt, random_indices)
    return {
        "claim_scope": "exact_toy_visual_token_contribution_game",
        "object_indices": object_indices,
        "background_indices": background_indices,
        "random_indices": random_indices,
        "clean_margin": margin(clean),
        "corrupt_margin": margin(corrupt),
        "object_patch_margin": margin(object_patched),
        "background_patch_margin": margin(background_patched),
        "random_patch_margin": margin(random_patched),
        "object_patch_flips": margin(object_patched) < 0,
        "background_patch_preserves": margin(background_patched) > 0,
        "random_patch_preserves": margin(random_patched) > 0,
        "object_beats_background": margin(background_patched) - margin(object_patched) > 5.0,
    }


def train_mini_vlm(
    *,
    steps: int = 320,
    lr: float = 3e-3,
    seed: int = DEFAULT_SEED,
    device: str | t.device = "cpu",
) -> MiniVLMTrainingResult:
    """Train on solid scenes and evaluate on the disjoint muted style."""

    target_device = t.device(device)
    if target_device.type == "cpu" and t.get_num_threads() > 4:
        t.set_num_threads(4)
    t.manual_seed(seed)
    encoder = FrozenPatchEncoder().to(target_device)
    train_batch = build_vqa_batch("train")
    heldout_batch = build_vqa_batch("heldout")
    train_cache = encode_visual_token_cache(encoder, train_batch.images, device=target_device)
    heldout_cache = encode_visual_token_cache(encoder, heldout_batch.images, device=target_device)
    train_questions = train_batch.question_ids.to(target_device)
    train_labels = train_batch.labels.to(target_device)
    heldout_questions = heldout_batch.question_ids.to(target_device)
    heldout_labels = heldout_batch.labels.to(target_device)

    model = MiniVLM().to(target_device)
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []
    for step in range(steps):
        loss = mini_vlm_loss(model, train_cache, train_questions, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 80) == 0 or step == steps - 1:
            losses.append(float(loss.detach().cpu().item()))

    model.eval()
    with t.no_grad():
        train_logits = model(train_cache, train_questions)
        heldout_logits = model(heldout_cache, heldout_questions)
    assert isinstance(train_logits, t.Tensor)
    assert isinstance(heldout_logits, t.Tensor)
    return MiniVLMTrainingResult(
        model=model,
        encoder=encoder,
        train_batch=train_batch,
        heldout_batch=heldout_batch,
        train_cache=train_cache.detach(),
        heldout_cache=heldout_cache.detach(),
        losses=tuple(losses),
        train_accuracy=vqa_accuracy(train_logits, train_labels),
        heldout_accuracy=vqa_accuracy(heldout_logits, heldout_labels),
    )


def modality_baseline_report(training: MiniVLMTrainingResult) -> dict[str, float]:
    """Compare joint VQA to text-only, image-only, and random-visual baselines."""

    model = training.model
    cache = training.heldout_cache
    question_ids = training.heldout_batch.question_ids.to(cache.device)
    labels = training.heldout_batch.labels.to(cache.device)
    generator = t.Generator(device="cpu").manual_seed(DEFAULT_SEED)
    permutation = t.randperm(cache.shape[0], generator=generator).to(cache.device)
    with t.no_grad():
        joint = model(cache, question_ids)
        text_only = model(t.zeros_like(cache), question_ids)
        image_only = model(cache, question_ids, ablate_question=True)
        random_visual = model(cache[permutation], question_ids)
    assert isinstance(joint, t.Tensor)
    assert isinstance(text_only, t.Tensor)
    assert isinstance(image_only, t.Tensor)
    assert isinstance(random_visual, t.Tensor)
    return {
        "joint_accuracy": vqa_accuracy(joint, labels),
        "text_only_accuracy": vqa_accuracy(text_only, labels),
        "image_only_accuracy": vqa_accuracy(image_only, labels),
        "random_visual_accuracy": vqa_accuracy(random_visual, labels),
    }


def build_single_example(
    *,
    color: str,
    shape: str,
    position: str,
    question_type: str,
    style: str = HELDOUT_STYLE,
    split: str = "patch",
) -> VQABatch:
    """Build a one-example VQA batch for intervention experiments."""

    image, bbox = render_controlled_scene(color, shape, position, style=style)
    answer = color if question_type == "color" else shape
    counterfactual = (
        _next_value(color, COLORS) if question_type == "color" else _next_value(shape, SHAPES)
    )
    example = VQAExample(
        image_id=f"{split}_{style}_{position}_{color}_{shape}_{question_type}",
        color=color,
        shape=shape,
        style=style,
        position=position,
        bbox=bbox,
        question_type=question_type,
        question="What color is the object?"
        if question_type == "color"
        else "What shape is the object?",
        answer=answer,
        counterfactual_answer=counterfactual,
        split=split,
    )
    return VQABatch(
        examples=(example,),
        images=image.unsqueeze(0),
        question_ids=t.tensor([QUESTION_TO_ID[question_type]], dtype=t.long),
        labels=t.tensor([ANSWER_TO_ID[answer]], dtype=t.long),
        counterfactual_labels=t.tensor([ANSWER_TO_ID[counterfactual]], dtype=t.long),
    )


def _patch_replacement_for_stage(
    model: MiniVLM,
    corrupt_cache: t.Tensor,
    question_ids: t.Tensor,
    patch_indices: tuple[int, ...],
    stage: str,
) -> t.Tensor:
    if stage == "cache":
        return corrupt_cache[:, patch_indices]
    with t.no_grad():
        _, caches = model(corrupt_cache, question_ids, return_cache=True)
    return caches[stage][:, patch_indices]


def forward_with_visual_patch(
    model: MiniVLM,
    clean_cache: t.Tensor,
    corrupt_cache: t.Tensor,
    question_ids: t.Tensor,
    patch_indices: tuple[int, ...],
    *,
    stage: str = "cache",
) -> t.Tensor:
    """Run clean example while replacing selected clean activations by corrupt ones."""

    replacement = _patch_replacement_for_stage(
        model,
        corrupt_cache,
        question_ids,
        patch_indices,
        stage,
    )
    logits = model(
        clean_cache,
        question_ids,
        patch_spec={"stage": stage, "indices": patch_indices, "values": replacement},
    )
    assert isinstance(logits, t.Tensor)
    return logits


def forward_with_random_activation_patch(
    model: MiniVLM,
    clean_cache: t.Tensor,
    question_ids: t.Tensor,
    patch_indices: tuple[int, ...],
    *,
    seed: int = DEFAULT_SEED,
) -> t.Tensor:
    """Patch object rows with same-scale random cache activations."""

    generator = t.Generator(device="cpu").manual_seed(seed)
    mean = clean_cache.detach().cpu().mean()
    std = clean_cache.detach().cpu().std().clamp_min(1e-3)
    replacement = mean + std * t.randn(
        clean_cache.shape[0],
        len(patch_indices),
        clean_cache.shape[-1],
        generator=generator,
    )
    logits = model(
        clean_cache,
        question_ids,
        patch_spec={
            "stage": "cache",
            "indices": patch_indices,
            "values": replacement.to(clean_cache.device),
        },
    )
    assert isinstance(logits, t.Tensor)
    return logits


def _margin_float(logits: t.Tensor, target: int, counterfactual: int) -> float:
    return float((logits[:, target] - logits[:, counterfactual]).detach().cpu().item())


def trained_patch_report(
    training: MiniVLMTrainingResult,
    *,
    clean_color: str = "red",
    corrupt_color: str = "blue",
    clean_shape: str = "square",
    corrupt_shape: str = "square",
    position: str = "top",
    question_type: str = "color",
    stage: str = "cache",
) -> dict[str, object]:
    """Run object/background/random patching for one controlled VQA pair."""

    model = training.model
    encoder = training.encoder
    clean = build_single_example(
        color=clean_color,
        shape=clean_shape,
        position=position,
        question_type=question_type,
    )
    corrupt = build_single_example(
        color=corrupt_color,
        shape=corrupt_shape,
        position=position,
        question_type=question_type,
    )
    device = next(model.parameters()).device
    clean_cache = encode_visual_token_cache(encoder, clean.images, device=device)
    corrupt_cache = encode_visual_token_cache(encoder, corrupt.images, device=device)
    question_ids = clean.question_ids.to(device)
    target = int(clean.labels.item())
    counterfactual = int(corrupt.labels.item())
    object_indices = patch_indices_from_bbox(clean.examples[0].bbox)
    background_indices = same_size_random_region_indices(
        object_indices,
        num_tokens=encoder.num_tokens,
        seed=0,
    )
    random_indices = same_size_random_region_indices(
        object_indices,
        num_tokens=encoder.num_tokens,
        seed=1,
    )
    full_indices = tuple(range(encoder.num_tokens))

    with t.no_grad():
        clean_logits = model(clean_cache, question_ids)
        corrupt_logits = model(corrupt_cache, question_ids)
        object_logits = forward_with_visual_patch(
            model,
            clean_cache,
            corrupt_cache,
            question_ids,
            object_indices,
            stage=stage,
        )
        background_logits = forward_with_visual_patch(
            model,
            clean_cache,
            corrupt_cache,
            question_ids,
            background_indices,
            stage=stage,
        )
        random_region_logits = forward_with_visual_patch(
            model,
            clean_cache,
            corrupt_cache,
            question_ids,
            random_indices,
            stage=stage,
        )
        full_logits = forward_with_visual_patch(
            model,
            clean_cache,
            corrupt_cache,
            question_ids,
            full_indices,
            stage=stage,
        )
    assert isinstance(clean_logits, t.Tensor)
    assert isinstance(corrupt_logits, t.Tensor)

    rows = {
        "clean": clean_logits,
        "corrupt": corrupt_logits,
        "full_sequence_patch": full_logits,
        "object_patch": object_logits,
        "background_patch": background_logits,
        "same_size_random_region_patch": random_region_logits,
    }
    margin_rows = [
        {
            "condition": condition,
            "target_minus_counterfactual": _margin_float(logits, target, counterfactual),
            "predicted_answer": ANSWER_VOCAB[int(logits.argmax(dim=-1).item())],
        }
        for condition, logits in rows.items()
    ]
    margin_by_condition = {
        row["condition"]: float(row["target_minus_counterfactual"])
        for row in margin_rows
    }
    return {
        "stage": stage,
        "question_type": question_type,
        "clean_example": asdict(clean.examples[0]),
        "corrupt_example": asdict(corrupt.examples[0]),
        "target_answer": clean.examples[0].answer,
        "counterfactual_answer": corrupt.examples[0].answer,
        "object_indices": object_indices,
        "background_indices": background_indices,
        "random_indices": random_indices,
        "rows": margin_rows,
        "clean_margin": margin_by_condition["clean"],
        "corrupt_margin": margin_by_condition["corrupt"],
        "object_patch_margin": margin_by_condition["object_patch"],
        "background_patch_margin": margin_by_condition["background_patch"],
        "random_region_patch_margin": margin_by_condition["same_size_random_region_patch"],
        "full_sequence_patch_margin": margin_by_condition["full_sequence_patch"],
        "object_patch_flips": margin_by_condition["object_patch"] < 0,
        "background_patch_preserves": margin_by_condition["background_patch"] > 0,
        "random_region_preserves": margin_by_condition["same_size_random_region_patch"] > 0,
        "full_sequence_matches_corrupt": abs(
            margin_by_condition["full_sequence_patch"] - margin_by_condition["corrupt"]
        )
        < 1e-4,
    }


def patching_effect_heatmap(
    training: MiniVLMTrainingResult,
    *,
    stages: tuple[str, ...] = ("cache", "projected", "value_0", "value_1"),
) -> dict[str, object]:
    """Patch each visual position and measure the target-margin drop."""

    base = trained_patch_report(training, stage="cache")
    clean = build_single_example(
        color=base["clean_example"]["color"],  # type: ignore[index]
        shape=base["clean_example"]["shape"],  # type: ignore[index]
        position=base["clean_example"]["position"],  # type: ignore[index]
        question_type=base["question_type"],  # type: ignore[arg-type]
    )
    corrupt = build_single_example(
        color=base["corrupt_example"]["color"],  # type: ignore[index]
        shape=base["corrupt_example"]["shape"],  # type: ignore[index]
        position=base["corrupt_example"]["position"],  # type: ignore[index]
        question_type=base["question_type"],  # type: ignore[arg-type]
    )
    model = training.model
    encoder = training.encoder
    device = next(model.parameters()).device
    clean_cache = encode_visual_token_cache(encoder, clean.images, device=device)
    corrupt_cache = encode_visual_token_cache(encoder, corrupt.images, device=device)
    question_ids = clean.question_ids.to(device)
    target = int(clean.labels.item())
    counterfactual = int(corrupt.labels.item())
    with t.no_grad():
        clean_logits = model(clean_cache, question_ids)
    assert isinstance(clean_logits, t.Tensor)
    clean_margin = _margin_float(clean_logits, target, counterfactual)

    heatmaps: dict[str, list[list[float]]] = {}
    for stage in stages:
        drops = []
        for token_index in range(encoder.num_tokens):
            with t.no_grad():
                patched_logits = forward_with_visual_patch(
                    model,
                    clean_cache,
                    corrupt_cache,
                    question_ids,
                    (token_index,),
                    stage=stage,
                )
            patched_margin = _margin_float(patched_logits, target, counterfactual)
            drops.append(clean_margin - patched_margin)
        grid = t.tensor(drops).reshape(PATCH_GRID, PATCH_GRID)
        heatmaps[stage] = grid.detach().cpu().tolist()
    return {
        "stages": stages,
        "clean_margin": clean_margin,
        "object_indices": base["object_indices"],
        "heatmaps": heatmaps,
    }


def build_signature_result(
    training: MiniVLMTrainingResult,
    *,
    save_assets: bool = False,
    assets_dir: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate one trained model and optionally save the visible figures."""

    baselines = modality_baseline_report(training)
    color_patch = trained_patch_report(training, stage="cache")
    shape_patch = trained_patch_report(
        training,
        clean_color="green",
        corrupt_color="green",
        clean_shape="square",
        corrupt_shape="circle",
        position="bottom",
        question_type="shape",
        stage="cache",
    )
    heatmap = patching_effect_heatmap(training)
    toy_oracle = toy_ground_truth_patch_report()
    result: dict[str, object] = {
        "claim_scope": "trained_mini_vlm_heldout_visual_style_and_visual_token_patching",
        "train_style": TRAIN_STYLE,
        "heldout_style": HELDOUT_STYLE,
        "train_examples": len(training.train_batch.examples),
        "heldout_examples": len(training.heldout_batch.examples),
        "loss_start": training.losses[0],
        "loss_end": training.losses[-1],
        "loss_curve": list(training.losses),
        "train_accuracy": training.train_accuracy,
        "heldout_accuracy": training.heldout_accuracy,
        "baselines": baselines,
        "toy_oracle": toy_oracle,
        "color_patch": color_patch,
        "shape_patch": shape_patch,
        "patching_heatmap": heatmap,
        "accepted": (
            training.heldout_accuracy >= 0.95
            and baselines["joint_accuracy"] >= 0.95
            and baselines["text_only_accuracy"] <= 0.45
            and baselines["image_only_accuracy"] <= 0.60
            and baselines["random_visual_accuracy"] <= 0.50
            and bool(color_patch["object_patch_flips"])
            and bool(shape_patch["object_patch_flips"])
            and bool(color_patch["background_patch_preserves"])
            and bool(color_patch["random_region_preserves"])
            and bool(toy_oracle["object_patch_flips"])
        ),
    }
    if save_assets:
        if assets_dir is None:
            raise ValueError("assets_dir is required when save_assets=True.")
        saved = save_signature_assets(result, Path(assets_dir))
        result["saved_assets"] = [str(path) for path in saved]
    return result


def run_signature_result(
    *,
    steps: int = 320,
    seed: int = DEFAULT_SEED,
    device: str | t.device = "cpu",
    save_assets: bool = False,
    assets_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train once, evaluate controls, and optionally save the visible figures."""

    training = train_mini_vlm(steps=steps, seed=seed, device=device)
    return build_signature_result(
        training,
        save_assets=save_assets,
        assets_dir=assets_dir,
    )


def save_signature_assets(result: dict[str, object], assets_dir: Path) -> tuple[Path, Path]:
    """Save the learner-facing signature result figures."""

    import matplotlib.pyplot as plt

    assets_dir.mkdir(parents=True, exist_ok=True)
    signature_path = assets_dir / "mini_vlm_signature_result.png"
    heatmap_path = assets_dir / "mini_vlm_layer_position_patching_heatmap.png"

    color_patch = result["color_patch"]  # type: ignore[assignment]
    baselines = result["baselines"]  # type: ignore[assignment]
    heatmap = result["patching_heatmap"]  # type: ignore[assignment]

    clean = color_patch["clean_example"]  # type: ignore[index]
    corrupt = color_patch["corrupt_example"]  # type: ignore[index]
    clean_image, _ = render_controlled_scene(
        clean["color"],  # type: ignore[index]
        clean["shape"],  # type: ignore[index]
        clean["position"],  # type: ignore[index]
        style=clean["style"],  # type: ignore[index]
    )
    corrupt_image, _ = render_controlled_scene(
        corrupt["color"],  # type: ignore[index]
        corrupt["shape"],  # type: ignore[index]
        corrupt["position"],  # type: ignore[index]
        style=corrupt["style"],  # type: ignore[index]
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].imshow(clean_image.permute(1, 2, 0).numpy())
    axes[0, 0].set_title(f"Held-out clean\n{clean['style']} {clean['answer']} {clean['shape']}")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(corrupt_image.permute(1, 2, 0).numpy())
    axes[0, 1].set_title(
        f"Held-out counterfactual\n{corrupt['style']} {corrupt['answer']} {corrupt['shape']}"
    )
    axes[0, 1].axis("off")
    axes[0, 2].plot(result["loss_curve"])  # type: ignore[arg-type]
    axes[0, 2].set_title("Training loss")
    axes[0, 2].set_xlabel("logged step")
    axes[0, 2].set_ylabel("cross entropy")

    baseline_names = ["joint", "text-only", "image-only", "random visual"]
    baseline_values = [
        baselines["joint_accuracy"],  # type: ignore[index]
        baselines["text_only_accuracy"],  # type: ignore[index]
        baselines["image_only_accuracy"],  # type: ignore[index]
        baselines["random_visual_accuracy"],  # type: ignore[index]
    ]
    axes[1, 0].bar(baseline_names, baseline_values, color=["#268bd2", "#b58900", "#859900", "#dc322f"])
    axes[1, 0].axhline(
        1 / len(ANSWER_VOCAB),
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="uniform chance",
    )
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("Held-out muted-style VQA")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].legend(frameon=False, loc="upper right")

    rows = color_patch["rows"]  # type: ignore[index]
    patch_labels = [
        {
            "clean": "clean",
            "corrupt": "counterfactual",
            "full_sequence_patch": "all visual tokens",
            "object_patch": "object tokens",
            "background_patch": "background tokens",
            "same_size_random_region_patch": "random region",
        }[row["condition"]]
        for row in rows
    ]
    patch_margins = [row["target_minus_counterfactual"] for row in rows]  # type: ignore[index]
    colors = ["#2aa198" if value > 0 else "#d33682" for value in patch_margins]
    axes[1, 1].barh(patch_labels, patch_margins, color=colors)
    axes[1, 1].axvline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Causal answer margin\nred minus blue")
    axes[1, 1].set_xlabel("target minus counterfactual logit")
    axes[1, 1].invert_yaxis()

    cache_heatmap = t.tensor(heatmap["heatmaps"]["cache"])  # type: ignore[index]
    im = axes[1, 2].imshow(cache_heatmap, cmap="magma")
    axes[1, 2].set_title("Single-token patch effect")
    axes[1, 2].set_xticks(range(PATCH_GRID))
    axes[1, 2].set_yticks(range(PATCH_GRID))
    object_rows = [index // PATCH_GRID for index in color_patch["object_indices"]]  # type: ignore[index]
    object_cols = [index % PATCH_GRID for index in color_patch["object_indices"]]  # type: ignore[index]
    from matplotlib.patches import Rectangle

    axes[1, 2].add_patch(
        Rectangle(
            (min(object_cols) - 0.5, min(object_rows) - 0.5),
            max(object_cols) - min(object_cols) + 1,
            max(object_rows) - min(object_rows) + 1,
            fill=False,
            edgecolor="#00ffff",
            linewidth=2,
        )
    )
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle("Mini VLM signature: visual tokens solve VQA and object patches flip answers")
    fig.tight_layout()
    fig.savefig(signature_path, dpi=180)
    plt.close(fig)

    stages = heatmap["stages"]  # type: ignore[index]
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 3.5))
    if len(stages) == 1:
        axes = [axes]
    vmax = max(
        abs(float(t.tensor(heatmap["heatmaps"][stage]).abs().max()))  # type: ignore[index]
        for stage in stages
    )
    for ax, stage in zip(axes, stages):
        grid = t.tensor(heatmap["heatmaps"][stage])  # type: ignore[index]
        im = ax.imshow(grid, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(stage)
        ax.set_xticks(range(PATCH_GRID))
        ax.set_yticks(range(PATCH_GRID))
        ax.add_patch(
            Rectangle(
                (min(object_cols) - 0.5, min(object_rows) - 0.5),
                max(object_cols) - min(object_cols) + 1,
                max(object_rows) - min(object_rows) + 1,
                fill=False,
                edgecolor="#00a6a6",
                linewidth=2,
            )
        )
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    fig.suptitle(
        "Layer/position patching: counterfactual margin drop (object box outlined)"
    )
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return signature_path, heatmap_path


def run_smoke_test(cpu: bool = True) -> dict[str, object]:
    """Compact contract used by tests and the notebook verification appendix."""

    device = "cpu" if cpu else "cuda"
    result = run_signature_result(steps=180, device=device)
    return {
        "accepted": result["accepted"],
        "heldout_accuracy": result["heldout_accuracy"],
        "baselines": result["baselines"],
        "toy_oracle": result["toy_oracle"],
        "color_patch": result["color_patch"],
        "shape_patch": result["shape_patch"],
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, object]:
    """Train and causally validate the MiniVLM on the release CUDA device."""

    if not t.cuda.is_available():
        raise RuntimeError("12.3 release verification requires CUDA.")
    t.cuda.reset_peak_memory_stats()
    result = run_signature_result(steps=260, device="cuda")
    baselines = result["baselines"]
    color_patch = result["color_patch"]
    shape_patch = result["shape_patch"]
    toy_oracle = result["toy_oracle"]
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "signature_accepted": result["accepted"],
        "train_example_count": result["train_examples"],
        "heldout_example_count": result["heldout_examples"],
        "train_accuracy": result["train_accuracy"],
        "heldout_accuracy": result["heldout_accuracy"],
        "joint_accuracy": baselines["joint_accuracy"],
        "text_only_accuracy": baselines["text_only_accuracy"],
        "image_only_accuracy": baselines["image_only_accuracy"],
        "random_visual_accuracy": baselines["random_visual_accuracy"],
        "color_object_patch_flips": color_patch["object_patch_flips"],
        "color_background_patch_preserves": color_patch["background_patch_preserves"],
        "color_random_region_preserves": color_patch["random_region_preserves"],
        "color_full_sequence_matches_corrupt": color_patch["full_sequence_matches_corrupt"],
        "color_clean_margin": color_patch["clean_margin"],
        "color_object_patch_margin": color_patch["object_patch_margin"],
        "shape_object_patch_flips": shape_patch["object_patch_flips"],
        "shape_background_patch_preserves": shape_patch["background_patch_preserves"],
        "shape_random_region_preserves": shape_patch["random_region_preserves"],
        "shape_full_sequence_matches_corrupt": shape_patch["full_sequence_matches_corrupt"],
        "shape_clean_margin": shape_patch["clean_margin"],
        "shape_object_patch_margin": shape_patch["object_patch_margin"],
        "toy_object_patch_flips": toy_oracle["object_patch_flips"],
        "toy_background_patch_preserves": toy_oracle["background_patch_preserves"],
        "toy_random_patch_preserves": toy_oracle["random_patch_preserves"],
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": "CUDA-trained MiniVLM with object/background/random visual-token patching.",
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, object]:
    """Run the complete release experiment rather than a reduced smoke path."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if __name__ == "__main__":
    section_dir = Path(__file__).resolve().parent
    assets = section_dir.parents[1] / "instructions" / "assets"
    report = run_signature_result(save_assets=True, assets_dir=assets)
    print(
        {
            "accepted": report["accepted"],
            "heldout_accuracy": report["heldout_accuracy"],
            "baselines": report["baselines"],
            "saved_assets": report.get("saved_assets", []),
        }
    )
