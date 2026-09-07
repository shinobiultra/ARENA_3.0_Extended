"""Visible reference bodies for the 12.1 pixel-and-token encoder exercises.

The notebook copies the taught functions/classes into its cells. Only image
rendering and deterministic dataset construction are provided as plumbing.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch as t
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn

PIXEL_COLORS = ("red", "blue", "green", "yellow")
PIXEL_SHAPES = ("square", "circle", "triangle")
PIXEL_RGB = ((230, 35, 30), (25, 65, 225), (25, 160, 55), (235, 200, 30))
PIXEL_CAPTIONS = tuple(f"a {color} {shape}" for color in PIXEL_COLORS for shape in PIXEL_SHAPES)
PIXEL_VOCAB = {
    word: i
    for i, word in enumerate(
        (
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
            "a",
            "the",
            "photo",
            "of",
            *PIXEL_COLORS,
            *PIXEL_SHAPES,
        )
    )
}


@dataclass(frozen=True)
class PixelDataset:
    images: t.Tensor
    labels: t.Tensor
    captions: tuple[str, ...]
    seed: int


def render_pixel_shape(class_id: int, *, generator: t.Generator, image_size: int = 32) -> t.Tensor:
    """Draw one RGB instance. Semantic metadata is used only by the renderer."""
    color, shape = divmod(class_id, len(PIXEL_SHAPES))
    background = int(t.randint(205, 256, (), generator=generator))
    radius = int(t.randint(6, 11, (), generator=generator))
    cx = int(t.randint(radius + 1, image_size - radius, (), generator=generator))
    cy = int(t.randint(radius + 1, image_size - radius, (), generator=generator))
    jitter = t.randint(-15, 16, (3,), generator=generator).tolist()
    rgb = tuple(
        max(0, min(255, channel + shift)) for channel, shift in zip(PIXEL_RGB[color], jitter)
    )
    image = Image.new("RGB", (image_size, image_size), (background,) * 3)
    draw = ImageDraw.Draw(image)
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    if shape == 0:
        draw.rectangle(box, fill=rgb)
    elif shape == 1:
        draw.ellipse(box, fill=rgb)
    else:
        draw.polygon(
            ((cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)), fill=rgb
        )
    pixels = t.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    # Independent sensor noise; zero byte overlap is not evidence of novel geometry.
    noise = 0.004 * t.randn(pixels.shape, generator=generator)
    return (pixels + noise).clamp(0, 1)


def make_pixel_dataset(*, examples_per_class: int, seed: int) -> PixelDataset:
    if examples_per_class < 1:
        raise ValueError("examples_per_class must be positive")
    generator = t.Generator().manual_seed(seed)
    labels = t.arange(len(PIXEL_CAPTIONS)).repeat_interleave(examples_per_class)
    images = t.stack([render_pixel_shape(int(label), generator=generator) for label in labels])
    return PixelDataset(images, labels, PIXEL_CAPTIONS, seed)


def dataset_overlap(train: PixelDataset, evaluation: PixelDataset) -> int:
    def keys(images: t.Tensor) -> set[str]:
        return {
            hashlib.sha256(image.contiguous().numpy().tobytes()).hexdigest() for image in images
        }

    return len(keys(train.images).intersection(keys(evaluation.images)))


def tokenize_pixel_captions(
    captions: Sequence[str], vocab: dict[str, int], *, context_length: int = 12
):
    """Right-pad BOS/word/EOS sequences and return an explicit boolean validity mask."""
    ids = t.full((len(captions), context_length), vocab["<pad>"], dtype=t.long)
    mask = t.zeros_like(ids, dtype=t.bool)
    for row, caption in enumerate(captions):
        tokens = [
            vocab["<bos>"],
            *(vocab.get(word, vocab["<unk>"]) for word in caption.lower().split()),
            vocab["<eos>"],
        ]
        if len(tokens) > context_length:
            raise ValueError("caption exceeds context_length; do not silently discard EOS")
        ids[row, : len(tokens)] = t.tensor(tokens)
        mask[row, : len(tokens)] = True
    return ids, mask


class PixelImageEncoder(nn.Module):
    """A complete small CNN: RGB pixels -> spatial features -> global representation."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.readout = nn.Linear(32 * 4 * 4, width)

    def forward(self, pixels: t.Tensor) -> t.Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, 32, 32):
            raise ValueError("image encoder expects RGB tensors of shape (batch, 3, 32, 32)")
        hidden = F.gelu(self.conv1(2 * pixels - 1))
        hidden = F.gelu(self.conv2(hidden))
        hidden = F.gelu(self.conv3(hidden))
        return self.readout(hidden.flatten(1))


class PixelTextBlock(nn.Module):
    """One visible pre-norm causal attention/MLP block, including padding exclusion."""

    def __init__(self, width: int = 32, heads: int = 4):
        super().__init__()
        if width % heads:
            raise ValueError("width must divide into complete attention heads")
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attention_out = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, width)

    def forward(self, hidden: t.Tensor, valid_mask: t.Tensor) -> t.Tensor:
        batch, length, width = hidden.shape
        if valid_mask.shape != (batch, length) or valid_mask.dtype != t.bool:
            raise ValueError("valid_mask must be a boolean batch-by-length matrix")
        if length == 0 or not bool(valid_mask[:, 0].all()):
            raise ValueError("every sequence needs a valid first token for causal attention")
        q, k, v = (
            self.qkv(self.norm1(hidden))
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .unbind(2)
        )
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        allowed = t.ones(length, length, dtype=t.bool, device=hidden.device).tril()
        allowed = allowed[None, None] & valid_mask[:, None, None, :]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(width // self.heads)
        attention = scores.masked_fill(~allowed, -t.inf).softmax(-1)
        mixed = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        hidden = hidden + self.attention_out(mixed)
        return hidden + self.fc2(F.gelu(self.fc1(self.norm2(hidden))))


class PixelTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, width: int = 32, context_length: int = 12, eos_id: int = 2):
        super().__init__()
        self.eos_id = eos_id
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = nn.Parameter(0.01 * t.randn(context_length, width))
        self.block = PixelTextBlock(width)
        self.final_norm = nn.LayerNorm(width)

    def forward(self, token_ids: t.Tensor, valid_mask: t.Tensor) -> t.Tensor:
        if token_ids.shape != valid_mask.shape or token_ids.ndim != 2:
            raise ValueError("token ids and validity mask must be equal-shaped matrices")
        if token_ids.shape[1] > len(self.position) or valid_mask.dtype != t.bool:
            raise ValueError("use a boolean mask and stay within the position context")
        eos = (token_ids == self.eos_id) & valid_mask
        if not bool((eos.sum(-1) == 1).all()):
            raise ValueError("each caption needs exactly one unmasked EOS")
        hidden = self.embedding(token_ids) + self.position[: token_ids.shape[1]]
        hidden = self.final_norm(self.block(hidden, valid_mask))
        positions = eos.long().argmax(-1)
        return hidden[t.arange(len(hidden), device=hidden.device), positions]


class PixelDualEncoder(nn.Module):
    def __init__(self, vocab_size: int, width: int = 32, embedding_dim: int = 16):
        super().__init__()
        self.image_encoder = PixelImageEncoder(width)
        self.text_encoder = PixelTextEncoder(vocab_size, width)
        self.image_projection = nn.Linear(width, embedding_dim, bias=False)
        self.text_projection = nn.Linear(width, embedding_dim, bias=False)
        self.logit_scale = nn.Parameter(t.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(t.tensor(-math.log(11.0)))

    def encode_images(self, pixels: t.Tensor) -> t.Tensor:
        return F.normalize(self.image_projection(self.image_encoder(pixels)), dim=-1)

    def encode_texts(self, token_ids: t.Tensor, valid_mask: t.Tensor) -> t.Tensor:
        return F.normalize(self.text_projection(self.text_encoder(token_ids, valid_mask)), dim=-1)

    def forward(self, pixels: t.Tensor, token_ids: t.Tensor, valid_mask: t.Tensor) -> t.Tensor:
        image = self.encode_images(pixels)
        text = self.encode_texts(token_ids, valid_mask)
        return self.logit_scale.exp() * image @ text.T + self.logit_bias


def pixel_alignment_loss(logits: t.Tensor, *, objective: str) -> t.Tensor:
    """CLIP symmetric cross entropy or SigLIP sum-over-pairs divided by batch size."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or len(logits) == 0:
        raise ValueError("training uses square batches with one example per semantic class")
    count = len(logits)
    labels = t.arange(count, device=logits.device)
    if objective == "clip":
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    if objective == "siglip":
        signed = 2 * t.eye(count, device=logits.device) - 1
        return F.softplus(-signed * logits).sum() / count
    raise ValueError("objective must be clip or siglip")


def semantic_retrieval_metrics(
    logits: t.Tensor, image_labels: t.Tensor, text_labels: t.Tensor
) -> dict[str, float]:
    """Score any semantically correct retrieved instance; split exact top ties fairly."""
    if logits.shape != (len(image_labels), len(text_labels)):
        raise ValueError("labels must describe the rectangular retrieval matrix")
    positive = image_labels[:, None] == text_labels[None, :]
    if not bool(positive.any(0).all() and positive.any(1).all()):
        raise ValueError("every query must have a positive in its gallery")
    top_text = logits == logits.max(1, keepdim=True).values
    top_image = logits == logits.max(0, keepdim=True).values
    i2t = (top_text & positive).sum(1) / top_text.sum(1)
    t2i = (top_image & positive).sum(0) / top_image.sum(0)
    return {"image_to_text": float(i2t.float().mean()), "text_to_image": float(t2i.float().mean())}


def train_pixel_dual_encoder(
    model: nn.Module,
    dataset: PixelDataset,
    *,
    objective: str,
    steps: int = 1200,
    seed: int = 0,
    learning_rate: float = 0.002,
    shuffle_pairs: bool = False,
) -> dict:
    """Train directly on RGB and token IDs, sampling one image per class each step."""
    device = next(model.parameters()).device
    generator = t.Generator().manual_seed(seed)
    pair_generator = t.Generator().manual_seed(seed + 10000)
    images = dataset.images.to(device)
    token_ids, valid_mask = tokenize_pixel_captions(dataset.captions, PIXEL_VOCAB)
    token_ids, valid_mask = token_ids.to(device), valid_mask.to(device)
    groups = [
        (dataset.labels == label).nonzero().flatten() for label in range(len(dataset.captions))
    ]
    if steps < 1 or any(len(group) == 0 for group in groups):
        raise ValueError("positive steps and at least one image per caption class are required")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = t.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    losses = []
    model.train()
    for _ in range(steps):
        indices = t.stack(
            [group[t.randint(len(group), (), generator=generator)] for group in groups]
        ).to(device)
        order = (
            t.randperm(len(groups), generator=pair_generator).to(device)
            if shuffle_pairs
            else t.arange(len(groups), device=device)
        )
        logits = model(images[indices], token_ids[order], valid_mask[order])
        loss = pixel_alignment_loss(logits, objective=objective)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        with t.no_grad():
            model.logit_scale.clamp_(0, math.log(100))
        losses.append(float(loss.detach()))
    changes = {}
    for prefix in ("image_encoder", "text_encoder", "image_projection", "text_projection"):
        changes[prefix] = math.sqrt(
            sum(
                float((p.detach() - before[n]).square().sum())
                for n, p in model.named_parameters()
                if n.startswith(prefix)
            )
        )
    return {
        "losses": losses,
        "parameter_change_norms": changes,
        "steps": steps,
        "examples_seen": steps * len(groups),
        "objective": objective,
        "shuffle_pairs": shuffle_pairs,
    }


@t.no_grad()
def evaluate_pixel_model(
    model: nn.Module, dataset: PixelDataset, *, image_override: t.Tensor | None = None
) -> dict:
    model.eval()
    device = next(model.parameters()).device
    ids, mask = tokenize_pixel_captions(dataset.captions, PIXEL_VOCAB)
    pixels = dataset.images if image_override is None else image_override
    text = model.encode_texts(ids.to(device), mask.to(device))
    image = t.cat([model.encode_images(batch.to(device)) for batch in pixels.split(64)])
    logits = (model.logit_scale.exp() * image @ text.T + model.logit_bias).cpu()
    return {
        **semantic_retrieval_metrics(logits, dataset.labels, t.arange(len(dataset.captions))),
        "logits": logits,
        "image_norm_max_error": float((image.norm(dim=-1) - 1).abs().max()),
        "text_norm_max_error": float((text.norm(dim=-1) - 1).abs().max()),
    }


def run_pixel_comparison(
    train_data: PixelDataset,
    evaluation_data: PixelDataset,
    *,
    device: str = "cpu",
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 1200,
) -> dict:
    """Visible experiment orchestration; all taught methods resolve in this namespace."""
    rows, histories, matrices, models = [], {}, {}, {}
    overlap = dataset_overlap(train_data, evaluation_data)
    if overlap:
        raise ValueError("evaluation contains training pixels")
    for objective in ("clip", "siglip"):
        for seed in seeds:
            t.manual_seed(seed)
            initial = PixelDualEncoder(len(PIXEL_VOCAB)).to(device)

            def record(condition, result):
                rows.append(
                    {
                        "objective": objective,
                        "seed": seed,
                        "condition": condition,
                        **{key: value for key, value in result.items() if key != "logits"},
                    }
                )
                if seed == seeds[0]:
                    matrices[f"{objective}/{condition}"] = result["logits"]

            record("untrained", evaluate_pixel_model(initial, evaluation_data))
            for condition in ("paired", "shuffled_train", "frozen_towers"):
                model = copy.deepcopy(initial)
                if condition == "frozen_towers":
                    for tower in (model.image_encoder, model.text_encoder):
                        tower.requires_grad_(False)
                history = train_pixel_dual_encoder(
                    model,
                    train_data,
                    objective=objective,
                    steps=steps,
                    seed=seed,
                    shuffle_pairs=condition == "shuffled_train",
                )
                histories[f"{objective}/{seed}/{condition}"] = history
                result = evaluate_pixel_model(model, evaluation_data)
                record(condition, result)
                if condition == "paired":
                    blank = t.ones_like(evaluation_data.images)
                    generator = t.Generator().manual_seed(seed + 700)
                    noise = t.rand(evaluation_data.images.shape, generator=generator)
                    record(
                        "blank_pixels",
                        evaluate_pixel_model(model, evaluation_data, image_override=blank),
                    )
                    record(
                        "noise_pixels",
                        evaluate_pixel_model(model, evaluation_data, image_override=noise),
                    )
                    # Replace every caption's color, retaining its original target label.
                    # This is a semantic mismatch control, not a reordering of the gallery labels.
                    permutation = t.arange(len(PIXEL_CAPTIONS)).roll(3)
                    wrong_logits = result["logits"][:, permutation]
                    record(
                        "wrong_color_words",
                        {
                            **semantic_retrieval_metrics(
                                wrong_logits, evaluation_data.labels, t.arange(len(PIXEL_CAPTIONS))
                            ),
                            "logits": wrong_logits,
                        },
                    )
                    if seed == seeds[0]:
                        models[objective] = model
                print(
                    f"{objective}, seed {seed}, {condition}: "
                    f"image→text {result['image_to_text']:.3f}; "
                    f"text→image {result['text_to_image']:.3f}",
                    flush=True,
                )
    return {
        "rows": rows,
        "histories": histories,
        "matrices": matrices,
        "models": models,
        "train_seed": train_data.seed,
        "evaluation_seed": evaluation_data.seed,
        "train_images": len(train_data.images),
        "evaluation_images": len(evaluation_data.images),
        "exact_pixel_overlap": overlap,
        "seeds": list(seeds),
        "steps": steps,
    }


def plot_pixel_comparison(comparison: dict, evaluation_data: PixelDataset):
    """Provided plotting: show every predeclared seed and the seed-zero matrix."""
    import matplotlib.pyplot as plt

    conditions = (
        "paired",
        "frozen_towers",
        "shuffled_train",
        "untrained",
        "blank_pixels",
        "noise_pixels",
        "wrong_color_words",
    )
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for row, objective in enumerate(("clip", "siglip")):
        for condition in ("paired", "frozen_towers", "shuffled_train"):
            losses = np.array(
                [
                    comparison["histories"][f"{objective}/{seed}/{condition}"]["losses"]
                    for seed in comparison["seeds"]
                ]
            )
            # Nonoverlapping 20-step windows smooth sampling variation, not seed variation.
            size = min(20, losses.shape[1])
            stop = losses.shape[1] // size * size
            smoothed = losses[:, :stop].reshape(len(losses), -1, size).mean(-1)
            x = np.arange(1, smoothed.shape[1] + 1) * size
            (line,) = axes[row, 0].plot(x, smoothed.mean(0), label=condition)
            axes[row, 0].fill_between(
                x, smoothed.min(0), smoothed.max(0), alpha=0.12, color=line.get_color()
            )
        axes[row, 0].set(
            title=f"{objective.upper()}: train loss",
            xlabel="Optimizer steps",
            ylabel="Objective (different scales)",
        )
        axes[row, 0].legend(fontsize=8)
        logits = comparison["matrices"][f"{objective}/paired"]
        probabilities = logits.softmax(1)
        mean = t.stack(
            [probabilities[evaluation_data.labels == label].mean(0) for label in range(12)]
        )
        image = axes[row, 1].imshow(mean, vmin=0, vmax=1, cmap="magma")
        axes[row, 1].set(
            title=f"Seed {comparison['seeds'][0]}: class-mean retrieval",
            xlabel="Caption class",
            ylabel="True image class",
        )
        axes[row, 1].set_xticks(
            range(12), [caption[2:] for caption in PIXEL_CAPTIONS], rotation=90, fontsize=7
        )
        axes[row, 1].set_yticks(range(12), [caption[2:] for caption in PIXEL_CAPTIONS], fontsize=7)
        figure.colorbar(
            image, ax=axes[row, 1], label="Relative row softmax (not calibrated)", shrink=0.75
        )
        for index, condition in enumerate(conditions):
            values = [
                item["image_to_text"]
                for item in comparison["rows"]
                if item["objective"] == objective and item["condition"] == condition
            ]
            axes[row, 2].bar(
                index, np.mean(values), color="tab:blue" if condition == "paired" else "#999999"
            )
            offsets = np.linspace(-0.15, 0.15, len(values))
            axes[row, 2].scatter(index + offsets, values, color="black", s=18, zorder=3)
        axes[row, 2].axhline(
            1 / 12, color="tab:red", linestyle="--", label="Balanced chance = 1/12"
        )
        axes[row, 2].set(
            ylim=(0, 1.05),
            title="Fresh-image classification; dots = seeds",
            ylabel="Image→text accuracy",
        )
        axes[row, 2].set_xticks(
            range(len(conditions)),
            [item.replace("_", " ") for item in conditions],
            rotation=50,
            ha="right",
            fontsize=8,
        )
        axes[row, 2].legend(fontsize=8)
    return figure


def pixel_comparison_payload(comparison: dict) -> dict:
    """JSON serialization only; no scores are computed or changed here."""
    return {key: value for key, value in comparison.items() if key not in ("matrices", "models")}
