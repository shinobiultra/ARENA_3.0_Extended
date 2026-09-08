"""Reference bodies for the word-level Mini VLM learner under construction.

These methods will be taught in notebook cells, not imported as a hidden model.
The existing categorical-question model remains a separate warm-up.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch as t
import torch.nn.functional as F
from torch import nn

from .language_data import VOCAB


@dataclass(frozen=True)
class ActivationPatch:
    stage: str
    row_mask: t.Tensor
    donor: t.Tensor


def replace_activation_rows(values: t.Tensor, donor: t.Tensor, row_mask: t.Tensor) -> t.Tensor:
    """Replace exactly the selected batch/position rows without mutating inputs."""
    if values.ndim != 3 or donor.shape != values.shape or row_mask.shape != values.shape[:2]:
        raise ValueError("Values/donor need [batch, position, width] and a matching row mask")
    if row_mask.dtype != t.bool or donor.dtype != values.dtype:
        raise ValueError("Use a boolean row mask and matching activation dtypes")
    if donor.device != values.device or row_mask.device != values.device:
        raise ValueError("All patch tensors must be on the activation device")
    return t.where(row_mask.unsqueeze(-1), donor, values)


def record_patch_stage(name, values, patch, cache):
    """Provided routing: the replacement method itself is a learner exercise."""
    if patch is not None and patch.stage == name:
        values = replace_activation_rows(values, patch.donor, patch.row_mask)
    if cache is not None:
        cache[name] = values.detach().clone()
    return values


class LearnedPatchEmbedding(nn.Module):
    """Non-overlapping RGB patches become a row-major learned token sequence."""

    def __init__(self, *, image_size: int = 32, patch_size: int = 4, width: int = 48):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("Patch size must divide the image size")
        self.image_size, self.patch_size = image_size, patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixels: t.Tensor) -> t.Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError("Patch embedding expects RGB pixels at the configured image size")
        return self.projection(2 * pixels - 1).flatten(2).transpose(1, 2)


class WordVisionBlock(nn.Module):
    """Explicit pre-norm attention/MLP; causal for words, bidirectional for vision."""

    def __init__(self, width: int = 48, heads: int = 4):
        super().__init__()
        if width % heads:
            raise ValueError("The residual width must divide into whole attention heads")
        self.width, self.heads = width, heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attention_out = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, hidden: t.Tensor, valid_mask: t.Tensor, *, causal: bool) -> t.Tensor:
        batch, positions, width = hidden.shape
        if width != self.width or valid_mask.shape != (batch, positions) or valid_mask.dtype != t.bool:
            raise ValueError("Attention needs [batch, position, width] and a boolean validity mask")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("Every example needs at least one valid attention key")
        if causal and not bool(valid_mask[:, 0].all()):
            raise ValueError("Causal batches must start with a valid token; use right padding")
        head_width = width // self.heads
        qkv = self.qkv(self.norm1(hidden)).reshape(batch, positions, 3, self.heads, head_width)
        query, key, value = [part.transpose(1, 2) for part in qkv.unbind(dim=2)]
        scores = query @ key.transpose(-1, -2) / head_width**.5
        allowed = valid_mask[:, None, None, :]
        if causal:
            allowed = allowed & t.ones(positions, positions, device=hidden.device, dtype=t.bool).tril()
        attention = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        context = (attention @ value).transpose(1, 2).reshape(batch, positions, width)
        hidden = hidden + self.attention_out(context)
        hidden = hidden + self.fc2(F.gelu(self.fc1(self.norm2(hidden))))
        return hidden.masked_fill(~valid_mask.unsqueeze(-1), 0)


class PatchVisionEncoder(nn.Module):
    """A trained vision transformer returning every contextualized patch token."""

    def __init__(self, *, image_size=32, patch_size=4, width=48, heads=4, layers=2):
        super().__init__()
        self.patch_embedding = LearnedPatchEmbedding(image_size=image_size, patch_size=patch_size, width=width)
        self.position = nn.Parameter(.02 * t.randn(1, self.patch_embedding.num_patches, width))
        self.blocks = nn.ModuleList([WordVisionBlock(width, heads) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)

    def forward(self, pixels, *, patch=None, cache=None):
        hidden = record_patch_stage("patches", self.patch_embedding(pixels), patch, cache)
        hidden = hidden + self.position
        valid = t.ones(hidden.shape[:2], dtype=t.bool, device=hidden.device)
        for block in self.blocks:
            hidden = block(hidden, valid, causal=False)
        return record_patch_stage("vision", self.final_norm(hidden), patch, cache)


def build_multimodal_word_sequence(model, visual_tokens, token_ids, valid_mask):
    """Insert explicit image markers and visual rows before the word tokens."""
    if token_ids.ndim != 2 or valid_mask.shape != token_ids.shape or valid_mask.dtype != t.bool:
        raise ValueError("Word IDs need a matching boolean validity mask")
    if visual_tokens.shape[:2] != (token_ids.shape[0], model.num_visual_tokens):
        raise ValueError("Visual and text batches must agree with the configured patch grid")
    if token_ids.shape[1] > model.max_text_tokens:
        raise ValueError("Text exceeds the declared context window")
    batch = token_ids.shape[0]
    start = model.word_embedding.weight[VOCAB["<image>"]].expand(batch, 1, -1)
    end = model.word_embedding.weight[VOCAB["</image>"]].expand(batch, 1, -1)
    sequence = t.cat((start, visual_tokens, end, model.word_embedding(token_ids)), dim=1)
    sequence = sequence + model.position[:, :sequence.shape[1]]
    prefix_valid = t.ones(batch, model.num_visual_tokens + 2, dtype=t.bool, device=token_ids.device)
    return sequence, t.cat((prefix_valid, valid_mask), dim=1)


class WordMiniVLM(nn.Module):
    """RGB ViT → learned connector → causal word decoder and vocabulary logits."""

    def __init__(self, *, image_size=32, patch_size=4, vision_width=48, width=48,
                 heads=4, vision_layers=2, decoder_layers=2, max_text_tokens=32):
        super().__init__()
        self.vision = PatchVisionEncoder(image_size=image_size, patch_size=patch_size,
                                        width=vision_width, heads=heads, layers=vision_layers)
        self.num_visual_tokens = self.vision.patch_embedding.num_patches
        self.max_text_tokens = max_text_tokens
        self.connector = nn.Sequential(nn.Linear(vision_width, width), nn.LayerNorm(width))
        self.word_embedding = nn.Embedding(len(VOCAB), width)
        self.position = nn.Parameter(.02 * t.randn(1, self.num_visual_tokens + 2 + max_text_tokens, width))
        self.decoder = nn.ModuleList([WordVisionBlock(width, heads) for _ in range(decoder_layers)])
        self.final_norm = nn.LayerNorm(width)
        self.unembed = nn.Linear(width, len(VOCAB), bias=False)

    def forward(self, pixels, token_ids, valid_mask, *, patch=None, return_cache=False):
        stages = {"patches", "vision", "projected", *(f"decoder_{layer}" for layer in range(len(self.decoder)))}
        if patch is not None and patch.stage not in stages:
            raise ValueError("Unknown activation-patching stage")
        cache = {} if return_cache else None
        visual = self.vision(pixels, patch=patch, cache=cache)
        visual = record_patch_stage("projected", self.connector(visual), patch, cache)
        hidden, sequence_valid = build_multimodal_word_sequence(self, visual, token_ids, valid_mask)
        for layer, block in enumerate(self.decoder):
            hidden = block(hidden, sequence_valid, causal=True)
            hidden = record_patch_stage(f"decoder_{layer}", hidden, patch, cache)
        word_states = self.final_norm(hidden[:, self.num_visual_tokens + 2:])
        logits = self.unembed(word_states)
        return (logits, cache) if return_cache else logits


def teacher_forced_answer_loss(logits: t.Tensor, token_ids: t.Tensor, target_mask: t.Tensor) -> t.Tensor:
    """Predict each answer word/EOS from the preceding prefix; ignore prompt/pad loss."""
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2] or target_mask.shape != token_ids.shape:
        raise ValueError("Word logits, token IDs and target masks must align")
    if target_mask.dtype != t.bool or not bool(target_mask[:, 1:].any()):
        raise ValueError("At least one answer/EOS target is required")
    labels = token_ids[:, 1:].masked_fill(~target_mask[:, 1:], -100)
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), labels.flatten(), ignore_index=-100)


@dataclass(frozen=True)
class WordGeneration:
    tokens: t.Tensor
    lengths: t.Tensor
    finished: t.Tensor


@t.inference_mode()
def greedy_word_generation(model, pixels, prompt_ids, prompt_valid, *, max_new_tokens=8):
    """Generate word tokens from pixels and prompt words without target answers."""
    if (max_new_tokens < 1 or prompt_ids.ndim != 2 or prompt_valid.shape != prompt_ids.shape
            or prompt_valid.dtype != t.bool):
        raise ValueError("Use nonempty prompt batches and a positive generation budget")
    lengths = prompt_valid.sum(1).long()
    expected = t.arange(prompt_ids.shape[1], device=prompt_ids.device)[None, :] < lengths[:, None]
    if not t.equal(prompt_valid, expected) or not bool((lengths > 0).all()):
        raise ValueError("Prompts must have at least one word and contiguous right padding")
    batch, prompt_width = prompt_ids.shape
    ids = F.pad(prompt_ids, (0, max_new_tokens), value=VOCAB["<pad>"])
    valid = F.pad(prompt_valid, (0, max_new_tokens), value=False)
    generated = t.full((batch, max_new_tokens), VOCAB["<pad>"], device=ids.device, dtype=ids.dtype)
    generated_lengths = t.zeros(batch, device=ids.device, dtype=t.long)
    finished = t.zeros(batch, device=ids.device, dtype=t.bool)
    rows = t.arange(batch, device=ids.device)
    previous_mode = model.training
    model.eval()
    try:
        for step in range(max_new_tokens):
            logits = model(pixels, ids, valid)
            next_ids = logits[rows, lengths - 1].argmax(-1)
            active = ~finished
            generated[:, step] = t.where(active, next_ids, VOCAB["<pad>"])
            ids[rows[active], lengths[active]] = next_ids[active]
            valid[rows[active], lengths[active]] = True
            lengths = lengths + active.long()
            generated_lengths = generated_lengths + active.long()
            finished = finished | (active & (next_ids == VOCAB["<eos>"]))
            if bool(finished.all()):
                break
    finally:
        model.train(previous_mode)
    return WordGeneration(generated, generated_lengths, finished)
