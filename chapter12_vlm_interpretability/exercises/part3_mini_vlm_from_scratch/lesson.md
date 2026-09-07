<!-- ARENA:cell markdown -->

# [12.3] Mini VLM from Scratch

**By the end of this notebook, you will have shown that a small learned RGB-to-word model solves familiar questions and captions better than trained modality controls, yet fails on unseen color–shape combinations; stage-specific donor patches reveal where image information reaches its words.**

```python
EXERCISE_ID = "12_3_mini_vlm_from_scratch"
GT_TIER = "GT-3"
DIFFICULTY = 4
IMPORTANCE = 5
EXPECTED_RUNTIME = "two to three study days; about ten minutes of CPU experiments"
REQUIRES_GPU = False
```

## Core Question

Can a small vision–language model generate the right words for the right visual
reason—and what would convince you that it has learned a shortcut instead?

We follow original ARENA's build–test–train–interpret progression. Start with
pixel rows and an exact patching game. Then replace engineered features with
learned RGB patch embeddings and a vision transformer, connect its visual
sequence to a causal word decoder, and train the complete image-to-language
path. The main result is actual generated text, not a seven-class answer head.

The model learns familiar images well but fails to compose some familiar
attributes in new combinations. You will keep that failure visible and use
matched counterfactuals to trace how image information reaches answer words.
The exact warm-up supplies GT-0 semantics; generated labels and independent
numerical oracles support the main GT-3 controlled-model experiment.

Prerequisites: complete original ARENA's [Transformer from Scratch](../../../chapter1_transformer_interp/exercises/part1_transformer_from_scratch/1.1_Transformer_from_Scratch_exercises.ipynb), including attention, language-model training and sampling. The exercises below reuse those ideas while adding image tokens and controlled interventions.

## Learning Objectives

1. Map pixels to ordered visual tokens and replace exactly selected rows.
2. Implement learned patch embeddings, bidirectional vision attention and a connector.
3. Assemble image markers, visual rows and word embeddings with correct masks.
4. Train with shifted answer/caption losses and generate through each example's EOS.
5. Freeze checkpoints on development, then measure every model seed and modality control.
6. Explain a visible compositional failure using actual questions, images and captions.
7. Compare object, background, unrestricted random and full-sequence patches across stages.

Original ARENA chapters remain unchanged. The renderer, token vocabulary,
collation and figure layouts are supplied; the model, loss, training, decoding,
metrics and intervention methods are yours to implement.

<!-- ARENA:cell code -->

```python
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display

chapter = "chapter12_vlm_interpretability"
section = "part3_mini_vlm_from_scratch"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
assets_dir = root_dir / chapter / "instructions" / "assets"
for path in (root_dir, exercises_dir):
    if str(path) not in sys.path:
        sys.path.append(str(path))
t.set_num_threads(2)
RUN_WORD_EXPERIMENTS = os.environ.get("ARENA_WORD_NUMERICAL_ONLY", "0") != "1"
WORD_DEVICE = "cpu"
word_run_directory = Path(os.environ.get("ARENA_WORD_OUTPUT_DIRECTORY") or tempfile.mkdtemp(prefix="arena-word-vlm-"))
word_run_directory.mkdir(parents=True, exist_ok=True)

import part3_mini_vlm_from_scratch.tests as tests
import part3_mini_vlm_from_scratch.solutions as warmup_reference
from part3_mini_vlm_from_scratch import language_data as words
from part3_mini_vlm_from_scratch import language_vlm_tests as model_tests
from part3_mini_vlm_from_scratch import language_training_tests as training_tests
from part3_mini_vlm_from_scratch import language_patching_tests as patch_tests
from part3_mini_vlm_from_scratch.language_vlm import ActivationPatch, WordGeneration
from part3_mini_vlm_from_scratch.language_training import WordTrainingRun, make_word_task_batch, TASKS
from part3_mini_vlm_from_scratch.language_data import VOCAB
from part3_mini_vlm_from_scratch.language_figures import (
    plot_word_learning, plot_word_transfer, plot_generated_gallery,
    plot_word_stage_patching, plot_word_random_patch_spread,
)

# Provided warm-up rendering, constants and a background-only sampler.
IMAGE_SIZE = warmup_reference.IMAGE_SIZE
PATCH_GRID = warmup_reference.PATCH_GRID
VISION_FEATURE_DIM = warmup_reference.VISION_FEATURE_DIM
OBJECT_SIZE = warmup_reference.OBJECT_SIZE
POSITION_ANCHORS = warmup_reference.POSITION_ANCHORS
ANSWER_VOCAB = warmup_reference.ANSWER_VOCAB
ANSWER_TO_ID = warmup_reference.ANSWER_TO_ID
HELDOUT_STYLE = warmup_reference.HELDOUT_STYLE
render_controlled_scene = warmup_reference.render_controlled_scene
same_size_random_region_indices = warmup_reference.same_size_random_region_indices
```

<!-- ARENA:cell markdown -->

## Cold Open — the same question, a different image

The question is **“What color is the object?”** The two images differ only in
object color. A model that reads the image should change its answer; a model
that has memorized a question prior need not. Before training, predict which
pixel patches contain all the changed evidence.

The warm-up deliberately exposes mean RGB and non-white occupancy. These are
engineered features with transparent geometry, not a pretrained or learned
vision encoder. Later you will learn image features directly from RGB.

<!-- ARENA:cell code -->

```python
clean_image, clean_bbox = render_controlled_scene("red", "square", "top", style=HELDOUT_STYLE)
corrupt_image, _ = render_controlled_scene("blue", "square", "top", style=HELDOUT_STYLE)
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
for ax, image, title in zip(axes, (clean_image, corrupt_image), ("clean: red square", "counterfactual: blue square")):
    ax.imshow(image.permute(1, 2, 0)); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
print("object bbox:", clean_bbox, "| fixed question: What color is the object?")
```

<!-- ARENA:cell markdown -->

### Exercise 1 — Make transparent RGB patch tokens

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 10–15 minutes on this exercise.
> ```

Reshape an image into a row-major `6 × 6` patch grid. For every patch, return
mean red, green and blue values plus the fraction of non-white pixels. Keep
all 36 rows: pooling them would erase the region identity needed for patching.

<details><summary>Help - approach</summary>

Separate grid rows from within-patch rows before permuting. Reduce only the two within-patch spatial axes; retain patch order.

</details>

<details><summary>Expected output</summary>

A `[1, 3, 48, 48]` image gives `[1, 36, 4]`. Pure object/background occupancy is exactly one/zero.

</details>

<details><summary>Solution</summary>

```python
class FrozenPatchEncoder(nn.Module):
    """Engineered RGB/occupancy warm-up with no learned weights."""

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
```

</details>

<!-- ARENA:cell code -->

```python
class FrozenPatchEncoder(nn.Module):
    """Engineered RGB/occupancy warm-up with no learned weights."""

    def __init__(self, *, grid_size: int=PATCH_GRID):
        raise NotImplementedError()

    @property
    def num_tokens(self) -> int:
        raise NotImplementedError()

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError()

    def forward(self, images: t.Tensor) -> t.Tensor:
        raise NotImplementedError()

tests.test_frozen_patch_encoder_extracts_rgb_and_occupancy(FrozenPatchEncoder)
print("Exercise 1 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 2 — Cache a frozen representation

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
> You should spend 5–10 minutes on this exercise.
> ```

Run the engineered encoder without gradient tracking and detach the result.
Caching is valid because this encoder has no learned parameters. You must not
reuse such detached visual features when training the learned vision tower later.

<details><summary>Help - approach</summary>

Move the encoder and pixels to the same device. `no_grad()` avoids building a graph; `detach()` makes the cache contract explicit.

</details>

<details><summary>Expected output</summary>

The supplied image batch gives detached `[120, 36, 4]` tokens containing both occupied and empty patches.

</details>

<details><summary>Solution</summary>

```python
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
```

</details>

<!-- ARENA:cell code -->

```python
def encode_visual_token_cache(encoder: FrozenPatchEncoder, images: t.Tensor, *, device: str | t.device='cpu') -> t.Tensor:
    """Encode images once and detach the visual tokens for fast VLM training."""
    raise NotImplementedError()

tests.test_visual_token_cache_detaches_and_preserves_patch_grid(encode_visual_token_cache, FrozenPatchEncoder)
print("Exercise 2 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 3 — Replace only the selected cache rows

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 10–15 minutes on this exercise.
> ```

Return a new cache in which the chosen clean rows become their donor rows.
Reject duplicate or invalid indices and leave both original tensors unchanged.
This operation, rather than an attention weight, is your first intervention.

<details><summary>Help - approach</summary>

Clone before assignment. Index the position axis, not the feature axis; match donor device and dtype explicitly.

</details>

<details><summary>Expected output</summary>

Exactly rows 1 and 3 change in the independent fixture. Unselected rows and original inputs stay unchanged.

</details>

<details><summary>Solution</summary>

```python
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
```

</details>

<!-- ARENA:cell code -->

```python
def patch_visual_tokens(clean_cache: t.Tensor, corrupt_cache: t.Tensor, patch_indices: tuple[int, ...] | list[int]) -> t.Tensor:
    """Return ``clean_cache`` with selected token rows replaced by ``corrupt_cache``."""
    raise NotImplementedError()

tests.test_patch_visual_tokens_replaces_only_selected_rows(patch_visual_tokens)
print("Exercise 3 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 4 — Map an object box to patch rows

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 10–15 minutes on this exercise.
> ```

Treat bounding-box upper limits as exclusive. Include a patch if its rectangle
has positive-area overlap with the box. Enumerate rows before columns.

<details><summary>Help - approach</summary>

Intervals overlap when the maximum lower endpoint is strictly smaller than the minimum upper endpoint. The row index is row × grid width + column.

</details>

<details><summary>Expected output</summary>

The centered object overlaps the known 4 × 4 block of patch rows. Boundary-only contact does not count.

</details>

<details><summary>Solution</summary>

```python
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
```

</details>

<!-- ARENA:cell code -->

```python
def patch_indices_from_bbox(bbox: tuple[int, int, int, int], *, image_size: int=IMAGE_SIZE, grid_size: int=PATCH_GRID) -> tuple[int, ...]:
    """Return patch-token indices whose image patches overlap ``bbox``."""
    raise NotImplementedError()

tests.test_patch_indices_from_bbox_matches_known_grid(patch_indices_from_bbox)
print("Exercise 4 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 5 — Recover an exact visual contribution game

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 15–20 minutes on this exercise.
> ```

Construct a cache whose known object rows each contribute +1 to the clean
answer and −0.25 to the donor answer; reverse those contributions in the donor.
Sum the rows to make answer logits. Compare object replacement with two
independent same-size **background-only** samples.

Both warm-up random sets exclude the object. They are background nulls, not
unrestricted spatial samples. The main experiment will implement the latter.

<details><summary>Help - approach</summary>

Use your bounding-box mapper and row replacer. Known contributions make the expected margin 1.25 times the number of object rows.

</details>

<details><summary>Expected output</summary>

The clean and donor margins have opposite signs. Replacing every object row exactly reproduces the donor margin; both background samples preserve the clean margin.

</details>

<details><summary>Solution</summary>

```python
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
```

</details>

<!-- ARENA:cell code -->

```python
def toy_ground_truth_patch_report(*, grid_size: int=PATCH_GRID, target_answer: str='red', counterfactual_answer: str='blue') -> dict[str, object]:
    """Exact oracle showing why object patching should flip this toy game."""
    raise NotImplementedError()

tests.test_toy_ground_truth_patch_report_has_exact_controls(toy_ground_truth_patch_report)
print("Exercise 5 passed.")
```

<!-- ARENA:cell code -->

```python
toy_report = toy_ground_truth_patch_report()
labels = ["Clean", "Donor", "Object patch", "Background draw 0", "Background draw 1"]
keys = ["clean_margin", "corrupt_margin", "object_patch_margin", "background_patch_margin", "random_patch_margin"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
encoder = FrozenPatchEncoder()
cache = encode_visual_token_cache(encoder, clean_image.unsqueeze(0))
axes[0].imshow(cache[0, :, 3].reshape(PATCH_GRID, PATCH_GRID), vmin=0, vmax=1, cmap="Blues")
axes[0].set(title="Engineered occupancy by patch row", xlabel="Patch column", ylabel="Patch row")
axes[1].barh(labels, [toy_report[key] for key in keys], color=["#2166ac", "#b2182b", "#b2182b", "#777777", "#777777"])
axes[1].axvline(0, color="black", linestyle=":")
axes[1].set(xlabel="Clean-answer minus donor-answer margin", title="Exact additive game with known object rows")
axes[1].invert_yaxis()
plt.show()
fig.savefig(assets_dir / "mini_vlm_exact_patch_warmup.png", dpi=150)
```

<!-- ARENA:cell markdown -->

## Try It Yourself — predict a boundary patch

Move the displayed warm-up object box by one pixel across a patch boundary.
Predict which row enters or leaves the object set before running. The geometry
view is a check of the index map, not a learned attention map.

<!-- ARENA:cell code -->

```python
PLAY_BBOX = (12, 12, 36, 36)
PLAY_SHIFT_X = 1
before = patch_indices_from_bbox(PLAY_BBOX)
after_box = (PLAY_BBOX[0] + PLAY_SHIFT_X, PLAY_BBOX[1], PLAY_BBOX[2] + PLAY_SHIFT_X, PLAY_BBOX[3])
after = patch_indices_from_bbox(after_box)
fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
for ax, rows, title in zip(axes, (before, after), ("Original box", "Shifted box"), strict=True):
    mask = t.zeros(PATCH_GRID * PATCH_GRID)
    mask[list(rows)] = 1
    ax.imshow(mask.reshape(PATCH_GRID, PATCH_GRID), vmin=0, vmax=1, cmap="Blues")
    ax.set_title(f"{title}: {len(rows)} rows")
plt.show()
print("Entered rows:", sorted(set(after) - set(before)), "Left rows:", sorted(set(before) - set(after)))
```

<!-- ARENA:cell markdown -->

## From RGB rows to a sequence of words

The main model receives a `32 × 32` RGB image and a word-level prompt. Four-pixel
patches give 64 visual tokens. Two bidirectional vision blocks contextualize
those tokens. A learned linear connector plus layer normalization maps them
into the decoder width. The causal decoder reads

`<image>, 64 visual rows, </image>, <bos>, question words, <answer>, answer words, <eos>`.

The word vocabulary has 24 entries, including image markers and punctuation.
There is no answer-class head and no categorical question ID. The same decoder
must answer `what color is the object ?`, answer `what shape is the object ?`,
and generate several words for `describe the image .`.

All model layers are trained from scratch. This is a small, inspectable
instance of the patch-transformer idea in [ViT](https://arxiv.org/abs/2010.11929)
and the vision-encoder/connector/language-decoder connection used by
[LLaVA](https://arxiv.org/abs/2304.08485). It uses neither paper's weights,
dataset, scale or complete training recipe.

### Know what the model is allowed to see

Provided scenes expose labels and masks for training targets and evaluation.
Only pixels and word IDs enter the model. Training and development renderer
seeds differ. Red triangles, blue circles and green squares are excluded from
both. Do not open final conditions while selecting checkpoints.

<!-- ARENA:cell code -->

```python
word_train = words.make_language_dataset("train", examples_per_combination=32)
word_development = words.make_language_dataset("development", examples_per_combination=12)
assert not ({(scene.color, scene.shape) for scene in word_train.scenes} & words.HELDOUT_COMBINATIONS)
assert {scene.scene_id for scene in word_train.scenes}.isdisjoint(scene.scene_id for scene in word_development.scenes)
sample_scene = word_train.scenes[0]
sample_tokens = words.tokenize_language_examples([sample_scene] * 3, ["color", "shape", "caption"])
display(pd.DataFrame({"request": list(TASKS),
    "word sequence": [" ".join(words.WORDS[int(token)] for token in row[valid]) for row, valid in zip(sample_tokens["token_ids"], sample_tokens["valid_mask"], strict=True)],
    "answer/EOS target positions": [mask.nonzero().flatten().tolist() for mask in sample_tokens["target_mask"]]}))
print(len(word_train.scenes), "training images;", len(word_development.scenes), "development images;", len(VOCAB), "word tokens")
```

<!-- ARENA:cell markdown -->

### Exercise 6 — Learn an RGB patch embedding

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 10–15 minutes on this exercise.
> ```

Create a stride-four convolution from three channels to the residual width.
Normalize pixels as `2 × pixels − 1`, flatten grid positions in row-major
order, and return `[batch, patch, width]`. This time the representation has
learned weights; the caption loss must be able to update them.

<details><summary>Help - approach</summary>

A non-overlapping convolution applies the same linear map to each flattened RGB window. Flatten the two output grid axes, then transpose channel and position.

</details>

<details><summary>Expected output</summary>

Values and pixel/weight/bias gradients match independently enumerated float64 pixel windows to 1e−11 or tighter.

</details>

<details><summary>Solution</summary>

```python
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
```

</details>

<!-- ARENA:cell code -->

```python
class LearnedPatchEmbedding(nn.Module):
    """Non-overlapping RGB patches become a row-major learned token sequence."""

    def __init__(self, *, image_size: int=32, patch_size: int=4, width: int=48):
        raise NotImplementedError()

    def forward(self, pixels: t.Tensor) -> t.Tensor:
        raise NotImplementedError()

model_tests.test_learned_patches_match_explicit_pixel_windows(LearnedPatchEmbedding)
print("Exercise 6 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 7 — Use bidirectional or causal attention

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 30–40 minutes on this exercise.
> ```

Implement a pre-normalized attention/MLP block with explicit Q, K and V,
scaled dot products, an output projection, and two residual additions. Mask
invalid keys. For the decoder, also mask every future key. Vision tokens use
the same block without the causal restriction.

Right-padded text starts with a valid token. Zero invalid query rows after the
block so pad states do not look like meaningful cached representations.

<details><summary>Help - approach</summary>

Reshape QKV to `[batch, position, 3, heads, head_width]`. Softmax is over keys. Combine the valid-key mask with a lower-triangular mask only when causal=True.

</details>

<details><summary>Expected output</summary>

Both modes match independent scaled-dot-product attention values and gradients. Future-token changes leave earlier causal outputs unchanged; vision outputs can change.

</details>

<details><summary>Solution</summary>

```python
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
```

</details>

<!-- ARENA:cell code -->

```python
class WordVisionBlock(nn.Module):
    """Explicit pre-norm attention/MLP; causal for words, bidirectional for vision."""

    def __init__(self, width: int=48, heads: int=4):
        raise NotImplementedError()

    def forward(self, hidden: t.Tensor, valid_mask: t.Tensor, *, causal: bool) -> t.Tensor:
        raise NotImplementedError()

model_tests.test_word_vision_attention_values_and_gradients(WordVisionBlock)
model_tests.test_word_attention_future_and_padding_invariance(WordVisionBlock)
print("Exercise 7 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 8 — Replace activation rows with an explicit batch mask

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 10–15 minutes on this exercise.
> ```

Extend the warm-up operation to a boolean `[batch, position]` mask, so every
image can select different patch rows. Preserve gradients to donor values on
selected rows and to clean values elsewhere. Require matching device and dtype.

<details><summary>Help - approach</summary>

Broadcast the boolean row mask across the width and use `torch.where`; no in-place mutation is needed.

</details>

<details><summary>Expected output</summary>

Every output row and its gradient routing match the independent assignment oracle exactly.

</details>

<details><summary>Solution</summary>

```python
def replace_activation_rows(values: t.Tensor, donor: t.Tensor, row_mask: t.Tensor) -> t.Tensor:
    """Replace exactly the selected batch/position rows without mutating inputs."""
    if values.ndim != 3 or donor.shape != values.shape or row_mask.shape != values.shape[:2]:
        raise ValueError("Values/donor need [batch, position, width] and a matching row mask")
    if row_mask.dtype != t.bool or donor.dtype != values.dtype:
        raise ValueError("Use a boolean row mask and matching activation dtypes")
    if donor.device != values.device or row_mask.device != values.device:
        raise ValueError("All patch tensors must be on the activation device")
    return t.where(row_mask.unsqueeze(-1), donor, values)
```

</details>

<!-- ARENA:cell code -->

```python
def replace_activation_rows(values: t.Tensor, donor: t.Tensor, row_mask: t.Tensor) -> t.Tensor:
    """Replace exactly the selected batch/position rows without mutating inputs."""
    raise NotImplementedError()

model_tests.test_activation_row_replacement_is_exact_and_differentiable(replace_activation_rows)
print("Exercise 8 passed.")
```

<!-- ARENA:cell markdown -->

### Provided stage routing

The small router below gives every intervention a named location. Your row
replacement implements the actual method. Cache values are detached copies;
ordinary forward values keep their gradient history. A patch is applied
before recording that stage so the cache describes the executed computation.

<!-- ARENA:cell code -->

```python
def record_patch_stage(name, values, patch, cache):
    """Provided routing: the replacement method itself is a learner exercise."""
    if patch is not None and patch.stage == name:
        values = replace_activation_rows(values, patch.donor, patch.row_mask)
    if cache is not None:
        cache[name] = values.detach().clone()
    return values
```

<!-- ARENA:cell markdown -->

### Exercise 9 — Build the vision transformer

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> You should spend 15–20 minutes on this exercise.
> ```

Add learned patch positions, run two bidirectional blocks, and apply final
layer normalization. Record `patches` before position addition and `vision`
after the last normalization. Keep all 64 rows for later region experiments.

<details><summary>Help - approach</summary>

Use `nn.Parameter` for positions and `nn.ModuleList` for blocks so the optimizer can see them. Every visual row is a valid attention key.

</details>

<details><summary>Expected output</summary>

A two-block float64 model matches independently enumerated patches, positional addition and attention to 1e−11.

</details>

<details><summary>Solution</summary>

```python
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
```

</details>

<!-- ARENA:cell code -->

```python
class PatchVisionEncoder(nn.Module):
    """A trained vision transformer returning every contextualized patch token."""

    def __init__(self, *, image_size=32, patch_size=4, width=48, heads=4, layers=2):
        raise NotImplementedError()

    def forward(self, pixels, *, patch=None, cache=None):
        raise NotImplementedError()

model_tests.test_vision_encoder_positions_and_blocks_match_explicit_oracle(PatchVisionEncoder)
print("Exercise 9 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 10 — Assemble image markers, visual rows and word tokens

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 15–20 minutes on this exercise.
> ```

Build the complete sequence with one image-start marker, all projected visual
rows, one image-end marker, and word embeddings. Add decoder positions only
after concatenation. Every visual/marker position is valid; word validity comes
from the right-padding mask. Enforce the declared text context length.

<details><summary>Help - approach</summary>

Image markers are learned rows of the same word embedding table. The text begins at `num_visual_tokens + 2`; do not count BOS as part of the image prefix.

</details>

<details><summary>Expected output</summary>

The exact slot-by-slot oracle agrees on marker embeddings, visual rows, words, positions, validity and visual gradients.

</details>

<details><summary>Solution</summary>

```python
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
```

</details>

<!-- ARENA:cell code -->

```python
def build_multimodal_word_sequence(model, visual_tokens, token_ids, valid_mask):
    """Insert explicit image markers and visual rows before the word tokens."""
    raise NotImplementedError()

model_tests.test_multimodal_word_positions_and_padding_are_explicit(build_multimodal_word_sequence)
print("Exercise 10 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 11 — Connect vision to a causal word decoder

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 25–35 minutes on this exercise.
> ```

Assemble your vision encoder, a linear-plus-LayerNorm connector, word
embeddings, learned decoder positions, two causal blocks, final normalization
and a full-vocabulary unembedding. Return logits only for text positions.

Cache `projected` visual rows before sequence assembly. Cache `decoder_0` and
`decoder_1` after each complete decoder block, including marker and text rows.
Accept one explicit `ActivationPatch` without permanently changing the model.

<details><summary>Help - approach</summary>

Pass the same patch/cache routing through the vision encoder and connector, then each decoder stage. After the blocks, slice out the image prefix before normalizing and unembedding text states.

</details>

<details><summary>Expected output</summary>

Every single-stage replacement matches an independent native module hook. Full-sequence donor replacement reproduces donor logits; gradients reach vision, connector and decoder.

</details>

<details><summary>Solution</summary>

```python
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
```

</details>

<!-- ARENA:cell code -->

```python
class WordMiniVLM(nn.Module):
    """RGB ViT → learned connector → causal word decoder and vocabulary logits."""

    def __init__(self, *, image_size=32, patch_size=4, vision_width=48, width=48, heads=4, vision_layers=2, decoder_layers=2, max_text_tokens=32):
        raise NotImplementedError()

    def forward(self, pixels, token_ids, valid_mask, *, patch=None, return_cache=False):
        raise NotImplementedError()

model_tests.test_word_model_native_patch_and_whole_sequence_parity(WordMiniVLM)
model_tests.test_full_word_loss_reaches_both_towers_and_connector(WordMiniVLM)
print("Exercise 11 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 12 — Shift the teacher-forced answer loss

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 10–15 minutes on this exercise.
> ```

**Common bug:** scoring a token against its own position can reward copying a
word already visible in the prefix. Shift predictions and labels explicitly.

A logit at text position `j` predicts the token at `j + 1`. Score answer words,
caption punctuation and EOS, while ignoring prompt, image-prefix and pad loss.
Average over all supervised target tokens in the batch, not over examples.

Caption examples have five targets (`a`, color, shape, `.`, EOS); one-word VQA
has two. This gives caption requests more total token weight by design.

<details><summary>Help - approach</summary>

Drop the last logit position and the first label/mask position. Set ignored labels to −100 and use cross entropy with that ignore index.

</details>

<details><summary>Expected output</summary>

Exactly four selected next-token terms contribute in the analytic test; every other logit gradient is zero.

</details>

<details><summary>Solution</summary>

```python
def teacher_forced_answer_loss(logits: t.Tensor, token_ids: t.Tensor, target_mask: t.Tensor) -> t.Tensor:
    """Predict each answer word/EOS from the preceding prefix; ignore prompt/pad loss."""
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2] or target_mask.shape != token_ids.shape:
        raise ValueError("Word logits, token IDs and target masks must align")
    if target_mask.dtype != t.bool or not bool(target_mask[:, 1:].any()):
        raise ValueError("At least one answer/EOS target is required")
    labels = token_ids[:, 1:].masked_fill(~target_mask[:, 1:], -100)
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), labels.flatten(), ignore_index=-100)
```

</details>

<!-- ARENA:cell code -->

```python
def teacher_forced_answer_loss(logits: t.Tensor, token_ids: t.Tensor, target_mask: t.Tensor) -> t.Tensor:
    """Predict each answer word/EOS from the preceding prefix; ignore prompt/pad loss."""
    raise NotImplementedError()

model_tests.test_teacher_forcing_shift_mask_and_analytic_gradient(teacher_forced_answer_loss)
print("Exercise 12 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 13 — Generate until each example reaches EOS

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 25–35 minutes on this exercise.
> ```

Start from answer-free prompts. At each step, use each row's last valid text
position, append its argmax word, and extend that row's validity mask. Stop
updating a row after EOS; other rows continue until they finish or reach the
budget. Return tokens, generated lengths and finished flags.

Restore the model's earlier train/eval mode even if a forward call fails.
Do not mutate the caller's prompt tensors. This small decoder recomputes the
prefix at every step; it does not use a KV cache.

<details><summary>Help - approach</summary>

Preallocate room for the generation budget. Index logits with `lengths − 1`, not the last padded column. Update lengths and tokens only where `~finished`.

</details>

<details><summary>Expected output</summary>

Ragged prompts generate different-length answers, retain EOS, pad finished rows and preserve original prompts/model mode. The exception path also restores mode.

</details>

<details><summary>Solution</summary>

```python
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
```

</details>

<!-- ARENA:cell code -->

```python
@t.inference_mode()
def greedy_word_generation(model, pixels, prompt_ids, prompt_valid, *, max_new_tokens=8):
    """Generate word tokens from pixels and prompt words without target answers."""
    raise NotImplementedError()

model_tests.test_greedy_words_stop_each_example_and_restore_mode(greedy_word_generation)
print("Exercise 13 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 14 — Score complete generated sequences

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 5–10 minutes on this exercise.
> ```

An answer is exact only if every generated token matches the reference,
including EOS, the generated length matches, and the decoder finished. A
truncated correct-looking word or a hidden extra special token is a failure.

<details><summary>Help - approach</summary>

Conjoin the finished flag, reference non-pad length and equality of the full padded token rows.

</details>

<details><summary>Expected output</summary>

Only the complete first answer passes; missing EOS, extra image markers and wrong lengths fail.

</details>

<details><summary>Solution</summary>

```python
def generated_word_exact_match(generation, expected_tokens):
    """Count complete word sequences including EOS, with no hidden bad-token cleanup."""
    if generation.tokens.shape != expected_tokens.shape:
        raise ValueError("Generated and reference token matrices must have the same padded width")
    expected_lengths = (expected_tokens != VOCAB["<pad>"]).sum(1)
    return (generation.finished & (generation.lengths == expected_lengths)
            & (generation.tokens == expected_tokens).all(dim=1))
```

</details>

<!-- ARENA:cell code -->

```python
def generated_word_exact_match(generation, expected_tokens):
    """Count complete word sequences including EOS, with no hidden bad-token cleanup."""
    raise NotImplementedError()

training_tests.test_generation_metrics_require_exact_words_and_eos(generated_word_exact_match)
print("Exercise 14 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 15 — Evaluate from answer-free word prompts

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 15–20 minutes on this exercise.
> ```

Build each batch's prompt mask from its declared prompt lengths, erase padded
positions, then call your generator. Report color, shape and caption exact
match separately and retain every predicted sequence. Evaluation must not
accidentally feed teacher-forced target answers to generation.

<details><summary>Help - approach</summary>

The training batch contains answers; slice to the maximum prompt length and mask each example beyond its own length. Mixed request lengths require both operations.

</details>

<details><summary>Expected output</summary>

An independent scripted red-square decoder gets color accuracy 1.0 and shape/caption 0.5 on two different scenes. It rejects any leaked target suffix.

</details>

<details><summary>Solution</summary>

```python
@t.inference_mode()
def evaluate_word_model(model, batch, *, batch_size=64, zero_images=False):
    """Generate from answer-free word prompts and report every objective separately."""
    device = next(model.parameters()).device
    matches, generated = [], []
    for start in range(0, len(batch["images"]), batch_size):
        indices = slice(start, start + batch_size)
        images = batch["images"][indices].to(device)
        if zero_images:
            images = t.zeros_like(images)
        prompt_lengths = batch["prompt_lengths"][indices].to(device)
        width = int(prompt_lengths.max())
        valid = t.arange(width, device=device)[None, :] < prompt_lengths[:, None]
        prompt_ids = batch["token_ids"][indices, :width].to(device).masked_fill(~valid, VOCAB["<pad>"])
        result = greedy_word_generation(model, images, prompt_ids, valid, max_new_tokens=8)
        matches.append(generated_word_exact_match(result, batch["expected_tokens"][indices].to(device)).cpu())
        generated.append(result.tokens.cpu())
    exact = t.cat(matches)
    return {"exact_match": float(exact.float().mean()),
            **{task + "_exact_match": float(exact[batch["task_ids"] == index].float().mean())
               for index, task in enumerate(TASKS)},
            "per_example_exact_match": exact.tolist(), "generated_tokens": t.cat(generated).tolist()}
```

</details>

<!-- ARENA:cell code -->

```python
@t.inference_mode()
def evaluate_word_model(model, batch, *, batch_size=64, zero_images=False):
    """Generate from answer-free word prompts and report every objective separately."""
    raise NotImplementedError()

training_tests.test_word_evaluation_generates_without_answer_leakage(evaluate_word_model)
print("Exercise 15 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 16 — Train both towers and select only on development

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 30–45 minutes on this exercise.
> ```

Use AdamW, fixed mini-batch sampling and norm clipping. Evaluate every 100
updates and keep the earliest checkpoint with the best development exact
sequence score. Never create or score final data inside training.

Implement four modes with the same initialization, image examples and update
budget: joint; text-only with zero images; image-only with question words
removed; and a fixed shuffled image–request pairing. The shuffled permutation
preserves all three requests belonging to an image. Image-only sequences are
shorter, so equal update budgets do not imply equal FLOP cost.

<details><summary>Help - approach</summary>

Keep all optimizer parameters connected to the loss, including vision. Copy the state dict at each improving development checkpoint. Use separate fixed generators for initialization, sampling and the shuffled-pair control.

</details>

<details><summary>Expected output</summary>

Two complete updates match independent moment estimates, bias correction, clipping and weight decay. All five learned parameter groups change.

</details>

<details><summary>Solution</summary>

```python
def train_word_model(train_dataset, development_dataset, *, steps=600, batch_size=48,
                     learning_rate=.003, seed=0, mode="joint", device="cpu",
                     evaluation_every=100, model_kwargs=None):
    """Fit on training instances and choose the best tested development checkpoint.

    No final familiar/compositional split is generated or scored here. Modes
    share initialization, examples and optimizer-step count; sequence lengths
    differ for image-only training, so FLOP costs are not claimed identical.
    """
    if mode not in ("joint", "text-only", "image-only", "shuffled-pairs"):
        raise ValueError("Choose a declared training/control mode")
    if train_dataset.split != "train" or development_dataset.split != "development":
        raise ValueError("Checkpoint selection must use training and development data only")
    if min(steps, batch_size, evaluation_every) < 1:
        raise ValueError("Training and evaluation counts must be positive")
    started = time.perf_counter()
    t.manual_seed(seed)
    model = WordMiniVLM(**(model_kwargs or {})).to(device)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    train = make_word_task_batch(train_dataset, omit_question=mode == "image-only")
    development = make_word_task_batch(development_dataset, omit_question=mode == "image-only")
    training_tensors = {key: value.to(device) for key, value in train.items()}
    if mode == "text-only":
        training_tensors["images"] = t.zeros_like(training_tensors["images"])
    if mode == "shuffled-pairs":
        generator = t.Generator().manual_seed(seed + 2000)
        permutation = t.randperm(len(train_dataset.scenes), generator=generator)
        image_indices = (permutation[:, None] * len(TASKS) + t.arange(len(TASKS))[None, :]).flatten().to(device)
        training_tensors["images"] = training_tensors["images"][image_indices]
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = t.Generator().manual_seed(seed + 1000)
    history, best_state, best_metric, best_step = [], None, -1., 0
    for step in range(steps):
        model.train()
        selected = t.randint(len(train["images"]), (batch_size,), generator=generator).to(device)
        logits = model(training_tensors["images"][selected], training_tensors["token_ids"][selected],
                       training_tensors["valid_mask"][selected])
        loss = teacher_forced_answer_loss(logits, training_tensors["token_ids"][selected],
                                          training_tensors["target_mask"][selected])
        optimizer.zero_grad()
        loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step + 1) % evaluation_every == 0 or step == steps - 1:
            measured = evaluate_word_model(model, development, zero_images=mode == "text-only")
            row = {"step": step + 1, "loss": float(loss.detach()),
                   **{key: value for key, value in measured.items() if key.endswith("exact_match") and not key.startswith("per_")}}
            history.append(row)
            if measured["exact_match"] > best_metric:
                best_metric, best_step = measured["exact_match"], step + 1
                best_state = copy.deepcopy(model.state_dict())
            print(f"{mode} seed {seed} step {step + 1}: loss {float(loss.detach()):.3f}; "
                  f"dev generation {measured['exact_match']:.3f} "
                  f"(color {measured['color_exact_match']:.3f}, shape {measured['shape_exact_match']:.3f}, caption {measured['caption_exact_match']:.3f})", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    changed = {prefix: any(not t.equal(value, initial[name]) for name, value in model.state_dict().items()
                           if name.startswith(prefix))
               for prefix in ("vision.", "connector.", "word_embedding.", "decoder.", "unembed.")}
    measured = evaluate_word_model(model, development, zero_images=mode == "text-only")
    return WordTrainingRun(model, mode, seed, best_step, history, measured, changed, time.perf_counter() - started)
```

</details>

<!-- ARENA:cell code -->

```python
def train_word_model(train_dataset, development_dataset, *, steps=600, batch_size=48, learning_rate=0.003, seed=0, mode='joint', device='cpu', evaluation_every=100, model_kwargs=None):
    """Fit on training instances and choose the best tested development checkpoint.

    No final familiar/compositional split is generated or scored here. Modes
    share initialization, examples and optimizer-step count; sequence lengths
    differ for image-only training, so FLOP costs are not claimed identical.
    """
    raise NotImplementedError()

training_tests.test_two_word_training_updates_match_manual_adamw(train_word_model)
training_tests.test_image_only_inputs_keep_targets_but_remove_question_words()
print("Exercise 16 passed.")
```

<!-- ARENA:cell markdown -->

## Fit the complete image-to-language path

Run all twelve combinations of three initialization seeds and four training
modes. Each sees 600 optimizer updates. Checkpoints are selected separately
using development exact generation, with the earliest maximum winning ties.
No seed is removed because it trains less well. The model has 127,392 learned
parameters; this small experiment runs on CPU.

The control questions matter. With question words removed, the same image
cannot specify whether the user wants color, shape or a caption. An image-only
model may still learn to answer one request type: do not expect every output
to fail. Text-only and shuffled-pair models can exploit answer priors. The
joint model must beat those useful baselines on independently rendered images.

<details><summary>Expected output</summary>

The reference joint development exact scores are about 0.951, 0.944 and 0.855
for seeds 0, 1 and 2. The weaker third seed remains in every figure. Stars mark
the actual selected checkpoint; all earlier and later tested scores remain
visible. Around five minutes of CPU training produces the twelve runs.

</details>

<!-- ARENA:cell code -->

```python
WORD_SEEDS = (0, 1, 2)
WORD_MODES = ("joint", "text-only", "image-only", "shuffled-pairs")
WORD_STEPS = 600

if RUN_WORD_EXPERIMENTS:
    word_runs, word_training_records = [], []
    for mode in WORD_MODES:
        for seed in WORD_SEEDS:
            run = train_word_model(word_train, word_development, steps=WORD_STEPS, batch_size=48,
                                   learning_rate=.003, seed=seed, mode=mode, device=WORD_DEVICE,
                                   evaluation_every=100)
            word_runs.append(run)
            word_training_records.append({"mode": mode, "seed": seed, "selected_step": run.selected_step,
                "history": run.history, "development": run.development,
                "changed_parameter_groups": run.changed_parameter_groups, "elapsed_seconds": run.elapsed_seconds})
    fig = plot_word_learning(word_training_records)
    plt.show()
    fig.savefig(assets_dir / "word_learning.png", dpi=150)
    display(pd.DataFrame([{key: row[key] for key in ("mode", "seed", "selected_step", "elapsed_seconds")}
                          | {"development_exact": row["development"]["exact_match"]} for row in word_training_records]))
```

<!-- ARENA:cell markdown -->

### Freeze before looking at new combinations

Keep every selected checkpoint. The provided digest function identifies exact
parameter values; writing the choices now makes the ordering reviewable.
The subsequent evaluations do not feed back into training or selection.

There are three final conditions: new familiar images in the training palette;
the **same** familiar geometries and sensor noise with muted colors; and muted
images containing the three never-trained color–shape combinations. Comparing
the first two isolates palette shift. Comparing familiar-muted to unseen-muted
keeps the palette fixed while testing the excluded combinations.

<!-- ARENA:cell code -->

```python
def word_model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

if RUN_WORD_EXPERIMENTS:
    word_frozen_choices = {
        "status": "frozen_before_final_evaluation", "model_seeds": list(WORD_SEEDS),
        "training_modes": list(WORD_MODES), "training_steps": WORD_STEPS,
        "train_seed": words.SPLIT_SEEDS["train"], "development_seed": words.SPLIT_SEEDS["development"],
        "train_images": len(word_train.scenes), "development_images": len(word_development.scenes),
        "all_seeds_retained": True,
        "choices": [{"mode": run.mode, "seed": run.seed, "selected_step": run.selected_step,
                     "state_sha256": word_model_digest(run.model), "development_exact": run.development["exact_match"]}
                    for run in word_runs],
        "final_conditions": {"familiar_solid": {"seed": 300, "images": 144},
                             "familiar_muted": {"seed": 300, "images": 144},
                             "compositional_muted": {"seed": 400, "images": 96}},
    }
    frozen_text = json.dumps(word_frozen_choices, sort_keys=True, indent=2) + "\n"
    word_freeze_path = word_run_directory / "frozen_selection.json"
    if word_freeze_path.exists() and word_freeze_path.read_text() != frozen_text:
        raise RuntimeError("Do not overwrite a different frozen selection; use a new run directory.")
    word_freeze_path.write_text(frozen_text)
    print("All twelve checkpoints frozen before final data evaluation:", word_freeze_path)
```

<!-- ARENA:cell markdown -->

### Generate on the final images

Every condition contains both word-level questions and a caption request for
each image. Use your answer-free evaluation function, then plot each objective
separately. Exact caption score requires the entire generated word sequence
and EOS; a correct color alone cannot pass a wrong caption.

<!-- ARENA:cell code -->

```python
if RUN_WORD_EXPERIMENTS:
    familiar_muted = words.make_language_dataset("familiar", examples_per_combination=16)
    solid_scenes = tuple(replace(scene, style="solid") for scene in familiar_muted.scenes)
    solid_images, solid_masks = zip(*(words.render_language_scene(scene) for scene in solid_scenes))
    familiar_solid = replace(familiar_muted, scenes=solid_scenes, images=t.stack(solid_images), object_masks=t.stack(solid_masks))
    compositional_muted = words.make_language_dataset("compositional", examples_per_combination=32)
    word_final_datasets = {"familiar_solid": familiar_solid, "familiar_muted": familiar_muted,
                           "compositional_muted": compositional_muted}
    word_final_rows = []
    for run, choice in zip(word_runs, word_frozen_choices["choices"], strict=True):
        assert word_model_digest(run.model) == choice["state_sha256"]
        for condition, dataset in word_final_datasets.items():
            batch = make_word_task_batch(dataset, omit_question=run.mode == "image-only")
            measured = evaluate_word_model(run.model, batch, zero_images=run.mode == "text-only")
            word_final_rows.append({"mode": run.mode, "seed": run.seed, "condition": condition,
                                    "image_count": len(dataset.scenes), **measured})
    word_final_evaluation = {"selection_sha256": hashlib.sha256(word_freeze_path.read_bytes()).hexdigest(),
                             "rows": word_final_rows}
    fig = plot_word_transfer(word_final_evaluation)
    plt.show()
    fig.savefig(assets_dir / "word_transfer.png", dpi=150)
```

<!-- ARENA:cell markdown -->

## Signature Result — familiar success, compositional failure

The reference joint models average 89.6% exact generation on new familiar
solid images and 85.8% on their muted counterparts. On excluded combinations,
color answers average 90.3%, shape answers 8.0%, and complete captions **0%**.
These are all-seed means, not the best run.

That failure is the main interpretive challenge. Good familiar accuracy,
image necessity and question necessity do not establish independent attribute
composition. Inspect the actual generated words before attributing the gap
to any internal mechanism. The gallery uses fixed first examples; it does not
search for attractive successes or dramatic failures.

<details><summary>Interpretation — what has and has not been shown</summary>

The three-seed comparison shows a reproducible familiar-task advantage over
trained unimodal and shuffled-pair controls. The paired palette comparison
measures one controlled appearance change. The absent caption combinations
expose a failure to combine learned words under the specified data split.

A plausible hypothesis is a learned color–shape prior. That is a hypothesis
about the failure, not a complete mechanism recovered from accuracy alone.
The intervention study next asks where donor-image information can change
words, with explicit endpoint correctness and structural controls.

</details>

<!-- ARENA:cell code -->

```python
if RUN_WORD_EXPERIMENTS:
    for split, condition, count in (("familiar", "familiar_muted", 16), ("compositional", "compositional_muted", 32)):
        dataset = word_final_datasets[condition]
        measured = next(row for row in word_final_rows if row["mode"] == "joint" and row["seed"] == 0 and row["condition"] == condition)
        title = ("New familiar images: fixed first image of all nine combinations" if split == "familiar" else
                 "Unseen combinations: fixed first image of each excluded color–shape pair")
        fig = plot_generated_gallery(dataset, measured, range(0, len(dataset.scenes), count), title=title)
        plt.show()
        fig.savefig(assets_dir / f"word_{split}_gallery.png", dpi=150)
```

<!-- ARENA:cell markdown -->

## Try It Yourself — ask the same image a different question

Edit the words, not a question-class ID. Try a color question, a shape question
and `describe the image .` on the same image. Compare blank-image and missing-
question controls with a matched image whose color changes. If you ask about
shape, a color-only image edit should ideally leave the answer unchanged.

The vocabulary is deliberately small; use spaces around punctuation and words
from the displayed vocabulary. This play cell does not train or replace any
frozen checkpoint.

<!-- ARENA:cell code -->

```python
PLAY_WORD_SEED = 0
PLAY_WORD_SCENE = 0
PLAY_WORD_PROMPT = "what shape is the object ?"

if RUN_WORD_EXPERIMENTS:
    play_model = next(run.model for run in word_runs if run.mode == "joint" and run.seed == PLAY_WORD_SEED)
    scene = familiar_muted.scenes[PLAY_WORD_SCENE]
    donor_scene = words.counterfactual_scene(scene, color=words.COLORS[(words.COLORS.index(scene.color) + 1) % len(words.COLORS)])
    clean_pixels = familiar_muted.images[PLAY_WORD_SCENE:PLAY_WORD_SCENE + 1].to(WORD_DEVICE)
    donor_pixels = words.render_language_scene(donor_scene)[0].unsqueeze(0).to(WORD_DEVICE)
    prompt_ids = t.tensor([[VOCAB[word] for word in ["<bos>", *PLAY_WORD_PROMPT.split(), "<answer>"]]], device=WORD_DEVICE)
    prompt_valid = t.ones_like(prompt_ids, dtype=t.bool)
    no_question = t.tensor([[VOCAB["<bos>"], VOCAB["<answer>"]]], device=WORD_DEVICE)
    play_word_rows = []
    for label, pixels, ids in (("Clean image + question", clean_pixels, prompt_ids),
                              ("Blank image + question", t.zeros_like(clean_pixels), prompt_ids),
                              ("Clean image, no question words", clean_pixels, no_question),
                              ("Counterfactual color + same question", donor_pixels, prompt_ids)):
        generated = greedy_word_generation(play_model, pixels, ids, t.ones_like(ids, dtype=t.bool))
        play_word_rows.append({"condition": label, "generated words": words.decode_answer(generated.tokens[0]),
                               "finished EOS": bool(generated.finished[0])})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    for ax, pixels, title in zip(axes, (clean_pixels, donor_pixels), (f"{scene.color} {scene.shape}", f"{donor_scene.color} {donor_scene.shape}"), strict=True):
        ax.imshow(pixels[0].cpu().permute(1, 2, 0)); ax.set_title(title); ax.axis("off")
    plt.show()
    display(pd.DataFrame(play_word_rows))
```

<!-- ARENA:cell markdown -->

## Where does image information become language information?

A patch means replacing selected clean-run activation rows with rows computed
from a matched donor image. At generation step two, donor activations must use
**the same current generated prefix** as the patched run. Caching a donor's
independently generated caption would change both image and text evidence.

Our five stages make a concrete prediction possible:

| Stage | What the selected visual row contains | What remains downstream |
|---|---|---|
| `patches` | One local learned RGB patch, before positions | Both vision blocks and decoder |
| `vision` | Context from every image patch | Connector and decoder |
| `projected` | Contextualized visual token in decoder width | Both decoder blocks |
| `decoder_0` | Visual state after the first causal block | One decoder block |
| `decoder_1` | Visual state after the last causal block | No further attention into word states |

Predict the final row of the heatmap before implementing it. The final
normalization and unembedding act independently on text positions: replacing
only visual rows after the final decoder block cannot change word logits.
Replacing the **full sequence**, including text states that have already read
the image, is a different intervention and should reproduce donor generation.

<!-- ARENA:cell markdown -->

### Exercise 17 — Map pixel masks and sample honest region controls

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 15–25 minutes on this exercise.
> ```

Map each exact occupied-pixel mask to all overlapping four-pixel patch rows.
For each image, sample eight background sets and eight unrestricted random
sets with the same number of rows as the object. The latter may overlap the
object; never remove a sample because its intervention works.

<details><summary>Help - approach</summary>

For pixel masks, reshape into grid and within-patch axes before reducing with `any`. Draw a random permutation of available rows, taking the required count without replacement.

</details>

<details><summary>Expected output</summary>

The exact pixel oracle gives four known patch rows for each image. Every sampled set has the required size; background sets exclude the object, unrestricted sets can include it.

</details>

<details><summary>Solution</summary>

```python
def visual_rows_from_pixel_mask(pixel_mask: t.Tensor, *, patch_size: int) -> t.Tensor:
    """Mark row-major patch tokens that overlap any selected image pixel."""
    if pixel_mask.ndim != 3 or pixel_mask.dtype != t.bool:
        raise ValueError("Pixel masks must be boolean [batch, height, width]")
    batch, height, width = pixel_mask.shape
    if patch_size < 1 or height % patch_size or width % patch_size:
        raise ValueError("Patch size must divide both image dimensions")
    patches = pixel_mask.reshape(batch, height // patch_size, patch_size, width // patch_size, patch_size)
    return patches.permute(0, 1, 3, 2, 4).flatten(3).any(-1).flatten(1)


def sample_visual_row_controls(object_rows: t.Tensor, *, samples: int = 8, seed: int = 701):
    """Sample equal-size background and unrestricted row sets without filtering outcomes."""
    if object_rows.ndim != 2 or object_rows.dtype != t.bool or samples < 1:
        raise ValueError("Use boolean [batch, patch] masks and positive sample count")
    generator = t.Generator().manual_seed(seed)
    result = {"background": [], "unrestricted_random": []}
    for _ in range(samples):
        background = t.zeros_like(object_rows)
        random = t.zeros_like(object_rows)
        for row, selected in enumerate(object_rows.cpu()):
            count = int(selected.sum())
            available = (~selected).nonzero().flatten()
            if count < 1 or len(available) < count:
                raise ValueError("Each example needs enough background rows for the same-size control")
            choices = available[t.randperm(len(available), generator=generator)[:count]]
            background[row, choices.to(background.device)] = True
            choices = t.randperm(len(selected), generator=generator)[:count]
            random[row, choices.to(random.device)] = True
        result["background"].append(background)
        result["unrestricted_random"].append(random)
    return result
```

</details>

<!-- ARENA:cell code -->

```python
def visual_rows_from_pixel_mask(pixel_mask: t.Tensor, *, patch_size: int) -> t.Tensor:
    """Mark row-major patch tokens that overlap any selected image pixel."""
    raise NotImplementedError()

def sample_visual_row_controls(object_rows: t.Tensor, *, samples: int=8, seed: int=701):
    """Sample equal-size background and unrestricted row sets without filtering outcomes."""
    raise NotImplementedError()

patch_tests.test_pixel_rows_and_unrestricted_controls_do_not_exclude_objects(visual_rows_from_pixel_mask, sample_visual_row_controls)
print("Exercise 17 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 18 — Patch during autoregressive generation

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 25–35 minutes on this exercise.
> ```

At each model call, recompute donor activations on the same word IDs and
validity mask. Replace the selected visual rows in the clean image run. For a
decoder stage, shift patch indices past the initial image marker; for a true
full-sequence patch, select all cached positions, including word states.

Use a temporary wrapper around the model rather than leaving hooks or a
persistent patch flag behind. Let your existing generator handle ragged EOS
and model-mode restoration.

<details><summary>Help - approach</summary>

The wrapper receives current `ids`/`valid` from the generator. Run the donor and patched clean model with those exact tensors. Decoder visual rows occupy positions `1:1 + num_visual_tokens`.

</details>

<details><summary>Expected output</summary>

The independent causal word oracle changes to blue under early visual or late full-sequence replacement. Late visual-only replacement still says red, and the original model mode is restored.

</details>

<details><summary>Solution</summary>

```python
def generate_with_visual_patch(model, clean_pixels, donor_pixels, prompt_ids, prompt_valid, *,
                               stage: str, visual_rows: t.Tensor, full_sequence: bool = False,
                               max_new_tokens: int = 8):
    """Recompute donor activations on the same current word prefix at every step.

    Visual masks index patch rows only. Later decoder stages translate these
    to their positions between image markers. full_sequence also replaces
    marker and text states, which may already contain image information.
    """
    if clean_pixels.shape != donor_pixels.shape:
        raise ValueError("Clean and donor image batches must have matching shapes")
    if visual_rows.shape != (len(clean_pixels), model.num_visual_tokens) or visual_rows.dtype != t.bool:
        raise ValueError("Select visual patch rows with a boolean batch mask")
    if visual_rows.device != clean_pixels.device or donor_pixels.device != clean_pixels.device:
        raise ValueError("Images and visual row masks must share a device")

    class PatchedGenerationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = model
            self.train(model.training)

        def forward(self, pixels, ids, valid):
            _, donor_cache = self.base(donor_pixels, ids, valid, return_cache=True)
            if stage not in donor_cache:
                raise ValueError("Unknown cached intervention stage")
            donor = donor_cache[stage]
            if full_sequence:
                rows = t.ones(donor.shape[:2], dtype=t.bool, device=donor.device)
            elif stage.startswith("decoder_"):
                rows = t.zeros(donor.shape[:2], dtype=t.bool, device=donor.device)
                rows[:, 1:1 + self.base.num_visual_tokens] = visual_rows
            else:
                rows = visual_rows
            return self.base(pixels, ids, valid, patch=ActivationPatch(stage, rows, donor))

    return greedy_word_generation(PatchedGenerationModel(), clean_pixels, prompt_ids, prompt_valid,
                                  max_new_tokens=max_new_tokens)
```

</details>

<!-- ARENA:cell code -->

```python
def generate_with_visual_patch(model, clean_pixels, donor_pixels, prompt_ids, prompt_valid, *, stage: str, visual_rows: t.Tensor, full_sequence: bool=False, max_new_tokens: int=8):
    """Recompute donor activations on the same current word prefix at every step.

    Visual masks index patch rows only. Later decoder stages translate these
    to their positions between image markers. full_sequence also replaces
    marker and text states, which may already contain image information.
    """
    raise NotImplementedError()

patch_tests.test_generation_full_sequence_includes_image_informed_text_states(generate_with_visual_patch)
print("Exercise 18 passed.")
```

<!-- ARENA:cell markdown -->

### Exercise 19 — State the targeted-flip denominator

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> You should spend 10–15 minutes on this exercise.
> ```

Count a targeted flip only among pairs for which the unpatched model answers
both the clean and donor questions correctly. Keep the total number of pairs,
the eligible count, successful flips and unconditional donor-answer rate.
Return `None` for the conditional rate if no pair is eligible.

<details><summary>Help - approach</summary>

Build eligibility as `clean_correct & donor_correct`. Do not count a pre-existing wrong answer as evidence that the patch changed the answer correctly.

</details>

<details><summary>Expected output</summary>

The four-pair oracle has two eligible pairs and one targeted flip: 0.5 conditional, despite 0.75 unconditional donor answers.

</details>

<details><summary>Solution</summary>

```python
def conditional_targeted_flips(clean_correct, donor_correct, patched_is_donor):
    """Report the declared joint-correct denominator and keep unconditional counts."""
    tensors = (clean_correct, donor_correct, patched_is_donor)
    if any(value.dtype != t.bool or value.ndim != 1 for value in tensors):
        raise ValueError("Use one boolean outcome per example")
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("Outcome vectors must describe the same examples")
    eligible = clean_correct & donor_correct
    count = int(eligible.sum())
    flips = int((eligible & patched_is_donor).sum())
    return {"examples": len(clean_correct), "eligible": count, "targeted_flips": flips,
            "conditional_flip_rate": flips / count if count else None,
            "unconditional_donor_answer_rate": float(patched_is_donor.float().mean())}
```

</details>

<!-- ARENA:cell code -->

```python
def conditional_targeted_flips(clean_correct, donor_correct, patched_is_donor):
    """Report the declared joint-correct denominator and keep unconditional counts."""
    raise NotImplementedError()

patch_tests.test_targeted_flip_denominators_exclude_failed_endpoints(conditional_targeted_flips)
print("Exercise 19 passed.")
```

<!-- ARENA:cell markdown -->

### Fix the pairs before measuring their patches

Use the first four instances of every familiar combination in the declared
16-per-combination final recipe. Ask both color and shape questions: 72 pairs
per seed. Donors cycle the queried attribute and preserve geometry, background
and sensor noise. The selected object mask is the union of occupied clean and
donor pixels. Pixels outside it match exactly.

Some donors fall into excluded color–shape combinations. Keep those pairs,
including incorrect endpoint predictions, in the raw results. Conditional
plots label their smaller joint-correct denominator. The reference denominators
are 50/72, 51/72 and 45/72 across seeds; they are not 72/72.

The supplied pair constructor only assembles scenes and labels. The experiment
below uses your model, generator, row selection and intervention functions.

<!-- ARENA:cell code -->

```python
from part3_mini_vlm_from_scratch.run_word_patching import prepare_counterfactual_pairs

if RUN_WORD_EXPERIMENTS:
    word_pairs = prepare_counterfactual_pairs()
    pair_clean_pixels = word_pairs["clean_pixels"].to(WORD_DEVICE)
    pair_donor_pixels = word_pairs["donor_pixels"].to(WORD_DEVICE)
    pair_prompt_ids = word_pairs["prompts"]["token_ids"].to(WORD_DEVICE)
    pair_prompt_valid = word_pairs["prompts"]["valid_mask"].to(WORD_DEVICE)
    pair_clean_expected = word_pairs["clean_expected"].to(WORD_DEVICE)
    pair_donor_expected = word_pairs["donor_expected"].to(WORD_DEVICE)
    object_rows = visual_rows_from_pixel_mask(word_pairs["pixel_mask"], patch_size=4).to(WORD_DEVICE)
    region_controls = sample_visual_row_controls(object_rows, samples=8, seed=701)
    region_conditions = [("object", 0, object_rows, False),
        *((name, index, rows, False) for name, samples in region_controls.items() for index, rows in enumerate(samples)),
        ("all_visual", 0, t.ones_like(object_rows), False),
        ("full_sequence", 0, t.ones_like(object_rows), True)]
    word_stages = ("patches", "vision", "projected", "decoder_0", "decoder_1")
    print(len(word_pairs["tasks"]), "pairs;", len(region_conditions), "conditions per stage; all 3 model seeds retained.")
```

<!-- ARENA:cell markdown -->

### Generate the stage-by-region heatmap

Keep every sample, even when a background or random patch changes the answer.
Vision attention mixes information across the image, so contextualized
background rows can carry object evidence. Their effect does not invalidate
the experiment; it changes what “background token” means at that stage.

The full-sequence control checks donor **generation parity on every pair**,
including pairs on which the donor is wrong. In contrast, the displayed
targeted-flip fraction uses only the explicitly correct endpoints.

<!-- ARENA:cell code -->

```python
def same_word_generation(left, right):
    return ((left.tokens == right.tokens).all(1) & (left.lengths == right.lengths)
            & (left.finished == right.finished))

if RUN_WORD_EXPERIMENTS:
    word_stage_rows, word_stage_endpoints = [], []
    for run in (run for run in word_runs if run.mode == "joint"):
        model = run.model
        choice = next(choice for choice in word_frozen_choices["choices"] if choice["mode"] == run.mode and choice["seed"] == run.seed)
        assert word_model_digest(model) == choice["state_sha256"]
        clean = greedy_word_generation(model, pair_clean_pixels, pair_prompt_ids, pair_prompt_valid)
        donor = greedy_word_generation(model, pair_donor_pixels, pair_prompt_ids, pair_prompt_valid)
        clean_correct = generated_word_exact_match(clean, pair_clean_expected)
        donor_correct = generated_word_exact_match(donor, pair_donor_expected)
        word_stage_endpoints.append({"seed": run.seed, "clean_generated_tokens": clean.tokens.tolist(),
            "donor_generated_tokens": donor.tokens.tolist(), "clean_correct": clean_correct.tolist(), "donor_correct": donor_correct.tolist()})
        for stage in word_stages:
            for condition, sample, selected_rows, full_sequence in region_conditions:
                generated = generate_with_visual_patch(model, pair_clean_pixels, pair_donor_pixels, pair_prompt_ids, pair_prompt_valid,
                    stage=stage, visual_rows=selected_rows, full_sequence=full_sequence)
                targeted = generated_word_exact_match(generated, pair_donor_expected)
                same_clean, same_donor = same_word_generation(generated, clean), same_word_generation(generated, donor)
                if full_sequence or (stage == "patches" and condition in ("object", "all_visual")):
                    assert bool(same_donor.all()), "Full intervention must reproduce actual donor generation."
                if (stage == "patches" and condition == "background") or (stage == "decoder_1" and not full_sequence):
                    assert bool(same_clean.all()), "A structurally inert patch changed the generated sequence."
                by_task = {}
                for task in ("color", "shape"):
                    subset = t.tensor([value == task for value in word_pairs["tasks"]], device=WORD_DEVICE)
                    by_task[task] = conditional_targeted_flips(clean_correct[subset], donor_correct[subset], targeted[subset])
                word_stage_rows.append({"seed": run.seed, "stage": stage, "condition": condition, "sample": sample,
                    **conditional_targeted_flips(clean_correct, donor_correct, targeted), "by_task": by_task,
                    "selected_visual_rows": selected_rows.nonzero().tolist(), "generated_tokens": generated.tokens.tolist(),
                    "generated_lengths": generated.lengths.tolist(), "generated_finished": generated.finished.tolist(),
                    "patched_is_donor": targeted.tolist(), "same_clean_generation": same_clean.tolist(), "same_donor_generation": same_donor.tolist()})
            print(f"Seed {run.seed}, {stage}: all 19 conditions measured.")
    word_stage_study = {"stages": list(word_stages), "control_seed": 701, "samples_per_random_control": 8,
        "pairs": [{"clean_scene_id": clean.scene_id, "task": task, "clean_answer": getattr(clean, task), "donor_answer": getattr(donor, task),
                   "clean_combination": [clean.color, clean.shape], "donor_combination": [donor.color, donor.shape],
                   "donor_combination_was_heldout": (donor.color, donor.shape) in words.HELDOUT_COMBINATIONS}
                  for clean, donor, task in zip(word_pairs["clean_scenes"], word_pairs["donor_scenes"], word_pairs["tasks"], strict=True)],
        "object_rows": object_rows.nonzero().tolist(), "endpoints": word_stage_endpoints, "rows": word_stage_rows}
    fig = plot_word_stage_patching(word_stage_study)
    plt.show()
    fig.savefig(assets_dir / "word_stage_region_patching.png", dpi=150)
    fig = plot_word_random_patch_spread(word_stage_study)
    plt.show()
    fig.savefig(assets_dir / "word_random_patch_spread.png", dpi=150)
```

<!-- ARENA:cell markdown -->

## Interpret the movement from pixels to words

At the input patch stage, replacing object rows substitutes all changed local
pixels. It reproduces donor generation. Background pixels match exactly, so
background input patches are inert. Unrestricted random sets sometimes include
object rows and can work; every such outcome remains in the plots.

After the vision transformer, information is distributed across rows. Replacing
only spatially local object rows is no longer equivalent to replacing the
object's complete internal representation. The similar `vision` and `projected`
results reflect a connector that transforms each row separately.

After the first decoder block, word states have already read the image. Even
replacing all visual rows leaves those word states intact. After the last
block, visual-only patches are structurally inert. Full-sequence replacement
still reproduces donor generation because it includes the informed word states.

<details><summary>Explain the apparently contradictory controls</summary>

“Background patch changes an answer” and “the image background was unchanged”
can both be true after visual attention mixes rows. Similarly, “all visual
rows were replaced” does not mean “all image information was replaced” after
word tokens have already attended to those rows. Name both the stage and the
selected positions before interpreting a causal effect.

The conditional heatmap asks whether a patch changes a correct clean answer
into a correct donor answer. It does not claim to repair wrong endpoints or
explain every compositional failure. Inspect the eligible counts and the raw
pairs whenever two seeds appear to disagree.

</details>

<!-- ARENA:cell markdown -->

## Try It Yourself — move the patch across the architecture

Choose a fixed pair, region and stage. Compare the generated clean, donor and
patched answers. Predict whether switching from visual-only to full-sequence
replacement should matter. Then inspect a pair where the donor itself is
wrong; explain why it was excluded from the conditional numerator/denominator
but retained in the experiment.

<!-- ARENA:cell code -->

```python
PLAY_PATCH_SEED = 0
PLAY_PAIR = 0
PLAY_STAGE = "decoder_1"
PLAY_REGION = "object"  # object, background, unrestricted_random, or all_visual
PLAY_FULL_SEQUENCE = False
PLAY_RANDOM_SAMPLE = 0

if RUN_WORD_EXPERIMENTS:
    play_model = next(run.model for run in word_runs if run.mode == "joint" and run.seed == PLAY_PATCH_SEED)
    all_rows = {"object": object_rows, "all_visual": t.ones_like(object_rows),
                "background": region_controls["background"][PLAY_RANDOM_SAMPLE],
                "unrestricted_random": region_controls["unrestricted_random"][PLAY_RANDOM_SAMPLE]}
    sl = slice(PLAY_PAIR, PLAY_PAIR + 1)
    selected_rows = all_rows[PLAY_REGION][sl]
    clean = greedy_word_generation(play_model, pair_clean_pixels[sl], pair_prompt_ids[sl], pair_prompt_valid[sl])
    donor = greedy_word_generation(play_model, pair_donor_pixels[sl], pair_prompt_ids[sl], pair_prompt_valid[sl])
    patched = generate_with_visual_patch(play_model, pair_clean_pixels[sl], pair_donor_pixels[sl], pair_prompt_ids[sl], pair_prompt_valid[sl],
        stage=PLAY_STAGE, visual_rows=selected_rows, full_sequence=PLAY_FULL_SEQUENCE)
    pair = word_stage_study["pairs"][PLAY_PAIR]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, pixels, title in zip(axes[:2], (pair_clean_pixels[sl], pair_donor_pixels[sl]), ("Clean image", "Donor image"), strict=True):
        ax.imshow(pixels[0].cpu().permute(1, 2, 0)); ax.set_title(title); ax.axis("off")
    axes[2].imshow(selected_rows[0].cpu().reshape(8, 8), vmin=0, vmax=1, cmap="Blues")
    axes[2].set_title(f"{PLAY_STAGE}: {PLAY_REGION} rows")
    plt.show()
    display(pd.DataFrame([{"condition": label, "generated words": words.decode_answer(result.tokens[0]),
                          "finished EOS": bool(result.finished[0])} for label, result in (("clean", clean), ("donor", donor), ("patched", patched))]))
    print("Question:", words.PROMPTS[pair["task"]], "Clean truth:", pair["clean_answer"], "Donor truth:", pair["donor_answer"])
    print("Full-sequence replacement:", PLAY_FULL_SEQUENCE)
```

<!-- ARENA:cell markdown -->

## What can you now defend?

You built the RGB-to-word path, checked its sub-functions, trained both towers,
and generated complete answers and captions. Both modalities help on familiar
images. Unseen attribute combinations expose a substantial failure that the
familiar metrics would have hidden. Stage-specific interventions show why
visual row identity changes meaning after attention and why final visual-only
patches cannot affect earlier-informed text states.

Before moving on, write two short claims: one supported by the familiar-task
controls, and one that those controls do **not** support. Explain one failed
caption and one stage-patching result using the actual image, words, stage,
selected rows and endpoint correctness.

### Limitations and next investigations

This model uses a tiny fixed vocabulary and a single-object renderer, not
open-domain language or photographs. The exact labels come from the renderer;
the learned mechanism is not analytically known. The three model seeds and
three excluded combinations do not establish broad generalization. A patch
can create an unusual hybrid activation; conditional flips are causal tests
under the specified replacement, not unique explanations.

Do not tune the frozen runs on the final images. A new experiment can vary
training-combination coverage, loss weighting, vision depth or connector width
using new training/development choices, then collect fresh final evidence.
An especially useful anomaly is a caption with the right color but wrong
shape: compare its shape question, caption prefix and stage patches before
choosing an architectural change.

### Reading links

- [ViT: An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) connects learned image patches with Transformer encoders.
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) connects a vision encoder to a language model. Our small from-scratch model illustrates that connection, without reproducing its training recipe or headline results.

### Supporting evidence

The generated galleries and causal plots are the teaching results. The final
cell saves the actual measurements, all seeds, random draws and frozen choices
for later checking. An explicit numerical-only run has no trained-model study
to publish.

<!-- ARENA:cell code -->

```python
if RUN_WORD_EXPERIMENTS:
    word_study_evidence = {
        "status": "completed_word_learner_experiments", "training_device": WORD_DEVICE,
        "torch_version": t.__version__, "threads": t.get_num_threads(),
        "fresh_training_runs": len(word_runs), "all_declared_seeds_retained": True,
        "frozen_choices": word_frozen_choices, "training": word_training_records,
        "final_evaluation": word_final_evaluation, "stage_patching": word_stage_study,
        "parameter_count": sum(parameter.numel() for parameter in word_runs[0].model.parameters()),
        "vocabulary": list(words.WORDS),
    }
    encoded = (json.dumps(word_study_evidence, sort_keys=True) + "\n").encode()
    path = word_run_directory / "completed_word_study.json.gz"
    path.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))
    print("Saved all learner measurements:", path)
```
