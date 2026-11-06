# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {
        "title": "Specify the planted pair",
        "icon": "1-circle-fill",
        "subtitle": ""
    },
    {
        "title": "Learn and identify latents",
        "icon": "2-circle-fill",
        "subtitle": ""
    },
    {
        "title": "Test behavior and intervention",
        "icon": "3-circle-fill",
        "subtitle": ""
    },
    {
        "title": "Inspect the full evidence ladder",
        "icon": "4-circle-fill",
        "subtitle": ""
    },
    {
        "title": "Bonus",
        "icon": "star",
        "subtitle": ""
    }
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [6.5] Crosscoders and Model Diffing
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/header-13-2.png" width="350">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Introduction
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Scope.** This lesson retains the full synthetic paired-activation study, including student-built
crosscoder training, planted-feature matching, behavior baselines and targeted interventions. It does not establish acausal across-layer or real base/chat crosscoders, BatchTopK,
complete-shrinkage/latent-decoupling studies, or a real base/fine-tuned learned-crosscoder replication. The separate pinned GELU-1L/SoLU-1L preflight is separate context.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ Specify the planted pair

> ##### Learning Objectives
>
> - Construct exact paired activations with separate training and held-out roles.

### 2️⃣ Learn and identify latents

> ##### Learning Objectives
>
> - Implement the sparse crosscoder class, objective, training and ownership matching.

### 3️⃣ Test behavior and intervention

> ##### Learning Objectives
>
> - Separate reconstruction, prediction and targeted causal interventions.

### 4️⃣ Inspect the full evidence ladder

> ##### Learning Objectives
>
> - Read all controls, inspect examples, and predict changed play settings.

### ☆ Bonus

Keep optional real-model context separate and investigate the anomaly prompts.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
By the end of this notebook, you will have shown that a small learned sparse crosscoder can recover planted shared and model-specific features from paired activations, predict
held-out model-behavior deltas, and pass a targeted ablation test that same-norm random and orthogonal controls fail.

The lesson takes 60-90 minutes and needs no GPU; the CPU signature run takes about 5-15 seconds. The optional TransformerLens preflight needs a CUDA GPU.

## Core Question

The question is not "can we find a large difference direction?" A top SVD direction can often do that. The question is: can a learned sparse crosscoder find features with
interpretable ownership, and can the model-specific feature survive prediction and intervention controls?

> **Common bug:** Low reconstruction error or a large SVD direction does not establish model-specific mechanism. Require held-out ownership recovery, behavior prediction, and a
targeted intervention that beats same-norm random and orthogonal directions.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## What you will implement

- Build an exact shared-plus-delta paired activation dataset with known ground truth.
- Train a tied-encoder sparse crosscoder rather than naming an SVD direction a crosscoder.
- Separate reconstruction, ownership, behavior prediction, and intervention as different claims.
- Compare against zero, shared-only, SVD, and label-shuffle baselines.
- Inspect top examples and use a play cell to change the feature, sparsity pressure, and random seed.

Cold open: the planted pair has two shared features, one model-A-specific feature, and one model-B-specific feature. The B-specific feature creates a behavior delta. Your final
result should visibly show the learned crosscoder recovering those owners and making the targeted B-specific ablation work while controls fail.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display

chapter = "chapter6_sparse_feature_methods"
section = "part5_crosscoders"
legacy_section = "part4_crosscoders_model_diffing"
# FILTERS: ~python
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
# END FILTERS
# FILTERS: python
root_dir = Path(__file__).resolve().parents[3]
# END FILTERS
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / legacy_section  # preserve historical lookup identity
canonical_section_dir = exercises_dir / section
asset_dir = canonical_section_dir / "media"
# FILTERS: python
MAIN = __name__ == "__main__"
# END FILTERS
if MAIN:
    asset_dir.mkdir(exist_ok=True)

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part4_crosscoders_model_diffing.tests as tests
from part5_crosscoders.crosscoder_plumbing import (
    CrosscoderInterventionReport,
    PlantedCrosscoderData,
    PlantedFeatureMatchReport,
    SparseCrosscoderForward,
    SparseCrosscoderLoss,
    SparseCrosscoderReconstructionReport,
    TrainedSparseCrosscoder,
)

FeatureOwner = Literal["shared", "model_a", "model_b"]
MAIN = __name__ == "__main__"
from part5_crosscoders.crosscoder_plumbing import roc_auc_binary

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ Specify the planted pair
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Exact Shared-Plus-Specific Ground Truth

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement the planted paired-model dataset. Each example has latent features `z`; model A and model B decode those features through different dictionaries. Shared features decode
into both models. The model-A-specific feature decodes only into model A. The model-B-specific feature decodes only into model B and determines the behavior-delta label.

The point of this cell is to start with an oracle. If the learned crosscoder later fails, you can tell whether it failed reconstruction, ownership, prediction, or intervention.

<details>
<summary>Expected output</summary>

The test should verify exact reconstruction when `noise_std=0.0`, owners `('shared', 'shared', 'model_a', 'model_b')`, one held-out split, and a roughly balanced behavior label.

```text
All tests in `test_make_planted_crosscoder_data_has_exact_ground_truth` passed!
```
</details>

<details>
<summary>Help - constructing the planted oracle</summary>

Make the clean activations with `true_latents @ true_decoder_a` and `true_latents @ true_decoder_b`. Set the A-specific decoder row to zero for model B and the B-specific decoder
row to zero for model A. Add noise only after the clean activations are built.
</details>

<details>
<summary>Worked solution</summary>

```python
def make_train_heldout_split(
    n_examples: int,
    *,
    train_fraction: float = 0.75,
    seed: int = 99,
) -> tuple[t.Tensor, t.Tensor]:
    """Return deterministic train and held-out indices."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    generator = t.Generator().manual_seed(seed)
    permutation = t.randperm(n_examples, generator=generator)
    n_train = int(train_fraction * n_examples)
    return permutation[:n_train], permutation[n_train:]

def _orthonormal_rows(n_rows: int, n_cols: int, generator: t.Generator) -> t.Tensor:
    if n_rows > n_cols:
        raise ValueError("n_rows must be <= n_cols for this toy orthogonal dictionary.")
    q, _ = t.linalg.qr(t.randn(n_cols, n_rows, generator=generator), mode="reduced")
    return q.T.contiguous()

def make_planted_crosscoder_data(
    *,
    n_examples: int = 1024,
    d_model: int = 12,
    seed: int = 6404,
    noise_std: float = 0.005,
    train_fraction: float = 0.75,
) -> PlantedCrosscoderData:
    """Generate paired activations from exact shared-plus-specific ground truth.

    The ground-truth dictionary has four features: two shared features, one
    model-A-specific feature, and one model-B-specific feature. The B-specific
    feature is also the behavior-delta feature used later for held-out AUC and
    intervention tests.
    """

    if d_model < 4:
        raise ValueError("d_model must be at least 4.")
    if n_examples < 64:
        raise ValueError("n_examples must be at least 64.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    generator = t.Generator().manual_seed(seed)
    n_features = 4
    base = _orthonormal_rows(n_features, d_model, generator)
    jitter = 0.03 * _orthonormal_rows(n_features, d_model, generator)

    true_decoder_a = t.zeros(n_features, d_model)
    true_decoder_b = t.zeros(n_features, d_model)
    for feature_id in range(2):
        true_decoder_a[feature_id] = base[feature_id]
        true_decoder_b[feature_id] = F.normalize(
            0.98 * base[feature_id] + jitter[feature_id],
            dim=0,
        )
    true_decoder_a[2] = base[2]
    true_decoder_b[3] = base[3]

    feature_probabilities = t.tensor([0.45, 0.35, 0.12, 0.18])
    present = t.rand(n_examples, n_features, generator=generator) < feature_probabilities
    magnitudes = 0.5 + t.rand(n_examples, n_features, generator=generator)
    true_latents = present.float() * magnitudes

    behavior_labels = t.rand(n_examples, generator=generator) < 0.5
    n_positive = int(behavior_labels.sum().item())
    n_negative = n_examples - n_positive
    true_latents[behavior_labels, 3] += (
        1.1 + 0.4 * t.rand(n_positive, generator=generator)
    )
    true_latents[~behavior_labels, 2] += (
        0.8 + 0.3 * t.rand(n_negative, generator=generator)
    )

    clean_a = true_latents @ true_decoder_a
    clean_b = true_latents @ true_decoder_b
    model_a_activations = clean_a + noise_std * t.randn(n_examples, d_model, generator=generator)
    model_b_activations = clean_b + noise_std * t.randn(n_examples, d_model, generator=generator)
    behavior_deltas = (
        1.6 * true_latents[:, 3]
        - 0.75 * true_latents[:, 2]
        + 0.12 * true_latents[:, 1]
        + 0.02 * t.randn(n_examples, generator=generator)
    )

    technical_terms = ("debugger", "compiler", "tensor", "kernel", "API", "cache")
    everyday_terms = ("garden", "recipe", "train", "letter", "market", "piano")
    prompts = []
    for index, is_positive in enumerate(behavior_labels.tolist()):
        terms = technical_terms if is_positive else everyday_terms
        prompts.append(f"The {terms[index % len(terms)]} example number {index:04d}")

    train_idx, heldout_idx = make_train_heldout_split(
        n_examples,
        train_fraction=train_fraction,
        seed=99,
    )
    return PlantedCrosscoderData(
        model_a_activations=model_a_activations,
        model_b_activations=model_b_activations,
        true_latents=true_latents,
        behavior_labels=behavior_labels,
        behavior_deltas=behavior_deltas,
        true_decoder_a=true_decoder_a,
        true_decoder_b=true_decoder_b,
        feature_names=(
            "shared topic feature",
            "shared syntax feature",
            "model A everyday feature",
            "model B technical feature",
        ),
        feature_owners=("shared", "shared", "model_a", "model_b"),
        prompts=tuple(prompts),
        train_idx=train_idx,
        heldout_idx=heldout_idx,
    )
```
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def make_train_heldout_split(
#     n_examples: int,
#     *,
#     train_fraction: float = 0.75,
#     seed: int = 99,
# ) -> tuple[t.Tensor, t.Tensor]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def make_train_heldout_split(
    n_examples: int,
    *,
    train_fraction: float = 0.75,
    seed: int = 99,
) -> tuple[t.Tensor, t.Tensor]:
    """Return deterministic train and held-out indices."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    generator = t.Generator().manual_seed(seed)
    permutation = t.randperm(n_examples, generator=generator)
    n_train = int(train_fraction * n_examples)
    return permutation[:n_train], permutation[n_train:]
# END SOLUTION

# EXERCISE
# def _orthonormal_rows(n_rows: int, n_cols: int, generator: t.Generator) -> t.Tensor:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def _orthonormal_rows(n_rows: int, n_cols: int, generator: t.Generator) -> t.Tensor:
    if n_rows > n_cols:
        raise ValueError("n_rows must be <= n_cols for this toy orthogonal dictionary.")
    q, _ = t.linalg.qr(t.randn(n_cols, n_rows, generator=generator), mode="reduced")
    return q.T.contiguous()
# END SOLUTION

# EXERCISE
# def make_planted_crosscoder_data(
#     *,
#     n_examples: int = 1024,
#     d_model: int = 12,
#     seed: int = 6404,
#     noise_std: float = 0.005,
#     train_fraction: float = 0.75,
# ) -> PlantedCrosscoderData:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def make_planted_crosscoder_data(
    *,
    n_examples: int = 1024,
    d_model: int = 12,
    seed: int = 6404,
    noise_std: float = 0.005,
    train_fraction: float = 0.75,
) -> PlantedCrosscoderData:
    """Generate paired activations from exact shared-plus-specific ground truth.

    The ground-truth dictionary has four features: two shared features, one
    model-A-specific feature, and one model-B-specific feature. The B-specific
    feature is also the behavior-delta feature used later for held-out AUC and
    intervention tests.
    """

    if d_model < 4:
        raise ValueError("d_model must be at least 4.")
    if n_examples < 64:
        raise ValueError("n_examples must be at least 64.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    generator = t.Generator().manual_seed(seed)
    n_features = 4
    base = _orthonormal_rows(n_features, d_model, generator)
    jitter = 0.03 * _orthonormal_rows(n_features, d_model, generator)

    true_decoder_a = t.zeros(n_features, d_model)
    true_decoder_b = t.zeros(n_features, d_model)
    for feature_id in range(2):
        true_decoder_a[feature_id] = base[feature_id]
        true_decoder_b[feature_id] = F.normalize(
            0.98 * base[feature_id] + jitter[feature_id],
            dim=0,
        )
    true_decoder_a[2] = base[2]
    true_decoder_b[3] = base[3]

    feature_probabilities = t.tensor([0.45, 0.35, 0.12, 0.18])
    present = t.rand(n_examples, n_features, generator=generator) < feature_probabilities
    magnitudes = 0.5 + t.rand(n_examples, n_features, generator=generator)
    true_latents = present.float() * magnitudes

    behavior_labels = t.rand(n_examples, generator=generator) < 0.5
    n_positive = int(behavior_labels.sum().item())
    n_negative = n_examples - n_positive
    true_latents[behavior_labels, 3] += (
        1.1 + 0.4 * t.rand(n_positive, generator=generator)
    )
    true_latents[~behavior_labels, 2] += (
        0.8 + 0.3 * t.rand(n_negative, generator=generator)
    )

    clean_a = true_latents @ true_decoder_a
    clean_b = true_latents @ true_decoder_b
    model_a_activations = clean_a + noise_std * t.randn(n_examples, d_model, generator=generator)
    model_b_activations = clean_b + noise_std * t.randn(n_examples, d_model, generator=generator)
    behavior_deltas = (
        1.6 * true_latents[:, 3]
        - 0.75 * true_latents[:, 2]
        + 0.12 * true_latents[:, 1]
        + 0.02 * t.randn(n_examples, generator=generator)
    )

    technical_terms = ("debugger", "compiler", "tensor", "kernel", "API", "cache")
    everyday_terms = ("garden", "recipe", "train", "letter", "market", "piano")
    prompts = []
    for index, is_positive in enumerate(behavior_labels.tolist()):
        terms = technical_terms if is_positive else everyday_terms
        prompts.append(f"The {terms[index % len(terms)]} example number {index:04d}")

    train_idx, heldout_idx = make_train_heldout_split(
        n_examples,
        train_fraction=train_fraction,
        seed=99,
    )
    return PlantedCrosscoderData(
        model_a_activations=model_a_activations,
        model_b_activations=model_b_activations,
        true_latents=true_latents,
        behavior_labels=behavior_labels,
        behavior_deltas=behavior_deltas,
        true_decoder_a=true_decoder_a,
        true_decoder_b=true_decoder_b,
        feature_names=(
            "shared topic feature",
            "shared syntax feature",
            "model A everyday feature",
            "model B technical feature",
        ),
        feature_owners=("shared", "shared", "model_a", "model_b"),
        prompts=tuple(prompts),
        train_idx=train_idx,
        heldout_idx=heldout_idx,
    )
# END SOLUTION


if MAIN:
    tests.test_make_planted_crosscoder_data_has_exact_ground_truth(make_planted_crosscoder_data)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ Learn and identify latents
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Sparse Crosscoder Forward Pass and Loss

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement a tied-encoder sparse crosscoder. The same encoder is applied to model-A activations and model-B activations, producing nonnegative latent activations for each model.
Separate decoders reconstruct each model's activation space.

This is intentionally small: the goal is to understand the claim boundary. The learned latents are only worth interpreting if they pass the later ownership and intervention tests.

<details>
<summary>Expected output</summary>

The test checks tensor shapes, nonnegative ReLU latents, and that total loss is reconstruction loss plus `l1_coefficient * l1_loss`.

```text
All tests in `test_sparse_crosscoder_forward_and_loss_shapes` passed!
```
</details>

<details>
<summary>Help - crosscoder shapes and loss</summary>

Use `nn.Linear(d_model, n_latents)` for the tied encoder, `decoder_a` and `decoder_b` as learned `(n_latents, d_model)` parameters, and `F.relu` for sparse nonnegative activations.
</details>

<details>
<summary>Worked solution</summary>

```python
class SparseCrosscoder(nn.Module):
    """A small tied-encoder sparse crosscoder for paired model activations."""

    def __init__(self, d_model: int, n_latents: int = 6, *, seed: int = 2):
        super().__init__()
        if d_model <= 0 or n_latents <= 0:
            raise ValueError("d_model and n_latents must be positive.")

        rng_state = t.random.get_rng_state()
        t.manual_seed(seed)
        self.encoder = nn.Linear(d_model, n_latents)
        self.decoder_a = nn.Parameter(0.08 * t.randn(n_latents, d_model))
        self.decoder_b = nn.Parameter(0.08 * t.randn(n_latents, d_model))
        self.bias_a = nn.Parameter(t.zeros(d_model))
        self.bias_b = nn.Parameter(t.zeros(d_model))
        t.random.set_rng_state(rng_state)

    def encode_model_a(self, model_a_activations: t.Tensor) -> t.Tensor:
        return F.relu(self.encoder(model_a_activations.float()))

    def encode_model_b(self, model_b_activations: t.Tensor) -> t.Tensor:
        return F.relu(self.encoder(model_b_activations.float()))

    def forward(self, model_a_activations: t.Tensor, model_b_activations: t.Tensor):
        if model_a_activations.shape != model_b_activations.shape:
            raise ValueError("paired activations must have matching shapes.")
        model_a_feature_acts = self.encode_model_a(model_a_activations)
        model_b_feature_acts = self.encode_model_b(model_b_activations)
        reconstructed_model_a = model_a_feature_acts @ self.decoder_a + self.bias_a
        reconstructed_model_b = model_b_feature_acts @ self.decoder_b + self.bias_b
        return SparseCrosscoderForward(
            model_a_feature_acts=model_a_feature_acts,
            model_b_feature_acts=model_b_feature_acts,
            reconstructed_model_a=reconstructed_model_a,
            reconstructed_model_b=reconstructed_model_b,
        )

def sparse_crosscoder_loss(
    output: SparseCrosscoderForward,
    model_a_activations: t.Tensor,
    model_b_activations: t.Tensor,
    *,
    l1_coefficient: float = 0.008,
) -> SparseCrosscoderLoss:
    """Compute reconstruction plus sparse-activation loss."""

    reconstruction_loss = F.mse_loss(
        output.reconstructed_model_a,
        model_a_activations.float(),
    ) + F.mse_loss(
        output.reconstructed_model_b,
        model_b_activations.float(),
    )
    l1_loss = output.model_a_feature_acts.mean() + output.model_b_feature_acts.mean()
    total_loss = reconstruction_loss + l1_coefficient * l1_loss
    return SparseCrosscoderLoss(
        reconstruction_loss=reconstruction_loss,
        l1_loss=l1_loss,
        total_loss=total_loss,
    )
```
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Predict what information each encoder sees and what the two decoders must reconstruct; identify how a shared latent can differ from a model-specific latent.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# class SparseCrosscoder(nn.Module):
#     def __init__(self, d_model: int, n_latents: int = 6, *, seed: int = 2):
#         super().__init__()
#         raise NotImplementedError()

#     def encode_model_a(self, model_a_activations: t.Tensor) -> t.Tensor:
#         raise NotImplementedError()

#     def encode_model_b(self, model_b_activations: t.Tensor) -> t.Tensor:
#         raise NotImplementedError()

#     def forward(self, model_a_activations: t.Tensor, model_b_activations: t.Tensor):
#         raise NotImplementedError()
# END EXERCISE
# SOLUTION
class SparseCrosscoder(nn.Module):
    """A small tied-encoder sparse crosscoder for paired model activations."""

    def __init__(self, d_model: int, n_latents: int = 6, *, seed: int = 2):
        super().__init__()
        if d_model <= 0 or n_latents <= 0:
            raise ValueError("d_model and n_latents must be positive.")

        rng_state = t.random.get_rng_state()
        t.manual_seed(seed)
        self.encoder = nn.Linear(d_model, n_latents)
        self.decoder_a = nn.Parameter(0.08 * t.randn(n_latents, d_model))
        self.decoder_b = nn.Parameter(0.08 * t.randn(n_latents, d_model))
        self.bias_a = nn.Parameter(t.zeros(d_model))
        self.bias_b = nn.Parameter(t.zeros(d_model))
        t.random.set_rng_state(rng_state)

    def encode_model_a(self, model_a_activations: t.Tensor) -> t.Tensor:
        return F.relu(self.encoder(model_a_activations.float()))

    def encode_model_b(self, model_b_activations: t.Tensor) -> t.Tensor:
        return F.relu(self.encoder(model_b_activations.float()))

    def forward(self, model_a_activations: t.Tensor, model_b_activations: t.Tensor):
        if model_a_activations.shape != model_b_activations.shape:
            raise ValueError("paired activations must have matching shapes.")
        model_a_feature_acts = self.encode_model_a(model_a_activations)
        model_b_feature_acts = self.encode_model_b(model_b_activations)
        reconstructed_model_a = model_a_feature_acts @ self.decoder_a + self.bias_a
        reconstructed_model_b = model_b_feature_acts @ self.decoder_b + self.bias_b
        return SparseCrosscoderForward(
            model_a_feature_acts=model_a_feature_acts,
            model_b_feature_acts=model_b_feature_acts,
            reconstructed_model_a=reconstructed_model_a,
            reconstructed_model_b=reconstructed_model_b,
        )
# END SOLUTION

# EXERCISE
# def sparse_crosscoder_loss(
#     output: SparseCrosscoderForward,
#     model_a_activations: t.Tensor,
#     model_b_activations: t.Tensor,
#     *,
#     l1_coefficient: float = 0.008,
# ) -> SparseCrosscoderLoss:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def sparse_crosscoder_loss(
    output: SparseCrosscoderForward,
    model_a_activations: t.Tensor,
    model_b_activations: t.Tensor,
    *,
    l1_coefficient: float = 0.008,
) -> SparseCrosscoderLoss:
    """Compute reconstruction plus sparse-activation loss."""

    reconstruction_loss = F.mse_loss(
        output.reconstructed_model_a,
        model_a_activations.float(),
    ) + F.mse_loss(
        output.reconstructed_model_b,
        model_b_activations.float(),
    )
    l1_loss = output.model_a_feature_acts.mean() + output.model_b_feature_acts.mean()
    total_loss = reconstruction_loss + l1_coefficient * l1_loss
    return SparseCrosscoderLoss(
        reconstruction_loss=reconstruction_loss,
        l1_loss=l1_loss,
        total_loss=total_loss,
    )
# END SOLUTION


if MAIN:
    tests.test_sparse_crosscoder_forward_and_loss_shapes(
        SparseCrosscoder,
        sparse_crosscoder_loss,
        make_planted_crosscoder_data,
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Train the Crosscoder and Check Held-Out Reconstruction

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Train the crosscoder on the training split and report held-out reconstruction. This is necessary evidence, not sufficient evidence: an overcomplete model can reconstruct while
mixing ownership. You will test that separately.

<details>
<summary>Expected output</summary>

The held-out reconstruction MSE should fall by far more than 50x on the small CPU test, beat the zero baseline, and land below `0.002`.

```text
All tests in `test_train_sparse_crosscoder_reduces_heldout_reconstruction` passed!
```
</details>

<details>
<summary>Help - training without held-out leakage</summary>

Full-batch AdamW is fine here. Log at step 0 before training so the curve shows learning. The reconstruction report should include model A MSE, model B MSE, average held-out MSE,
zero-baseline MSE, and L0 for both models.
</details>

<details>
<summary>Worked solution</summary>

```python
def evaluate_sparse_crosscoder_reconstruction(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    indices: t.Tensor | None = None,
) -> SparseCrosscoderReconstructionReport:
    """Evaluate held-out reconstruction and sparsity against a zero baseline."""

    if indices is None:
        indices = data.heldout_idx
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    model_a_mse = F.mse_loss(
        output.reconstructed_model_a,
        data.model_a_activations[indices],
    ).item()
    model_b_mse = F.mse_loss(
        output.reconstructed_model_b,
        data.model_b_activations[indices],
    ).item()
    heldout_mse = 0.5 * (model_a_mse + model_b_mse)
    zero_mse = 0.5 * (
        data.model_a_activations[indices].pow(2).mean().item()
        + data.model_b_activations[indices].pow(2).mean().item()
    )
    model_a_l0 = output.model_a_feature_acts.gt(1e-4).float().sum(dim=-1).mean().item()
    model_b_l0 = output.model_b_feature_acts.gt(1e-4).float().sum(dim=-1).mean().item()
    return SparseCrosscoderReconstructionReport(
        model_a_mse=model_a_mse,
        model_b_mse=model_b_mse,
        heldout_reconstruction_mse=heldout_mse,
        zero_reconstruction_mse=zero_mse,
        model_a_l0=model_a_l0,
        model_b_l0=model_b_l0,
        beats_zero_baseline=heldout_mse < 0.01 * zero_mse,
    )

def train_sparse_crosscoder(
    data: PlantedCrosscoderData,
    *,
    n_latents: int = 6,
    steps: int = 500,
    learning_rate: float = 4e-3,
    l1_coefficient: float = 0.008,
    log_every: int = 50,
    seed: int = 2,
) -> TrainedSparseCrosscoder:
    """Train the toy sparse crosscoder on CPU-sized paired activations."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    model = SparseCrosscoder(
        d_model=data.model_a_activations.shape[-1],
        n_latents=n_latents,
        seed=seed,
    )
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_idx = data.train_idx
    heldout_idx = data.heldout_idx
    history: list[dict[str, float]] = []

    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            with t.no_grad():
                train_output = model(
                    data.model_a_activations[train_idx],
                    data.model_b_activations[train_idx],
                )
                heldout_output = model(
                    data.model_a_activations[heldout_idx],
                    data.model_b_activations[heldout_idx],
                )
                train_loss = sparse_crosscoder_loss(
                    train_output,
                    data.model_a_activations[train_idx],
                    data.model_b_activations[train_idx],
                    l1_coefficient=l1_coefficient,
                )
                heldout_loss = sparse_crosscoder_loss(
                    heldout_output,
                    data.model_a_activations[heldout_idx],
                    data.model_b_activations[heldout_idx],
                    l1_coefficient=l1_coefficient,
                )
                heldout_mse = 0.5 * (
                    F.mse_loss(
                        heldout_output.reconstructed_model_a,
                        data.model_a_activations[heldout_idx],
                    ).item()
                    + F.mse_loss(
                        heldout_output.reconstructed_model_b,
                        data.model_b_activations[heldout_idx],
                    ).item()
                )
                history.append(
                    {
                        "step": float(step),
                        "train_total_loss": float(train_loss.total_loss.item()),
                        "heldout_total_loss": float(heldout_loss.total_loss.item()),
                        "heldout_reconstruction_mse": heldout_mse,
                        "model_a_l0": float(
                            heldout_output.model_a_feature_acts.gt(1e-4)
                            .float()
                            .sum(dim=-1)
                            .mean()
                            .item()
                        ),
                        "model_b_l0": float(
                            heldout_output.model_b_feature_acts.gt(1e-4)
                            .float()
                            .sum(dim=-1)
                            .mean()
                            .item()
                        ),
                    }
                )
        if step == steps:
            break

        optimizer.zero_grad()
        output = model(
            data.model_a_activations[train_idx],
            data.model_b_activations[train_idx],
        )
        loss = sparse_crosscoder_loss(
            output,
            data.model_a_activations[train_idx],
            data.model_b_activations[train_idx],
            l1_coefficient=l1_coefficient,
        ).total_loss
        loss.backward()
        optimizer.step()

    return TrainedSparseCrosscoder(
        model=model,
        history=history,
        train_idx=train_idx,
        heldout_idx=heldout_idx,
    )
```
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def evaluate_sparse_crosscoder_reconstruction(
#     model: SparseCrosscoder,
#     data: PlantedCrosscoderData,
#     indices: t.Tensor | None = None,
# ) -> SparseCrosscoderReconstructionReport:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def evaluate_sparse_crosscoder_reconstruction(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    indices: t.Tensor | None = None,
) -> SparseCrosscoderReconstructionReport:
    """Evaluate held-out reconstruction and sparsity against a zero baseline."""

    if indices is None:
        indices = data.heldout_idx
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    model_a_mse = F.mse_loss(
        output.reconstructed_model_a,
        data.model_a_activations[indices],
    ).item()
    model_b_mse = F.mse_loss(
        output.reconstructed_model_b,
        data.model_b_activations[indices],
    ).item()
    heldout_mse = 0.5 * (model_a_mse + model_b_mse)
    zero_mse = 0.5 * (
        data.model_a_activations[indices].pow(2).mean().item()
        + data.model_b_activations[indices].pow(2).mean().item()
    )
    model_a_l0 = output.model_a_feature_acts.gt(1e-4).float().sum(dim=-1).mean().item()
    model_b_l0 = output.model_b_feature_acts.gt(1e-4).float().sum(dim=-1).mean().item()
    return SparseCrosscoderReconstructionReport(
        model_a_mse=model_a_mse,
        model_b_mse=model_b_mse,
        heldout_reconstruction_mse=heldout_mse,
        zero_reconstruction_mse=zero_mse,
        model_a_l0=model_a_l0,
        model_b_l0=model_b_l0,
        beats_zero_baseline=heldout_mse < 0.01 * zero_mse,
    )
# END SOLUTION

# EXERCISE
# def train_sparse_crosscoder(
#     data: PlantedCrosscoderData,
#     *,
#     n_latents: int = 6,
#     steps: int = 500,
#     learning_rate: float = 4e-3,
#     l1_coefficient: float = 0.008,
#     log_every: int = 50,
#     seed: int = 2,
# ) -> TrainedSparseCrosscoder:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def train_sparse_crosscoder(
    data: PlantedCrosscoderData,
    *,
    n_latents: int = 6,
    steps: int = 500,
    learning_rate: float = 4e-3,
    l1_coefficient: float = 0.008,
    log_every: int = 50,
    seed: int = 2,
) -> TrainedSparseCrosscoder:
    """Train the toy sparse crosscoder on CPU-sized paired activations."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    model = SparseCrosscoder(
        d_model=data.model_a_activations.shape[-1],
        n_latents=n_latents,
        seed=seed,
    )
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_idx = data.train_idx
    heldout_idx = data.heldout_idx
    history: list[dict[str, float]] = []

    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            with t.no_grad():
                train_output = model(
                    data.model_a_activations[train_idx],
                    data.model_b_activations[train_idx],
                )
                heldout_output = model(
                    data.model_a_activations[heldout_idx],
                    data.model_b_activations[heldout_idx],
                )
                train_loss = sparse_crosscoder_loss(
                    train_output,
                    data.model_a_activations[train_idx],
                    data.model_b_activations[train_idx],
                    l1_coefficient=l1_coefficient,
                )
                heldout_loss = sparse_crosscoder_loss(
                    heldout_output,
                    data.model_a_activations[heldout_idx],
                    data.model_b_activations[heldout_idx],
                    l1_coefficient=l1_coefficient,
                )
                heldout_mse = 0.5 * (
                    F.mse_loss(
                        heldout_output.reconstructed_model_a,
                        data.model_a_activations[heldout_idx],
                    ).item()
                    + F.mse_loss(
                        heldout_output.reconstructed_model_b,
                        data.model_b_activations[heldout_idx],
                    ).item()
                )
                history.append(
                    {
                        "step": float(step),
                        "train_total_loss": float(train_loss.total_loss.item()),
                        "heldout_total_loss": float(heldout_loss.total_loss.item()),
                        "heldout_reconstruction_mse": heldout_mse,
                        "model_a_l0": float(
                            heldout_output.model_a_feature_acts.gt(1e-4)
                            .float()
                            .sum(dim=-1)
                            .mean()
                            .item()
                        ),
                        "model_b_l0": float(
                            heldout_output.model_b_feature_acts.gt(1e-4)
                            .float()
                            .sum(dim=-1)
                            .mean()
                            .item()
                        ),
                    }
                )
        if step == steps:
            break

        optimizer.zero_grad()
        output = model(
            data.model_a_activations[train_idx],
            data.model_b_activations[train_idx],
        )
        loss = sparse_crosscoder_loss(
            output,
            data.model_a_activations[train_idx],
            data.model_b_activations[train_idx],
            l1_coefficient=l1_coefficient,
        ).total_loss
        loss.backward()
        optimizer.step()

    return TrainedSparseCrosscoder(
        model=model,
        history=history,
        train_idx=train_idx,
        heldout_idx=heldout_idx,
    )
# END SOLUTION


if MAIN:
    tests.test_train_sparse_crosscoder_reduces_heldout_reconstruction(
        train_sparse_crosscoder,
        evaluate_sparse_crosscoder_reconstruction,
        make_planted_crosscoder_data,
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Latent Ownership and Ground-Truth Recovery

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Now test the actual crosscoder claim. A learned latent should not just reconstruct; it should line up with a planted feature and have the right ownership. Use held-out activation
correlations to match learned latents to planted features, then classify ownership by model-A versus model-B activation means.

<details>
<summary>Expected output</summary>

The cheap ownership test classifies a handcrafted activation table as `shared`, `model_a`, and `model_b`. The trained recovery test should recover all four planted owners with
minimum held-out correlation above `0.9`.

```text
All tests in `test_latent_ownership_table_classifies_specificity` passed!
All tests in `test_feature_matching_recovers_planted_owners` passed!
```
</details>

<details>
<summary>Help - classifying latent ownership</summary>

For a shared planted feature, compare its true latent to the average of learned A and B activations. For a model-specific planted feature, compare against that model's learned
activations. Use absolute correlation because dictionary features can be represented with sign conventions in the decoder.
</details>

<details>
<summary>Worked solution</summary>

```python
def latent_ownership_table(
    output: SparseCrosscoderForward,
    *,
    shared_threshold: float = 0.08,
) -> list[dict[str, float | str | int]]:
    """Classify learned latents by model-A vs model-B activation means."""

    model_a_means = output.model_a_feature_acts.mean(dim=0)
    model_b_means = output.model_b_feature_acts.mean(dim=0)
    rows: list[dict[str, float | str | int]] = []
    for latent_id, (model_a_mean, model_b_mean) in enumerate(zip(model_a_means, model_b_means)):
        specificity = float((model_b_mean - model_a_mean).item())
        if abs(specificity) <= shared_threshold:
            owner: FeatureOwner = "shared"
        elif specificity > 0:
            owner = "model_b"
        else:
            owner = "model_a"
        rows.append(
            {
                "latent": latent_id,
                "model_a_mean": float(model_a_mean.item()),
                "model_b_mean": float(model_b_mean.item()),
                "specificity": specificity,
                "predicted_owner": owner,
            }
        )
    return rows

def _columnwise_abs_correlation(reference: t.Tensor, candidates: t.Tensor) -> t.Tensor:
    reference = reference.float()
    candidates = candidates.float()
    reference = (reference - reference.mean()) / reference.std().clamp_min(1e-8)
    candidates = (candidates - candidates.mean(dim=0)) / candidates.std(dim=0).clamp_min(1e-8)
    return (reference[:, None] * candidates).mean(dim=0).abs()

def match_learned_to_planted_features(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    indices: t.Tensor | None = None,
    *,
    shared_threshold: float = 0.08,
    target_model_b_feature_id: int = 3,
) -> PlantedFeatureMatchReport:
    """Match learned latents to planted features by held-out activation correlation."""

    if indices is None:
        indices = data.heldout_idx
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    owner_by_latent = {
        int(row["latent"]): str(row["predicted_owner"])
        for row in latent_ownership_table(output, shared_threshold=shared_threshold)
    }
    matched_latents: list[int] = []
    predicted_owners: list[FeatureOwner] = []
    correlations: list[float] = []
    for feature_id, true_owner in enumerate(data.feature_owners):
        true_activation = data.true_latents[indices, feature_id]
        if true_owner == "shared":
            learned_activations = 0.5 * (
                output.model_a_feature_acts + output.model_b_feature_acts
            )
        elif true_owner == "model_a":
            learned_activations = output.model_a_feature_acts
        else:
            learned_activations = output.model_b_feature_acts
        correlation = _columnwise_abs_correlation(true_activation, learned_activations)
        matched_latent = int(correlation.argmax().item())
        matched_latents.append(matched_latent)
        predicted_owners.append(owner_by_latent[matched_latent])  # type: ignore[arg-type]
        correlations.append(float(correlation[matched_latent].item()))

    correct = [
        predicted == true
        for predicted, true in zip(predicted_owners, data.feature_owners, strict=True)
    ]
    if target_model_b_feature_id < 0 or target_model_b_feature_id >= len(data.feature_owners):
        raise IndexError("target_model_b_feature_id is out of range.")
    return PlantedFeatureMatchReport(
        feature_names=data.feature_names,
        true_owners=data.feature_owners,
        matched_latents=tuple(matched_latents),
        predicted_owners=tuple(predicted_owners),
        correlations=tuple(correlations),
        ownership_accuracy=sum(correct) / len(correct),
        min_correlation=min(correlations),
        target_model_b_feature_id=target_model_b_feature_id,
        target_model_b_latent=matched_latents[target_model_b_feature_id],
    )
```
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Predict whether good reconstruction forces correct latent ownership. State which planted-oracle comparison could falsify that implication.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def latent_ownership_table(
#     output: SparseCrosscoderForward,
#     *,
#     shared_threshold: float = 0.08,
# ) -> list[dict[str, float | str | int]]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def latent_ownership_table(
    output: SparseCrosscoderForward,
    *,
    shared_threshold: float = 0.08,
) -> list[dict[str, float | str | int]]:
    """Classify learned latents by model-A vs model-B activation means."""

    model_a_means = output.model_a_feature_acts.mean(dim=0)
    model_b_means = output.model_b_feature_acts.mean(dim=0)
    rows: list[dict[str, float | str | int]] = []
    for latent_id, (model_a_mean, model_b_mean) in enumerate(zip(model_a_means, model_b_means)):
        specificity = float((model_b_mean - model_a_mean).item())
        if abs(specificity) <= shared_threshold:
            owner: FeatureOwner = "shared"
        elif specificity > 0:
            owner = "model_b"
        else:
            owner = "model_a"
        rows.append(
            {
                "latent": latent_id,
                "model_a_mean": float(model_a_mean.item()),
                "model_b_mean": float(model_b_mean.item()),
                "specificity": specificity,
                "predicted_owner": owner,
            }
        )
    return rows
# END SOLUTION

# EXERCISE
# def _columnwise_abs_correlation(reference: t.Tensor, candidates: t.Tensor) -> t.Tensor:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def _columnwise_abs_correlation(reference: t.Tensor, candidates: t.Tensor) -> t.Tensor:
    reference = reference.float()
    candidates = candidates.float()
    reference = (reference - reference.mean()) / reference.std().clamp_min(1e-8)
    candidates = (candidates - candidates.mean(dim=0)) / candidates.std(dim=0).clamp_min(1e-8)
    return (reference[:, None] * candidates).mean(dim=0).abs()
# END SOLUTION

# EXERCISE
# def match_learned_to_planted_features(
#     model: SparseCrosscoder,
#     data: PlantedCrosscoderData,
#     indices: t.Tensor | None = None,
#     *,
#     shared_threshold: float = 0.08,
#     target_model_b_feature_id: int = 3,
# ) -> PlantedFeatureMatchReport:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def match_learned_to_planted_features(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    indices: t.Tensor | None = None,
    *,
    shared_threshold: float = 0.08,
    target_model_b_feature_id: int = 3,
) -> PlantedFeatureMatchReport:
    """Match learned latents to planted features by held-out activation correlation."""

    if indices is None:
        indices = data.heldout_idx
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    owner_by_latent = {
        int(row["latent"]): str(row["predicted_owner"])
        for row in latent_ownership_table(output, shared_threshold=shared_threshold)
    }
    matched_latents: list[int] = []
    predicted_owners: list[FeatureOwner] = []
    correlations: list[float] = []
    for feature_id, true_owner in enumerate(data.feature_owners):
        true_activation = data.true_latents[indices, feature_id]
        if true_owner == "shared":
            learned_activations = 0.5 * (
                output.model_a_feature_acts + output.model_b_feature_acts
            )
        elif true_owner == "model_a":
            learned_activations = output.model_a_feature_acts
        else:
            learned_activations = output.model_b_feature_acts
        correlation = _columnwise_abs_correlation(true_activation, learned_activations)
        matched_latent = int(correlation.argmax().item())
        matched_latents.append(matched_latent)
        predicted_owners.append(owner_by_latent[matched_latent])  # type: ignore[arg-type]
        correlations.append(float(correlation[matched_latent].item()))

    correct = [
        predicted == true
        for predicted, true in zip(predicted_owners, data.feature_owners, strict=True)
    ]
    if target_model_b_feature_id < 0 or target_model_b_feature_id >= len(data.feature_owners):
        raise IndexError("target_model_b_feature_id is out of range.")
    return PlantedFeatureMatchReport(
        feature_names=data.feature_names,
        true_owners=data.feature_owners,
        matched_latents=tuple(matched_latents),
        predicted_owners=tuple(predicted_owners),
        correlations=tuple(correlations),
        ownership_accuracy=sum(correct) / len(correct),
        min_correlation=min(correlations),
        target_model_b_feature_id=target_model_b_feature_id,
        target_model_b_latent=matched_latents[target_model_b_feature_id],
    )
# END SOLUTION


if MAIN:
    tests.test_latent_ownership_table_classifies_specificity(latent_ownership_table)
    tests.test_feature_matching_recovers_planted_owners(
        train_sparse_crosscoder,
        match_learned_to_planted_features,
        make_planted_crosscoder_data,
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 3️⃣ Test behavior and intervention
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Behavior Prediction and Baselines

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

The model-B-specific latent should predict held-out behavior deltas. Compare it to baselines:

- zero reconstruction;
- shared-only learned latents;
- top SVD direction on train-set model deltas;
- label-shuffled behavior labels.

SVD is allowed to find a useful dense direction. It is not allowed to substitute for sparse feature ownership and targeted intervention.

<details>
<summary>Expected output</summary>

The learned target latent should get behavior AUC above `0.95` on the compact CPU test. Label-shuffling and shared-only controls should be much lower.

```text
All tests in `test_behavior_baselines_include_real_controls` passed!
```
</details>

<details>
<summary>Help - matching learned and planted features</summary>

Use signed AUC for directions, but use the same labels for every baseline. For the SVD baseline, fit the top delta direction on train examples and score held-out deltas; this
avoids fitting the baseline on the test labels.
</details>

<details>
<summary>Worked solution</summary>

```python
def signed_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    """Return max(AUC, 1-AUC) so opposite signed features still count."""

    auc = roc_auc_binary(scores, labels)
    return max(auc, 1.0 - auc)

def model_behavior_delta_from_reconstruction(
    reconstructed_model_a: t.Tensor,
    reconstructed_model_b: t.Tensor,
    data: PlantedCrosscoderData,
    *,
    target_feature_id: int = 3,
) -> t.Tensor:
    """Read out the planted behavior difference from reconstructed activations."""

    return (
        reconstructed_model_b.float() @ data.true_decoder_b[target_feature_id].float()
        - reconstructed_model_a.float() @ data.true_decoder_a[target_feature_id].float()
    )

def ablate_latents(
    output: SparseCrosscoderForward,
    model: SparseCrosscoder,
    latent_ids: list[int] | tuple[int, ...],
) -> tuple[t.Tensor, t.Tensor]:
    """Reconstruct both models after zeroing selected learned latents."""

    model_a_acts = output.model_a_feature_acts.clone()
    model_b_acts = output.model_b_feature_acts.clone()
    if latent_ids:
        model_a_acts[:, list(latent_ids)] = 0
        model_b_acts[:, list(latent_ids)] = 0
    reconstructed_a = model_a_acts @ model.decoder_a + model.bias_a
    reconstructed_b = model_b_acts @ model.decoder_b + model.bias_b
    return reconstructed_a, reconstructed_b

def shared_only_reconstruction(
    output: SparseCrosscoderForward,
    model: SparseCrosscoder,
    owner_rows: list[dict[str, float | str | int]],
) -> tuple[t.Tensor, t.Tensor]:
    """Keep only latents classified as shared and reconstruct both models."""

    non_shared = [
        int(row["latent"])
        for row in owner_rows
        if row["predicted_owner"] != "shared"
    ]
    return ablate_latents(output, model, non_shared)

def svd_delta_direction_scores(
    data: PlantedCrosscoderData,
    *,
    train_idx: t.Tensor | None = None,
    heldout_idx: t.Tensor | None = None,
) -> t.Tensor:
    """Fit a top SVD direction on train deltas, then score held-out deltas."""

    if train_idx is None:
        train_idx = data.train_idx
    if heldout_idx is None:
        heldout_idx = data.heldout_idx
    train_delta = (
        data.model_b_activations[train_idx] - data.model_a_activations[train_idx]
    ).float()
    heldout_delta = (
        data.model_b_activations[heldout_idx] - data.model_a_activations[heldout_idx]
    ).float()
    _, _, right_vectors = t.linalg.svd(train_delta, full_matrices=False)
    return heldout_delta @ right_vectors[0]

def svd_pair_reconstruction_mse(
    data: PlantedCrosscoderData,
    *,
    rank: int = 4,
    train_idx: t.Tensor | None = None,
    heldout_idx: t.Tensor | None = None,
) -> float:
    """Evaluate a train-fitted low-rank SVD baseline on concatenated held-out pairs."""

    if train_idx is None:
        train_idx = data.train_idx
    if heldout_idx is None:
        heldout_idx = data.heldout_idx
    train_pair = t.cat(
        [data.model_a_activations[train_idx], data.model_b_activations[train_idx]],
        dim=-1,
    ).float()
    heldout_pair = t.cat(
        [data.model_a_activations[heldout_idx], data.model_b_activations[heldout_idx]],
        dim=-1,
    ).float()
    center = train_pair.mean(dim=0, keepdim=True)
    _, _, right_vectors = t.linalg.svd(train_pair - center, full_matrices=False)
    basis = right_vectors[:rank]
    reconstructed = (heldout_pair - center) @ basis.T @ basis + center
    return F.mse_loss(reconstructed, heldout_pair).item()

def behavior_baseline_table(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    feature_match: PlantedFeatureMatchReport,
    indices: t.Tensor | None = None,
    *,
    shuffle_seed: int = 12,
) -> list[dict[str, float | str]]:
    """Compare learned target-latent behavior against simple baselines."""

    if indices is None:
        indices = data.heldout_idx
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    labels = data.behavior_labels[indices]
    reconstruction = evaluate_sparse_crosscoder_reconstruction(model, data, indices)
    target_scores = output.model_b_feature_acts[:, feature_match.target_model_b_latent]
    target_auc = signed_auc_binary(target_scores, labels)

    owner_rows = latent_ownership_table(output)
    shared_a, shared_b = shared_only_reconstruction(output, model, owner_rows)
    shared_delta = model_behavior_delta_from_reconstruction(shared_a, shared_b, data)
    shared_auc = signed_auc_binary(shared_delta, labels)
    shared_mse = 0.5 * (
        F.mse_loss(shared_a, data.model_a_activations[indices]).item()
        + F.mse_loss(shared_b, data.model_b_activations[indices]).item()
    )

    svd_scores = svd_delta_direction_scores(data, heldout_idx=indices)
    svd_auc = signed_auc_binary(svd_scores, labels)
    svd_mse = svd_pair_reconstruction_mse(data, heldout_idx=indices)

    generator = t.Generator().manual_seed(shuffle_seed)
    shuffled_labels = labels[t.randperm(labels.numel(), generator=generator)]
    shuffled_auc = signed_auc_binary(target_scores, shuffled_labels)

    return [
        {
            "method": "learned sparse crosscoder target latent",
            "heldout_reconstruction_mse": reconstruction.heldout_reconstruction_mse,
            "behavior_auc": target_auc,
            "role": "main result",
        },
        {
            "method": "zero reconstruction",
            "heldout_reconstruction_mse": reconstruction.zero_reconstruction_mse,
            "behavior_auc": 0.5,
            "role": "reconstruction floor",
        },
        {
            "method": "shared-only learned latents",
            "heldout_reconstruction_mse": shared_mse,
            "behavior_auc": shared_auc,
            "role": "removes model-specific latents",
        },
        {
            "method": "SVD top delta direction",
            "heldout_reconstruction_mse": svd_mse,
            "behavior_auc": svd_auc,
            "role": "non-sparse direction baseline",
        },
        {
            "method": "label-shuffled target latent",
            "heldout_reconstruction_mse": reconstruction.heldout_reconstruction_mse,
            "behavior_auc": shuffled_auc,
            "role": "negative control",
        },
    ]
```
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def signed_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def signed_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    """Return max(AUC, 1-AUC) so opposite signed features still count."""

    auc = roc_auc_binary(scores, labels)
    return max(auc, 1.0 - auc)
# END SOLUTION

# EXERCISE
# def model_behavior_delta_from_reconstruction(
#     reconstructed_model_a: t.Tensor,
#     reconstructed_model_b: t.Tensor,
#     data: PlantedCrosscoderData,
#     *,
#     target_feature_id: int = 3,
# ) -> t.Tensor:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def model_behavior_delta_from_reconstruction(
    reconstructed_model_a: t.Tensor,
    reconstructed_model_b: t.Tensor,
    data: PlantedCrosscoderData,
    *,
    target_feature_id: int = 3,
) -> t.Tensor:
    """Read out the planted behavior difference from reconstructed activations."""

    return (
        reconstructed_model_b.float() @ data.true_decoder_b[target_feature_id].float()
        - reconstructed_model_a.float() @ data.true_decoder_a[target_feature_id].float()
    )
# END SOLUTION

# EXERCISE
# def ablate_latents(
#     output: SparseCrosscoderForward,
#     model: SparseCrosscoder,
#     latent_ids: list[int] | tuple[int, ...],
# ) -> tuple[t.Tensor, t.Tensor]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def ablate_latents(
    output: SparseCrosscoderForward,
    model: SparseCrosscoder,
    latent_ids: list[int] | tuple[int, ...],
) -> tuple[t.Tensor, t.Tensor]:
    """Reconstruct both models after zeroing selected learned latents."""

    model_a_acts = output.model_a_feature_acts.clone()
    model_b_acts = output.model_b_feature_acts.clone()
    if latent_ids:
        model_a_acts[:, list(latent_ids)] = 0
        model_b_acts[:, list(latent_ids)] = 0
    reconstructed_a = model_a_acts @ model.decoder_a + model.bias_a
    reconstructed_b = model_b_acts @ model.decoder_b + model.bias_b
    return reconstructed_a, reconstructed_b
# END SOLUTION

# EXERCISE
# def shared_only_reconstruction(
#     output: SparseCrosscoderForward,
#     model: SparseCrosscoder,
#     owner_rows: list[dict[str, float | str | int]],
# ) -> tuple[t.Tensor, t.Tensor]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def shared_only_reconstruction(
    output: SparseCrosscoderForward,
    model: SparseCrosscoder,
    owner_rows: list[dict[str, float | str | int]],
) -> tuple[t.Tensor, t.Tensor]:
    """Keep only latents classified as shared and reconstruct both models."""

    non_shared = [
        int(row["latent"])
        for row in owner_rows
        if row["predicted_owner"] != "shared"
    ]
    return ablate_latents(output, model, non_shared)
# END SOLUTION

# EXERCISE
# def svd_delta_direction_scores(data: PlantedCrosscoderData, *, train_idx=None, heldout_idx=None):
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def svd_delta_direction_scores(
    data: PlantedCrosscoderData,
    *,
    train_idx: t.Tensor | None = None,
    heldout_idx: t.Tensor | None = None,
) -> t.Tensor:
    """Fit a top SVD direction on train deltas, then score held-out deltas."""

    if train_idx is None:
        train_idx = data.train_idx
    if heldout_idx is None:
        heldout_idx = data.heldout_idx
    train_delta = (
        data.model_b_activations[train_idx] - data.model_a_activations[train_idx]
    ).float()
    heldout_delta = (
        data.model_b_activations[heldout_idx] - data.model_a_activations[heldout_idx]
    ).float()
    _, _, right_vectors = t.linalg.svd(train_delta, full_matrices=False)
    return heldout_delta @ right_vectors[0]
# END SOLUTION

# EXERCISE
# def svd_pair_reconstruction_mse(data: PlantedCrosscoderData, *, rank: int = 4, train_idx=None, heldout_idx=None) -> float:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def svd_pair_reconstruction_mse(
    data: PlantedCrosscoderData,
    *,
    rank: int = 4,
    train_idx: t.Tensor | None = None,
    heldout_idx: t.Tensor | None = None,
) -> float:
    """Evaluate a train-fitted low-rank SVD baseline on concatenated held-out pairs."""

    if train_idx is None:
        train_idx = data.train_idx
    if heldout_idx is None:
        heldout_idx = data.heldout_idx
    train_pair = t.cat(
        [data.model_a_activations[train_idx], data.model_b_activations[train_idx]],
        dim=-1,
    ).float()
    heldout_pair = t.cat(
        [data.model_a_activations[heldout_idx], data.model_b_activations[heldout_idx]],
        dim=-1,
    ).float()
    center = train_pair.mean(dim=0, keepdim=True)
    _, _, right_vectors = t.linalg.svd(train_pair - center, full_matrices=False)
    basis = right_vectors[:rank]
    reconstructed = (heldout_pair - center) @ basis.T @ basis + center
    return F.mse_loss(reconstructed, heldout_pair).item()
# END SOLUTION

# EXERCISE
# def behavior_baseline_table(
#     model: SparseCrosscoder,
#     data: PlantedCrosscoderData,
#     feature_match: PlantedFeatureMatchReport,
#     indices: t.Tensor | None = None,
#     *,
#     shuffle_seed: int = 12,
# ) -> list[dict[str, float | str]]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def behavior_baseline_table(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    feature_match: PlantedFeatureMatchReport,
    indices: t.Tensor | None = None,
    *,
    shuffle_seed: int = 12,
) -> list[dict[str, float | str]]:
    """Compare learned target-latent behavior against simple baselines."""

    if indices is None:
        indices = data.heldout_idx
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    labels = data.behavior_labels[indices]
    reconstruction = evaluate_sparse_crosscoder_reconstruction(model, data, indices)
    target_scores = output.model_b_feature_acts[:, feature_match.target_model_b_latent]
    target_auc = signed_auc_binary(target_scores, labels)

    owner_rows = latent_ownership_table(output)
    shared_a, shared_b = shared_only_reconstruction(output, model, owner_rows)
    shared_delta = model_behavior_delta_from_reconstruction(shared_a, shared_b, data)
    shared_auc = signed_auc_binary(shared_delta, labels)
    shared_mse = 0.5 * (
        F.mse_loss(shared_a, data.model_a_activations[indices]).item()
        + F.mse_loss(shared_b, data.model_b_activations[indices]).item()
    )

    svd_scores = svd_delta_direction_scores(data, heldout_idx=indices)
    svd_auc = signed_auc_binary(svd_scores, labels)
    svd_mse = svd_pair_reconstruction_mse(data, heldout_idx=indices)

    generator = t.Generator().manual_seed(shuffle_seed)
    shuffled_labels = labels[t.randperm(labels.numel(), generator=generator)]
    shuffled_auc = signed_auc_binary(target_scores, shuffled_labels)

    return [
        {
            "method": "learned sparse crosscoder target latent",
            "heldout_reconstruction_mse": reconstruction.heldout_reconstruction_mse,
            "behavior_auc": target_auc,
            "role": "main result",
        },
        {
            "method": "zero reconstruction",
            "heldout_reconstruction_mse": reconstruction.zero_reconstruction_mse,
            "behavior_auc": 0.5,
            "role": "reconstruction floor",
        },
        {
            "method": "shared-only learned latents",
            "heldout_reconstruction_mse": shared_mse,
            "behavior_auc": shared_auc,
            "role": "removes model-specific latents",
        },
        {
            "method": "SVD top delta direction",
            "heldout_reconstruction_mse": svd_mse,
            "behavior_auc": svd_auc,
            "role": "non-sparse direction baseline",
        },
        {
            "method": "label-shuffled target latent",
            "heldout_reconstruction_mse": reconstruction.heldout_reconstruction_mse,
            "behavior_auc": shuffled_auc,
            "role": "negative control",
        },
    ]
# END SOLUTION


if MAIN:
    tests.test_behavior_baselines_include_real_controls(
        train_sparse_crosscoder,
        match_learned_to_planted_features,
        behavior_baseline_table,
        make_planted_crosscoder_data,
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Targeted Ablation and Controls

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Prediction is still not enough. Remove the matched model-B-specific latent and measure the positive behavior-delta reduction. Then subtract two controls with the same activation
scale and decoder-pair norm: a random direction and an orthogonal direction. The target should reduce the behavior delta; controls should not.

<details>
<summary>Expected output</summary>

The target ablation should beat both controls by at least `0.25` absolute delta reduction on the compact CPU test.

```text
All tests in `test_targeted_ablation_beats_same_norm_and_orthogonal_controls` passed!
```
</details>

<details>
<summary>Help - building causal controls</summary>

Restrict the headline reduction to positive behavior-delta examples. Otherwise the A-specific negative examples partially cancel the B-specific effect and make the intervention
harder to interpret.
</details>

<details>
<summary>Worked solution</summary>

```python
def crosscoder_intervention_report(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    feature_match: PlantedFeatureMatchReport,
    indices: t.Tensor | None = None,
    *,
    random_seed: int = 111,
    control_margin: float = 0.25,
) -> CrosscoderInterventionReport:
    """Compare targeted model-specific ablation against random/orthogonal controls."""

    if indices is None:
        indices = data.heldout_idx
    target_latent = feature_match.target_model_b_latent
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
        baseline_delta = model_behavior_delta_from_reconstruction(
            output.reconstructed_model_a,
            output.reconstructed_model_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )
        target_a, target_b = ablate_latents(output, model, [target_latent])
        target_delta = model_behavior_delta_from_reconstruction(
            target_a,
            target_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )

        target_pair = t.cat([model.decoder_a[target_latent], model.decoder_b[target_latent]])
        generator = t.Generator().manual_seed(random_seed)
        random_pair = t.randn(target_pair.shape, generator=generator)
        random_pair = random_pair / random_pair.norm().clamp_min(1e-12)
        random_pair = random_pair * target_pair.norm().clamp_min(1e-12)
        random_a = random_pair[: model.decoder_a.shape[-1]]
        random_b = random_pair[model.decoder_a.shape[-1] :]
        random_reconstructed_a = (
            output.reconstructed_model_a
            - output.model_a_feature_acts[:, target_latent : target_latent + 1] * random_a
        )
        random_reconstructed_b = (
            output.reconstructed_model_b
            - output.model_b_feature_acts[:, target_latent : target_latent + 1] * random_b
        )
        random_delta = model_behavior_delta_from_reconstruction(
            random_reconstructed_a,
            random_reconstructed_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )

        orthogonal_pair = random_pair - (
            (random_pair @ target_pair) / target_pair.pow(2).sum().clamp_min(1e-12)
        ) * target_pair
        orthogonal_pair = orthogonal_pair / orthogonal_pair.norm().clamp_min(1e-12)
        orthogonal_pair = orthogonal_pair * target_pair.norm().clamp_min(1e-12)
        orthogonal_a = orthogonal_pair[: model.decoder_a.shape[-1]]
        orthogonal_b = orthogonal_pair[model.decoder_a.shape[-1] :]
        orthogonal_reconstructed_a = (
            output.reconstructed_model_a
            - output.model_a_feature_acts[:, target_latent : target_latent + 1] * orthogonal_a
        )
        orthogonal_reconstructed_b = (
            output.reconstructed_model_b
            - output.model_b_feature_acts[:, target_latent : target_latent + 1] * orthogonal_b
        )
        orthogonal_delta = model_behavior_delta_from_reconstruction(
            orthogonal_reconstructed_a,
            orthogonal_reconstructed_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )

    positive = data.behavior_labels[indices]
    baseline_abs = baseline_delta[positive].abs().mean().item()
    target_abs = target_delta[positive].abs().mean().item()
    random_abs = random_delta[positive].abs().mean().item()
    orthogonal_abs = orthogonal_delta[positive].abs().mean().item()
    target_reduction = baseline_abs - target_abs
    random_reduction = baseline_abs - random_abs
    orthogonal_reduction = baseline_abs - orthogonal_abs
    return CrosscoderInterventionReport(
        target_latent=target_latent,
        baseline_abs_delta=baseline_abs,
        target_ablated_abs_delta=target_abs,
        same_norm_random_abs_delta=random_abs,
        orthogonal_abs_delta=orthogonal_abs,
        target_reduction=target_reduction,
        same_norm_random_reduction=random_reduction,
        orthogonal_reduction=orthogonal_reduction,
        passes_controls=(
            target_reduction > random_reduction + control_margin
            and target_reduction > orthogonal_reduction + control_margin
        ),
    )

def top_activating_examples(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    feature_match: PlantedFeatureMatchReport,
    indices: t.Tensor | None = None,
    *,
    k: int = 6,
) -> list[dict[str, float | str | int | bool]]:
    """Return the top held-out examples for the matched model-B-specific latent."""

    if indices is None:
        indices = data.heldout_idx
    target_latent = feature_match.target_model_b_latent
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    scores = output.model_b_feature_acts[:, target_latent]
    top_local = scores.argsort(descending=True)[:k]
    rows: list[dict[str, float | str | int | bool]] = []
    for rank, local_index in enumerate(top_local.tolist(), start=1):
        global_index = int(indices[local_index].item())
        rows.append(
            {
                "rank": rank,
                "example_index": global_index,
                "prompt": data.prompts[global_index],
                "is_behavior_delta_positive": bool(data.behavior_labels[global_index].item()),
                "learned_activation": float(scores[local_index].item()),
                "true_model_b_feature": float(
                    data.true_latents[global_index, feature_match.target_model_b_feature_id].item()
                ),
                "behavior_delta": float(data.behavior_deltas[global_index].item()),
            }
        )
    return rows
```
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Predict whether target-latent ablation should exceed both norm-matched controls by the declared margin on the positive held-out subset. Define the denominator before inspection.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def crosscoder_intervention_report(
#     model: SparseCrosscoder,
#     data: PlantedCrosscoderData,
#     feature_match: PlantedFeatureMatchReport,
#     indices: t.Tensor | None = None,
#     *,
#     random_seed: int = 111,
#     control_margin: float = 0.25,
# ) -> CrosscoderInterventionReport:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def crosscoder_intervention_report(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    feature_match: PlantedFeatureMatchReport,
    indices: t.Tensor | None = None,
    *,
    random_seed: int = 111,
    control_margin: float = 0.25,
) -> CrosscoderInterventionReport:
    """Compare targeted model-specific ablation against random/orthogonal controls."""

    if indices is None:
        indices = data.heldout_idx
    target_latent = feature_match.target_model_b_latent
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
        baseline_delta = model_behavior_delta_from_reconstruction(
            output.reconstructed_model_a,
            output.reconstructed_model_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )
        target_a, target_b = ablate_latents(output, model, [target_latent])
        target_delta = model_behavior_delta_from_reconstruction(
            target_a,
            target_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )

        target_pair = t.cat([model.decoder_a[target_latent], model.decoder_b[target_latent]])
        generator = t.Generator().manual_seed(random_seed)
        random_pair = t.randn(target_pair.shape, generator=generator)
        random_pair = random_pair / random_pair.norm().clamp_min(1e-12)
        random_pair = random_pair * target_pair.norm().clamp_min(1e-12)
        random_a = random_pair[: model.decoder_a.shape[-1]]
        random_b = random_pair[model.decoder_a.shape[-1] :]
        random_reconstructed_a = (
            output.reconstructed_model_a
            - output.model_a_feature_acts[:, target_latent : target_latent + 1] * random_a
        )
        random_reconstructed_b = (
            output.reconstructed_model_b
            - output.model_b_feature_acts[:, target_latent : target_latent + 1] * random_b
        )
        random_delta = model_behavior_delta_from_reconstruction(
            random_reconstructed_a,
            random_reconstructed_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )

        orthogonal_pair = random_pair - (
            (random_pair @ target_pair) / target_pair.pow(2).sum().clamp_min(1e-12)
        ) * target_pair
        orthogonal_pair = orthogonal_pair / orthogonal_pair.norm().clamp_min(1e-12)
        orthogonal_pair = orthogonal_pair * target_pair.norm().clamp_min(1e-12)
        orthogonal_a = orthogonal_pair[: model.decoder_a.shape[-1]]
        orthogonal_b = orthogonal_pair[model.decoder_a.shape[-1] :]
        orthogonal_reconstructed_a = (
            output.reconstructed_model_a
            - output.model_a_feature_acts[:, target_latent : target_latent + 1] * orthogonal_a
        )
        orthogonal_reconstructed_b = (
            output.reconstructed_model_b
            - output.model_b_feature_acts[:, target_latent : target_latent + 1] * orthogonal_b
        )
        orthogonal_delta = model_behavior_delta_from_reconstruction(
            orthogonal_reconstructed_a,
            orthogonal_reconstructed_b,
            data,
            target_feature_id=feature_match.target_model_b_feature_id,
        )

    positive = data.behavior_labels[indices]
    baseline_abs = baseline_delta[positive].abs().mean().item()
    target_abs = target_delta[positive].abs().mean().item()
    random_abs = random_delta[positive].abs().mean().item()
    orthogonal_abs = orthogonal_delta[positive].abs().mean().item()
    target_reduction = baseline_abs - target_abs
    random_reduction = baseline_abs - random_abs
    orthogonal_reduction = baseline_abs - orthogonal_abs
    return CrosscoderInterventionReport(
        target_latent=target_latent,
        baseline_abs_delta=baseline_abs,
        target_ablated_abs_delta=target_abs,
        same_norm_random_abs_delta=random_abs,
        orthogonal_abs_delta=orthogonal_abs,
        target_reduction=target_reduction,
        same_norm_random_reduction=random_reduction,
        orthogonal_reduction=orthogonal_reduction,
        passes_controls=(
            target_reduction > random_reduction + control_margin
            and target_reduction > orthogonal_reduction + control_margin
        ),
    )
# END SOLUTION

# EXERCISE
# def top_activating_examples(
#     model: SparseCrosscoder,
#     data: PlantedCrosscoderData,
#     feature_match: PlantedFeatureMatchReport,
#     indices: t.Tensor | None = None,
#     *,
#     k: int = 6,
# ) -> list[dict[str, float | str | int | bool]]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def top_activating_examples(
    model: SparseCrosscoder,
    data: PlantedCrosscoderData,
    feature_match: PlantedFeatureMatchReport,
    indices: t.Tensor | None = None,
    *,
    k: int = 6,
) -> list[dict[str, float | str | int | bool]]:
    """Return the top held-out examples for the matched model-B-specific latent."""

    if indices is None:
        indices = data.heldout_idx
    target_latent = feature_match.target_model_b_latent
    with t.no_grad():
        output = model(
            data.model_a_activations[indices],
            data.model_b_activations[indices],
        )
    scores = output.model_b_feature_acts[:, target_latent]
    top_local = scores.argsort(descending=True)[:k]
    rows: list[dict[str, float | str | int | bool]] = []
    for rank, local_index in enumerate(top_local.tolist(), start=1):
        global_index = int(indices[local_index].item())
        rows.append(
            {
                "rank": rank,
                "example_index": global_index,
                "prompt": data.prompts[global_index],
                "is_behavior_delta_positive": bool(data.behavior_labels[global_index].item()),
                "learned_activation": float(scores[local_index].item()),
                "true_model_b_feature": float(
                    data.true_latents[global_index, feature_match.target_model_b_feature_id].item()
                ),
                "behavior_delta": float(data.behavior_deltas[global_index].item()),
            }
        )
    return rows
# END SOLUTION


if MAIN:
    tests.test_targeted_ablation_beats_same_norm_and_orthogonal_controls(
        train_sparse_crosscoder,
        match_learned_to_planted_features,
        crosscoder_intervention_report,
        make_planted_crosscoder_data,
    )

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 4️⃣ Inspect the full evidence ladder
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Signature Result

### Composition exercise - Compose the Evidence Ladder

Put the ladder together. This cell should train a fresh crosscoder, show the training curve, recover latent ownership, compare behavior baselines, inspect top examples, and run
intervention controls. This is the result a reviewer should look at first.

<details>
<summary>Expected output</summary>

The contract should pass with held-out reconstruction MSE below `3e-4`, ownership accuracy `1.0`, minimum planted-feature correlation above `0.9`, learned target-latent behavior
AUC above `0.93`, label-shuffled AUC much lower, and target ablation beating both controls.
</details>

<details>
<summary>Interpretation</summary>

If reconstruction passes but ownership fails, you have a model-diffing autoencoder, not an interpretable crosscoder. If ownership and AUC pass but intervention fails, the feature
is predictive but not causal for this readout. Only the conjunction earns the section claim.
</details>

<details>
<summary>Worked solution</summary>

```python
def run_crosscoder_signature_result(
    *,
    steps: int = 500,
    n_latents: int = 6,
) -> dict[str, object]:
    """Train and evaluate the CPU learned-crosscoder signature result."""

    data = make_planted_crosscoder_data()
    trained = train_sparse_crosscoder(data, n_latents=n_latents, steps=steps)
    reconstruction = evaluate_sparse_crosscoder_reconstruction(
        trained.model,
        data,
        data.heldout_idx,
    )
    feature_match = match_learned_to_planted_features(
        trained.model,
        data,
        data.heldout_idx,
    )
    baselines = behavior_baseline_table(
        trained.model,
        data,
        feature_match,
        data.heldout_idx,
    )
    intervention = crosscoder_intervention_report(
        trained.model,
        data,
        feature_match,
        data.heldout_idx,
    )
    top_examples = top_activating_examples(
        trained.model,
        data,
        feature_match,
        data.heldout_idx,
    )
    learned_auc = next(
        float(row["behavior_auc"])
        for row in baselines
        if row["method"] == "learned sparse crosscoder target latent"
    )
    label_shuffle_auc = next(
        float(row["behavior_auc"])
        for row in baselines
        if row["method"] == "label-shuffled target latent"
    )
    contract_passed = (
        reconstruction.beats_zero_baseline
        and reconstruction.heldout_reconstruction_mse < 3e-4
        and feature_match.ownership_accuracy >= 0.99
        and feature_match.min_correlation > 0.9
        and learned_auc > 0.93
        and label_shuffle_auc < learned_auc - 0.2
        and intervention.passes_controls
    )
    return {
        "contract_passed": contract_passed,
        "accepted": contract_passed,
        "tests_passed": contract_passed,
        "dataset": {
            "n_examples": int(data.model_a_activations.shape[0]),
            "d_model": int(data.model_a_activations.shape[1]),
            "n_train": int(data.train_idx.numel()),
            "n_heldout": int(data.heldout_idx.numel()),
            "feature_names": list(data.feature_names),
            "feature_owners": list(data.feature_owners),
        },
        "training_history": trained.history,
        "reconstruction": reconstruction.__dict__,
        "feature_match": feature_match.__dict__,
        "baselines": baselines,
        "intervention": intervention.__dict__,
        "top_examples": top_examples,
    }
```
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Contract reminder

```python
# Uncomment after you have completed the previous exercises.
# tests.test_notebook_contract(run_smoke_test)
```
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# EXERCISE
# def run_crosscoder_signature_result(*, steps: int = 500, n_latents: int = 6) -> dict[str, object]:
#     raise NotImplementedError()
# END EXERCISE
# SOLUTION
def run_crosscoder_signature_result(
    *,
    steps: int = 500,
    n_latents: int = 6,
) -> dict[str, object]:
    """Train and evaluate the CPU learned-crosscoder signature result."""

    data = make_planted_crosscoder_data()
    trained = train_sparse_crosscoder(data, n_latents=n_latents, steps=steps)
    reconstruction = evaluate_sparse_crosscoder_reconstruction(
        trained.model,
        data,
        data.heldout_idx,
    )
    feature_match = match_learned_to_planted_features(
        trained.model,
        data,
        data.heldout_idx,
    )
    baselines = behavior_baseline_table(
        trained.model,
        data,
        feature_match,
        data.heldout_idx,
    )
    intervention = crosscoder_intervention_report(
        trained.model,
        data,
        feature_match,
        data.heldout_idx,
    )
    top_examples = top_activating_examples(
        trained.model,
        data,
        feature_match,
        data.heldout_idx,
    )
    learned_auc = next(
        float(row["behavior_auc"])
        for row in baselines
        if row["method"] == "learned sparse crosscoder target latent"
    )
    label_shuffle_auc = next(
        float(row["behavior_auc"])
        for row in baselines
        if row["method"] == "label-shuffled target latent"
    )
    contract_passed = (
        reconstruction.beats_zero_baseline
        and reconstruction.heldout_reconstruction_mse < 3e-4
        and feature_match.ownership_accuracy >= 0.99
        and feature_match.min_correlation > 0.9
        and learned_auc > 0.93
        and label_shuffle_auc < learned_auc - 0.2
        and intervention.passes_controls
    )
    return {
        "contract_passed": contract_passed,
        "accepted": contract_passed,
        "tests_passed": contract_passed,
        "n_heldout": int(data.heldout_idx.numel()),
        "heldout_reconstruction_mse": reconstruction.heldout_reconstruction_mse,
        "zero_reconstruction_mse": reconstruction.zero_reconstruction_mse,
        "ownership_accuracy": feature_match.ownership_accuracy,
        "min_correlation": feature_match.min_correlation,
        "learned_behavior_auc": learned_auc,
        "label_shuffle_behavior_auc": label_shuffle_auc,
        "target_reduction": intervention.target_reduction,
        "same_norm_random_reduction": intervention.same_norm_random_reduction,
        "orthogonal_reduction": intervention.orthogonal_reduction,
        "dataset": {
            "n_examples": int(data.model_a_activations.shape[0]),
            "d_model": int(data.model_a_activations.shape[1]),
            "n_train": int(data.train_idx.numel()),
            "n_heldout": int(data.heldout_idx.numel()),
            "feature_names": list(data.feature_names),
            "feature_owners": list(data.feature_owners),
        },
        "training_history": trained.history,
        "reconstruction": reconstruction.__dict__,
        "feature_match": feature_match.__dict__,
        "baselines": baselines,
        "intervention": intervention.__dict__,
        "top_examples": top_examples,
    }
# END SOLUTION


def run_smoke_test(cpu: bool = True) -> dict[str, object]:
    _ = cpu
    return run_crosscoder_signature_result(steps=300)

if MAIN:
    data = make_planted_crosscoder_data(n_examples=1024, d_model=12)
    trained = train_sparse_crosscoder(data, n_latents=6, steps=500, log_every=50)
    reconstruction = evaluate_sparse_crosscoder_reconstruction(trained.model, data)
    feature_match = match_learned_to_planted_features(trained.model, data)
    intervention = crosscoder_intervention_report(trained.model, data, feature_match)
    baselines = behavior_baseline_table(trained.model, data, feature_match)
    top_examples = top_activating_examples(trained.model, data, feature_match, k=8)

    print("contract-ready", reconstruction.beats_zero_baseline and feature_match.ownership_accuracy == 1.0 and intervention.passes_controls)
    print("heldout reconstruction mse", reconstruction.heldout_reconstruction_mse)
    print("ownership accuracy", feature_match.ownership_accuracy)
    print("target latent", feature_match.target_model_b_latent)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Training and reconstruction controls

The held-out curve must fall with the training curve and beat the zero-reconstruction baseline. Low training loss without held-out improvement would invalidate every later
interpretation.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    history_df = pd.DataFrame(trained.history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(history_df["step"], history_df["train_total_loss"], label="train total")
    axes[0].plot(history_df["step"], history_df["heldout_total_loss"], label="held-out total")
    axes[0].set_yscale("log")
    axes[0].set_title("Sparse crosscoder training")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(history_df["step"], history_df["heldout_reconstruction_mse"], color="#0f766e")
    axes[1].axhline(reconstruction.zero_reconstruction_mse, color="#7f1d1d", linestyle="--", label="zero baseline")
    axes[1].set_yscale("log")
    axes[1].set_title("Held-out reconstruction")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(asset_dir / "crosscoder_training_reconstruction_curves.png", dpi=180, bbox_inches="tight")
    plt.show()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/crosscoder_legacy_cell_18_output_0.png" width="1100">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Latent ownership

This table and heatmap compare every matched learned latent with the independent planted-feature oracle. Reconstruction alone cannot establish ownership.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Before opening the saved ownership heatmap, predict how splitting a planted feature across learned latents could affect ownership and reconstruction differently.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    with t.no_grad():
        heldout_output = trained.model(
            data.model_a_activations[data.heldout_idx],
            data.model_b_activations[data.heldout_idx],
        )
    ownership_rows = latent_ownership_table(heldout_output)
    ownership_df = pd.DataFrame(ownership_rows)
    match_df = pd.DataFrame({
        "planted_feature": feature_match.feature_names,
        "true_owner": feature_match.true_owners,
        "matched_latent": feature_match.matched_latents,
        "predicted_owner": feature_match.predicted_owners,
        "heldout_abs_corr": feature_match.correlations,
    })
    display(match_df)

    heat = ownership_df[["model_a_mean", "model_b_mean"]].to_numpy()
    row_labels = [f"latent {int(row.latent)} ({row.predicted_owner})" for row in ownership_df.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(heat, aspect="auto", cmap="viridis")
    ax.set_xticks([0, 1], ["model A mean", "model B mean"])
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title("Learned latent ownership")
    for row_id in range(heat.shape[0]):
        for col_id in range(heat.shape[1]):
            color = "white" if heat[row_id, col_id] < heat.max() * 0.45 else "black"
            ax.text(col_id, row_id, f"{heat[row_id, col_id]:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean activation")
    fig.tight_layout()
    fig.savefig(asset_dir / "crosscoder_latent_ownership_heatmap.png", dpi=180, bbox_inches="tight")
    plt.show()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/crosscoder_legacy_cell_20_output_1.png" width="1100">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Behavior baselines

The learned target latent is compared with zero reconstruction, shared-only latents, a dense SVD direction, and shuffled labels. The dashed line is chance ranking.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    baseline_df = pd.DataFrame(baselines)
    display(baseline_df)
    short_labels = ["learned target", "zero", "shared-only", "SVD delta", "label shuffle"]
    colors = ["#0f766e", "#94a3b8", "#64748b", "#2563eb", "#b91c1c"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    y = range(len(baseline_df))
    ax.barh(list(y), baseline_df["behavior_auc"], color=colors)
    ax.axvline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(0.45, 1.02)
    ax.set_yticks(list(y), short_labels)
    ax.invert_yaxis()
    ax.set_xlabel("signed held-out behavior AUC")
    ax.set_title("Behavior prediction requires controls")
    for row_id, auc in enumerate(baseline_df["behavior_auc"]):
        ax.text(min(float(auc) + 0.01, 1.0), row_id, f"{auc:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(asset_dir / "crosscoder_behavior_auc_baselines.png", dpi=180, bbox_inches="tight")
    plt.show()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/crosscoder_legacy_cell_22_output_1.png" width="1100">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Inspect the top held-out examples

The matched model-B-specific latent should fire on the planted technical feature and agree with the behavior delta. Inspecting rows exposes anomalies that aggregate AUC can hide.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    top_examples_df = pd.DataFrame(top_examples)
    display(top_examples_df)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Causal intervention controls

Targeted ablation must reduce the behavior delta more than same-norm random and orthogonal controls. Predictive AUC without this separation is not the causal result.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Forecast the target-versus-control predicate for the published main-study cohort before inspecting the intervention bars.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    intervention_df = pd.DataFrame([
        {"condition": "baseline", "abs_behavior_delta": intervention.baseline_abs_delta, "reduction": 0.0},
        {"condition": "target latent ablated", "abs_behavior_delta": intervention.target_ablated_abs_delta, "reduction": intervention.target_reduction},
        {"condition": "same-norm random", "abs_behavior_delta": intervention.same_norm_random_abs_delta, "reduction": intervention.same_norm_random_reduction},
        {"condition": "orthogonal control", "abs_behavior_delta": intervention.orthogonal_abs_delta, "reduction": intervention.orthogonal_reduction},
    ])
    display(intervention_df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(intervention_df["condition"], intervention_df["abs_behavior_delta"], color=["#64748b", "#0f766e", "#b91c1c", "#b91c1c"])
    axes[0].set_ylabel("positive examples mean |delta|")
    axes[0].set_title("Ablation effect")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(intervention_df["condition"][1:], intervention_df["reduction"][1:], color=["#0f766e", "#b91c1c", "#b91c1c"])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("delta reduction")
    axes[1].set_title("Target beats controls")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(asset_dir / "crosscoder_intervention_bars.png", dpi=180, bbox_inches="tight")
    plt.show()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/crosscoder_legacy_cell_26_output_1.png" width="1100">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try It Yourself

After the signature result passes, change exactly one of these and rerun:

- `play_l1 = 0.002`, `0.008`, or `0.02`;
- `play_seed = 2`, `7`, or `11`;
- `play_latents = 4`, `6`, or `8`.

Watch for the failure modes: dense reconstruction with weak ownership, high AUC with weak controls, or controls that accidentally hit the same direction.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Play suggestion

This commented alternative uses the original main-study-sized cohort and training budget. It is a distinct exploratory suggestion from the recorded play below, not an additional
executed result.

```python
# Play cell. Run this only after your implementation works.
# play_data = make_planted_crosscoder_data(n_examples=1024, d_model=12)
# play_trained = train_sparse_crosscoder(
#     play_data,
#     n_latents=6,
#     steps=500,
#     l1_coefficient=0.008,
#     seed=2,
# )
# play_match = match_learned_to_planted_features(play_trained.model, play_data)
# play_rows = behavior_baseline_table(play_trained.model, play_data, play_match)
# display(pd.DataFrame(play_rows))
# display(pd.DataFrame(top_activating_examples(play_trained.model, play_data, play_match, k=8)))
# crosscoder_intervention_report(play_trained.model, play_data, play_match)
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Choose one existing play knob before inspecting its output. Predict how changing sparsity pressure or latent count might alter ownership versus reconstruction; do not select a
favorable seed after inspection.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    play_l1 = 0.008
    play_seed = 2
    play_latents = 6
    play_steps = 300

    play_data = make_planted_crosscoder_data(n_examples=512, d_model=12)
    play_trained = train_sparse_crosscoder(
        play_data,
        n_latents=play_latents,
        steps=play_steps,
        l1_coefficient=play_l1,
        seed=play_seed,
        log_every=100,
    )
    play_match = match_learned_to_planted_features(play_trained.model, play_data)
    play_intervention = crosscoder_intervention_report(play_trained.model, play_data, play_match)
    print({
        "ownership_accuracy": play_match.ownership_accuracy,
        "min_correlation": play_match.min_correlation,
        "target_latent": play_match.target_model_b_latent,
        "target_reduction": play_intervention.target_reduction,
        "random_reduction": play_intervention.same_norm_random_reduction,
        "orthogonal_reduction": play_intervention.orthogonal_reduction,
    })
    display(pd.DataFrame(behavior_baseline_table(play_trained.model, play_data, play_match)))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# ☆ Bonus
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## GPU entry points

These wrappers are not executed in the learner notebook. They refresh the optional pinned TransformerLens preflight, which does not replace the visible learned-crosscoder result above.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def run_gpu_test(max_vram_gb: float = 24.0):
    """Optional parent-run CUDA preflight; this notebook does not call it."""

    import part4_crosscoders_model_diffing.solutions as solutions

    return solutions.run_gpu_test(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0):
    """Verification entry point for the parent GPU runner."""

    return run_gpu_test(max_vram_gb=max_vram_gb)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Anomaly Hunt

Try to produce one negative result and explain it. Good targets:

- too little L1 pressure gives many active latents and ambiguous ownership;
- too much L1 pressure hurts reconstruction before ownership becomes cleaner;
- too many latents can split one planted feature across several learned latents;
- an SVD direction can get decent AUC while still failing the sparse ownership story.

## Optional Real Paired-Model Evidence

The existing `solutions.run_gpu_test(max_vram_gb=24.0)` path loads pinned TransformerLens `gelu-1l` and `solu-1l` checkpoints and
runs a bounded paired residual-delta preflight. Treat that as real-model context, not the signature result of this notebook. The signature result above is the learned sparse
crosscoder with exact planted ground truth.

## Limitations

This section proves the method on a small planted model organism. It does not prove that a large base-vs-chat crosscoder has clean features, that dense SVD directions are useless,
or that the optional TransformerLens preflight is a full trained-crosscoder replication. Those are follow-up experiments for the parent GPU pass.
'''

