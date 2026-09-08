"""Rebuild the paired 14.3 notebooks and their instruction-page export."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import textwrap
from pathlib import Path
from pprint import pformat

import solutions as reference


SECTION_DIR = Path(__file__).resolve().parent
EXERCISE_PATH = SECTION_DIR / "14.3_V_JEPA_from_Scratch_exercises.ipynb"
SOLUTION_PATH = SECTION_DIR / "14.3_V_JEPA_from_Scratch_solutions.ipynb"
PAGE_PATH = SECTION_DIR.parents[1] / "instructions" / "pages" / "03_[14.3]_V_JEPA_from_Scratch.md"


def clean(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def source_for(*names: str) -> str:
    return "\n\n".join(inspect.getsource(getattr(reference, name)).rstrip() for name in names) + "\n"


def solution_details(*names: str) -> str:
    return (
        "<details>\n<summary>Solution</summary>\n\n```python\n"
        + source_for(*names)
        + "```\n\n</details>\n"
    )


def complete_exercise(source: str) -> str:
    """Fill only learner stubs; keep provided code and immediate tests identical."""

    source = clean(source)
    lines = source.splitlines(keepends=True)
    replacements = []
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
        if "YOUR CODE HERE" in "".join(lines[start:node.end_lineno]):
            replacements.append((start, node.end_lineno, source_for(node.name)))
    for start, end, answer in reversed(replacements):
        lines[start:end] = [answer]
    return "".join(lines)


def cell_id(kind: str, source: str) -> str:
    return hashlib.sha1(f"{kind}:{source}".encode()).hexdigest()[:12]


def markdown(source: str) -> dict[str, object]:
    source = clean(source)
    return {"cell_type": "markdown", "id": cell_id("markdown", source), "metadata": {}, "source": source.splitlines(True)}


EXERCISE_GUIDANCE = {
    1: ("Every forward and reversed collision count is zero.",
        "Key equality must compare all four coordinates, not just color or state."),
    2: ("A batch of two clips becomes (2, 48, d_model) tokens on a (3, 4, 4) grid.",
        "Check divisibility before Conv3D; flatten time, y, x in raster order."),
    3: ("The training mask has 12 targets per clip; the future mask has 16.",
        "Sample a spatial origin per clip, then broadcast the same block across time."),
    4: ("Visible context has 36 tokens; replacing masked pixels changes none of them.",
        "Add position vectors before packing. Equal visible counts permit a batched reshape."),
    5: ("Predictions and targets have shape (batch, 12, 24); only predictions require gradients.",
        "Scatter projected context into visible slots, use mask tokens elsewhere, then select targets."),
    6: ("For target=0, online=2, momentum=0.75, EMA gives 0.5. L1 averages absolute errors.",
        "Detach target features, leave predictions unnormalized, and update matching parameter names."),
    7: ("The short-run test reduces L1 by at least 20%; the three-seed run comes later.",
        "Clear gradients, compute L1, backpropagate, clip and step online weights, then update EMA."),
    8: ("State features have 96 columns and rank at most 72; constant features have rank zero.",
        "Compute rank entropy from normalized squared singular values; handle the zero matrix explicitly."),
    9: ("An exact one-hot fixture is classified perfectly; a permuted label mapping fails.",
        "Append a bias column and solve the regularized normal equations; do not penalize the bias."),
    10: ("Two equal top scores give 1/2 credit only when the true key is one of them.",
         "Match keys first. Reordering the gallery must not change accuracy."),
    11: ("Exact dynamics reproduces every future pixel; blank-future keeps only frames 0–3.",
         "The final two frames form one temporal tubelet. Normalize teacher features before selecting it."),
    12: ("Reversing keys twice recovers the original keys; all required intersections are empty.",
         "The reversed start is the forward final position, and the velocity changes sign."),
    13: ("The fit subset has equal counts for all eight states; score color, direction, and state separately.",
         "Fit on training features only. Derive reverse physical labels before computing accuracy."),
    14: ("Five methods use the same 128-key gallery; exact dynamics has cosine and retrieval one.",
         "Encode pixel baselines through the frozen teacher. Do not substitute a different gallery per method."),
    15: ("The dependency check rejects imported method implementations; each interval is mean and observed range.",
         "Compose your tested functions. Audit split and reversal keys before interpreting any probe score."),
}


def exercise_markdown(source: str) -> dict[str, object]:
    number = int(re.search(r"### Exercise (\d+):", source)[1])
    expected, help_text = EXERCISE_GUIDANCE[number]
    additions = []
    if "<summary>Expected output</summary>" not in source:
        additions.append(f"<details><summary>Expected output</summary>\n\n{expected}\n\n</details>")
    if "<summary>Help</summary>" not in source:
        additions.append(f"<details><summary>Help</summary>\n\n{help_text}\n\n</details>")
    marker = source.index("<summary>Solution</summary>")
    insert_at = source.rfind("<details>", 0, marker)
    source = source[:insert_at] + "\n\n".join(additions) + "\n\n" + source[insert_at:]
    return markdown(source)


def code(source: str) -> dict[str, object]:
    source = clean(source)
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id("code", source),
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


INTRO = """
# [14.3] V-JEPA from Scratch: A Persistence-Shortcut Lab

This notebook asks one deliberately adversarial question: **does a paper-mechanics V-JEPA trained on a tiny moving-object world learn enough dynamics to beat persistence on an unseen future mask?**

You will implement the released method's core mechanics, then try to falsify the stronger world-model claim.
The checked result is a negative result: this toy protocol can verify masked latent learning, but it does not
establish dynamics beyond copy-last-frame.

The evidence chain is visible and learner-owned:

> canonical clips -> collision-free splits -> tubelets -> spatial tube masks -> stop-gradient target -> L1 predictor -> EMA teacher -> future-mask stress test

The future-only mask is explicitly out of distribution relative to pretraining. Exact dynamics is an oracle,
persistence is the serious shortcut baseline, and all predictors are scored in the same EMA-teacher space. A
blank-future control tests whether the teacher's final-slice features retain context even when every future
pixel is erased.
"""


OBJECTIVES = """
## Learning Objectives

You will be able to:

- construct unique clip keys and physically canonical reversed keys;
- exclude forward and reversed collisions from pretraining, probe-fit, and held-out splits;
- implement Conv3D tubelets and spatial blocks repeated through time;
- verify that masked pixels cannot enter the online encoder;
- implement stop-gradient, target layer normalization, plain L1 prediction, and EMA;
- write and test the full `train_small_vjepa` loop yourself;
- separate an in-distribution spatial-tube task from an out-of-distribution future-mask stress test;
- fit balanced probe suites and score temporal reversal under physical labels;
- report tie-aware retrieval and tie counts against persistence, exact dynamics, and random controls;
- prove why the redundant 96-column temporal summary has effective-rank maximum 72.
"""


SETUP = """
import copy
import hashlib
import inspect
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch as t
import torch.nn.functional as F
from IPython.display import Markdown, display
from torch import Tensor, nn

chapter = "chapter14_jepa_world_models"
section = "part3_vjepa_from_scratch"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
section_dir = root_dir / chapter / "exercises" / section
assets_dir = root_dir / chapter / "instructions" / "assets"
if str(section_dir) not in sys.path:
    sys.path.insert(0, str(section_dir))

import tests
import solutions as reference
from solutions import (
    CollapseDiagnostics,
    DuplicateAudit,
    MetricInterval,
    PredictorMetrics,
    RetrievalDetails,
    ReversalCollisionAudit,
    VideoWorldBatch,
    VJEPAMetrics,
)

t.set_num_threads(min(8, os.cpu_count() or 1))
device = t.device(os.environ.get("VJEPA_DEVICE", "cpu"))
print(f"device={device}; torch={t.__version__}")

GT_TIER = "GT-1"
EXERCISE_ID = "14_3_vjepa_from_scratch"
DIFFICULTY = 4
IMPORTANCE = 5
EXPECTED_RUNTIME = "three 1,000-step reference seeds; typically under five minutes on CPU"
REQUIRES_GPU = False
EVALUATION_PROTOCOL_VERSION = "vjepa-14.3-paper-l1-tube-mask-v4"

DIRECTIONS = t.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=t.long)
REVERSED_DIRECTIONS = t.tensor([1, 0, 3, 2], dtype=t.long)
"""


WORLD_MD = f"""
### Exercise 1: Build Canonical, Collision-Free Video Worlds

A clip key is `(color, direction, start_x, start_y)`. Its physical reverse starts at the original final
position and moves in the opposite direction. Implement both key systems, then construct the held-out world so
forward and reversed test keys avoid pretraining keys and avoid each other.

<details><summary>Expected output</summary>

```text
train duplicates=0; test duplicates=0; train/test overlap=0
reversed/test and reversed/pretraining overlap=0
```
</details>

<details><summary>Help</summary>

For key `(c, d, x, y)`, the reversed start is `(x, y) + (frames - 1) * velocity[d]` modulo image size. Reversal is an involution: applying it twice must recover the original key.
</details>

{solution_details('reverse_canonical_keys', 'make_moving_object_batch', 'make_disjoint_worlds', 'audit_clip_splits')}
"""


WORLD_EXERCISE = """
def reverse_canonical_keys(clip_keys, *, frames, image_size, speed=1):
    # YOUR CODE HERE
    raise NotImplementedError()

def make_moving_object_batch(
    n_samples,
    *,
    frames=6,
    image_size=16,
    object_size=4,
    speed=1,
    seed=0,
    excluded_clip_keys=None,
    excluded_reverse_against_keys=None,
    disallow_reverse_pairs=False,
    device="cpu",
):
    # YOUR CODE HERE
    raise NotImplementedError()

def make_disjoint_worlds(train_samples, test_samples, *, seed=0, test_seed=None, device="cpu"):
    # YOUR CODE HERE
    raise NotImplementedError()

def audit_clip_splits(train_world, test_world):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_make_moving_object_batch_exact(make_moving_object_batch)
tests.test_make_moving_object_batch_rejects_invalid_world(make_moving_object_batch)
tests.test_disjoint_world_splits_and_duplicate_audit(make_disjoint_worlds, audit_clip_splits)
"""


WORLD_PREVIEW = """
_, world_preview = make_disjoint_worlds(64, 64, seed=0)
fig, axes = plt.subplots(1, 6, figsize=(12, 2.1), constrained_layout=True)
for frame_index, axis in enumerate(axes):
    axis.imshow(world_preview.videos[24, frame_index].cpu().permute(1, 2, 0))
    axis.set_title(f"t={frame_index}")
    axis.set_xticks([])
    axis.set_yticks([])
fig.suptitle(f"Exact toroidal clip: key={tuple(world_preview.clip_keys[24].tolist())}")
plt.show()
"""


TUBELET_MD = f"""
### Exercise 2: Turn Video Into Tubelet Tokens

Implement the grid calculation and a strided `Conv3d`. For `(6, 16, 16)` video and `(2, 4, 4)` tubelets, the grid is `(3, 4, 4)`, or 48 tokens.

<details><summary>Help</summary>

PyTorch `Conv3d` expects `(batch, channels, time, height, width)`. Reject indivisible dimensions instead of silently truncating them.
</details>

{solution_details('tubelet_grid', 'TubeletEmbedding')}
"""


TUBELET_EXERCISE = """
def tubelet_grid(video_shape, tubelet_size):
    # YOUR CODE HERE
    raise NotImplementedError()

class TubeletEmbedding(nn.Module):
    def __init__(self, in_channels=3, d_model=24, tubelet_size=(2, 4, 4)):
        super().__init__()
        # YOUR CODE HERE

    def forward(self, video):
        # YOUR CODE HERE
        raise NotImplementedError()

tests.test_tubelet_grid_and_embedding(tubelet_grid, TubeletEmbedding)
tests.test_tubelet_embedding_rejects_wrong_rank(tubelet_grid, TubeletEmbedding)
"""


MASK_MD = f"""
### Exercise 3: Implement V-JEPA Spatial Tube Masks

Sample one contiguous spatial target block and repeat exactly the same block through all temporal tubelets. Also implement a separate full-future mask used only for the later stress test.

> **Paper connection.** Figure 2 of [Bardes et al.](https://arxiv.org/abs/2404.08471) uses spatial blocks
> repeated across the temporal dimension. The [released mask
> generator](https://github.com/facebookresearch/jepa/blob/main/src/masks/multiblock3d.py) defaults its
> temporal prediction scale to the full clip. Our tiny version uses one block rather than the release's
> multi-block sampling.

<details><summary>Expected output</summary>

```text
each 2x2 spatial block contributes 3 * 2 * 2 = 12 target tubelets
all temporal mask slices are identical
```
</details>

{solution_details('spatiotemporal_sincos', 'sample_spatial_tube_mask', 'future_prediction_mask', 'expand_tubelet_mask')}
"""


MASK_EXERCISE = """
def spatiotemporal_sincos(grid, d_model, *, device):
    # YOUR CODE HERE
    raise NotImplementedError()

def sample_spatial_tube_mask(batch_size, grid, *, block_shape=(2, 2), generator=None, device="cpu"):
    # YOUR CODE HERE
    raise NotImplementedError()

def future_prediction_mask(batch_size, grid, *, future_steps=1, device="cpu"):
    # YOUR CODE HERE
    raise NotImplementedError()

def expand_tubelet_mask(target_mask, grid, tubelet_size):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_spatiotemporal_sincos_is_factorized(spatiotemporal_sincos)
tests.test_spatial_tube_masks_repeat_through_time(sample_spatial_tube_mask)
tests.test_future_prediction_mask_is_separate_diagnostic(future_prediction_mask)
tests.test_expand_tubelet_mask_matches_pixels(expand_tubelet_mask)
"""


ENCODER_MD = f"""
### Exercise 4: Encode Only Visible Context

Remove target tokens before the online transformer. Then prove masked-pixel invariance: replacing every masked pixel with random values must leave the visible representation exactly unchanged.

<details><summary>Interpretation</summary>

This semantic intervention catches target leakage that shape tests cannot. It works because the Conv3D tubelets do not overlap.
</details>

{solution_details('pack_visible_tokens', 'TokenEncoder', 'VideoEncoder')}
"""


ENCODER_EXERCISE = """
def pack_visible_tokens(tokens, target_mask):
    # YOUR CODE HERE
    raise NotImplementedError()

class TokenEncoder(nn.Module):
    def __init__(self, d_model=24, n_heads=3, depth=1, mlp_ratio=2):
        super().__init__()
        # YOUR CODE HERE

    def forward(self, tokens):
        # YOUR CODE HERE
        raise NotImplementedError()

class VideoEncoder(nn.Module):
    def __init__(self, d_model=24, n_heads=3, depth=1, tubelet_size=(2, 4, 4)):
        super().__init__()
        # YOUR CODE HERE

    def tokenize(self, video):
        # YOUR CODE HERE
        raise NotImplementedError()

    def encode_all(self, video):
        # YOUR CODE HERE
        raise NotImplementedError()

    def encode_visible(self, video, target_mask):
        # YOUR CODE HERE
        raise NotImplementedError()

tests.test_pack_visible_tokens_preserves_order(pack_visible_tokens)
tests.test_video_encoder_visible_shape(VideoEncoder)
tests.test_masked_pixels_cannot_change_visible_context(VideoEncoder)
"""


MODEL_MD = f"""
### Exercise 5: Assemble Online Encoder, Predictor, and Teacher

Reinsert learned mask tokens in the predictor, freeze a deep-copied target encoder, and layer-normalize the full target encoder output under `no_grad` before selecting target positions.

> **Paper connection.** The V-JEPA target sees the full unmasked clip; masking is applied after target encoding. The online encoder sees only context tokens. The target path is stop-gradient and follows the online path by EMA.

{solution_details('VideoPredictor', 'SmallVJEPA')}
"""


MODEL_EXERCISE = """
class VideoPredictor(nn.Module):
    def __init__(self, d_model=24, n_heads=3, depth=1):
        super().__init__()
        # YOUR CODE HERE

    def forward(self, visible, target_mask, positions):
        # YOUR CODE HERE
        raise NotImplementedError()

class SmallVJEPA(nn.Module):
    def __init__(self, d_model=24, n_heads=3, encoder_depth=1, predictor_depth=1, tubelet_size=(2, 4, 4)):
        super().__init__()
        # YOUR CODE HERE

    def forward(self, video, target_mask):
        # YOUR CODE HERE
        raise NotImplementedError()

tests.test_predictor_places_visible_and_mask_tokens(VideoPredictor)
tests.test_small_vjepa_forward_stops_target_grad(SmallVJEPA)
"""


OBJECTIVE_MD = f"""
### Exercise 6: Implement EMA and the Paper L1 Objective

Implement `target = momentum * target + (1 - momentum) * online` and mean absolute error from predictor outputs to detached, layer-normalized target features.

> **Paper connection.** The [paper architecture](https://arxiv.org/abs/2404.08471) labels this edge `L1` and
> stop-gradient; the [released training
> loop](https://github.com/facebookresearch/jepa/blob/main/app/vjepa/train.py) computes a power-1 absolute
> feature loss for the V-JEPA term. This lab uses that term alone. There is no variance or covariance penalty
> in the primary objective.

<details><summary>Common bugs</summary>

Normalizing the prediction as well as the target, using Smooth L1, letting gradients enter the teacher, or applying EMA before the online optimizer step.
</details>

{solution_details('ema_update', 'vjepa_l1_loss')}
"""


OBJECTIVE_EXERCISE = """
@t.no_grad()
def ema_update(target_encoder, context_encoder, momentum):
    # YOUR CODE HERE
    raise NotImplementedError()

def vjepa_l1_loss(predicted, target):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_ema_update_exact(ema_update)
tests.test_vjepa_l1_loss_is_exact_and_stops_target_grad(vjepa_l1_loss)
"""


TRAIN_MD = f"""
### Exercise 7: Write the Training Step and Loop

This is the P0 method exercise. Implement one optimizer/EMA step and `train_small_vjepa`; do not import either from `solutions.py`. The history has one key, `vjepa_l1`, because there is only one active loss.

<details><summary>Expected output</summary>

```text
history keys: {{'vjepa_l1'}}
short-run L1 decreases by at least 20 percent
the recorded training mask repeats through all three temporal slices
```
</details>

{solution_details('set_seed', 'vjepa_train_step', 'train_small_vjepa')}
"""


TRAIN_EXERCISE = """
def set_seed(seed):
    # YOUR CODE HERE
    raise NotImplementedError()

def vjepa_train_step(model, video, optimizer, *, generator, ema_momentum, block_shape=(2, 2)):
    # YOUR CODE HERE
    raise NotImplementedError()

def train_small_vjepa(
    train_world,
    *,
    steps=240,
    batch_size=32,
    learning_rate=2e-3,
    ema_momentum=0.97,
    block_shape=(2, 2),
    seed=0,
    model=None,
):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_training_updates_online_not_target_directly(train_small_vjepa, vjepa_train_step)
tests.test_training_contract_is_l1_only(train_small_vjepa, vjepa_train_step)
tests.test_short_training_improves_latent_objective(train_small_vjepa, SmallVJEPA)
"""


READOUT_MD = f"""
### Exercise 8: Diagnose Collapse and Prove the Rank Bound

Average spatial tokens separately at each of three temporal slices, flatten those `3 * 24 = 72` coordinates, then append `final - initial` as a useful but redundant 24-column summary.

The resulting matrix has 96 columns but maximum rank 72: every appended coordinate is exactly one final-slice coordinate minus one initial-slice coordinate. Feature-wise centering and nonzero rescaling preserve that linear dependence.

{solution_details('collapse_diagnostics', 'temporal_state_features', 'temporal_state_feature_rank_bound')}
"""


READOUT_EXERCISE = """
def collapse_diagnostics(features, *, min_std=0.05, min_rank=2.0):
    # YOUR CODE HERE
    raise NotImplementedError()

def temporal_state_features(tokens, grid):
    # YOUR CODE HERE
    raise NotImplementedError()

def temporal_state_feature_rank_bound(grid, d_model):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_collapse_diagnostics_exact(collapse_diagnostics)
tests.test_temporal_state_features_keep_order(temporal_state_features, temporal_state_feature_rank_bound)
"""


PROBE_BASE_MD = f"""
### Exercise 9: Implement Ridge Probes

Keep color, direction, and combined state separate. A perfect color probe is not evidence of motion understanding.

{solution_details('fit_ridge_probe', 'probe_accuracy')}
"""


PROBE_BASE_EXERCISE = """
def fit_ridge_probe(features, labels, n_classes, ridge=1e-2):
    # YOUR CODE HERE
    raise NotImplementedError()

def probe_accuracy(features, labels, weights):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_ridge_probe_and_random_labels(fit_ridge_probe, probe_accuracy)
"""


RETRIEVAL_MD = f"""
### Exercise 10: Implement Key-Matched, Tie-Aware Retrieval

Each query key must occur exactly once in the gallery. Award fractional credit when the true target shares the top score, and retain every per-query tie count for reporting.

{solution_details('retrieval_details', 'retrieval_accuracy')}
"""


RETRIEVAL_EXERCISE = """
def retrieval_details(query, gallery, query_keys, gallery_keys, *, tie_atol=1e-7):
    # YOUR CODE HERE
    raise NotImplementedError()

def retrieval_accuracy(query, gallery, query_keys, gallery_keys):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_retrieval_is_key_matched_and_tie_aware(retrieval_details, retrieval_accuracy)
"""


FEATURE_EVAL_MD = f"""
### Exercise 11: Encode Future Targets and Physical Baselines

The future-only mask is an out-of-distribution diagnostic, not the pretraining mask. Convert the learned predictor, blank-future, copy-last-frame, exact dynamics, and a random predictor into the same final-slice EMA-teacher latent space.

**Why blank the future?** The teacher attends to the entire clip before we select its final tokens. Those
tokens can identify a clip from the past. Keep frames 0–3, zero frames 4–5, and measure how much retrieval
survives. This control has no future image prediction at all.

The exact-dynamics oracle may read the generator's velocity labels; the learned predictor, blank-future, and persistence controls may only use visible pixels. An oracle score of one checks the evaluator, not a learned capability.

{solution_details('encode_state_features', 'future_teacher_latents', 'predict_future_features', 'blank_future_video', 'copy_last_frame_video', 'exact_dynamics_video', 'exact_dynamics_prediction', 'reverse_direction_labels')}
"""


FEATURE_EVAL_EXERCISE = """
@t.no_grad()
def encode_state_features(encoder, videos, batch_size=64):
    # YOUR CODE HERE
    raise NotImplementedError()

@t.no_grad()
def future_teacher_latents(encoder, videos, batch_size=64):
    # YOUR CODE HERE
    raise NotImplementedError()

@t.no_grad()
def predict_future_features(model, videos, batch_size=64):
    # YOUR CODE HERE
    raise NotImplementedError()

@t.no_grad()
def blank_future_video(world, context_frames=4):
    # YOUR CODE HERE
    raise NotImplementedError()

@t.no_grad()
def copy_last_frame_video(world, context_frames=4):
    # YOUR CODE HERE
    raise NotImplementedError()

@t.no_grad()
def exact_dynamics_video(world, context_frames=4):
    # YOUR CODE HERE
    raise NotImplementedError()

@t.no_grad()
def exact_dynamics_prediction(world):
    # YOUR CODE HERE
    raise NotImplementedError()

def reverse_direction_labels(direction_labels):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_encode_and_future_helpers_match_direct_model_calls(
    encode_state_features, future_teacher_latents, predict_future_features
)
tests.test_future_video_baselines_are_semantic(
    copy_last_frame_video, exact_dynamics_video, exact_dynamics_prediction
)
tests.test_reverse_direction_labels_are_physical(reverse_direction_labels)
tests.test_blank_future_removes_future_without_mutation(blank_future_video)
tests.test_future_teacher_features_can_retain_context(future_teacher_latents)
"""


REVERSAL_MD = f"""
### Exercise 12: Audit Physical Reversal Before Scoring It

Derive reversed keys from actual start positions and velocities. Reject the run unless reversed held-out keys are unique and have zero overlap with pretraining, the explicit probe-fit subset, and forward held-out keys.

{solution_details('physical_reverse_keys', 'audit_reversal_collisions')}
"""


REVERSAL_EXERCISE = """
def physical_reverse_keys(world):
    # YOUR CODE HERE
    raise NotImplementedError()

def audit_reversal_collisions(pretraining_world, probe_fit_keys, test_world):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_physical_reverse_keys_and_collision_audit(
    reverse_canonical_keys, physical_reverse_keys, audit_reversal_collisions
)
"""


PROBE_SUITE_MD = f"""
### Exercise 13: Build the Evaluation in Tested Probe Subfunctions

Select a balanced, explicit probe-fit subset; fit and score the three probes; then run shuffled-label, reversal, and random-encoder controls as separate functions.

{solution_details('balanced_probe_fit_indices', 'fit_probe_suite', 'score_probe_suite', 'shuffled_state_probe_accuracy', 'score_reversed_probes', '_model_shape', 'random_encoder_probe_scores')}
"""


PROBE_SUITE_EXERCISE = f"""
def balanced_probe_fit_indices(world, examples):
    # YOUR CODE HERE
    raise NotImplementedError()

def fit_probe_suite(features, world, fit_indices):
    # YOUR CODE HERE
    raise NotImplementedError()

def score_probe_suite(features, world, probes):
    # YOUR CODE HERE
    raise NotImplementedError()

def shuffled_state_probe_accuracy(train_features, train_world, fit_indices, test_features, test_world, *, seed):
    # YOUR CODE HERE
    raise NotImplementedError()

@t.no_grad()
def score_reversed_probes(encoder, test_world, probes):
    # YOUR CODE HERE
    raise NotImplementedError()

{source_for('_model_shape').rstrip()}

@t.no_grad()
def random_encoder_probe_scores(model, train_world, test_world, fit_indices, *, seed):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_probe_suite_progression(balanced_probe_fit_indices, fit_probe_suite, score_probe_suite)
"""


PREDICTOR_SCORE_MD = f"""
### Exercise 14: Score Predictor Controls in One Space

Use an independently initialized predictor receiving the trained visible context. For every method, report matched teacher cosine, tie-aware retrieval, number of tied queries, and maximum tie size.

{solution_details('random_predictor_features', 'score_future_predictors')}
"""


PREDICTOR_SCORE_EXERCISE = """
@t.no_grad()
def random_predictor_features(model, videos, *, seed, batch_size=64):
    # YOUR CODE HERE
    raise NotImplementedError()

@t.no_grad()
def score_future_predictors(model, world, *, seed):
    # YOUR CODE HERE
    raise NotImplementedError()

tests.test_random_predictor_and_shared_teacher_scoring(random_predictor_features, score_future_predictors)
"""


ASSEMBLY_MD = f"""
### Exercise 15: Assemble and Validate the Protocol

The final evaluator now composes the tested pieces above. The dependency validator checks object identity and required call edges, so the long final run cannot silently fall back to imported reference functions.

{solution_details('evaluate_small_vjepa', '_flatten_numeric_metrics', 'summarize_multiseed', 'evaluation_fingerprint', 'validate_learner_pipeline_dependencies')}
"""


DEPENDENCY_GRAPH_SOURCE = f"LEARNER_DEPENDENCY_GRAPH = {reference.LEARNER_DEPENDENCY_GRAPH!r}\n"


ASSEMBLY_EXERCISE = f"""
@t.no_grad()
def evaluate_small_vjepa(model, train_world, test_world, history, *, seed=0, probe_train_examples=64):
    # YOUR CODE HERE
    raise NotImplementedError()

{source_for('_flatten_numeric_metrics').rstrip()}

def summarize_multiseed(metrics_by_seed):
    # YOUR CODE HERE
    raise NotImplementedError()

def evaluation_fingerprint(metrics):
    # YOUR CODE HERE
    raise NotImplementedError()

{DEPENDENCY_GRAPH_SOURCE.rstrip()}

{source_for('validate_learner_pipeline_dependencies').rstrip()}

tests.test_complete_evaluator_intervals_and_fingerprint(
    evaluate_small_vjepa, summarize_multiseed, evaluation_fingerprint
)
tests.test_dependency_validator_rejects_reference_fallback()
"""


SIGNATURE_MD = """
## Signature Result: The Persistence-Shortcut Stress Test

Run three complete CPU seeds. The image row selects the worst seed-0 future-mask retrieval by true rank and
matched cosine. Bars show means and observed ranges, not confidence intervals. Predictor differences are
descriptive only; three seeds do not support a population claim.
"""


SIGNATURE_RUN = """
EVAL_SEEDS = (0, 1, 2)
validate_learner_pipeline_dependencies(globals(), reference)

metrics_by_seed = {}
seed_zero_run = None
for seed in EVAL_SEEDS:
    train_world, test_world = make_disjoint_worlds(512, 128, seed=seed, device=device)
    model, history = train_small_vjepa(train_world, steps=1_000, batch_size=64, seed=seed)
    metrics = evaluate_small_vjepa(
        model, train_world, test_world, history, seed=seed, probe_train_examples=128
    )
    metrics_by_seed[seed] = metrics
    if seed == 0:
        seed_zero_run = (model, history, metrics, train_world, test_world)

intervals = summarize_multiseed(metrics_by_seed)
model, history, metrics, train_world, test_world = seed_zero_run

rows = [
    ("final / initial V-JEPA L1", "train_loss_ratio"),
    ("color probe", "color_probe_accuracy"),
    ("direction probe", "direction_probe_accuracy"),
    ("combined-state probe", "state_probe_accuracy"),
    ("random-encoder direction", "random_encoder_direction_accuracy"),
    ("random-encoder state", "random_encoder_state_accuracy"),
    ("reversed / original state", "reversed_original_state_accuracy"),
    ("reversed / physical state", "reversed_physical_state_accuracy"),
    ("future-mask V-JEPA cosine", "predictors.vjepa.teacher_latent_cosine"),
    ("blank-future cosine", "predictors.blank_future.teacher_latent_cosine"),
    ("copy-last-frame cosine", "predictors.copy_last_frame.teacher_latent_cosine"),
    ("exact-dynamics cosine", "predictors.exact_dynamics.teacher_latent_cosine"),
    ("random-predictor cosine", "predictors.random_predictor.teacher_latent_cosine"),
    ("future-mask V-JEPA retrieval", "predictors.vjepa.retrieval_accuracy"),
    ("blank-future retrieval", "predictors.blank_future.retrieval_accuracy"),
    ("copy-last-frame retrieval", "predictors.copy_last_frame.retrieval_accuracy"),
    ("exact-dynamics retrieval", "predictors.exact_dynamics.retrieval_accuracy"),
    ("random-predictor retrieval", "predictors.random_predictor.retrieval_accuracy"),
    ("effective rank / 72", "effective_rank"),
]
table = ["| metric | three-seed mean | observed range |", "|---|---:|---:|"]
for label, name in rows:
    interval = intervals[name]
    table.append(f"| {label} | {interval.mean:.3f} | [{interval.minimum:.3f}, {interval.maximum:.3f}] |")
display(Markdown("\\n".join(table)))

tie_table = ["| method | tied queries mean [range] | maximum tie size [range] |", "|---|---:|---:|"]
for label, key in (
    ("future-mask V-JEPA", "vjepa"), ("blank future", "blank_future"),
    ("copy last", "copy_last_frame"), ("exact", "exact_dynamics"), ("random", "random_predictor"),
):
    tied = intervals[f"predictors.{key}.tied_queries"]
    maximum = intervals[f"predictors.{key}.max_tie_count"]
    tie_table.append(
        f"| {label} | {tied.mean:.1f} [{tied.minimum:.0f}, {tied.maximum:.0f}] | "
        f"{maximum.mean:.1f} [{maximum.minimum:.0f}, {maximum.maximum:.0f}] |"
    )
display(Markdown("\\n".join(tie_table)))

audit_table = [
    "| seed | train/test overlap | reversed/pretrain | reversed/probe-fit | reversed/test | collision free |",
    "|---:|---:|---:|---:|---:|:---:|",
]
for seed, seed_metrics in metrics_by_seed.items():
    duplicate = seed_metrics.duplicate_audit
    reversal = seed_metrics.reversal_audit
    audit_table.append(
        f"| {seed} | {duplicate.train_test_overlap} | {reversal.reversed_pretraining_overlap} | "
        f"{reversal.reversed_probe_fit_overlap} | {reversal.reversed_test_overlap} | {reversal.collision_free} |"
    )
display(Markdown("\\n".join(audit_table)))

for seed_metrics in metrics_by_seed.values():
    assert seed_metrics.duplicate_audit.is_disjoint
    assert seed_metrics.reversal_audit.collision_free
    assert seed_metrics.effective_rank_max == 72
    assert seed_metrics.predictors["exact_dynamics"].teacher_latent_cosine >= 0.999999
    assert seed_metrics.predictors["exact_dynamics"].retrieval_accuracy == 1.0

predicted, target = predict_future_features(model, test_world.videos)
retrieval = retrieval_details(predicted, target, test_world.clip_keys, test_world.clip_keys)
signature_path = assets_dir / "vjepa_from_scratch_signature.png"
fig = make_signature_figure(history, metrics, test_world, retrieval, intervals)
fingerprint_payload = {
    "seed_metrics": asdict(metrics),
    "intervals": {name: asdict(interval) for name, interval in intervals.items()},
}
fig.savefig(
    signature_path,
    dpi=180,
    bbox_inches="tight",
    metadata={
        "EvaluationProtocol": EVALUATION_PROTOCOL_VERSION,
        "EvaluationFingerprint": evaluation_fingerprint(fingerprint_payload),
    },
)
plt.show()
"""


SIGNATURE_INTERPRETATION = """
<details open>
<summary>How to interpret the checked result</summary>

- Across seeds 0, 1, and 2, final/initial L1 is `0.388 [0.343, 0.456]`.
- Future-mask V-JEPA reaches cosine `0.889 [0.838, 0.946]` and retrieval `0.531 [0.414, 0.672]`; persistence reaches cosine `0.922 [0.898, 0.964]` and retrieval `0.703 [0.609, 0.852]`.
- Blank-future reaches cosine `0.913 [0.880, 0.959]` and retrieval `0.721 [0.602, 0.945]` despite having no future pixels. This is the strongest warning against interpreting clip retrieval as future prediction.
- Exact dynamics is `1.000` on both metrics. Random-predictor retrieval is `0.005 [0.000, 0.008]`.
- Effective rank is `9.27 [7.61, 10.97]` out of the attainable 72. Every collision count and tied-query count is zero in all checked seeds.
- The training curve verifies optimization of the paper L1 term under spatial tube masking; it does not by itself verify future extrapolation.
- The future-mask V-JEPA bar is an out-of-distribution stress test. Compare it only with controls scored in the same teacher space.
- Blank-future retrieval measures surviving clip identity without any future pixels. High retrieval on this control exposes context carryover; neither teacher cosine nor clip retrieval isolates predictive dynamics.
- Exact dynamics must reach the teacher target exactly. Persistence is the relevant shortcut, not a straw baseline.
- Tie counts are part of every retrieval claim. Zero split collisions do not imply zero representation ties.
- Reversal metrics are admissible only because the canonical-key audit is collision-free against pretraining and probe-fit keys.
- The observed V-JEPA-minus-persistence differences across three seeds are descriptive, not a confidence interval or an aggregate superiority claim.

The lab's conclusion remains negative. A positive follow-up needs to beat persistence and blank-future
controls in a preregistered setting whose targets distinguish futures for matched contexts; improving these
clip-level metrics alone does not establish dynamics.

</details>
"""


PLAY_MD = """
## Try It Yourself

Switch between the in-distribution spatial tube mask and the out-of-distribution future mask, then inspect one query, its target pixels, its retrieved clip, true rank, and tie count.

After inspecting a query, compare its last two real frames with the same clip's blank-future and copy-last-frame controls. None of the teacher-space controls produces a decoded video.
"""


PLAY_CODE = """
play_mode = "tube"  # "tube" or "future"
play_block = (2, 2)
play_query_index = 7

_, play_grid = model.target_encoder.encode_all(test_world.videos[:1])
if play_mode == "tube":
    one_mask = sample_spatial_tube_mask(
        1,
        play_grid,
        block_shape=play_block,
        generator=t.Generator().manual_seed(17),
        device=device,
    )
elif play_mode == "future":
    one_mask = future_prediction_mask(1, play_grid, device=device)
else:
    raise ValueError("play_mode must be 'tube' or 'future'.")

gallery_mask = one_mask.expand(len(test_world.videos), -1).clone()
with t.no_grad():
    gallery_prediction, gallery_target, _ = model(test_world.videos, gallery_mask)
play_retrieval = retrieval_details(
    gallery_prediction.flatten(1),
    gallery_target.flatten(1),
    test_world.clip_keys,
    test_world.clip_keys,
)
top1_index = int(play_retrieval.top1_indices[play_query_index])
pixel_mask = expand_tubelet_mask(one_mask, play_grid, model.context_encoder.tubelet.tubelet_size)[0]

fig, axes = plt.subplots(3, 6, figsize=(12, 6.2), constrained_layout=True)
for frame_index in range(6):
    axes[0, frame_index].imshow(test_world.videos[play_query_index, frame_index].cpu().permute(1, 2, 0))
    overlay = t.zeros(*pixel_mask[frame_index].shape, 4)
    overlay[..., 0] = 1.0
    overlay[..., 3] = pixel_mask[frame_index].cpu().float() * 0.55
    axes[0, frame_index].imshow(overlay)
    axes[0, frame_index].set_title(f"masked t={frame_index}")

    target_pixels = test_world.videos[play_query_index, frame_index] * pixel_mask[frame_index].to(device)
    axes[1, frame_index].imshow(target_pixels.cpu().permute(1, 2, 0))
    axes[1, frame_index].set_title(f"target t={frame_index}")

    axes[2, frame_index].imshow(test_world.videos[top1_index, frame_index].cpu().permute(1, 2, 0))
    axes[2, frame_index].set_title(f"top-1 t={frame_index}")

for axis in axes.flat:
    axis.set_xticks([])
    axis.set_yticks([])
fig.suptitle(
    f"mode={play_mode}; query={tuple(test_world.clip_keys[play_query_index].tolist())}; "
    f"top-1={tuple(test_world.clip_keys[top1_index].tolist())}; "
    f"rank={int(play_retrieval.true_ranks[play_query_index])}; "
    f"ties={int(play_retrieval.tie_counts[play_query_index])}; "
    f"cosine={float(play_retrieval.matched_cosines[play_query_index]):.3f}"
)
plt.show()

play_controls = {
    "true future": test_world.videos,
    "blank future": blank_future_video(test_world),
    "copy last": copy_last_frame_video(test_world),
}
fig, axes = plt.subplots(3, 2, figsize=(5, 6), constrained_layout=True)
for row, (name, video) in enumerate(play_controls.items()):
    for offset, frame in enumerate((4, 5)):
        axes[row, offset].imshow(video[play_query_index, frame].cpu().permute(1, 2, 0))
        axes[row, offset].set_title(f"{name}, t={frame}")
        axes[row, offset].axis("off")
plt.show()
"""


ENDING = """
<details>
<summary>Interpretation guide</summary>

- Strong tube-mask retrieval with weak future-mask retrieval indicates interpolation without demonstrated extrapolation.
- If persistence matches or wins, keep the negative conclusion; do not rescue the claim with probe accuracy.
- If physical reversal does not outperform original-label reversal, inspect static position shortcuts.
- Inspect ties and wraparound failures rather than averaging them away.

</details>

## Further Research / Anomaly Hunting

A positive follow-up needs a preregistered generated process where the visible context contains a predictive
latent state but a frozen final frame does not. Specify the process, metric, persistence margin, seed count,
and stopping rule before training. That is a new experiment, not evidence supplied by this notebook.

## Limitations

- This is a tiny one-object toroidal world with four constant velocities, no occlusion, and no interaction.
- Spatial tube pretraining and full-future evaluation use different mask distributions by design.
- The primary method has no variance or covariance penalty; collapse diagnostics observe the result but do not prevent collapse.
- A random encoder can expose appearance and position, so trained-minus-random probe margins remain descriptive.
- The 96-column readout has an attainable rank bound of 72 because its final 24 columns are algebraically dependent.
- Three seeds provide observed ranges only, not uncertainty estimates.
- Teacher final-slice features attend to the context. Blank-future retrieval tests that shortcut; a matched-context counterfactual-future metric is still needed for a positive dynamics claim.
- This notebook does not reproduce V-JEPA scale, multi-block masking, data augmentation, optimizer schedules, or downstream attentive probes.

## Reading

- [V-JEPA paper](https://arxiv.org/abs/2404.08471)
- [Official V-JEPA repository](https://github.com/facebookresearch/jepa)
- [Released V-JEPA training loop](https://github.com/facebookresearch/jepa/blob/main/app/vjepa/train.py)
- [Released 3D mask generator](https://github.com/facebookresearch/jepa/blob/main/src/masks/multiblock3d.py)
"""


def build_notebook(path: Path, *, solved: bool) -> dict[str, object]:
    notebook = json.loads(path.read_text())
    choose = lambda names, tests_source, exercise, prefix="": (
        complete_exercise(exercise) if solved else exercise
    )

    world_code = choose(
        ("reverse_canonical_keys", "make_moving_object_batch", "make_disjoint_worlds", "audit_clip_splits"),
        """tests.test_make_moving_object_batch_exact(make_moving_object_batch)
tests.test_make_moving_object_batch_rejects_invalid_world(make_moving_object_batch)
tests.test_disjoint_world_splits_and_duplicate_audit(make_disjoint_worlds, audit_clip_splits)""",
        WORLD_EXERCISE,
    )
    tubelet_code = choose(
        ("tubelet_grid", "TubeletEmbedding"),
        """tests.test_tubelet_grid_and_embedding(tubelet_grid, TubeletEmbedding)
tests.test_tubelet_embedding_rejects_wrong_rank(tubelet_grid, TubeletEmbedding)""",
        TUBELET_EXERCISE,
    )
    mask_code = choose(
        ("spatiotemporal_sincos", "sample_spatial_tube_mask", "future_prediction_mask", "expand_tubelet_mask"),
        """tests.test_spatiotemporal_sincos_is_factorized(spatiotemporal_sincos)
tests.test_spatial_tube_masks_repeat_through_time(sample_spatial_tube_mask)
tests.test_future_prediction_mask_is_separate_diagnostic(future_prediction_mask)
tests.test_expand_tubelet_mask_matches_pixels(expand_tubelet_mask)""",
        MASK_EXERCISE,
    )
    encoder_code = choose(
        ("pack_visible_tokens", "TokenEncoder", "VideoEncoder"),
        """tests.test_pack_visible_tokens_preserves_order(pack_visible_tokens)
tests.test_video_encoder_visible_shape(VideoEncoder)
tests.test_masked_pixels_cannot_change_visible_context(VideoEncoder)""",
        ENCODER_EXERCISE,
    )
    model_code = choose(
        ("VideoPredictor", "SmallVJEPA"),
        """tests.test_predictor_places_visible_and_mask_tokens(VideoPredictor)
tests.test_small_vjepa_forward_stops_target_grad(SmallVJEPA)""",
        MODEL_EXERCISE,
    )
    objective_code = choose(
        ("ema_update", "vjepa_l1_loss"),
        """tests.test_ema_update_exact(ema_update)
tests.test_vjepa_l1_loss_is_exact_and_stops_target_grad(vjepa_l1_loss)""",
        OBJECTIVE_EXERCISE,
    )
    train_code = choose(
        ("set_seed", "vjepa_train_step", "train_small_vjepa"),
        """tests.test_training_updates_online_not_target_directly(train_small_vjepa, vjepa_train_step)
tests.test_training_contract_is_l1_only(train_small_vjepa, vjepa_train_step)
tests.test_short_training_improves_latent_objective(train_small_vjepa, SmallVJEPA)""",
        TRAIN_EXERCISE,
    )
    readout_code = choose(
        ("collapse_diagnostics", "temporal_state_features", "temporal_state_feature_rank_bound"),
        """tests.test_collapse_diagnostics_exact(collapse_diagnostics)
tests.test_temporal_state_features_keep_order(temporal_state_features, temporal_state_feature_rank_bound)""",
        READOUT_EXERCISE,
    )
    probe_base_code = choose(
        ("fit_ridge_probe", "probe_accuracy"),
        "tests.test_ridge_probe_and_random_labels(fit_ridge_probe, probe_accuracy)",
        PROBE_BASE_EXERCISE,
    )
    retrieval_code = choose(
        ("retrieval_details", "retrieval_accuracy"),
        "tests.test_retrieval_is_key_matched_and_tie_aware(retrieval_details, retrieval_accuracy)",
        RETRIEVAL_EXERCISE,
    )
    feature_eval_code = choose(
        ("encode_state_features", "future_teacher_latents", "predict_future_features", "copy_last_frame_video", "exact_dynamics_video", "exact_dynamics_prediction", "reverse_direction_labels"),
        """tests.test_encode_and_future_helpers_match_direct_model_calls(
    encode_state_features, future_teacher_latents, predict_future_features
)
tests.test_future_video_baselines_are_semantic(
    copy_last_frame_video, exact_dynamics_video, exact_dynamics_prediction
)
tests.test_reverse_direction_labels_are_physical(reverse_direction_labels)""",
        FEATURE_EVAL_EXERCISE,
    )
    reversal_code = choose(
        ("physical_reverse_keys", "audit_reversal_collisions"),
        """tests.test_physical_reverse_keys_and_collision_audit(
    reverse_canonical_keys, physical_reverse_keys, audit_reversal_collisions
)""",
        REVERSAL_EXERCISE,
    )
    probe_suite_code = choose(
        ("balanced_probe_fit_indices", "fit_probe_suite", "score_probe_suite", "shuffled_state_probe_accuracy", "score_reversed_probes", "_model_shape", "random_encoder_probe_scores"),
        "tests.test_probe_suite_progression(balanced_probe_fit_indices, fit_probe_suite, score_probe_suite)",
        PROBE_SUITE_EXERCISE,
    )
    predictor_score_code = choose(
        ("random_predictor_features", "score_future_predictors"),
        "tests.test_random_predictor_and_shared_teacher_scoring(random_predictor_features, score_future_predictors)",
        PREDICTOR_SCORE_EXERCISE,
    )
    assembly_prefix = DEPENDENCY_GRAPH_SOURCE + "\n" + source_for("validate_learner_pipeline_dependencies") + "\n"
    assembly_code = choose(
        ("evaluate_small_vjepa", "_flatten_numeric_metrics", "summarize_multiseed", "evaluation_fingerprint"),
        """tests.test_complete_evaluator_intervals_and_fingerprint(
    evaluate_small_vjepa, summarize_multiseed, evaluation_fingerprint
)
tests.test_dependency_validator_rejects_reference_fallback()""",
        ASSEMBLY_EXERCISE,
        assembly_prefix,
    )

    notebook["cells"] = [
        markdown(INTRO), markdown(OBJECTIVES), code(SETUP), markdown("## Part 1: Implement Paper Mechanics"),
        exercise_markdown(WORLD_MD), code(world_code), code(WORLD_PREVIEW),
        markdown("![Expected exact video world](../../instructions/assets/vjepa_from_scratch_world.png)"),
        exercise_markdown(TUBELET_MD), code(tubelet_code), exercise_markdown(MASK_MD), code(mask_code),
        exercise_markdown(ENCODER_MD), code(encoder_code), exercise_markdown(MODEL_MD), code(model_code),
        exercise_markdown(OBJECTIVE_MD), code(objective_code), exercise_markdown(TRAIN_MD), code(train_code),
        exercise_markdown(READOUT_MD), code(readout_code), exercise_markdown(PROBE_BASE_MD), code(probe_base_code),
        exercise_markdown(RETRIEVAL_MD), code(retrieval_code), markdown("## Part 2: Falsify the Dynamics Claim"),
        exercise_markdown(FEATURE_EVAL_MD), code(feature_eval_code), exercise_markdown(REVERSAL_MD), code(reversal_code),
        exercise_markdown(PROBE_SUITE_MD), code(probe_suite_code), exercise_markdown(PREDICTOR_SCORE_MD), code(predictor_score_code),
        exercise_markdown(ASSEMBLY_MD), code(assembly_code), markdown(SIGNATURE_MD),
        code(source_for("make_signature_figure") + "\n" + clean("tests.test_signature_figure_uses_honest_axes(make_signature_figure)")),
        code(SIGNATURE_RUN),
        markdown("![Expected V-JEPA signature result](../../instructions/assets/vjepa_from_scratch_signature.png)"),
        markdown(SIGNATURE_INTERPRETATION), markdown(PLAY_MD), code(PLAY_CODE), markdown(ENDING),
    ]
    return notebook


def export_instruction_page(notebook: dict[str, object]) -> str:
    rendered = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"]).rstrip()
        if cell["cell_type"] == "markdown":
            source = source.replace("../../instructions/assets/", "../assets/")
            if index == 0:
                title, rest = source.split("\n", 1)
                links = (
                    "> **Notebooks: [exercises](../../exercises/part3_vjepa_from_scratch/14.3_V_JEPA_from_Scratch_exercises.ipynb) "
                    "| [solutions](../../exercises/part3_vjepa_from_scratch/14.3_V_JEPA_from_Scratch_solutions.ipynb)**"
                )
                source = f"{title}\n\n{links}\n{rest}"
            rendered.append(source)
        else:
            source = source.replace(
                DEPENDENCY_GRAPH_SOURCE.strip(),
                "LEARNER_DEPENDENCY_GRAPH = "
                + pformat(reference.LEARNER_DEPENDENCY_GRAPH, width=95, sort_dicts=False),
            )
            rendered.append(f"```python\n{source}\n```")
    return "\n\n".join(rendered) + "\n"


def main() -> None:
    old_solution = json.loads(SOLUTION_PATH.read_text())
    prior_outputs = {
        cell["id"]: cell for cell in old_solution["cells"] if cell["cell_type"] == "code"
    }
    exercise = build_notebook(EXERCISE_PATH, solved=False)
    solution = build_notebook(SOLUTION_PATH, solved=True)
    for cell in solution["cells"]:
        previous = prior_outputs.get(cell["id"])
        if previous is not None and "".join(previous["source"]) == "".join(cell["source"]):
            cell["outputs"] = previous["outputs"]
            cell["execution_count"] = previous["execution_count"]
    for notebook in (exercise, solution):
        for cell in notebook["cells"]:
            cell_source = cell["source"]
            if isinstance(cell_source, list):
                cell_source = "".join(cell_source)
            cell["source"] = cell_source.splitlines(keepends=True)
    EXERCISE_PATH.write_text(json.dumps(exercise, indent=1) + "\n")
    SOLUTION_PATH.write_text(json.dumps(solution, indent=1) + "\n")
    PAGE_PATH.write_text(export_instruction_page(exercise))
    print(f"rebuilt {len(exercise['cells'])} exercise cells and {len(solution['cells'])} solution cells")


if __name__ == "__main__":
    main()
