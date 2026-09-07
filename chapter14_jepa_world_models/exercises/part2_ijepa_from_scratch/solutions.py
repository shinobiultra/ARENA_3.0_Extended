"""Reference implementation for [14.2] I-JEPA from Scratch.

The experiment deliberately uses an exact generated image world. Every image
contains one known object, one known location, and a visible context key for
the object. The learner predicts a masked object block in representation space;
object labels are used only for evaluation and position labels only choose the
target block. No pixel-reconstruction target is used.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch as t
from torch import Tensor, nn
from torch.nn import functional as F


IMAGE_SIZE = 32
PATCH_SIZE = 4
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
N_PATCHES = GRID_SIZE**2
OBJECT_NAMES = ("red square", "blue circle", "green diamond")
POSITION_NAMES = ("top left", "top right", "bottom left", "bottom right")
KEY_ROW_SLICE = slice(0, PATCH_SIZE)
KEY_COLUMN_SLICE = slice(3 * PATCH_SIZE, 4 * PATCH_SIZE)
SIGNATURE_ASSET = (
    Path(__file__).resolve().parents[2] / "instructions/assets/ijepa_signature_result.png"
)
CONTRACT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class SceneBatch:
    images: Tensor
    object_labels: Tensor
    position_labels: Tensor


@dataclass(frozen=True)
class CollapseDiagnostics:
    feature_std: float
    effective_rank: float
    mean_pairwise_cosine: float


@dataclass(frozen=True)
class ProbeMetrics:
    object_accuracy: float
    position_accuracy: float


@dataclass(frozen=True)
class PredictionMetrics:
    latent_loss: float
    mismatched_context_loss: float
    key_erased_context_loss: float
    zero_context_loss: float
    target_object_accuracy: float
    predicted_object_accuracy: float
    mismatched_context_object_accuracy: float
    alternate_key_object_accuracy: float
    key_erased_context_object_accuracy: float
    non_key_pixel_max_abs_diff: float
    key_erased_non_key_pixel_max_abs_diff: float


@dataclass(frozen=True)
class TrainingTrace:
    losses: list[float]
    feature_std: list[float]
    effective_rank: list[float]


@dataclass(frozen=True)
class IJEPARun:
    device: str
    seed: int
    initial_loss: float
    final_loss: float
    loss_ratio: float
    probes: ProbeMetrics
    shuffled_label_probes: ProbeMetrics
    collapsed_control_probes: ProbeMetrics
    joint_target_probes: ProbeMetrics
    random_encoder_probes: ProbeMetrics
    prediction: PredictionMetrics
    joint_target_prediction: PredictionMetrics
    random_prediction: PredictionMetrics
    diagnostics: CollapseDiagnostics
    joint_target_diagnostics: CollapseDiagnostics
    collapsed_control_diagnostics: CollapseDiagnostics
    target_parameter_change: float
    target_has_gradients: bool
    peak_vram_gb: float
    trace: TrainingTrace
    joint_target_trace: TrainingTrace
    train_eval_exact_overlap: int
    train_semantic_template_count: int
    evaluation_semantic_template_count: int
    shared_semantic_template_count: int


def _draw_scene(object_id: int, position_id: int, *, jitter: int, seed: int) -> Tensor:
    """Draw one exact scene in ``[channels, height, width]`` format."""

    generator = t.Generator().manual_seed(seed)
    image = t.full((3, IMAGE_SIZE, IMAGE_SIZE), 0.04)
    image += 0.015 * t.rand(image.shape, generator=generator)

    # Radius three with the production jitter fixed at zero fits completely
    # inside the 2x2 target-patch mask. The separated key is therefore the only
    # class cue left in the online encoder's visible pixels.
    base_centres = ((8, 8), (23, 8), (8, 23), (23, 23))
    cx, cy = base_centres[position_id]
    dx = int(t.randint(-jitter, jitter + 1, (), generator=generator).item())
    dy = int(t.randint(-jitter, jitter + 1, (), generator=generator).item())
    cx, cy = cx + dx, cy + dy

    yy, xx = t.meshgrid(t.arange(IMAGE_SIZE), t.arange(IMAGE_SIZE), indexing="ij")
    if object_id == 0:
        region = (xx - cx).abs() <= 3
        region &= (yy - cy).abs() <= 3
        colour = t.tensor([0.95, 0.10, 0.10])
    elif object_id == 1:
        region = (xx - cx).square() + (yy - cy).square() <= 3**2
        colour = t.tensor([0.10, 0.35, 0.98])
    elif object_id == 2:
        region = (xx - cx).abs() + (yy - cy).abs() <= 3
        colour = t.tensor([0.10, 0.90, 0.32])
    else:
        raise ValueError(f"object_id must be 0, 1, or 2; got {object_id}")

    image[:, region] = colour[:, None]
    # One separated patch is a visible predictive key. The target mask never
    # covers it, so a correct prediction must use context rather than pixels
    # from the hidden object block.
    image[:, KEY_ROW_SLICE, KEY_COLUMN_SLICE] = colour[:, None, None]
    return image.clamp(0.0, 1.0)


def build_scene_dataset(
    examples_per_combination: int = 24,
    *,
    seed: int = 0,
    jitter: int = 0,
) -> SceneBatch:
    """Return a balanced, deterministic image world with exact semantic labels."""

    if examples_per_combination < 2:
        raise ValueError("examples_per_combination must be at least 2")
    images: list[Tensor] = []
    objects: list[int] = []
    positions: list[int] = []
    for object_id in range(len(OBJECT_NAMES)):
        for position_id in range(len(POSITION_NAMES)):
            for example_id in range(examples_per_combination):
                scene_seed = seed + 10_000 * object_id + 1_000 * position_id + example_id
                images.append(
                    _draw_scene(object_id, position_id, jitter=jitter, seed=scene_seed)
                )
                objects.append(object_id)
                positions.append(position_id)
    return SceneBatch(
        images=t.stack(images),
        object_labels=t.tensor(objects, dtype=t.long),
        position_labels=t.tensor(positions, dtype=t.long),
    )


def patchify(images: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
    """Convert ``[batch, channels, height, width]`` images into flattened patches."""

    if images.ndim != 4:
        raise ValueError(f"expected rank-4 images, got shape {tuple(images.shape)}")
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(batch, -1, channels * patch_size * patch_size)


def sample_block_masks(
    batch_size: int,
    *,
    grid_size: int = GRID_SIZE,
    block_size: int = 2,
    generator: t.Generator | None = None,
    device: t.device | str = "cpu",
) -> Tensor:
    """Sample one contiguous square target block for each image."""

    if not 1 <= block_size < grid_size:
        raise ValueError("block_size must satisfy 1 <= block_size < grid_size")
    upper = grid_size - block_size + 1
    starts = t.randint(0, upper, (batch_size, 2), generator=generator)
    masks = t.zeros(batch_size, grid_size, grid_size, dtype=t.bool)
    for row, (top, left) in enumerate(starts.tolist()):
        masks[row, top : top + block_size, left : left + block_size] = True
    return masks.flatten(1).to(device)


def object_target_masks(
    position_labels: Tensor,
    *,
    grid_size: int = GRID_SIZE,
    block_size: int = 2,
    device: t.device | str | None = None,
) -> Tensor:
    """Mask the object-bearing block while leaving its context key visible."""

    if grid_size != GRID_SIZE or block_size != 2:
        raise ValueError("the exact organism uses an 8x8 grid and 2x2 object targets")
    labels = position_labels.detach().cpu().long()
    if labels.ndim != 1 or not t.all((0 <= labels) & (labels < len(POSITION_NAMES))):
        raise ValueError("position_labels must be a rank-1 tensor with values in [0, 3]")
    near, far = 1, grid_size - block_size - 1
    starts = ((near, near), (near, far), (far, near), (far, far))
    masks = t.zeros(len(labels), grid_size, grid_size, dtype=t.bool)
    for row, label in enumerate(labels.tolist()):
        top, left = starts[label]
        masks[row, top : top + block_size, left : left + block_size] = True
    return masks.flatten(1).to(device or position_labels.device)


def gather_masked_tokens(tokens: Tensor, mask: Tensor) -> Tensor:
    """Gather the same number of selected tokens from every batch row."""

    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError("tokens must be [batch, patches, width] and mask [batch, patches]")
    counts = mask.sum(dim=1)
    if not t.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("every row must select the same number of tokens")
    return tokens[mask].view(tokens.shape[0], int(counts[0].item()), tokens.shape[-1])


def normalized_prediction_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """Mean squared error between unit-normalized latent target vectors."""

    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(predicted.shape)} != {tuple(target.shape)}")
    predicted_unit = F.normalize(predicted, dim=-1)
    target_unit = F.normalize(target.detach(), dim=-1)
    return (predicted_unit - target_unit).square().sum(dim=-1).mean()


@t.no_grad()
def update_target_encoder(online: nn.Module, target: nn.Module, ema_decay: float) -> None:
    """Apply ``target <- decay * target + (1-decay) * online`` in place."""

    if not 0.0 <= ema_decay <= 1.0:
        raise ValueError("ema_decay must lie in [0, 1]")
    for online_parameter, target_parameter in zip(online.parameters(), target.parameters()):
        target_parameter.lerp_(online_parameter, 1.0 - ema_decay)


class PatchTransformerEncoder(nn.Module):
    """Small vision transformer encoder used by both I-JEPA branches."""

    def __init__(self, patch_dim: int = 3 * PATCH_SIZE**2, d_model: int = 48) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(patch_dim, d_model)
        self.position = nn.Parameter(t.randn(1, N_PATCHES, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patches: Tensor, visible_mask: Tensor | None = None) -> Tensor:
        tokens = self.patch_projection(patches) + self.position[:, : patches.shape[1]]
        if visible_mask is not None:
            tokens = gather_masked_tokens(tokens, visible_mask)
        return self.norm(self.transformer(tokens))


class LatentPredictor(nn.Module):
    """Predict target positions after placing visible context into the full grid."""

    def __init__(self, d_model: int = 48) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(t.zeros(1, 1, d_model))
        self.position = nn.Parameter(t.randn(1, N_PATCHES, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context: Tensor, target_mask: Tensor) -> Tensor:
        batch, patches = target_mask.shape
        visible_mask = ~target_mask
        full = self.mask_token.expand(batch, patches, -1).clone()
        full[visible_mask] = context.reshape(-1, context.shape[-1])
        full = self.norm(self.transformer(full + self.position[:, :patches]))
        return gather_masked_tokens(full, target_mask)


class IJEPAModel(nn.Module):
    """Online context encoder, EMA target encoder, and latent predictor."""

    def __init__(self, d_model: int = 48) -> None:
        super().__init__()
        self.online_encoder = PatchTransformerEncoder(d_model=d_model)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = LatentPredictor(d_model=d_model)
        self.target_encoder.requires_grad_(False)

    def forward(self, patches: Tensor, target_mask: Tensor) -> tuple[Tensor, Tensor]:
        context = self.online_encoder(patches, ~target_mask)
        predicted = self.predictor(context, target_mask)
        with t.no_grad():
            target = gather_masked_tokens(self.target_encoder(patches), target_mask)
        return predicted, target

    @t.no_grad()
    def representations(self, patches: Tensor) -> Tensor:
        return self.target_encoder(patches).mean(dim=1)


def collapse_diagnostics(features: Tensor) -> CollapseDiagnostics:
    """Quantify variance, covariance effective rank, and pairwise similarity."""

    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must be [examples, width] with at least two examples")
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    feature_std = centered.std(dim=0, unbiased=False).mean().item()
    singular_values = t.linalg.svdvals(centered)
    spectrum = singular_values.square()
    probabilities = spectrum / spectrum.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    effective_rank = entropy.exp().item()
    unit = F.normalize(centered, dim=-1)
    similarities = unit @ unit.T
    off_diagonal = similarities[~t.eye(features.shape[0], dtype=t.bool, device=features.device)]
    return CollapseDiagnostics(
        feature_std=feature_std,
        effective_rank=effective_rank,
        mean_pairwise_cosine=off_diagonal.mean().item(),
    )


def fit_ridge_classifier(
    train_features: Tensor, train_labels: Tensor, ridge: float = 1e-4
) -> Tensor:
    """Fit a closed-form multiclass ridge probe, including a bias feature."""

    features = train_features.float()
    design = t.cat((features, t.ones(features.shape[0], 1, device=features.device)), dim=1)
    targets = F.one_hot(train_labels, int(train_labels.max().item()) + 1).float()
    identity = t.eye(design.shape[1], device=design.device)
    identity[-1, -1] = 0.0
    return t.linalg.solve(design.T @ design + ridge * identity, design.T @ targets)


def _probe_accuracy(
    train_features: Tensor,
    train_labels: Tensor,
    test_features: Tensor,
    test_labels: Tensor,
) -> float:
    train_features = train_features.float()
    test_features = test_features.float()
    weights = fit_ridge_classifier(train_features, train_labels)
    test_design = t.cat(
        (test_features, t.ones(test_features.shape[0], 1, device=test_features.device)),
        dim=1,
    )
    return (test_design @ weights).argmax(-1).eq(test_labels).float().mean().item()


def evaluate_probes(
    features: Tensor,
    object_labels: Tensor,
    position_labels: Tensor,
    *,
    seed: int = 0,
    train_fraction: float = 0.65,
) -> ProbeMetrics:
    """Fit split-sample linear probes for exact object and position labels."""

    generator = t.Generator(device=features.device).manual_seed(seed)
    permutation = t.randperm(features.shape[0], generator=generator, device=features.device)
    split = int(train_fraction * features.shape[0])
    train_indices, test_indices = permutation[:split], permutation[split:]
    return ProbeMetrics(
        object_accuracy=_probe_accuracy(
            features[train_indices],
            object_labels[train_indices],
            features[test_indices],
            object_labels[test_indices],
        ),
        position_accuracy=_probe_accuracy(
            features[train_indices],
            position_labels[train_indices],
            features[test_indices],
            position_labels[test_indices],
        ),
    )


def evaluate_transfer_probes(
    train_features: Tensor,
    train_object_labels: Tensor,
    train_position_labels: Tensor,
    test_features: Tensor,
    test_object_labels: Tensor,
    test_position_labels: Tensor,
) -> ProbeMetrics:
    """Fit probes on training noise draws and score pixel-distinct draws."""

    return ProbeMetrics(
        object_accuracy=_probe_accuracy(
            train_features,
            train_object_labels,
            test_features,
            test_object_labels,
        ),
        position_accuracy=_probe_accuracy(
            train_features,
            train_position_labels,
            test_features,
            test_position_labels,
        ),
    )


def cyclic_object_mismatch_indices(
    object_labels: Tensor, position_labels: Tensor, *, shift: int = 1
) -> Tensor:
    """Map each row to another object while preserving position and example index."""

    if object_labels.shape != position_labels.shape or object_labels.ndim != 1:
        raise ValueError("object and position labels must be matching rank-1 tensors")
    if shift % len(OBJECT_NAMES) == 0:
        raise ValueError("shift must select a different object class")
    rows: list[int] = []
    for index, (object_id, position_id) in enumerate(
        zip(object_labels.detach().cpu().tolist(), position_labels.detach().cpu().tolist())
    ):
        candidates = (
            (object_labels == (object_id + shift) % len(OBJECT_NAMES))
            & (position_labels == position_id)
        ).nonzero(as_tuple=False).flatten()
        source_ordinal = int(
            (
                (object_labels[:index] == object_id)
                & (position_labels[:index] == position_id)
            ).sum()
        )
        if source_ordinal >= len(candidates):
            raise ValueError("each object-position cell must have the same number of examples")
        rows.append(int(candidates[source_ordinal]))
    indices = t.tensor(rows, dtype=t.long, device=object_labels.device)
    if not t.all(object_labels[indices] != object_labels):
        raise AssertionError("mismatched contexts must always come from a different object")
    if not t.equal(position_labels[indices], position_labels):
        raise AssertionError("mismatched contexts must preserve object position")
    return indices


def replace_scene_keys(scenes: SceneBatch, source_indices: Tensor) -> SceneBatch:
    """Replace only the visible key patch, preserving every other image pixel."""

    if source_indices.shape != scenes.object_labels.shape or source_indices.ndim != 1:
        raise ValueError("source_indices must provide one source row per scene")
    source_indices = source_indices.to(scenes.images.device)
    if not t.all((0 <= source_indices) & (source_indices < len(scenes.images))):
        raise ValueError("source_indices contains an out-of-range row")
    images = scenes.images.clone()
    images[:, :, KEY_ROW_SLICE, KEY_COLUMN_SLICE] = scenes.images[
        source_indices, :, KEY_ROW_SLICE, KEY_COLUMN_SLICE
    ]
    return SceneBatch(
        images=images,
        object_labels=scenes.object_labels.clone(),
        position_labels=scenes.position_labels.clone(),
    )


def erase_scene_keys(scenes: SceneBatch) -> SceneBatch:
    """Replace the key with neutral grayscale sampled from a safe background patch."""

    images = scenes.images.clone()
    background_level = scenes.images[:, :, :PATCH_SIZE, :PATCH_SIZE].mean(
        dim=(1, 2, 3), keepdim=True
    )
    images[:, :, KEY_ROW_SLICE, KEY_COLUMN_SLICE] = background_level
    return SceneBatch(
        images=images,
        object_labels=scenes.object_labels.clone(),
        position_labels=scenes.position_labels.clone(),
    )


@t.no_grad()
def evaluate_prediction(
    model: IJEPAModel,
    centroid_scenes: SceneBatch,
    evaluation_scenes: SceneBatch,
    *,
    device: t.device,
    mismatch_shift: int = 1,
) -> PredictionMetrics:
    """Score prediction and controls on pixel-distinct shared-template scenes."""

    model.eval()

    def target_and_prediction(
        scenes: SceneBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float]:
        patches = patchify(scenes.images).to(device)
        target_mask = object_target_masks(scenes.position_labels, device=device)
        context = model.online_encoder(patches, ~target_mask)
        target = gather_masked_tokens(model.target_encoder(patches), target_mask)
        predicted = model.predictor(context, target_mask)
        mismatch_indices = cyclic_object_mismatch_indices(
            scenes.object_labels.to(device),
            scenes.position_labels.to(device),
            shift=mismatch_shift,
        )
        key_swapped_scenes = replace_scene_keys(scenes, mismatch_indices.cpu())
        key_swapped_patches = patchify(key_swapped_scenes.images).to(device)
        key_swapped_context = model.online_encoder(key_swapped_patches, ~target_mask)
        mismatched = model.predictor(key_swapped_context, target_mask)
        key_erased_scenes = erase_scene_keys(scenes)
        key_erased_patches = patchify(key_erased_scenes.images).to(device)
        key_erased_context = model.online_encoder(key_erased_patches, ~target_mask)
        key_erased = model.predictor(key_erased_context, target_mask)
        zero_context = model.predictor(t.zeros_like(context), target_mask)
        non_key_mask = t.ones_like(scenes.images, dtype=t.bool)
        non_key_mask[:, :, KEY_ROW_SLICE, KEY_COLUMN_SLICE] = False
        swapped_non_key_max_diff = (
            scenes.images[non_key_mask] - key_swapped_scenes.images[non_key_mask]
        ).abs().max().item()
        erased_non_key_max_diff = (
            scenes.images[non_key_mask] - key_erased_scenes.images[non_key_mask]
        ).abs().max().item()
        return (
            target,
            predicted,
            mismatched,
            key_erased,
            zero_context,
            mismatch_indices,
            swapped_non_key_max_diff,
            erased_non_key_max_diff,
        )

    centroid_target, *_ = target_and_prediction(centroid_scenes)
    (
        target,
        predicted,
        mismatched,
        key_erased,
        zero_context,
        mismatch_indices,
        non_key_max_diff,
        erased_non_key_max_diff,
    ) = target_and_prediction(evaluation_scenes)
    centroid_labels = centroid_scenes.object_labels.to(device)
    centroids = t.stack(
        [
            centroid_target.mean(dim=1)[centroid_labels == object_id].mean(dim=0)
            for object_id in range(len(OBJECT_NAMES))
        ]
    )
    centroids = F.normalize(centroids, dim=-1)
    evaluation_labels = evaluation_scenes.object_labels.to(device)

    def centroid_predictions(values: Tensor) -> Tensor:
        pooled = F.normalize(values.mean(dim=1), dim=-1)
        return (pooled @ centroids.T).argmax(dim=-1)

    def centroid_accuracy(values: Tensor, labels: Tensor = evaluation_labels) -> float:
        return centroid_predictions(values).eq(labels).float().mean().item()

    alternate_labels = evaluation_labels[mismatch_indices]

    return PredictionMetrics(
        latent_loss=normalized_prediction_loss(predicted, target).item(),
        mismatched_context_loss=normalized_prediction_loss(mismatched, target).item(),
        key_erased_context_loss=normalized_prediction_loss(key_erased, target).item(),
        zero_context_loss=normalized_prediction_loss(zero_context, target).item(),
        target_object_accuracy=centroid_accuracy(target),
        predicted_object_accuracy=centroid_accuracy(predicted),
        mismatched_context_object_accuracy=centroid_accuracy(mismatched),
        alternate_key_object_accuracy=centroid_accuracy(mismatched, alternate_labels),
        key_erased_context_object_accuracy=centroid_accuracy(key_erased),
        non_key_pixel_max_abs_diff=non_key_max_diff,
        key_erased_non_key_pixel_max_abs_diff=erased_non_key_max_diff,
    )


def _extract_features(model: IJEPAModel, patches: Tensor, batch_size: int = 96) -> Tensor:
    chunks = [model.representations(chunk) for chunk in patches.split(batch_size)]
    return t.cat(chunks)


def _train_model(
    scenes: SceneBatch,
    *,
    device: t.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    ema_decay: float,
    joint_target_ablation: bool,
) -> tuple[IJEPAModel, TrainingTrace, float]:
    t.manual_seed(seed)
    model = IJEPAModel().to(device)
    if joint_target_ablation:
        model.target_encoder.requires_grad_(True)
    initial_target = t.cat(
        [parameter.detach().flatten().cpu() for parameter in model.target_encoder.parameters()]
    )
    trainable = list(model.online_encoder.parameters()) + list(model.predictor.parameters())
    if joint_target_ablation:
        trainable += list(model.target_encoder.parameters())
    optimizer = t.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    patches = patchify(scenes.images).to(device)
    generator = t.Generator().manual_seed(seed + 1)
    trace_losses: list[float] = []
    trace_std: list[float] = []
    trace_rank: list[float] = []

    for epoch in range(epochs):
        ordering = t.randperm(patches.shape[0], generator=generator)
        epoch_losses: list[float] = []
        model.train()
        for indices in ordering.split(batch_size):
            batch = patches[indices.to(device)]
            target_mask = object_target_masks(
                scenes.position_labels[indices], device=device
            )
            if joint_target_ablation:
                context = model.online_encoder(batch, ~target_mask)
                predicted = model.predictor(context, target_mask)
                target = gather_masked_tokens(model.target_encoder(batch), target_mask)
                # This compound ablation removes both stop-gradient and EMA by
                # optimizing the target branch directly.
                loss = (
                    F.normalize(predicted, dim=-1)
                    - F.normalize(target, dim=-1)
                ).square().sum(dim=-1).mean()
            else:
                predicted, target = model(batch, target_mask)
                loss = normalized_prediction_loss(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            shared_parameters = list(model.online_encoder.parameters()) + list(
                model.predictor.parameters()
            )
            t.nn.utils.clip_grad_norm_(shared_parameters, 1.0)
            if joint_target_ablation:
                # Keep clipping of shared parameters identical to the EMA run.
                # The target branch is clipped separately so the compound
                # treatment cannot rescale the shared update through clipping.
                t.nn.utils.clip_grad_norm_(model.target_encoder.parameters(), 1.0)
            optimizer.step()
            if not joint_target_ablation:
                update_target_encoder(model.online_encoder, model.target_encoder, ema_decay)
            epoch_losses.append(loss.detach().item())

        model.eval()
        with t.no_grad():
            diagnostic_features = _extract_features(model, patches[: min(96, len(patches))])
        diagnostic = collapse_diagnostics(diagnostic_features)
        trace_losses.append(sum(epoch_losses) / len(epoch_losses))
        trace_std.append(diagnostic.feature_std)
        trace_rank.append(diagnostic.effective_rank)

    final_target = t.cat(
        [parameter.detach().flatten().cpu() for parameter in model.target_encoder.parameters()]
    )
    target_change = (final_target - initial_target).norm().item()
    return model, TrainingTrace(trace_losses, trace_std, trace_rank), target_change


def run_experiment(
    *,
    device: str | t.device = "cpu",
    seed: int = 0,
    examples_per_combination: int = 20,
    epochs: int = 30,
) -> IJEPARun:
    """Train both objectives and evaluate pixel-distinct shared-template scenes."""

    device = t.device(device)
    if device.type == "cuda":
        t.cuda.reset_peak_memory_stats(device)
    scenes = build_scene_dataset(examples_per_combination, seed=seed)
    evaluation_scenes = build_scene_dataset(examples_per_combination, seed=seed + 50_000)
    model, trace, target_change = _train_model(
        scenes,
        device=device,
        seed=seed,
        epochs=epochs,
        batch_size=48,
        learning_rate=2e-3,
        ema_decay=0.995,
        joint_target_ablation=False,
    )
    joint_target_model, joint_target_trace, _ = _train_model(
        scenes,
        device=device,
        seed=seed,
        epochs=epochs,
        batch_size=48,
        learning_rate=2e-3,
        ema_decay=0.0,
        joint_target_ablation=True,
    )
    t.manual_seed(seed)
    random_model = IJEPAModel().to(device)
    train_patches = patchify(scenes.images).to(device)
    evaluation_patches = patchify(evaluation_scenes.images).to(device)
    train_object_labels = scenes.object_labels.to(device)
    train_position_labels = scenes.position_labels.to(device)
    evaluation_object_labels = evaluation_scenes.object_labels.to(device)
    evaluation_position_labels = evaluation_scenes.position_labels.to(device)
    model.eval()
    joint_target_model.eval()
    random_model.eval()
    with t.no_grad():
        train_features = _extract_features(model, train_patches)
        features = _extract_features(model, evaluation_patches)
        joint_target_train_features = _extract_features(joint_target_model, train_patches)
        joint_target_features = _extract_features(joint_target_model, evaluation_patches)
        random_train_features = _extract_features(random_model, train_patches)
        random_features = _extract_features(random_model, evaluation_patches)
    probes = evaluate_transfer_probes(
        train_features,
        train_object_labels,
        train_position_labels,
        features,
        evaluation_object_labels,
        evaluation_position_labels,
    )
    joint_target_probes = evaluate_transfer_probes(
        joint_target_train_features,
        train_object_labels,
        train_position_labels,
        joint_target_features,
        evaluation_object_labels,
        evaluation_position_labels,
    )
    random_probes = evaluate_transfer_probes(
        random_train_features,
        train_object_labels,
        train_position_labels,
        random_features,
        evaluation_object_labels,
        evaluation_position_labels,
    )

    label_generator = t.Generator(device=device).manual_seed(seed + 3)
    shuffled_objects = train_object_labels[
        t.randperm(len(train_object_labels), generator=label_generator, device=device)
    ]
    shuffled_positions = train_position_labels[
        t.randperm(len(train_position_labels), generator=label_generator, device=device)
    ]
    shuffled = evaluate_transfer_probes(
        train_features,
        shuffled_objects,
        shuffled_positions,
        features,
        evaluation_object_labels,
        evaluation_position_labels,
    )
    collapsed_train_features = t.ones_like(train_features)
    collapsed_features = t.ones_like(features)
    collapsed = evaluate_transfer_probes(
        collapsed_train_features,
        train_object_labels,
        train_position_labels,
        collapsed_features,
        evaluation_object_labels,
        evaluation_position_labels,
    )
    prediction = evaluate_prediction(model, scenes, evaluation_scenes, device=device)
    joint_target_prediction = evaluate_prediction(
        joint_target_model, scenes, evaluation_scenes, device=device
    )
    random_prediction = evaluate_prediction(
        random_model, scenes, evaluation_scenes, device=device
    )
    train_eval_exact_overlap = int(
        (
            scenes.images[:, None].eq(evaluation_scenes.images[None])
            .flatten(2)
            .all(dim=-1)
        ).sum()
    )
    train_support = set(zip(scenes.object_labels.tolist(), scenes.position_labels.tolist()))
    evaluation_support = set(
        zip(evaluation_scenes.object_labels.tolist(), evaluation_scenes.position_labels.tolist())
    )

    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = t.cuda.max_memory_allocated(device) / 2**30
    initial_loss = trace.losses[0]
    final_loss = sum(trace.losses[-3:]) / min(3, len(trace.losses))
    return IJEPARun(
        device=str(device),
        seed=seed,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_ratio=final_loss / initial_loss,
        probes=probes,
        shuffled_label_probes=shuffled,
        collapsed_control_probes=collapsed,
        joint_target_probes=joint_target_probes,
        random_encoder_probes=random_probes,
        prediction=prediction,
        joint_target_prediction=joint_target_prediction,
        random_prediction=random_prediction,
        diagnostics=collapse_diagnostics(features),
        joint_target_diagnostics=collapse_diagnostics(joint_target_features),
        collapsed_control_diagnostics=collapse_diagnostics(collapsed_features),
        target_parameter_change=target_change,
        target_has_gradients=any(
            parameter.grad is not None for parameter in model.target_encoder.parameters()
        ),
        peak_vram_gb=peak_vram_gb,
        trace=trace,
        joint_target_trace=joint_target_trace,
        train_eval_exact_overlap=train_eval_exact_overlap,
        train_semantic_template_count=len(train_support),
        evaluation_semantic_template_count=len(evaluation_support),
        shared_semantic_template_count=len(train_support & evaluation_support),
    )


def validate_run(run: IJEPARun, *, require_cuda: bool = False, max_vram_gb: float = 24.0) -> None:
    """Raise a teaching-oriented assertion when a signature claim does not hold."""

    if require_cuda:
        assert run.device.startswith("cuda"), (
            "GPU verification requires a CUDA device; CPU fallback is forbidden."
        )
    assert run.loss_ratio < 0.20, f"Latent loss did not fall enough: ratio={run.loss_ratio:.3f}."
    assert run.diagnostics.feature_std > 0.02, (
        "EMA representations collapsed to near-constant features."
    )
    assert run.prediction.target_object_accuracy >= 0.95, (
        "The target encoder must expose the exact object factor before predictions are scored."
    )
    assert run.prediction.predicted_object_accuracy >= 0.90, (
        "Held-out predictions must recover the masked object's exact class."
    )
    assert run.prediction.mismatched_context_object_accuracy <= 0.10, (
        "A key-only cyclic mismatch should stop predicting the original object class."
    )
    assert run.prediction.alternate_key_object_accuracy >= 0.90, (
        "A key-only cyclic mismatch should predict the alternate key's object class."
    )
    assert run.prediction.key_erased_context_object_accuracy <= 0.50, (
        "Removing the key in image space should reduce prediction to near chance."
    )
    assert run.prediction.non_key_pixel_max_abs_diff == 0.0, (
        "The causal intervention changed pixels outside the visible key patch."
    )
    assert run.prediction.key_erased_non_key_pixel_max_abs_diff == 0.0, (
        "The key-erasure control changed pixels outside the visible key patch."
    )
    assert run.random_prediction.predicted_object_accuracy <= 0.50, (
        "An untrained encoder-predictor must not solve the pixel-distinct prediction task."
    )
    assert run.joint_target_prediction.predicted_object_accuracy <= 0.75, (
        "The compound joint-target ablation should not match the main prediction accuracy."
    )
    assert run.prediction.mismatched_context_loss > 1.10 * run.prediction.latent_loss, (
        "Correct context must beat the cyclic object-mismatch intervention in latent loss."
    )
    assert run.prediction.key_erased_context_loss > 1.10 * run.prediction.latent_loss, (
        "Correct context must beat the image-space key-erasure control in latent loss."
    )
    assert run.prediction.zero_context_loss > 5.0 * run.prediction.latent_loss, (
        "The predictor must use visible context rather than only target positions."
    )
    assert run.random_prediction.latent_loss > 20.0 * run.prediction.latent_loss, (
        "Training must beat the same initialized architecture before semantic claims are made."
    )
    assert run.diagnostics.feature_std > 5.0 * run.joint_target_diagnostics.feature_std, (
        "Main-model features must retain at least five times the compound ablation's spread."
    )
    assert run.collapsed_control_diagnostics.feature_std < 1e-7, (
        "The exact constant-feature control should have numerically zero feature spread."
    )
    assert run.collapsed_control_diagnostics.effective_rank <= 1.01, (
        "The exact constant-feature control should have effective rank at most one."
    )
    assert run.target_parameter_change > 0.0, "EMA target parameters never moved."
    assert not run.target_has_gradients, "EMA target encoder unexpectedly received gradients."
    assert run.train_eval_exact_overlap == 0, "Training and evaluation tensors must be distinct."
    expected_templates = len(OBJECT_NAMES) * len(POSITION_NAMES)
    assert run.train_semantic_template_count == expected_templates, "Training must contain all twelve declared templates."
    assert run.evaluation_semantic_template_count == expected_templates, "Evaluation must contain the same twelve declared templates."
    assert run.shared_semantic_template_count == expected_templates, (
        "The GT-0 contract intentionally shares every semantic template across the pixel split."
    )
    assert run.peak_vram_gb <= max_vram_gb, (
        f"Peak VRAM {run.peak_vram_gb:.3f} GB exceeded the {max_vram_gb:.1f} GB budget."
    )


def _report_metrics(run: IJEPARun) -> dict[str, Any]:
    report = asdict(run)
    report.update(
        {
            "loss_ratio": run.loss_ratio,
            "feature_std": run.diagnostics.feature_std,
            "effective_rank": run.diagnostics.effective_rank,
            "heldout_prediction_accuracy": run.prediction.predicted_object_accuracy,
            "mismatched_context_accuracy": run.prediction.mismatched_context_object_accuracy,
            "alternate_key_accuracy": run.prediction.alternate_key_object_accuracy,
            "key_erased_context_accuracy": run.prediction.key_erased_context_object_accuracy,
            "non_key_pixel_max_abs_diff": run.prediction.non_key_pixel_max_abs_diff,
            "key_erased_non_key_pixel_max_abs_diff": (
                run.prediction.key_erased_non_key_pixel_max_abs_diff
            ),
            "random_prediction_accuracy": run.random_prediction.predicted_object_accuracy,
            "joint_target_ablation_prediction_accuracy": (
                run.joint_target_prediction.predicted_object_accuracy
            ),
            "mismatched_to_correct_loss_ratio": (
                run.prediction.mismatched_context_loss / run.prediction.latent_loss
            ),
            "zero_to_correct_loss_ratio": (
                run.prediction.zero_context_loss / run.prediction.latent_loss
            ),
            "key_erased_to_correct_loss_ratio": (
                run.prediction.key_erased_context_loss / run.prediction.latent_loss
            ),
            "random_to_trained_loss_ratio": (
                run.random_prediction.latent_loss / run.prediction.latent_loss
            ),
            "random_encoder_object_probe_accuracy": run.random_encoder_probes.object_accuracy,
            "random_encoder_position_probe_accuracy": run.random_encoder_probes.position_accuracy,
            "object_probe_margin_over_random": (
                run.probes.object_accuracy - run.random_encoder_probes.object_accuracy
            ),
            "position_probe_margin_over_random": (
                run.probes.position_accuracy - run.random_encoder_probes.position_accuracy
            ),
            "main_to_joint_target_feature_std_ratio": (
                run.diagnostics.feature_std / run.joint_target_diagnostics.feature_std
            ),
            "collapsed_feature_std": run.collapsed_control_diagnostics.feature_std,
            "collapsed_effective_rank": run.collapsed_control_diagnostics.effective_rank,
            "train_eval_exact_overlap": run.train_eval_exact_overlap,
            "shared_semantic_template_count": run.shared_semantic_template_count,
            "tests_passed": True,
            "contract_passed": True,
            "accepted": True,
        }
    )
    return report


def run_seed_suite(
    *,
    device: str | t.device,
    seeds: tuple[int, ...] = CONTRACT_SEEDS,
    examples_per_combination: int = 10,
    epochs: int = 30,
    max_vram_gb: float = 24.0,
) -> tuple[list[IJEPARun], dict[str, Any]]:
    """Run and validate every predeclared seed, returning per-seed evidence."""

    if tuple(seeds) != CONTRACT_SEEDS:
        raise ValueError(f"the release contract requires exactly seeds {CONTRACT_SEEDS}")
    runs: list[IJEPARun] = []
    for seed in seeds:
        run = run_experiment(
            device=device,
            seed=seed,
            examples_per_combination=examples_per_combination,
            epochs=epochs,
        )
        validate_run(
            run,
            require_cuda=t.device(device).type == "cuda",
            max_vram_gb=max_vram_gb,
        )
        runs.append(run)

    report = _report_metrics(runs[0])
    report.update(
        {
            "contract_seeds": list(seeds),
            "all_seeds_accepted": True,
            "per_seed": [
                {
                    "seed": run.seed,
                    "prediction_accuracy": run.prediction.predicted_object_accuracy,
                    "mismatched_original_accuracy": (
                        run.prediction.mismatched_context_object_accuracy
                    ),
                    "alternate_key_accuracy": run.prediction.alternate_key_object_accuracy,
                    "key_erased_accuracy": (
                        run.prediction.key_erased_context_object_accuracy
                    ),
                    "latent_loss": run.prediction.latent_loss,
                    "effective_rank_report_only": run.diagnostics.effective_rank,
                    "feature_std": run.diagnostics.feature_std,
                    "shared_semantic_templates": run.shared_semantic_template_count,
                }
                for run in runs
            ],
            "worst_seed_prediction_accuracy": min(
                run.prediction.predicted_object_accuracy for run in runs
            ),
            "worst_seed_key_erased_accuracy": max(
                run.prediction.key_erased_context_object_accuracy for run in runs
            ),
            "minimum_reported_effective_rank": min(
                run.diagnostics.effective_rank for run in runs
            ),
            "peak_vram_gb": max(run.peak_vram_gb for run in runs),
        }
    )
    return runs, report


def plot_signature_result(run: IJEPARun, output_path: str | Path | None = None):
    """Show the predictive result, causal controls, collapse checks, and probe trap."""

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    example = _draw_scene(2, 3, jitter=0, seed=0)
    target_mask = object_target_masks(t.tensor([3])).view(GRID_SIZE, GRID_SIZE)
    axes[0, 0].imshow(example.permute(1, 2, 0))
    for row, column in target_mask.nonzero().tolist():
        axes[0, 0].add_patch(
            Rectangle(
                (column * PATCH_SIZE, row * PATCH_SIZE),
                PATCH_SIZE,
                PATCH_SIZE,
                color="#f0b429",
                alpha=0.65,
            )
        )
    axes[0, 0].set_title("Visible key, masked object target")
    axes[0, 0].axis("off")

    axes[0, 1].plot(run.trace.losses, label="EMA target", color="#2364aa")
    axes[0, 1].plot(
        run.joint_target_trace.losses,
        label="joint target (no stop-grad, no EMA)",
        color="#d1495b",
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("normalized latent loss")
    axes[0, 1].set_title(f"Training loss ratio: {run.loss_ratio:.4f}")
    axes[0, 1].legend(fontsize=8)

    prediction_names = [
        "target",
        "correct",
        "swap: original",
        "swap: alternate",
        "key erased",
        "joint target",
        "untrained",
    ]
    prediction_values = [
        run.prediction.target_object_accuracy,
        run.prediction.predicted_object_accuracy,
        run.prediction.mismatched_context_object_accuracy,
        run.prediction.alternate_key_object_accuracy,
        run.prediction.key_erased_context_object_accuracy,
        run.joint_target_prediction.predicted_object_accuracy,
        run.random_prediction.predicted_object_accuracy,
    ]
    axes[0, 2].bar(
        prediction_names,
        prediction_values,
        color=[
            "#2a9d8f",
            "#2364aa",
            "#d1495b",
            "#e76f51",
            "#e9c46a",
            "#8d6a9f",
            "#7a7a7a",
        ],
    )
    axes[0, 2].axhline(1 / 3, color="black", linestyle=":", label="chance")
    axes[0, 2].set_ylim(0, 1.05)
    axes[0, 2].tick_params(axis="x", rotation=20)
    axes[0, 2].set_ylabel("pixel-distinct evaluation accuracy")
    axes[0, 2].set_title("Key-only swap changes the predicted class")
    axes[0, 2].legend(fontsize=8)

    loss_names = ["correct", "mismatch", "key erased", "zero latent", "untrained"]
    loss_values = [
        run.prediction.latent_loss,
        run.prediction.mismatched_context_loss,
        run.prediction.key_erased_context_loss,
        run.prediction.zero_context_loss,
        run.random_prediction.latent_loss,
    ]
    axes[1, 0].bar(
        loss_names,
        loss_values,
        color=["#2364aa", "#d1495b", "#e9c46a", "#8d6a9f", "#7a7a7a"],
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].set_ylabel("pixel-distinct evaluation latent loss")
    axes[1, 0].set_title("Correct context beats causal controls")

    spread_names = ["main", "joint target", "constant"]
    spread_values = [
        run.diagnostics.feature_std,
        run.joint_target_diagnostics.feature_std,
        run.collapsed_control_diagnostics.feature_std,
    ]
    axes[1, 1].bar(spread_names, spread_values, color=["#2364aa", "#d1495b", "#7a7a7a"])
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].set_ylabel("mean feature standard deviation")
    axes[1, 1].set_title(
        "Compound ablation spread ratio: "
        f"{run.diagnostics.feature_std / run.joint_target_diagnostics.feature_std:.1f}x"
    )

    probe_names = ["trained", "random encoder", "shuffled labels", "constant"]
    object_scores = [
        run.probes.object_accuracy,
        run.random_encoder_probes.object_accuracy,
        run.shuffled_label_probes.object_accuracy,
        run.collapsed_control_probes.object_accuracy,
    ]
    position_scores = [
        run.probes.position_accuracy,
        run.random_encoder_probes.position_accuracy,
        run.shuffled_label_probes.position_accuracy,
        run.collapsed_control_probes.position_accuracy,
    ]
    x = np.arange(len(probe_names))
    width = 0.36
    axes[1, 2].bar(x - width / 2, object_scores, width, label="object", color="#2a9d8f")
    axes[1, 2].bar(x + width / 2, position_scores, width, label="position", color="#e9c46a")
    axes[1, 2].set_xticks(x, probe_names, rotation=20, ha="right")
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].set_ylabel("pixel-distinct evaluation probe accuracy")
    axes[1, 2].set_title("Negative result: random features already decode")
    axes[1, 2].legend(fontsize=8)

    figure.suptitle(
        "GT-0 I-JEPA mechanism: a visible key controls masked-target prediction",
        fontsize=15,
        fontweight="bold",
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
    return figure


def run_smoke_test(cpu: bool = True) -> dict[str, Any]:
    """Run the complete three-seed contract on the requested device."""

    device = "cpu" if cpu else "cuda"
    if not cpu and not t.cuda.is_available():
        raise RuntimeError("CUDA was requested; no CPU fallback is used")
    _, report = run_seed_suite(device=device)
    report.update({"cuda_available": not cpu, "within_vram_budget": True})
    return report


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, Any]:
    """Run all three contract seeds on CUDA, refusing to fall back to CPU."""

    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for run_gpu_test; CPU fallback is forbidden")
    runs, report = run_seed_suite(device="cuda", max_vram_gb=max_vram_gb)
    plot_signature_result(runs[0], SIGNATURE_ASSET)
    report.update(
        {
            "cuda_available": True,
            "device": t.cuda.get_device_name(),
            "gpu_name": t.cuda.get_device_name(),
            "torch_version": t.__version__,
            "cuda_version": t.version.cuda,
            "max_allowed_gpu_gb": max_vram_gb,
            "within_vram_budget": report["peak_vram_gb"] <= max_vram_gb,
        }
    )
    return report


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, Any]:
    """Run the release-grade CUDA experiment with no CPU fallback."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


def save_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_smoke_test(), indent=2))
