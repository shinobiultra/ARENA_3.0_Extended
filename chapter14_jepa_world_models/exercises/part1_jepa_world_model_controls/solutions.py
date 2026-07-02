# %%
"""Reference solutions for [14.1] JEPA and World-Model Controls."""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t
import torch.nn.functional as F

chapter = "chapter14_jepa_world_models"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.jepa_world_models import (
    causal_latent_patch_report,
    collapse_diagnostics_report,
    jepa_prediction_report,
    latent_rollout_report,
    loss_decrease_report,
    object_permanence_report,
    transition_consistency_report,
    world_state_probe_report,
)

MAIN = __name__ == "__main__"

REAL_VJEPA2_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
REAL_VJEPA2_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
VJEPA2_VIDEO_KINDS = (
    "red_square",
    "red_square_shifted",
    "blue_circle",
    "red_square_late_occluded",
    "red_square_absent_occluder",
)
VJEPA2_ACTIONS = (
    (0, 0),
    (8, 0),
    (-8, 0),
    (0, 8),
    (0, -8),
)


def paired_cosine(left: t.Tensor, right: t.Tensor, *, eps: float = 1e-8) -> t.Tensor:
    """Return row-wise cosine similarities for two equally-shaped embedding batches."""

    if left.shape != right.shape:
        raise ValueError("left and right must have the same shape.")
    if left.ndim < 2:
        raise ValueError("left and right must have shape (..., d_model).")
    left_float = left.float()
    right_float = right.float()
    numerator = (left_float * right_float).sum(dim=-1)
    denominator = left_float.norm(dim=-1) * right_float.norm(dim=-1)
    return numerator / denominator.clamp_min(eps)


@dataclass(frozen=True)
class SyntheticVJEPAWorldBatch:
    videos: t.Tensor
    labels: t.Tensor
    x_buckets: t.Tensor
    action_ids: t.Tensor
    next_videos: t.Tensor
    occluded_videos: t.Tensor
    absent_videos: t.Tensor
    bboxes: list[tuple[int, int, int, int]]


# %%
def jepa_prediction_smoke_test() -> dict:
    target_embeddings = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    predicted_targets = target_embeddings.clone()
    return jepa_prediction_report(
        predicted_targets,
        target_embeddings,
        min_cosine=0.99,
        max_mse=0.01,
    ).__dict__


def state_probe_smoke_test() -> dict:
    probe_logits = t.tensor([[3.0, 0.0], [0.0, 4.0], [2.0, 1.0]])
    labels = t.tensor([0, 1, 0])
    return world_state_probe_report(
        probe_logits,
        labels,
        min_accuracy=1.0,
    ).__dict__


def transition_smoke_test() -> dict:
    state_embeddings = t.tensor([[0.0, 0.0], [1.0, 1.0]])
    action_deltas = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    next_state_embeddings = t.tensor([[1.0, 0.0], [1.0, 2.0]])
    return transition_consistency_report(
        state_embeddings,
        action_deltas,
        next_state_embeddings,
        min_cosine=0.99,
    ).__dict__


def object_permanence_smoke_test() -> dict:
    return object_permanence_report(
        visible_scores=t.tensor([0.9, 0.8]),
        occluded_scores=t.tensor([0.75, 0.7]),
        absent_scores=t.tensor([0.1, 0.2]),
        min_occluded_score=0.6,
        min_absent_gap=0.4,
    ).__dict__


def collapse_diagnostics_smoke_test() -> dict:
    structured_features = t.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    collapsed_features = t.ones(6, 4)
    structured = collapse_diagnostics_report(
        structured_features,
        min_feature_std=0.1,
        min_effective_rank=2.0,
    )
    collapsed = collapse_diagnostics_report(
        collapsed_features,
        min_feature_std=0.1,
        min_effective_rank=2.0,
    )
    return {
        "structured": structured.__dict__,
        "collapsed": collapsed.__dict__,
        "collapsed_control_rejected": not collapsed.non_collapsed,
    }


def state_probe_control_smoke_test() -> dict:
    probe_logits = t.tensor(
        [
            [4.0, 0.0],
            [3.0, 0.0],
            [0.0, 4.0],
            [0.0, 3.0],
        ]
    )
    labels = t.tensor([0, 0, 1, 1])
    shuffled_labels = t.tensor([1, 1, 0, 0])
    probe = world_state_probe_report(probe_logits, labels, min_accuracy=0.9)
    shuffled = world_state_probe_report(
        probe_logits,
        shuffled_labels,
        min_accuracy=0.9,
    )
    return {
        "probe": probe.__dict__,
        "shuffled": shuffled.__dict__,
        "shuffled_control_rejected": not shuffled.predicts_state,
        "accuracy_margin": probe.accuracy - shuffled.accuracy,
    }


def rollout_control_smoke_test() -> dict:
    rollout = latent_rollout_report(
        rollout_loss=0.10,
        copy_baseline_loss=1.00,
        shuffled_action_loss=0.90,
        max_rollout_to_copy=0.8,
        max_rollout_to_shuffled=0.8,
    )
    failed = latent_rollout_report(
        rollout_loss=0.75,
        copy_baseline_loss=0.80,
        shuffled_action_loss=0.70,
        max_rollout_to_copy=0.8,
        max_rollout_to_shuffled=0.8,
    )
    return {
        "rollout": rollout.__dict__,
        "failed_control": failed.__dict__,
        "copy_and_shuffled_controls_rejected": not failed.rollout_passes,
    }


def object_permanence_control_smoke_test() -> dict:
    preserved = object_permanence_report(
        visible_scores=t.tensor([0.95, 0.9]),
        occluded_scores=t.tensor([0.8, 0.75]),
        absent_scores=t.tensor([0.15, 0.2]),
        min_occluded_score=0.6,
        min_absent_gap=0.4,
    )
    absent_like = object_permanence_report(
        visible_scores=t.tensor([0.95, 0.9]),
        occluded_scores=t.tensor([0.52, 0.48]),
        absent_scores=t.tensor([0.46, 0.44]),
        min_occluded_score=0.6,
        min_absent_gap=0.4,
    )
    different_object = object_permanence_report(
        visible_scores=t.tensor([0.95, 0.9]),
        occluded_scores=t.tensor([0.78, 0.76]),
        absent_scores=t.tensor([0.72, 0.7]),
        min_occluded_score=0.6,
        min_absent_gap=0.4,
    )
    return {
        "preserved": preserved.__dict__,
        "absent_like": absent_like.__dict__,
        "different_object": different_object.__dict__,
        "absent_like_rejected": not absent_like.preserves_occluded_object,
        "different_object_rejected": not different_object.preserves_occluded_object,
    }


def _synthetic_vjepa_video(kind: str, *, frames: int = 8, size: int = 96) -> t.Tensor:
    """Create a deterministic normalized video tensor with shape (frames, channels, height, width)."""

    video_frames = []
    yy, xx = t.meshgrid(t.arange(size), t.arange(size), indexing="ij")
    for frame_index in range(frames):
        image = t.zeros(3, size, size)
        if kind in {"red_square", "red_square_late_occluded"}:
            x0 = 8 + frame_index * 4
            y0 = 32
            image[0, y0 : y0 + 24, x0 : x0 + 24] = 1.0
        elif kind == "red_square_shifted":
            x0 = 10 + frame_index * 4
            y0 = 34
            image[0, y0 : y0 + 24, x0 : x0 + 24] = 1.0
        elif kind == "blue_circle":
            center_x = 48
            center_y = 24 + frame_index * 4
            mask = (xx - center_x).pow(2) + (yy - center_y).pow(2) <= 12**2
            image[2, mask] = 1.0
        elif kind == "red_square_absent_occluder":
            pass
        else:
            raise ValueError(f"Unknown synthetic V-JEPA video kind: {kind}")
        if kind in {"red_square_late_occluded", "red_square_absent_occluder"} and frame_index >= 4:
            image[:, 28:62, 4:66] = 0.5
        video_frames.append((image - 0.5) / 0.5)
    return t.stack(video_frames)


def _draw_object_video(
    object_kind: str,
    x: int,
    y: int,
    *,
    dx: int = 0,
    dy: int = 0,
    frames: int = 8,
    size: int = 96,
    occlude_late: bool = False,
    absent: bool = False,
) -> t.Tensor:
    """Create a deterministic video for one labeled object/action state."""

    yy, xx = t.meshgrid(t.arange(size), t.arange(size), indexing="ij")
    video_frames = []
    for frame_index in range(frames):
        image = t.zeros(3, size, size)
        if not absent:
            progress = frame_index / max(frames - 1, 1)
            x0 = int(x + dx * progress)
            y0 = int(y + dy * progress)
            if object_kind == "red_square":
                image[0, y0 : y0 + 20, x0 : x0 + 20] = 1.0
            elif object_kind == "blue_circle":
                center_x = x0 + 10
                center_y = y0 + 10
                mask = (xx - center_x).pow(2) + (yy - center_y).pow(2) <= 10**2
                image[2, mask] = 1.0
            else:
                raise ValueError(f"Unknown object kind: {object_kind}")
        if occlude_late and frame_index >= frames // 2:
            image[:, max(0, y - 4) : min(size, y + 28), max(0, x - 4) : min(size, x + 28)] = 0.5
        video_frames.append((image - 0.5) / 0.5)
    return t.stack(video_frames)


def _build_synthetic_vjepa_world_batch(*, size: int = 96) -> SyntheticVJEPAWorldBatch:
    """Build a balanced generated video dataset with exact state/action labels."""

    x_positions = [8, 20, 32, 44, 56]
    y_positions = [12, 28, 44, 60]
    videos = []
    labels = []
    x_buckets = []
    action_ids = []
    next_videos = []
    occluded_videos = []
    absent_videos = []
    bboxes: list[tuple[int, int, int, int]] = []
    for object_kind, label in (("red_square", 0), ("blue_circle", 1)):
        for x_bucket, x in enumerate(x_positions):
            for y in y_positions:
                for action_id, (dx, dy) in enumerate(VJEPA2_ACTIONS):
                    next_x = max(4, min(size - 24, x + dx))
                    next_y = max(4, min(size - 24, y + dy))
                    videos.append(_draw_object_video(object_kind, x, y, size=size))
                    labels.append(label)
                    x_buckets.append(x_bucket)
                    action_ids.append(action_id)
                    next_videos.append(_draw_object_video(object_kind, next_x, next_y, size=size))
                    occluded_videos.append(
                        _draw_object_video(object_kind, x, y, size=size, occlude_late=True)
                    )
                    absent_videos.append(
                        _draw_object_video(
                            object_kind,
                            x,
                            y,
                            size=size,
                            occlude_late=True,
                            absent=True,
                        )
                    )
                    bboxes.append((x, y, x + 20, y + 20))
    return SyntheticVJEPAWorldBatch(
        videos=t.stack(videos),
        labels=t.tensor(labels, dtype=t.long),
        x_buckets=t.tensor(x_buckets, dtype=t.long),
        action_ids=t.tensor(action_ids, dtype=t.long),
        next_videos=t.stack(next_videos),
        occluded_videos=t.stack(occluded_videos),
        absent_videos=t.stack(absent_videos),
        bboxes=bboxes,
    )


def _bbox_to_vjepa_tokens(
    bbox: tuple[int, int, int, int],
    *,
    image_size: int = 96,
    token_grid: int = 12,
    pad: int = 4,
) -> list[int]:
    """Map an image-space object box to the corresponding 12x12 V-JEPA token grid."""

    x0, y0, x1, y1 = bbox
    cell = image_size / token_grid
    tokens = []
    for gy in range(token_grid):
        for gx in range(token_grid):
            cx = (gx + 0.5) * cell
            cy = (gy + 0.5) * cell
            if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
                tokens.append(gy * token_grid + gx)
    return tokens or [token_grid * token_grid // 2]


def _extract_vjepa2_latents(
    model: t.nn.Module,
    videos: t.Tensor,
    *,
    device: t.device,
    batch_size: int = 16,
) -> t.Tensor:
    """Extract frozen V-JEPA 2 encoder latents for a generated video batch."""

    outputs = []
    for start in range(0, videos.shape[0], batch_size):
        batch = videos[start : start + batch_size].to(device=device, dtype=t.float16)
        with t.inference_mode():
            output = model(pixel_values_videos=batch, skip_predictor=True)
        outputs.append(output.last_hidden_state.detach().float().cpu())
    return t.cat(outputs, dim=0)


def _stratified_split(labels: t.Tensor, *, train_per_label: int = 75) -> tuple[t.Tensor, t.Tensor]:
    """Return deterministic balanced train/test indices."""

    train_indices = []
    test_indices = []
    for label in sorted(labels.unique().tolist()):
        group = t.nonzero(labels == label, as_tuple=False).flatten()
        generator = t.Generator().manual_seed(int(label))
        shuffled = group[t.randperm(group.numel(), generator=generator)]
        train_indices.append(shuffled[:train_per_label])
        test_indices.append(shuffled[train_per_label:])
    return t.cat(train_indices), t.cat(test_indices)


def _nearest_neighbor_accuracy(
    train_features: t.Tensor,
    train_labels: t.Tensor,
    test_features: t.Tensor,
    test_labels: t.Tensor,
) -> float:
    """Classify held-out features with nearest train feature under cosine distance."""

    train_norm = F.normalize(train_features.float(), dim=-1)
    test_norm = F.normalize(test_features.float(), dim=-1)
    nearest = (test_norm @ train_norm.T).argmax(dim=-1)
    predictions = train_labels[nearest]
    return predictions.eq(test_labels).float().mean().item()


def _train_linear_classifier(
    train_features: t.Tensor,
    train_labels: t.Tensor,
    test_features: t.Tensor,
    test_labels: t.Tensor,
    *,
    classes: int,
    steps: int = 240,
    lr: float = 2e-2,
) -> tuple[t.nn.Linear, float]:
    """Train a small held-out probe on frozen latents."""

    probe = t.nn.Linear(train_features.shape[-1], classes, device=train_features.device)
    optimizer = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        loss = F.cross_entropy(probe(train_features), train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with t.inference_mode():
        accuracy = probe(test_features).argmax(dim=-1).eq(test_labels).float().mean().item()
    return probe, accuracy


def _train_mlp_predictor(
    train_inputs: t.Tensor,
    train_targets: t.Tensor,
    test_inputs: t.Tensor,
    test_targets: t.Tensor,
    *,
    action_count: int | None = None,
    train_actions: t.Tensor | None = None,
    test_actions: t.Tensor | None = None,
    steps: int = 500,
) -> tuple[t.nn.Module, float, float]:
    """Train a small MLP predictor and return initial/final held-out MSE."""

    input_dim = train_inputs.shape[-1] + (action_count or 0)
    predictor = t.nn.Sequential(
        t.nn.Linear(input_dim, 256, device=train_inputs.device),
        t.nn.GELU(),
        t.nn.Linear(256, train_targets.shape[-1], device=train_inputs.device),
    )
    optimizer = t.optim.AdamW(predictor.parameters(), lr=2e-3, weight_decay=1e-4)

    def _concat_actions(features: t.Tensor, actions: t.Tensor | None) -> t.Tensor:
        if action_count is None:
            return features
        if actions is None:
            raise ValueError("actions are required when action_count is set.")
        action_features = F.one_hot(actions, num_classes=action_count).float()
        return t.cat([features, action_features], dim=-1)

    train_x = _concat_actions(train_inputs, train_actions)
    test_x = _concat_actions(test_inputs, test_actions)
    with t.inference_mode():
        initial_loss = F.mse_loss(predictor(test_x), test_targets).item()
    for _ in range(steps):
        loss = F.mse_loss(predictor(train_x), train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with t.inference_mode():
        final_loss = F.mse_loss(predictor(test_x), test_targets).item()
    return predictor, initial_loss, final_loss


def vjepa2_feature_extraction_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run pinned V-JEPA 2 ViT-L feature extraction on deterministic synthetic videos."""

    if not t.cuda.is_available():
        raise RuntimeError("14.1 V-JEPA 2 feature extraction preflight requires CUDA.")

    from huggingface_hub import snapshot_download
    from transformers import VJEPA2Model

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    local_snapshot = snapshot_download(
        REAL_VJEPA2_MODEL_ID,
        revision=REAL_VJEPA2_REVISION,
        allow_patterns=[
            "config.json",
            "model.safetensors",
        ],
    )
    model = VJEPA2Model.from_pretrained(
        local_snapshot,
        dtype=t.float16,
    ).to(device)
    model.eval()
    videos = t.stack([_synthetic_vjepa_video(kind) for kind in VJEPA2_VIDEO_KINDS]).to(
        device=device,
        dtype=t.float16,
    )
    with t.inference_mode():
        output = model(pixel_values_videos=videos, skip_predictor=True)
    features = output.last_hidden_state.detach().float()
    pooled = F.normalize(features.mean(dim=1), dim=-1)
    cosine = pooled @ pooled.T
    same_object_cosine = cosine[0, 1].item()
    different_object_cosine = t.stack([cosine[0, 2], cosine[1, 2]]).max().item()
    same_object_margin = same_object_cosine - different_object_cosine
    late_occluded_cosine = cosine[0, 3].item()
    absent_occluder_cosine = cosine[0, 4].item()
    occluded_vs_absent_gap = late_occluded_cosine - absent_occluder_cosine
    occluded_vs_blue_gap = late_occluded_cosine - cosine[0, 2].item()
    permanence = object_permanence_report(
        visible_scores=t.tensor([same_object_cosine]),
        occluded_scores=t.tensor([late_occluded_cosine]),
        absent_scores=t.tensor([absent_occluder_cosine]),
        min_occluded_score=0.9,
        min_absent_gap=0.2,
    )
    finite_features = bool(t.isfinite(features).all().item())
    feature_std = features.std().item()
    feature_shape = list(features.shape)
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3

    del output, features, pooled, cosine, videos, model
    t.cuda.empty_cache()

    preflight_passed = (
        finite_features
        and feature_std > 0.1
        and same_object_margin >= 0.03
        and permanence.preserves_occluded_object
        and occluded_vs_blue_gap >= 0.03
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "cuda_available": True,
        "model_id": REAL_VJEPA2_MODEL_ID,
        "revision": REAL_VJEPA2_REVISION,
        "local_snapshot": str(local_snapshot),
        "claim_scope": "pinned_vjepa2_vitl_synthetic_video_feature_extraction_preflight",
        "video_kinds": list(VJEPA2_VIDEO_KINDS),
        "video_count": len(VJEPA2_VIDEO_KINDS),
        "frames_per_video": 8,
        "input_size": 96,
        "feature_shape": feature_shape,
        "tokens_per_video": feature_shape[1],
        "hidden_size": feature_shape[2],
        "finite_features": finite_features,
        "feature_std": feature_std,
        "same_object_cosine": same_object_cosine,
        "different_object_cosine": different_object_cosine,
        "same_object_margin": same_object_margin,
        "same_object_beats_different_object": same_object_margin >= 0.03,
        "late_occluded_cosine": late_occluded_cosine,
        "absent_occluder_cosine": absent_occluder_cosine,
        "occluded_vs_absent_gap": occluded_vs_absent_gap,
        "occluded_vs_blue_gap": occluded_vs_blue_gap,
        "synthetic_occlusion_permanence": permanence.__dict__,
        "synthetic_occlusion_permanence_passed": permanence.preserves_occluded_object
        and occluded_vs_blue_gap >= 0.03,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": preflight_passed,
    }


def vjepa2_world_model_control_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run frozen V-JEPA 2 latent probes, rollout heads, and causal patch controls."""

    if not t.cuda.is_available():
        raise RuntimeError("14.1 V-JEPA 2 world-model control preflight requires CUDA.")

    from huggingface_hub import snapshot_download
    from transformers import VJEPA2Model

    device = t.device("cuda")
    t.manual_seed(0)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(0)
    t.cuda.reset_peak_memory_stats()
    local_snapshot = snapshot_download(
        REAL_VJEPA2_MODEL_ID,
        revision=REAL_VJEPA2_REVISION,
        allow_patterns=[
            "config.json",
            "model.safetensors",
        ],
    )
    model = VJEPA2Model.from_pretrained(local_snapshot, dtype=t.float16).to(device)
    model.eval()

    batch = _build_synthetic_vjepa_world_batch()
    features = _extract_vjepa2_latents(model, batch.videos, device=device)
    next_features = _extract_vjepa2_latents(model, batch.next_videos, device=device)
    occluded_features = _extract_vjepa2_latents(model, batch.occluded_videos, device=device)
    absent_features = _extract_vjepa2_latents(model, batch.absent_videos, device=device)
    del model

    raw_pooled = features.mean(dim=1)
    pooled = F.normalize(raw_pooled, dim=-1)
    next_pooled = F.normalize(next_features.mean(dim=1), dim=-1)
    occluded_pooled = F.normalize(occluded_features.mean(dim=1), dim=-1)
    absent_pooled = F.normalize(absent_features.mean(dim=1), dim=-1)
    train_idx_cpu, test_idx_cpu = _stratified_split(batch.labels)
    train_idx = train_idx_cpu.to(device)
    test_idx = test_idx_cpu.to(device)

    collapse = collapse_diagnostics_report(
        raw_pooled,
        min_feature_std=0.5,
        min_effective_rank=1.5,
    )

    pooled_gpu = pooled.to(device)
    next_pooled_gpu = next_pooled.to(device)
    occluded_pooled_gpu = occluded_pooled.to(device)
    absent_pooled_gpu = absent_pooled.to(device)
    labels = batch.labels.to(device)
    action_ids = batch.action_ids.to(device)
    x_buckets = batch.x_buckets.to(device)

    state_probe, state_probe_accuracy = _train_linear_classifier(
        pooled_gpu[train_idx],
        labels[train_idx],
        pooled_gpu[test_idx],
        labels[test_idx],
        classes=2,
    )
    _ = state_probe
    state_probe_knn_accuracy = _nearest_neighbor_accuracy(
        pooled_gpu[train_idx],
        labels[train_idx],
        pooled_gpu[test_idx],
        labels[test_idx],
    )
    shuffled_train_labels = labels[train_idx][t.randperm(train_idx.numel(), device=device)]
    state_probe_shuffled_knn_accuracy = _nearest_neighbor_accuracy(
        pooled_gpu[train_idx],
        shuffled_train_labels,
        pooled_gpu[test_idx],
        labels[test_idx],
    )
    state_probe_margin = state_probe_knn_accuracy - state_probe_shuffled_knn_accuracy

    masked_predictor, masked_initial_loss, masked_final_loss = _train_mlp_predictor(
        occluded_pooled_gpu[train_idx],
        pooled_gpu[train_idx],
        occluded_pooled_gpu[test_idx],
        pooled_gpu[test_idx],
        steps=500,
    )
    with t.inference_mode():
        masked_copy_loss = F.mse_loss(
            occluded_pooled_gpu[test_idx],
            pooled_gpu[test_idx],
        ).item()
    masked_report = loss_decrease_report(
        masked_initial_loss,
        masked_final_loss,
        masked_copy_loss,
        min_relative_reduction=0.5,
        max_final_to_baseline=0.8,
    )

    transition_head, transition_initial_loss, transition_final_loss = _train_mlp_predictor(
        pooled_gpu[train_idx],
        next_pooled_gpu[train_idx],
        pooled_gpu[test_idx],
        next_pooled_gpu[test_idx],
        action_count=len(VJEPA2_ACTIONS),
        train_actions=action_ids[train_idx],
        test_actions=action_ids[test_idx],
        steps=700,
    )
    with t.inference_mode():
        transition_copy_loss = F.mse_loss(
            pooled_gpu[test_idx],
            next_pooled_gpu[test_idx],
        ).item()
        shuffled_test_actions = action_ids[test_idx][
            t.randperm(test_idx.numel(), device=device)
        ]
        shuffled_inputs = t.cat(
            [
                pooled_gpu[test_idx],
                F.one_hot(shuffled_test_actions, num_classes=len(VJEPA2_ACTIONS)).float(),
            ],
            dim=-1,
        )
        transition_shuffled_loss = F.mse_loss(
            transition_head(shuffled_inputs),
            next_pooled_gpu[test_idx],
        ).item()
    rollout = latent_rollout_report(
        transition_final_loss,
        transition_copy_loss,
        transition_shuffled_loss,
        max_rollout_to_copy=0.8,
        max_rollout_to_shuffled=0.8,
    )

    visible_occluded_cosine = (pooled_gpu * occluded_pooled_gpu).sum(dim=-1)
    visible_absent_cosine = (pooled_gpu * absent_pooled_gpu).sum(dim=-1)
    visible_self_cosine = (pooled_gpu * pooled_gpu).sum(dim=-1)
    permanence = object_permanence_report(
        visible_scores=visible_self_cosine,
        occluded_scores=visible_occluded_cosine,
        absent_scores=visible_absent_cosine,
        min_occluded_score=0.85,
        min_absent_gap=0.05,
    )

    feature_tokens = features.to(device)
    local_position_features = t.stack(
        [
            feature_tokens[index, _bbox_to_vjepa_tokens(bbox)].mean(dim=0)
            for index, bbox in enumerate(batch.bboxes)
        ]
    )
    position_probe, position_probe_accuracy = _train_linear_classifier(
        local_position_features[train_idx],
        x_buckets[train_idx],
        local_position_features[test_idx],
        x_buckets[test_idx],
        classes=5,
        steps=320,
    )
    object_patch_effects = []
    random_patch_effects = []
    for object_label in (0, 1):
        for y_position in (12, 28, 44, 60):
            source_candidates = [
                index
                for index, bbox in enumerate(batch.bboxes)
                if batch.labels[index].item() == object_label
                and bbox[0] == 56
                and bbox[1] == y_position
            ]
            target_candidates = [
                index
                for index, bbox in enumerate(batch.bboxes)
                if batch.labels[index].item() == object_label
                and bbox[0] == 8
                and bbox[1] == y_position
            ]
            for source_index, target_index in zip(source_candidates[:2], target_candidates[:2]):
                source_tokens = _bbox_to_vjepa_tokens(batch.bboxes[source_index])
                target_tokens = _bbox_to_vjepa_tokens(batch.bboxes[target_index])
                token_count = min(len(source_tokens), len(target_tokens))
                source_tokens = source_tokens[:token_count]
                target_tokens = target_tokens[:token_count]
                patched = feature_tokens[target_index].clone()
                patched[target_tokens] = feature_tokens[source_index, source_tokens]
                random_target_tokens = t.randperm(feature_tokens.shape[1], device=device)[
                    :token_count
                ]
                random_source_tokens = t.randperm(feature_tokens.shape[1], device=device)[
                    :token_count
                ]
                random_patched = feature_tokens[target_index].clone()
                random_patched[random_target_tokens] = feature_tokens[
                    source_index,
                    random_source_tokens,
                ]
                with t.inference_mode():
                    base_source_x_prob = F.softmax(
                        position_probe(
                            feature_tokens[target_index, target_tokens]
                            .mean(dim=0, keepdim=True)
                        ),
                        dim=-1,
                    )[0, 4]
                    object_source_x_prob = F.softmax(
                        position_probe(patched[target_tokens].mean(dim=0, keepdim=True)),
                        dim=-1,
                    )[0, 4]
                    random_source_x_prob = F.softmax(
                        position_probe(
                            random_patched[target_tokens].mean(dim=0, keepdim=True)
                        ),
                        dim=-1,
                    )[0, 4]
                object_patch_effects.append(object_source_x_prob - base_source_x_prob)
                random_patch_effects.append(random_source_x_prob - base_source_x_prob)
    patching = causal_latent_patch_report(
        t.stack(object_patch_effects),
        t.stack(random_patch_effects),
        min_object_patch_effect=0.5,
        min_patch_random_gap=0.4,
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        collapse.non_collapsed
        and masked_report.loss_decreases
        and state_probe_accuracy >= 0.8
        and state_probe_margin >= 0.2
        and rollout.rollout_passes
        and permanence.preserves_occluded_object
        and position_probe_accuracy >= 0.7
        and patching.causal_patch_passes
        and within_vram_budget
    )

    del (
        features,
        next_features,
        occluded_features,
        absent_features,
        feature_tokens,
        pooled_gpu,
        next_pooled_gpu,
        occluded_pooled_gpu,
        absent_pooled_gpu,
        masked_predictor,
        transition_head,
        position_probe,
    )
    t.cuda.empty_cache()

    return {
        "cuda_available": True,
        "model_id": REAL_VJEPA2_MODEL_ID,
        "revision": REAL_VJEPA2_REVISION,
        "claim_scope": "pinned_vjepa2_vitl_frozen_latent_world_model_controls",
        "video_count": int(batch.videos.shape[0]),
        "frames_per_video": int(batch.videos.shape[1]),
        "input_size": 96,
        "feature_shape": list(raw_pooled.shape),
        "token_feature_shape": [int(v) for v in [batch.videos.shape[0], 144, 1024]],
        "collapse_diagnostics": collapse.__dict__,
        "vjepa2_non_collapsed": collapse.non_collapsed,
        "masked_prediction": masked_report.__dict__,
        "masked_prediction_loss_reduction": masked_report.relative_reduction,
        "masked_prediction_final_loss": masked_report.final_loss,
        "masked_prediction_copy_baseline_loss": masked_report.baseline_loss,
        "masked_prediction_passed": masked_report.loss_decreases,
        "state_probe_accuracy": state_probe_accuracy,
        "state_probe_knn_accuracy": state_probe_knn_accuracy,
        "state_probe_shuffled_knn_accuracy": state_probe_shuffled_knn_accuracy,
        "state_probe_margin_over_random": state_probe_margin,
        "state_probe_beats_random_baseline": state_probe_margin >= 0.2,
        "latent_rollout": rollout.__dict__,
        "latent_rollout_initial_loss": transition_initial_loss,
        "latent_rollout_final_loss": rollout.rollout_loss,
        "latent_rollout_copy_baseline_loss": rollout.copy_baseline_loss,
        "latent_rollout_shuffled_action_loss": rollout.shuffled_action_loss,
        "latent_rollout_passed": rollout.rollout_passes,
        "shuffled_action_baseline_failed": rollout.shuffled_action_fails,
        "object_permanence": permanence.__dict__,
        "real_latent_object_permanence_passed": permanence.preserves_occluded_object,
        "position_probe_accuracy": position_probe_accuracy,
        "causal_latent_patching": patching.__dict__,
        "causal_latent_patching_passed": patching.causal_patch_passes,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "preflight_passed": preflight_passed,
    }


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "jepa_prediction": jepa_prediction_smoke_test(),
        "collapse": collapse_diagnostics_smoke_test(),
        "state_probe": state_probe_smoke_test(),
        "state_probe_control": state_probe_control_smoke_test(),
        "transition": transition_smoke_test(),
        "rollout_control": rollout_control_smoke_test(),
        "object_permanence": object_permanence_smoke_test(),
        "object_permanence_control": object_permanence_control_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("14.1 GPU verification requires CUDA.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    target_embeddings = t.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    prediction = jepa_prediction_report(
        target_embeddings.clone(),
        target_embeddings,
        min_cosine=0.99,
        max_mse=0.01,
    )
    transition = transition_consistency_report(
        t.tensor([[0.0, 0.0], [1.0, 1.0]], device=device),
        t.tensor([[1.0, 0.0], [0.0, 1.0]], device=device),
        t.tensor([[1.0, 0.0], [1.0, 2.0]], device=device),
        min_cosine=0.99,
    )
    t.cuda.synchronize()
    synthetic_peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    vjepa2 = vjepa2_feature_extraction_preflight(max_vram_gb=max_vram_gb)
    world_controls = vjepa2_world_model_control_preflight(max_vram_gb=max_vram_gb)
    peak_vram_gb = max(
        synthetic_peak_vram_gb,
        vjepa2["peak_vram_gb"],
        world_controls["peak_vram_gb"],
    )
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "mean_cosine": prediction.mean_cosine,
        "prediction_mse": prediction.mse,
        "predicts_target": prediction.predicts_target,
        "jepa_predicts_target": prediction.predicts_target,
        "transition_consistent": transition.transition_consistent,
        "vjepa2_preflight_passed": vjepa2["preflight_passed"],
        "vjepa2_world_model_controls_passed": world_controls["preflight_passed"],
        "vjepa2_feature_shape": vjepa2["feature_shape"],
        "vjepa2_feature_std": vjepa2["feature_std"],
        "vjepa2_world_feature_shape": world_controls["feature_shape"],
        "vjepa2_token_feature_shape": world_controls["token_feature_shape"],
        "vjepa2_non_collapsed": world_controls["vjepa2_non_collapsed"],
        "vjepa2_same_object_cosine": vjepa2["same_object_cosine"],
        "vjepa2_different_object_cosine": vjepa2["different_object_cosine"],
        "vjepa2_same_object_margin": vjepa2["same_object_margin"],
        "vjepa2_same_object_beats_different_object": vjepa2[
            "same_object_beats_different_object"
        ],
        "vjepa2_late_occluded_cosine": vjepa2["late_occluded_cosine"],
        "vjepa2_absent_occluder_cosine": vjepa2["absent_occluder_cosine"],
        "vjepa2_occluded_vs_absent_gap": vjepa2["occluded_vs_absent_gap"],
        "vjepa2_occluded_vs_blue_gap": vjepa2["occluded_vs_blue_gap"],
        "vjepa2_synthetic_occlusion_permanence_passed": vjepa2[
            "synthetic_occlusion_permanence_passed"
        ],
        "masked_prediction_loss_reduction": world_controls[
            "masked_prediction_loss_reduction"
        ],
        "masked_prediction_final_loss": world_controls["masked_prediction_final_loss"],
        "masked_prediction_copy_baseline_loss": world_controls[
            "masked_prediction_copy_baseline_loss"
        ],
        "masked_prediction_passed": world_controls["masked_prediction_passed"],
        "state_probe_accuracy": world_controls["state_probe_accuracy"],
        "state_probe_knn_accuracy": world_controls["state_probe_knn_accuracy"],
        "state_probe_shuffled_knn_accuracy": world_controls[
            "state_probe_shuffled_knn_accuracy"
        ],
        "state_probe_margin_over_random": world_controls[
            "state_probe_margin_over_random"
        ],
        "state_probe_beats_random_baseline": world_controls[
            "state_probe_beats_random_baseline"
        ],
        "latent_rollout_final_loss": world_controls["latent_rollout_final_loss"],
        "latent_rollout_copy_baseline_loss": world_controls[
            "latent_rollout_copy_baseline_loss"
        ],
        "latent_rollout_shuffled_action_loss": world_controls[
            "latent_rollout_shuffled_action_loss"
        ],
        "latent_rollout_passed": world_controls["latent_rollout_passed"],
        "shuffled_action_baseline_failed": world_controls[
            "shuffled_action_baseline_failed"
        ],
        "real_latent_object_permanence_passed": world_controls[
            "real_latent_object_permanence_passed"
        ],
        "position_probe_accuracy": world_controls["position_probe_accuracy"],
        "causal_latent_patching_passed": world_controls[
            "causal_latent_patching_passed"
        ],
        "causal_latent_patch_effect": world_controls["causal_latent_patching"][
            "object_patch_effect"
        ],
        "causal_latent_random_patch_effect": world_controls["causal_latent_patching"][
            "random_patch_effect"
        ],
        "causal_latent_patch_random_gap": world_controls["causal_latent_patching"][
            "patch_random_gap"
        ],
        "vjepa2_peak_vram_gb": vjepa2["peak_vram_gb"],
        "vjepa2_world_controls_peak_vram_gb": world_controls["peak_vram_gb"],
        "vjepa2_preflight": vjepa2,
        "vjepa2_world_controls": world_controls,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": (
            peak_vram_gb <= max_vram_gb
            and vjepa2["within_vram_budget"]
            and world_controls["within_vram_budget"]
        ),
        "full_path": (
            "Validated JEPA target prediction, transition consistency, pinned "
            "V-JEPA 2 ViT-L synthetic-video feature extraction, frozen-latent "
            "masked prediction, held-out state probes, action-conditioned rollout, "
            "object-permanence controls, and causal latent-token patching."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
