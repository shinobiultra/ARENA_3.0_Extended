from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch as t
from torch import Tensor, nn
from torch.nn import functional as F


IMAGE_SIZE = 16
N_DIFFUSION_STEPS = 8
N_SHAPES = 2
N_COLORS = 3
N_ROWS = 2
N_COLUMNS = 2
PATCH_LAYERS = ("early", "concept", "middle", "late")
REAL_SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REAL_SD15_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
REAL_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
REAL_CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
REAL_SIGNATURE_ASSET = (
    Path(__file__).resolve().parents[2]
    / "instructions/assets/denoising_time_sd15_signature.png"
)
REAL_SD15_NEGATIVE_PROMPT = (
    "multiple objects, repeated pattern, tiled, grid, texture, text, letters, "
    "watermark, frame, border, shadow, clutter, photo, realistic, wood"
)
REAL_SD15_CASES = (
    {
        "case_id": "red_into_blue_square",
        "seed": 4,
        "donor_prompt": (
            "a single centered solid red square, plain white background, flat vector "
            "icon, minimal geometric shape"
        ),
        "recipient_prompt": (
            "a single centered solid blue square, plain white background, flat vector "
            "icon, minimal geometric shape"
        ),
        "donor_text": "a red square icon",
        "recipient_text": "a blue square icon",
        "target_color": "red",
    },
    {
        "case_id": "blue_into_red_circle",
        "seed": 2,
        "donor_prompt": (
            "a single centered solid blue circle, plain white background, flat vector "
            "icon, minimal geometric shape"
        ),
        "recipient_prompt": (
            "a single centered solid red circle, plain white background, flat vector "
            "icon, minimal geometric shape"
        ),
        "donor_text": "a blue circle icon",
        "recipient_text": "a red circle icon",
        "target_color": "blue",
    },
)
COLOR_VALUES = t.tensor(
    [
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
)


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: Tensor
    alphas: Tensor
    alpha_bars: Tensor


@dataclass
class ActivationCache:
    values: dict[tuple[int, str], Tensor]
    timestep: int | None = None


@dataclass(frozen=True)
class ActivationPatch:
    layer: str
    apply_timestep: int
    donor_timestep: int
    donor_cache: dict[tuple[int, str], Tensor]
    channels: Tensor
    spatial_mask: Tensor


@dataclass(frozen=True)
class RegionalCausalMetric:
    recovery: float
    outside_change: float
    selectivity: float


@dataclass(frozen=True)
class TrainingTrace:
    first_loss: float
    final_loss: float
    losses: tuple[float, ...]


@dataclass(frozen=True)
class LatentTransferMetric:
    donor_distance: float
    recipient_distance: float
    recovery: float
    outside_change: float
    selectivity: float


@dataclass(frozen=True)
class SD15Trajectory:
    image: Any
    image_tensor: Tensor
    latent_states: tuple[Tensor, ...]
    scheduler_timesteps: tuple[int, ...]


@dataclass(frozen=True)
class SD15PatchControlReport:
    best_step_index: int
    best_scheduler_timestep: int
    best_clip_recovery: float
    best_regional_selectivity: float
    wrong_timestep_recovery: float
    wrong_region_recovery: float
    random_latent_recovery: float
    unpatched_recovery: float
    target_beats_controls: bool


def make_diffusion_schedule(
    n_steps: int = N_DIFFUSION_STEPS,
    beta_start: float = 0.03,
    beta_end: float = 0.22,
    *,
    device: t.device | str = "cpu",
) -> DiffusionSchedule:
    """Return a schedule with an exact clean endpoint at timestep zero."""
    if n_steps < 2:
        raise ValueError("n_steps must be at least two so early/late controls exist.")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("Require 0 < beta_start < beta_end < 1.")
    betas = t.cat(
        [
            t.zeros(1, device=device),
            t.linspace(beta_start, beta_end, n_steps, device=device),
        ]
    )
    alphas = 1.0 - betas
    alpha_bars = t.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def all_world_labels(*, device: t.device | str = "cpu") -> Tensor:
    """Enumerate shape, color, row, and column labels without sampling."""
    labels = [
        (shape, color, row, column)
        for shape in range(N_SHAPES)
        for color in range(N_COLORS)
        for row in range(N_ROWS)
        for column in range(N_COLUMNS)
    ]
    return t.tensor(labels, dtype=t.long, device=device)


def render_object_masks(labels: Tensor, size: int = IMAGE_SIZE) -> Tensor:
    """Render exact square/circle masks from integer world-state labels."""
    if labels.ndim != 2 or labels.shape[1] != 4:
        raise ValueError("labels must have shape [batch, 4].")
    if size < 8:
        raise ValueError("size must be at least eight pixels.")
    if labels.numel() == 0:
        raise ValueError("labels cannot be empty.")
    limits = t.tensor(
        [N_SHAPES, N_COLORS, N_ROWS, N_COLUMNS], device=labels.device
    )
    if ((labels < 0) | (labels >= limits)).any():
        raise ValueError("A shape/color/row/column label is out of range.")

    coordinates = t.arange(size, device=labels.device, dtype=t.float32)
    yy, xx = t.meshgrid(coordinates, coordinates, indexing="ij")
    quarter = size / 4.0
    centers_y = quarter + labels[:, 2].float() * (size / 2.0)
    centers_x = quarter + labels[:, 3].float() * (size / 2.0)
    dy = yy.unsqueeze(0) - centers_y[:, None, None]
    dx = xx.unsqueeze(0) - centers_x[:, None, None]
    radius = size * 0.17
    square = (dx.abs() <= radius) & (dy.abs() <= radius)
    circle = dx.square() + dy.square() <= radius**2
    return t.where(labels[:, 0, None, None] == 0, square, circle)


def render_object_world(labels: Tensor, size: int = IMAGE_SIZE) -> tuple[Tensor, Tensor]:
    """Render deterministic RGB images in [-1, 1] and their object masks."""
    masks = render_object_masks(labels, size)
    colors = COLOR_VALUES.to(labels.device)[labels[:, 1]]
    images = -t.ones(labels.shape[0], 3, size, size, device=labels.device)
    images = t.where(masks[:, None], colors[:, :, None, None], images)
    return images, masks


def q_sample(
    x_start: Tensor,
    timesteps: Tensor,
    noise: Tensor,
    schedule: DiffusionSchedule,
) -> Tensor:
    """Sample q(x_t | x_0) with one independently testable broadcast operation."""
    if x_start.shape != noise.shape:
        raise ValueError("x_start and noise must have identical shapes.")
    if timesteps.shape != (x_start.shape[0],):
        raise ValueError("timesteps must have shape [batch].")
    if timesteps.min() < 0 or timesteps.max() >= len(schedule.alpha_bars):
        raise ValueError("A timestep lies outside the schedule.")
    alpha_bar = schedule.alpha_bars[timesteps].view(-1, 1, 1, 1)
    return alpha_bar.sqrt() * x_start + (1.0 - alpha_bar).sqrt() * noise


def sinusoidal_timestep_embedding(timesteps: Tensor, width: int) -> Tensor:
    """Create the standard paired sine/cosine diffusion timestep embedding."""
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be rank one.")
    if width < 4 or width % 2:
        raise ValueError("width must be an even integer of at least four.")
    half = width // 2
    frequencies = t.exp(
        -math.log(10_000.0)
        * t.arange(half, device=timesteps.device, dtype=t.float32)
        / max(half - 1, 1)
    )
    phases = timesteps.float()[:, None] * frequencies[None, :]
    return t.cat([phases.sin(), phases.cos()], dim=-1)


class ResidualConvBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )

    def forward(self, activations: Tensor) -> Tensor:
        return activations + self.net(activations)


class HookableObjectDenoiser(nn.Module):
    """Small x0-predicting denoiser with an explicit object-feature bottleneck."""

    def __init__(self, width: int = 12, concept_channels: int = 4):
        super().__init__()
        if width % 4:
            raise ValueError("width must be divisible by four for GroupNorm.")
        if not 1 <= concept_channels <= width // 2:
            raise ValueError("concept_channels must occupy at most half the model width.")
        self.width = width
        self.concept_channels = concept_channels
        self.input_projection = nn.Conv2d(5, width, kernel_size=3, padding=1)
        self.time_projection = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.shape_embedding = nn.Embedding(N_SHAPES, concept_channels)
        self.color_embedding = nn.Embedding(N_COLORS, concept_channels)
        self.blocks = nn.ModuleDict(
            {
                "early": ResidualConvBlock(width),
                "concept": nn.Identity(),
                "middle": ResidualConvBlock(width),
                "late": ResidualConvBlock(width),
            }
        )
        self.output_projection = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(width, 3, kernel_size=3, padding=1),
        )

    def patch_layer(self, name: str) -> nn.Module:
        if name not in self.blocks:
            raise KeyError(f"Unknown patch layer {name!r}; choose from {PATCH_LAYERS}.")
        return self.blocks[name]

    def condition_feature(self, labels: Tensor, spatial_size: int) -> Tensor:
        masks = render_object_masks(labels, spatial_size).float()
        vectors = self.shape_embedding(labels[:, 0]) + self.color_embedding(labels[:, 1])
        feature = vectors[:, :, None, None] * masks[:, None]
        padding = self.width - self.concept_channels
        return F.pad(feature, (0, 0, 0, 0, 0, padding))

    def forward(self, noisy_images: Tensor, timesteps: Tensor, labels: Tensor) -> Tensor:
        if noisy_images.ndim != 4 or noisy_images.shape[1] != 3:
            raise ValueError("noisy_images must have shape [batch, 3, height, width].")
        if labels.shape != (noisy_images.shape[0], 4):
            raise ValueError("labels must have shape [batch, 4].")
        batch, _, height, width = noisy_images.shape
        y = t.linspace(-1.0, 1.0, height, device=noisy_images.device)
        x = t.linspace(-1.0, 1.0, width, device=noisy_images.device)
        yy, xx = t.meshgrid(y, x, indexing="ij")
        coordinates = t.stack([yy, xx]).expand(batch, -1, -1, -1)
        hidden = self.input_projection(t.cat([noisy_images, coordinates], dim=1))
        time_bias = self.time_projection(
            sinusoidal_timestep_embedding(timesteps, self.width)
        )[:, :, None, None]
        hidden = self.blocks["early"](hidden + time_bias)
        hidden = self.blocks["concept"](
            hidden + self.condition_feature(labels, height)
        )
        hidden = self.blocks["middle"](hidden)
        hidden = self.blocks["late"](hidden)
        return self.output_projection(hidden)


def register_activation_cache(
    model: HookableObjectDenoiser,
    cache: ActivationCache,
    layers: Iterable[str] = PATCH_LAYERS,
) -> list[Any]:
    """Register learner-visible forward hooks keyed by timestep and layer."""
    handles = []
    for layer in layers:
        module = model.patch_layer(layer)

        def save_activation(_module, _inputs, output, *, layer_name=layer):
            if cache.timestep is None:
                raise RuntimeError("Set cache.timestep before running the denoiser.")
            cache.values[(cache.timestep, layer_name)] = output.detach().clone()

        handles.append(module.register_forward_hook(save_activation))
    return handles


def make_activation_replacement_hook(
    donor_activation: Tensor,
    channels: Tensor,
    spatial_mask: Tensor,
):
    """Return a hook replacing exactly the requested channels and locations."""
    if donor_activation.ndim != 4:
        raise ValueError("donor_activation must have shape [batch, channels, height, width].")
    channels = channels.long().flatten()
    if channels.numel() == 0 or channels.min() < 0 or channels.max() >= donor_activation.shape[1]:
        raise ValueError("channels must be a non-empty valid channel index set.")
    mask = spatial_mask.bool()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape != (donor_activation.shape[0], *donor_activation.shape[-2:]):
        raise ValueError("spatial_mask must align with donor batch and spatial dimensions.")

    def replace(_module, _inputs, recipient_activation):
        if recipient_activation.shape != donor_activation.shape:
            raise ValueError("Donor and recipient activations must have identical shapes.")
        patched = recipient_activation.clone()
        selector = mask[:, None].expand(-1, channels.numel(), -1, -1)
        recipient_slice = patched[:, channels]
        donor_slice = donor_activation.to(patched.device)[:, channels]
        patched[:, channels] = t.where(selector, donor_slice, recipient_slice)
        return patched

    return replace


@t.inference_mode()
def denoise_trajectory(
    model: HookableObjectDenoiser,
    labels: Tensor,
    initial_noise: Tensor,
    schedule: DiffusionSchedule,
    *,
    cache: ActivationCache | None = None,
    patch: ActivationPatch | None = None,
) -> tuple[Tensor, dict[int, Tensor]]:
    """Run deterministic DDIM-style x0-prediction sampling with optional patching."""
    if initial_noise.shape[0] != labels.shape[0]:
        raise ValueError("initial_noise and labels must share a batch dimension.")
    device = next(model.parameters()).device
    labels = labels.to(device)
    current = initial_noise.to(device).clone()
    states = {N_DIFFUSION_STEPS: current.detach().cpu().clone()}
    handles = register_activation_cache(model, cache) if cache is not None else []
    try:
        for step in range(N_DIFFUSION_STEPS, 0, -1):
            timestep = t.full((labels.shape[0],), step, device=device, dtype=t.long)
            if cache is not None:
                cache.timestep = step
            patch_handle = None
            if patch is not None and step == patch.apply_timestep:
                donor = patch.donor_cache[(patch.donor_timestep, patch.layer)]
                hook = make_activation_replacement_hook(
                    donor.to(device), patch.channels.to(device), patch.spatial_mask.to(device)
                )
                patch_handle = model.patch_layer(patch.layer).register_forward_hook(hook)
            predicted_x0 = model(current, timestep, labels).clamp(-1.0, 1.0)
            if patch_handle is not None:
                patch_handle.remove()
            if step == 1:
                current = predicted_x0
            else:
                alpha_bar = schedule.alpha_bars[step].to(device)
                previous_alpha_bar = schedule.alpha_bars[step - 1].to(device)
                predicted_noise = (
                    current - alpha_bar.sqrt() * predicted_x0
                ) / (1.0 - alpha_bar).sqrt().clamp_min(1e-6)
                current = (
                    previous_alpha_bar.sqrt() * predicted_x0
                    + (1.0 - previous_alpha_bar).sqrt() * predicted_noise
                )
            states[step - 1] = current.detach().cpu().clone()
    finally:
        for handle in handles:
            handle.remove()
    return current.clamp(-1.0, 1.0), states


def regional_causal_metric(
    clean_image: Tensor,
    corrupt_image: Tensor,
    patched_image: Tensor,
    target_mask: Tensor,
) -> RegionalCausalMetric:
    """Measure normalized target recovery and penalize off-target image changes."""
    if clean_image.shape != corrupt_image.shape or clean_image.shape != patched_image.shape:
        raise ValueError("clean, corrupt, and patched images must have the same shape.")
    mask = target_mask.bool()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape != (clean_image.shape[0], *clean_image.shape[-2:]):
        raise ValueError("target_mask must align with the image batch and spatial dimensions.")
    inside = mask[:, None].expand_as(clean_image)
    outside = ~inside
    baseline_error = (corrupt_image[inside] - clean_image[inside]).square().mean()
    patched_error = (patched_image[inside] - clean_image[inside]).square().mean()
    recovery = ((baseline_error - patched_error) / baseline_error.clamp_min(1e-8)).item()
    outside_change = (
        (patched_image[outside] - corrupt_image[outside]).square().mean().sqrt().item()
        if outside.any()
        else 0.0
    )
    return RegionalCausalMetric(
        recovery=recovery,
        outside_change=outside_change,
        selectivity=recovery - outside_change,
    )


def train_object_denoiser(
    *,
    device: t.device | str = "cpu",
    steps: int = 350,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    seed: int = 0,
    width: int = 12,
    concept_channels: int = 4,
) -> tuple[HookableObjectDenoiser, TrainingTrace]:
    """Train the complete denoiser on generated labeled images, with no pretrained weights."""
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive.")
    device = t.device(device)
    t.manual_seed(seed)
    generator = t.Generator(device=device).manual_seed(seed + 1)
    schedule = make_diffusion_schedule(device=device)
    labels_table = all_world_labels(device=device)
    model = HookableObjectDenoiser(width, concept_channels).to(device)
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    previous_threads = t.get_num_threads()
    if device.type == "cpu" and previous_threads > 2:
        t.set_num_threads(2)
    model.train()
    try:
        for _ in range(steps):
            indices = t.randint(
                len(labels_table), (batch_size,), generator=generator, device=device
            )
            labels = labels_table[indices]
            clean, object_mask = render_object_world(labels)
            timesteps = t.randint(
                1,
                N_DIFFUSION_STEPS + 1,
                (batch_size,),
                generator=generator,
                device=device,
            )
            noise = t.randn(clean.shape, generator=generator, device=device)
            noisy = q_sample(clean, timesteps, noise, schedule)
            prediction = model(noisy, timesteps, labels)
            weights = 1.0 + 5.0 * object_mask[:, None].float()
            loss = ((prediction - clean).square() * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    finally:
        if device.type == "cpu" and t.get_num_threads() != previous_threads:
            t.set_num_threads(previous_threads)
    model.eval()
    window = min(25, len(losses))
    return model, TrainingTrace(
        first_loss=float(sum(losses[:window]) / window),
        final_loss=float(sum(losses[-window:]) / window),
        losses=tuple(losses),
    )


def _single_label(values: tuple[int, int, int, int], device: t.device) -> Tensor:
    return t.tensor([values], device=device, dtype=t.long)


def _patch_result(
    model: HookableObjectDenoiser,
    labels: Tensor,
    initial_noise: Tensor,
    schedule: DiffusionSchedule,
    donor_cache: ActivationCache,
    *,
    layer: str,
    apply_timestep: int,
    donor_timestep: int,
    channels: Tensor,
    mask: Tensor,
) -> Tensor:
    patch = ActivationPatch(
        layer=layer,
        apply_timestep=apply_timestep,
        donor_timestep=donor_timestep,
        donor_cache=donor_cache.values,
        channels=channels,
        spatial_mask=mask,
    )
    image, _ = denoise_trajectory(
        model, labels, initial_noise, schedule, patch=patch
    )
    return image


@t.inference_mode()
def causal_patch_sweep(
    model: HookableObjectDenoiser,
    clean_labels: Tensor,
    corrupt_labels: Tensor,
    initial_noise: Tensor,
    schedule: DiffusionSchedule,
) -> dict[str, Any]:
    """Run the full layer-by-timestep sweep and preregistered negative controls."""
    device = next(model.parameters()).device
    clean_labels = clean_labels.to(device)
    corrupt_labels = corrupt_labels.to(device)
    initial_noise = initial_noise.to(device)
    clean_target, target_mask = render_object_world(clean_labels)
    corrupt_target, _ = render_object_world(corrupt_labels)

    clean_cache = ActivationCache(values={})
    clean_generated, clean_states = denoise_trajectory(
        model, clean_labels, initial_noise, schedule, cache=clean_cache
    )
    corrupt_generated, corrupt_states = denoise_trajectory(
        model, corrupt_labels, initial_noise, schedule
    )
    concept_channels = t.arange(model.concept_channels, device=device)
    heatmap = t.empty(len(PATCH_LAYERS), N_DIFFUSION_STEPS)
    patched_images: dict[tuple[str, int], Tensor] = {}
    metrics: dict[tuple[str, int], RegionalCausalMetric] = {}
    for layer_index, layer in enumerate(PATCH_LAYERS):
        for step in range(1, N_DIFFUSION_STEPS + 1):
            patched = _patch_result(
                model,
                corrupt_labels,
                initial_noise,
                schedule,
                clean_cache,
                layer=layer,
                apply_timestep=step,
                donor_timestep=step,
                channels=concept_channels,
                mask=target_mask,
            )
            metric = regional_causal_metric(
                clean_target, corrupt_generated, patched, target_mask
            )
            heatmap[layer_index, step - 1] = metric.selectivity
            patched_images[(layer, step)] = patched.detach().cpu()
            metrics[(layer, step)] = metric

    best_layer = "concept"
    best_layer_index = PATCH_LAYERS.index(best_layer)
    best_step = int(heatmap[best_layer_index].argmax()) + 1
    best_image = patched_images[(best_layer, best_step)].to(device)
    best_metric = metrics[(best_layer, best_step)]

    random_channels = t.arange(
        model.concept_channels,
        2 * model.concept_channels,
        device=device,
    )
    random_channel_image = _patch_result(
        model,
        corrupt_labels,
        initial_noise,
        schedule,
        clean_cache,
        layer=best_layer,
        apply_timestep=best_step,
        donor_timestep=best_step,
        channels=random_channels,
        mask=target_mask,
    )
    random_location_mask = t.roll(target_mask, shifts=(IMAGE_SIZE // 2, IMAGE_SIZE // 2), dims=(-2, -1))
    random_location_image = _patch_result(
        model,
        corrupt_labels,
        initial_noise,
        schedule,
        clean_cache,
        layer=best_layer,
        apply_timestep=best_step,
        donor_timestep=best_step,
        channels=concept_channels,
        mask=random_location_mask,
    )
    wrong_step = 1 if best_step > N_DIFFUSION_STEPS // 2 else N_DIFFUSION_STEPS
    wrong_timestep_image = _patch_result(
        model,
        corrupt_labels,
        initial_noise,
        schedule,
        clean_cache,
        layer=best_layer,
        apply_timestep=wrong_step,
        donor_timestep=wrong_step,
        channels=concept_channels,
        mask=target_mask,
    )

    shuffled_labels = _single_label((1, 1, 1, 0), device)
    shuffled_cache = ActivationCache(values={})
    denoise_trajectory(model, shuffled_labels, initial_noise, schedule, cache=shuffled_cache)
    shuffled_image = _patch_result(
        model,
        corrupt_labels,
        initial_noise,
        schedule,
        shuffled_cache,
        layer=best_layer,
        apply_timestep=best_step,
        donor_timestep=best_step,
        channels=concept_channels,
        mask=target_mask,
    )

    pixel_patch = corrupt_generated.clone()
    expanded_mask = target_mask[:, None].expand_as(pixel_patch)
    pixel_patch[expanded_mask] = clean_target[expanded_mask]

    controls = {
        "matched_seed_unpatched": regional_causal_metric(
            clean_target, corrupt_generated, corrupt_generated, target_mask
        ),
        "same_size_random_channels": regional_causal_metric(
            clean_target, corrupt_generated, random_channel_image, target_mask
        ),
        "same_size_random_location": regional_causal_metric(
            clean_target, corrupt_generated, random_location_image, target_mask
        ),
        "wrong_timestep": regional_causal_metric(
            clean_target, corrupt_generated, wrong_timestep_image, target_mask
        ),
        "shuffled_labels": regional_causal_metric(
            clean_target, corrupt_generated, shuffled_image, target_mask
        ),
        "pixel_patch_upper_bound": regional_causal_metric(
            clean_target, corrupt_generated, pixel_patch, target_mask
        ),
    }
    return {
        "clean_target": clean_target.detach().cpu(),
        "corrupt_target": corrupt_target.detach().cpu(),
        "clean_generated": clean_generated.detach().cpu(),
        "corrupt_generated": corrupt_generated.detach().cpu(),
        "best_patched": best_image.detach().cpu(),
        "target_mask": target_mask.detach().cpu(),
        "heatmap": heatmap.detach().cpu(),
        "best_layer": best_layer,
        "best_timestep": best_step,
        "best_metric": best_metric,
        "controls": controls,
        "clean_states": clean_states,
        "corrupt_states": corrupt_states,
    }


def _untrained_control(
    trained_model: HookableObjectDenoiser,
    clean_labels: Tensor,
    corrupt_labels: Tensor,
    initial_noise: Tensor,
    schedule: DiffusionSchedule,
    best_layer: str,
    best_timestep: int,
) -> RegionalCausalMetric:
    device = next(trained_model.parameters()).device
    t.manual_seed(19)
    model = HookableObjectDenoiser(
        trained_model.width, trained_model.concept_channels
    ).to(device).eval()
    clean_labels = clean_labels.to(device)
    corrupt_labels = corrupt_labels.to(device)
    initial_noise = initial_noise.to(device)
    clean_target, target_mask = render_object_world(clean_labels)
    donor_cache = ActivationCache(values={})
    denoise_trajectory(model, clean_labels, initial_noise, schedule, cache=donor_cache)
    corrupt, _ = denoise_trajectory(model, corrupt_labels, initial_noise, schedule)
    patched = _patch_result(
        model,
        corrupt_labels,
        initial_noise,
        schedule,
        donor_cache,
        layer=best_layer,
        apply_timestep=best_timestep,
        donor_timestep=best_timestep,
        channels=t.arange(model.concept_channels, device=device),
        mask=target_mask,
    )
    return regional_causal_metric(clean_target, corrupt, patched, target_mask)


def _run_toy_experiment_impl(
    *,
    device: t.device | str = "cpu",
    training_steps: int = 350,
    seed: int = 0,
) -> dict[str, Any]:
    """Train, generate, patch, sweep, and score the complete model organism."""
    device = t.device(device)
    started = time.perf_counter()
    model, trace = train_object_denoiser(device=device, steps=training_steps, seed=seed)
    schedule = make_diffusion_schedule(device=device)
    clean_labels = _single_label((0, 0, 0, 0), device)
    corrupt_labels = _single_label((1, 2, 1, 1), device)
    generator = t.Generator(device=device).manual_seed(seed + 2026)
    initial_noise = t.randn(
        1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator, device=device
    )
    sweep = causal_patch_sweep(
        model, clean_labels, corrupt_labels, initial_noise, schedule
    )
    untrained = _untrained_control(
        model,
        clean_labels,
        corrupt_labels,
        initial_noise,
        schedule,
        sweep["best_layer"],
        sweep["best_timestep"],
    )
    sweep["controls"]["untrained_model"] = untrained
    best_control = max(
        sweep["controls"][name].selectivity
        for name in (
            "matched_seed_unpatched",
            "same_size_random_channels",
            "same_size_random_location",
            "shuffled_labels",
            "untrained_model",
        )
    )
    accepted = (
        trace.final_loss < trace.first_loss * 0.35
        and sweep["best_metric"].selectivity > best_control + 0.08
        and sweep["best_metric"].recovery > 0.12
    )
    sweep.update(
        {
            "model": model,
            "training_trace": trace,
            "clean_labels": clean_labels.detach().cpu(),
            "corrupt_labels": corrupt_labels.detach().cpu(),
            "runtime_seconds": time.perf_counter() - started,
            "accepted": bool(accepted),
        }
    )
    return sweep


def run_toy_experiment(
    *,
    device: t.device | str = "cpu",
    training_steps: int = 350,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the full experiment while avoiding tiny-convolution CPU oversubscription."""
    device = t.device(device)
    previous_threads = t.get_num_threads()
    if device.type == "cpu" and previous_threads > 2:
        t.set_num_threads(2)
    try:
        return _run_toy_experiment_impl(
            device=device, training_steps=training_steps, seed=seed
        )
    finally:
        if device.type == "cpu" and t.get_num_threads() != previous_threads:
            t.set_num_threads(previous_threads)


def serializable_metrics(result: dict[str, Any]) -> dict[str, Any]:
    controls = {
        name: asdict(metric) for name, metric in result["controls"].items()
    }
    trace: TrainingTrace = result["training_trace"]
    best: RegionalCausalMetric = result["best_metric"]
    return {
        "accepted": result["accepted"],
        "training_first_loss": trace.first_loss,
        "training_final_loss": trace.final_loss,
        "training_loss_ratio": trace.final_loss / max(trace.first_loss, 1e-12),
        "best_layer": result["best_layer"],
        "best_timestep": result["best_timestep"],
        "best_recovery": best.recovery,
        "best_outside_change": best.outside_change,
        "best_selectivity": best.selectivity,
        "heatmap_max": float(result["heatmap"].max()),
        "heatmap_min": float(result["heatmap"].min()),
        "controls": controls,
        "runtime_seconds": result["runtime_seconds"],
    }


def plot_signature_result(result: dict[str, Any], output_path: str | Path | None = None):
    """Render the visible image trajectory, causal heatmap, and control comparison."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 1.15))
    panels = (
        ("clean_target", "Clean target"),
        ("corrupt_target", "Corrupt target"),
        ("clean_generated", "Clean denoised"),
        ("corrupt_generated", "Matched-seed corrupt"),
        ("best_patched", "Activation patched"),
    )
    for column, (key, title) in enumerate(panels):
        axis = figure.add_subplot(grid[0, column])
        image = result[key][0].permute(1, 2, 0).clamp(-1, 1)
        axis.imshow((image + 1.0) / 2.0, interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    loss_axis = figure.add_subplot(grid[0, 5])
    losses = result["training_trace"].losses
    loss_axis.plot(losses, color="#2b6cb0", linewidth=1.5)
    loss_axis.set_title("End-to-end training", fontsize=10)
    loss_axis.set_xlabel("step")
    loss_axis.set_ylabel("weighted MSE")
    loss_axis.grid(alpha=0.25)

    heat_axis = figure.add_subplot(grid[1, :4])
    heatmap = result["heatmap"].numpy()
    image = heat_axis.imshow(heatmap, cmap="RdBu_r", aspect="auto", vmin=-1.0, vmax=1.0)
    heat_axis.set_yticks(range(len(PATCH_LAYERS)), PATCH_LAYERS)
    heat_axis.set_xticks(range(N_DIFFUSION_STEPS), range(1, N_DIFFUSION_STEPS + 1))
    heat_axis.set_xlabel("patched denoising timestep")
    heat_axis.set_ylabel("patched layer")
    heat_axis.set_title("Target-region recovery minus off-target change")
    figure.colorbar(image, ax=heat_axis, shrink=0.8, label="causal selectivity")

    control_axis = figure.add_subplot(grid[1, 4:])
    names = ["target"] + list(result["controls"])
    values = [result["best_metric"].selectivity] + [
        result["controls"][name].selectivity for name in result["controls"]
    ]
    colors = ["#c53030"] + ["#718096"] * (len(names) - 1)
    control_axis.barh(range(len(names)), values, color=colors)
    control_axis.set_yticks(range(len(names)), [name.replace("_", " ") for name in names], fontsize=8)
    control_axis.axvline(0.0, color="black", linewidth=0.8)
    control_axis.set_xlabel("causal selectivity")
    control_axis.set_title("Preregistered controls")
    control_axis.invert_yaxis()
    figure.suptitle(
        f"Denoising-time causal patching: {result['best_layer']} at t={result['best_timestep']}",
        fontsize=15,
        fontweight="bold",
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
    return figure


def make_center_mask(
    height: int,
    width: int,
    *,
    fraction: float = 0.5,
    device: t.device | str = "cpu",
) -> Tensor:
    """Return a preregistered centered spatial mask."""
    if height < 2 or width < 2:
        raise ValueError("height and width must both be at least two.")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between zero and one.")
    selected_height = max(1, round(height * fraction))
    selected_width = max(1, round(width * fraction))
    row_start = (height - selected_height) // 2
    column_start = (width - selected_width) // 2
    mask = t.zeros(height, width, dtype=t.bool, device=device)
    mask[
        row_start : row_start + selected_height,
        column_start : column_start + selected_width,
    ] = True
    return mask


def _broadcast_spatial_mask(mask: Tensor, tensor: Tensor) -> Tensor:
    if tensor.ndim != 4:
        raise ValueError("tensor must have shape [batch, channels, height, width].")
    mask = mask.to(device=tensor.device, dtype=t.bool)
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape[-2:] != tensor.shape[-2:]:
        raise ValueError("spatial_mask must match the tensor's spatial dimensions.")
    if mask.shape[0] not in (1, tensor.shape[0]):
        raise ValueError("spatial_mask batch must be one or match the tensor batch.")
    return mask.expand(tensor.shape[0], tensor.shape[1], -1, -1)


def apply_latent_patch(
    recipient: Tensor,
    donor: Tensor,
    spatial_mask: Tensor,
    *,
    mix: float = 1.0,
) -> Tensor:
    """Replace a selected latent region while leaving its complement bit-exact."""
    if recipient.shape != donor.shape:
        raise ValueError("recipient and donor latents must have identical shapes.")
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must lie in [0, 1].")
    selector = _broadcast_spatial_mask(spatial_mask, recipient)
    replacement = recipient.lerp(donor.to(recipient), mix)
    return t.where(selector, replacement, recipient)


def apply_random_latent_patch(
    recipient: Tensor,
    donor: Tensor,
    spatial_mask: Tensor,
    *,
    seed: int,
) -> Tensor:
    """Apply a same-size Gaussian control matched to donor-region mean and scale."""
    if recipient.shape != donor.shape:
        raise ValueError("recipient and donor latents must have identical shapes.")
    selector = _broadcast_spatial_mask(spatial_mask, recipient)
    donor_values = donor.to(recipient)[selector]
    if donor_values.numel() < 2:
        raise ValueError("spatial_mask must select at least two latent values.")
    generator = t.Generator(device=recipient.device).manual_seed(seed)
    noise = t.randn(
        recipient.shape,
        dtype=recipient.dtype,
        device=recipient.device,
        generator=generator,
    )
    noise = noise * donor_values.float().std().to(recipient.dtype)
    noise = noise + donor_values.float().mean().to(recipient.dtype)
    return t.where(selector, noise, recipient)


def calibrated_recovery(
    recipient_score: float,
    donor_score: float,
    patched_score: float,
) -> float:
    """Calibrate recipient to zero and donor to one on an oriented scalar score."""
    denominator = donor_score - recipient_score
    if abs(denominator) < 1e-8:
        raise ValueError("donor and recipient scores must define a nonzero contrast.")
    return (patched_score - recipient_score) / denominator


def latent_transfer_metric(
    donor_image: Tensor,
    recipient_image: Tensor,
    patched_image: Tensor,
    spatial_mask: Tensor,
) -> LatentTransferMetric:
    """Measure donor recovery inside a region and recipient preservation outside it."""
    if donor_image.shape != recipient_image.shape or donor_image.shape != patched_image.shape:
        raise ValueError("all images must have identical [batch, channels, height, width] shapes.")
    selector = _broadcast_spatial_mask(spatial_mask, donor_image)
    outside = ~selector
    donor_distance = (
        (recipient_image[selector] - donor_image[selector]).square().mean().sqrt().item()
    )
    patched_distance = (
        (patched_image[selector] - donor_image[selector]).square().mean().sqrt().item()
    )
    recovery = 1.0 - patched_distance / max(donor_distance, 1e-8)
    recipient_distance = (
        (patched_image[selector] - recipient_image[selector]).square().mean().sqrt().item()
    )
    outside_change = (
        (patched_image[outside] - recipient_image[outside]).square().mean().sqrt().item()
        if outside.any()
        else 0.0
    )
    return LatentTransferMetric(
        donor_distance=donor_distance,
        recipient_distance=recipient_distance,
        recovery=recovery,
        outside_change=outside_change,
        selectivity=recovery - outside_change,
    )


def load_pinned_sd15_ddim_pipeline(device: str | t.device):
    """Load the pinned SD1.5 weights and replace only the scheduler with DDIM."""
    from diffusers import DDIMScheduler

    from chapter13_image_generation_interpretability.exercises.part1_diffusion_image_controls import (
        solutions as controls_reference,
    )

    pipe = controls_reference.load_pinned_sd15_pipeline(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_pinned_clip_components(device: str | t.device):
    """Reuse 13.1's pinned checkpoint loader; scoring remains visible here."""
    from chapter13_image_generation_interpretability.exercises.part1_diffusion_image_controls import (
        solutions as controls_reference,
    )

    return controls_reference.load_pinned_clip_components(device)


def encode_sd15_prompt(
    pipe,
    prompt: str,
    negative_prompt: str,
    *,
    device: str | t.device,
) -> Tensor:
    """Encode one classifier-free-guidance prompt in UNet batch order."""
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    if negative_prompt_embeds is None:
        raise RuntimeError("classifier-free guidance requires negative prompt embeddings.")
    return t.cat([negative_prompt_embeds, prompt_embeds])


def prepare_same_seed_sd15_latents(
    pipe,
    *,
    seed: int,
    device: str | t.device,
    height: int = 512,
    width: int = 512,
) -> Tensor:
    """Create the exact initial latent shared by donor, recipient, and controls."""
    device = t.device(device)
    generator = t.Generator(device=device).manual_seed(seed)
    return pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.unet.config.in_channels,
        height=height,
        width=width,
        dtype=pipe.unet.dtype,
        device=device,
        generator=generator,
    )


def decode_sd15_latents(pipe, latents: Tensor) -> tuple[Any, Tensor]:
    """Decode one latent to both a PIL image and a [1, 3, H, W] float tensor."""
    scaled = latents / pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(scaled, return_dict=False)[0]
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    width, height = image.size
    image_tensor = (
        t.tensor(list(image.convert("RGB").getdata()), dtype=t.float32)
        .reshape(height, width, 3)
        .permute(2, 0, 1)[None]
        / 255.0
    )
    return image, image_tensor


@t.inference_mode()
def run_sd15_latent_trajectory(
    pipe,
    prompt: str,
    negative_prompt: str,
    initial_latents: Tensor,
    *,
    num_inference_steps: int = 20,
    guidance_scale: float = 9.0,
    patch_step_index: int | None = None,
    donor_states: tuple[Tensor, ...] | None = None,
    spatial_mask: Tensor | None = None,
    patch_kind: Literal["donor", "random"] = "donor",
    patch_mix: float = 1.0,
    random_seed: int = 0,
    cache_states: bool = True,
) -> SD15Trajectory:
    """Run the explicit DDIM loop, optionally replacing one latent state."""
    if num_inference_steps < 2:
        raise ValueError("num_inference_steps must be at least two.")
    device = initial_latents.device
    prompt_embeds = encode_sd15_prompt(
        pipe, prompt, negative_prompt, device=device
    )
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = tuple(int(step.item()) for step in pipe.scheduler.timesteps)
    if patch_step_index is not None:
        if not 0 <= patch_step_index < num_inference_steps:
            raise ValueError("patch_step_index must index a denoising model call.")
        if donor_states is None or len(donor_states) != num_inference_steps + 1:
            raise ValueError("donor_states must contain the initial state and every DDIM update.")
        if spatial_mask is None:
            raise ValueError("spatial_mask is required when patching.")
    latents = initial_latents.clone()
    states: list[Tensor] = [latents.detach().clone()] if cache_states else []
    for step_index, timestep in enumerate(pipe.scheduler.timesteps):
        if step_index == patch_step_index:
            donor = donor_states[step_index].to(latents)
            if patch_kind == "donor":
                latents = apply_latent_patch(
                    latents, donor, spatial_mask, mix=patch_mix
                )
            elif patch_kind == "random":
                latents = apply_random_latent_patch(
                    latents, donor, spatial_mask, seed=random_seed
                )
            else:
                raise ValueError("patch_kind must be 'donor' or 'random'.")
        model_input = t.cat([latents, latents])
        model_input = pipe.scheduler.scale_model_input(model_input, timestep)
        noise_prediction = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        noise_unconditional, noise_conditional = noise_prediction.chunk(2)
        guided_noise = noise_unconditional + guidance_scale * (
            noise_conditional - noise_unconditional
        )
        latents = pipe.scheduler.step(
            guided_noise,
            timestep,
            latents,
            eta=0.0,
            return_dict=False,
        )[0]
        if cache_states:
            states.append(latents.detach().clone())
    image, image_tensor = decode_sd15_latents(pipe, latents)
    return SD15Trajectory(
        image=image,
        image_tensor=image_tensor,
        latent_states=tuple(states),
        scheduler_timesteps=timesteps,
    )


@t.inference_mode()
def clip_donor_margins(
    clip_model,
    clip_processor,
    images: list[Any],
    *,
    donor_text: str,
    recipient_text: str,
    device: str | t.device,
) -> Tensor:
    """Return CLIP donor-minus-recipient logits for every generated image."""
    inputs = clip_processor(
        text=[donor_text, recipient_text],
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)
    logits = clip_model(**inputs).logits_per_image.detach().float().cpu()
    if logits.shape != (len(images), 2):
        raise RuntimeError("CLIP should return one donor/recipient logit pair per image.")
    return logits[:, 0] - logits[:, 1]


def _wrong_region_mask(mask: Tensor) -> Tensor:
    return t.roll(mask, shifts=(mask.shape[-2] // 2, mask.shape[-1] // 2), dims=(-2, -1))


@t.inference_mode()
def run_sd15_patch_case(
    pipe,
    clip_model,
    clip_processor,
    case: dict[str, Any],
    *,
    device: str | t.device,
    num_inference_steps: int = 20,
    guidance_scale: float = 9.0,
    patch_fraction: float = 0.5,
) -> dict[str, Any]:
    """Sweep a real same-seed donor latent over denoising time and controls."""
    device = t.device(device)
    initial = prepare_same_seed_sd15_latents(
        pipe, seed=int(case["seed"]), device=device
    )
    latent_mask = make_center_mask(
        initial.shape[-2], initial.shape[-1], fraction=patch_fraction, device=device
    )
    image_mask = make_center_mask(512, 512, fraction=patch_fraction)[None]
    donor = run_sd15_latent_trajectory(
        pipe,
        str(case["donor_prompt"]),
        REAL_SD15_NEGATIVE_PROMPT,
        initial,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    recipient = run_sd15_latent_trajectory(
        pipe,
        str(case["recipient_prompt"]),
        REAL_SD15_NEGATIVE_PROMPT,
        initial,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    patched: list[SD15Trajectory] = []
    for step_index in range(1, num_inference_steps):
        patched.append(
            run_sd15_latent_trajectory(
                pipe,
                str(case["recipient_prompt"]),
                REAL_SD15_NEGATIVE_PROMPT,
                initial,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                patch_step_index=step_index,
                donor_states=donor.latent_states,
                spatial_mask=latent_mask,
                cache_states=False,
            )
        )
    sweep_images = [donor.image, recipient.image] + [item.image for item in patched]
    sweep_margins = clip_donor_margins(
        clip_model,
        clip_processor,
        sweep_images,
        donor_text=str(case["donor_text"]),
        recipient_text=str(case["recipient_text"]),
        device=device,
    )
    donor_margin = float(sweep_margins[0])
    recipient_margin = float(sweep_margins[1])
    clip_recoveries = [
        calibrated_recovery(recipient_margin, donor_margin, float(score))
        for score in sweep_margins[2:]
    ]
    regional_metrics = [
        latent_transfer_metric(
            donor.image_tensor,
            recipient.image_tensor,
            item.image_tensor,
            image_mask,
        )
        for item in patched
    ]
    best_offset = max(range(len(clip_recoveries)), key=clip_recoveries.__getitem__)
    best_step = best_offset + 1
    best = patched[best_offset]
    wrong_step = 1 if best_step >= num_inference_steps // 2 else num_inference_steps - 1
    wrong_time = patched[wrong_step - 1]
    wrong_region = run_sd15_latent_trajectory(
        pipe,
        str(case["recipient_prompt"]),
        REAL_SD15_NEGATIVE_PROMPT,
        initial,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        patch_step_index=best_step,
        donor_states=donor.latent_states,
        spatial_mask=_wrong_region_mask(latent_mask),
        cache_states=False,
    )
    random_latent = run_sd15_latent_trajectory(
        pipe,
        str(case["recipient_prompt"]),
        REAL_SD15_NEGATIVE_PROMPT,
        initial,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        patch_step_index=best_step,
        donor_states=donor.latent_states,
        spatial_mask=latent_mask,
        patch_kind="random",
        random_seed=1000 + int(case["seed"]),
        cache_states=False,
    )
    control_images = [wrong_region.image, random_latent.image]
    control_margins = clip_donor_margins(
        clip_model,
        clip_processor,
        control_images,
        donor_text=str(case["donor_text"]),
        recipient_text=str(case["recipient_text"]),
        device=device,
    )
    wrong_timestep_recovery = clip_recoveries[wrong_step - 1]
    wrong_region_recovery = calibrated_recovery(
        recipient_margin, donor_margin, float(control_margins[0])
    )
    random_recovery = calibrated_recovery(
        recipient_margin, donor_margin, float(control_margins[1])
    )
    best_recovery = clip_recoveries[best_offset]
    controls = [0.0, wrong_timestep_recovery, wrong_region_recovery, random_recovery]
    target_beats_controls = best_recovery >= max(controls) + 0.05
    report = SD15PatchControlReport(
        best_step_index=best_step,
        best_scheduler_timestep=donor.scheduler_timesteps[best_step],
        best_clip_recovery=best_recovery,
        best_regional_selectivity=regional_metrics[best_offset].selectivity,
        wrong_timestep_recovery=wrong_timestep_recovery,
        wrong_region_recovery=wrong_region_recovery,
        random_latent_recovery=random_recovery,
        unpatched_recovery=0.0,
        target_beats_controls=target_beats_controls,
    )
    return {
        "case": dict(case),
        "donor": donor,
        "recipient": recipient,
        "patched_sweep": patched,
        "clip_recoveries": clip_recoveries,
        "regional_metrics": regional_metrics,
        "best": best,
        "wrong_timestep": wrong_time,
        "wrong_region": wrong_region,
        "random_latent": random_latent,
        "report": report,
        "latent_mask": latent_mask.detach().cpu(),
        "image_mask": image_mask.detach().cpu(),
    }


@t.inference_mode()
def run_real_sd15_experiment(
    *,
    max_vram_gb: float = 24.0,
    num_inference_steps: int = 20,
) -> dict[str, Any]:
    """Run both pinned SD1.5 causal-patching cases on CUDA with no fallback."""
    if not t.cuda.is_available():
        raise RuntimeError("The real SD1.5 causal experiment requires CUDA.")
    if not 0.0 < max_vram_gb <= 24.0:
        raise ValueError("max_vram_gb must lie in (0, 24].")
    device = t.device("cuda")
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pipe = load_pinned_sd15_ddim_pipeline(device)
    clip_model, clip_processor = load_pinned_clip_components(device)
    cases = [
        run_sd15_patch_case(
            pipe,
            clip_model,
            clip_processor,
            case,
            device=device,
            num_inference_steps=num_inference_steps,
        )
        for case in REAL_SD15_CASES
    ]
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 2**30
    reports = [case["report"] for case in cases]
    accepted = (
        peak_vram_gb <= max_vram_gb
        and all(report.target_beats_controls for report in reports)
        and all(report.best_clip_recovery >= 0.10 for report in reports)
        and all(report.best_regional_selectivity >= 0.05 for report in reports)
    )
    result = {
        "model_id": REAL_SD15_MODEL_ID,
        "revision": REAL_SD15_REVISION,
        "clip_model_id": REAL_CLIP_MODEL_ID,
        "clip_revision": REAL_CLIP_REVISION,
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "gpu_name": t.cuda.get_device_name(),
        "num_inference_steps": num_inference_steps,
        "cases": cases,
        "peak_vram_gb": peak_vram_gb,
        "runtime_seconds": time.perf_counter() - started,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "accepted": accepted,
    }
    return result


def serializable_real_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Drop image and latent payloads while preserving every causal metric."""
    return {
        "model_id": result["model_id"],
        "revision": result["revision"],
        "clip_model_id": result["clip_model_id"],
        "clip_revision": result["clip_revision"],
        "torch_version": result["torch_version"],
        "cuda_version": result["cuda_version"],
        "gpu_name": result["gpu_name"],
        "num_inference_steps": result["num_inference_steps"],
        "peak_vram_gb": result["peak_vram_gb"],
        "runtime_seconds": result["runtime_seconds"],
        "within_vram_budget": result["within_vram_budget"],
        "accepted": result["accepted"],
        "cases": [
            {
                "case_id": case["case"]["case_id"],
                "seed": case["case"]["seed"],
                "clip_recoveries": case["clip_recoveries"],
                "regional_recoveries": [m.recovery for m in case["regional_metrics"]],
                "regional_selectivities": [m.selectivity for m in case["regional_metrics"]],
                "target_control_margin": case["report"].best_clip_recovery
                - max(
                    case["report"].unpatched_recovery,
                    case["report"].wrong_timestep_recovery,
                    case["report"].wrong_region_recovery,
                    case["report"].random_latent_recovery,
                ),
                **asdict(case["report"]),
            }
            for case in result["cases"]
        ],
    }


def plot_real_signature_result(
    result: dict[str, Any], output_path: str | Path | None = None
):
    """Plot real images, time-localized recovery curves, and matched controls."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(result["cases"]),
        8,
        figsize=(20, 4.2 * len(result["cases"])),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1.8, 1.4]},
    )
    if len(result["cases"]) == 1:
        axes = axes[None]
    for row, case in enumerate(result["cases"]):
        report: SD15PatchControlReport = case["report"]
        image_panels = (
            (case["donor"].image, "Donor"),
            (case["recipient"].image, "Recipient"),
            (case["best"].image, "Correct time + region"),
            (case["wrong_timestep"].image, "Wrong time"),
            (case["wrong_region"].image, "Wrong region"),
            (case["random_latent"].image, "Random latent"),
        )
        for column, (image, title) in enumerate(image_panels):
            axes[row, column].imshow(image)
            axes[row, column].set_title(title, fontsize=9)
            axes[row, column].axis("off")
        curve_axis = axes[row, 6]
        indices = list(range(1, result["num_inference_steps"]))
        curve_axis.plot(indices, case["clip_recoveries"], marker="o", label="CLIP recovery")
        curve_axis.plot(
            indices,
            [metric.selectivity for metric in case["regional_metrics"]],
            marker="s",
            label="regional selectivity",
        )
        curve_axis.axvline(report.best_step_index, color="#c53030", linestyle="--")
        curve_axis.axhline(0.0, color="black", linewidth=0.8)
        curve_axis.set_xlabel("denoising call index")
        curve_axis.set_ylabel("calibrated effect")
        curve_axis.set_title("When does the donor matter?")
        curve_axis.legend(fontsize=7)
        control_axis = axes[row, 7]
        names = ["target", "unpatched", "wrong time", "wrong region", "random"]
        values = [
            report.best_clip_recovery,
            report.unpatched_recovery,
            report.wrong_timestep_recovery,
            report.wrong_region_recovery,
            report.random_latent_recovery,
        ]
        control_axis.barh(names, values, color=["#c53030"] + ["#718096"] * 4)
        control_axis.axvline(0.0, color="black", linewidth=0.8)
        control_axis.set_title("Matched controls")
        control_axis.set_xlabel("CLIP recovery")
        axes[row, 0].set_ylabel(str(case["case"]["case_id"]).replace("_", " "))
    figure.suptitle(
        "Pinned SD1.5 denoising-time causal patching: donor concept transfer is time and region specific",
        fontsize=15,
        fontweight="bold",
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def run_smoke_test(cpu: bool = True) -> dict[str, Any]:
    """Run exact CPU contracts without pretending to validate trained behavior."""
    _ = cpu
    schedule = make_diffusion_schedule()
    labels = t.tensor([[0, 0, 0, 0], [1, 2, 1, 1]])
    images, masks = render_object_world(labels)
    noised = q_sample(images, t.zeros(2, dtype=t.long), t.randn_like(images), schedule)
    return {
        "tests_passed": bool(t.equal(noised, images)),
        "dataset_size": int(len(all_world_labels())),
        "image_shape": list(images.shape),
        "mask_pixels": [int(mask.sum()) for mask in masks],
        "alpha_bar_t0": float(schedule.alpha_bars[0]),
    }


def _cuda_report(max_vram_gb: float, training_steps: int) -> dict[str, Any]:
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required; this verification path has no CPU fallback.")
    if max_vram_gb <= 0 or max_vram_gb > 24.0:
        raise ValueError("max_vram_gb must be in (0, 24].")
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()
    result = run_toy_experiment(device="cuda", training_steps=training_steps)
    peak_vram_gb = t.cuda.max_memory_allocated() / 2**30
    if peak_vram_gb > max_vram_gb:
        raise RuntimeError(
            f"Peak allocation {peak_vram_gb:.2f} GiB exceeded {max_vram_gb:.2f} GiB."
        )
    metrics = serializable_metrics(result)
    metrics.update(
        {
            "gpu_name": t.cuda.get_device_name(),
            "peak_vram_gb": peak_vram_gb,
            "max_vram_gb": max_vram_gb,
        }
    )
    if not metrics["accepted"]:
        raise AssertionError("The trained causal-patching signature did not beat controls.")
    return metrics


def _package_gpu_result(toy: dict[str, Any], real: dict[str, Any]) -> dict[str, Any]:
    """Expose numeric real-model acceptance metrics at stable report keys."""
    sd15 = serializable_real_metrics(real)
    cases = sd15["cases"]
    controls = toy["controls"]
    peak_vram_gb = max(float(toy["peak_vram_gb"]), float(sd15["peak_vram_gb"]))
    return {
        "cuda_available": True,
        "torch_version": sd15["torch_version"],
        "cuda_version": sd15["cuda_version"],
        "gpu_name": sd15["gpu_name"],
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": bool(
            toy["peak_vram_gb"] <= toy["max_vram_gb"]
            and sd15["within_vram_budget"]
        ),
        "dataset_size": len(all_world_labels()),
        "layer_count": len(PATCH_LAYERS),
        "diffusion_steps": N_DIFFUSION_STEPS,
        "heatmap_cell_count": len(PATCH_LAYERS) * N_DIFFUSION_STEPS,
        "random_channel_selectivity": controls["same_size_random_channels"][
            "selectivity"
        ],
        "wrong_timestep_selectivity": controls["wrong_timestep"]["selectivity"],
        "shuffled_label_selectivity": controls["shuffled_labels"]["selectivity"],
        "untrained_model_selectivity": controls["untrained_model"]["selectivity"],
        "pixel_patch_recovery": controls["pixel_patch_upper_bound"]["recovery"],
        "sd15_case_count": len(cases),
        "sd15_num_inference_steps": sd15["num_inference_steps"],
        "sd15_min_best_clip_recovery": min(case["best_clip_recovery"] for case in cases),
        "sd15_min_best_regional_selectivity": min(
            case["best_regional_selectivity"] for case in cases
        ),
        "sd15_min_target_control_margin": min(
            case["target_control_margin"] for case in cases
        ),
        "sd15_accepted": bool(sd15["accepted"]),
        "toy": toy,
        "sd15": sd15,
        "accepted": bool(toy["accepted"] and sd15["accepted"]),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict[str, Any]:
    toy = _cuda_report(max_vram_gb=max_vram_gb, training_steps=900)
    real = run_real_sd15_experiment(max_vram_gb=max_vram_gb)
    plot_real_signature_result(real, REAL_SIGNATURE_ASSET)
    return _package_gpu_result(toy, real)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict[str, Any]:
    toy = _cuda_report(max_vram_gb=max_vram_gb, training_steps=1_500)
    real = run_real_sd15_experiment(
        max_vram_gb=max_vram_gb,
        num_inference_steps=30,
    )
    plot_real_signature_result(real, REAL_SIGNATURE_ASSET)
    return _package_gpu_result(toy, real)


def save_metrics(result: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(serializable_metrics(result), indent=2) + "\n")
