# %%
"""Reference solutions for [13.1] Diffusion and Image-Generation Controls."""

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = "chapter13_image_generation_interpretability"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.image_generation_interpretability import (
    attention_region_report,
    color_region_mask,
    daam_region_report,
    denoising_circuit_report,
    image_quality_report,
    latent_direction_effect_report,
    prompt_region_causal_report,
    sd15_strict_acceptance_report,
    token_ablation_region_report,
    white_noise_image_control_report,
)
from arena_ext.vlm_interpretability import contrastive_alignment_report

MAIN = __name__ == "__main__"

REAL_SD_TURBO_MODEL_ID = "stabilityai/sd-turbo"
REAL_SD_TURBO_REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
REAL_SD_TURBO_PROMPTS = (
    "a simple red square icon centered on a clean white background, flat vector art",
    "a simple blue circle icon centered on a clean white background, flat vector art",
)
REAL_SD_TURBO_CLIP_TEXTS = (
    "a red square icon",
    "a blue circle icon",
)
REAL_SD_TURBO_TARGET_TERMS = (
    ("red", "square"),
    ("blue", "circle"),
)
REAL_SD_TURBO_CONTROL_TERMS = (
    ("white", "background"),
    ("white", "background"),
)
REAL_SD_TURBO_REGION_KINDS = ("red_square", "blue_circle")

REAL_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
REAL_CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

REAL_SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REAL_SD15_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
REAL_SD15_NEGATIVE_PROMPT = (
    "multiple objects, repeated pattern, tiled, grid, texture, text, letters, "
    "watermark, frame, border, shadow, clutter, photo, realistic, wood"
)
REAL_SD15_CASES = (
    {
        "case_id": "red_square",
        "seed": 4,
        "target_color": "red",
        "prompt": (
            "a single centered solid red square, plain white background, flat vector "
            "icon, minimal geometric shape"
        ),
        "target_ablated_prompt": (
            "a single centered solid square, plain white background, flat vector icon, "
            "minimal geometric shape"
        ),
        "control_ablated_prompt": (
            "a single centered solid red square, plain white foreground, flat vector "
            "icon, minimal geometric shape"
        ),
        "target_terms": ("red", "square"),
        "control_terms": ("geometric",),
        "clip_text": "a red square icon",
    },
    {
        "case_id": "blue_circle",
        "seed": 2,
        "target_color": "blue",
        "prompt": (
            "a single centered solid blue circle, plain white background, flat vector "
            "icon, minimal geometric shape"
        ),
        "target_ablated_prompt": (
            "a single centered solid circle, plain white background, flat vector icon, "
            "minimal geometric shape"
        ),
        "control_ablated_prompt": (
            "a single centered solid blue circle, plain white foreground, flat vector "
            "icon, minimal geometric shape"
        ),
        "target_terms": ("blue", "circle"),
        "control_terms": ("geometric",),
        "clip_text": "a blue circle icon",
    },
)


# %%
def attention_region_smoke_test() -> dict:
    attention_map = t.tensor([[0.1, 0.1], [0.2, 0.6]])
    region_mask = t.tensor([[False, False], [False, True]])
    return attention_region_report(
        attention_map,
        region_mask,
        min_region_mass=0.5,
    ).__dict__


def denoising_circuit_smoke_test() -> dict:
    return denoising_circuit_report(
        baseline_loss=0.2,
        ablated_loss=0.7,
        random_control_loss=0.35,
        min_loss_increase=0.3,
        min_control_gap=0.2,
    ).__dict__


def latent_direction_smoke_test() -> dict:
    baseline = t.tensor([0.1, 0.2])
    steered = t.tensor([0.7, 0.8])
    random_control = t.tensor([0.25, 0.15])
    return latent_direction_effect_report(
        baseline,
        steered,
        random_control,
        expected_direction="increase",
        min_effect=0.5,
        min_random_margin=0.2,
    ).__dict__


def prompt_region_smoke_test() -> dict:
    return prompt_region_causal_report(
        original_region_score=0.9,
        ablated_region_score=0.3,
        control_region_score=0.75,
        min_target_drop=0.4,
        min_control_margin=0.2,
    ).__dict__


@dataclass
class CrossAttentionCaptureStore:
    token_groups: dict[str, list[int]]
    maps: dict[str, list[t.Tensor]]
    resolutions: list[int]

    @classmethod
    def for_groups(cls, token_groups: dict[str, list[int]]) -> "CrossAttentionCaptureStore":
        return cls(token_groups=token_groups, maps={key: [] for key in token_groups}, resolutions=[])


class CrossAttentionCaptureProcessor:
    """Diffusers attention processor that stores token-group cross-attention maps."""

    def __init__(self, name: str, store: CrossAttentionCaptureStore):
        self.name = name
        self.store = store

    def __call__(
        self,
        attn,
        hidden_states: t.Tensor,
        encoder_hidden_states: t.Tensor | None = None,
        attention_mask: t.Tensor | None = None,
        temb: t.Tensor | None = None,
        *args,
        **kwargs,
    ) -> t.Tensor:
        _ = args, kwargs
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = channel = height = width = None

        effective_batch, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, effective_batch)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if is_cross_attention and self.name.endswith("attn2.processor"):
            query_length = attention_probs.shape[1]
            side = math.isqrt(query_length)
            if side * side == query_length and query_length >= 64:
                mean_attention = attention_probs.detach().float().mean(dim=0)
                for group_name, token_indices in self.store.token_groups.items():
                    group_attention = mean_attention[:, token_indices].mean(dim=-1)
                    self.store.maps[group_name].append(group_attention.reshape(side, side).cpu())
                self.store.resolutions.append(side)

        hidden_states = t.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


def _token_positions_for_terms(tokenizer, prompt: str, terms: tuple[str, ...]) -> list[int]:
    input_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    positions: list[int] = []
    for term in terms:
        term_tokens = tokenizer.tokenize(term)
        for index in range(len(tokens) - len(term_tokens) + 1):
            if tokens[index : index + len(term_tokens)] == term_tokens:
                positions.extend(range(index, index + len(term_tokens)))
    if not positions:
        raise RuntimeError(f"could not find token positions for terms {terms!r}")
    return sorted(set(positions))


def _aggregate_attention_maps(maps: list[t.Tensor], *, output_size: int = 64) -> t.Tensor:
    if not maps:
        raise RuntimeError("no cross-attention maps were captured")
    resized = [
        t.nn.functional.interpolate(
            attention_map[None, None],
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        for attention_map in maps
    ]
    aggregate = t.stack(resized).mean(dim=0).clamp_min(0)
    total = aggregate.sum()
    if total.item() <= 0:
        raise RuntimeError("captured cross-attention map has no positive mass")
    return aggregate / total


def _generated_shape_region_mask(image, region_kind: str, *, output_size: int = 64) -> t.Tensor:
    rgb = image.convert("RGB").resize((output_size, output_size))
    pixels = t.tensor(list(rgb.getdata()), dtype=t.float32).reshape(output_size, output_size, 3)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    if region_kind == "red_square":
        mask = (red > 120) & (red > green * 1.15) & (red > blue * 1.15)
    elif region_kind == "blue_circle":
        mask = (blue > 110) & (blue > red * 1.10) & (blue > green * 1.05)
    else:
        raise ValueError(f"unknown region kind {region_kind!r}")
    if not mask.any():
        raise RuntimeError(f"generated {region_kind} mask is empty")
    return mask


def _attention_region_mass(attention_map: t.Tensor, region_mask: t.Tensor) -> float:
    if attention_map.shape != region_mask.shape:
        raise ValueError("attention_map and region_mask must have the same shape")
    return attention_map[region_mask].sum().item()


def sd_turbo_clip_alignment_preflight(max_vram_gb: float = 24.0) -> dict:
    """Generate safe shape prompts, score CLIP alignment, and localize cross-attention."""

    if not t.cuda.is_available():
        raise RuntimeError("SD-Turbo image-generation preflight requires CUDA.")

    import logging

    os.environ.setdefault("BNB_CUDA_VERSION", "130")
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)

    from diffusers import AutoPipelineForText2Image
    from huggingface_hub import snapshot_download
    from transformers import CLIPModel, CLIPProcessor

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    pipe = AutoPipelineForText2Image.from_pretrained(
        REAL_SD_TURBO_MODEL_ID,
        revision=REAL_SD_TURBO_REVISION,
        torch_dtype=t.float16,
        variant="fp16",
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    images = []
    attention_reports = []
    attention_resolutions: set[int] = set()
    for seed, prompt in enumerate(REAL_SD_TURBO_PROMPTS):
        token_groups = {
            "target": _token_positions_for_terms(
                pipe.tokenizer,
                prompt,
                REAL_SD_TURBO_TARGET_TERMS[seed],
            ),
            "control": _token_positions_for_terms(
                pipe.tokenizer,
                prompt,
                REAL_SD_TURBO_CONTROL_TERMS[seed],
            ),
        }
        store = CrossAttentionCaptureStore.for_groups(token_groups)
        pipe.unet.set_attn_processor(
            {
                name: CrossAttentionCaptureProcessor(name, store)
                for name in pipe.unet.attn_processors
            }
        )
        generator = t.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
            height=512,
            width=512,
            generator=generator,
        ).images[0]
        images.append(image)
        target_attention = _aggregate_attention_maps(store.maps["target"])
        control_attention = _aggregate_attention_maps(store.maps["control"])
        region_mask = _generated_shape_region_mask(image, REAL_SD_TURBO_REGION_KINDS[seed])
        target_region_mass = _attention_region_mass(target_attention, region_mask)
        control_region_mass = _attention_region_mass(control_attention, region_mask)
        mask_fraction = region_mask.float().mean().item()
        attention_resolutions.update(store.resolutions)
        attention_reports.append(
            {
                "prompt": prompt,
                "target_terms": list(REAL_SD_TURBO_TARGET_TERMS[seed]),
                "control_terms": list(REAL_SD_TURBO_CONTROL_TERMS[seed]),
                "target_token_positions": token_groups["target"],
                "control_token_positions": token_groups["control"],
                "region_kind": REAL_SD_TURBO_REGION_KINDS[seed],
                "region_mask_fraction": mask_fraction,
                "captured_map_count": len(store.maps["target"]),
                "target_region_mass": target_region_mass,
                "control_region_mass": control_region_mass,
                "target_control_gap": target_region_mass - control_region_mass,
                "target_lift_over_mask_fraction": target_region_mass - mask_fraction,
            }
        )

    local_clip_snapshot = snapshot_download(
        REAL_CLIP_MODEL_ID,
        revision=REAL_CLIP_REVISION,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "pytorch_model.bin",
        ],
    )
    processor = CLIPProcessor.from_pretrained(local_clip_snapshot, use_fast=False)
    clip_model = CLIPModel.from_pretrained(
        local_clip_snapshot,
        use_safetensors=False,
    ).to(device)
    clip_model.eval()
    inputs = processor(
        text=list(REAL_SD_TURBO_CLIP_TEXTS),
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with t.inference_mode():
        output = clip_model(**inputs)
    logits = output.logits_per_image.detach().float().cpu()
    report = contrastive_alignment_report(
        logits,
        min_accuracy=1.0,
        min_positive_margin=2.0,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    image_sizes = [list(image.size) for image in images]
    target_attention_masses = [row["target_region_mass"] for row in attention_reports]
    control_attention_masses = [row["control_region_mass"] for row in attention_reports]
    target_control_gaps = [row["target_control_gap"] for row in attention_reports]
    min_target_control_gap = min(target_control_gaps)
    mean_target_control_gap = sum(target_control_gaps) / len(target_control_gaps)
    min_target_lift_over_mask_fraction = min(
        row["target_lift_over_mask_fraction"] for row in attention_reports
    )
    min_captured_map_count = min(row["captured_map_count"] for row in attention_reports)
    attention_localized = (
        min_target_control_gap >= 0.02
        and min_target_lift_over_mask_fraction >= 0.02
        and min_captured_map_count >= 16
    )

    del inputs, output, clip_model, processor, pipe, images
    t.cuda.empty_cache()

    return {
        "cuda_available": True,
        "model_id": REAL_SD_TURBO_MODEL_ID,
        "revision": REAL_SD_TURBO_REVISION,
        "clip_model_id": REAL_CLIP_MODEL_ID,
        "clip_revision": REAL_CLIP_REVISION,
        "claim_scope": (
            "pinned_sd_turbo_safe_shape_generation_clip_alignment_and_cross_attention_"
            "localization_preflight"
        ),
        "prompt_count": len(REAL_SD_TURBO_PROMPTS),
        "image_count": len(image_sizes),
        "image_sizes": image_sizes,
        "num_inference_steps": 2,
        "guidance_scale": 0.0,
        "seeds": list(range(len(REAL_SD_TURBO_PROMPTS))),
        "image_to_text_accuracy": report.image_to_text_accuracy,
        "text_to_image_accuracy": report.text_to_image_accuracy,
        "mean_positive_margin": report.mean_positive_margin,
        "aligned": report.aligned,
        "logits": logits.tolist(),
        "attention_reports": attention_reports,
        "attention_resolutions": sorted(attention_resolutions),
        "target_attention_region_masses": target_attention_masses,
        "control_attention_region_masses": control_attention_masses,
        "min_target_control_attention_gap": min_target_control_gap,
        "mean_target_control_attention_gap": mean_target_control_gap,
        "min_target_lift_over_mask_fraction": min_target_lift_over_mask_fraction,
        "min_captured_cross_attention_map_count": min_captured_map_count,
        "cross_attention_localized": attention_localized,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": report.aligned and attention_localized and peak_vram_gb <= max_vram_gb,
    }


def _pil_rgb_tensor(image, *, output_size: int | None = None) -> t.Tensor:
    rgb = image.convert("RGB")
    if output_size is not None:
        rgb = rgb.resize((output_size, output_size))
    width, height = rgb.size
    return t.tensor(list(rgb.getdata()), dtype=t.float32).reshape(height, width, 3)


def _target_color_region_fraction(image, target_color: str) -> float:
    mask = color_region_mask(_pil_rgb_tensor(image), target_color)  # type: ignore[arg-type]
    return mask.float().mean().item()


def _target_color_region_mask_64(image, target_color: str) -> t.Tensor:
    return color_region_mask(
        _pil_rgb_tensor(image, output_size=64),
        target_color,  # type: ignore[arg-type]
    )


def sd15_daam_token_ablation_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run strict SD1.5 DAAM-style attention, ablation, quality, and noise controls."""

    if not t.cuda.is_available():
        raise RuntimeError("SD1.5 strict image-generation preflight requires CUDA.")

    import logging

    os.environ.setdefault("BNB_CUDA_VERSION", "130")
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)

    from diffusers import StableDiffusionPipeline
    from huggingface_hub import snapshot_download
    from transformers import CLIPModel, CLIPProcessor

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    pipe = StableDiffusionPipeline.from_pretrained(
        REAL_SD15_MODEL_ID,
        revision=REAL_SD15_REVISION,
        torch_dtype=t.float16,
        variant="fp16",
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    original_images = []
    case_reports = []
    daam_reports = []
    token_ablation_reports = []
    image_quality_reports = []
    white_noise_reports = []
    attention_resolutions: set[int] = set()
    for case in REAL_SD15_CASES:
        token_groups = {
            "target": _token_positions_for_terms(
                pipe.tokenizer,
                str(case["prompt"]),
                case["target_terms"],  # type: ignore[arg-type]
            ),
            "control": _token_positions_for_terms(
                pipe.tokenizer,
                str(case["prompt"]),
                case["control_terms"],  # type: ignore[arg-type]
            ),
        }
        store = CrossAttentionCaptureStore.for_groups(token_groups)
        pipe.unet.set_attn_processor(
            {
                name: CrossAttentionCaptureProcessor(name, store)
                for name in pipe.unet.attn_processors
            }
        )
        generator = t.Generator(device=device).manual_seed(int(case["seed"]))
        original_image = pipe(
            str(case["prompt"]),
            negative_prompt=REAL_SD15_NEGATIVE_PROMPT,
            num_inference_steps=20,
            guidance_scale=9.0,
            height=512,
            width=512,
            generator=generator,
        ).images[0]
        original_images.append(original_image)

        target_attention = _aggregate_attention_maps(store.maps["target"])
        control_attention = _aggregate_attention_maps(store.maps["control"])
        region_mask = _target_color_region_mask_64(
            original_image,
            str(case["target_color"]),
        )
        target_region_mass = _attention_region_mass(target_attention, region_mask)
        control_region_mass = _attention_region_mass(control_attention, region_mask)
        mask_fraction = region_mask.float().mean().item()
        attention_resolutions.update(store.resolutions)
        daam_report = daam_region_report(
            target_region_mass=target_region_mass,
            control_region_mass=control_region_mass,
            mask_fraction=mask_fraction,
            captured_map_count=len(store.maps["target"]),
            min_target_control_gap=0.005,
            min_lift_over_mask_fraction=0.01,
            min_captured_map_count=32,
        )

        target_ablated_image = pipe(
            str(case["target_ablated_prompt"]),
            negative_prompt=REAL_SD15_NEGATIVE_PROMPT,
            num_inference_steps=20,
            guidance_scale=9.0,
            height=512,
            width=512,
            generator=t.Generator(device=device).manual_seed(int(case["seed"])),
        ).images[0]
        control_ablated_image = pipe(
            str(case["control_ablated_prompt"]),
            negative_prompt=REAL_SD15_NEGATIVE_PROMPT,
            num_inference_steps=20,
            guidance_scale=9.0,
            height=512,
            width=512,
            generator=t.Generator(device=device).manual_seed(int(case["seed"])),
        ).images[0]
        original_region_score = _target_color_region_fraction(
            original_image,
            str(case["target_color"]),
        )
        target_ablated_region_score = _target_color_region_fraction(
            target_ablated_image,
            str(case["target_color"]),
        )
        control_region_score = _target_color_region_fraction(
            control_ablated_image,
            str(case["target_color"]),
        )
        token_report = token_ablation_region_report(
            original_region_score=original_region_score,
            target_ablated_region_score=target_ablated_region_score,
            random_control_region_score=control_region_score,
            min_target_drop=0.05,
            min_random_margin=0.05,
        )
        quality_report = image_quality_report(
            _pil_rgb_tensor(original_image),
            target_color=str(case["target_color"]),  # type: ignore[arg-type]
            min_target_region_fraction=0.02,
            max_high_frequency_energy=0.12,
        )
        white_noise = t.randint(
            0,
            256,
            _pil_rgb_tensor(original_image).shape,
            generator=t.Generator(device="cpu").manual_seed(1000 + int(case["seed"])),
        )
        noise_report = white_noise_image_control_report(
            quality_report,
            white_noise,
            target_color=str(case["target_color"]),  # type: ignore[arg-type]
            max_high_frequency_energy=0.12,
            min_noise_gap=0.12,
        )

        daam_reports.append(daam_report)
        token_ablation_reports.append(token_report)
        image_quality_reports.append(quality_report)
        white_noise_reports.append(noise_report)
        case_reports.append(
            {
                "case_id": case["case_id"],
                "seed": case["seed"],
                "prompt": case["prompt"],
                "target_ablated_prompt": case["target_ablated_prompt"],
                "control_ablated_prompt": case["control_ablated_prompt"],
                "target_terms": list(case["target_terms"]),
                "control_terms": list(case["control_terms"]),
                "target_token_positions": token_groups["target"],
                "control_token_positions": token_groups["control"],
                "target_color": case["target_color"],
                "mask_fraction": mask_fraction,
                "attention_resolutions": sorted(set(store.resolutions)),
                "captured_cross_attention_map_count": len(store.maps["target"]),
                "target_region_mass": daam_report.target_region_mass,
                "control_region_mass": daam_report.control_region_mass,
                "target_control_gap": daam_report.target_control_gap,
                "target_lift_over_mask_fraction": (
                    daam_report.target_lift_over_mask_fraction
                ),
                "daam_localized": daam_report.daam_localized,
                "original_region_score": token_report.original_region_score,
                "target_ablated_region_score": (
                    token_report.target_ablated_region_score
                ),
                "control_region_score": token_report.random_control_region_score,
                "target_drop": token_report.target_drop,
                "random_control_drop": token_report.random_control_drop,
                "target_ablation_passed": token_report.target_ablation_passed,
                "random_token_ablation_weaker": (
                    token_report.random_token_ablation_weaker
                ),
                "target_region_fraction": quality_report.target_region_fraction,
                "rgb_std": quality_report.rgb_std,
                "high_frequency_energy": quality_report.high_frequency_energy,
                "saturation_fraction": quality_report.saturation_fraction,
                "image_quality_preserved": quality_report.image_quality_preserved,
                "white_noise_high_frequency_energy": (
                    noise_report.white_noise_high_frequency_energy
                ),
                "white_noise_rejected": noise_report.white_noise_rejected,
            }
        )

    strict_report = sd15_strict_acceptance_report(
        daam_reports=daam_reports,
        token_ablation_reports=token_ablation_reports,
        image_quality_reports=image_quality_reports,
        white_noise_reports=white_noise_reports,
    )

    local_clip_snapshot = snapshot_download(
        REAL_CLIP_MODEL_ID,
        revision=REAL_CLIP_REVISION,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "pytorch_model.bin",
        ],
    )
    processor = CLIPProcessor.from_pretrained(local_clip_snapshot, use_fast=False)
    clip_model = CLIPModel.from_pretrained(
        local_clip_snapshot,
        use_safetensors=False,
    ).to(device)
    clip_model.eval()
    inputs = processor(
        text=[str(case["clip_text"]) for case in REAL_SD15_CASES],
        images=original_images,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with t.inference_mode():
        output = clip_model(**inputs)
    logits = output.logits_per_image.detach().float().cpu()
    clip_report = contrastive_alignment_report(
        logits,
        min_accuracy=1.0,
        min_positive_margin=2.0,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3

    del inputs, output, clip_model, processor, pipe, original_images
    t.cuda.empty_cache()

    return {
        "cuda_available": True,
        "model_id": REAL_SD15_MODEL_ID,
        "revision": REAL_SD15_REVISION,
        "clip_model_id": REAL_CLIP_MODEL_ID,
        "clip_revision": REAL_CLIP_REVISION,
        "claim_scope": (
            "pinned_sd15_safe_shape_generation_daam_style_attention_token_ablation_"
            "image_quality_and_white_noise_controls"
        ),
        "prompt_count": len(REAL_SD15_CASES),
        "image_count": len(case_reports),
        "image_size": [512, 512],
        "num_inference_steps": 20,
        "guidance_scale": 9.0,
        "negative_prompt": REAL_SD15_NEGATIVE_PROMPT,
        "seeds": [case["seed"] for case in REAL_SD15_CASES],
        "case_reports": case_reports,
        "attention_resolutions": sorted(attention_resolutions),
        "min_target_control_attention_gap": min(
            report.target_control_gap for report in daam_reports
        ),
        "mean_target_control_attention_gap": sum(
            report.target_control_gap for report in daam_reports
        )
        / len(daam_reports),
        "min_target_lift_over_mask_fraction": min(
            report.target_lift_over_mask_fraction for report in daam_reports
        ),
        "min_captured_cross_attention_map_count": min(
            report.captured_map_count for report in daam_reports
        ),
        "min_target_ablation_drop": min(
            report.target_drop for report in token_ablation_reports
        ),
        "max_random_control_drop": max(
            report.random_control_drop for report in token_ablation_reports
        ),
        "min_target_region_fraction": min(
            report.target_region_fraction for report in image_quality_reports
        ),
        "max_high_frequency_energy": max(
            report.high_frequency_energy for report in image_quality_reports
        ),
        "min_white_noise_high_frequency_gap": min(
            noise.white_noise_high_frequency_energy - quality.high_frequency_energy
            for noise, quality in zip(white_noise_reports, image_quality_reports)
        ),
        "daam_passed": strict_report.daam_passed,
        "token_ablation_passed": strict_report.token_ablation_passed,
        "random_token_ablation_weaker": strict_report.random_token_ablation_weaker,
        "image_quality_preserved": strict_report.image_quality_preserved,
        "white_noise_rejected": strict_report.white_noise_rejected,
        "strict_experiment_passed": strict_report.sd15_strict_experiment_passed,
        "clip_image_to_text_accuracy": clip_report.image_to_text_accuracy,
        "clip_text_to_image_accuracy": clip_report.text_to_image_accuracy,
        "clip_mean_positive_margin": clip_report.mean_positive_margin,
        "clip_logits": logits.tolist(),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": (
            strict_report.sd15_strict_experiment_passed
            and clip_report.aligned
            and peak_vram_gb <= max_vram_gb
        ),
    }


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "attention_region": attention_region_smoke_test(),
        "denoising_circuit": denoising_circuit_smoke_test(),
        "latent_direction": latent_direction_smoke_test(),
        "prompt_region": prompt_region_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("Section 13.1 GPU verification requires CUDA.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    attention = attention_region_report(
        t.tensor([[0.1, 0.1], [0.2, 0.6]], device=device),
        t.tensor([[False, False], [False, True]], device=device),
        min_region_mass=0.5,
    )
    latent = latent_direction_effect_report(
        t.tensor([0.1, 0.2], device=device),
        t.tensor([0.7, 0.8], device=device),
        t.tensor([0.25, 0.15], device=device),
        expected_direction="increase",
        min_effect=0.5,
        min_random_margin=0.2,
    )
    t.cuda.synchronize()
    synthetic_peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    sd_turbo = sd_turbo_clip_alignment_preflight(max_vram_gb=max_vram_gb)
    sd15 = sd15_daam_token_ablation_experiment(max_vram_gb=max_vram_gb)
    peak_vram_gb = max(
        synthetic_peak_vram_gb,
        sd_turbo["peak_vram_gb"],
        sd15["peak_vram_gb"],
    )
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "region_mass": attention.region_mass,
        "region_selective": attention.region_selective,
        "latent_observed_delta": latent.observed_delta,
        "latent_random_delta": latent.random_delta,
        "has_directional_effect": latent.has_directional_effect,
        "latent_direction_effect": latent.has_directional_effect,
        "sd_turbo_preflight_passed": sd_turbo["preflight_passed"],
        "sd_turbo_image_to_text_accuracy": sd_turbo["image_to_text_accuracy"],
        "sd_turbo_text_to_image_accuracy": sd_turbo["text_to_image_accuracy"],
        "sd_turbo_mean_positive_margin": sd_turbo["mean_positive_margin"],
        "sd_turbo_cross_attention_localized": sd_turbo["cross_attention_localized"],
        "sd_turbo_attention_resolutions": sd_turbo["attention_resolutions"],
        "sd_turbo_min_target_control_attention_gap": sd_turbo[
            "min_target_control_attention_gap"
        ],
        "sd_turbo_mean_target_control_attention_gap": sd_turbo[
            "mean_target_control_attention_gap"
        ],
        "sd_turbo_min_target_lift_over_mask_fraction": sd_turbo[
            "min_target_lift_over_mask_fraction"
        ],
        "sd_turbo_min_captured_cross_attention_map_count": sd_turbo[
            "min_captured_cross_attention_map_count"
        ],
        "sd_turbo_target_attention_region_masses": sd_turbo[
            "target_attention_region_masses"
        ],
        "sd_turbo_control_attention_region_masses": sd_turbo[
            "control_attention_region_masses"
        ],
        "sd_turbo_peak_vram_gb": sd_turbo["peak_vram_gb"],
        "sd_turbo_preflight": sd_turbo,
        "sd15_strict_experiment_passed": sd15["preflight_passed"],
        "sd15_model_id": sd15["model_id"],
        "sd15_revision": sd15["revision"],
        "sd15_fixed_seed_generation_passed": sd15["image_count"] == len(REAL_SD15_CASES),
        "sd15_daam_baseline_included": sd15["daam_passed"],
        "sd15_cross_attention_maps_captured": (
            sd15["min_captured_cross_attention_map_count"] >= 32
        ),
        "sd15_token_ablation_passed": sd15["token_ablation_passed"],
        "sd15_random_token_ablation_weaker": sd15["random_token_ablation_weaker"],
        "sd15_image_quality_preserved": sd15["image_quality_preserved"],
        "sd15_white_noise_rejected": sd15["white_noise_rejected"],
        "sd15_attention_resolutions": sd15["attention_resolutions"],
        "sd15_min_target_control_attention_gap": sd15[
            "min_target_control_attention_gap"
        ],
        "sd15_mean_target_control_attention_gap": sd15[
            "mean_target_control_attention_gap"
        ],
        "sd15_min_target_lift_over_mask_fraction": sd15[
            "min_target_lift_over_mask_fraction"
        ],
        "sd15_min_captured_cross_attention_map_count": sd15[
            "min_captured_cross_attention_map_count"
        ],
        "sd15_min_target_ablation_drop": sd15["min_target_ablation_drop"],
        "sd15_max_random_control_drop": sd15["max_random_control_drop"],
        "sd15_min_target_region_fraction": sd15["min_target_region_fraction"],
        "sd15_max_high_frequency_energy": sd15["max_high_frequency_energy"],
        "sd15_min_white_noise_high_frequency_gap": sd15[
            "min_white_noise_high_frequency_gap"
        ],
        "sd15_clip_image_to_text_accuracy": sd15["clip_image_to_text_accuracy"],
        "sd15_clip_text_to_image_accuracy": sd15["clip_text_to_image_accuracy"],
        "sd15_clip_mean_positive_margin": sd15["clip_mean_positive_margin"],
        "sd15_peak_vram_gb": sd15["peak_vram_gb"],
        "sd15_case_reports": sd15["case_reports"],
        "sd15_preflight": sd15,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": (
            peak_vram_gb <= max_vram_gb
            and sd_turbo["within_vram_budget"]
            and sd15["within_vram_budget"]
        ),
        "full_path": (
            "Validated diffusion attention region controls, latent-direction "
            "effects over random controls, pinned SD-Turbo safe-shape generation "
            "scored by real CLIP, SD-Turbo cross-attention localization, and "
            "pinned SD1.5 DAAM-style attention, target-token ablation, image-quality, "
            "and white-noise controls on deterministic safe shape prompts."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
