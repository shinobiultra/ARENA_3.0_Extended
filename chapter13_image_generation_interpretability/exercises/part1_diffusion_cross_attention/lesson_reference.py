"""Exact pinned constants and checkpoint/PIL plumbing for the historical 13.1 lesson.

No model, download or tensor allocation happens at import. Scientific attention,
intervention, metrics and controls remain in the learner notebook. The appendix
smoke wrapper explicitly retains its original legacy identity.
"""
from __future__ import annotations
import os
import torch as t

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


def _pil_rgb_tensor(image, *, output_size: int | None = None) -> t.Tensor:
    rgb = image.convert("RGB")
    if output_size is not None:
        rgb = rgb.resize((output_size, output_size))
    width, height = rgb.size
    return t.tensor(list(rgb.getdata()), dtype=t.float32).reshape(height, width, 3)


def load_pinned_sd15_pipeline(device: str | t.device):
    """Load only the pinned SD1.5 checkpoint; the experiment remains learner code."""

    import logging

    os.environ.setdefault("BNB_CUDA_VERSION", "130")
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.ERROR)
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        REAL_SD15_MODEL_ID,
        revision=REAL_SD15_REVISION,
        dtype=t.float16,
        variant="fp16",
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_pinned_clip_components(device: str | t.device):
    """Load the pinned CLIP scorer without hiding its downstream metric computation."""

    from huggingface_hub import snapshot_download
    from transformers import CLIPModel, CLIPProcessor

    local_snapshot = snapshot_download(
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
    processor = CLIPProcessor.from_pretrained(local_snapshot, backend="pil")
    model = CLIPModel.from_pretrained(
        local_snapshot,
        use_safetensors=False,
    ).to(device)
    model.eval()
    return model, processor


def shuffled_region_control_smoke_test():
    """Explicit lazy delegation to the unchanged legacy appendix smoke helper."""
    from chapter13_image_generation_interpretability.exercises.part1_diffusion_image_controls.solutions import shuffled_region_control_smoke_test as legacy_smoke
    return legacy_smoke()
