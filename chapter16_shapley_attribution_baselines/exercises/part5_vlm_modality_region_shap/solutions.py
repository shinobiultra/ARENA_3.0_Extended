# %%
"""Reference solutions for [16.5] VLM Modality and Region SHAP."""

import itertools
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"
Coalition = frozenset[int]

REAL_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
REAL_CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
REAL_CLIP_NEUTRAL_TEXT = "a photo"
REAL_CLIP_TARGET_TEXT = "a red square"
REAL_CLIP_DISTRACTOR_TEXT = "a blue circle"
REAL_CLIP_MODALITY_SYNERGY_MIN = 2.0
REAL_CLIP_REGION_MARGIN_MIN = 1.0
REAL_CLIP_TARGET_DISTRACTOR_MARGIN_MIN = 2.0


@dataclass(frozen=True)
class ShapleyEfficiencyReport:
    shapley_sum: float
    total_value_delta: float
    efficiency_error: float
    satisfies_efficiency: bool


@dataclass(frozen=True)
class VLMModalitySHAPReport:
    modality_values: t.Tensor
    baseline_score: float
    image_only_score: float
    text_only_score: float
    full_score: float
    synergy: float
    detects_synergy: bool
    satisfies_efficiency: bool


@dataclass(frozen=True)
class VLMRegionSHAPReport:
    region_values: t.Tensor
    region_names: tuple[str, ...]
    target_region: str
    target_value: float
    max_background_value: float
    localizes_target: bool
    satisfies_efficiency: bool


# %%
def all_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Return every player subset for a finite cooperative game."""

    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    players = range(num_players)
    coalitions: list[Coalition] = []
    for size in range(num_players + 1):
        coalitions.extend(frozenset(group) for group in itertools.combinations(players, size))
    return tuple(coalitions)


def normalize_coalition_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[Coalition, float]:
    """Normalize coalition keys and require a complete value table."""

    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(all_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")
    return values


def coalition_values_from_function(
    num_players: int,
    value_fn: Callable[[Coalition], float],
) -> dict[Coalition, float]:
    """Evaluate `value_fn` on the complete coalition table."""

    return {coalition: float(value_fn(coalition)) for coalition in all_coalitions(num_players)}


def exact_shapley_values(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact Shapley values by summing weighted marginal effects."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    shapley = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        others = [candidate for candidate in range(num_players) if candidate != player]
        for size in range(num_players):
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                shapley[player] += weight * (values[coalition | {player}] - values[coalition])
    return shapley


def shapley_efficiency_report(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
    tolerance: float = 1e-9,
) -> ShapleyEfficiencyReport:
    """Check that Shapley values sum to `full_coalition - empty_coalition`."""

    values = normalize_coalition_values(coalition_values, num_players=num_players)
    shapley = exact_shapley_values(values, num_players=num_players)
    total_delta = values[frozenset(range(num_players))] - values[frozenset()]
    shapley_sum = float(shapley.sum().item())
    efficiency_error = abs(shapley_sum - total_delta)
    return ShapleyEfficiencyReport(
        shapley_sum=shapley_sum,
        total_value_delta=total_delta,
        efficiency_error=efficiency_error,
        satisfies_efficiency=efficiency_error <= tolerance,
    )


def vlm_modality_game(
    *,
    image_weight: float = 1.0,
    text_weight: float = 0.5,
    synergy_weight: float = 2.0,
) -> dict[Coalition, float]:
    """Return a two-player image/text game with multimodal synergy."""

    def value_fn(coalition: Coalition) -> float:
        image_present = 0 in coalition
        text_present = 1 in coalition
        score = 0.0
        if image_present:
            score += image_weight
        if text_present:
            score += text_weight
        if image_present and text_present:
            score += synergy_weight
        return score

    return coalition_values_from_function(2, value_fn)


def vlm_modality_shap_report(
    *,
    image_weight: float = 1.0,
    text_weight: float = 0.5,
    synergy_weight: float = 2.0,
    min_synergy: float = 1.0,
    tolerance: float = 1e-9,
) -> VLMModalitySHAPReport:
    """Compute modality Shapley values and detect image/text synergy."""

    values = vlm_modality_game(
        image_weight=image_weight,
        text_weight=text_weight,
        synergy_weight=synergy_weight,
    )
    modality_values = exact_shapley_values(values, num_players=2)
    baseline = values[frozenset()]
    image_only = values[frozenset({0})]
    text_only = values[frozenset({1})]
    full = values[frozenset({0, 1})]
    synergy = full - image_only - text_only + baseline
    efficiency_error = abs(float(modality_values.sum().item()) - (full - baseline))
    return VLMModalitySHAPReport(
        modality_values=modality_values,
        baseline_score=baseline,
        image_only_score=image_only,
        text_only_score=text_only,
        full_score=full,
        synergy=synergy,
        detects_synergy=synergy >= min_synergy,
        satisfies_efficiency=efficiency_error <= tolerance,
    )


def vlm_region_game(
    *,
    object_weight: float = 2.0,
    ocr_weight: float = 0.75,
    object_ocr_interaction: float = 0.5,
) -> dict[Coalition, float]:
    """Return a three-player object/background/OCR region game."""

    def value_fn(coalition: Coalition) -> float:
        object_present = 0 in coalition
        ocr_present = 2 in coalition
        score = 0.0
        if object_present:
            score += object_weight
        if ocr_present:
            score += ocr_weight
        if object_present and ocr_present:
            score += object_ocr_interaction
        return score

    return coalition_values_from_function(3, value_fn)


def vlm_region_shap_report(
    *,
    region_names: tuple[str, ...] = ("object", "background", "ocr_text"),
    target_region: str = "object",
    min_margin: float = 0.5,
    tolerance: float = 1e-9,
) -> VLMRegionSHAPReport:
    """Compute structured region Shapley values and check target localization."""

    if len(region_names) != 3:
        raise ValueError("region_names must name object, background, and OCR regions.")
    if target_region not in region_names:
        raise ValueError("target_region must be one of region_names.")

    values = vlm_region_game()
    region_values = exact_shapley_values(values, num_players=3)
    target_index = region_names.index(target_region)
    target_value = float(region_values[target_index].item())
    max_background = max(
        abs(float(value.item()))
        for index, value in enumerate(region_values)
        if index != target_index
    )
    efficiency = shapley_efficiency_report(values, num_players=3, tolerance=tolerance)
    return VLMRegionSHAPReport(
        region_values=region_values,
        region_names=region_names,
        target_region=target_region,
        target_value=target_value,
        max_background_value=max_background,
        localizes_target=target_value >= max_background + min_margin,
        satisfies_efficiency=efficiency.satisfies_efficiency,
    )


# %%
def _tensor_report(report) -> dict:
    result = report.__dict__.copy()
    for key, value in list(result.items()):
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
    return result


def modality_shap_smoke_test() -> dict:
    return _tensor_report(vlm_modality_shap_report())


def region_shap_smoke_test() -> dict:
    return _tensor_report(vlm_region_shap_report())


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "modality": modality_shap_smoke_test(),
        "region": region_shap_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("16.5 GPU preflight requires CUDA; no CPU fallback is accepted.")

    return run_real_clip_vlm_shap_preflight(max_vram_gb=max_vram_gb)


def render_shape_image(color: str, shape: str):
    """Render a deterministic 224x224 colored-shape image for CLIP preflights."""

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    bbox = [55, 55, 169, 169]
    if shape == "square":
        draw.rectangle(bbox, fill=color)
    elif shape == "circle":
        draw.ellipse(bbox, fill=color)
    else:
        raise ValueError("shape must be 'square' or 'circle'.")
    return image


def render_region_clip_image(
    *,
    object_present: bool,
    background_present: bool,
    ocr_present: bool,
):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (224, 224), (230, 230, 230) if background_present else "white")
    draw = ImageDraw.Draw(image)
    if object_present:
        draw.rectangle([55, 45, 169, 159], fill="red")
    if ocr_present:
        draw.text((88, 178), "RED", fill="black")
    return image


def _clip_scores(model, processor, *, images, texts, device: t.device) -> list[float]:
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
    ).to(device)
    with t.inference_mode():
        output = model(**inputs)
    logits = output.logits_per_image.detach().float()
    return [float(logits[index, index].item()) for index in range(len(images))]


def run_real_clip_vlm_shap_preflight(
    max_vram_gb: float = 24.0,
    *,
    include_visuals: bool = False,
) -> dict:
    from huggingface_hub import snapshot_download
    from transformers import CLIPModel, CLIPProcessor

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
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
    model = CLIPModel.from_pretrained(local_snapshot, use_safetensors=False).to(device)
    model.eval()

    target_image = render_shape_image("red", "square")
    distractor_image = render_shape_image("blue", "circle")
    modality_coalitions = [frozenset(), frozenset({0}), frozenset({1}), frozenset({0, 1})]
    modality_images = [
        target_image if 0 in coalition else distractor_image
        for coalition in modality_coalitions
    ]
    modality_texts = [
        REAL_CLIP_TARGET_TEXT if 1 in coalition else REAL_CLIP_NEUTRAL_TEXT
        for coalition in modality_coalitions
    ]
    modality_raw_scores = _clip_scores(
        model,
        processor,
        images=modality_images,
        texts=modality_texts,
        device=device,
    )
    modality_table = {
        coalition: score for coalition, score in zip(modality_coalitions, modality_raw_scores)
    }
    modality_values = exact_shapley_values(modality_table, num_players=2)
    modality_synergy = (
        modality_table[frozenset({0, 1})]
        - modality_table[frozenset({0})]
        - modality_table[frozenset({1})]
        + modality_table[frozenset()]
    )
    modality_efficiency = shapley_efficiency_report(modality_table, num_players=2)

    region_coalitions = [
        frozenset(group)
        for size in range(4)
        for group in itertools.combinations(range(3), size)
    ]
    region_images = [
        render_region_clip_image(
            object_present=0 in coalition,
            background_present=1 in coalition,
            ocr_present=2 in coalition,
        )
        for coalition in region_coalitions
    ]
    region_scores = _clip_scores(
        model,
        processor,
        images=region_images,
        texts=[REAL_CLIP_TARGET_TEXT] * len(region_images),
        device=device,
    )
    region_table = {
        coalition: score for coalition, score in zip(region_coalitions, region_scores)
    }
    region_values = exact_shapley_values(region_table, num_players=3)
    object_value = float(region_values[0].item())
    max_non_object = max(abs(float(value.item())) for value in region_values[1:])
    object_margin = object_value - max_non_object
    region_efficiency = shapley_efficiency_report(region_table, num_players=3)

    target_and_distractor_scores = _clip_scores(
        model,
        processor,
        images=[target_image, target_image],
        texts=[REAL_CLIP_TARGET_TEXT, REAL_CLIP_DISTRACTOR_TEXT],
        device=device,
    )
    target_distractor_margin = target_and_distractor_scores[0] - target_and_distractor_scores[1]

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    preflight_passed = (
        modality_synergy >= REAL_CLIP_MODALITY_SYNERGY_MIN
        and modality_efficiency.satisfies_efficiency
        and object_margin >= REAL_CLIP_REGION_MARGIN_MIN
        and region_efficiency.satisfies_efficiency
        and target_distractor_margin >= REAL_CLIP_TARGET_DISTRACTOR_MARGIN_MIN
        and peak_vram_gb <= max_vram_gb
    )

    visual_payload = None
    if include_visuals:
        visual_payload = {
            "target_image": target_image,
            "distractor_image": distractor_image,
            "target_text": REAL_CLIP_TARGET_TEXT,
            "distractor_text": REAL_CLIP_DISTRACTOR_TEXT,
            "neutral_text": REAL_CLIP_NEUTRAL_TEXT,
            "modality_coalitions": modality_coalitions,
            "modality_images": modality_images,
            "modality_texts": modality_texts,
            "modality_scores": modality_raw_scores,
            "region_coalitions": region_coalitions,
            "region_images": region_images,
            "region_scores": region_scores,
            "target_and_distractor_scores": target_and_distractor_scores,
        }

    del model, processor
    t.cuda.empty_cache()

    result = {
        "preflight_passed": preflight_passed,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "model_id": REAL_CLIP_MODEL_ID,
        "revision": REAL_CLIP_REVISION,
        "local_snapshot": str(local_snapshot),
        "claim_scope": "pinned_real_clip_rendered_vlm_shap_preflight",
        "modality_players": ["image", "text"],
        "modality_values": modality_values.tolist(),
        "modality_synergy": modality_synergy,
        "modality_satisfies_efficiency": modality_efficiency.satisfies_efficiency,
        "modality_scores": {str(sorted(coalition)): score for coalition, score in modality_table.items()},
        "region_names": ["object", "background", "ocr_text"],
        "region_values": region_values.tolist(),
        "object_value": object_value,
        "max_non_object_value": max_non_object,
        "object_margin": object_margin,
        "region_satisfies_efficiency": region_efficiency.satisfies_efficiency,
        "target_distractor_margin": target_distractor_margin,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": "Run modality and region SHAP on a pinned CLIP checkpoint with deterministic rendered image/text coalitions.",
    }
    if visual_payload is not None:
        result["visual_payload"] = visual_payload
    return result


def run_real_clip_vlm_shap_signature_result(max_vram_gb: float = 24.0) -> dict:
    """Run the pinned CLIP experiment and retain every learner-facing coalition."""

    return run_real_clip_vlm_shap_preflight(
        max_vram_gb=max_vram_gb,
        include_visuals=True,
    )


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
