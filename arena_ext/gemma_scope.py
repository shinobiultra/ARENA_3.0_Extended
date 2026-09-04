"""Validation helpers for Gemma Scope-style feature artifact notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch as t

from arena_ext.features import feature_detection_report, roc_auc_binary


@dataclass(frozen=True)
class FeatureArtifactMetadata:
    model_name: str
    artifact_name: str
    artifact_type: Literal["sae", "transcoder"]
    layer: int
    hook_name: str
    d_model: int
    n_features: int


@dataclass(frozen=True)
class TaggedFeatureSpec:
    feature_id: int
    layer: int
    tags: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class GemmaScopeArtifactSpec:
    repo_id: str
    revision: str
    artifact_path: str
    model_name: str
    hook_point: str
    layer: int
    d_model: int
    width: int
    base_model_revision: str = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    architecture: str = "jump_relu"
    artifact_type: str = "sae"


GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL = GemmaScopeArtifactSpec(
    repo_id="google/gemma-scope-2-1b-it",
    revision="b0fa29457c3601df0a70c48a15534c738d7c10e0",
    artifact_path="resid_post/layer_13_width_16k_l0_small",
    model_name="google/gemma-3-1b-it",
    hook_point="model.layers.13.output",
    layer=13,
    d_model=1152,
    width=16384,
)


REAL_GEMMA_TECHNICAL_PROMPTS = (
    "Explain how a Python function validates a JSON schema before calling a tool.",
    "Describe a tensor shape check for a batched matrix multiplication in PyTorch.",
    "Write a concise note about masking invalid API actions before decoding.",
    "Explain how an embedding retrieval system ranks documents with cosine similarity.",
    "Describe a unit test for parsing function-call arguments from model output.",
    "Explain why attention masks should ignore padding tokens during pooling.",
    "Write a short note about checking CUDA tensor dtype and device placement.",
    "Describe how to compute a sparse autoencoder reconstruction error.",
)
REAL_GEMMA_NARRATIVE_PROMPTS = (
    "Describe a quiet garden with rain on green leaves in the early morning.",
    "Tell a calm story about a child watching clouds move over a hill.",
    "Describe the color of sunset light across a lake and a wooden dock.",
    "Write a peaceful paragraph about walking through fresh snow at night.",
    "Describe a small kitchen where bread cools beside a window.",
    "Tell a gentle scene about friends listening to music in a park.",
    "Describe the sound of waves reaching a beach under a clear sky.",
    "Write a short scene about lanterns glowing during an evening festival.",
)


def _hf_cache_repo_dir(repo_id: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"


def _safe_error_text(error: BaseException, *, limit: int = 400) -> str:
    text = str(error).replace("\n", " ")
    return text[:limit]


def gemma_base_model_access_report(
    *,
    spec: GemmaScopeArtifactSpec = GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL,
    allow_network: bool = True,
) -> dict[str, Any]:
    """Report whether the exact Gemma base model for a Scope artifact is usable.

    This is an access gate, not a fallback. It only checks the model named by
    the Gemma Scope config and marks the semantic activation path ready when
    authenticated access exposes config/tokenizer/weight files for that model.
    """

    from huggingface_hub import HfApi, list_repo_files

    cache_dir = _hf_cache_repo_dir(spec.model_name)
    local_non_ref_files = []
    if cache_dir.exists():
        local_non_ref_files = [
            path
            for path in cache_dir.rglob("*")
            if path.is_file() and "refs" not in path.parts
        ]
    local_file_names = {path.name for path in local_non_ref_files}
    local_config_available = "config.json" in local_file_names
    local_tokenizer_available = bool(
        {"tokenizer.json", "tokenizer.model", "tokenizer_config.json"} & local_file_names
    )
    local_weight_available = any(
        name.endswith((".safetensors", ".bin")) for name in local_file_names
    )

    authenticated = False
    auth_error_type = None
    auth_error = None
    try:
        HfApi().whoami()
        authenticated = True
    except Exception as error:  # pragma: no cover - environment dependent
        auth_error_type = type(error).__name__
        auth_error = _safe_error_text(error)

    repo_files: list[str] = []
    repo_listed = False
    access_error_type = None
    access_error = None
    if allow_network:
        try:
            repo_files = list(
                list_repo_files(
                    spec.model_name,
                    repo_type="model",
                    revision=spec.base_model_revision,
                )
            )
            repo_listed = True
        except Exception as error:  # pragma: no cover - environment dependent
            access_error_type = type(error).__name__
            access_error = _safe_error_text(error)

    remote_file_names = {Path(path).name for path in repo_files}
    remote_config_available = "config.json" in remote_file_names
    remote_tokenizer_available = bool(
        {"tokenizer.json", "tokenizer.model", "tokenizer_config.json"} & remote_file_names
    )
    remote_weight_available = any(
        name.endswith((".safetensors", ".bin")) for name in remote_file_names
    )
    remote_ready = (
        authenticated
        and repo_listed
        and remote_config_available
        and remote_tokenizer_available
        and remote_weight_available
    )
    missing_local_patterns = []
    if not local_config_available:
        missing_local_patterns.append("config.json")
    if not local_tokenizer_available:
        missing_local_patterns.append("tokenizer.json|tokenizer.model|tokenizer_config.json")
    if not local_weight_available:
        missing_local_patterns.append("*.safetensors|*.bin")

    missing_remote_patterns = []
    if not remote_config_available:
        missing_remote_patterns.append("config.json")
    if not remote_tokenizer_available:
        missing_remote_patterns.append("tokenizer.json|tokenizer.model|tokenizer_config.json")
    if not remote_weight_available:
        missing_remote_patterns.append("*.safetensors|*.bin")

    local_ready = local_config_available and local_tokenizer_available and local_weight_available
    ready_for_real_activations = remote_ready or local_ready
    return {
        "model_id": spec.model_name,
        "model_revision": spec.base_model_revision,
        "required_hook_point": spec.hook_point,
        "required_layer": spec.layer,
        "required_d_model": spec.d_model,
        "allow_network": allow_network,
        "authenticated": authenticated,
        "auth_error_type": auth_error_type,
        "auth_error": auth_error,
        "cache_dir": str(cache_dir),
        "local_non_ref_file_count": len(local_non_ref_files),
        "local_config_available": local_config_available,
        "local_tokenizer_available": local_tokenizer_available,
        "local_weight_available": local_weight_available,
        "missing_local_patterns": missing_local_patterns,
        "repo_listed": repo_listed,
        "repo_file_count": len(repo_files),
        "remote_config_available": remote_config_available,
        "remote_tokenizer_available": remote_tokenizer_available,
        "remote_weight_available": remote_weight_available,
        "missing_remote_patterns": missing_remote_patterns,
        "access_error_type": access_error_type,
        "access_error": access_error,
        "remote_ready_for_real_activations": remote_ready,
        "local_ready_for_real_activations": local_ready,
        "ready_for_real_activations": ready_for_real_activations,
        "semantic_feature_claimed": False,
    }


@dataclass(frozen=True)
class FeatureValidationSuiteReport:
    feature_auc: float
    baseline_auc: float
    auc_margin: float
    threshold_accuracy: float
    positive_mean: float
    negative_mean: float
    passes_baseline: bool


@dataclass(frozen=True)
class BaseInstructionFeatureDelta:
    base_mean: float
    instruction_mean: float
    delta: float
    abs_delta: float


@dataclass(frozen=True)
class AblationControlReport:
    baseline_mean: float
    ablated_mean: float
    random_ablated_mean: float
    ablation_delta: float
    random_delta: float
    passes_control: bool


@dataclass(frozen=True)
class SteeringSafetyReport:
    baseline_mean: float
    steered_mean: float
    random_mean: float
    steered_delta: float
    random_delta: float
    perplexity_ratio: float
    passes_control: bool
    passes_perplexity_guard: bool


def gemma_scope_jump_relu_forward(
    activations: t.Tensor,
    *,
    w_enc: t.Tensor,
    w_dec: t.Tensor,
    b_enc: t.Tensor,
    b_dec: t.Tensor,
    threshold: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Run the Gemma Scope JumpReLU SAE forward pass.

    Gemma Scope JumpReLU artifacts store an encoder matrix with shape
    ``[d_model, width]`` and a decoder matrix with shape ``[width, d_model]``.
    The feature activation is ``relu(pre)`` gated by ``pre > threshold``.
    """

    pre_acts = (activations.float() - b_dec.float()) @ w_enc.float() + b_enc.float()
    feature_acts = t.relu(pre_acts) * (pre_acts > threshold.float())
    reconstructed = feature_acts @ w_dec.float() + b_dec.float()
    return feature_acts, reconstructed


def validate_gemma_scope_artifact_state(
    config: dict[str, Any],
    state: dict[str, t.Tensor],
    spec: GemmaScopeArtifactSpec = GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL,
) -> dict[str, Any]:
    """Validate config fields and tensor shapes for a pinned Gemma Scope artifact."""

    required_keys = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
    tensor_shapes = {key: list(value.shape) for key, value in state.items()}
    tensor_dtypes = {key: str(value.dtype) for key, value in state.items()}
    expected_shapes = {
        "w_enc": [spec.d_model, spec.width],
        "w_dec": [spec.width, spec.d_model],
        "b_enc": [spec.width],
        "b_dec": [spec.d_model],
        "threshold": [spec.width],
    }
    required_tensors_present = required_keys.issubset(state)
    tensor_shapes_match = required_tensors_present and all(
        tensor_shapes[key] == expected_shape for key, expected_shape in expected_shapes.items()
    )
    config_matches = (
        config.get("model_name") == spec.model_name
        and config.get("architecture") == spec.architecture
        and config.get("type") == spec.artifact_type
        and config.get("hf_hook_point_in") == spec.hook_point
        and config.get("hf_hook_point_out") == spec.hook_point
        and int(config.get("width", -1)) == spec.width
    )
    finite_tensors = required_tensors_present and all(
        bool(t.isfinite(state[key]).all().item()) for key in required_keys
    )
    metadata = FeatureArtifactMetadata(
        model_name=str(config.get("model_name", "")),
        artifact_name=spec.artifact_path,
        artifact_type=str(config.get("type", "")),  # type: ignore[arg-type]
        layer=spec.layer,
        hook_name=str(config.get("hf_hook_point_in", "")),
        d_model=spec.d_model,
        n_features=spec.width,
    )
    return {
        "required_tensors_present": required_tensors_present,
        "tensor_shapes_match_config": tensor_shapes_match,
        "config_matches_expected": config_matches,
        "tensors_finite": finite_tensors,
        "metadata_complete": metadata_is_complete(metadata),
        "tensor_shapes": tensor_shapes,
        "tensor_dtypes": tensor_dtypes,
        "expected_shapes": expected_shapes,
    }


def gemma_scope_artifact_preflight(
    *,
    max_vram_gb: float = 24.0,
    spec: GemmaScopeArtifactSpec = GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL,
) -> dict[str, Any]:
    """Load a pinned Gemma Scope SAE artifact and run a CUDA JumpReLU forward pass."""

    if not t.cuda.is_available():
        raise RuntimeError("Gemma Scope artifact preflight requires CUDA; no CPU fallback.")

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config_path = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            filename=f"{spec.artifact_path}/config.json",
        )
    )
    params_path = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            filename=f"{spec.artifact_path}/params.safetensors",
        )
    )
    config = json.loads(config_path.read_text())
    state_cpu = load_file(params_path, device="cpu")
    validation = validate_gemma_scope_artifact_state(config, state_cpu, spec)
    if not (
        validation["required_tensors_present"]
        and validation["tensor_shapes_match_config"]
        and validation["config_matches_expected"]
        and validation["tensors_finite"]
        and validation["metadata_complete"]
    ):
        raise ValueError(f"Gemma Scope artifact validation failed: {validation}")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats(device)
    state = {key: value.to(device) for key, value in state_cpu.items()}

    w_enc = state["w_enc"]
    b_enc = state["b_enc"]
    b_dec = state["b_dec"]
    threshold = state["threshold"]
    column_norm_sq = w_enc.square().sum(dim=0)
    candidate_mask = (
        t.isfinite(threshold)
        & t.isfinite(b_enc)
        & t.isfinite(column_norm_sq)
        & (column_norm_sq > 1e-8)
        & (threshold > b_enc)
    )
    candidate_ids = t.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_ids.numel() < 4:
        raise ValueError("Gemma Scope artifact has too few finite probeable encoder columns.")
    feature_ids = candidate_ids[:4]

    constructed_activations = []
    for feature_id in feature_ids:
        column = w_enc[:, feature_id]
        scale = (threshold[feature_id] - b_enc[feature_id] + 5.0) / column_norm_sq[
            feature_id
        ].clamp_min(1e-8)
        constructed_activations.append(b_dec + scale * column)
    activations = t.stack(constructed_activations)
    feature_acts, reconstructed = gemma_scope_jump_relu_forward(
        activations,
        w_enc=state["w_enc"],
        w_dec=state["w_dec"],
        b_enc=state["b_enc"],
        b_dec=state["b_dec"],
        threshold=state["threshold"],
    )
    selected_margins = (
        feature_acts[t.arange(feature_ids.numel(), device=device), feature_ids]
        - threshold[feature_ids]
    )
    forward_passed = bool(
        t.isfinite(feature_acts).all().item()
        and t.isfinite(reconstructed).all().item()
        and bool((selected_margins > 0).all().item())
    )
    active_feature_counts = (feature_acts > 0).sum(dim=-1)
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated(device) / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = forward_passed and within_vram_budget

    return {
        "preflight_passed": preflight_passed,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "artifact_path": spec.artifact_path,
        "config_path": str(config_path),
        "params_path": str(params_path),
        "model_name": config["model_name"],
        "architecture": config["architecture"],
        "artifact_type": config["type"],
        "hook_point_in": config["hf_hook_point_in"],
        "hook_point_out": config["hf_hook_point_out"],
        "layer": spec.layer,
        "width": spec.width,
        "d_model": spec.d_model,
        "artifact_l0_target": int(config["l0"]),
        "validation": validation,
        "forward_passed": forward_passed,
        "selected_feature_ids": [int(x) for x in feature_ids.detach().cpu().tolist()],
        "selected_feature_margins": [
            float(x) for x in selected_margins.detach().cpu().tolist()
        ],
        "constructed_probe_active_features_mean": float(
            active_feature_counts.float().mean().detach().cpu().item()
        ),
        "constructed_probe_active_features_min": int(active_feature_counts.min().cpu().item()),
        "feature_max": float(feature_acts.max().detach().cpu().item()),
        "reconstruction_norm_mean": float(
            reconstructed.norm(dim=-1).mean().detach().cpu().item()
        ),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "semantic_feature_claimed": False,
        "next_required_full_path": (
            "Run this pinned SAE on real Gemma 3 1B IT residual activations once gated "
            "model access is available, then validate features on held-out positives, "
            "matched negatives, ablation, steering, and random-feature controls."
        ),
    }


def gemma_scope_real_activation_preflight(
    *,
    max_vram_gb: float = 24.0,
    spec: GemmaScopeArtifactSpec = GEMMA_SCOPE_2_1B_IT_LAYER13_SMALL,
    max_length: int = 64,
) -> dict[str, Any]:
    """Run the pinned Gemma Scope SAE on real Gemma 3 residual activations.

    The validation split is deliberately benign: technical ML/tool/API prompts
    versus narrative prose prompts. The selected SAE feature is chosen only from
    the training half by mean activation difference, then evaluated on held-out
    prompts against a deterministic random-feature baseline and a label-shuffle
    control.
    """

    if not t.cuda.is_available():
        raise RuntimeError("Gemma Scope real-activation preflight requires CUDA.")

    base_access = gemma_base_model_access_report(spec=spec)
    if not base_access["ready_for_real_activations"]:
        raise RuntimeError(
            "Gemma 3 base model is not ready for real activation capture; "
            f"access report: {base_access}"
        )

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            filename=f"{spec.artifact_path}/config.json",
        )
    )
    params_path = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            filename=f"{spec.artifact_path}/params.safetensors",
        )
    )
    config = json.loads(config_path.read_text())
    state_cpu = load_file(params_path, device="cpu")
    validation = validate_gemma_scope_artifact_state(config, state_cpu, spec)
    if not (
        validation["required_tensors_present"]
        and validation["tensor_shapes_match_config"]
        and validation["config_matches_expected"]
        and validation["tensors_finite"]
        and validation["metadata_complete"]
    ):
        raise ValueError(f"Gemma Scope artifact validation failed: {validation}")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats(device)
    state = {key: value.to(device) for key, value in state_cpu.items()}
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_name,
        revision=spec.base_model_revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_name,
        revision=spec.base_model_revision,
        dtype=t.bfloat16,
        device_map="cuda",
    )
    model.eval()

    prompts = list(REAL_GEMMA_TECHNICAL_PROMPTS + REAL_GEMMA_NARRATIVE_PROMPTS)
    labels = t.tensor(
        [1] * len(REAL_GEMMA_TECHNICAL_PROMPTS) + [0] * len(REAL_GEMMA_NARRATIVE_PROMPTS),
        dtype=t.bool,
        device=device,
    )
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with t.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
        residual = outputs.hidden_states[spec.layer + 1].float()
        feature_acts, reconstructed = gemma_scope_jump_relu_forward(
            residual,
            w_enc=state["w_enc"],
            w_dec=state["w_dec"],
            b_enc=state["b_enc"],
            b_dec=state["b_dec"],
            threshold=state["threshold"],
        )

    attention_mask = inputs["attention_mask"].bool()
    feature_acts = feature_acts.masked_fill(~attention_mask[..., None], 0)
    feature_scores = feature_acts.max(dim=1).values
    train_indices = t.tensor([0, 1, 2, 3, 8, 9, 10, 11], device=device)
    heldout_indices = t.tensor([4, 5, 6, 7, 12, 13, 14, 15], device=device)
    train_scores = feature_scores[train_indices]
    train_labels = labels[train_indices]
    heldout_scores = feature_scores[heldout_indices]
    heldout_labels = labels[heldout_indices]

    positive_mean = train_scores[train_labels].mean(dim=0)
    negative_mean = train_scores[~train_labels].mean(dim=0)
    train_diff = positive_mean - negative_mean
    selected_feature_id = int(train_diff.argmax().item())
    random_feature_id = int((selected_feature_id + 7919) % spec.width)
    validation_report = validate_feature_scores(
        heldout_scores[:, selected_feature_id],
        heldout_labels,
        heldout_scores[:, random_feature_id],
        min_auc_margin=0.25,
    )
    label_shuffle_auc = roc_auc_binary(
        heldout_scores[:, selected_feature_id],
        ~heldout_labels,
    )
    prompted_reconstruction_mse = (
        (reconstructed[attention_mask].float() - residual[attention_mask].float())
        .square()
        .mean()
        .item()
    )
    residual_shape = list(residual.shape)
    active_features_per_prompt_token = (
        (feature_acts > 0).sum(dim=-1).float()[attention_mask].mean().item()
    )
    t.cuda.synchronize(device)
    peak_vram_gb = t.cuda.max_memory_allocated(device) / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    real_activation_forward_passed = bool(
        t.isfinite(residual).all().item()
        and t.isfinite(feature_acts).all().item()
        and t.isfinite(reconstructed).all().item()
    )
    semantic_feature_claimed = bool(
        real_activation_forward_passed
        and validation_report.passes_baseline
        and validation_report.feature_auc >= 0.95
        and label_shuffle_auc <= 0.05
        and within_vram_budget
    )

    del model, outputs, residual, feature_acts, reconstructed, state
    t.cuda.empty_cache()

    return {
        "preflight_passed": semantic_feature_claimed,
        "model_id": spec.model_name,
        "model_revision": spec.base_model_revision,
        "model_layer": spec.layer,
        "hook_point": spec.hook_point,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "artifact_path": spec.artifact_path,
        "semantic_axis": "technical_ml_tool_text_vs_narrative_prose",
        "prompt_count": len(prompts),
        "train_prompt_count": int(train_indices.numel()),
        "heldout_prompt_count": int(heldout_indices.numel()),
        "technical_prompt_count": len(REAL_GEMMA_TECHNICAL_PROMPTS),
        "narrative_prompt_count": len(REAL_GEMMA_NARRATIVE_PROMPTS),
        "max_length": max_length,
        "residual_shape": residual_shape,
        "feature_score_shape": list(feature_scores.shape),
        "selected_feature_id": selected_feature_id,
        "random_control_feature_id": random_feature_id,
        "selected_feature_train_margin": float(train_diff[selected_feature_id].item()),
        "feature_auc": validation_report.feature_auc,
        "baseline_auc": validation_report.baseline_auc,
        "auc_margin": validation_report.auc_margin,
        "threshold_accuracy": validation_report.threshold_accuracy,
        "positive_mean": validation_report.positive_mean,
        "negative_mean": validation_report.negative_mean,
        "passes_random_feature_baseline": validation_report.passes_baseline,
        "label_shuffle_auc": label_shuffle_auc,
        "label_shuffle_control_passed": label_shuffle_auc <= 0.05,
        "active_features_per_prompt_token": active_features_per_prompt_token,
        "prompted_reconstruction_mse": prompted_reconstruction_mse,
        "real_activation_forward_passed": real_activation_forward_passed,
        "semantic_feature_claimed": semantic_feature_claimed,
        "base_access_report": base_access,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
    }


def metadata_is_complete(metadata: FeatureArtifactMetadata) -> bool:
    """Check minimum artifact metadata needed for reproducible feature work."""

    return (
        bool(metadata.model_name)
        and bool(metadata.artifact_name)
        and metadata.artifact_type in {"sae", "transcoder"}
        and metadata.layer >= 0
        and bool(metadata.hook_name)
        and metadata.d_model > 0
        and metadata.n_features > 0
    )


def features_with_tag(
    features: list[TaggedFeatureSpec],
    tag: str,
) -> list[TaggedFeatureSpec]:
    """Return feature specs carrying a tag such as refusal, code, or sentiment."""

    return [feature for feature in features if tag in feature.tags]


def feature_score_vector(
    feature_acts: t.Tensor,
    feature_id: int,
    *,
    reduction: Literal["max", "mean", "last"] = "max",
) -> t.Tensor:
    """Reduce per-token feature activations to one score per example."""

    if feature_acts.ndim == 2:
        scores = feature_acts[:, feature_id]
        return scores.float()
    if feature_acts.ndim != 3:
        raise ValueError(
            "feature_acts must have shape (examples, features) or (examples, seq, features)."
        )
    if feature_id < 0 or feature_id >= feature_acts.shape[-1]:
        raise IndexError("feature_id is out of range.")

    per_token = feature_acts[..., feature_id].float()
    if reduction == "max":
        return per_token.max(dim=-1).values
    if reduction == "mean":
        return per_token.mean(dim=-1)
    if reduction == "last":
        return per_token[:, -1]
    raise ValueError("reduction must be 'max', 'mean', or 'last'.")


def validate_feature_scores(
    feature_scores: t.Tensor,
    labels: t.Tensor,
    baseline_scores: t.Tensor,
    *,
    min_auc_margin: float = 0.1,
) -> FeatureValidationSuiteReport:
    """Validate a feature against matched negatives and a baseline score vector."""

    feature_report = feature_detection_report(feature_scores, labels)
    baseline_auc = roc_auc_binary(baseline_scores.flatten().float(), labels.flatten().bool())
    auc_margin = feature_report.auc - baseline_auc
    return FeatureValidationSuiteReport(
        feature_auc=feature_report.auc,
        baseline_auc=baseline_auc,
        auc_margin=auc_margin,
        threshold_accuracy=feature_report.threshold_accuracy,
        positive_mean=feature_report.positive_mean,
        negative_mean=feature_report.negative_mean,
        passes_baseline=auc_margin >= min_auc_margin,
    )


def base_instruction_feature_delta(
    base_scores: t.Tensor,
    instruction_scores: t.Tensor,
) -> BaseInstructionFeatureDelta:
    """Compare mean feature activation between base and instruction-tuned models."""

    if base_scores.shape != instruction_scores.shape:
        raise ValueError("base_scores and instruction_scores must have matching shapes.")
    base_mean = base_scores.float().mean().item()
    instruction_mean = instruction_scores.float().mean().item()
    delta = instruction_mean - base_mean
    return BaseInstructionFeatureDelta(
        base_mean=base_mean,
        instruction_mean=instruction_mean,
        delta=delta,
        abs_delta=abs(delta),
    )


def ablation_control_report(
    baseline_scores: t.Tensor,
    ablated_scores: t.Tensor,
    random_ablated_scores: t.Tensor,
) -> AblationControlReport:
    """Check whether ablating a feature reduces a target behavior more than control."""

    baseline_mean = baseline_scores.float().mean().item()
    ablated_mean = ablated_scores.float().mean().item()
    random_mean = random_ablated_scores.float().mean().item()
    ablation_delta = baseline_mean - ablated_mean
    random_delta = baseline_mean - random_mean
    return AblationControlReport(
        baseline_mean=baseline_mean,
        ablated_mean=ablated_mean,
        random_ablated_mean=random_mean,
        ablation_delta=ablation_delta,
        random_delta=random_delta,
        passes_control=ablation_delta > random_delta,
    )


def steering_safety_report(
    baseline_scores: t.Tensor,
    steered_scores: t.Tensor,
    random_control_scores: t.Tensor,
    *,
    baseline_perplexity: float,
    steered_perplexity: float,
    max_perplexity_ratio: float = 1.2,
) -> SteeringSafetyReport:
    """Check steering target increase while bounding perplexity degradation."""

    if baseline_perplexity <= 0:
        raise ValueError("baseline_perplexity must be positive.")
    baseline_mean = baseline_scores.float().mean().item()
    steered_mean = steered_scores.float().mean().item()
    random_mean = random_control_scores.float().mean().item()
    steered_delta = steered_mean - baseline_mean
    random_delta = random_mean - baseline_mean
    perplexity_ratio = steered_perplexity / baseline_perplexity
    return SteeringSafetyReport(
        baseline_mean=baseline_mean,
        steered_mean=steered_mean,
        random_mean=random_mean,
        steered_delta=steered_delta,
        random_delta=random_delta,
        perplexity_ratio=perplexity_ratio,
        passes_control=steered_delta > random_delta,
        passes_perplexity_guard=perplexity_ratio <= max_perplexity_ratio,
    )
