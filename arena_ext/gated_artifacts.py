"""Access and preparation helpers for required gated Hugging Face artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HFGatedArtifactSpec:
    repo_id: str
    revision: str
    required_patterns: tuple[str, ...]
    download_patterns: tuple[str, ...]
    purpose: str


GEMMA3_1B_IT_BASE = HFGatedArtifactSpec(
    repo_id="google/gemma-3-1b-it",
    revision="dcc83ea841ab6100d6b47a070329e1ba4cf78752",
    required_patterns=("config.json", "tokenizer.json", "model.safetensors"),
    download_patterns=(
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "model.safetensors",
    ),
    purpose="Gemma Scope semantic activation capture",
)

EMBEDDINGGEMMA_300M = HFGatedArtifactSpec(
    repo_id="google/embeddinggemma-300m",
    revision="57c266a740f537b4dc058e1b0cda161fd15afa75",
    required_patterns=("config.json", "model.safetensors", "tokenizer.json"),
    download_patterns=(
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentence_bert_config.json",
        "modules.json",
        "1_Pooling/config.json",
        "2_Dense/config.json",
        "2_Dense/model.safetensors",
        "3_Dense/config.json",
        "3_Dense/model.safetensors",
    ),
    purpose="direct EmbeddingGemma retrieval baseline",
)

FUNCTIONGEMMA_270M_IT_BASE = HFGatedArtifactSpec(
    repo_id="google/functiongemma-270m-it",
    revision="39eccb091651513a5dfb56892d3714c1b5b8276c",
    required_patterns=("config.json", "tokenizer.json", "model.safetensors"),
    download_patterns=(
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "model.safetensors",
    ),
    purpose="direct FunctionGemma base-model validation",
)

DIFFUSIONGEMMA_26B_A4B_IT = HFGatedArtifactSpec(
    repo_id="google/diffusiongemma-26B-A4B-it",
    revision="0f28bc42f588fbd8f71e08102b1c3960298a1358",
    required_patterns=(
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-00001-of-00011.safetensors",
        "model-00002-of-00011.safetensors",
        "model-00003-of-00011.safetensors",
        "model-00004-of-00011.safetensors",
        "model-00005-of-00011.safetensors",
        "model-00006-of-00011.safetensors",
        "model-00007-of-00011.safetensors",
        "model-00008-of-00011.safetensors",
        "model-00009-of-00011.safetensors",
        "model-00010-of-00011.safetensors",
        "model-00011-of-00011.safetensors",
    ),
    download_patterns=(
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-*.safetensors",
    ),
    purpose="direct DiffusionGemma BF16 block-diffusion generation",
)

NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4 = HFGatedArtifactSpec(
    repo_id="nvidia/diffusiongemma-26B-A4B-it-NVFP4",
    revision="2ea837236295d617ac27f8c17d61228081932c40",
    required_patterns=(
        "config.json",
        "generation_config.json",
        "hf_quant_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ),
    download_patterns=(
        "config.json",
        "generation_config.json",
        "hf_quant_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-*.safetensors",
    ),
    purpose="DiffusionGemma NVFP4 local inference through a compatible vLLM runtime",
)

REQUIRED_GEMMA_FAMILY_ARTIFACTS = (
    GEMMA3_1B_IT_BASE,
    EMBEDDINGGEMMA_300M,
    FUNCTIONGEMMA_270M_IT_BASE,
)

REQUIRED_DIFFUSIONGEMMA_ARTIFACTS = (
    DIFFUSIONGEMMA_26B_A4B_IT,
    NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4,
)


def _safe_error_text(error: BaseException, *, limit: int = 400) -> str:
    return str(error).replace("\n", " ")[:limit]


def hf_cache_repo_dir(repo_id: str, *, cache_root: Path | None = None) -> Path:
    if cache_root is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_root = Path(HF_HUB_CACHE)
    return cache_root / f"models--{repo_id.replace('/', '--')}"


def _matches_required_pattern(path: Path, pattern: str) -> bool:
    if "/" in pattern:
        return path.match(f"**/{pattern}")
    return path.name == pattern


def _missing_patterns(paths: list[Path], patterns: tuple[str, ...]) -> list[str]:
    missing = []
    for pattern in patterns:
        if not any(_matches_required_pattern(path, pattern) for path in paths):
            missing.append(pattern)
    return missing


def hf_model_artifact_access_report(
    spec: HFGatedArtifactSpec,
    *,
    allow_network: bool = True,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Report whether a required HF artifact is actually usable.

    Listing a gated repo's filenames is not enough: this reports authenticated
    status, local cache readiness, and remote download readiness separately.
    """

    from huggingface_hub import HfApi

    cache_dir = hf_cache_repo_dir(spec.repo_id, cache_root=cache_root)
    local_files = []
    if cache_dir.exists():
        local_files = [
            path
            for path in cache_dir.rglob("*")
            if path.is_file() and "refs" not in path.parts
        ]
    local_relative_files = [path.relative_to(cache_dir) for path in local_files]
    missing_local = _missing_patterns(local_relative_files, spec.required_patterns)

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
                HfApi().list_repo_files(
                    spec.repo_id,
                    repo_type="model",
                    revision=spec.revision,
                )
            )
            repo_listed = True
        except Exception as error:  # pragma: no cover - environment dependent
            access_error_type = type(error).__name__
            access_error = _safe_error_text(error)

    remote_paths = [Path(path) for path in repo_files]
    missing_remote = _missing_patterns(remote_paths, spec.required_patterns)
    local_ready = not missing_local
    remote_download_ready = authenticated and repo_listed and not missing_remote

    return {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "purpose": spec.purpose,
        "required_patterns": list(spec.required_patterns),
        "download_patterns": list(spec.download_patterns),
        "cache_dir": str(cache_dir),
        "local_non_ref_file_count": len(local_files),
        "missing_local_patterns": missing_local,
        "local_ready_for_direct_loading": local_ready,
        "authenticated": authenticated,
        "auth_error_type": auth_error_type,
        "auth_error": auth_error,
        "repo_listed": repo_listed,
        "repo_file_count": len(repo_files),
        "missing_remote_patterns": missing_remote,
        "access_error_type": access_error_type,
        "access_error": access_error,
        "remote_download_ready": remote_download_ready,
        "ready_for_direct_loading": local_ready or remote_download_ready,
    }


def download_required_artifact(spec: HFGatedArtifactSpec, *, max_workers: int | None = None) -> str:
    """Download the exact required artifact; requires accepted/authenticated access."""

    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {}
    if max_workers is not None:
        kwargs["max_workers"] = max_workers

    return snapshot_download(
        spec.repo_id,
        repo_type="model",
        revision=spec.revision,
        allow_patterns=list(spec.download_patterns),
        token=True,
        **kwargs,
    )
