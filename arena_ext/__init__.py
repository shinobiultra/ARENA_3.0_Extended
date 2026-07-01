"""Reusable helpers for ARENA frontier extension notebooks.

The extension package intentionally keeps imports lazy. Hosted CPU audits inspect
report metadata without installing torch, while notebook and CUDA paths still
import the torch-backed modules when their symbols are actually used.
"""

from importlib import import_module
from typing import Any


_SUBMODULES = {
    "activation_language",
    "activation_oracles",
    "activation_patching",
    "activation_store",
    "attribution_patching",
    "capstone",
    "circuit_metrics",
    "circuit_tracing",
    "cot_faithfulness",
    "crosscoders",
    "data_contracts",
    "diffusion_lm",
    "environment",
    "fake_interpretability",
    "feature_verbalizers",
    "features",
    "gated_artifacts",
    "gemma",
    "gemma_scope",
    "hf_parity",
    "image_generation_interpretability",
    "jepa_world_models",
    "mamba",
    "memory_budget",
    "natural_language_autoencoders",
    "peft_interpretability",
    "predictive_concept_decoders",
    "proxy_drift_detection",
    "refusal_steering",
    "representation_geometry",
    "sae_variants",
    "shapley_attribution",
    "shapley_neural_game",
    "sparse_feature_circuits",
    "specialist_models",
    "state_tracking",
    "streamlit_home",
    "training_dynamics",
    "transcoders",
    "verification",
    "vlm_interpretability",
    "white_box_monitors",
}

_EXPORTS = {
    "ActivationRecord": ("activation_store", "ActivationRecord"),
    "DiskActivationStore": ("activation_store", "DiskActivationStore"),
    "EnvironmentReport": ("environment", "EnvironmentReport"),
    "MemoryBudget": ("memory_budget", "MemoryBudget"),
    "ParityReport": ("hf_parity", "ParityReport"),
    "VerificationReport": ("verification", "VerificationReport"),
    "build_verification_report": ("verification", "build_verification_report"),
    "compare_logits": ("hf_parity", "compare_logits"),
    "cuda_environment": ("verification", "cuda_environment"),
    "current_git_commit": ("verification", "current_git_commit"),
    "deterministic_generation_equal": (
        "hf_parity",
        "deterministic_generation_equal",
    ),
    "estimate_inference_memory": ("memory_budget", "estimate_inference_memory"),
    "get_environment_report": ("environment", "get_environment_report"),
    "write_verification_report": ("verification", "write_verification_report"),
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _EXPORTS:
        module_name, symbol_name = _EXPORTS[name]
        module = import_module(f"{__name__}.{module_name}")
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted([*_SUBMODULES, *_EXPORTS])
