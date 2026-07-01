# %%
"""Reference solutions for [8.5] Sparse Feature Circuits."""

import json
import math
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = "chapter8_automated_circuits"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"

OFFICIAL_FEATURE_CIRCUITS_REPO = "https://github.com/saprmarks/feature-circuits"
OFFICIAL_FEATURE_CIRCUITS_HEAD = "7fbd82b895ae16294f4e6fc7bfc675d1d680d659"
OFFICIAL_SFC_HF_REPO = "saprmarks/pythia-70m-deduped-saes"
OFFICIAL_SFC_ZIP = "dictionaries_pythia-70m-deduped_10.zip"
OFFICIAL_SFC_HF_COMMIT = "50a434461d36ed78d1b0b901944e6edc829f1dce"
OFFICIAL_SFC_ZIP_SIZE_BYTES = 2_369_891_306
OFFICIAL_SFC_GRAPH_CIRCUIT = "pythia-70m-deduped_simple_train_n100_aggsum_node0.2.pt"
OFFICIAL_SFC_GRAPH_FIGURE = (
    "pythia-70m-deduped_simple_train_n100_aggsum_node0.2_edge0.1_n100_aggsum.png"
)
OFFICIAL_SFC_HELDOUT_EXAMPLES = 40
OFFICIAL_SFC_FAITHFULNESS_START_LAYER = 2
OFFICIAL_SFC_FAITHFULNESS_NODE_THRESHOLD = 0.0


# %%
@dataclass(frozen=True)
class FeatureNodePatchingReport:
    selected_feature_ids: tuple[int, ...]
    full_logit_diff: float
    graph_logit_diff: float
    recovered_fraction: float
    passes_recovery: bool


@dataclass(frozen=True)
class FeatureEdgePatchingReport:
    selected_edges: tuple[tuple[int, int], ...]
    full_edge_score: float
    graph_edge_score: float
    recovered_fraction: float
    passes_recovery: bool


@dataclass(frozen=True)
class EAPIGComparisonReport:
    exact_error: float
    eap_error: float
    eap_ig_error: float
    eap_passes: bool
    eap_ig_improves: bool


@dataclass(frozen=True)
class FeatureGraphThresholdReport:
    selected_feature_ids: tuple[int, ...]
    threshold: float
    full_logit_diff: float
    graph_logit_diff: float
    recovered_fraction: float
    passes_threshold: bool


@dataclass(frozen=True)
class RandomFeatureGraphControlReport:
    target_graph_logit_diff: float
    random_graph_logit_diff: float
    target_recovered_fraction: float
    random_recovered_fraction: float
    margin: float
    random_graph_fails: bool


@dataclass(frozen=True)
class SparseFeatureEditingReport:
    target_feature_ids: tuple[int, ...]
    spurious_feature_ids: tuple[int, ...]
    random_feature_ids: tuple[int, ...]
    baseline_train_accuracy: float
    baseline_ood_accuracy: float
    edited_train_accuracy: float
    edited_ood_accuracy: float
    random_edit_ood_accuracy: float
    black_box_baseline_ood_accuracy: float
    spurious_reliance_before: float
    spurious_reliance_after: float
    target_reliance_after: float
    ood_improvement: float
    random_edit_improvement: float
    target_accuracy_drop: float
    spurious_reliance_decreases: bool
    target_task_preserved: bool
    ood_generalization_improves: bool
    random_edit_control_fails: bool
    editing_passes: bool


@dataclass(frozen=True)
class ResidualFeaturePreflightReport:
    clean_logit_diff: float
    corrupt_logit_diff: float
    total_effect: float
    linearization_error: float
    selected_feature_ids: tuple[int, ...]
    selected_effect: float
    recovered_fraction: float
    random_feature_ids: tuple[int, ...]
    random_effect: float
    random_recovered_fraction: float
    random_control_fails: bool


@dataclass(frozen=True)
class OfficialSparseFeatureArtifactReadinessReport:
    official_repo_commit: str
    hf_commit_hash: str
    hf_zip_size_gb: float
    expected_dictionary_count: int
    missing_dictionary_paths: tuple[str, ...]
    missing_repo_files: tuple[str, ...]
    hf_zip_present: bool
    remote_manifest_passed: bool
    local_dictionaries_ready: bool
    ready_for_gt2_replication: bool


@dataclass(frozen=True)
class SparseAutoencoderStateDictSmokeReport:
    activation_dim: int
    dict_size: int
    feature_l0: int
    feature_max: float
    reconstruction_mse: float
    relative_l2_error: float
    shapes_match: bool
    tensors_finite: bool
    passes_smoke: bool


@dataclass(frozen=True)
class SparseFeatureAttributionSmokeReport:
    total_residual_effect: float
    decoded_feature_effect: float
    sae_error_effect: float
    selected_feature_ids: tuple[int, ...]
    selected_effect: float
    recovered_fraction: float
    random_feature_ids: tuple[int, ...]
    random_effect: float
    random_recovered_fraction: float
    clean_feature_l0: int
    corrupt_feature_l0: int
    random_control_fails: bool
    passes_smoke: bool


@dataclass(frozen=True)
class OfficialSparseFeatureCircuitArtifactReport:
    circuit_path: str
    figure_path: str
    circuit_present: bool
    figure_present: bool
    examples: int
    node_submodule_count: int
    edge_group_count: int
    thresholded_node_count: int
    thresholded_edge_count: int
    max_abs_node_effect: float
    max_abs_edge_effect: float
    passes_smoke: bool


def _index_tensor(indices: t.Tensor | list[int] | tuple[int, ...], *, device: t.device) -> t.Tensor:
    if isinstance(indices, t.Tensor):
        return indices.to(device=device, dtype=t.long).flatten()
    return t.tensor(list(indices), device=device, dtype=t.long)


def _require_finite_tensor(name: str, tensor: t.Tensor) -> t.Tensor:
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not t.isfinite(tensor.float()).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _require_finite_nonnegative(name: str, value: float) -> float:
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be finite.")
    if value_float < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value_float


def _validate_index_tensor(ids: t.Tensor, *, name: str, upper_bound: int) -> t.Tensor:
    if ids.numel() == 0:
        raise ValueError(f"at least one {name} id is required.")
    if ids.min().item() < 0 or ids.max().item() >= upper_bound:
        raise ValueError(f"{name} id is out of range.")
    if ids.unique().numel() != ids.numel():
        raise ValueError(f"{name} ids must be unique.")
    return ids


def _fraction(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise ValueError("reference score must be nonzero.")
    return numerator / denominator


def _flatten_effect_tensor(tensor: t.Tensor) -> t.Tensor:
    if tensor.is_sparse:
        return tensor.coalesce().values().detach().float().flatten()
    return tensor.detach().float().flatten()


def _flatten_effect_values(value) -> tuple[t.Tensor, ...]:
    if isinstance(value, t.Tensor):
        return (_flatten_effect_tensor(value),)
    tensors = []
    for attribute in ("act", "resc", "res"):
        tensor = getattr(value, attribute, None)
        if isinstance(tensor, t.Tensor):
            tensors.append(_flatten_effect_tensor(tensor))
    return tuple(tensor for tensor in tensors if tensor.numel() > 0)


def _count_thresholded(tensors: list[t.Tensor], threshold: float) -> tuple[int, float]:
    if not tensors:
        return 0, 0.0
    values = t.cat([tensor.abs() for tensor in tensors if tensor.numel() > 0])
    if values.numel() == 0:
        return 0, 0.0
    values = _require_finite_tensor("effect values", values)
    return int(values.gt(threshold).sum().item()), float(values.max().item())


def expected_pythia_sfc_dictionary_paths(num_layers: int = 6) -> tuple[str, ...]:
    """Return expected unzipped Sparse Feature Circuits Pythia SAE paths."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    paths = ["embed/10_32768/ae.pt"]
    for layer in range(num_layers):
        paths.extend(
            [
                f"attn_out_layer{layer}/10_32768/ae.pt",
                f"mlp_out_layer{layer}/10_32768/ae.pt",
                f"resid_out_layer{layer}/10_32768/ae.pt",
            ]
        )
    return tuple(paths)


def exact_feature_node_patching_report(
    feature_contributions: t.Tensor,
    feature_ids: t.Tensor | list[int] | tuple[int, ...],
    *,
    min_recovered_fraction: float = 0.75,
) -> FeatureNodePatchingReport:
    """Treat feature contributions as an exact node-patching oracle."""

    min_recovered_fraction = _require_finite_nonnegative(
        "min_recovered_fraction",
        min_recovered_fraction,
    )
    contributions = _require_finite_tensor(
        "feature_contributions",
        feature_contributions.flatten().float(),
    )
    ids = _validate_index_tensor(
        _index_tensor(feature_ids, device=contributions.device),
        name="feature",
        upper_bound=contributions.numel(),
    )
    full_logit_diff = contributions.sum().item()
    graph_logit_diff = contributions[ids].sum().item()
    recovered_fraction = _fraction(graph_logit_diff, full_logit_diff)
    return FeatureNodePatchingReport(
        selected_feature_ids=tuple(int(index) for index in ids.tolist()),
        full_logit_diff=full_logit_diff,
        graph_logit_diff=graph_logit_diff,
        recovered_fraction=recovered_fraction,
        passes_recovery=recovered_fraction >= min_recovered_fraction,
    )


def exact_feature_edge_patching_report(
    edge_scores: t.Tensor,
    selected_edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    min_recovered_fraction: float = 0.75,
) -> FeatureEdgePatchingReport:
    """Treat absolute edge scores as an exact edge-patching oracle."""

    min_recovered_fraction = _require_finite_nonnegative(
        "min_recovered_fraction",
        min_recovered_fraction,
    )
    if edge_scores.ndim != 2:
        raise ValueError("edge_scores must have shape (sources, features).")
    edge_scores = _require_finite_tensor("edge_scores", edge_scores.float())
    if not selected_edges:
        raise ValueError("at least one edge is required.")
    edge_magnitudes = edge_scores.abs()
    selected_score = 0.0
    normalized_edges = []
    for source_id, feature_id in selected_edges:
        if not 0 <= source_id < edge_scores.shape[0]:
            raise ValueError("source id is out of range.")
        if not 0 <= feature_id < edge_scores.shape[1]:
            raise ValueError("feature id is out of range.")
        selected_score += edge_magnitudes[source_id, feature_id].item()
        normalized_edges.append((int(source_id), int(feature_id)))
    if len(set(normalized_edges)) != len(normalized_edges):
        raise ValueError("selected edges must be unique.")
    full_score = edge_magnitudes.sum().item()
    recovered_fraction = _fraction(selected_score, full_score)
    return FeatureEdgePatchingReport(
        selected_edges=tuple(normalized_edges),
        full_edge_score=full_score,
        graph_edge_score=selected_score,
        recovered_fraction=recovered_fraction,
        passes_recovery=recovered_fraction >= min_recovered_fraction,
    )


def eap_ig_comparison_report(
    exact_scores: t.Tensor,
    eap_scores: t.Tensor,
    eap_ig_scores: t.Tensor,
    *,
    max_eap_ig_error: float = 0.05,
) -> EAPIGComparisonReport:
    """Compare EAP and EAP-IG approximations against exact patching scores."""

    max_eap_ig_error = _require_finite_nonnegative(
        "max_eap_ig_error",
        max_eap_ig_error,
    )
    if exact_scores.shape != eap_scores.shape or exact_scores.shape != eap_ig_scores.shape:
        raise ValueError("exact, EAP, and EAP-IG scores must have matching shapes.")
    exact = _require_finite_tensor("exact_scores", exact_scores.float())
    eap = _require_finite_tensor("eap_scores", eap_scores.float())
    eap_ig = _require_finite_tensor("eap_ig_scores", eap_ig_scores.float())
    eap_error = (exact - eap).abs().mean().item()
    eap_ig_error = (exact - eap_ig).abs().mean().item()
    return EAPIGComparisonReport(
        exact_error=0.0,
        eap_error=eap_error,
        eap_ig_error=eap_ig_error,
        eap_passes=eap_error <= max_eap_ig_error,
        eap_ig_improves=eap_ig_error < eap_error and eap_ig_error <= max_eap_ig_error,
    )


def threshold_feature_graph_report(
    feature_contributions: t.Tensor,
    *,
    threshold: float,
    min_recovered_fraction: float = 0.8,
) -> FeatureGraphThresholdReport:
    """Select features above a contribution threshold and check preservation."""

    threshold = _require_finite_nonnegative("threshold", threshold)
    min_recovered_fraction = _require_finite_nonnegative(
        "min_recovered_fraction",
        min_recovered_fraction,
    )
    contributions = _require_finite_tensor(
        "feature_contributions",
        feature_contributions.flatten().float(),
    )
    selected = contributions.abs().ge(threshold).nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        raise ValueError("threshold selected no features.")
    node_report = exact_feature_node_patching_report(
        contributions,
        selected,
        min_recovered_fraction=min_recovered_fraction,
    )
    return FeatureGraphThresholdReport(
        selected_feature_ids=node_report.selected_feature_ids,
        threshold=threshold,
        full_logit_diff=node_report.full_logit_diff,
        graph_logit_diff=node_report.graph_logit_diff,
        recovered_fraction=node_report.recovered_fraction,
        passes_threshold=node_report.passes_recovery,
    )


def random_feature_graph_control_report(
    feature_contributions: t.Tensor,
    target_feature_ids: t.Tensor | list[int] | tuple[int, ...],
    random_feature_ids: t.Tensor | list[int] | tuple[int, ...],
    *,
    min_margin: float = 0.2,
) -> RandomFeatureGraphControlReport:
    """Check that the target sparse-feature graph beats a random graph."""

    min_margin = _require_finite_nonnegative("min_margin", min_margin)
    contributions = _require_finite_tensor(
        "feature_contributions",
        feature_contributions.flatten().float(),
    )
    target_ids = _validate_index_tensor(
        _index_tensor(target_feature_ids, device=contributions.device),
        name="target feature",
        upper_bound=contributions.numel(),
    )
    random_ids = _validate_index_tensor(
        _index_tensor(random_feature_ids, device=contributions.device),
        name="random feature",
        upper_bound=contributions.numel(),
    )
    if target_ids.numel() != random_ids.numel():
        raise ValueError("target and random graphs must have the same number of features.")
    if set(target_ids.tolist()) & set(random_ids.tolist()):
        raise ValueError("random graph control must not overlap target features.")
    target = exact_feature_node_patching_report(contributions, target_ids)
    random = exact_feature_node_patching_report(contributions, random_ids)
    margin = target.recovered_fraction - random.recovered_fraction
    return RandomFeatureGraphControlReport(
        target_graph_logit_diff=target.graph_logit_diff,
        random_graph_logit_diff=random.graph_logit_diff,
        target_recovered_fraction=target.recovered_fraction,
        random_recovered_fraction=random.recovered_fraction,
        margin=margin,
        random_graph_fails=margin >= min_margin,
    )


def _binary_accuracy(logits: t.Tensor, labels: t.Tensor) -> float:
    predictions = t.where(logits >= 0, t.ones_like(labels), -t.ones_like(labels))
    return predictions.eq(labels).float().mean().item()


def shift_style_sparse_feature_editing_report(
    train_features: t.Tensor,
    train_labels: t.Tensor,
    ood_features: t.Tensor,
    ood_labels: t.Tensor,
    classifier_weights: t.Tensor,
    *,
    target_feature_ids: t.Tensor | list[int] | tuple[int, ...],
    spurious_feature_ids: t.Tensor | list[int] | tuple[int, ...],
    random_feature_ids: t.Tensor | list[int] | tuple[int, ...],
    suppression: float = 0.0,
    max_target_accuracy_drop: float = 0.05,
    min_ood_improvement: float = 0.5,
    min_random_edit_gap: float = 0.5,
) -> SparseFeatureEditingReport:
    """Run a safe toy SHIFT-style sparse-feature edit."""

    suppression = float(suppression)
    if not math.isfinite(suppression) or suppression < 0 or suppression > 1:
        raise ValueError("suppression must be between 0 and 1.")
    max_target_accuracy_drop = _require_finite_nonnegative(
        "max_target_accuracy_drop",
        max_target_accuracy_drop,
    )
    min_ood_improvement = _require_finite_nonnegative(
        "min_ood_improvement",
        min_ood_improvement,
    )
    min_random_edit_gap = _require_finite_nonnegative(
        "min_random_edit_gap",
        min_random_edit_gap,
    )
    train = _require_finite_tensor("train_features", train_features.float())
    ood = _require_finite_tensor("ood_features", ood_features.float())
    labels_train = _require_finite_tensor("train_labels", train_labels.flatten().float())
    labels_ood = _require_finite_tensor("ood_labels", ood_labels.flatten().float())
    weights = _require_finite_tensor(
        "classifier_weights",
        classifier_weights.flatten().float(),
    )
    if train.ndim != 2 or ood.ndim != 2:
        raise ValueError("train_features and ood_features must have shape [batch, features].")
    if train.shape[1] != weights.numel() or ood.shape[1] != weights.numel():
        raise ValueError("feature tensors must share classifier_weights feature dimension.")
    if train.shape[0] != labels_train.numel() or ood.shape[0] != labels_ood.numel():
        raise ValueError("labels must have one entry per feature row.")
    if not set(labels_train.tolist() + labels_ood.tolist()).issubset({-1.0, 1.0}):
        raise ValueError("labels must be encoded as -1 or +1.")

    device = weights.device
    target_ids = _validate_index_tensor(
        _index_tensor(target_feature_ids, device=device),
        name="target feature",
        upper_bound=weights.numel(),
    )
    spurious_ids = _validate_index_tensor(
        _index_tensor(spurious_feature_ids, device=device),
        name="spurious feature",
        upper_bound=weights.numel(),
    )
    random_ids = _validate_index_tensor(
        _index_tensor(random_feature_ids, device=device),
        name="random feature",
        upper_bound=weights.numel(),
    )
    if spurious_ids.numel() != random_ids.numel():
        raise ValueError("random edit control must edit the same number of features.")
    if set(spurious_ids.tolist()) & set(random_ids.tolist()):
        raise ValueError("random edit control must not edit the spurious features.")
    if set(target_ids.tolist()) & set(spurious_ids.tolist()):
        raise ValueError("target and spurious features must be distinct.")

    baseline_train_accuracy = _binary_accuracy(train @ weights, labels_train)
    baseline_ood_accuracy = _binary_accuracy(ood @ weights, labels_ood)
    edited_weights = weights.clone()
    edited_weights[spurious_ids] *= suppression
    random_weights = weights.clone()
    random_weights[random_ids] *= suppression
    edited_train_accuracy = _binary_accuracy(train @ edited_weights, labels_train)
    edited_ood_accuracy = _binary_accuracy(ood @ edited_weights, labels_ood)
    random_edit_ood_accuracy = _binary_accuracy(ood @ random_weights, labels_ood)

    spurious_reliance_before = weights[spurious_ids].abs().sum().item()
    spurious_reliance_after = edited_weights[spurious_ids].abs().sum().item()
    target_reliance_after = edited_weights[target_ids].abs().sum().item()
    ood_improvement = edited_ood_accuracy - baseline_ood_accuracy
    random_edit_improvement = random_edit_ood_accuracy - baseline_ood_accuracy
    target_accuracy_drop = baseline_train_accuracy - edited_train_accuracy
    spurious_reliance_decreases = spurious_reliance_after < spurious_reliance_before
    target_task_preserved = target_accuracy_drop <= max_target_accuracy_drop
    ood_generalization_improves = ood_improvement >= min_ood_improvement
    random_edit_control_fails = (
        edited_ood_accuracy - random_edit_ood_accuracy >= min_random_edit_gap
    )
    return SparseFeatureEditingReport(
        target_feature_ids=tuple(int(index) for index in target_ids.tolist()),
        spurious_feature_ids=tuple(int(index) for index in spurious_ids.tolist()),
        random_feature_ids=tuple(int(index) for index in random_ids.tolist()),
        baseline_train_accuracy=baseline_train_accuracy,
        baseline_ood_accuracy=baseline_ood_accuracy,
        edited_train_accuracy=edited_train_accuracy,
        edited_ood_accuracy=edited_ood_accuracy,
        random_edit_ood_accuracy=random_edit_ood_accuracy,
        black_box_baseline_ood_accuracy=baseline_ood_accuracy,
        spurious_reliance_before=spurious_reliance_before,
        spurious_reliance_after=spurious_reliance_after,
        target_reliance_after=target_reliance_after,
        ood_improvement=ood_improvement,
        random_edit_improvement=random_edit_improvement,
        target_accuracy_drop=target_accuracy_drop,
        spurious_reliance_decreases=spurious_reliance_decreases,
        target_task_preserved=target_task_preserved,
        ood_generalization_improves=ood_generalization_improves,
        random_edit_control_fails=random_edit_control_fails,
        editing_passes=(
            spurious_reliance_decreases
            and target_task_preserved
            and ood_generalization_improves
            and random_edit_control_fails
        ),
    )


def official_sparse_feature_artifact_readiness_report(
    *,
    official_repo_files: set[str] | list[str] | tuple[str, ...],
    hf_repo_files: set[str] | list[str] | tuple[str, ...],
    local_dictionary_files: set[str] | list[str] | tuple[str, ...] = (),
    official_repo_commit: str = "",
    hf_commit_hash: str = "",
    hf_zip_size_bytes: int = 0,
) -> OfficialSparseFeatureArtifactReadinessReport:
    """Check readiness for the official Sparse Feature Circuits GT-2 path."""

    repo_files = set(official_repo_files)
    hf_files = set(hf_repo_files)
    local_files = set(local_dictionary_files)
    expected_repo_files = {
        "README.md",
        "attribution.py",
        "activation_utils.py",
        "ablation.py",
        "circuit.py",
        "circuit_plotting.py",
        "dictionary_loading_utils.py",
        "loading_utils.py",
        "data/simple_train.json",
        "data/simple_test.json",
        "data/rc_train.json",
        "data/rc_test.json",
        "annotations/pythia-70m-deduped.jsonl",
        "scripts/get_circuit.sh",
        "scripts/evaluate_circuit.sh",
    }
    dictionary_paths = expected_pythia_sfc_dictionary_paths()
    hf_zip_present = OFFICIAL_SFC_ZIP in hf_files
    missing_repo_files = tuple(sorted(expected_repo_files - repo_files))
    missing_dictionary_paths = tuple(path for path in dictionary_paths if path not in local_files)
    remote_manifest_passed = (
        not missing_repo_files
        and hf_zip_present
        and bool(official_repo_commit)
        and bool(hf_commit_hash)
        and hf_zip_size_bytes > 0
    )
    local_dictionaries_ready = not missing_dictionary_paths
    return OfficialSparseFeatureArtifactReadinessReport(
        official_repo_commit=official_repo_commit,
        hf_commit_hash=hf_commit_hash,
        hf_zip_size_gb=hf_zip_size_bytes / 1024**3,
        expected_dictionary_count=len(dictionary_paths),
        missing_dictionary_paths=missing_dictionary_paths,
        missing_repo_files=missing_repo_files,
        hf_zip_present=hf_zip_present,
        remote_manifest_passed=remote_manifest_passed,
        local_dictionaries_ready=local_dictionaries_ready,
        ready_for_gt2_replication=remote_manifest_passed and local_dictionaries_ready,
    )


def sparse_autoencoder_state_dict_smoke_report(
    state_dict: dict[str, t.Tensor],
    activation: t.Tensor,
) -> SparseAutoencoderStateDictSmokeReport:
    """Run a dependency-free smoke test for released SFC AutoEncoder weights."""

    required_keys = {"bias", "encoder.weight", "encoder.bias", "decoder.weight"}
    missing = required_keys - set(state_dict)
    if missing:
        raise ValueError(f"autoencoder state dict is missing keys: {sorted(missing)}")
    x = _require_finite_tensor("activation", activation.flatten().float())
    bias = _require_finite_tensor("bias", state_dict["bias"].float())
    encoder_weight = _require_finite_tensor(
        "encoder.weight",
        state_dict["encoder.weight"].float(),
    )
    encoder_bias = _require_finite_tensor(
        "encoder.bias",
        state_dict["encoder.bias"].float(),
    )
    decoder_weight = _require_finite_tensor(
        "decoder.weight",
        state_dict["decoder.weight"].float(),
    )
    dict_size, activation_dim = encoder_weight.shape
    shapes_match = (
        x.shape == (activation_dim,)
        and bias.shape == (activation_dim,)
        and encoder_bias.shape == (dict_size,)
        and decoder_weight.shape == (activation_dim, dict_size)
    )
    if not shapes_match:
        raise ValueError(
            "expected activation [d_model], encoder.weight [dict, d_model], "
            "encoder.bias [dict], decoder.weight [d_model, dict], and bias [d_model]."
        )
    features = t.relu((x - bias) @ encoder_weight.T + encoder_bias)
    reconstruction = features @ decoder_weight.T + bias
    error = reconstruction - x
    reconstruction_mse = error.pow(2).mean().item()
    denominator = x.norm().clamp_min(1e-12)
    relative_l2_error = (error.norm() / denominator).item()
    tensors_finite = bool(t.isfinite(features).all() and t.isfinite(reconstruction).all())
    feature_l0 = int(features.gt(0).sum().item())
    feature_max = features.max().item() if features.numel() else 0.0
    return SparseAutoencoderStateDictSmokeReport(
        activation_dim=activation_dim,
        dict_size=dict_size,
        feature_l0=feature_l0,
        feature_max=feature_max,
        reconstruction_mse=reconstruction_mse,
        relative_l2_error=relative_l2_error,
        shapes_match=shapes_match,
        tensors_finite=tensors_finite,
        passes_smoke=shapes_match and tensors_finite and feature_l0 > 0,
    )


def sparse_feature_attribution_smoke_report(
    clean_activation: t.Tensor,
    corrupt_activation: t.Tensor,
    state_dict: dict[str, t.Tensor],
    unembedding: t.Tensor,
    *,
    target_token_id: int,
    distractor_token_id: int,
    top_k: int = 16,
    random_seed: int = 0,
    min_random_margin: float = 0.05,
) -> SparseFeatureAttributionSmokeReport:
    """Attribute a real residual-stream contrast through one SAE dictionary."""

    min_random_margin = _require_finite_nonnegative(
        "min_random_margin",
        min_random_margin,
    )
    clean = _require_finite_tensor("clean_activation", clean_activation.flatten().float())
    corrupt = _require_finite_tensor("corrupt_activation", corrupt_activation.flatten().float())
    weights = _require_finite_tensor("unembedding", unembedding.float())
    if clean.shape != corrupt.shape:
        raise ValueError("clean_activation and corrupt_activation must have matching shapes.")
    if weights.ndim != 2 or weights.shape[1] != clean.numel():
        raise ValueError("unembedding must have shape (vocab, d_model).")
    if not 0 <= target_token_id < weights.shape[0]:
        raise ValueError("target token id is out of range.")
    if not 0 <= distractor_token_id < weights.shape[0]:
        raise ValueError("distractor token id is out of range.")
    if target_token_id == distractor_token_id:
        raise ValueError("target and distractor token ids must be distinct.")
    smoke = sparse_autoencoder_state_dict_smoke_report(state_dict, clean)
    if top_k <= 0 or top_k >= smoke.dict_size:
        raise ValueError("top_k must be positive and smaller than the dictionary size.")

    bias = _require_finite_tensor("bias", state_dict["bias"].float())
    encoder_weight = _require_finite_tensor(
        "encoder.weight",
        state_dict["encoder.weight"].float(),
    )
    encoder_bias = _require_finite_tensor(
        "encoder.bias",
        state_dict["encoder.bias"].float(),
    )
    decoder_weight = _require_finite_tensor(
        "decoder.weight",
        state_dict["decoder.weight"].float(),
    )
    clean_features = t.relu((clean - bias) @ encoder_weight.T + encoder_bias)
    corrupt_features = t.relu((corrupt - bias) @ encoder_weight.T + encoder_bias)
    target_direction = weights[target_token_id] - weights[distractor_token_id]
    total_residual_effect = ((clean - corrupt) * target_direction).sum().item()
    if abs(total_residual_effect) < 1e-12:
        raise ValueError("total residual effect must be nonzero.")
    feature_logit_weights = target_direction @ decoder_weight
    feature_contributions = (clean_features - corrupt_features) * feature_logit_weights
    decoded_feature_effect = feature_contributions.sum().item()
    sae_error_effect = total_residual_effect - decoded_feature_effect

    effect_sign = 1.0 if total_residual_effect > 0 else -1.0
    aligned = feature_contributions * effect_sign
    aligned_ids = aligned.gt(0).nonzero(as_tuple=False).flatten()
    if aligned_ids.numel() < top_k:
        raise ValueError("not enough aligned SAE features for the requested top_k.")
    ranked_aligned = aligned_ids[t.argsort(aligned[aligned_ids], descending=True)]
    selected_ids = ranked_aligned[:top_k]
    selected_effect = feature_contributions[selected_ids].sum().item()
    recovered_fraction = _fraction(selected_effect, total_residual_effect)
    selected_set = set(int(index) for index in selected_ids.tolist())
    candidates = t.tensor(
        [index for index in range(smoke.dict_size) if index not in selected_set],
        device=clean.device,
        dtype=t.long,
    )
    generator = t.Generator(device=clean.device)
    generator.manual_seed(random_seed)
    random_ids = candidates[
        t.randperm(candidates.numel(), generator=generator, device=clean.device)[:top_k]
    ]
    random_effect = feature_contributions[random_ids].sum().item()
    random_recovered_fraction = _fraction(random_effect, total_residual_effect)
    random_control_fails = recovered_fraction - random_recovered_fraction >= min_random_margin
    tensors_finite = bool(
        t.isfinite(clean_features).all()
        and t.isfinite(corrupt_features).all()
        and t.isfinite(feature_contributions).all()
    )
    return SparseFeatureAttributionSmokeReport(
        total_residual_effect=total_residual_effect,
        decoded_feature_effect=decoded_feature_effect,
        sae_error_effect=sae_error_effect,
        selected_feature_ids=tuple(int(index) for index in selected_ids.tolist()),
        selected_effect=selected_effect,
        recovered_fraction=recovered_fraction,
        random_feature_ids=tuple(int(index) for index in random_ids.tolist()),
        random_effect=random_effect,
        random_recovered_fraction=random_recovered_fraction,
        clean_feature_l0=int(clean_features.gt(0).sum().item()),
        corrupt_feature_l0=int(corrupt_features.gt(0).sum().item()),
        random_control_fails=random_control_fails,
        passes_smoke=tensors_finite and smoke.passes_smoke and random_control_fails,
    )


def summarize_official_sparse_feature_circuit_artifact(
    artifact: dict,
    *,
    circuit_path: str = "",
    figure_path: str = "",
    circuit_present: bool = True,
    figure_present: bool = True,
    node_threshold: float = 0.2,
    edge_threshold: float = 0.1,
    min_examples: int = 1,
    expected_node_submodule_count: int = 19,
) -> OfficialSparseFeatureCircuitArtifactReport:
    """Summarize an official feature-circuits saved graph artifact."""

    node_threshold = _require_finite_nonnegative("node_threshold", node_threshold)
    edge_threshold = _require_finite_nonnegative("edge_threshold", edge_threshold)
    if min_examples <= 0:
        raise ValueError("min_examples must be positive.")
    if expected_node_submodule_count <= 0:
        raise ValueError("expected_node_submodule_count must be positive.")
    nodes = artifact.get("nodes", {})
    edges = artifact.get("edges", {})
    examples = artifact.get("examples", [])
    node_tensors: list[t.Tensor] = []
    for value in nodes.values():
        node_tensors.extend(_flatten_effect_values(value))
    edge_tensors: list[t.Tensor] = []
    edge_group_count = 0
    for destinations in edges.values():
        if not isinstance(destinations, dict):
            continue
        for value in destinations.values():
            edge_group_count += 1
            edge_tensors.extend(_flatten_effect_values(value))
    thresholded_node_count, max_abs_node_effect = _count_thresholded(
        node_tensors,
        node_threshold,
    )
    thresholded_edge_count, max_abs_edge_effect = _count_thresholded(
        edge_tensors,
        edge_threshold,
    )
    example_count = len(examples) if isinstance(examples, list) else 0
    node_submodule_count = len(nodes) if isinstance(nodes, dict) else 0
    passes_smoke = (
        circuit_present
        and figure_present
        and example_count >= min_examples
        and node_submodule_count >= expected_node_submodule_count
        and edge_group_count > 0
        and thresholded_node_count > 0
        and thresholded_edge_count > 0
        and max_abs_node_effect > 0
        and max_abs_edge_effect > 0
    )
    return OfficialSparseFeatureCircuitArtifactReport(
        circuit_path=circuit_path,
        figure_path=figure_path,
        circuit_present=circuit_present,
        figure_present=figure_present,
        examples=example_count,
        node_submodule_count=node_submodule_count,
        edge_group_count=edge_group_count,
        thresholded_node_count=thresholded_node_count,
        thresholded_edge_count=thresholded_edge_count,
        max_abs_node_effect=max_abs_node_effect,
        max_abs_edge_effect=max_abs_edge_effect,
        passes_smoke=passes_smoke,
    )


def official_sparse_feature_circuit_artifact_report(
    circuit_path: str | Path,
    *,
    figure_path: str | Path | None = None,
    official_repo_path: str | Path | None = None,
    node_threshold: float = 0.2,
    edge_threshold: float = 0.1,
    min_examples: int = 1,
    expected_node_submodule_count: int = 19,
) -> OfficialSparseFeatureCircuitArtifactReport:
    """Load and summarize a tiny official-code Sparse Feature Circuit graph."""

    circuit = Path(circuit_path)
    figure = Path(figure_path) if figure_path is not None else Path("")
    circuit_present = circuit.exists()
    figure_present = figure.exists() if figure_path is not None else False
    if not circuit_present:
        return OfficialSparseFeatureCircuitArtifactReport(
            circuit_path=str(circuit),
            figure_path=str(figure) if figure_path is not None else "",
            circuit_present=False,
            figure_present=figure_present,
            examples=0,
            node_submodule_count=0,
            edge_group_count=0,
            thresholded_node_count=0,
            thresholded_edge_count=0,
            max_abs_node_effect=0.0,
            max_abs_edge_effect=0.0,
            passes_smoke=False,
        )
    inserted_path = None
    if official_repo_path is not None:
        inserted_path = str(Path(official_repo_path).resolve())
        if inserted_path not in sys.path:
            sys.path.insert(0, inserted_path)
    try:
        with t.sparse.check_sparse_tensor_invariants():
            artifact = t.load(circuit, map_location="cpu", weights_only=False)
    finally:
        if inserted_path is not None and sys.path and sys.path[0] == inserted_path:
            sys.path.pop(0)
    return summarize_official_sparse_feature_circuit_artifact(
        artifact,
        circuit_path=str(circuit),
        figure_path=str(figure) if figure_path is not None else "",
        circuit_present=circuit_present,
        figure_present=figure_present,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        min_examples=min_examples,
        expected_node_submodule_count=expected_node_submodule_count,
    )


def residual_feature_preflight_report(
    clean_hidden: t.Tensor,
    corrupt_hidden: t.Tensor,
    unembedding: t.Tensor,
    *,
    target_token_id: int,
    distractor_token_id: int,
    clean_logits: t.Tensor,
    corrupt_logits: t.Tensor,
    top_k: int = 16,
    random_seed: int = 0,
    min_recovered_fraction: float = 0.25,
    min_random_margin: float = 0.10,
) -> ResidualFeaturePreflightReport:
    """Use residual dimensions as exact bridge features for a real-model preflight."""

    min_recovered_fraction = _require_finite_nonnegative(
        "min_recovered_fraction",
        min_recovered_fraction,
    )
    min_random_margin = _require_finite_nonnegative(
        "min_random_margin",
        min_random_margin,
    )
    clean = _require_finite_tensor("clean_hidden", clean_hidden.flatten().float())
    corrupt = _require_finite_tensor("corrupt_hidden", corrupt_hidden.flatten().float())
    weights = _require_finite_tensor("unembedding", unembedding.float())
    if clean.shape != corrupt.shape:
        raise ValueError("clean_hidden and corrupt_hidden must have matching shape.")
    if weights.ndim != 2 or weights.shape[1] != clean.numel():
        raise ValueError("unembedding must have shape (vocab, d_model).")
    if not 0 <= target_token_id < weights.shape[0]:
        raise ValueError("target token id is out of range.")
    if not 0 <= distractor_token_id < weights.shape[0]:
        raise ValueError("distractor token id is out of range.")
    if target_token_id == distractor_token_id:
        raise ValueError("target and distractor token ids must be distinct.")
    if top_k <= 0 or top_k >= clean.numel():
        raise ValueError("top_k must be positive and smaller than d_model.")
    clean_logits = _require_finite_tensor("clean_logits", clean_logits.flatten().float())
    corrupt_logits = _require_finite_tensor("corrupt_logits", corrupt_logits.flatten().float())
    if target_token_id >= clean_logits.numel() or target_token_id >= corrupt_logits.numel():
        raise ValueError("target token id is out of range for logits.")
    if distractor_token_id >= clean_logits.numel() or distractor_token_id >= corrupt_logits.numel():
        raise ValueError("distractor token id is out of range for logits.")
    clean_logit_diff = (
        clean_logits[target_token_id] - clean_logits[distractor_token_id]
    ).float().item()
    corrupt_logit_diff = (
        corrupt_logits[target_token_id] - corrupt_logits[distractor_token_id]
    ).float().item()
    total_effect = clean_logit_diff - corrupt_logit_diff
    target_direction = weights[target_token_id] - weights[distractor_token_id]
    feature_effects = (clean - corrupt) * target_direction
    linearized_total = feature_effects.sum().item()
    linearization_error = abs(linearized_total - total_effect)
    if abs(total_effect) < 1e-12:
        raise ValueError("total effect must be nonzero.")
    effect_sign = 1.0 if total_effect > 0 else -1.0
    aligned = feature_effects * effect_sign
    positive_ids = aligned.gt(0).nonzero(as_tuple=False).flatten()
    if positive_ids.numel() < top_k:
        raise ValueError("not enough aligned residual dimensions for top_k.")
    ranked = positive_ids[t.argsort(aligned[positive_ids], descending=True)]
    selected_ids = ranked[:top_k]
    selected_effect = feature_effects[selected_ids].sum().item()
    recovered_fraction = _fraction(selected_effect, total_effect)
    selected_set = set(int(index) for index in selected_ids.tolist())
    candidates = t.tensor(
        [index for index in range(clean.numel()) if index not in selected_set],
        device=clean.device,
        dtype=t.long,
    )
    generator = t.Generator(device=clean.device)
    generator.manual_seed(random_seed)
    random_ids = candidates[
        t.randperm(candidates.numel(), generator=generator, device=clean.device)[:top_k]
    ]
    random_effect = feature_effects[random_ids].sum().item()
    random_recovered_fraction = _fraction(random_effect, total_effect)
    random_control_fails = recovered_fraction - random_recovered_fraction >= min_random_margin
    return ResidualFeaturePreflightReport(
        clean_logit_diff=clean_logit_diff,
        corrupt_logit_diff=corrupt_logit_diff,
        total_effect=total_effect,
        linearization_error=linearization_error,
        selected_feature_ids=tuple(int(index) for index in selected_ids.tolist()),
        selected_effect=selected_effect,
        recovered_fraction=recovered_fraction,
        random_feature_ids=tuple(int(index) for index in random_ids.tolist()),
        random_effect=random_effect,
        random_recovered_fraction=random_recovered_fraction,
        random_control_fails=(
            recovered_fraction >= min_recovered_fraction and random_control_fails
        ),
    )


def toy_sparse_feature_fixture(device: str | t.device = "cpu") -> dict[str, t.Tensor]:
    """Return a tiny graph with known node and edge ground truth."""

    device = t.device(device)
    return {
        "feature_contributions": t.tensor([0.7, 0.2, 0.05, 0.05], device=device),
        "edge_scores": t.tensor([[0.05, 0.8], [0.05, 0.1]], device=device),
        "exact_scores": t.tensor([0.7, 0.2, 0.05, 0.05], device=device),
        "eap_scores": t.tensor([0.5, 0.35, 0.1, 0.05], device=device),
        "eap_ig_scores": t.tensor([0.69, 0.21, 0.04, 0.06], device=device),
    }


def toy_shift_sparse_feature_editing_fixture(
    device: str | t.device = "cpu",
) -> dict[str, t.Tensor]:
    """Return a generated feature organism for SHIFT-style editing."""

    device = t.device(device)
    labels = t.tensor([1.0, 1.0, -1.0, -1.0], device=device)
    distractor = t.tensor([1.0, -1.0, 1.0, -1.0], device=device)
    train_features = t.stack(
        [labels, labels, distractor, t.zeros_like(labels)],
        dim=1,
    )
    ood_features = t.stack(
        [labels, -labels, distractor, t.zeros_like(labels)],
        dim=1,
    )
    return {
        "train_features": train_features,
        "train_labels": labels,
        "ood_features": ood_features,
        "ood_labels": labels,
        "classifier_weights": t.tensor([0.7, 1.0, 0.05, 0.0], device=device),
    }


def encode_decode_shape_smoke_test() -> dict:
    feature_acts = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    decoder = t.eye(2)
    reconstructed = feature_acts @ decoder
    return {
        "feature_shape": list(feature_acts.shape),
        "reconstructed_shape": list(reconstructed.shape),
        "matches_input": bool(t.equal(feature_acts, reconstructed)),
    }


def exact_node_patching_smoke_test() -> dict:
    fixture = toy_sparse_feature_fixture()
    return exact_feature_node_patching_report(
        fixture["feature_contributions"],
        [0, 1],
        min_recovered_fraction=0.8,
    ).__dict__


def exact_edge_patching_smoke_test() -> dict:
    fixture = toy_sparse_feature_fixture()
    return exact_feature_edge_patching_report(
        fixture["edge_scores"],
        [(0, 1)],
        min_recovered_fraction=0.75,
    ).__dict__


def eap_ig_smoke_test() -> dict:
    fixture = toy_sparse_feature_fixture()
    return eap_ig_comparison_report(
        fixture["exact_scores"],
        fixture["eap_scores"],
        fixture["eap_ig_scores"],
        max_eap_ig_error=0.02,
    ).__dict__


def threshold_graph_smoke_test() -> dict:
    fixture = toy_sparse_feature_fixture()
    return threshold_feature_graph_report(
        fixture["feature_contributions"],
        threshold=0.2,
        min_recovered_fraction=0.8,
    ).__dict__


def random_graph_control_smoke_test() -> dict:
    fixture = toy_sparse_feature_fixture()
    return random_feature_graph_control_report(
        fixture["feature_contributions"],
        target_feature_ids=[0, 1],
        random_feature_ids=[2, 3],
        min_margin=0.5,
    ).__dict__


def shift_editing_smoke_test() -> dict:
    fixture = toy_shift_sparse_feature_editing_fixture()
    return shift_style_sparse_feature_editing_report(
        fixture["train_features"],
        fixture["train_labels"],
        fixture["ood_features"],
        fixture["ood_labels"],
        fixture["classifier_weights"],
        target_feature_ids=[0],
        spurious_feature_ids=[1],
        random_feature_ids=[2],
        suppression=0.0,
        max_target_accuracy_drop=0.05,
        min_ood_improvement=0.5,
        min_random_edit_gap=0.5,
    ).__dict__


def _read_github_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ARENA-3-Extended-verification",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _local_official_repo_files(repo_dir: Path) -> set[str]:
    if not repo_dir.exists():
        return set()
    return {
        str(path.relative_to(repo_dir))
        for path in repo_dir.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }


def _local_dictionary_files(local_dictionary_dir: str | Path | None) -> tuple[str, ...]:
    if local_dictionary_dir is None:
        env_dir = os.environ.get("ARENA_SFC_DICTIONARY_DIR")
        local_dictionary_dir = (
            Path(env_dir)
            if env_dir
            else root_dir / "external" / "feature-circuits" / "dictionaries" / "pythia-70m-deduped"
        )
    base = Path(local_dictionary_dir)
    if not base.exists():
        return ()
    return tuple(
        sorted(
            str(path.relative_to(base))
            for path in base.rglob("ae.pt")
            if path.is_file()
        )
    )


def official_artifact_readiness_smoke_test(
    *,
    check_remote: bool = False,
    local_dictionary_dir: str | Path | None = None,
) -> dict:
    """Check official Sparse Feature Circuits repo and SAE artifact readiness."""

    expected_repo_files = {
        "README.md",
        "attribution.py",
        "activation_utils.py",
        "ablation.py",
        "circuit.py",
        "circuit_plotting.py",
        "dictionary_loading_utils.py",
        "loading_utils.py",
        "data/simple_train.json",
        "data/simple_test.json",
        "data/rc_train.json",
        "data/rc_test.json",
        "annotations/pythia-70m-deduped.jsonl",
        "scripts/get_circuit.sh",
        "scripts/evaluate_circuit.sh",
    }
    if check_remote:
        artifact_root = root_dir / "external" / "feature-circuits"
        cached_report_path = artifact_root / "artifact_readiness_report.json"
        if cached_report_path.exists():
            cached_report = json.loads(cached_report_path.read_text())
            official_repo_path = artifact_root / "feature-circuits"
            official_commit = str(cached_report["official_repo"]["head"])
            official_files = _local_official_repo_files(official_repo_path)
            hf_report = cached_report["huggingface"]
            hf_files = set(hf_report["files"])
            hf_commit_hash = str(hf_report["commit_hash"])
            hf_zip_size_bytes = int(hf_report["size"])
            metadata_source = "cached_prepare_sparse_feature_circuits_artifacts_report"
            cached_ready_for_gt2 = bool(cached_report["ready_for_gt2_replication"])
            official_repo_pinned = bool(cached_report["official_repo"]["pinned"])
            hf_metadata_matches_pin = bool(hf_report["metadata_matches_pin"])
        else:
            commit_data = _read_github_json(
                "https://api.github.com/repos/saprmarks/feature-circuits/commits/main"
            )
            tree_data = _read_github_json(
                "https://api.github.com/repos/saprmarks/feature-circuits/git/trees/main?recursive=1"
            )
            official_commit = commit_data["sha"]
            official_files = {
                item["path"] for item in tree_data["tree"] if item.get("type") == "blob"
            }

            from huggingface_hub import get_hf_file_metadata, hf_hub_url, list_repo_files

            hf_files = set(list_repo_files(OFFICIAL_SFC_HF_REPO, repo_type="model"))
            metadata = get_hf_file_metadata(hf_hub_url(OFFICIAL_SFC_HF_REPO, OFFICIAL_SFC_ZIP))
            hf_commit_hash = metadata.commit_hash or ""
            hf_zip_size_bytes = int(metadata.size or 0)
            metadata_source = "live_remote_manifest"
            cached_ready_for_gt2 = None
            official_repo_pinned = official_commit == OFFICIAL_FEATURE_CIRCUITS_HEAD
            hf_metadata_matches_pin = hf_commit_hash == OFFICIAL_SFC_HF_COMMIT
    else:
        official_commit = OFFICIAL_FEATURE_CIRCUITS_HEAD
        official_files = expected_repo_files
        hf_files = {OFFICIAL_SFC_ZIP}
        hf_commit_hash = OFFICIAL_SFC_HF_COMMIT
        hf_zip_size_bytes = OFFICIAL_SFC_ZIP_SIZE_BYTES
        metadata_source = "pinned_manifest_constants"
        cached_ready_for_gt2 = None
        official_repo_pinned = True
        hf_metadata_matches_pin = True

    report = official_sparse_feature_artifact_readiness_report(
        official_repo_files=official_files,
        hf_repo_files=hf_files,
        local_dictionary_files=_local_dictionary_files(local_dictionary_dir),
        official_repo_commit=official_commit,
        hf_commit_hash=hf_commit_hash,
        hf_zip_size_bytes=hf_zip_size_bytes,
    )
    data = report.__dict__
    data.update(
        {
            "official_repo": OFFICIAL_FEATURE_CIRCUITS_REPO,
            "hf_repo": OFFICIAL_SFC_HF_REPO,
            "hf_zip": OFFICIAL_SFC_ZIP,
            "dictionary_download_required": not report.local_dictionaries_ready,
            "missing_dictionary_count": len(report.missing_dictionary_paths),
            "metadata_source": metadata_source,
            "cached_ready_for_gt2_replication": cached_ready_for_gt2,
            "official_repo_pinned": official_repo_pinned,
            "hf_metadata_matches_pin": hf_metadata_matches_pin,
            "claim_scope": "artifact_manifest_readiness_not_sparse_feature_replication",
        }
    )
    return data


def _single_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"expected {text!r} to tokenize to one token, got {token_ids}.")
    return int(token_ids[0])


def pythia_subject_verb_residual_preflight(
    *,
    model_id: str = "EleutherAI/pythia-70m-deduped",
    clean_prompt: str = "The cats",
    corrupt_prompt: str = "The cat",
    target_token: str = " are",
    distractor_token: str = " is",
    top_k: int = 16,
    max_vram_gb: float = 24.0,
) -> dict:
    """Run a real Pythia-70M subject/verb residual-feature preflight.

    This deliberately remains below the Sparse Feature Circuits GT-2 claim:
    it uses residual dimensions as exact bridge features, not released SAE
    dictionaries. The point is to prove the real model/tokenizer/GPU path
    before the official sparse-feature artifact replication is attempted.
    """

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "model_id": model_id,
            "claim_scope": "Pythia preflight requires CUDA in this course environment.",
        }

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = t.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=t.float32).to(device).eval()
    target_id = _single_token_id(tokenizer, target_token)
    distractor_id = _single_token_id(tokenizer, distractor_token)

    def forward_last(text: str) -> tuple[t.Tensor, t.Tensor, list[str]]:
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with t.inference_mode():
            output = model(input_ids, output_hidden_states=True)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        hidden = output.hidden_states[-1][0, -1].detach()
        logits = output.logits[0, -1].detach()
        return hidden, logits, tokens

    clean_hidden, clean_logits, clean_tokens = forward_last(clean_prompt)
    corrupt_hidden, corrupt_logits, corrupt_tokens = forward_last(corrupt_prompt)
    unembedding = model.get_output_embeddings().weight.detach()
    report = residual_feature_preflight_report(
        clean_hidden,
        corrupt_hidden,
        unembedding,
        target_token_id=target_id,
        distractor_token_id=distractor_id,
        clean_logits=clean_logits,
        corrupt_logits=corrupt_logits,
        top_k=top_k,
        random_seed=0,
        min_recovered_fraction=0.25,
        min_random_margin=0.10,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    return {
        "cuda_available": True,
        "model_id": model_id,
        "device": t.cuda.get_device_name(0),
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "target_token": target_token,
        "target_token_id": target_id,
        "distractor_token": distractor_token,
        "distractor_token_id": distractor_id,
        "top_k": top_k,
        "clean_logit_diff": report.clean_logit_diff,
        "corrupt_logit_diff": report.corrupt_logit_diff,
        "total_effect": report.total_effect,
        "linearization_error": report.linearization_error,
        "selected_feature_ids": report.selected_feature_ids,
        "recovered_fraction": report.recovered_fraction,
        "random_feature_ids": report.random_feature_ids,
        "random_recovered_fraction": report.random_recovered_fraction,
        "random_control_fails": report.random_control_fails,
        "preflight_passed": (
            report.clean_logit_diff > report.corrupt_logit_diff
            and report.linearization_error < 1e-3
            and report.random_control_fails
        ),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "claim_scope": "real_model_residual_preflight_not_official_sae_replication",
    }


def official_sae_state_dict_smoke_test(
    *,
    model_id: str = "EleutherAI/pythia-70m-deduped",
    prompt: str = "The cats",
    dictionary_relative_path: str = "resid_out_layer5/10_32768/ae.pt",
    local_dictionary_dir: str | Path | None = None,
    max_vram_gb: float = 24.0,
) -> dict:
    """Load one official SFC SAE state dict and apply it to a real activation."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "claim_scope": "official SAE state-dict smoke requires CUDA.",
        }

    if local_dictionary_dir is None:
        env_dir = os.environ.get("ARENA_SFC_DICTIONARY_DIR")
        base = (
            Path(env_dir)
            if env_dir
            else root_dir / "external" / "feature-circuits" / "dictionaries" / "pythia-70m-deduped"
        )
    else:
        base = Path(local_dictionary_dir)
    ae_path = base / dictionary_relative_path
    if not ae_path.exists():
        return {
            "cuda_available": True,
            "model_id": model_id,
            "dictionary_path": str(ae_path),
            "state_dict_present": False,
            "passes_smoke": False,
            "claim_scope": "official_sae_state_dict_not_present",
        }

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = t.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=t.float32).to(device).eval()
    activation_cache: dict[str, t.Tensor] = {}

    def save_layer_output(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        activation_cache["resid_out_layer5"] = tensor.detach()

    handle = model.gpt_neox.layers[5].register_forward_hook(save_layer_output)
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with t.inference_mode():
            _ = model(input_ids)
    finally:
        handle.remove()

    activation = activation_cache["resid_out_layer5"][0, -1]
    state_dict = t.load(ae_path, map_location=device, weights_only=True)
    report = sparse_autoencoder_state_dict_smoke_report(state_dict, activation)
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    return {
        "cuda_available": True,
        "model_id": model_id,
        "prompt": prompt,
        "tokens": tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
        "dictionary_path": str(ae_path),
        "dictionary_relative_path": dictionary_relative_path,
        "state_dict_present": True,
        "activation_dim": report.activation_dim,
        "dict_size": report.dict_size,
        "feature_l0": report.feature_l0,
        "feature_max": report.feature_max,
        "reconstruction_mse": report.reconstruction_mse,
        "relative_l2_error": report.relative_l2_error,
        "shapes_match": report.shapes_match,
        "tensors_finite": report.tensors_finite,
        "passes_smoke": report.passes_smoke,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "claim_scope": "official_sae_state_dict_shape_and_finiteness_smoke_only",
    }


def official_sae_feature_attribution_smoke_test(
    *,
    model_id: str = "EleutherAI/pythia-70m-deduped",
    clean_prompt: str = "The cats",
    corrupt_prompt: str = "The cat",
    target_token: str = " are",
    distractor_token: str = " is",
    dictionary_relative_path: str = "resid_out_layer5/10_32768/ae.pt",
    local_dictionary_dir: str | Path | None = None,
    top_k: int = 16,
    max_vram_gb: float = 24.0,
) -> dict:
    """Attribute the subject/verb residual effect through one official SAE."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "claim_scope": "official SAE feature attribution smoke requires CUDA.",
        }

    if local_dictionary_dir is None:
        env_dir = os.environ.get("ARENA_SFC_DICTIONARY_DIR")
        base = (
            Path(env_dir)
            if env_dir
            else root_dir / "external" / "feature-circuits" / "dictionaries" / "pythia-70m-deduped"
        )
    else:
        base = Path(local_dictionary_dir)
    ae_path = base / dictionary_relative_path
    if not ae_path.exists():
        return {
            "cuda_available": True,
            "model_id": model_id,
            "dictionary_path": str(ae_path),
            "state_dict_present": False,
            "passes_smoke": False,
            "claim_scope": "official_sae_state_dict_not_present",
        }

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = t.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=t.float32).to(device).eval()
    target_id = _single_token_id(tokenizer, target_token)
    distractor_id = _single_token_id(tokenizer, distractor_token)

    def layer_activation(text: str) -> tuple[t.Tensor, list[str]]:
        cache: dict[str, t.Tensor] = {}

        def save_layer_output(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            cache["resid_out_layer5"] = tensor.detach()

        handle = model.gpt_neox.layers[5].register_forward_hook(save_layer_output)
        try:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            with t.inference_mode():
                _ = model(input_ids)
        finally:
            handle.remove()
        return cache["resid_out_layer5"][0, -1], tokenizer.convert_ids_to_tokens(
            input_ids[0].tolist()
        )

    clean_activation, clean_tokens = layer_activation(clean_prompt)
    corrupt_activation, corrupt_tokens = layer_activation(corrupt_prompt)
    state_dict = t.load(ae_path, map_location=device, weights_only=True)
    unembedding = model.get_output_embeddings().weight.detach()
    report = sparse_feature_attribution_smoke_report(
        clean_activation,
        corrupt_activation,
        state_dict,
        unembedding,
        target_token_id=target_id,
        distractor_token_id=distractor_id,
        top_k=top_k,
        random_seed=0,
        min_random_margin=0.05,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    return {
        "cuda_available": True,
        "model_id": model_id,
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "target_token": target_token,
        "target_token_id": target_id,
        "distractor_token": distractor_token,
        "distractor_token_id": distractor_id,
        "dictionary_path": str(ae_path),
        "dictionary_relative_path": dictionary_relative_path,
        "top_k": top_k,
        "total_residual_effect": report.total_residual_effect,
        "decoded_feature_effect": report.decoded_feature_effect,
        "sae_error_effect": report.sae_error_effect,
        "selected_feature_ids": report.selected_feature_ids,
        "selected_effect": report.selected_effect,
        "recovered_fraction": report.recovered_fraction,
        "random_feature_ids": report.random_feature_ids,
        "random_effect": report.random_effect,
        "random_recovered_fraction": report.random_recovered_fraction,
        "clean_feature_l0": report.clean_feature_l0,
        "corrupt_feature_l0": report.corrupt_feature_l0,
        "random_control_fails": report.random_control_fails,
        "passes_smoke": report.passes_smoke,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "claim_scope": "one_layer_official_sae_feature_attribution_smoke_not_full_graph",
    }


def official_sparse_feature_circuit_replication_test(
    *,
    circuit_path: str | Path | None = None,
    figure_path: str | Path | None = None,
    official_repo_path: str | Path | None = None,
    node_threshold: float = 0.2,
    edge_threshold: float = 0.1,
) -> dict:
    """Summarize the 100-example graph produced by the pinned official SFC code."""

    artifact_root = root_dir / "external" / "feature-circuits"
    if circuit_path is None:
        circuit_path = artifact_root / "official-runs" / "circuits" / OFFICIAL_SFC_GRAPH_CIRCUIT
    if figure_path is None:
        figure_path = artifact_root / "official-runs" / "figures" / OFFICIAL_SFC_GRAPH_FIGURE
    if official_repo_path is None:
        official_repo_path = artifact_root / "feature-circuits"

    report = official_sparse_feature_circuit_artifact_report(
        circuit_path,
        figure_path=figure_path,
        official_repo_path=official_repo_path,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        min_examples=100,
        expected_node_submodule_count=19,
    )
    data = report.__dict__
    data.update(
        {
            "official_repo": OFFICIAL_FEATURE_CIRCUITS_REPO,
            "official_repo_commit": OFFICIAL_FEATURE_CIRCUITS_HEAD,
            "node_threshold": node_threshold,
            "edge_threshold": edge_threshold,
            "claim_scope": "official_code_100_example_sparse_feature_graph_replication_artifact",
        }
    )
    return data


def _parse_official_faithfulness_output(output: str, field: str) -> float:
    pattern = rf"^{re.escape(field)}:\s*([-+0-9.eE]+)\s*$"
    for line in output.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            return float(match.group(1))
    raise ValueError(f"could not parse {field!r} from official evaluator output.")


def official_sparse_feature_circuit_faithfulness_test(
    *,
    circuit_path: str | Path | None = None,
    min_faithfulness: float = 0.9,
    max_vram_gb: float = 24.0,
) -> dict:
    """Run the official SFC held-out faithfulness evaluator on CUDA."""

    artifact_root = root_dir / "external" / "feature-circuits"
    official_repo_path = artifact_root / "feature-circuits"
    python_path = artifact_root / ".venv-official" / "bin" / "python"
    if circuit_path is None:
        circuit_path = artifact_root / "official-runs" / "circuits" / OFFICIAL_SFC_GRAPH_CIRCUIT
    circuit_path = Path(circuit_path)
    dictionary_link = official_repo_path / "dictionaries"
    dictionary_target = artifact_root / "dictionaries"
    if not dictionary_link.exists() and dictionary_target.exists():
        dictionary_link.symlink_to(dictionary_target, target_is_directory=True)

    command = [
        str(python_path),
        "ablation.py",
        "--model",
        "EleutherAI/pythia-70m-deduped",
        "--circuit",
        str(circuit_path),
        "--data",
        "simple_test",
        "--examples",
        str(OFFICIAL_SFC_HELDOUT_EXAMPLES),
        "--threshold",
        str(OFFICIAL_SFC_FAITHFULNESS_NODE_THRESHOLD),
        "--ablation",
        "mean",
        "--handle_errors",
        "default",
        "--start_layer",
        str(OFFICIAL_SFC_FAITHFULNESS_START_LAYER),
        "--device",
        "cuda:0",
    ]
    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "passes_faithfulness": False,
            "claim_scope": "official_sfc_faithfulness_requires_cuda",
        }
    if not python_path.exists() or not circuit_path.exists():
        return {
            "cuda_available": True,
            "official_env_python": str(python_path),
            "circuit_path": str(circuit_path),
            "passes_faithfulness": False,
            "claim_scope": "official_sfc_faithfulness_artifacts_missing",
        }

    result = subprocess.run(
        command,
        cwd=official_repo_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    try:
        faithfulness = _parse_official_faithfulness_output(output, "Faithfulness")
        f_model = _parse_official_faithfulness_output(output, "F(M)")
        f_circuit = _parse_official_faithfulness_output(output, "F(C)")
        f_empty = _parse_official_faithfulness_output(output, "F(∅)")
    except ValueError:
        faithfulness = f_model = f_circuit = f_empty = float("nan")
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    return {
        "cuda_available": True,
        "official_env_python": str(python_path),
        "circuit_path": str(circuit_path),
        "heldout_dataset": "simple_test",
        "heldout_examples": OFFICIAL_SFC_HELDOUT_EXAMPLES,
        "start_layer": OFFICIAL_SFC_FAITHFULNESS_START_LAYER,
        "node_threshold": OFFICIAL_SFC_FAITHFULNESS_NODE_THRESHOLD,
        "faithfulness": faithfulness,
        "f_model": f_model,
        "f_circuit": f_circuit,
        "f_empty": f_empty,
        "returncode": result.returncode,
        "passes_faithfulness": (
            result.returncode == 0
            and faithfulness >= min_faithfulness
            and peak_vram_gb <= max_vram_gb
        ),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "claim_scope": "official_code_heldout_sparse_feature_circuit_faithfulness",
    }


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "encode_decode": encode_decode_shape_smoke_test(),
        "exact_node_patching": exact_node_patching_smoke_test(),
        "exact_edge_patching": exact_edge_patching_smoke_test(),
        "eap_ig": eap_ig_smoke_test(),
        "threshold_graph": threshold_graph_smoke_test(),
        "random_graph_control": random_graph_control_smoke_test(),
        "shift_editing": shift_editing_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "full_path": "Run Pythia-70M subject-verb sparse-feature circuit replication.",
        }

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    fixture = toy_sparse_feature_fixture(device)
    node = exact_feature_node_patching_report(
        fixture["feature_contributions"],
        [0, 1],
        min_recovered_fraction=0.8,
    )
    edge = exact_feature_edge_patching_report(
        fixture["edge_scores"],
        [(0, 1)],
        min_recovered_fraction=0.75,
    )
    eap_ig = eap_ig_comparison_report(
        fixture["exact_scores"],
        fixture["eap_scores"],
        fixture["eap_ig_scores"],
        max_eap_ig_error=0.02,
    )
    random = random_feature_graph_control_report(
        fixture["feature_contributions"],
        target_feature_ids=[0, 1],
        random_feature_ids=[2, 3],
        min_margin=0.5,
    )
    shift_fixture = toy_shift_sparse_feature_editing_fixture(device)
    shift_editing = shift_style_sparse_feature_editing_report(
        shift_fixture["train_features"],
        shift_fixture["train_labels"],
        shift_fixture["ood_features"],
        shift_fixture["ood_labels"],
        shift_fixture["classifier_weights"],
        target_feature_ids=[0],
        spurious_feature_ids=[1],
        random_feature_ids=[2],
        suppression=0.0,
        max_target_accuracy_drop=0.05,
        min_ood_improvement=0.5,
        min_random_edit_gap=0.5,
    )
    pythia = pythia_subject_verb_residual_preflight(max_vram_gb=max_vram_gb)
    official = official_artifact_readiness_smoke_test(check_remote=True)
    sae_smoke = official_sae_state_dict_smoke_test(max_vram_gb=max_vram_gb)
    sae_attribution = official_sae_feature_attribution_smoke_test(max_vram_gb=max_vram_gb)
    official_graph = official_sparse_feature_circuit_replication_test()
    official_faithfulness = official_sparse_feature_circuit_faithfulness_test(
        max_vram_gb=max_vram_gb
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "node_recovered_fraction": node.recovered_fraction,
        "edge_recovered_fraction": edge.recovered_fraction,
        "eap_ig_error": eap_ig.eap_ig_error,
        "eap_ig_improves": eap_ig.eap_ig_improves,
        "random_graph_fails": random.random_graph_fails,
        "shift_editing_passed": shift_editing.editing_passes,
        "shift_editing_ood_improvement": shift_editing.ood_improvement,
        "shift_editing_target_accuracy_drop": shift_editing.target_accuracy_drop,
        "shift_editing_random_edit_control_fails": shift_editing.random_edit_control_fails,
        "shift_editing_spurious_reliance_before": shift_editing.spurious_reliance_before,
        "shift_editing_spurious_reliance_after": shift_editing.spurious_reliance_after,
        "shift_editing": shift_editing.__dict__,
        "real_model_preflight_passed": pythia["preflight_passed"],
        "real_model_total_effect": pythia["total_effect"],
        "real_model_recovered_fraction": pythia["recovered_fraction"],
        "real_model_linearization_error": pythia["linearization_error"],
        "real_model_random_control_fails": pythia["random_control_fails"],
        "real_model_peak_vram_gb": pythia["peak_vram_gb"],
        "real_model_claim_scope": pythia["claim_scope"],
        "pythia_subject_verb_preflight": pythia,
        "official_artifact_remote_manifest_passed": official["remote_manifest_passed"],
        "official_artifact_local_dictionaries_ready": official["local_dictionaries_ready"],
        "official_artifact_ready_for_gt2": official["ready_for_gt2_replication"],
        "official_artifact_expected_dictionary_count": official[
            "expected_dictionary_count"
        ],
        "official_artifact_missing_dictionary_count": official["missing_dictionary_count"],
        "official_artifact_hf_zip_size_gb": official["hf_zip_size_gb"],
        "official_artifact_readiness": official,
        "official_sae_state_dict_smoke_passed": sae_smoke["passes_smoke"],
        "official_sae_feature_l0": sae_smoke.get("feature_l0"),
        "official_sae_relative_l2_error": sae_smoke.get("relative_l2_error"),
        "official_sae_state_dict_smoke": sae_smoke,
        "official_sae_feature_attribution_passed": sae_attribution["passes_smoke"],
        "official_sae_feature_attribution_recovered_fraction": sae_attribution[
            "recovered_fraction"
        ],
        "official_sae_feature_attribution_random_fraction": sae_attribution[
            "random_recovered_fraction"
        ],
        "official_sae_feature_attribution": sae_attribution,
        "official_sparse_feature_circuit_replication_passed": official_graph["passes_smoke"],
        "official_sparse_feature_circuit_examples": official_graph["examples"],
        "official_sparse_feature_circuit_thresholded_nodes": official_graph[
            "thresholded_node_count"
        ],
        "official_sparse_feature_circuit_thresholded_edges": official_graph[
            "thresholded_edge_count"
        ],
        "official_sparse_feature_circuit": official_graph,
        "official_sparse_feature_circuit_faithfulness_passed": official_faithfulness[
            "passes_faithfulness"
        ],
        "official_sparse_feature_circuit_faithfulness": official_faithfulness[
            "faithfulness"
        ],
        "official_sparse_feature_circuit_faithfulness_report": official_faithfulness,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "full_path": (
            "Validated toy sparse-feature circuits, Pythia-70M residual-feature "
            "preflight, official SAE state-dict checks, a 100-example official "
            "sparse-feature graph artifact, held-out official faithfulness, and "
            "safe generated-data SHIFT-style sparse-feature editing."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
