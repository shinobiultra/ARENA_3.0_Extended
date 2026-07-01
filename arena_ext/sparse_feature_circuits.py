"""Sparse Feature Circuit toy oracles and graph controls."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch as t


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


def official_sparse_feature_artifact_readiness_report(
    *,
    official_repo_files: set[str] | list[str] | tuple[str, ...],
    hf_repo_files: set[str] | list[str] | tuple[str, ...],
    local_dictionary_files: set[str] | list[str] | tuple[str, ...] = (),
    official_repo_commit: str = "",
    hf_commit_hash: str = "",
    hf_zip_size_bytes: int = 0,
) -> OfficialSparseFeatureArtifactReadinessReport:
    """Check readiness for the official Sparse Feature Circuits GT-2 path.

    This validates manifests and local dictionary presence; it does not load
    the released autoencoders or claim replication of the paper metrics.
    """

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
    hf_zip_present = "dictionaries_pythia-70m-deduped_10.zip" in hf_files
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
    total_residual_effect_tensor = ((clean - corrupt) * target_direction).sum()
    total_residual_effect = total_residual_effect_tensor.item()
    if abs(total_residual_effect) < 1e-12:
        raise ValueError("total residual effect must be nonzero.")

    feature_logit_weights = target_direction @ decoder_weight
    feature_contributions = (clean_features - corrupt_features) * feature_logit_weights
    decoded_feature_effect_tensor = feature_contributions.sum()
    decoded_feature_effect = decoded_feature_effect_tensor.item()
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
    """Summarize an official feature-circuits saved graph artifact.

    This deliberately inspects only the saved graph object. It does not claim
    paper-level faithfulness/completeness/minimality replication.
    """

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
    """Run a safe toy SHIFT-style sparse-feature edit.

    The generated task has a target feature and a correlated-but-spurious
    feature. Editing suppresses only the spurious feature weights, then checks
    that target accuracy is preserved while OOD accuracy improves. The random
    edit is a same-size negative control.
    """

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


def residual_feature_preflight_report(
    clean_hidden: t.Tensor,
    corrupt_hidden: t.Tensor,
    unembedding: t.Tensor,
    *,
    target_token_id: int,
    distractor_token_id: int,
    clean_logits: t.Tensor | None = None,
    corrupt_logits: t.Tensor | None = None,
    top_k: int = 16,
    random_seed: int = 0,
    min_recovered_fraction: float = 0.25,
    min_random_margin: float = 0.10,
) -> ResidualFeaturePreflightReport:
    """Preflight a real-model subject/verb contrast in residual dimensions.

    This is not an SAE feature-circuit replication. It checks the real model,
    tokenizer, hidden state, and unembedding path by decomposing the final
    residual logit-difference effect into exact residual-dimension
    contributions. The selected dimensions are a bridge fixture for the later
    official sparse-feature artifact path.
    """

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
        raise ValueError("clean_hidden and corrupt_hidden must have matching shapes.")
    if weights.ndim != 2 or weights.shape[1] != clean.numel():
        raise ValueError("unembedding must have shape (vocab, d_model).")
    if not 0 <= target_token_id < weights.shape[0]:
        raise ValueError("target token id is out of range.")
    if not 0 <= distractor_token_id < weights.shape[0]:
        raise ValueError("distractor token id is out of range.")
    if target_token_id == distractor_token_id:
        raise ValueError("target and distractor token ids must be distinct.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if top_k >= clean.numel():
        raise ValueError("top_k must be smaller than the hidden dimension.")

    target_direction = weights[target_token_id] - weights[distractor_token_id]
    contributions = (clean - corrupt) * target_direction
    total_effect_tensor = contributions.sum()
    total_effect = total_effect_tensor.item()
    if abs(total_effect) < 1e-12:
        raise ValueError("total residual effect must be nonzero.")

    effect_sign = 1.0 if total_effect > 0 else -1.0
    aligned = contributions * effect_sign
    aligned_ids = aligned.gt(0).nonzero(as_tuple=False).flatten()
    if aligned_ids.numel() < top_k:
        raise ValueError("not enough aligned residual features for the requested top_k.")
    ranked_aligned = aligned_ids[t.argsort(aligned[aligned_ids], descending=True)]
    selected_ids = ranked_aligned[:top_k]

    selected_effect = contributions[selected_ids].sum().item()
    recovered_fraction = _fraction(selected_effect, total_effect)

    generator = t.Generator(device=clean.device)
    generator.manual_seed(random_seed)
    selected_set = set(int(index) for index in selected_ids.tolist())
    candidates = t.tensor(
        [index for index in range(clean.numel()) if index not in selected_set],
        device=clean.device,
        dtype=t.long,
    )
    random_positions = t.randperm(candidates.numel(), generator=generator, device=clean.device)[
        :top_k
    ]
    random_ids = candidates[random_positions]
    random_effect = contributions[random_ids].sum().item()
    random_recovered_fraction = _fraction(random_effect, total_effect)

    if clean_logits is not None and corrupt_logits is not None:
        clean_logits = _require_finite_tensor("clean_logits", clean_logits.flatten().float())
        corrupt_logits = _require_finite_tensor("corrupt_logits", corrupt_logits.flatten().float())
        if target_token_id >= clean_logits.numel() or target_token_id >= corrupt_logits.numel():
            raise ValueError("target token id is out of range for logits.")
        if distractor_token_id >= clean_logits.numel() or distractor_token_id >= corrupt_logits.numel():
            raise ValueError("distractor token id is out of range for logits.")
        clean_logit_diff = (
            clean_logits[target_token_id] - clean_logits[distractor_token_id]
        ).float()
        corrupt_logit_diff = (
            corrupt_logits[target_token_id] - corrupt_logits[distractor_token_id]
        ).float()
        logit_delta = clean_logit_diff - corrupt_logit_diff
        linearization_error = (logit_delta - total_effect_tensor).abs().item()
        clean_logit_diff_value = clean_logit_diff.item()
        corrupt_logit_diff_value = corrupt_logit_diff.item()
    else:
        clean_logit_diff_value = float("nan")
        corrupt_logit_diff_value = float("nan")
        linearization_error = 0.0

    random_control_fails = (
        recovered_fraction >= min_recovered_fraction
        and recovered_fraction - random_recovered_fraction >= min_random_margin
    )
    return ResidualFeaturePreflightReport(
        clean_logit_diff=clean_logit_diff_value,
        corrupt_logit_diff=corrupt_logit_diff_value,
        total_effect=total_effect,
        linearization_error=linearization_error,
        selected_feature_ids=tuple(int(index) for index in selected_ids.tolist()),
        selected_effect=selected_effect,
        recovered_fraction=recovered_fraction,
        random_feature_ids=tuple(int(index) for index in random_ids.tolist()),
        random_effect=random_effect,
        random_recovered_fraction=random_recovered_fraction,
        random_control_fails=random_control_fails,
    )
