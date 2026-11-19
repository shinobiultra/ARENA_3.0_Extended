# %%
"""Reference solutions for [8.5] Sparse Feature Circuits."""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = "chapter8_automated_circuits"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

MAIN = __name__ == "__main__"


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
class SAEReconstructionReport:
    feature_shape: tuple[int, ...]
    reconstructed_shape: tuple[int, ...]
    l0_mean: float
    density: float
    dead_feature_fraction: float
    reconstruction_mse: float
    relative_l2_error: float
    loss_recovered: float
    passes_reconstruction: bool


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


def toy_sae_encode(
    activations: t.Tensor,
    encoder_weight: t.Tensor,
    encoder_bias: t.Tensor,
) -> t.Tensor:
    """Encode activations with the ReLU SAE used by the planted circuit."""

    acts = _require_finite_tensor("activations", activations.float())
    weight = _require_finite_tensor("encoder_weight", encoder_weight.float())
    bias = _require_finite_tensor("encoder_bias", encoder_bias.float())
    if weight.ndim != 2:
        raise ValueError("encoder_weight must have shape [features, activation].")
    if bias.shape != (weight.shape[0],):
        raise ValueError("encoder_bias must have shape [features].")
    if acts.shape[-1] != weight.shape[1]:
        raise ValueError("activation dimension must match encoder_weight.")
    return t.relu(acts @ weight.T + bias)


def toy_sae_decode(
    feature_acts: t.Tensor,
    decoder_weight: t.Tensor,
    decoder_bias: t.Tensor,
) -> t.Tensor:
    """Decode SAE features back into activation space."""

    features = _require_finite_tensor("feature_acts", feature_acts.float())
    weight = _require_finite_tensor("decoder_weight", decoder_weight.float())
    bias = _require_finite_tensor("decoder_bias", decoder_bias.float())
    if weight.ndim != 2:
        raise ValueError("decoder_weight must have shape [activation, features].")
    if bias.shape != (weight.shape[0],):
        raise ValueError("decoder_bias must have shape [activation].")
    if features.shape[-1] != weight.shape[1]:
        raise ValueError("feature dimension must match decoder_weight.")
    return features @ weight.T + bias


def sae_reconstruction_report(
    activations: t.Tensor,
    feature_acts: t.Tensor,
    reconstructions: t.Tensor,
    *,
    active_threshold: float = 1e-6,
    max_relative_l2_error: float = 1e-6,
) -> SAEReconstructionReport:
    """Compute the SAE metrics students inspect before trusting feature nodes."""

    active_threshold = _require_finite_nonnegative("active_threshold", active_threshold)
    max_relative_l2_error = _require_finite_nonnegative(
        "max_relative_l2_error",
        max_relative_l2_error,
    )
    acts = _require_finite_tensor("activations", activations.float())
    features = _require_finite_tensor("feature_acts", feature_acts.float())
    recon = _require_finite_tensor("reconstructions", reconstructions.float())
    if acts.shape != recon.shape:
        raise ValueError("activations and reconstructions must have matching shapes.")
    if acts.ndim == 1:
        acts_2d = acts.unsqueeze(0)
        features_2d = features.unsqueeze(0)
        recon_2d = recon.unsqueeze(0)
    elif acts.ndim == 2:
        acts_2d = acts
        features_2d = features
        recon_2d = recon
    else:
        raise ValueError("activations must be rank-1 or rank-2.")
    if features_2d.ndim != 2 or features_2d.shape[0] != acts_2d.shape[0]:
        raise ValueError("feature_acts must be rank-2 with the same batch size.")

    error = recon_2d - acts_2d
    mse = error.pow(2).mean().item()
    relative_l2_error = (
        error.norm(dim=-1) / acts_2d.norm(dim=-1).clamp_min(1e-12)
    ).mean().item()
    zero_baseline_mse = acts_2d.pow(2).mean().clamp_min(1e-12).item()
    active = features_2d.abs().gt(active_threshold)
    l0_mean = active.sum(dim=-1).float().mean().item()
    density = active.float().mean().item()
    dead_feature_fraction = active.any(dim=0).logical_not().float().mean().item()
    loss_recovered = 1.0 - mse / zero_baseline_mse
    return SAEReconstructionReport(
        feature_shape=tuple(int(x) for x in features.shape),
        reconstructed_shape=tuple(int(x) for x in recon.shape),
        l0_mean=l0_mean,
        density=density,
        dead_feature_fraction=dead_feature_fraction,
        reconstruction_mse=mse,
        relative_l2_error=relative_l2_error,
        loss_recovered=loss_recovered,
        passes_reconstruction=relative_l2_error <= max_relative_l2_error,
    )


def planted_sparse_feature_circuit_fixture(device: str | t.device = "cpu") -> dict[str, object]:
    """Return an exact sparse-feature graph with known node and edge ground truth."""

    device = t.device(device)
    clean_activation = t.tensor([1.0, 1.0, 0.0, 0.3, 0.0, 0.0], device=device)
    corrupt_activation = t.tensor([1.0, 0.0, 1.0, 0.1, 0.0, 0.2], device=device)
    encoder_weight = t.eye(6, device=device)
    encoder_bias = t.zeros(6, device=device)
    decoder_weight = t.eye(6, device=device)
    decoder_bias = t.zeros(6, device=device)
    edge_weights = t.zeros(6, 6, device=device)
    edge_weights[1, 0] = 1.20
    edge_weights[2, 1] = 0.80
    edge_weights[0, 2] = 0.30
    edge_weights[3, 3] = 0.20
    edge_weights[5, 5] = 1.00
    readout_weights = t.tensor([1.0, -0.9, 0.0, 0.2, 0.0, -0.1], device=device)
    return {
        "clean_activation": clean_activation,
        "corrupt_activation": corrupt_activation,
        "encoder_weight": encoder_weight,
        "encoder_bias": encoder_bias,
        "decoder_weight": decoder_weight,
        "decoder_bias": decoder_bias,
        "edge_weights": edge_weights,
        "readout_weights": readout_weights,
        "feature_names": (
            "shared subject token",
            "plural subject feature",
            "singular subject feature",
            "format feature",
            "dead distractor feature",
            "SAE error feature",
        ),
        "receiver_names": (
            "plural relay",
            "singular relay",
            "shared syntax relay",
            "format readout",
            "dead receiver",
            "SAE error receiver",
        ),
        "ground_truth_node_ids": (0, 1),
        "ground_truth_edges": ((1, 0), (2, 1)),
        "same_size_random_node_ids": (3, 5),
        "same_size_random_edges": ((3, 3), (5, 5)),
        "node_threshold": 0.10,
        "edge_threshold": 0.10,
    }


def planted_receiver_activations(
    source_features: t.Tensor,
    edge_weights: t.Tensor,
) -> t.Tensor:
    """Apply the toy source-feature-to-receiver-feature edge map."""

    source = _require_finite_tensor("source_features", source_features.float())
    edges = _require_finite_tensor("edge_weights", edge_weights.float())
    if edges.ndim != 2:
        raise ValueError("edge_weights must have shape [source_features, receiver_features].")
    if source.shape[-1] != edges.shape[0]:
        raise ValueError("source feature dimension must match edge_weights.")
    return source @ edges


def planted_circuit_metric(receiver_features: t.Tensor, readout_weights: t.Tensor) -> float:
    """Return the plural-minus-singular metric for receiver features."""

    receiver = _require_finite_tensor("receiver_features", receiver_features.float())
    readout = _require_finite_tensor("readout_weights", readout_weights.float())
    if receiver.shape[-1] != readout.numel():
        raise ValueError("receiver_features and readout_weights must align.")
    return float((receiver * readout).sum(dim=-1).mean().item())


def exact_planted_node_patch_scores(
    clean_receiver_features: t.Tensor,
    corrupt_receiver_features: t.Tensor,
    readout_weights: t.Tensor,
) -> t.Tensor:
    """Return exact receiver-feature patching effects for the planted graph."""

    clean = _require_finite_tensor("clean_receiver_features", clean_receiver_features.float())
    corrupt = _require_finite_tensor(
        "corrupt_receiver_features",
        corrupt_receiver_features.float(),
    )
    readout = _require_finite_tensor("readout_weights", readout_weights.float())
    if clean.shape != corrupt.shape:
        raise ValueError("clean and corrupt receiver features must match.")
    if clean.shape[-1] != readout.numel():
        raise ValueError("receiver features and readout weights must align.")
    return (clean - corrupt) * readout


def exact_planted_edge_patch_scores(
    clean_source_features: t.Tensor,
    corrupt_source_features: t.Tensor,
    edge_weights: t.Tensor,
    readout_weights: t.Tensor,
) -> t.Tensor:
    """Return exact source-feature-to-receiver-feature edge patching effects."""

    clean = _require_finite_tensor("clean_source_features", clean_source_features.float())
    corrupt = _require_finite_tensor("corrupt_source_features", corrupt_source_features.float())
    edges = _require_finite_tensor("edge_weights", edge_weights.float())
    readout = _require_finite_tensor("readout_weights", readout_weights.float())
    if clean.shape != corrupt.shape:
        raise ValueError("clean and corrupt source features must match.")
    if edges.ndim != 2 or edges.shape[0] != clean.numel():
        raise ValueError("edge_weights must have one row per source feature.")
    if edges.shape[1] != readout.numel():
        raise ValueError("edge_weights columns must align with readout_weights.")
    return (clean - corrupt).unsqueeze(-1) * edges * readout.unsqueeze(0)


def patch_planted_nodes(
    corrupt_receiver_features: t.Tensor,
    clean_receiver_features: t.Tensor,
    readout_weights: t.Tensor,
    node_ids: t.Tensor | list[int] | tuple[int, ...],
) -> float:
    """Patch selected clean receiver-feature nodes into the corrupt graph."""

    clean = _require_finite_tensor("clean_receiver_features", clean_receiver_features.float())
    corrupt = _require_finite_tensor(
        "corrupt_receiver_features",
        corrupt_receiver_features.float(),
    )
    if clean.shape != corrupt.shape:
        raise ValueError("clean and corrupt receiver features must match.")
    ids = _validate_index_tensor(
        _index_tensor(node_ids, device=corrupt.device),
        name="receiver node",
        upper_bound=corrupt.numel(),
    )
    patched = corrupt.clone()
    patched[ids] = clean[ids]
    return planted_circuit_metric(patched, readout_weights)


def patch_planted_edges(
    corrupt_source_features: t.Tensor,
    clean_source_features: t.Tensor,
    edge_weights: t.Tensor,
    readout_weights: t.Tensor,
    selected_edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> float:
    """Patch selected clean source-to-receiver edge contributions into the corrupt graph."""

    clean = _require_finite_tensor("clean_source_features", clean_source_features.float())
    corrupt = _require_finite_tensor("corrupt_source_features", corrupt_source_features.float())
    edges = _require_finite_tensor("edge_weights", edge_weights.float())
    if not selected_edges:
        raise ValueError("at least one edge is required.")
    if clean.shape != corrupt.shape or edges.shape[0] != clean.numel():
        raise ValueError("source features and edge rows must align.")
    normalized_edges = []
    edge_contribs = corrupt.unsqueeze(-1) * edges
    for source_id, receiver_id in selected_edges:
        if not 0 <= source_id < edges.shape[0]:
            raise ValueError("source id is out of range.")
        if not 0 <= receiver_id < edges.shape[1]:
            raise ValueError("receiver id is out of range.")
        edge_contribs[source_id, receiver_id] = clean[source_id] * edges[source_id, receiver_id]
        normalized_edges.append((int(source_id), int(receiver_id)))
    if len(set(normalized_edges)) != len(normalized_edges):
        raise ValueError("selected edges must be unique.")
    receiver = edge_contribs.sum(dim=0)
    return planted_circuit_metric(receiver, readout_weights)


def nonlinear_eap_ig_edge_attribution_report(
    clean_source_features: t.Tensor,
    corrupt_source_features: t.Tensor,
    edge_weights: t.Tensor,
    readout_weights: t.Tensor,
    *,
    steps: int = 256,
) -> dict[str, object]:
    """Compare plain EAP to EAP-IG on a saturated nonlinear readout."""

    if steps <= 1:
        raise ValueError("steps must be greater than 1.")
    edge_delta = exact_planted_edge_patch_scores(
        clean_source_features,
        corrupt_source_features,
        edge_weights,
        readout_weights,
    )
    clean_receiver = planted_receiver_activations(clean_source_features, edge_weights)
    corrupt_receiver = planted_receiver_activations(corrupt_source_features, edge_weights)
    clean_linear = t.tensor(planted_circuit_metric(clean_receiver, readout_weights))
    corrupt_linear = t.tensor(planted_circuit_metric(corrupt_receiver, readout_weights))
    total_linear_delta = edge_delta.sum()
    if total_linear_delta.abs().item() < 1e-12:
        raise ValueError("total linear edge effect must be nonzero.")
    exact_total = t.tanh(clean_linear) - t.tanh(corrupt_linear)
    exact_path_scores = edge_delta * (exact_total / total_linear_delta)
    corrupt_grad = 1.0 - t.tanh(corrupt_linear).pow(2)
    eap_scores = edge_delta * corrupt_grad
    alphas = (t.arange(steps, dtype=edge_delta.dtype, device=edge_delta.device) + 0.5) / steps
    path_values = corrupt_linear + alphas * total_linear_delta
    path_grads = 1.0 - t.tanh(path_values).pow(2)
    eap_ig_scores = edge_delta * path_grads.mean()
    eap_error = (exact_path_scores - eap_scores).abs().mean().item()
    eap_ig_error = (exact_path_scores - eap_ig_scores).abs().mean().item()
    return {
        "exact_total_effect": float(exact_total.item()),
        "linear_total_effect": float(total_linear_delta.item()),
        "exact_path_scores": exact_path_scores,
        "eap_scores": eap_scores,
        "eap_ig_scores": eap_ig_scores,
        "eap_error": eap_error,
        "eap_ig_error": eap_ig_error,
        "eap_ig_improves": eap_ig_error < eap_error,
        "eap_total": float(eap_scores.sum().item()),
        "eap_ig_total": float(eap_ig_scores.sum().item()),
    }


def threshold_sparse_feature_graph(
    node_scores: t.Tensor,
    edge_scores: t.Tensor,
    *,
    node_threshold: float,
    edge_threshold: float,
    min_node_recovery: float = 0.9,
    min_edge_recovery: float = 0.9,
) -> dict[str, object]:
    """Threshold node and edge scores, then score recovered exact effect mass."""

    node_threshold = _require_finite_nonnegative("node_threshold", node_threshold)
    edge_threshold = _require_finite_nonnegative("edge_threshold", edge_threshold)
    nodes = _require_finite_tensor("node_scores", node_scores.flatten().float())
    edges = _require_finite_tensor("edge_scores", edge_scores.float())
    if edges.ndim != 2:
        raise ValueError("edge_scores must have shape [source, receiver].")
    selected_nodes = nodes.abs().ge(node_threshold).nonzero(as_tuple=False).flatten()
    selected_edge_tensor = edges.abs().ge(edge_threshold).nonzero(as_tuple=False)
    if selected_nodes.numel() == 0:
        raise ValueError("node threshold selected no features.")
    if selected_edge_tensor.numel() == 0:
        raise ValueError("edge threshold selected no edges.")
    selected_edges = tuple((int(src), int(dst)) for src, dst in selected_edge_tensor.tolist())
    node_report = exact_feature_node_patching_report(
        nodes,
        selected_nodes,
        min_recovered_fraction=min_node_recovery,
    )
    edge_report = exact_feature_edge_patching_report(
        edges,
        selected_edges,
        min_recovered_fraction=min_edge_recovery,
    )
    return {
        "selected_node_ids": node_report.selected_feature_ids,
        "selected_edges": selected_edges,
        "node_recovered_fraction": node_report.recovered_fraction,
        "edge_recovered_fraction": edge_report.recovered_fraction,
        "passes_threshold": node_report.passes_recovery and edge_report.passes_recovery,
    }


def faithfulness_minimality_completeness_report(
    feature_contributions: t.Tensor,
    selected_feature_ids: t.Tensor | list[int] | tuple[int, ...],
    random_feature_ids: t.Tensor | list[int] | tuple[int, ...],
    *,
    min_faithfulness: float = 0.9,
    min_random_margin: float = 0.5,
) -> dict[str, object]:
    """Measure graph quality and compare with a disjoint same-size random graph."""

    contributions = _require_finite_tensor(
        "feature_contributions",
        feature_contributions.flatten().float(),
    )
    selected = _validate_index_tensor(
        _index_tensor(selected_feature_ids, device=contributions.device),
        name="selected feature",
        upper_bound=contributions.numel(),
    )
    random_ids = _validate_index_tensor(
        _index_tensor(random_feature_ids, device=contributions.device),
        name="random feature",
        upper_bound=contributions.numel(),
    )
    if selected.numel() != random_ids.numel():
        raise ValueError("selected and random graphs must have the same size.")
    if set(selected.tolist()) & set(random_ids.tolist()):
        raise ValueError("random graph must be disjoint from selected features.")
    full_score = contributions.sum().item()
    selected_score = contributions[selected].sum().item()
    random_score = contributions[random_ids].sum().item()
    if abs(full_score) < 1e-12:
        raise ValueError("full feature contribution must be nonzero.")
    faithfulness = selected_score / full_score
    completeness = 1.0 - abs(full_score - selected_score) / abs(full_score)
    minimality = (contributions[selected].abs().min() / abs(full_score)).item()
    random_fraction = random_score / full_score
    random_margin = faithfulness - random_fraction
    return {
        "selected_feature_ids": tuple(int(x) for x in selected.tolist()),
        "random_feature_ids": tuple(int(x) for x in random_ids.tolist()),
        "full_score": full_score,
        "selected_score": selected_score,
        "faithfulness": faithfulness,
        "minimality": minimality,
        "completeness": completeness,
        "random_fraction": random_fraction,
        "random_margin": random_margin,
        "passes": faithfulness >= min_faithfulness and random_margin >= min_random_margin,
    }


def threshold_sweep_rows(
    feature_contributions: t.Tensor,
    thresholds: list[float] | tuple[float, ...] = (0.0, 0.02, 0.10, 0.50, 1.00),
) -> list[dict[str, float]]:
    """Return rows for faithfulness/completeness/minimality threshold curves."""

    contributions = _require_finite_tensor(
        "feature_contributions",
        feature_contributions.flatten().float(),
    )
    rows = []
    full_score = contributions.sum().item()
    if abs(full_score) < 1e-12:
        raise ValueError("full feature contribution must be nonzero.")
    for threshold in thresholds:
        threshold = _require_finite_nonnegative("threshold", threshold)
        selected = contributions.abs().ge(threshold).nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            selected_score = 0.0
            minimality = 0.0
        else:
            selected_score = contributions[selected].sum().item()
            minimality = (contributions[selected].abs().min() / abs(full_score)).item()
        faithfulness = selected_score / full_score
        completeness = 1.0 - abs(full_score - selected_score) / abs(full_score)
        rows.append(
            {
                "threshold": float(threshold),
                "num_features": float(selected.numel()),
                "faithfulness": float(faithfulness),
                "minimality": float(minimality),
                "completeness": float(completeness),
            }
        )
    return rows


def run_planted_sparse_feature_signature() -> dict[str, object]:
    """Run the exact GT-0 sparse-feature circuit signature result."""

    fixture = planted_sparse_feature_circuit_fixture()
    clean = fixture["clean_activation"]
    corrupt = fixture["corrupt_activation"]
    clean_features = toy_sae_encode(clean, fixture["encoder_weight"], fixture["encoder_bias"])
    corrupt_features = toy_sae_encode(corrupt, fixture["encoder_weight"], fixture["encoder_bias"])
    feature_batch = t.stack([clean_features, corrupt_features])
    recon_batch = toy_sae_decode(
        feature_batch,
        fixture["decoder_weight"],
        fixture["decoder_bias"],
    )
    sae_report = sae_reconstruction_report(t.stack([clean, corrupt]), feature_batch, recon_batch)
    clean_receiver = planted_receiver_activations(clean_features, fixture["edge_weights"])
    corrupt_receiver = planted_receiver_activations(corrupt_features, fixture["edge_weights"])
    node_scores = exact_planted_node_patch_scores(
        clean_receiver,
        corrupt_receiver,
        fixture["readout_weights"],
    )
    edge_scores = exact_planted_edge_patch_scores(
        clean_features,
        corrupt_features,
        fixture["edge_weights"],
        fixture["readout_weights"],
    )
    graph = threshold_sparse_feature_graph(
        node_scores,
        edge_scores,
        node_threshold=fixture["node_threshold"],
        edge_threshold=fixture["edge_threshold"],
        min_node_recovery=0.95,
        min_edge_recovery=0.95,
    )
    circuit_metrics = faithfulness_minimality_completeness_report(
        node_scores,
        graph["selected_node_ids"],
        fixture["same_size_random_node_ids"],
        min_faithfulness=0.95,
        min_random_margin=0.9,
    )
    eap_ig = nonlinear_eap_ig_edge_attribution_report(
        clean_features,
        corrupt_features,
        fixture["edge_weights"],
        fixture["readout_weights"],
        steps=256,
    )
    random_control = random_feature_graph_control_report(
        node_scores,
        target_feature_ids=graph["selected_node_ids"],
        random_feature_ids=fixture["same_size_random_node_ids"],
        min_margin=0.9,
    )
    return {
        "sae": sae_report.__dict__,
        "clean_metric": planted_circuit_metric(clean_receiver, fixture["readout_weights"]),
        "corrupt_metric": planted_circuit_metric(corrupt_receiver, fixture["readout_weights"]),
        "node_scores": node_scores.tolist(),
        "edge_scores": edge_scores.tolist(),
        "graph": graph,
        "circuit_metrics": circuit_metrics,
        "threshold_rows": threshold_sweep_rows(node_scores),
        "random_control": random_control.__dict__,
        "eap_ig": {
            key: value.tolist() if isinstance(value, t.Tensor) else value
            for key, value in eap_ig.items()
        },
        "accepted": (
            sae_report.passes_reconstruction
            and graph["passes_threshold"]
            and circuit_metrics["passes"]
            and random_control.random_graph_fails
            and eap_ig["eap_ig_improves"]
        ),
        "claim_scope": "GT-0 exact planted sparse-feature graph; released artifacts are separate evidence.",
    }


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


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "planted_signature": run_planted_sparse_feature_signature(),
        "encode_decode": encode_decode_shape_smoke_test(),
        "exact_node_patching": exact_node_patching_smoke_test(),
        "exact_edge_patching": exact_edge_patching_smoke_test(),
        "eap_ig": eap_ig_smoke_test(),
        "threshold_graph": threshold_graph_smoke_test(),
        "random_graph_control": random_graph_control_smoke_test(),
        "shift_editing": shift_editing_smoke_test(),
    }


if MAIN:
    print(run_smoke_test())
