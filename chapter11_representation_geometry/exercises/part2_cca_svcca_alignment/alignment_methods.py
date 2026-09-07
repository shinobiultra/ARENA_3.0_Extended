"""Transparent CCA/SVCCA and transport primitives, to be implemented in the lesson."""

from __future__ import annotations
import torch as t


def svd_whitener(x: t.Tensor, variance_fraction: float = 1.0, max_rank: int | None = None):
    """Training-only centered coordinates with identity covariance on retained rank."""
    if x.ndim != 2 or len(x) < 3 or not x.is_floating_point() or not t.isfinite(x).all():
        raise ValueError(
            "Use a finite floating [examples, features] matrix with at least three rows."
        )
    if not 0 < variance_fraction <= 1 or (max_rank is not None and max_rank < 1):
        raise ValueError("Keep a positive fraction of variance and a positive maximum rank.")
    mean = x.mean(0)
    centered = x - mean
    _, singular, vh = t.linalg.svd(centered, full_matrices=False)
    tolerance = max(x.shape) * t.finfo(x.dtype).eps * singular[0]
    rank = int((singular > tolerance).sum())
    if rank == 0:
        raise ValueError("A constant representation has no identifiable canonical coordinate.")
    fractions = singular[:rank].square() / singular[:rank].square().sum()
    retained = min(
        rank,
        int(
            t.searchsorted(
                fractions.cumsum(0), t.tensor(variance_fraction, device=x.device, dtype=x.dtype)
            ).item()
        )
        + 1,
    )
    if max_rank is not None:
        retained = min(retained, max_rank)
    basis = vh[:retained].T
    whitening = basis * ((len(x) - 1) ** 0.5 / singular[:retained])
    return {
        "mean": mean,
        "basis": basis,
        "whitening": whitening,
        "variance_fractions": fractions,
        "retained_rank": retained,
        "numerical_rank": rank,
    }


def fit_cca(
    x: t.Tensor,
    y: t.Tensor,
    components: int | None = None,
    variance_fraction: float = 1.0,
    max_rank: int | None = None,
):
    """Whiten each training view, then align by SVD of the cross-covariance."""
    if len(x) != len(y) or x.dtype != y.dtype or x.device != y.device:
        raise ValueError("CCA needs paired rows with matching dtype and device.")
    left = svd_whitener(x, variance_fraction, max_rank)
    right = svd_whitener(y, variance_fraction, max_rank)
    qx = (x - left["mean"]) @ left["whitening"]
    qy = (y - right["mean"]) @ right["whitening"]
    u, correlations, vh = t.linalg.svd(qx.T @ qy / (len(x) - 1), full_matrices=False)
    available = len(correlations)
    count = available if components is None else components
    if not 1 <= count <= available:
        raise ValueError("Canonical component count must fit both retained ranks.")
    x_map = left["whitening"] @ u[:, :count]
    y_map = right["whitening"] @ vh[:count].T
    canonical_y = (y - right["mean"]) @ y_map
    # Unit-variance canonical targets have this least-squares loading back to Y.
    y_loading = canonical_y.T @ (y - right["mean"]) / (len(x) - 1)
    return {
        "left": left,
        "right": right,
        "x_map": x_map,
        "y_map": y_map,
        "correlations": correlations[:count],
        "y_loading": y_loading,
        "train_rows": len(x),
    }


def paired_column_correlations(x: t.Tensor, y: t.Tensor):
    """Signed Pearson correlations; constant columns are explicitly undefined."""
    if x.ndim != 2 or x.shape != y.shape or len(x) < 3:
        raise ValueError("Use equally sized paired coordinate matrices.")
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    denominator = xc.square().sum(0).sqrt() * yc.square().sum(0).sqrt()
    return t.where(denominator > 0, (xc * yc).sum(0) / denominator, t.full_like(denominator, t.nan))


def heldout_correlations(x: t.Tensor, y: t.Tensor, fit: dict):
    """Reuse both fitted transformations; never find new test-set directions."""
    return paired_column_correlations(
        (x - fit["left"]["mean"]) @ fit["x_map"], (y - fit["right"]["mean"]) @ fit["y_map"]
    )


def fit_procrustes(x: t.Tensor, y: t.Tensor):
    """Minimize paired training squared error using translation and an orthogonal map."""
    if x.ndim != 2 or x.shape != y.shape or len(x) < 3:
        raise ValueError("Procrustes needs paired matrices of equal feature width.")
    mean_x = x.mean(0)
    mean_y = y.mean(0)
    u, _, vh = t.linalg.svd((x - mean_x).T @ (y - mean_y), full_matrices=False)
    return {"mean_x": mean_x, "mean_y": mean_y, "rotation": u @ vh}


def cca_difference_transport(delta_x: t.Tensor, fit: dict, components: int | None = None):
    """Predict a target difference through a truncated canonical regression map.

    This is a specified linear transport rule, not a causal conclusion from CCA.
    """
    count = len(fit["correlations"]) if components is None else components
    if not 0 <= count <= len(fit["correlations"]):
        raise ValueError("Transport rank must fit the learned coordinates.")
    return ((delta_x @ fit["x_map"][:, :count]) * fit["correlations"][:count]) @ fit["y_loading"][
        :count
    ]


def fit_ridge_transport(x: t.Tensor, y: t.Tensor, ridge: float = 1e-3):
    """Training-only linear prediction baseline, with an unpenalized intercept."""
    if len(x) != len(y) or ridge < 0:
        raise ValueError("Use paired training rows and a nonnegative ridge.")
    mx = x.mean(0)
    my = y.mean(0)
    xc = x - mx
    yc = y - my
    if ridge == 0:
        weights = t.linalg.lstsq(
            xc, yc, driver="gelsd" if x.device.type == "cpu" else "gels"
        ).solution
    else:
        # Dual form avoids a feature-by-feature matrix when examples are scarce.
        weights = xc.T @ t.linalg.solve(
            xc @ xc.T + ridge * t.eye(len(x), device=x.device, dtype=x.dtype), yc
        )
    return {"mean_x": mx, "mean_y": my, "weights": weights}


def distribution_kl(target_logits: t.Tensor, actual_logits: t.Tensor):
    """Per-example KL(target || actual), using all vocabulary tokens."""
    if target_logits.shape != actual_logits.shape or target_logits.ndim != 2:
        raise ValueError("Use aligned [examples, vocabulary] logits.")
    logp = target_logits.log_softmax(-1)
    logq = actual_logits.log_softmax(-1)
    return (logp.exp() * (logp - logq)).sum(-1)
