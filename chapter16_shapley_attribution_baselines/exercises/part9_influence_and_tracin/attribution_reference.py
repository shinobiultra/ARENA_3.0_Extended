"""Independent NumPy/SciPy checks for actual digit refits."""
from __future__ import annotations

import numpy as np
from scipy.optimize import root
from scipy.special import expit
import torch


def scipy_logistic_optimum(x, y, weights, l2):
    """Root solving, rather than the learner's Newton/Armijo implementation."""
    features = x.detach().cpu().numpy()
    labels = y.detach().cpu().numpy()
    mass = weights.detach().cpu().numpy()
    mass = mass / mass.sum()

    def gradient(theta):
        return features.T @ (mass * (expit(features @ theta) - labels)) + l2 * theta

    def hessian(theta):
        probability = expit(features @ theta)
        curvature = mass * probability * (1 - probability)
        return features.T @ (curvature[:, None] * features) + l2 * np.eye(features.shape[1])

    result = root(gradient, np.zeros(features.shape[1]), jac=hessian, tol=1e-11)
    residual = float(np.abs(gradient(result.x)).max())
    if residual > 1e-10:
        raise ValueError(f"Independent SciPy optimum is insufficiently converged: {residual}.")
    return torch.from_numpy(result.x.copy()), residual
