"""Bounded float64 reference for the analytic 2-D heavy-ball teaching toy.

This module is a reference, not a substitute for the notebook's student method.
A completed finite horizon does not imply asymptotic convergence. The stability
oracle describes the deterministic recurrence only; noise need not vanish.
"""
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

SIMULATOR_VERSION = 'quadratic-heavy-ball-v1'
LOSS_CAP = 1e12
StopReason = Literal['completed', 'loss_cap', 'nonfinite']


def _real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number, not a bool')
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must fit finite float64') from exc
    if not isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number


@dataclass(frozen=True)
class QuadraticConfig:
    lambda_min: float = 1.0
    lambda_max: float = 10.0
    angle_deg: float = 0.0
    x0: tuple[float, float] = (1.0, 1.0)
    eta: float = 0.1
    beta: float = 0.0
    noise_std: float = 0.0
    seed: int = 0
    steps: int = 50

    def __post_init__(self) -> None:
        for name in ('lambda_min', 'lambda_max', 'angle_deg', 'eta', 'beta', 'noise_std'):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        if not 0 < self.lambda_min <= self.lambda_max:
            raise ValueError('require 0 < lambda_min <= lambda_max')
        if not 0 <= self.angle_deg < 180:
            raise ValueError('angle_deg must be in [0, 180) degrees')
        if self.eta < 0 or not 0 <= self.beta < 1 or self.noise_std < 0:
            raise ValueError('require eta >= 0, beta in [0, 1), noise_std >= 0')
        try:
            coordinates = tuple(self.x0)
        except TypeError as exc:
            raise ValueError('x0 must contain exactly two finite coordinates') from exc
        if len(coordinates) != 2:
            raise ValueError('x0 must contain exactly two finite coordinates')
        object.__setattr__(self, 'x0', tuple(_real(x, 'x0 coordinate') for x in coordinates))
        for name, lower, upper in (('seed', 0, 2**32 - 1), ('steps', 1, 1000)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lower <= value <= upper:
                raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class QuadraticTrace:
    """Initial and accepted states, never a rejected/clipped proposal.

    Arrays have lengths n+1 for n accepted updates; update_norms[0] is zero.
    Exception: if initial loss cannot be represented in float64, all arrays are
    empty (states shape (0, 2)) and stop_reason is nonfinite. An initial finite
    loss above LOSS_CAP is retained for inspection, but no update is attempted.
    """
    states: NDArray[np.float64]
    losses: NDArray[np.float64]
    update_norms: NDArray[np.float64]
    stop_reason: StopReason


def _rotation(config: QuadraticConfig) -> NDArray[np.float64]:
    angle = np.deg2rad(config.angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def hessian(config: QuadraticConfig) -> NDArray[np.float64]:
    """R diag(lambda_min, lambda_max) R.T, avoiding sum overflow.

    As with any float64 matrix, extreme condition numbers can lose the smaller
    eigenvalue on materialization. Simulation uses the supplied eigensystem.
    """
    rotation = _rotation(config)
    normalized = rotation @ np.diag([config.lambda_min / config.lambda_max, 1.0]) @ rotation.T
    # Roundoff in orthogonal products can otherwise exceed float64 max for a
    # representable isotropic Hessian at the largest supported curvature.
    return np.clip(normalized, -1.0, 1.0) * config.lambda_max


def _loss(x: NDArray[np.float64], rotation: NDArray[np.float64], eigenvalues: NDArray[np.float64]) -> float:
    # Weighted norm avoids squaring a huge coordinate before multiplying by a
    # tiny curvature. No epsilon: an exact optimum retains exactly zero loss.
    scale = float(np.max(np.abs(x)))
    if scale == 0:
        return 0.0
    # Normalize before rotation: R.T@x itself can overflow for huge coordinates
    # even when tiny eigenvalues make the final weighted norm representable.
    weighted = ((rotation.T @ (x / scale)) * (np.sqrt(eigenvalues) / sqrt(2.0))) * scale
    return float(weighted @ weighted)


def _trace(states: list, losses: list, updates: list, reason: StopReason) -> QuadraticTrace:
    arrays = (np.array(states, dtype=np.float64).reshape(-1, 2),
              np.array(losses, dtype=np.float64), np.array(updates, dtype=np.float64))
    for array in arrays:
        array.flags.writeable = False
    return QuadraticTrace(*arrays, reason)


def simulate(config: QuadraticConfig) -> QuadraticTrace:
    """v' = beta*v + H*x + noise; x' = x - eta*v' (unnormalized).

    A local generator draws independent N(0, noise_std**2) gradient coordinates.
    Proposed nonfinite state/loss/norm stops without accepting that update;
    finite proposed loss > 1e12 stops with loss_cap, also before accepting it.
    Only the first unrepresentable quantity determines the failure reason.
    """
    rotation = _rotation(config)
    eigenvalues = np.array([config.lambda_min, config.lambda_max], dtype=np.float64)
    x = np.array(config.x0, dtype=np.float64)
    velocity = np.zeros(2, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        initial_loss = _loss(x, rotation, eigenvalues)
        if not isfinite(initial_loss):
            return _trace([], [], [], 'nonfinite')
        states, losses, updates = [x.copy()], [initial_loss], [0.0]
        if initial_loss > LOSS_CAP:
            return _trace(states, losses, updates, 'loss_cap')
        rng = np.random.default_rng(config.seed)
        for _ in range(config.steps):
            # eta=0 is the exact no-movement control, including finite extreme
            # gradients. No unnecessary overflowing velocity can alter it.
            if config.eta == 0:
                states.append(x.copy())
                losses.append(initial_loss)
                updates.append(0.0)
                continue
            noise = rng.normal(0.0, config.noise_std, size=2) if config.noise_std else np.zeros(2)
            velocity_next = config.beta * velocity + rotation @ (eigenvalues * (rotation.T @ x)) + noise
            x_next = x - config.eta * velocity_next
            if not np.isfinite(velocity_next).all() or not np.isfinite(x_next).all():
                return _trace(states, losses, updates, 'nonfinite')
            loss_next = _loss(x_next, rotation, eigenvalues)
            displacement = x_next - x
            update_norm = float(np.hypot(displacement[0], displacement[1]))
            if not isfinite(loss_next) or not isfinite(update_norm):
                return _trace(states, losses, updates, 'nonfinite')
            if loss_next > LOSS_CAP:
                return _trace(states, losses, updates, 'loss_cap')
            x, velocity = x_next, velocity_next
            states.append(x.copy())
            losses.append(loss_next)
            updates.append(update_norm)
    return _trace(states, losses, updates, 'completed')


def spectral_radius(config: QuadraticConfig) -> float:
    """Largest root magnitude of r²-(1+beta-eta*lambda)r+beta.

    Independent of a simulated path and of rotation/x0. rho < 1 describes strict
    deterministic asymptotic stability; rho=1 includes nondecaying boundaries.
    Return infinity on unrepresentable eta*lambda, never a false stable value.
    """
    if config.eta == 0:
        return 1.0
    radii = []
    for curvature in (config.lambda_min, config.lambda_max):
        product = config.eta * curvature
        if not isfinite(product):
            return float('inf')
        coefficient = abs(1.0 + config.beta - product)
        half = coefficient / 2.0
        if half <= sqrt(config.beta):
            radius = sqrt(config.beta)
        else:
            # The ratio form avoids overflowing coefficient**2.
            radius = half + half * sqrt(max(0.0, 1.0 - (sqrt(config.beta) / half)**2))
        # The neutral roots are exactly magnitude one. Preserve that known
        # boundary rather than allowing root-formula rounding to label it stable.
        if config.eta >= 2.0 * (1.0 + config.beta) / curvature:
            radius = max(1.0, radius)
        radii.append(radius)
    return max(radii)
