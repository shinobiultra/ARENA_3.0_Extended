"""An exact, rotated model with a predictive feature that its readout ignores.

Development source: the eventual learner will expose these short mechanisms.
No pretrained-model result or release is claimed by this experiment.
"""

from __future__ import annotations

import itertools

import torch as t

from . import das_methods as m


def make_organism(width: int = 8, seed: int = 17) -> dict:
    if width < 3:
        raise ValueError("The organism needs two causal variables and one unused copy.")
    generator = t.Generator().manual_seed(seed)
    rotation = t.linalg.qr(t.randn(width, width, dtype=t.float64, generator=generator)).Q
    classes = t.tensor(list(itertools.product((-1, 1), repeat=2)), dtype=t.float64)
    return {"rotation": rotation, "readout": 2.0 * rotation[:, :2] @ classes.T}


def organism_pairs(organism: dict, repeats: int, seed: int) -> dict:
    """All 16 base/source variable pairs, each with fresh independent nuisances."""
    variables = t.tensor(list(itertools.product((0, 1), repeat=2)))
    base_variables = variables.repeat_interleave(4, 0).repeat(repeats, 1)
    source_variables = variables.repeat(4, 1).repeat(repeats, 1)
    generator = t.Generator().manual_seed(seed)
    rotation = organism["rotation"]

    def encode(values):
        latent = t.randn(len(values), rotation.shape[0], dtype=t.float64, generator=generator)
        latent[:, :2] = 2 * values - 1
        latent[:, 2] = latent[:, 0]  # Perfectly predictive; the readout ignores this column.
        return latent @ rotation.T

    return {
        "base": encode(base_variables),
        "source": encode(source_variables),
        "base_variables": base_variables,
        "source_variables": source_variables,
    }


def evaluate_alignment(organism: dict, pairs: dict, basis: t.Tensor) -> dict:
    """Keep output agreement, coordinate errors and collateral changes distinct."""
    target_variables = m.high_level_interchange(
        pairs["base_variables"], pairs["source_variables"], (0,)
    )
    targets = m.high_level_output(target_variables)
    patched = m.subspace_interchange(pairs["base"], pairs["source"], basis)
    logits = patched @ organism["readout"]
    latent = patched @ organism["rotation"]
    changed = pairs["base_variables"][:, 0] != pairs["source_variables"][:, 0]
    expected = (2 * target_variables - 1).double()
    return {
        "iia": m.interchange_accuracy(logits, targets),
        "changed_iia": m.interchange_accuracy(logits, targets, changed),
        "unchanged_iia": m.interchange_accuracy(logits, targets, ~changed),
        "loss": float(m.intervention_loss(logits, targets)),
        "side_decision_accuracy": float(
            ((latent[:, 1] > 0) == target_variables[:, 1]).double().mean()
        ),
        "routing_coordinate_mae": float((latent[:, 0] - expected[:, 0]).abs().mean()),
        "side_coordinate_mae": float((latent[:, 1] - expected[:, 1]).abs().mean()),
        "unused_copy_constraint_error": float((latent[:, 0] - latent[:, 2]).abs().mean()),
        "examples": len(targets),
        "changed_examples": int(changed.sum()),
    }
