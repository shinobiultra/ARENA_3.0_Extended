"""Mutation checks for rejecting incomplete or altered reproduction payloads."""

import pytest
import torch as t

from .das_reproduction import validate_payload
from . import das_protocol as protocol


def payload():
    basis = t.eye(768, protocol.RANK)
    bases = {
        f"{kind}/seed{seed}": basis.clone()
        for kind in ("DAS", "Shuffled")
        for seed in protocol.SEEDS
    }
    bases.update({f"Random/seed{seed}": basis.clone() for seed in protocol.RANDOM_SEEDS})
    bases.update(
        {
            "Role probe": basis[:, :1],
            "Mean direction": basis[:, :1],
            "No intervention": t.empty(768, 0),
        }
    )
    logits = {
        name: t.zeros(2, 50257) for name in set(bases) | {"Whole donor", "Explicit counterfactual"}
    }
    return dict(final_pairs=[{}, {}], final_bases=bases, final_logits=logits)


def test_complete_payload():
    validate_payload(payload())


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_random",
        "extra_condition",
        "missing_row",
        "nan_logit",
        "wrong_rank",
        "nonorthogonal",
    ],
)
def test_payload_mutations_rejected(mutation):
    saved = payload()
    if mutation == "missing_random":
        del saved["final_logits"]["Random/seed731"]
    elif mutation == "extra_condition":
        saved["final_logits"]["Selected favorable seed"] = saved["final_logits"]["DAS/seed0"]
    elif mutation == "missing_row":
        saved["final_logits"]["DAS/seed0"] = saved["final_logits"]["DAS/seed0"][:1]
    elif mutation == "nan_logit":
        saved["final_logits"]["DAS/seed0"][0, 4] = t.nan
    elif mutation == "wrong_rank":
        saved["final_bases"]["DAS/seed0"] = t.eye(768, 1)
    elif mutation == "nonorthogonal":
        saved["final_bases"]["DAS/seed0"][:, 0] *= 2
    with pytest.raises((ValueError, AssertionError)):
        validate_payload(saved)
