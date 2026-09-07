"""Run the bounded exact-model development experiment; never publish a lesson."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import torch as t

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das import (
    das_methods as m,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.das_organism import (
    make_organism,
    organism_pairs,
    evaluate_alignment,
)


def run(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    t.set_num_threads(2)
    names = ["das_methods.py", "das_methods_tests.py", "das_organism.py", Path(__file__).name]
    here = Path(__file__).parent
    source_inputs = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in names}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in names:
        (snapshot / name).write_bytes((here / name).read_bytes())
    state = {"status": "RUNNING", "source_inputs": source_inputs, "pretrained_model_used": False}

    def save():
        (output / "development.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    started = time.perf_counter()
    try:
        organism = make_organism()
        initial_readout = organism["readout"].clone()
        train, heldout = organism_pairs(organism, 16, 101), organism_pairs(organism, 64, 202)
        target_variables = m.high_level_interchange(
            train["base_variables"], train["source_variables"], (0,)
        )
        target = m.high_level_output(target_variables)
        rotation = organism["rotation"]
        bases = {
            "no_op": rotation[:, :0],
            "known_causal": rotation[:, :1],
            "unused_predictive_copy": rotation[:, 2:3],
            "full_donor": rotation,
            "observational_mean_direction": (rotation[:, 0:1] + rotation[:, 2:3]) / 2**0.5,
        }
        histories = {}
        for seed in (0, 1, 2):
            for kind in ("DAS", "shuffled_counterfactuals"):
                targets = target
                if kind != "DAS":
                    permutation = t.randperm(
                        len(target), generator=t.Generator().manual_seed(400 + seed)
                    )
                    targets = target[permutation]
                fitted = m.fit_alignment(
                    train["base"],
                    train["source"],
                    targets,
                    lambda h: h @ organism["readout"],
                    rank=1,
                    steps=300,
                    learning_rate=0.03,
                    seed=seed,
                )
                key = f"{kind}/seed{seed}"
                bases[key], histories[key] = fitted["basis"], fitted["history"]
        for draw in range(32):
            bases[f"random_rank1/draw{draw}"] = m.orthonormal_columns(
                t.randn(8, 1, dtype=t.float64, generator=t.Generator().manual_seed(600 + draw))
            )
        results = {key: evaluate_alignment(organism, heldout, q) for key, q in bases.items()}
        t.save(
            {"organism": organism, "train": train, "heldout": heldout, "bases": bases},
            output / "actual_tensors.pt",
        )
        state.update(
            status="PASS",
            results=results,
            histories=histories,
            train_pairs=len(target),
            heldout_pairs=len(heldout["base"]),
            all_seeds_retained=True,
            control_draws=32,
            final_low_level_weights_unchanged=t.equal(initial_readout, organism["readout"]),
        )
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        state["changed_sources"] = [
            name
            for name, digest in source_inputs.items()
            if hashlib.sha256((here / name).read_bytes()).hexdigest() != digest
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        save()
    for name, result in results.items():
        if not name.startswith("random"):
            print(
                name,
                {
                    k: round(v, 5)
                    for k, v in result.items()
                    if k
                    in ("iia", "changed_iia", "routing_coordinate_mae", "side_decision_accuracy")
                },
            )
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    run(parser.parse_args().output_directory)
