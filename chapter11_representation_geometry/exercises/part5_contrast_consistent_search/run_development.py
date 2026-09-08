"""Capture and fit the frozen CCS train/calibration/development protocol, never final."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_data as data
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_methods as m
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_native as native
from chapter11_representation_geometry.exercises.part4_concept_erasure.erasure_methods import fit_binary_probe


def score_record(scores, rows, orientation):
    result = {"target": m.binary_metrics(scores, torch.tensor([row["label"] for row in rows]), orientation)}
    if rows[0]["distractor"] is not None:
        result["distractor"] = m.binary_metrics(scores, torch.tensor([row["distractor"] for row in rows]), orientation)
    return result


def fit_case(caches, layer):
    values = {split: (cache["plus"]["features"][layer].double(), cache["minus"]["features"][layer].double())
              for split, cache in caches.items()}
    centered = m.fit_pair_centering(*values["train"])
    transformed = {key: m.apply_pair_centering(*pair, centered) for key, pair in values.items()}
    calibration_labels = torch.tensor([row["label"] for row in caches["calibration"]["rows"]])
    records, parameters = [], {}

    def retain(name, scores, extra=None):
        orientation = m.calibrate_orientation(scores["calibration"], calibration_labels)
        record = {"name": name, "orientation": orientation,
                  "calibration": score_record(scores["calibration"], caches["calibration"]["rows"], orientation),
                  "development": score_record(scores["development"], caches["development"]["rows"], orientation)}
        if extra:
            record.update(extra)
        records.append(record)
        return orientation

    selected = None
    for mode in ("centered", "raw", "shuffled_pairs"):
        inputs = values if mode == "raw" else transformed
        plus, minus = inputs["train"]
        permutation = None
        if mode == "shuffled_pairs":
            permutation = torch.randperm(len(minus), generator=torch.Generator().manual_seed(904))
            minus = minus[permutation]
        fitted = m.fit_ccs_restarts(plus, minus)
        best = m.select_restart(fitted)
        scores = {key: m.paired_probabilities(*pair, fitted["weights"], fitted["biases"])[0]
                  for key, pair in inputs.items()}
        orientations = []
        for index, seed in enumerate(fitted["seeds"]):
            orientations.append(retain(f"CCS {mode} seed {seed}", {key: value[:, index] for key, value in scores.items()},
                                       {"training_loss": float(fitted["history"][-1, index, 0]), "selected": index == best["index"]}))
        parameters[mode] = {"fitted": fitted, "selected": best, "orientations": orientations,
                            "scores": scores, "training_permutation": permutation}
        if mode == "centered":
            selected = best

    # Keep one PCA baseline and the supervised fitting process explicitly separate.
    pca = m.fit_difference_pca(*transformed["train"])
    pca_scores = {key: ((pair[0] - pair[1] - pca["mean"]) @ pca["weight"]).sigmoid()
                  for key, pair in transformed.items()}
    pca["orientation"] = retain("Paired-difference PCA", pca_scores)
    pca["scores"] = pca_scores
    parameters["pca"] = pca
    training_labels = torch.tensor([row["label"] for row in caches["train"]["rows"]])
    supervised = fit_binary_probe(torch.cat(transformed["train"], dim=1), training_labels)
    supervised_scores = {key: (torch.cat(pair, dim=1) @ supervised["coefficient"] + supervised["intercept"]).sigmoid()
                         for key, pair in transformed.items()}
    # Supervised labels already define orientation; retain that fact explicitly.
    records.append({"name": "Supervised linear probe", "orientation": 1,
                    **{key: score_record(supervised_scores[key], caches[key]["rows"], 1)
                       for key in ("calibration", "development")}})
    parameters["supervised"] = {"probe": supervised, "scores": supervised_scores}

    random_weights, random_orientations = [], []
    random_scores = {key: [] for key in values}
    for seed in range(1000, 1032):
        weight = torch.randn(selected["weight"].shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
        weight *= selected["weight"].norm() / weight.norm()
        scores = {key: m.paired_probabilities(*pair, weight, selected["bias"])[0] for key, pair in transformed.items()}
        random_weights.append(weight)
        random_orientations.append(retain(f"Random direction {seed}", scores))
        for key in random_scores:
            random_scores[key].append(scores[key])
    parameters["random"] = {"weights": torch.stack(random_weights), "bias": selected["bias"],
                             "orientations": random_orientations,
                             "scores": {key: torch.stack(value, dim=1) for key, value in random_scores.items()}}
    parameters["centering"] = centered
    return records, parameters


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    data.check_data_contract()
    names = ["ccs_methods.py", "ccs_data.py", "ccs_native.py", "run_development.py", "native_protocol.md"]
    sources = [HERE / name for name in names]
    sources.append(ROOT / "chapter11_representation_geometry/exercises/part4_concept_erasure/erasure_methods.py")
    hashes = {}
    for path in sources:
        relative = str(path.relative_to(ROOT))
        snapshot = directory / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    start = time.perf_counter()
    state = {"status": "RUNNING", "final_evaluated": False, "source_inputs": hashes, "cases": [], "captures": []}

    def save():
        state["elapsed_seconds"] = time.perf_counter() - start
        (directory / "development.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        model, tokenizer = native.load_model()
        before = native.weight_digest(model)
        examples = data.paired_reviews("development", "distracting_reader", "customer")
        state["capture_checks"] = native.check_capture(model, tokenizer, [row["plus"] for row in examples[:3]] + [examples[-1]["minus"]])
        save()
        for variant, wrapper in itertools.product(data.VARIANTS, data.WRAPPERS):
            caches = {}
            for split in ("train", "calibration", "development"):
                rows = data.paired_reviews(split, variant, wrapper)
                caches[split] = native.capture_pairs(model, tokenizer, rows)
                path = directory / f"capture-{variant}-{wrapper}-{split}.pt"
                torch.save(caches[split], path)
                cache = caches[split]
                ids = cache["candidate_token_ids"]
                probabilities = cache["prefix"]["logits"][:, ids].double().softmax(-1)[:, 1]
                winners = cache["prefix"]["logits"].argmax(-1)
                state["captures"].append({"variant": variant, "wrapper": wrapper, "split": split, "rows": len(rows),
                                           "candidate_token_ids": ids, "candidate_accuracy": score_record(probabilities, rows, 1),
                                           "full_vocabulary_correct": int((winners == torch.tensor([ids[row["label"]] for row in rows])).sum()),
                                           "capture_file": path.name})
                save()
            for layer in native.LAYERS:
                records, parameters = fit_case(caches, layer)
                path = directory / f"fits-{variant}-{wrapper}-layer{layer}.pt"
                torch.save(parameters, path)
                state["cases"].append({"variant": variant, "wrapper": wrapper, "layer": layer, "reporters": records,
                                        "parameters_file": path.name})
                print(f"Finished {variant}, {wrapper}, layer {layer}: {len(records)} retained reporters", flush=True)
                save()
        state.update(status="PASS", weights_unchanged=native.weight_digest(model) == before,
                     model_gradients_absent=all(p.grad is None for p in model.parameters()), model_weight_sha256=before)
        if not state["weights_unchanged"] or not state["model_gradients_absent"]:
            raise ValueError("The model must remain frozen.")
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["changed_sources"] = [name for name, value in hashes.items()
                                    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        save()
    if state["status"] != "PASS":
        raise ValueError("A native development input changed during execution.")
    print(f"PASS: {len(state['cases'])} cases in {state['elapsed_seconds']:.3f}s; final not evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    run(parser.parse_args().output_directory.resolve())
