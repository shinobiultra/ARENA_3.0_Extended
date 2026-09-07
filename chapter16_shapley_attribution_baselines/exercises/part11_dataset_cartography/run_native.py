"""Cold CPU experiment with recorded source, selection boundary and independent checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography import (
    cartography_methods as methods,
)
from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography.cartography_data import (
    load_training,
    load_evaluation,
    data_manifest,
)
from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography.cartography_study import (
    map_training_data,
    retrain_selections,
    evaluate_study,
    retraining_play,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths():
    return [
        HERE / name
        for name in (
            "protocol.md",
            "protocol_freeze.json",
            "data_manifest.json",
            "cartography_data.py",
            "cartography_methods.py",
            "cartography_methods_tests.py",
            "cartography_reference.py",
            "cartography_study.py",
            "run_native.py",
        )
    ] + [HERE.parent / "part9_influence_and_tracin/attribution_data.py"]


def validate_freeze():
    freeze = json.loads((HERE / "protocol_freeze.json").read_text())
    for name, checksum in freeze["files"].items():
        if digest(HERE / name) != checksum:
            raise ValueError(f"Frozen pre-experiment input changed: {name}")
    if data_manifest() != json.loads((HERE / "data_manifest.json").read_text()):
        raise ValueError("Loaded source data differs from the frozen row and pixel manifest.")


def reference_checks(train, mapping, implementation=methods):
    results = dict(
        separate_runs=[],
        numpy_metrics=[],
        independent_small_loop="cartography_methods_tests.test_train_ensemble",
    )
    for name, width in (("clean", 64), ("linear", 0)):
        separate = implementation.train_ensemble(
            train["x"][None], train["y"][None], [161100], width=width
        )
        batched = mapping["runs"][name]
        errors = []
        for key in ("online_logits", "end_logits", "loss"):
            actual, expected = separate[key][:, 0], batched[key][:, 0]
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            errors.append(float((actual - expected).abs().max()))
        for key in separate["parameters"]:
            actual, expected = separate["parameters"][key][0], batched["parameters"][key][0]
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            errors.append(float((actual - expected).abs().max()))
        results["separate_runs"].append(
            dict(model=name, seed=161100, maximum_absolute_error=max(errors))
        )
    for name, label_name in (
        ("clean", "y"),
        ("linear", "y"),
        ("noisy", "noisy_y"),
        ("shuffled", "shuffled_y"),
    ):
        labels = train[label_name].numpy()
        for mode in ("online", "end"):
            logits = mapping["runs"][name][f"{mode}_logits"].numpy().astype(np.float64)
            exponential = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probability = exponential / exponential.sum(axis=-1, keepdims=True)
            gold = np.take_along_axis(probability, labels[None, None, :, None], axis=-1)[..., 0]
            expected = dict(
                confidence=gold.mean(axis=0),
                variability=gold.std(axis=0, ddof=0),
                correctness=(logits.argmax(axis=-1) == labels).mean(axis=0),
            )
            errors = {}
            for key, value in expected.items():
                actual = mapping["statistics"][name][mode][key].numpy()
                np.testing.assert_allclose(actual, value, rtol=1e-6, atol=1e-6)
                errors[key] = float(np.max(np.abs(actual - value)))
            results["numpy_metrics"].append(dict(model=name, logging=mode, maximum_errors=errors))
    return results


def summarize(mapping, prepared, evaluated, play, native_directory):
    return dict(
        native_directory=str(native_directory),
        mapping_runs=20,
        fresh_refits=prepared["refits"],
        main_training_runs=140,
        separate_training_checks=2,
        play_refits=2,
        train_images=3000,
        same_source_images=823,
        separate_writer_images=1797,
        full_pool_updates=prepared["fits"]["full"]["steps"],
        main_optimizer_updates=sum(
            run["steps"] * len(run["seeds"]) for run in mapping["runs"].values()
        )
        + sum(run["steps"] * len(run["seeds"]) for run in prepared["fits"].values()),
        evaluation_used_for_selection=evaluated["evaluation_used_for_selection"],
        accuracy={
            name: {split: values["accuracy"].tolist() for split, values in results.items()}
            for name, results in evaluated["refits"].items()
        },
        cross_entropy={
            name: {split: values["cross_entropy"].tolist() for split, values in results.items()}
            for name, results in evaluated["refits"].items()
        },
        mapping_accuracy={
            name: {split: values["accuracy"].tolist() for split, values in results.items()}
            for name, results in evaluated["mapping"].items()
        },
        injected_error_precision_at_300=[
            float(mapping["diagnosis"][str(seed)]["noisy"]["precision"][2]) for seed in range(5)
        ],
        play_accuracy={
            name: value["accuracy"].tolist() for name, value in play["evaluation"].items()
        },
    )


def run(output_directory: Path, preparation_only=False):
    output_directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    inputs = {}
    for source in source_paths():
        relative = str(source.relative_to(ROOT))
        target = output_directory / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        inputs[relative] = digest(source)
    state = dict(status="RUNNING", source_inputs=inputs, native_model_retrained=False)
    started = time.perf_counter()
    native = output_directory / "native"
    try:
        validate_freeze()
        train = load_training()
        print("Training 20 mapping models and recording both logging conventions.", flush=True)
        mapping = map_training_data(train, methods, native)
        print("Maps fixed. Selecting every region/control before 120 fresh refits.", flush=True)
        prepared = retrain_selections(train, mapping, methods, native)
        boundary = dict(
            phase=prepared["phase"],
            heldout_predictions_computed=False,
            mapping_sha256=digest(native / "mapping.pt"),
            selections_sha256=digest(native / "selections_before_refits.pt"),
            prepared_sha256=digest(native / "prepared.pt"),
        )
        (output_directory / "before_evaluation.json").write_text(
            json.dumps(boundary, indent=2) + "\n"
        )
        if preparation_only:
            state.update(status="PASS", preparation_only=True, native_model_retrained=True)
        else:
            print("All 140 runs complete; opening held-out predictions now.", flush=True)
            evaluation = load_evaluation()
            evaluated = evaluate_study(mapping, prepared, evaluation, methods, native)
            play = retraining_play(train, mapping, evaluation, methods)
            torch.save(play, native / "play.pt")
            checks = reference_checks(train, mapping)
            (output_directory / "reference_checks.json").write_text(
                json.dumps(checks, indent=2) + "\n"
            )
            summary = summarize(mapping, prepared, evaluated, play, native)
            (output_directory / "native_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            state.update(status="PASS", native_model_retrained=True, main_training_runs=140)
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["changed_sources"] = [
            name for name, checksum in inputs.items() if digest(ROOT / name) != checksum
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        state["elapsed_seconds"] = time.perf_counter() - started
        state["tensor_artifacts"] = {
            str(path.relative_to(output_directory)): dict(
                sha256=digest(path), size_bytes=path.stat().st_size
            )
            for path in native.glob("*.pt")
        }
        (output_directory / "execution.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["status"] != "PASS":
        raise ValueError("Source changed during the experiment.")
    print(f"{state['status']} in {state['elapsed_seconds']:.3f} seconds", flush=True)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--preparation-only", action="store_true")
    args = parser.parse_args()
    run(args.output_directory.resolve(), args.preparation_only)
