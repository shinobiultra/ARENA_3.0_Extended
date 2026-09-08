"""Reload pinned data and rebuild every native learner artifact in a new process."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin import attribution_methods as m
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_data import load_study_data, data_manifest
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_study import prepare_study, evaluate_final
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_play import query_deletion_play
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_reference import scipy_logistic_optimum


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, path, counts):
    """Compare every tensor element and non-tensor value, without dropping cases."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"Tensor metadata mismatch: {path}")
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11, msg=lambda message: f"{path}: {message}")
        counts["tensors"] += 1
        counts["tensor_elements"] += expected.numel()
        if expected.is_floating_point() and expected.numel():
            counts["max_absolute_error"] = max(counts["max_absolute_error"], float((actual - expected).abs().max()))
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Dictionary keys mismatch: {path}")
        for key in expected:
            compare(actual[key], expected[key], f"{path}/{key}", counts)
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"Sequence mismatch: {path}")
        for index, value in enumerate(expected):
            compare(actual[index], value, f"{path}/{index}", counts)
    else:
        if actual != expected:
            raise ValueError(f"Scalar mismatch: {path}: {actual!r} != {expected!r}")
        counts["scalars"] += 1


def reproduce(learner_directory, output_directory):
    torch.set_num_threads(4)
    output_directory.mkdir(parents=True, exist_ok=False)
    learner = json.loads((learner_directory / "execution.json").read_text())
    if learner.get("status") != "PASS" or not learner.get("native_model_executed"):
        raise ValueError("A completed real-data own-answer learner is required.")
    inputs = dict(learner["source_inputs"])
    inputs[str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__))
    for name, expected in inputs.items():
        source = ROOT / name
        if digest(source) != expected:
            raise ValueError(f"Learner/source mismatch before reproduction: {name}")
        target = output_directory / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    original_native = Path(learner["native_directory"])
    for name, evidence in learner["tensor_artifacts"].items():
        original = learner_directory / name
        if digest(original) != evidence["sha256"] or original.stat().st_size != evidence["size_bytes"]:
            raise ValueError(f"Learner tensor changed before reproduction: {name}")
    state = {
        "status": "RUNNING", "learner_directory": str(learner_directory),
        "learner_execution_sha256": digest(learner_directory / "execution.json"),
        "source_inputs": inputs, "native_model_retrained": True,
        "reference_refits": 930, "sgd_paths": 6, "sgd_updates": 30720,
        "comparisons": {"tensors": 0, "tensor_elements": 0, "scalars": 0, "max_absolute_error": 0.},
    }
    started = time.perf_counter()
    def save():
        state["elapsed_seconds"] = time.perf_counter() - started
        (output_directory / "reproduction.json").write_text(json.dumps(state, indent=2) + "\n")
    save()
    try:
        # Reparse the pinned source ZIP. No cached fit or score is supplied.
        data = load_study_data()
        manifest = data_manifest(data)
        if manifest != json.loads((HERE / "data_manifest.json").read_text()):
            raise ValueError("Freshly loaded data differ from the frozen split manifest.")
        (output_directory / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        native = output_directory / "native"
        native.mkdir()
        fits, _ = prepare_study(data, m, native, progress=lambda message: print(message, flush=True))
        references = []
        for condition, value in fits.items():
            n = len(data["train"]["y"])
            cases = [("base", torch.ones(n, dtype=torch.float64), value["base"]["theta"][0])]
            for index in (0, 1, 2):
                weights = torch.ones(n, dtype=torch.float64)
                weights[index] = 0
                cases.append((f"loo_{index}", weights, value["loo"]["theta"][index]))
            weights = torch.ones(n, dtype=torch.float64)
            weights[data["upweight_probe_indices"][0]] = .5
            cases.append(("half_weight", weights, value["perturbations"]["theta"][1]))
            for name, weights, actual in cases:
                expected, residual = scipy_logistic_optimum(data["train"]["x"], value["train_y"], weights, .02)
                error = float((actual - expected).abs().max())
                if error >= 1e-8 or residual > 1e-10:
                    raise ValueError(f"Independent SciPy mismatch: {condition}/{name}")
                references.append({"condition": condition, "case": name, "parameter_error": error, "residual": residual})
        evaluate_final(fits, data["train"], data["final"], m, native)
        play = learner["native_summary"]["exploratory_play"]
        index = play["query_index"]
        query = {"x": data["ranking"]["x"][index:index + 1], "y": data["ranking"]["y"][index:index + 1], "ids": [data["ranking"]["ids"][index]]}
        play_result = query_deletion_play(data["train"], query, fits[play["condition"]], m, play["budget"])
        torch.save(play_result, native / "query_play.pt")
        expected_files = sorted(p.relative_to(original_native) for p in original_native.rglob("*.pt"))
        actual_files = sorted(p.relative_to(native) for p in native.rglob("*.pt"))
        if expected_files != actual_files or len(actual_files) != 15:
            raise ValueError("Fresh and learner artifact sets differ.")
        for name in actual_files:
            compare(torch.load(native / name, weights_only=True), torch.load(original_native / name, weights_only=True), str(name), state["comparisons"])
        compare(references, learner["native_summary"]["independent_scipy_checks"], "independent_scipy", state["comparisons"])
        state.update(status="PASS", independent_scipy_checks=references, exploratory_refits=33,
                     separate_refit_checks=6, data_manifest_sha256=digest(output_directory / "data_manifest.json"),
                     artifacts={str(p.relative_to(output_directory)): {"sha256": digest(p), "size_bytes": p.stat().st_size} for p in native.rglob("*.pt")})
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["changed_sources"] = [name for name, expected in inputs.items() if digest(ROOT / name) != expected]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        save()
    if state["status"] != "PASS":
        raise ValueError("Source changed during fresh reproduction.")
    print(json.dumps({key: value for key, value in state.items() if key not in ("source_inputs", "artifacts", "independent_scipy_checks")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.learner_directory.resolve(), args.output_directory.resolve())
