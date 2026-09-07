"""Fresh model recapture and independent refitting of every prescribed CCS case."""

from __future__ import annotations

import argparse
import ast
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
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_native as native
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search import ccs_study as study
from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.build_notebooks import lesson_cells, complete_learner

SOURCE_NAMES = (
    "lesson.md", "ccs_methods.py", "ccs_methods_tests.py", "reference_parity.py",
    "build_notebooks.py", "execute_learner.py", "ccs_native_methods.py", "ccs_native_methods_tests.py",
    "ccs_native.py", "ccs_data.py", "ccs_study.py", "native_protocol.md",
)
SHARED_SOURCE = ROOT / "chapter11_representation_geometry/exercises/part4_concept_erasure/erasure_methods.py"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def expected_artifacts():
    captures = {f"capture-{v}-{w}-{split}.pt" for v, w, split in itertools.product(data.VARIANTS, data.WRAPPERS, data.FRAMES)}
    fits = {f"fits-{v}-{w}-layer{layer}.pt" for v, w, layer in itertools.product(data.VARIANTS, data.WRAPPERS, native.LAYERS)}
    return captures | fits | {"native-play.pt"}


def validate_learner(directory):
    execution = json.loads((directory / "execution.json").read_text())
    if execution.get("status") != "PASS" or not execution.get("completed_from_own_answers") or execution.get("changed_sources"):
        raise ValueError("A complete, unchanged own-answer learner is required.")
    if not execution.get("native_model_executed") or execution.get("final_reporter_scores") != 98304:
        raise ValueError("The own-answer learner must execute all final native reporter conditions.")
    paths = [HERE / name for name in SOURCE_NAMES] + [SHARED_SOURCE]
    if set(execution["source_inputs"]) != {str(p.relative_to(ROOT)) for p in paths}:
        raise ValueError("The learner source-input set is incomplete or unexpected.")
    prose_changed = False
    for path in paths:
        relative = str(path.relative_to(ROOT))
        expected = execution["source_inputs"][relative]
        if digest(directory / "source_snapshot" / relative) != expected:
            raise ValueError(f"The own-answer source snapshot changed: {relative}")
        if digest(path) != expected:
            if path.name != "lesson.md":
                raise ValueError(f"Scientific source changed since own-answer execution: {relative}")
            prose_changed = True
    own = json.loads((directory / "completed_learner.ipynb").read_text())
    current = complete_learner({"cells": lesson_cells()})
    code_ast = lambda nb: [ast.dump(ast.parse("".join(c["source"])), include_attributes=False)
                           for c in nb["cells"] if c["cell_type"] == "code"]
    if code_ast(own) != code_ast(current):
        raise ValueError("Current own answers or experiment code differ from the executed learner.")
    summary_path = directory / "figures/native_summary.json"
    if digest(summary_path) != execution["native_summary_sha256"]:
        raise ValueError("The learner native summary changed.")
    artifacts = execution["native_artifacts"]
    if set(artifacts) != {"native/" + name for name in expected_artifacts()}:
        raise ValueError("A native capture, fitted case or play artifact is missing or unexpected.")
    if {p.name for p in (directory / "figures/native").glob("*.pt")} != expected_artifacts():
        raise ValueError("Native files do not match the complete artifact set.")
    for relative, record in artifacts.items():
        path = directory / "figures" / relative
        if path.stat().st_size != record["size_bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"A learner tensor artifact changed: {relative}")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "PASS" or not summary.get("weights_unchanged") or not summary.get("model_gradients_absent"):
        raise ValueError("The learner native model was not confirmed frozen.")
    if summary["split_manifest"] != data.split_manifest() or summary.get("final_used_for_selection"):
        raise ValueError("The frozen data or final-selection boundary differs from the protocol.")
    if not summary.get("final_started_only_after_all_fits"):
        raise ValueError("Final evaluation must follow every fit and calibration.")
    return execution, summary, prose_changed


def compare_nested(actual, expected, *, name="root", errors=None):
    """Reject missing data, different shapes/types and nonfinite numerical evidence."""
    if errors is None:
        errors = {"max_tensor_difference": 0.0, "max_scalar_difference": 0.0, "compared_tensors": 0}
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"Tensor shape or dtype mismatch: {name}")
        errors["compared_tensors"] += 1
        if expected.is_floating_point():
            if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                raise ValueError(f"Nonfinite tensor: {name}")
            difference = float((actual - expected).abs().max()) if actual.numel() else 0.0
            errors["max_tensor_difference"] = max(errors["max_tensor_difference"], difference)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6, msg=lambda msg: name + ": " + msg)
        elif not torch.equal(actual, expected):
            raise ValueError(f"Token or discrete tensor mismatch: {name}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Missing or extra dictionary entries: {name}")
        for key in expected:
            compare_nested(actual[key], expected[key], name=f"{name}.{key}", errors=errors)
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"Sequence mismatch: {name}")
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare_nested(a, b, name=f"{name}[{index}]", errors=errors)
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not torch.isfinite(torch.tensor([actual, expected])).all():
            raise ValueError(f"Nonfinite or invalid scalar: {name}")
        difference = abs(actual - expected)
        errors["max_scalar_difference"] = max(errors["max_scalar_difference"], difference)
        if difference > 1e-7:
            raise ValueError(f"Scalar mismatch: {name}: {actual} versus {expected}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Value mismatch: {name}")
    return errors


def validate_complete_result(result):
    expected_cases = {(v, w, layer) for v, w, layer in itertools.product(data.VARIANTS, data.WRAPPERS, native.LAYERS)}
    actual_cases = [(c["variant"], c["wrapper"], c["layer"]) for c in result["cases"]]
    if len(actual_cases) != 18 or set(actual_cases) != expected_cases:
        raise ValueError("Fresh reproduction must contain exactly all 18 cases.")
    expected_captures = {(v, w, split) for v, w, split in itertools.product(data.VARIANTS, data.WRAPPERS, data.FRAMES)}
    actual_captures = [(c["variant"], c["wrapper"], c["split"]) for c in result["captures"]]
    if len(actual_captures) != 24 or set(actual_captures) != expected_captures:
        raise ValueError("Fresh reproduction must contain every split and prompt capture.")
    required = {"final_reporter_scores": 98304, "final_prefix_examples": 512, "all_prefix_examples_including_play": 3586,
                "refitted_reporters": 1152, "winning_token_mismatches": 0}
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError("A fresh count or token-identity check does not cover the complete protocol.")
    if any(not result.get(key) for key in ("weights_unchanged", "model_gradients_absent", "play_reproduced", "all_reporters_refitted")):
        raise ValueError("The fresh model, refits or play did not pass.")
    numerical = result.get("numerical_comparison", {})
    if numerical.get("compared_tensors", 0) <= 0:
        raise ValueError("Fresh comparison must contain actual tensors.")
    for key, limit in (("max_tensor_difference", 1e-5), ("max_scalar_difference", 1e-7)):
        value = numerical.get(key, float("nan"))
        if not torch.isfinite(torch.tensor(value)) or value < 0 or value > limit:
            raise ValueError("Fresh numerical comparison is missing, nonfinite or outside tolerance.")
    if result.get("changed_sources"):
        raise ValueError("Fresh scientific inputs changed during execution.")


def reproduce(learner_directory, output_directory):
    output_directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    execution, own_summary, prose_changed = validate_learner(learner_directory)
    expected_dir = learner_directory / "figures/native"
    paths = [HERE / name for name in SOURCE_NAMES] + [SHARED_SOURCE, Path(__file__).resolve()]
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        snapshot = output_directory / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        hashes[relative] = digest(path)
    started = time.perf_counter()
    result = {"status": "RUNNING", "source_inputs": hashes, "learner_directory": str(learner_directory),
              "learner_execution_sha256": digest(learner_directory / "execution.json"), "prose_only_lesson_changes": prose_changed,
              "cases": [], "captures": [], "final_reporter_scores": 0, "final_prefix_examples": 0,
              "all_prefix_examples_including_play": 0, "winning_token_mismatches": 0}
    errors = {"max_tensor_difference": 0.0, "max_scalar_difference": 0.0, "compared_tensors": 0}

    def save():
        result.update(elapsed_seconds=time.perf_counter() - started, numerical_comparison=errors)
        (output_directory / "reproduction.json").write_text(json.dumps(result, indent=2) + "\n")

    def capture(split, variant, wrapper):
        cache = native.capture_pairs(model, tokenizer, data.paired_reviews(split, variant, wrapper))
        filename = f"capture-{variant}-{wrapper}-{split}.pt"
        expected = torch.load(expected_dir / filename, weights_only=False)
        compare_nested(cache, expected, name=filename, errors=errors)
        result["winning_token_mismatches"] += int((cache["prefix"]["logits"].argmax(-1) != expected["prefix"]["logits"].argmax(-1)).sum())
        result["all_prefix_examples_including_play"] += len(cache["rows"])
        if split == "final":
            result["final_prefix_examples"] += len(cache["rows"])
        torch.save(cache, output_directory / filename)
        metrics = study.summarize_native_outputs(cache)
        own = next(c for c in own_summary["captures"] if (c["variant"], c["wrapper"], c["split"]) == (variant, wrapper, split))
        compare_nested(metrics, own["native_outputs"], name=filename + ".metrics", errors=errors)
        result["captures"].append({"variant": variant, "wrapper": wrapper, "split": split, "rows": len(cache["rows"]), "filename": filename})
        save()
        return cache

    save()
    try:
        model, tokenizer = native.load_model()
        before = native.weight_digest(model)
        check_rows = data.paired_reviews("development", "distracting_reader", "customer")
        result["capture_checks"] = native.check_capture(model, tokenizer, [r["plus"] for r in check_rows[:3]] + [check_rows[-1]["minus"]])
        for variant, wrapper in itertools.product(data.VARIANTS, data.WRAPPERS):
            caches = {split: capture(split, variant, wrapper) for split in ("train", "calibration", "development")}
            for layer in native.LAYERS:
                train = caches["train"]
                fitted = study.fit_reporters(train["plus"]["features"][layer].double(), train["minus"]["features"][layer].double(),
                                             torch.tensor([r["label"] for r in train["rows"]]))
                calibration = caches["calibration"]
                orientations = study.calibrate_reporters(fitted, calibration["plus"]["features"][layer].double(),
                    calibration["minus"]["features"][layer].double(), torch.tensor([r["label"] for r in calibration["rows"]]))
                own_case = next(c for c in own_summary["cases"] if (c["variant"], c["wrapper"], c["layer"]) == (variant, wrapper, layer))
                saved_scores = {}
                for split, cache in caches.items():
                    scores = study.reporter_scores(fitted, cache["plus"]["features"][layer].double(), cache["minus"]["features"][layer].double())
                    saved_scores[split] = scores
                    metrics = study.evaluate_reporters(fitted, scores, cache["rows"], orientations)
                    compare_nested(metrics, own_case["evaluations"][split], name=f"{variant}.{wrapper}.{layer}.{split}.metrics", errors=errors)
                torch.save({"fitted": fitted, "orientations": orientations, "scores": saved_scores}, output_directory / own_case["filename"])
                result["cases"].append({"variant": variant, "wrapper": wrapper, "layer": layer, "filename": own_case["filename"]})
                save()
            print(f"Fresh fits complete: {variant}, {wrapper}", flush=True)
        for variant, wrapper in itertools.product(data.VARIANTS, data.WRAPPERS):
            cache = capture("final", variant, wrapper)
            for layer in native.LAYERS:
                filename = f"fits-{variant}-{wrapper}-layer{layer}.pt"
                payload = torch.load(output_directory / filename, weights_only=False)
                scores = study.reporter_scores(payload["fitted"], cache["plus"]["features"][layer].double(), cache["minus"]["features"][layer].double())
                payload["scores"]["final"] = scores
                expected = torch.load(expected_dir / filename, weights_only=False)
                compare_nested(payload, expected, name=filename, errors=errors)
                metrics = study.evaluate_reporters(payload["fitted"], scores, cache["rows"], payload["orientations"])
                own_case = next(c for c in own_summary["cases"] if (c["variant"], c["wrapper"], c["layer"]) == (variant, wrapper, layer))
                compare_nested(metrics, own_case["evaluations"]["final"], name=filename + ".final_metrics", errors=errors)
                result["final_reporter_scores"] += sum(len(value) for value in scores.values())
                torch.save(payload, output_directory / filename)
                save()
        expected_play = torch.load(expected_dir / "native-play.pt", weights_only=False)
        play_rows = []
        for opinion in (0, 1):
            prefix = data.render_prefix(own_summary["play_review"], "distracting_reader", "customer", opinion)
            play_rows.append({"prefix": prefix, "plus": prefix + " positive", "minus": prefix + " negative",
                              "review": own_summary["play_review"], "distractor": opinion, "base_id": "native-play"})
        play_cache = native.capture_pairs(model, tokenizer, play_rows)
        payload = torch.load(output_directory / "fits-distracting_reader-customer-layer8.pt", weights_only=False)
        play_scores = study.reporter_scores(payload["fitted"], play_cache["plus"]["features"][8].double(), play_cache["minus"]["features"][8].double())
        play = {"cache": play_cache, "scores": play_scores, "case": "fits-distracting_reader-customer-layer8.pt"}
        compare_nested(play, expected_play, name="native-play", errors=errors)
        result["winning_token_mismatches"] += int((play_cache["prefix"]["logits"].argmax(-1) != expected_play["cache"]["prefix"]["logits"].argmax(-1)).sum())
        result["all_prefix_examples_including_play"] += 2
        torch.save(play, output_directory / "native-play.pt")
        result.update(weights_unchanged=native.weight_digest(model) == before,
                      model_gradients_absent=all(p.grad is None for p in model.parameters()), model_weight_sha256=before,
                      all_reporters_refitted=True, refitted_reporters=18 * 64, play_reproduced=True)
        if before != own_summary["model_weight_sha256"]:
            raise ValueError("Fresh model weights do not match the learner checkpoint.")
        validate_learner(learner_directory)
        result["numerical_comparison"] = errors
        result["changed_sources"] = [name for name, checksum in hashes.items() if digest(ROOT / name) != checksum]
        validate_complete_result(result)
        result["artifact_hashes"] = {p.name: {"sha256": digest(p), "size_bytes": p.stat().st_size}
                                     for p in sorted(output_directory.glob("*.pt"))}
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=str(error))
        raise
    finally:
        result["changed_sources"] = [name for name, checksum in hashes.items() if digest(ROOT / name) != checksum]
        if result["changed_sources"]:
            result["status"] = "FAIL"
        save()
    if result["status"] != "PASS":
        raise ValueError("Fresh sources changed during execution.")
    print(f"PASS: 18 fresh cases, 1152 refits, 98304 final reporter scores, {result['elapsed_seconds']:.3f}s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.learner_directory.resolve(), args.output_directory.resolve())
