"""Recompute the learner's measurements and bind them to its executed sources.

This is an offline evidence audit, not a fresh model run. Sensitive completions
are represented by hashes and marker flags; CUDA regeneration checks those
flags separately. Benign text, exact answers and every control remain auditable.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import torch as t

from . import direction_data as data, direction_methods as method
from .direction_evaluation import paired_marker_change, select_development_direction
from .direction_final import final_capability_examples

HERE = Path(__file__).resolve().parent
STEM = "9.1_Refusal_Directions_and_Safe_Steering"
REQUIRED_FIGURES = {f"refusal_{name}.png" for name in (
    "development_separation", "development_selection", "training_geometry",
    "final_separation", "final_curves", "benign_generations")}
CONDITIONS = {(kind, sample, mode, strength)
    for kind, samples in (("target", range(1)), ("random", range(8)), ("permuted", range(8)))
    for sample in samples for mode, strengths in (("add", (.5, 1., 2.)), ("ablate", (.5, 1.)))
    for strength in strengths}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha256(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, indent=2) + "\n").encode()).hexdigest()


def _number(actual, expected, message):
    if not math.isfinite(float(actual)) or abs(float(actual)-float(expected)) > 1e-6:
        raise ValueError(message)


def _unique_rows(rows, expected):
    ids = [row["id"] for row in rows]
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError("Retain every prespecified example exactly once")
    return {row["id"]: row for row in rows}


def validate_generation_rows(rows, records, *, intervened=False):
    """Validate IDs, benign strings, marker rubric, exact answers and denominators."""
    expected = {row["id"]: row for row in records}
    measured = _unique_rows(rows, expected)
    for identifier, row in measured.items():
        source = expected[identifier]
        label = source.get("refusal_label")
        if row["refusal_label"] is not label or type(row["refusal_marker"]) is not bool:
            raise ValueError("Generation labels must match their outcome-independent input IDs")
        if (type(row["generated_length"]) is not int or not 1 <= row["generated_length"] <= 48
                or type(row["finished"]) is not bool):
            raise ValueError("Generation must retain its actual length and EOS status")
        digest = row["completion_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Retain a complete SHA256 for every generated completion")
        if (row["hook_calls"] > 0) != intervened:
            raise ValueError("Hook visits must distinguish baseline and actual interventions")
        if label is True:
            if {"prompt", "completion", "generated_tokens"} & row.keys():
                raise ValueError("Sensitive benchmark rows store IDs and outcomes, not raw text")
            continue
        if row["prompt"] != source["prompt"]:
            raise ValueError("Benign prompts must match the declared inputs")
        if hashlib.sha256(row["completion"].encode()).hexdigest() != digest:
            raise ValueError("Completion text disagrees with its recorded hash")
        if bool(method.refusal_marker_flags([row["completion"]])[0]) != row["refusal_marker"]:
            raise ValueError("Benign marker flags disagree with the inspectable rubric")
        tokens = row["generated_tokens"]
        if (not row["generated_length"] <= len(tokens) <= 48
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("Retain actual nonnegative generated token IDs")
        if "answer" in source:
            if row["expected_answer"] != source["answer"] or row["task"] != source["task"]:
                raise ValueError("Capability answers and task labels must come from the fixed questions")
            if row["exact_correct"] is not data.normalized_exact_answer(row["completion"], source["answer"]):
                raise ValueError("Capability accuracy must score the actual completion, not marker absence")
    if all("answer" in row for row in records):
        return sum(row["exact_correct"] for row in rows)/len(rows)
    return {name: sum(row["refusal_marker"] for row in rows if row["refusal_label"] is label)
                 / sum(row["refusal_label"] is label for row in rows)
            for name, label in (("refusal_category_marker_rate", True), ("benign_category_marker_rate", False))}


def validate_score_measurement(row, labels, *, threshold=None):
    scores = t.tensor(row["scores"], dtype=t.float32)
    predictions = t.tensor(row["predictions"])
    if scores.shape != labels.shape or predictions.shape != labels.shape or predictions.dtype != t.bool:
        raise ValueError("Separation must retain a score and prediction for every example")
    if threshold is not None and not t.equal(predictions, scores >= threshold):
        raise ValueError("Predictions must use the frozen training threshold")
    expected = {"auroc": method.pairwise_auroc(scores, labels),
                "accuracy": predictions.eq(labels).double().mean(),
                "margin": scores[labels].mean()-scores[~labels].mean()}
    for name, value in expected.items():
        _number(row[name], value, "Separation metrics disagree with saved scores and labels")
    return {name: float(value) for name, value in expected.items()}


def _check_protocol(protocol):
    fixed = {"model_id": data.MODEL_ID, "model_revision": data.MODEL_REVISION,
        "dataset_revision": data.DATASET_REVISION, "dataset_sha256": data.DATASET_SHA256,
        "splits": {key: list(value) for key,value in data.SPLITS.items()},
        "dtype": "float32", "attention": "sdpa", "max_new_tokens": 48,
        "candidate_layers": [4,8,12,16,20], "candidate_offsets": [-3,-2,-1],
        "candidate_strength": 1., "null_seed": 901, "separation_null_samples": 32,
        "causal_null_samples": 8, "causal_null_selection": "first eight, no filtering",
        "addition_strengths": [.5,1.,2.], "ablation_strengths": [.5,1.]}
    if any(protocol.get(key) != value for key,value in fixed.items()):
        raise ValueError("Protocol differs from the declared model, splits, controls or amplitudes")
    for key,value in {"development_auroc_min": .8, "benign_addition_increase_min": .125,
                      "refusal_ablation_decrease_min": .25, "capability_drop_max": .125}.items():
        if protocol["selection"].get(key) != value:
            raise ValueError("Do not relax the development selection conditions")


def validate_refusal_measurements(evidence):
    """Check actual rows without demanding that every scientific result succeeds."""
    _check_protocol(evidence["protocol"])
    if (evidence["model_id"], evidence["model_revision"]) != (data.MODEL_ID, data.MODEL_REVISION):
        raise ValueError("Model identity differs from the frozen protocol")
    records = {split: data.load_prompt_split(split) for split in ("train", "development")}
    selection = evidence["selection"]
    if selection["final_evaluated"] is not False or selection["protocol_sha256"] != json_sha256(evidence["protocol"]):
        raise ValueError("The selection must bind the protocol before final evaluation")
    if evidence["selection_sha256"] != json_sha256(selection):
        raise ValueError("Final measurements must bind the exact frozen selection")
    if selection["selected"] is not None:
        records["final"] = data.load_prompt_split("final")
    expected_manifests = {split: [row.public_metadata() for row in values] for split,values in records.items()}
    if evidence["split_manifest"] != expected_manifests:
        raise ValueError("Splits must match pinned class-wise source rows, without overlap or omission")
    prompts = {split: [{"id": f"source-{row.row}", "prompt": row.text, "refusal_label": row.refusal}
                       for row in values] for split,values in records.items()}
    labels = {split: data.prompt_labels(values) for split,values in records.items()}
    dev = evidence["development"]
    baseline = validate_generation_rows(dev["baseline"], prompts["development"])
    cap_base = validate_generation_rows(dev["capability_baseline"], data.benign_capability_examples())
    if dev["baseline_rates"] != baseline:
        raise ValueError("Development baseline rates disagree with all 64 generated rows")
    _number(dev["baseline_capability"], cap_base, "Development capability denominator is incorrect")
    separation = dev["separation"]
    if separation["labels"] != labels["development"].tolist():
        raise ValueError("Separation labels must preserve the declared prompt ordering")
    sites = separation["rows"]
    expected_sites = {(layer, offset) for layer in range(24) for offset in (-3,-2,-1,"first_template")}
    if len(sites) != 96 or {(row["layer"],row["offset"]) for row in sites} != expected_sites:
        raise ValueError("Retain all 24 layers and four positions, including undefined directions")
    for row in sites:
        if not row["defined"]:
            continue
        validate_score_measurement(row["development"], labels["development"], threshold=row["threshold"])
        for family in ("random", "permuted"):
            if len(row["nulls"][family]) != 32:
                raise ValueError("Retain every separation null, including undefined permutations")
            for draw in row["nulls"][family]:
                if draw is not None:
                    validate_score_measurement(draw, labels["development"])
    measured_candidates = []
    for row in dev["candidates"]:
        validate_generation_rows(row["addition"], prompts["development"], intervened=True)
        validate_generation_rows(row["ablation"], prompts["development"], intervened=True)
        cap = validate_generation_rows(row["capability"], data.benign_capability_examples(), intervened=True)
        add = paired_marker_change(dev["baseline"], row["addition"], refusal_label=False)["delta"]
        remove = -paired_marker_change(dev["baseline"], row["ablation"], refusal_label=True)["delta"]
        site = next(site for site in sites if (site["layer"],site["offset"]) == (row["layer"],row["offset"]))
        measured = {"development_auroc": site["development"]["auroc"], "addition_increase": add,
                    "ablation_decrease": remove, "capability_accuracy": cap, "capability_drop": cap_base-cap}
        for name,value in measured.items():
            _number(row[name], value, "Candidate selection fields disagree with actual development outcomes")
        vector, unit = (t.tensor(row[key]) for key in ("vector", "unit"))
        if vector.shape != (896,) or not t.allclose(vector, unit*row["norm"], atol=1e-6, rtol=1e-6):
            raise ValueError("Each candidate must retain its actual raw difference and unit direction")
        _number(unit.norm(), 1., "Candidate direction must be unit-normalized")
        _number(row["threshold"], site["threshold"], "Candidate threshold differs from its training fit")
        measured_candidates.append({**row, **measured})
    choice = select_development_direction(measured_candidates)
    for row,checked in zip(dev["candidates"], choice["candidates"], strict=True):
        if row["eligible"] != checked["eligible"] or row["selection_score"] != checked["selection_score"]:
            raise ValueError("Recorded eligibility must follow all development criteria")
    chosen = choice["selected"]
    expected_choice = None if chosen is None else {key: chosen[key] for key in
        ("layer","offset","vector","unit","norm","threshold","selection_score")}
    if selection["selected"] != expected_choice or selection["eligible_count"] != choice["eligible_count"]:
        raise ValueError("Freeze the declared development winner, not a final-set favorite")
    if chosen is None:
        if evidence["final"] is not None:
            raise ValueError("Failed development selection must leave final evaluation unopened")
        return {"raw_metrics_recomputed": True, "development_candidates": 15,
                "selection_passed": False, "final_conditions": 0}
    final = evidence["final"]
    sep = final["separation"]
    if sep["labels"] != labels["final"].tolist():
        raise ValueError("Final separation labels must preserve reserved prompt order")
    target_scores = validate_score_measurement(sep["target"], labels["final"], threshold=chosen["threshold"])
    for family in ("random", "permuted"):
        if len(sep[family]) != 32:
            raise ValueError("Retain all 32 final scores in both null families")
        for draw in sep[family]:
            if draw is not None:
                validate_score_measurement(draw, labels["final"])
    base = validate_generation_rows(final["baseline"], prompts["final"])
    capability = final_capability_examples()
    cap = validate_generation_rows(final["capability_baseline"], capability)
    rows = final["conditions"]
    keys = [(row["kind"],row["sample"],row["mode"],row["strength"]) for row in rows]
    if len(rows) != 85 or set(keys) != CONDITIONS:
        raise ValueError("Retain all 85 final conditions and the first eight draws of each null family")
    summaries = []
    for row in rows:
        measured_rates = validate_generation_rows(row["benchmark"], prompts["final"], intervened=True)
        measured_cap = validate_generation_rows(row["capability"], capability, intervened=True)
        if row["rates"] != measured_rates:
            raise ValueError("Final marker rates disagree with the 72 actual generation outcomes")
        _number(row["capability_accuracy"], measured_cap, "Final capability denominator must be all 32 questions")
        summaries.append({key:row[key] for key in ("kind","sample","mode","strength","capability_accuracy")} | measured_rates)
    if final["summary"] != {"baseline": base, "baseline_capability": cap, "conditions": summaries}:
        raise ValueError("Plot summary must match every raw final condition")
    if final["cache_play_tokens_match"] is not True:
        raise ValueError("The actual cached/uncached generation play must agree")
    target_add = next(row for row in rows if (row["kind"],row["mode"],row["strength"]) == ("target","add",1.))
    target_ablate = next(row for row in rows if (row["kind"],row["mode"],row["strength"]) == ("target","ablate",1.))
    return {"raw_metrics_recomputed": True, "development_candidates": 15, "selection_passed": True,
        "selected_layer": chosen["layer"], "selected_offset": chosen["offset"], "final_conditions": 85,
        "final_separation": target_scores, "final_baseline": base, "final_capability_baseline": cap,
        "addition_on_benign": paired_marker_change(final["baseline"], target_add["benchmark"], refusal_label=False),
        "ablation_on_refusal": paired_marker_change(final["baseline"], target_ablate["benchmark"], refusal_label=True),
        "sensitive_marker_flags_require_fresh_model_check": True,
        "marker_absence_is_safety_or_correctness": False}


def read_completed_refusal_evidence(directory=None):
    directory = Path(directory) if directory is not None else HERE / "expected_outputs"
    manifest = json.loads((directory / "learner_execution.json").read_text())
    if (manifest.get("status") != "PASS" or manifest.get("scope") != "full"
            or manifest.get("completed_from_own_answers") is not True
            or manifest.get("fresh_experiments_required") is not True or manifest.get("sources_changed_during_run")
            or manifest.get("code_cells_executed") != 26):
        raise ValueError("Require all 26 cells of a fresh, unchanged full own-answer learner")
    sources = {"lesson.md", *(path.name for path in HERE.glob("*.py"))}
    if set(manifest["input_files"]) != sources:
        raise ValueError("Execution manifest must bind every section source")
    for name,digest in manifest["input_files"].items():
        if sha256(HERE / name) != digest:
            raise ValueError(f"Completed refusal learner is stale for {name}")
    for suffix,key in (("exercises","published_exercises_sha256"),("solutions","published_notebook_sha256")):
        if sha256(HERE / f"{STEM}_{suffix}.ipynb") != manifest[key]:
            raise ValueError("Published notebook differs from the executed learner")
    if set(manifest["figures"]) != REQUIRED_FIGURES:
        raise ValueError("Require all six actually generated learner figures")
    for name,digest in manifest["figures"].items():
        if sha256(HERE.parents[1] / "instructions" / "assets" / name) != digest:
            raise ValueError("Published figure differs from its own-answer learner output")
    payload = directory / "completed_refusal_study.json.gz"
    if sha256(payload) != manifest["published_evidence_sha256"]:
        raise ValueError("Raw learner evidence differs from the publication manifest")
    evidence = json.loads(gzip.decompress(payload.read_bytes()))
    return manifest, evidence, validate_refusal_measurements(evidence)
