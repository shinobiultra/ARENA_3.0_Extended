"""Validate complete learner provenance and recompute its metrics from raw words.

Stored measurements are supporting evidence, never described as a fresh model
run. The full executor trains all twelve models from the learner's own answers.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch as t

from . import language_data as data, language_patching as patching, language_training as training
from .run_word_patching import prepare_counterfactual_pairs

HERE = Path(__file__).resolve().parent
STEM = "12.3_Mini_VLM_from_Scratch"
MODES = ("joint", "text-only", "image-only", "shuffled-pairs")
SEEDS = (0, 1, 2)
STAGES = ("patches", "vision", "projected", "decoder_0", "decoder_1")
REQUIRED_FIGURES = {"mini_vlm_exact_patch_warmup.png", "word_learning.png", "word_transfer.png",
                    "word_familiar_gallery.png", "word_compositional_gallery.png",
                    "word_stage_region_patching.png", "word_random_patch_spread.png"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_word_datasets():
    """Reconstruct declared final renderer conditions without reading model outputs."""
    muted = data.make_language_dataset("familiar", examples_per_combination=16)
    solid_scenes = tuple(replace(scene, style="solid") for scene in muted.scenes)
    images, masks = zip(*(data.render_language_scene(scene) for scene in solid_scenes))
    solid = replace(muted, scenes=solid_scenes, images=t.stack(images), object_masks=t.stack(masks))
    return {"familiar_solid": solid, "familiar_muted": muted,
            "compositional_muted": data.make_language_dataset("compositional", examples_per_combination=32)}


def _same_number(actual, expected):
    return actual is None if expected is None else actual is not None and abs(float(actual) - expected) <= 1e-6


def _check_generated_metrics(row, expected_tokens):
    tokens = t.tensor(row["generated_tokens"])
    if tokens.shape != expected_tokens.shape:
        raise ValueError("Every example needs a complete padded generated-word row")
    exact = (tokens == expected_tokens).all(1)
    if row["per_example_exact_match"] != exact.tolist():
        raise ValueError("Reported exact matches disagree with actual generated words and EOS")
    measured = {"exact_match": float(exact.float().mean()),
                **{task + "_exact_match": float(exact[index::3].float().mean())
                   for index, task in enumerate(training.TASKS)}}
    if any(not _same_number(row[key], value) for key, value in measured.items()):
        raise ValueError("Reported word metrics disagree with their per-example denominator")
    return measured


def _flip_counts(clean, donor, targeted):
    eligible = clean & donor
    count, flips = int(eligible.sum()), int((eligible & targeted).sum())
    return {"examples": len(clean), "eligible": count, "targeted_flips": flips,
            "conditional_flip_rate": flips / count if count else None,
            "unconditional_donor_answer_rate": float(targeted.float().mean())}


def validate_word_measurements(evidence):
    """Reject omitted seeds/controls, outcome-selected denominators or false metrics."""
    frozen = evidence["frozen_choices"]
    if (frozen["status"] != "frozen_before_final_evaluation" or frozen["training_steps"] != 600
            or frozen["model_seeds"] != list(SEEDS) or frozen["training_modes"] != list(MODES)
            or frozen["all_seeds_retained"] is not True):
        raise ValueError("The frozen protocol must retain all declared modes, seeds and training steps")
    frozen_text = json.dumps(frozen, sort_keys=True, indent=2) + "\n"
    if evidence["final_evaluation"]["selection_sha256"] != hashlib.sha256(frozen_text.encode()).hexdigest():
        raise ValueError("Final word measurements must bind the exact frozen selection")
    if evidence["fresh_training_runs"] != 12 or evidence["all_declared_seeds_retained"] is not True:
        raise ValueError("Keep all twelve fresh training runs and every declared seed")
    expected_runs = {(mode, seed) for mode in MODES for seed in SEEDS}
    runs, choices = evidence["training"], evidence["frozen_choices"]["choices"]
    if len(runs) != 12 or {(row["mode"], row["seed"]) for row in runs} != expected_runs:
        raise ValueError("Training evidence must contain every mode/seed exactly once")
    if len(choices) != 12 or {(row["mode"], row["seed"]) for row in choices} != expected_runs:
        raise ValueError("Freeze every mode/seed before final evaluation")
    development = training.make_word_task_batch(data.make_language_dataset("development", examples_per_combination=12))
    for row in runs:
        if [point["step"] for point in row["history"]] != list(range(100, 601, 100)):
            raise ValueError("Retain the complete six-checkpoint development history")
        best = max(row["history"], key=lambda point: (point["exact_match"], -point["step"]))
        choice = next(choice for choice in choices if (choice["mode"], choice["seed"]) == (row["mode"], row["seed"]))
        if row["selected_step"] != best["step"] or choice["selected_step"] != best["step"]:
            raise ValueError("Choose the earliest best development checkpoint")
        measured = _check_generated_metrics(row["development"], development["expected_tokens"])
        if not _same_number(measured["exact_match"], best["exact_match"]) or not _same_number(choice["development_exact"], best["exact_match"]):
            raise ValueError("Frozen checkpoint and recorded development results disagree")
    final = evidence["final_evaluation"]["rows"]
    datasets = final_word_datasets()
    expected_final = {(mode, seed, condition) for mode, seed in expected_runs for condition in datasets}
    if len(final) != 36 or {(row["mode"], row["seed"], row["condition"]) for row in final} != expected_final:
        raise ValueError("Retain all 36 final mode/seed/condition evaluations")
    for row in final:
        dataset = datasets[row["condition"]]
        if row["image_count"] != len(dataset.scenes):
            raise ValueError("Final image counts must match the declared renderer recipe")
        _check_generated_metrics(row, training.make_word_task_batch(dataset)["expected_tokens"])

    study = evidence["stage_patching"]
    if study["stages"] != list(STAGES) or study["control_seed"] != 701 or study["samples_per_random_control"] != 8:
        raise ValueError("Use every declared stage and all eight draws of each random control")
    pairs = prepare_counterfactual_pairs()
    if len(study["pairs"]) != 72:
        raise ValueError("No counterfactual pair may be omitted")
    for recorded, clean, donor, task in zip(study["pairs"], pairs["clean_scenes"], pairs["donor_scenes"], pairs["tasks"], strict=True):
        expected = {"clean_scene_id": clean.scene_id, "task": task, "clean_answer": getattr(clean, task),
                    "donor_answer": getattr(donor, task), "clean_combination": [clean.color, clean.shape],
                    "donor_combination": [donor.color, donor.shape],
                    "donor_combination_was_heldout": (donor.color, donor.shape) in data.HELDOUT_COMBINATIONS}
        if recorded != expected:
            raise ValueError("Counterfactual pairs must match the fixed outcome-independent recipe")
    objects = patching.visual_rows_from_pixel_mask(pairs["pixel_mask"], patch_size=4)
    controls = patching.sample_visual_row_controls(objects, samples=8, seed=701)
    regions = {("object", 0): objects, ("all_visual", 0): t.ones_like(objects), ("full_sequence", 0): t.ones_like(objects),
               **{(name, index): mask for name, masks in controls.items() for index, mask in enumerate(masks)}}
    if study["object_rows"] != objects.nonzero().tolist():
        raise ValueError("Object rows must cover all changed pixels")
    rows = study["rows"]
    expected_rows = {(seed, stage, name, sample) for seed in SEEDS for stage in STAGES for name, sample in regions}
    if len(rows) != 285 or {(row["seed"], row["stage"], row["condition"], row["sample"]) for row in rows} != expected_rows:
        raise ValueError("Retain all 285 stage/region/seed outcomes")
    if len(study["endpoints"]) != 3 or {row["seed"] for row in study["endpoints"]} != set(SEEDS):
        raise ValueError("Every model seed needs its unpatched clean and donor endpoints")
    for endpoint in study["endpoints"]:
        clean_tokens, donor_tokens = (t.tensor(endpoint[key]) for key in ("clean_generated_tokens", "donor_generated_tokens"))
        clean_correct = (clean_tokens == pairs["clean_expected"]).all(1)
        donor_correct = (donor_tokens == pairs["donor_expected"]).all(1)
        if endpoint["clean_correct"] != clean_correct.tolist() or endpoint["donor_correct"] != donor_correct.tolist():
            raise ValueError("Endpoint eligibility must follow actual generated answers")
        for row in (row for row in rows if row["seed"] == endpoint["seed"]):
            tokens = t.tensor(row["generated_tokens"])
            if tokens.shape != pairs["donor_expected"].shape:
                raise ValueError("Each patch must retain every generated sequence")
            is_eos = tokens == data.VOCAB["<eos>"]
            finished = is_eos.any(1)
            lengths = t.where(finished, is_eos.long().argmax(1) + 1, tokens.shape[1])
            if row["generated_lengths"] != lengths.tolist() or row["generated_finished"] != finished.tolist():
                raise ValueError("Patched EOS/length metadata disagrees with actual words")
            targeted = (tokens == pairs["donor_expected"]).all(1)
            same_clean, same_donor = (tokens == clean_tokens).all(1), (tokens == donor_tokens).all(1)
            for key, expected in (("patched_is_donor", targeted), ("same_clean_generation", same_clean), ("same_donor_generation", same_donor)):
                if row[key] != expected.tolist():
                    raise ValueError("Patched generation flags disagree with actual words")
            if row["selected_visual_rows"] != regions[(row["condition"], row["sample"])].nonzero().tolist():
                raise ValueError("Random or object rows differ from the declared fixed sample")
            expected = _flip_counts(clean_correct, donor_correct, targeted)
            if any(not _same_number(row[key], value) for key, value in expected.items()):
                raise ValueError("Targeted-flip numerator or denominator is incorrect")
            for task in ("color", "shape"):
                selected = t.tensor([value == task for value in pairs["tasks"]])
                expected = _flip_counts(clean_correct[selected], donor_correct[selected], targeted[selected])
                if any(not _same_number(row["by_task"][task][key], value) for key, value in expected.items()):
                    raise ValueError("Task-specific targeted-flip denominator is incorrect")
            if row["condition"] == "full_sequence" or (row["stage"] == "patches" and row["condition"] in ("object", "all_visual")):
                if not bool(same_donor.all()):
                    raise ValueError("A complete image-information replacement must reproduce donor generation")
            if (row["stage"] == "patches" and row["condition"] == "background") or (row["stage"] == "decoder_1" and row["condition"] != "full_sequence"):
                if not bool(same_clean.all()):
                    raise ValueError("Structurally inert patches must preserve the clean generation")
    means = {condition: {mode: {metric: sum(row[metric] for row in final if row["mode"] == mode and row["condition"] == condition) / 3
                                for metric in ("exact_match", "color_exact_match", "shape_exact_match", "caption_exact_match")}
                         for mode in MODES} for condition in datasets}
    return {"raw_word_metrics_recomputed": True, "training_runs": 12, "final_evaluations": 36,
            "stage_region_experiments": 285, "final_means": means}


def read_completed_word_evidence(directory=None):
    directory = Path(directory) if directory is not None else HERE / "expected_outputs"
    manifest = json.loads((directory / "learner_execution.json").read_text())
    if (manifest.get("status") != "PASS" or manifest.get("scope") != "full"
            or manifest.get("completed_from_own_answers") is not True
            or manifest.get("fresh_training_required") is not True or manifest.get("sources_changed_during_run")):
        raise ValueError("Require a successful full learner with fresh training and unchanged source")
    sources = {"lesson.md", *(path.name for path in HERE.glob("*.py"))}
    if set(manifest["input_files"]) != sources:
        raise ValueError("The execution manifest must include every section source")
    for name, digest in manifest["input_files"].items():
        if sha256(HERE / name) != digest:
            raise ValueError(f"Completed word learner is stale for {name}")
    for suffix, key in (("exercises", "published_exercises_sha256"), ("solutions", "published_notebook_sha256")):
        if sha256(HERE / f"{STEM}_{suffix}.ipynb") != manifest[key]:
            raise ValueError(f"Published {suffix} differs from the completed learner")
    if set(manifest["figures"]) != REQUIRED_FIGURES:
        raise ValueError("Require every learner gallery, learning/control curve and causal figure")
    for name, digest in manifest["figures"].items():
        if sha256(HERE.parents[1] / "instructions" / "assets" / name) != digest:
            raise ValueError("A published figure differs from its measured learner output")
    payload = directory / "completed_word_study.json.gz"
    if sha256(payload) != manifest["published_evidence_sha256"]:
        raise ValueError("Raw learner evidence differs from the execution manifest")
    evidence = json.loads(gzip.decompress(payload.read_bytes()))
    summary = validate_word_measurements(evidence)
    return manifest, evidence, summary
