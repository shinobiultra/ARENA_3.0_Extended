"""Measure frozen word models on fixed counterfactual pairs; never select weights.

This candidate runner supplies orchestration, not the learner implementation.
The lesson will run the same study with its own stage/row/generation methods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch as t

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter12_vlm_interpretability.exercises.part3_mini_vlm_from_scratch import (
    language_data as data,
    language_patching as patching,
    language_training as training,
    language_vlm as vlm,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_counterfactual_pairs():
    """First four instances of every familiar combination, both VQA questions.

    Use the exact final familiar recipe (16 instances per combination), then
    take fixed positions. No prediction, correctness or patch result selects
    a pair. Donors cycle the queried attribute and retain all nuisance values.
    """
    familiar = data.make_language_dataset("familiar", examples_per_combination=16)
    indices = [group * 16 + instance for group in range(9) for instance in range(4)]
    clean_scenes, donor_scenes, tasks, clean_pixels, donor_pixels, occupied = [], [], [], [], [], []
    for index in indices:
        scene = familiar.scenes[index]
        for task, vocabulary in (("color", data.COLORS), ("shape", data.SHAPES)):
            current = getattr(scene, task)
            changed = vocabulary[(vocabulary.index(current) + 1) % len(vocabulary)]
            donor = data.counterfactual_scene(scene, **{task: changed})
            pixels, mask = data.render_language_scene(donor)
            union = familiar.object_masks[index] | mask
            if not t.equal(familiar.images[index][:, ~union], pixels[:, ~union]):
                raise AssertionError("Counterfactuals must have identical pixels outside the object union")
            clean_scenes.append(scene)
            donor_scenes.append(donor)
            tasks.append(task)
            clean_pixels.append(familiar.images[index])
            donor_pixels.append(pixels)
            occupied.append(union)
    prompts = data.tokenize_language_examples(clean_scenes, tasks, include_answers=False)
    expected = []
    for scenes in (clean_scenes, donor_scenes):
        ids = t.full((len(scenes), 8), data.VOCAB["<pad>"], dtype=t.long)
        for row, (scene, task) in enumerate(zip(scenes, tasks, strict=True)):
            ids[row, :2] = t.tensor([data.VOCAB[getattr(scene, task)], data.VOCAB["<eos>"]])
        expected.append(ids)
    return {
        "clean_scenes": clean_scenes, "donor_scenes": donor_scenes, "tasks": tasks,
        "clean_pixels": t.stack(clean_pixels), "donor_pixels": t.stack(donor_pixels),
        "pixel_mask": t.stack(occupied), "prompts": prompts,
        "clean_expected": expected[0], "donor_expected": expected[1],
    }


def run_frozen_patching(selection_path, *, device="cpu"):
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text())
    configuration = json.loads((selection_path.parent / "configuration.json").read_text())
    if selection["configuration_sha256"] != sha256(selection_path.parent / "configuration.json"):
        raise ValueError("The configuration must match the pre-evaluation freeze")
    for module in (data, vlm, training):
        if configuration["source_sha256"][Path(module.__file__).name] != sha256(module.__file__):
            raise ValueError("Frozen model, training and data sources must remain unchanged")
    pairs = prepare_counterfactual_pairs()
    clean_pixels, donor_pixels = (pairs[key].to(device) for key in ("clean_pixels", "donor_pixels"))
    prompt_ids = pairs["prompts"]["token_ids"].to(device)
    prompt_valid = pairs["prompts"]["valid_mask"].to(device)
    clean_expected, donor_expected = (pairs[key].to(device) for key in ("clean_expected", "donor_expected"))
    object_rows = patching.visual_rows_from_pixel_mask(pairs["pixel_mask"], patch_size=4).to(device)
    controls = patching.sample_visual_row_controls(object_rows, samples=8, seed=701)
    conditions = [("object", 0, object_rows, False),
                  *((name, index, rows, False) for name, samples in controls.items() for index, rows in enumerate(samples)),
                  ("all_visual", 0, t.ones_like(object_rows), False),
                  ("full_sequence", 0, t.ones_like(object_rows), True)]
    rows, endpoints = [], []
    stages = ("patches", "vision", "projected", "decoder_0", "decoder_1")
    for choice in selection["choices"]:
        if choice["mode"] != "joint":
            continue
        weights_path = selection_path.parent / choice["weights_file"]
        if sha256(weights_path) != choice["weights_sha256"]:
            raise ValueError("Weights must match the pre-evaluation freeze")
        model = vlm.WordMiniVLM().to(device)
        model.load_state_dict(t.load(weights_path, map_location=device, weights_only=True))
        model.eval()
        clean = vlm.greedy_word_generation(model, clean_pixels, prompt_ids, prompt_valid)
        donor = vlm.greedy_word_generation(model, donor_pixels, prompt_ids, prompt_valid)
        clean_correct = training.generated_word_exact_match(clean, clean_expected)
        donor_correct = training.generated_word_exact_match(donor, donor_expected)
        endpoints.append({"seed": choice["seed"], "clean_generated_tokens": clean.tokens.tolist(),
                          "donor_generated_tokens": donor.tokens.tolist(),
                          "clean_correct": clean_correct.tolist(), "donor_correct": donor_correct.tolist()})
        for stage in stages:
            started = time.perf_counter()
            for condition, sample, selected_rows, full_sequence in conditions:
                generated = patching.generate_with_visual_patch(
                    model, clean_pixels, donor_pixels, prompt_ids, prompt_valid,
                    stage=stage, visual_rows=selected_rows, full_sequence=full_sequence,
                )
                targeted = training.generated_word_exact_match(generated, donor_expected)
                same_clean = ((generated.tokens == clean.tokens).all(1) & (generated.lengths == clean.lengths)
                              & (generated.finished == clean.finished))
                same_donor = ((generated.tokens == donor.tokens).all(1) & (generated.lengths == donor.lengths)
                              & (generated.finished == donor.finished))
                if full_sequence or (stage == "patches" and condition in ("object", "all_visual")):
                    if not bool(same_donor.all()):
                        raise AssertionError(f"{stage}/{condition}: exact donor-generation endpoint failed")
                if (stage == "patches" and condition == "background") or (stage == "decoder_1" and not full_sequence):
                    if not bool(same_clean.all()):
                        raise AssertionError(f"{stage}/{condition}: structurally inert generation changed")
                stratified = {}
                for task in ("color", "shape"):
                    subset = t.tensor([value == task for value in pairs["tasks"]], device=device)
                    stratified[task] = patching.conditional_targeted_flips(
                        clean_correct[subset], donor_correct[subset], targeted[subset])
                rows.append({"seed": choice["seed"], "stage": stage, "condition": condition, "sample": sample,
                             **patching.conditional_targeted_flips(clean_correct, donor_correct, targeted),
                             "by_task": stratified, "selected_visual_rows": selected_rows.nonzero().tolist(),
                             "generated_tokens": generated.tokens.tolist(), "generated_lengths": generated.lengths.tolist(),
                             "generated_finished": generated.finished.tolist(), "patched_is_donor": targeted.tolist(),
                             "same_clean_generation": same_clean.tolist(), "same_donor_generation": same_donor.tolist()})
            print(f"seed {choice['seed']} {stage}: all 19 region/full-sequence conditions; "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
    return {"status": "frozen_stage_region_generation_study", "selection_sha256": sha256(selection_path),
            "source_sha256": {Path(module.__file__).name: sha256(module.__file__)
                              for module in (data, vlm, training, patching)},
            "pair_rule": "First four of each familiar combination in the fixed 16-per-combination final recipe; both VQA questions",
            "control_seed": 701, "samples_per_random_control": 8, "stages": list(stages),
            "pairs": [{"clean_scene_id": clean.scene_id, "task": task,
                       "clean_answer": getattr(clean, task), "donor_answer": getattr(donor, task),
                       "clean_combination": [clean.color, clean.shape], "donor_combination": [donor.color, donor.shape],
                       "donor_combination_was_heldout": (donor.color, donor.shape) in data.HELDOUT_COMBINATIONS}
                      for clean, donor, task in zip(pairs["clean_scenes"], pairs["donor_scenes"], pairs["tasks"], strict=True)],
            "object_rows": object_rows.nonzero().tolist(), "endpoints": endpoints, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous outcomes; choose a new result path")
    result = run_frozen_patching(args.selection, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
