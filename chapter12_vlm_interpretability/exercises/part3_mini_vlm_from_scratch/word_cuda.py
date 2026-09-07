"""Fresh CUDA training/generation controls supporting the complete CPU learner.

This is a separate four-mode seed-0 replication on CUDA. It does not relabel
stored learner metrics as a CUDA run or select hardware runs from final scores.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import torch as t

from . import language_data as data, language_patching as patching, language_training as training, language_vlm as vlm
from .run_word_patching import prepare_counterfactual_pairs
from .word_evidence import MODES, STAGES, final_word_datasets, read_completed_word_evidence


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_fresh_word_cuda_check(max_vram_gb=24.):
    if not t.cuda.is_available():
        raise RuntimeError("The separate word-model CUDA check requires CUDA")
    manifest, _, learner_summary = read_completed_word_evidence()
    t.cuda.reset_peak_memory_stats()
    train = data.make_language_dataset("train", examples_per_combination=32)
    development = data.make_language_dataset("development", examples_per_combination=12)
    runs = [training.train_word_model(train, development, steps=600, batch_size=48,
                                      learning_rate=.003, seed=0, mode=mode, device="cuda", evaluation_every=100)
            for mode in MODES]
    choices = [{"mode": run.mode, "seed": run.seed, "selected_step": run.selected_step,
                "state_sha256": model_digest(run.model), "development_exact": run.development["exact_match"]}
               for run in runs]
    frozen = {"scope": "fresh_cuda_four_mode_seed_zero_check", "choices": choices,
              "training_steps": 600, "all_modes_retained": True,
              "final_conditions": ["familiar_solid", "familiar_muted", "compositional_muted"]}
    directory = Path(tempfile.mkdtemp(prefix="arena-word-cuda-"))
    selection_path = directory / "frozen_selection.json"
    selection_path.write_text(json.dumps(frozen, sort_keys=True, indent=2) + "\n")
    final_rows = []
    for run, choice in zip(runs, choices, strict=True):
        assert model_digest(run.model) == choice["state_sha256"]
        for condition, dataset in final_word_datasets().items():
            batch = training.make_word_task_batch(dataset, omit_question=run.mode == "image-only")
            measured = training.evaluate_word_model(run.model, batch, zero_images=run.mode == "text-only")
            final_rows.append({"mode": run.mode, "seed": 0, "condition": condition,
                               "image_count": len(dataset.scenes), **measured})
            print(f"Fresh CUDA {run.mode}, {condition}: exact generated sequence {measured['exact_match']:.3f}", flush=True)

    pairs = prepare_counterfactual_pairs()
    clean_pixels, donor_pixels = (pairs[key].cuda() for key in ("clean_pixels", "donor_pixels"))
    ids, valid = (pairs["prompts"][key].cuda() for key in ("token_ids", "valid_mask"))
    clean_expected, donor_expected = (pairs[key].cuda() for key in ("clean_expected", "donor_expected"))
    objects = patching.visual_rows_from_pixel_mask(pairs["pixel_mask"], patch_size=4).cuda()
    controls = patching.sample_visual_row_controls(objects, samples=8, seed=701)
    regions = [("object", 0, objects, False),
               *((name, index, mask, False) for name, masks in controls.items() for index, mask in enumerate(masks)),
               ("all_visual", 0, t.ones_like(objects), False), ("full_sequence", 0, t.ones_like(objects), True)]
    model = runs[0].model
    clean = vlm.greedy_word_generation(model, clean_pixels, ids, valid)
    donor = vlm.greedy_word_generation(model, donor_pixels, ids, valid)
    clean_correct = training.generated_word_exact_match(clean, clean_expected)
    donor_correct = training.generated_word_exact_match(donor, donor_expected)
    patch_rows = []
    for stage in STAGES:
        for condition, sample, selected, full_sequence in regions:
            generated = patching.generate_with_visual_patch(model, clean_pixels, donor_pixels, ids, valid,
                stage=stage, visual_rows=selected, full_sequence=full_sequence)
            targeted = training.generated_word_exact_match(generated, donor_expected)
            same_clean = ((generated.tokens == clean.tokens).all(1) & (generated.lengths == clean.lengths)
                          & (generated.finished == clean.finished))
            same_donor = ((generated.tokens == donor.tokens).all(1) & (generated.lengths == donor.lengths)
                          & (generated.finished == donor.finished))
            if full_sequence or (stage == "patches" and condition in ("object", "all_visual")):
                if not bool(same_donor.all()):
                    raise AssertionError("Fresh CUDA full-information patch must reproduce donor generation")
            if (stage == "patches" and condition == "background") or (stage == "decoder_1" and not full_sequence):
                if not bool(same_clean.all()):
                    raise AssertionError("Fresh CUDA structurally inert patch changed generation")
            patch_rows.append({"stage": stage, "condition": condition, "sample": sample,
                **patching.conditional_targeted_flips(clean_correct, donor_correct, targeted),
                "selected_visual_rows": selected.nonzero().tolist(), "generated_tokens": generated.tokens.tolist(),
                "generated_lengths": generated.lengths.tolist(), "generated_finished": generated.finished.tolist(),
                "same_clean_generation": same_clean.tolist(), "same_donor_generation": same_donor.tolist()})
        print(f"Fresh CUDA {stage}: all nineteen region/full-sequence conditions checked", flush=True)
    familiar = {row["mode"]: row["exact_match"] for row in final_rows if row["condition"] == "familiar_muted"}
    best_control = max(value for mode, value in familiar.items() if mode != "joint")
    # This tests the intended familiar-task signal; no compositional score selects or rejects a checkpoint.
    if familiar["joint"] < .75 or familiar["joint"] - best_control < .20:
        raise AssertionError("Fresh CUDA joint generation did not establish the declared familiar-task control separation")
    peak = t.cuda.max_memory_allocated() / 1024**3
    if peak > max_vram_gb:
        raise RuntimeError(f"Word-model CUDA check used {peak:.3f} GiB, over the {max_vram_gb:g} GiB budget")
    result = {"accepted": True, "cuda_available": True, "device": t.cuda.get_device_name(0),
        "peak_vram_gb": peak, "within_vram_budget": True,
        "word_learner_completed_from_own_answers": True, "word_learner_code_cells": manifest["code_cells_executed"],
        "word_learner_evidence_sha256": manifest["published_evidence_sha256"],
        "word_learner_metrics_recomputed_from_tokens": True, "word_learner_summary": learner_summary,
        "word_fresh_cuda_training_runs": 4, "word_fresh_cuda_final_evaluations": 12,
        "word_fresh_cuda_stage_region_experiments": len(patch_rows),
        "word_cuda_joint_familiar_exact": familiar["joint"], "word_cuda_best_control_familiar_exact": best_control,
        "word_cuda_familiar_control_gap": familiar["joint"] - best_control,
        "word_cuda_structural_patch_controls_passed": True,
        "cuda_selection": frozen, "cuda_selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "cuda_training": [{"mode": run.mode, "seed": run.seed, "selected_step": run.selected_step,
                           "history": run.history, "development": run.development,
                           "changed_parameter_groups": run.changed_parameter_groups} for run in runs],
        "cuda_final_rows": final_rows,
        "cuda_patch_endpoints": {"clean_generated_tokens": clean.tokens.tolist(), "donor_generated_tokens": donor.tokens.tolist(),
                                 "clean_correct": clean_correct.tolist(), "donor_correct": donor_correct.tolist()},
        "cuda_patch_rows": patch_rows,
        "full_path": "Complete three-seed CPU learner plus separate fresh four-mode seed-0 CUDA training, frozen generation and stage/region controls. Compositional failures are retained.",
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
