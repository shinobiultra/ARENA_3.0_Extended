"""Validate completed learner evidence and recheck its frozen circuits on CUDA.

The long searches are executed by execute_learner.py, from the notebook's own
answers. This supporting check verifies their provenance and freshly measures
the frozen circuits and controls; it never presents stored metrics as a run.
"""
from __future__ import annotations

from arena_ext.readability_compatibility import source_matches_execution

import gzip
import hashlib
import json
from pathlib import Path

import torch as t

from . import edge_metrics, greater_than, real_edges

HERE = Path(__file__).resolve().parent
STEM = "8.3_ACDC_and_Circuit_Metrics"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_completed_learner_evidence(directory: Path | None = None) -> tuple[dict, dict]:
    """Require a current, successful full execution and its exact raw payload."""
    directory = directory or HERE / "expected_outputs"
    manifest = json.loads((directory / "learner_execution.json").read_text())
    if (manifest.get("status") != "PASS" or manifest.get("scope") != "full"
            or manifest.get("completed_from_own_answers") is not True
            or manifest.get("fresh_searches_required") is not True
            or manifest.get("sources_changed_during_run")):
        raise ValueError("A successful full learner with fresh searches and unchanged source is required")
    required_sources = {"lesson.md", *(path.name for path in HERE.glob("*.py"))}
    if set(manifest["input_files"]) != required_sources:
        raise ValueError("The execution record must bind every section source file")
    for name, expected in manifest["input_files"].items():
        if not source_matches_execution(HERE / name, expected):
            raise ValueError(f"Completed learner evidence is stale for {name}")
    for suffix, key in (("exercises", "published_exercises_sha256"), ("solutions", "published_notebook_sha256")):
        if sha256(HERE / f"{STEM}_{suffix}.ipynb") != manifest[key]:
            raise ValueError(f"Published {suffix} notebook differs from the completed learner")
    payload = directory / "completed_edge_study.json.gz"
    if sha256(payload) != manifest["published_evidence_sha256"]:
        raise ValueError("The raw learner evidence does not match its execution record")
    evidence = json.loads(gzip.decompress(payload.read_bytes()))
    if evidence["fresh_searches"] is not True or evidence["model_revision"] != real_edges.GPT2_REVISION:
        raise ValueError("Learner search mode or model revision differs from the declared protocol")
    searches = evidence["ioi"]["searches"] + [evidence["greater_than"]["search"]]
    if [search["threshold"] for search in searches] != [.05, .004, .001, .0005, .004]:
        raise ValueError("The complete declared threshold sweep must be preserved")
    for search in searches:
        if not search["complete"] or not search["trace"] or abs(search["initial_loss"]) > 1e-4:
            raise ValueError("Every search must start at the full graph and complete actual edge trials")
    ioi = evidence["ioi"]
    qualified = [row for row in ioi["development_rows"] if row["recovery"] >= .85
                 and row["pairwise_accuracy"] >= row["clean_pairwise_accuracy"] - .10
                 and row["kl_clean_to_circuit"] <= .5]
    if ioi["selection_passed"] != bool(qualified):
        raise ValueError("IOI selection must respect all fixed development criteria")
    if not qualified:
        if ioi["heldout"] is not None or ioi["frozen_selection"] is not None or ioi["evaluation_split"] != "development":
            raise ValueError("Failed IOI selection must leave the holdout unopened")
        expected = max(ioi["development_rows"], key=lambda row: row["recovery"])
        if ioi["diagnostic_candidate"]["threshold"] != expected["threshold"]:
            raise ValueError("The IOI diagnostic must be the declared maximum-development-recovery candidate")
    elif ioi["frozen_selection"] is None or ioi["evaluation_split"] != "heldout":
        raise ValueError("Qualified IOI transfer requires a frozen selection")
    return manifest, evidence


def _check_behavior(actual: dict, expected: dict, *, task: str) -> float:
    raw = "mean_logit_difference" if task == "ioi" else "mean_probability_difference"
    positive = "pairwise_accuracy" if task == "ioi" else "fraction_positive_probability_difference"
    keys = ("recovery", raw, positive, "kl_clean_to_circuit")
    error = max(abs(actual[key] - expected[key]) for key in keys)
    if error > 1e-4:
        raise AssertionError(f"Fresh {task} behavior differs from the completed learner by {error:g}")
    return error


def run_frozen_edge_cuda_recheck(max_vram_gb: float = 24.0) -> dict:
    """Recompute the IOI diagnostic, four GT transfer circuits and all 320 nulls."""
    if not t.cuda.is_available():
        raise RuntimeError("Full residual-edge verification requires CUDA")
    manifest, evidence = read_completed_learner_evidence()
    model = real_edges.load_pinned_gpt2()
    results = {}
    max_behavior_error = 0.
    max_endpoint_error = 0.
    control_count = 0
    for task, key in (("ioi", "ioi"), ("greater-than", "greater_than")):
        study = evidence[key]
        ioi_transfer = task == "ioi" and study["selection_passed"]
        data = (real_edges.prepare_ioi_data(model, count=64 if ioi_transfer else 32,
                                           seed=3 if ioi_transfer else 2, shifted_templates=ioi_transfer)
                if task == "ioi" else greater_than.prepare_greater_than_data(model, count=64, seed=13, split="heldout"))
        recorded_data = study["evaluation"] if task == "ioi" else study["heldout"]
        for field in ("prompts", "corrupt_prompts", "clean_tokens", "corrupt_tokens", "end_positions"):
            actual = data[field].tolist() if isinstance(data[field], t.Tensor) else data[field]
            if actual != recorded_data[field]:
                raise ValueError(f"Evaluation {task} {field} differs from the frozen learner study")
        engine = real_edges.EdgeIntervention(model, data["clean_tokens"], data["corrupt_tokens"], data["end_positions"])
        if engine.graph.legal_mask.nonzero().tolist() != evidence["legal_edges"]:
            raise ValueError("The full candidate graph differs from the completed learner")
        full, empty = engine(engine.graph.legal_mask), engine(t.zeros_like(engine.graph.legal_mask))
        endpoint_error = max(float((full - engine.clean_logits).abs().max()),
                             float((empty - engine.corrupt_logits).abs().max()))
        if endpoint_error > 1e-4:
            raise AssertionError("Frozen-circuit verification failed full/empty endpoint parity")
        max_endpoint_error = max(max_endpoint_error, endpoint_error)

        def behavior(logits):
            if task == "ioi":
                return edge_metrics.circuit_behavior(logits, engine.clean_logits, engine.corrupt_logits, data["answer_ids"])
            return greater_than.greater_than_behavior(logits, engine.clean_logits, engine.corrupt_logits,
                                                      data["suffix_ids"], data["start_suffixes"])

        rows = [study["evaluation_row"]] if task == "ioi" else study["transfer"]
        task_results = []
        for row in rows:
            frozen = (study["diagnostic_candidate"] if task == "ioi" else
                      study["frozen_selection"]["choices"][row["method"]])
            mask = real_edges.mask_from_result(engine.graph, frozen)
            candidate = behavior(engine(mask))
            max_behavior_error = max(max_behavior_error, _check_behavior(candidate, row["candidate"], task=task))
            controls = {}
            for name, matched in (("uniform_same_size", False), ("receiver_indegree_matched", True)):
                expected_controls = row["controls"][name]
                if expected_controls["count"] != 32 or expected_controls["seed"] != 701:
                    raise ValueError("All 32 samples of each declared null must be retained")
                random_masks = edge_metrics.same_size_random_masks(engine.graph, mask, count=32, seed=701, match_receivers=matched)
                measurements = []
                for random_mask, expected in zip(random_masks, expected_controls["results"], strict=True):
                    if random_mask.nonzero().tolist() != expected["active_edges"]:
                        raise ValueError("A random sample differs from its frozen edge set")
                    measured = behavior(engine(random_mask))
                    max_behavior_error = max(max_behavior_error, _check_behavior(measured, expected, task=task))
                    measurements.append(measured)
                    control_count += 1
                controls[name] = measurements
            task_results.append({"method": row["method"], "retained_edges": int(mask.sum()),
                                 "candidate": candidate, "controls": controls})
            print(f"Fresh {task} {row['method']}: {int(mask.sum())} edges, recovery {candidate['recovery']:.4f}", flush=True)
        results[key] = {"full": behavior(full), "empty": behavior(empty), "frozen_methods": task_results,
                        "split": study["evaluation_split"] if task == "ioi" else "heldout"}
        del engine
    peak = t.cuda.max_memory_allocated() / 1024**3
    if max(peak, evidence["peak_vram_gb"]) > max_vram_gb:
        raise RuntimeError(f"Frozen edge recheck used {peak:.2f} GiB, exceeding {max_vram_gb:g}")
    return {
        "accepted": True, "cuda_available": True, "full_edge_graph": True,
        "model_name": "gpt2-small", "hf_revision": real_edges.GPT2_REVISION,
        "legal_additive_edges": len(evidence["legal_edges"]),
        "fresh_frozen_circuit_cuda_recheck": True, "fresh_random_circuits_rechecked": control_count,
        "completed_learner_fresh_searches": True, "completed_learner_code_cells": manifest["code_cells_executed"],
        "ioi_development_selection_passed": evidence["ioi"]["selection_passed"],
        "ioi_holdout_opened": evidence["ioi"]["heldout"] is not None,
        "completed_learner_evidence_sha256": manifest["published_evidence_sha256"],
        "max_full_empty_logit_error": max_endpoint_error,
        "max_live_vs_learner_behavior_error": max_behavior_error,
        "peak_vram_gb": peak, "within_vram_budget": True,
        "learner_search_peak_vram_gb": evidence["peak_vram_gb"],
        "ioi": results["ioi"], "greater_than": results["greater_than"],
        "claim_scope": (
            "Full additive-edge methods, failed IOI selection with development diagnostics, and "
            "greater-than controlled transfer; fresh search provenance plus live circuit checks. "
            "No successful IOI transfer is claimed when selection fails."
        ),
    }
