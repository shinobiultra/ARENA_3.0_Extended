"""Evaluate a completed candidate on development or sealed held-out IOI prompts.

The held-out route requires a previously written selection record identifying
the frozen candidate. All random circuits and negative diagnostic results are
retained. This runner does not upgrade the section's teaching/release status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch as t

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter8_automated_circuits.exercises.part3_acdc_circuit_metrics import (
    edge_metrics as metrics,
    greater_than as gt,
    real_edges as edge,
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_candidate(candidate_path: Path, *, split: str, random_count: int = 32,
                       diagnostics: bool = False, selection_path: Path | None = None) -> dict:
    candidate = json.loads(candidate_path.read_text())
    if not candidate["complete"] or candidate["context"]["source_sha256"] != file_sha256(edge.__file__):
        raise ValueError("only a completed candidate from this exact intervention source is eligible")
    task = "greater-than" if candidate["context"].get("task") == "greater_than" else "ioi"
    if task == "greater-than" and candidate["context"]["data_metric_source_sha256"] != file_sha256(gt.__file__):
        raise ValueError("greater-than data and metric source must match the frozen candidate")
    if split not in ("development", "heldout"):
        raise ValueError("choose development or heldout")
    selection = None
    if split == "heldout":
        if selection_path is None:
            raise ValueError("freeze a candidate in a selection record before opening held-out prompts")
        selection = json.loads(selection_path.read_text())
        if selection["candidate_sha256"] != file_sha256(candidate_path):
            raise ValueError("selection record must identify this exact candidate")
        if not selection["development_evidence"] or not selection["reason"]:
            raise ValueError("selection needs development evidence and an explicit reason")
    seed, count = (2, 32) if split == "development" else (3, 64)
    if task == "greater-than":
        seed += 10
    model = edge.load_pinned_gpt2()
    data = (edge.prepare_ioi_data(model, count=count, seed=seed, shifted_templates=split == "heldout")
            if task == "ioi" else gt.prepare_greater_than_data(model, count=count, seed=seed, split=split))
    engine = edge.EdgeIntervention(model, data["clean_tokens"], data["corrupt_tokens"], data["end_positions"])
    mask = edge.mask_from_result(engine.graph, candidate)
    graph = engine.graph
    full_logits, empty_logits = engine(graph.legal_mask), engine(t.zeros_like(mask))
    full_error = float((full_logits - engine.clean_logits).abs().max())
    empty_error = float((empty_logits - engine.corrupt_logits).abs().max())
    if full_error > 1e-4 or empty_error > 1e-4:
        raise AssertionError("full/empty intervention endpoints must match on each evaluation split")

    def behavior(logits):
        if task == "ioi":
            return metrics.circuit_behavior(logits, engine.clean_logits, engine.corrupt_logits, data["answer_ids"])
        return gt.greater_than_behavior(logits, engine.clean_logits, engine.corrupt_logits,
                                        data["suffix_ids"], data["start_suffixes"])

    task_metric = (data["answer_ids"] if task == "ioi" else
                   lambda logits: gt.greater_than_probability_difference(logits, data["suffix_ids"], data["start_suffixes"]))

    chosen = behavior(engine(mask))
    controls = {}
    for name, matched in (("uniform_same_size", False), ("receiver_indegree_matched", True)):
        rows = []
        for index, random_mask in enumerate(metrics.same_size_random_masks(
            graph, mask, count=random_count, seed=701, match_receivers=matched,
        )):
            row = behavior(engine(random_mask))
            row["active_edges"] = random_mask.nonzero().tolist()
            rows.append(row)
        controls[name] = {
            "seed": 701, "count": random_count, "results": rows,
            "fraction_at_least_candidate_recovery": sum(row["recovery"] >= chosen["recovery"] for row in rows) / random_count,
            "fraction_at_most_candidate_kl": sum(row["kl_clean_to_circuit"] <= chosen["kl_clean_to_circuit"] for row in rows) / random_count,
        }
        print(f"{split} {name}: all {random_count} sampled circuits scored", flush=True)
    result = {
        "status": "candidate_evaluation_not_a_release_upgrade", "split": split, "task": task,
        "seed": seed, "count": count, "shifted_templates": data["shifted_templates"],
        "candidate_sha256": file_sha256(candidate_path), "candidate_path": str(candidate_path),
        "intervention_source_sha256": file_sha256(edge.__file__),
        "metric_source_sha256": file_sha256(metrics.__file__),
        "threshold": candidate["threshold"], "retained_edges": int(mask.sum()),
        "templates": data["templates"] if task == "ioi" else [data["template"]], "prompts": data["prompts"],
        "corrupt_prompts": data["corrupt_prompts"],
        "end_positions": data["end_positions"].tolist(),
        "candidate": chosen, "full": behavior(full_logits), "empty": behavior(empty_logits),
        "controls": controls, "full_graph_max_abs_error": full_error,
        "empty_graph_max_abs_error": empty_error, "selection": selection,
    }
    if task == "ioi":
        result["answer_ids"] = data["answer_ids"].tolist()
    else:
        result["suffix_ids"] = data["suffix_ids"].tolist()
        result["start_suffixes"] = data["start_suffixes"].tolist()
        result["data_metric_source_sha256"] = file_sha256(gt.__file__)
    if diagnostics:
        result["minimality"] = metrics.minimality_diagnostics(engine, mask, task_metric)
        print(f"{split}: all {int(mask.sum())} retained-edge deletions recomputed", flush=True)
        result["sampled_completeness"] = metrics.sampled_completeness_diagnostics(engine, mask, task_metric)
    result["peak_vram_gib"] = t.cuda.max_memory_allocated() / 1024**3
    accuracy_key = "pairwise_accuracy" if task == "ioi" else "fraction_positive_probability_difference"
    print(f"{task} {split}: {int(mask.sum())} edges, recovery {chosen['recovery']:.4f}, "
          f"{accuracy_key} {chosen[accuracy_key]:.4f}, KL {chosen['kl_clean_to_circuit']:.4f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "heldout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--random-count", type=int, default=32)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the previous evaluation; choose a new output path")
    result = evaluate_candidate(args.candidate, split=args.split, random_count=args.random_count,
                                diagnostics=args.diagnostics, selection_path=args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
