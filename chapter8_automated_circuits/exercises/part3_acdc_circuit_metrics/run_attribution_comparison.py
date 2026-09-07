"""Fit EAP/EAP-IG on discovery prompts and compare fixed sizes on development.

This does not open held-out templates or select a release claim. Scores,
selected edges and every random-control result are written for inspection.
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
    edge_attribution as attribution,
    edge_metrics as metrics,
    greater_than as gt,
    ioi_controls,
    real_edges as edge,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("ioi", "greater-than"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 100, 300, 1000, 3000])
    parser.add_argument("--random-count", type=int, default=16)
    parser.add_argument("--ig-steps", type=int, default=8)
    parser.add_argument("--selection", choices=("topk", "greedy"), default="topk")
    parser.add_argument("--ioi-corruption", choices=("all-names", "second-subject"), default="all-names")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the previous comparison; choose a new output")
    model = edge.load_pinned_gpt2()
    if args.task == "ioi":
        prepare = edge.prepare_ioi_data if args.ioi_corruption == "all-names" else ioi_controls.prepare_ioi_second_subject_data
        discovery = prepare(model, count=32, seed=1)
        development = prepare(model, count=32, seed=2)
        def loss(logits):
            return -edge.answer_logit_differences(logits, discovery["answer_ids"]).mean()
        def behavior(logits, engine):
            return metrics.circuit_behavior(logits, engine.clean_logits, engine.corrupt_logits, development["answer_ids"])
    else:
        discovery = gt.prepare_greater_than_data(model, count=32, seed=11)
        development = gt.prepare_greater_than_data(model, count=32, seed=12, split="development")
        def loss(logits):
            return -gt.greater_than_probability_difference(logits, discovery["suffix_ids"], discovery["start_suffixes"]).mean()
        def behavior(logits, engine):
            return gt.greater_than_behavior(logits, engine.clean_logits, engine.corrupt_logits,
                                            development["suffix_ids"], development["start_suffixes"])
    overlap = set(discovery["prompts"]) & set(development["prompts"])
    if overlap:
        raise AssertionError("discovery and development prompts must be distinct")
    graph = edge.build_residual_graph(model.cfg.n_layers, model.cfg.n_heads)
    with t.inference_mode():
        residual = model(discovery["clean_tokens"].to(next(model.parameters()).device),
                         stop_at_layer=graph.n_layers, return_type=None)
        clean_reference = edge.final_position_logits(model, residual, discovery["end_positions"])
    def kl_loss(logits):
        return edge.clean_kl_divergence(logits, clean_reference)
    scores_by_method = {}
    for label, method, metric in (("eap", "eap", loss), ("eap-ig-inputs", "eap-ig-inputs", loss),
                                  ("eap-ig-kl", "eap-ig-inputs", kl_loss)):
        scores_by_method[label] = attribution.edge_attribution_scores(
            model, discovery["clean_tokens"], discovery["corrupt_tokens"], discovery["end_positions"],
            metric, method=method, steps=args.ig_steps,
        )
        print(f"{args.task}: scored all {int(graph.legal_mask.sum())} edges with {label}", flush=True)
    engine = edge.EdgeIntervention(model, development["clean_tokens"], development["corrupt_tokens"], development["end_positions"])
    full, empty = engine(graph.legal_mask), engine(t.zeros_like(graph.legal_mask))
    full_error = float((full - engine.clean_logits).abs().max())
    empty_error = float((empty - engine.corrupt_logits).abs().max())
    if full_error > 1e-4 or empty_error > 1e-4:
        raise AssertionError("development endpoint parity failed")
    rows = []
    for size in args.sizes:
        entry = {"edge_count": size, "methods": {}, "uniform_same_size": []}
        for method, scores in scores_by_method.items():
            selector = attribution.top_edge_mask if args.selection == "topk" else attribution.greedy_score_mask
            mask = selector(graph, scores, size)
            result = behavior(engine(mask), engine)
            result["active_edges"] = mask.nonzero().tolist()
            entry["methods"][method] = result
        for random_mask in metrics.same_size_random_masks(graph, mask, count=args.random_count, seed=701):
            result = behavior(engine(random_mask), engine)
            result["active_edges"] = random_mask.nonzero().tolist()
            entry["uniform_same_size"].append(result)
        rows.append(entry)
        print(f"{args.task} {size} edges: recovery EAP {entry['methods']['eap']['recovery']:.3f}, "
              f"EAP-IG {entry['methods']['eap-ig-inputs']['recovery']:.3f}, "
              f"EAP-IG-KL {entry['methods']['eap-ig-kl']['recovery']:.3f}", flush=True)
    result = {
        "status": "development_comparison_only", "task": args.task,
        "corruption": args.ioi_corruption if args.task == "ioi" else "start_year_suffix_01",
        "model": "gpt2-small", "revision": edge.GPT2_REVISION, "ig_steps": args.ig_steps,
        "quadrature": "alpha=0,1/m,...,(m-1)/m", "score_selection": args.selection,
        "selection_detail": "absolute_scores_exact_budget_no_input_path_pruning",
        "source_sha256": {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for module in (attribution, metrics, gt, edge, ioi_controls)},
        "discovery": {key: value.tolist() if isinstance(value, t.Tensor) else value for key, value in discovery.items()},
        "development": {key: value.tolist() if isinstance(value, t.Tensor) else value for key, value in development.items()},
        "legal_edges": graph.legal_mask.nonzero().tolist(),
        "scores": {method: scores[graph.legal_mask].tolist() for method, scores in scores_by_method.items()},
        "full": behavior(full, engine), "empty": behavior(empty, engine), "rows": rows,
        "random_seed": 701, "random_count_per_size": args.random_count,
        "full_graph_max_abs_error": full_error, "empty_graph_max_abs_error": empty_error,
        "peak_vram_gib": t.cuda.max_memory_allocated() / 1024**3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
