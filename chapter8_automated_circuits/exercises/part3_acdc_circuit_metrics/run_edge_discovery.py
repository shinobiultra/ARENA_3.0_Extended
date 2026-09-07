"""Run a full GPT-2 IOI additive-edge candidate with durable, validated checkpoints.

This is candidate evidence while the full learner and remaining roadmap tasks
are being built. It never upgrades a release report or roadmap status.
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
from chapter8_automated_circuits.exercises.part3_acdc_circuit_metrics import real_edges as edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=.05)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_suffix(".checkpoint.json")
    context_path = args.output.with_suffix(".context.json")
    context = {
        "model": "gpt2-small", "revision": edge.GPT2_REVISION,
        "threshold": args.threshold, "discovery_count": args.count, "discovery_seed": 1,
        "templates": "original_IOI_ABBA_and_BABA_indices_0_to_2",
        "metric": "mean_KL_clean_to_circuit_final_position_full_vocabulary",
        "candidate_graph": "all_additive_QKV_MLP_and_final_residual_edges",
        "source_sha256": hashlib.sha256(Path(edge.__file__).read_bytes()).hexdigest(),
        "torch_version": t.__version__,
    }
    resume = None
    if args.resume:
        if json.loads(context_path.read_text()) != context:
            raise ValueError("resuming requires identical source, model, data and settings")
        resume = json.loads(checkpoint_path.read_text())
    else:
        if context_path.exists() or checkpoint_path.exists() or args.output.exists():
            raise FileExistsError("use a new output or explicitly resume the existing candidate")
        context_path.write_text(json.dumps(context, indent=2) + "\n")
    model = edge.load_pinned_gpt2()
    data = edge.prepare_ioi_data(model, count=args.count, seed=1)
    engine = edge.EdgeIntervention(model, data["clean_tokens"], data["corrupt_tokens"], data["end_positions"])
    clean_diff = edge.answer_logit_differences(engine.clean_logits, data["answer_ids"])
    corrupt_diff = edge.answer_logit_differences(engine.corrupt_logits, data["answer_ids"])
    full_error = float((engine(engine.graph.legal_mask) - engine.clean_logits).abs().max())
    empty_error = float((engine(t.zeros_like(engine.graph.legal_mask)) - engine.corrupt_logits).abs().max())
    if full_error > 1e-4 or empty_error > 1e-4:
        raise AssertionError(f"endpoint parity failed: full={full_error}, empty={empty_error}")
    print(f"GPT-2 IOI: {args.count} discovery prompts; {int(engine.graph.legal_mask.sum())} legal edges; "
          f"clean/corrupt logit difference {float(clean_diff.mean()):.4f}/{float(corrupt_diff.mean()):.4f}", flush=True)
    print(f"Endpoint max errors: full={full_error:.3g}, empty={empty_error:.3g}", flush=True)
    def evaluate(mask):
        return float(edge.clean_kl_divergence(engine(mask), engine.clean_logits))
    result = edge.greedy_edge_acdc(engine.graph, evaluate, threshold=args.threshold,
                                  checkpoint_path=checkpoint_path, resume=resume)
    active = edge.mask_from_result(engine.graph, result)
    circuit_logits = engine(active)
    circuit_diff = edge.answer_logit_differences(circuit_logits, data["answer_ids"])
    result.update({
        "context": context, "prompts": data["prompts"], "corrupt_prompts": data["corrupt_prompts"],
        "answer_ids": data["answer_ids"].tolist(), "end_positions": data["end_positions"].tolist(),
        "sources": list(engine.graph.sources), "receivers": [r.name for r in engine.graph.receivers],
        "clean_logit_differences": clean_diff.tolist(), "corrupt_logit_differences": corrupt_diff.tolist(),
        "circuit_logit_differences": circuit_diff.tolist(),
        "discovery_recovery": float((circuit_diff.mean() - corrupt_diff.mean()) / (clean_diff.mean() - corrupt_diff.mean())),
        "discovery_accuracy": float((circuit_diff > 0).float().mean()),
        "clean_accuracy": float((clean_diff > 0).float().mean()),
        "full_graph_max_abs_error": full_error, "empty_graph_max_abs_error": empty_error,
        "peak_vram_gib": t.cuda.max_memory_allocated() / 1024**3,
        "release_status": "candidate_requires_heldout_controls_and_completed_learner",
    })
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Complete: {result['retained_edges']} edges, discovery recovery "
          f"{result['discovery_recovery']:.3f}, accuracy {result['discovery_accuracy']:.3f}; "
          f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
