"""Run full-graph ACDC on the fixed greater-than discovery task."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch as t

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter8_automated_circuits.exercises.part3_acdc_circuit_metrics import greater_than as gt, real_edges as edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=.004)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".checkpoint.json")
    context_path = args.output.with_suffix(".context.json")
    context = {
        "model": "gpt2-small", "revision": edge.GPT2_REVISION, "task": "greater_than",
        "threshold": args.threshold, "discovery_count": 32, "discovery_seed": 11,
        "metric": "mean_KL_clean_to_circuit_final_position_full_vocabulary",
        "corruption": "start_year_suffix_01", "candidate_graph": "all_additive_QKV_MLP_and_final_residual_edges",
        "source_sha256": hashlib.sha256(Path(edge.__file__).read_bytes()).hexdigest(),
        "data_metric_source_sha256": hashlib.sha256(Path(gt.__file__).read_bytes()).hexdigest(),
        "torch_version": t.__version__,
    }
    resume = None
    if args.resume:
        if json.loads(context_path.read_text()) != context:
            raise ValueError("resume requires the exact original source, model, task and settings")
        resume = json.loads(checkpoint.read_text())
    else:
        if context_path.exists() or checkpoint.exists() or args.output.exists():
            raise FileExistsError("use a new output path or explicitly resume the candidate")
        context_path.write_text(json.dumps(context, indent=2) + "\n")
    model = edge.load_pinned_gpt2()
    data = gt.prepare_greater_than_data(model, count=32, seed=11)
    engine = edge.EdgeIntervention(model, data["clean_tokens"], data["corrupt_tokens"], data["end_positions"])
    full_error = float((engine(engine.graph.legal_mask) - engine.clean_logits).abs().max())
    empty_error = float((engine(t.zeros_like(engine.graph.legal_mask)) - engine.corrupt_logits).abs().max())
    if full_error > 1e-4 or empty_error > 1e-4:
        raise AssertionError("greater-than full/empty endpoint parity failed")
    def behavior(logits):
        return gt.greater_than_behavior(logits, engine.clean_logits, engine.corrupt_logits,
                                        data["suffix_ids"], data["start_suffixes"])
    print(f"Greater-than: 32 discovery prompts, {int(engine.graph.legal_mask.sum())} legal edges", flush=True)
    def loss(mask):
        return float(edge.clean_kl_divergence(engine(mask), engine.clean_logits))
    result = edge.greedy_edge_acdc(engine.graph, loss, threshold=args.threshold,
                                  checkpoint_path=checkpoint, resume=resume)
    mask = edge.mask_from_result(engine.graph, result)
    result.update({
        "context": context, "discovery": behavior(engine(mask)),
        "data": {key: value.tolist() if isinstance(value, t.Tensor) else value for key, value in data.items()},
        "sources": list(engine.graph.sources), "receivers": [receiver.name for receiver in engine.graph.receivers],
        "full_graph_max_abs_error": full_error, "empty_graph_max_abs_error": empty_error,
        "peak_vram_gib": t.cuda.max_memory_allocated() / 1024**3,
        "release_status": "candidate_requires_heldout_controls_and_completed_learner",
    })
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Complete: {int(mask.sum())} edges; discovery recovery {result['discovery']['recovery']:.4f}", flush=True)


if __name__ == "__main__":
    main()
