"""Compare every legal EAP/EAP-IG score with pinned, unedited author code.

The only adapter supplies our already-tokenized aligned examples and lengths;
the upstream attribution, graph and hook implementations remain byte-identical.
Run with --gpt2 for a pinned checkpoint comparison as well as the small model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch as t

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter8_automated_circuits.exercises.part3_acdc_circuit_metrics import (
    edge_attribution as learner,
    real_edges as edge,
)
from chapter8_automated_circuits.exercises.part3_acdc_circuit_metrics.edge_tests import example_inputs, small_model

UPSTREAM_COMMIT = "e24bafd3d22af2c59ac5a83e0a6581b89b5c05dd"
UPSTREAM_HASHES = {
    "attribute.py": "c4c30ecac4d28dd58fc2e9f2a0d7c94d6b0165d4c8c17ac45fdadf0c03566009",
    "evaluate.py": "d29e702b931e717c9992f0a9efcd8c7ce307b77341460af28dbac2a3ad7ff718",
    "graph.py": "3ec51d2e112ad7ef9118756f210872793c07d842f773afe233972be9be2faa92",
    "utils.py": "1bea32178bfd0313f5c5b460c9965c7a1e218a73de47ec5918928d61cdd5d7c3",
    "visualization.py": "98e178bfba53d68d89da0014ee62d4714de925565e3b0668a7b6519f47e41814",
}


def load_upstream():
    cache = Path(tempfile.gettempdir()) / f"arena-eapig-{UPSTREAM_COMMIT}" / "eap"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "__init__.py").write_text("")
    for name, expected in UPSTREAM_HASHES.items():
        path = cache / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/hannamw/EAP-IG/{UPSTREAM_COMMIT}/src/eap/{name}"
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"upstream source hash mismatch: {name}")
    if "eap" in sys.modules:
        raise RuntimeError("run parity in a fresh process to avoid importing a different eap package")
    sys.path.insert(0, str(cache.parent))
    return importlib.import_module("eap.attribute"), importlib.import_module("eap.graph").Graph


def compare(model, clean, corrupt, ends, answers, *, steps=8):
    upstream, Graph = load_upstream()
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    graph = Graph.from_model(model)
    ours = edge.build_residual_graph(model.cfg.n_layers, model.cfg.n_heads)
    t.testing.assert_close(graph.real_edge_mask, ours.legal_mask)
    device = next(model.parameters()).device

    def token_adapter(unused_model, inputs):
        tokens = inputs.to(device)
        mask = (t.arange(tokens.shape[1], device=device)[None, :] <= ends.to(device)[:, None]).long()
        return tokens, mask, ends.to(device) + 1, tokens.shape[1]

    upstream.tokenize_plus = token_adapter
    def author_loss(logits, clean_logits, lengths, labels):
        del clean_logits
        final = logits[t.arange(len(logits), device=logits.device), lengths - 1]
        # Author code divides by batch size after accumulating summed-loss gradients.
        return -edge.answer_logit_differences(final, labels).sum()
    def learner_loss(logits):
        return -edge.answer_logit_differences(logits, answers).mean()

    with t.no_grad():
        residual = model(clean.to(device), stop_at_layer=ours.n_layers, return_type=None)
        clean_reference = edge.final_position_logits(model, residual, ends)
    def learner_kl(logits):
        return edge.clean_kl_divergence(logits, clean_reference)
    def author_kl(logits, clean_logits, lengths, labels):
        del labels
        indices = t.arange(len(logits), device=logits.device)
        final, clean_final = logits[indices, lengths - 1], clean_logits[indices, lengths - 1]
        return edge.clean_kl_divergence(final, clean_final) * len(logits)

    rows = []
    for method in ("eap", "eap-ig-inputs", "eap-ig-kl"):
        loss = learner_kl if method == "eap-ig-kl" else learner_loss
        actual = learner.edge_attribution_scores(model, clean, corrupt, ends, loss,
                                                 method="eap" if method == "eap" else "eap-ig-inputs", steps=steps)
        model.zero_grad(set_to_none=True)
        if method == "eap":
            expected = upstream.get_scores_eap(model, graph, [(clean, corrupt, answers)], author_loss, quiet=True)
        else:
            metric = author_kl if method == "eap-ig-kl" else author_loss
            expected = upstream.get_scores_eap_ig(model, graph, [(clean, corrupt, answers)], metric, steps=steps, quiet=True)
        expected = expected.detach().cpu()
        model.zero_grad(set_to_none=True)
        error = float((actual - expected).abs().max())
        t.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        rows.append({"method": method, "steps": 1 if method == "eap" else steps,
                     "max_abs_error": error, "legal_edges": int(ours.legal_mask.sum()),
                     "atol": 2e-5, "rtol": 2e-4})
        print(f"{method}: {int(ours.legal_mask.sum())} legal scores; max error {error:.3g}", flush=True)
    # Unique magnitudes remove set-iteration tie ambiguity in upstream heaps.
    selector_scores = t.zeros_like(graph.scores)
    values = t.arange(1, int(ours.legal_mask.sum()) + 1, dtype=selector_scores.dtype)
    values[::2] *= -1
    selector_scores[ours.legal_mask] = values
    graph.scores[:] = selector_scores
    sizes = sorted({0, 2, 7, min(37, len(values)), min(113, len(values))})
    for count in sizes:
        graph.apply_greedy(count, absolute=True, reset=True, prune=False)
        t.testing.assert_close(learner.greedy_score_mask(ours, selector_scores, count), graph.in_graph)
    rows.append({"method": "output_frontier_greedy_selection", "tested_edge_counts": sizes,
                 "unique_magnitudes": True, "upstream_prune": False, "exact_masks_match": True})
    print(f"Greedy selector: exact author-code mask parity at sizes {sizes}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new parity result path")
    if args.gpt2:
        model = edge.load_pinned_gpt2()
        data = edge.prepare_ioi_data(model, count=8, seed=1)
        clean, corrupt, ends, answers = (data[key] for key in ("clean_tokens", "corrupt_tokens", "end_positions", "answer_ids"))
    else:
        model = small_model().to("cuda")
        clean, corrupt, ends = example_inputs()
        answers = t.tensor([[0, 1], [2, 3], [4, 5]])
    results = compare(model, clean, corrupt, ends, answers)
    result = {
        "status": "PASS", "model": "gpt2-small" if args.gpt2 else "seed4_two_layer_two_head_model",
        "model_revision": edge.GPT2_REVISION if args.gpt2 else None,
        "upstream_commit": UPSTREAM_COMMIT, "upstream_sha256": UPSTREAM_HASHES,
        "boundary_adapter": "pretokenized_inputs_right_padding_mask_and_actual_lengths_only",
        "source_sha256": hashlib.sha256(Path(learner.__file__).read_bytes()).hexdigest(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
