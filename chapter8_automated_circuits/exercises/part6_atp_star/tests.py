"""Visible semantic tests for [8.6] AtP* from Scratch."""

from __future__ import annotations

import math

import torch as t

from . import solutions as reference


def _toy_inputs():
    model = reference.ToyAtPStarModel()
    clean = reference.toy_site_tensor(reference.TOY_CLEAN_VALUES)
    corrupt = reference.toy_site_tensor(reference.TOY_CORRUPT_VALUES)
    return model, clean, corrupt


def test_toy_site_tensor(toy_site_tensor=reference.toy_site_tensor):
    tensor = toy_site_tensor(reference.TOY_CLEAN_VALUES)
    assert tensor.shape == (8,), "The site tensor must preserve all eight named patch sites."
    assert tensor.dtype == t.float64, "The exact toy oracle uses float64 to make tight checks meaningful."
    assert float(tensor[0]) == 3.0 and float(tensor[-1]) == 0.5, (
        "Named values must follow TOY_COMPONENTS order; an order mismatch invalidates every patch label."
    )
    try:
        toy_site_tensor({"L0H0.Q@dest": 3.0})
    except ValueError:
        pass
    else:
        raise AssertionError("A partial component mapping must fail instead of silently inserting defaults.")
    print("All tests in `test_toy_site_tensor` passed!")


def test_clean_corrupt_caches(cache_clean_and_corrupt=reference.cache_clean_and_corrupt):
    model, clean, corrupt = _toy_inputs()
    clean_run, corrupt_run = cache_clean_and_corrupt(model, clean, corrupt)
    assert math.isclose(float(clean_run.metric), 2.9, abs_tol=1e-7), (
        "The clean metric should be attention 2.0 + cancellation pair 0.5 + linear path 0.4."
    )
    assert math.isclose(float(corrupt_run.metric), 0.0, abs_tol=1e-9), (
        "The fully corrupt endpoint should be exactly zero."
    )
    assert t.allclose(
        clean_run.activations["attention_pattern"],
        t.tensor([1.0, 0.0], dtype=t.float64),
        atol=2e-8,
    ), "The clean softmax should be saturated on the signal value."
    assert set(clean_run.activations) >= {
        "attention_logits",
        "attention_pattern",
        "attention_output",
        "cancellation_output",
    }, "The cache must expose the nonlinear intermediates used by AtP*."
    print("All tests in `test_clean_corrupt_caches` passed!")


def test_batched_exact_patch_effects(
    batched_exact_patch_effects=reference.batched_exact_patch_effects,
):
    model, clean, corrupt = _toy_inputs()
    serial = batched_exact_patch_effects(model, clean, corrupt, component_batch_size=1)
    batched = batched_exact_patch_effects(model, clean, corrupt, component_batch_size=4)
    assert serial.keys() == batched.keys(), "Batching must not drop or rename components."
    for name in serial:
        assert math.isclose(serial[name], batched[name], abs_tol=1e-12), (
            f"Batching changed the exact intervention effect for {name}."
        )
    assert math.isclose(batched["L0H0.Q@dest"], -3.0, abs_tol=1e-6), (
        "Patching the saturated query should flip attention and remove three metric units."
    )
    assert math.isclose(batched["L0M0.cancel_source"], -0.5, abs_tol=1e-12), (
        "The cancellation source has a finite intervention effect despite zero local gradient."
    )
    assert batched["L1M1.null_source"] == 0.0, "The declared null component must remain an exact control."
    print("All tests in `test_batched_exact_patch_effects` passed!")


def test_batched_exact_patch_validation(
    batched_exact_patch_effects=reference.batched_exact_patch_effects,
):
    model, clean, corrupt = _toy_inputs()
    for bad_batch in (0, -1):
        try:
            batched_exact_patch_effects(model, clean, corrupt, component_batch_size=bad_batch)
        except ValueError:
            pass
        else:
            raise AssertionError("Non-positive component batches must fail clearly.")
    try:
        batched_exact_patch_effects(model, clean.unsqueeze(0), corrupt.unsqueeze(0))
    except ValueError:
        pass
    else:
        raise AssertionError("The one-pair toy oracle must reject an accidental example batch dimension.")
    print("All tests in `test_batched_exact_patch_validation` passed!")


def test_vanilla_attribution_patching(
    vanilla_attribution_patching=reference.vanilla_attribution_patching,
):
    model, clean, corrupt = _toy_inputs()
    scores = vanilla_attribution_patching(model, clean, corrupt)
    assert abs(scores["L0H0.Q@dest"]) < 2e-6, (
        "Vanilla AtP should expose the intended saturated-softmax false negative."
    )
    assert scores["L0M0.cancel_source"] == 0.0, (
        "The direct and nonlinear gradients should cancel exactly at the clean endpoint."
    )
    assert math.isclose(scores["L0H0.V@signal"], -2.0, abs_tol=1e-6), (
        "The linear value path is a positive control which vanilla AtP should estimate accurately."
    )
    print("All tests in `test_vanilla_attribution_patching` passed!")


def test_qk_softmax_correction(qk_softmax_correction=reference.qk_softmax_correction):
    model, clean, corrupt = _toy_inputs()
    run, _ = reference._toy_gradient_run(model, clean)
    corrected = qk_softmax_correction(run, clean, corrupt)
    exact = reference.batched_exact_patch_effects(model, clean, corrupt)
    for name in reference.TOY_QK_COMPONENTS:
        assert math.isclose(corrected[name], exact[name], abs_tol=1e-9), (
            f"Recomputing the softmax should make the {name} estimate exact in this linear downstream graph."
        )
    print("All tests in `test_qk_softmax_correction` passed!")


def test_qk_softmax_correction_requires_gradient(
    qk_softmax_correction=reference.qk_softmax_correction,
):
    model, clean, corrupt = _toy_inputs()
    run = model(clean.detach().clone().requires_grad_(True))
    try:
        qk_softmax_correction(run, clean, corrupt)
    except ValueError as exc:
        assert "grad" in str(exc), "The error should identify the missing backward-pass gradient."
    else:
        raise AssertionError("QK correction without a pattern gradient must fail explicitly.")
    print("All tests in `test_qk_softmax_correction_requires_gradient` passed!")


def test_atp_star_scores(atp_star_scores=reference.atp_star_scores):
    model, clean, corrupt = _toy_inputs()
    scores, per_drop = atp_star_scores(model, clean, corrupt)
    exact = reference.batched_exact_patch_effects(model, clean, corrupt)
    assert tuple(per_drop) == reference.TOY_GRAD_PATHS, (
        "GradDrop must expose one independently inspectable estimate per residual contribution."
    )
    for name in reference.TOY_COMPONENTS:
        assert math.isclose(scores[name], abs(exact[name]), abs_tol=1e-8), (
            f"The complete AtP* toy estimator should match the exact magnitude for {name}."
        )
    assert per_drop["cancellation"]["L0M0.cancel_source"] == -1.0, (
        "Dropping the nonlinear path should reveal the hidden direct gradient."
    )
    print("All tests in `test_atp_star_scores` passed!")


def test_atp_star_grad_path_validation(atp_star_scores=reference.atp_star_scores):
    model, clean, corrupt = _toy_inputs()
    for paths in (("attention",), ("attention", "attention")):
        try:
            atp_star_scores(model, clean, corrupt, grad_paths=paths)
        except ValueError:
            pass
        else:
            raise AssertionError("GradDrop must reject too few or duplicate gradient-control paths.")
    print("All tests in `test_atp_star_grad_path_validation` passed!")


def test_eap_style_edge_scores(eap_style_edge_scores=reference.eap_style_edge_scores):
    model, clean, corrupt = _toy_inputs()
    edges, by_component = eap_style_edge_scores(model, clean, corrupt)
    assert len(edges) == 10, "The edge baseline must expose its immediate graph-edge scores."
    assert math.isclose(by_component["L0M0.cancel_source"], 1.0, abs_tol=1e-12), (
        "Absolute edge aggregation should expose opposing direct and indirect cancellation paths."
    )
    assert by_component["L0H0.Q@dest"] < 2e-6, (
        "EAP-style local edge gradients should still fail at the saturated attention logits."
    )
    print("All tests in `test_eap_style_edge_scores` passed!")


def test_rank_components(rank_components=reference.rank_components):
    exact = {name: float(index) for index, name in enumerate(reference.TOY_COMPONENTS)}
    ranking = rank_components(exact)
    assert ranking[0] == "L1M1.null_source" and ranking[-1] == "L0H0.Q@dest", (
        "Ranking must sort by descending magnitude and preserve every named component."
    )
    try:
        rank_components({"only": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("A partial ranking must fail rather than silently changing the candidate set.")
    print("All tests in `test_rank_components` passed!")


def test_spearman_rank_correlation(
    spearman_rank_correlation=reference.spearman_rank_correlation,
):
    ascending = {name: float(i) for i, name in enumerate(reference.TOY_COMPONENTS)}
    descending = {name: float(len(reference.TOY_COMPONENTS) - i) for i, name in enumerate(reference.TOY_COMPONENTS)}
    assert math.isclose(spearman_rank_correlation(ascending, ascending), 1.0, abs_tol=1e-12), (
        "A ranking must correlate perfectly with itself."
    )
    assert spearman_rank_correlation(ascending, descending) < -0.9, (
        "Reversing a strict ranking should produce strong negative correlation."
    )
    print("All tests in `test_spearman_rank_correlation` passed!")


def test_recovery_at_k(recovery_at_k=reference.recovery_at_k):
    model, clean, corrupt = _toy_inputs()
    exact = reference.batched_exact_patch_effects(model, clean, corrupt)
    ranking = reference.rank_components(exact)
    curve = recovery_at_k(exact, ranking)
    assert all(a <= b for a, b in zip(curve, curve[1:])), (
        "Recovery@k must be cumulative and therefore monotone."
    )
    assert math.isclose(curve[-1], 1.0, abs_tol=1e-12), (
        "Selecting every component must recover all exact-effect mass."
    )
    print("All tests in `test_recovery_at_k` passed!")


def test_topk_faithfulness_is_not_additive(
    topk_faithfulness_curve=reference.topk_faithfulness_curve,
):
    model, clean, corrupt = _toy_inputs()
    exact = reference.batched_exact_patch_effects(model, clean, corrupt)
    curve = topk_faithfulness_curve(model, clean, corrupt, reference.rank_components(exact))
    assert curve[0] > 1.0, (
        "The top query patch should overshoot full-corruption effect, making non-additivity visible."
    )
    assert curve[2] < curve[1], (
        "Joint top-k faithfulness need not be monotone when components interact."
    )
    assert math.isclose(curve[-1], 1.0, abs_tol=1e-12), (
        "Patching every site must reproduce the corrupt endpoint exactly."
    )
    print("All tests in `test_topk_faithfulness_is_not_additive` passed!")


def test_toy_study_contract(run_toy_study=reference.run_toy_study):
    report = run_toy_study(component_batch_size=4)
    assert report["accepted"] is True, "The exact toy study should satisfy every declared acceptance check."
    assert report["reports"]["AtP*"]["spearman"] > 0.99, (
        "AtP* should recover the exact toy ordering."
    )
    assert report["reports"]["Vanilla AtP"]["recovery_at_k"][3] < 0.8, (
        "The baseline must visibly miss exact effect mass at the four-component budget."
    )
    assert report["reports"]["Random"]["recovery_at_k"][3] < 0.2, (
        "The fixed random-order control must fail visibly."
    )
    print("All tests in `test_toy_study_contract` passed!")


def test_grammar_pairs(build_grammar_pairs=reference.build_grammar_pairs):
    pairs = build_grammar_pairs()
    assert len(pairs) == 24, "The real path must evaluate at least twenty examples, not one prompt."
    assert {pair.family for pair in pairs} == {"copula", "possession", "agreement"}, (
        "The task should vary answer tokens so label-shuffled controls are meaningful."
    )
    assert len({(pair.clean_prompt, pair.corrupt_prompt) for pair in pairs}) == 24, (
        "Every clean/corrupt pair should be distinct."
    )
    print("All tests in `test_grammar_pairs` passed!")


def test_per_prompt_logit_diff(per_prompt_logit_diff=reference.per_prompt_logit_diff):
    logits = t.zeros(2, 3, 7)
    logits[0, -1, 2], logits[0, -1, 4] = 5.0, 1.5
    logits[1, -1, 3], logits[1, -1, 1] = -1.0, 2.0
    result = per_prompt_logit_diff(logits, t.tensor([2, 3]), t.tensor([4, 1]))
    assert t.equal(result, t.tensor([3.5, -3.0])), (
        "The real task metric must select per-example answer tokens at the final position."
    )
    try:
        per_prompt_logit_diff(logits, t.tensor([2]), t.tensor([4]))
    except ValueError:
        pass
    else:
        raise AssertionError("Metric shape mismatches must fail before producing a misleading average.")
    print("All tests in `test_per_prompt_logit_diff` passed!")


def test_real_component_names_and_parser(
    real_component_names=reference.real_component_names,
    parse_real_component=reference.parse_real_component,
):
    names = real_component_names(2, 3)
    assert len(names) == 12 and names[:2] == ("L0H0.Q@end", "L0H0.K@end"), (
        "Each layer/head must expose distinct final-position query and key sites."
    )
    assert parse_real_component("L5H7.K@end") == (5, 7, "K"), (
        "The parser must preserve layer, head, and Q/K kind."
    )
    for bad in ("L5H7", "5.7.Q", "L5H7.V@end", "L5H7.Q@start"):
        try:
            parse_real_component(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Malformed component {bad!r} must fail explicitly.")
    print("All tests in `test_real_component_names_and_parser` passed!")


def test_notebook_contract(run_smoke_test=reference.run_smoke_test):
    report = run_smoke_test()
    required = {
        "accepted",
        "exact_effects",
        "vanilla_atp",
        "atp_star",
        "grad_drop_estimates",
        "eap_edges",
        "reports",
    }
    assert required <= set(report), (
        "The notebook contract must expose the oracle, every approximation, controls, and evaluation."
    )
    assert report["accepted"] is True, "The smoke contract must represent a passing exact toy result."
    print("All tests in `test_notebook_contract` passed!")


class _LinearAttentionChain:
    """Independent scalar chain oracle with the same named hook interface.

    Forward: residual <- residual + a_l * q_l * k_l, with q_l=residual,
    k_l=1 and a=(2,3,5). All derivatives are hand-computable products.
    This is a unit-test graph, not evidence about a trained model.
    """

    def __init__(self):
        from types import SimpleNamespace
        self.cfg = SimpleNamespace(n_layers=3, n_heads=1)
        self.seen = {}

    def run_with_hooks(self, tokens, *, fwd_hooks):
        hooks = dict(fwd_hooks)
        residual = tokens[:, 0].to(t.float64)

        def apply(name, value):
            if name in hooks:
                value = hooks[name](value, None)
            self.seen[name] = value.detach().clone()
            return value

        for layer, coefficient in enumerate((2.0, 3.0, 5.0)):
            q = apply(f"blocks.{layer}.attn.hook_rot_q", residual[:, None, None, None])
            k = apply(f"blocks.{layer}.attn.hook_rot_k", t.ones_like(q))
            pattern = apply(f"blocks.{layer}.attn.hook_pattern", q * k)
            attention = apply(f"blocks.{layer}.hook_attn_out", coefficient * pattern[:, 0, 0, 0])
            mlp = apply(f"blocks.{layer}.hook_mlp_out", t.zeros_like(attention))
            residual = residual + attention + mlp
        return t.stack([residual, t.zeros_like(residual)], dim=-1)[:, None]


def _chain_task():
    return {"clean_tokens": t.tensor([[1.0]], dtype=t.float64),
            "positive_ids": t.tensor([0]), "negative_ids": t.tensor([1])}


def test_real_vanilla_gradients_preserve_later_qk_paths(vanilla_fn=None):
    fn = vanilla_fn or reference.vanilla_real_atp_scores
    model = _LinearAttentionChain()
    task = _chain_task()
    model.run_with_hooks(task["clean_tokens"], fwd_hooks=[])
    clean = {name: value for name, value in model.seen.items() if name.endswith(("hook_rot_q", "hook_rot_k"))}
    cache = {"clean_cache": clean, "corrupt_cache": {name: value - 1 for name, value in clean.items()}}
    scores = fn(model, task, cache)
    # d output / d q_0 = 2 * (1+3) * (1+5) = 48. Detaching q_1/q_2
    # incorrectly retains only the direct residual contribution, giving 2.
    expected = {"L0H0.Q@end": 48.0, "L0H0.K@end": 48.0,
                "L1H0.Q@end": 18.0, "L1H0.K@end": 54.0,
                "L2H0.Q@end": 5.0, "L2H0.K@end": 60.0}
    for name, value in expected.items():
        assert math.isclose(scores[name], value, rel_tol=0, abs_tol=1e-10), (
            f"{name}: preserve downstream Q/K paths; expected {value}, got {scores[name]}."
        )
    print("All tests in `test_real_vanilla_gradients_preserve_later_qk_paths` passed!")


def test_graddrop_only_cuts_the_requested_layer(pattern_gradients_fn=None):
    fn = pattern_gradients_fn or reference._pattern_gradients
    model = _LinearAttentionChain()
    task = _chain_task()
    gradients = fn(model, task, dropped_layer=2,
                   positive_ids=task["positive_ids"], negative_ids=task["negative_ids"])
    # Dropping layer 2 leaves the layer 0 -> layer 1 path. Its gradient is
    # 2 * (1+3) = 8, not the detached-pattern result 2.
    for layer, expected in enumerate((8.0, 3.0, 0.0)):
        actual = gradients[f"blocks.{layer}.attn.hook_pattern"]
        t.testing.assert_close(actual, t.full_like(actual, expected), rtol=0, atol=1e-10)
    print("All tests in `test_graddrop_only_cuts_the_requested_layer` passed!")


def test_real_qk_correction_matches_two_token_oracle(correction=None):
    from types import SimpleNamespace
    fn = correction or reference.qk_corrected_real_estimates
    model = SimpleNamespace(cfg=SimpleNamespace(n_layers=1, n_heads=1, d_head=1))
    q = t.ones(1, 2, 1, 1, dtype=t.float64)
    k = t.tensor([2.0, -2.0], dtype=t.float64).reshape(1, 2, 1, 1)
    q_corrupt = q.clone(); q_corrupt[:, -1] = -1
    k_corrupt = k.clone(); k_corrupt[:, -1] = 2
    pattern = t.tensor([2.0, -2.0], dtype=t.float64).softmax(-1).repeat(1, 1, 2, 1)
    gradient = t.tensor([3.0, -1.0], dtype=t.float64).repeat(1, 1, 2, 1)
    cache = {
        "clean_metric": t.zeros(1),
        "clean_cache": {"blocks.0.attn.hook_rot_q": q, "blocks.0.attn.hook_rot_k": k,
                        "blocks.0.attn.hook_pattern": pattern},
        "corrupt_cache": {"blocks.0.attn.hook_rot_q": q_corrupt, "blocks.0.attn.hook_rot_k": k_corrupt},
    }
    result = fn(model, cache, {"blocks.0.attn.hook_pattern": gradient})
    # Original first-key probability is sigmoid(4). The Q swap reverses it;
    # the K swap makes the two keys equal, yielding exactly one half.
    original_probability = 1 / (1 + math.exp(-4))
    expected_q = 4 * (1 - 2 * original_probability)
    expected_k = 4 * (0.5 - original_probability)
    assert math.isclose(float(result["L0H0.Q@end"][0]), expected_q, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(float(result["L0H0.K@end"][0]), expected_k, rel_tol=0, abs_tol=1e-12)
    print("All tests in `test_real_qk_correction_matches_two_token_oracle` passed!")


def test_real_graddrop_aggregation(aggregate=None):
    from types import SimpleNamespace
    from unittest.mock import patch
    fn = aggregate or reference.real_atp_star_scores
    model = SimpleNamespace(cfg=SimpleNamespace(n_layers=3, n_heads=1))
    task = {"clean_tokens": t.zeros(2, 1, dtype=t.long),
            "positive_ids": t.zeros(2, dtype=t.long), "negative_ids": t.ones(2, dtype=t.long)}
    names = reference.real_component_names(3, 1)

    def gradients(_model, _task, *, dropped_layer, **kwargs):
        return {"drop": dropped_layer}

    def corrected(_model, _cache, grad, **kwargs):
        drop = grad["drop"]
        return {name: t.tensor([(-1) ** drop * (drop + 1), 2 * (drop + 1)], dtype=t.float32) for name in names}

    with patch.dict(fn.__globals__, {"_pattern_gradients": gradients, "qk_corrected_real_estimates": corrected}):
        actual = fn(model, task, {})
    # |1|+|-2|+|3| = 6 and |2|+|4|+|6| = 12; divide by two, then average = 4.5.
    assert set(actual) == set(names)
    assert all(math.isclose(value, 4.5, rel_tol=0, abs_tol=1e-7) for value in actual.values()), (
        "Take absolute values per example/drop, normalize by L-1, then average examples."
    )
    print("All tests in `test_real_graddrop_aggregation` passed!")


def test_built_answers_match_canonical_and_real_runner_uses_local_methods():
    import ast
    import json
    import re
    from pathlib import Path
    from . import build_notebooks as builder

    section = Path(__file__).resolve().parent
    learner = json.loads((section / "8.6_AtP-Star_from_Scratch_exercises.ipynb").read_text())
    solved = json.loads((section / "8.6_AtP-Star_from_Scratch_solutions.ipynb").read_text())
    expected = builder.build_cells(solved=True)
    checked = 0
    for index, (cell, answer, canonical) in enumerate(zip(learner["cells"], solved["cells"], expected, strict=True)):
        assert "".join(answer["source"]) == canonical.source, f"Cell {index} must be rebuilt from the current lesson and source."
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        solution = "".join(answer["source"])
        if source == solution:
            continue
        prior = "".join(learner["cells"][index - 1]["source"])
        dropdown = re.search(r"<summary>Solution</summary>(.*?)</details>", prior, re.S)
        assert dropdown, f"Exercise cell {index} needs an adjacent solution."
        code = "\n".join(re.findall(r"```python\n(.*?)```", dropdown[1], re.S))
        answers = {node.name: node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
        tree = ast.parse(source)
        for offset, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef):
                assert node.name in answers, f"Cell {index} missing {node.name} answer."
                tree.body[offset] = answers[node.name]
        assert ast.dump(tree) == ast.dump(ast.parse(solution)), f"Cell {index} answer does not complete the actual learner."
        checked += 1
    assert checked == 11
    all_code = "\n".join("".join(cell["source"]) for cell in solved["cells"] if cell["cell_type"] == "code")
    tree = ast.parse(all_code)
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"vanilla_real_atp_scores", "_pattern_gradients", "qk_corrected_real_estimates", "real_atp_star_scores",
            "exact_real_patch_effects", "run_real_transformerlens_study"} <= definitions
    calls = [node.value for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "real" for target in node.targets)]
    assert len(calls) == 1 and isinstance(calls[0].func, ast.Name) and calls[0].func.id == "run_real_transformerlens_study"
    assert any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PLAY_TOP_K"
               for target in node.targets) for node in tree.body), "The playground must be executable code."
    print("All tests in `test_built_answers_match_canonical_and_real_runner_uses_local_methods` passed!")


def test_candidate_report_requires_cuda_and_current_inputs():
    import json
    from pathlib import Path
    from jsonschema import Draft202012Validator
    from arena_ext.verification import verify_report_input_manifest

    section = Path(__file__).resolve().parent
    report = json.loads((section / "verification_report.json").read_text())
    schema = json.loads((section / "verification_report.schema.json").read_text())
    Draft202012Validator(schema).validate(report)
    assert not report["accepted"] and not report["gpu_evidence"]["uses_cuda"], "CPU evidence does not satisfy the CUDA release contract."
    assert report["cpu_evidence"]["code_cells"] == 20
    assert verify_report_input_manifest(section.parents[2], report["report_inputs"]) == []
