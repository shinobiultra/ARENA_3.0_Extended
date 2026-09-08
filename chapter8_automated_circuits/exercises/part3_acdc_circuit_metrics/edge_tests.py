"""Independent numerical tests for the full residual-edge learner functions."""
from __future__ import annotations

import math

import pytest
import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

from . import real_edges as reference


def small_model():
    return HookedTransformer(HookedTransformerConfig(
        n_layers=2, n_heads=2, d_head=8, d_model=16, d_mlp=32, n_ctx=8,
        d_vocab=31, act_fn="gelu", normalization_type="LN", device="cpu", seed=4,
    ))


def example_inputs():
    generator = t.Generator().manual_seed(8)
    clean = t.randint(0, 31, (3, 8), generator=generator)
    return clean, clean.roll(1, 0), t.tensor([7, 5, 6])


def test_complete_residual_graph(fn=None):
    fn = fn or reference.build_residual_graph
    graph = fn(2, 2)
    assert graph.sources == ("input", "L0H0", "L0H1", "L0MLP", "L1H0", "L1H1", "L1MLP")
    assert graph.legal_mask.shape == (7, 15)
    assert int(graph.legal_mask.sum()) == 46, "Include every eligible sender, including input and MLPs."
    lookup = {receiver.name: i for i, receiver in enumerate(graph.receivers)}
    for port in ("q", "k", "v"):
        for head in (0, 1):
            assert graph.legal_mask[:, lookup[f"L0H{head}.{port}"]].nonzero().flatten().tolist() == [0]
            assert graph.legal_mask[:, lookup[f"L1H{head}.{port}"]].nonzero().flatten().tolist() == [0, 1, 2, 3]
    assert graph.legal_mask[:, lookup["L0MLP.in"]].nonzero().flatten().tolist() == [0, 1, 2]
    assert graph.legal_mask[:, lookup["L1MLP.in"]].nonzero().flatten().tolist() == [0, 1, 2, 3, 4, 5]
    assert bool(graph.legal_mask[:, -1].all()), "Final residual receives all sources."
    large = fn(12, 12)
    assert int(large.legal_mask.sum()) == 32491, "Do not restrict GPT-2 to published circuit heads."
    with pytest.raises(ValueError):
        fn(0, 2)


def test_receiver_reconstruction(fn=None):
    fn = fn or reference.reconstruct_receiver
    baseline = t.tensor([[[10., 20.], [30., 40.]]])
    deltas = t.tensor([[[[1., 2.], [3., -4.], [-5., 6.]],
                        [[7., 8.], [9., 10.], [11., -12.]]]])
    mask = t.tensor([[1., 0.], [0., 1.], [1., 1.]])
    expected = t.stack((baseline + deltas[:, :, 0] + deltas[:, :, 2],
                        baseline + deltas[:, :, 1] + deltas[:, :, 2]), dim=2)
    untouched = baseline.clone(), deltas.clone(), mask.clone()
    t.testing.assert_close(fn(baseline, deltas, mask), expected)
    t.testing.assert_close(fn(baseline, deltas, mask[:, 0]), expected[:, :, 0])
    t.testing.assert_close(fn(baseline, deltas, t.zeros(3)), baseline)
    # Continuous weights are useful for later interpolation; no boolean coercion.
    t.testing.assert_close(fn(baseline, deltas, t.tensor([.5, 0., 0.])), baseline + .5 * deltas[:, :, 0])
    for actual, original in zip((baseline, deltas, mask), untouched, strict=True):
        t.testing.assert_close(actual, original)
    with pytest.raises(ValueError):
        fn(baseline, deltas, t.zeros(4))


def test_end_positions_and_metrics(logit_fn=None, difference_fn=None, kl_fn=None):
    logit_fn = logit_fn or reference.final_position_logits
    difference_fn = difference_fn or reference.answer_logit_differences
    kl_fn = kl_fn or reference.clean_kl_divergence
    model = small_model()
    clean, _, ends = example_inputs()
    with t.inference_mode():
        residual = model(clean, stop_at_layer=2)
        expected = model(clean)[t.arange(3), ends]
        t.testing.assert_close(logit_fn(model, residual, ends), expected, atol=2e-6, rtol=2e-6)
    logits = t.tensor([[3., 0., 1.], [0., 4., 2.]])
    answers = t.tensor([[0, 2], [2, 1]])
    t.testing.assert_close(difference_fn(logits, answers), t.tensor([2., -2.]))
    clean_logits = t.tensor([[math.log(.5), math.log(.3), math.log(.2)],
                             [math.log(.2), math.log(.3), math.log(.5)]], dtype=t.float64)
    alternate = t.log(t.tensor([[.2, .5, .3], [.3, .5, .2]], dtype=t.float64))
    exact = sum(p * math.log(p / q) for p, q in zip((.5,.3,.2,.2,.3,.5),(.2,.5,.3,.3,.5,.2))) / 2
    assert float(kl_fn(alternate, clean_logits)) == pytest.approx(exact)
    assert abs(float(kl_fn(clean_logits, clean_logits))) < 1e-12


def _slow_edge_oracle(model, graph, clean, corrupt, ends, active, *, freeze_senders=False):
    """Independent explicit per-edge sums; do not call the vectorized reconstructor."""
    with t.inference_mode():
        _, corrupt_cache = model.run_with_cache(corrupt)
        _, clean_cache = model.run_with_cache(clean)
        corrupt_sources = {"input": corrupt_cache["blocks.0.hook_resid_pre"]}
        frozen_sources = {"input": clean_cache["blocks.0.hook_resid_pre"]}
        for layer in range(graph.n_layers):
            for head in range(graph.n_heads):
                corrupt_sources[f"L{layer}H{head}"] = corrupt_cache[f"blocks.{layer}.attn.hook_result"][:, :, head]
                frozen_sources[f"L{layer}H{head}"] = clean_cache[f"blocks.{layer}.attn.hook_result"][:, :, head]
            corrupt_sources[f"L{layer}MLP"] = corrupt_cache[f"blocks.{layer}.hook_mlp_out"]
            frozen_sources[f"L{layer}MLP"] = clean_cache[f"blocks.{layer}.hook_mlp_out"]
        current = {}
        hooks = []

        def remember_input(value, hook):
            current["input"] = value.clone()
            return value
        hooks.append(("blocks.0.hook_resid_pre", remember_input))
        for layer in range(graph.n_layers):
            def remember_heads(value, hook, layer=layer):
                for head in range(graph.n_heads):
                    current[f"L{layer}H{head}"] = value[:, :, head].clone()
                return value
            def remember_mlp(value, hook, layer=layer):
                current[f"L{layer}MLP"] = value.clone()
                return value
            hooks.extend(((f"blocks.{layer}.attn.hook_result", remember_heads),
                          (f"blocks.{layer}.hook_mlp_out", remember_mlp)))
        for column, receiver in enumerate(graph.receivers):
            def patch(value, hook, column=column, receiver=receiver):
                rebuilt = corrupt_cache[receiver.baseline_hook].clone()
                for row, source in enumerate(graph.sources[:receiver.source_count]):
                    if active[row, column]:
                        sender = frozen_sources[source] if freeze_senders else current[source]
                        rebuilt = rebuilt + sender - corrupt_sources[source]
                if receiver.head is None:
                    return rebuilt
                value = value.clone()
                value[:, :, receiver.head] = rebuilt
                return value
            hooks.append((receiver.hook_name, patch))
        logits = model.run_with_hooks(clean, fwd_hooks=hooks)
        return logits[t.arange(len(clean)), ends]


def test_real_edge_endpoints(cls=None):
    cls = cls or reference.EdgeIntervention
    model = small_model()
    clean, corrupt, ends = example_inputs()
    engine = cls(model, clean, corrupt, ends)
    full = engine.graph.legal_mask.clone()
    for mask, expected in ((full, engine.clean_logits), (t.zeros_like(full), engine.corrupt_logits)):
        t.testing.assert_close(engine(mask), expected, atol=1e-5, rtol=1e-5)
    assert not any(h.fwd_hooks for h in model.hook_dict.values()), "Temporary edge hooks must be removed."


def test_single_edge_receiver_isolation(cls=None):
    cls = cls or reference.EdgeIntervention
    model = small_model()
    clean, corrupt, ends = example_inputs()
    engine = cls(model, clean, corrupt, ends)
    graph = engine.graph
    lookup = {r.name: i for i, r in enumerate(graph.receivers)}
    with t.inference_mode():
        _, clean_cache = model.run_with_cache(clean)
        _, corrupt_cache = model.run_with_cache(corrupt)
        for source, receiver_name in ((0, "L0H0.q"), (1, "L1H1.k"),
                                      (2, "L1H0.v"), (1, "L0MLP.in"), (3, "output")):
            column = lookup[receiver_name]
            receiver = graph.receivers[column]
            source_name = graph.sources[source]
            if source_name == "input":
                delta = corrupt_cache["blocks.0.hook_resid_pre"] - clean_cache["blocks.0.hook_resid_pre"]
            elif "MLP" in source_name:
                layer = int(source_name[1:source_name.index("MLP")])
                delta = corrupt_cache[f"blocks.{layer}.hook_mlp_out"] - clean_cache[f"blocks.{layer}.hook_mlp_out"]
            else:
                layer, head = map(int, source_name[1:].split("H"))
                delta = (corrupt_cache[f"blocks.{layer}.attn.hook_result"] - clean_cache[f"blocks.{layer}.attn.hook_result"])[:, :, head]
            def manual_patch(value, hook):
                patched = value.clone()
                if receiver.head is None:
                    patched += delta
                else:
                    patched[:, :, receiver.head] += delta
                return patched
            expected = model.run_with_hooks(clean, fwd_hooks=[(receiver.hook_name, manual_patch)])[t.arange(3), ends]
            mask = graph.legal_mask.clone()
            mask[source, column] = False
            t.testing.assert_close(engine(mask), expected, atol=1e-5, rtol=1e-5)


def test_hybrid_senders_and_slow_oracle(cls=None):
    cls = cls or reference.EdgeIntervention
    model = small_model()
    clean, corrupt, ends = example_inputs()
    engine = cls(model, clean, corrupt, ends)
    graph = engine.graph
    generator = t.Generator().manual_seed(99)
    for _ in range(4):
        mask = (t.rand(graph.legal_mask.shape, generator=generator) > .4) & graph.legal_mask
        expected = _slow_edge_oracle(model, graph, clean, corrupt, ends, mask)
        t.testing.assert_close(engine(mask), expected, atol=2e-5, rtol=2e-5)
    mask = graph.legal_mask.clone()
    mask[0, 0] = False
    # A changed early head is still transmitted to later receivers through retained edges.
    actual = engine(mask)
    frozen = _slow_edge_oracle(model, graph, clean, corrupt, ends, mask, freeze_senders=True)
    assert float((actual - frozen).abs().max()) > .01, "Fixture must reject frozen-clean sender reuse."


def test_edge_hook_cleanup_on_failure(cls=None):
    cls = cls or reference.EdgeIntervention
    model = small_model()
    clean, corrupt, ends = example_inputs()
    engine = cls(model, clean, corrupt, ends)
    def fail(value, hook):
        raise RuntimeError("deliberate receiver failure")
    with model.hooks(fwd_hooks=[("blocks.1.hook_mlp_in", fail)]):
        before = {name: len(hook.fwd_hooks) for name, hook in model.hook_dict.items()}
        with pytest.raises(RuntimeError, match="deliberate receiver failure"):
            engine(engine.graph.legal_mask)
        assert before == {name: len(hook.fwd_hooks) for name, hook in model.hook_dict.items()}
    t.testing.assert_close(engine(engine.graph.legal_mask), engine.clean_logits, atol=1e-5, rtol=1e-5)


def test_reachability(fn=None):
    fn = fn or reference.output_reachable_sources
    graph = reference.build_residual_graph(2, 2)
    assert fn(graph, graph.legal_mask) == set(range(7))
    assert fn(graph, t.zeros_like(graph.legal_mask)) == set()
    mask = t.zeros_like(graph.legal_mask)
    lookup = {receiver.name: i for i, receiver in enumerate(graph.receivers)}
    mask[5, lookup["output"]] = True
    mask[1, lookup["L1H1.v"]] = True
    mask[0, lookup["L0H0.q"]] = True
    assert fn(graph, mask) == {0, 1, 5}, "Walk fixed port-to-component dependencies as well as retained additive edges."


def test_greedy_edge_recomputation(fn=None):
    fn = fn or reference.greedy_edge_acdc
    graph = reference.build_residual_graph(1, 1)
    calls = []
    def loss(mask):
        calls.append(mask.clone())
        primary = float(mask[0, 0] and mask[1, 4])
        backup = float(mask[0, 3] and mask[2, 4])
        return (1.2 - primary - .2 * backup) ** 2
    result = fn(graph, loss, threshold=.01, progress_every=1000)
    mask = reference.mask_from_result(graph, result)
    expected = t.zeros_like(mask)
    expected[0, 0] = expected[1, 4] = expected[0, 3] = expected[2, 4] = True
    t.testing.assert_close(mask, expected)
    assert result["retained_edges"] == 4 and result["final_loss"] == pytest.approx(0)
    current = graph.legal_mask.clone()
    # Replay every recorded trial; a static full-graph score cannot satisfy this.
    for row in result["trace"]:
        assert row["before_loss"] == pytest.approx(loss(current))
        trial = current.clone()
        trial[row["source"], row["receiver"]] = False
        assert row["trial_loss"] == pytest.approx(loss(trial))
        if row["removed"]:
            current = trial
    with pytest.raises(ValueError):
        fn(graph, lambda mask: float("nan"), threshold=.01)


def test_broken_receiver_and_metric_rejected():
    with pytest.raises(AssertionError):
        test_receiver_reconstruction(lambda baseline, deltas, mask: baseline)
    with pytest.raises(AssertionError):
        test_end_positions_and_metrics(difference_fn=lambda logits, answers: logits.max(1).values)


@pytest.mark.requires_model
def test_original_ioi_boundary_excludes_answer():
    from types import SimpleNamespace
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", revision=reference.GPT2_REVISION,
                                             local_files_only=True)
    tokenizer.add_bos_token = False
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    data = reference.prepare_ioi_data(SimpleNamespace(tokenizer=tokenizer), count=8, seed=41)
    for row, prompt in enumerate(data["prompts"]):
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        end = int(data["end_positions"][row])
        assert end == len(prefix) - 1, "Score the final prefix token, not the answer or padding."
        assert tokenizer.decode([prefix[-1]]) == " to", "The generated prompt ends before its IO answer."
        assert data["clean_tokens"][row, :end + 1].tolist() == prefix
        assert bool((data["clean_tokens"][row, end + 1:] == tokenizer.pad_token_id).all())
        assert data["answer_ids"][row, 0] != data["answer_ids"][row, 1]


@pytest.mark.requires_model
def test_second_subject_control_preserves_first_names():
    from types import SimpleNamespace
    from transformers import AutoTokenizer
    from .ioi_controls import prepare_ioi_second_subject_data

    tokenizer = AutoTokenizer.from_pretrained("gpt2", revision=reference.GPT2_REVISION,
                                             local_files_only=True)
    tokenizer.add_bos_token = False
    tokenizer.pad_token = tokenizer.eos_token
    model = SimpleNamespace(tokenizer=tokenizer)
    all_names = reference.prepare_ioi_data(model, count=8, seed=41)
    second = prepare_ioi_second_subject_data(model, count=8, seed=41)
    for key in ("clean_tokens", "end_positions", "answer_ids"):
        t.testing.assert_close(second[key], all_names[key])
    for row, (io_id, subject_id) in enumerate(second["answer_ids"].tolist()):
        changed = (second["clean_tokens"][row] != second["corrupt_tokens"][row]).nonzero().flatten()
        assert changed.tolist() == [second["changed_positions"][row]]
        assert second["clean_tokens"][row, changed[0]] == subject_id
        assert int(second["corrupt_tokens"][row, changed[0]]) not in (io_id, subject_id)
        assert int((second["corrupt_tokens"][row] == io_id).sum()) == 1
        assert int((second["corrupt_tokens"][row] == subject_id).sum()) == 1


def test_checkpoint_resume_preserves_search(tmp_path):
    import json

    graph = reference.build_residual_graph(1, 1)
    def loss(mask):
        primary = float(mask[0, 0] and mask[1, 4])
        backup = float(mask[0, 3] and mask[2, 4])
        return (1.2 - primary - .2 * backup) ** 2
    expected = reference.greedy_edge_acdc(graph, loss, threshold=.01, progress_every=1000)
    calls = 0
    def interrupted(mask):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("simulated process interruption")
        return loss(mask)
    path = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        reference.greedy_edge_acdc(graph, interrupted, threshold=.01,
                                   progress_every=1, checkpoint_path=path)
    resumed = reference.greedy_edge_acdc(graph, loss, threshold=.01,
                                         progress_every=1000, resume=json.loads(path.read_text()))
    assert resumed["active_edges"] == expected["active_edges"], "Resume must preserve the exact circuit."
    assert resumed["trace"] == expected["trace"], "Do not repeat or skip accepted/retained decisions."
    assert resumed["skipped"] == expected["skipped"]
