"""Tests for [8.1] Activation Patching Refresher.

Wherever possible these use an independent oracle: TransformerLens's own `patching` module, its
`stack_head_results` direct logit attribution, the exact linear formula for a direct effect, or a
small hand-worked example.
"""

import contextlib
from typing import Callable

import pandas as pd
import torch as t
import transformer_lens.patching as tl_patching
from transformer_lens import ActivationCache, HookedTransformer

# ----------------------------------------------------------------------------------------------------------------
# Helpers (independent of the student's code)
# ----------------------------------------------------------------------------------------------------------------

_CLEAN = [
    "When Mary and John went to the store, John gave a drink to",
    "When Tom and Anna went to the park, Anna gave a book to",
    "After Kate and Paul got to the beach, Paul handed a ball to",
]
_CORRUPT = [
    "When Mary and John went to the store, Lisa gave a drink to",
    "When Tom and Anna went to the park, Mark gave a book to",
    "After Kate and Paul got to the beach, Ryan handed a ball to",
]
_ANSWERS = [(" Mary", " John"), (" Tom", " Anna"), (" Kate", " Paul")]


@contextlib.contextmanager
def _quiet_tl_patching():
    """TransformerLens's patching functions print a progress bar; silence it while testing."""
    original = tl_patching.tqdm
    tl_patching.tqdm = lambda iterable, *args, **kwargs: iterable
    try:
        yield
    finally:
        tl_patching.tqdm = original


def _small_task_parts(model: HookedTransformer):
    clean = model.to_tokens(_CLEAN)
    corrupt = model.to_tokens(_CORRUPT)
    answers = t.tensor([[model.to_single_token(a), model.to_single_token(b)] for a, b in _ANSWERS], device=clean.device)

    def metric(logits):  # per-prompt logit difference, written independently of the student's version
        final = logits[:, -1]
        return final[t.arange(len(answers)), answers[:, 0]] - final[t.arange(len(answers)), answers[:, 1]]

    return clean, corrupt, answers, metric


def _make_task(PatchingTask, model):
    clean, corrupt, answers, metric = _small_task_parts(model)
    labels = [f"p{i}" for i in range(clean.shape[1])]
    return PatchingTask(name="test", clean_tokens=clean, corrupted_tokens=corrupt, metric=metric, labels=labels), answers


def _direct_effect_oracle(model, clean_cache, corrupt_cache, heads, answers):
    """Exact direct effect on the mean logit diff of replacing these heads' z (all positions) with corrupted values."""
    direction = (model.W_U[:, answers[:, 0]] - model.W_U[:, answers[:, 1]]).T  # [batch d_model]
    total = 0.0
    for layer, head in heads:
        dz = corrupt_cache["z", layer][:, -1, head] - clean_cache["z", layer][:, -1, head]
        dout = dz @ model.W_O[layer, head]
        dout = dout - dout.mean(-1, keepdim=True)  # the final LayerNorm centres its input
        total = total + ((dout / clean_cache["ln_final.hook_scale"][:, -1]) * direction).sum(-1).mean()
    return float(total)


# ----------------------------------------------------------------------------------------------------------------
# Part 1
# ----------------------------------------------------------------------------------------------------------------


def test_ioi_prompt(ioi_prompt: Callable, IOIExample: type):
    ex = IOIExample("When {A} and {B} went to the store, {C} gave a drink to", "Mary", "John", "Tom", ("Ann", "Bob", "Kate"))
    expected = {
        "ABB": "When Mary and John went to the store, John gave a drink to",
        "ABA": "When Mary and John went to the store, Mary gave a drink to",
        "ABC": "When Mary and John went to the store, Tom gave a drink to",
        "XYZ": "When Ann and Bob went to the store, Kate gave a drink to",
    }
    for scheme, prompt in expected.items():
        got = ioi_prompt(ex, scheme)
        assert got == prompt, f"Scheme {scheme!r}: expected {prompt!r}, got {got!r}"
    # BABA-ordered template: the slots, not the positions, define the roles
    baba = IOIExample("When {B} and {A} went to the park, {C} gave a ball to", "Mary", "John", "Tom", ("Ann", "Bob", "Kate"))
    assert ioi_prompt(baba, "ABB") == "When John and Mary went to the park, John gave a ball to", (
        "Fill the slots by name ({A} = io, {B} = s, {C} = s for ABB), not by position in the sentence"
    )
    print("All tests in `test_ioi_prompt` passed!")


def test_logit_diff(logit_diff: Callable):
    t.manual_seed(0)
    logits = t.randn(2, 4, 10)
    logits[:, :-1] = 100.0  # earlier positions must be ignored
    answers = t.tensor([[1, 2], [7, 3]])
    result = logit_diff(logits, answers)
    assert result.shape == (2,), f"Expected one value per prompt, shape (2,), got {tuple(result.shape)}"
    expected = t.stack([logits[0, -1, 1] - logits[0, -1, 2], logits[1, -1, 7] - logits[1, -1, 3]])
    t.testing.assert_close(result, expected, msg="Should be logit[correct] - logit[incorrect] at the final position")
    print("All tests in `test_logit_diff` passed!")


def test_patching_score(patching_score: Callable):
    assert abs(patching_score(0.5, 0.0, 1.0) - 0.5) < 1e-9
    assert abs(patching_score(2.0, 1.0, 3.0) - 0.5) < 1e-9
    # Noising direction: base = clean (3.0), source = corrupted (1.0); patched all the way to corrupted -> 1
    assert abs(patching_score(1.0, 3.0, 1.0) - 1.0) < 1e-9, "Patching all the way to the source should give 1"
    assert abs(patching_score(3.0, 3.0, 1.0)) < 1e-9, "No change from the base should give 0"
    assert patching_score(4.0, 3.0, 1.0) < 0, "Moving away from the source should give a negative score"
    x = t.tensor([[1.0, 2.0], [3.0, 5.0]])
    t.testing.assert_close(patching_score(x, 1.0, 5.0), (x - 1.0) / 4.0, msg="Should work elementwise on tensors")
    print("All tests in `test_patching_score` passed!")


def test_patch_run(patch_run: Callable, model: HookedTransformer):
    clean, corrupt, answers, _ = _small_task_parts(model)
    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupt_logits = model(corrupt)

    # Patching every position of resid_pre at layer 0 turns the corrupted run into the clean run
    hook_name = "blocks.0.hook_resid_pre"
    out = patch_run(model, corrupt, hook_name, clean_cache[hook_name], (slice(None), slice(None)))
    assert out.shape == clean_logits.shape, f"Should return full logits {tuple(clean_logits.shape)}, got {tuple(out.shape)}"
    t.testing.assert_close(out, clean_logits, atol=1e-3, rtol=1e-3, msg="Patching all of resid_pre at layer 0 should give the source run's logits")

    # Patching a single head at all positions agrees with TransformerLens's own head patching
    token = int(answers[0, 0])
    source = clean_cache["blocks.5.attn.hook_z"].clone()
    out = patch_run(model, corrupt, "blocks.5.attn.hook_z", source, (slice(None), slice(None), 3))
    with _quiet_tl_patching():
        oracle = tl_patching.get_act_patch_attn_head_out_all_pos(model, corrupt, clean_cache, lambda lg: lg[0, -1, token])
    t.testing.assert_close(out[0, -1, token], oracle[5, 3], atol=1e-3, rtol=1e-3, msg="Head patch disagrees with transformer_lens.patching")
    t.testing.assert_close(source, clean_cache["blocks.5.attn.hook_z"], msg="patch_run modified the source activation!")

    # Patching a position where clean and corrupted are identical (position 3 = ' and') at layer 0 changes nothing
    out = patch_run(model, corrupt, hook_name, clean_cache[hook_name], (slice(None), 3))
    t.testing.assert_close(out, corrupt_logits, atol=1e-3, rtol=1e-3, msg="Patching an identical activation should change nothing")

    # No hooks left behind
    t.testing.assert_close(model(corrupt), corrupt_logits, atol=1e-4, rtol=1e-4, msg="patch_run left a hook attached")
    print("All tests in `test_patch_run` passed!")


def test_layer_position_sweep(layer_position_sweep: Callable, PatchingTask: type, model: HookedTransformer):
    task, _ = _make_task(PatchingTask, model)
    clean_logits, clean_cache = model.run_with_cache(task.clean_tokens)
    corrupt_logits, corrupt_cache = model.run_with_cache(task.corrupted_tokens)
    m_clean, m_corr = task.metric(clean_logits).mean(), task.metric(corrupt_logits).mean()

    with _quiet_tl_patching():
        # Denoising resid_pre
        result = layer_position_sweep(model, task, "resid_pre", "denoise")
        assert result.shape == (model.cfg.n_layers, task.seq_len), f"Expected (n_layers, seq), got {tuple(result.shape)}"
        oracle = tl_patching.get_act_patch_resid_pre(
            model, task.corrupted_tokens, clean_cache, lambda lg: (task.metric(lg).mean() - m_corr) / (m_clean - m_corr)
        )
        t.testing.assert_close(result.cpu(), oracle.cpu(), atol=2e-3, rtol=1e-3, msg="Denoising resid_pre sweep disagrees with transformer_lens.patching")

        # Noising mlp_out
        result = layer_position_sweep(model, task, "mlp_out", "noise")
        oracle = tl_patching.get_act_patch_mlp_out(
            model, task.clean_tokens, corrupt_cache, lambda lg: (task.metric(lg).mean() - m_clean) / (m_corr - m_clean)
        )
        t.testing.assert_close(result.cpu(), oracle.cpu(), atol=2e-3, rtol=1e-3, msg="Noising mlp_out sweep disagrees with transformer_lens.patching")

        # Shuffled source, still normalised with the unshuffled metrics
        perm = t.tensor([1, 2, 0])
        result = layer_position_sweep(model, task, "attn_out", "denoise", source_perm=perm)
        permuted_cache = ActivationCache({k: v[perm.to(v.device)] for k, v in clean_cache.items()}, model)
        oracle = tl_patching.get_act_patch_attn_out(
            model, task.corrupted_tokens, permuted_cache, lambda lg: (task.metric(lg).mean() - m_corr) / (m_clean - m_corr)
        )
        t.testing.assert_close(result.cpu(), oracle.cpu(), atol=2e-3, rtol=1e-3, msg="Shuffled (source_perm) sweep is wrong")
    print("All tests in `test_layer_position_sweep` passed!")


# ----------------------------------------------------------------------------------------------------------------
# Part 2
# ----------------------------------------------------------------------------------------------------------------


def test_head_sweep(head_sweep: Callable, model: HookedTransformer):
    clean, corrupt, _, metric = _small_task_parts(model)
    _, corrupt_cache = model.run_with_cache(corrupt)
    source_z = [corrupt_cache["z", layer] for layer in range(model.cfg.n_layers)]
    result = head_sweep(model, clean, source_z, metric)
    assert result.shape == (model.cfg.n_layers, model.cfg.n_heads), f"Expected (n_layers, n_heads), got {tuple(result.shape)}"
    with _quiet_tl_patching():
        oracle = tl_patching.get_act_patch_attn_head_out_all_pos(model, clean, corrupt_cache, lambda lg: metric(lg).mean())
    t.testing.assert_close(result.cpu(), oracle.cpu(), atol=2e-3, rtol=1e-3, msg="Head sweep (mean metric) disagrees with transformer_lens.patching")
    print("All tests in `test_head_sweep` passed!")


def test_head_position_sweep(head_position_sweep: Callable, model: HookedTransformer):
    clean, corrupt, _, metric = _small_task_parts(model)
    _, corrupt_cache = model.run_with_cache(corrupt)
    source_z = [corrupt_cache["z", layer] for layer in range(model.cfg.n_layers)]
    heads = ["9.9", "3.0"]
    result = head_position_sweep(model, clean, source_z, metric, heads)
    seq = clean.shape[1]
    assert result.shape == (2, seq), f"Expected (len(heads), seq) = (2, {seq}), got {tuple(result.shape)}"
    index_df = pd.DataFrame(
        [{"layer": int(h.split(".")[0]), "pos": pos, "head": int(h.split(".")[1])} for h in heads for pos in range(seq)]
    )
    with _quiet_tl_patching():
        oracle = tl_patching.get_act_patch_attn_head_out_by_pos(
            model, clean, corrupt_cache, lambda lg: metric(lg).mean(), index_axis_names=None, index_df=index_df
        )
    t.testing.assert_close(result.cpu().flatten(), oracle.cpu().flatten(), atol=2e-3, rtol=1e-3, msg="Head-by-position sweep disagrees with transformer_lens.patching")
    print("All tests in `test_head_position_sweep` passed!")


# ----------------------------------------------------------------------------------------------------------------
# Part 3
# ----------------------------------------------------------------------------------------------------------------


def test_ablation_source(ablation_source: Callable):
    t.manual_seed(0)
    z = t.randn(7, 3, 2, 4)
    zero = ablation_source(z, "zero")
    assert zero.shape == z.shape and (zero == 0).all(), "Zero ablation should return zeros of the same shape"

    mean = ablation_source(z, "mean")
    assert mean.shape == z.shape, f"Mean ablation should return shape {tuple(z.shape)}, got {tuple(mean.shape)}"
    for i in range(7):
        t.testing.assert_close(mean[i], z.mean(0), msg="Every prompt should get the batch mean (per position)")

    res = ablation_source(z, "resample", seed=3)
    assert res.shape == z.shape
    sources = []
    for i in range(7):
        matches = [j for j in range(7) if t.equal(res[i], z[j])]
        assert len(matches) == 1, f"Resampled row {i} should be exactly one of the original rows"
        assert matches[0] != i, f"Prompt {i} was resampled with itself: the permutation must have no fixed points"
        sources.append(matches[0])
    assert sorted(sources) == list(range(7)), "Resampling should be a permutation (each prompt used exactly once)"
    t.testing.assert_close(ablation_source(z, "resample", seed=3), res, msg="Resampling should be deterministic given the seed")
    print("All tests in `test_ablation_source` passed!")


# ----------------------------------------------------------------------------------------------------------------
# Part 4
# ----------------------------------------------------------------------------------------------------------------


def test_run_with_frozen(run_with_frozen: Callable, all_nodes: Callable, frozen_run_cache: Callable, model: HookedTransformer):
    clean, corrupt, answers, metric = _small_task_parts(model)
    clean_logits = model(clean)
    clean_cache = frozen_run_cache(model, clean)
    full_clean_cache = model.run_with_cache(clean)[1]
    corrupt_cache = frozen_run_cache(model, corrupt)

    # Nothing patched, nothing recomputed: the clean run
    out = run_with_frozen(model, clean, clean_cache, {}, set())
    t.testing.assert_close(out, clean_logits, atol=1e-3, rtol=1e-3, msg="With no patches, a frozen run should equal the base run")

    # Everything recomputed: an ordinary patched run
    patches = {"9.9": corrupt_cache["z", 9][:, :, 9]}

    def manual_hook(z, hook):
        z[:, :, 9] = corrupt_cache["z", 9][:, :, 9]
        return z

    expected = model.run_with_hooks(clean, fwd_hooks=[("blocks.9.attn.hook_z", manual_hook)])
    out = run_with_frozen(model, clean, clean_cache, patches, all_nodes(model))
    t.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3, msg="With recompute = all_nodes, should equal an ordinary patched run")

    # Nothing recomputed: the direct effect, which is exactly linear
    out = run_with_frozen(model, clean, clean_cache, patches, set())
    got = (metric(out) - metric(clean_logits)).mean().item()
    oracle = _direct_effect_oracle(model, full_clean_cache, corrupt_cache, [(9, 9)], answers)
    assert abs(got - oracle) < 2e-3, (
        f"Direct effect of 9.9 should be {oracle:.4f} (its direct logit attribution with the final LN scale frozen), got {got:.4f}. "
        "Did you freeze ln_final.hook_scale and every MLP?"
    )
    # Recomputing only the final LayerNorm should give a (slightly) different answer
    out_ln = run_with_frozen(model, clean, clean_cache, patches, {"ln_final"})
    assert not t.allclose(out_ln, out, atol=1e-4), "Putting 'ln_final' in recompute should let the LayerNorm scale recompute"

    t.testing.assert_close(model(clean), clean_logits, atol=1e-4, rtol=1e-4, msg="run_with_frozen left a hook attached")
    print("All tests in `test_run_with_frozen` passed!")


def test_total_and_direct_effect(total_and_direct_effect: Callable, PatchingTask: type, model: HookedTransformer):
    import part1_activation_patching_refresher.solutions as solutions

    task, answers = _make_task(PatchingTask, model)
    clean_logits, full_clean_cache = model.run_with_cache(task.clean_tokens)
    _, full_corrupt_cache = model.run_with_cache(task.corrupted_tokens)
    clean_cache = solutions.frozen_run_cache(model, task.clean_tokens)
    corrupt_cache = solutions.frozen_run_cache(model, task.corrupted_tokens)
    m_clean = task.metric(clean_logits).mean()

    te, de = total_and_direct_effect(model, task, clean_cache, corrupt_cache, ["10.0"])
    assert isinstance(te, float) and isinstance(de, float), "Should return two floats"
    with _quiet_tl_patching():
        te_grid = tl_patching.get_act_patch_attn_head_out_all_pos(
            model, task.clean_tokens, full_corrupt_cache, lambda lg: task.metric(lg).mean() - m_clean
        )
    assert abs(te - te_grid[10, 0].item()) < 2e-3, f"Total effect of 10.0 should be {te_grid[10, 0].item():.4f}, got {te:.4f}"
    de_oracle = _direct_effect_oracle(model, full_clean_cache, full_corrupt_cache, [(10, 0)], answers)
    assert abs(de - de_oracle) < 2e-3, f"Direct effect of 10.0 should be {de_oracle:.4f}, got {de:.4f}"

    # Direct effects of a set of heads add up (the frozen run is linear in the patched outputs)
    _, de_pair = total_and_direct_effect(model, task, clean_cache, corrupt_cache, ["10.0", "9.9"])
    de_pair_oracle = _direct_effect_oracle(model, full_clean_cache, full_corrupt_cache, [(10, 0), (9, 9)], answers)
    assert abs(de_pair - de_pair_oracle) < 2e-3, f"Joint direct effect of 10.0 and 9.9 should be {de_pair_oracle:.4f}, got {de_pair:.4f}"
    print("All tests in `test_total_and_direct_effect` passed!")


def test_path_restricted_effects(
    path_restricted_effects: Callable, all_nodes: Callable, frozen_run_cache: Callable, PatchingTask: type, model: HookedTransformer
):
    task, answers = _make_task(PatchingTask, model)
    clean_logits, full_clean_cache = model.run_with_cache(task.clean_tokens)
    _, full_corrupt_cache = model.run_with_cache(task.corrupted_tokens)
    clean_cache, corrupt_cache = frozen_run_cache(model, task.clean_tokens), frozen_run_cache(model, task.corrupted_tokens)
    m_clean = task.metric(clean_logits).mean()

    routes = {"everything": all_nodes(model), "nothing": set(), "via 9.9": {"9.9"}}
    result = path_restricted_effects(model, task, clean_cache, corrupt_cache, "8.10", routes)
    assert set(result) == set(routes), f"Should return one value per route, got keys {list(result)}"
    with _quiet_tl_patching():
        te_grid = tl_patching.get_act_patch_attn_head_out_all_pos(
            model, task.clean_tokens, full_corrupt_cache, lambda lg: task.metric(lg).mean() - m_clean
        )
    assert abs(result["everything"] - te_grid[8, 10].item()) < 2e-3, "Recomputing everything should give the total effect"
    de_oracle = _direct_effect_oracle(model, full_clean_cache, full_corrupt_cache, [(8, 10)], answers)
    assert abs(result["nothing"] - de_oracle) < 2e-3, "Recomputing nothing should give the direct effect"
    assert abs(result["via 9.9"] - result["nothing"]) > 1e-4, "Letting 9.9 recompute should change the effect of 8.10"
    print("All tests in `test_path_restricted_effects` passed!")


def test_head_direct_contributions(head_direct_contributions: Callable, model: HookedTransformer):
    clean, _, answers, metric = _small_task_parts(model)
    logits, cache = model.run_with_cache(clean)
    result = head_direct_contributions(model, cache, answers)
    assert result.shape == (model.cfg.n_layers, model.cfg.n_heads), f"Expected (n_layers, n_heads), got {tuple(result.shape)}"
    # Oracle: TransformerLens's stack_head_results with the final LayerNorm applied
    stacked = cache.stack_head_results(layer=-1, pos_slice=-1, apply_ln=True)  # [144 batch d_model]
    direction = (model.W_U[:, answers[:, 0]] - model.W_U[:, answers[:, 1]]).T  # [batch d_model]
    oracle = (stacked * direction).sum(-1).mean(-1).reshape(model.cfg.n_layers, model.cfg.n_heads)
    t.testing.assert_close(result.cpu().float(), oracle.cpu().float(), atol=2e-3, rtol=1e-3, msg="Disagrees with cache.stack_head_results(apply_ln=True)")
    # It should only need hook_z and the final LN scale
    small_cache = ActivationCache({k: v for k, v in cache.items() if k.endswith(("hook_z", "ln_final.hook_scale"))}, model)
    t.testing.assert_close(head_direct_contributions(model, small_cache, answers).cpu(), result.cpu())
    print("All tests in `test_head_direct_contributions` passed!")


# ----------------------------------------------------------------------------------------------------------------
# Part 5
# ----------------------------------------------------------------------------------------------------------------


def test_year_prob_diff(year_prob_diff: Callable):
    vocab = 300
    year_tokens = t.arange(100) + 150
    logits = t.full((4, 3, vocab), -1e4)
    logits[:, :-1, 0] = 1e4  # earlier positions must be ignored
    logits[0, -1, 150 + 50] = 10.0  # all mass on year 50
    logits[1, -1, 150 + 50] = 10.0
    logits[2, -1, 150 + 50] = 10.0
    logits[3, -1, 150 + 80] = 0.0  # half the mass on year 80, half on a non-year token
    logits[3, -1, 3] = 0.0
    start_years = t.tensor([30, 70, 50, 40])
    result = year_prob_diff(logits, start_years, year_tokens)
    assert result.shape == (4,), f"Expected one value per prompt, got shape {tuple(result.shape)}"
    expected = t.tensor([1.0, -1.0, -1.0, 0.5])
    t.testing.assert_close(result.cpu(), expected, atol=1e-4, rtol=1e-4,
                           msg="Expected PD = [1, -1, -1, 0.5]: years equal to YY count as 'not greater', and use the full-vocab softmax")
    print("All tests in `test_year_prob_diff` passed!")


def test_per_prompt_head_effects(per_prompt_head_effects: Callable, PatchingTask: type, model: HookedTransformer):
    task, _ = _make_task(PatchingTask, model)
    clean_logits = model(task.clean_tokens)
    _, corrupt_cache = model.run_with_cache(task.corrupted_tokens)
    source_z = [corrupt_cache["z", layer] for layer in range(model.cfg.n_layers)]
    result = per_prompt_head_effects(model, task, source_z, ["9.9", "5.5"])
    assert result.shape == (2, task.n_prompts), f"Expected (n_heads, batch) = (2, {task.n_prompts}), got {tuple(result.shape)}"
    m_clean = task.metric(clean_logits).mean()
    with _quiet_tl_patching():
        oracle = tl_patching.get_act_patch_attn_head_out_all_pos(
            model, task.clean_tokens, corrupt_cache, lambda lg: task.metric(lg).mean() - m_clean
        )
    t.testing.assert_close(result.mean(-1).cpu(), t.stack([oracle[9, 9], oracle[5, 5]]).cpu(), atol=2e-3, rtol=1e-3,
                           msg="The mean of the per-prompt effects should equal the ordinary head patching effect")
    print("All tests in `test_per_prompt_head_effects` passed!")


def test_bootstrap_mean_ci(bootstrap_mean_ci: Callable):
    g = t.Generator().manual_seed(0)
    values = t.randn(400, generator=g) + 3.0
    lo, hi = bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, seed=0)
    assert lo < values.mean().item() < hi, "The interval should contain the sample mean"
    width = hi - lo
    expected_width = 2 * 1.96 * values.std().item() / 400**0.5
    assert 0.8 * expected_width < width < 1.2 * expected_width, f"Interval width {width:.3f} should be close to {expected_width:.3f}"
    assert bootstrap_mean_ci(values, seed=0) == bootstrap_mean_ci(values, seed=0), "Should be deterministic given the seed"
    lo90, hi90 = bootstrap_mean_ci(values, alpha=0.1, seed=0)
    assert hi90 - lo90 < width, "A 90% interval should be narrower than a 95% interval"
    print("All tests in `test_bootstrap_mean_ci` passed!")


def test_answer_probability(answer_probability: Callable):
    logits = t.zeros(2, 3, 4)
    logits[:, :-1, 0] = 100.0  # earlier positions must be ignored
    logits[0, -1] = t.tensor([0.0, 0.0, 0.0, 0.0])
    logits[1, -1] = t.tensor([t.log(t.tensor(3.0)).item(), 0.0, 0.0, 0.0])
    answers = t.tensor([[1, 2], [0, 1]])
    result = answer_probability(logits, answers)
    t.testing.assert_close(result, t.tensor([0.25, 0.5]), msg="Should be softmax(final logits)[correct answer]")
    print("All tests in `test_answer_probability` passed!")
