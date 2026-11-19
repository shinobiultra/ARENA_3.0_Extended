"""7.1 Logit lens, tuned lens and Patchscopes: CPU fixtures for the master method definitions, no model loads."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch as t

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = 'chapter7_activation_to_language'
PART = 'part1_lenses_patchscopes'
MASTER = 'master_7_1.py'
METHODS = ('train_tuned_lens', 'evaluate_lenses', 'shifted_translator_agreement', 'patchscope_controls',
           'patchscope_accuracy_grid')


def section_tests():
    path = ROOT / CHAPTER / 'exercises' / PART / 'tests.py'
    spec = importlib.util.spec_from_file_location('lesson_tests_' + PART, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method_ast(name):
    path = ROOT / 'infrastructure/chapters' / CHAPTER / MASTER
    return next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == name)


def method(name, *helpers):
    """Compile one master function (and the helper functions it calls) without importing the master."""
    # Future annotations avoid needing model classes. Decorators still execute normally.
    namespace = {'t': t, 'tqdm': lambda iterable, **kwargs: iterable}
    for fn_name in (*helpers, name):
        source = ast.unparse(ast.Module(body=[method_ast(fn_name)], type_ignores=[]))
        exec(compile('from __future__ import annotations\n' + source, MASTER, 'exec'), namespace)
    return namespace[name]


def method_variant(name, old, new, *helpers):
    """The master function with one piece of its source replaced: a deliberately wrong version."""
    source = ast.unparse(ast.Module(body=[method_ast(name)], type_ignores=[]))
    assert old in source, f'{old!r} not found in {name}; update this wrong variant'
    namespace = {'t': t, 'tqdm': lambda iterable, **kwargs: iterable}
    for helper in helpers:
        exec(compile('from __future__ import annotations\n' + ast.unparse(method_ast(helper)), MASTER, 'exec'), namespace)
    exec(compile('from __future__ import annotations\n' + source.replace(old, new), MASTER, 'exec'), namespace)
    return namespace[name]


def tiny_model(seed=0):
    """A random 3-block transformer on CPU with a plain final layernorm (as after folding) and non-zero biases."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(n_layers=3, d_model=16, n_heads=2, d_head=8, n_ctx=8, d_vocab=30, d_mlp=32,
                                  act_fn='gelu', normalization_type='LNPre', seed=seed, device='cpu')
    model = HookedTransformer(cfg)
    generator = t.Generator().manual_seed(seed)
    with t.no_grad():
        for p in model.parameters():
            p.copy_(0.4 * t.randn(p.shape, generator=generator))
    return model.requires_grad_(False)


def fails(test, *args):
    try:
        test(*args)
    except AssertionError:
        return True
    return False


@pytest.mark.parametrize('name', METHODS)
def test_independent_method_fixture(name):
    getattr(section_tests(), 'test_' + name)(method(name))




def test_invalid_lens_denominators_are_rejected():
    from inspect import unwrap
    from unittest.mock import patch

    fixture = section_tests()._metric_fixture()
    model, lens, tokens, _, _, get_resid = fixture
    for name in ('evaluate_lenses', 'shifted_translator_agreement'):
        fn = method(name)
        with patch.dict(unwrap(fn).__globals__, get_resid_and_logprobs=get_resid):
            for bad_tokens, batch_size in ((tokens[:0], 2), (tokens[:, :1], 2), (tokens, 0), (tokens, -1)):
                with pytest.raises(ValueError, match='positive batch size'):
                    fn(model, lens, bad_tokens, batch_size=batch_size, **({'shift': 0} if 'shifted' in name else {}))


def test_training_feedback_rejects_accumulated_gradient_mutation():
    """The learner check must fail if zero_grad is accidentally omitted."""
    import copy

    node = copy.deepcopy(method_ast('train_tuned_lens'))

    class OmitZeroGrad(ast.NodeTransformer):
        removed = 0

        def visit_Expr(self, expression):
            if (isinstance(expression.value, ast.Call)
                    and isinstance(expression.value.func, ast.Attribute)
                    and expression.value.func.attr == 'zero_grad'):
                self.removed += 1
                return None
            return self.generic_visit(expression)

    mutation = OmitZeroGrad()
    node = ast.fix_missing_locations(mutation.visit(node))
    assert mutation.removed == 1
    namespace = {'t': t, 'tqdm': lambda iterable, **kwargs: iterable}
    source = ast.unparse(ast.Module(body=[node], type_ignores=[]))
    exec('from __future__ import annotations\n' + source, namespace)
    with pytest.raises(AssertionError, match='Clear gradients'):
        section_tests().test_train_tuned_lens(namespace['train_tuned_lens'])


# --- Part 1: attention lens, and the wrong readouts the existing tests must catch -------------------------------


def test_attention_lens_sums_to_the_final_logit():
    section_tests().test_attention_lens(method('attention_lens'), tiny_model())


def test_attention_lens_wrong_variants_fail():
    tests = section_tests()
    model = tiny_model()

    def attention_lens_no_scale(model, cache, answer_tokens):
        z = t.stack([cache['z', layer][:, -1] for layer in range(model.cfg.n_layers)])
        out = t.einsum('lbhd,lhdm->lhbm', z, model.W_O)
        out = out - out.mean(-1, keepdim=True)
        return t.einsum('lhbm,mb->lhb', out, model.W_U[:, answer_tokens])  # forgot to divide by ln_final's scale

    def attention_lens_own_scale(model, cache, answer_tokens):
        z = t.stack([cache['z', layer][:, -1] for layer in range(model.cfg.n_layers)])
        out = t.einsum('lbhd,lhdm->lhbm', z, model.W_O)
        out = out - out.mean(-1, keepdim=True)
        out = out / out.std(-1, keepdim=True)  # each head normalised by its own size
        return t.einsum('lhbm,mb->lhb', out, model.W_U[:, answer_tokens])

    assert fails(tests.test_attention_lens, attention_lens_no_scale, model), 'skipping ln_final scale must fail'
    assert fails(tests.test_attention_lens, attention_lens_own_scale, model), 'per-head scale must fail'


def test_logit_lens_without_ln_final_fails():
    model = tiny_model()
    tests = section_tests()
    tests.test_logit_lens(method('logit_lens'), model)
    wrong = method_variant('logit_lens', 'model.ln_final(resid_stack)', 'resid_stack')
    assert fails(tests.test_logit_lens, wrong, model), 'a logit lens without ln_final must fail its test'


def test_reversed_kl_fails():
    tests = section_tests()
    tests.test_kl_to_final(method('kl_to_final'))
    wrong = method_variant('kl_to_final', 'final.exp() * (final - logprobs)', 'logprobs.exp() * (logprobs - final)')
    assert fails(tests.test_kl_to_final, wrong), 'KL(layer || final) must fail the KL(final || layer) test'


# --- Part 2: the contextual-lens bridge --------------------------------------------------------------------------


def test_normalize_rows_fixture():
    tests = section_tests()
    tests.test_normalize_rows(method('normalize_rows'))
    no_clamp = method_variant('normalize_rows', '.clamp_min(eps)', '')
    assert fails(tests.test_normalize_rows, no_clamp), 'zero rows must not turn into NaN'
    columns = method_variant('normalize_rows', 'dim=-1', 'dim=0')
    assert fails(tests.test_normalize_rows, columns), 'normalising columns must fail'


def test_contextual_topk_fixture():
    tests = section_tests()
    tests.test_contextual_topk(method('contextual_topk', 'normalize_rows'))


def test_contextual_topk_wrong_variants_fail():
    tests = section_tests()

    def raw_dot_product(queries, bank, k=1, chunk_size=1024):
        scores = queries @ bank.T
        order = scores.sort(dim=-1, descending=True, stable=True)
        return order.values[:, :k], order.indices[:, :k]

    def ties_to_highest_index(queries, bank, k=1, chunk_size=1024):
        scores = method('normalize_rows')(queries) @ method('normalize_rows')(bank).T
        order = scores.flip(-1).sort(dim=-1, descending=True, stable=True)
        return order.values[:, :k], bank.shape[0] - 1 - order.indices[:, :k]

    def drops_last_chunk(queries, bank, k=1, chunk_size=1024):
        scores, indices = method('contextual_topk', 'normalize_rows')(queries, bank, k, chunk_size)
        keep = len(queries) - len(queries) % chunk_size if len(queries) > chunk_size else len(queries)
        if len(queries) % chunk_size:
            scores, indices = scores.clone(), indices.clone()
            scores[keep:], indices[keep:] = 0, 0
        return scores, indices

    assert fails(tests.test_contextual_topk, raw_dot_product), 'an unnormalised dot product must fail'
    assert fails(tests.test_contextual_topk, ties_to_highest_index), 'ties must go to the lowest bank index'
    assert fails(tests.test_contextual_topk, drops_last_chunk), 'a search that mishandles the last chunk must fail'


def test_layer_alignment_grid_fixture():
    tests = section_tests()
    fn = method('layer_alignment_grid', 'normalize_rows', 'contextual_topk')
    tests.test_layer_alignment_grid(fn)

    def transposed(*args, **kwargs):
        return fn(*args, **kwargs).T

    def same_layer_only(queries, bank, targets, labels, chunk_size=1024):
        grid = fn(queries, bank, targets, labels, chunk_size)
        return t.diag(grid.diag())

    assert fails(tests.test_layer_alignment_grid, transposed), '[bank layer, query layer] order must fail'
    assert fails(tests.test_layer_alignment_grid, same_layer_only), 'a grid with only the diagonal must fail'


# --- Part 3: Patchscope grid (independent of the solutions) and the necessity exercise ------------------------


def test_patchscope_accuracy_grid_wrong_variants_fail():
    tests = section_tests()

    def make(transpose, use_position_not_layer):
        def patchscope_accuracy_grid(model, source_vectors, target_tokens, target_pos, answer_tokens, target_layers):
            grid = t.zeros(source_vectors.shape[0], len(target_layers))
            for i, vectors in enumerate(source_vectors):
                for j, target_layer in enumerate(target_layers):
                    layer = j if use_position_not_layer else target_layer
                    logits = run_patchscope(model, vectors, target_tokens, target_pos, layer)
                    grid[i, j] = (logits.argmax(dim=-1) == answer_tokens).float().mean()
            return grid.T if transpose else grid

        return patchscope_accuracy_grid

    for transpose, use_position in ((True, False), (False, True)):
        fn = make(transpose, use_position)
        assert fails(tests.test_patchscope_accuracy_grid, fn), f'wrong variant {(transpose, use_position)} must fail'


def test_ablate_blocks_recall_fixture():
    section_tests().test_ablate_blocks_recall(method('ablate_blocks_recall'), tiny_model())


def test_ablate_blocks_recall_wrong_variants_fail():
    tests = section_tests()
    model = tiny_model()
    zero = method_variant('ablate_blocks_recall', '.mean(dim=0)', '.mul(0).sum(dim=0)')
    assert fails(tests.test_ablate_blocks_recall, zero, model), 'zero-ablation instead of mean-ablation must fail'
    wrong_position = method_variant('ablate_blocks_recall', 'mlp_out[:, pos] = mean', 'mlp_out[:, -1] = mean')
    assert fails(tests.test_ablate_blocks_recall, wrong_position, model), 'ablating the final position must fail'
    online_mean = method_variant('ablate_blocks_recall', "clean_cache[name][:, pos].mean(dim=0)", "t.zeros(model.cfg.d_model)")
    assert fails(tests.test_ablate_blocks_recall, online_mean, model), 'a wrong mean must fail'
