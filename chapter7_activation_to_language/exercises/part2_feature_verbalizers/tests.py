"""Focused tests for [7.2] Feature Verbalizers.

The ``check_*`` functions are called immediately from the notebooks. The
``test_*`` wrappers make the same checks discoverable by pytest.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
from types import SimpleNamespace
from pathlib import Path

import nbformat
import pytest
import torch as t

SECTION_DIR = Path(__file__).resolve().parent
PAGE = SECTION_DIR.parents[1] / "instructions" / "pages" / "02_[7.2]_Feature_Verbalizers.md"
ASSET = SECTION_DIR.parents[1] / "instructions" / "assets" / "feature_verbalizers_number_signature.png"


def _solutions():
    from chapter7_activation_to_language.exercises.part2_feature_verbalizers import solutions

    return solutions


def _unfinished_functions(code: str) -> list[str]:
    """Detect empty exercise bodies structurally, including raise-only stubs."""
    unfinished = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [statement for statement in node.body if not (
            isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )]
        if not body or all(isinstance(statement, (ast.Raise, ast.Pass)) or (
            isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
            and statement.value.value is Ellipsis
        ) for statement in body):
            unfinished.append(node.name)
    return unfinished


def test_mean_difference_decoder(fit_fn, score_fn) -> None:
    labels = t.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=t.bool)
    activations = t.tensor(
        [
            [-2.0, 1.0, 0.0],
            [-2.0, -1.0, 0.0],
            [-2.0, 0.0, 1.0],
            [-2.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 0.0, -1.0],
        ]
    )
    decoder = fit_fn(activations, labels)
    t.testing.assert_close(decoder.direction, t.tensor([1.0, 0.0, 0.0]))
    assert decoder.threshold == pytest.approx(0.0), "The midpoint threshold should be zero."
    t.testing.assert_close(score_fn(activations, decoder), activations[:, 0])
    print("Mean-difference decoder: exact toy oracle passed.")
    print("All tests in `test_mean_difference_decoder` passed!")


def test_component_replacement(replace_fn) -> None:
    activations = t.tensor([[1.0, 2.0, -1.0], [-3.0, 4.0, 5.0]])
    direction = t.tensor([1.0, 0.0, 0.0])
    target_scores = t.tensor([7.0, -2.0])
    replaced = replace_fn(activations, direction, target_scores)
    t.testing.assert_close(replaced @ direction, target_scores)
    t.testing.assert_close(replaced[:, 1:], activations[:, 1:])
    assert not t.equal(replaced, activations), "Replacement must change the direction component."
    print("Direction-component replacement: score and orthogonal-component oracles passed.")
    print("All tests in `test_component_replacement` passed!")


def test_hypothesis_order_invariance(score_fn, select_fn) -> None:
    solutions = _solutions()
    _, validation, _ = solutions.split_number_dataset(solutions.make_number_dataset())
    activation_predictions = solutions.number_labels(validation)
    rows_seed_1 = score_fn(validation, activation_predictions, solutions.HYPOTHESES, seed=1)
    rows_seed_9 = score_fn(validation, activation_predictions, solutions.HYPOTHESES, seed=9)
    scores_1 = {row["hypothesis_id"]: row["agreement"] for row in rows_seed_1}
    scores_9 = {row["hypothesis_id"]: row["agreement"] for row in rows_seed_9}
    assert [row["hypothesis_id"] for row in rows_seed_1] != [
        row["hypothesis_id"] for row in rows_seed_9
    ], "Different seeds should alter display order."
    assert scores_1 == scores_9, "Semantic scores must not depend on display order."
    assert select_fn(rows_seed_1) == select_fn(rows_seed_9) == "subject_number_plural", (
        "Both orderings should select the subject-number hypothesis."
    )
    assert all(not re.fullmatch(r"[A-Z]", str(row["hypothesis_id"])) for row in rows_seed_1), (
        "Candidates must use semantic IDs rather than answer letters."
    )
    print("Candidate display order changed; semantic scores and selection did not.")
    print("All tests in `test_hypothesis_order_invariance` passed!")


def test_matched_subset_comparison(evaluate_fn) -> None:
    labels = t.tensor([1, 0, 0, 0], dtype=t.bool)
    predictions = t.tensor([1, 0, 0, 1], dtype=t.bool)
    baseline = t.tensor([0, 0, 0, 0], dtype=t.bool)
    subset = t.tensor([0, 0, 1, 1], dtype=t.bool)
    report = evaluate_fn(predictions, labels, baseline, subset)
    assert report.accuracy == pytest.approx(0.75), "Full-set method accuracy is incorrect."
    assert report.baseline_accuracy == pytest.approx(0.75), "Full-set baseline accuracy is incorrect."
    assert report.subset_accuracy == pytest.approx(0.5), "Subset method accuracy is incorrect."
    assert report.subset_baseline_accuracy == pytest.approx(1.0), (
        "The baseline must be evaluated on the identical subset."
    )
    assert not report.survives_subset, "A method below its matched baseline must not survive."
    print("Contrastive comparison uses the baseline on the same selected rows.")
    print("All tests in `test_matched_subset_comparison` passed!")


# These are notebook-facing ARENA tests whose implementations are passed explicitly.
# Pytest exercises them through the zero-argument wrappers below.
test_mean_difference_decoder.__test__ = False
test_component_replacement.__test__ = False
test_hypothesis_order_invariance.__test__ = False
test_matched_subset_comparison.__test__ = False


def test_mean_difference_decoder_exact_toy_oracle() -> None:
    solutions = _solutions()
    test_mean_difference_decoder(
        solutions.fit_mean_difference_decoder,
        solutions.direction_scores,
    )
    print("All tests in `test_mean_difference_decoder_exact_toy_oracle` passed!")


def test_direction_component_replacement_exact_toy_oracle() -> None:
    test_component_replacement(_solutions().replace_direction_component)
    print("All tests in `test_direction_component_replacement_exact_toy_oracle` passed!")


def test_number_dataset_is_balanced_and_vocab_disjoint() -> None:
    solutions = _solutions()
    dataset = solutions.make_number_dataset()
    train, validation, heldout = solutions.split_number_dataset(dataset)
    assert (len(train), len(validation), len(heldout)) == (24, 12, 24), (
        "Expected the precommitted 24/12/24 split."
    )
    noun_sets = [{example.noun for example in split} for split in (train, validation, heldout)]
    assert not noun_sets[0] & noun_sets[1], "Train and validation noun vocabularies must be disjoint."
    assert not noun_sets[0] & noun_sets[2], "Train and held-out noun vocabularies must be disjoint."
    assert not noun_sets[1] & noun_sets[2], (
        "Validation and held-out noun vocabularies must be disjoint."
    )
    assert {example.morphology for example in heldout} == {"regular", "irregular", "nonce"}, (
        "Held-out evaluation must include every declared morphology."
    )
    assert sum(example.morphology == "nonce" for example in heldout) == 8, (
        "Held-out evaluation must contain eight nonce examples."
    )
    print("All tests in `test_number_dataset_is_balanced_and_vocab_disjoint` passed!")


def test_hypothesis_scoring_is_not_letter_or_order_dependent() -> None:
    solutions = _solutions()
    test_hypothesis_order_invariance(
        solutions.score_hypotheses,
        solutions.select_hypothesis,
    )
    print("All tests in `test_hypothesis_scoring_is_not_letter_or_order_dependent` passed!")


def test_contrastive_metric_uses_matched_baseline_subset() -> None:
    test_matched_subset_comparison(_solutions().evaluate_binary_predictions)
    print("All tests in `test_contrastive_metric_uses_matched_baseline_subset` passed!")


def test_pair_preserving_null_exact_oracle(fn=None) -> None:
    solutions = _solutions()
    fn = fn or solutions.paired_swap_decoder_accuracies
    labels = t.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=t.bool)
    train = t.tensor(
        [
            [-4.0, 1.0],
            [4.0, 1.0],
            [-3.0, -1.0],
            [3.0, -1.0],
            [-2.0, 0.5],
            [2.0, 0.5],
            [-1.0, -0.5],
            [1.0, -0.5],
        ]
    )
    target = solutions.fit_mean_difference_decoder(train, labels)
    target_accuracy = float(solutions.direction_predictions(train, target).eq(labels).float().mean())
    paired_null = fn(train, labels, train, labels)
    assert target_accuracy == 1.0, "The exact toy signal should be perfectly decodable."
    assert len(paired_null) == 16, "Four matched pairs require all 2**4 exact swaps."
    assert sum(paired_null) / len(paired_null) == pytest.approx(0.5), (
        "The exact pair-preserving toy null should be symmetric around chance."
    )
    print("All tests in `test_pair_preserving_null_exact_oracle` passed!")


def test_token_id_controls(tokenizer, final_id_fn, lookup_fn) -> None:
    solutions = _solutions()
    train, _, heldout = solutions.split_number_dataset(solutions.make_number_dataset())
    train_ids = final_id_fn(tokenizer, train)
    heldout_ids = final_id_fn(tokenizer, heldout)
    predictions = lookup_fn(tokenizer, train, heldout)
    assert len(train_ids) == len(train) and len(heldout_ids) == len(heldout), (
        "Return exactly one final token ID per noun while preserving multi-token nouns."
    )
    assert not set(train_ids) & set(heldout_ids), (
        "This exact-ID memorization control requires disjoint held-out token IDs."
    )
    assert predictions.shape == (len(heldout),), "Lookup predictions must align with held-out rows."
    assert predictions.dtype == t.bool, "Lookup predictions must be a Boolean number decision."
    print("All tests in `test_token_id_controls` passed!")


def test_orthogonal_control_construction(direction_fn, directions_fn) -> None:
    direction = t.nn.functional.normalize(t.arange(1.0, 9.0), dim=0)
    one = direction_fn(direction, seed=3)
    many = directions_fn(direction, seeds=tuple(range(16)))
    assert one.shape == direction.shape and many.shape == (16, len(direction)), (
        "Orthogonal helpers returned the wrong direction-bank shape."
    )
    assert float((many @ direction).abs().max()) < 1e-5, (
        "Every matched control must be orthogonal to the target direction."
    )
    t.testing.assert_close(many.norm(dim=-1), t.ones(16), rtol=1e-5, atol=1e-5)
    print("All tests in `test_orthogonal_control_construction` passed!")


test_token_id_controls.__test__ = False
test_orthogonal_control_construction.__test__ = False


def test_feature_evidence_loop(
    rank_fn,
    counterexample_fn,
    frontier_fn,
) -> None:
    solutions = _solutions()
    _, validation, _ = solutions.split_number_dataset(solutions.make_number_dataset())
    scores = t.tensor([-3.0, 3.0, -2.0, 2.0, -1.0, 1.0, -0.2, 0.2, -0.1, 0.1, -0.3, 0.3])
    rows = rank_fn(validation, scores, 0.0, n=2, seed=0)
    assert {row["group"] for row in rows} == {"top", "bottom", "random", "near miss"}, (
        "The evidence table must expose top, bottom, random, and near-threshold rows."
    )
    activation_predictions = scores >= 0
    suffix = next(h for h in solutions.HYPOTHESES if h.hypothesis_id == "surface_suffix_s")
    counterexamples = counterexample_fn(validation, activation_predictions, suffix)
    assert {row["prompt"] for row in counterexamples} == {"The mice", "The geese"}, (
        "Irregular plurals must falsify the suffix explanation."
    )
    scored = solutions.score_hypotheses(validation, activation_predictions)
    assert "subject_number_plural" in frontier_fn(scored), (
        "The accurate grammatical explanation must remain on the brevity frontier."
    )
    print("Feature evidence loop: four sampling groups, counterexamples, and brevity passed.")
    print("All tests in `test_feature_evidence_loop` passed!")


test_feature_evidence_loop.__test__ = False


class _TensorBatch(dict):
    def to(self, device):
        return _TensorBatch({key: value.to(device) for key, value in self.items()})


class _ExactTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, prompts, padding=True, return_tensors="pt"):
        del padding, return_tensors
        return _TensorBatch(
            input_ids=t.zeros((len(prompts), 1), dtype=t.long),
            attention_mask=t.ones((len(prompts), 1), dtype=t.long),
        )

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [0 if text == " is" else 1]


class _ExactBlock(t.nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _ExactBackbone(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = t.nn.ModuleList([_ExactBlock()])


class _ExactHookModel(t.nn.Module):
    """A two-dimensional causal readout with a genuine registered block hook."""

    def __init__(self):
        super().__init__()
        self.anchor = t.nn.Parameter(t.zeros(()))
        self.gpt_neox = _ExactBackbone()
        self.config = SimpleNamespace(hidden_size=2)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = t.zeros((len(input_ids), 1, 2), device=self.anchor.device) + self.anchor * 0
        hidden = self.gpt_neox.layers[0](hidden)[0]
        logits = t.zeros((len(input_ids), 1, 2), device=hidden.device)
        logits[:, :, 1] = hidden[:, :, 0] + 2 * hidden[:, :, 1]
        return SimpleNamespace(logits=logits)


def test_batched_control_intervention_exact_hook_oracle(run_fn=None) -> None:
    solutions = _solutions()
    run_fn = run_fn or solutions.run_direction_control_distribution
    report = run_fn(
        _ExactHookModel(),
        _ExactTokenizer(),
        ("first", "second"),
        t.tensor([2.0, 4.0]),
        t.eye(2),
        0.0,
        effect_sign=1.0,
        chunk_size=1,
        layer=0,
    )
    assert report["effects"] == pytest.approx((3.0, 6.0)), (
        "Each direction must receive the supplied per-example coefficients before the readout."
    )
    assert report["hook_calls"] == (1, 1), (
        "Every chunk must pass through the registered block hook exactly once."
    )
    assert report["coefficient_checksum"] == pytest.approx(6.0), (
        "The control report must retain an exact checksum of supplied coefficients."
    )
    print("Batched paired controls used the exact coefficients and real hook path.")
    print("All tests in `test_batched_control_intervention_exact_hook_oracle` passed!")


test_batched_control_intervention_exact_hook_oracle.__test__ = False


def test_paired_causal_direction_bank_exact_oracle(fn=None) -> None:
    solutions = _solutions()
    fn = fn or solutions.paired_swap_decoder_directions
    labels = t.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=t.bool)
    activations = t.tensor(
        [[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
         [0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0]]
    )
    patterns, directions = fn(activations, labels, count=6, seed=17)
    assert len(patterns) == len(set(patterns)) == 6 and all(0 < p < 15 for p in patterns), (
        "The causal null must exclude the target and global inversion patterns."
    )
    for pattern, direction in zip(patterns, directions, strict=True):
        swapped = labels.view(-1, 2).clone()
        mask = t.tensor([(pattern >> index) & 1 for index in range(4)], dtype=t.bool)
        swapped[mask] = swapped[mask].flip(1)
        expected = solutions.fit_mean_difference_decoder(activations, swapped.flatten()).direction
        t.testing.assert_close(direction, expected)
    print("Every precommitted causal-null direction matches its within-pair label swap.")
    print("All tests in `test_paired_causal_direction_bank_exact_oracle` passed!")


test_paired_causal_direction_bank_exact_oracle.__test__ = False


def test_runtime_dependency_provenance(entrypoint, required_functions, model, tokenizer):
    """Execute the real learner graph and prove required code objects actually ran."""

    forbidden_names = {"reference", "import_module", "eval", "exec"}
    assert forbidden_names.isdisjoint(entrypoint.__code__.co_names), (
        "The learner entrypoint must not delegate acceptance to a reference implementation."
    )
    seen_codes = set()

    required_codes = {fn.__code__ for fn in required_functions}

    def tracer(frame, event, arg):
        del arg
        if event == "call" and frame.f_code in required_codes:
            seen_codes.add(frame.f_code)
        return None

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = entrypoint(model, tokenizer, seed=0)
    finally:
        sys.settrace(previous)
    missing = [fn.__name__ for fn in required_functions if fn.__code__ not in seen_codes]
    assert not missing, f"Required learner functions were not executed: {missing}"
    assert result["signature_passed"], "The learner graph ran but failed its scientific contract."
    print("Runtime provenance passed:", ", ".join(fn.__name__ for fn in required_functions))
    print("All tests in `test_runtime_dependency_provenance` passed!")
    return result


test_runtime_dependency_provenance.__test__ = False


def test_cache_api_cannot_receive_labels() -> None:
    signature = inspect.signature(_solutions().cache_final_token_activations)
    assert "labels" not in signature.parameters, (
        "Activation caching must not receive labels or leak supervision."
    )
    print("All tests in `test_cache_api_cannot_receive_labels` passed!")


def test_layer_sweep_exact_oracle(sweep_fn=None) -> None:
    solutions = _solutions()
    sweep_fn = sweep_fn or solutions.layer_sweep_accuracies
    labels = t.tensor([False, True, False, True])
    states = t.tensor([[-2.0, 1.0], [2.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    train = solutions.LayerSweepCache(("embedding", "block 0"), (states, states))
    heldout = solutions.LayerSweepCache(("embedding", "block 0"), (states, -states))
    assert sweep_fn(train, labels, heldout, labels) == (1.0, 0.0), (
        "Fit each layer on train only: the held-out reversal must fail."
    )
    mismatched = solutions.LayerSweepCache(("embedding", "block 1"), (states, states))
    with pytest.raises(ValueError, match="same layers"):
        sweep_fn(train, labels, mismatched, labels)
    print("All tests in `test_layer_sweep_exact_oracle` passed!")


def test_feature_evidence_loop_exact_oracles() -> None:
    solutions = _solutions()
    test_feature_evidence_loop(
        solutions.rank_feature_examples,
        solutions.mine_counterexamples,
        solutions.brevity_frontier,
    )
    print("All tests in `test_feature_evidence_loop_exact_oracles` passed!")


def test_batched_control_intervention_uses_coefficients_and_hook() -> None:
    test_batched_control_intervention_exact_hook_oracle(
        _solutions().run_direction_control_distribution
    )
    print("All tests in `test_batched_control_intervention_uses_coefficients_and_hook` passed!")


def test_precommitted_paired_causal_direction_bank() -> None:
    test_paired_causal_direction_bank_exact_oracle(
        _solutions().paired_swap_decoder_directions
    )
    print("All tests in `test_precommitted_paired_causal_direction_bank` passed!")


@pytest.mark.cpu
@pytest.mark.requires_model
def test_real_cpu_signature_uses_cached_model_and_real_hook() -> None:
    solutions = _solutions()
    model, tokenizer = solutions.load_pinned_model()
    required = (
        solutions.cache_final_token_activations,
        solutions.cache_layer_sweep,
        solutions.fit_mean_difference_decoder,
        solutions.rank_feature_examples,
        solutions.mine_counterexamples,
        solutions.orthographic_ridge_predictions,
        solutions.paired_swap_decoder_directions,
        solutions.run_residual_edit,
        solutions.run_direction_control_distribution,
    )
    result = test_runtime_dependency_provenance(
        solutions.run_real_model_signature,
        required,
        model,
        tokenizer,
    )
    assert result["model_id"] == "EleutherAI/pythia-70m", "The released model ID changed."
    assert result["model_revision"] == "a39f36b100fe8a5377810d56c3f4789b9c53ac42", (
        "The released model revision changed."
    )
    assert result["selected_hypothesis"] == "subject_number_plural", (
        "Validation should select the subject-number verbalizer."
    )
    assert result["metrics"]["heldout_activation_accuracy"] >= 0.95, (
        "The direction should generalize to the disjoint held-out vocabulary."
    )
    assert result["metrics"]["subset_accuracy"] > result["metrics"]["subset_baseline_accuracy"], (
        "The direction must beat the suffix baseline on the matched contrastive subset."
    )
    assert result["metrics"]["heldout_activation_accuracy"] > result["metrics"]["suffix_accuracy"], (
        "Overall direction accuracy must exceed the suffix heuristic."
    )
    assert result["metrics"]["paired_null_count"] == 4096, (
        "The release result must enumerate every one of 2**12 pair-preserving swaps."
    )
    assert result["metrics"]["paired_null_accuracy_p95"] == pytest.approx(1.0), (
        "The exact paired null's perfect-accuracy tail must remain visible."
    )
    assert result["metrics"]["paired_null_exact_p"] >= 0.05, (
        "The decoding result must be reported as non-significant under the paired null."
    )
    assert result["metrics"]["embedding_accuracy"] == pytest.approx(1.0), (
        "The embedding shortcut must remain visible instead of being credited to block 3."
    )
    assert result["metrics"]["orthographic_accuracy"] <= result["metrics"]["heldout_activation_accuracy"], (
        "The spelling baseline must be reported on the same held-out rows."
    )
    assert result["suffix_counterexamples"], (
        "Validation counterexamples must force revision of the suffix explanation."
    )
    assert result["selected_hypothesis"] in result["explanation_frontier"], (
        "The revised explanation must survive the agreement-versus-brevity comparison."
    )
    assert result["metrics"]["train_heldout_final_token_overlap"] == 0, (
        "The final-token lookup control requires disjoint train and held-out token ids."
    )
    assert result["metrics"]["heldout_multi_token_noun_count"] > 0, (
        "The tokenizer audit must expose, rather than silently ignore, multi-token nouns."
    )
    assert result["morphology_breakdown"]["irregular"]["suffix_accuracy"] < 1.0, (
        "Irregular nouns must expose errors in the suffix heuristic."
    )
    assert result["morphology_breakdown"]["nonce"]["suffix_accuracy"] == 1.0, (
        "Report honestly that nonce nouns do not defeat this suffix heuristic."
    )
    intervention = result["intervention"]
    assert intervention["target_delta"] > 2.0, "The forward target edit is too weak."
    controls = intervention["orthogonal_distribution"]
    reverse_controls = intervention["reverse_orthogonal_distribution"]
    assert controls["orthogonality_max_abs_dot"] < 1e-5, (
        "Every control direction must be orthogonal to the target direction."
    )
    assert controls["norm_error_max"] < 1e-5, (
        "Control edits must preserve each target edit coefficient norm."
    )
    assert intervention["target_delta"] > controls["absolute_p95"] + 1.0, (
        "The forward target edit must clear the 64-control absolute p95."
    )
    assert intervention["reverse_target_delta"] > 1.0, "The reverse target edit is too weak."
    assert intervention["reverse_target_delta"] > reverse_controls["absolute_p95"] + 1.0, (
        "The reverse target edit must clear its own matched 64-direction null."
    )
    paired_controls = intervention["paired_causal_distribution"]
    reverse_paired_controls = intervention["reverse_paired_causal_distribution"]
    assert result["metrics"]["paired_causal_count"] == 256, (
        "The precommitted causal randomization test must evaluate 256 decoder directions."
    )
    assert intervention["target_delta"] > paired_controls["p95"], (
        "Forward intervention must clear the stronger within-pair decoder null."
    )
    assert intervention["reverse_target_delta"] > reverse_paired_controls["p95"], (
        "Reverse intervention must clear its own within-pair decoder null."
    )
    assert result["metrics"]["forward_paired_causal_p"] < 0.05, (
        "Forward paired causal randomization must reject at alpha 0.05."
    )
    assert result["metrics"]["reverse_paired_causal_p"] < 0.05, (
        "Reverse paired causal randomization must reject at alpha 0.05."
    )
    assert all(count == 1 for count in paired_controls["hook_calls"]), (
        "Every paired-control batch must execute its real hook once."
    )
    assert all(count == 1 for count in reverse_paired_controls["hook_calls"]), (
        "Every reverse paired-control batch must execute its real hook once."
    )
    assert intervention["reverse_singular_preference_rate"] >= 0.75, (
        "Reverse edits should make most plural prompts prefer the singular continuation."
    )
    assert intervention["target_plural_preference_rate"] >= 0.9, (
        "Forward edits should make most singular prompts prefer the plural continuation."
    )
    assert all(count == 1 for count in intervention["hook_calls"]), (
        "Each dose-response forward pass must call the registered hook exactly once."
    )
    assert result["signature_passed"], "The complete scoped signature contract failed."
    print("All tests in `test_real_cpu_signature_uses_cached_model_and_real_hook` passed!")


def test_invalid_fabricated_paths_are_absent() -> None:
    targets = [
        SECTION_DIR / "solutions.py",
        SECTION_DIR / "7.2_Feature_Verbalizers_exercises.ipynb",
        SECTION_DIR / "7.2_Feature_Verbalizers_solutions.ipynb",
        PAGE,
    ]
    joined = "\n".join(path.read_text() for path in targets)
    forbidden = (
        "_planted_feature_score",
        "planted_intervention_direction_test",
        "DECOY_TERMS",
        "Answer with one letter only",
        "Qwen2.5-0.5B-Instruct",
        "baseline_scores + alpha",
    )
    for phrase in forbidden:
        assert phrase not in joined, f"Rejected legacy evidence remains learner-visible: {phrase}"
    assert "register_forward_hook" in joined, "The learner surface must contain a real hook."
    assert "output_hidden_states=True" in joined, (
        "The learner surface must cache model-produced hidden states."
    )
    print("All tests in `test_invalid_fabricated_paths_are_absent` passed!")


def test_notebooks_and_page_have_complete_learner_arc() -> None:
    exercise = nbformat.read(SECTION_DIR / "7.2_Feature_Verbalizers_exercises.ipynb", as_version=4)
    solution = nbformat.read(SECTION_DIR / "7.2_Feature_Verbalizers_solutions.ipynb", as_version=4)
    exercise_text = "\n".join("".join(cell.source) for cell in exercise.cells)
    solution_text = "\n".join("".join(cell.source) for cell in solution.cells)
    page_text = PAGE.read_text()
    required = (
        "Core Question",
        "Cold Open",
        "Expected output",
        "Help",
        "Solution",
        "Interpretation",
        "Signature Result",
        "Try It Yourself",
        "Hunt an Anomaly",
        "Limitations",
    )
    for phrase in required:
        assert phrase in exercise_text, f"Exercise notebook is missing learner element: {phrase}"
        assert phrase in solution_text, f"Solution notebook is missing learner element: {phrase}"
        assert phrase in page_text, f"Instruction page is missing learner element: {phrase}"
    exercise_code = "\n".join(cell.source for cell in exercise.cells if cell.cell_type == "code")
    assert _unfinished_functions(exercise_code), (
        "Exercise notebook must leave core methods for the learner to implement."
    )
    solution_code = "\n".join(cell.source for cell in solution.cells if cell.cell_type == "code")
    assert not _unfinished_functions(solution_code), (
        "Solution notebook must provide every core implementation."
    )
    assert 'load_pinned_model("cuda")' in solution_code, (
        "The learner signature must explicitly use CUDA with no fallback."
    )
    assert "run_residual_edit(" in solution_code, "The learner must execute real residual edits."
    assert "2**pair_count" in solution_code, (
        "The learner-visible null must enumerate all pair-preserving label swaps."
    )
    assert "orthogonal_distribution" in solution_code, (
        "The learner result must compare a distribution of matched orthogonal controls."
    )
    assert "reverse_replacements" in solution_code, (
        "The learner result must perform the reverse causal intervention."
    )
    assert "reverse_orthogonal_distribution" in solution_code, (
        "The reverse intervention must have its own coefficient-matched control distribution."
    )
    assert "custom_intervention = run_residual_edit" in solution_code, (
        "The play cell must execute, not merely describe, a custom hook intervention."
    )
    assert "verification_report.json" not in solution_code, (
        "The notebook must generate its result rather than load release evidence."
    )
    assert ASSET.exists() and ASSET.stat().st_size > 20_000, (
        "The instruction-page signature asset is missing or implausibly small."
    )
    print("All tests in `test_notebooks_and_page_have_complete_learner_arc` passed!")


def test_executed_solution_notebook_contains_visible_signature() -> None:
    notebook = nbformat.read(SECTION_DIR / "7.2_Feature_Verbalizers_solutions.ipynb", as_version=4)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert not errors, "The stored solution notebook contains an execution error."
    rendered = json.dumps(notebook)
    assert "SIGNATURE PASSED: True" in rendered, "The executed notebook did not pass its signature."
    assert '"image/png"' in rendered, "The executed notebook is missing its visible signature figure."
    print("All tests in `test_executed_solution_notebook_contains_visible_signature` passed!")


def test_all_solution_dropdowns_parse_and_supply_core_functions() -> None:
    notebook = nbformat.read(SECTION_DIR / "7.2_Feature_Verbalizers_exercises.ipynb", as_version=4)
    answered = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown" or "<summary>Solution</summary>" not in cell.source:
            continue
        dropdown = re.search(r"<summary>Solution</summary>(.*?)</details>", cell.source, re.S)[1]
        answer = "\n".join(re.findall(r"```python\n(.*?)```", dropdown, re.S))
        answer_tree = ast.parse(answer)
        assert not _unfinished_functions(answer), "A solution must contain implemented function bodies."
        names = {node.name for node in answer_tree.body if isinstance(node, ast.FunctionDef)}
        required = set(_unfinished_functions(notebook.cells[index + 1].source))
        assert required <= names, f"Solution missing learner functions: {required - names}"
        answered.extend(required)
    assert len(answered) >= 20, "The completed learner must implement the full experiment."
    print("All tests in `test_all_solution_dropdowns_parse_and_supply_core_functions` passed!")
