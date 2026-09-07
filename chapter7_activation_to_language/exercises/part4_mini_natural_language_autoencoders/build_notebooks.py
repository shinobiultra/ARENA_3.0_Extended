"""Build the paired learner and solution notebooks for section 7.4."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import nbformat as nbf

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from chapter7_activation_to_language.exercises.part4_mini_natural_language_autoencoders import (
    solutions,
)

SECTION_DIR = Path(__file__).resolve().parent
EXERCISE_PATH = SECTION_DIR / "7.4_Mini_Natural_Language_Autoencoders_exercises.ipynb"
SOLUTION_PATH = SECTION_DIR / "7.4_Mini_Natural_Language_Autoencoders_solutions.ipynb"
PAGE_PATH = (
    SECTION_DIR.parents[1]
    / "instructions"
    / "pages"
    / "04_[7.4]_Mini_Natural_Language_Autoencoders.md"
)


def source_for(*names: str) -> str:
    return "\n\n".join(inspect.getsource(getattr(solutions, name)) for name in names)


def stub_for(name: str) -> str:
    node = ast.parse(inspect.getsource(getattr(solutions, name))).body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    node.body = [
        ast.Raise(exc=ast.Call(func=ast.Name(id="NotImplementedError"), args=[], keywords=[]))
    ]
    return ast.unparse(ast.fix_missing_locations(node))


def exercise_cell(
    title: str,
    difficulty: str,
    minutes: str,
    task: str,
    expected: str,
    help_text: str,
    interpretation: str,
    solution_names: tuple[str, ...],
) -> str:
    solution = source_for(*solution_names)
    ast.parse(solution)
    return f"""### Exercise - {title}

> ```yaml
> Difficulty: {difficulty}
> Importance: high
> You should spend up to {minutes} minutes on this exercise.
> ```

{task}

<details>
<summary>Expected output</summary>

{expected}

</details>

<details>
<summary>Help - reveal guidance</summary>

{help_text}

</details>

<details>
<summary>Interpretation</summary>

{interpretation}

</details>

<details>
<summary>Solution</summary>

```python
{solution}
```

</details>"""


SETUP = f"""import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
import torch.nn.functional as F
from IPython.display import Image, display

chapter = "chapter7_activation_to_language"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from chapter7_activation_to_language.exercises.part4_mini_natural_language_autoencoders import tests
# tests.test_* wrappers run the same focused checks under pytest.

PYTHIA_MODEL_ID = {solutions.PYTHIA_MODEL_ID!r}
PYTHIA_REVISION = {solutions.PYTHIA_REVISION!r}
PYTHIA_HIDDEN_STATE_INDEX = {solutions.PYTHIA_HIDDEN_STATE_INDEX}
PYTHIA_PATCH_BLOCK_INDEX = {solutions.PYTHIA_PATCH_BLOCK_INDEX}
AGREEMENT_CANDIDATES = {solutions.AGREEMENT_CANDIDATES!r}
FACTOR_STATES = {solutions.FACTOR_STATES!r}
AR_PROMPT = {solutions.AR_PROMPT!r}
NOUN_PAIRS = {solutions.NOUN_PAIRS!r}
TEMPLATE_PAIRS = {solutions.TEMPLATE_PAIRS!r}

{source_for('AgreementExample', 'PythiaNLADataset', 'ExplanationTokens', 'TokenVerbalizer', 'GeneratedExplanations', 'TokenReconstructor', 'ReconstructionMetrics', 'InterventionResult', 'MiniNLAReport', 'MiniNLAExperiment')}

{source_for('_require_informative_matrix', '_batched_final_hidden', '_candidate_indices', 'load_pythia', 'load_pythia_cpu', 'cache_pythia_nla_dataset')}

t.set_num_threads(min(8, t.get_num_threads()))
"""


INTRO = """# [7.4] Mini Natural Language Autoencoders

By the end of this notebook, you will have shown that a supervised text bottleneck recovers held-out
Pythia states and behavior because its contextual reconstructor beats an order-blind bag-of-tokens
bypass even when factor states share the same token multiset within each lexical context.

```python
GT_TIER = "GT-3"
EXERCISE_ID = "7_4_mini_natural_language_autoencoders"
EXPECTED_RUNTIME = "90-120 minutes; approximately one minute for the complete CUDA experiment"
REQUIRES_GPU = True
```

## Core Question

Does readable text preserve row-specific information in an activation, or only name a class whose mean vector is enough for one narrow behavior?

## Learning Objectives

- create a precommitted train/eval split with one composition omitted per lexical context;
- predict GPT-NeoX token sequences directly from activations;
- pass generated text through a frozen LM before reconstructing activations;
- implement the paper's row-normalized FVE rather than raw-vector FVE;
- build an order-blind bag-of-tokens reconstructor and make it fail visibly;
- reject factor-only, direct-factor, prototype, mean-conditioned, relabeled, and shuffled shortcuts;
- verify hook identity parity, row locality, and a nonzero causal text edit.

## Cold Open

Consider four Pythia states associated with `Today, the key`, `Today, the keys`, `Yesterday, the
key`, and `Yesterday, the keys`. We withhold one combination. Every explanation uses exactly the
same token multiset within a lexical context; only word order carries number and time:

> The topic is key. The wording is plain. Quantity is one not many. Time is current not earlier.

The opposite factor state swaps `one` with `many` and `current` with `earlier`. A bag-of-tokens
model therefore sees identical counts. The contextual reconstructor receives the ordered tokens,
wrapped in the paper's AR prompt. If the bag-of-tokens path matches it, the language model did no
necessary work.

## Background

The NLA paper defines an activation verbalizer (AV), `activation -> text`, and an activation
reconstructor (AR), `text -> activation`. Its AR runs explanation tokens through an LM and applies
an affine head to the final hidden state. Our smaller reconstructor reads Pythia's layer-4 state at
the final token and fits an affine head. The AV here is a supervised token-level ridge model; the
paper instead fine-tunes an LM and trains it with RL.

This is not an unsupervised reproduction of the paper. It is a controlled, inspectable warm-start designed to make the language path and its strongest obvious bypass executable on one local GPU.

The paper normalizes every gold and reconstructed activation before computing FVE. Skipping
prediction normalization changes the metric. Within-factor FVE then subtracts each number/time mean
before scoring, so a four-prototype decoder gets zero by construction.

<details>
<summary>Common bugs</summary>

- Comparing raw activation magnitudes instead of row-normalized directions.
- Letting a control use a shorter prompt format, which confounds content removal with distribution shift.
- Calling text a bottleneck without testing an order-blind lexical decoder.
- Reporting behavior parity alone even though four class prototypes can preserve the same answer.

</details>
"""


EXERCISES = [
    (
        exercise_cell(
            "precommit a compositional split",
            "easy",
            "10-15",
            (
                "Implement `natural_language_explanation` and `build_agreement_examples`. Each "
                "of 32 noun/style contexts must contribute three training rows and one "
                "evaluation row. Rotate the withheld state so all four number/time states "
                "occur eight times in evaluation."
            ),
            "`96` training examples, `32` evaluation examples, and `8` held-out rows for each of the four states.",
            "Use `context_id % 4` to select the held-out state. Encode each binary factor by swapping word order inside a fixed two-word contrast, so all four states retain identical token counts.",
            "Every evaluation sequence is an unseen composition, and a count-based model receives exactly the same lexical evidence for all four states in one context.",
            ("natural_language_explanation", "build_agreement_examples"),
        ),
        ("natural_language_explanation", "build_agreement_examples"),
        "tests.check_compositional_split(build_agreement_examples)",
    ),
    (
        exercise_cell(
            "tokenize complete explanations",
            "easy",
            "10-15",
            "Implement `tokenize_explanations`. Keep actual GPT-NeoX token IDs, add EOS, and map only the observed tokens into a compact local vocabulary for the small AV output head.",
            "The 96 training explanations form a `[96, 24]` token tensor over `33` actual tokenizer tokens.",
            "Pad with EOS and use an equality broadcast against `vocabulary_token_ids` to obtain local IDs. Reject evaluation tokens absent from the training vocabulary.",
            "A compact token vocabulary is ordinary language-model machinery. It is not a phrase lookup: the model predicts 24 token positions and never receives an explanation ID.",
            ("tokenize_explanations",),
        ),
        ("tokenize_explanations",),
        "tests.check_tokenization(tokenizer, dataset.train_examples, tokenize_explanations)",
    ),
    (
        exercise_cell(
            "fit an activation-to-token verbalizer",
            "medium",
            "15-20",
            "Implement `fit_token_verbalizer`, `generate_token_explanations`, and `token_generation_accuracy`. Fit token logits at every position from normalized layer-4 activations, then decode token IDs back to text.",
            "`792` logits per row (`24 x 33`) and `32/32` exact held-out explanations.",
            "Flatten position and vocabulary only for the ridge target. Reshape before `argmax`, recover actual token IDs through the vocabulary tensor, and stop decoding at EOS.",
            "The AV is supervised and position-wise, but it emits a real sequence. No factor tensor or fixed explanation selector lies between activation and text.",
            ("fit_token_verbalizer", "generate_token_explanations", "token_generation_accuracy"),
        ),
        ("fit_token_verbalizer", "generate_token_explanations", "token_generation_accuracy"),
        "tests.check_token_verbalizer(tokenizer, dataset, tokenize_explanations, fit_token_verbalizer, generate_token_explanations, token_generation_accuracy)",
    ),
    (
        exercise_cell(
            "run explanation tokens through the AR language model",
            "medium",
            "10-15",
            "Implement `encode_reconstructor_text`. Wrap each explanation in the paper's fixed AR prompt, run Pythia, and read the layer-4 hidden state at the final non-padding token.",
            "A `[batch, 512]` feature matrix; reordering the same words changes the feature by a nonzero amount.",
            "Reuse `_batched_final_hidden` with `PYTHIA_HIDDEN_STATE_INDEX`. Do not parse words into factors before this call.",
            "This LM pass is the mechanism missing from the failed version. The affine head comes after contextual token processing, as in the paper.",
            ("encode_reconstructor_text",),
        ),
        ("encode_reconstructor_text",),
        "tests.check_reconstructor_language_features(tokenizer, model, encode_reconstructor_text)",
    ),
    (
        exercise_cell(
            "fit the normalized activation reconstructor",
            "medium",
            "15-20",
            "Implement `fit_token_reconstructor`, `decode_text_features`, and `reconstruct_from_explanations`. Fit normalized targets, then restore the mean training activation norm for patching.",
            "A `[32, 512]` held-out reconstruction matrix with more than four distinct rows.",
            "Center normalized text features and activation targets separately. Solve the ridge problem in the 96-row dual space.",
            "The output head is affine, but its input is a contextual LM representation of all 24 text tokens. Replacing their order changes the reconstructed vector.",
            ("fit_token_reconstructor", "decode_text_features", "reconstruct_from_explanations"),
        ),
        ("fit_token_reconstructor", "decode_text_features", "reconstruct_from_explanations"),
        "assert reconstructed.shape == dataset.eval_activations.shape\nassert count_distinct_rows(reconstructed) == 32",
    ),
    (
        exercise_cell(
            "measure variation and reject prototype collapse",
            "medium",
            "15-20",
            "Implement FVE, within-factor FVE, row counting, the combined metric, and `assert_nonprototype_reconstruction`.",
            "Held-out FVE above `0.75`, within-factor FVE above `0.75`, and `32` distinct rows.",
            "Normalize every original and reconstructed row first. For within-factor FVE, subtract the two group means separately before comparing residuals. Guard both denominators against zero.",
            "Agreement behavior can survive class means. Within-factor FVE makes that shortcut visible instead of rewarding it.",
            (
                "fraction_variance_explained",
                "within_factor_fve",
                "count_distinct_rows",
                "reconstruction_metrics",
                "assert_nonprototype_reconstruction",
            ),
        ),
        (
            "fraction_variance_explained",
            "within_factor_fve",
            "count_distinct_rows",
            "reconstruction_metrics",
            "assert_nonprototype_reconstruction",
        ),
        "tests.check_nonprototype_metrics(dataset.eval_activations, reconstructed, dataset.eval_factors, reconstruction_metrics, assert_nonprototype_reconstruction)",
    ),
    (
        exercise_cell(
            "build the order-blind lexical bypass",
            "medium",
            "15-20",
            "Implement `bag_of_tokens_features`. Count actual tokenizer IDs, fit the same ridge reconstruction head to those counts, and verify that all four factor states in each lexical context are indistinguishable to this control.",
            "All 32 contexts have identical token-count rows across their four states. The held-out bag-of-tokens FVE is about `0.64`, below the contextual path by more than `0.10`.",
            "Build the vocabulary from training text only. Ignore unseen evaluation IDs rather than giving the control privileged features. Reuse the same normalized ridge head for a fair comparison.",
            "The bypass still reconstructs noun/style context well, so its within-factor FVE can look strong. It fails on the order-coded factors and causal behavior; both views belong in the final figure.",
            ("bag_of_tokens_features",),
        ),
        ("bag_of_tokens_features",),
        "tests.check_bag_of_tokens_control(tokenizer, dataset, bag_of_tokens_features, fit_token_reconstructor, decode_text_features, fraction_variance_explained)",
    ),
    (
        exercise_cell(
            "construct shortcut controls",
            "medium",
            "15-20",
            "Implement factor-only text, lexical relabeling, class-mean replacement, and the direct two-factor decoder. These are explicit negative baselines and must never feed the learned NLA path.",
            "The format-matched fixed-context control and direct-factor decoder remain far below the contextual path; class means have exactly four rows; noun relabeling damages row alignment.",
            "Use the full explanation template with a fixed known noun/style for the factor-text control. For prototypes, average only within an explicit factor state.",
            "These controls separate three claims: carrying factor labels, preserving behavior, and reconstructing row-specific activation variation.",
            (
                "prompt_only_factor_explanations",
                "lexically_relabel_explanations",
                "factor_prototype_reconstruction",
                "replace_with_factor_prototypes",
                "fit_direct_factor_decoder",
                "decode_direct_factors",
            ),
        ),
        (
            "prompt_only_factor_explanations",
            "lexically_relabel_explanations",
            "factor_prototype_reconstruction",
            "replace_with_factor_prototypes",
            "fit_direct_factor_decoder",
            "decode_direct_factors",
        ),
        "prototype = factor_prototype_reconstruction(dataset.train_activations, dataset.train_factors, dataset.eval_factors)\nassert count_distinct_rows(prototype) == 4",
    ),
    (
        exercise_cell(
            "patch one real row safely",
            "hard",
            "20-25",
            "Implement `patch_pythia_candidate_logits`, `candidate_accuracy`, `behavior_parity`, and `run_text_edit_intervention`. The hook must support all-row replacement and a declared row subset.",
            "Identity replacement reproduces logits exactly; a one-row edit changes row 0 by more than `1e-3` and changes every other row by exactly zero.",
            "Clone hook output before writing. Index both the batch row and that row's final-token position. Always remove the hook in `finally`.",
            "Identity parity validates the hook site. Single-row locality validates indexing. The AR edit difference supplies a real causal vector rather than a displayed text-only toy.",
            (
                "patch_pythia_candidate_logits",
                "candidate_accuracy",
                "behavior_parity",
                "run_text_edit_intervention",
            ),
        ),
        (
            "patch_pythia_candidate_logits",
            "candidate_accuracy",
            "behavior_parity",
            "run_text_edit_intervention",
        ),
        "tests.check_identity_and_locality(model, dataset, patch_pythia_candidate_logits)",
    ),
]


PROVIDED_BEFORE_AR = source_for("permute_explanation_tokens")
PROVIDED_FINAL = source_for(
    "_counterfactual_explanation",
    "_metrics_for",
    "evaluate_mini_nla",
    "save_signature_figure",
    "run_gpu_test",
    "run_full_experiment",
)


def build_notebook(solution: bool):
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(SETUP)]

    for index, (markdown, names, check) in enumerate(EXERCISES):
        cells.append(nbf.v4.new_markdown_cell(markdown))
        code = source_for(*names) if solution else "\n\n".join(stub_for(name) for name in names)
        if index == 0:
            code += "\n\n" + check
            code += "\n\nassert t.cuda.is_available(), 'This notebook requires CUDA.'"
            code += "\ntokenizer, model = load_pythia('cuda')"
            code += "\ndataset = cache_pythia_nla_dataset(tokenizer, model)"
        elif index == 1:
            code = PROVIDED_BEFORE_AR + "\n\n" + code + "\n\n" + check
        elif index == 4:
            code += "\n\ntrain_texts = tuple(x.explanation for x in dataset.train_examples)"
            code += "\ntrain_features = encode_reconstructor_text(tokenizer, model, train_texts)"
            code += "\nreconstructor = fit_token_reconstructor(train_features, dataset.train_activations)"
            code += "\neval_texts = tuple(x.explanation for x in dataset.eval_examples)"
            code += "\nreconstructed = reconstruct_from_explanations(tokenizer, model, eval_texts, reconstructor)"
            code += "\nassert reconstructed.shape == dataset.eval_activations.shape"
            code += "\nassert t.unique(reconstructed.round(decimals=4), dim=0).shape[0] == 32"
        else:
            code += "\n\n" + check
        cells.append(nbf.v4.new_code_cell(code))

    cells.extend(
        [
            nbf.v4.new_markdown_cell(
                """## Signature Result

Run the full pipeline once. The first table exposes actual prompts, generated token sequences,
original Pythia choices, reconstructed choices, and per-row cosine. The figure then makes the
shortcut comparison visible. No result is loaded from a verification report."""
                + "\n\n![Token-sequence NLA signature result](../../instructions/assets/mini_nla_sequence_reconstruction_signature.png)"
            ),
            nbf.v4.new_code_cell(
                PROVIDED_FINAL
                + "\n\nexperiment = evaluate_mini_nla(tokenizer, model, dataset)"
                + "\nreport = experiment.report"
                + "\ndisplay(pd.DataFrame(report.rows).head(12))"
                + "\nsummary = pd.DataFrame({"
                + "\n    'path': ['token NLA', 'bag of tokens', 'factor text', 'direct factors', 'class mean', 'mean-conditioned', 'lexical relabel'],"
                + "\n    'FVE': [report.reconstruction.fve, report.bag_of_tokens.fve, report.prompt_only_factor.fve, report.direct_factor.fve, report.class_mean.fve, report.mean_conditioned.fve, report.lexical_relabel.fve],"
                + (
                    "\n"
                    "    'within-factor FVE': [report.reconstruction.within_factor_fve, "
                    "report.bag_of_tokens.within_factor_fve, "
                    "report.prompt_only_factor.within_factor_fve, "
                    "report.direct_factor.within_factor_fve, report.class_mean.within_factor_fve, "
                    "report.mean_conditioned.within_factor_fve, "
                    "report.lexical_relabel.within_factor_fve],"
                )
                + (
                    "\n"
                    "    'behavior parity': [report.behavior_parity_accuracy, "
                    "report.bag_of_tokens_behavior_parity, "
                    "report.prompt_only_factor_behavior_parity, "
                    "report.direct_factor_behavior_parity, report.class_mean_behavior_parity, "
                    "float('nan'), report.lexical_relabel_behavior_parity],"
                )
                + (
                    "\n"
                    "    'distinct rows': [report.reconstruction.distinct_rows, "
                    "report.bag_of_tokens.distinct_rows, report.prompt_only_factor.distinct_rows, "
                    "report.direct_factor.distinct_rows, report.class_mean.distinct_rows, "
                    "report.mean_conditioned.distinct_rows, "
                    "report.lexical_relabel.distinct_rows],"
                )
                + "\n})"
                + "\ndisplay(summary)"
                + "\nasset = root_dir / 'chapter7_activation_to_language/instructions/assets/mini_nla_sequence_reconstruction_signature.png'"
                + "\nsave_signature_figure(report, asset)"
                + "\ndisplay(Image(filename=str(asset)))"
            ),
            nbf.v4.new_markdown_cell("""<details>
<summary>Interpretation</summary>

The contextual path beats the bag-of-tokens reconstructor on full FVE and causal behavior even
though every factor state has the same token counts. The bag model still reconstructs much of the
noun/style variation, which is why its within-factor score can be competitive. The figure therefore
separates lexical context recovery from order-sensitive factor recovery.

</details>

## Try It Yourself

Edit one generated explanation below. The cell computes `AR(edited) - AR(base)`, adds that vector to
one original Pythia state, patches only that row, and displays the before/after candidate logits.
This is the real intervention used by the test, not a text-only preview."""),
            nbf.v4.new_code_cell("""PLAY_ROW = 0
PLAY_STRENGTH = 1.0
PLAY_TEXT = "The topic is key. The wording is plain. Quantity is many not one. Time is current not earlier."

play = run_text_edit_intervention(
    tokenizer,
    model,
    dataset,
    experiment.reconstructor,
    row_index=PLAY_ROW,
    base_text=report.generated_explanations[PLAY_ROW],
    edited_text=PLAY_TEXT,
    strength=PLAY_STRENGTH,
)
display(pd.DataFrame({
    "candidate": [word.strip() for word in AGREEMENT_CANDIDATES],
    "original logit": dataset.original_candidate_logits[PLAY_ROW].tolist(),
    "intervened logit": play.candidate_logits.tolist(),
    "logit change": (play.candidate_logits - dataset.original_candidate_logits[PLAY_ROW]).tolist(),
}))
print(f"Selected-row maximum change: {play.selected_logit_change:.3f}")
# Strength zero or unchanged text is an identity control, and is allowed.
assert play.other_rows_max_logit_change == 0.0
"""),
            nbf.v4.new_markdown_cell("""## Controls and Baselines

- **Prompt-only factor text:** removes noun and style content while retaining quantity and time words.
- **Bag of tokens:** uses the same training texts and ridge head but discards order; all four factor states in a context become identical.
- **Direct factors:** fits the failed two-scalar affine shortcut explicitly as a negative baseline.
- **Class mean:** emits one activation prototype per factor state.
- **Mean-conditioned AR:** trains the same LM-based AR head after replacing every target with its factor mean.
- **Lexical relabeling:** cycles noun meanings while preserving factor words and sentence structure.
- **Shuffled AV:** independently permutes token-sequence targets for eight seeds and reports every seed's FVE and behavior parity.
- **Identity and locality:** exact identity-hook parity and zero change outside a selected patch row.

The seed distribution is visible below; no mean-only summary can hide a favorable shuffled run."""),
            nbf.v4.new_code_cell("""display(pd.DataFrame({
    "seed": [seed for seed, _ in report.shuffled_fve_by_seed],
    "shuffled FVE": [value for _, value in report.shuffled_fve_by_seed],
    "behavior parity": [value for _, value in report.shuffled_behavior_parity_by_seed],
}))
assert report.identity_max_logit_error == 0.0
"""),
            nbf.v4.new_markdown_cell("""## Anomaly Hunt

Sort the row table by cosine and inspect the lowest five cases. Does the error track noun
morphology, wording style, or one held-out state? Then change a noun pair or template pair before
rerunning. Precommit the change: do not relabel a row after seeing Pythia's answer.
"""),
            nbf.v4.new_code_cell(
                """failures = pd.DataFrame(report.rows).sort_values("row cosine").head(5)
display(failures)
"""
            ),
            nbf.v4.new_markdown_cell("""## Interpreting the Result

This result establishes a narrow mechanism claim. The AV predicts actual tokenizer sequences. The AR
receives those tokens alone, uses Pythia's contextual language features, and beats a bag-of-tokens
ridge decoder even though their lexical evidence is identical within each context. The 32 evaluation
rows are omitted combinations within known lexical contexts, not unseen-vocabulary generalization.
Pythia itself gets 30/32 grammatical targets correct, while the NLA preserves all 32 of Pythia's
original choices; these are deliberately reported as different metrics.

## Verification Evidence

The independent release job calls `run_full_experiment` and records a
hash-bound `verification_report.json`. That report is supporting provenance for
the CUDA run; it is never loaded to create, grade, or accept the learner result.

## Limitations

- This is a supervised warm-start with templated 24-token explanations, not unsupervised NLA discovery or RL.
- The AV is position-wise ridge regression, not an autoregressive fine-tuned language model.
- Nouns, styles, and individual factor words all appear during training; only their context-specific composition is held out.
- Pythia-70m's layer-4 state and four agreement tokens are a narrow behavior slice.
- The bag-of-tokens model still has strong within-factor FVE because noun/style tokens identify lexical context; the positive result is specifically order-coded factor recovery and behavior.
- High FVE does not establish that every word is faithful. Lexical edits and independent causal tests remain necessary.
- The full Pythia forward computes later layers even though AR features are read at layer 4; this keeps the lesson simple rather than runtime-optimal.

## Further Research

Replace the position-wise AV with a small autoregressive decoder, hold out whole noun families,
compare multiple reconstruction layers, or train the AR language trunk rather than freezing it. Keep
the class-mean and within-factor tests: a more capable model does not make shortcut checks optional.

## Reading

- [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/)
- [Open-source NLA training code](https://github.com/kitft/natural_language_autoencoders)
"""),
        ]
    )
    for index, cell in enumerate(cells):
        cell.id = f"mini-nla-{index:02d}"
    notebook.cells = cells
    return notebook


def render_instruction_page(notebook) -> str:
    """Render the exercise notebook into the chapter's Markdown page."""

    parts: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            parts.append(cell.source.rstrip())
        elif cell.cell_type == "code":
            parts.append(f"```python\n{cell.source.rstrip()}\n```")
    page = "\n\n".join(parts) + "\n"
    return page.replace(
        "../../instructions/assets/mini_nla_sequence_reconstruction_signature.png",
        "../assets/mini_nla_sequence_reconstruction_signature.png",
    )


def main() -> None:
    exercise_notebook = build_notebook(solution=False)
    nbf.write(exercise_notebook, EXERCISE_PATH)
    nbf.write(build_notebook(solution=True), SOLUTION_PATH)
    PAGE_PATH.write_text(render_instruction_page(exercise_notebook))


if __name__ == "__main__":
    main()
