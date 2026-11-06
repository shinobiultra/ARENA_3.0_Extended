# Blank public Practice: 7.1 and 9.1

Use the first-session route below before opening
[7.1 Logit Lens, Tuned Lens and Patchscopes](7_1_exercises.ipynb) or
[9.1 Refusal Directions and Representation Engineering](9_1_exercises.ipynb).
Each route has a useful stopping point; neither requires finishing the full lesson.

## Start here: question, exposure, then implementation

Read the question and record what you already know or have seen. An informal
prediction, confidence and reason are optional while you are learning the method;
"I don't know yet" is a useful starting point. Save a personal working copy.

The opening record is optional and informal. Exact method hashes belong beside
the later selected experiment, after you have implemented and checked its methods.
**Use this order:**

1. Read the question and record prior exposure. Leave the exact method/config
   fingerprint pending while you learn; it is not a prerequisite for reading on.
2. Implement the functions in the original exercise cells and run their adjacent
   semantic tests. Use hints and solutions when needed, recording that exposure.
   Setup, baseline outputs and expected-answer dropdowns also count as exposure.
3. Inspect the first-session result with its controls, then stop and explain it.
   This is public, exposed practice, including any exploratory changes you make.
4. When ready for a **selected new experiment**, first finish and test every
   method it uses. Then save an exact method/config commitment and a prediction
   **before running or opening that selected experiment's result**. Include the
   SHA-256 of your completed method cells and used helpers, recoverable ordered
   inputs, model/tokenizer identities, data roles, metric/selection rules, seeds
   and runtime settings, following the preserved configuration fields. Record
   unknown historical details as unknown; never hash a placeholder as if it were
   your completed method. Freeze the changed-case plan before its result too.

Do not treat earlier exploration or results you already read as a newly frozen
prediction. Preserve the existing registered question, controls and protocol for
the later full exercise; a first-session question below is a learning milestone.
Record any inspected or selected-on data as monitored rather than untouched TEST.
These are manual public Practice records, with no software-enforced freeze and
no claim of unseen mastery. They add neither checkpoints nor blind companions.

## 7.1: first session

**Question:** At which layers does a correct capital become readable, and how do
other plausible capitals and random activations change that interpretation?

Start in **1️⃣ The Logit Lens**. Use the retained setup and capital-city example
when you choose to run a model. Implement `logit_lens`, `top_tokens_table`,
`answer_ranks` and `other_answer_ranks`, checking each with the adjacent
`tests.test_*` call before using it in the next exercise. The rank test fixes the
tie rule: one plus the number of strictly greater logits. The lens tests use the
model; these are not all model-free checks.

Your visible result is the layer-by-layer decoded-token table, followed by the
rank plot for the correct capital, other countries' capitals and norm-matched
random vectors. Read **Help - how should I interpret this plot?** beside that
plot. Identify one layer where a capital is readable but the identity is still
ambiguous. The original plot selects prompts the model answers correctly; keep
that denominator and selection visible, and do not generalize to every prompt.

**Stop after this rank plot and its interpretation.** KL divergence is a later
optional exercise. Do not enter **2️⃣ The Tuned Lens**, load TinyStories, or train
translators to finish this first session. For a small play task, inspect one
different retained capital prompt and explain how its table differs; record it
as exploratory if you have already inspected its answer.

Return for tuned-lens fitting and its held-out/bigram/wrong-layer comparisons in
a separate session. Patchscopes and activation interventions are another session:
only after `get_source_vectors`, `run_patchscope` and `patchscope_controls` pass
their semantic checks, complete the exact commitment before the selected
patched-versus-text-only experiment. Keep the norm-matched random and wrong-BOS
controls. Intervention/readout compatibility does not establish native necessity;
the preserved grid qualification remains applicable.

## 9.1: first session

**Question:** Can a mean-difference readout distinguish these prompt groups, and
how much separation remains for random or label-shuffled directions?

Start in **1️⃣ Finding a Refusal Direction**. Work through the retained chat
template and data explanation when you choose to run the setup. Implement
`is_refusal`, `refusal_score`, `get_activations`, `difference_of_means`, `auroc`
and `shuffled_label_direction`, running each adjacent `tests.test_*` check.
The activation-cache test uses the model/tokenizer; the rubric and tensor
exercises let you reason about the measurement before interpreting a model run.

Your first visible result is the layerwise AUROC readout plot, with random,
naive-shuffle and stratified-shuffle direction bands. Read its adjacent help and
**Summary of 1️⃣**. Explain what the rubric measures, what first-token scores miss,
and why a high readout AUROC does not establish a causal refusal mechanism.
Shuffle controls can retain prompt-distribution signal; do not require every
control to land at chance or change its rules to make it fail. The bands summarize
direction draws, not uncertainty from new datasets. Once inspected, these public
test-split results are exposed/monitored for your practice.

**Stop at the end of section 1.** Do not enter **2️⃣ Ablation, Addition and
Controls** or call `sweep_directions` for this session. A small play task is to
write a harmless completion that challenges the marker rubric, label it by hand,
and compare the rubric's answer without changing the rubric during evaluation.

Return for projection, ablation/addition, direction sweeps and their selection
rules in a later intervention session. Complete the relevant method checks and
exact commitment before the selected new sweep/intervention, preserving the
development/selection/evaluation roles. Weight edits and side effects, geometry
and PCA, and representation-engineering comparisons are later sessions. Readout
separation alone is not a steering result or a broad safety conclusion.

## Preserved reference and exposure

A blank reflection follows the preserved controls, including a changed-case
forecast. Complete those records when you reach the corresponding experiment;
they are not entrance requirements for the first-session exercises.

Every original learner cell remains in order, including implementations/stubs,
immediate tests, help/solutions, controls and play. Only saved **code outputs** and
**execution counts** are cleared in these copies. Original metadata, published
numbers, expected-output text/dropdowns and links remain: reading them later exposes
answers. The original provenance metadata describes the public reference; it does
not mean these blank copies ran. Keep track of what you already saw.

7.1 also places a conspicuous interpretation qualification immediately after the
original grid help. The grid shows intervention/readout compatibility in that target
prompt; downstream computation and donor/target compatibility both affect success.
It motivates a localization hypothesis, and does not establish necessity of blocks
1–5 for native factual recall. The historical paragraph and later summary stay
verbatim, and the qualification applies to that preserved necessity wording too.

## Intact originals

| Section | Original exercises | Original solutions | Instruction page |
| --- | --- | --- | --- |
| 7.1 | [original exercises](../../../chapter7_activation_to_language/exercises/part1_lenses_patchscopes/7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_exercises.ipynb) | [original solutions](../../../chapter7_activation_to_language/exercises/part1_lenses_patchscopes/7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_solutions.ipynb) | [original page](../../../chapter7_activation_to_language/instructions/pages/01_%5B7.1%5D_Logit_Lens_Tuned_Lens_and_Patchscopes.md) |
| 9.1 | [original exercises](../../../chapter9_alignment_interpretability/exercises/part1_refusal_directions/9.1_Refusal_Directions_and_Representation_Engineering_exercises.ipynb) | [original solutions](../../../chapter9_alignment_interpretability/exercises/part1_refusal_directions/9.1_Refusal_Directions_and_Representation_Engineering_solutions.ipynb) | [original page](../../../chapter9_alignment_interpretability/instructions/pages/01_%5B9.1%5D_Refusal_Directions_and_Representation_Engineering.md) |

All original masters, books, pages, modules and media are unchanged. The bounded
`original_sources.tar.gz` retains their exact bytes, with a strict review contract
and [source pins](original_source_manifest.json). The generator and projection map
are ordinary JSON/source plumbing, not hidden taught methods.

## Open and work

From the repository root, use the existing environment; no dependency changes are
needed to inspect or generate these copies:

```bash
.venv/bin/python mastery_hub/units/pilot_practice/generate.py --check
.venv/bin/python mastery_hub/units/pilot_practice/generate.py --guide 7.1
.venv/bin/python -m jupyter lab mastery_hub/units/pilot_practice/7_1_exercises.ipynb
```

For 9.1 use `--guide 9.1` and replace the filename with `9_1_exercises.ipynb`.
`--guide` prints the common opening and your selected route from this README; it
does not generate/check projections, access lesson inputs, or execute any cells.
Keep the notebook/kernel
inside this repository and use its existing environment. The retained setup finds
the repository from cwd or its ancestors, and imports the original chapter helpers.
The retained Colab links open the **original** reference notebooks. Original setup
and experiment cells can load models, download data or train when you choose to run
them; generation, `--check` and the projection tests execute none of those cells.

Save a personal working copy before editing. The generator defaults to exclusive
creation of **missing** projection files. It first checks every pinned source and
every existing target, then refuses any different bytes, including learner source,
metadata, saved outputs or mere formatting changes. There is no force/overwrite
switch. Exact matching files are a no-op, preserving mtime. `--check` is read-only
and fails for missing or edited files. If your notebook is edited, keep it and start
from a separate clean checkout rather than discarding work to satisfy generation.
The opening revision adds Markdown guidance, an explicit first-session stop and
a later exact commitment beside the selected experiment. A controlled migration
first verifies exact old machine-generated bytes and both original inverses;
learner-edited copies are not migration targets. The old map remains a separate
preserved receipt. Original source/archive/protocol pins are unchanged, and the
default generator still cannot overwrite any differing target.

## Exact inverse and limits

`projection_map.json` records every original/projected cell position and typed hash,
the full added Markdown cells, and the complete original code output/count states.
`invert` removes only those additions, restores those states and verifies the whole
original typed-object hash. All source/metadata/IDs/stubs/order are preserved.
The archive recovers the original serialized bytes separately. Notebook format4.2
and its existing absence of cell IDs stay intact; these copies do not normalize IDs
or schema. No runtime receipt, scientific result or learner response is fabricated.

The preserved methods/signatures, plots, controls, play and scientific limitations
are those of the original lessons. Public predictions and good source tests do not
establish a newly reproduced experiment, broad safety, a native recall circuit or
human learner acceptance.
