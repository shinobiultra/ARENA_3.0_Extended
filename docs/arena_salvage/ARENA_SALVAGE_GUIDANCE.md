# ARENA Extension Salvage Guidance

## Purpose

The current extension is not failing mainly because of missing tests. It is failing because it does not feel like a learner-facing ARENA course. It often produces a verification-shaped artifact instead of a notebook where a student can build a mental model by writing code, seeing a compelling result, poking it, breaking it, and then understanding why the method matters.

This document tells agents how to salvage the extension.

## Diagnosis: what went wrong

### 1. The extension optimized for coverage instead of pedagogy

The existing extension tries to cover too many topics at once. Many notebooks are thin wrappers around toy fixtures and committed verification reports. They may satisfy structural audits, but they do not teach the student what is happening.

Bad pattern:

```text
import tests
write stub
run tests
load verification_report.json
print dictionary/table
state limitations
```

ARENA pattern:

```text
introduce a concrete mystery
load or create a model where the mystery is visible
implement a small function
immediately test it
visualize a real intermediate object
interpret the visualization
change something and see the effect
add controls and baselines
connect to the paper / real model
invite exploration
```

### 2. The current 7.1-style output is pedagogically dead

A result like:

```python
{
  'logit_lens': {'logits': [[2.0, 0.0, 1.0], ...]},
  'tuned_lens': {'logit_lens_accuracy': 0.0, 'tuned_lens_accuracy': 1.0},
  'patchscope': {'patchscope_accuracy': 1.0, 'text_only_accuracy': 0.5},
}
```

may be mathematically testable, but it is not an ARENA-style result. It does not tell the student:

- what activation was decoded;
- what model produced it;
- which token or concept it represented;
- why the result is surprising;
- what would happen if they changed the prompt;
- how this relates to logit lens / tuned lens / Patchscopes papers;
- what the model is doing internally.

This should be replaced by an interactive, concrete task. For 7.1, use a tiny real transformer or a trained toy transformer, real prompts, real tokens, and visual tables showing how decoded predictions evolve across layers or positions. The student should be able to change the prompt and watch the decoded answer change.

### 3. Too much implementation is hidden elsewhere

It is fine for shared utilities to exist, but the pedagogical core must be in the notebook. The student must implement the important pieces. A notebook where the real implementation lives in hidden `solutions.py`, `utils.py`, or a verification report is not a course notebook.

Allowed hidden support:

- plotting helpers;
- boring data-loading helpers;
- small test fixtures;
- reusable model wrappers after the student implemented the core mechanism once.

Not allowed hidden support:

- the method being taught;
- the main experimental loop;
- the core patching / attribution / decoding logic;
- the only code that produces the signature result;
- the only code that uses a real model.

### 4. The notebooks often prove too-weak claims

A course notebook needs a claim worth seeing. Some current notebooks end by saying, essentially, “this is just a preflight, not the actual method.” That is acceptable for infrastructure pages, but not as the main course section.

If the result is only a preflight, either:

1. strengthen it into a real section, or
2. move it to `docs/internal_preflights/`, or
3. make it a small prerequisite cell inside a stronger notebook.

Example: “ACDC and circuit metrics” should not end with final-position localization in `gelu-1l` and then say it is not ACDC. It should at least contain a toy graph with known ground-truth circuit and a small real circuit fragment where ACDC-style pruning visibly recovers a meaningful subgraph.

### 5. The current outputs are not convincing enough

The bar is not “the test passes.” The bar is “a skeptical student sees the result and believes something nontrivial.”

Weak result:

```text
Toy accuracy 1.0 on 3 examples.
```

Convincing result:

```text
A trained toy model reaches >95% held-out accuracy on 500 generated examples;
random-label control stays near chance;
activation patching localizes the variable to a specific layer/position;
patching the variable flips the prediction;
three hand-picked and three random held-out examples are shown;
the plot makes the mechanism legible.
```

### 6. The writing is too defensive and report-like

Claim boundaries are important, but if every section says “this is not X, not Y, not Z,” the student experiences the notebook as an audit artifact. ARENA instead teaches a concrete thing well, then clearly states limitations.

Use:

```text
Here is the thing we can show cleanly.
Here is why it is cool.
Here is the evidence.
Here is what it does not show.
```

Do not make the limitations the main story.

## What went right and should be kept

Keep these parts:

- append-only preservation of original ARENA;
- strict claim boundaries;
- ground-truth-first methodology;
- sub-function tests;
- negative controls;
- local 24GB GPU budget;
- verification reports as reproducibility artifacts;
- artifact locks;
- no fake / mock / placeholder claims;
- human-review questions about pedagogy.

But these should support the notebook, not replace it.

## What ARENA style actually means

ARENA is not merely a collection of tests. ARENA is an interactive learning path.

A good ARENA-style notebook has:

1. **A concrete hook.** The student knows what mystery they are investigating.
2. **A real object.** A model, circuit, dataset, image, activation, SAE feature, or prompt that can be inspected.
3. **Small implementation steps.** Students implement functions that are short enough to reason about.
4. **Immediate tests.** Every hard function is tested before it is used.
5. **Visible expected outputs.** Not just a dict; a table, plot, attention pattern, decoded token table, graph, example image, or failure case.
6. **Interpretation help.** The notebook explains what the result means and what it does not mean.
7. **Play affordances.** The student can change prompts, layer, position, image region, patch site, threshold, seed, or feature id and see what happens.
8. **Controls.** Random baseline, shuffled labels, text-only/image-only baseline, same-size random circuit, or counterfactual pairs.
9. **Escalation.** Start toy/controlled, then connect to a real model or published paper.
10. **Research taste.** It teaches how to notice anomalies and ask better follow-up questions.

## The non-negotiable rewrite rule

Every student-facing section must have exactly one sentence of the form:

```text
By the end of this notebook, you will have shown that <specific phenomenon> happens in <specific model/task>, because <specific evidence> beats <specific control>.
```

Examples:

```text
By the end of 7.1, you will have shown that intermediate residual activations in a tiny transformer become increasingly decodable through the unembedding across layers, because logit-lens/tuned-lens tables recover the correct next token on held-out prompts while random activations and text-only Patchscope prompts fail.
```

```text
By the end of 8.3, you will have shown that ACDC-style pruning can recover a known two-edge toy circuit and a small real-model path-patching fragment, because the discovered circuit preserves task logit difference, is minimal, is complete, and beats same-size random circuits.
```

If this sentence cannot be made compelling, the section is not ready.

## Minimum convincing result standard

A section’s signature result must include:

- a visible plot/table/image/graph generated in the notebook;
- at least 20 held-out examples for toy tasks unless mathematically exact;
- at least one negative control that fails visibly;
- at least one baseline that is meaningfully weaker;
- at least one intervention or counterfactual when the method claims causal relevance;
- a short interpretation paragraph written for a learner;
- at least one “try changing this” cell.

For tiny exact mathematical tasks, fewer examples are allowed, but the exactness must be obvious and explained.

## “Looks good” does not mean “white noise with a test pass”

Reject these as section-ready:

- a UMAP that does not predict labels;
- a generated image grid where most images are noise or collapse;
- a toy task trained on so few examples that memorization is plausible;
- a model-loading proof presented as interpretability;
- a dict of scalar metrics with no visible phenomenon;
- a “signature result” loaded only from `verification_report.json`;
- a circuit metric preflight that does not discover or validate a real circuit;
- a VLM result without object-region / background-region controls;
- a tuned lens that only improves on synthetic logits unrelated to a model;
- a Patchscope result where the text-only prompt already solves the task.

## Allowed uses of verification reports

Verification reports are reproducibility artifacts. They may be used to:

- cache a long GPU run;
- confirm exact model revision and GPU budget;
- store release metrics;
- prove that the notebook can be rerun offline if the model is already cached.

They may not be the only source of the lesson’s core result.

A student-facing notebook should have a live or lightweight path that produces the same kind of result, even if the heavy path is cached.

## Required rewrite process for every section

1. Read the corresponding original ARENA notebook closest in spirit.
2. Write the one-sentence section claim.
3. Identify the signature result figure/table.
4. Write the toy/controlled task.
5. Implement the smallest model/path that makes the phenomenon visible.
6. Add sub-function exercises and tests.
7. Add visible expected outputs.
8. Add controls and baselines.
9. Add “try changing this” cells.
10. Add the real-model/paper connection.
11. Add limitations.
12. Only then regenerate verification reports.

Do not start from reports or tests. Start from the learner experience.

## Rewrite priority

Stop expanding the course. Salvage in this order:

1. 7.1 Logit Lens / Tuned Lens / Patchscopes — because it is visibly broken as a learner experience.
2. 12.1 CLIP/SigLIP/VLM controls — because VLMs are the main interest and should be the flagship.
3. 8.3 ACDC and Circuit Metrics — because it has a decent skeleton but weak result.
4. 8.5 Sparse Feature Circuits — because it should be a flagship replication, not a checklist.
5. 9.1 Refusal Directions — because it can be simple, convincing, and beautiful.
6. 5.5 Diffusion Language Models — keep claim boundaries, add better visuals and play.
7. 11.1 Representation Geometry — enforce “plots must predict or intervene.”

All other sections should be frozen until at least three pilot sections feel genuinely ARENA-quality.

## Final acceptance test

A human reviewer should open a notebook cold and answer yes to all of these:

- Did I understand the question in the first minute?
- Did I implement meaningful code myself?
- Did I see intermediate tests catch likely bugs?
- Did I see a result that made me go “oh, cool, I get it”?
- Could I change something and see the result change?
- Did the notebook explain what the result means?
- Did a baseline or negative control fail?
- Did the result connect to a paper or real model?
- Did the notebook avoid overclaiming?
- Would I recommend this to someone learning mech interp?

If any answer is no, the section is not done.
