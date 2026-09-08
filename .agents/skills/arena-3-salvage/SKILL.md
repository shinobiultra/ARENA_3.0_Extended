---
name: arena-3-salvage
description: "Rewrite ARENA_3.0_Extended notebooks so they genuinely match original ARENA style: concrete learner arc, student-implemented code, visible convincing results, controls, and no verification-report-only sections. Use when modifying, auditing, or creating ARENA extension notebooks, instruction pages, solution notebooks, tests, or learner-facing docs."
---

# ARENA 3.0 Extension Salvage Skill

## Mission

Your job is not to add more topics. Your job is to make a small number of sections feel like original ARENA.

Original ARENA is sacred. Preserve it append-only. Never modify original chapters except explicit compatibility patches approved by the maintainer.

The current extension failed because it produced many verification-shaped notebooks rather than beautiful learner-facing notebooks. Fix that.

## Core standard

A notebook is ARENA-style only if a student can:

1. understand the concrete question immediately;
2. implement meaningful functions themselves;
3. get visible sub-function test feedback;
4. see a convincing result, not just a JSON report;
5. play with the model/input/layer/feature/patch;
6. understand the result through prose and figures;
7. see controls and baselines fail;
8. connect the toy result to a real paper/model;
9. know the limitations without the section becoming a defensive disclaimer.

## First action for every task

Before editing, inspect:

- the target notebook;
- the target instruction page;
- the target solution notebook;
- the closest original ARENA notebook in `chapter1_transformer_interp` or relevant original chapter.

Then write a short plan containing:

```text
Current learner failure:
Target section claim:
Signature result to create:
Student-implemented functions:
Controls/baselines:
Play cells:
Files to modify:
Tests to run:
```

## Forbidden patterns

Do not ship:

- signature result loaded only from `verification_report.json`;
- main output that is only a Python dict or scalar table;
- implementation hidden entirely in `utils.py` or `solutions.py`;
- toy tensor fixture unrelated to a model/task as the only result;
- UMAP/t-SNE without probe/kNN/seed/random-label checks;
- image-generation grids where most examples are white noise or collapsed;
- VLM results without image-region/background-region or text-only/image-only controls;
- ACDC notebook whose main result is final-position localization while admitting it is not ACDC;
- “course_ready” status based only on tests and verification report.

## Required notebook shape

Each rewritten notebook must contain:

```text
# Title
One-sentence claim
Learning objectives
Cold open concrete example
Background needed for first exercise
Exercise 1 + tests + expected output + help + solution
Exercise 2 + tests + expected output + help + solution
...
Signature result generated visibly in notebook
Try It Yourself cell
Controls and baselines
Interpreting the result
Limitations
Further research / anomaly hunting
Reading links
```

## Signature result requirement

Every section needs a visible result:

- plot;
- heatmap;
- table with real tokens/images/examples;
- circuit graph;
- generated examples with independent scoring;
- layer sweep;
- patching effect curve;
- retrieval matrix;
- denoising trajectory.

A JSON report may support this, but cannot be the result.

## Student-implemented core

Keep boring plumbing in helpers. Keep the method in the notebook.

Students must implement the core operation. Examples:

- 7.1: `logit_lens`, top-token table, simple tuned/ridge lens, activation replacement, Patchscope eval, random/text-only controls.
- 8.3: exact toy patching, pruning, faithfulness, minimality, completeness, same-size random circuits.
- 12.1: CLIP/SigLIP losses, retrieval eval, random-caption control.
- 12.3: visual-token cache, object/background patching, VQA metric.
- 9.1: mean-difference refusal direction, projection/addition, layer sweep, random/label-shuffled controls.

## Result-quality bar

Use at least one of:

- exact ground truth;
- held-out task with >=20 examples;
- reference implementation parity;
- real model behavior with counterfactual pairs;
- visible control failure.

If the result is white noise, collapsed, or only a one-prompt loading proof, label it as a negative/preflight result and do not make it the main course section.

## Chapter-specific guidance

Read `references/section_rewrite_blueprints.md` before rewriting high-priority sections.

Priority order:

1. 7.1 Logit Lens / Tuned Lens / Patchscopes.
2. 12.1/12.3 VLM flagship basics.
3. 8.3 ACDC and Circuit Metrics.
4. 9.1 Refusal Direction.
5. 5.5 Diffusion LM visuals.
6. 11.1 Representation Geometry.

## Finish criteria

When done, report:

```text
Changed files:
Original ARENA preservation status:
Student-facing improvements:
Signature result:
Controls/baselines:
Play cells:
Tests run:
Remaining limitations:
```

Do not claim the whole extension is fixed after one notebook. Mark unrevised sections as prototype/roadmap if they fail this standard.
