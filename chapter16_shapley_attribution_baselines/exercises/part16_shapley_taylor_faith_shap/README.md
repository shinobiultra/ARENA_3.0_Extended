# 16.16 — Where Does Higher-Order Credit Go?

This extension lesson asks one concrete question: when an explanation retains only singletons and
pairs, where should the credit for a genuinely higher-order mechanism go? The learner implements
Shapley-Taylor interactions and Faith-Shap on complete games, keeps pair SII as a separate contextual
index, and then uses the three definitions to read a prespecified eight-feature Concrete-strength
model game.

The section is a **fully executed development candidate awaiting independent review**.
The v7 learner implements ten visible methods across eight exercises and produces eleven figures
from the full native study. Fresh source-only reproduction matches all deterministic values exactly.
The zero-run freeze permits development execution and explicitly does not grant publication approval.
Reviewer agents exhausted their account allowance before final review; no independent approval is
inferred from the author's own checks. Canonical notebooks, page, navigation and registry promotion
remain pending.

The complete candidate, rendered exercise draft, author visual review and exact evidence are linked
from `/home/bitwise/.cache/arena_extended/shapley_interactions/review-packet-v7/README.md`.

## Learner arc

The cold open is the exact four-player unanimity game. At order two, Shapley-Taylor assigns `0` to
every singleton and `1/6` to every pair; Faith-Shap assigns `-1/5` and `3/10`. Pair SII is `1/3`, while
ordinary singleton Shapley is `1/4`. The lesson derives why these answers differ before moving to the
real-model study.

Eight implementation exercises expose ten student-owned functions:

1. `mobius_transform`
2. `shapley_taylor_interactions`
3. `faith_design_matrix` and `faith_kernel_weights`
4. `faith_shap_interactions`
5. `faith_shap_closed_form`
6. `reconstruct_game` and `faithfulness_error`
7. `masked_model_game`
8. `contextual_pair_synergy`

Each exercise has an immediate focused test, expected output, help and interpretation dropdowns,
and an adjacent full solution. The exact diagnostics accumulate as the learner completes the methods,
so the first two figures are produced from functions the learner just wrote.

The guided native capstone refits the known seven Concrete predictors, requires all 21 known model
state identities, builds every coalition with one intact donor row, seals explanation evidence before
the literal audit load, and retains complete signed method outputs. Its eleven executed figures cover
the exact allocation ledger, the reason Faith-Shap moves singleton credit, predictive and game
controls, a fixed-row interaction graph, all-row pair matrices, held-out order capture, order-two
reconstruction, background sensitivity, contextual interventions, and both editable plays.

## Controls and interpretation boundaries

The exact controls include constant, additive, signed pair, pure triple, pure four, degree-two mixed,
and higher-order mixed games. The model study retains three clean MLPs, three paired-label-shuffled
MLPs, and ridge. It compares nested and independent donor banks, a fixed explanation-derived pair
order against 256 random orders, 128 within-coalition-size shuffles, direct four-forward contextual
differences, ridge pair terms, and distance-to-training diagnostics.

The background defines the empirical coalition game; moving it changes the estimand. Faith-Shap is
optimal only for its declared normalized kernel objective. Shapley-Taylor, Faith-Shap, and SII average
higher-order structure differently, and none is treated as causal ground truth. Predictive quality is
shown as context and does not select models or explanation results.

The two `Try It Yourself` cells are:

- a dividend router where the player set, retained order, sign, and value are editable; and
- a donor-context microscope where the learner can change the model, row, donor bank, pair, context,
  or intact donor and inspect the four model values behind one local second difference.

## Checkpoint commands

Run the method tests without changing the protocol boundary:

```bash
uv run pytest -q chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap/interaction_methods_tests.py
```

Parse the lesson, complete every exercise from its own adjacent answer, and list every unresolved
strict-build requirement:

```bash
uv run python chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap/build_notebooks.py --checkpoint-check
```

To inspect paired notebook structure as an unpublished draft, choose a new disposable directory. Draft
filenames carry a `DRAFT_` prefix and cannot be mistaken for canonical course surfaces:

```bash
uv run python chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap/build_notebooks.py --draft-output /tmp/arena-16-16-draft
```

Strict mode checks the executable source and canonical instruction-page parity. It refuses publication
while the page is absent. The execution harness can build an unpublished learner after the zero-run
freeze; the publisher additionally requires independent review and actual native/visual evidence:

```bash
uv run python chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap/build_notebooks.py --check
```

Do not run `run_native.py` before the recorded zero-run freeze. After that freeze, the execution
owner will run the complete learner and fresh reproduction under the bounded CPU protocol; passing
method or checkpoint tests alone does not authorize native execution or publication.

## Sources

The primary definitions are:

- Sundararajan, Dhamdhere, and Agarwal, “The Shapley Taylor Interaction Index” (ICML 2020).
- Tsai, Yeh, and Ravikumar, “Faith-Shap: The Faithful Shapley Interaction Index” (JMLR 2023).

`protocol.md` records the frozen scientific design; `implementation_amendments.json` records six
dated implementation corrections without changing that design.
`data_manifest.json` pins the intended Concrete rows, splits, donor banks, seeds, and known model
states. Verification reports and package comparisons will support the learner artifact; they do not
replace the visible derivations, tests, figures, or plays.

## Current evidence and limitations

- Complete own-answer learner v7: 141 cells, 55 code cells, eleven embedded figures; all seven known
  models refit and all 21 model-state hashes match. Monitored wall time 53.69 seconds, sampled peak
  process-tree RSS 2.640 GB, outputs 743.54 MB before browser derivatives.
- Fresh subprocess: 1,797 arrays / 94,604,271 values and 22,556 scalar leaves match exactly, with
  no learner fitted payloads supplied to the child. This is independent implementation reproduction,
  not an independent reviewer verdict or a new predictive study.
- Payload checks: 1,772 entries pass, including 1,735 required tensor paths and all physical shards.
- Tests: 119 passed, one explicit optional full-fit skip; full fitting was exercised separately.
- Real browser review by the author: 65 rendered math nodes, 54 working dropdowns, eight solutions,
  eleven loaded figures, no math errors or page horizontal overflow. Both play cells have checked
  parameter variations and invalid-input cases.
- Independent final review, canonical publication, normal registered report execution and integration
  audits remain open. `publish_study.validate_publication_review` correctly refuses this development
  state. The builder deliberately cannot create canonical surfaces by bypassing that gate.
- Model/data outcomes are known replications. Intact-donor replacement is observational; changing
  donors changes the game, and an interaction coefficient is not a physical causal effect.

Failed development runs and exact source snapshots are retained in the same cache. The current
candidate is v7; do not use v6's failed strict reproduction as passing evidence.
