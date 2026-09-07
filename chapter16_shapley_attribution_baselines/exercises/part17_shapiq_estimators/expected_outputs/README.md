# Execution evidence for 16.17

The teaching artifacts are the notebook pair and instruction page. These receipts support their
execution and provenance; they are not the lesson.

`learner_execution.json` binds the published pair, page and ten figures to the complete own-answer
learner. `own_answer_execution.json` preserves its original execution receipt.
`fresh_reproduction.json` records a separate source-only child that refits seven models and
recomputes every deterministic artifact; it compares dtype, shape, values and container order.
`reference_metrics.json` is written by the normal registered report from its fresh arrays.
`author_visual_review.json` records the author's browser, figure and editable-play review.

Large arrays remain in the local cache at the absolute paths in these receipts. Each canonical
run retains six payloads: fitted states/traces, all donor predictions and games, sampling plans,
every estimate/SE/rank/constraint residual, exact references/variance, and actual package outputs.
Physical hashes and recursive value hashes cover all of them. The arrays are never replaced by
the small JSON receipts. Missing cache evidence makes verification fail; use the source commands
in `execute_learner.py`, `reproduce_native.py`, `review_learner.py` and `publish_study.py` to produce
and review a new complete run.

The original development run and corrected figure-axis run are retained under
`~/.cache/arena_extended/shapiq_estimators/`. Earlier analytic failures and their fixes remain there
too. Scientific seeds, budgets, splits, controls and tolerances were unchanged after native results.
The registered report always creates another fresh output directory. It does not load fitted models
into its child. Reproducing known seeded predictors is not independent model replication.
