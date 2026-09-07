# [8.6] AtP* Expected Outputs

These fixtures pin the results a learner should understand, not just reproduce.

## Exact toy

The clean metric is approximately `2.90` and the corrupt metric is `0.00`.
Exhaustive patching assigns the saturated query an effect magnitude of `3.00`
and the cancelling source `0.50`. Vanilla AtP assigns them only `1.64e-6` and
`0.00`, respectively. The Q/K softmax correction recovers the query and key
effects; GradDrop recovers the cancelled source. Their combination, AtP*, has
Spearman correlation `1.00` with exact patching and recovery@4 `0.899`.

The EAP-style baseline aggregates immediate path contributions and is
intentionally not an alias for AtP*: it still misses saturated Q/K but exposes
the two cancelling paths separately. Joint top-k faithfulness is visibly
non-monotonic, including an overshoot above `1.0`, so learners cannot replace
causal re-evaluation with a sum of individual effects.

## Pinned TransformerLens result

The solution notebook loads the exact pinned Pythia-70M revision on CPU and
evaluates 96 named final-position rotary Q/K sites over 24 fixed grammar pairs.
AtP* reaches approximately `0.993` Spearman correlation with exact patching,
against `0.943` for vanilla AtP. The exact top site is `L5H5.K@end`.

The causal effect of the AtP* top-eight set is approximately `0.489`. It ranks
first against 24 same-size random sets (mean `0.189`, best `0.417`). Rankings
learned from clean-to-clean shuffled donors and independently shuffled labels
produce weaker true-task top-eight effects, approximately `0.266` and `0.395`.
The label-shuffled rank correlation can remain high (`0.987`); the notebook
calls this anomaly out rather than pretending that every descriptive statistic
must fail.

## Claim boundary

This establishes exact behavior on the toy and scoped final-token Q/K ranking
evidence on one pinned small model. It does not identify a complete grammatical
circuit, assign semantic roles to heads, or implement the paper's subset-sampling
false-negative diagnostic.


## Numerical provenance

On 2026-09-05, all twenty learner code cells were executed after completing the
eleven answer dropdowns. `reference_metrics.json` contains the complete corrected
CPU real-model run, including per-component estimates and control effects. The
original numerical acceptance thresholds in the lock remain unchanged.
The derivative-chain and two-token Q/K tests are independent float64 oracles with
absolute tolerances 1e-10 and 1e-12 respectively. The real comparison uses pinned
float32 weights, twenty-four fixed prompt pairs, and seed-zero matched controls.
The same prompt suite is used for screening and causal verification; this is a
descriptive component comparison, not held-out circuit generalization.
