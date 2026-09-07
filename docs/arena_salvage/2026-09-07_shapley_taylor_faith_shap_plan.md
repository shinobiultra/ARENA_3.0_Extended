# Shapley-Taylor and Faith-Shap learner plan

Current learner failure: the selected `shapley.shapley_taylor_faith_shap`
requirement has no student implementation or native comparison. Section 16.3
teaches SII; it does not provide an efficient Shapley-Taylor decomposition or
an exact constrained Faith-Shap surrogate. The new target pair/page does not
yet exist. The original 0.4 Backprop pair/page and the adjacent 16.3 pair/page
were inspected; the original ARENA prose, exercises, hints and result arc remain
the style authority. Preserve original chapters and completed 16.14/16.15.

Target section claim: when we retain only singles and pairs, where should credit
for a three-way or four-way mechanism go? Shapley-Taylor distributes omitted
dividends into the highest retained order. Faith-Shap jointly chooses a faithful
polynomial under the declared Shapley kernel and exact endpoint constraints.
Both can be efficient while assigning different coefficients. SII is a labeled
comparator, not an interchangeable efficient decomposition.

Section identity: provisional **16.16 — Where Does Higher-Order Credit Go?**,
directory `chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap`.
Implement only this selected requirement. SHAP-IQ/KernelSHAP-IQ estimator work
and SPEX remain distinct roadmap requirements.

Signature result: derive an exact pure-four unanimity game in which order-two
Shapley-Taylor has zero singletons and pair coefficients 1/6, while Faith-Shap
has singletons -1/5 and pair coefficients 3/10. Show endpoint efficiency beside
all-coalition reconstruction and contextual pair effects. Then compare signed
interaction matrices, order truncation, residual by coalition size and all-row
background sensitivity on a real fitted prediction game. Method disagreement
is a scientific result, not an error to hide.

Student-implemented functions: `mobius_transform`,
`shapley_taylor_interactions`, `faith_design_matrix`, `faith_kernel_weights`,
`faith_shap_interactions`, `faith_shap_closed_form`, `reconstruct_game`,
`faithfulness_error`, `masked_model_game` and `contextual_pair_synergy`.
Use complete CPU-float64 bitmask-indexed games with batch-safe leading axes,
an explicit intercept, exact zeros above the retained order and exact endpoint
constraints. The native core must execute from student-visible definitions;
reference adapters, plotting and storage may remain named plumbing. The method
contract is in `~/.cache/arena_extended/shapley_interactions/resume-spec-v1.md`.
Its unrun D5 fitting proposal is superseded by the D8 choice below.

Controls/baselines: exact additive, constant, dummy, pair, triple and pure-four
games; Möbius/order-n reconstruction; order-one Shapley identity; independent
discrete-derivative and primary-paper formula references; pinned shapiq exact
indices with its endpoint-penalty limitation; ridge zero higher-order credit;
all shuffled-target fits; declared within-cardinality shuffles; all random-pair
controls; background sensitivity. Distinguish a kernel-specific reconstruction
guarantee from unweighted error, predictive quality and causal interpretation.

Play cells: edit a tiny game's dividends/order to watch credit redistribution;
choose a real held-out recipe, donor background, method/order and feature pair,
then recompute all four contextual predictions and the resulting interaction.
Every exercise needs immediate semantic tests, expected outputs, a useful hint,
interpretation and a following Solution dropdown in original ARENA form.

Native design choice: use the D8 Concrete intact-donor empirical game, all seven
known predictors, all 272 held-out rows and every one of 256 coalitions under
the primary 64 intact training donors. The lesson refits the known predictors
self-contained and checks the established 21 state hashes. This is not an
independent model or dataset replication; known predictive outcomes cannot be
presented as newly discovered. Do not add the proposed D5 toy-physics fit. The
exact tiny-game controls remain useful without another synthetic training study.
Study/visual owners are specifying complete background/order/pair diagnostics,
retained arrays, seeds and audit boundaries before an independent protocol
review. No 16.16 native fit or attribution is authorized before the final freeze.

Continuation amendment: every reviewer agent hit the account usage limit after
the first independent review. Root is closing its six concrete findings with
no-fit checks. A development-only freeze may then permit native execution while
retaining `independent_approval=false` and `publication_authorized=false`.
The original blocked review remains preserved. This changes review timing only;
all scientific scope, seeds, controls and tolerances remain fixed before native
outcomes. Independent recheck, actual visual QA and fresh reproduction still
block publication and promotion. See the current protocol for the exact gate.

Files to modify: the new methods/tests, data/protocol/study, reference adapters,
diagnostics/figures, lesson/builder and execution harness in part16; paired
notebooks, instruction page and namespaced assets only after full execution and
review. Append navigation and the single selected registry row only after the
actual native, visual and report gates pass. No original-course or completed
section edits are planned.

Tests to run: hand-derived exact fractions and signs; dummy/symmetry/linearity;
Möbius-zeta inverse; exact endpoints/order-one/order-n; constrained-vs-closed-form
parity; per-coalition versus cardinality weights; arbitrary leading batches;
input/model/RNG restoration; donor averaging and explicit contextual forward
checks; full own-answer execution, independent fresh reproduction, same-game
primary-paper/package comparisons, actual browser/figure QA and final global
style/course/report/ladder/preservation audits. Benchmarks before freeze may
use analytic or random/untrained games only, never native outcomes for tuning.

Current outcome (September 7): final development candidate v7 passes complete own-answer
execution, exact fresh CPU reproduction (1,797 arrays / 94,604,271 values), 1,772
prospective payload checks, 119 tests and original preservation. Root inspected all
eleven actual figures in the rendered exercise page, all 54 dropdowns and 65 math
renderers. Two plays pass parameter variation checks. Six implementation corrections
are append-only and leave the original scientific freeze unchanged. The complete
review packet is `~/.cache/arena_extended/shapley_interactions/review-packet-v7/README.md`.
Independent publication review is still pending; the selected requirement remains
unregistered/unpublished and is not promoted. Continue other selected work while
reviewer execution is unavailable. No native process is still running.
