# TreeSHAP completed checkpoint — 2026-09-06

Section 16.13 is published. Final learner-v7 runs all 87 cells / 38 code cells,
implements nine functions from their own solution dropdowns, and generates 17
figures and two editable plays in 26.18 seconds. Independent fresh fitting
reproduces all 3,961 arrays / 1,055,546 elements / 916 scalars exactly. Browser
review opens 40 dropdowns with 59 math expressions and no math errors.

All 18 tree fits and 486 component trees remain in the frozen protocol. Raw
SHAP 0.49.1 interventional parity fails on 469 model-row vectors and is disclosed;
a separately threshold-corrected reference passes. Student-to-exhaustive parity
remains near float64 precision. Thirty-five scientific tests pass; the global
suite has 72 passes and one historical readability failure (78 old blockers,
zero TreeSHAP blockers). Course/style/ladder/report/preservation audits pass.
The full course is still open on 86 unimplemented requirements and partial work.

See the current `HANDOFF.md` checkpoint and the section's `expected_outputs/`
archives for exact source, notebook, figure, report and run hashes. This original
plan is retained below to show the pre-fit intent and the falsified raw-library
expectation. The three frozen experiment-defining files were not changed.

---

# TreeSHAP learner plan

Current learner failure: the selected `shapley.treeshap` requirement has no
tree-path algorithm or fitted-tree attribution lab. Section 16.2 teaches a
six-player KernelSHAP/Owen game; it cannot teach TreeSHAP by renaming a call.
Before this plan, both 16.2 notebooks and its page were inspected, together
with original 0.1 Ray Tracing's implementation/test/render/solution sequence.
Original chapters 0–4 remain pinned and untouched.

Target section claim: implement exact TreeSHAP for a specified missing-feature
game, explain real tree ensembles, and distinguish algorithmic exactness from
the assumptions used to define the game. Append section 16.13; only the already
selected TreeSHAP requirement is in scope.

Signature result to create: a small tree and its leaf/path contributions lead
to real concrete-strength prediction waterfalls, attribution heatmaps and
runtime curves. Path-dependent and finite-background interventional results
are compared on identical models and rows. Repeated features remain one player.

Student-implemented functions: tree prediction, recursive path-dependent
coalition values, exact Shapley contraction, extend/unwind path weights,
recursive TreeSHAP, ensemble linearity, explicit interventional games and
Saabas path differences. The complete fitting and study code is also visible.
Use array batches for query/donor rows without hiding the path algorithm.

Controls/baselines: exhaustive subset and independent SHAP-library parity;
constant/dummy features; repeated-split fixtures; correlated two-feature
conditional/path/interventional games; Saabas that conserves totals but can
misallocate credit; shuffled training targets; all three predetermined seeds;
training-mean prediction; alternate training backgrounds; held-out recipe
groups and random feature-deletion orderings. Explicit float32 routing matches
scikit-learn prediction semantics. Additivity alone is not sufficient.

Play cells: alter the tiny tree's coverage/query; select a real held-out recipe,
model, missing-feature game and background size, then recompute its explanation.

Files to modify: new part13_treeshap sources, paired notebooks, readable lesson,
chapter page and actual-output assets; append navigation and selected-topic
registries only after complete execution and visual review. Do not edit the
older SAGE experiment or original ARENA content.

Tests to run: immediate tests for every student method; exhaustive path-weight
enumeration and repeated-feature fixtures; actual fitted sklearn prediction
and SHAP parity; complete own-answer notebook in a clean CPU kernel; independent
fresh fitting/reproduction; browser rendering/dropdowns/math/figures; original
preservation and applicable course/style/evidence/completeness audits.

Primary reading: Lundberg, Erion & Lee, arXiv:1802.03888, Section 3 and Algorithms
1–2; official SHAP TreeExplainer documentation; installed SHAP 0.49.1 Python
path recurrence and C++ implementation. Tree-path cover weighting is described
as a specific game, not silently identified with the population conditional.

Experiment decisions will be frozen before the first native tree fit. Reuse
SAGE's pinned UCI concrete source and recipe-group split for comparison, with
fresh tree fits and a separately pinned background ordering. Earlier SAGE
results are already known; there is no claim that this is a new unseen dataset.
