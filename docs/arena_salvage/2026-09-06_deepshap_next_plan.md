# DeepSHAP learner plan

## Completed checkpoint

Section 16.14 is now published and registered. The final own-answer learner-v3
executes 114 cells, 70 code cells, nine exercises, 17 figures and two plays in
17.277692300 seconds on CPU. Fresh reproduction refits all seven models and
matches 842 arrays, 5,887,125 elements and 603 scalars exactly. The normal
section verifier also refits all seven models and passes.

Actual browser review covers 34 rendered equations, all 42 dropdowns, all 17
figures and 39 screenshots. The registered style, course, hard-exercise,
evidence and original-preservation audits pass. All 54 focused scientific and
harness tests pass. The full-course gate remains open with 85 unimplemented
requirements; its other partial/deferred rows remain explicit.

Canonical evidence is under `part14_deepshap/expected_outputs`; complete native
tensors remain in `~/.cache/arena_extended/deepshap/learner-v3`,
`reproduction-v3`, and the normal report's recorded fresh directory. No frozen
protocol, data manifest or data implementation was changed. The first learner
run's missing gradient-helper definition and seven unhelpful test failures were
fixed without changing scientific methods, results or tolerances. Earlier
development evidence is retained with its original source hashes.

The plan below records the design made before fitting. The next selected topic
has its own plan at `2026-09-06_gradientshap_next_plan.md`.

Current learner failure: the selected `shapley.deepshap` requirement has no
modified-backpropagation lesson, empirical-background approximation study or
known computational intervention comparison. Section16.13 teaches tree paths;
its exactness does not transfer to a nonlinear neural network.

Before this plan, original 0.4 Backprop exercises, solutions and instruction
page were inspected, together with the adjacent SAGE and TreeSHAP learner
surfaces and their visible training/data code. The new target has no existing
notebook/page/solution to preserve. Original chapters0–4 remain untouched.

Target section claim: implement DeepSHAP's Rescale-style backward multipliers
for a scalar affine/ReLU network, then distinguish completeness, agreement with
an independent implementation, and agreement with exact feature Shapley.
Append only the already selected DeepSHAP requirement as provisional16.14.
GradientSHAP/Expected Gradients remains a separate selected requirement.

Signature result: start with a two-input ReLU gate whose two necessary features
receive unequal Rescale credit; derive and implement the modified chain rule,
then show exact-versus-DeepSHAP waterfalls, signed scatters and all-row error
heatmaps for actual held-out concrete-strength predictions. Every row and
predeclared model remains in the aggregate results. No high-error row is chosen
as a representative example after seeing the data.

Student implementations: paired forward states, ReLU secant slopes, backward
multiplier propagation, a single-reference explanation, background averaging,
the full empirical coalition table, exact Shapley contraction, bounded
RevealCancel group allocation, and insertion/deletion prefix evaluation.
Each gets an immediate test, expected output, help/interpretation and its own
Solution dropdown. Keep complete fitting, prediction and evaluation visible in
small coherent cells, following original0.4's progression from local backward
rules to a full neural computation.

Controls/baselines: all256 coalitions for the identical empirical donor game;
independent installed SHAP DeepExplainer parity; exact affine/ridge control;
constant/dummy inputs; threshold and cancellation circuits with known input
interventions; gradient-times-input-difference; random feature orders; paired
shuffled-target networks; training-mean prediction; all three fixed seeds;
nested and independent backgrounds with their own exact games. The near-zero
secant fallback is explicit and has a tiny crossing counterexample. Reject
unsupported operators and reused modules rather than silently claiming general
DeepSHAP support.

Play cells: change a tiny gate's weights/bias/query/reference and watch
completeness and exactness diverge; choose a held-out recipe/model/background,
then recompute its approximation, exact game and feature-replacement curve.
Audit play is exploratory reuse, not a new evaluation set.

Files to modify: new `part14_deepshap` sources, readable lesson, paired
notebooks, chapter page and namespaced figures; then append navigation and
selected-topic/method/hard-exercise registries only after execution, fresh
reproduction and actual browser review. Keep TreeSHAP and original ARENA fixed.

Tests to run: immediate mathematical tests, known exact games and cancellation
controls, frozen data and input/output identities, independent author parity,
full own-answer notebook in a fresh CPU kernel, all-model fresh reproduction,
mutation tests for meaningful acceptance boundaries, visual/math/dropdown
review, course/style/ladder/evidence/original-preservation audits and the honest
full-course completeness gate.

Native design to freeze before the first fit: retain the UCI concrete data and
recipe-group split already used in16.12/16.13 (1030 rows;758/153/119). Train six
fresh float64 8→32 ReLU→32 ReLU→1 networks (three clean, three shuffled) and one
ridge control, keeping all models and all272 held-out rows. Use the same fixed
800-step optimizer recipe as16.12 with a new seed namespace and a separately
fixed64-donor background. Train and compute rankings on train/explanation only;
persist them before the literal audit load. Fold training standardization into
the first/last affine weights for a pure Sequential model operating in raw
input units and output MPa. Exact background games cover every model and row;
background variants recompute the exact oracle for the primary clean model.

Primary reading: Lundberg & Lee (NeurIPS2017), the DeepSHAP/DeepLIFT discussion;
Shrikumar, Greenside & Kundaje (ICML2017), Rescale and RevealCancel with the
supplement; official SHAP DeepExplainer PyTorch source. Computational input
replacement in tiny circuits supplies known intervention effects. Observational
concrete hybrids do not establish physical causal effects. Source review and
protocol review precede any native training.
