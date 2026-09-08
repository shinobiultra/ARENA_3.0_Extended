# DeepSHAP execution evidence

## How this fixture was produced

The publication fixture is created only from a complete own-answer learner in
a fresh CPU kernel. The learner defines the nine DeepSHAP methods in its own
cells, fits all six predeclared float64 MLPs plus the affine ridge control, and
evaluates every displayed result. It saves fitted numerical states and the
prepared explanation rankings before the literal audit load. A second process
then refits all seven models without receiving learner weights or fitted
payloads and compares every deterministic value recursively.

The data are the byte-pinned UCI Concrete Compressive Strength rows. The
758/153/119 train/explanation/audit split occupies 298/64/65 disjoint recipe
groups. Each of the seven models receives exact 256-coalition and DeepSHAP
explanations for all 272 held-out rows. Full per-reference contributions for
the 64-row primary background, five background variants, three attribution
rankings, and 100 random replacement orders remain in the numerical payloads.

No canonical fixture exists until the learner, independent fresh reproduction,
and actual browser review all pass. Missing outputs, changed source hashes, or
an absent visual review cannot be published as passing evidence.

## Trusted implementation

The exact scientific oracle evaluates the declared intact-donor empirical game
at all 256 coalitions and contracts it with an independently written Shapley
formula. Hand-computable affine, threshold, background-average, dummy, and
cancellation circuits give additional independent checks. The scalar-output
author reference is pinned SHAP 0.49.1 `DeepExplainer` on the exact supported
`nn.Sequential` Linear/ReLU architecture; it is supporting implementation
parity rather than the definition of exact feature Shapley.

DeepSHAP is a compositional approximation. The evidence must retain its signed
differences from exact Shapley over all 1,904 model-row combinations even when
its completeness and author-reference checks pass. The near-zero ReLU fixture
also retains the primary `epsilon=1e-6` residual of `4e-7` MPa as an expected
finite-epsilon failure. Recomputing it with `epsilon=0` establishes the exact
nonzero-secant control; neither result may be hidden by a larger tolerance.

## Random seed

Clean and shuffled MLP pairs reset model initialization to seeds 161410,
161411, and 161412. Shuffled-target permutations use 161420, 161421, and
161422. Primary and alternate background orders use 161460, 161461, and
161462. The 100 random replacement orders use 161480. Any diagnostic-only seed
must remain recorded in the deterministic diagnostic payload. Recipe splits
are byte-pinned, and background ordering is a deterministic hash of the stated
seed and source-row identity.

## Allowed tolerances

Tiny exact algebra, exact coalition efficiency, affine/ridge analytic parity,
and raw-state folding use `rtol=1e-10` and `atol=1e-10`. Real-model local
accuracy and author-reference parity use `rtol=1e-8` and `atol=1e-7` as frozen
in `protocol.md`. Fresh reproduction of deterministic arrays is exact with
`rtol=0` and `atol=0`. This lesson has no measured timing experiment, so
`timings.pt` is an intentionally empty mapping. Overall execution duration is
descriptive run metadata and is excluded from numerical reproduction equality.

The primary near-zero diagnostic is intentionally outside the real-model local
accuracy bound: its residual is `4e-7` MPa by construction. DeepSHAP-versus-
exact attribution error has no pass tolerance because disagreement is the
quantity being measured, not an implementation failure.

## When to regenerate

Regenerate after an intentional change to the lesson, visible methods, study
recipe, frozen data or protocol, reference adapter, diagnostics, plots, or
execution pipeline. Run the full own-answer learner from a fresh kernel, refit
all seven models in a second CPU process without cached fitted inputs, compare
all deterministic payload values, and repeat browser review before publishing
new hashes. Do not regenerate to erase DeepSHAP's approximation error, the
finite-epsilon diagnostic failure, or an unexplained package, data, seed,
tolerance, or source-hash drift.

## Expected records

- `learner_execution.json` binds the published notebook pair, page, namespaced
  figures, source snapshots, browser screenshots, and native artifacts.
- `native_summary.json` indexes the seven actual fits and complete study.
- `fresh_reproduction.json` records the independent seven-model refit and
  recursive full-value comparison. The completed reproduction compared 842 arrays,
  5,887,125 array elements and 603 scalars with zero numerical difference. Its
  separate `timings.pt` payload is empty by design.
- Archive manifests bind exact source and output bytes. Large `.pt` payloads
  remain at the persistent cache paths recorded by those manifests.

The six deterministic payloads are `fitted.pt`, `prepared.pt`, `evaluated.pt`,
`diagnostics.pt`, `reference.pt`, and `play.pt`. `timings.pt` reserves a separate
nondeterministic branch and is empty because this lesson declares no timing study. Fitted evidence
contains raw and standardized state
dictionaries, loss traces, target permutations, training targets and
predictions; it contains no opaque live PyTorch model object.

## Reproduce

After publication, run from the repository root with new output directories:

```bash
uv run python chapter16_shapley_attribution_baselines/exercises/part14_deepshap/execute_learner.py --output-directory /tmp/deepshap-own-new
CUDA_VISIBLE_DEVICES='' uv run python chapter16_shapley_attribution_baselines/exercises/part14_deepshap/reproduce_native.py --learner-directory /tmp/deepshap-own-new --output-directory /tmp/deepshap-fresh-new
CUDA_VISIBLE_DEVICES='' uv run python scripts/run_extension_verification_reports.py --section 16.14
```

The normal report performs another fresh CPU refit and verifies current
publication hashes. Original chapters 0–4 remain append-only. Passing this
section does not establish completion of other course requirements.
