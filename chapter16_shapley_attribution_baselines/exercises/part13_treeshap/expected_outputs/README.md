# TreeSHAP execution evidence

## How this fixture was produced

The primary artifact is the executed learner, which completes every student
function solely from its own following Solution dropdown. It fits all 18
predeclared tree models from the byte-pinned UCI concrete data and generates
every displayed result. These files support that lesson; they are not a
precomputed substitute for its algorithms or training.

`protocol.md`, `data_manifest.json` and `treeshap_data.py` were hashed before
native fitting and remain unchanged. The 758/153/119 train/explanation/audit
rows occupy 298/64/65 disjoint seven-ingredient recipe groups. All 486 component
trees, three families, three seeds, nine clean and nine shuffled fits remain
in the study. Prepared fits and explanation rankings are saved before the
literal audit load.

For each model, every one of 272 held-out rows receives both path-dependent
and 64-donor interventional explanations. Three primary clean families have
all 256 coalition values in both games. Controls include Saabas, 100 random
deletion orderings, alternate backgrounds, shuffled targets, a training-mean
predictor, repeated features, constants, dummy inputs and exact tiny games.

The installed SHAP 0.49.1 interventional reference fails the original strict
tolerance. Keep that result: 469 model-row vectors and 18 prediction sums are
affected. The separately corrected copied explainer floors each internal
threshold onto the float32 lattice without changing sklearn's input partition.
Its tolerance remains unchanged. See `reference_boundary_addendum.md` for the
post-freeze finding, complete counts and distinction between the compiled and
float64 exhaustive comparisons.

## Trusted implementation

The scientific oracle is the declared missing-feature game evaluated at all
256 coalitions, followed by an independently written subset contraction. Tiny
fixtures are also checked by averaging marginal contributions over every
feature ordering. These checks establish the student path-dependent and
finite-background interventional algorithms without treating a library call as
ground truth. The path-dependent SHAP 0.49.1 C++ result is a supporting
reference.

The unmodified SHAP 0.49.1 interventional result is retained as an expected
failed comparison. Its float32 threshold storage changes routing for some
sklearn float32 inputs, so it is not a trusted oracle for those rows. A separate
copied explainer replaces each internal threshold with the greatest float32
value at or below the original float64 threshold. This preserves sklearn's
input partition and passes the original compiled-reference bounds. The fitted
sklearn models, student outputs and individual examples are never adjusted.
The pinned UCI Concrete Compressive Strength bytes, package versions, source
hashes and independent reference hashes are recorded in the manifest and
execution evidence.

## Random seed

Model seeds are 161310, 161311 and 161312. Their matched shuffled-target
permutations use 161320, 161321 and 161322. Background variants use 161360,
161361 and 161362; the 100 random deletion orders use 161380; and the synthetic
runtime-tree leaf values use 161390. The immediate randomized method oracles use
131 and 132. Recipe splits themselves are byte-pinned inputs, and the
background ordering is a deterministic hash of the stated seed and row ID.

## Allowed tolerances

Exhaustive declared-game and path-reference comparisons use `rtol=1e-8` and
`atol=1e-8`. The corrected compiled interventional comparison retains
`rtol=2e-6` and `atol=2e-5` for SHAP's float32 C++ arithmetic. Fresh
reproduction of stored deterministic arrays is exact, with `rtol=0` and
`atol=0`; wall-clock timing values are validated separately. The unmodified
interventional reference has no relaxed acceptance tolerance: its failure,
affected rows and errors remain evidence.

## When to regenerate

Regenerate after an intentional change to the lesson, visible methods, study
recipe, frozen data or protocol, references, diagnostics, plots, or execution
pipeline. Run the complete own-answer learner in a fresh kernel, refit all 18
models in a second process without cached fitted inputs, compare every
deterministic array, and repeat the browser visual review before publishing new
hashes. Do not regenerate merely to remove the expected raw SHAP failure or to
accept an unexplained package, data, seed, tolerance or source-hash drift.

## Current records

- `learner_execution.json` binds the current notebook pair, page, namespaced
  chapter figures, exact own-answer source snapshots, actual browser screenshots
  and native outputs. It records 87 cells, 38 code cells, nine implementation
  exercises, 17 figures and two editable plays.
- `native_summary.json` describes actual fitted-model results and independent
  checks. Complete numerical arrays remain in the seven payloads below.
- `fresh_reproduction.json` identifies a separate run that refits every model
  from source data with no learner weights or other fitted inputs supplied.
  It compares all 3,961 arrays, 1,055,546 elements and 916 scalars; descriptive
  wall-clock timing arrays are checked separately.
- Archive directories contain exact source/output snapshots with SHA256
  manifests. Every tar member is read back and verified. Large `.pt` payloads
  stay at the persistent cache paths recorded in those manifests.

The six deterministic payloads are `fitted.pt`, `prepared.pt`, `evaluated.pt`,
`diagnostics.pt`, `reference.pt` and `play.pt`. `timings.pt` retains all measured
runtime repetitions. Fitted payloads contain plain tree arrays and training
targets/predictions, not opaque sklearn estimator objects. Reference payloads
retain raw and corrected vectors, all threshold changes and routing controls.

## Reproduce

From the repository root, choose new output directories:

```bash
uv run python chapter16_shapley_attribution_baselines/exercises/part13_treeshap/execute_learner.py --output-directory /tmp/treeshap-own-new
CUDA_VISIBLE_DEVICES='' uv run python chapter16_shapley_attribution_baselines/exercises/part13_treeshap/reproduce_native.py --learner-directory /tmp/treeshap-own-new --output-directory /tmp/treeshap-fresh-new
CUDA_VISIBLE_DEVICES='' uv run python scripts/run_extension_verification_reports.py --section 16.13
```

The normal report performs another fresh CPU reproduction and verifies current
publication hashes. Original chapters 0–4 remain append-only. Passing this
section's checks does not establish completion of the remaining course topics.
