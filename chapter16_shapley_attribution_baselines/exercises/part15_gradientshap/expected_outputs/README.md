# Expected Gradients / GradientSHAP execution evidence

This directory contains the published release evidence created by `publish_study.py` from the
passing learner-v3, fresh-reproduction-v3 and browser-review records. The canonical records are
`learner_execution.json`, `native_summary.json`, `fresh_reproduction.json` and
`execution.tar.gz`. This README explains their claim boundary and regeneration contract; it is not
itself a passing release record.

## How this fixture was produced

The fixture comes from a complete own-answer learner executed in a fresh CPU kernel with CUDA
hidden and four OpenMP, MKL and OpenBLAS threads. The learner defines and tests all nine Expected
Gradients methods in its own cells, refits six predeclared float64 MLPs plus ridge, verifies all 21
known 16.14 state hashes, and computes every displayed result. It persists fitted and prepared state
before the literal audit load. SHAP 0.49.1 and Captum 0.9.0 reference checks, diagnostics and both
editable plays run afterward.

The byte-pinned UCI Concrete data retain all 1,030 rows and use a recipe-disjoint 758/153/119
train/explanation/audit split. The complete study covers seven models by 272 held-out rows, five
backgrounds, three noise conditions, K=`8,32,128,512,2048` nested Monte Carlo prefixes, 256- and
512-node path quadrature, all 256 intact-donor coalitions, 100 random replacement orders and 64
independent sampling schedules.

The actual v3 learner evidence is
`~/.cache/arena_extended/gradientshap/native-dev-v3/execution.json` (SHA256
`d9b0506e72ff849d17eb76ddba23f424ea33824dd9086fcbba954888b1af6973`). It records PASS, 126
executed cells, 79 code cells, 20 embedded figures, seven fits, completion from own answers, CPU
execution and no changed source. Its completed notebook SHA256 is
`6c2c458fa715f52a62f123ce19c8e0d88c9776f4255fe32b2d1d68a1a8546055`.

## Trusted implementation and independent oracles

The mathematical oracle for the declared feature game evaluates every one of the 256
intact-donor coalitions and performs an independently written exact Shapley contraction. Analytic
affine, quadratic, ReLU, threshold, constant, dummy, background-average and Gaussian-smoothed ReLU
fixtures check individual operations. The ridge model is an affine real-data control: its path and
feature-game allocations agree to floating-point precision.

Pinned SHAP 0.49.1 `GradientExplainer` and Captum 0.9.0 `GradientShap` are implementation
references for their own draw conventions. Parity replays the exact donors, alphas and noise drawn
by each package for all seven models and all 272 rows. The packages use different draw tables, so an
equal integer seed is not evidence of same-draw parity. SHAP's raw `phi_vars` is
`samples.var(ddof=0)/sqrt(K)`, neither variance of the mean nor ordinary SE. Captum's convergence
delta has one value per repeated sample and is not the averaged-vector completeness residual. Both
are retained as diagnostics and excluded from parity acceptance.

The v3 reference passed for 1,904 vectors and 15,232 coordinates per package. The maximum same-draw
SHAP error was `5.684341886080802e-14`; Captum's was zero. Positive-noise parity uses the separate
analytic smoothed-ReLU fixture.

## Random seed and frozen protocol

[protocol.md](../protocol.md) defines the estimator, data boundary, preparation-before-audit order,
model fits, backgrounds, noise, comparisons and payloads. [protocol_freeze.json](../protocol_freeze.json)
binds that protocol, [data_manifest.json](../data_manifest.json) and the data implementation before
any 16.15 native fit or attribution. The root namespace is
`arena16.15/gradientshap/v1`.

The fixed namespace seeds are:

| Purpose | Seed |
|---|---:|
| Primary donor/alpha schedule | `161510` |
| 64 independent schedule replicates | `161511` |
| Standard-normal query noise | `161512` |
| SHAP author NumPy stream | `161513` |
| Captum donor/alpha NumPy stream | `161514` |
| SHAP and Captum Torch smoothing streams, reset separately | `161515` |
| Analytic diagnostics | `161590` |
| Untrained benchmark only | `161599` |

Model initialization uses `161410`–`161412`, shuffled targets use `161420`–`161422`, background
orders use `161460`–`161462`, and random replacement uses `161480`. Each source-specific stream is
NumPy PCG64 seeded from the first 16 bytes, little-endian, of
`SHA256("{namespace}/{seed}/{source_id}/{stream_tag}")`; the literal tags are `baseline`, `alpha`
and `standard_normal`. Replicate `r` uses the first eight digest bytes, little-endian, of
`SHA256("arena16.15/gradientshap/v1/161511/replicate/{r}")`.

## Allowed tolerances

Tiny analytic/algebraic checks and same-draw SHAP/Captum parity use `rtol=1e-10` and
`atol=1e-10`. Exact model-state identities and sigma-zero sample identity are byte-exact. Fresh
reproduction compares every deterministic array and scalar exactly (`rtol=0`, `atol=0`) through
recursive value hashes. `timings.pt` is a separate nondeterministic branch; durations are descriptive
metadata and are excluded from numerical equality.

There is no acceptance tolerance for observed Expected-Gradient versus exact-Shapley disagreement,
background movement, 256-versus-512 quadrature drift, Monte Carlo residuals, distance diagnostics
or audit replacement curves. Those differences are the measurements. The ordinary Monte Carlo SE
is the plug-in estimate `sqrt(sample_variance(ddof=1) / K)`; it is distinct from discrete-game error,
SHAP's variance proxy and Captum's per-draw delta.

## Retained payloads and archive limit

Deterministic payloads are `fitted.pt`, `prepared.pt`, `evaluated.pt`, `reference.pt`,
`diagnostics.pt`, `play.pt` and `replicate_samples.pt`. Every payload has a recursive
path/type/shape/dtype/value manifest plus serialized-byte SHA256 and size. `timings.pt` remains a
separate branch.

The external `replicate_samples.pt` retains the complete float64 contribution tensor of shape
`(64,3,64,2048,8)`: 201,326,592 coordinates and 1,610,612,736 raw tensor bytes before
serialization. In v3 its serialized file is 1,610,614,447 bytes with SHA256
`46238cad0348f56be4807507256233884af3588037a434197f0cbe0f0aa1a0fa`. The publication archive
inlines only payload files no larger than 128 MiB. In this release `prepared.pt`, `evaluated.pt` and
`replicate_samples.pt` exceed that limit and remain at the persistent cache paths recorded in the
archive manifest. The archive records their absolute paths, byte sizes, artifact hashes and
manifest hashes and verifies every included member on read-back. The archive is therefore a review
bundle, not a self-contained copy of multi-gigabyte numerical evidence.

## Fresh reproduction scope

The fresh process receives no learner weights or fitted payloads. It reloads the pinned source data,
refits all seven models on CPU, recomputes schedules, samples, quadrature pairs, coalition games,
baseline methods, package-reference replays, sampling replicates, distances, curves and both plays,
then compares every deterministic payload recursively.

V3 fresh reproduction is PASS at
`~/.cache/arena_extended/gradientshap/reproduction-v3/reproduction.json` (SHA256
`0a7eb03c59a6266d5904abd4f73e63e1f927ace2e8f32323b4aaaf3437e97b9d`). It used no cached
model inputs and matched 1,442 arrays, 464,237,288 elements, 2,730 records and 872 scalars with
maximum absolute error zero. This establishes exact reproduction of the same seeded known models
and data; it is not an independent model or dataset replication.

Actual browser review is also PASS. It checked 63 rendered math nodes, opened all 43 dropdowns,
inspected all 20 figures, found no horizontal overflow or console/page errors, and retained 42
screenshots. The review record is
`~/.cache/arena_extended/gradientshap/native-dev-v3/visual_review.json` (SHA256
`88cf020dd7c39cccd2097f36496c8d54df468a3d4077de9c33c819be2085f5f9`). Independent figure
review is recorded in `visual_review_v3.md` beside it (SHA256
`1264775bd00b9309be63da772be5db0541b8f4efb18279c59d8747642877d02d`).

## When to regenerate

Regenerate native evidence after an intentional change to the lesson, student methods or tests,
visible study recipe, frozen data or protocol, reference adapter, diagnostics, plots, or the
execution/reproduction pipeline. Use new output directories, execute the complete own-answer learner
from a fresh CPU kernel, perform a second seven-model CPU refit without cached fitted inputs, compare
every deterministic value, and repeat the actual browser and full-resolution figure review. A
documentation-only edit to either README does not invalidate unchanged learner evidence, but the
publication/report manifest must be refreshed if it binds those README bytes. Do not regenerate to
erase method disagreement, Monte Carlo error, quadrature drift, background or noise sensitivity,
off-manifold distances, or unexplained source/package drift.

## Published records

- [learner_execution.json](learner_execution.json) binds the published notebook pair, instruction
  page, 20 namespaced figures, reviewed learner evidence, screenshots, source identities, native
  artifacts, reproduction record and archive receipt. Its SHA256 is
  `ddac3a2451f543ec2fa1532f63b762b81526aeb26305f9437d29181ff7491d69`.
- [native_summary.json](native_summary.json) indexes the seven fits and complete study dimensions
  without replacing the retained numerical payloads. Its SHA256 is
  `e44e1f0b7bc98ca4ab9c714cffe89121e74e1e0bfae002d12626abc3da72daa4`.
- [fresh_reproduction.json](fresh_reproduction.json) records the independent seven-model refit and
  exact recursive comparison. Its SHA256 is
  `0a7eb03c59a6266d5904abd4f73e63e1f927ace2e8f32323b4aaaf3437e97b9d`.
- [execution.tar.gz](execution.tar.gz) contains the reviewable execution record, completed notebook,
  before-audit seal, summaries, source snapshots, figures, manifests and payloads within the 128 MiB
  inline limit. Its manifest binds the three cache-resident larger payloads. Its SHA256 is
  `b37b9875e7d6e725146adfb2ca4cb96efe50df5164dff1c5df290a0ac7548ece`.

## Reproduce and republish

Run from the repository root with new output directories:

```bash
section="chapter16_shapley_attribution_baselines/exercises/part15_gradientshap"
learner_output="$HOME/.cache/arena_extended/gradientshap/learner-evidence-new"
fresh_output="$HOME/.cache/arena_extended/gradientshap/reproduction-evidence-new"

uv run python "${section}/build_notebooks.py" --check
uv run python -m pytest -q "${section}/gradientshap_contracts.py"

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run python "${section}/execute_learner.py" \
  --output-directory "$learner_output"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run python "${section}/reproduce_native.py" \
  --learner-directory "$learner_output" \
  --output-directory "$fresh_output"
```

After an actual browser review has written a passing `visual_review.json` into the learner output,
the release owner may run:

```bash
uv run python "${section}/publish_study.py" \
  --learner-directory "$learner_output" \
  --fresh-directory "$fresh_output"
```

Publication validates the learner, reproduction and visual evidence again before replacing
canonical course surfaces. Original ARENA chapters remain append-only, and this section's PASS does
not establish completion of other course requirements.
