# September 7 GradientSHAP graceful-stop checkpoint

The user asked to finish running work and stop for today. All seven workers
finished their assignments; the accidental duplicate child was interrupted
without a known edit. Native execution and reproduction are terminal, and the
browser-review server on port 8770 was stopped. No publication, registration,
new topic, commit, push, cleanup or restart was started. Wait for the user to
resume. The full-course goal is unfinished; voluntary stopping is neither goal
completion nor a technical blocker.

## Current result and location

Canonical sources are in
`chapter16_shapley_attribution_baselines/exercises/part15_gradientshap`.
The section title is **Expected Gradients: Which Expectation Does GradientSHAP
Estimate?**, provisional number 16.15. Its nine core methods remain visible in
the learner, with immediate semantic tests, expected outputs, help and Solution
dropdowns. Training and scientific evaluation are visible too. The strict build
contains 126 cells, 79 code cells, nine exercises, 20 figures and two editable
plays. This is a successfully executed and visually reviewed learner source;
the canonical paired notebooks and instruction page are not yet published.

The completed executable notebook is
`/home/bitwise/.cache/arena_extended/gradientshap/native-dev-v3/completed_learner.ipynb`.
The rendered review copy is under its `html_review/` directory. Both are cache
evidence, not a substitute for final course publication.

The signature result compares sampled path credit with exact feature-coalition
credit, through convergence panels, seven-model heatmaps and an actual audit
recipe waterfall. In the fixed clean_0/source-1 audit example, the 64-donor base
is 35.55 MPa, prediction 70.67 MPa, K=2048 Expected Gradients sum 34.78 MPa and
finite-sample completeness residual 0.35 MPa. The two plays edit a threshold
path and a real held-out recipe/model/background/noise setting.

Controls retain three clean networks, three paired shuffled-target networks,
ridge, exact 256-coalition games, DeepSHAP, endpoint gradient, quadrature at
256/512 nodes, five backgrounds, three noise branches, 64 replicates, geometric
distance diagnostics and 100 random replacement orders. Ordinary plug-in SE
uses unbiased sample variance; neither sample SD nor SE is claimed unbiased.
Quadrature is approximate at ReLU jumps, finite Monte Carlo need not be complete,
and noise changes the population target. These observational attributions do not
establish causal effects of changing concrete ingredients. The data and seven
models reproduce 16.14 and are not an independent dataset/model replication.

## Final execution, reproduction and review evidence

All paths below are relative to
`/home/bitwise/.cache/arena_extended/gradientshap/` unless stated otherwise.

| Artifact | SHA-256 | Result |
| --- | --- | --- |
| `native-dev-v3/execution.json` | `d9b0506e72ff849d17eb76ddba23f424ea33824dd9086fcbba954888b1af6973` | PASS; CPU, 60.045227675 s, 126/79/20, no changed sources |
| `native-dev-v3/completed_learner.ipynb` | `6c2c458fa715f52a62f123ce19c8e0d88c9776f4255fe32b2d1d68a1a8546055` | Completed solely from its own answers |
| `reproduction-v3/reproduction.json` | `0a7eb03c59a6266d5904abd4f73e63e1f927ace2e8f32323b4aaaf3437e97b9d` | PASS; actual fresh subprocess, seven refits, no cache reuse |
| `native-dev-v3.execution_audit.json` | `6ff682be629d32f1a777af09ae239857c1bce3285d065eb498d1f0b92fa31529` | Scientific scope and run lineage |
| `native-dev-v3/visual_review.json` | `88cf020dd7c39cccd2097f36496c8d54df468a3d4077de9c33c819be2085f5f9` | Browser/human review PASS |
| `native-dev-v3/html_review/browser_review.json` | `c44aad979edbd753b4ce39d57f88bad4cdf507aa9164bf502957e279eb3de2a3` | 63 math nodes, 43 opened dropdowns, 20 loaded figures |
| `native-dev-v3/visual_review_v3.md` | `1264775bd00b9309be63da772be5db0541b8f4efb18279c59d8747642877d02d` | All 20 actual PNGs inspected at original and 1200-pixel width |

Fresh reproduction compared 1,442 arrays containing 464,237,288 elements,
2,730 records and 872 scalars, with maximum deterministic error zero. All seven
deterministic serialized payloads also match. It took 53.8348 seconds in the
wrapper, including a 48.6004-second native child. Timings are descriptive, not a
benchmark or part of the deterministic comparison. The direct learner's timing
payload is empty; fresh phase timings are retained separately.

All 21 initial/standardized/raw model-state hashes match before attribution.
All seven models and all 272 held-out rows are retained. SHAP and Captum compare
3,808 vectors / 30,464 coordinates in total; maximum same-draw errors are
5.684341886080802e-14 and zero respectively, against frozen atol=rtol=1e-10.

Browser review at 1440x1000 reported zero MathJax, console or page errors and
zero horizontal overflow. The record binds 42 screenshot files and eight
inspected contact sheets. Seventeen unchanged figure screenshots reuse the
completed v2 inspection; the three corrected figures and updated execution
paragraph were re-inspected directly in v3. The independent figure reviewer
also inspected every actual v3 PNG, including dense figure 15 at notebook width.

## Data, immutable freeze and storage

The protocol freeze was written at `2026-09-07T02:01:38.970418+00:00`, with zero
native fits and attribution runs before freeze. Do not edit or reformat:

- `protocol.md`: `4ed1d1d45a0b5db304cb45e7c5abe1c85c97434e53b4c9c156ce05b2b489a83b`
- `data_manifest.json`: `bcf41f96c9f6839ee4cbf24df82ca8467b35a6ed9090a9c4c56fce0b1a23120c`
- `gradientshap_data.py`: `9c9c1016ffd6fe839a004c06e801cdcd24d4d71a85ccd3f0707168dec390c7ce`
- Freeze record: `protocol_freeze.json`, SHA `79fc20a6b600cc17b9e942927d44ac135e192538e124fcf7b7889407940afcf5`

Independent pre-fit review is `review-resume-v1/independent_protocol_review.md`,
SHA `8ff9ae80ae42665a4f33fe70df1d0f581a814986d9d7dddbfc4071899f31c38c`.
The frozen manifest specifies every RNG namespace, all 64 replicate seeds,
package draw ordering, smoothing seeds and reference batches. Do not change
scientific thresholds after viewing the outcomes.

Native payloads are in `native-dev-v3/figures/native/`; fresh equivalents are in
`reproduction-v3/fresh_run/`. Each contains seven deterministic `.pt` artifacts
plus separate `timings.pt`, totaling 3,548,279,909 bytes on the learner side.
The full `(64,3,64,2048,8)` float64 replicate tensor is externalized in
`replicate_samples.pt`, not reduced or replaced by summary statistics:

- Serialized bytes: 1,610,614,447.
- File SHA: `46238cad0348f56be4807507256233884af3588037a434197f0cbe0f0aa1a0fa`.
- Value SHA: `521ce9f2e1f23c239e6863166333fb0a3f5f52dbec980a4a527118cee7391c6f`.

Recursive manifests bind every retained array, including full primary samples,
hybrid predictions and shared donor/alpha schedules. Fresh validation rereads
the original and fresh large payloads, recomputes value hashes and checks the
shallow descriptor in `evaluated.pt`. The successful archive dry run is
`archive-check-v3/execution.tar.gz` (37,376,425 bytes, SHA prefix `3a825689`).
It verifies every member after writing and binds cache-resident prepared,
evaluated and replicate files larger than 128 MiB. Preserve the cache: that tar
is not a self-contained copy of those large files. A direct own-answer learner
has no native `run.json`; the publisher now correctly treats that file as optional.

## Preserved development history

- `fit-check-v1`: fit-only PASS for all 21 state hashes; no native attribution.
- `native-dev-v1`: own-answer route, FAIL before fitting in cell 19. A negative
  dtype fixture inherited the lesson's float64 default. The fixture now constructs
  an explicitly float32 module; the nine public method ASTs did not change.
- `native-dev-v2`: full own-answer PASS, 64.3529 seconds, 126/79/20, unchanged
  sources. `reproduction-v2` independently matches the same full array scope.
- V2 visual review found label overlaps in figures 16/18, excessive figure-19
  width, and stale prose claiming execution was still pending. Those historical
  failures remain recorded. Presentation/prose fixes led to v3; no scientific
  outcome or threshold was altered. V2 visual status must not be relabeled PASS.

The v2 report-payload helper result is `native-dev-v2.report_payload_metrics.json`.
It is diagnostic evidence, not an accepted normal verification report. Its
165.7736 MPa maximum Monte Carlo completeness residual covers all five budgets,
including K=8; do not mislabel it as a K=2048-only result. Exact feature-credit
gaps, quadrature drift and replacement superiority are observations, not gates.

## Changed files and validation at the stop

In `part15_gradientshap`, saved work covers the methods/tests, data/protocol/
manifest/freeze, study/tests, author adapters/tests, diagnostics/plots/tests,
lesson/builder/README, native/learner/reproduction/publication harness and
contracts. Root also added `solutions.py`, `tests.py`, `artifacts.lock.yml`,
`verification_report.schema.json` and `author_reference.json` for the pending
normal report. Exact source hashes used by v3 are in `execution.json` and its
source snapshot, and are cross-bound by the fresh reproduction record.

Global changes are `scripts/audit_course_surface.py`,
`tests/test_course_surface_audit.py`, `scripts/audit_report_evidence_contracts.py`
and `tests/test_report_evidence_contract_audit.py`. The course audit checks all
nine visible methods, own-answer AST parity, the actual ordered scientific path,
the state gate and complete external sample/reference scope. The new report
audit checks real evidence bindings and normalized same-draw numerical errors.
At the stop, `docs/roadmap_requirement_registry.yml` received an accurate evidence
pointer and remaining-work description for GradientSHAP; its status stays
UNIMPLEMENTED and its section remains unassigned until final registration.

Completed focused checks include 47 section tests with one explicit native-fit
skip, 14 harness contracts, 21 course-surface tests and 17 report-evidence audit
tests. Method tests pass under both global float32 and float64 defaults. Strict
126-cell/9-exercise build, compilation, course-surface CLI, full learner,
independent reproduction and both visual gates pass. These are overlapping test
suites; do not add their counts into an invented unique-test total.

Original chapters 0–4 remain append-only against
`f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0`. At the stop, the refreshed
preservation audit passed with zero blockers (1,805 changed paths before this
new checkpoint file). Completed 16.14 source and evidence were not edited.
The existing dirty worktree is intentional; no reset, clean or broad staging.

## Exact resume route and remaining limitations

1. Read this checkpoint and the final evidence, then confirm the v3 source hashes
   still match. Preserve the immutable freeze and existing output directories.
   Use only `gpt-5.6-sol` / `xhigh` for authorized workers/reviewers. Keep native
   work on CPU with four threads; leave unrelated GPU work untouched.
2. Independently re-review root's last report-frontend fixes. The earlier review
   at `review-resume-v1/report_frontend_review.md` (SHA `159594547085dc4f4a87a4fd4d1341599a4683b12454ab8709f11169cc336a64`)
   found three issues: original/fresh fit counts sharing fresh evidence; author
   coverage trusting declared counters; and bogus hash/large parity-error
   acceptance. Root fixed all three and 17 audit tests passed, but no independent
   re-review has yet accepted the fixes. `_study_metrics` and `_reference_metrics`
   successfully read actual retained v2 payloads after those changes. The normal
   `run_full_experiment` path still needs execution after publication. A final
   formatting attempt failed because `uv run black` is not installed; formatting
   is not verified. Do not alter held scientific files for cosmetic cleanup.
3. Once review is satisfactory, publish the verified v3 pair/page/figures using
   the command below. The harness's live fresh validator and archive dry run
   already pass. Publication was deferred for the user's stop, not a native or
   visual failure. Keep archival evidence and large external bindings intact.
4. Bootstrap the normal report with the repository's existing report runner API,
   following 16.14's approach if registration is needed first. Discover current
   `run_record` arguments from source; do not invent them. The report must launch
   its own fresh subprocess, validate actual payload coverage/hashes and bind
   the published artifacts. Its legacy `run_gpu_test` name still runs on CPU.
5. Inspect `~/.cache/arena_extended/gradientshap/integrate_published.py` before
   applying it. It was only compiled, never applied. It checks publication/report
   evidence and then appends config/style/method/ladder/roadmap/navigation entries.
   It raises on an existing 16.15 entry: do not treat it as idempotent. Check actual
   publication field names and fix missing spaces in its proposed human-facing
   descriptions. Never rerun the historical non-idempotent generators
   `revise_protocol_before_freeze.py`, `build_report_contract.py`,
   `add_report_audit.py` or `fix_report_frontend.py`; they can overwrite final fixes.
6. Build/check the merged config and run the normal registered 16.15 report, style,
   course-surface, report-evidence, hard-exercise-ladder, original-preservation
   and full-roadmap audits. Update only the selected GradientSHAP expectation in
   tests after its actual promotion. Archive final reports/gates/source evidence,
   update the section/root README and HANDOFF, and inspect published learner links.
   Do not rerun v3 merely to fill time; new source changes or failed validation
   are what justify a fresh bound execution.

Deferred publication command, from the repository root:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
uv run python chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/publish_study.py \
  --learner-directory /home/bitwise/.cache/arena_extended/gradientshap/native-dev-v3 \
  --fresh-directory /home/bitwise/.cache/arena_extended/gradientshap/reproduction-v3
```

No GradientSHAP registry promotion has occurred. The live course registry remains
205 requirements: 51 TAUGHT, 36 PREFLIGHT_ONLY, 20 TOY_ONLY, 13 DEFERRED and 85
UNIMPLEMENTED. The previous global seven-file test run had one existing
readability failure covering 78 legacy source issues, none in 16.14; that run
was not repeated at this stop. Passing the present section does not complete
the course. The selected Shapley-Taylor/Faith-Shap research packet remains in
`~/.cache/arena_extended/shapley_interactions/`; no implementation or promotion
started. Continue the full selected-topic goal only after the user resumes.
