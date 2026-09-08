# Native report assembly for SPEX and ProxySPEX

Current learner failure: the analytic drafts and interrupted first native attempt
do not establish the completed native lesson. This adapter must reject them.

Target claim: a complete fixed-query GPT-2 study, strict cold reproduction, and
actual portable student replays support the disclosed lesson scope. Predictive
superiority and author-reference success are not acceptance conditions.

The signature results, student implementations, controls and play cells belong
to the two notebooks. This module verifies their evidence and writes supporting
reports outside the repository. It adds no learner-facing verification cells.

Owned files: `native_report.py`, `native_report_tests.py`, this API document.
No frozen native input, method implementation, shared gate or publisher is owned
by this adapter. Its tests use synthetic evidence and do not approve publication.

## Entry point

```python
report = native_report.build_report(
    section,                # "16.18" or "16.19"
    native_run,             # completed cold native run directory
    fresh_run,              # separate completed cold reproduction directory
    learner_execution,      # actual portable own-answer receipt JSON
    fixture,                # exact shared native_query_fixture.json.gz
    artifact_inputs,        # {final_repo_relative_path: actual_existing_file}
    output,                 # new external directory; never a canonical path
)
```

The command-line interface uses `--section`, `--native-run`, `--fresh-run`,
`--learner-execution`, `--fixture`, `--artifact-inputs`, and `--output`.
`--artifact-inputs` is a JSON file containing that canonical-to-physical mapping.

Supply the final exercise/solution notebook pair, exported instruction page,
every actual prefixed figure, and data/provenance instructions. The adapter adds
current native/learner/report source files, the frozen receipt source inventory,
project environment files, portable data and its sidecar, and the native/fresh/
learner/comparison receipts. Missing files and substituted native source bytes
are rejected. Input manifests use the existing `path`, `sha256`, `size_bytes`
schema with canonical repository-relative paths, even while the exact bytes are
staged externally. They therefore remain usable after installation.

The output contains `verification_report.json`, `artifacts.lock.yml`,
`verification_report.schema.json`, `native_summary.json`, and
`report_artifacts.json`. The last file supplies physical artifact records for the
publisher's `verification_report`, `artifacts_lock`, `verification_schema`, and
`native_summary` roles, plus the complete physical report-input mapping. Merge
those roles into the author's section publication manifest. The report does not
bind its downstream review archive; the publisher binds that archive separately.

The adapter invokes the publisher's complete-native-pair validator, verifies the
portable data contract and semantic schema, checks actual learner sources and
replay receipts, and streams saved prediction arrays to recompute every exported
condition metric and reference outcome. It does not load fitted states, query
GPT-2, fit estimators, launch a reproduction, install artifacts or approve review.

## Stable integration hooks

```python
has_native_evidence(report) -> bool
receipt_blockers(section, summary, learner, fresh, report) -> list[str]
```

Root owns the explicit `has_native_cpu_model_evidence` branch and fixture gate.
The branch must retain its existing CPU lock/device preconditions and call
`has_native_evidence` only for these two notebook identities:

- `16_18_find_the_interactions_without_trying_every_subset`
- `16_19_explain_the_proxy_check_the_model`

The fixture gate retains common section identity, file existence, exactly one
manifest binding, SHA256, and byte-size checks. For these sections only, it then
delegates to `receipt_blockers`, instead of imposing classifier refits or exact
zero-error reproduction. The required triplet remains `native_summary.json`,
`learner_execution.json`, and `fresh_reproduction.json`. The report also binds
`native_execution.json`, `fresh_native_execution.json`,
`native_comparison_summary.json`, and the shared portable fixture/sidecar.

The current generic `run_record` assumes a `solutions.py` smoke/GPU entry point.
Keep this adapter a standalone explicit entry point until root adds a truthful
dispatch using its completed evidence inputs. Do not wrap stored metrics in a
fake GPU execution or claim new model work during report assembly.

## Receipt fields

`native_summary.json` uses schema
`arena.sparse-interaction.native-summary.v1`, study
`16.18_16.19_spex_proxyspex`, section identity, and status
`COMPLETE_VERIFIED_STUDY`. It contains:

- `inventory`: all 32 prompts, 1,152 distinct conditions, 512 audit masks per
  prompt and nine ordered reference stages, plus measured fits and physical
  queries checked against the fixed prospective inventory.
- `bindings`: `native_execution_sha256`, `fresh_execution_sha256`,
  `fresh_reproduction_sha256`, `learner_execution_sha256`, `fixture_sha256`.
- `condition_outcomes`, `reference_outcomes`, `prompt_outcomes`: every original
  recorded outcome, with negative fidelity, undefined zero-variance metrics,
  shuffled results, reference exceptions and incorrect unmasked scores retained.
- `comparison`: original/fresh array/element/scalar/container counts, status,
  mismatch count, exactness, number of changed numerical leaves and the full
  external comparison's hash. A toleranced match is never relabelled bitwise exact.
- `portable_replay`: the unchanged learner field.
- `replay_outcomes`: expanded actual replay records with `record_path`, `hashes`
  and `replay`, checked during assembly and usable without the original cache.

The two learner receipt formats are retained. Section 16.18 records two absolute
native-record directories in `portable_replay.replay_records`. Section 16.19
records two named JSON hashes in top-level `replay_receipts` and zero queries in
top-level `model_queries`. Both require the actual real/shuffled comparisons,
every estimator/baseline stage, the declared pointwise tolerance, zero new model
rows, and a fit boundary that never receives audit labels.

The report's `metrics.gpu_test` contains the same measured inventory and digest
bindings, `fresh_compared_arrays`, `fresh_compared_elements`,
`fresh_bitwise_exact`, `fresh_numerical_difference_leaves`,
`own_answer_code_cells`, and `embedded_figures`. Explicit fields state:

```text
native_model_queries_in_report = 0
native_estimator_fits_in_report = 0
recorded_study_reexecuted_in_notebook = false
method_superiority_is_acceptance_gate = false
reference_success_is_acceptance_gate = false
publication_approval_claimed = false
```

Report acceptance means complete checked evidence for this bounded claim. The
generic `known_failures` field describes report-contract failures; observed
reference failures remain explicitly in `metrics.native_outcomes`, the summary,
the portable fixture and original external study. No source-parity or
control-quality success flag is invented. Measurement timestamps come from the
native receipt, while report-assembly time is recorded separately. Report wall
time is the sum of the two cold native children; their full resource receipts
and the separate reproduction/learner timings remain hash-bound.

## Current validation boundary

Run the focused `native_report_tests.py` suite. It exercises complete synthetic
coverage, poor fidelity, failed references, nonexact tolerated reproduction,
missing/corrupt/mismatched receipts, both replay formats, physical prediction
array checks, standard report schema/freshness with a synthetic canonical tree,
and actual CLI rejection of an incomplete native receipt.

The approved 180-minute resource-only amendment permits a future fresh attempt.
The first interrupted run remains failed. No complete native run, portable
native learner execution, actual accepted native report or publication approval
was produced by this adapter's implementation or tests.
