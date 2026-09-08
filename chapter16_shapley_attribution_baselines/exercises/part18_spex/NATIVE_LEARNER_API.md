# Native learner integration

Current learner failure: the reviewed analytic draft stops before actual GPT-2
results. Its permanently false native switch cannot be the final learner route.

Target section claim: use a completed, fixed-query native study to test how well
the student's recovered signed spectrum predicts held-out word-mask scores.

Signature result: a visibly recomputed prompt-0, seed-0 SPEX fit at 1,824 requests,
beside actual model scores, query-matched baselines and shuffled-response failure;
all 32 declared prompts remain in the recorded budget curves.

Student functions: the same 22 visible exercise functions, including complete
SPEX recovery and whole-word fixed-position masking. The shared replay helper
only validates data, selects an already declared condition and passes the
student namespace through the existing fitting boundary.

Controls and baselines: real/shuffled refits, constant/additive predictors, saved
other-method metrics, subject-only counterfactuals and all incorrect unmasked
answers. Recorded aggregate curves are labelled as recorded study results.

Plays: retain both analytic plays; change a native prompt, seed or budget and
recompute the student's estimator on the actual saved training responses. Each
replay makes two estimator fits and four simple baseline fits, and zero model
queries. Replay after inspecting audit outcomes is exploratory.

Files: new `native_learner.py`, `native_learner_tests.py`; existing lesson prose,
cell definitions, builder, plots and learner tests. Frozen native/core files and
original ARENA remain unchanged.

Tests: synthetic full-inventory serialization and tampering controls; exact
float round-trip and ordered-label reconstruction; spy estimator verification
that no audit labels cross the fit boundary; own-answer namespace execution;
actual portable replay and rendered figures after the coordinated study completes.

## Shared API

```python
evidence = load_native_evidence(path=None, *, binding=None)
replayed = replay_condition(
    evidence, method="spex", methods=student_namespace,
    prompt_index=0, seed=0, budget=1824, output_dir=None,
)
```

The default data path is
`part18_spex/expected_outputs/native_query_fixture.json.gz`, overridable by
`ARENA_NATIVE_QUERY_FIXTURE`. Its adjacent `.data-contract.json` uses
`scripts.evidence_bundles.validate_json_data`: the existing limits are 16 MiB
encoded and 64 MiB decoded. The fixture uses plain JSON and exact finite float64
round trips, without fitted objects, NPZ or SQLite files. Both sections share
these bytes. Optional `binding={"fixture_sha256": ...}` pins an additional
explicit identity.

`evidence` contains `payload`, `plan`, `arrays`, `training_values` and `binding`.
The payload has all 1,152 condition records with measured metrics, query counts,
the largest 12 signed terms and explicit truncation flags. Fitted trees are omitted.
Full prediction vectors are retained only for the examples selected before outcomes:
`(prompt, seed, budget)` of `(0, 0, 1824)`, `(1, 0, 912)` and `(0, 0, 456)`. These
retain every stage's 512 predictions for pointwise replay checks at the frozen
`atol=1e-8, rtol=1e-7`. Proxy records retain all seven stage-error
scalars. All nine reference-stage statuses, errors and measured prediction
metrics are retained. Audit records include the 512 actual scores, unmasked
score and subject-only counterfactual score for every prompt.

`replayed` contains the selected student's real and shuffled records with full
predictions and spectra; `recorded_comparators` contains the other method's
corresponding recorded metrics. `replay` reports raw/unique mask counts, actual
refit counts, prediction-digest comparisons and zero additional model rows.

For the external full study, pass its directory and an explicit binding with
`native_execution_sha256`, `summary_hashes`, `freeze_sha256`,
`query_plan_directory` and `query_plan_manifest_sha256`. The exporter additionally
requires `fresh_reproduction_sha256`. No function starts a cold native run.

## Status

The first approved native run was interrupted and retained as a failure when
its measured runtime made the original 90-minute cap infeasible. The runtime
amendment and a future complete run are root-owned work. No actual fixture has
been exported and no native learner fit has been performed. Historical analytic
drafts remain intact. Actual fixture size, completed execution hashes and learner
results remain pending the complete study and coordinated reproduction.

The integration checkpoint is intentionally an unexecuted native draft. It is
not publication-ready, and no native prediction, method comparison or portability
claim about actual completed outcomes is established by its synthetic tests.
