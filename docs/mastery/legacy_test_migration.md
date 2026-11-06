# Legacy test migrations — 2026-10-15

The initial offline repository pytest run failed **during collection**, with five errors.
At that checkpoint the affected test files were unchanged versus starting HEAD
`dec028ec8ed8accb40bf0d0a7315bb72f0c1fcc6`.
[Raw output](evidence/cpu-regression-collection.txt).

The five migrations are now implemented: **114 tests passed in 3.49 seconds**
across the five affected root test files; see [combined output](evidence/legacy-migration-checks.txt).
Repository collection succeeds with **1,346 tests** at this checkpoint;
[collection output](evidence/legacy-collection-after.txt). Collection is not a full-suite pass.
After the first geometry/circuit integration checkpoint, collection
reaches **1,420 tests with no errors**; [first continuation output](evidence/continued-collection.txt).

The final continuation collects **1,522 tests without errors** and passes **622
focused integration tests**, including the five migrated files. See
[collection](evidence/second-continuation-collection.txt) and
[integration output](evidence/second-continuation-final-checks.txt). The legacy report
audit described below remains failing; no whole-repository pass is claimed.

| Root test | Missing old API | Current learner location | Starting-state evidence |
| --- | --- | --- | --- |
| `test_capstone_research_sprint.py` | `sprint.py`, `sprint_tests.py` | 10.1 same directory, `solutions.py` / callable `tests.py` | Missing already in HEAD, index and working tree. |
| `test_data_shapley.py` | 16.7 `part7_data_shapley_in_one_training_run` | 16.9 `part9_data_shapley` | Old files present in HEAD, staged deleted before this work. |
| `test_inrun_data_shapley.py` | Old `inrun_methods`, `inrun_tests`, `inrun_report` | 16.9 `part9_data_shapley` | Old files present in HEAD, staged deleted before this work. |
| `test_sparse_concurrent_publication.py` | 16.18 `publish_study`, `run_native` | 16.6 `part6_spex_proxyspex` for learner methods only | Old files present in HEAD, staged deleted before this work. |
| `test_sparse_verification_integration.py` | 16.18 `native_report`, `native_report_tests` | 16.6 learner methods; no replacement native receipt API | Old files present in HEAD, staged deleted before this work. |

These are API migrations, not import-path typos. Current capstone functions include
`mean_difference_direction`, `train_model`, `paired_bootstrap`, `ResearchLog` and
`run_pilot`. Data Shapley now teaches `backprop_factors`, `per_example_grads`,
`first_order_credit`, `second_order_credit`, `local_game_values` and
`run_in_run_shapley`, checked with independent autograd/permutation oracles.
The current SPEX lesson's method tests do not certify the removed native
publication/receipt protocol. Do not silently replace those guarantees with a
method test or restore the old prototypes merely to make collection green.

## Retained guarantees

- [Capstone](migrations/capstone.md): visible registration binds data/splits and
  enforces one-time DEV/confirmation stages and actual readout budgets. The current
  CPU solution ran afresh: 46 code cells and 10 figures in six seconds. Historical
  notebooks are archived with hashes.
- [Shapley](migrations/data_shapley.md): current visible numerical methods retain
  independent historical tolerance checks, including rejection of a `+1e-8` ghost-product
  mutation. Historical receipt comparisons have an explicit timing-path allowance.
  The current CPU solution ran afresh: 56 code cells and 13 figures in 30 seconds;
  scientific outputs match the archive, with measured timings refreshed.
- [Sparse evidence](migrations/sparse_evidence.md): a minimal provenance-bound
  historical validator/stager preserves receipt and publication guarantees. It
  neither restores retired learner prototypes nor certifies current 16.6 scientific results.

Independent review verified the retained sparse source bodies and provenance,
then closed both Shapley tolerance and capstone budget-test findings. No tests were
skipped or removed to resolve collection.

## Further runtime debt

A broader related diagnostic reached **398 passed, one failure and 40 errors**.
The failure concerns missing chapter-5 `artifacts.lock.yml`; the errors arose from
a synthetic learner-contract fixture importing deleted 16.18/16.19 notebook
builders. That fixture is now migrated to explicit temporary synthetic trees,
preserving all five original test bodies and all 38 mutation cases. An independent
review found no weakening; 100 sparse tests pass. The broader rerun now reaches
**438 passed, one failure and zero errors**. The missing chapter-5 lock remains
the first exception, not the full remaining debt. A [complete declared-report
inventory](migrations/report_inventory.md) found 73 of 88 legacy records missing
one or both files (145 paths). Of 15 existing pairs, 13 have no contract blockers and
the preserved 11.1 and 11.2 legacy reports each have nine freshness blockers after
canonical generation. Its original receipts remain archived; no old report was rewritten
to certify new source. Historical identities in
`scripts/sparse_interaction_learner_contract.py` remain intentional. This diagnostic
does not establish a whole-repository pass.

Generated notebooks also had empty code-only fields on Markdown cells. The
extension build wrapper now removes those invalid fields and validates the schema.
[The normalization receipt](evidence/notebook_schema_normalization/manifest.json)
proves all 34 existing artifacts retained every complete code cell, output and
execution count. Exact fresh capstone/Shapley artifacts from before normalization
are archived beside it, keeping the original execution-receipt hashes recoverable.
