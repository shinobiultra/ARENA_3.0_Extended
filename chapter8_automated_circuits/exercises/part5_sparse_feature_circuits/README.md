# [8.5] Sparse Feature Circuits

Course-ready support files for the Sparse Feature Circuits section.

- `artifacts.lock.yml` pins the current notebook-contract artifacts.
- `8.5_Sparse_Feature_Circuits_exercises.ipynb` is the local learner notebook
  with stubs, expected outputs, help dropdowns, visible tests, and the
  report-backed verification contract.
- `8.5_Sparse_Feature_Circuits_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA/official-artifact report
  checks with saved outputs.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the historical `run_smoke_test` contract hook.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The current GPU report checks the toy ladder, Pythia-70M-deduped residual
effects, released SAE state-dict and feature-attribution controls, a
100-example official-code sparse feature graph, and held-out official
faithfulness on 40 `simple_test` examples.

Scope: this is a GT-0 learner implementation contract plus pinned local CUDA
evidence. The residual-feature preflight, one-layer SAE attribution smoke, and
official saved-graph checks are deliberately described with claim boundaries;
they are not a full paper-level Sparse Feature Circuits replication claim.
