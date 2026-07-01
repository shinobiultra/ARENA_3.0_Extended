# [8.4] Circuit Tracing with Attribution Graphs Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current artifact contract.
- `8.4_Circuit_Tracing_with_Attribution_Graphs_exercises.ipynb` is the local
  learner notebook with stubs and visible tests.
- `8.4_Circuit_Tracing_with_Attribution_Graphs_solutions.ipynb` runs the
  reference implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded report path loads the pinned `NeelNanda/GELU_1L512W_C4_Code`
checkpoint through TransformerLens on CUDA, builds a top-1 graph from the real
EAP position-edge matrix, and verifies target-metric explanation, top-path
perturbation, an alternative wrong-position graph baseline, and a
summary-predicted counterfactual.
