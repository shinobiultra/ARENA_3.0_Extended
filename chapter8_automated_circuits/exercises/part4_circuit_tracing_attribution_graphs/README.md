# [8.4] Circuit Tracing with Attribution Graphs Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current artifact contract.
- `8.4_Circuit_Tracing_with_Attribution_Graphs_exercises.ipynb` is the local
  learner notebook with seven tested steps from the planted component DAG to
  exact path interventions and matched graph controls.
- `8.4_Circuit_Tracing_with_Attribution_Graphs_solutions.ipynb` exposes every
  taught implementation inline and regenerates the graph, heatmap, and metric
  figures from those visible functions.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The flagship claim is the exact planted component theorem: integrated-gradient
edge scores recover the country-feature to Paris-logit path, path-only and
path-removed interventions quantify faithfulness/completeness/minimality, and
same-size random plus reversed-edge controls fail. The graded report also loads
the pinned `NeelNanda/GELU_1L512W_C4_Code` checkpoint through TransformerLens on
CUDA. Its top-1 `position_5 -> position_5` graph remains a mechanics preflight,
not a meaningful component-level circuit claim.
