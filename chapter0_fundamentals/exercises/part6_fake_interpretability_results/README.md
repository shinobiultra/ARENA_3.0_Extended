# [0.6] How to Know When an Interpretability Result Is Fake Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current diagnostic contract and CUDA diagnostic
  preflight.
- `0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb`
  is the local learner notebook with stubs and visible tests.
- `0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb`
  runs the reference implementation, visible tests, and committed CUDA report
  checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the historical `run_smoke_test` contract hook.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

This section is a GT-0 diagnostic lab: the bogus results are synthetic and
known in advance, so every detector should flag its assigned failure mode.
It is not evidence that a full real-model mechanism has been validated.
The graded GPU path records label-leakage accuracy, cherry-pick inflation,
probe train-vs-heldout gap, and random-direction control rejection on CUDA.
