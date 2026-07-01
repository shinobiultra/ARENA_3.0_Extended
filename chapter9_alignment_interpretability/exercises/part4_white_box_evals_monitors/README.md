# [9.4] White-box Evals and Monitors Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the toy contract plus the Pythia-70M CUDA preflight.
- `9.4_White-box_Evals_and_Monitors_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `9.4_White-box_Evals_and_Monitors_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records the expected white-box monitor
  metrics and control slots.

The graded local path loads pinned `EleutherAI/pythia-70m-deduped`, extracts
hidden states on safe generated eval records, trains a thresholded white-box
monitor direction, calibrates held-out failure scores, compares against a real
next-token `pass`/`fail` black-box proxy, validates label-shuffled and fixed
random-direction controls, and records false-positive documentation status. It
uses hidden states and logits only; it does not generate completions or claim a
broad harmful-content monitor benchmark.
