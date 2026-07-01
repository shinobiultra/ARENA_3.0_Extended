# [9.3] Emergent Misalignment Detection Verification Assets

Generated support files for the roadmap verification contract.

- `9.3_Emergent_Misalignment_Detection_exercises.ipynb` is the learner-facing
  notebook with section-local stubs and visible tests.
- `9.3_Emergent_Misalignment_Detection_solutions.ipynb` runs the reference
  visible tests and asserts the committed Pythia hidden-state verification
  report fields.
- `artifacts.lock.yml` pins the toy contract plus the Pythia-70M CUDA preflight.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records the expected proxy-drift
  hidden-state metrics and control slots.

The graded local path loads pinned `EleutherAI/pythia-70m-deduped`, extracts
hidden states on safe generated proxy-drift prompt pairs, trains a thresholded
hidden-state drift direction, checks held-out detection across five benign proxy
kinds, compares label-shuffled and fixed random directions, aligns feature
scores with a safe next-token behavior proxy, and tests projection-style
mitigation through the LM head. It uses hidden states and logits only; it does
not generate completions, train adapters, or claim a harmful finetune or full
emergent-misalignment reproduction.
