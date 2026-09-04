# [9.3] Emergent Misalignment Detection

ARENA-style learner files for the safe proxy-drift detection section.

- `9.3_Emergent_Misalignment_Detection_exercises.ipynb` is the learner-facing
  notebook with visible stubs, expected outputs, help, interpretation, solution
  dropdowns, controls, a play cell, and anomaly hunting.
- `9.3_Emergent_Misalignment_Detection_solutions.ipynb` mirrors the complete
  eight-exercise learner progression, defines every taught method inline,
  runs the immediate semantic tests, renders the signature plot, and asserts
  the committed Pythia hidden-state verification report fields.
- `artifacts.lock.yml` pins the known-onset toy organism plus the Pythia-70M
  CUDA preflight.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records the expected proxy-drift
  toy metrics, hidden-state metrics, and control slots.

The learner path starts with a generated model organism where the true causal
drift direction is activation dimension 0, the true onset is checkpoint 3, and a
delayed behavior proxy fires at checkpoint 4. Students implement the
activation-difference direction, held-out thresholded detector, onset metric,
behavior-alignment check, projection mitigation, capability-cost calculation,
and random-direction / label-shuffled controls.

The graded real-model path loads pinned `EleutherAI/pythia-70m-deduped`,
extracts hidden states on safe generated proxy-drift prompt pairs, trains a
thresholded hidden-state drift direction, checks held-out detection across five
benign proxy kinds, compares label-shuffled and fixed random directions, aligns
feature scores with a safe next-token behavior proxy, and tests
projection-style mitigation through the LM head. It uses hidden states and
logits only; it does not generate completions, train adapters, or claim a
harmful finetune or full emergent-misalignment reproduction.
