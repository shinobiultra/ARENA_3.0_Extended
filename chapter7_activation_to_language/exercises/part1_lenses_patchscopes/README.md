# [7.1] Logit Lens, Tuned Lens, and Patchscopes Verification Assets

Generated support files for the roadmap verification contract.

- `7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_exercises.ipynb` is the learner
  notebook with local stubs for logit lens, tuned lens, attention lens,
  Patchscope templates, counterfactual controls, and random-activation controls.
- `7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_solutions.ipynb` executes the
  visible tests against `solutions.py` and checks the committed CUDA report
  highlights.
- `artifacts.lock.yml` pins the current artifact contract.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded report path loads the pinned `NeelNanda/GELU_1L512W_C4_Code`
checkpoint through TransformerLens on CUDA, trains a ridge affine lens on cached
safe activations, evaluates held-out next-token decoding, runs a real
attention-lens diagnostic, compares activation-conditioned clean logits against
corrupt text-only baselines, and checks counterfactual plus random-activation
controls.
