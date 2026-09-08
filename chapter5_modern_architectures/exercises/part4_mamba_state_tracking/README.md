# [5.4] Mamba State Tracking Verification Assets

Generated support files for the roadmap verification contract.

- `5.4_Mamba_State_Tracking_exercises.ipynb` is the learner-facing notebook
  with local implementation stubs, direct tests, expected outputs, and common
  bug notes.
- `5.4_Mamba_State_Tracking_solutions.ipynb` validates the section-local
  reference implementation against the visible tests.
- `artifacts.lock.yml` pins the current smoke-test artifact contract.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The current GPU verification trains a tiny Mamba bracket-depth state classifier,
checks longer-sequence generalization, verifies that a random-label training
control fails, compares against a trained tiny causal Transformer baseline,
performs learned Mamba hidden-state interventions with matched random-direction
controls, and loads pinned `state-spaces/mamba-130m-hf` hidden states on CUDA.
The lock and report should record exact revisions, seeds, dtypes, controls, and
measured VRAM.

This evidence is scoped: the official checkpoint path is a hidden-state
extraction preflight, while the trained Mamba, Transformer baseline, and
learned-state intervention paths use generated bracket-depth model organisms.
