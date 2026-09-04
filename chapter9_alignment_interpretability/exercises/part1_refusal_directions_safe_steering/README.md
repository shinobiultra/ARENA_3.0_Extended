# [9.1] Refusal Directions and Safe Steering Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current smoke-test artifact contract.
- `9.1_Refusal_Directions_and_Safe_Steering_exercises.ipynb` is the local
  learner notebook with stubs, visible tests, safe prompt-category tables, a
  toy layer sweep, PCA/SVD geometry check, steering/projection curves, and
  random-direction / label-shuffle / capability controls.
- `9.1_Refusal_Directions_and_Safe_Steering_solutions.ipynb` is the solved
  version of the same path. Outputs are intentionally cleared in git; execute it
  locally to see the generated tables and plots.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The current GPU verification loads pinned Pythia-70M-deduped hidden states and a
pinned Qwen2.5-0.5B-Instruct checkpoint on CUDA. It keeps the original sanitized
meta-prompt ladder, then runs a GT-2 aggregate refusal-direction path on the
public `josephmayo/refusal-compliance-pairs` dataset. The report checks held-out
direction separation, layer/position sweeps, PCA/SVD structure, label-shuffle and
random-direction controls, Qwen addition/projection-out behavioral effects,
exact revisions, dtypes, and measured VRAM. The CPU notebook contract also
checks the generated toy signature result. Raw dataset prompts and generated
completion text are not saved; the report stores aggregate metrics and prompt
hashes only.
