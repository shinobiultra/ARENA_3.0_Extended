# [16.5] VLM Modality and Region SHAP Verification Assets

Generated support files for the roadmap verification contract and the pinned
real CLIP VLM SHAP preflight.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  rendered-shape CLIP modality/region SHAP preflight.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.
- `16.5_VLM_Modality_and_Region_SHAP_exercises.ipynb` is the learner notebook
  with local exact Shapley, modality SHAP, and region SHAP exercises.
- `16.5_VLM_Modality_and_Region_SHAP_solutions.ipynb` executes the reference
  implementation, visible tests, and committed CUDA report checks.

The graded GPU path in `solutions.py` loads a pinned CLIP ViT-B/32 checkpoint,
renders safe shape controls, computes modality and structured-region SHAP from
real CLIP logits, verifies target-vs-distractor and object-localization margins,
and records measured VRAM.
