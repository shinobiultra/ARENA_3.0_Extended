# [11.1] PCA, SVD, and Geometry Controls Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current smoke-test artifact contract.
- `11.1_PCA_SVD_and_Geometry_Controls_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `11.1_PCA_SVD_and_Geometry_Controls_solutions.ipynb` runs the local reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The current GPU verification loads pinned `EleutherAI/pythia-70m-deduped`
hidden states for generated weekday and month prompt splits. It checks raw
template-dominated transfer failure, template-centered held-out label transfer,
matched-pair retrieval, permuted-label controls, white-noise controls, no text
generation, and measured VRAM.
