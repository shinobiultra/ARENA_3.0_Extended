# [14.1] JEPA and World-Model Controls Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current smoke-test artifact contract.
- `14.1_JEPA_and_World_Model_Controls_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `14.1_JEPA_and_World_Model_Controls_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The current GPU verification loads pinned `facebook/vjepa2-vitl-fpc64-256` on
CUDA and extracts encoder features from generated 8-frame videos. It records
same-object versus different-object similarity, a late-occlusion object
permanence contrast against an absent-object same-occluder control, feature
shape/finiteness/non-collapse checks, exact revisions, dtypes, and measured
VRAM.
