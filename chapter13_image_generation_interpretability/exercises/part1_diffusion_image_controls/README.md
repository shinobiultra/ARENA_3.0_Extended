# [13.1] Diffusion and Image-Generation Controls Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current smoke-test artifact contract.
- `13.1_Diffusion_and_Image_Generation_Controls_exercises.ipynb` is the local
  learner notebook with stubs and visible tests.
- `13.1_Diffusion_and_Image_Generation_Controls_solutions.ipynb` runs the
  reference implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The current graded path uses pinned Stable Diffusion 1.5, supplemental pinned
SD-Turbo, and pinned CLIP revisions on CUDA. The acceptance path uses safe
generated shape prompts, CLIP alignment scoring, captured SD1.5 DAAM-style
cross-attention maps, target-token ablation over random/control-token ablations,
simple image-quality metrics, and white-noise rejection.
