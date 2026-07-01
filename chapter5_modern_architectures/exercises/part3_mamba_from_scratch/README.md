# [5.3] Mamba from Scratch Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current toy scan/cache and official Mamba artifact contract.
- `5.3_Mamba_from_Scratch_exercises.ipynb` and
  `5.3_Mamba_from_Scratch_solutions.ipynb` provide the first ARENA-style
  learner notebook pair for the selective-scan implementation ladder.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded GPU path loads pinned `state-spaces/mamba-130m-hf`, checks logits and
deterministic generation on CUDA fast kernels, and records measured VRAM. Update
`artifacts.lock.yml` whenever the checkpoint revision, prompt set, dtype, or
expected metric thresholds change.
