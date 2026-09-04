# [17.1] Checkpoint Archaeology and Mechanism Emergence Verification Assets

ARENA-style learner path and verification assets for the checkpoint archaeology
and mechanism-emergence section.

- `17.1_Checkpoint_Archaeology_and_Mechanism_Emergence_exercises.ipynb` is the
  learner notebook with stubs and visible ARENA-style tests.
- `17.1_Checkpoint_Archaeology_and_Mechanism_Emergence_solutions.ipynb` runs the
  reference implementation and committed-report checks.
- `artifacts.lock.yml` pins the toy trajectory contract plus the real modular-addition checkpoint preflight.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the historical `run_smoke_test` contract hook.
- `expected_outputs/reference_metrics.json` records baseline/control slots.
- `../../instructions/assets/checkpoint_archaeology_validation_loop.svg` shows
  the train/save/reload/control validation loop.
- `../../instructions/assets/checkpoint_archaeology_signature_result.svg`
  summarizes the scoped finite-table signature result.

The learner-facing live path is `live_checkpoint_archaeology_smoke_test`: it
trains a tiny mod-13 addition MLP on CPU, writes and reloads real checkpoints,
analyzes the reloaded trajectory, and rejects a random-label checkpoint control.

The real graded release path is `run_gpu_test`: it runs the same finite-table
checkpoint archaeology preflight on CUDA. There is no CPU fallback for GPU
acceptance. In both paths, "generalization" means exhaustive evaluation over
the complete finite mod-13 table (`169/169` input pairs), not OOD extrapolation.
