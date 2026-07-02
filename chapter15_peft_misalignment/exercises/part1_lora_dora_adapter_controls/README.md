# [15.1] LoRA, DoRA, and Adapter Controls Verification Assets

ARENA-style learner path and verification assets for the roadmap contract.

- `artifacts.lock.yml` pins the current smoke-test artifact contract.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.
- `15.1_LoRA_DoRA_and_Adapter_Controls_exercises.ipynb` is the learner-facing
  notebook with local stubs and visible tests.
- `15.1_LoRA_DoRA_and_Adapter_Controls_solutions.ipynb` executes the reference
  solution, visible tests, and committed verification-report assertions.
- `../../instructions/assets/lora_dora_adapter_controls_validation_loop.svg`
  shows the exact tensor-to-CUDA validation loop.
- `../../instructions/assets/lora_dora_adapter_controls_signature_result.svg`
  summarizes the visible result and scoped CUDA controls.

The current GPU verification trains generated rank-1 LoRA, rank-1 DoRA, and
full-finetune safe proxy classifiers on a planted target-direction
classification task. The report records adapter accuracy, frozen-baseline
accuracy, merge/unmerge parity, learned rank, target-direction alignment,
random-label failure, same-norm random-adapter failure, DoRA norm preservation
on the learned delta, matched LoRA-vs-DoRA-vs-full-finetune accuracy and
directional controls, and measured VRAM.

This evidence is scoped: it is a generated safe proxy adapter, not a public
unsafe adapter, not a refusal-suppression adapter, and not a claim about
public adapter side effects.
