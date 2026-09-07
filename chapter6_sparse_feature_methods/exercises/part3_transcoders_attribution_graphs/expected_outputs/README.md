# Expected outputs for [6.3]

`reference_metrics.json` records the exact CPU model-organism result and the
separately pinned GELU-1L preflight. `smoke_test.json` records the CPU notebook
contract.

The exact CPU result is the learner-facing evidence: a target score of `1.6`, a
ten-edge graph over features `[4, 0, 2]`, retention faithfulness
`[0.78125, 0.90625, 1.0]`, and matched controls that remain substantially below
the recovered graph. The committed GPU metrics are provenance-bound supporting
evidence and are not used to generate the signature figure.

Float32 tensor checks use PyTorch's default close tolerances unless a semantic
test states an exact value. Integer ids, edge counts, and control identities are
checked exactly.

## How this fixture was produced

`solutions.toy_experiment()` evaluates the explicit colored-shape ReLU MLP;
`run_smoke_test(cpu=True)` serializes its activations, signed graph, intervention
curves, and controls. The real GELU-1L preflight is a separately executed result.

## Trusted implementation

The trusted oracle is the declared encoder/decoder/readout matrices, not the
transcoder's own output. Tests hand-check preactivations `[0.5, -0.5, 0.5, -0.5,
0.5, -1.5]`, active features `[0.5, 0, 0.5, 0, 0.5, 0]`, reconstruction
`[1.35, -0.25, 0.10]`, and exact edge decomposition before the final graph.

## Random seed

The toy organism and matched graph controls are explicitly specified tensors;
there is no fitted random toy model. The separate real-model preflight sets
Torch seed 6303. Preserve that distinction when comparing references.

## Allowed tolerances

The float32 `torch.testing.assert_close` checks use `rtol=1.3e-6, atol=1e-5`
under the pinned runtime. Integer graph identities are exact. Follow explicit
per-test overrides and the real-model bounds in `artifacts.lock.yml`.

## When to regenerate

Regenerate after changing the organism, readout, edge definition, intervention,
control, or pinned real model. Execute the solution notebook, verify the signed
graph and control curves visually, run tests, then regenerate the supporting
report. A report cannot substitute for the learner's live graph.
