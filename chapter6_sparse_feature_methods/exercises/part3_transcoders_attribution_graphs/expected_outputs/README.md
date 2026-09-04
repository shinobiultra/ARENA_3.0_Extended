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
