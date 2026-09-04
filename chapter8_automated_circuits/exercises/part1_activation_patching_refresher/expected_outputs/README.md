# Expected Outputs

The signature result is the notebook-generated activation-patching panel, not
the verification report.

For the exact two-hop copy model:

```text
denoising recovery = noising damage =
[[0, 1, 0, 0, 0],
 [0, 0, 0, 1, 0],
 [0, 0, 0, 0, 1]]
```

The rows are `embed`, `route`, and `readout`; columns are `<BOS>`, `source`,
`distractor`, `query`, and `answer`. The wrong-position donor control must be
all zeros.

The real-model CUDA path is a secondary `gelu-1l` residual-position mechanics
preflight. Its report must be regenerated in the parent serialized GPU pass.
