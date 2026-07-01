# Falsification Tests

Write these before running the main experiment.

| Test | Claim It Attacks | Pass Condition | Failure Means |
| --- | --- | --- | --- |
| Held-out split | Generalization | Metric beats baseline | Claim is overfit |
| Random control | Causality | Random effect is small | Effect is nonspecific |
| OOD templates | Robustness | Direction of effect survives | Claim is template-bound |
| Causal patch / ablation | Mechanism | Intervention changes target | Evidence is correlational |

A project is not ready for a paper-style writeup until at least one real
falsification test could have failed.
