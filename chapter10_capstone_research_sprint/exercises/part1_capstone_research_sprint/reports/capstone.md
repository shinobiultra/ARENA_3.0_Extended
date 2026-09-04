# Exact XOR-Direction Capstone Result

## Preregistered claim

A ridge direction fitted on balanced train templates will recover the exact distributed XOR mediator, generalize to held-out templates, and causally transfer counterfactual donor answers beyond matched controls.

## Result

| Metric | Value |
| --- | ---: |
| Held-out activation accuracy | 1.000 |
| Raw-bit linear baseline | 0.500 |
| Template-only baseline | 0.500 |
| Shuffled-label mean | 0.525 |
| Exact-direction cosine | 1.000 |
| Paired accuracy delta | +0.500 |
| Paired bootstrap 95% interval | [+0.312, +0.688] |
| Learned patch donor-target accuracy | 1.000 |
| Random-direction patch mean | 0.057 |
| Accuracy after direction ablation | 0.500 |

## Controls and failure analysis

Raw-bit and template-only probes remain at chance, and shuffled-label probes average near chance. The learned intervention is compared with 256 isotropic directions of the same dimensionality.

The strongest random direction is an instructive anomaly: it reaches 1.000 donor-target accuracy because its absolute cosine with the exact direction is 0.810.

The activation-noise stress curve is:

| Sigma | Accuracy |
| ---: | ---: |
| 0.00 | 1.000 |
| 0.25 | 1.000 |
| 0.50 | 0.979 |
| 1.00 | 0.841 |
| 2.00 | 0.686 |
| 3.00 | 0.622 |

## Limitations

The organism computes parity exactly and applies a fixed orthogonal mix. This supports a method-validation claim, not a discovery about a released transformer. Held-out templates vary nuisance features but preserve the task rule. Real-model representations may be nonlinear, contextual, and unstable across inputs or layers.
