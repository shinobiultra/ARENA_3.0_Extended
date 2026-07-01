# Mini Activation-Oracle Capstone Report

## Claim

A question-conditioned MLP activation oracle can recover a controlled latent state from synthetic residual-stream activations, including a nonlinear XOR question that a bank of linear probes does not solve. The claim is scoped to this generated model-organism benchmark.

## Setup

- Benchmark: `synthetic_activation_oracle_latent_questions_v1`
- Dataset: `balanced_latent_bits_with_heldout_template_nuisance_v1`
- Device: `cuda`
- Seeds: [0, 1, 2]
- Train examples per seed: 576
- IID test examples per seed: 576
- Held-out template examples per seed: 384
- Questions: color_bit, shape_bit, material_bit, color_xor_shape

## Results

| Metric | Mean |
| --- | ---: |
| Oracle accuracy | 1.000 |
| Text-only accuracy | 0.500 |
| Linear-probe-bank accuracy | 0.878 |
| Oracle XOR-question accuracy | 1.000 |
| Linear-probe XOR-question accuracy | 0.514 |
| Held-out-template accuracy | 1.000 |
| Relevant-dimension ablation drop | 0.480 |
| Counterfactual patch answer-change rate | 1.000 |
| Counterfactual patch target accuracy | 1.000 |
| Random-dimension patch change rate | 0.000 |
| Random-activation accuracy | 0.495 |
| Random-activation confidence | 0.918 |
| Label-shuffle oracle accuracy | 0.481 |

## Causal Validation

Ablating the latent dimensions used by the asked question drops oracle accuracy, while patching those dimensions from a donor example usually changes the answer to the donor answer. Patching randomly sampled non-latent control dimensions has a much smaller effect.

## Per-Question Means

| Question | Oracle accuracy | Linear probe accuracy |
| --- | ---: | ---: |
| color_bit | 1.000 | 1.000 |
| shape_bit | 1.000 | 1.000 |
| material_bit | 1.000 | 1.000 |
| color_xor_shape | 1.000 | 0.514 |

## Limitations

This is a generated model-organism sprint, not evidence about a released transformer. Random activations are scored by accuracy, not abstention, because the binary oracle can be confidently wrong off distribution. The high random-activation confidence is recorded as a calibration limitation for the next iteration. Random-patch controls sample non-latent dimensions, so they show that unrelated control coordinates do less than targeted latent patches.

## Failure Cases

No held-out-template failures were observed in the committed run.
