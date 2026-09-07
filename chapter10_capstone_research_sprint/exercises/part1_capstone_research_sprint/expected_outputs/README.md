# [10.1] Expected Outputs

These fixtures describe the CPU-executable exact XOR-direction study used in
the learner notebook. They are review aids, not the source of the notebook's
scientific result.

The smoke-test fixture records the preregistered claim and decision thresholds.
The reference-metrics fixture records the deterministic CPU reference values.
The solution notebook recomputes every metric from model-organism activations
before building its figures.

The fixed seeds are 7 for the orthogonal activation rotation, 11 for 128
shuffled-label fits, 13 for 256 isotropic random patch directions, and 123 for
the activation-noise stress test.

Exact structural identities use float64 and atol 1e-10. Behavioral gates use
the explicit preregistered thresholds because a later CUDA repetition may differ
at negligible floating-point boundaries.

## How this fixture was produced

`solutions.run_study(device="cpu")` enumerates the rotated XOR organism and
runs the preregistered held-out recovery, causal intervention, shuffled-label,
random-direction, bootstrap, and noise comparisons. CUDA repeats the same study.

## Trusted implementation

The signed XOR mediator is latent coordinate two before the known orthogonal
rotation. Its exact direction is column two of that rotation. The tests verify
orthogonality, all four bit combinations per template, balanced labels, and
projection onto the known mediator before fitting the ridge estimate.

## Random seed

Use rotation seed 7, shuffled-fit seed 11, isotropic-control seed 13, and
noise seed 123 as listed above. The bootstrap reproducibility unit oracle uses
seed 3 and 2,000 resamples. Seeds are part of the experiment, not tunable knobs
for meeting the preregistered threshold.

## Allowed tolerances

The float64 structural `torch.allclose` checks use `atol=1e-10` and its default
`rtol=1e-5`. Exact template and label identities require equality. Recovery,
causal, and stress-test gates use their explicit scientific thresholds.

## When to regenerate

Regenerate after changing the organism, train/evaluation templates, estimator,
intervention, preregistration, seed, or control. Recompute the complete study
and inspect its figures; retain failed preregistered claims as failures instead
of changing thresholds after seeing the outcome.
