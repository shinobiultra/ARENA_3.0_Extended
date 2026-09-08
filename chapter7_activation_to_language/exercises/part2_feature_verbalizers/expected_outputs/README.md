# Expected outputs for [7.2]

The solution notebook should finish with `SIGNATURE PASSED: True` and display a
six-panel figure generated during the same CUDA run:

1. embedding-through-final-layer decoding accuracy;
2. candidate-explanation agreement on validation examples, with word counts;
3. all 4,096 matched-pair label-swap accuracies, including 548 perfect null fits;
4. scores of the displayed top, bottom, random, and near-threshold examples;
5. the forward target effect against 256 matched within-pair causal directions;
6. the reverse target effect against the same 256 directions with its own coefficients.

Validation evidence tables show top, bottom, random, and near-threshold
examples. `The mice` and `The geese` falsify the suffix explanation before
evaluation on the held-out vocabulary.

`reference_metrics.json` records the current pinned CUDA values. Small logit
differences may vary with PyTorch kernels, but the qualitative gates are:

- held-out activation accuracy at least `0.95`;
- held-out accuracy `1.0` but paired-null p95 `1.0` and exact `p >= 0.05`, explicitly marked non-significant;
- zero overlap between train and held-out final noun token IDs;
- explicit regular, irregular, and nonce results, including the nonce suffix failure-to-fail;
- both intervention directions change the mean margin by more than `8.0`;
- each target edit beats its own 64-control absolute p95 by more than `1.0`;
- every edited prompt switches to the intervention's target continuation; and
- every intervention forward pass calls the registered hook exactly once.
- both causal effects beat their within-pair null's p95, with plus-one p-values below 0.05.

## How this fixture was produced

The tables come from the pinned Pythia-70M checkpoint and exact 24/12/24
training/validation/held-out noun split in `solutions.py`. Predictions are
computed by the notebook's functions and real transformer hooks.

**Trusted implementation:** hand-computable mean differences and projection
replacement; exhaustive four-pair tests; a two-dimensional hooked model with
readout `x + 2*y`; and the pinned Transformers Pythia forward pass.

**Random seed:** display order 0, paired causal seed 1729, 256 unique swap
patterns; orthogonal controls use seeds 101–164 forward and 201–264 reverse.
The decoding null enumerates all 4096 patterns and is not sampled.

**Allowed tolerances:** the real zero-edit hook uses `rtol=1e-4`, `atol=5e-4`;
unit-vector and orthogonality checks use `rtol=1e-5`, `atol=1e-5`. Exact counts
and split membership have no tolerance. Small kernel-dependent logit changes
must still satisfy every frozen effect and control threshold in the lock.

**When to regenerate:** rebuild and execute both learner and solution paths
after a source or dependency change; inspect the actual examples and figure,
then regenerate expected values and the report from the same pinned run.

The historical planted and answer-letter results are invalid and must not be
used as expected outputs.
