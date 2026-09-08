# Frozen concept-erasure native evaluation

The generated `erasure_data.py` splits predate the first GPT-2 run. Role tasks
contain 512 eraser-fitting, 320 independent probe-fitting, 160 development and
288 final prompts. The unrelated object task has 320 development and 384 final
prompts. All final frames and rows remain closed during development.

The retained development run is
`~/.cache/arena_extended/erasure/native-development-v1`: PASS, 27.0585 seconds,
all six source inputs unchanged. It evaluates ten site/method combinations,
with explicit native full-hook parity and unchanged pretrained weights.
Block 6 input/repeated subject and block 9 input/final token were specified
before that run. Both remain in the final experiment. The empirical eraser,
fixed .01 spherical covariance mixture, orthogonal cross-covariance removal,
projection of the original probe coefficient and no-op are all retained.
No setting is chosen because it produces the strongest final effect.

The fixed native checkpoint is GPT-2 revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32, eager attention,
eval mode with all model weights frozen. Moments, erasers and probe fitting use
float64; edited activations are explicitly cast to the model dtype. Numerical
rank uses width times float64 epsilon; the cross-covariance SVD cutoff is 1e-10.
The .01 covariance mixture changes the fitting objective and is not claimed to
be the exact unregularized optimum. Both cases remain visible.

Probe evaluation uses an independently fitted reader: after fitting an eraser
on the 512 training prompts, apply it to the separate 320 probe-training prompts
and fit a new logistic reader there. It is evaluated on new development/final
sentences. Probing the eraser-fitting examples separately checks the empirical
constraint and cannot stand in for this new-data test. Logistic fitting uses
training-only standardization, an intercept, L2=.001 and at most 150 LBFGS
iterations. Report both log loss and thresholded accuracy; loss worse than the
constant predictor is a failed reader, not stronger evidence of erasure.

Each site has 16 seeded random rank-one orthogonal removals, plus the same 16
directions with per-example displacement magnitudes matched separately to the
empirical and .01-mixture erasers. Seed ranges are 1000–1015 and 2000–2015.
The matched controls use no final labels and are nonlinear transforms, not
claimed affine erasers. Retain all draws. Rank and actual displacement are both
measured, and every condition gets its own independently refitted reader.

There are 53 conditions per site, 106 total. Each condition is evaluated on all
288 role and 384 object final prompts: 71,232 next-token outcomes in total.
Record the complete vocabulary logits for each condition, separately on disk,
plus actual intervention vectors, probe parameters/probabilities and fitted maps.
Keep all native errors. Accuracy, target log loss, native-distribution KL,
representation distortion and unrelated-object damage are distinct outcomes.
The first role example in each template is selected for display by row order;
its success or failure cannot alter inclusion.

The headline claim is deliberately tied to the learner's question: an exact
fitting-distribution constraint need not remove information on new inputs, and
an intervention can damage behavior without isolating a unique causal concept.
This finite, single-position experiment does not reproduce all-layer concept
scrubbing, remove all nonlinear information or establish a unique mechanism.
Any later investigation must label this final set as already inspected and
specify fresh data before making a new untouched-evaluation claim.
