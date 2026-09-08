# Expected outputs

The deterministic signature run should show five facts together:

1. Normalized latent prediction loss falls without reconstructing pixels.
2. Correct visible context recovers the fully masked object's class on disjoint scenes.
3. Swapping only the visible key changes the prediction to the alternate key's class, changes no other pixel, and raises loss.
4. Zero context and the same untrained architecture fail; the latter stays at chance.
5. Random encoder features already support strong probes, so probe accuracy is a negative result rather than the headline evidence.

Pinned CUDA 13.2 seed-0 values are approximately:

- correct prediction accuracy `1.000`; original class after key swap `0.000`; alternate key class `1.000`; untrained near `0.333`;
- correct latent loss `0.00648`; mismatch `0.03771`; zero context `0.05684`; untrained `1.93`;
- EMA feature spread `0.02821`; matched joint-target/no-EMA spread `0.00246`;
- non-key pixel difference `0`; exact train/evaluation overlap `0`; peak VRAM well below `1 GiB`.

The exact values in `smoke_test.json` are a pinned CPU reference, not universal
constants. Acceptance uses the conservative thresholds in `artifacts.lock.yml`.

Inspect `instructions/assets/ijepa_signature_result.png`. A valid figure must show
the target mask and key, loss traces, prediction controls, latent-loss controls,
feature spread, and the random-probe negative result. A decorative embedding plot
or a probe-only result is not accepted.

## How this fixture was produced

`solutions.run_smoke_test(cpu=True)` trains the actual tiny transformers on CPU
for seeds 0, 1, and 2 and writes `smoke_test.json`. The separate strict CUDA run
uses the same 30 epochs, 48-example batches, learning rate 0.002, EMA 0.995, and
120 training plus 120 evaluation scenes per seed. The notebook computes its own
result; it does not read either file. The reference metric table records the
measured CPU run, not constructed target values; the notebook and release report
record the independent CUDA run.

## Trusted implementation

Hand calculations anchor patch ordering, context scatter, EMA updates, and the
exact key-reading oracle (correct/mismatch/zero latent losses 0/2/1; erased-key
loss 4/3 and accuracy 1/3). The learner supplies the full model and training loop.
The [official implementation](https://github.com/facebookresearch/ijepa/blob/main/src/train.py)
is the architectural comparator, not a numerical checkpoint oracle: this toy
uses a different loss, position embedding, mask policy, and training world.

## Random seed

Contract seeds are fixed at `[0, 1, 2]`, evaluation seeds at `[50000, 50001, 50002]`.
Pixel overlap is zero, but all twelve semantic templates occur in both splits.
The optional play cell defaults to seed 17 and the other nontrivial key shift.

## Allowed tolerances

Integer selections, intervention locality, and copied EMA parameters use
`rtol=0, atol=0`. The hand-computed normalized losses allow `atol=1e-7` for
float32 rounding. Learned metrics use the explicit bounds in the lock, not
bitwise equality across hardware. Every declared seed must recover the correct
and alternate class at least 90%, reduce original-class accuracy after a swap
to at most 10%, and reduce erased-key accuracy to at most 50%.

The existing seed-0 effective-rank floor of 1.5 is retained. Across seeds,
measured ranks are about 2.658, 1.429, and 1.850: rank is reported separately
from causal recovery and feature spread. Seed 1 does not satisfy a hypothetical
all-seed rank floor of 1.5. Successful three-class decoding does not require a
high-dimensional representation; this lab makes no such claim. The compound
ablation even has higher effective rank than the main model while nearly
collapsing feature scale, so rank alone is an inadequate collapse diagnostic.

## When to regenerate

After changing generation, masking, model, objective, optimizer, seed list,
control, evaluator, or thresholds, rerun semantic tests and the completed
learner notebook, inspect its figure, regenerate CPU/GPU reference values,
then run the hash-bound report generator last. Never raise tolerances simply
to make a changed experiment pass.
