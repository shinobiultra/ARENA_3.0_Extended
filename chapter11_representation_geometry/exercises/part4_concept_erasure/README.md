# 11.4 Concept Erasure

The learner asks what is removed when a probe stops working. Eight implementation
groups derive paired moments, support-aware whitening, orthogonal removal, the
LEACE affine eraser, refitted logistic probes, magnitude-matched random controls
and full-vocabulary outcome metrics. Immediate checks include an independent
constrained quadratic oracle and a nonlinear equal-centroid counterexample.

The native stage edits frozen GPT-2 at two prescribed sites. Erasers use 512
fitting sentences; new readers use a separate 320-sentence split. All 106
conditions remain in the final comparison on 288 role sentences and 384 unrelated
object sentences. The 12 figures connect exact geometry, new-reader recovery,
actual token predictions and collateral damage. Two play cells change covariance
anisotropy and native intervention strength.

`lesson.md` is the canonical source. Each function is completed from its own
solution dropdown; `build_notebooks.py` validates the paired draft. Execute it
from the repository root with the pinned GPT-2 checkpoint cached locally:

```bash
uv run python chapter11_representation_geometry/exercises/part4_concept_erasure/execute_learner.py --output-directory ~/.cache/arena_extended/erasure/my-own-run
```

Use a new output directory. The complete full-vocabulary record needs about
16 GB of disk; independent reproduction writes another complete record. Model
execution uses CPU float32, while moments, affine fits and probes use float64.
The experiment keeps all native errors and every fixed random control.

`erasure_reproduction.py --learner-directory DIRECTORY --output-directory
NEW_DIRECTORY` independently reloads the pretrained model, refits every map and
reader, and compares all 71,232 full-vocabulary outputs and their winning tokens.
The publisher requires that evidence plus an actual rendered math/dropdown and
figure review. The normal section report uses explicit CPU execution:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/run_extension_verification_reports.py --section 11.4
```

The exact guarantee concerns a chosen distribution and convex-loss affine
predictors. It does not establish statistical independence, generalization,
all-layer concept scrubbing or a unique causal concept inside GPT-2.
