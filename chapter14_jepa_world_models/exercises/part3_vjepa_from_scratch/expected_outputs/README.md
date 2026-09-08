# [14.3] Expected Outputs

## Reference protocol

`regenerate_reference.py` runs seeds 0, 1, and 2 for 1,000 steps each on CPU. Every seed creates 512 unique pretraining clips and a 128-item test/retrieval gallery. Forward test keys and physically canonical reversed test keys are excluded from pretraining; reversed keys are also disjoint from the explicit 128-key probe-fit subset and the forward test gallery.

Pretraining uses only target-normalized, stop-gradient L1 prediction, EMA, and a contiguous 2x2 spatial mask repeated through all three temporal tubelet slices. No variance or covariance penalty is active. The full-future mask is a separate out-of-distribution diagnostic.

All future predictors are evaluated in the same final-tubelet EMA-teacher latent space:

- the trained V-JEPA predictor;
- blank-future with frames 4–5 erased, encoded by the same teacher;
- copy-last-frame, encoded by the same teacher;
- exact dynamics, encoded by the same teacher;
- an independently initialized predictor receiving the trained visible context.

Each receives matched teacher-latent cosine and tie-aware top-1 retrieval. Exact top-score ties receive fractional credit, and both tied-query counts and maximum tie size are reported. Probe results are separate for color, direction, and combined state; reversed clips are scored against original and physical labels only after collision audit.

## Checked three-seed result

- Final/initial L1 ratio: `0.388 [0.343, 0.456]`.
- Future-mask V-JEPA cosine: `0.889 [0.838, 0.946]`; retrieval: `0.531 [0.414, 0.672]`.
- Blank-future cosine: `0.913 [0.880, 0.959]`; retrieval: `0.721 [0.602, 0.945]`.
- Copy-last-frame cosine: `0.922 [0.898, 0.964]`; retrieval: `0.703 [0.609, 0.852]`.
- Exact dynamics: `1.000` cosine and retrieval for every seed.
- Random predictor cosine: `-0.022 [-0.148, 0.075]`; retrieval: `0.005 [0.000, 0.008]`.
- Color/direction/state probes: `1.000`, `0.555`, and `0.549` means.
- Random-encoder direction/state probes: `0.458` and `0.438` means.
- Reversed original/physical state: `0.120` and `0.560` means.
- Effective rank: `9.27 [7.61, 10.97]` on the attainable maximum of 72; the readout still has 96 columns, but 24 are exact differences of existing columns.
- Every predictor has zero tied queries and maximum tie size 1 in all checked seeds.
- Train/test, reversed/pretraining, reversed/probe-fit, and reversed/test overlap are all zero in every seed.

These metrics support optimization, temporal label semantics, and non-collapse under the stated diagnostics. They do not support a claim that the future-mask predictor beats persistence. Differences across three seeds are descriptive only.

The blank-future control retains high clip retrieval without a predicted future
image. Teacher attention mixes information across the full clip before target
selection, so future-token retrieval can reflect past context. A positive dynamics
claim requires a future-specific evaluation that controls for that shared context.

## Freshness

`reference_metrics.json` records hashes for `solutions.py`, `regenerate_reference.py`, and both PNG assets. The signature PNG embeds the evaluation protocol and a fingerprint of seed-0 metrics plus all multi-seed intervals. `test_reference_assets_and_metrics_are_fresh` rejects stale code, metrics, or assets.

Regenerate after changing the world, split, architecture, objective, seeds, evaluator, metrics, or plotting code. Do not loosen a threshold merely to accept a changed result.
