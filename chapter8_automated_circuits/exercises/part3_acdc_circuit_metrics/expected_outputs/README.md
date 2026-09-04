# [8.3] ACDC and Circuit Metrics Expected Outputs

These fixtures freeze the results learners should see after completing the section. They describe the section's actual teaching surface, not a generic report schema.

## Exact toy result

The toy graph has ten named candidate edges and a declared eight-edge ground-truth circuit. One-shot insertion assigns zero recovery to every edge because the primary mechanism contains a product gate and the backup mechanism needs two edges. Iterative greedy deletion at threshold `0.10` removes only the two matched decoys, retains all eight ground-truth edges, and reaches recovery `1.0`.

The threshold sweep has two interpretable cliffs: at `0.17` it removes the two-edge backup path and falls to recovery `5/6`; at `0.84` it removes the six-edge primary path. Pairwise completeness is required because single-edge addition cannot reveal the omitted two-edge backup interaction.

All `10 choose 8 = 45` same-size circuits are enumerated. The recovered circuit ranks first with exact empirical `p = 1/45`; the best wrong circuit reaches only `5/6`. Its recovery remains exactly `1.0` in all three held-out toy regimes.

## Pinned real-model result

The CPU solution run loads the pinned TransformerLens `gelu-1l` checkpoint and patches named slices of `blocks.0.attn.hook_result`. Greedy deletion at threshold `0.05` retains `L0H0`, `L0H4`, and `L0H6`, recovering `0.946` of the clean-corrupt logit-difference gap. This set ranks first among all 56 three-head sets and stays above `0.75` recovery on three held-out prompt pairs.

Patching the clean MLP output alone reaches `0.991` recovery. This is deliberately presented as a downstream bottleneck control: it does not show that the upstream heads are irrelevant, and the section does not claim an edge-level IOI or greater-than circuit.

## Tolerances

The analytic toy checks use `rtol=1e-9` and `atol=1e-9`. The pinned float32 real-model acceptance thresholds are declared in `artifacts.lock.yml`; exact observed CPU values are recorded in `reference_metrics.json` for review and regression diagnosis.
