# Full-edge candidate evidence

These files preserve completed experiments while the 8.3 learner is being
finished. They do not upgrade the section's published release report or
roadmap status. The draft learner implements and runs its own methods; these
archived values are not its main teaching output.

Each `.json.gz` is a lossless copy of a completed raw result. The manifest
records both compressed and original JSON SHA256 values. Original source
hashes inside results are retained as recorded, not rebound to later edits.

| Candidate | Discovery recovery | Development recovery | Decision |
|---|---:|---:|---|
| IOI ACDC, threshold 0.05, 32 edges | −0.2769 | −0.2347 | Failed |
| IOI ACDC, threshold 0.004, 383 edges | 0.9494 | 0.5226 | Overfits discovery; failed selection |
| IOI ACDC, threshold 0.001, 737 edges | 1.023 | 0.8815 | Failed accuracy criterion: 27/32 versus required 28/32 |
| Greater-than EAP-IG, 100 globally ranked edges | Not used for selection | 0.8788 | Promising development result |
| Greater-than EAP-IG, 300 globally ranked edges | Not used for selection | 0.9670 | Promising development result |

IOI ACDC uses all-name corruption and mean full-vocabulary clean-to-circuit KL.
The threshold 0.004 circuit has development accuracy 0.75, versus clean 0.96875,
and KL 0.7706. It fails the predefined selection criteria even though it beats
both random nulls. All 383 retained-edge deletions and the sampled retained-
subset completeness probes remain in its development file.

The attribution comparisons retain every tested size and uniform random
sample, including the failed IOI curves. `three-method` adds KL-loss EAP-IG;
`second-subject` changes only the repeated subject as a separate corruption
control. These variants are not pooled into a single favorable score.

Author-code parity files identify the immutable upstream commit and every
downloaded source hash. The only adapter supplies already-tokenized examples,
right-padding masks and actual lengths. The attribution, graph and hook
implementations are not edited. The full GPT-2 three-method parity record also covers KL-loss EAP-IG and
output-frontier selection at five budgets.

The four greater-than choices were frozen before any held-out scores in
`greater_than_frozen_selection.json` and its individual selection records.
The 64-example seed-13 holdout changes the wording; no choice was retuned.

| Frozen method | Edges | Development recovery | Held-out recovery |
|---|---:|---:|---:|
| ACDC, threshold 0.004 | 78 | 0.8550 | 0.8318 |
| EAP, task loss | 300 | 0.9272 | 0.8952 |
| EAP-IG, task loss | 100 | 0.8788 | 0.8586 |
| EAP-IG, KL loss | 300 | 0.9226 | 0.9090 |

ACDC falls below the development selection floor on transfer. Every held-out
file retains 32 uniform and 32 receiver-indegree-matched random circuits,
every retained-edge deletion, and 72 sampled completeness probes. A positive
probability margin is not top-token accuracy. These are small controlled
studies, not exact replications of the papers' dataset sizes.

The manifest archives twenty-two completed raw JSON results. The IOI threshold
0.0005 search is running under the documented development-only amendment;
partial checkpoints are not completed evidence. IOI held-out prompts remain
unscored. `learner_validation` separately preserves three executed learner
subsets, their exact source snapshots and declared partial scopes.
