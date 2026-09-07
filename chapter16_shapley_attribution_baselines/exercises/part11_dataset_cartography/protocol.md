# 16.11 Dataset Cartography — frozen experiment v1

This protocol is fixed before any native digit classifier is trained for 16.11.
Primary method: Swayamdipta et al., EMNLP 2020, Sections 2–5 and Appendix A.1.
The adjacent plan records the inspected primary PDF and author implementation.

## Data and questions

Use the byte-pinned UCI Optdigits archive taught in 16.9. Use all ten classes.
Select 300 examples per class from the original training file by increasing
SHA256 of `arena16.11/v1/train/{zero_based_source_row}`; sort selected rows by
source row. The 3000 selected examples are training data. The remaining 823
training-file examples are a same-source holdout. All 1797 original test-file
examples form the separate-writer holdout. No individual writer IDs exist in
these files. No fitted normalization, augmentation, filtering or hidden split
tuning. Inputs are 64 original pixel counts divided by 16, float32 CPU.

For the paired noise experiment, choose 30 rows per original class by the same
hash scheme with namespace `noise`, change each label to `(label + 1) % 10`, and
retain every original label and changed-row identity for diagnosis only. Counts
remain balanced. A separate random-label control permutes labels with CPU Torch
Generator seed 161199; coincidentally unchanged labels remain unchanged.

## Model and logging

Primary architecture: 64 input → 64 ReLU hidden → 10 output. Comparison:
64 input → 10 linear output. Normal weights: hidden std sqrt(2/64), output
std 1/sqrt(hidden); linear std 1/sqrt(64). All biases zero. Each run uses its own
Torch CPU Generator initialized by its run seed, with weight draws then one
fresh randperm per epoch. All runs cold-start. SGD learning rate 0.05, momentum
0.9 with first buffer equal to gradient, weight decay 0.0001 on all parameters,
minibatch size 128, no scheduler, dropout or augmentation. Thirty full epochs,
including the smaller final batch. No early stopping or hyperparameter selection.
All training is CPU float32, four threads, deterministic algorithms enabled.

Five mapping seeds: 161100 through 161104. Train each on clean, noisy and shuffled
labels (15 MLPs), and five clean linear models (20 mapping runs total). Each run
retains logits by stable source ID at every epoch in two conventions: predictions
before each visited minibatch update, and a full deterministic epoch-end pass.
Primary maps use epoch-end logits. The paper's formula specifies epoch-end model
parameters; the released training loop collects pre-update minibatch predictions.
Record both instead of silently equating them. Save final parameters, epoch losses,
step counts, both logit traces and all derived metrics.

Confidence is mean assigned-label probability, variability is population standard
deviation (divide by E), correctness is the fraction of argmax-correct epochs.
Argmax ties choose the first class index. Maps use all 30 epochs; prescribed
window comparisons use epochs 1–10 and 21–30. A constant predictor and a repeated
final snapshot demonstrate zero temporal variability. Random-label maps are a
control for semantic interpretation, not a guarantee that a network cannot fit
random labels. Compare every seed and the linear model without picking a winner.

## Selection and actual retraining

Use only the clean MLP map from seed 161100, all 30 epoch-end snapshots. For each
retained size 300, 1000 and 2000, select globally by decreasing confidence (easy),
increasing confidence (hard), or decreasing variability (ambiguous). Ties break
by training source order. Regions are rankings, not disjoint universal classes.
Do not use clean/noisy identities, held-out images or held-out outcomes to select.

Five independent retraining seeds: 161200 through 161204. For each region/size,
train five new MLPs on that selected set. For each corresponding seed, independently
sample a random set with exactly the selected set's original observed-label
counts, using seed `161300 + 1000 * region_index + 10 * size_index + replicate`.
Region order: easy, hard, ambiguous. Size order: 300, 1000, 2000. Train these five
matched random sets with the same five retraining seeds. Also train five uniform
random subsets per size, sample seed `165000 + 10 * size_index + replicate`.
Train five full-pool MLP baselines with the same retraining seeds. These are 110
fresh main refits. Retain all final weights, epoch losses and actual step counts.

At size 1000 only, add a training-budget control for ambiguous and its five
matched random sets: 90 epochs each (720 updates, equal to 30 full-pool epochs),
with the same initialization and first-30-epoch path as the corresponding
30-epoch fit. Ten additional refits. Main equal-epoch comparisons do not control
optimizer-update count. Count mapping cost separately; data selection is not free.

All maps, selection masks, random controls and 120 retrained parameter sets must
be saved before any held-out model predictions are computed. Then evaluate every
model on all 823 same-source and all 1797 separate-writer examples, without choosing
models, epochs or examples from these results. Report full-seed values, mean/range,
cross entropy and accuracy; classwise accuracy and paired correctness changes.
No natural OOD benchmark or RoBERTa replication claim is permitted.

## Diagnosis, independent checks and play

For each noisy MLP map, rank ascending assigned-label confidence and report
precision/recall for the 300 known injected errors over budgets 30/100/300/600/1000,
with random rankings and ascending clean-run confidence as controls. These
synthetic error labels are evaluation truth, not evidence of natural mislabeling.
Display the first four ranked images per easy/hard/ambiguous region and first four
noisy hard examples; include source IDs, labels and full probability trajectories.
Compare map rank overlap and correlations across all five seeds, both logging
conventions, early/full/late windows and linear/MLP models.

Independent checks: NumPy softmax/metric oracle; ordinary torch.nn plus
torch.optim.SGD reference loop on small deterministic fixtures; separately scheduled
full clean mapping seed 161100 (MLP and linear) against batched results, including
all logits, losses and final weights, rtol/atol 2e-5. No trained-cache inputs for
native reproduction. Own-answer versus fresh native full payloads compared at
rtol 1e-6, atol 1e-7, with exact identifiers, shapes, dtypes and metadata.

Two editable plays: probability-trajectory algebra, and actual new region versus
matched-random training. Default real play: ambiguous, 1000 retained, map seed
161100, retraining seed 166000, random selection seed 166001, 30 epochs, two new
MLPs, evaluated on the same frozen held-out sets. This is exploratory reuse of
evaluation data and is not included in confirmatory ranking/selection.

Total frozen main training runs: 20 mapping + 120 retraining = 140; plus two
separate mapping checks and two default play fits. No run selected by its outcome.
