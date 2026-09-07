# Section 16.10: protocol fixed before digit fitting

## Scientific question and estimand

For each fixed labeled target image, predict the correct-class logit margin
after fitting a regularized logistic classifier to a subset of a fixed training
pool. A separate sparse linear datamodel maps subset membership to that margin.
This follows the framework and fixed-cardinality sampling in Ilyas et al. (2022),
equations 5–8. It is a CPU-scale application to a deterministic learning
algorithm, not a replication of the paper's stochastic ResNets or their scale.

## Public data and selection

Use the exact UCI Optical Recognition of Handwritten Digits archive already
pinned in section 16.9. Archive SHA-256:
`0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0`.
The source data has separate train/test writer groups. Individual writer IDs
are unavailable. Attribution is to Alpaydin and Kaynak (1998), DOI
10.24432/C50P49, CC BY 4.0. Retain source row IDs and content digests.

Digits **3 and 8** provide a fresh binary image task relative to 16.9's 1/7.
Selection is independent of model outputs. Within each class, sort original
training rows by SHA-256 of `arena16.10/v1/train/{row}` and take the first 64.
Within each class, sort original test rows by SHA-256 of
`arena16.10/v1/query/{row}` and take the first 32. Sort each resulting split by
original row number. Thus there are 128 training images and 64 labeled queries.
Scale pixels by 1/16, append a constant bias coordinate. No fitted preprocessing,
label corruption, data augmentation or example filtering. Check exact duplicate
images between splits, retaining and reporting any rather than deleting them.

The queries are excluded from **classifier** training, but their labeled margins
are explicitly used as **datamodel** regression targets. They are not unseen
images for the datamodel. The primary held-out unit is a training subset/model.

## Learning algorithm and subset designs

Fit mean binary cross-entropy plus `0.02 * ||theta||² / 2`, including the bias,
from zero, with the independently converged Newton solver taught in 16.9.
Every model starts from zero; no warm starts or model-response caches. Require
gradient infinity norm at most 1e-10. Float64 CPU, four Torch threads, batches
of at most 128 models. Batch size affects scheduling, not the scientific fit.
Check six separately fit models and ten independent NumPy/SciPy optima.

For each fraction **0.25, 0.5, 0.75**, draw **1024 training**, **128 validation**
and **256 test** membership masks with exactly 32, 64 or 96 included images.
Use Torch CPU Generator seeds `161000 + 100*a + split`, with a=0,1,2 in the
stated fraction order and split=0,1,2. Sample by independent random permutations;
do not stratify by class. Reject duplicate masks across the three splits and
report the check. Hold out test responses until all paths and deletion masks
have been selected. The 4224 resulting models are actual independent refits.

## Datamodel estimation

For each query fit `b + mask @ w` by minimizing
`mean((prediction - margin)**2)/2 + lambda * abs(w).sum()`.
The intercept is unpenalized and fitted by centering on training rows only.
This loss has half the squared-error scale of equation 8, so its lambda is half
the paper's lambda for the same problem. Do not normalize feature variances.

Use a 16-point descending path `lambda_max * logspace(0, -4, 16)`, separately
for each query, where `lambda_max = max(abs(X_centered.T @ y_centered))/m`.
Fit using accelerated proximal gradient with warm starts along the regression
path, but no classifier warm starts. Require a KKT residual at most 1e-8;
use at most 20000 steps per path point and report any nonconvergence as failure.
Pick lambda separately per query by minimum validation MSE, first index on an
exact tie. Never use test responses to choose lambda, fractions or fit count.

Retain every path coefficient, intercept, KKT residual, iteration count and
validation loss. Compare intercept-only and centered minimum-norm least squares.
Fit a permuted-response control with seed 161440, permuting training subset
rows jointly across queries, and select its lambda on true validation responses.
For alpha=0.5 also fit training prefixes 64,128,256,512; use the same validation
split and retain all points plus the full 1024-model fit. Sample-complexity
curves have no outcome-selected stopping rule.

## Counterfactuals and galleries

Before any test-subset or counterfactual response is inspected, construct
deletions for the first twelve source-ordered queries of each true class (24
queries total). Use budgets **2,8,16,32**, and four rankings: largest datamodel
weight for each of the three fractions, and largest same-label signed pixel
cosine similarity. Ties use the source order. Remove high supportive weights
to attempt to reduce the query's margin. Do not claim every positive coefficient
is an individually causal or distribution-independent contribution.

For each ranking/query/budget, create **16 matched random deletions** with the
same number removed from each observed training class as that selected set.
Use seed `162000 + 10000*method + 100*query_position + budget`, sampling without
replacement independently for each random replicate. Keep all controls. This
is 24*4*4*17 = **6528 counterfactual fits**, plus one full-pool base fit.
Report true margin and its change from the full pool, predicted change, actual
classification flips, per-query errors and class counts. A tested deletion
causing misclassification is only an upper bound on the smallest necessary
deletion; absence of a flip is not robustness beyond tested budgets.

Main image gallery: first two source-ordered queries of each class, always all
four, with supportive/opposing datamodel images and pixel neighbors. Use no
best-case selection. The interactive cell may choose any of the 64 queries;
its later exploratory refits never modify the frozen reference experiment.

## Identifiability, validation and publication

Fixed-cardinality masks obey `mask @ 1 = alpha*n`. Adding a constant c to every
weight and subtracting `alpha*n*c` from the intercept preserves predictions on
that slice. L1 regularization chooses a representative; this does not establish
a unique causal offset. Demonstrate unchanged same-size predictions, changed
different-size predictions and invariant rankings/centered similarities.
Cross-fraction evaluation and deletions are distribution shifts, even when they
work. Keep the alpha choices and adverse cases visible.

Validate core functions with exact fixtures and sklearn's independently coded
coordinate-descent Lasso. On fixed-cardinality designs compare predictions,
objective and KKT residual rather than insisting on a unique coefficient vector.
Do not claim an estimated stochastic-training noise floor: this algorithm is
deterministic. Require fresh own-answer notebook execution, a complete independent
rerun from the pinned source archive, actual plots and rendered visual review
before publication or any TAUGHT status. Keep revisions and failures traceable.
