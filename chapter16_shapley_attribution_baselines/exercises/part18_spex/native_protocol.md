# Prospective SPEX / ProxySPEX GPT-2 comparison

**Draft, before native outcomes.** This document fixes the proposed scientific
question and budget. It is not an execution approval or a completed experiment.
A separate source-bound freeze and independent protocol review are required
before querying the language model. Tokenizer-only construction has checked
all 32 prompts; native model queries and native proxy fits remain zero.

## Question and scope

For a fixed next-name logit difference, how accurately do SPEX and ProxySPEX
predict responses to held-out word masks at the same requested-query budget?
Which recovered signed word interactions explain that prediction, and where
do sparse recovery and proxy fitting fail?

The claim concerns a defined 28-feature, fixed-position GPT-2 masking game.
It is neither a natural-language causal interpretation nor evidence for native
thousand-token scalability. High-dimensional scaling belongs to separately
labelled analytic games with known spectra. Every fixed prompt contributes to
the results, irrespective of the model's unmasked answer.

## Model and input inventory

Use `openai-community/gpt2`, revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, the 12-layer, 768-dimensional
checkpoint already cached locally. Freeze physical hashes of the weights,
config and tokenizer inputs, not only the revision string. Use the project's
locked environment, CPU float32, deterministic algorithms and evaluation
mode, eager causal attention and the explicit fixed absolute position IDs.
Only the final hidden vector and the two candidate unembedding rows are needed
for the scalar target. A full LM-head check below validates this reduced head.
No model training, fine-tuning or checkpoint selection occurs.

`native_data.py` defines four 28-word templates, four name pairs and both
name orders: 32 primary prompts, each with one subject-only counterfactual.
All prompts end with the fixed suffix ` to`. Tokenizer preflight found 30 BPE
tokens in each prompt; every candidate space-prefixed name is one BPE token.
The freeze will bind every prompt, token ID, offset and word-group assignment.

The value is the logit for the first name minus the logit for the second name
at the final suffix position. A positive unmasked difference is the intended
IOI answer; changing only the final subject to the first name reverses that
intended sign. The outcome is a continuous logit difference, not a correctness
indicator or sampled generation. Name correctness and approximation fidelity
are reported separately.

Each displayed word position is one feature, including repeated names as
distinct positions. Retained words keep all their BPE tokens. Absent words
have all their token IDs replaced by EOS and all their attention-key entries
set to zero. The suffix remains present; absolute positions remain unchanged.
The fully absent coalition therefore still has a valid final token. The word
masking function is visible to the learner. Empty/full, repeated-mask and
replacement-token invariance checks will test the defined intervention.
The exact probe table uses prompt zero and three masks: empty, full and
alternating with feature zero present. Evaluate a three-row reduced-head batch,
its repeated batch, three singleton repeats, a three-row batch using replacement
token zero, and a three-row full-LM-head batch. These are **15 extra physical
row evaluations in seven batches**, kept outside the training cache. Compare
both candidate logits with `atol=1e-5, rtol=1e-6`. These tolerances are fixed
before any model outcomes; a failure is retained and stops the run.

## Fourier convention and methods

Use `chi_S(x) = (-1)**sum(x[i] for i in S)` with `x_i=1` retaining word `i`.
Feature zero is the first displayed word and the most significant bit of an
integer query ID. Supports are sorted tuples; `()` is the intercept. All
recovery, proxy and attribution calculations use float64 after native logits
are produced. Never label Fourier coefficients as SII or Shapley values
without applying the checked conversion.

SPEX uses the actual author's systematic shortened BCH construction for
`n=28,t=3`: BCH(63,45), shortening 17, 18 parity rows plus the zero delay.
There are three random binary subsampling matrices. Hash sizes `b=3,4,5`
require **456, 912 and 1,824 raw mask requests**. For each seed and budget,
query the student's exact design and decode with the source-aligned bounded
Chase search, signed amplitude, bin check and iterative message passing.
The freeze binds the tested source and all recovery defaults: 100 iterations,
Chase depth `2*t=6`, author Chase variant, residual fraction 0.9, minimum
amplitude `1e-8`, bin energy floor `1e-9` and stagnation window 15. No native
response may determine them. Unlike two recorded author edge cases,
reject a failed BCH decode and avoid dividing by a zero-shift bin amplitude
below `1e-8`. Retain the resulting small amplitude/sweep differences and
failed-decode discrepancies in comparisons; do not claim whole-estimator
byte identity. An expanded Chase play is a labelled
variant, not the default or an author-equivalent result.

ProxySPEX uses the student's squared-residual boosted trees and exact spectrum
extraction. `DecisionTreeRegressor` is the supplied tree-fitting primitive.
The prospective grid is depth `{1,3}` and tree count `{128,512}`, learning
rate 0.1, minimum leaf size 2. Three-fold training-only CV selects the proxy
by MSE. Supports are selected with an always-kept intercept and the minimal
energy-sorted prefix reaching at least 95% of nonconstant squared energy,
with deterministic tie order. This is a specified, robust boundary convention;
the actual author and maintained implementations have separately recorded
edge failures. Do not silently attribute the correction to those sources.

Ridge refinement selects alpha from `{1e-6,1e-3,0.1,1,10}` by three-fold
training-only CV. Each fold learns its proxy and selected supports from that
fold's training rows. The proxy hyperparameter is selected using these same
folds before ridge selection; this is development CV, not a nested unbiased
estimate of selection performance. No single fold's validation labels fit
that fold's tree or its coefficient support. Repeated input masks must stay
in one fold. Record exact fold assignments,
candidate scores, selected settings and all fit counts. After selection,
refit on the complete query training set. No held-out audit value is a fit
input, a stopping condition or a hyperparameter-selection input.

Actual source-pinned author/maintained algorithms are independent reference
comparators on the predeclared subset **prompt 0, seed 0, all three budgets**.
They consume the same fixed query rows and training values; their public
samplers differ, so this is a fixed-query algorithm comparison, not a claim
that their public APIs issue identical queries. Show the limited subset next
to the full 32-prompt student-method benchmark. The tested reference interface
and source-bound contract below fix the fitting grids and stage caps.
The educational boosting loop is not
claimed to reproduce LightGBM's split construction or optimizer. Report
model-to-proxy, proxy-to-spectrum, truncation and refinement errors separately.

The actual adapter is `reference_bridge.py`, contract
`3680dfd90c6b3cf88f6d7f454f978d031a81584e6b9f2ad666ea8e77b0fd949b`.
Author SPEX consumes saved subsampling matrices/delays and ordered value replay
through `SubsampledSignal` and actual `support_recovery('soft', ...)`. Author
ProxySPEX calls actual `spectralexplain.utils.proxy_spex` on the supplied
training rows; a read-only return-frame observer retains its intermediate stages.
The maintained shapiq comparator calls its actual model fit, tree conversion,
Fourier extraction and `_refine` stages. Public sampling is bypassed explicitly.

Author proxy selection uses depth `{1,3}`, trees `{500,1000,5000}`, learning
rate `{0.01,0.1}` and five-fold R2 CV: 60 CV ensemble fits plus one refit per
budget. The maintained grid is depth `{3,5}`, actual iterations `{500,1000}`,
rates `{0.01,0.1}` and five-fold R2 CV: 40 fits plus one refit. Both retain their
source's 100-alpha `RidgeCV`, intercept convention and full-training basis
selection. These are different procedures from the student's development CV.
Exact proxy-to-full-spectrum predictions use `atol=1e-8, rtol=1e-7`.
Native course/reference spectra have no required equality threshold: known
guard and floating tie differences remain measured and disclosed.

The bridge accepts only the specified scalar metadata, SPEX matrices/delays,
ordered training masks/values and audit masks. Audit labels and extra fields
are rejected. The pinned separate Python 3.12 environment and source contract
must validate before every actual reference stage. Fifteen genuine analytical
stage checks, including three exact saved-design SPEX replays, predate native
execution. The author parity failure remains a source outcome. The additional
host dispatch test is explicitly mocked; it is not an actual reference run.

## Queries, repetitions and held-out audit

Use seeds 0, 1 and 2 for each method and each of the three budgets. A seed is
an algorithm repetition, not an independent language-model checkpoint. The
32 prompts are a fixed factorial set, not independent random draws from a
natural-language population. Show all prompt-level errors and the spread
across seeds; do not imply a population confidence interval.

For each prompt, generate one ordered uniform-mask stream per seed from
`PCG64(9200 + seed)`, keeping unique masks and including empty/full first
without advancing the RNG. `native_query_plan.py` draws one 28-element uint8
Bernoulli row per RNG call, rejecting duplicates by the big-endian integer ID.
The draw dtype, shape and call order are part of the fixed design.
Proxy budgets take prefixes of lengths 456, 912 and 1,824. Its raw training
requests therefore equal its unique training masks. SPEX has repeated requests
by design; preserve those values in design order and count both raw and unique
queries. Any cross-budget or cross-method physical cache reuse is recorded
separately from the algorithm's logical requested-query cost.

Generate one common 512-mask held-out audit for every prompt using
`PCG64(7719)`. Before any response is queried, construct all planned training
designs and draw unique Bernoulli masks while rejecting their entire union
and the empty/full coalitions. Freeze audit IDs. These exact same audit masks
compare every method/budget/seed. Their labels are queried and exposed to the
evaluation stage only after all training choices and fitted predictions are
saved. Audit queries are additional evaluation cost and never disappear from
the physical-call inventory.

Keep a disk-backed oracle cache keyed by model, prompt/token/masking identity
and the full mask ID. Cache only requested masks, never the 2**28 game. Record
each request's stage, method, seed, budget, ordered mask IDs, cache-hit count,
unique count and new physical inference count. Store every unique mask and its
two target logits, difference and producing batch identity. Bound prediction
batch size at 256 and use fixed order; changed batching needs explicit repeat
validation, not an unrecorded adjustment after numerical differences appear.

## Controls and visible results

For each method's own training queries, fit a constant mean and an additive
linear response model; keep the intercept. Use unique query rows for these
baselines. The additive solver is float64 `numpy.linalg.lstsq` with
`rcond=None`, an explicit intercept and no regularization (the minimum-norm
solution if rank deficient). Retain rank and singular values. Fit both baselines
for both real and shuffled conditions, then report their held-out error beside
the method. A shuffled-response control uses `PCG64(3300 + seed)` to permute
values ordered by ascending unique training-mask ID, then maps those labels
back to the raw request order. This preserves duplicate-value consistency.
It runs the same recovery or proxy-selection procedure with the same settings.
These controls reuse model values but incur their own fitting/recovery cost.

Headline plots show per-prompt RMSE and normalized RMSE versus requested and
unique queries. Normalization uses the fixed held-out response standard
deviation; if it is zero, show unnormalized error and identify the constant
game. This statistic never drives model selection. Include constant/additive
and shuffled curves. Show R2 only with the same zero-variance handling.

Show all 32 unmasked/counterfactual logit differences beside intended signs.
Use fixed prompt index zero for the initial signed word-interaction display;
provide an all-prompt selector and report if its approximation is poor. Do
not choose the most convincing prompt after seeing results. Show strongest
absolute recovered terms with their signs and reconstructed held-out behavior.
Unknown native ground-truth support is not scored as correct or incorrect.

Separate analytic plots establish support precision/recall, signed coefficient
error, query scaling, collision resolution, BCH corruption, wrong-degree and
dense-spectrum failures. Those known-truth results cannot certify native
ground-truth support.

## Resources and failure behavior

CPU only; `CUDA_VISIBLE_DEVICES=''`. Native PyTorch inference uses eight threads
and four inter-op threads; BLAS/NumPy and reference fitting retain four threads.
The eight-thread choice was made before any native outcome, after inspecting
the available 24-core CPU. Earlier four-thread identity preflight is retained
as an unexecuted design checkpoint. No unrelated process is interrupted.
Use a maximum 8 GiB sampled process-tree RSS, 8 GiB
of new retained output and 90 minutes for one complete joint native execution.
Reference fitting is separately timed **inside** the same 90-minute joint cap,
not hidden model query cost. Its nine fixed variant/budget stages each have a
300-second cap, 8 GiB process-tree RSS cap and 256 MiB output cap. A reference
stage's actual failure or timeout is retained as that comparator's outcome,
with no fallback estimate or invented successful result. A breach of a joint
cap stops the whole execution. Reference processes inherit the joint deadline;
nine per-stage caps cannot extend the outer deadline.
A full independent source-only reproduction receives no fitted
learner proxy or oracle values and has the same per-run resource caps.

The executable `query_plan_v1` preflight contains 183 bound arrays and the
following no-outcome inventory. It uses the frozen masks, folds and permutations;
the counts are not estimates derived from a native run.

| Stage | Planned count |
| --- | ---: |
| Primary prompt games | 32 |
| Method/seed/budget combinations per prompt | 18 |
| Training union per prompt | 14,549 |
| Common disjoint audit masks per prompt | 512 |
| Primary training plus audit physical rows | 481,952 |
| Full-mask subject counterfactual rows | 32 |
| Invariance and head-comparison rows | 15 |
| Total planned physical native rows | 481,999 |
| SPEX recoveries, including shuffled controls | 576 |
| Proxy pipelines, including shuffled controls | 576 |
| Student proxy ensemble fits | 7,488 |
| Student residual-tree fits, depending on selected candidate | 2,285,568–2,506,752 |
| Ridge candidate solutions / SVDs | 9,216 / 2,304 |
| Constant and additive baseline fits, real and shuffled | 2,304 |
| Fixed-query reference stages / additional native rows | 9 / 0 |

The 32 final primary-sign requests reuse already queried full masks. Every
SPEX intra-request duplicate, cross-design cache reuse and shuffled-label reuse
is separately counted in actual oracle receipts. Fixed-query reference requests
reuse declared values; an undeclared mask fails without querying the model.
Reference fitting grids, actual fit counts and failures have their own receipts
and are additional to the student fit counts above.

Retain the complete ordered plan, all unique native two-logit values and batch
identities, primary/counterfactual probes, every fitted prediction, final signed
spectra, complete proxy CV/selection/fold records, extraction/truncation spectra
and baseline coefficients/rank. For native SPEX, keep initial/residual bins and
per-sweep energies, signed updates, acceptance/rejection counts and support counts.
Do not retain duplicate per-sweep full-spectrum copies or every decoder candidate;
the analytical collision lesson retains its full explanatory trace separately.
Fitted tree objects are discarded after predictions and exact spectrum extraction;
fresh reproduction fits them again. No pickle input or hidden fitted artifact is
passed to reproduction.

Source-only reproduction compares all shapes, dtypes and ordering; integer IDs,
supports, selected settings, folds and query counts must match exactly. Native
logits use `atol=1e-5, rtol=1e-6`; floating estimator predictions use
`atol=1e-8, rtol=1e-7`. Report exact matches and every within-tolerance difference
separately. A tolerated logit difference does not excuse a changed discrete
support or selection. Timing, absolute output paths and process IDs are measured
afresh and explicitly excluded from scientific equality. Poor prediction error
is a measured result and has no passing threshold.

If a resource cap or numerical check fails, retain the run, its source snapshot
and partial inventory. Do not silently reduce prompts, seeds, budgets or
controls; propose and record a prospective amendment before another run.
Poor scientific performance is a result, not grounds for changing the design.

## Publication boundary

Before native execution: independent protocol review, fixed method/reference
interfaces and constants, passing analytical checks, exact model/input/source
hashes and an immutable freeze with zero native-query and native-fit counts.
After execution: complete inventory and resource checks, independent source
reproduction, all own-answer cells executed, actual figure and browser review,
all help/solution/play checks, and truthful primary-source citations.

The joint requirement stays UNIMPLEMENTED until both learner sections pass.
Reports and review archives are evidence supporting those notebooks; they are
not their signature result or a substitute for instruction.
