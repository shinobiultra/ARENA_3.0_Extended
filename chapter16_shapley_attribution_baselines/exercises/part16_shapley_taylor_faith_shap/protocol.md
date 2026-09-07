# 16.16 protocol: Where Does Higher-Order Credit Go?

Status: **Development execution requires a reviewed zero-native-run freeze;
publication additionally requires independent approval**.
No 16.16 native fit, Concrete model-game calculation, interaction result or audit
load begins until `protocol_freeze.json` records the zero-run freeze and its actual
review disposition. This protocol may change before that freeze; it may not be tuned
after native outcomes. The selected requirement is
`shapley.shapley_taylor_faith_shap`, not SHAP-IQ estimators or SPEX.

## Question and mathematical targets

When an explanation retains only singletons and pairs, where should credit for
a higher-order mechanism go? Compare two efficient but distinct decompositions
on the same complete finite game. SII is a separate contextual derivative
average, not a joint efficient singleton-plus-pair expansion.

Games use CPU float64 arrays indexed by ascending integer coalition masks.
Bit i denotes feature i; mask zero is empty and mask 255 is full for the native
eight-feature game. The intercept is always `v(empty)` and is kept separate
from the sum of nonempty interaction coefficients.

Let a(U) be the Möbius dividend of U. At order two, Shapley-Taylor preserves
`I({i}) = a({i})` and sets
`I(S) = sum_{U superseteq S} a(U) / choose(|U|,2)` for each pair S.
All higher-order returned coefficients are zero. At order one it gives ordinary
Shapley values, and at full order it gives the complete Möbius expansion.

Faith-Shap minimizes the interior-coalition weighted squared error of
`v_hat(S) = sum_{T subseteq S, |T|<=2} beta(T)`, subject to exact constraints
`beta(empty)=v(empty)` and `sum_T beta(T)=v(full)`. The finite weight per
interior coalition of size s is proportional to
`(n-1)/(choose(n,s)*s*(n-s))` and normalized to sum to one. Endpoint zeros in the
stored finite-weight vector exclude equality-constrained rows; they are not the
mathematical endpoint penalty. The student solves exact constrained least squares
and separately implements the primary-paper closed form. A package's finite
endpoint penalty is a qualified numerical comparison, not this definition.

Pair SII averages the pair's second difference over all disjoint contexts T with
weight `|T|! * (n-|T|-2)! / (n-1)!`. Store ordinary Shapley singleton values
separately. Do not add that singleton vector and SII pairs to claim efficiency.
Kernel-optimal reconstruction does not imply better unweighted reconstruction,
causal discovery or a preferred explanation target.

The primary definitions are Sundararajan, Dhamdhere and Agarwal (2020),
[Shapley-Taylor](https://proceedings.mlr.press/v119/sundararajan20a.html), and
Tsai, Yeh and Ravikumar (2023),
[Faith-Shap](https://www.jmlr.org/papers/v24/22-0202.html).
Pinned author code and shapiq support independent implementation comparisons.

## Data, predictors and known prior outcomes

Use the byte-pinned 1,030-row UCI Concrete data and known recipe-disjoint split:
758 training / 153 explanation / 119 audit rows in 298 / 64 / 65 recipe groups.
Keep all original rows, including duplicates and conflicting outcomes. Feature
order is Cement, Slag, Fly ash, Water, Superplasticizer, Coarse aggregate,
Fine aggregate and Age; the scalar target/game output is strength in MPa.

Refit the same six float64 8-32-32-1 affine/ReLU MLPs (three clean and three
paired shuffled targets, 800 updates each) plus ridge. Training normalization,
seeds, target permutations and raw-unit folding follow `data_manifest.json`.
Require all 21 known initial/standardized/raw model-state hashes to match before
any native game calculation. No loaded previous weights may replace the refit.
The data and predictive results are already known from 16.14/16.15. This is a
self-contained reproduction of the same models and data, not independent
prediction or dataset replication. No predictive-performance cutoff selects
which models are analyzed.

Model order is `clean_0`, `clean_1`, `clean_2`, `shuffled_0`, `shuffled_1`,
`shuffled_2`, `ridge`. The native study uses order two only. Higher-order exact
small games provide the order-one/order-three/full-order method checks.

## Complete empirical game and retained scope

For each query x, intact training donor z and mask S, replace coordinates in S
by x and retain other coordinates from that same z. Average scalar model outputs
over the equal-weight donor bank to define v(S). The background is part of the
target game, not a sampling-error estimate or a conditional population model.

Retain every donor prediction for all seven models, all 272 held-out rows, all
256 masks and all primary64 donors: 1,904 games, 487,424 averaged coalition
values and 31,195,136 hybrid predictions. Model/game method outputs, reconstructions,
residuals, all 28 pairs and all 64 contexts per pair remain accessible.

For clean_0/all 272 rows, use primary64, nested16, nested128, independent64_a and
independent64_b, preserving exact inherited donor order. Primary64 aliases its
already-computed clean_0 game; the four alternatives add 1,088 distinct games
and 18,939,904 donor predictions. Thus there are 2,992 unique empirical games.
Background movement is a change of estimand, not a confidence interval.

Store full-context second differences for every model, row and pair, with an
independent direct four-forward check using the same donor bank. The four masks
are context, context-plus-first, context-plus-second and context-plus-both, and
the signed difference is fourth minus second minus third plus first. Keep the
raw four predictions as well as their difference.

## Exact controls and random designs

Use the eight fixed four-player games specified literally by rational Möbius
dividends in `data_manifest.json` under `fixed_exact_games`. Unlisted dividends
are zero and coalition values are the subset sums of these dividends. The games
are constant, additive, positive_pair, negative_pair, pure_triple, pure_four,
mixed_degree2 and mixed_high. A degree-two game is an agreement
control, not an additive zero-pair control. Ridge and the additive game supply
the zero-pair controls. The shuffled predictors need not have zero interactions.

For a unit pure triple, order-two Shapley-Taylor gives pair 1/3 and singleton 0;
Faith-Shap gives pair 1/2 and singleton -1/6. For a unit pure four, the respective
values are pair 1/6 / singleton 0 and pair 3/10 / singleton -1/5. The negative
Faith singleton is a surrogate coefficient, not evidence that a positive
mechanism physically harms the outcome. Store exact rational oracles.

Random controls use NumPy 1.26.4 PCG64 and namespace
`arena16.16/shapley-interactions/v1`: seed 161610 for 15 exact random games
(three each at n=3..7), 161611 for 128 within-size audit-game shuffles, and
161612 for 256 random pair-order permutations shared across models/methods.
Each private stream receives the unsigned little-endian integer represented by
the first 16 bytes of SHA256 over one of these exact UTF-8 messages, with decimal
integer substitutions and no trailing newline:

- `arena16.16/shapley-interactions/v1/161610/{n_players}/{replicate}`;
- `arena16.16/shapley-interactions/v1/161611/{source_id}/{replicate}`;
- `arena16.16/shapley-interactions/v1/161612/{replicate}`.

The retained `derived_seed_hex` or `derived_seeds_hex` values are 32-character,
zero-padded hexadecimal representations of that resulting integer. They are
not `digest[:16].hex()`, which displays the digest bytes in the opposite order.
No global NumPy or Torch random stream is consumed by these control designs.
For random exact games, iterate n=3..7 outside replicate=0..2 and make one
`standard_normal(2**n)` float64 call per private stream, assigning in ascending
mask order. Retain the 15 literal vectors and derived seeds; test each at orders
one, two and three, giving 45 game/order cases **per method** and 90 STII-plus-FSII
comparisons. SII is compared on the 15 distinct games. Further full-order checks
use replicate zero at n=3,4,5,6,7, each at order n, and are retained as five
separately labeled boundary cases. The n=3/order3 boundary intentionally repeats
one of the 45 grid cases. It is a separate package instance, not a new random
game. The eight fixed four-player games each receive one order-two reference.
Thus exact validation has 58 direct game instances and 174 package index calls.

For each clean_0 audit row, each within-size shuffle independently permutes
values among masks of cardinality 1..7, keeping empty/full fixed. Retain all
128 x 119 x 256 int16 mask permutations and all three methods' pair values.
Iterate replicate=0..127 outside the pinned ascending audit source IDs; within
each private stream visit cardinalities one through seven in order and call
`permutation` on that cardinality's ascending mask IDs. The selected values at
these destination positions are the source game values at the resulting indices.
Retain the derived seed hex strings as a nested `[128][119]` record. The manifest
prospectively binds the complete permutation values using `tensor_mapping_sha256`
with the literal mapping key `permutation`, as defined by the pinned data module.
This is
a structural null, not another donor-model game and not an expected-zero index.
Show observed and all null top-three-share/peak-to-mean values with explicit
zero-denominator flags. No localization or method superiority is a pass gate.

## Audit boundary, aggregation and display

After freezing, load training/explanation only, refit and check all model states,
compute exact/explanation games and all explanation-derived orders, then persist
complete fitted/prepared artifacts and physical leaf manifests. Only after that
seal may the literal audit loader run. Keep pre-audit shards immutable; do not
retroactively concatenate audit data into the prepared artifact.

Rank every pair for each model/method by explanation recipe-group-weighted mean
absolute pair magnitude, with lexicographic ties. Average rows within each group,
then groups equally. On audit, compare cumulative capture of group-weighted mean
absolute full-context pair synergy against all 256 random pair orders. A zero
denominator gives a zero curve and a separate flag. All models, rows and pairs
are retained regardless of apparent agreement, null results or outliers.

For each random pair order, create its private PCG64 stream and make one
permutation of the ascending 28 lexicographic pair indices. Retain all 256
integer seeds and `[256,28]` int64 permutations. Their prospective value hash
uses the literal tensor mapping key `orders`. Use the same orders for every
model and method, retaining the complete `[7,256,29]` random capture curves.

Content-blind local defaults are model clean_0, source ID 0, donor ID 173,
pair (0,1) and full-context mask 252. Source ID 0 is in the explanation split;
173 is the first primary64 donor. Their membership and order are declared in
the manifest and independently checked before freezing. Native displays use fixed
IDs and complete seed/group summaries; no audit result chooses a headline row,
method, background, pair, threshold or axis inclusion. Editable post-study plays
are explicitly exploratory and do not change the frozen primary results.

For support geometry, use every clean_0 query/mask and the first eight primary
donors. Standardize by training population mean/SD and retain the exact Euclidean
nearest-training-row distance and source ID against all 758 training rows:
`[272,256,8]` each. This bounded geometric diagnostic is neither density nor
causal validation and does not reduce the primary64 game. Compute float64 squared
differences feature by feature in index order, sum in that order, and take the
square root. If the computed minimum is exactly tied, return the smallest
training source ID. The original data contain duplicated feature vectors, so
this tie rule is part of the result definition.

## Execution, references and physical evidence

Use CPU float64 and four Torch threads. Raw fitted states, normalization,
permutations and loss traces must be serialized rather than opaque live models.
Physical donor shards preserve explanation/audit separation, and primary64
aliases avoid duplicate arrays. Every leaf records path, shape, dtype, element
count and value hash; every artifact records file hash/size. External references
also identify their exact physical shard and value hash. Archive large tensors
by exact persistent-cache bindings while retaining their numerical bytes there.

The ten physical float64 donor-prediction shards are fixed below. Paths are
relative to the corresponding prepared/evaluated payload, using dotted mapping
components. Every `.pt` has its own full leaf/file manifest. Explanation files
are written before audit access; audit files are written afterward.

| Artifact | Logical payload path | Shape |
| --- | --- | --- |
| `primary_explain_hybrid.pt` | `primary64.hybrid_prediction` | `[7,153,256,64]` |
| `nested16_explain_hybrid.pt` | `background_variants.nested16.hybrid_prediction` | `[153,256,16]` |
| `nested128_explain_hybrid.pt` | `background_variants.nested128.hybrid_prediction` | `[153,256,128]` |
| `independent64_a_explain_hybrid.pt` | `background_variants.independent64_a.hybrid_prediction` | `[153,256,64]` |
| `independent64_b_explain_hybrid.pt` | `background_variants.independent64_b.hybrid_prediction` | `[153,256,64]` |
| `primary_audit_hybrid.pt` | `primary64_audit.hybrid_prediction` | `[7,119,256,64]` |
| `nested16_audit_hybrid.pt` | `background_variants_audit.nested16.hybrid_prediction` | `[119,256,16]` |
| `nested128_audit_hybrid.pt` | `background_variants_audit.nested128.hybrid_prediction` | `[119,256,128]` |
| `independent64_a_audit_hybrid.pt` | `background_variants_audit.independent64_a.hybrid_prediction` | `[119,256,64]` |
| `independent64_b_audit_hybrid.pt` | `background_variants_audit.independent64_b.hybrid_prediction` | `[119,256,64]` |

The shards contain 50,135,040 unique raw predictions (401,080,320 numerical
bytes). Primary64 in each background-variant mapping is a checked view at model
index zero into the corresponding primary shard. All-row hybrid access is a
logical explanation/audit concatenation of these physical arrays, preserving
the split boundary without another physical copy. Descriptors retain storage,
artifact, manifest, payload path, kind, shape, dtype, elements, value SHA256,
artifact SHA256 and artifact size; a view also retains its selector and selected
value SHA256. Direct descriptor fields are `kind`, `array_kind`, `artifact`,
`manifest`, `payload_path`, `shape`, `dtype`, `elements`, `value_sha256`,
`artifact_sha256` and `size_bytes`; views also contain `target_payload_path` and
`selector`. Descriptor payload paths use slash-separated mapping components;
the prospective table uses the equivalent dotted paths for readability.
Logical all-row mappings use `storage=split_external_shards` and their two
phase descriptors, with no concatenated donor tensor on disk.

The storage/RAM/runtime limits are 0.75 GiB artifact bytes, 4 GiB peak
RAM and two CPU hours; engineering checks must pass before freezing. Batch
changes may control memory, but cannot drop rows, models, donors, contexts or
controls. A measured timeout is not proof a process ended; inspect its live
handle and do not start duplicate runs.

Independent derivative-STII, closed-form Faith-Shap and pair-SII references
cover all 2,992 empirical games. Separate NumPy formulas supply the independent
operators by evaluating complete-game basis vectors; a separate equality-
constrained KKT solve checks the Faith formula. These independent basis calls
are explicitly counted as such. Every empirical game receives one direct
`shapiq==1.5.2` ExactComputer instance and three index calls (STII, FSII, SII),
giving 8,976 direct index calls. Package calls never use a package basis operator.
The complete source/package identities are pinned by the reference adapter and
the manifest. The Faith-Shap author solver is pinned at commit
`a5226d600a8ea5761a95da9e23bd58eee5aa85a5` as primary implementation provenance.

Subtract `v(empty)` before every independent/package nonempty computation and
restore the empty coefficient afterward. For each game let
`scale = max(1, max_S(abs(v(S)-v(empty))))`, with the maximum over all coalition
masks. Native errors and their absolute floors are in MPa; analytic games use
their declared scalar game units. Compare each coordinate separately using the
following inclusive inequalities, then report the maximum error/bound ratio:

| Comparison family | Bound on absolute error |
| --- | --- |
| Student versus independent STII/FSII/SII, independent KKT versus closed FSII | `1e-10 + 1e-12 * scale` |
| Direct shapiq STII and pair-SII versus independent values | `1e-10 + 1e-12 * scale` |
| Direct shapiq FSII versus independent FSII coordinates | `1e-8 + 5e-9 * scale` |
| Student/independent/STII package efficiency residuals | `1e-10 + 1e-12 * scale` |
| Direct shapiq FSII efficiency residual | `1e-8 + 3e-8 * scale` |

Errors are student-minus-independent or package-minus-independent; the KKT
comparison is KKT-minus-closed-form. Efficiency residual is the sum of nonempty
coefficients minus `v(full)-v(empty)`. All returned coefficients above requested
order are exactly zero as a separate structural test. Exact endpoint constraints
define the mathematical Faith problem; solved float64 endpoints satisfy the
strict numerical rule rather than a bit-exact summation promise. Package FSII
uses a finite `1e8` endpoint penalty, so its efficiency diagnostic and coordinate
parity have distinct looser bounds. Those package bounds do not weaken the exact
KKT/closed-form checks or establish an alternative scientific target.

For each native game, retain student/independent STII and FSII full `[256]`
axes plus SII `[28]` pair values, giving 1,615,680 coordinates per side over
the complete scope. Also retain the independent KKT `[2992,256]` values.
The package returns `[2992,37]` for each index: intercept, eight singletons and
28 pairs in ascending mask order. All 332,112 returned coordinates are retained;
package SII singletons/intercept are outside the pair-SII comparison, so actual
package comparisons cover 305,184 coordinates. The lexicographic pair axis is
mapped explicitly into these ascending package masks. Preserve all signed error
arrays and efficiency residuals alongside summaries, masks and source-game hashes.

Across exact and empirical evidence, there are 3,050 direct game instances and
9,150 package index calls, 1,621,494 full-axis student comparison coordinates,
308,074 relevant package comparison coordinates and 335,342 raw returned package
coordinates. Compute the exact-suite references once during preparation and the
all-game empirical reference once after evaluation. The visible learner calls
`reference_check` for that empirical array and `finalize_study_reference` to
validate/augment it with the already-retained exact suite. The finalizer makes
no package call. Standalone reproduction calls `validate_study_reference`, which
performs the same one empirical computation followed by the same finalizer.

These bounds were chosen before any native outcome using the coherent seeded
random/untrained feasibility study: script SHA256
`f6a01486633102e2a19b15c7de9611011c27f13cfa6a539cde7d76ca5bf2c628`,
JSON `f8a0a7a3c945d3209fb6317593337ca4350ae83e682119428af5deeb9fce8166`,
NPZ `ce0e629457ce1e5c22dc8309418f42001010c661dafc610e83121affd7e35ba0`.
It evaluated all 2,992 synthetic games directly in about 24.4 seconds. The D8
exact KKT condition number was about 354; the package finite-penalty weighted
design was about 149,053. These are feasibility results, not Concrete results.
This comparison policy supersedes the earlier blanket `1e-8`, normalized
`5e-8`, and `atol=rtol=1e-10` proposals. Never widen a bound after observing a
failing native value.

Execute the complete learner from its own answers, with all taught methods and
scientific orchestration visible. A separate fresh process refits the seven
models without receiving learner fitted inputs and reproduces every deterministic
leaf exactly. Timing is a separate descriptive branch. Actual generated figures,
math, dropdowns and editable plays require source-bound visual review before
publishing the paired notebooks, page and namespaced figures. Reports support
the learning artifact and cannot replace it.

## Authoritative contract and zero-run review

The literal `retained_payload_contract`, reference scope, random designs,
physical shard/view definitions and logical split mappings in `data_manifest.json`
are normative. Every declared required leaf has an explicit prospective shape,
dtype and count; post-execution recursive manifests additionally bind its actual
values and bytes. The cache design documents are development provenance. This
protocol and the final manifest supersede their earlier additive-label, seed,
reference-bound and call-count proposals. All package and lock source hashes in
the manifest must agree with the installed files checked by the reference adapter.

The independent pre-freeze review identified six specific resolutions and approved
the mathematical targets, exact fractions, complete empirical scope, controls and
reference feasibility. Its final recheck stopped because the reviewer agents hit
their account usage limit. The root agent may resolve those six findings with
source-bound no-fit checks and create a **development-only** zero-run freeze.
That record must retain `independent_approval=false`, identify the root resolution,
and leave publication unauthorized. It must not relabel the blocked independent
review as a pass. This workflow change implements the user's authorization to
continue development and revise legacy process; it changes no game, data split,
model, seed, retained coordinate, tolerance, control, or display selection.

`pre_run_resolution.json` records the original independent review, each resolution,
the pre-native checks, and the exact source snapshot. The timestamped freeze binds
the protocol, manifest, data loader and resolution record before any native outcome.
The source snapshot includes the learner, methods, study, references, diagnostics,
plots and execution pipeline. Until this freeze, only analytic, seeded random or
untrained checks run. A development execution and its fresh reproduction are usable
engineering evidence, but do not authorize publication or registry promotion.

An independent recheck after development must explicitly state its timing, approve
publication, and bind both the unchanged frozen design and the actual executed
source set in `independent_recheck.json`. The publisher enforces that additional
gate. Do not register 16.16 as taught until that review and the complete native,
visual, reproduction and report gates actually pass.
