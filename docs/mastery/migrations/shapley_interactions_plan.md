# Canonical 16.5 Shapley Interactions: full three-curriculum migration plan

Status: read-only investigation plus this plan only. No candidate master, canonical section, source archive, notebook, shared registry/config/workbook or
existing 16.11 artifact was changed. No taught method, package estimator, data loader, model, training, native study, cache probe or test was executed.
Inspection used standard-library source/JSON/AST/hash and archive-directory reads; no numerical archive payload was deserialized. Future implementation requires
root GO and an independently reviewed frozen candidate before adoption.

## Required salvage plan

- **Current learner failure:** canonical 16.5 is missing, while its syllabus material lives in three complete historical curricula with different targets and
  evidence scopes. A short replacement would erase substantial teaching and conflate exact interaction definitions with approximation algorithms.
- **Target section claim:** understand pair SII in a complete game; distinguish Shapley–Taylor and exact constrained Faith-Shap allocation of omitted
  higher-order dividends; estimate ordinary Shapley and pair SII under explicitly charged shared-query budgets. These are three related experiments, not one
  interchangeable definition.
- **Signature results:** preserve the toy four-panel recovery/control figure; all eleven allocation/native-game figures; and all ten shared-query/budget
  figures. Preserve each native study completely, including negative findings, exact controls and separately scoped reference comparisons.
- **Student implementations:** retain all seven toy exercise groups, eight allocation-definition groups teaching ten core functions, and eight estimator groups
  teaching nine core functions, with immediate checks and their own answers. Do not replace notebook implementations with imports from `solutions.py` or
  reference modules.
- **Controls:** exact constant/additive/signed/high-order/dummy games, within-size structural shuffles, independent/package definitions, ridge and
  shuffled-target model controls, random pair orders, zero-pair estimates, variance/rank/constraint diagnostics, and all fixed proposals/budgets/replicates.
- **Play:** retain the toy editable polynomial/sampling play and anomaly hunt; allocation dividend and donor/context plays; estimator single-query and
  native-budget plays. Preserve actual source defaults and existing computation pathways without running them.
- **Files after GO:** new `master_16_5.py` and derivative master notebook; canonical `part5_shapley_interactions` notebooks/module/README/media and explicitly
  necessary plumbing; generated instruction page; dedicated migration tests/docs/evidence. Root alone owns shared registry, navigation, configuration and
  workbook adoption.
- **Checks after GO:** strict flat archive coverage; immutable source/report/figure hashes; three-way code mapping and reversible namespace proof; own-answer
  reconstruction; import safety; output/widget preservation; helper/test acyclicity; expected-only exercise displays; local media/routes; bounded offline
  static/tiny-fixture selector if authorized; standard guided-case validation; freeze and independent review. No native reproduction follows implicitly.

The salvage skill, all six source notebooks, all three instruction pages and section `solutions.py` files were inspected. Original ARENA
`chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/1.2_Intro_to_Mech_Interp_{exercises,solutions}.ipynb` supplies the comparison: concrete
model/question, visible learner functions, immediate method checks, interpretations, controls and exploration. Preserve original ARENA append-only.

## Exact publication inventory

All paths below are beneath `chapter16_shapley_attribution_baselines/`.

| Historical curriculum | Exercise cells | Solution cells | Code cells in each | Solution output records | PNG output records | Notebook bytes, exercises / solutions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `exercises/part3_shapley_interactions_shapiq/16.3_Shapley_Interactions_with_shapiq` | 32 | 32 | 13 | 14 | 0 | 53,021 / 78,556 |
| `exercises/part16_shapley_taylor_faith_shap/16.16_Where_Does_Higher_Order_Credit_Go` | 149 | 141 | 55 | 42 | 11 | 2,050,447 / 2,046,943 |
| `exercises/part17_shapiq_estimators/16.17_One_Query_Many_Interactions` | 57 | 49 | 22 | 23 | 10 | 2,267,567 / 2,268,428 |
| Total | 238 | 222 | 90 | 79 | 21 | 8,764,962 across all six notebooks |

The solution pair-to-exercise mapping must use track plus code-cell ordinal and own-answer semantics, not identical notebook-cell indices: the latter two
exercise notebooks contain eight additional markdown cells each. Their source exercises already contain historical outputs (34 and 15 records respectively);
canonical exercise counts must be `None` and outputs must be regenerated expected displays only. Preserve their exact original bytes in the historical bundle,
rather than silently presenting actual executions as learner work.

The toy signature is **saved to an asset and closed without a notebook PNG display** in source solution cell 19. Its notebook's fourteen records comprise nine
streams and five HTML/table displays. Therefore the full signature count is **22 figures**, consisting of 21 native PNG outputs plus
`instructions/assets/shapley_interactions_exact_signature.png`. Do not report 21 and lose the original toy visual. Add its expected display in canonical output
without pretending it was an original PNG output record. Historical solutions retain all 79 original output records; any additional expected visual display must
be separately labeled, never counted as fresh execution.

The original pages are `03_[16.3]_Shapley_Interactions_with_shapiq.md` (39,913 bytes), `16_[16.16]_Where_Does_Higher_Order_Credit_Go.md` (123,245), and
`17_[16.17]_One_Query_Many_Interactions.md` (47,608). Their local image references identify exactly 22 assets: one toy signature, eleven
`shapley_interactions_01..11` assets, and ten `shapiq_01..10` assets. Preserve original assets and carry their exact bytes into namespaced canonical media. No
source notebook currently has notebook-level widget metadata, but the preservation check must explicitly detect and retain any graph if that changes before
implementation.

## Track A: original game and pair-SII intuition

Retain the four-player polynomial game with player names `new_device`, `foreign_login`, `high_amount`, `many_attempts`; additive weights `[.2,.2,5,4]`, pair
`(0,1)` with weight 3, and triple `(0,1,2)` with weight 2. Keep all sixteen coalitions, complete normalization, discrete second differences, pair SII, ordinary
Shapley values, merged-player permutation sampling, within-size value permutation, direct `shapiq` parity and recovery reports visible.

Core definitions are `all_coalitions`, `normalize_coalition_values`, `polynomial_game`, `discrete_second_difference`, `pairwise_shapley_interactions`,
`exact_shapley_values`, `sampled_pair_interaction`, `permute_coalition_values_within_sizes`, `shapiq_pairwise_interactions`, and `interaction_recovery_report`.
Seven groups immediately call eight semantic checks; sampling/permutation share a group. Keep the mean/10th/90th-percentile sampling curves at budgets
`[2,4,8,16,32,64,128,256]` and all forty seeds, permutation seed 0, additive zero-pair control, large-main-effect ranking failure and triple-induced contextual
anomaly. A within-size permutation is a structural null, not an expected-zero interaction guarantee.

The play keeps its original polynomial knobs, budget 32 and seed 7. The anomaly table and pinned neural-target connection remain. The full legacy `solutions.py`
also contains the distinct CUDA neural-game pathway (`NeuralCoalitionGame`, trained-game records, four-feature truth/ablation tables, shuffled targets,
preflight and release entry points). Its recorded defaults include seed 1621, hidden width 64, 1,200 steps and learning rate .05. Retain this source/helper
identity and historical evidence; do not run it, replace the CPU signature with it, import it eagerly, or reinterpret its CUDA evidence as a new canonical
native run.

## Track B: exact allocation definitions and complete native scope

Preserve the separate definitions of Möbius dividends, Shapley–Taylor, exact equality-constrained Faith-Shap, its independent closed form,
reconstruction/weighted error, donor-replacement games and contextual pair interventions. Core functions are `mobius_transform`, `shapley_taylor_interactions`,
`faith_design_matrix`, `faith_kernel_weights`, `faith_shap_interactions`, `faith_shap_closed_form`, `reconstruct_game`, `faithfulness_error`,
`masked_model_game`, and `contextual_pair_synergy`. Keep diagnostic functions, all visible fitting/game/ranking/preparation/evaluation/reference orchestration
and adjacent answers, rather than a simplified wrapper around a hidden method.

Do not equate pair SII plus ordinary Shapley singletons with an efficient joint expansion. Faith-Shap's finite interior kernel and exact endpoints differ from
the package's finite endpoint penalty. Negative singleton surrogate coefficients do not imply physical harm. Reconstruction, efficiency, package parity,
contextual synergy and held-out pair capture remain different metrics.

The immutable protocol specifies 1,030 Concrete rows, splits 758 train / 153 explanation / 119 audit in 298 / 64 / 65 recipe groups; six known float64 8–32–32–1
ReLU models, 800 updates each, plus ridge, with 21 initial/standardized/raw state-hash checks. These reproduce known predictors from historical 16.14/16.15, not
independent discovery. Preserve model order, source IDs, target permutations, normalization and all `data_manifest.json` parameters. The audit loader must stay
behind the original persisted preparation boundary.

Preserve all seven models × 272 rows × 256 masks × 64 donors: 1,904 primary games, 487,424 coalition values and 31,195,136 donor predictions. Four additional
clean_0 backgrounds add 1,088 games; primary64 aliases the existing game, giving 2,992 unique empirical games. Keep every pair/context, full-context independent
four-forward check, all 256 shared random pair orders, 128 within-size shuffles of all 119 audit rows, explanation-group-weighted pair rankings, audit
recipe-group weighting, and all support-geometry/source-ID tie rules. The exact suite, direct per-game package calls, independent KKT/closed form and
scale-dependent predeclared bounds remain separate. Never widen bounds after a failure.

Keep literal random namespace `arena16.16/shapley-interactions/v1` with its exact SHA-derived little-endian PCG64 stream rules and seeds 161610/161611/161612.
Native display/play defaults remain clean_0, source ID 0, primary64, pair `(0,1)`, context mask 252 and inherited donor handling. `DONOR_SOURCE_ID=None` in the
notebook must not be silently replaced with a guessed donor; manifest-derived first donor 173 is a separate explanatory fact.

The original zero-run freeze records development-only/no independent approval. A separate `independent_recheck.json` records `APPROVE_PUBLICATION`,
after-development timing and exact reviewed sources. Preserve both chronological claims plus the prior blocked-publication record. Do not rewrite the initial
freeze to appear independently approved, or transfer that historical approval to the new combined migration.

## Track C: shared queries and fixed estimator-budget study

Preserve all nine student functions: `coalition_design`, `sii_query_weights`, `sample_coalitions`, `shapiq_estimate`, `query_sampled_game`, `constrained_wls`,
`pair_residual_design`, `kernel_weights`, `kernelshap_iq`. Keep the one-query four-player cold open, exact boundary treatment, importance correction/sample SE,
deduplicating oracle receipt, constrained/nullspace least squares, sequential main-effect/pair stages, finite-penalty boundary-limit derivation and all
immediate checks. The proved pairwise KernelSHAP-IQ limit is not exact interpolation of every boundary row or a general higher-order claim.

The complete native study reproduces the same seven known Concrete predictors on all 272 held-out rows with 64 intact donors, giving 1,904 games. Exact boundary
sizes 0,1,7,8 give eighteen unique coalitions. Interior sizes 2..6 have 238 possible coalitions. Preserve both kernel and uniform proposals, 32 replicates per
proposal with seeds `161750+r`, one 1,024-draw plan per proposal/replicate, and prefixes `[16,32,64,128,256,512,1024]`. All duplicates remain in estimator
means/SE; both estimators receive identical queried masks/values within each plan. All plans are shared across rows, so rows are not independent sampling
replicates.

Charge K draws, 18+K nominal requests, actual unique masks and unique masks × 64 donor predictions separately. Keep exact-reference construction cost 487,424
coalitions / 31,195,136 donor predictions. These are logical queries to a precomputed oracle, not measured inference speedups. Retain all four
estimator/proposal combinations, every signed pair estimate, exact pair-SII/zero baseline, predicted iid variance versus estimated SE squared versus empirical
replicate error, free rank/singular values/equality residuals, fixed SVD threshold and minimum-norm solution. No implicit ridge, clipping, rank filtering, seed
selection or audit-selected winner.

Defaults remain clean_0, source ID 0, pair `(0,1)`, replicate 0 and 128 draws for detailed display; play seed 161750 and kernel proposal. Keep both plays and
their audit-reuse warning. The historical review scope is author review, distinct from Track B's historical independent after-execution review.

## Source/report closure and exact archive proposal

Current static SHA checks match every `verification_report.json` input: **14/14** toy, **51/51** allocation/native and **67/67** estimator. Allocation
own-answer/learner/fresh receipts have **22/22** current source inputs; estimator equivalents have **32/32**. These are source-identity checks only: no old
model result was recomputed, accepted status renewed or cache availability established.

The three complete source trees have **90 files**, excluding `__pycache__`, totaling **30,874,453 bytes**. Add all three pages and their 22 exact image assets.
The union of report dependencies contains 127 paths; 25 are additional paths outside those trees/pages/assets. Thus the report/publication preservation universe
is **140 distinct file identities**. The toy semantic-test dependency `arena_ext/shapley_attribution.py` adds one explicit transitive source identity, giving
the implementation archive universe **141 files**; its imports are standard library plus Torch, with no further repository-source import. Include each original
tree file, including failed/review/publication histories, rather than only whichever receipt looks successful.

Before implementation writes, use `scripts/evidence_bundles.py` to seal a strict flat bundle and a disjoint union manifest. Two original nested archives remain
external exact-path/hash references:

- `part16_shapley_taylor_faith_shap/expected_outputs/execution.tar.gz`: 65 regular members, 12,328,113 expanded bytes, no `.pt`/`.npy`/`.npz`/pickle/raw-weight
  members.
- `part17_shapiq_estimators/expected_outputs/execution_review.tar.gz`: 92 members, 19,436,594 expanded bytes, likewise no raw numerical/model payloads.

Use **113 internal flat members + 28 external references** as the planned coverage: the two nested archives, 25 supplemental report-source dependencies and
`arena_ext/shapley_attribution.py` remain externally hash-bound. Do not nest or duplicate them. Preserve original archive sidecar contracts as ordinary JSON
evidence. Assert internal/external disjointness and equality of their union with all 141 original identities; verify every source hash and strict bundle
contract. Recheck inventory immediately before packaging because others may change shared dependencies.

The 25 supplemental references include `arena_ext/verification.py`, `uv.lock`, two report/contract scripts, the contract test and twenty historical
`part15_gradientshap` files. Preserve exact source identities for its `gradientshap_data`, `gradientshap_study`, contracts, model-state checker and references.
These are historical known-model dependencies; current canonical section numbers must not reroute them. The toy semantic tests additionally use
`arena_ext.shapley_attribution`; its explicit transitive dependency is already included in the 141-file universe above. Any further source-only imports
discovered during implementation require a separately named closure expansion, not an undeclared omission.

No external cache path was statted, opened or enumerated. Receipts name historical locations under `~/.cache/arena_extended/shapley_interactions/` and
`~/.cache/arena_extended/shapiq_estimators/`; presence is **unprobed**. Track B describes ten split donor-prediction shards, 50,135,040 raw float64 predictions
/ 401,080,320 numerical bytes with exact external shard/view descriptors and preparation/audit separation. Track C describes six full cached payload families
and roughly .95 GB written per historical run. Textual JSON receipts do not contain those arrays. Preserve locators, physical/value hashes, shape/dtype/leaf
manifests and aliases without loading tensors or claiming recoverability. The original tensor-reproduction code stays available; any future reproduction needs
separate execution authority and its original complete protocol.

## Namespace, source visibility and converter hazards

Use three explicit track identities, for example `toy_`, `allocation_`, `estimator_`; map actual global bindings lexically rather than blindly replacing words.
Top-level assigned/loop/import/function collision sets currently include:

- Toy/allocation: `Mapping`, `Path`, `math`, `np`, `pair`, `pd`, `plt`, `random`, `row`, `t`, `tests`.
- Toy/estimator: `Path`, `chapter`, `itertools`, `math`, `np`, `pd`, `plt`, `root_dir`, `sys`, `t`, `tests`, `value`.
- Allocation/estimator: `Any`, `MODEL_KEY`, `Path`, `SOURCE_ID`, `SimpleNamespace`, `Tensor`, `display`, `figure`, `games`, `math`, `methods`, `np`, `os`, `pd`,
  `play`, `plt`, `show_figure`, `t`, `tests`.

`from __future__ import annotations` is a directive, not a bound global named `annotations`: exclude it from name maps. It appears after earlier notebook work,
valid across notebook cells but not as a late import in a concatenated Python module. Preserve notebook directives and explicitly hoist/filter the Python-only
directive at the module start. Record and mechanically invert that exact converter adaptation separately from namespace-only rewriting. Do not advertise raw AST
equality while silently deleting or altering directives.

The toy's `import part3_shapley_interactions_shapiq.tests` relies on the original exercises-root path setup. Import-safe canonical routing should use an
explicit fully qualified legacy module plus its own alias and an independently recorded reversible import adaptation. Its lazy CUDA release delegation must
continue pointing to the historical toy module, never canonical `solutions.py`, which would recurse. Keep future imports, decorators/dataclasses, nested scopes,
comprehensions, argument defaults, annotations and function-local bindings correct.

Neither notebook exposes `globals()[name]` core lookup, but allocation helpers enforce `METHOD_NAMES` through `getattr(methods,name,None)` and `reference_check`
dynamically; estimator study similarly checks its nine named methods, and the report wrapper uses dynamic `getattr(tests,'test_'+name)`. Preserve
`SimpleNamespace` field names, exact public strings, keywords and helper interfaces; only aliased variable references change. A renamed function assigned to an
original interface field must retain that original field key.

Keep tests one-way: toy semantic tests depend on `arena_ext.shapley_attribution`; allocation tests depend on legacy `.interaction_methods`; estimator tests
depend on legacy `estimator_methods`. No test/reference import may reach canonical generated `solutions.py`. Imported data/plumbing/reference modules remain
explicit with exact preserved sources. Plot helpers may remain plumbing only where the plotted statistic is defined visibly or linked to its retained complete
source; no taught estimator/decomposition may be hidden behind them.

For all ninety code cells, store original track/ordinal, original and transformed source/AST hashes, output hash and execution count. Invert exact namespace
maps plus enumerated import/directive adaptations and require original AST equality. Add a negative proof check that changing a metric, coefficient, solver
tolerance, seed or budget fails. Complete each stub only from its own answer, not a different track's matching function name. Preserve all original Markdown
content and controls; normalize only named converter headings/answer placement/links, with a prose coverage ledger.

Expected exercise displays contain figures and inert HTML representations only, with counts `None`; no real historical stdout or execute-result answer is copied
into learner execution evidence. Solutions retain original counts/outputs explicitly labeled historical. Single PNG emission versus the toy's saved-only asset
and multi-output cells require separate figure-routing assertions. Python-only bootstrap/`MAIN` guards protect module import without disabling notebook
loading/training code. Preserve all default directories, empty-native-directory requirements, seeds, four-thread deterministic CPU protocols and optional CUDA
entry points; the migration selector itself uses one thread, offline mode and hidden CUDA.

## Forecast/reflection design and standard case proposal

Add paired, specific commitment prompts before each important result, followed by observations, control contrasts and changed-condition forecasts. Require
numeric intervals/signs/rankings, confidence, causal mechanism and prior exposure; never fabricate learner responses.

- Toy: predict main-effect ranking versus planted pair recovery; compare empty/revealing-context derivatives for a triple; forecast sampling-error bands and
  permutation-control behavior before changing budget or dividend.
- Allocation: forecast singleton/pair allocation for a unit high-order dividend; distinguish efficiency from kernel-weighted and unweighted error; predict
  ridge/shuffled-game controls; commit explanation-derived pair orders before audit capture; forecast effects of changing donors/background/context while
  recognizing a changed estimand.
- Estimators: forecast the signs from one queried coalition, per-set sampling probabilities, duplicate/unique query counts, SE scaling and additive-game
  finite-sample noise; distinguish boundary residual from equality constraints; predict rank deficiency and compare all estimator/proposal/budget curves without
  selecting favorable audit outcomes.

Propose one strict standard `validate_case` candidate only after implementation GO; keep additional detailed forecasts in a separate queue and the lesson. A
clean primary is **numeric order-two Faith-Shap coefficient of pair `(0,1)` in the original four-player unit pure-four game**, with no answer prefill. Freeze
all sixteen coalition values by the original literal Möbius game definition, separate `v(empty)` intercept, original pair mask/order, exact equality-constrained
interior-kernel objective, float64 solver definition/tolerances and exact independent closed-form/KKT references. Denominator: one exhaustive finite game and
one selected coefficient; six symmetric pairs/sixteen coalitions are not stochastic replicates, source rows or training examples. Controls retain
constant/additive zero-pair, signed/degree-two agreement, and distinct STII/pair-SII targets; do not label the latter identical estimands.

Use the exact standard case key set from `mastery_hub.units.extension_predictions.lesson.validate_case`: section/title/source paths/question; numeric prediction
with empty `forecast_items`; complete string-valued `config_fields`; controls/methods; evidence class/role/limitations/status; experiment/observation prompt;
binary transfer. Set `evidence_class=analytic_toy`, `data_role=historical` and `status=guided_practice`: this is an exact finite-game analytic exercise with an
exposed historical allocation ledger, not an empirical native case. Never claim a fresh native result. All responses/observations/answers remain absent from the
standard case rather than added as unsupported schema keys.

Binary transfer can change only the nonzero dividend from the four-way set to the original pure-triple set, retaining four players/order two/pair `(0,1)`, and
ask whether that pair coefficient increases. This variant and the original are already exposed in historical rational-ledger teaching; label it guided
historical transfer, not unseen mastery. Do not prefill either numeric or boolean answer. Observation receipts name the exact allocation-ledger figure/source
hash and explain exact rational versus visual precision. This analytic case does not replace the full native and estimator curricula or their forecast queues.
Root owns adoption into the shared workbook.

## Remaining boundary

This is a plan, not candidate acceptance. No files outside this note were written. The 16.11 frozen packet and its separate proposal addendum remain untouched.
Original ARENA preservation, current source hashes and later candidate tests are necessary but do not establish current external-cache availability, full
reproduction or human acceptance. After GO, finish the complete reviewable three-track candidate, freeze it, request independent review and preserve any failed
checks before asking root to adopt it.

## Implemented candidate, pending independent review

Root authorized the complete candidate described above. The canonical source is
`infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_5.py`; generated teaching artifacts live in `part5_shapley_interactions` and
`05_[16.5]_Shapley_Interactions.{md,py}`. The standard extension converter now produces 376 exercise cells and 353 solution cells, each with all 90 original
code cells. The additional prose consists of specific forecast/reflection pairs, historical execution boundaries and explicit cross-protocol comparisons. All 79
actual solution output records, original execution counts and code-cell metadata are preserved exactly. Exercises have null counts and exactly 22 expected
figure displays with inert HTML-object representations. Original notebook and every cell's metadata/ID are retained in namespaced provenance; no source widgets
were present.

The original 141-file universe was rechecked and sealed before master/canonical writes: 113 flat members plus 28 exact-path/hash external references. The sealed
archive is 11,916,958 bytes. An initially overbroad local 5MiB archive guard stopped the first attempt. Root clarified that the established strict review-bundle
validator already permits up to 64MiB compressed / 128MiB expanded; no exception or partition was introduced. `layout_fork.json` preserves that guard stop and
resolution. All generated notebooks, module and page remain below 5MiB.

Only named presentation/import adaptations were made: lexical colliding-global maps; explicit qualification of the toy's historical test module; Python-only
future directive hoisting, root/exercises-root bootstrap and display import; import-safe `MAIN` guards; historical heading normalization; numeric
difficulty/importance rendered as equivalent icons; split compound Markdown before Exercise headings; and regenerated own-answer dropdown placement. Notebook
future directives, all scientific code/defaults/public field names, original helper identities and source paths remain unchanged. The inverse AST proof covers
every code cell and rejects both an added scientific statement and an actual 800-to-801 training-budget mutation. No original/shared converter file was changed.

The first focused selector preserved a real local rendering failure: commented blank lines in multiline stub docstrings prevented the unchanged shared converter
from uncommenting those stubs. The local migration generator now emits truly blank lines for blank stub lines. An independent focused regression passes such a
docstring through `Cell.process_inline_filters` and asserts a live function, raise and intact docstring. Static scanning found zero susceptible blocks across
all 38 existing configured masters; no other section was regenerated. A separate overconstrained negative-fixture lookup was corrected to the actual
`TRAINING_STEPS = 800` cell. The initial log records 6 passed / 2 failed in 1.21 seconds; the repaired full suite records 8 passed in 1.11 seconds, followed by
**9 passed in 1.38 seconds** with the new narrow converter regression. These are static/source/format checks, not scientific executions.

`proposed_case.json` passes the standard `validate_case` schema with `evidence_class=analytic_toy`, historical guided-practice classification, numeric pure-four
Faith coefficient target and binary pure-triple transfer. It contains no learner forecast, observation or answer. All other native/estimator forecasts and
controls remain in the full lesson. No shared workbook, registry or config was edited. No method, model, data/cache loader, native computation, training,
package estimator, GPU or download was executed. The candidate requires independent review before adoption.
