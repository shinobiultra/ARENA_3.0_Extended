# Draft M2–M3 shared empirical protocol — 2026-11-03

**Candidate for human review; source-only work. No launch, adoption, learner-readiness or release claim.** Owner decisions: [#32](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/32), [#23](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/23), Q001 and Q002. Full #39 remains required before v1.0; this proposal does not reduce its 205 requirements or substitute one organism for the whole extension.

The current learner failure is the missing empirical bridge: M1's quadratic is correctly labelled `analytic_toy`, but cannot show whether its optimizer intuition transfers to an actual trained language model. The concrete target is an original, small natural-text next-byte model reused for basics/scaling, optimizer comparisons, landscapes, depth/stability and capacity. Negative results and failed runs are publishable observations, not a reason to retune indefinitely. This is not a replication of Stanford's organism or assignment answers. Generalization/exotic upstream content is not invented; M3 generalization needs a separately reviewed source-grounded protocol.

## Authority, provenance and open decisions

This draft used the local design/plan, `docs/mastery/record_schema.md`, M1 exercise/solution source, original ARENA 1.1's opening, `full_completion_frontier/FRONTIER.md` and captured issue #23/#32 bodies. Original ARENA, canonical lessons, catalog/readiness, lockfiles and pinned foundations are untouched. Neither model weights nor cached models, corpus contents, protected targets, or Stanford handout/answer contents were accessed. Only public upstream **dataset metadata** was consulted online; there was no dataset/weight download, training, inference, tensor-library import or GPU/cluster use.

| Stable item | Status and next boundary |
|---|---|
| V1-01 | Settled: full #39 before v1.0, no partial teaching release. |
| Q001 | Direct human walkthrough reported a usability failure: “the UI is too spread out and not very intuitive. try harder with the UI”. Compact-UI repair and a fresh human walkthrough remain prerequisites; Q001 is not passed. Source tests cannot answer this. |
| Q007 / #35 | Settled: supersede obsolete exact Claude review identities with latest available tiers. Preserve failed-auth history; independent review remains separate from human acceptance. |
| M2-R01 / #23 | Pending human decision on the exact proposed new local envelope below. No historical allowance is reused. |
| M2-D01 | Pending adoption of natural-text organism, source revision, rights/provenance and leakage audit. Controlled synthetic fixtures are engineering checks only. |
| M2-P01 / Q002 | Pending final exact protocol/manifest and explicit launch approval; depends on Q001, #23, data custody and reviewed numerics. |
| M2-N01 | Pending independent technical review of numerical parity/precision, optimizer equations, parameter/FLOP counting and comparison design. No empirical acceptance tolerances are invented here. |
| M2-L01 | Pending focused unit source designs, evidence envelope/replay design, independent learner review and direct human acceptance after actual evidence exists. |

Recommend accepting **the shape for further source preparation**, then resolving M2-D01/N01 and Q001 before any Q002 launch request. An affirmative resource decision alone must not be interpreted as authorization to access TEST, download data or start training. The final launch record must identify the reviewed artifact hash, complete run IDs, data/model/code/environment identities and actual human grant. Agent-authored JSON is not a grant.

## Recommended organism and data candidate

Use one decoder-only transformer from scratch, implemented visibly in a new companion notebook. No pretrained checkpoint, tokenizer fit, external model code or copied course solution is necessary. The proposed dataset is **Salesforce/WikiText, `wikitext-2-raw-v1`**. The primary dataset card identifies that configuration, English text-generation scope and CC-BY-SA-3.0/GFDL metadata. These are source facts, not a claim that local bytes have been verified. [Primary metadata, fixed README revision](https://huggingface.co/datasets/Salesforce/wikitext/blob/288b475496d0b1896dd57db862fcda46a0d26fd2/README.md).

The README commit above is a **metadata citation only**, not the adopted data revision. Dataset RELEASE commit, three exact shard paths/byte hashes, article reconstruction version, source/redirect-family metadata, attribution/license files, local availability and complete split manifest remain **UNSET: launch blocked**. No use of `main`, `latest`, dataset viewer examples, mutable cache aliases, or a guessed hash. Metadata shows a small dataset candidate; it supplies no measured device runtime or verified download allowance. Preserve attribution/license notices with any later authorized derived artifact; distribution of corpus/weights is a separate decision, not implied by local training.

An alternative controlled-language corpus could be written and trained as an additional diagnostic later, but it is **not** the sole M2 natural-language bridge. Today's guard tests contain only fictional hashes/IDs and exercise no LM, corpus generator or scientific target.

### Operationally disjoint data protocol, before fitting

1. A separately authorized data-custody stage pins and hashes the source files, reconstructs whole articles before tokenization, and assigns every article to its upstream TRAIN/validation/TEST role. Validation becomes DEV. No line/window random split; no concatenation between articles or splits. Article-title/redirect families are the independence groups. Their canonicalization and source-family metadata must be reviewed and frozen before use; missing identifiers or unresolved aliases are a hard stop.
2. Compute exact raw-document hashes, canonical-document hashes (UTF-8 decode; CRLF to LF only), article-family identity and every contiguous 128-byte span hash. Any cross-role source family, document or span overlap aborts. Do not move articles, silently drop duplicates or keep resampling until it passes. An amended cohort requires a separately reviewed protocol and new manifest. Shared byte vocabulary is intended; it is not an independent-observation claim. Hash checks cannot establish semantic independence or rule out undeclared near-duplicates; inspect the grouping procedure independently and state that limitation.
3. TRAIN alone feeds gradient updates and any empirical preprocessing. No learned tokenizer: encode UTF-8 bytes 0–255 plus BOS=256, PAD=257. No Unicode/whitespace cleanup, lowercasing or vocabulary selection based on DEV/TEST. Mask PAD and BOS targets; no next-article prediction targets. A sample is a contiguous 129-byte window within one article, yielding exactly 128 next-byte targets. Short articles are excluded by the predeclared length rule in **all roles**; report excluded identities/counts. Fail if any role cannot supply the required unique target windows.
4. TRAIN sampling: counter-keyed deterministic schedule `(seed, step, batch_slot)` over sorted permitted article/window IDs, uniform over eligible windows with replacement. Every condition in a seed shares this schedule; initialization and data-order RNG streams have distinct named seeds. DEV/TEST windows are fixed, sorted by SHA-256 of `(manifest hash, role, article ID, window start, cohort seed=20261103)`, selected greedily without overlapping 129-byte windows, until exactly 64 windows/8192 targets. TRAIN diagnostic cohort uses the same algorithm and size with cohort seed=20261104. No training example can share an article group with either evaluation role.
5. DEV is used for the declared learning-rate selection only. TEST windows/outcomes remain sealed from learner/implementer during development and tuning. An authorized custodian may compute integrity metadata before training without exposing content/outcomes. TEST access is a distinct final evaluation action after code, cohorts, predictions, candidates, failure handling and selections are frozen. Retain an exposure ledger; any prior access changes the untouched-TEST claim. A held-out width/budget condition is a separate prediction holdout, not a substitute for independent TEST data.

These checks define disjoint units operationally and fail closed. This draft has **not** certified that WikiText passes them. The source-only `validate_splits` checks declared metadata identities only; the real article/span audit, byte verification and sealed evaluator are unimplemented launch prerequisites.

## Exact finite architecture and recipe proposal

All numbers in this section are proposed design choices, not measured optima or adopted defaults.

- Context length 128; batch 16; gradient accumulation 1; exactly 2048 valid next-byte targets/update; no packed cross-article transitions. Base depth 4, width 128, four attention heads (head dimension width/4), MLP hidden width 4d, learned 128-position embeddings, tied 258-row input/output embeddings, causal full attention, GELU, dropout zero. Bias-free attention/MLP projections and output head. Pre-LayerNorm on attention and MLP, affine LayerNorm with epsilon 1e-5, residual coefficient 1, final LayerNorm. Width variants 64/128/192/256; all divisible by four. Depth variants 2/4/8. No rotary embeddings, tokenizer training or external weights.
- Initialization: embeddings/linears independent normal std 0.02, LayerNorm scales 1 and offsets 0; tied matrices initialized once. Exact parameter accounting for standard depth L, width d, MLP ratio r: `258d + 128d + L*((4+2r)*d*d + 4d) + 2d`; base `(L=4,d=128,r=4)` is **838,144** parameters. Maximum width256/depth4 is **3,249,152**. The depth8/width128 model is **1,626,624**. LayerNorm-removal ablation removes both block norms and final norm; residual-removal leaves other tensors unchanged; ratio2 changes only MLP hidden width. Verify formula against named tensors before training.
- Base optimizer AdamW: LR 3e-4, betas (0.9,0.95), epsilon 1e-8, decoupled decay 0.01 on projection/embedding matrices only (tied matrix once), norm affine parameters excluded, no clipping/scheduler/warmup. Explicit bias correction and decay/update ordering must be reference-checked before launch. Comparator: SGD with unnormalized momentum `v=0.9*v+g`, `theta=(1-lr*decay)*theta-lr*v`, initial v=0; same decay mask. Optimizer convention remains visible beside every plot.
- Seeds **11,23,37** are fixed *training replicates*, not population samples or a power calculation. Use the same seed set for every condition; display every seed and paired differences. Reserve 101 solely for an eight-update preflight. Nonfinite loss/gradient/parameter or an exception ends that condition immediately; retain last finite state, error and actual spend; no repair/retry/resume or seed replacement. Planned remaining work stays `not_started` if a global cap is reached.
- Proposed execution reference: existing locked environment, FP32 eager; exact Python/torch/CUDA/device/runtime and dirty-source hashes still UNSET until read-only metadata freeze. Do not change pins. Candidate BF16/compile/max-autotune requires its own bounded parity/performance comparison including compile and warmup cost and a human-reviewed delta; it is not auto-enabled on this tiny run. No current import/device/cache inspection is authorized by this draft.

The parameter formula is an architectural derivation; it is not an inference-time memory guarantee. FLOPs must be computed from the exact shapes/operators including attention's context term, backward, tied embeddings and eval, and independently checked. Until that counter is implemented/reviewed, show parameters, updates, valid tokens and measured device time; do not call `6NT` exact compute or claim compute matching.

## Frozen run grid and reuse

A fit means one from-scratch initialization/optimizer trajectory. Reuse exact checkpoint prefixes where indicated; never silently change optimizer state or replay a training run to improve a result. All counts below include all seeds. No cross-method winner is chosen on TEST.

| IDs / purpose | Conditions, each seeds 11/23/37 | New fits | Updates/fit | Training targets |
|---|---|---:|---:|---:|
| A / scaling + shared base | width64/128/256, depth4, base AdamW; save prefixes128/256/512 | 9 | 512 | 9,437,184 |
| B / precommitted held-out condition | width192/depth4, base AdamW; held-out budget384 | 3 | 384 | 2,359,296 |
| C / AdamW retuning supplement | base width128/depth4; LR1e-4/1e-3; LR3e-4 prefix256 reused from A | 6 | 256 | 3,145,728 |
| D / SGD fixed + retuned | base architecture, momentum0.9; fixed LR3e-4 plus retuning LR0.01/0.03/0.1 | 12 | 256 | 6,291,456 |
| E / M3 fixed-recipe | width128, depth2/depth8/no-LayerNorm/no-residual/MLP-ratio2, LR3e-4; base reused A | 15 | 512 | 15,728,640 |
| F / M3 retuning supplement | same five E variants, LR1e-4/1e-3; LR3e-4 prefix256 reused E; base reused A/C | 30 | 256 | 15,728,640 |
| G / failure controls | base architecture: zero LR and target-byte permutation using fixed seed73 each batch, otherwise base AdamW | 6 | 256 | 3,145,728 |
| P / preflight | base architecture, seed101, no reuse as selected scientific evidence | 1 | 8 | 16,384 |
| **Total maximum** | **81 scientific fits plus one preflight** | **82** | **27,272 aggregate** | **55,853,056** |

The target-byte permutation is over the flattened 2048 target positions, with the input byte sequences unchanged; preserve marginal bytes, break input-target alignment, record RNG derivation. A noise/shuffle control need not visibly fail if the real model has learned only unigram frequency: that result falsifies the stronger contextual-learning explanation. A zero-update control must leave parameters unchanged by exact construction; weight decay is disabled in that control and explicitly accounted as its defining difference. Controls do not receive extra tuning.

**Fixed recipe:** compare A/E under exactly base LR/optimizer/batch/data schedule and report tokens, parameter count and nonmatched compute. Show A prefixes128/256/512 versus B384 only after the learner has predicted B. Precommit a simple interpolant of DEV loss in log(width), log(tokens) on A prefixes128/256 only; predict the prefix512 budget extrapolation and width192 interpolation separately, with residuals for every seed. Do not use B or prefix512 to select the interpolation form. This is a diagnostic interpolation exercise, not a universal scaling-law fit.

**Equal-effort retuning:** exactly six AdamW architectures (base width128/depth4 and the five E variants) use three candidates `(1e-4,3e-4,1e-3)` × three seeds × 256 updates, including charged reused prefixes. Widths64/256 in A and width192 in B are fixed-recipe scaling conditions only and are not retuned. SGD uses exactly three candidates `(0.01,0.03,0.1)` × three seeds × 256 updates; the fixed LR3e-4 diagnostic is separately charged, never an extra candidate. Select one LR per method/architecture by mean final DEV next-byte NLL over its three seeds; ties choose smaller LR. A candidate with any failed seed is ineligible, but all failures remain visible; if none are eligible, report no selected method. No new retraining of a winner. Reuse is labelled both physical cost (once) and each comparison's logical allowance. Report equal candidate/seed/update/token effort; this is **not equal FLOPs or equal walltime** across widths/depths, and it is not a best-possible-method claim. A future compute-matched retuning study requires a new accepted cap/design.

## Evaluation, landscape and proposed resource envelope

Metric: mean next-byte negative log likelihood in nats over explicitly counted valid targets; optional bits/byte = NLL/log(2). No word perplexity claims. TRAIN-probe/DEV curves use fixed cohorts at step0 and every128 updates; 512 fits have five checkpoints, 384 fits four, 256 fits three. Across A–G this is 294 checkpoints × two cohorts = **588 forward evaluation calls ×8192 targets**. The eight-update P run gets one DEV call only. TEST uses exactly 99 declared fit/checkpoint calls after global freeze: the final checkpoint for every A–G fit (81 calls, including controls), plus step256 for the three A width128 fits and all fifteen E fits (18 calls). These extra prefix calls supply the equal-256-update retuning comparison; their step512 outputs belong only to the separately identified fixed-recipe/longer-budget comparisons. Every call uses the same8192-target TEST cohort; no exploratory stepwise TEST peeking or choosing a checkpoint on TEST. For a failed trajectory report each unavailable declared TEST checkpoint as missing; if prefix256 exists but final512 does not, retain that distinction. Never substitute the last available checkpoint, impute a finite score or exclude failures from the attempted denominator.

The exact TEST set is expressed below without relying on a mutable “final” alias. Each brace product expands over seeds `{11,23,37}`; the fit table fixes architectures/optimizer conventions for every ID. Names are protocol identifiers, not existing checkpoint files.

```text
A = A-w{64,128,256}-s{11,23,37}                            end=512  (9)
B = B-w192-s{11,23,37}                                   end=384  (3)
C = C-lr{1e-4,1e-3}-s{11,23,37}                          end=256  (6)
D = D-lr{3e-4,0.01,0.03,0.1}-s{11,23,37}                 end=256 (12)
E = E-{depth2,depth8,noLN,noResidual,mlp2}-s{11,23,37}      end=512 (15)
F = F-{depth2,depth8,noLN,noResidual,mlp2}-lr{1e-4,1e-3}
      -s{11,23,37}                                      end=256 (30)
G = G-{zeroLR,permutedTargets}-s{11,23,37}                 end=256  (6)
TEST tuples = {(fit_id, end, TEST, cohort_sha256, 8192) for fit_id in A..G}
            union {(A-w128-s{11,23,37}, 256, TEST, cohort_sha256, 8192)}
            union {(E-{depth2,depth8,noLN,noResidual,mlp2}-s{11,23,37},
                    256, TEST, cohort_sha256, 8192)}
# 81 + 3 + 15 = 99 distinct fit/checkpoint tuples; cohort_sha256 remains UNSET.
```

The final reviewed run settings must declare the exact evaluation checkpoints by role and cohort hash, alongside the **exact checkpoint-byte hash and recorded optimizer/data-order step** for every selected comparison. Comparing retuned settings requires step256 for all six AdamW architectures and SGD, never a step512 base substituted against step256 alternatives. Store a comparison as `(fit_id, checkpoint_step, checkpoint_sha256, cohort_sha256, metric_version)` in its evidence and cache selection. Global protocol/run-settings hashing detects changed checkpoint declarations, but the current metadata helper **does not validate checkpoint bytes, checkpoint availability, step identity or evidence-envelope checkpoint tuples**. Those checks and the model-specific schema are implementation/launch prerequisites; a cache hit alone cannot establish them.

Landscapes reuse A's width128/step256 checkpoint for each seed: two independently seeded orthonormal parameter directions (seed401) and their 45-degree rotation, 9×9 coordinate grid per plane, radii `(-1,-.75,-.5,-.25,0,.25,.5,.75,1)*0.01*||theta||`. Evaluate on the fixed **4096-target prefix of the DEV cohort**. Add five positions per direction at radii `(-1,-.5,0,.5,1)*0.001*||theta||` for central directional finite differences: `(162+10)*3=516` forward calls, 2,113,536 targets. Directions and rotations are fixed before outcomes; no hunt for a prettier plane. Directional differences are descriptive; finite-difference numerical adequacy and radius interpretation require M2-N01 review. Layerwise activation/gradient/update/weight norms are captured from existing steps without new gradient updates. No protected TEST loss slice.

Maximum proposed totals are **1,204 evaluation calls, 7,749,632 evaluation targets** (588 TRAIN/DEV curve calls +1 preflight DEV call +99 TEST calls +516 landscape calls), plus 55,853,056 training targets. Failed/not-started calls reduce actual counts but do not erase reservations. Save step0/128/256/512 checkpoints for 512-update fits and step384 for B; other fits retain initial/final states and full scalar traces, enough to reuse declared prefixes. Every checkpoint gets content hash and exact model/optimizer/data-order step identity.

**This is a proposed new ceiling, not allocated compute and not a measured runtime estimate:**

| Resource | Proposed aggregate hard cap for this entire grid, controls, preflight, compilation, evaluation and failures |
|---|---|
| Hardware route | Existing local RTX 5090 Laptop, physical24GB; at most one GPU process; no clusters/jobs/paid services |
| GPU allocation/time | One device, peak20GiB allocated VRAM; 3,600 device-active seconds total |
| Elapsed walltime | 7,200 seconds from first preparation/authorized corpus access to final cleanup; no timer reset by stage/restart |
| CPU | At most24 threads, peak32GiB process-tree RSS on host with94GiB physical RAM; 86,400 CPU core-seconds aggregate |
| Training | 82 fits maximum, 27,272 optimizer-update attempts, 55,853,056 valid training targets maximum |
| Evaluation | 1,204 forward cohort calls, 7,749,632 valid evaluation targets maximum; no open-ended generation |
| Local writes | 4GiB cumulative bytes written including temporary/failure artifacts; 4GiB retained ceiling |
| Dataset transfer | At most32MiB new source-data bytes **only if separately authorized**, exact pinned shards/license files only; zero weights/model transfers |
| Retry | Zero; first unsafe/nonfinite condition stops that run, exhausted/global unsafe cap cancels the remainder; no automatic resume |
| Shutdown | First limit wins; stop further dispatch immediately, cooperative cancel then bounded termination. Reserve120 wall seconds **inside**7200 for stop/receipt preservation; never spend cleanup as training. |

Fit-specific reservations must be bound in the final manifest such that their complete sums fit this envelope. Runtime/VRAM cannot be guaranteed from architecture alone. If preflight indicates the full grid cannot fit, stop and report predicted incompleteness with measured evidence; do not shrink the model, discard seeds, increase caps, change precision or launch a new grid without a reviewed amendment. CPU-only synthetic reference checks remain a verification path; CPU training of the entire empirical grid is not silently substituted for the GPU request. Historical SSD/SAGE/Shapley/ACDC/MiniVLM/setup-only budgets remain separate and consumed exactly as recorded; no unused fragments fund this grid.

## Learner design and source implementation boundary

| Companion | Student-visible methods / immediate tests | Signature result and play |
|---|---|---|
| Basics/scaling | UTF-8 byte/window target alignment; valid-token accounting; mean masked NLL; freeze/aggregate a seed sweep; paired comparisons and interpolation residuals. Tests: hand-count targets, known logits, no-cross-article masks, independent 3-point interpolation fixture. | Every-seed loss versus steps **and** valid tokens, parameters and measured compute/time; deliberately mixed-budget panel repaired beside matched-token comparison. Predict B before reveal; replay seed/width/checkpoint choices, inspect failed runs, explain which extrapolation failed. |
| Optimization bridge | Explicit SGD-momentum and AdamW equations; norm/weight update accounting; fixed-vs-retuned selection ledger. Tests: hand-calculated two-update scalar states, independent reference parity before native work, full grid accounting. | Paired seed-level loss/update-to-weight curves, zero-update and shuffled-target controls, separate fixed/retuned panels with all candidates/failures. Play cell modifies a proposed LR in a **draft config**; it invalidates commitment but starts no training. |
| Landscapes | Parameter flatten/unflatten, normalized orthogonal directions, directional loss grid and finite difference; identity/rotation/norm tests on a small non-target quadratic. | Two rotated slices plus directional-curvature profiles from the actual accepted LM checkpoint, stating projection dependence. Replay plane/radius selectors read only declared stored measurements; out-of-grid request displays missing evidence. |
| M3 stability/capacity | Named ablation checks, per-layer propagation norms and parameter/token/FLOP accounting; fixed-vs-retuned separation. | Depth/norm/residual panels and all-seed matched-token capacity curves; same recipe and retuned results remain distinct. Counterexample asks whether a width advantage survives explicit compute accounting. |

Expected engineering output is an explicit fixture pass or a rejection with reason. **No expected scientific winner or numeric loss threshold is supplied.** Show actual controls, failures and nulls once legitimately measured; do not synthesize curves to fill these designs. Retuning on DEV precludes calling DEV independent generalization evidence. Three seeds show repeatability on this finite setup; do not imply a powered population test. Report paired seed points and ranges; any inferential interval/claim requires a justified independence/resampling plan first.

Future learner files (not authored by this draft): focused designs and then `mastery_hub/units/u01_basics/lesson.py`, the separately identified empirical bridge in `u02_optimization/`, `u03_landscapes/lesson.py`, `u04_stability/lesson.py`, `u05_capacity/lesson.py` and generated practice/solution notebooks. Each needs the full orient/predict/explore/implement/verify/falsify/transfer/reflect arc, expandable help/solutions, immediate function tests and keyboard-readable plot tables. No launch through sliders or notebook imports. Readable student code owns the core training/optimizer/comparison/slicing methods; helpers own only data custody, records, cache validation and display plumbing.

The existing M1 portable schema accepts **only** `analytic_toy`. This proposal does not widen it. Before UI integration, design a separate strict empirical evidence envelope carrying exact protocol/run/cohort/checkpoint/source/environment hashes, predictions, exposure history, full failed-run ledger and measured resource receipts. `recorded_real_run` must mean validated stored real measurements; replay changes visualization only. Fresh execution requires explicit Run with the displayed, approved immutable config and cancel support; neither agent fields nor a replay button supplies launch permission.

## What the new source-only guards actually establish

`mastery_hub/empirical_protocol.py` supplies immutable metadata value objects, finite strict canonical JSON with unknown-field rejection for protocol/run/budget schemas, canonical protocol/cache identity, declared group/source/content split-overlap rejection, cumulative whole-run reservations without refunds, and fixed/retuned pair checks with type-preserving canonical JSON equality. Both tuning grids and selected learning rates must be positive finite numbers, excluding booleans; boolean model factors can still be deliberately varied. Tests use fictional three-document hashes and tiny integer budgets, never data/model access.

A cache key includes **the entire** source/model/environment/split identities, run list/settings/seeds and caps, including declared evaluation-checkpoint lists when present in settings. It does not inspect any checkpoint file. Any change invalidates it conservatively. Run settings must include the architecture, optimizer convention, full training/eval configuration and provenance for the eventual reviewed protocol. A reservation is charged before dispatch and stays charged after failure; an unknown run, stale ledger, or any over-budget axis is rejected without partially modifying the ledger. The caller must preserve immutable snapshots durably; this pure module has no concurrent or filesystem journal. Its cap logic is **not a process watchdog**, proof of actual resource spend, a sealed evaluation system or a security boundary. The pair guard enforces matched steps/tokens/eval effort and declared settings only; it does not prove complete search-ledger fairness or compute matching. Unknown settings require a future model-specific schema; arbitrary finite settings JSON is fingerprinted here, not scientifically validated.

Current source tests do not establish native parity, GPU feasibility, data leakage absence, learner success, M2/M3 completion or permission. Full data/source/environment identities remain UNSET, so this public candidate is intentionally not an executable frozen launch manifest.

## Verification and next concrete review

Changed files for this increment: this proposal, `mastery_hub/empirical_protocol.py`, `tests/mastery/test_empirical_protocol.py`, and local receipts under `.scratch/studiable-20261102/v1-quality/m2_protocol/`. No original/canonical notebook or lesson was modified.

Focused source-only check: `UV_CACHE_DIR=/tmp/arena-m2-uv-cache uv run --offline --no-sync pytest -q tests/mastery/test_empirical_protocol.py`. The cache override avoids writing the read-only default uv cache and installs nothing. Receipt records actual outcome. Broader tests, native execution, training, GPU, downloads and preservation-wide audit are not claimed by this increment.

Human review is of the exact organism/data/split/grid/resource proposal above. Technical review must first resolve M2-D01 and M2-N01, followed by the real Q001 walkthrough and explicit Q002 launch decision. Null/failed evidence is acceptable; lowering the scientific scope or dropping required human acceptance to call this v1.0 is not.

### Review repair receipt — 2026-11-03

The first draft omitted the18 TEST prefix256 evaluations needed by the reused A/E retuning candidates. This draft preserves both the81 final-checkpoint reports and those18 prefix reports as99 exact TEST tuples, raising the aggregate from1186 to1204 evaluation calls and from7,602,176 to7,749,632 evaluation targets. These remain proposed caps; no scientific evaluation ran. The grid-metadata test independently enumerates82 fits,27,272 updates,588 curve calls and99 distinct TEST fit/checkpoint tuples.

Independent review also identified Python boolean/numeric equality hiding undeclared settings changes and malformed right-hand retuning grids. Regression tests retain those counterexamples; guards now validate both sides and compare canonical JSON after the explicitly varied factor/selected LR are removed. Pre-repair source/proposal/test bytes and hashes remain in the owned scratch directory. The revised snapshot still needs independent re-review and the previously listed human/data/numerical gates.
