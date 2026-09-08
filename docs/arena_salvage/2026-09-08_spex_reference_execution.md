# SPEX / ProxySPEX: executed primary reference on analytic games

September 8, 2026. This completes the isolated **analytic reference preflight**
requested after the September 7 research checkpoint. It does not teach or close
`shapley.spex_proxyspex`, freeze an LLM study, or establish native-model quality.
No language model was queried.

The actual author SPEX implementation recovers the tested sparse signed spectra
and fails on useful dense-spectrum / degree controls. The actual author ProxySPEX
implementation fits genuine LightGBM ensembles whose extracted spectra reproduce
the trees accurately, but its final strict truncation rule materially degrades two
games and crashes on pure parity. These outcomes are retained without patching the
author checkout or installed algorithm source.

## Scope and ownership

- Research scripts, isolated environment and all raw evidence:
  `/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/`.
- Repository change owned by this subtask: this report only.
- Original ARENA notebooks, pages and assets: **no changes by this subtask**.
  A repository-wide preservation result must come from the parent task's current
  preservation audit, not this statement about scoped edits.
- Shared `pyproject.toml`, `uv.lock`, lesson notebooks/pages/answers, registry and
  learner receipts: no changes by this subtask.
- Student functions, lesson signature result and play cells: not authored here.
  The evidence below is a prerequisite for designing them, not a substitute.

## Reproduction and provenance

The existing isolated environment was inspected, reused and checked with
`uv sync --frozen` (34 packages checked). Its lockfile binds the installed versions.
The author package is built from the acquired local source, not an editable install.

```bash
cd /home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1
uv sync --frozen
.venv/bin/python run_reference.py conventions > artifacts_conventions.log 2>&1
.venv/bin/python run_reference.py spex > artifacts_spex.log 2>&1
.venv/bin/python run_reference.py proxy > artifacts_proxy.log 2>&1
.venv/bin/python audit_mechanisms.py > artifacts_mechanisms.log 2>&1
.venv/bin/python make_manifest.py
```

Both scripts set `CUDA_VISIBLE_DEVICES=''`, numerical thread limits of four, and
process affinity to four CPUs before importing numerical libraries. The fitted
LightGBM boosters report `num_threads=4`. Runs were sequential. The model API
itself uses `n_jobs=-1`; the affinity restriction is therefore material.

| Component | Version / identity |
| --- | --- |
| Python | 3.12.12 |
| Author `spectralexplain` | 0.0.4, commit `61400b5388b8ecbf18fe18c4d821ffe3e8a5f9c8` |
| `sparse-transform` | 0.2.1 |
| `galois` | 0.4.11 |
| LightGBM | 4.7.0 |
| NumPy / SciPy | 2.5.3 / 1.18.1 |
| scikit-learn / pandas | 1.9.0 / 3.0.5 |
| numba / llvmlite | 0.67.0 / 0.49.0 |
| `shapiq` | 1.5.2; installed for compatibility context, not the reference tested here |

`artifacts/environment.json` records all distribution versions, platform, source
hashes for every installed `spectralexplain` and `sparse_transform` Python file,
the clean author checkout status, author commit, scripts/environment input hashes,
and the hash of the prior `references/sources.json`. `artifact_manifest.json`
binds the scripts, lockfile, logs and individual evidence files. Author source and
both papers remain in the sibling `references/` directory identified by the
September 7 checkpoint.

Sources: [SPEX paper](https://proceedings.mlr.press/v267/kang25a.html),
[ProxySPEX paper](https://papers.neurips.cc/paper_files/paper/2025/hash/68dced1638d71d4d21598f79ae91262e-Abstract-Conference.html),
[pinned author source](https://github.com/basics-lab/spectral-explain/tree/61400b5388b8ecbf18fe18c4d821ffe3e8a5f9c8).

## Fourier signs, conversion and BCH checks

The convention is `chi_S(x) = (-1) ** sum(x[i] for i in S)`, with binary `1`
retaining a feature. Spectra are signed coefficients in this basis. Query IDs use
big-endian binary integers: feature 0 is the most significant bit. These are not
Shapley interaction coefficients until explicitly converted.

All **62 basis functions** across `n=1,...,5` were checked independently:

- Author forward Fourier transform and inverse: maximum error **0**.
- Author Fourier → Möbius → SII versus the factorial-weighted finite-difference
  definition, for every nonempty interaction: maximum error **1.34e-15**.
- Singleton SII/Shapley value agrees with
  `SV(i) = -2 * sum(F(S)/len(S) for S containing i with odd len(S))`:
  maximum error **1.45e-15**.

SPEX Table 1 omits the `-2` factor in its displayed Shapley-value expression.
Appendix C equation (46) includes it and agrees with these tests. Use equation
(46), not the table expression, for the learner conversion.

The installed BCH implementation uses the **systematic generator**, with identity
message columns followed by parity columns. For `n` features it takes the final
`n` generator rows, transposes their parity portion, and adds a zero delay row.
`parity_decode` supplies zeros in the message part of a shortened codeword, then
decodes the supplied parity observations. This is not a generic parity-check
syndrome decoder. In row-vector package notation, `G` is `k_c × n_c`; the paper's
column-vector convention transposes it.

All **92 cases** for `n=5,t=2` satisfying
`support_weight + parity_error_weight <= t` decode correctly. For `n=31,t=3`,
support `{1,3,7}` decodes noiselessly with three corrected message positions. One
deliberate low-confidence parity-sign error exceeds that hard-decoding bound:
hard decoding reports `-1`, while the actual author soft decoder with Chase depth
6 recovers the correct support. The returned successful Chase record has three
entries in this case; callers should not assume a fixed two-element tuple.

For a negative Fourier coefficient `-1.25`, the installed SISO operation
`real(U[1:] / U[0])` returns the correct parity signs. The package amplitude
estimate is the normalized signed correlation
`rho = conjugate(signature) @ U / len(signature)` and recovers `-1.25` with
residual norm **5.92e-16**. The package accepts the candidate using bin consistency
and a residual-energy fraction below `0.9`. It does not directly implement the
paper's displayed positive-correlation comparison for a negative coefficient.

The paper's Algorithm 4 variable-message formula is displayed as a sum, whereas
Appendix A.4.2 describes averaging. The installed `peel_average=True` implements
averaging. The code parameter search uses designed distance `2*t+1`; this avoids
the paper paragraph's confused `t_c = 2d+1` wording. These source differences must
be explained rather than silently presented as exact pseudocode equivalence.

Evidence: `artifacts/conventions.json`, `artifacts/mechanisms.json`.

## Feasible BCH designs and actual game queries

There are three random binary subsampling matrices, one repeat and a shared
systematic BCH delay matrix. Raw requests are `3 * (parity_bits+1) * 2**b`.

| Features | Degree parameter | BCH `(n_c,k_c)` | Delay rows | b=3 | b=4 | b=5 | b=6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 31 | 1 | (63,57) | 7 | 168 | 336 | 672 | 1,344 |
| 31 | 2 | (63,51) | 13 | 312 | 624 | 1,248 | 2,496 |
| 31 | 3 | (63,45) | 19 | 456 | 912 | 1,824 | 3,648 |
| 31 | 4 | (63,39) | 25 | 600 | 1,200 | 2,400 | 4,800 |
| 31 | 5 | (63,36) | 28 | 672 | 1,344 | 2,688 | 5,376 |

Additional dimensions `5,12,24,32,40`, each at `t=1,...,5`, are recorded in
`conventions.json`. These are design costs, not an approved native-study budget.

For the two tested seeds at `n=31,t=3`, requested/unique masks were respectively
`456/418`, `912/874`, and `1824/1786`. Both counts are preserved. Each source
sampling design calls the analytic oracle in three batches. No deduplication was
inserted into the reference algorithm. A native cached oracle would need to report
requested masks, unique masks and real model evaluations separately.

The public `spectralexplain.Explainer` SPEX API was also exercised on the parity
game with requested budget 456 and order 3. Its constructor first creates a
separate dummy zero-valued sampling design through `b=12` to determine thresholds,
which advances the global NumPy RNG before real sampling. That dummy stage does
not query the game. Exact public-API results and actual game request counts are
recorded under `public_spex_explainer` in `mechanisms.json`.

## SPEX recovery and failure controls

Twenty-six unmodified `sampling_strategy` + `support_recovery('soft')` runs used
`n=31`, seeds 0 and 1, and `b=3,4,5` for the three sparse games. Controls used
`b=3,5`. Audit masks were 4,096 independent Bernoulli masks from seed 7701.
They were evaluated only by the analytic truth, outside the recovery oracle.
No `2**31` coalition table was constructed.

| Analytic game | Ground truth | Observed recovery |
| --- | --- | --- |
| Additive | constant and six signed singleton Fourier coefficients | all seven supports, no sign errors; worst RMSE `1.25e-5`, other runs near machine precision |
| Parity | `-1.25 * chi_{1,3,7}` | exact signed recovery in all six runs, RMSE 0 |
| Sparse polynomial | `.3 + x0 - .7*x3 + 1.2*x1*x7 - 2.4*x2*x4*x9`; independently expanded into 13 Fourier terms | all 13 supports, no sign errors; worst RMSE `2.90e-5`; both b=5 runs near machine precision |
| Wrong degree | same third-degree parity with `t=1` | no recovered coefficient, RMSE 1.25 in all four runs |
| Dense spectrum | 256 random signed degree-three terms, fixed truth seed 884 | only 3–63 terms recovered; RMSE 0.666–1.021, demonstrating insufficient sparsity/budget |

Every saved design's transform was compared to the independent identity

`U_c[d,j] = sum(F(k) * (-1)**(d @ k) for k if M_c.T @ k % 2 == j)`.

Across all 26 designs the maximum error was **4.72e-16**. Exact truth-based bin
occupancies distinguish zero bins, singletons and collisions. The saved sparse
polynomial `seed=1,b=3` trace records **six message-passing iterations**.

There is a consequential source/API discrepancy: the author's
`support_recovery.py:47` passes `regress='lasso'`, but the installed transform
returns before its regression branch when `report=False` (the default). A runtime
observer confirms **zero regression calls** for the real author wrapper. A
separately labeled `report=True` diagnostic calls Lasso once and worsens this
example's RMSE from `2.90e-5` to `0.00360`. Do not assert that the author wrapper's
output includes final Lasso refinement or that turning refinement on necessarily
improves the result.

Evidence: `spex_results.json`; each `*_queries.npz` and `*_design.npz`; independent
audit masks; and the trace / regression probe in `mechanisms.json`.

## ProxySPEX: separate prediction, extraction and selection errors

Each game used the **actual public author Explainer**, 12 features, seed 0 and
1,024 sampled Bernoulli mask rows (901 distinct masks). The reference retained its
full grid: depths `{1,3}`, tree counts `{500,1000,5000}`, learning rates
`{.01,.1}`, five-fold R² cross-validation and RidgeCV over 100 alphas. A subclass
observed the completed `GridSearchCV.fit` result for stage receipts and did not
change its estimator, grid, data or returned result. Installed source was unchanged.

The independent audit enumerated all 4,096 analytic masks **after fitting**.
This exhaustive audit is small-game ground truth, not training queries and not a
scalable native-model result. Random sampling allows duplicate masks to cross CV
folds; consequently CV scores should not be interpreted as an independent mask
generalization audit. The exhaustive stage audit exposes the actual error.

| Game | Tree RMSE vs truth | Fourier extraction RMSE vs fitted tree | Author final RMSE / outcome |
| --- | --- | --- | --- |
| Additive | `1.20e-8` | `2.76e-14` | `0.51507` |
| Parity | `0.02322` | `4.16e-15` | crashes after empty support selection |
| Sparse polynomial | `0.001921` | `5.26e-15` | `0.30162` |

All three best models used 5,000 LightGBM trees and learning rate .1; additive
chose depth 1 and the other two chose depth 3. All three boosters report four
threads. No single-tree fallback was used. The tree-to-Fourier extraction was
therefore tested on real boosted ensembles, not a stand-in fixture.

The failure is in author `src/spectralexplain/utils.py:423–440`: it locates the
coefficient at the cumulative 95% energy boundary and then retains coefficients
**strictly larger** than that boundary. This drops the boundary coefficient and
all ties. Nonconstant energy retained was 92.409% (additive), 0% (parity), and
92.942% (polynomial), not at least 95%. The constant coefficient is excluded when
computing the energy threshold but is not explicitly retained in the output;
Ridge has `fit_intercept=False`, so dropping it introduces additional error.

The exact preserved parity failure is:

```text
ValueError: Dot product shape mismatch, (1024, 12) vs (0,)
```

Separate diagnostic alternatives reuse the same fitted spectra and original
training queries. They are explicitly **changed truncation/refinement behavior**,
not relabeled author results:

- Including the 95% boundary and always retaining the constant produces parity
  and polynomial Ridge RMSE around `1e-9`. The additive game still intentionally
  truncates two small terms; retained energy 97.800%, RMSE `0.22483`.
- Keeping all extracted coefficients and applying the same RidgeCV grid gives
  all three games RMSE below `2.1e-9`.

These diagnostics locate the error and provide test cases for an explicit learner
implementation. They do not approve a particular native-study estimator. In
particular, changing the truncation rule must be described as a correction or
variant, with the untouched reference failures still visible.

The installed `shapiq==1.5.2` source has different short-spectrum and threshold
handling. Its result must not be assumed identical to this author checkout;
this report does not claim to have executed its full approximation methods.

Evidence: `proxyspex_results.json`, per-game raw query arrays, `*_cv.json`, saved
`*_lightgbm.txt` boosters, `*_exhaustive_audit.npz`, and the separately labeled
`proxy_threshold_diagnostics.json`.

## Remaining decisions and honest completion boundary

An actual LLM task, feature grouping, scalar response, masking operation, contexts,
train/audit split, seeds, budgets, baselines and query-cache semantics still need
to be frozen before native evaluation. The 31-feature BCH designs are feasible
analytic preflight evidence only. Proxy depth/order is a meaningful limitation,
and pure parity is an explicit warning that tree fitting and coefficient selection
can fail for different reasons.

The lesson must expose the binary subtransform, systematic BCH support decoding,
confidence-based Chase trials, signed amplitude estimate, collision/message
passing, tree spectrum recursion, coefficient selection and chosen refinement.
Notebook exercises, immediate tests, controls, visible plots and editable play
cells remain required. Neither passing analytic checks nor this report closes
the joint requirement or establishes course/release readiness.

## Continuation: actual maintained `shapiq==1.5.2` execution

The earlier statement that `shapiq` had only been inspected describes the first
phase above. This continuation **executes its public ProxySPEX approximation** on
the same three analytic games. It supersedes that source-only status. No native
LLM outcome was evaluated and no shared environment or lesson input was changed.

```bash
cd /home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1
.venv/bin/python run_shapiq_reference.py > artifacts_shapiq.log 2>&1
.venv/bin/python shapiq_constant_probe.py
.venv/bin/python bch_reference_fixtures.py > artifacts_bch_fixtures.log 2>&1
.venv/bin/python make_manifest.py
```

`shapiq_1_5_2_source_snapshot.tar.gz` preserves every installed `shapiq` Python
source file. `shapiq_environment.json` records their individual hashes, the locked
environment, script and snapshot hashes, and CPU/GPU constraints. The observing
subclass records the input/output of the actual spectrum extraction and refinement
methods, returning their original result unchanged. It does not replace the
sampler, model, selection rule, RidgeCV or interaction conversion.

All three calls use `ProxySPEX(n=12, max_order=3, index='SII',
proxy_model='lightgbm', hpo=True, random_state=0).approximate(1024, game)`.
The method's actual behavior differs from the author wrapper:

- **Sampling:** it evaluates 1,024 distinct masks, prioritizing empty and full
  coalitions. At seed 0 these require 1,179 internal draws/counts; the callback
  receives one batch of 1,024 unique masks. The author's 1,024 sampled rows had
  only 901 unique masks. These are different training sets even at the same seed
  and nominal row budget. The maintained sampler's internal counts are retained,
  but ProxySPEX fits each distinct row once without those counts as fit weights.
- **Search:** its default grid has eight candidates: depths `{3,5}`, `max_iter`
  `{500,1000}`, and learning rates `{.01,.1}`, with five-fold R² CV. This grid is
  independent of requested interaction `max_order=3`; the parity fit uses depth 5.
- **Tree-count alias:** the fitted scikit-learn estimator still exposes
  `n_estimators=100`, but LightGBM's `max_iter` alias sets the actual booster to
  **1,000 iterations/trees** for all three best models. Actual booster fields and
  saved models establish the tree count; inspecting `n_estimators` alone would
  misreport it. All boosters report four threads.
- **Selection:** `_refine` moves the threshold index one entry past the 95%
  cumulative-energy boundary, then still uses a strict `>` comparison. This
  retains the boundary when neighboring magnitudes differ. It is not a generally
  tie-safe inclusive rule. Spectra with at most `n` entries bypass both selection
  and regression entirely.

The independent post-fit audit again enumerates all 4,096 analytic masks. SII
errors below compare nonempty returned interactions through order 3 to the exact
ground-truth conversion already checked against finite differences.

| Game | Tree RMSE | Extraction RMSE vs tree | Final Fourier RMSE | Retained nonconstant energy | Maximum nonempty SII error |
| --- | --- | --- | --- | --- | --- |
| Additive | `0.00084335` | `3.00e-15` | `0.2236969` | 97.8003% | `0.4` |
| Parity | `0.1031905` | `2.46e-15` | `1.22e-9` | 99.3320% | `9.77e-9` |
| Sparse polynomial | `0.0112908` | `2.26e-15` | `1.21e-9` | 99.9911% | `2.58e-9` |

The best depth is 3 for additive and polynomial and 5 for parity; all best rates
are .1. The final support contains respectively 5, 1 and 13 coefficients. The
constant survives in additive and polynomial; parity's true constant is zero.

Thus the maintained implementation resolves the observed author parity crash
and polynomial error on these games. The additive residual is principally the
intentional removal of the true `+0.2` and `-0.1` singleton Fourier coefficients.
Even 97.8% retained spectral energy can omit a true Shapley value of magnitude
0.4 under the signed Fourier-to-SV conversion. Energy retention and interaction
accuracy should therefore be shown separately.

To separate the changed cutoff from changed sampling/tree fitting, the maintained
`_refine` method was also called directly on the **previously saved author full
spectra and original author query rows**. RMSE becomes `0.2248346` for additive,
`1.22e-9` for parity, and `1.14e-9` for polynomial. This independently confirms that
the cutoff/refinement change resolves the specific author failures; it is not an
inference from different end-to-end training sets.

### Remaining maintained-refinement failures

Direct execution of the untouched `_refine` method retains the following edge
cases, separately labeled from the three successful full approximation calls:

| Exact input spectrum | Observed behavior |
| --- | --- |
| n=12, seven-entry exact additive spectrum | returns unchanged because entry count is at most n; no Ridge or truncation |
| n=3, all seven nonempty supports with coefficient 1, no constant | `IndexError: index 7 is out of bounds for axis 0 with size 7` after shifting past the final threshold entry |
| n=8, all 28 pair supports with coefficient 1, constant .01 | threshold remains 1 because of ties; strict `>` retains nothing, then NumPy matrix multiplication fails on empty support |
| n=3, constant 3 and seven nonconstant coefficients 10,9,...,4 | constant is not protected; threshold 4 drops both the constant and final nonconstant term. Truncated RMSE is 5; final RidgeCV RMSE is 6.2694 |

These outcomes rule out calling the maintained rule universally inclusive or
safe for ties, empty selection and nonzero baselines. A lesson correction must
retain an explicit reference-versus-correction distinction. The general
fourier-spectrum and exact analytic controls should remain available even when
the selected native study happens not to trigger these cases.

Receipts: `shapiq_proxyspex_results.json`, per-game `*_sampler.npz`, raw query
arrays, CV results, boosters and exhaustive audits; `shapiq_cutoff_probes.json`;
`shapiq_constant_probe.json`. One earlier instrumentation attempt stopped after
the first fit because the harness read the sampler property as `coalition_sizes`
instead of `coalitions_size`. Its log is retained as
`artifacts_shapiq_attempt1_harness_error.log`; it is a harness error, not an
algorithm failure. The corrected three full runs all completed.

### 28-feature costs and independent BCH fixtures for implementation parity

Actual author zero-oracle designs for `n=28,t=3,seed=0` use BCH(63,45), shortening
17, and a `19 × 28` delay matrix. At `b=3,4,5`, raw/unique masks are
`456/418`, `912/874`, and `1824/1786`. Matrices and complete ordered mask/query-ID
arrays are stored in `bch_n28_t3_b*_seed0_design.npz` and indexed by
`bch_n28_designs.json`. Constructing these designs performs no model evaluations.

A request from the methods implementation agent led to **70 independent author
generator/delay fixtures**: 14 feature dimensions
`{1,2,5,12,28,31,32,63,64,127,128,255,256,511}` × requested `t=1,...,5`.
This is representative coverage through 511, not exhaustive coverage of every
dimension. Full `G` and `D` arrays are saved as uint8, with raw row-major SHA256
hashes, code parameters, shortening, generator polynomials and field identifiers.
The fixture index includes 132 successful author-decoder cases: the earlier 92
exhaustive small cases and representative cases at n=28 and n=31.

This check caught a concrete compatibility error in a proposed generic finite
field table. The actual `BCH.extension_field.irreducible_poly` integers are:

| Extension degree m | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Author BCH field polynomial | 7 | 11 | 19 | 37 | **67** | **137** | 285 | 529 | **1033** |

The generic proposed values 91, 131 and 1135 at degrees 6, 7 and 10 do not match.
For example, the actual n=28,t=3 author code uses `x^6+x+1` (67). Codebook parity
must use the actual BCH field polynomial, not assume `galois.GF(2**m)` defaults
are identical. The implementation agent was given these fixtures before claiming
parity. This report establishes the independent reference; it does not itself
claim that the new course implementation has passed against every fixture.

Query order is explicitly stored: `qary_ints(b,2)` lists big-endian binary
columns; masks are `(M @ L + d[:,None]) % 2`, transposed into rows, concatenated
group first, then delay, then the L column. For an independent design at each b,
`np.random.seed(seed)` and three successive `randint(2,size=(n,b))` calls determine
the matrices. Fixture index: `bch_author_fixture_index.json`.

No tokenizer template selection was added in this reference subtask. The parent
task owns the prospective task/grouping protocol. The native-study and learner
completion boundaries stated above remain in force.

### Subsequent independent course-core parity check

The course implementation was subsequently checked against the saved author
fixtures. **All 70 generator/delay comparisons, 132 decoder cases (including
corrected shortened-codeword positions), and three ordered query designs pass.**
`course_bch_independent_parity.json` binds exact source hashes and retained source
snapshots. This verifies codebook/decoder/design compatibility, not lesson quality.

The full recovery comparison replays the exact 26 saved author query designs and
values into the course implementation, without querying any model or changing the
analytic data. Source snapshot `spex_methods.py` SHA256
`7545eb3727e1f696d4f3e8642686b4482d790f66b5865a90693084c98b20d56b`, with
`spex_coding.py` SHA256
`dc2e4b564e03504a02da7a0ed5f1e3570c31f530883aa200cc3d667187d3fdc4`, yields
**23/26 spectra matching within 1e-8** after filtering author coefficients of
magnitude at most 1e-8 for comparison. Three discrepancies remain explicit:

- Polynomial `seed=1,b=3`: maximum coefficient difference `3.9331e-5`, with seven
  course sweeps versus six author sweeps. The first differing decisions are in
  sweep zero: the author divides by reference-shift values of approximately
  `-5.55e-17`, `1.25e-16` and `-1.53e-16`, producing reliability magnitudes around
  `1e16`. The course's explicit `abs(U[0]) <= 1e-8` guard rejects those candidates.
- Polynomial `seed=1,b=4`: maximum coefficient difference `7.2897e-6`. One author
  candidate uses `U[0]=8.33e-17` and is rejected by the same guard. A second
  difference at original group 1, bin 2 is sensitivity to the transform's final
  floating-point bits: real FWHT and author FFT differ by at most `1.11e-16` in
  that bin. Both actual author and course soft decoders fail on the author
  reliability vector, and both decode `{1,7}` on the course vector. The near-equal
  reliability ordering, not a different BCH/Chase algorithm for identical inputs,
  changes this decision.
- Dense spectrum `seed=1,b=3`: the author adds a spurious constant `+0.1228906`
  after its decoder explicitly reports **error count -1**. An instrumented replay
  locates the accepted failed decode at sweep zero, original group 2, bin 0.
  The course rejects failed decoding and does not add that constant. The other
  four recovered terms agree. This is a deliberate validity check beyond the
  author's wrapper behavior, not strict full-output reference equivalence.

All three instrumented author replays reproduce the original saved spectra
exactly. The near-zero guard and failed-decode rejection must remain disclosed
when explaining source parity; numerical differences must not be erased by
silently increasing the equality tolerance.

Receipts: `course_spex_independent_parity.json`, every `*_course_trace.json` and
`*_course_residual.npz`, `author_spex_discrepancy_traces.json`, and
`polynomial_first_sweep_roundoff_probe.json`. Reproduce these checks with
`check_course_bch_parity.py`, `check_course_spex_parity.py`,
`trace_author_spex_discrepancies.py`, and `probe_polynomial_roundoff.py` in the
isolated reference environment. A source snapshot is a bounded compatibility
result and is not a publication or native-study completion claim.


### Bounded fixed-query reference bridge

The new `chapter16_shapley_attribution_baselines/exercises/part18_spex/reference_bridge.py`
provides `run_fixed_query_stages(output_dir, inputs, arrays, freeze)` for the
prospectively bounded reference comparison. It is plumbing around actual source
algorithms, not an additional teaching implementation. Module import requires only
the standard library; the host path uses NumPy and the shared `run_native`
supervisor. Scientific dependencies import inside the isolated Python worker.

The accompanying `reference_environment/pyproject.toml` and `uv.lock` pin Python
3.12.12, spectralexplain 0.0.4 from author Git commit
`61400b5388b8ecbf18fe18c4d821ffe3e8a5f9c8`, sparse-transform 0.2.1,
shapiq 1.5.2, LightGBM 4.7.0, and the previously validated exact transitive
versions. The Git install's `direct_url.json` commit and installed scientific
Python source hashes are checked. No root lock or root environment was edited.
The tested isolated executable is:

```
/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/bridge_environment/bin/python
```

The tested bridge contract SHA256 is
`4f073c7c9ff2deab22d1e936b770123f4096e692ad1fc26c631ad17c679f7873`.
Its full source/version/grid/cap specification is retained at
`reference_execution_v1/bridge_analytic_v1/contract.json`. The hash binds the
bridge, isolated pyproject/lock, exact installed distribution versions, and all
Python sources in spectralexplain, sparse_transform and shapiq. It must be
recomputed if any of those change. The outer native freeze independently binds
its supervisor and native-study sources; the bridge does not freeze or run a
native experiment by itself.

The host accepts exactly six `(condition, training_values)` pairs, with methods
`spex` and `proxyspex`, seed zero, and `b` 3, 4, 5. Each condition supplies its
`key`, `method`, `seed`, `b`, `budget`, and `unique_count`. Query-plan arrays use
`{method}_s0_k{budget}_masks`, SPEX additionally `_matrices` and `_delays`, and
one common `audit_masks`. The freeze supplies `reference_python_path`,
`reference_contract_sha256`, and these nine ordered stage IDs:

```
author_spex_b3, author_proxy_b3, maintained_proxy_b3,
author_spex_b4, author_proxy_b4, maintained_proxy_b4,
author_spex_b5, author_proxy_b5, maintained_proxy_b5
```

The host writes one exact NPZ per budget before any stage starts. Its complete
field allowlist is `schema_version,n,t,b,seed,prompt_index,spex_matrices,
spex_delays,spex_masks,spex_values,proxy_masks,proxy_values,audit_masks`.
Extra fields, including audit labels, are rejected. In `native_fixed` mode,
validation requires n=28,t=3,prompt=0,seed=0,C=3, the exact 456/912/1824 requested
rows, 512 unique audit masks, unique ProxySPEX training rows, and audit masks
disjoint from both training designs. Duplicate SPEX masks must have exactly equal
saved values. Masks must match the complete group-delay-subcube sequence implied
by the supplied matrices and delays. The host never reads audit values or calls
an oracle. `analytic` mode allows the tiny exhaustive diagnostic games and is
separately labeled in every receipt.

Actual dispatch is explicit:

- SPEX: the source-supported `Ms_and_Ds.pickle` path supplies the frozen matrices
  and delays to actual `SubsampledSignal`. The replay callback compares every
  requested row to the supplied sequence before returning its saved value; no
  other query is possible. Actual `spectralexplain.support_recovery("soft",
  signal,b,t)` then executes with the author Chase/message-passing behavior. The
  legacy RNG is reset and advanced by the original C*n*b matrix draws before
  recovery, preserving the original random group order. The author's ineffective
  `regress="lasso",report=False` combination is retained, with no final fit.
- Author ProxySPEX: actual `spectralexplain.utils.proxy_spex((X,y),max_order=t)`
  executes its fixed-samples branch. A read-only Python return/exception-frame
  observer captures the function's actual fitted model, search, full spectrum,
  selected spectrum and RidgeCV object. It does not replace a function, alter a
  global, modify source, change a return value, or suppress the retained source
  exception. This observer was explicitly accepted for preserving the monolithic
  source function's intermediate evidence.
- Maintained ProxySPEX: actual `ProxySPEX(n=n,max_order=t,index="SII",
  proxy_model="lightgbm",hpo=True,random_state=seed)` is constructed; its actual
  `proxy_model.fit`, `convert_tree_model`, `_sklearn_to_fourier`, and `_refine`
  functions operate on the fixed training rows. The public sampler and subsequent
  SII conversion are bypassed because this bridge returns signed Fourier spectra.
  The public sampler does not provide this exact fixed-design interface. These
  direct method calls are disclosed as an adapter, not asserted to be an
  unchanged public `approximate` call.

The source grids remain different. At t=3 the author uses max_depth {1,3},
n_estimators {500,1000,5000}, learning_rate {.01,.1}: 12 candidates × five R²
folds plus one full-data refit. Maintained shapiq uses max_depth {3,5}, max_iter
{500,1000}, learning_rate {.01,.1}: eight candidates × five R² folds plus one
refit. Maintained `max_iter` is a LightGBM alias; receipts record actual booster
iterations as well as the misleading sklearn `n_estimators=100` attribute. Both
sources retain their original truncation and RidgeCV with 100 alphas
`np.logspace(-6,6,100)`, no intercept and its default CV. This source refinement
uses a basis learned from all supplied training rows; it is not the course
implementation's grouped out-of-fold basis selection. No shared-optimizer or
nested-validation equivalence is claimed.

`run_native.run_reference_stage` is the sole supervisor. Each fixed stage has a
300-second wall cap, shortened by the inherited joint deadline; each ID can be
used once. Nine stages have at most 2700 seconds of nominal stage time. The shared
supervisor samples summed process-tree RSS and output at 0.05 seconds with 8 GiB
RSS/output limits, and enforces the parent's full-study budget. Workers set
four CPU threads/first-four available CPU affinity and disable CUDA. An
individual timeout/failure is retained without a replacement result. A process
failure exposes no substitute predictions even when partial files exist. The
outer supervisor owns the consequence of exhausting the joint deadline.

Every stage retains `execution.json`, `child.log`, `result.json`, `spectra.json`
and `audit_predictions.npz` when produced, and actual proxy model/CV records.
Successful source stages produce signed full/selected/final spectra as available,
plus source tree predictions. Author parity failure retains full, empty selected,
and tree predictions but no final prediction. It remains `SOURCE_FAILURE`, not a
successful fallback. Extraction compares the actual tree and its full Fourier
representation at atol=1e-8, rtol=1e-7. There is no native course-versus-author
success tolerance that erases the documented source discrepancies.

Returned JSON-serializable receipts keep deterministic statuses, supports,
coefficients, predictions, selections, fit/alpha decisions, query counts, hashes
and exception type/message. Paths, elapsed time, sampled RSS and full tracebacks
are under an explicit `operational` field for the shared reproduction comparator.
Those original/fresh operational fields remain evidence even when excluded from
numerical reproduction comparison. Each prediction artifact is written before
return to the parent study's prediction boundary, before audit values are queried.

### Bridge validation, without native model outcomes

`prepare_bridge_analytic.py`, `test_bridge_analytic.py` and
`verify_bridge_analytic.py` executed 12 actual isolated workers under the shared
supervisor on the same additive, triple-parity and sparse-polynomial analytic
games. For each game, all three reference variants used the author's saved 1024
ProxySPEX query rows, with a source BCH n=12,t=3,b=3 SPEX design and exhaustive
4096-mask prediction audit. A fourth maintained-only stage used the actual
maintained public sampler's earlier saved 1024 rows. No audit response was passed
to the bridge; analytic ground truth was read only by the separate validator.

All author tree predictions are exactly identical to the earlier saved source
runs; full, selected and successful final predictions differ by at most
`4.885e-15`. The exact author parity exception and lack of final output are
preserved. All maintained own-query tree predictions are exactly identical to
the earlier actual public-interface runs, and full/selected/final differences
are at most `8.216e-15`. All actual boosters report four threads. Every full
Fourier/tree extraction passes the declared tolerance. No scientific source was
patched for these bridge checks.

On identical author training rows, final audit RMSE is:

| Analytic game | Author SPEX | Author ProxySPEX | Maintained ProxySPEX |
|---|---:|---:|---:|
| Additive | 2.18e-16 | .515070 | .224835 |
| Triple parity | 0 | retained source failure | 1.221e-9 |
| Sparse polynomial | 3.17e-16 | .301618 | 1.139e-9 |

Maintained final RMSE on its own earlier public-sampler rows remains .223697,
1.221e-9 and 1.213e-9 respectively. Thus fixed training rows preserve the known
source truncation discrepancy; differences are not solely sampler effects.
These are exhaustive analytic diagnostics, not new native model outcomes.

`test_bridge_host.py` separately tests the native-shaped host input allowlist and
nine-stage order using analytic parity labels and an explicitly mocked dispatch.
This test performs zero reference fits and zero native calls; its receipt is
marked `MOCKED_HOST_TEST_ONLY`. It verifies that the host reads `audit_masks`
only, never an audit-label key. The actual NPZ validator separately rejects an
archive containing an added `audit_values` field. The 12 real scientific stages
above, not this host mock, establish execution compatibility.

The complete bridge evidence is retained in
`reference_execution_v1/bridge_analytic_v1/`, with all per-stage process and
scientific receipts, original fixed NPZ inputs, source contract, grid/model files,
`verification.json`, `host_validation.json`, and test logs in its parent cache.
This work adds no learner-facing notebook result or play cell and makes no
TAUGHT, native-study, course-ready or publication claim. Original ARENA files,
root dependency locks and root model environments remain untouched by this
reference subtask.


An additional three actual SPEX bridge stages replay the original saved
n=31,t=3,b=3,seed=0 query designs and values for the same games. All three signed
spectra are **exactly identical** to the original author results (maximum
coefficient difference zero). `test_bridge_saved_spex.py` and
`bridge_analytic_v1/saved_spex_parity.json` retain this stronger saved-design
check. Across the bridge checks, 15 actual source workers were run, one retaining
the expected author ProxySPEX failure.

To install/recheck the isolated environment from the repository root, without
using the root environment as the installation target:

```bash
UV_PROJECT_ENVIRONMENT=/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/bridge_environment uv sync --project chapter16_shapley_attribution_baselines/exercises/part18_spex/reference_environment --frozen --python 3.12.12
/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/bridge_environment/bin/python chapter16_shapley_attribution_baselines/exercises/part18_spex/reference_bridge.py contract --output /tmp/spex_reference_contract.json
```

The worker CLI is intentionally internal: `_worker --input fixed_queries.npz
--output-dir existing_stage_dir --mode analytic|native_fixed --variant
{author_spex,author_proxy,maintained_proxy} --expected-contract SHA256`.
`run_supervised_stage` or the parent `run_fixed_query_stages` supplies a new output
directory through the shared supervisor and invokes this exact command. A second
independent resource monitor is not present in the bridge.

To rerun the analytic evidence into a **new** cache directory, set the same new
`ARENA_SPEX_ANALYTIC_BRIDGE_OUTPUT` path for these five commands. Existing evidence
is never overwritten by the stage supervisor:

```bash
export ARENA_SPEX_ANALYTIC_BRIDGE_OUTPUT=/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/bridge_analytic_rerun
/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/bridge_environment/bin/python /home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/prepare_bridge_analytic.py
.venv/bin/python /home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/test_bridge_analytic.py
.venv/bin/python /home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/verify_bridge_analytic.py
.venv/bin/python /home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/test_bridge_host.py
.venv/bin/python /home/bitwise/.cache/arena_extended/sparse_interaction_discovery/reference_execution_v1/test_bridge_saved_spex.py
```

The preparation command uses the isolated scientific environment; the subsequent
supervisors/validators use the root environment only to orchestrate fresh isolated
workers and compare already analytic values. They never load model weights. The
first analytic test script creates a fresh stage registry per game with the
shared nine allowed IDs, while the fixed native host uses the one inherited
joint-run registry. This difference is explicit test setup, not a native cap
bypass.


### Prospective resource-cap amendment: reference output is 256 MiB

Independent review identified a mismatch: the prospective native protocol
specified **256 MiB of retained output per reference stage**, but the first
bridge contract and shared supervisor inherited the whole-study 8 GiB output
cap. The earlier 15 analytic stages, original bridge bytes, original
`4f073c7c9ff2deab22d1e936b770123f4096e692ad1fc26c631ad17c679f7873`
contract and their manifest are retained as historical evidence. Their successful
analytic behavior does not establish compliance with the corrected cap.

The amended limits are unambiguous:

| Scope | Wall time | Summed process-tree RSS | Retained output |
|---|---:|---:|---:|
| Each reference stage | 300 seconds, shortened by joint time remaining | 8 GiB | **256 MiB** |
| Whole native study | 5400 seconds | 8 GiB | 8 GiB |

The bridge now declares `STAGE_OUTPUT_BYTES = 256 * 1024**2`; the shared
supervisor independently passes the same explicit value into each reference
`ResourceLimits`. Thus this is enforced behavior, not solely a metadata change.
The earlier paragraph describing an 8 GiB reference output limit and the earlier
prospective bridge hash are superseded by this amendment. Native execution
remains at **zero model queries** in this reference subtask.

The amended bridge source SHA256 is
`28ac06250cf11c0c25251d228fe2fde327fdefd6275e595ec288b9db70d27fae`.
The **new prospective reference contract SHA256** is
`3680dfd90c6b3cf88f6d7f454f978d031a81584e6b9f2ad666ea8e77b0fd949b`.
The exact isolated Python executable and dependency lock remain unchanged.
The root task will bind this new contract and the corrected supervisor into a
new prospective candidate; it retains the earlier candidate rather than
rewriting its history.

`reference_execution_v1/bridge_resource_cap_v2/` preserves before/after bridge
sources, both the previous manifest and the new contract, and an AST comparison.
After removing only the assignment to `STAGE_OUTPUT_BYTES`, the entire old and
new bridge ASTs are identical. The two contracts are identical after excluding
only the bridge source hash and retained-output-cap field. Scientific algorithms,
query adapters, source failures, grids and numerical tolerances did not change.


`test_bridge_resource_cap_v2.py` then executed one actual author SPEX worker on
the previously saved n=31,t=3,b=3,seed=0 parity input, under the corrected shared
supervisor. It completed with **exactly equal** signed spectra and audit
predictions to the historical source stage. Its actual `execution.json` records
wall_seconds=300, rss_bytes=8589934592, output_bytes=268435456,
sample_seconds=.05. The worker performed zero native queries. This is a resource
amendment smoke test; the earlier 15 scientific checks are preserved rather than
claimed to have been rerun. `amendment_verification.json` binds the actual stage,
new contract, structural-equality proof and supervisor source hash
`134fc5bc559591f8f5d9d9c5b09dd99e8a1e6d414d8512d438d8d23156970a2f`.

The supervisor owner separately reports 40 passing synthetic checks, including
the exact 268435456-byte reference-cap assertion and an actual short subprocess
that breaches a deliberately smaller test output cap. Those synthetic results
are the supervisor owner's evidence, separate from this actual analytic smoke.
