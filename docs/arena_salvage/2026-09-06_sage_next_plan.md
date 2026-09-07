## Current checkpoint — 16.12 published and reproduced; full course goal remains active

Section **16.12 SAGE: Which Features Help a Model Predict?** is complete in
`chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance`.
Learner files are `16.12_SAGE_Global_Importance_{exercises,solutions}.ipynb`,
`lesson.md`, and
`chapter16_shapley_attribution_baselines/instructions/pages/12_[16.12]_SAGE_Global_Importance.md`.
Eighteen actual-output figures are in that chapter's `instructions/assets/sage_*.png`.
The README links all learner files, primary sources and execution commands.
Original chapters 0–4 remain pinned to
`f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0`, with zero preservation violations.
Do not reset/clean the dirty tree, disturb unrelated GPU work, or use subagents.
All new work ran on CPU through `uv run python`. The unlimited course goal is
still active; this checkpoint completes one selected requirement, not the course.

### Learner result and scientific scope

The lesson has **80 cells, 33 executed own-answer code cells, nine implementation
exercises, nine student functions, eighteen figures and two editable plays**.
Every stub is completed solely from its own following Solution dropdown. All
core methods, complete training, model forward passes and study orchestration
are visible. Nine immediate tests provide feedback; twelve contract cases
protect real boundaries and complete reproduction. Browser checks opened all
32 dropdowns and found 81 rendered math expressions with zero math errors.
Original 0.3 Optimization and existing 16.1 exercises/solutions/page were
inspected before the written plan in
`docs/arena_salvage/2026-09-06_sage_next_plan.md`.

Students implement coalition masks, background-averaged restricted predictions,
per-row squared-loss games, exact Shapley contraction, conditional and marginal
Gaussian games, permutation-path contributions, direct SAGE sampling, outer
standard errors and remaining-feature prefix evaluation. Exact Gaussian examples
separate information in an unused correlated feature from direct functional use;
XOR reveals interaction allocation; a misspecified model gives negative SAGE.
The native regressor distinguishes prediction SHAP from local loss SHAP and
verifies mean local loss Shapley equals global SAGE. The lesson separates
outer Monte Carlo precision from finite-background choice, conditional/marginal
assumptions, inner-loss bias, model seed and recipe weighting.

Primary method: Covert, Lundberg & Lee (NeurIPS 2020),
https://papers.neurips.cc/paper_files/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html .
Read main Sections 2–5, Supplemental A–B and D/Algorithm 1, and author
MarginalImputer / PermutationEstimator path code. Author parity uses unmodified
imputers.py and utils.py at commit
`0e34e6c7e2a6b1666e39043e83b9929e68ee68f3`, with hashes/MIT provenance in
`author_reference.json`. The real experiment uses a finite empirical marginal
background; it is not exact conditional SAGE for the unknown concrete population,
a causal ingredient effect, universal optimal-model importance or a reproduction
of the full paper benchmark.

### Frozen data and training

The experiment was frozen at **2026-09-06T11:27:00.245079+00:00** with zero native
fits. These three files remain byte-unchanged:

- protocol.md: `d19cbb3593dc512b1e211badf7092d77ba3ee9418445406d5ee8c90679c39c39`
- data_manifest.json: `4262d471486ed5b26cbcaa804950e878b1ef00318b4141eaa52b554eb26de47a`
- sage_data.py: `5d9d12b16444f29ac4cf51b6f5e0995a105665c4c86d955d3f3990fd77c5350c`

`protocol_readable.md` is an exact token-preserving, line-wrapped copy; it is
bound into the normal report. Do not casually reformat the frozen source files.
Data: Yeh (1998), UCI Concrete Compressive Strength, DOI10.24432/C5PK67, CC BY4.0.
Official UCI API points to
`https://archive.ics.uci.edu/static/public/165/data.csv` (41472 bytes), SHA256
`8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af`.
All 1030 complete rows are retained. There are 992 distinct eight-input rows and
1005 distinct full nine-column rows; do not deduplicate conflicting outcomes.
Group identical seven-ingredient recipes regardless of age. Encode seven input
float64 values little-endian, SHA256 their bytes, then sort groups by SHA256
`arena16.12/split/` plus group hex. Split the 427 groups at floor(.7G) and
floor(.85G): **298/64/65 groups, 758/153/119 rows** for train/explanation/audit.
Group overlaps are zero. Laboratory/experimental identities are unavailable.

Input/target means and population SD use training rows only. Display predictions
in MPa and losses in MPa². Six CPU float64 networks have architecture
8→32 ReLU→32 ReLU→1, default torch.nn.Linear initialization. Clean/shuffled pairs
reset seeds161210..161212; shuffled targets use NumPy seeds161220..161222.
Each trains 800 full-batch Adam steps, lr.003, wd.0001 on all parameters, no
selection or early stopping. Four CPU threads; all six models retained. One
ridge baseline uses standardized data, mean squared risk and coefficient ridge
.01 excluding intercept. Seven actual fits, **4800 optimizer updates**.

Primary background64 uses training source rows sorted by SHA256
`arena16.12/background/161260/{source_row}`. Nested16/128 and independent64 sets
with seeds161261/161262 are fixed controls. Missing features come from one intact
donor row. Predict each hybrid, average predictions, then take loss. Every model
has all256 coalitions on all153 explanation and119 audit rows. Rankings are
saved with final models before the audit split is passed to evaluation.

Direct sampling uses4096 paths, seed161290. Thirty-two table-based schedules
seed161300..161331 each contain8192 independent row/permutation paths; budgets
32/128/512/2048/8192. Random ordering baseline:100 orders, seed161280. Evaluation
target permutation161270; analytic inner-bias illustration161340. Equal-recipe
weighting, exact ignored ninth feature and nearest-training-distance diagnostics
are retained. The real play recomputes all local coalitions for audit row index0
(source row1), selected feature0, background64; it is exploratory audit reuse.

### Results and independent references

Audit MSE for clean seeds0/1/2 is **36.2886 /32.1915 /37.0454 MPa²**; ridge99.655,
training-mean target244.679; shuffled networks480.045/511.116/557.622. Primary
explanation SAGE in natural feature order is
`[143.0495,14.6550,-8.2012,16.7032,2.3625,-11.4649,-9.7806,78.5034]`.
Audit SAGE is
`[131.4851,11.0853,-2.9313,22.0535,-3.2890,0.6109,-10.4913,63.4377]`.
Cement/age dominate, while small values can be negative or change sign. Prefix
plots retain SAGE/leave-one-out disagreements and all three clean models plus
ridge. Random-order bands describe ordering variation, not population intervals.

Independent checks enumerate all40320 feature orderings, solve constrained
weighted regression on254 proper coalitions, compare64 actual predictions with
NumPy, and64 restricted predictions with the author imputer. Maximum errors are
1.805e-11 /8.527e-14 /1.421e-14 /7.105e-15 respectively. Every permutation path
must conserve its endpoints. Direct paths/table paths differ by at most6.821e-13;
mean local/global attribution by8.527e-14; ignored ninth feature exactly0.

### Current execution and archives

Cache root: `/home/bitwise/.cache/arena_extended/sage`.
No SAGE model/test jobs are left running. The local preview server is the
existing localhost8768 server rooted in the alignment rendered-preview cache.
Verify it is live before reuse; do not interfere with unrelated jobs.

- **own-full-v4** PASS11.6174s, 80/33/18 cells/code/figures, 15 unchanged sources.
  Native payloads: `figures/native-83cv2nzi/study`.
  Completed notebook SHA256:
  `2de7a4227e7a944d26479644a37f62c4fe50348ece4c181a8e0a0cf4d8de4c79`.
  execution.json SHA256:
  `15eda55e57ca6e402f0d6ea509b503ad761aa32479135f720c9fde515584f783`.
- **fresh-native-v4** PASS3.7834s; all464 tensors /3786320 elements /61 scalars
  compare exactly, maximum error0. No weights or cached masks supplied to training.
  reproduction.json SHA256:
  `f350832ae1492c36cd2ff145fcbea8a32d95d8713037b3e0a77a3039cc210b99`.
- **Current normal report** accepted,49 input hashes verified unchanged.
  Actual new training in `report_reproductions/report-8s8c9vjy/fresh_cpu`:
  native3.51085s, full reproduction3.58084s. expected_outputs/fresh_reproduction.json
  points to this report run; learner_execution.json retains fresh-native-v4.
- Five complete own-answer payloads: diagnostics.pt21537991 bytes,
  play.pt19587, evaluated.pt3672087, fitted.pt289389, prepared.pt4898394.
  Payloads stay in cache; archive manifests bind their exact paths, sizes/hashes.

Current expected_outputs archives: pinned_source_evidence (7 readable members),
full_own_answer_v4 (51/5 external tensors), fresh_native_v4 (29/5),
normal_report_source_v2 (51/0, including all49 verified current source inputs),
normal_report_native_v2 (29/5). Development archives preserve native-full-v1,
own-full-v1, failed own-full-v2, trace-v2, checker diagnostics, own-full-v3,
fresh-native-v3, and early normal-report/native evidence. Archive helper reads
back every tar member and verifies its hash. The early report input manifest
became stale after editorial changes; it is explicitly development evidence.
The accepted current normal report has a complete source snapshot.

The failed v2 independent checker was diagnosed, not dismissed. Under the tested
Python3.14 /NumPy1.26 Jupyter kernel, `added = current | (1 << features)` reused
and mutated the initial current buffer. Kernel evidence shows previous values0,
new values1..128, shares_memory=True, and every first-step contribution zero.
Saved model/game arrays were unchanged after every notebook cell. The reference
now uses an explicit `np.bitwise_or(..., out=separate_array)` and checks every
path's endpoint conservation. The fitted experiment and frozen decisions did
not change. Do not replace this with an operator expression casually.

### Integration, global gates and continuation

Append navigation and conversion_map in `infrastructure/core/config_extension.yaml`,
then generate config.yaml through scripts/build_merged_config.py. SAGE is appended
to all three method and hard-ladder registry surfaces. Only `shapley.sage` was
promoted from UNIMPLEMENTED to TAUGHT. Root README and style status link the
reviewed learner. No original course content was edited.

Course-surface auditing follows actual SAGE training, loss, sampling, evaluation
and independent-check calls, with mutation tests. It recognizes called
plt.gcf().savefig helpers. CPU report evidence has a distinct regression/recipe
audit contract; it does not manufacture token, SciPy-optimizer or image metrics.
The hard-ladder audit now counts genuine NumPy/PyTorch numerical assertions,
not only Python assert statements. Readability auditing preserves a byte-frozen
protocol only if a bound, token-identical readable mirror passes line checks.
Mutations of the mirror, frozen source or checksum must fail this exception.

Final focused suite: **84 passed,1 deselected,9.90s**. The deselected test is the
known repository-wide no-minified-source assertion; running it still fails on
78 older issues. That failure is preserved separately, not reclassified as a
pass. The readability audit has no new SAGE issues.

Final gates: original preservation PASS; course surface PASS13 chapters; style
PASS54 pages; hard ladders PASS53 rows; report evidence PASS56 reports. Registry
has205 entries: **49 TAUGHT,36 PREFLIGHT_ONLY,20 TOY_ONLY,13 DEFERRED,
87 UNIMPLEMENTED**. Final completeness fails on exactly those87 unimplemented
items. Partial/placeholder sections still require their scoped upgrades; don't
redefine the full goal around these passing section gates.

Commands (from repository root):

```bash
uv run python chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance/execute_learner.py --output-directory /tmp/sage-own-new
CUDA_VISIBLE_DEVICES='' uv run python chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance/reproduce_native.py --learner-directory /tmp/sage-own-new --output-directory /tmp/sage-fresh-new
CUDA_VISIBLE_DEVICES='' uv run python scripts/run_extension_verification_reports.py --section 16.12
uv run python scripts/audit_original_arena_preservation.py
uv run python scripts/audit_course_surface.py
uv run python scripts/audit_arena_style_depth.py
uv run python scripts/audit_hard_exercise_ladders.py
uv run python scripts/audit_report_evidence_contracts.py
uv run python scripts/audit_roadmap_final_completeness.py
uv run python scripts/audit_no_minified_files.py
```

Continue with existing selected UNIMPLEMENTED requirements. TreeSHAP, DeepSHAP
and GradientSHAP remain concrete next candidates in this comparison track;
inspect the current registry and closest original lessons before choosing the
next bounded section. Keep original preservation, frozen experiment decisions,
visible student implementations, controlled real results and complete execution.
Do not rerun the cache's one-off author/polish/integration scripts over final
sources; they were version-specific construction tools.


# 16.12 SAGE — learner rewrite plan

Current learner failure: the selected SAGE requirement is unimplemented. Existing 16.1 explains finite Shapley
games, but offers no model-loss game, conditional restriction, practical marginal approximation, or real
global-versus-local comparison. Inspected both 16.1 notebook variants and its page, the adjacent 16.11
builder, and original 0.3 Optimization's exercise, stub, immediate experiment, help and following solution
structure. Original chapters remain append-only.

Target section claim: feature importance depends on the model, evaluation distribution, loss and
missing-feature convention. Students can derive conditional SAGE on an exactly solvable correlated Gaussian
model, implement practical marginal SAGE for a freshly trained concrete-strength regressor, and separate
estimation precision from assumption error. No claim of optimal-model information importance for a learned
regressor.

Signature result to create: feature-by-feature loss-reduction bars, local prediction versus local loss
explanations, the exact mean-loss-Shapley/global-SAGE identity, correlated-feature credit transfer, sampled
convergence against all 256 coalitions, and held-out feature-removal curves with random order controls.
Present model/seed/background variability and failed controls rather than selecting outcomes.

Student-implemented functions: coalition masks; marginal prediction averaging; per-instance loss games; exact
Shapley contraction; Gaussian conditional restriction and analytic risks; permutation path contributions and
sampling; uncertainty summaries; subset-prefix evaluation. Model training and complete study orchestration
remain visible, with independent ordinary torch reference calculations.

Controls/baselines: constant regressor, shuffled-training-target regressors, mean-imputed versus
background-averaged prediction, prediction-SHAP mean/mean absolute values, leave-one-feature-out, random
feature orders, exact ignored-feature dummy, conditional/marginal correlation sweep, background sizes and
independent sets, fixed-model negative-importance example. Every substantive figure has an explicit
comparison.

Play cells: vary correlation and a model's erroneous coefficient in the analytic example; change a real
held-out case/background/feature and rerun actual model attributions. The real play is exploratory reuse of
held-out data.

Files to modify: new part12_sage_global_importance lesson, notebook pair, methods and immediate tests, frozen
data/protocol, study and execution/reproduction tools, page and assets. Only after fresh execution and visual
review: append navigation, method/ladder/status registries, requirement-specific SAGE evidence and handoff.

Tests to run: independently enumerated permutation and weighted-regression Shapley parity; Gaussian
conditional/MSE oracles; averaging-before-loss and missingness controls; direct-sampling versus enumerated
finite-game checks; source/own-answer boundaries; complete clean-kernel execution; fresh CPU retraining and
tensor-level reproduction; direct figure/browser inspection; original preservation and relevant global gates.

Primary sources read: Covert, Lundberg & Lee (NeurIPS 2020), main paper Sections 2–5; Supplemental A–B and D,
Algorithm 1. Author MarginalImputer and PermutationEstimator._process_sample at commit
0e34e6c7e2a6b1666e39043e83b9929e68ee68f3. The latter evaluates a fixed empirical background before loss. UCI
Concrete Compressive Strength (Yeh, 1998), DOI 10.24432/C5PK67, 1030 rows, 8 inputs, CC BY 4.0; official API
points to static/public/165/data.csv. Dataset choice precedes all native fits. Freeze detailed split,
preprocessing, seeds, training, background, estimator and control choices before fitting any native model.
