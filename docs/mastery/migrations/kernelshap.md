# 16.2 KernelSHAP and PartitionSHAP canonical migration

## Preservation plan

Current learner failure: the complete finite-game lesson lives in a legacy source directory and bypasses canonical authoring and durable prediction/reflection practice.

Target section claim: full-table constrained KernelSHAP matches an independent exact dividend oracle; partial sampling and a priori player unions require separate error controls. Exact Owen values may differ from ordinary Shapley while preserving efficiency.

Signature result to preserve: the full three-panel sampling convergence, attribution comparison and seeded random-partition histogram, including all original negative findings and limits.

Student methods: dividend game/oracle, finite Shapley kernel/design rows, constrained KKT regression, uniform/complement-paired sampling, partition validation/group expansion and exact Owen values. Keep the core methods visible and independently tested.

Controls/baselines: exact dividend oracle, full interior table, equal-budget uniform sampling, shuffled interior values with endpoints fixed, singleton/aligned/misaligned partitions, 100 random balanced partitions and all 50 sampler seeds.

Play cells: preserve the live budget/seed/strategy/partition play. Add an optional small CPU plot if needed to make knob changes visible, preserving its exact historical source and output separately.

Files to modify: new master_16_2.py, new canonical part2_kernelshap_partitionshap directory, registered page, focused migration tests, this note and scoped evidence. Root owns
shared registry/config/workbook. Legacy source directory and old page remain physically unchanged; archive all exact pre-edit sources/receipts and declare any nested archive
inventory explicitly. No pretrained model, GPU, training or download work.

Tests: exact archive/source/output/figure parity with named route changes; own-answer student completion; independent finite-game/permutation/KKT oracles; import safety and dtype restoration; fresh analytic-play execution provenance if added. No commits, staging or pushes.

## Canonical artifacts and preservation

The canonical source is `infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_2.py`. Generated artifacts live in **`part2_kernelshap_partitionshap`**, using stem `16.2_KernelSHAP_and_PartitionSHAP`. The registered page is `02_[16.2]_KernelSHAP_and_PartitionSHAP.{md,py}`.

The complete old **`part2_kernelshap_partition_shap_controls`** directory and `02_[16.2]_KernelSHAP_and_PartitionSHAP_Controls.md` remain physically unchanged. Before any canonical
write, all original section files, original page/signature asset and all fourteen original `report_inputs` were archived exactly in `published_before_master.tar.gz`. `before.json`
records every member and archive hash. Tests compare the archive member set with the full manifest and recheck every original live file and report-input hash. The explicit
nested-archive inventory is empty: none existed in this source, and no nested archive was silently omitted. No original ARENA chapter 0–4 material was edited.

The original lesson has **26 cells, twelve code cells, seven exercises, twelve output entries across ten output-bearing cells, one three-panel signature and a working
budget/seed/strategy/partition play**. All original teaching prose, equations, controls, expected outputs, help, solutions, interpretations, negative findings and limitations
survive. Six implementation cells still complete from their own answers; the seventh exercise's complete nonstub control answer is retained. All 22 original notebook function/class
definitions remain identifiable. Five summaries and ten specific blank forecast/reflection pairs are additive.

The current generated lesson has **69 cells, thirteen code cells, seven exercises and 4,740 prose words** at the last build. Canonical headers, standard objective/rating blocks and
the current CPU evidence boundary are additive. The old learning-objectives heading becomes “Learning goals” to respect the converter's reserved syntax, without changing its
objective lines. The original title's publication suffix is normalized for exercises/solutions.

Three narrowly identified code routes change: setup points to the canonical section/tests/media, Python-only root lookup uses `__file__`, and the verification appendix uses a fully
qualified import of the unchanged legacy compatibility module. Future canonical plotting writes only canonical media, protecting the original asset. `tests.py` retains every
original test body/assertion, while its loader targets canonical methods and its report lookup explicitly targets the unchanged legacy report. Its old prototype-library oracle
import is replaced by visible, independent permutation-order and restricted-order Owen oracles in the test file. `utils.py` is a byte copy. No taught method is hidden behind an
import.

## Failed regression and declared numerical correction

Independent review caught a genuine intermittent failure in the preserved partial-design solver. For the additive three-player game `[1,2,3]` with one observed coalition `{0}`, the
declared minimum-norm constrained answer is `[1,2.5,2.5]`. The float64 KKT system has rank three. Default/`gelsy` least squares intermittently reported rank two and returned
approximately `[1.93103448,2.03448276,2.03448276]`, with KKT residual about .31034483. The independent suite recorded **19 passing / one failing** check. The failure and exact
small system are preserved in `rank_deficient_review_failure.json`.

The reviewer reproduced the instability with five calls per driver at 24, one and two CPU threads. Both `gelsd` and `gelss` consistently returned rank three with residual below
6e−15. The oracle was retained without relaxation, thread pinning as a workaround or retry selection. The canonical partial branch now uses **`torch.linalg.lstsq(kkt.cpu(),
rhs.cpu(), driver="gelsd")`**, then returns coefficients to the original KKT device. This handles CUDA's lack of an SVD least-squares driver by solving the tiny system on CPU. The
complete-table direct `solve` branch is unchanged. Actual CUDA execution was not performed.

This is a named numerical correction to implement the lesson's already-declared minimum-norm KKT semantics. It is **not** treated as routing-only AST identity. `output_parity.json`
records exact original/current code hashes and the explicit correction at ordinal four. Reversing only that correction and the declared routes recovers every original nonblank
source line and AST. The original wrong-default source and every original output remain in the unchanged legacy notebook and exact archive.

The partial estimator's historical per-seed results cannot be assumed unaffected by a faulty rank-revealing solve. Therefore all current canonical outputs were genuinely rerun on
CPU after independent source review. No current numerical output is represented as carried historical execution. Historical figures remain visible in the original opening preview
and as `kernelshap_partition_controls_signature_historical.png`; corrected current signature/play figures occupy separate canonical output assets.

## Fresh complete CPU execution and retained attempts

`fresh_cpu_execute.py` runs the inspected complete notebook with a **120-second total bound**, GPUs hidden, two CPU threads and a frozen runtime/source/configuration manifest. It
executes thirteen lesson code cells. The appendix invokes only five unchanged finite CPU compatibility contracts; its cell is marked `legacy_finite_contract_rerun`. The CUDA neural
preflight is neither called nor claimed. An evidence-only end probe captures actual runtime and exact seed-error arrays; it is retained in the execution-input archive and excluded
from the generated lesson artifact.

The initial launch failed **before kernel startup** because the filesystem sandbox blocked local socket creation. Its failed notebook and traceback remain in
`fresh_cpu_socket_blocked.*`. A normal sandbox escalation allowed the authorized local Jupyter sockets. Two complete same-source CPU runs then succeeded: **4.074255738 seconds**
and **3.911051705 seconds**, an aggregate **7.985307443 seconds** of complete-run wall time. The first success has its own immutable notebook archive, receipt, input and freeze.
The second adds exact array capture. `execution_attempts.json` lists both, and no transfer game was executed.

`fresh_cpu_execution.json` binds all thirteen current code-cell sources/outputs/counts, actual runtime, figure hashes and complete matrices to `fresh_cpu_freeze.json`. Both
successful executed notebooks and the exact before/input notebooks are retained as compressed artifacts. The runtime is Python 3.14.2, PyTorch 2.12.1+cu132, NumPy 1.26.4, ambient
float32 with explicitly float64 finite-game calculations, two CPU threads, and **CUDA uninitialized**. No checkpoint, optimizer, model training, download or external service was
used. Original verification reports were not rewritten.

The fresh budget-24 paired-design mean maximum error is **.24212010859344338**, uniform is **.47716847873371226**, and paired minus uniform is **−.2350483701402689 attribution
units**. The six-by-fifty error matrices, means/standard deviations, shuffled-value errors and all hundred partition draws/errors are retained. Rounded headline values agree with
the old publication; this does not prove identity of every old per-seed row. Full-table error is below 1e−12, the fixed misaligned partition error remains .300, and 91/100 random
balanced partitions differ from ordinary Shapley.

The main comparison combines complement pairing, kernel-proportional pair sampling without replacement and kernel row weighting without inverse-inclusion correction. It compares
those complete estimator designs on this fixed game. It does not isolate pairing alone or establish an unbiased generic sampling estimator. Plot bands show seed standard
deviations, not confidence intervals. Exact Owen values are correct for their restricted-ordering game even when they differ from ordinary Shapley.

## Runnable play and prediction contract

The original live play remains at ordinal ten. The additive plot is ordinal twelve: explicit budget, seed, highlighted sampler and groups recompute both samplers, full-table
regression and requested/aligned/singleton Owen values. A visible independent enumeration of all 720 player orderings checks the dividend oracle, while efficiency, complement
inclusion and full-table parity are checked immediately. The two-panel plot shows each player credit and every maximum-error control. Its current output is bound to the fresh
complete run through `play_execution.json`; earlier isolated pre-correction play evidence remains separately preserved.

The default play uses budget 24, seed four, paired-kernel highlighting and groups `((0,3,4),(1,2,5))`. Its current maximum errors are about .132088 for paired sampling, .417317 for
uniform sampling and .300 for the requested Owen grouping; full-table/aligned/singleton controls recover ordinary Shapley. These appear after forecast commitment, with guided
exposure made explicit. No learner answer, confidence or observation is fabricated.

The adopted guided primary question is the mean over seeds **0–49** of the six-player maximum absolute oracle error for `paired_kernel` minus that for `uniform`, with **24 distinct
interior coalitions per estimator**. Freeze baseline zero; singleton dividends `[.6,−.2,.5,.1,.4,−.1]`; triple dividends `{0,1,2}:1.8` and `{3,4,5}:−.9`; all coalition
masks/orderings; KKT constraints/numerical driver; all seeds; paired and uniform sampling algorithms; value-shuffle seeds 100–149; partition seeds 0–99; and
aligned/misaligned/singleton group definitions. Exact inputs and implementation hashes are available in the freeze and generated source.

A distinct transfer adds **one new four-way dividend `{0,1,2,3}:+.8`**, preserving the original dividends, budgets, masks, seeds and groups. Forecast both the same error gap and
the formerly aligned Owen error. This changes the game rather than merely replacing a seed or repeating an already-observed primary negative control. **This transfer was not
executed.** Root owns workbook/registry integration; the current reference remains monitored guided practice, not unseen mastery or untouched evaluation data.

## Validation and remaining limits

Focused validation uses:

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv MPLCONFIGDIR=/tmp/arena-mastery-kernelshap/mpl \
uv run --no-sync --offline pytest -q tests/mastery/test_kernelshap_master_migration.py
```

The final corrected suite reports **21 passed in 4.97 seconds**. Twelve independent CPU-oracle lanes cover permutation/dividend parity, finite kernel weights, baseline
shifts/design rows, full-table parity, an independent nullspace regression solution, the retained rank-deficient regression (five repetitions at each of three thread counts),
sampling controls, restricted-order Owen enumeration, partition rejection/group expansion, endpoint-preserving shuffled values and twelve live-play knob combinations. Other checks
cover archive/member/source/report hashes, all original source correspondence, fresh output provenance, original and current figures, own-answer completion, every teaching line,
copied independent tests, import safety with preserved grad mode, complete execution archives/runtime/seed arrays and retained failures. The tests restore ambient dtype and any
changed thread count.

The two current figures and historical signature were inspected visually. Generated module import performs no experiment, plotting, global grad/style change, filesystem write or
CUDA query. The current canonical directory contains only standard generated artifacts, tests/helpers and media; no retained-directory structural exception is required. The old
source directory is outside that canonical surface and remains wholly preserved.

Remaining brief gaps are explicit: six finite synthetic players; supplied fixed partitions rather than learned hierarchies; no token/region masking, background distribution,
conditional expectations or large-model attribution. The old four-player CUDA neural preflight remains separate historical supporting evidence. No new CUDA acceptance,
pretrained-model replication, transfer result, human learner acceptance, readiness promotion or whole-course completion is claimed.

Proposed chapter description: **Implement constrained KernelSHAP and exact Owen values, then distinguish sampling error from changed credit under declared player groups using exact finite-game controls.**
