# Next selected learner: influence, checkpoints and actual retraining

## Current checkpoint — 16.9 published and reproduced; course goal still active

Section **16.9 Influence Functions and TracIn** is now a complete, published
ARENA-style learner in
`chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin`.
Original ARENA remains the pinned append-only base. Preserve the dirty tree,
all previous checkpoints and the user's unrelated GPU process. No subagents
were requested or used; all new training and verification ran on CPU.

The exercise and executed solution notebooks contain **80 cells, 40 code cells,
eleven exercise groups, thirteen student functions, fourteen immediate test
calls, eighteen controlled figures and two editable plays**. The canonical
`lesson.md` supplies both notebooks and the instruction page
`chapter16_shapley_attribution_baselines/instructions/pages/09_[16.9]_Influence_Functions_and_TracIn.md`.
Every core method and the full study orchestration are visible in the notebook.
Python blocks were formatted for reading without changing executable ASTs;
only docstring indentation was normalized. All 42 dropdowns open and 43 math
expressions render without errors. Eight current browser views and every figure
were inspected, including figures byte-identical to already inspected versions.

### What the student learns and sees

The learner derives regularized logistic losses, weighted objectives and
curvature, Hessian-vector products, conjugate gradients, independent batched
Newton refits, implicit upweight influence, exact SGD trajectories, exact and
Taylor path allocations, TracInCP, affine ghost products and class-matched
removals. Immediate checks use independent autograd, NumPy/SciPy, exact quadratic
and actual update oracles. A fixture play changes learning rate; the real-data
play changes query/label condition/budget and actually retrains 33 models.

The real-data experiment uses the frozen UCI source archive, 128 training images,
48 ranking queries and 64 final queries from the separate source-test writer
group. Both clean and 12/128-flipped training-label conditions remain. The
manifest SHA-256 is
`a009ed80855932e774a427cc0b727cb30cfdfa7c228b1d5d7b7dee8f569b0a7c`.
The protocol was frozen before digit fitting; no outcome-driven changes were
made to model, optimizer, data, corruption, seeds, checkpoints or final galleries.

Every complete run retains 930 independently converged reference fits, six
separate-versus-batched refit checks, ten independent SciPy optima and six full
SGD trajectories (30720 updates). All 48 rankers and 240 deletion-policy/budget
refits per label condition finish before final evaluation. Ranking queries alone
choose deletion sets. All damping values, twelve perturbation probes, eight
weight changes, four checkpoint choices, three SGD orders and 32 matched random
rankings remain. The later exploratory query play uses a ranking query and does
not change the reference experiment.

Both full-data models score 64/64 final accuracy, with mean losses 0.0821795 and
0.1731935. Influence is closely ranked but has **54/8192 clean and 34/8192
corrupted wrong-sign pairs**. At deletion budget 8, influence changes final mean
loss by -0.00222964 and -0.0668859; all 32 random changes are above these values
at that budget. Some checkpoint choices hurt loss or accuracy. Clean deletion
accuracy stays 64/64; corrupted deletion accuracy ranges from 62/64 to 64/64.

The image galleries show genuine disagreement: on clean query `optdigits.tes:27`,
TracInCP's highest image `optdigits.tra:2794` is harmful under actual removal,
whereas influence's `optdigits.tra:3112` is helpful. A source digit 1 trained as 7
can hurt a digit-1 query and help a digit-7 query. Exact path sums telescope, but
Taylor approximations can substantially overestimate accumulated loss reduction.
Wrong normalization can preserve ranking while changing effect sizes. Ghost
products match complete gradients; rolled residuals visibly fail.

Claims remain limited to a small regularized binary classifier and the stated
writer-separated subset. No exact Data Shapley, general nonconvex attribution,
tied-token ghost formula, optional GhostSuite package integration or complete
paper replication is claimed. In-Run Data Shapley beyond the earlier one-step
lesson is still a separate open requirement.

### Exact execution and archives

Persistent cache root:
`/home/bitwise/.cache/arena_extended/data_attribution`.

- `own-full-v3`: complete current own-answer run, **13.535 s**, all thirteen
  source inputs unchanged, eighteen figures. Current publication comes from it.
- `fresh-native-v3`: independent data reload and complete refit, **1.661 s**.
  All **1048 tensors / 16726855 elements and 7272 scalar values** match exactly.
- `report_reproductions/report-2dmugp85/fresh_cpu`: the normal report's separate
  complete native retraining. `verification_report.json` is accepted with no
  verification failures and binds fifty exact source inputs. This report ran
  the study again; it did not count cached weights as retraining.
- Every complete native run has fifteen `.pt` payloads totaling **165630181
  bytes**. Preserve these persistent files; the archives record external paths,
  hashes and sizes instead of duplicating tensor payloads into Git.

Read-back archives are in the part's `expected_outputs`: `full_own_answer_v3`
(52 verified members), `fresh_native_v3` (16), `report_fresh_native_v1` (16),
`normal_report_source_v1` (52 exact source/report members) and `release_gate_v1`.
Development archives preserve full own v1 (35), own v2 (51), fresh v1/v2 (16
each), ranking-only native development and all earlier mathematical foundations.
The v1 visual defects and the formatter's docstring-only guard failure were
resolved without altering scientific settings. Keep these earlier records.

### Integration and remaining work

Navigation is defined in `infrastructure/core/config_extension.yaml`; generate
`config.yaml` with `uv run python scripts/build_merged_config.py --write --check`.
Never edit `config_original.yaml`. The initial integration audit caught a missing
overlay update; the corrected merged configuration now passes preservation.
The surface and report audits now recognize actual scratch-model refits, while
keeping their previous pretrained/CUDA contracts. Mutation tests reject missing
refits, comparisons, source loading, held-out evaluation and required evidence.

Final focused validation passes **77 tests in 9.96 s**. The course surface
(13 chapters), style depth (51 pages), hard-exercise ladder (50 rows), report
evidence (53 reports) and original preservation audits pass. All fifty current
16.9 report inputs still match their recorded hashes.

The current registry has **205 requirements: 46 TAUGHT, 36 PREFLIGHT_ONLY,
20 TOY_ONLY, 13 DEFERRED and 90 UNIMPLEMENTED**. The three newly taught rows are
influence functions, TracIn and the required affine ghost dot product; optional
GhostSuite integration remains explicitly outside the claim. All other topic
statuses are preserved. The global completeness audit still fails on those
ninety unimplemented requirements; passing local gates is not course completion.
The readability audit still reports 78 older issues, with none in 16.9.

Continue the selected syllabus using `docs/roadmap_requirement_registry.yml`
and the governing roadmap, while keeping original chapters unchanged. A coherent
next unit is Datamodels, followed by dataset cartography / richer In-Run Data
Shapley. Before implementing it, read the primary method and relevant original
ARENA example, write the learner arc, and freeze development/final selection.
Reuse only boring data and fitting plumbing; keep the newly taught method
student-visible. Demand actual refitting outcomes, random/matched baselines,
interpreted plots, editable experiments and complete own-answer plus independent
execution before updating its requirement. Also return to partial native sections
and the remaining global audits before declaring the full extension complete.

The checkpoint below is historical and is superseded by this section.

## Current checkpoint — real-data development passed; full learner under construction

The 16.9 native development run has now completed all 930 prescribed converged
reference fits and all six 5120-update SGD paths on CPU. Ten independent SciPy
checks, separate-versus-batched refits and ghost-gradient parity passed. The 21
method, data and train/final-boundary tests pass. Ranking-query results retain
wrong-sign influence errors and substantial accumulated Taylor error, especially
with corrupted labels. No digit final-query evaluation has run at this checkpoint.

Exact development inputs, results and tensor digests are archived in the part's
`development_evidence/native_development_v1`; tensor payloads remain at
`/home/bitwise/.cache/arena_extended/data_attribution/native-development-v1`.
`lesson.md` is being extended from the reviewed foundation to the complete study.
Do not execute it until the final figures, play, summary and full-scope executor
are finished. The old foundation executor is being replaced. Then execute own
answers in a fresh kernel, independently rebuild all native fits and predictions,
inspect the rendered lesson, and publish/integrate only after those checks pass.
Selected-topic rows remain UNIMPLEMENTED until that work finishes. The global
course goal remains open with 93 selected requirements still UNIMPLEMENTED.

The following foundation/protocol checkpoint is historical; its statements that
no digit model has been fitted have been superseded by the paragraph above.

## Current work — 16.9 foundation complete, digit study pending

This is the next active work after the completed 11.5 checkpoint below. The full
course goal remains active. Original chapters and the selected syllabus remain
binding. No subagents were requested or used. Preserve the dirty tree and the
user's unrelated GPU process. The new work runs on CPU.

The new development directory is
`chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin`.
The canonical `lesson.md` has **39 cells, 17 code cells, ten exercise groups,
twelve student functions, thirteen immediate check calls, six controlled figures
and one editable play**. This is a mathematical foundation, not a released
native section. Influence, TracIn and GhostSuite remain UNIMPLEMENTED.

`attribution_methods.py` exposes per-example logistic losses/gradients,
independently weighted objectives and derivatives, a Hessian-vector product,
checked conjugate gradients, independent batched Newton refits, signed upweight
influence, exact SGD states, exact/Taylor path accounting, TracInCP and affine
gradient products. `class_matched_removals` is also implemented and tested but
still needs its learner exercise. All core method bodies used by the foundation
are visible in the lesson's solution dropdowns. `build_notebooks.py` fills stubs
only from those answers; `execute_learner.py` currently reports foundation scope.

The own-answer run at
`/home/bitwise/.cache/arena_extended/data_attribution/own-foundation-v3`
passes all seventeen code cells in **4.127 s**, with all eight inputs unchanged
during execution. The preceding polished v2 run passes in 4.517 s. All six
current figures are byte-identical to inspected v2 figures. Browser checks open
33 dropdowns and render 36 math expressions without errors. All seven v2 views
and the credited v3 data/derivation views were inspected. The 17 current method
and data tests pass, including independent SciPy root, autograd, exact quadratic,
telescoping, corrupted-trajectory and source-data checks.

The six figures show real labelled digit images, curvature and CG residuals,
actual refit versus influence curves with a local-derivative zoom, exact versus
Taylor training-path contributions, all declared foundation checkpoint choices,
and a learning-rate/query play. Controls include identity curvature, zero solve,
wrong regularization normalization, omitted regularizer in SGD, exact path and
smaller learning rate. In particular, the local zoom exposes a nonvanishing
error for the mismatched normalization. These fixtures establish the mathematics;
they do not yet establish real-digit attribution performance.

The first test collection failed on a module import path, which was corrected.
An independent five-row refit test then exposed a Newton line search stalling at
3.56e-10 because rounded losses rejected the final useful step. An eight-machine-
epsilon objective allowance fixes that issue while retaining the 1e-10 gradient
criterion. The initial figure review found overlapping image titles and an
insufficiently resolved derivative comparison; both were repaired. Preserve the
failed/intermediate evidence rather than overwriting it.

Read-back archives are under the part's `development_evidence`:
`own_foundation_v1` (17 members), `own_foundation_v2` and `own_foundation_v3`
(27 members each), plus `checkpoint_v1` for tests, current source/protocol and
integration logs. No external tensor payload is needed for these foundation
archives. Raw UCI source files stay in
`/home/bitwise/.cache/arena_extended/data_attribution` with pinned hashes.
Do not rerun the one-time cache writer over `lesson.md`; it predates the polished
figures, wrapped prose and explicit dataset credit.

### Prospective real-data protocol

The full primary methods and relevant supplements for Koh/Liang influence and
Pruthi et al. TracIn were read, including the regularization convention, exact
single-row path accounting, minibatch limitation, checkpoint heuristic and
factored affine gradient product. The original 0.3 optimization learner and
existing 16.7 exercise/solution/page were inspected before the plan was written.
Keep the released 16.7 lesson unchanged.

`native_protocol.md`, `attribution_data.py` and `data_manifest.json` specify the
next study before any digit-model fit. UCI's original training/test files are
pinned by SHA-256. The test file matches sklearn's independent bundled copy.
Digit 1/7 selection is outcome-independent: 128 training, 48 ranking and 64
final examples, with original train/test writer groups separated. No exact
pixel duplicates occur among selected images. The clean and deliberately
12/128-flipped training-label conditions retain original labels and row IDs.
The manifest SHA-256 is
`a009ed80855932e774a427cc0b727cb30cfdfa7c228b1d5d7b7dee8f569b0a7c`.

Use 65 parameters (pixels / 16 plus bias), float64, lambda .02 including the
bias, and independently converged Newton refits from zero. Six complete SGD
paths cover clean/corrupted labels and order seeds 0/1/2, each with 40 epochs,
rate .05 for twenty epochs and .01 for twenty. Retain all/early/late/last epoch
checkpoint choices. Query losses omit regularization; the training row gradient
includes it so deletion matches the normalized fixed-penalty objective.

For each label condition, refit all 128 leave-one-out models and twelve prescribed
weight-perturbation examples at eight sizes. Retain all damping and convergence
controls. Forty-eight class-matched deletion policies cover influence, gradient,
signed pixel similarity, actual ranking-query leave-one-out, twelve checkpoint
policies and 32 random seeds. Evaluate budgets 2/4/8/16/32. Ranking queries alone
choose deletion sets; fixed final galleries use the first two source rows per
class. All cases, errors and random draws remain. Details and exact source
citations are in the protocol. No digit model has been fitted, and no digit-query
loss has been evaluated. Raw final data/labels are loaded for hashing only.

After own v3, the protocol's opening was clarified to say "digit-model" fitting
and distinguish the five-row mathematical fixtures. This changes no scientific
setting or learner code. `protocol-scope-clarification.json` records that exact
prose-only difference. The original protocol bytes remain in all run snapshots.
No repeated model execution is needed for that clarification.

### Concrete completion route

1. Implement a readable native orchestration module whose method calls can be
   copied into the notebook and bound to the student's functions. Snapshot every
   scientific source and the prospective manifest before fitting. Begin with
   ranking queries only; do not select settings from final outcomes.
2. Run both label conditions, all six SGD paths, all prescribed reference,
   deletion and reweighting refits. Verify each convergence residual, exact path
   telescoping, all first-order steps and individual-versus-batched refit parity.
   Keep the actual states, IDs, scores, losses and probabilities in run artifacts.
3. Append the class-matched removal exercise and visible native analysis to the
   learner: actual-image helpful/harmful galleries, all leave-one-out scatter
   panels, wrong-sign cases, damping/convergence curves, checkpoint/path comparison
   and final matched-deletion curves with every random control. Add a real query
   play. Derive the normalization and separate the different estimands in prose.
4. Finish and execute the complete own-answer notebook, including the predeclared
   final stage only after every fit and deletion policy is frozen. Do not claim
   checkpoint scores are exact removal effects or that this convex digit model
   reproduces large nonconvex neural experiments.
5. Add an independent fresh source reload, retraining and every-counterfactual
   reproduction. Compare actual outputs, not a saved scalar report. Inspect the
   complete rendered learner and all native figures, then publish paired notebooks,
   page/assets and the normal CPU report through the existing course gates.
6. Only then update the corresponding selected-topic rows. GhostSuite is
   conditional on a full visible gradient-product learner/reference comparison;
   this foundation alone is not sufficient. Datamodels, dataset cartography,
   multi-step In-Run Shapley and Adam-aware attribution remain separate open work.

The last complete course checkpoint has 43 TAUGHT, 36 PREFLIGHT_ONLY, 20 TOY_ONLY,
13 DEFERRED and 93 UNIMPLEMENTED requirements. The separate readability gate
still reports 78 older issues, with none in this new part. Do not narrow the full
course goal to this foundation or to passing structural checks.

## Historical initial proposal

This is a proposed continuation of the existing syllabus, not a completed
section or a frozen experiment. Finish the 11.5 publication gates first. Keep
the full course goal active and original chapters unchanged.

## Existing material inspected

The exercises and solutions for 16.7, its instruction page and README were read.
That learner already includes an eight-row logistic organism, exact coalition
training, a checkpoint-gradient score, a damped influence baseline and matched
leave-one-out controls. The roadmap's literal claim that no influence code
exists is therefore too broad. It still lacks a deep, separately validated
influence/TracIn lesson with identifiable real training examples and a full
retraining comparison. Do not promote those requirements from a README claim.

Original `chapter0_fundamentals/exercises/part3_optimization/0.3_Optimization_exercises.ipynb`
cells 4, 9, 22 and 27 were inspected for the learner arc: derive the update,
implement it visibly, inspect the trajectory, then change a parameter yourself.
Preserve all original files. The closest extension prerequisite is 16.7.

## Learner plan

Current learner failure: the distinction between a local infinitesimal change,
a recorded gradient trajectory and actually leaving out a training example is
buried in a small Shapley baseline. No gallery links those differences to real
training images and measured retraining outcomes.

Target section claim: a training example can help one prediction and hurt
another; an approximation must be judged against its stated perturbation and
an actual retrained model. Influence and checkpoint-gradient scores answer
related but different questions. This proposed section must not equate either
one with exact Data Shapley.

Signature result to create: a real handwritten-digit training-image gallery
with helpful/harmful rankings, predicted versus measured leave-one-out effects,
and retraining curves after removing selected or random groups. Keep failed
rankings, ties and wrong-sign predictions visible. Begin with an exact quadratic
oracle before moving to a trained classifier.

Student-implemented functions: per-example losses/gradients; regularized Hessian
and Hessian-vector product; a checked linear solve or conjugate-gradient routine;
implicit upweight influence; weighted checkpoint gradient alignment; independent
refit and effect metrics. A ghost dot-product identity can be included if it
remains part of the same visible gradient calculation, with autograd parity.
Do not claim that optional library integration or multi-step In-Run Data Shapley
has been taught merely because a gradient product was implemented.

Controls/baselines: exact quadratic response, small positive and negative
upweight finite differences, converged versus deliberately unfinished fits,
damping/conditioning checks, leave-one-out refits, all declared checkpoint
choices, input-neighbor and random-selection baselines, controlled label changes,
and class-matched deletion budgets. Separate ranking data from final evaluation.
Explicitly derive the objective normalization after removal; fixed-denominator
upweighting is not silently the same objective as mean-loss leave-one-out.

Play cells: change the evaluation example and watch the helpful/harmful gallery
change; change perturbation strength or deletion budget and observe where the
linear approximation breaks. These must run the relevant computation.

Files to modify: create a new append-only 16.9 companion (proposed directory
`chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin`),
its canonical lesson, visible methods, checks, data protocol, executor,
reproduction and instruction page. Link the existing 16.7 prerequisite. Keep
that released lesson's scientific code and evidence intact. Update only the
corresponding selected-topic registry rows after full learner, visual and
independent real-data verification passes.

Tests to run: exact derivatives and gradients against independent formulas;
Hessian-vector versus dense/autograd products; solver residual and failure
cases; influence versus infinitesimal independently refitted models; exact
one-step loss-change approximation; checkpoint weighting and invariance checks;
full own-answer execution; fresh data/model training and all prescribed
retraining comparisons; original preservation and normal course gates.

## Sources and open decisions

Primary source discovery is complete, but the full methods and supplements
must be read before implementing or freezing the protocol:

- [Koh and Liang, Understanding Black-box Predictions via Influence Functions](https://proceedings.mlr.press/v70/koh17a.html).
  The local perturbation and inverse-Hessian formulation should be derived with
  its assumptions, rather than presented as an exact deletion identity.
- [Pruthi et al., Estimating Training Data Influence by Tracing Gradient Descent](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html).
  Distinguish the path-based definition from checkpoint and gradient
  approximations, and verify learning-rate and batch normalization conventions.
- [scikit-learn load_digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
  A candidate local dataset. Inspect the actual bundled data, provenance and
  license before choosing splits or fitting any model. No data split, label
  corruption, model size, seed count or final example has been chosen yet.

The narrow first targets are `data_attribution.influence_functions` and
`data_attribution.tracin`. `data_attribution.ghostsuite` is conditional on a
complete visible ghost-gradient exercise and reference comparison. Datamodels,
dataset cartography and other selected data-attribution topics remain separate
requirements unless their full learner arcs are actually implemented.
