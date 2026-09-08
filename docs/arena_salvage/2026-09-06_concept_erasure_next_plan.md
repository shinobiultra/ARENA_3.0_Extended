# Concept erasure: completed learner and original plan

## Completed checkpoint, 2026-09-06

The plan below has been implemented in
`chapter11_representation_geometry/exercises/part4_concept_erasure` and the
paired 11.4 notebooks/page are published in both course configurations. The
canonical `lesson.md` has 52 cells, 24 code cells, eight implementation groups,
11 immediate checks, 12 controlled figures and two play cells. Original ARENA
1.5.3's probe explanations, exercise blocks and interpretation dropdowns were
inspected before authoring. Original chapter files remain unchanged.

Author parity uses revision `9f51753821316a1edacf78b52b464ab26d40e60a`
of `EleutherAI/concept-erasure`, with explicit no-shrinkage/no-trace-constraint
settings and a fixed singular-value tolerance. Three reference cases and an
independent constrained quadratic oracle pass. The four-point construction
separates old-probe failure, covariance removal and least-distortion erasure.
The nonlinear counterexample explicitly distinguishes convex-loss guardedness
from hard-threshold accuracy and statistical independence.

Frozen native protocol: GPT-2 revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32, erasers and probes
float64. The 512 eraser-fitting rows, separate 320 probe-fitting rows, 160
development rows, 288 final role rows and 384 final unrelated-object rows were
defined before native access. Two prescribed sites, block 6 subject and block 9
end, retain all five base methods and all 16 random draws for each of three
control families. There are 106 conditions and 71,232 final native outcomes.
All original model errors are retained. Empirical covariance and the prescribed
.01 covariance mixture remain separate fits, not final-set choices.

Actual controls are no intervention, old-probe coefficient removal, orthogonal
cross-covariance removal, random rank-one edits, per-example displacement-norm
matched random edits for both LEACE fits, independent probe fitting and an
unrelated object task. Shuffled-label fitting is suggested as further research;
it is not one of the executed 106 conditions. Plays vary covariance anisotropy
and native intervention strength. This supersedes the broader provisional play
and control list below.

The important native result is a limitation students can investigate. Early
empirical LEACE eliminates fitting covariance, yet a newly fitted reader gets
286/288 final roles correct. Late empirical LEACE has 50% reader accuracy but
poor cross-entropy (5.205), so that scalar is not proof of removed information.
Native recipient accuracy changes from 260/288 to 228/288 early or 214/288 late;
unrelated-object accuracy changes from 192/384 to 191/384 or 96/384. The lesson
shows every random control, refitting loss, edit size, native KL and actual token
predictions alongside these counts. No all-layer scrubbing, unique mechanism or
unrestricted information-erasure claim is made.

Evidence root: `/home/bitwise/.cache/arena_extended/erasure`.

- `full-own-answer-v1`: full learner PASS, 324.554 s, all 12 scientific inputs
  unchanged while executing. Later prose interpretation changes preserve every
  executed code cell's AST and have their own current rendered review.
- `fresh-native-v1`: independently reloads the model, recaptures activations and
  refits every eraser and probe, PASS 329.615 s. All 71,232 full-vocabulary
  outputs and winners agree exactly; all 13 reproduction inputs are unchanged.
- `full-own-answer-v1/visual_review.json`: all 12 figures inspected; current
  rendered lesson has 28 working dropdowns and 42 math expressions without
  render errors. Actual screenshots and figure hashes are retained.
- Normal report's `fresh_reproductions/run-uv_8ru7l/fresh_cpu`: another complete
  fresh refit/reproduction, PASS 302.198 s, zero logit difference and winner
  mismatches. All weights are unchanged and model gradients are absent.

Readable evidence archives under `expected_outputs` are read back and hash
verified. Large native tensors remain at manifest-named persistent paths outside
Git. The normal report passes, all 70 integration/reproduction-guard tests pass,
and preservation, course surface, style, hard ladders and report evidence pass.
Only `representation.concept_erasure` is promoted: 205 requirements now contain
42 TAUGHT, 36 PREFLIGHT_ONLY, 20 TOY_ONLY, 13 DEFERRED and 94 UNIMPLEMENTED.
The separate readability audit still has 78 older issues, none in this part.
The full course remains unfinished. Next selected work: `representation.ccs`.

## Original proposal (historical)

This continues `representation.concept_erasure`, already selected in the final
amended syllabus (`Extension-Roadmap.md:2871-2874`). It does not add a new topic
or promote the requirement. Proposed home: chapter 11, part 4, after the completed
geometry, alignment and DAS lessons. Original chapters remain append-only.

The concrete question is: **if a representation stops supporting a linear probe,
what has actually been removed, and what happens to the model's answers?**

Use LEACE as a closed-form implementation of the selected concept-erasure topic.
The primary paper's definitions and Theorems 3.1–4.3 distinguish convex-loss
linear guardedness from unrestricted information removal. Its eraser generally
uses an oblique projection. Study its singular-covariance derivation and official
implementation before coding numerical cutoffs or making uniqueness claims.
Sources inspected during planning:

- [Belrose et al., LEACE (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/d066d21c619d0a78c5b557fa3291a8f4-Paper-Conference.pdf), definitions, central derivation, oblique projections, native experiments, sequential scrubbing and limitations.
- [Author repository](https://github.com/EleutherAI/concept-erasure); this was still uninspected at proposal time. The completed checkpoint above records the pinned comparison.

## Learner progression

Start with a small anisotropic representation in which projecting out a fitted
probe's coefficient differs from equalizing class means. Have the student derive
and implement centered cross-covariance, a tolerance-aware whitening operator,
the eraser and its affine application. Plot original and edited points with
arrows, class centroids and a useful side variable. Compare distortion against
orthogonal removal of the cross-covariance span and an independent constrained
least-squares oracle. Include singular and translated data explicitly.

Then build an exact counterexample with equal class centroids but nonlinear
information. Compare a freshly fitted convex-loss linear probe, a nonlinear
readout, majority prediction and shuffled labels. Do not equate a zero empirical
covariance with statistical independence, chance under every possible metric,
or guaranteed erasure on new data.

The native extension should use frozen pinned GPT-2 and the already-tested
capture/full-forward plumbing from 11.3, while keeping every core erasure method
visible in the notebook. Specify new train/development/final prompt families
before inspecting intervention results. A narrow name-role task provides a
connection to DAS: edit one role representation, refit probes after erasure,
continue the model and show actual next-token probabilities and answers.
Separate reduced probe performance, target-task damage, unrelated-task damage
and global output-distribution change. The old DAS final examples are prior
knowledge, not a new untouched final set.

## Controls, plots and checks

Keep no-op, orthogonal cross-covariance removal, projection of the fitted probe,
matched-rank random removals, shuffled training labels and useful-side-variable
checks. For oblique edits, show both rank and actual edit magnitude; rank alone
does not match intervention size. Choose settings on development only. Retain
all random draws, native errors and final cases. Compare fitting-set guarantees
with new-template/name-pair results and track rank/covariance conditioning.

The main native figure should put post-erasure probe loss, correct-name behavior
and collateral changes beside each other. Include a per-example token panel and
plays for covariance anisotropy, nonlinear information and a new native prompt.
Single-site erasure is a scoped experiment; it does not establish all-layer
scrubbing or complete removal from the transformer.

Immediate tests must cover hand-computed centroids, affine mean preservation,
zero cross-covariance, idempotence, zero-rank and singular covariance, independent
constrained-fit optimality, unchanged unrelated coordinates when justified,
refitting after erasure, native no-op and full-forward parity. Compare the visible
implementation against a pinned author implementation on well-conditioned and
rank-deficient examples. Preserve all differences rather than loosening cutoffs
to pass a single fixture.

## Execution and release

Reuse native plumbing as plumbing, not as a hidden method. Read the closest
original ARENA probe/toy-model lesson before writing the new learner. Develop
small exact examples first; freeze a native protocol before final evaluation.
Require full own-answer execution, independent fresh native output comparison,
rendered equations/dropdowns and controlled figure inspection, normal section
report, navigation and the same preservation/style/course audits as 11.3.
Keep source snapshots, failures and actual tensors in persistent cache.

At proposal time there was no concept-erasure code, fitted eraser or native final
evaluation. The completed checkpoint above supersedes that state.
