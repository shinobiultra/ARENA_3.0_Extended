# Next selected learner: Contrast-Consistent Search

## Current checkpoint — 11.5 Contrast-Consistent Search complete

This checkpoint supersedes all earlier CCS development and publication status
below. Continue the full active goal. Original chapters and the selected syllabus
are binding; legacy extension implementation may be replaced. Preserve the dirty
tree. No subagents were requested or used. Do not broadly stage, reset, commit
or push. The user's unrelated GPU job remains untouched; CCS runs on CPU.

### Current learner

`chapter11_representation_geometry/exercises/part5_contrast_consistent_search/11.5_Contrast_Consistent_Search_exercises.ipynb`
is published with its paired solutions notebook, fifteen assets and
`chapter11_representation_geometry/instructions/pages/05_[11.5]_Contrast_Consistent_Search.md`.
The canonical source is the part's `lesson.md`. Both course configurations,
root README, method registry, hard-exercise ladder and style registry include it.

The 65-cell learner contains 32 code cells, nine exercise groups, twelve
student-implemented functions with immediate checks, fifteen controlled figures
and two editable plays. Help, expected-output, interpretation and solution
dropdowns follow the original ARENA examples. The method is visible: mean-only
branch centering, the original asymmetric loss, independent restart fitting,
unlabelled selection, calibration-only orientation, PCA, padding-safe token
capture, paired opinion changes and actual full-vocabulary metrics. Every
restart and random control remains visible. Plays vary the exact distractor's
strength and an actual GPT-2 review with an independently changed reader opinion.

### Scientific result and scope

Frozen GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` runs on CPU in
float32 with eager attention; reporter fitting uses float64. The predeclared
split has 256 training base reviews and 64 each for calibration, development
and final. Final uses new frames and adjective pairs. All three sites, two
wrappers and three text conditions are retained. Ten centered, ten raw and ten
shuffled-pair CCS restarts, PCA, supervised logistic and 32 random directions
produce 64 reporters per condition. Training loss selects restarts; only
calibration labels orient unsupervised reporters. Final is now evaluated and
was never used for selection, refitting, recentering or orientation.

Plain customer layer 8 CCS gets 64/64 final targets right. When an independent
reader's opinion is added, every selected CCS condition has 50% target accuracy
and flips on all 64 paired opinion changes. Supervised controls retain
103/128 to 128/128 target accuracy. The figures expose this failure beside raw,
shuffled-pair, PCA and every matched random control; those controls sometimes
find useful structure too. Restricted two-answer output accuracy is separate
from GPT-2's actual next token: none of the 512 final unrestricted winners is
the requested sentiment word, and mean candidate probability mass ranges from
about 0.044% to 1.14%. The actual review gallery shows the examples and outputs.

These are controlled adjective-based reviews, not a truth or belief benchmark,
causal necessity result, whole-paper replication or claim about larger models.
There are 64 distinct final base reviews, 512 repeated contrast pairs, 98,304
reporter scores, and 3,586 actual full-vocabulary prefix outputs across all
splits plus play. Do not call the reporter scores full-vocabulary predictions.

### Current execution and evidence

All paths in this paragraph are under `/home/bitwise/.cache/arena_extended/ccs`.
`full-own-answer-v3` passes all 32 code cells in **224.500 s** with all thirteen
scientific inputs unchanged. `fresh-native-v4` independently reloads the model,
recaptures features and refits all **1,152 reporters** in **228.792 s**. All
**5,408 tensor comparisons**, scalar comparisons and token winners match exactly;
all fourteen fresh inputs and model weights remain unchanged, with no gradients.
Every one of the fifteen PNGs and the complete native summary is byte-identical
to the polished v2 learner. V3 only improves method-test explanations and success
feedback; test inputs, oracles and mathematical predicates are unchanged.
The current browser render opens all 34 dropdowns and renders 33 math expressions
without errors. Fifteen figures and eight screenshots have been inspected.

The current normal `verification_report.json` is accepted with no known failures.
It explicitly reuses `fresh-native-v4`: `native_evidence_reused=true` and
`native_model_reloaded_in_this_report=false`. Before reuse, it verifies all
fourteen scientific sources, all 43 fresh tensor files, learner identity and
published notebook/page/figure hashes. Running without the reuse environment
variable performs another independent native run. This avoids redundant model
execution during a publication-only refresh without claiming a new reload.

Read-back-verified archives in the part's `expected_outputs` are
`full_own_answer_v3`, `fresh_native_v4`, `normal_report_source_v2` and
`release_gate_v3`. The source archive preserves all 47 exact normal-report input
files plus the report. Each full model run has 43 external tensor files totaling
851,618,524 bytes; their hashes, sizes and persistent cache paths are recorded
in the archive manifest. Do not delete these cache directories.
Current normal-report fresh evidence SHA-256:
`34d376e70a5868ce3204eb73534eb6b702708a8b33cd536ef28f371aba916df3`.
Current learner evidence SHA-256:
`1f1407df131b90e097b5aad5b51f64c26f48b3e6a85a03561cbef30030d32cb3`.

Historical own v1/v2 and fresh v1/v3/normal-v1 archives remain. Fresh v2 was
interrupted by SIGTERM after sixteen cases; the cause is unknown. Its original
RUNNING record and explicit interruption record are archived under
`development_evidence/fresh_native_interrupted_v2`; it is not release evidence.
No CCS model process remains running.

### Checks and remaining work

The current focused suite passes **96 tests**. Original preservation, course
surface (13 chapters), style depth (50 pages), exercise ladder (49 rows) and
evidence contracts (52 reports) pass. Original chapters 0–4 remain unchanged
against `f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0`.
The 205-item roadmap now has **43 TAUGHT, 36 PREFLIGHT_ONLY, 20 TOY_ONLY,
13 DEFERRED and 93 UNIMPLEMENTED**. Its 93 blockers are exactly the remaining
UNIMPLEMENTED items. The separate readability audit has 78 older issues and
none in CCS. Passing the current section's checks does not complete the course.

Next selected work is influence functions and TracIn. The concrete preparation
is in `docs/arena_salvage/2026-09-06_data_attribution_next_plan.md`: extend the
existing data-attribution material with visible influence and checkpoint-gradient
implementations, real training images, all leave-one-out retraining effects and
matched deletion controls. Primary papers and the original optimization learner
have been located; the full paper reading and a prospective native protocol
remain to do. Do not promote those requirements on the existing 16.7 toy study.

## Historical foundation checkpoint, 2026-09-06

Part 5 now exists at
`chapter11_representation_geometry/exercises/part5_contrast_consistent_search`.
It is an unpublished development section. The roadmap entry remains
UNIMPLEMENTED: a complete native learner and final reproduction do not exist yet.

`lesson.md` has a complete 32-cell / 16-code-cell foundation, six implementation
groups, nine immediate check calls, seven figures and one play. Methods in
`ccs_methods.py` are visible in its own solution dropdowns. Ten standalone
method tests pass. `reference_parity.py` compares mean-only centering, original
loss and gradients with three pinned-author cases (maximum errors below 1e-12).
The original asymmetric loss is retained, including its lack of global soft-score
complement symmetry. Orientation and its labelled calibration are separate.

Independent restarts are batched to reduce overhead; tests compare them against
separate runs, including histories and the final post-update loss. The default
exact experiment retains all ten seeds. Seed 0 reads the target; the other nine
read the independent distractor. Unlabelled loss selects seed 6, loss approximately
.000149, with 50% target accuracy even after retrospective orientation.

Evidence root: `/home/bitwise/.cache/arena_extended/ccs`.

- `own-foundation-v1`: complete own-answer kernel PASS, 5.430 s, 7 figures,
  six source inputs unchanged. Its crowded seed labels were refined afterward.
- `own-foundation-v2`: current foundation PASS, 9.172 s, 7 figures, unchanged
  sources. All figures inspected individually; current browser review opens all
  23 dropdowns and renders 31 math expressions without errors. Four screenshots
  and exact source/figure hashes are recorded in `visual_review.json`.
- `native-development-v1`: actual GPT-2 capture and fitting PASS, 197.444 s,
  all 18 site/wrapper/variant cases and all 64 reporters per case retained.
  All six source inputs and model weights unchanged; no model gradients.
  It contains 18 capture files and 18 fit files. Final was not evaluated.
- `author-source-v1`: pinned upstream source, notebook, MIT license and hashes.
- `data-manifest-v1.json` and `data-manifest-v2.json`: only development text
  differs, due to the grammatical correction described below.

Read-back-verified archives for current foundation and native development are
in `part5_contrast_consistent_search/development_evidence`. Native tensors remain
at manifest-named persistent paths outside Git. Never delete them to tidy up.

## Frozen native design and observed development

`ccs_data.py`, `ccs_native.py`, `run_development.py` and `native_protocol.md`
implement the predeclared protocol. Pinned GPT-2, CPU float32/eager attention,
reporters float64. Base reviews: 256 train, 64 calibration, 64 development,
64 final. Final adjectives and frames differ from training. Three text variants
(plain, neutral reader, independent opinion) cross two wrappers and three sites
(hidden-state indices 4, 8 and 12). The distracting opinion takes both signs for
every review; repeated opinions/wrappers are paired measurements of the same base.

All 18 cases retain ten centered CCS fits, ten raw fits, ten corrupted-pair fits,
one PCA, one supervised probe and 32 matched-norm random directions. Each
unsupervised reporter's orientation uses calibration labels only. Restart
selection uses final unlabelled training loss. No settings are selected on final.
The data contain 512 final contrast pairs, representing 64 base reviews; across
three sites and 64 reporters there will be 98,304 reporter scores, not 98,304
independent native full-vocabulary predictions. Keep these quantities distinct.

V1 development, layer 8: selected CCS gets 62/64 plain customer reviews and
64/64 plain quoted reviews correct. With the independent reader, every selected
centered CCS reporter at every site/wrapper has 50% target accuracy and 100%
retrospectively oriented distractor agreement. Every supervised probe in those
six distracting cases gets 100% target accuracy. Other sites and controls are
retained, including low-loss plain fits that miss the target. Do not describe
these as final or broad knowledge-elicitation results.

After the completed v1 run, one development frame was changed from
`What a {adjective} {item} this turned out to be.` to
`This {item} turned out to be {adjective}.` to remove the incorrect article before
vowel-initial adjectives. Train, calibration and final manifests match v1
exactly. V1 data/results remain preserved. No scientific setting changed. The
next complete own-answer run must freshly recapture the corrected development
sentences. No additional development search is needed before that run.

## Concrete completion route

Current native-writing plan: the learner's failure is the missing bridge from
the executed foundation to actual paired model activations. Keep the section's
claim about feature identification. Add student implementations of last-token
gathering, paired opinion-flip measurements and full-vocabulary versus restricted
answer metrics, with immediate checks. Show all fitting/control/evaluation loops
in the notebook using its own CCS functions. Add controlled site/seed plots and
row-ordered actual text comparisons, plus a native play using a development
example. Modify only part 5 source and its eventual publication files. Execute
the entire current learner, then fresh recapture/refitting, rendered review and
the same course/preservation gates as 11.4. This implements the existing plan;
no native scientific settings or final choices change.

1. Append the native stage to `lesson.md`. The model/data setup, fit/evaluate
   loops and plots must call the learner's visible functions. `run_development.py`
   is an experiment reference, not a substitute for notebook implementation.
   Add an immediate native token/activation check and a paired opinion-flip
   metric so the student can see which factor changes a reporter's answer.
2. Preserve all 18 cases and 64 reporters per case. Show target/distractor
   recovery, all restarts, raw/centered/corrupted-pair behavior, PCA/supervised/
   random controls, full native winners and restricted answer probabilities.
   Freeze the text-gallery row IDs and default play on development before
   first final evaluation. Keep calibration orientation fixed on final.
3. Extend `execute_learner.py`'s six-input foundation snapshot to cover all native
   source/data/protocol files and the unchanged prior lesson's supervised-probe
   implementation. Execute a fresh kernel with own answers, one frozen model,
   current train/calibration/development/final capture and every fit. Record
   full-vocabulary prefix logits, paired activations/tokens, parameters and
   actual reporter scores. Final has not been accessed by a model yet.
4. Implement independent fresh model recapture and refitting of every reporter.
   Compare actual outputs/tokens/activations, maps, orientations and all final
   scores; source-bound archives cannot be accepted as fresh execution. Add
   meaningful guards for omitted/extra cases, bad tokens/shapes/nonfinite data,
   changed sources and incorrect metrics. Keep frozen model weights unchanged.
5. Inspect every native figure and current rendered math/dropdowns. Publish
   paired notebooks, page/assets, normal CPU report and course navigation only
   after own-answer and fresh evidence pass. Follow 11.4's publication/evidence
   contract while using CCS-specific quantities and claims. Then promote only
   `representation.ccs` and rerun the existing integration/original-preservation
   gates. Until then the roadmap remains at 94 UNIMPLEMENTED requirements.

Do not rerun the one-time cache script `write_foundation.py` over the current
lesson: it predates the reviewed annotation/prose improvements. The canonical
source is now the part's `lesson.md`. No CCS native process remains running at
this checkpoint. The local preview is `http://127.0.0.1:8768/ccs-foundation-v2.html`.

## Original design record

Selected requirement: `representation.ccs`, in the final amended syllabus at
`Extension-Roadmap.md:2840-2874`. Proposed home: chapter 11, part 5. This adds no
new syllabus topic. The requirement remains UNIMPLEMENTED until the complete
learner, native experiment and release evidence exist.

Current learner failure: there is no CCS exercise notebook, paired solution or
instruction page. The closest originals inspected are 1.5.3 OthelloGPT's probe
motivation, cosine-similarity and direction exercises, and help dropdowns. Use
their concrete examples, progressive tensor reasoning, exercise timing blocks
and interpretive questions. Do not replace instruction with experiment reports.

Target section claim: opposite-answer consistency can find a useful linear
feature without fitting labels, but low CCS loss does not identify that feature
as truth. Fitting without labels, assigning semantic orientation, selecting a
model and evaluating against labels are distinct operations.

Signature result to create: an exact two-feature example in which either the
target or an independent distractor satisfies the objective, followed by an
actual pretrained-model comparison of unlabelled loss, target accuracy,
distractor agreement and output behavior. Retain all restart outcomes. Show
actual contrast-pair sentences and probabilities, not just aggregate counts.

Student-implemented functions: fit/apply separate branch centering; original
CCS loss and its two terms; paired inference; independent-restart optimization;
label-free restart selection; calibration-only orientation; paired-difference
PCA; supervised probe baseline and evaluation. Keep these operations visible in
the notebook. Batch independent restarts and cache model activations once to
reduce execution time without replacing fresh reproduction.

Controls/baselines: constant half probabilities, constant confident predictions,
answer-token offsets before/after centering, independent distractor feature,
global sign alternatives, random directions, paired-difference PCA, supervised
linear probes, corrupted pairings and model-output scoring. Native choices and
all final splits must be frozen before final evaluation. Use an independently
balanced distractor rather than inferring truth from the selected projection.

Play cells: vary distractor strength and initialization; compare ordinary and
distracting native prompts using a fixed fitted reporter. Do not orient a new
reporter using the same final labels whose accuracy is reported.

Files to modify: append new `part5_contrast_consistent_search` methods, tests,
data/protocol, canonical lesson, paired notebooks, native capture/reproduction
and publication files. Add its page, figures, navigation and registries only
after actual learner/native execution and rendered review. Preserve all
completed chapter 11 scientific files and original ARENA chapters unchanged.

Tests to run: exact loss values and analytic gradients; branch centering with
held-out data; paired inference and ties; label-free optimization and restart
selection; independent-restart batching equivalence; pinned author primitives;
two-feature counterexample and information limits; native capture/token
alignment, repeated fresh fitting and full output comparison. Then run normal
section reporting and preservation/style/course/evidence checks.

## Sources inspected and implementation decisions

- [Burns et al., original CCS](https://arxiv.org/html/2212.03827v2): section 2.2,
  normalization and optimization details in appendix G, and the reported use of
  the better test-set orientation. Teach the original asymmetric confidence
  term; it is not exactly invariant to complementing arbitrary soft scores.
- [Farquhar et al., identification limits](https://arxiv.org/html/2312.10029v2):
  arbitrary-feature counterexample, the distinction between representability
  and actually learning a feature, and the corrected symmetric loss used for
  their second theorem. Do not attribute that theorem to the unmodified loss.
- [Pinned author source](https://github.com/collin-burns/discovering_latent_knowledge/tree/be10d7b912b17e8eacb45de794e9dca044e4ed37):
  `CCS.ipynb`, `utils.py` and MIT license are cached with SHA-256 hashes at
  `/home/bitwise/.cache/arena_extended/ccs/author-source-v1`.

The author's demonstration centers train and test independently and reports the
better test orientation. Our primary held-out evaluation will reuse training
means and orient using a separate labelled calibration split. Show these as
deliberate protocol differences, not numerical parity claims. Primitive parity
can compare loss/gradients and training centering. The pinned `utils.py`
constructor assigns `self.probe = self.initialize_probe()` even though that
method returns nothing; the notebook version avoids that assignment. Do not
claim the complete unmodified script class executed successfully. Extract only
the inspected, hash-verified primitives for the independent reference check.

No native model/data result has been observed for CCS at this checkpoint.
Choose a narrow task whose ground truth is explicit; do not describe a small
model exercise as a replication of the paper's large-model benchmark or a
general-purpose truth detector.
