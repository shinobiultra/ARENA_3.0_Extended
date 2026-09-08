## Current checkpoint — 16.11 published and reproduced; full course goal remains active

Section **16.11 Dataset Cartography** is complete in
`chapter16_shapley_attribution_baselines/exercises/part11_dataset_cartography`.
The learner files are `16.11_Dataset_Cartography_exercises.ipynb`,
`16.11_Dataset_Cartography_solutions.ipynb`, readable `lesson.md`, and
`chapter16_shapley_attribution_baselines/instructions/pages/11_[16.11]_Dataset_Cartography.md`.
Twenty PNG figures are in that chapter's `instructions/assets/cartography_*.png`.
The section README links the lesson, notebook pair, primary sources and run commands.
Original chapters 0–4 remain pinned to
`f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0`, with zero preservation violations.
Do not reset/clean the dirty tree or disturb unrelated GPU work. No subagents
were used. All new model work ran on CPU using `uv run python`.

### Learner result and scientific scope

The published lesson has **71 cells, 30 executed own-answer code cells, nine
exercise groups, ten student functions, twenty figures and two editable plays**.
Each stub is completed only from its own following Solution dropdown. All core
methods and all study orchestration are visible and editable. Ten immediate
learner test calls give educational feedback; ten method tests and ten contract
cases support the learner. Browser review checked **32 open dropdowns, 16 rendered
math expressions and zero math errors**. Original 0.3 Optimization and adjacent
16.10 notebooks/pages were inspected before the learner plan.

Students implement stable epoch-ID alignment, assigned-label probability,
confidence/population variability/correctness, independent model-axis forward
passes, SGD with both logging conventions, stable region ranking, error retrieval,
set overlap, class-matched sampling and held-out evaluation. The forward/training
loop matches an independent ordinary torch.nn + torch.optim.SGD implementation.
NumPy independently checks actual native map coordinates. A mathematical play
shows that a probability history and its reversal have identical map coordinates.
A second play actually retrains a selected subset and a matched random subset.

The primary source is Swayamdipta et al. (EMNLP 2020),
https://aclanthology.org/2020.emnlp-main.746/ . Sections 2–5 and Appendix A.1
were read, as were the released author training and filtering implementations.
The paper defines epoch-end model parameters; the released training loop records
visited minibatch logits before that batch's update. The lesson retains both and
uses epoch-end measurements for the primary map. Author correctness is stored as
a count; the notebook displays its fraction of epochs. No silent equivalence is
claimed. This is a real small digit-network method experiment, not a reproduction
of RoBERTa NLP, natural OOD datasets or the full paper.

### Frozen data and protocol

The experiment was frozen at **2026-09-06T10:52:59.610544+00:00**, with zero native
16.11 digit fits. Frozen checksums:

- `protocol.md`: `91d9857f36b162730773cce3868b5ffe8145a7ea40f9516b0f58c7849a545637`
- `data_manifest.json`: `4677e7425a872b6cbc1caab44fa99f72480c95aeef6f34c6b13f2b10f4b84be4`
- `cartography_data.py`: `e31bebc991561d103c1e50bf3399c21361ac4201adffcce1fdf88007d5d9068e`

Do not casually reformat the byte-frozen data helper. The protocol and manifest
remain unchanged after observing outcomes. The data helper reuses only the pinned
source-download/checksum plumbing from 16.9, not its model or task selection.
UCI archive SHA256 remains
`0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0`.
Source training/test files contain 3823/1797 rows, 64 integer pixels plus label.
The source is Alpaydin and Kaynak (1998), DOI 10.24432/C50P49, CC BY 4.0.

Use all ten classes, selecting 300 training-file examples per class by sorting
SHA256 `arena16.11/v1/train/{zero_based_source_row}`. Sort selected rows by source
row before model training. The **3000 training / 823 same-source holdout / 1797
separate-writer holdout** splits have zero exact cross-training pixel overlaps.
Individual writer IDs are unavailable; the 823-image holdout is not guaranteed
writer-disjoint. Divide pixel counts by 16; no fitted preprocessing or augmentation.

For label-error diagnosis, choose 30 source rows per class by namespace `noise`
and change their label to `(label + 1) % 10`. This produces exactly 300 errors
while preserving class counts. Original labels and changed identities remain
available for diagnostic scoring only. A separate label permutation uses seed
161199, retaining coincidental unchanged labels.

Model: 64 → 64 ReLU → 10 MLP, plus a linear comparison. Float32 CPU, four threads,
deterministic algorithms, cold local-generator initialization. SGD lr .05,
momentum .9, weight decay .0001 on all parameters, batch 128 including the partial
last batch, thirty epochs, no scheduler/dropout/early stopping. Hidden weight
std sqrt(2/64), output std 1/sqrt(width), biases zero. The same local generator
subsequently supplies independent epoch permutations. Independent model losses
are summed after averaging within each model, preserving each model's gradient.

Five mapping seeds 161100–161104 train clean/noisy/shuffled MLPs and clean linear
models: **20 mapping runs**. Save every epoch's visited and epoch-end logits,
final weights, losses and metadata. Main map statistics use all thirty epochs;
additional prescribed windows are epochs 1–10 and 21–30, plus repeated final
snapshot. The latter has zero variability, whose correlation is left undefined.

Choose subsets only from clean MLP seed 161100's epoch-end map. Retain 300, 1000
or 2000 by highest confidence, lowest confidence or highest variability. Global
ranking ties use source order. Five new seeds 161200–161204 train each selected
set and five independently sampled, exact-label-count matched random controls;
five uniform random subsets per size and five full-pool baselines also run.
All sampling seeds are explicit in the protocol. This is **110 primary refits**.
Ten further refits train the one-third ambiguous and its matched random sets for
90 epochs, matching the full pool's 720 updates. Their first thirty-epoch loss
paths exactly match the corresponding thirty-epoch runs.

All selections and **120 final retrained parameter sets** are saved before any
holdout model predictions are computed. Then evaluate all 140 models on every
image of both holdouts. Main experiment: **140 training runs / 53550 optimizer
updates**, plus two separate full mapping checks and two default real-play fits.
The default play uses ambiguous/1000/seed166000, random selection seed166001,
thirty epochs; it explicitly reuses the holdouts for exploration.

### What actually happened

The full-pool retraining baseline averages **95.45% separate-writer accuracy**.
At 1000 retained images, high variability averages **86.27%**, its class-matched
random control **89.68%**, and uniform random **93.91%**. Low confidence averages
**94.95%**. At equal 720 updates, high variability improves to **88.46%**, while
its matched random control reaches **91.29%**. All five runs and every budget
remain in the plots. This is not a manufactured ambiguous-region win.

The easy 300-image set contains no digits 3, 5, 8 or 9. The one-third ambiguous
set has class counts `[5,73,80,255,53,129,28,25,224,128]`. Class-count controls,
classwise accuracy and paired gained/lost examples expose these differences.
Low-confidence noise retrieval at budget300 has precision/recall 0.8833–0.8967,
versus random expected precision0.1. Misses and false positives remain. A hard
source image is not declared naturally mislabeled.

Clean-map confidence seed correlations range about .966–.978; variability about
.761–.873. Seed0 online/epoch-end correlations are about .997 for confidence and
.718 for variability. The maps, image galleries and trajectories show why temporal
variation can include a smooth learning transition. The shuffled-label models
score near chance on original-label holdouts: separate-writer mean about9.35%.
The full mapping, selection and held-out outcomes are native saved tensors,
not reconstructed from verification-report metrics.

### Current execution, publication and archives

Persistent cache root: `/home/bitwise/.cache/arena_extended/cartography`.
Do not rerun `author_lesson_v1.py` or `polish_lesson_v3.py`: these are historical
one-off authoring scripts. Current canonical source is the part's `lesson.md`.

- `native-full-v1`: PASS, **4.232 s**, first complete scientific study.
- `own-full-v1`: FAILED before execution because a following conceptual proof
  also had the exact `Solution` label. The proof label was clarified; the own-
  exercise answer requirement was preserved. The failure/source are archived.
- `own-full-v2`: PASS **12.966 s**, 67 cells/28 code/18 figures. Visual review
  requested two more explicit measurement/generalization comparisons.
- `fresh-native-v2`: PASS **4.989 s**, complete fresh refits and exact comparison.
- **`own-full-v3` CURRENT**: PASS **13.115 s**, 71 cells/30 code/20 figures,
  fifteen source inputs unchanged. Eighteen prior figures are byte-identical;
  two added figures and the full image gallery were inspected directly. Four
  browser views were inspected, all32 dropdowns opened, all16 math expressions
  rendered. Native path: `figures/native-6dge0kjr/study` within this run.
- Executed notebook SHA256:
  `e05a24e40afbeb53f817f0f8d93b4a8ca1c6e2a0defcd4bef10d42472edc6333`.
  Execution JSON SHA256:
  `0ec5b37b95a75f09b113185c755120942bee072ab0a5f7ef2bec81ce0aef94e7`.
- **`fresh-native-v3` CURRENT**: PASS **4.889 s**; all **675 tensors /
  41903198 elements / 8904 scalar and ID values** agree exactly. Full native
  execution is4.142s. Reproduction JSON SHA256:
  `a97f82e44818eb07e6d2f81bdc77071f92a73bb09973dd7d6b55f224d8739273`.
- Normal verification report is **accepted**, no known failures, **50 source
  inputs** checked unchanged. It performs an entirely new CPU study at
  `report_reproductions/report-omyurfdg/fresh_cpu`, native4.485s/reproduce4.891s.
  `expected_outputs/fresh_reproduction.json` references this current normal run.
  Publication `learner_execution.json` also retains the original fresh-native-v3
  link; both are correct and complete.

Full reproduction checks use rtol1e-6/atol1e-7, exact metadata/types/IDs. Separate
full MLP/linear comparisons have max absolute differences6.68e-6/2.87e-6 within
rtol/atol2e-5. Eight condition/logging combinations × three map metrics are
checked independently in NumPy; max discrepancy about2.01e-7. All native payloads
are retained: `mapping.pt`, `selections_before_refits.pt`, `prepared.pt`,
`evaluated.pt`, `play.pt`, totaling **168015763 bytes** per complete study.

Readback-verified archive directories under the part:

- `development_evidence/native_full_v1` (14 readable members /5 external PT)
- `development_evidence/failed_own_answer_v1` (16/0)
- `development_evidence/full_own_answer_v2` (40/5)
- `development_evidence/fresh_native_v2` (30/5)
- `development_evidence/normal_report_source_v1` (52/0)
- `development_evidence/normal_report_native_v1` (30/5)
- `expected_outputs/full_own_answer_v3` (54/5)
- `expected_outputs/fresh_native_v3` (30/5)
- `expected_outputs/normal_report_source_v2` (52/0)
- `expected_outputs/normal_report_native_v2` (30/5)
- `expected_outputs/release_gate_v1` (final focused tests, initial/final gates,
  checkpoint and source snapshots).

PT files remain at their manifest's persistent cache path with exact size/hash;
they are not duplicated into Git or falsely described as inside the tarball.

### Integration and verified current global state

Added both conversion mapping and navigation in `config_extension.yaml`, then
regenerated `config.yaml` with `scripts/build_merged_config.py --write --check`.
Added rootREADME navigation, the style-status row, method-registry entry, all
three hard-ladder registry forms and five evidence links for **only** the selected
`data_attribution.dataset_cartography` requirement, now TAUGHT in16.11.
No other scientific topic was marked complete by this work.

The course-surface audit now follows the actual cartography SGD/data/selection/
evaluation/reference path. The report audit accepts independent SGD/NumPy checks
for training dynamics while preserving SciPy requirements for prior optimum-based
studies. Mutation tests reject missing operations, changed counts, missing controls
and hidden held-out selection. The style audit now recognizes an actual plotted
Markdown image or embedded notebook output without requiring the literal phrase
“signature result.” Source-only plotting code or an image without a plotting
operation does not satisfy that alternative. Existing controls/execution checks
remain separate. This fixed an actual20-image false positive without adding
pipeline vocabulary to the learner's original-style prose.

**Final focused tests: 80 PASS in11.65s.** Test set comprises the new ten method
and ten contract cases plus course-surface, report-evidence, roadmap-completeness,
style-depth, hard-ladder and original-preservation tests. Initial tests ran before
the first report existed and recorded three missing-report failures; final tests
run after complete publication/report pass. Initial/final gate logs are retained.

Current global gates:

- Original preservation PASS, zero violations.
- Course surface PASS,13 extension chapters.
- Style depth PASS,53 pages.
- Hard exercise ladder PASS,52 rows.
- Report evidence PASS,55 reports.
- Roadmap205: **48 TAUGHT /36 PREFLIGHT_ONLY /20 TOY_ONLY /13 DEFERRED /
  88 UNIMPLEMENTED**. Final completeness fails on exactly88 unimplemented
  requirements; this section contributes no remaining blocker.
- Readability still flags78 older issues, none in16.11. They remain full-course
  work; passing this section does not close them or promote partial topics.

The shared legacy report runner initializes CUDA before dispatching even a CPU
section. An initial unmasked invocation failed with CUDA OOM at
`cleanup_cuda_allocations -> torch.cuda.synchronize`, before the section ran.
Do not kill other GPU processes. For CPU report regeneration use:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/run_extension_verification_reports.py --section 16.11
```

The common report runner was left unchanged to preserve all previous report
input hashes. The section's own executor/native/reproducer run CPU-only. No
training/test jobs remain running. A local preview HTTP server was started on
127.0.0.1:8768, root
`/home/bitwise/.cache/arena_extended/alignment/rendered-preview` (session23844
at this checkpoint; verify it is still live before use). Current preview is
`cartography-full-v3.html`; render/browser scripts are in the cartography cache.

Primary source cache `sources/`: `paper.pdf` SHA256
`e3d807f0e007ff03a22f6b80176f4bbc9f734e7804061c61792c6ff1b97c550f`,
`train_dy_filtering.py` SHA256
`03bd34ef421ddf615124ce7b5575adb15631c169fd4e2be3845d3c07790d8db6`,
`run_glue.py` SHA256
`ea002ddc421d18cfe404fd6908f4c26e5f05466504205ad0562fb9bc3fd3d083`.
The attempted author `train_dy_metrics.py` path was404; the actual implementation
read is `train_dy_filtering.py`.

### Continue the complete goal

The unbounded goal remains active. Read the current requirement registry, not
old completion labels. A concrete next selected topic is **`shapley.sage`**, still
UNIMPLEMENTED with no SAGE/global-versus-local lesson. The separate broad
`data_attribution.training_run_simulations` requirement also remains UNIMPLEMENTED;
do not count cartography or a report as an implementation of that method family.
No new SAGE plan, protocol, data choice or source research has been done yet.

For the next section, read its primary method and closest original ARENA lesson,
write the learner arc before implementation, fix scientific choices before native
outcomes, then deliver student core functions, immediate independent tests,
controlled real results, examples, editable play, interpretation and limits.
Execute from own answers, independently reproduce the complete actual experiment,
inspect figures and rendered math/dropdowns, publish, rerun focused/global gates,
archive exact source/results and update this handoff. Preserve original chapters
and all existing evidence. Finish all selected requirements and resolve partial
and readability gaps before declaring the entire extension complete.


---

# 16.11 Dataset Cartography — learner and experiment plan

Current learner failure: the selected dataset-cartography requirement has no section,
confidence/variability map, example-level trajectories or data-selection retraining.
There are no target exercise, solution or instruction files to salvage. Inspected
adjacent 16.10 notebooks/page and original 0.3 Optimization before this plan.

Target section claim: implement training-dynamics maps, inspect their dependence on
logging convention, model, seed and time window, and test region selection through
actual fresh training. A hard example is not automatically a labeling error.

Signature result to create: maps with correctness colors and the feasible variance
boundary; image/trajectory galleries; paired clean/noisy maps; known-error retrieval;
seed and model agreement; held-out accuracy/loss curves after region and matched
random selection; equal-update control and visible editable retraining.

Student-implemented functions: stable epoch-ID alignment, gold-label probability,
confidence/population variability/correctness, SGD logging with an independent
model axis, rank selection, label-count matched sampling, noise-retrieval curves,
set agreement, and explicit test-set evaluation. All study orchestration remains
visible. Only source download/checksum plumbing may be imported.

Controls/baselines: constant predictor and repeated final snapshot, label permutation,
paired clean versus 10 percent injected cyclic label errors, five initialization/order
seeds, linear versus MLP, online versus epoch-end logs, early/full/late windows,
full training pool, size-matched uniform random and label-count-matched random,
and equal-update retraining at the one-third ambiguous selection budget.

Play cells: reorder/repeat/alter a short probability trajectory; choose a real map
region, retained size and fresh seed, inspect examples and actually retrain versus
matched random. No settings chosen by held-out performance.

Files to modify: new part11_dataset_cartography, corresponding instruction page/assets;
append navigation/status/roadmap evidence only after native, learner and independent
reproduction and visual review pass. Preserve chapters 0–4 byte-for-byte.

Tests to run: independent NumPy scalar metric oracle, correct epoch alignment under
shuffle/duplicates/missing IDs, tiny torch.nn/SGD loop parity, class-count and rank
invariants, own-following-answer notebook completion, no held-out access during
selection/training, fresh CPU tensor reproduction, visual inspection, focused gates
and original preservation. Archive exact source and outcomes, including failures.

Primary source: Swayamdipta et al. (EMNLP 2020),
https://aclanthology.org/2020.emnlp-main.746/ (Sections 2–5, Appendix A.1).
PDF SHA256 e3d807f0e007ff03a22f6b80176f4bbc9f734e7804061c61792c6ff1b97c550f.
Released author code https://github.com/allenai/cartography, inspected
cartography/selection/train_dy_filtering.py and classification/run_glue.py.
Paper says epoch-end parameters; code stores visited minibatch logits before its
update. We explicitly retain both, using epoch-end as primary. Author correctness
is stored as a count; our displayed fraction divides by the number of epochs.
This CPU digit experiment tests the method, not the paper's RoBERTa NLP/OOD claims.
