# Your study guide to ARENA Extended

Choose one question, build the method that answers it, and see whether its controls support your
explanation. You do not need to read the whole extension before starting.

To read the lessons locally, run this from the repository root in the existing environment:

```bash
uv run --no-sync --offline streamlit run mastery_hub/read.py --server.address=127.0.0.1 --server.port=18505 --server.headless=true --browser.gatherUsageStats=false
```

Choose a chapter, lesson and part in the sidebar. Start with the first button, **Download blank Practice exercises**,
to work in your notebook environment. The published exercises and worked reference retain their saved outputs;
help and expected answers remain public Practice. Use the study routes below to choose where to begin.

## Open your first working notebook

1. Open [the local reader](http://127.0.0.1:18505/). For a small mathematical start,
   select [16.1 Exact Shapley](http://127.0.0.1:18505/?section=16.1&part=1).
2. Download blank Practice and save a personal copy in the **exact lesson directory**
   shown by the reader. A notebook opened from Downloads will not necessarily find
   this checkout's helpers and assets. Keep the published reference unchanged.
3. From the repository root, run `uv run --no-sync --offline jupyter lab`. Open your
   saved copy in that browser. Select a kernel whose `sys.executable` is this
   checkout's `.venv/bin/python`; the Jupyter server and kernel must use the same
   ARENA environment. Follow the selected lesson's setup rather than installing
   packages from a different environment.
4. Read the first question, implement one function, and run its immediate test.
   A notebook full of intentional exercise stubs is not meant to pass Run All
   before you have implemented them. Use one section at a time.
5. Bookmark the reader's current lesson/part URL. Save the personal notebook after
   adding a short next-action note; the bookmark resumes reading, while that file
   preserves your implementation. It does not record mastery or a completed run.

For 7.1 and 9.1, first read the [pilot Practice session guide][study-017]. Prepare
and test the method before recording its exact implementation fingerprint; an
informal initial prediction can come first. The exact forecast for a chosen new
experiment is frozen after preparation and before that experiment is run.

The original ARENA course remains the foundation. This guide gives exact routes to **86 covered extension
lessons**, their reference notebooks and reading pages. Coverage means that these materials exist; it does
not mean that all 86 have native/model execution evidence, independent evaluation or learner acceptance.
The earlier 60 routes retain their historical evidence scope. **26 new CPU study editions** add visible
methods, controlled mathematical or small-model organisms and independent fixtures; their native/model
claims remain pending. There are now **zero missing configured lessons**. The new editions are listed
separately below so you can see their scope before choosing a route.

## Pick your first question

- **Still getting comfortable with tensors?** Start with original [0.1 Ray Tracing][study-001]. Its small
  ray-intersection functions build toward a visible renderer. Then follow original [1.1 Transformer from
  Scratch][study-002] and [1.2 Intro to Mech Interp][study-003]. Use the [original course index][study-004]
  for the rest of the original curriculum.
- **Want a small mathematical start?** Take [16.1 Exact Shapley][study-005] → [16.2 KernelSHAP and
  PartitionSHAP][study-006]. Ask how two players can receive credit for the same interaction, then compare
  exact credit with an estimator and a different grouping of players. The finite-game exercises and plots
  have a CPU route.
- **Want to understand a model’s internal answer?** After original 1.2, use [7.1 blank Practice][study-007].
  Watch capital-city tokens emerge across layers, implement the lenses, then investigate what a patched
  vector can make an inspection prompt say. Continue through [8.1 activation patching][study-008] → [8.2
  attribution patching][study-009] → [8.3 AtP*][study-010] → [8.4 ACDC][study-011]. ACDC is currently
  **8.4**; older 8.3 ACDC files are historical material. Original [1.4.1 IOI][study-012] is a useful earlier
  route from exploratory plots to stronger interventions.
- **Want to build image–text matching?** Start with [12.1 CLIP and SigLIP][study-013]: similarity matrix →
  two losses → tiny dual encoder → retrieval controls. Finish the tiny-model comparison before moving to its
  released-checkpoint part. Then try [12.3 Mini VLM][study-014]: build the projector and input assembly,
  compare against image-blind controls, and ask which regions affect an answer. Read the checked-recovery
  note below before making a new recovery curve.
- **Want to study a causal direction?** Use [9.1 blank Practice][study-015] after learning residual-stream
  hooks. Implement the direction and interventions, inspect the refusal-strength sweep, and compare random
  and label-shuffled directions alongside benign side effects. A refusal rubric is not a general safety
  classifier.

Choose one branch for your first session. Stop after one implemented function and its immediate test if that
is your available time; next time, build toward its first interpretable figure. The chapter tables below
follow the current configured lesson identities; historical topics remain available in their original
locations.

## Work through a lesson

1. **Forecast.** Read the concrete question and prerequisites. Before opening saved results or answer
   dropdowns, write what you expect, how confident you are, and why. Include the control: what would make
   the same attractive picture misleading?
2. **Implement and test.** Write the core operation in the exercise notebook. Run the nearby sub-function
   checks when you are ready to execute; use the help before the solution if you get stuck. Preparing and
   testing an implementation comes before committing a forecast for its exact target experiment.
3. **Compare a visible result with its control.** Inspect tokens, images, a heatmap, graph or curve. Explain
   the axes, denominator and baseline. A successful test or a printed report does not by itself answer the
   scientific question. If a control succeeds, investigate it; if none succeeds, record that honestly.
4. **Reflect and play.** Say what changed your mind, name a competing explanation, and make a new prediction
   before changing one scientific condition in a “Try It Yourself” cell. A changed case stays a prediction
   until you actually measure it.

The [guided prediction workbook][study-016] supplies a checkpoint for each of the 86 lessons. Use its
section number to record the exact experiment, forecast, observations and transfer question. It accompanies
the method notebook; it does not replace it. Follow the lesson’s existing environment and setup when you
choose to run it, and preserve the published reference while working in your own copy.

These are **guided public Practice** materials. Blank code outputs do not hide cold-open examples, expected
answers or public solutions. Use them to strengthen understanding; recording a forecast after seeing the
answer does not make it an unseen assessment. The [Practice instructions][study-017] explain the 7.1 and 9.1
copies.

## Two reading notes before interpreting the pilots

Keep the [7.1 interpretation note][study-018], [7.1 control-reflection note][study-019] and [12.3 recovery
note][study-020] beside the corresponding exercises.

**7.1 — a readable donor is not a necessity result.** A Patchscope can show that a donor state carries
useful information and that an inspection prompt can use it at a given insertion layer. The late-layer
cutoff alone does not establish which layers native factual recall needs: donor/recipient compatibility and
the computation remaining in the changed prompt also matter. When you encounter the stronger historical
wording, keep this distinction and propose an additional native-recall necessity test. For the control
reflection, inspect a correct control if there is one; otherwise write **none**, then inspect a
patched-success/control-failure case. All three saved controls—text-only, norm-matched random vector and
wrong position—have 0% accuracy, so a correct control example is not required.

**12.3 — check the entire cohort before averaging recovery.** Before changing scene pairs or making a new
recovery plot, complete the [Mini VLM checked recovery exercises][study-021] and read their [short
instructions][study-022]. The inline exercises test finite inputs and recovery and require a positive
clean-minus-corrupt gap for every pair in the declared cohort before aggregation. Zero or reversed gaps must
be reported with their pair indices rather than silently dropped. A tiny positive gap remains eligible;
finite recovery outside [0,1] is retained. The literal CPU exercises need no model. The final bridge
requires an already prepared 12.3 session; historical Mini weights were not saved as recoverable
checkpoints. The highest-attention comparison in the prediction workbook is an additional learner exercise,
not an outcome supplied by the historical curves.

## Choose a route that fits your machine

Reading and handwritten predictions can begin before model setup. For execution, distinguish the learner
exercise from an optional pretrained bridge or release benchmark; a chapter-wide GPU flag can refer to that
later benchmark.

| Source-backed route | What to plan for |
| --- | --- |
| 16.1 exact finite game; 16.2 estimator/partition exercises | CPU mathematical exercises and visible figures. Separate CUDA verification hooks are not required for this learner route. |
| 5.3 selective-memory exercises; 5.4 fixed-filter SSD exercises and analytic play | CPU learner paths. Cached pretrained parity and fused CUDA/release checks are separate parts; read their setup before choosing them. CPU timing does not establish fused-kernel speed. |
| 12.1 losses and tiny dual encoder | The lesson provides CPU and accelerator selection and describes the tiny model as suitable for a recent multi-core CPU. Its later released CLIP/SigLIP comparison needs the specified checkpoint/data setup. |
| 7.1, 8.4 and 9.1 model studies | The notebooks select MPS/CUDA/CPU. Their documented model recipes include GPU runs; 8.4 performs thousands of forward passes and 9.1 repeated generation. 7.1 gives a reduced CPU lens recipe, which changes the training setting. No current runtime or cache availability is promised here. |
| 12.3 full Mini VLM and optional real-VLM bridge | The default lesson describes GPU featurisation/training, including about 1.5 GB for cached vision features. It offers reduced CPU settings. Its later real-VLM part quotes about 6 GB of GPU memory. Those are lesson estimates, not a fresh machine preflight. |
| Six optional companions below; Mini VLM guard fixtures | Bounded CPU/analytic exercises. They do not reproduce the corresponding trained-model studies. |
| The 26 new study editions listed below | Small CPU mathematical or explicitly constructed model organisms, immediate independent tests and controlled visible results. Their native/pretrained bridges, large-model comparisons and learner acceptance remain pending. |

For other lessons, check the linked setup and limitations rather than assuming a GPU budget from the title.
This guide changes no dependencies or execution settings.

## All 86 covered lessons

Each row follows the current configured section identity. **Exercises** is the working entrypoint;
**reference** retains worked answers and saved historical outputs where present; **reading** is the matching
page; **source** is the authoring master. Some titles describe a wider ambition than the implemented study.
The result column tells you the concrete object to inspect, and the scope notes below identify important
gaps. The chapter tables retain the earlier 60 routes; the 26 new CPU study editions have their own table below.

### 0. Foundations addendum

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **0.6 How to Know When an Interpretability Result Is Fake (Practice)** | Leakage/cherry-picking/probe-overfit/direction audit panel. | [exercises][study-023] · [reference][study-024] · [reading][study-025] · [source][study-026] |

### 1. Infrastructure addendum

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **1.6 Local Frontier ML Infrastructure (Practice)** | Memory, parity and activation-persistence figures. | [exercises][study-027] · [reference][study-028] · [reading][study-029] · [source][study-030] |

### 5. Modern architectures

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **5.1 Gemma from Scratch** | GELU/RoPE convention parity and cache-mask comparison. | [exercises][study-031] · [reference][study-032] · [reading][study-033] · [source][study-034] |
| **5.2 Mixture of Experts from Scratch** | Accuracy versus routing-health and intervention panels. | [exercises][study-035] · [reference][study-036] · [reading][study-037] · [source][study-038] |
| **5.3 Mamba from Scratch** | Selective-memory intervention and hold/erase trajectories. | [exercises][study-039] · [reference][study-040] · [reading][study-041] · [source][study-042] |
| **5.4 Mamba-2 and State-Space Duality** | Causal SSD matrix and carry-effect curves. | [exercises][study-043] · [reference][study-044] · [reading][study-045] · [source][study-046] |
| **5.5 State Tracking in SSMs and Transformers** | Eight-position state-transplant trajectories. | [exercises][study-047] · [reference][study-048] · [reading][study-049] · [source][study-050] |
| **5.8 Discrete Diffusion Language Models** | Token denoising trajectories and baseline comparisons. | [exercises][study-051] · [reference][study-052] · [reading][study-053] · [source][study-054] |
| **5.9 Embedding, Function-Calling and Specialist Models** | Retrieval heatmap and availability-aware abstention outputs. | [exercises][study-055] · [reference][study-056] · [reading][study-057] · [source][study-058] |

### 6. Sparse feature methods

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **6.1 SAE Variants** | Sparsity/reconstruction/recovery and intervention panel. | [exercises][study-059] · [reference][study-060] · [reading][study-061] · [source][study-062] |
| **6.2 Gemma Scope and Feature Steering** | Exact feature steering and confound panels in two organisms. | [exercises][study-063] · [reference][study-064] · [reading][study-065] · [source][study-066] |
| **6.4 Transcoders and Skip Transcoders** | Signed transcoder graph and behavioral faithfulness. | [exercises][study-067] · [reference][study-068] · [reading][study-069] · [source][study-070] |
| **6.5 Crosscoders and Model Diffing** | Crosscoder ownership and intervention curves. | [exercises][study-071] · [reference][study-072] · [reading][study-073] · [source][study-074] |

### 7. Activation to language

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **7.1 Logit Lens, Tuned Lens and Patchscopes** | Layer-by-layer tokens/ranks, tuned lens and Patchscope plots. | [exercises][study-075] · [reference][study-076] · [reading][study-077] · [source][study-078] |
| **7.2 Feature Verbalizers** | 30-latent simulation/control plot and counterexamples. | [exercises][study-079] · [reference][study-080] · [reading][study-081] · [source][study-082] |
| **7.3 Mini Activation Oracles** | Oracle question accuracy, layer/probe and intervention panels. | [exercises][study-083] · [reference][study-084] · [reading][study-085] · [source][study-086] |
| **7.4 Mini Natural Language Autoencoders** | Reconstruction FVE versus behavior transfer and descriptions. | [exercises][study-087] · [reference][study-088] · [reading][study-089] · [source][study-090] |
| **7.5 Predictive Concept Decoders** | PCD versus same-size SAE and all-position concept ablation. | [exercises][study-091] · [reference][study-092] · [reading][study-093] · [source][study-094] |

### 8. Circuit discovery

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **8.1 Activation Patching Refresher** | IOI/greater-than patch heatmaps and Hydra self-repair. | [exercises][study-008] · [reference][study-095] · [reading][study-096] · [source][study-097] |
| **8.2 Attribution Patching and EAP** | Exact/AP/IG scatter and edge-budget faithfulness curves. | [exercises][study-009] · [reference][study-098] · [reading][study-099] · [source][study-100] |
| **8.3 AtP* from Scratch** | 576-node recall and QK-fix/GradDrop ablations. | [exercises][study-010] · [reference][study-101] · [reading][study-102] · [source][study-103] |
| **8.4 ACDC and Circuit Metrics** | Automatically pruned circuit and faithfulness/completeness/minimality curves. | [exercises][study-011] · [reference][study-104] · [reading][study-105] · [source][study-106] |
| **8.5 Causal Scrubbing** | Induction loss recovery for competing causal hypotheses. | [exercises][study-107] · [reference][study-108] · [reading][study-109] · [source][study-110] |
| **8.6 Sparse Feature Circuits and SHIFT** | Planted sparse graph and exact-versus-IG curves. | [exercises][study-111] · [reference][study-112] · [reading][study-113] · [source][study-114] |
| **8.7 Circuit Tracing with Attribution Graphs** | Planted component DAG, edge heatmap and top-path interventions. | [exercises][study-115] · [reference][study-116] · [reading][study-117] · [source][study-118] |
| **8.8 Benchmarking Circuit Discovery on Compiled Transformers** | Compiled-transformer ground-truth edge AUC and faithfulness. | [exercises][study-119] · [reference][study-120] · [reading][study-121] · [source][study-122] |

### 9. Alignment interpretability

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **9.1 Refusal Directions and Representation Engineering** | Direction sweep, refusal controls and benign cost curves. | [exercises][study-123] · [reference][study-124] · [reading][study-125] · [source][study-126] |
| **9.3 Chain-of-Thought Faithfulness** | Layer/position decodability and answer-patching heatmaps. | [exercises][study-127] · [reference][study-128] · [reading][study-129] · [source][study-130] |
| **9.4 Emergent Misalignment Detection** | Known-onset drift and mitigation-versus-capability curves. | [exercises][study-131] · [reference][study-132] · [reading][study-133] · [source][study-134] |
| **9.5 White-Box Monitors and Red-Teaming with Internals** | Threshold/AUROC monitor dashboard and output-only misses. | [exercises][study-135] · [reference][study-136] · [reading][study-137] · [source][study-138] |
| **9.6 Model Editing with ROME and MEMIT** | Localization heatmap, target continuation and collateral edit table. | [exercises][study-139] · [reference][study-140] · [reading][study-141] · [source][study-142] |

### 10. Research capstone

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **10.1 Capstone Research Sprint** | Preregistered digit-patching test and uncertainty/claim verdict. | [exercises][study-143] · [reference][study-144] · [reading][study-145] · [source][study-146] |

### 11. Representation geometry

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **11.1 PCA, SVD and Geometry Controls** | Calendar coordinates and actual-token intervention figures. | [exercises][study-147] · [reference][study-148] · [reading][study-149] · [source][study-150] |
| **11.2 CCA, SVCCA and Representation Alignment** | Representation alignment and donor-distribution transport. | [exercises][study-151] · [reference][study-152] · [reading][study-153] · [source][study-154] |
| **11.3 Causal Abstraction and Distributed Alignment Search** | Interchange distributions and native role counterfactuals. | [exercises][study-155] · [reference][study-156] · [reading][study-157] · [source][study-158] |
| **11.4 Concept Erasure** | Recipient/object token tables after LEACE and reader refit. | [exercises][study-159] · [reference][study-160] · [reading][study-161] · [source][study-162] |
| **11.5 Contrast-Consistent Search** | Actual review predictions and opinion-change panels. | [exercises][study-163] · [reference][study-164] · [reading][study-165] · [source][study-166] |

### 12. Vision–language interpretability

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **12.1 CLIP and SigLIP from Scratch** | Tiny and released CLIP/SigLIP retrieval/binding matrices. | [exercises][study-013] · [reference][study-167] · [reading][study-168] · [source][study-169] |
| **12.3 Mini VLM from Scratch** | VQA examples, image/text controls and region-patching sweep. | [exercises][study-014] · [reference][study-170] · [reading][study-171] · [source][study-172] |
| **12.5 Object Hallucination and Modality Arbitration** | Object hallucination/arbitration heatmaps and patch figures. | [exercises][study-173] · [reference][study-174] · [reading][study-175] · [source][study-176] |

### 13. Diffusion interpretability

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **13.1 Diffusion Models and Cross-Attention Attribution** | Token-region attention and same-seed image-edit grids. | [exercises][study-177] · [reference][study-178] · [reading][study-179] · [source][study-180] |
| **13.2 Denoising-Time Causal Patching** | Time/place intervention curves and SD1.5 patch image grid. | [exercises][study-181] · [reference][study-182] · [reading][study-183] · [source][study-184] |

### 14. JEPA and world models

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **14.1 I-JEPA from Scratch** | Masked-latent prediction/probe figures and key-swap table. | [exercises][study-185] · [reference][study-186] · [reading][study-187] · [source][study-188] |
| **14.2 V-JEPA and Video World Models** | Persistence-shortcut stress test and separate released controls. | [exercises][study-189] · [reference][study-190] · [reading][study-191] · [source][study-192] |
| **14.5 Maze Transformers and World Models** | Maze rollout, state-probe and causal patch plots. | [exercises][study-193] · [reference][study-194] · [reading][study-195] · [source][study-196] |

### 15. Parameter-efficient adaptation

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **15.1 LoRA and DoRA from Scratch** | Adapter singular spectra and behavior intervention panel. | [exercises][study-197] · [reference][study-198] · [reading][study-199] · [source][study-200] |
| **15.2 LoRA vs Full Fine-Tuning** | LoRA/full update spectra and behavior/collateral comparisons. | [exercises][study-201] · [reference][study-202] · [reading][study-203] · [source][study-204] |
| **15.3 QLoRA from Scratch** | NF4 outlier/storage and retained QLoRA training panels. | [exercises][study-205] · [reference][study-206] · [reading][study-207] · [source][study-208] |

### 16. Shapley and attribution baselines

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **16.1 Exact Shapley on Ground-Truth Games** | Exact credit and leave-one-out comparison on interaction game. | [exercises][study-005] · [reference][study-209] · [reading][study-210] · [source][study-211] |
| **16.2 KernelSHAP and PartitionSHAP** | KernelSHAP error and aligned/misaligned partition allocation. | [exercises][study-006] · [reference][study-212] · [reading][study-213] · [source][study-214] |
| **16.3 TreeSHAP, DeepSHAP and GradientSHAP (Practice)** | Tree/Deep/GradientSHAP full historical curricula and figures. | [exercises][study-215] · [reference][study-216] · [reading][study-217] · [source][study-218] |
| **16.4 SAGE Global Importance** | Conditional versus marginal SAGE and coalition-risk curves. | [exercises][study-219] · [reference][study-220] · [reading][study-221] · [source][study-222] |
| **16.5 Shapley Interactions** | Pair, Shapley-Taylor, Faith-Shap and shared-budget estimator figures. | [exercises][study-223] · [reference][study-224] · [reading][study-225] · [source][study-226] |
| **16.6 Sparse Interaction Discovery with SPEX and ProxySPEX** | Exact spectrum, SPEX/ProxySPEX budget and recovery panels. | [exercises][study-227] · [reference][study-228] · [reading][study-229] · [source][study-230] |
| **16.7 TokenSHAP and Region SHAP for LLMs and VLMs** | Token credit and rendered-region/whole-modality coalitions. | [exercises][study-231] · [reference][study-232] · [reading][study-233] · [source][study-234] |
| **16.8 SHAP vs Activation Patching** | SHAP/patching agreement and interaction/deletion-insertion sweeps. | [exercises][study-235] · [reference][study-236] · [reading][study-237] · [source][study-238] |
| **16.9 Data Shapley in One Training Run** | Data credit, ghost-gradient parity and delete/retrain panels. | [exercises][study-239] · [reference][study-240] · [reading][study-241] · [source][study-242] |
| **16.10 Influence Functions and TracIn** | Influence/refit scatter and checkpoint-gradient deletion results. | [exercises][study-243] · [reference][study-244] · [reading][study-245] · [source][study-246] |
| **16.11 Datamodels and Dataset Cartography** | Subset-response prediction and training-history/deletion plots. | [exercises][study-247] · [reference][study-248] · [reading][study-249] · [source][study-250] |

### 17. Training dynamics

| Lesson | Result to inspect | Open |
| --- | --- | --- |
| **17.1 Checkpoint Archaeology and Grokking** | Finite-domain checkpoint/Fourier emergence and intervention panels. | [exercises][study-251] · [reference][study-252] · [reading][study-253] · [source][study-254] |

## Keep the question within the implemented scope

- **Architectures and sparse features:** 5.1 starts with tiny fixed-random architecture parity. 5.2’s
  preferred-expert necessity hypothesis was rejected. 5.8 retains weak-generation results. 5.9’s main
  retrieval study uses a hand lexicon; specialist-model evaluation is unfinished. 6.1 compares four planted
  forward rules, not a learned-SAE frontier including BatchTopK; 6.4 implements an ordinary transcoder, with
  skip transcoders unfinished; 6.5 does not establish real base/chat model differences.
- **Interpreting representations and circuits:** descriptions and probes in 7.2–7.5 remain bounded to their
  tasks and scorers. In 8.1, single-component effects are not an additive decomposition. Attribution in 8.2
  is a local approximation; 8.6’s full SHIFT/released-SAE expansion and 8.7’s released-transcoder expansion
  remain unfinished. Coordinate alignment, linear erasure and contrast consistency in chapter 11 do not by
  themselves establish correct behavior, absence of nonlinear information or truthfulness.
- **Alignment and multimodal claims:** 9.3 does not supply the planned generated instruct-model CoT studies.
  9.4/9.5 use planted proxy/monitoring tasks; 9.6’s rank-one course edit is not official ROME parity or
  MEMIT mass editing. In 12.1, a consistent wrong relabeling can be learned, so it differs from a genuinely
  mismatched-pair control. 12.5 is an analytic arbitration lab; its neutral-prior visual preflight is not
  hallucination repair.
- **Read these as historical or bounded references:** 14.1’s retained figures and seed tables agree internally:
  correct/original-after-swap/alternate accuracy is 1/0/1 for seeds 0, 1 and 2. The third, original-class bar is
  zero; the fourth, alternate-class bar is full. The old disagreement warning preserves an axis misreading;
  this correction is not a new reproduction. 14.2 separates
  generated-world experiments from frozen released-model controls and does not establish real-video
  transfer. 14.5 retains failed historical CUDA gates. 15.1/15.2 are controlled adaptation studies. 15.3
  retains failed BF16 parity and remains unreleased.
- **Attribution and dynamics:** chapter 16’s games have different players, backgrounds, losses and
  normalization conventions; keep those choices explicit when comparing methods. Edited-input credit does
  not automatically identify an internal mechanism. 17.1 trains/evaluates on all 169 mod-13 pairs: it
  studies finite-domain checkpoint behavior, not held-out grokking. For grokking, follow original [1.5.2
  Grokking and Modular Arithmetic][study-255].

## Intact earlier source books

The 0.6, 1.6 and 16.3 Practice routes preserve the earlier teaching material and link back to it. Use these
intact books when you want the original source and historical output context:

- [0.6 How to Know When an Interpretability Result Is Fake — exercises][study-256]
- [0.6 How to Know When an Interpretability Result Is Fake — reference][study-257]
- [1.6 Local Frontier ML Infrastructure — exercises][study-258]
- [1.6 Local Frontier ML Infrastructure — reference][study-259]
- [16.13 TreeSHAP from First Principles — exercises][study-260]
- [16.13 TreeSHAP from First Principles — reference][study-261]
- [16.14 DeepSHAP When Does Fast Attribution Match Exact SHAP — exercises][study-262]
- [16.14 DeepSHAP When Does Fast Attribution Match Exact SHAP — reference][study-263]
- [16.15 Expected Gradients Which Expectation Does GradientSHAP Estimate — exercises][study-264]
- [16.15 Expected Gradients Which Expectation Does GradientSHAP Estimate — reference][study-265]

For the original course and upstream resources, keep the [original-materials shelf][study-266] alongside
this guide. Original ARENA chapters 0–4 remain the base; 0.6 and 1.6 are extension addenda.

## Optional prediction companions

After learning the method, choose one companion if you want a focused forecast with visible controls. Each
starts without saved code outputs. Declare prior exposure honestly; a blank notebook does not make a public
setup unseen. Pass its separate practice fixtures before committing to the target, and use its own
instructions for observation, reflection and replay.

| Companion | Question to forecast |
| --- | --- |
| [8.8 circuit prediction][study-267] | How does changing readout gain affect circuit attribution? |
| [Data Shapley prediction][study-268] | How does coalition-training normalization change one example’s credit? |
| [SAGE prediction][study-269] | How do conditional information and marginal model use allocate global credit? |
| [SSD prediction][study-270] | How does weakening state carry change a signed final readout? |
| [8.7 graph/path prediction][study-271] | How do graph attribution and path recovery behave in a fixed synthetic graph? |
| [6.1 fixed-forward SAE prediction][study-272] | How do four fixed rules change codes and reconstructions with the same weights? |

These are bounded CPU studies, with their own target and exposure boundaries. The graph/path and SAE public
targets remain unexecuted in the current record. Fixed-rule SAE mechanics do not rank trained SAEs, and
fixed-filter SSD does not establish CUDA performance. Completing a companion record is practice evidence
rather than automatic unseen mastery.

## The 26 new CPU study editions (2026-11-02)

At the earlier 60-lesson checkpoint, these 26 configured destinations had no current canonical
lesson. They now complete the **86 available / 0 missing** lesson inventory in the [syllabus][study-273].
Each is a CPU study edition with student-implemented methods, immediate independent fixtures,
controlled visible results and changed-condition play. Its small organism or analytic example
teaches the mechanism; **native/pretrained evidence and learner acceptance remain pending**.
Read each lesson's precise boundary before treating its title as a claim about a released model.

| Lesson | Open the CPU study edition |
| --- | --- |
| **5.6 Linear Attention, RetNet and RWKV** | [exercises][study-274] · [reference][study-275] · [reading][study-276] · [source][study-277] |
| **5.7 Griffin, xLSTM, Hyena and Hybrid Models** | [exercises][study-278] · [reference][study-279] · [reading][study-280] · [source][study-281] |
| **6.3 Feature Splitting, Absorption and Dashboards** | [exercises][study-282] · [reference][study-283] · [reading][study-284] · [source][study-285] |
| **7.6 Automated Interpretability Agents** | [exercises][study-286] · [reference][study-287] · [reading][study-288] · [source][study-289] |
| **9.2 Truthfulness Directions and Hallucination Probes** | [exercises][study-290] · [reference][study-291] · [reading][study-292] · [source][study-293] |
| **9.7 Unlearning and How to Evaluate It** | [exercises][study-294] · [reference][study-295] · [reading][study-296] · [source][study-297] |
| **11.6 Feature Manifolds - Days, Months, Space and Time** | [exercises][study-298] · [reference][study-299] · [reading][study-300] · [source][study-301] |
| **12.2 CLIP Features, SAEs and Typographic Attacks** | [exercises][study-302] · [reference][study-303] · [reading][study-304] · [source][study-305] |
| **12.4 Visual Token Flow in Real VLMs** | [exercises][study-306] · [reference][study-307] · [reading][study-308] · [source][study-309] |
| **12.6 Multimodal SAEs, Transcoders and Crosscoders** | [exercises][study-310] · [reference][study-311] · [reading][study-312] · [source][study-313] |
| **12.7 Dense Vision Features - DINO and SAM** | [exercises][study-314] · [reference][study-315] · [reading][study-316] · [source][study-317] |
| **13.3 Concept Directions and Diffusion LoRA** | [exercises][study-318] · [reference][study-319] · [reading][study-320] · [source][study-321] |
| **13.4 Sparse Autoencoders for Diffusion Models** | [exercises][study-322] · [reference][study-323] · [reading][study-324] · [source][study-325] |
| **13.5 Diffusion Transformers and ConceptAttention** | [exercises][study-326] · [reference][study-327] · [reading][study-328] · [source][study-329] |
| **13.6 Autoregressive Image Tokens** | [exercises][study-330] · [reference][study-331] · [reading][study-332] · [source][study-333] |
| **14.3 Action-Conditioned World Models and Latent Planning** | [exercises][study-334] · [reference][study-335] · [reading][study-336] · [source][study-337] |
| **14.4 The JEPA Family** | [exercises][study-338] · [reference][study-339] · [reading][study-340] · [source][study-341] |
| **14.6 Sudoku and Constraint World Models** | [exercises][study-342] · [reference][study-343] · [reading][study-344] · [source][study-345] |
| **14.7 Recurrent World Models - Dreamer, TD-MPC2 and IRIS** | [exercises][study-346] · [reference][study-347] · [reading][study-348] · [source][study-349] |
| **15.4 LoRA Phase Transitions and Misalignment Proxies** | [exercises][study-350] · [reference][study-351] · [reading][study-352] · [source][study-353] |
| **15.5 VLM LoRA** | [exercises][study-354] · [reference][study-355] · [reading][study-356] · [source][study-357] |
| **17.2 Induction Head Emergence** | [exercises][study-358] · [reference][study-359] · [reading][study-360] · [source][study-361] |
| **17.3 Fine-Tuning Diffs and Capability Emergence** | [exercises][study-362] · [reference][study-363] · [reading][study-364] · [source][study-365] |
| **17.4 Training Dynamics Across Architectures** | [exercises][study-366] · [reference][study-367] · [reading][study-368] · [source][study-369] |
| **18.1 Concept Bottlenecks, Backpacks and Interchange Intervention Training** | [exercises][study-370] · [reference][study-371] · [reading][study-372] · [source][study-373] |
| **18.2 Sparse and Modular Training, KANs and Sparse Trees** | [exercises][study-374] · [reference][study-375] · [reading][study-376] · [source][study-377] |

Return to your chosen question after each session: what can your implementation now show, which control
challenges the explanation, and what would you change next? That is a useful stopping point even when a
later model bridge or a planned topic remains unfinished.

[study-001]: ../chapter0_fundamentals/exercises/part1_ray_tracing/0.1_Ray_Tracing_exercises.ipynb
[study-002]: ../chapter1_transformer_interp/exercises/part1_transformer_from_scratch/1.1_Transformer_from_Scratch_exercises.ipynb
[study-003]: ../chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/1.2_Intro_to_Mech_Interp_exercises.ipynb
[study-004]: ../README.md
[study-005]: ../chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley/16.1_Exact_Shapley_on_Ground-Truth_Games_exercises.ipynb
[study-006]: ../chapter16_shapley_attribution_baselines/exercises/part2_kernelshap_partitionshap/16.2_KernelSHAP_and_PartitionSHAP_exercises.ipynb
[study-007]: units/pilot_practice/7_1_exercises.ipynb
[study-008]: ../chapter8_automated_circuits/exercises/part1_activation_patching_refresher/8.1_Activation_Patching_Refresher_exercises.ipynb
[study-009]: ../chapter8_automated_circuits/exercises/part2_attribution_patching_eap/8.2_Attribution_Patching_and_EAP_exercises.ipynb
[study-010]: ../chapter8_automated_circuits/exercises/part3_atp_star/8.3_AtP_from_Scratch_exercises.ipynb
[study-011]: ../chapter8_automated_circuits/exercises/part4_acdc_circuit_metrics/8.4_ACDC_and_Circuit_Metrics_exercises.ipynb
[study-012]: ../chapter1_transformer_interp/exercises/part41_indirect_object_identification/1.4.1_Indirect_Object_Identification_exercises.ipynb
[study-013]: ../chapter12_vlm_interpretability/exercises/part1_clip_siglip/12.1_CLIP_and_SigLIP_from_Scratch_exercises.ipynb
[study-014]: ../chapter12_vlm_interpretability/exercises/part3_mini_vlm_from_scratch/12.3_Mini_VLM_from_Scratch_exercises.ipynb
[study-015]: units/pilot_practice/9_1_exercises.ipynb
[study-016]: units/extension_predictions/exercises.ipynb
[study-017]: units/pilot_practice/README.md
[study-018]: ../docs/mastery/pilot_reader_notes_20261023.md#7-1-interpretation
[study-019]: ../docs/mastery/pilot_reader_notes_20261023.md#7-1-controls-reflection
[study-020]: ../docs/mastery/pilot_reader_notes_20261023.md#12-3-recovery
[study-021]: practice_addons/mini_vlm_checked_recovery/exercises.ipynb
[study-022]: practice_addons/mini_vlm_checked_recovery/README.md
[study-023]: ../chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice_exercises.ipynb
[study-024]: ../chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice_solutions.ipynb
[study-025]: ../chapter0_fundamentals/instructions/pages/06_%5B0.6%5D_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice.md
[study-026]: ../infrastructure/chapters/chapter0_fundamentals/master_0_6.py
[study-027]: ../chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_Practice_exercises.ipynb
[study-028]: ../chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/1.6_Local_Frontier_ML_Infrastructure_Practice_solutions.ipynb
[study-029]: ../chapter1_transformer_interp/instructions/pages/40_%5B1.6%5D_Local_Frontier_ML_Infrastructure_Practice.md
[study-030]: ../infrastructure/chapters/chapter1_transformer_interp/master_1_6.py
[study-031]: ../chapter5_modern_architectures/exercises/part1_gemma_from_scratch/5.1_Gemma_from_Scratch_exercises.ipynb
[study-032]: ../chapter5_modern_architectures/exercises/part1_gemma_from_scratch/5.1_Gemma_from_Scratch_solutions.ipynb
[study-033]: ../chapter5_modern_architectures/instructions/pages/01_%5B5.1%5D_Gemma_from_Scratch.md
[study-034]: ../infrastructure/chapters/chapter5_modern_architectures/master_5_1.py
[study-035]: ../chapter5_modern_architectures/exercises/part2_mixture_of_experts/5.2_Mixture_of_Experts_from_Scratch_exercises.ipynb
[study-036]: ../chapter5_modern_architectures/exercises/part2_mixture_of_experts/5.2_Mixture_of_Experts_from_Scratch_solutions.ipynb
[study-037]: ../chapter5_modern_architectures/instructions/pages/02_%5B5.2%5D_Mixture_of_Experts_from_Scratch.md
[study-038]: ../infrastructure/chapters/chapter5_modern_architectures/master_5_2.py
[study-039]: ../chapter5_modern_architectures/exercises/part3_mamba_from_scratch/5.3_Mamba_from_Scratch_exercises.ipynb
[study-040]: ../chapter5_modern_architectures/exercises/part3_mamba_from_scratch/5.3_Mamba_from_Scratch_solutions.ipynb
[study-041]: ../chapter5_modern_architectures/instructions/pages/03_%5B5.3%5D_Mamba_from_Scratch.md
[study-042]: ../infrastructure/chapters/chapter5_modern_architectures/master_5_3.py
[study-043]: ../chapter5_modern_architectures/exercises/part4_mamba2_ssd/5.4_Mamba-2_and_State-Space_Duality_exercises.ipynb
[study-044]: ../chapter5_modern_architectures/exercises/part4_mamba2_ssd/5.4_Mamba-2_and_State-Space_Duality_solutions.ipynb
[study-045]: ../chapter5_modern_architectures/instructions/pages/04_%5B5.4%5D_Mamba-2_and_State-Space_Duality.md
[study-046]: ../infrastructure/chapters/chapter5_modern_architectures/master_5_4.py
[study-047]: ../chapter5_modern_architectures/exercises/part5_state_tracking/5.5_State_Tracking_in_SSMs_and_Transformers_exercises.ipynb
[study-048]: ../chapter5_modern_architectures/exercises/part5_state_tracking/5.5_State_Tracking_in_SSMs_and_Transformers_solutions.ipynb
[study-049]: ../chapter5_modern_architectures/instructions/pages/05_%5B5.5%5D_State_Tracking_in_SSMs_and_Transformers.md
[study-050]: ../infrastructure/chapters/chapter5_modern_architectures/master_5_5.py
[study-051]: ../chapter5_modern_architectures/exercises/part8_diffusion_language_models/5.8_Discrete_Diffusion_Language_Models_exercises.ipynb
[study-052]: ../chapter5_modern_architectures/exercises/part8_diffusion_language_models/5.8_Discrete_Diffusion_Language_Models_solutions.ipynb
[study-053]: ../chapter5_modern_architectures/instructions/pages/08_%5B5.8%5D_Discrete_Diffusion_Language_Models.md
[study-054]: ../infrastructure/chapters/chapter5_modern_architectures/master_5_8.py
[study-055]: ../chapter5_modern_architectures/exercises/part9_specialist_models/5.9_Embedding_Function-Calling_and_Specialist_Models_exercises.ipynb
[study-056]: ../chapter5_modern_architectures/exercises/part9_specialist_models/5.9_Embedding_Function-Calling_and_Specialist_Models_solutions.ipynb
[study-057]: ../chapter5_modern_architectures/instructions/pages/09_%5B5.9%5D_Embedding_Function-Calling_and_Specialist_Models.md
[study-058]: ../infrastructure/chapters/chapter5_modern_architectures/master_5_9.py
[study-059]: ../chapter6_sparse_feature_methods/exercises/part1_sae_variants/6.1_SAE_Variants_exercises.ipynb
[study-060]: ../chapter6_sparse_feature_methods/exercises/part1_sae_variants/6.1_SAE_Variants_solutions.ipynb
[study-061]: ../chapter6_sparse_feature_methods/instructions/pages/01_%5B6.1%5D_SAE_Variants.md
[study-062]: ../infrastructure/chapters/chapter6_sparse_feature_methods/master_6_1.py
[study-063]: ../chapter6_sparse_feature_methods/exercises/part2_gemma_scope/6.2_Gemma_Scope_and_Feature_Steering_exercises.ipynb
[study-064]: ../chapter6_sparse_feature_methods/exercises/part2_gemma_scope/6.2_Gemma_Scope_and_Feature_Steering_solutions.ipynb
[study-065]: ../chapter6_sparse_feature_methods/instructions/pages/02_%5B6.2%5D_Gemma_Scope_and_Feature_Steering.md
[study-066]: ../infrastructure/chapters/chapter6_sparse_feature_methods/master_6_2.py
[study-067]: ../chapter6_sparse_feature_methods/exercises/part4_transcoders/6.4_Transcoders_and_Skip_Transcoders_exercises.ipynb
[study-068]: ../chapter6_sparse_feature_methods/exercises/part4_transcoders/6.4_Transcoders_and_Skip_Transcoders_solutions.ipynb
[study-069]: ../chapter6_sparse_feature_methods/instructions/pages/04_%5B6.4%5D_Transcoders_and_Skip_Transcoders.md
[study-070]: ../infrastructure/chapters/chapter6_sparse_feature_methods/master_6_4.py
[study-071]: ../chapter6_sparse_feature_methods/exercises/part5_crosscoders/6.5_Crosscoders_and_Model_Diffing_exercises.ipynb
[study-072]: ../chapter6_sparse_feature_methods/exercises/part5_crosscoders/6.5_Crosscoders_and_Model_Diffing_solutions.ipynb
[study-073]: ../chapter6_sparse_feature_methods/instructions/pages/05_%5B6.5%5D_Crosscoders_and_Model_Diffing.md
[study-074]: ../infrastructure/chapters/chapter6_sparse_feature_methods/master_6_5.py
[study-075]: ../chapter7_activation_to_language/exercises/part1_lenses_patchscopes/7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_exercises.ipynb
[study-076]: ../chapter7_activation_to_language/exercises/part1_lenses_patchscopes/7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_solutions.ipynb
[study-077]: ../chapter7_activation_to_language/instructions/pages/01_%5B7.1%5D_Logit_Lens_Tuned_Lens_and_Patchscopes.md
[study-078]: ../infrastructure/chapters/chapter7_activation_to_language/master_7_1.py
[study-079]: ../chapter7_activation_to_language/exercises/part2_feature_verbalizers/7.2_Feature_Verbalizers_exercises.ipynb
[study-080]: ../chapter7_activation_to_language/exercises/part2_feature_verbalizers/7.2_Feature_Verbalizers_solutions.ipynb
[study-081]: ../chapter7_activation_to_language/instructions/pages/02_%5B7.2%5D_Feature_Verbalizers.md
[study-082]: ../infrastructure/chapters/chapter7_activation_to_language/master_7_2.py
[study-083]: ../chapter7_activation_to_language/exercises/part3_mini_activation_oracles/7.3_Mini_Activation_Oracles_exercises.ipynb
[study-084]: ../chapter7_activation_to_language/exercises/part3_mini_activation_oracles/7.3_Mini_Activation_Oracles_solutions.ipynb
[study-085]: ../chapter7_activation_to_language/instructions/pages/03_%5B7.3%5D_Mini_Activation_Oracles.md
[study-086]: ../infrastructure/chapters/chapter7_activation_to_language/master_7_3.py
[study-087]: ../chapter7_activation_to_language/exercises/part4_mini_natural_language_autoencoders/7.4_Mini_Natural_Language_Autoencoders_exercises.ipynb
[study-088]: ../chapter7_activation_to_language/exercises/part4_mini_natural_language_autoencoders/7.4_Mini_Natural_Language_Autoencoders_solutions.ipynb
[study-089]: ../chapter7_activation_to_language/instructions/pages/04_%5B7.4%5D_Mini_Natural_Language_Autoencoders.md
[study-090]: ../infrastructure/chapters/chapter7_activation_to_language/master_7_4.py
[study-091]: ../chapter7_activation_to_language/exercises/part5_predictive_concept_decoders/7.5_Predictive_Concept_Decoders_exercises.ipynb
[study-092]: ../chapter7_activation_to_language/exercises/part5_predictive_concept_decoders/7.5_Predictive_Concept_Decoders_solutions.ipynb
[study-093]: ../chapter7_activation_to_language/instructions/pages/05_%5B7.5%5D_Predictive_Concept_Decoders.md
[study-094]: ../infrastructure/chapters/chapter7_activation_to_language/master_7_5.py
[study-095]: ../chapter8_automated_circuits/exercises/part1_activation_patching_refresher/8.1_Activation_Patching_Refresher_solutions.ipynb
[study-096]: ../chapter8_automated_circuits/instructions/pages/01_%5B8.1%5D_Activation_Patching_Refresher.md
[study-097]: ../infrastructure/chapters/chapter8_automated_circuits/master_8_1.py
[study-098]: ../chapter8_automated_circuits/exercises/part2_attribution_patching_eap/8.2_Attribution_Patching_and_EAP_solutions.ipynb
[study-099]: ../chapter8_automated_circuits/instructions/pages/02_%5B8.2%5D_Attribution_Patching_and_EAP.md
[study-100]: ../infrastructure/chapters/chapter8_automated_circuits/master_8_2.py
[study-101]: ../chapter8_automated_circuits/exercises/part3_atp_star/8.3_AtP_from_Scratch_solutions.ipynb
[study-102]: ../chapter8_automated_circuits/instructions/pages/03_%5B8.3%5D_AtP_from_Scratch.md
[study-103]: ../infrastructure/chapters/chapter8_automated_circuits/master_8_3.py
[study-104]: ../chapter8_automated_circuits/exercises/part4_acdc_circuit_metrics/8.4_ACDC_and_Circuit_Metrics_solutions.ipynb
[study-105]: ../chapter8_automated_circuits/instructions/pages/04_%5B8.4%5D_ACDC_and_Circuit_Metrics.md
[study-106]: ../infrastructure/chapters/chapter8_automated_circuits/master_8_4.py
[study-107]: ../chapter8_automated_circuits/exercises/part5_causal_scrubbing/8.5_Causal_Scrubbing_exercises.ipynb
[study-108]: ../chapter8_automated_circuits/exercises/part5_causal_scrubbing/8.5_Causal_Scrubbing_solutions.ipynb
[study-109]: ../chapter8_automated_circuits/instructions/pages/05_%5B8.5%5D_Causal_Scrubbing.md
[study-110]: ../infrastructure/chapters/chapter8_automated_circuits/master_8_5.py
[study-111]: ../chapter8_automated_circuits/exercises/part6_sparse_feature_circuits/8.6_Sparse_Feature_Circuits_and_SHIFT_exercises.ipynb
[study-112]: ../chapter8_automated_circuits/exercises/part6_sparse_feature_circuits/8.6_Sparse_Feature_Circuits_and_SHIFT_solutions.ipynb
[study-113]: ../chapter8_automated_circuits/instructions/pages/06_%5B8.6%5D_Sparse_Feature_Circuits_and_SHIFT.md
[study-114]: ../infrastructure/chapters/chapter8_automated_circuits/master_8_6.py
[study-115]: ../chapter8_automated_circuits/exercises/part7_attribution_graphs/8.7_Circuit_Tracing_with_Attribution_Graphs_exercises.ipynb
[study-116]: ../chapter8_automated_circuits/exercises/part7_attribution_graphs/8.7_Circuit_Tracing_with_Attribution_Graphs_solutions.ipynb
[study-117]: ../chapter8_automated_circuits/instructions/pages/07_%5B8.7%5D_Circuit_Tracing_with_Attribution_Graphs.md
[study-118]: ../infrastructure/chapters/chapter8_automated_circuits/master_8_7.py
[study-119]: ../chapter8_automated_circuits/exercises/part8_circuit_benchmarks/8.8_Benchmarking_Circuit_Discovery_on_Compiled_Transformers_exercises.ipynb
[study-120]: ../chapter8_automated_circuits/exercises/part8_circuit_benchmarks/8.8_Benchmarking_Circuit_Discovery_on_Compiled_Transformers_solutions.ipynb
[study-121]: ../chapter8_automated_circuits/instructions/pages/08_%5B8.8%5D_Benchmarking_Circuit_Discovery_on_Compiled_Transformers.md
[study-122]: ../infrastructure/chapters/chapter8_automated_circuits/master_8_8.py
[study-123]: ../chapter9_alignment_interpretability/exercises/part1_refusal_directions/9.1_Refusal_Directions_and_Representation_Engineering_exercises.ipynb
[study-124]: ../chapter9_alignment_interpretability/exercises/part1_refusal_directions/9.1_Refusal_Directions_and_Representation_Engineering_solutions.ipynb
[study-125]: ../chapter9_alignment_interpretability/instructions/pages/01_%5B9.1%5D_Refusal_Directions_and_Representation_Engineering.md
[study-126]: ../infrastructure/chapters/chapter9_alignment_interpretability/master_9_1.py
[study-127]: ../chapter9_alignment_interpretability/exercises/part3_cot_faithfulness/9.3_Chain-of-Thought_Faithfulness_exercises.ipynb
[study-128]: ../chapter9_alignment_interpretability/exercises/part3_cot_faithfulness/9.3_Chain-of-Thought_Faithfulness_solutions.ipynb
[study-129]: ../chapter9_alignment_interpretability/instructions/pages/03_%5B9.3%5D_Chain-of-Thought_Faithfulness.md
[study-130]: ../infrastructure/chapters/chapter9_alignment_interpretability/master_9_3.py
[study-131]: ../chapter9_alignment_interpretability/exercises/part4_emergent_misalignment/9.4_Emergent_Misalignment_Detection_exercises.ipynb
[study-132]: ../chapter9_alignment_interpretability/exercises/part4_emergent_misalignment/9.4_Emergent_Misalignment_Detection_solutions.ipynb
[study-133]: ../chapter9_alignment_interpretability/instructions/pages/04_%5B9.4%5D_Emergent_Misalignment_Detection.md
[study-134]: ../infrastructure/chapters/chapter9_alignment_interpretability/master_9_4.py
[study-135]: ../chapter9_alignment_interpretability/exercises/part5_white_box_monitors/9.5_White-Box_Monitors_and_Red-Teaming_with_Internals_exercises.ipynb
[study-136]: ../chapter9_alignment_interpretability/exercises/part5_white_box_monitors/9.5_White-Box_Monitors_and_Red-Teaming_with_Internals_solutions.ipynb
[study-137]: ../chapter9_alignment_interpretability/instructions/pages/05_%5B9.5%5D_White-Box_Monitors_and_Red-Teaming_with_Internals.md
[study-138]: ../infrastructure/chapters/chapter9_alignment_interpretability/master_9_5.py
[study-139]: ../chapter9_alignment_interpretability/exercises/part6_rome_memit/9.6_Model_Editing_with_ROME_and_MEMIT_exercises.ipynb
[study-140]: ../chapter9_alignment_interpretability/exercises/part6_rome_memit/9.6_Model_Editing_with_ROME_and_MEMIT_solutions.ipynb
[study-141]: ../chapter9_alignment_interpretability/instructions/pages/06_%5B9.6%5D_Model_Editing_with_ROME_and_MEMIT.md
[study-142]: ../infrastructure/chapters/chapter9_alignment_interpretability/master_9_6.py
[study-143]: ../chapter10_capstone_research_sprint/exercises/part1_capstone_research_sprint/10.1_Capstone_Research_Sprint_exercises.ipynb
[study-144]: ../chapter10_capstone_research_sprint/exercises/part1_capstone_research_sprint/10.1_Capstone_Research_Sprint_solutions.ipynb
[study-145]: ../chapter10_capstone_research_sprint/instructions/pages/01_%5B10.1%5D_Capstone_Research_Sprint.md
[study-146]: ../infrastructure/chapters/chapter10_capstone_research_sprint/master_10_1.py
[study-147]: ../chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/11.1_PCA_SVD_and_Geometry_Controls_exercises.ipynb
[study-148]: ../chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/11.1_PCA_SVD_and_Geometry_Controls_solutions.ipynb
[study-149]: ../chapter11_representation_geometry/instructions/pages/01_%5B11.1%5D_PCA_SVD_and_Geometry_Controls.md
[study-150]: ../infrastructure/chapters/chapter11_representation_geometry/master_11_1.py
[study-151]: ../chapter11_representation_geometry/exercises/part2_cca_svcca_alignment/11.2_CCA_SVCCA_and_Representation_Alignment_exercises.ipynb
[study-152]: ../chapter11_representation_geometry/exercises/part2_cca_svcca_alignment/11.2_CCA_SVCCA_and_Representation_Alignment_solutions.ipynb
[study-153]: ../chapter11_representation_geometry/instructions/pages/02_%5B11.2%5D_CCA_SVCCA_and_Representation_Alignment.md
[study-154]: ../infrastructure/chapters/chapter11_representation_geometry/master_11_2.py
[study-155]: ../chapter11_representation_geometry/exercises/part3_causal_abstraction_das/11.3_Causal_Abstraction_and_Distributed_Alignment_Search_exercises.ipynb
[study-156]: ../chapter11_representation_geometry/exercises/part3_causal_abstraction_das/11.3_Causal_Abstraction_and_Distributed_Alignment_Search_solutions.ipynb
[study-157]: ../chapter11_representation_geometry/instructions/pages/03_%5B11.3%5D_Causal_Abstraction_and_Distributed_Alignment_Search.md
[study-158]: ../infrastructure/chapters/chapter11_representation_geometry/master_11_3.py
[study-159]: ../chapter11_representation_geometry/exercises/part4_concept_erasure/11.4_Concept_Erasure_exercises.ipynb
[study-160]: ../chapter11_representation_geometry/exercises/part4_concept_erasure/11.4_Concept_Erasure_solutions.ipynb
[study-161]: ../chapter11_representation_geometry/instructions/pages/04_%5B11.4%5D_Concept_Erasure.md
[study-162]: ../infrastructure/chapters/chapter11_representation_geometry/master_11_4.py
[study-163]: ../chapter11_representation_geometry/exercises/part5_contrast_consistent_search/11.5_Contrast-Consistent_Search_exercises.ipynb
[study-164]: ../chapter11_representation_geometry/exercises/part5_contrast_consistent_search/11.5_Contrast-Consistent_Search_solutions.ipynb
[study-165]: ../chapter11_representation_geometry/instructions/pages/05_%5B11.5%5D_Contrast-Consistent_Search.md
[study-166]: ../infrastructure/chapters/chapter11_representation_geometry/master_11_5.py
[study-167]: ../chapter12_vlm_interpretability/exercises/part1_clip_siglip/12.1_CLIP_and_SigLIP_from_Scratch_solutions.ipynb
[study-168]: ../chapter12_vlm_interpretability/instructions/pages/01_%5B12.1%5D_CLIP_and_SigLIP_from_Scratch.md
[study-169]: ../infrastructure/chapters/chapter12_vlm_interpretability/master_12_1.py
[study-170]: ../chapter12_vlm_interpretability/exercises/part3_mini_vlm_from_scratch/12.3_Mini_VLM_from_Scratch_solutions.ipynb
[study-171]: ../chapter12_vlm_interpretability/instructions/pages/03_%5B12.3%5D_Mini_VLM_from_Scratch.md
[study-172]: ../infrastructure/chapters/chapter12_vlm_interpretability/master_12_3.py
[study-173]: ../chapter12_vlm_interpretability/exercises/part5_object_hallucination/12.5_Object_Hallucination_and_Modality_Arbitration_exercises.ipynb
[study-174]: ../chapter12_vlm_interpretability/exercises/part5_object_hallucination/12.5_Object_Hallucination_and_Modality_Arbitration_solutions.ipynb
[study-175]: ../chapter12_vlm_interpretability/instructions/pages/05_%5B12.5%5D_Object_Hallucination_and_Modality_Arbitration.md
[study-176]: ../infrastructure/chapters/chapter12_vlm_interpretability/master_12_5.py
[study-177]: ../chapter13_image_generation_interpretability/exercises/part1_diffusion_cross_attention/13.1_Diffusion_Models_and_Cross-Attention_Attribution_exercises.ipynb
[study-178]: ../chapter13_image_generation_interpretability/exercises/part1_diffusion_cross_attention/13.1_Diffusion_Models_and_Cross-Attention_Attribution_solutions.ipynb
[study-179]: ../chapter13_image_generation_interpretability/instructions/pages/01_%5B13.1%5D_Diffusion_Models_and_Cross-Attention_Attribution.md
[study-180]: ../infrastructure/chapters/chapter13_image_generation_interpretability/master_13_1.py
[study-181]: ../chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching/13.2_Denoising-Time_Causal_Patching_exercises.ipynb
[study-182]: ../chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching/13.2_Denoising-Time_Causal_Patching_solutions.ipynb
[study-183]: ../chapter13_image_generation_interpretability/instructions/pages/02_%5B13.2%5D_Denoising-Time_Causal_Patching.md
[study-184]: ../infrastructure/chapters/chapter13_image_generation_interpretability/master_13_2.py
[study-185]: ../chapter14_jepa_world_models/exercises/part1_ijepa_from_scratch/14.1_I-JEPA_from_Scratch_exercises.ipynb
[study-186]: ../chapter14_jepa_world_models/exercises/part1_ijepa_from_scratch/14.1_I-JEPA_from_Scratch_solutions.ipynb
[study-187]: ../chapter14_jepa_world_models/instructions/pages/01_%5B14.1%5D_I-JEPA_from_Scratch.md
[study-188]: ../infrastructure/chapters/chapter14_jepa_world_models/master_14_1.py
[study-189]: ../chapter14_jepa_world_models/exercises/part2_vjepa/14.2_V-JEPA_and_Video_World_Models_exercises.ipynb
[study-190]: ../chapter14_jepa_world_models/exercises/part2_vjepa/14.2_V-JEPA_and_Video_World_Models_solutions.ipynb
[study-191]: ../chapter14_jepa_world_models/instructions/pages/02_%5B14.2%5D_V-JEPA_and_Video_World_Models.md
[study-192]: ../infrastructure/chapters/chapter14_jepa_world_models/master_14_2.py
[study-193]: ../chapter14_jepa_world_models/exercises/part5_maze_transformers/14.5_Maze_Transformers_and_World_Models_exercises.ipynb
[study-194]: ../chapter14_jepa_world_models/exercises/part5_maze_transformers/14.5_Maze_Transformers_and_World_Models_solutions.ipynb
[study-195]: ../chapter14_jepa_world_models/instructions/pages/05_%5B14.5%5D_Maze_Transformers_and_World_Models.md
[study-196]: ../infrastructure/chapters/chapter14_jepa_world_models/master_14_5.py
[study-197]: ../chapter15_peft_misalignment/exercises/part1_lora_dora/15.1_LoRA_and_DoRA_from_Scratch_exercises.ipynb
[study-198]: ../chapter15_peft_misalignment/exercises/part1_lora_dora/15.1_LoRA_and_DoRA_from_Scratch_solutions.ipynb
[study-199]: ../chapter15_peft_misalignment/instructions/pages/01_%5B15.1%5D_LoRA_and_DoRA_from_Scratch.md
[study-200]: ../infrastructure/chapters/chapter15_peft_misalignment/master_15_1.py
[study-201]: ../chapter15_peft_misalignment/exercises/part2_lora_vs_full_finetuning/15.2_LoRA_vs_Full_Fine-Tuning_exercises.ipynb
[study-202]: ../chapter15_peft_misalignment/exercises/part2_lora_vs_full_finetuning/15.2_LoRA_vs_Full_Fine-Tuning_solutions.ipynb
[study-203]: ../chapter15_peft_misalignment/instructions/pages/02_%5B15.2%5D_LoRA_vs_Full_Fine-Tuning.md
[study-204]: ../infrastructure/chapters/chapter15_peft_misalignment/master_15_2.py
[study-205]: ../chapter15_peft_misalignment/exercises/part3_qlora_from_scratch/15.3_QLoRA_from_Scratch_exercises.ipynb
[study-206]: ../chapter15_peft_misalignment/exercises/part3_qlora_from_scratch/15.3_QLoRA_from_Scratch_solutions.ipynb
[study-207]: ../chapter15_peft_misalignment/instructions/pages/03_%5B15.3%5D_QLoRA_from_Scratch.md
[study-208]: ../infrastructure/chapters/chapter15_peft_misalignment/master_15_3.py
[study-209]: ../chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley/16.1_Exact_Shapley_on_Ground-Truth_Games_solutions.ipynb
[study-210]: ../chapter16_shapley_attribution_baselines/instructions/pages/01_%5B16.1%5D_Exact_Shapley_on_Ground-Truth_Games.md
[study-211]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_1.py
[study-212]: ../chapter16_shapley_attribution_baselines/exercises/part2_kernelshap_partitionshap/16.2_KernelSHAP_and_PartitionSHAP_solutions.ipynb
[study-213]: ../chapter16_shapley_attribution_baselines/instructions/pages/02_%5B16.2%5D_KernelSHAP_and_PartitionSHAP.md
[study-214]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_2.py
[study-215]: ../chapter16_shapley_attribution_baselines/exercises/part3_model_specific_shap/16.3_TreeSHAP_DeepSHAP_and_GradientSHAP_Practice_exercises.ipynb
[study-216]: ../chapter16_shapley_attribution_baselines/exercises/part3_model_specific_shap/16.3_TreeSHAP_DeepSHAP_and_GradientSHAP_Practice_solutions.ipynb
[study-217]: ../chapter16_shapley_attribution_baselines/instructions/pages/03_%5B16.3%5D_TreeSHAP_DeepSHAP_and_GradientSHAP_Practice.md
[study-218]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_3.py
[study-219]: ../chapter16_shapley_attribution_baselines/exercises/part4_sage/16.4_SAGE_Global_Importance_exercises.ipynb
[study-220]: ../chapter16_shapley_attribution_baselines/exercises/part4_sage/16.4_SAGE_Global_Importance_solutions.ipynb
[study-221]: ../chapter16_shapley_attribution_baselines/instructions/pages/04_%5B16.4%5D_SAGE_Global_Importance.md
[study-222]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_4.py
[study-223]: ../chapter16_shapley_attribution_baselines/exercises/part5_shapley_interactions/16.5_Shapley_Interactions_exercises.ipynb
[study-224]: ../chapter16_shapley_attribution_baselines/exercises/part5_shapley_interactions/16.5_Shapley_Interactions_solutions.ipynb
[study-225]: ../chapter16_shapley_attribution_baselines/instructions/pages/05_%5B16.5%5D_Shapley_Interactions.md
[study-226]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_5.py
[study-227]: ../chapter16_shapley_attribution_baselines/exercises/part6_spex_proxyspex/16.6_Sparse_Interaction_Discovery_with_SPEX_and_ProxySPEX_exercises.ipynb
[study-228]: ../chapter16_shapley_attribution_baselines/exercises/part6_spex_proxyspex/16.6_Sparse_Interaction_Discovery_with_SPEX_and_ProxySPEX_solutions.ipynb
[study-229]: ../chapter16_shapley_attribution_baselines/instructions/pages/06_%5B16.6%5D_Sparse_Interaction_Discovery_with_SPEX_and_ProxySPEX.md
[study-230]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_6.py
[study-231]: ../chapter16_shapley_attribution_baselines/exercises/part7_token_region_shap/16.7_TokenSHAP_and_Region_SHAP_for_LLMs_and_VLMs_exercises.ipynb
[study-232]: ../chapter16_shapley_attribution_baselines/exercises/part7_token_region_shap/16.7_TokenSHAP_and_Region_SHAP_for_LLMs_and_VLMs_solutions.ipynb
[study-233]: ../chapter16_shapley_attribution_baselines/instructions/pages/07_%5B16.7%5D_TokenSHAP_and_Region_SHAP_for_LLMs_and_VLMs.md
[study-234]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_7.py
[study-235]: ../chapter16_shapley_attribution_baselines/exercises/part8_shap_vs_patching/16.8_SHAP_vs_Activation_Patching_exercises.ipynb
[study-236]: ../chapter16_shapley_attribution_baselines/exercises/part8_shap_vs_patching/16.8_SHAP_vs_Activation_Patching_solutions.ipynb
[study-237]: ../chapter16_shapley_attribution_baselines/instructions/pages/08_%5B16.8%5D_SHAP_vs_Activation_Patching.md
[study-238]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_8.py
[study-239]: ../chapter16_shapley_attribution_baselines/exercises/part9_data_shapley/16.9_Data_Shapley_in_One_Training_Run_exercises.ipynb
[study-240]: ../chapter16_shapley_attribution_baselines/exercises/part9_data_shapley/16.9_Data_Shapley_in_One_Training_Run_solutions.ipynb
[study-241]: ../chapter16_shapley_attribution_baselines/instructions/pages/09_%5B16.9%5D_Data_Shapley_in_One_Training_Run.md
[study-242]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_9.py
[study-243]: ../chapter16_shapley_attribution_baselines/exercises/part10_influence_tracin/16.10_Influence_Functions_and_TracIn_exercises.ipynb
[study-244]: ../chapter16_shapley_attribution_baselines/exercises/part10_influence_tracin/16.10_Influence_Functions_and_TracIn_solutions.ipynb
[study-245]: ../chapter16_shapley_attribution_baselines/instructions/pages/10_%5B16.10%5D_Influence_Functions_and_TracIn.md
[study-246]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_10.py
[study-247]: ../chapter16_shapley_attribution_baselines/exercises/part11_datamodels_cartography/16.11_Datamodels_and_Dataset_Cartography_exercises.ipynb
[study-248]: ../chapter16_shapley_attribution_baselines/exercises/part11_datamodels_cartography/16.11_Datamodels_and_Dataset_Cartography_solutions.ipynb
[study-249]: ../chapter16_shapley_attribution_baselines/instructions/pages/11_%5B16.11%5D_Datamodels_and_Dataset_Cartography.md
[study-250]: ../infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_11.py
[study-251]: ../chapter17_training_dynamics/exercises/part1_checkpoint_archaeology/17.1_Checkpoint_Archaeology_and_Grokking_exercises.ipynb
[study-252]: ../chapter17_training_dynamics/exercises/part1_checkpoint_archaeology/17.1_Checkpoint_Archaeology_and_Grokking_solutions.ipynb
[study-253]: ../chapter17_training_dynamics/instructions/pages/01_%5B17.1%5D_Checkpoint_Archaeology_and_Grokking.md
[study-254]: ../infrastructure/chapters/chapter17_training_dynamics/master_17_1.py
[study-255]: ../chapter1_transformer_interp/exercises/part52_grokking_and_modular_arithmetic/1.5.2_Grokking_%26_Modular_Arithmetic_exercises.ipynb
[study-256]: ../chapter0_fundamentals/exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb
[study-257]: ../chapter0_fundamentals/exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb
[study-258]: ../chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb
[study-259]: ../chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb
[study-260]: ../chapter16_shapley_attribution_baselines/exercises/part13_treeshap/16.13_TreeSHAP_from_First_Principles_exercises.ipynb
[study-261]: ../chapter16_shapley_attribution_baselines/exercises/part13_treeshap/16.13_TreeSHAP_from_First_Principles_solutions.ipynb
[study-262]: ../chapter16_shapley_attribution_baselines/exercises/part14_deepshap/16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP_exercises.ipynb
[study-263]: ../chapter16_shapley_attribution_baselines/exercises/part14_deepshap/16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP_solutions.ipynb
[study-264]: ../chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/16.15_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate_exercises.ipynb
[study-265]: ../chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/16.15_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate_solutions.ipynb
[study-266]: ../docs/mastery/original_materials.md
[study-267]: units/circuit_prediction/exercises.ipynb
[study-268]: units/shapley_prediction/exercises.ipynb
[study-269]: units/sage_prediction/exercises.ipynb
[study-270]: units/ssd_prediction/exercises.ipynb
[study-271]: units/graph_prediction/exercises.ipynb
[study-272]: units/sae_prediction/exercises.ipynb
[study-273]: ../infrastructure/extension/syllabus.yaml

## More practice for 8.4 ACDC

After the 8.4 lesson, use the [own-method guided public Practice notebook][acdc-own-practice] and its
[instructions][acdc-own-instructions] to build the GPT-2 callback yourself, with visible sender/edge methods,
immediate CPU checks, controls and play. Both are available as native downloads when you select 8.4 in the reader.
The own-method GPT-2 run and its panel including 20 random-circuit controls remain pending; native parity
tolerances are **UNSET** and must be resolved before that model run. Reference answers remain public.

[acdc-own-practice]: units/acdc_own_method/exercises.ipynb
[acdc-own-instructions]: units/acdc_own_method/README.md


[study-274]: ../chapter5_modern_architectures/exercises/part6_linear_attention_retnet_rwkv/5.6_Linear_Attention_RetNet_and_RWKV_exercises.ipynb
[study-275]: ../chapter5_modern_architectures/exercises/part6_linear_attention_retnet_rwkv/5.6_Linear_Attention_RetNet_and_RWKV_solutions.ipynb
[study-276]: ../chapter5_modern_architectures/instructions/pages/06_%5B5.6%5D_Linear_Attention_RetNet_and_RWKV.md
[study-277]: ../infrastructure/chapters/chapter5_modern_architectures/master_5_6.py
[study-278]: ../chapter5_modern_architectures/exercises/part7_griffin_xlstm_hyena/5.7_Griffin_xLSTM_Hyena_and_Hybrid_Models_exercises.ipynb
[study-279]: ../chapter5_modern_architectures/exercises/part7_griffin_xlstm_hyena/5.7_Griffin_xLSTM_Hyena_and_Hybrid_Models_solutions.ipynb
[study-280]: ../chapter5_modern_architectures/instructions/pages/07_%5B5.7%5D_Griffin_xLSTM_Hyena_and_Hybrid_Models.md
[study-281]: ../infrastructure/chapters/chapter5_modern_architectures/master_5_7.py
[study-282]: ../chapter6_sparse_feature_methods/exercises/part3_sae_pathologies_dashboards/6.3_Feature_Splitting_Absorption_and_Dashboards_exercises.ipynb
[study-283]: ../chapter6_sparse_feature_methods/exercises/part3_sae_pathologies_dashboards/6.3_Feature_Splitting_Absorption_and_Dashboards_solutions.ipynb
[study-284]: ../chapter6_sparse_feature_methods/instructions/pages/03_%5B6.3%5D_Feature_Splitting_Absorption_and_Dashboards.md
[study-285]: ../infrastructure/chapters/chapter6_sparse_feature_methods/master_6_3.py
[study-286]: ../chapter7_activation_to_language/exercises/part6_interpretability_agents/7.6_Automated_Interpretability_Agents_exercises.ipynb
[study-287]: ../chapter7_activation_to_language/exercises/part6_interpretability_agents/7.6_Automated_Interpretability_Agents_solutions.ipynb
[study-288]: ../chapter7_activation_to_language/instructions/pages/06_%5B7.6%5D_Automated_Interpretability_Agents.md
[study-289]: ../infrastructure/chapters/chapter7_activation_to_language/master_7_6.py
[study-290]: ../chapter9_alignment_interpretability/exercises/part2_truthfulness_hallucination/9.2_Truthfulness_Directions_and_Hallucination_Probes_exercises.ipynb
[study-291]: ../chapter9_alignment_interpretability/exercises/part2_truthfulness_hallucination/9.2_Truthfulness_Directions_and_Hallucination_Probes_solutions.ipynb
[study-292]: ../chapter9_alignment_interpretability/instructions/pages/02_%5B9.2%5D_Truthfulness_Directions_and_Hallucination_Probes.md
[study-293]: ../infrastructure/chapters/chapter9_alignment_interpretability/master_9_2.py
[study-294]: ../chapter9_alignment_interpretability/exercises/part7_unlearning/9.7_Unlearning_and_How_to_Evaluate_It_exercises.ipynb
[study-295]: ../chapter9_alignment_interpretability/exercises/part7_unlearning/9.7_Unlearning_and_How_to_Evaluate_It_solutions.ipynb
[study-296]: ../chapter9_alignment_interpretability/instructions/pages/07_%5B9.7%5D_Unlearning_and_How_to_Evaluate_It.md
[study-297]: ../infrastructure/chapters/chapter9_alignment_interpretability/master_9_7.py
[study-298]: ../chapter11_representation_geometry/exercises/part6_feature_manifolds/11.6_Feature_Manifolds_Days_Months_Space_and_Time_exercises.ipynb
[study-299]: ../chapter11_representation_geometry/exercises/part6_feature_manifolds/11.6_Feature_Manifolds_Days_Months_Space_and_Time_solutions.ipynb
[study-300]: ../chapter11_representation_geometry/instructions/pages/06_%5B11.6%5D_Feature_Manifolds_Days_Months_Space_and_Time.md
[study-301]: ../infrastructure/chapters/chapter11_representation_geometry/master_11_6.py
[study-302]: ../chapter12_vlm_interpretability/exercises/part2_clip_saes_typographic/12.2_CLIP_Features_SAEs_and_Typographic_Attacks_exercises.ipynb
[study-303]: ../chapter12_vlm_interpretability/exercises/part2_clip_saes_typographic/12.2_CLIP_Features_SAEs_and_Typographic_Attacks_solutions.ipynb
[study-304]: ../chapter12_vlm_interpretability/instructions/pages/02_%5B12.2%5D_CLIP_Features_SAEs_and_Typographic_Attacks.md
[study-305]: ../infrastructure/chapters/chapter12_vlm_interpretability/master_12_2.py
[study-306]: ../chapter12_vlm_interpretability/exercises/part4_visual_token_flow/12.4_Visual_Token_Flow_in_Real_VLMs_exercises.ipynb
[study-307]: ../chapter12_vlm_interpretability/exercises/part4_visual_token_flow/12.4_Visual_Token_Flow_in_Real_VLMs_solutions.ipynb
[study-308]: ../chapter12_vlm_interpretability/instructions/pages/04_%5B12.4%5D_Visual_Token_Flow_in_Real_VLMs.md
[study-309]: ../infrastructure/chapters/chapter12_vlm_interpretability/master_12_4.py
[study-310]: ../chapter12_vlm_interpretability/exercises/part6_multimodal_sparse_features/12.6_Multimodal_SAEs_Transcoders_and_Crosscoders_exercises.ipynb
[study-311]: ../chapter12_vlm_interpretability/exercises/part6_multimodal_sparse_features/12.6_Multimodal_SAEs_Transcoders_and_Crosscoders_solutions.ipynb
[study-312]: ../chapter12_vlm_interpretability/instructions/pages/06_%5B12.6%5D_Multimodal_SAEs_Transcoders_and_Crosscoders.md
[study-313]: ../infrastructure/chapters/chapter12_vlm_interpretability/master_12_6.py
[study-314]: ../chapter12_vlm_interpretability/exercises/part7_dense_vision_features/12.7_Dense_Vision_Features_DINO_and_SAM_exercises.ipynb
[study-315]: ../chapter12_vlm_interpretability/exercises/part7_dense_vision_features/12.7_Dense_Vision_Features_DINO_and_SAM_solutions.ipynb
[study-316]: ../chapter12_vlm_interpretability/instructions/pages/07_%5B12.7%5D_Dense_Vision_Features_DINO_and_SAM.md
[study-317]: ../infrastructure/chapters/chapter12_vlm_interpretability/master_12_7.py
[study-318]: ../chapter13_image_generation_interpretability/exercises/part3_concept_directions_lora/13.3_Concept_Directions_and_Diffusion_LoRA_exercises.ipynb
[study-319]: ../chapter13_image_generation_interpretability/exercises/part3_concept_directions_lora/13.3_Concept_Directions_and_Diffusion_LoRA_solutions.ipynb
[study-320]: ../chapter13_image_generation_interpretability/instructions/pages/03_%5B13.3%5D_Concept_Directions_and_Diffusion_LoRA.md
[study-321]: ../infrastructure/chapters/chapter13_image_generation_interpretability/master_13_3.py
[study-322]: ../chapter13_image_generation_interpretability/exercises/part4_diffusion_saes/13.4_Sparse_Autoencoders_for_Diffusion_Models_exercises.ipynb
[study-323]: ../chapter13_image_generation_interpretability/exercises/part4_diffusion_saes/13.4_Sparse_Autoencoders_for_Diffusion_Models_solutions.ipynb
[study-324]: ../chapter13_image_generation_interpretability/instructions/pages/04_%5B13.4%5D_Sparse_Autoencoders_for_Diffusion_Models.md
[study-325]: ../infrastructure/chapters/chapter13_image_generation_interpretability/master_13_4.py
[study-326]: ../chapter13_image_generation_interpretability/exercises/part5_dit_concept_attention/13.5_Diffusion_Transformers_and_ConceptAttention_exercises.ipynb
[study-327]: ../chapter13_image_generation_interpretability/exercises/part5_dit_concept_attention/13.5_Diffusion_Transformers_and_ConceptAttention_solutions.ipynb
[study-328]: ../chapter13_image_generation_interpretability/instructions/pages/05_%5B13.5%5D_Diffusion_Transformers_and_ConceptAttention.md
[study-329]: ../infrastructure/chapters/chapter13_image_generation_interpretability/master_13_5.py
[study-330]: ../chapter13_image_generation_interpretability/exercises/part6_autoregressive_image_tokens/13.6_Autoregressive_Image_Tokens_exercises.ipynb
[study-331]: ../chapter13_image_generation_interpretability/exercises/part6_autoregressive_image_tokens/13.6_Autoregressive_Image_Tokens_solutions.ipynb
[study-332]: ../chapter13_image_generation_interpretability/instructions/pages/06_%5B13.6%5D_Autoregressive_Image_Tokens.md
[study-333]: ../infrastructure/chapters/chapter13_image_generation_interpretability/master_13_6.py
[study-334]: ../chapter14_jepa_world_models/exercises/part3_action_conditioned_planning/14.3_Action-Conditioned_World_Models_and_Latent_Planning_exercises.ipynb
[study-335]: ../chapter14_jepa_world_models/exercises/part3_action_conditioned_planning/14.3_Action-Conditioned_World_Models_and_Latent_Planning_solutions.ipynb
[study-336]: ../chapter14_jepa_world_models/instructions/pages/03_%5B14.3%5D_Action-Conditioned_World_Models_and_Latent_Planning.md
[study-337]: ../infrastructure/chapters/chapter14_jepa_world_models/master_14_3.py
[study-338]: ../chapter14_jepa_world_models/exercises/part4_jepa_family/14.4_The_JEPA_Family_exercises.ipynb
[study-339]: ../chapter14_jepa_world_models/exercises/part4_jepa_family/14.4_The_JEPA_Family_solutions.ipynb
[study-340]: ../chapter14_jepa_world_models/instructions/pages/04_%5B14.4%5D_The_JEPA_Family.md
[study-341]: ../infrastructure/chapters/chapter14_jepa_world_models/master_14_4.py
[study-342]: ../chapter14_jepa_world_models/exercises/part6_sudoku_transformers/14.6_Sudoku_and_Constraint_World_Models_exercises.ipynb
[study-343]: ../chapter14_jepa_world_models/exercises/part6_sudoku_transformers/14.6_Sudoku_and_Constraint_World_Models_solutions.ipynb
[study-344]: ../chapter14_jepa_world_models/instructions/pages/06_%5B14.6%5D_Sudoku_and_Constraint_World_Models.md
[study-345]: ../infrastructure/chapters/chapter14_jepa_world_models/master_14_6.py
[study-346]: ../chapter14_jepa_world_models/exercises/part7_recurrent_world_models/14.7_Recurrent_World_Models_Dreamer_TD-MPC2_and_IRIS_exercises.ipynb
[study-347]: ../chapter14_jepa_world_models/exercises/part7_recurrent_world_models/14.7_Recurrent_World_Models_Dreamer_TD-MPC2_and_IRIS_solutions.ipynb
[study-348]: ../chapter14_jepa_world_models/instructions/pages/07_%5B14.7%5D_Recurrent_World_Models_Dreamer_TD-MPC2_and_IRIS.md
[study-349]: ../infrastructure/chapters/chapter14_jepa_world_models/master_14_7.py
[study-350]: ../chapter15_peft_misalignment/exercises/part4_lora_phase_transitions/15.4_LoRA_Phase_Transitions_and_Misalignment_Proxies_exercises.ipynb
[study-351]: ../chapter15_peft_misalignment/exercises/part4_lora_phase_transitions/15.4_LoRA_Phase_Transitions_and_Misalignment_Proxies_solutions.ipynb
[study-352]: ../chapter15_peft_misalignment/instructions/pages/04_%5B15.4%5D_LoRA_Phase_Transitions_and_Misalignment_Proxies.md
[study-353]: ../infrastructure/chapters/chapter15_peft_misalignment/master_15_4.py
[study-354]: ../chapter15_peft_misalignment/exercises/part5_vlm_lora/15.5_VLM_LoRA_exercises.ipynb
[study-355]: ../chapter15_peft_misalignment/exercises/part5_vlm_lora/15.5_VLM_LoRA_solutions.ipynb
[study-356]: ../chapter15_peft_misalignment/instructions/pages/05_%5B15.5%5D_VLM_LoRA.md
[study-357]: ../infrastructure/chapters/chapter15_peft_misalignment/master_15_5.py
[study-358]: ../chapter17_training_dynamics/exercises/part2_induction_head_emergence/17.2_Induction_Head_Emergence_exercises.ipynb
[study-359]: ../chapter17_training_dynamics/exercises/part2_induction_head_emergence/17.2_Induction_Head_Emergence_solutions.ipynb
[study-360]: ../chapter17_training_dynamics/instructions/pages/02_%5B17.2%5D_Induction_Head_Emergence.md
[study-361]: ../infrastructure/chapters/chapter17_training_dynamics/master_17_2.py
[study-362]: ../chapter17_training_dynamics/exercises/part3_finetuning_diffs_emergence/17.3_Fine-Tuning_Diffs_and_Capability_Emergence_exercises.ipynb
[study-363]: ../chapter17_training_dynamics/exercises/part3_finetuning_diffs_emergence/17.3_Fine-Tuning_Diffs_and_Capability_Emergence_solutions.ipynb
[study-364]: ../chapter17_training_dynamics/instructions/pages/03_%5B17.3%5D_Fine-Tuning_Diffs_and_Capability_Emergence.md
[study-365]: ../infrastructure/chapters/chapter17_training_dynamics/master_17_3.py
[study-366]: ../chapter17_training_dynamics/exercises/part4_cross_architecture_dynamics/17.4_Training_Dynamics_Across_Architectures_exercises.ipynb
[study-367]: ../chapter17_training_dynamics/exercises/part4_cross_architecture_dynamics/17.4_Training_Dynamics_Across_Architectures_solutions.ipynb
[study-368]: ../chapter17_training_dynamics/instructions/pages/04_%5B17.4%5D_Training_Dynamics_Across_Architectures.md
[study-369]: ../infrastructure/chapters/chapter17_training_dynamics/master_17_4.py
[study-370]: ../chapter18_intrinsic_interpretability/exercises/part1_concept_bottlenecks_backpacks_iit/18.1_Concept_Bottlenecks_Backpacks_and_Interchange_Intervention_Training_exercises.ipynb
[study-371]: ../chapter18_intrinsic_interpretability/exercises/part1_concept_bottlenecks_backpacks_iit/18.1_Concept_Bottlenecks_Backpacks_and_Interchange_Intervention_Training_solutions.ipynb
[study-372]: ../chapter18_intrinsic_interpretability/instructions/pages/01_%5B18.1%5D_Concept_Bottlenecks_Backpacks_and_Interchange_Intervention_Training.md
[study-373]: ../infrastructure/chapters/chapter18_intrinsic_interpretability/master_18_1.py
[study-374]: ../chapter18_intrinsic_interpretability/exercises/part2_sparse_modular_kans/18.2_Sparse_and_Modular_Training_KANs_and_Sparse_Trees_exercises.ipynb
[study-375]: ../chapter18_intrinsic_interpretability/exercises/part2_sparse_modular_kans/18.2_Sparse_and_Modular_Training_KANs_and_Sparse_Trees_solutions.ipynb
[study-376]: ../chapter18_intrinsic_interpretability/instructions/pages/02_%5B18.2%5D_Sparse_and_Modular_Training_KANs_and_Sparse_Trees.md
[study-377]: ../infrastructure/chapters/chapter18_intrinsic_interpretability/master_18_2.py
