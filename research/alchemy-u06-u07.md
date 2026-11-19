# Alchemy units 6 and 7: what CS 312 has released (checked 2026-11-19)

Research for the "Not yet specified" entry of [the decision map (#4)](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/4) and the "Alchemy units 6 and 7: pending release" note of [Spec #39](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/39). It follows the earlier [availability research (#5)](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/5). Facts only: no design choices, no Decisions line, no training, no GPU jobs. Primary sources only; every fact carries a link. Where a statement is my reading or arithmetic rather than something a source says, it is marked **(inference)** or **(arithmetic)**.

## Verdict

| Local unit | Upstream unit | Released? | Evidence |
| --- | --- | --- | --- |
| u06 | Assignment 6, **Generalization** (data-efficient algorithms), modality **DNA** | **Yes.** Public from 2026-11-11 (the schedule dates it 10 Nov). Lectures 10 and 11 are up. Its quiz is not. | [Release commit](https://github.com/deep-learning-alchemy/deep-learning-alchemy.github.io/commit/1b28b8be5da40f415b9c0b342a7d264efe2d61ed), [handout](https://deep-learning-alchemy.github.io/assets/assignments/a6_generalization.pdf), [code at a6_release](https://github.com/deep-learning-alchemy/assignments/tree/7f0db017cb89e31c7d387b907480f977dd033127) |
| u07 | Assignment 7, **"Exotic: ???"** | **No, not yet.** Scheduled "Assignment 7 out" on Thu 19 Nov. No handout, code branch, lecture or quiz exists. | [Schedule](https://deep-learning-alchemy.github.io/#schedule); [branch list](https://github.com/deep-learning-alchemy/assignments/branches) has `a1_release`–`a6_release` only |

**Timing caveat for u07.** My checks ran at about 2026-11-19 01:20 UTC, which is 2026-11-18 17:20 Pacific, so the scheduled 19 Nov release had not yet arrived on the course's own clock. "Not released" therefore means "not yet", and it should be re-checked after 19 Nov Pacific (commands at the end).

## Source snapshot (what was checked, and its pins)

- Course homepage, served last modified 2026-11-17 21:37 UTC: [deep-learning-alchemy.github.io](https://deep-learning-alchemy.github.io/), source repo HEAD [`45ea5f68`](https://github.com/deep-learning-alchemy/deep-learning-alchemy.github.io/commit/45ea5f685d47cf49f210646f055e470a074feaef) (2026-11-17, "Add Generalization Lecture 2 compression slides"). The page says it is Stanford CS 312, Fall 2026, instructors Tatsunori Hashimoto and Suhas Kotha.
- Assignments repository: [`main` 6e6c1544](https://github.com/deep-learning-alchemy/assignments/commit/6e6c1544c25509497dfa8af62fb67fcab827dac0) (2026-11-12) and [`a6_release` 7f0db017](https://github.com/deep-learning-alchemy/assignments/commit/7f0db017cb89e31c7d387b907480f977dd033127) (2026-11-12, "Document A6 Hugging Face artifacts and local GPU setup"). The release before it is [05e2b982](https://github.com/deep-learning-alchemy/assignments/commit/05e2b9828c7a5aa69aea88507d42b295e861dc0f) "Release Assignment 6 generalization starter and references" (2026-11-11).
- A6 handout PDF (PDF creation date 2026-11-11), SHA-256 `21076534bcde35b81205744dcb9d1ad97bedba9652209e590131d88db576d6df`. Lecture 10 PDF SHA-256 `340d07e475bc5d1047593df1dfb38608f0e68cdfc94ebb5e0e7241412995aeed`; lecture 11 `8cae11ee1f444f4fa4a9fa227eb03fbdea46ea724bc97a25079947a8c411f6b5`.
- A6 data bundle: Hugging Face dataset [`kothasuhas/dl_alchemy_a6_generalization`](https://huggingface.co/datasets/kothasuhas/dl_alchemy_a6_generalization) at revision [`145bebd1`](https://huggingface.co/datasets/kothasuhas/dl_alchemy_a6_generalization/tree/145bebd16541824ae099735b57a830aeb410c716), last modified 2026-11-12; it lists a [manifest.json](https://huggingface.co/datasets/kothasuhas/dl_alchemy_a6_generalization/raw/145bebd16541824ae099735b57a830aeb410c716/manifest.json) with sizes and SHA-256 checksums. The assignment README pins this same revision in its download command.
- Everything was read, not executed: no `uv sync`, no checkpoint loads, no training. Only the PDFs, slides, small JSON bundles, one history file and the manifest were downloaded, into a scratch directory outside the repository.

## Numbering and dates

The Spec's labels map to the course's seven units as: u01 Basics, u02 Optimization 1, u03 Optimization 2 (landscapes), u04 Architecture 1, u05 Architecture 2, u06 Generalization, u07 Exotic. The homepage lists exactly that sequence ([assignments list](https://deep-learning-alchemy.github.io/#schedule)): "Assignments 1 through 5 will study language model pre-training at a small scale… Starting with Assignment 6, we will switch to a new modality: DNA", and "Exotic: ???". Grading is 7 quizzes with 2 dropped, 85% of the grade; assignments are ungraded and used only as partial credit when cited in a quiz ([Coursework](https://deep-learning-alchemy.github.io/#schedule)).

Schedule rows for these units, copied as dates only:

| Date (2026) | Row | Status |
| --- | --- | --- |
| Tue 10 Nov | Quiz: Architecture 2 (u05); **Assignment 6 out**, due 19 Nov 1:00pm | A6 repository and handout went public on 11 Nov ([release commit](https://github.com/deep-learning-alchemy/deep-learning-alchemy.github.io/commit/1b28b8be5da40f415b9c0b342a7d264efe2d61ed), 2026-11-11 09:44 UTC) |
| Thu 12 Nov | Lecture: Generalization ([slides](https://deep-learning-alchemy.github.io/assets/slides/lecture_10.pdf)) | Published; added by [commit 24db8c82](https://github.com/deep-learning-alchemy/deep-learning-alchemy.github.io/commit/24db8c82d6681f0a2050273069c081b815d5cae5) on 12 Nov |
| Tue 17 Nov | Lecture: Generalization, part 2 ([slides](https://deep-learning-alchemy.github.io/assets/slides/lecture_11.pdf)) | Published 17 Nov |
| Thu 19 Nov | **Quiz: Generalization**, details "TBA"; **Assignment 7 out**, due 3 Dec 1:00pm | Neither published when checked |
| Tue 1 Dec | Lecture: Exotic, "TBA" | Not published |
| Thu 3 Dec | Quiz: Exotic, "TBA" | Not published |

The Spec's note says unit 6 "is due 10 November"; upstream's 10 Nov is the A6 **release** date and the A6 report is due 19 Nov.

## Unit 6: Generalization (released)

### Title and the questions it asks

Handout title: [Assignment 6: Generalization](https://deep-learning-alchemy.github.io/assets/assignments/a6_generalization.pdf) (homepage tag: "Generalization: Data-efficient algorithms"). Its framing: instead of one pass over a ~600M-token set, train on a fixed 30M-token DNA pool "as many times as we like" and "get the best possible held-out loss using these 30M tokens". The two baseline levers are repeating the data (epochs) and scaling parameters; the problems then test other ways to buy data efficiency.

Four problems (handout pp. 4–7):

1. **Architecture.** (a) Recall: parameter scaling from a supplied tuned dense d4–d10 suite; fit L(N) = L∞ + A·N^−α; is it monotone, do returns diminish, how do epochs and compute vary with size. (b) Recall: the A5 four-expert top-1 MoE vs dense d8, ΔL at equal active compute. (c) Recall: the A5 looped model (one prelude, six passes of a shared core, one coda) vs dense d8. (d) Exploration of data-efficient architectures at a fixed parameter budget.
2. **Regularization.** (a)(i) Recall: the LR × WD grid at a fixed 16 epochs, quadratic fit in log LR and log WD as in A2, and loss against LR×WD; how much larger is the best WD than the old 0.1 reference. (a)(ii) Hyperball vs AdamW with and without WD tuning. (b)(i) Attention-probability dropout (0.05, 0.15, 0.25) on a single model, dropout off at evaluation. (b)(ii) Independent ensembles from five supplied no-dropout models: loss vs ensemble size K = 1…5 averaged over all member subsets, a scaling law L(K) = L∞ + A·K^−α, paired resampling, "five members may not identify the infinite-ensemble limit". (b)(iii) How ensembling and dropout compose.
3. **Optimization.** (a) Linear vs constant LR schedules at several horizons, with sharpness reasoning. (b) Recall: batch size B ∈ {16, 64, 256, 1024, 4096} with LR and WD scaled by √(B/64). (c) SAM vs AdamW with the supplied sharpness probe. (d) Exploration of other optimizers.
4. **Fine-tuning.** Four starting points: related DNA (eukaryotic genic windows from OpenGenome2, excluding target and validation chunks), DCLM text (body transferred, DNA embedding and head re-initialized), nucleotide noise (IID A/C/G/T) and scratch. (a) Compare all four at 1, 4 and 16 passes on the 30k pool, tuning LR. (b) Fine-tuning pools of 1k, 3k, 10k, 30k at 16 passes; one scaling law per initialization; map pretraining benefit to an effective sample count. (c) Adaptations of the related-DNA checkpoint: WD tuning, decay toward the pretrained weights, frozen head, frozen body. (d) Exploration.

Lectures (summary of what they cover, not a copy; slide numbers are PDF pages): [lecture 10](https://deep-learning-alchemy.github.io/assets/slides/lecture_10.pdf), "Regularization and fine-tuning", 116 pp. — data-constrained regime (pp. 6–10), bias–variance, double descent and ridge (pp. 15–24), epoching and parameter scaling at 200M tokens of DCLM (pp. 28–43; the slides say the assignment itself is on DNA, p. 28), ensemble member scaling and seed science (pp. 46–52), dropout (p. 53), composing recipes (pp. 56–60), distillation (pp. 76–85), downstream and OOD (pp. 86–89), fine-tuning, effective samples, forgetting (pp. 91–115). [Lecture 11](https://deep-learning-alchemy.github.io/assets/slides/lecture_11.pdf), "Compression", 104 pp. — information theory, Kolmogorov complexity, Solomonoff induction and MDL (pp. 5–48), description length in architecture: looping, MoE, sequence methods (pp. 49–65), sharpness: batch size, learning rate, SAM (pp. 66–80), data filtering and augmentation (pp. 81–103). The handout cites Kim et al. (2025), Hernandez, Kaplan et al. (2021), Keskar et al. (2017) and Jiang et al. (2020). Two of these are linked from the schedule: [Pre-training under infinite compute](https://arxiv.org/abs/2509.14786) and [Scaling Laws for Transfer](https://arxiv.org/abs/2102.01293). The handout's own reference list is not in the PDF text, so I did not verify the other two. The handout also mentions "the DART task (Patel et al., 2024)" for a shuffled-sequence downstream check (handout p. 3); I identified this as [DART-Eval, arXiv 2412.05430](https://arxiv.org/abs/2412.05430) **(inference)**.

### Experiments, models and datasets

- **Model.** d8: eight blocks, width 512, QK normalization, 30,437,888 parameters, MLP intermediate 1792 ([`parameter_scaling.json`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/references/parameter_scaling.json), [`lr_wd_runs.json`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/references/lr_wd_runs.json)). It is the same d8 family as units 1–5. The A6 code imports the **A5** model classes for the MoE and looping variants ([`models.py`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/models.py)), so u06's code depends on u05's.
- **Recipe.** AdamW (0.9, 0.95), LR .003, WD 3.2, batch 64 (micro-batch 8), clipping 1, 32 epochs, 4,800 sequence exposures of warmup then linear decay to zero; "the default baseline is the sweep-selected 32-epoch D8" ([A6 README](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/README.md)). 32 epochs is 983.04M tokens of exposure over 30.72M unique tokens (handout p. 1).
- **Data.** A subset of the Arc Institute's [OpenGenome2](https://huggingface.co/datasets/arcinstitute/opengenome2) (the Evo 2 training corpus): metagenomes, eukaryotic genic windows, bacterial and archaeal genomes, mRNA, viral sequences, noncoding RNA, organelles, promoters. Character-level tokens, 19-token vocabulary (A/C/G/T, eleven ambiguity symbols, U, X, BOS, EOS), 1,024-token chunks, 30,000 training chunks (30.72M tokens) and a disjoint 1,000-chunk validation set (handout p. 2). The handout warns that neighbouring or homologous sequences can cross splits, so this tests generalization within the mixture and not to held-out species or genomes. There is **no student test split**: selection and reporting use the same validation set, and the handout says repeated selection "can make validation results optimistic" (handout p. 1). The prepared-data repository is pinned at [`620f5e89`](https://huggingface.co/datasets/kothasuhas/opengenome2_tokenized_n12361963_ctx1024_char19_bos_eos_iid/tree/620f5e892cd428813cfb233e54802f43cc8b892a).
- **Metric.** Next-token validation NLL (the supplied files call it `validation_nll`). One downstream check (DART, shuffled-vs-real regulatory sequences) appears only as a handout figure and "does not select recipes" (handout Fig. 1 caption).
- **Seeds.** Reference runs use model seed 42 and data seed 42, except the ensemble bank (seeds 42–46). The handout asks students to confirm main comparisons with a second seed and to pair seeds and data order across methods.

### What is supplied and what is not (the release's own list)

From [`RELEASE_PLAN.md`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/RELEASE_PLAN.md) and the [bundle manifest](https://huggingface.co/datasets/kothasuhas/dl_alchemy_a6_generalization/raw/145bebd16541824ae099735b57a830aeb410c716/manifest.json): 48 distinct recorded training runs, with histories mirrored to a public W&B project and as `histories/*.json.gz`.

| Topic | Supplied (recorded runs / files) | Not supplied (students train it) |
| --- | --- | --- |
| P1(a) parameter scaling | Seven tuned dense recipes d4–d10 with histories and checkpoints; the 8/16/32/64-epoch reference runs | — |
| P1(b–d) MoE, looping, exploration | Starter architecture code only | All runs, histories and checkpoints |
| P2(a)(i) LR × WD | The completed 16-epoch grid (21 runs in the small bundle) | — |
| P2(a)(ii) Hyperball | The two AdamW controls (WD 0.1 and WD 3.2, 32 epochs) | Hyperball runs |
| P2(b) dropout, ensembles | Five no-dropout members (seeds 42–46) with validation logits and per-K chunk losses | Dropout models, logits, ensemble banks, composition curves |
| P3(a) schedules | Five linear horizons and five endpoints of one continuous constant-LR run | Extra schedules |
| P3(b) batch size | B = 16/64/256/1024/4096 under √ scaling, with checkpoints and a sharpness probe | — |
| P3(c–d) SAM, other optimizers | The sharpness probe (`directional_sharpness`) | All SAM and optimizer runs |
| P4 fine-tuning | The three 10B-token pretrained checkpoints (`related.pt`, `dclm.pt`, `noise.pt`); the 30k and 1k nested-prefix splits | All fine-tuning runs, and the pretraining corpora and related-DNA selection indices |
| Held-out | — | "Instructor held-out metrics" are listed as **excluded** from the release |

Bundle size is about 2.25 GB; no account or token is needed to download it ([A6 README](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/README.md#use-your-own-gpu), [dataset card](https://huggingface.co/datasets/kothasuhas/dl_alchemy_a6_generalization)). The small reference JSONs (about 80 KB in total) are in the Git repository itself, so the scaling, LR×WD, batch, ensemble-curve and schedule analyses run with no GPU ("Analyze supplied data without a GPU" in the A6 README).

### What a held-out prediction would look like

**The unit 6 quiz is not published.** The schedule shows "Quiz: Generalization — TBA" for 19 Nov, there is no `quiz-generalization.html`, and no quiz PDF in the [site tree at `45ea5f68`](https://github.com/deep-learning-alchemy/deep-learning-alchemy.github.io/tree/45ea5f685d47cf49f210646f055e470a074feaef). What is established is the course's format and the shape of the A6 evidence.

Format, from the [homepage](https://deep-learning-alchemy.github.io/#schedule) and the latest published quiz, [Capacity (u05)](https://deep-learning-alchemy.github.io/quiz-capacity.html):

- A 40-minute quiz; "each quiz question will correspond to a thematic held-out experiment run by the instructors"; students get the description and a code diff and predict the outcome; questions span easy, medium and hard "depending on how much extrapolation they require from the assignment suggestions". The Capacity quiz has six questions and 90 points, states a reference recipe, applies each diff to the released starter at a named commit, and lets "evidence and reasoning" recover up to 75% of missed points.
- Example items on the homepage: rank variants by final validation loss, with "=" when within 0.01; predict the loss difference between a modified and a default model, taking the best of a stated LR set.
- Quiz pages appear after the quiz: the Stability quiz (29 Oct) was added on [2 Nov](https://github.com/deep-learning-alchemy/deep-learning-alchemy.github.io/commit/ad25525034e77cabe7ccd5d1a4847c2e03b3e78d) and the Capacity quiz (10 Nov) on [15 Nov](https://github.com/deep-learning-alchemy/deep-learning-alchemy.github.io/commit/fc03c8b5fcba9e80d555db1689d1598b42f3844d). The Generalization quiz page may therefore appear a few days after 19 Nov **(inference from two data points)**.

What the A6 numbers imply about question shape **(arithmetic on the supplied bundle, rounded; for scoping only, and the files are linked, not copied)**:

- Most A6 effects are small in NLL. Across the supplied dense suite d4–d10 the best final NLLs span about 1.259–1.270 (parameters 3.8M–59M, one seed); the five no-dropout seeds of the d8 baseline span 1.2608–1.2629; K = 1→5 ensembling moves 1.2619→1.2543; linear vs constant LR endpoints differ by at most about 0.007 across the five horizons. Against that, WD 0.1 vs 3.2 is 1.4057 vs 1.2627, and B = 4096 vs 64 is 1.3947 vs 1.2627. The course's own "= when within 0.01" rule fits this range.
- Source: [`parameter_scaling.json`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/references/parameter_scaling.json), [`ensemble_bank.json`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/references/ensemble_bank.json), [`ensemble_curves.json`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/references/ensemble_curves.json), [`schedules.json`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/references/schedules.json), [`wd_tuning.json`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/references/wd_tuning.json), [`batch_scaling.json`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/references/batch_scaling.json).
- Questions that could be answered from supplied runs alone are the ones in the "Supplied" column above (scaling, LR×WD, WD, batch, ensembles, schedules, and checkpoint-based probes). The MoE, looping, dropout, Hyperball, SAM and fine-tuning topics have **no supplied outcomes**; the quiz page for them would show the instructors' held-out results, which are not released.
- The handout has no answer key (unlike the A2 handout noted in [#5's research](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/5)).

### Compute needs

- **Course default:** "a single NVIDIA H100 on Modal" for d8 runs; Modal listed the H100 SXM5 at $3.95/hour on 2 Oct 2026, billed per second ([homepage](https://deep-learning-alchemy.github.io/#schedule)). Course students get a 48 GPU-hour budget per assignment and 2 concurrent GPUs in the course Modal workspace ([repo README](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/README.md)); these limits apply to students of the course, not to self-study.
- **One d8 baseline:** 30,437,888 parameters, about 15,000 optimizer steps (the manifest records 15,008), 983.04M tokens of exposure, and a published `training_flops` of 2.29×10¹⁷ ([`parameter_scaling.json`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/references/parameter_scaling.json)). **No wall-clock time is published, and none was measured here.** At sustained 50, 100 or 200 TFLOP/s that FLOP count is about 76, 38 or 19 minutes **(arithmetic)**; achieved throughput on a 24 GB laptop GPU or the Julia L40S is unmeasured.
- **Per-problem scale (arithmetic from the handout):** a fine-tuning run of 16 passes over 30k chunks is half a baseline's steps; P4(a) is 4 initializations × 3 pass counts with LR tuning; P4(b) is 4 × 4 pools at 16 passes; the Hyperball LR search, SAM comparison, dropout at three rates plus ensemble members, and the A5 architecture comparisons at three LRs are each several baseline-sized runs. Most "Recall" parts need no training.
- **Memory and software:** d8 checkpoints are about 122 MB ([manifest entry](https://huggingface.co/datasets/kothasuhas/dl_alchemy_a6_generalization/raw/145bebd16541824ae099735b57a830aeb410c716/manifest.json)); micro-batch 8 × 1,024 tokens is the starter default. The repository pins Python 3.10 (`requires-python >=3.10,<3.13`), `torch==2.6.0` and `wandb==0.31.0` ([`pyproject.toml`](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/pyproject.toml)); the non-Modal path runs on one local CUDA GPU with `torch.compile(mode="reduce-overhead")` and `DataParallel` on several ([GPU guide](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/gpu/README.md)). Our project's own stack choices are not settled here.
- **Inference-only use:** loading a supplied checkpoint and scoring the 1,000 validation chunks is a forward pass over about 1.0M tokens at 30M parameters, about 6×10¹³ FLOPs **(arithmetic)**. Whether the supplied `.pt` files load without the course code was not tested.

### Licence and linking terms

- **No licence is stated for the course material.** GitHub's metadata reports `license: null` for both the [assignments repository](https://api.github.com/repos/deep-learning-alchemy/assignments) and the [website repository](https://api.github.com/repos/deep-learning-alchemy/deep-learning-alchemy.github.io) (checked 2026-11-19). The recursive trees of [`a6_release`](https://api.github.com/repos/deep-learning-alchemy/assignments/git/trees/a6_release?recursive=1) and [`main`](https://api.github.com/repos/deep-learning-alchemy/assignments/git/trees/main?recursive=1) contain no file named like LICENSE, COPYING or NOTICE. The A6 handout carries no licence statement. This is the same state that [the earlier availability research](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/5) recorded for A1–A4; the homepage's honor-code section governs course students and is not a reuse licence.
- **The Hugging Face A6 bundle states no licence for itself.** Its [card](https://huggingface.co/datasets/kothasuhas/dl_alchemy_a6_generalization) has no `license` field. It says the upstream OpenGenome2 DNA data "is released under Apache 2.0 (Brixi et al., 2025)", and the [prepared-data card](https://huggingface.co/datasets/kothasuhas/opengenome2_tokenized_n12361963_ctx1024_char19_bos_eos_iid) is tagged `license:apache-2.0`, as is [`arcinstitute/opengenome2`](https://huggingface.co/datasets/arcinstitute/opengenome2). The terms of the course-made checkpoints, reference runs and figures are not stated.
- **What that supports:** linking to pinned official sources is fine. Copying handout text, figures, code, checkpoints, reference measurements or quiz content into this repository is not established as permitted. This matches the Spec's rule "linked at pinned revisions, never copied". Reference numbers quoted in this note are scoping facts with their source file linked; they are not a licence to vendor the files.
- **Access terms for learners:** the HF bundle needs no account or token; the Modal mount `cs312-shared-data` / `hard-dl-dclm-v1` is for the course workspace ([README](https://github.com/deep-learning-alchemy/assignments/blob/7f0db017cb89e31c7d387b907480f977dd033127/experiments/a6_generalization/README.md)).

## Unit 7: "Exotic" (not released)

What exists as of the check:

- The homepage lists "Exotic: ???" as the seventh assignment topic and states "Current assignment topics are tentative" ([assignments list](https://deep-learning-alchemy.github.io/#schedule)).
- Schedule: Assignment 7 out Thu 19 Nov (due 3 Dec 1:00pm); Lecture: Exotic Tue 1 Dec, "TBA"; Quiz: Exotic Thu 3 Dec, "TBA"; 24 Nov and 26 Nov are Thanksgiving recess ([schedule](https://deep-learning-alchemy.github.io/#schedule)).
- The A6 handout says DNA starts with Assignment 6 and names no modality for A7, so unit 7's modality is **unknown**; nothing in the A6 handout, lectures 10 and 11, or the A6 repository mentions an Assignment 7 (searched).
- No `a7_release` branch (`git ls-remote` showed `a1_release`…`a6_release` and `main` only), no A7 PDF in the website tree at `45ea5f68`, and the latest website commit before the check was 2026-11-17.

Nothing else can be stated about unit 7's title, questions, experiments, models, datasets, held-out predictions, compute or licence until it is published. Its licence situation will most likely repeat the "no stated licence" state above **(inference)**, but that is unchecked.

## Open items

- u06: the unit 6 quiz is unpublished; instructor held-out metrics are not released; wall-clock time per run on our hardware is unmeasured; whether the supplied checkpoints load outside the course code is untested; reuse terms for checkpoints, references and figures are unstated.
- u07: everything above.
- The handout's reference list (Keskar 2017, Jiang 2020, the DART citation) is not in the PDF text, so those are unverified.

## How to re-check unit 7 and the u06 quiz

```bash
git ls-remote https://github.com/deep-learning-alchemy/assignments 'refs/heads/a7*'
gh api 'repos/deep-learning-alchemy/deep-learning-alchemy.github.io/commits?per_page=5' --jq '.[]|[.commit.committer.date,.commit.message]|@tsv'
gh api 'repos/deep-learning-alchemy/deep-learning-alchemy.github.io/git/trees/HEAD?recursive=1' --jq '.tree[].path' | grep -Ei 'a7|exotic|generalization|lecture_1[2-9]'
curl -s https://deep-learning-alchemy.github.io/ | grep -n 'Assignment 7\|Exotic'
```
