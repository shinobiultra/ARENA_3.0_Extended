# PROTOTYPE (throwaway, #235): Chapter 19 — The Platonic Representation Hypothesis: What Converges, Why, and When?

A rough outline for reacting to, not a lesson. Toy numbers come from `prh_toy_probe.py` (`run_output.txt`, CPU, random data). Nothing about real models was run, downloaded or trained. Model sizes and times below are estimates.

Sources (#233 and `docs/mastery/research/inbox_20261103/prh_jepa.md`):
S01 [Huh et al., *The Platonic Representation Hypothesis*](https://arxiv.org/abs/2405.07987) ·
S02 [Balestriero & LeCun, *Proving and Improving the PRH with JEPAs*](https://arxiv.org/abs/2610.39302) ·
S03 [Gröger et al., *An Aristotelian View*](https://arxiv.org/abs/2602.14486) ·
S04 [You et al., *Structure over Geometry*](https://arxiv.org/abs/2609.27252) ·
S05 [Schnaus et al., *Rosetta Stone*](https://arxiv.org/abs/2610.09411) ·
S06 [Ziyin & Chuang, *perfect PRH*](https://arxiv.org/abs/2507.01098)

---

## Where it sits

A new chapter 19, appended after 18. Nothing is renumbered. The section count goes from 87 (with 0.7) to 90.

| Section | Title | Runs on | Time | Spec rows |
| --- | --- | --- | --- | --- |
| 19.1 | Measuring Convergence: Kernels, CKA and Mutual Neighbours | CPU | ~3 h | PRH01–03 |
| 19.2 | Do Real Models Converge? Replicate and Critique the PRH | 24 GB GPU | ~4 h | new real-model tier; PRH02–03 on real features |
| 19.3 | Proving and Improving the Platonic Representation Hypothesis with JEPAs | CPU | ~3.5 h | PRH04–06 |

Every prerequisite has a lower number, so the JEPA section comes after chapter 14 without a cycle. The study route (#30) can still send a learner from chapter 11 and 10.2 straight into 19.1–19.2, and into 19.3 after 14.4.

```
0.7 parts 2,3,5 ──┐
11.1, 11.2 (1-2) ─┼─> 19.1 ──┬─> 19.2  (also needs 12.1 and 10.2)
0.6 (recommended)─┘          └─> 19.3  (also needs 14.1 and 14.4; 0.7 part 4)
```

Intuition at first use (#234 rule): CKA and mutual k-NN are shared tools (12.7 already uses mutual k-NN, 17.4 uses CKA). So 0.7 Part 2 gets a short "CKA is a correlation between two similarity matrices" block, and Part 3 gets "mutual k-NN asks whether two maps keep the same neighbours". 19.1 implements both and breaks them. Kernels, the debiased HSIC estimator and spectral embedding are single-home tools, so they get inline blocks in 19.1 and 19.3. KL appears only in 19.2's reading of S01's contrastive/PMI argument, and links to 0.7 Part 1.

---

## 19.1 Measuring Convergence: Kernels, CKA and Mutual Neighbours

**First screen (draft prose).**

> Here are two matrices of pure Gaussian noise, drawn independently: 1,024 rows (the "examples") by 16,384 columns (the "neurons"). They share nothing. Their linear CKA, the most popular score for "these two networks represent the data the same way", is **0.94**.
>
> Make a prediction first: what is it at 4,096 columns? (It's 0.80.) At 64? (0.06.) For isotropic noise it is almost exactly d/(d+n), so any comparison where wider models score higher is suspect before it starts.
>
> The Platonic Representation Hypothesis says that as models get bigger and better, their representations converge, across architectures and even across vision and language. Before you can test that, you need scores that don't fall for noise. In this section you'll build three, and break each of them once.

**1️⃣ Same examples, different coordinates** (~50 min)
- A representation, for our purposes, is the n×n similarity matrix of its examples (the kernel view S01 uses). Implement `center_gram`, `linear_cka`, `knn_indices` (cosine, self excluded), `mutual_knn` (S01's m_NN), `fit_procrustes`, `retrieval_at_1`.
- Rotation organism (10 clusters, 64-d, n = 1,024): CKA **1.000**, mutual k-NN **1.000**, but direct retrieval@1 without a map **0.000** (chance 0.002). After Procrustes fitted on 512 training pairs, retrieval@1 on the other 512 is **1.000**. Shuffled rows: CKA **0.009**, m_NN **0.011**.
- Tests: rotation preserves the Gram matrix; a feature permutation is not a row shuffle; constant vectors trip the tie/collapse guard.

**2️⃣ Noise that looks converged** (~60 min)
- Width: biased CKA on noise is d/(d+n): **0.016, 0.059, 0.200, 0.501, 0.800, 0.941** for d = 16 … 16,384 at n = 1,024. Implement the debiased HSIC estimator: **0.0006 … −0.0013**. Mutual k-NN stays at chance (k/(n−1) = 0.0098) at every width.
- Layer search (residual-stream-like stacks, n = 512, k = 10, no shared signal): the raw max of m_NN over all layer pairs rises **0.0178 → 0.0225 → 0.0236** for 1, 4 and 12 layers per model. Calibrated against a single layer pair's null, p falls **0.76 → 0.17 → 0.07**. Against the null of the whole search (one row permutation shared by every layer pair), p stays at **0.77, 0.71, 0.94**. With a planted shared signal, raw m_NN ≈ **0.50** and p = **0.005** (the smallest possible with 200 permutations) in every row.
- Implement `permute_rows`, `null_of_max`, `calibrated_p`. Links 0.7 Part 5 and 0.6's forking paths.

**3️⃣ Local or global? Structure or metric?** (~45 min)
- 8 clusters × 128 points, 32-d:

  | Variant | CKA | mutual k-NN | MST edge overlap |
  | --- | --- | --- | --- |
  | (a) same neighbours, new layout | 0.778 | 0.756 | 0.993 |
  | (b) same layout, new neighbours | **0.972** | **0.070** | 0.023 |
  | rotated copy | 1.000 | 1.000 | 1.000 |

- Variant (b) is the point: a global score of 0.97 while only 7% of neighbours agree. Implement `mst_edges` and `edge_overlap`, then write two sentences: S03's reading (calibrated support is local) and S04's (structure, not metric, is what agrees).

**Signature result:** panel 1 of `prh_toy_probe.png` (two unrelated noise matrices, CKA against width) beside the 2×2 table drawn as edge pictures.

**🎯 Held-out Quiz** (~20 min), then **☆ Bonus:** CKNNA (S01's appendix metric between m_NN and CKA); unpaired Rosetta alignment (S05) on the rotation organism, including the symmetric world where it can't work.

---

## 19.2 Do Real Models Converge? Replicate and Critique the PRH

It follows 10.2's loop, Read → Shrink → Predict → Reproduce → Attack → Write, and its verdict rules.

**Setup** (~15 min). Models and data facts live here, not in the introduction.
- Captions: the Pythia ladder 70M, 160M, 410M, 1B, 1.4B (widths 512 → 2048; same data, same order). Images: DINOv2-B. Positive control: CLIP ViT-B/32's own text and image towers, which were trained to agree. Negative controls: shuffled pairs, and an untrained Pythia-1.4B with the same architecture.
- Data: the 1,024 WIT image–caption pairs from S01's code, loaded at runtime from the authors' Hugging Face dataset. Link it, don't vendor it. Check the licence when the batch opens.
- Estimates, not measured: about 7 GB of downloads, under 5 GB of VRAM, and a few minutes of feature extraction.

**1️⃣ Read, shrink, predict** (~35 min)
- The claim card: who claimed what, in their words. S01 §2.3/Fig. 3 says language models that are better at language align more with vision models. S01 §6 adds caption density and its own limitations. S03 replicated this and found the calibrated support is mostly local. S04 replies that it is structure, not metric. S02's theorem has conditions. Each one is linked, with its section.
- Shrink: one language family × one vision model × 1,024 pairs, on one GPU.
- Predict: the sign and rough slope of alignment against model size, and the size of the calibrated effect. Fix the verdict rule before anything runs.

**2️⃣ Reproduce the headline** (~60 min)
- Extract per-layer features: mean-pooled caption tokens for each language model, the CLS token at each DINOv2 layer. Take m_NN (k = 10), maximised over layer pairs as S01 does. Plot it against the language models' bits per byte on a held-out text sample, which the learner measures.
- The positive control (CLIP towers) must clear the null by a wide margin before anything else counts.

**3️⃣ Attack it** (~70 min)
1. Run the whole-pipeline permutation null from 19.1 Part 2 on real features.
2. Compare biased and debiased CKA across the ladder. Width grows from 512 to 2048, so the isotropic-noise CKA null alone would rise from 0.33 to 0.67 at n = 1,024. Measure the real null instead of assuming it.
3. Run the untrained Pythia: does any "alignment" appear without learning?
4. Caption density: truncate the captions to 5, 10 and 20 words, and use all of them.
5. Bootstrap the confidence intervals over items.
6. Compare the last layer alone with the max over layer pairs.

**4️⃣ Verdict and critique note** (~35 min). Use 10.1's `claim_verdict` with #246's four verdicts for each sub-claim. The note comes from `write_note()`. "Not replicated at 1.4B on 1,024 pairs" is a finished result once the positive control passed.

**Claims the learner tests (honest nulls included):**

| ID | Claim (source) | Test on 24 GB | What a null looks like, and why it's still a result |
| --- | --- | --- | --- |
| C1 | Better language models align more with vision models (S01 §2.3, Fig. 3) | Pythia 70M–1.4B × DINOv2-B, calibrated m_NN against bits per byte | The slope's interval includes 0. S01's curve spans much larger models, so the note records the size range tested. |
| C3 | Denser captions align better (S01 §6) | Truncation sweep on the same pairs | No monotone change. Truncation isn't S01's protocol, and the note says so. |
| C4 | After calibration, local agreement survives better than global (S03) | Calibrated m_NN and debiased CKA, side by side across the ladder | If CKA survives too, S03's conclusion doesn't carry to this pair, which is a finding in itself. |
| C5 | Structure agrees even at global scale (S04) | Try it: MST edge overlap on the real features | Overlap at chance after calibration |
| C2 | Strong vision models align with each other (S01 Fig. 2) | Bonus: a timm/DINOv2 ladder | — |
| C8 | 42× fewer samples, and a 109M text JEPA matching LMs up to 72B on alignment (S02 §4.2–4.3) | Author-reported only. Bonus: the authors' Flickr arm on the Julia worker (author estimate 1–2 GPU-h), with the user's permission | — |

**Signature result:** a 2×2 grid with the ladder on the x-axis. The top row is raw, the bottom row calibrated; the left column is m_NN, the right column CKA. Each panel has null bands, the untrained-Pythia point and the CLIP positive control, with the learner's verdict printed under each column.

**🎯 Held-out Quiz** (~25 min), then **☆ Bonus:** the full ladder to 2.8B/6.9B (6.9B in fp16 is about 14 GB, which fits); m_NN across Pythia's training checkpoints for one size; a Qwen3 ladder, where the small models are distilled from larger ones, which confounds "convergence with scale"; the vision–vision C2; Rosetta (S05) on real features; real-model behaviour transfer, which 11.2 Part 3 already does on GPT-2.

---

## 19.3 Proving and Improving the Platonic Representation Hypothesis with JEPAs

Subtitle: *Which positive pairs make two observers agree?* S02's theorem is taught as a conditional result. Its empirical claims are listed as the authors' reports.

**First screen (draft).** A world of 24 states in four groups. Two observers see each state through different renderings, but they are trained with the same rule for which pairs count as "the same scene". Their 1-D representations agree up to a sign, with residual **0.000**. Now nudge which pairs count as positive by ε = 0.001. When one distinction is clearly rewarded most (eigengap 0.107), the residual is **0.047**. When two distinctions are rewarded equally (gap 0), it is **0.77**, even at ε = 0.0001. Make a prediction first: what happens with a gap of 0.011? (**0.213**.)

**1️⃣ One world, two observers** (~70 min). Implement `validate_positive_kernel`, `normalized_pair_operator`, `spectral_embedding` (dropping the constant mode), `weighted_procrustes` and `population_prediction_loss`, and check them against a brute-force small-world oracle. Write out Theorem 1's assumptions in a table: finite world, injective renderings, common symmetric positives and marginals, population minimisers, an identity predictor, and rotation-unique solutions.

**2️⃣ Break each assumption** (~60 min)

| Break | Residual (toy) |
| --- | --- |
| Gap 0.107, ε = 1e-4 / 1e-3 / 3e-3 | 0.004 / 0.047 / 0.237 |
| Gap 0.011, same ε | 0.004 / 0.213 / 1.320 |
| Tied (gap 0), same ε | 0.766 / 0.769 / 0.737 |
| Lossy rendering: merge two states in the same group | 0.000 |
| Lossy rendering: merge two states across the kept split | 0.557 |
| No anti-collapse | a constant embedding has zero prediction loss |

Plot S02's Eq. 3 bound only after transcribing it with its actual norms and weights. Our toy's ε is not the paper's quantity.

**3️⃣ Good views, bad views, and does it transfer?** (~60 min). Add state plus nuisance variables, with independent or shared nuisance (PRH05). Then do a finite-world edit transport using an exact state oracle, with shuffled-anchor and random-edit controls (PRH06).

**Signature result:** the mismatch-against-perturbation sweep for the three gaps (panel 3 of `prh_toy_probe.png`), with the two lossy observers marked.

**🎯 Held-out Quiz**, then **☆ Bonus:** the authors' Flickr arm (on the Julia worker, with permission); S06's deep-linear proof compared assumption by assumption; LeJEPA's SIGReg as an anti-collapse alternative. The 14.4 references follow #236's corrections.

---

## Held-out quizzes (CS 312 shape, per #245)

Answers marked "key run" weren't run in this prototype. Each quiz's last cell regenerates its answers.

**19.1**
- Q1 (Easy). *From the section:* biased CKA of two noise matrices, n = 1,024, d = 4,096 → 0.80. *Diff:* n = 4,096. *Answer bins:* [0, .2) [.2, .4) [.4, .6) [.6, .8) [.8, 1]. *Answer:* 0.50 → [.4, .6).
- Q2 (Medium). *Diff:* the rotated copy, with 50% of its rows shuffled. *Predict:* mutual k-NN, as an interval. (key run)
- Q3 (Hard). *Diff:* 36 layers per model, no signal, the single-pair null. *Predict:* the false-positive rate at α = 0.05 over 20 datasets, as an interval. (key run)

**19.2**
- Q1 (Easy). *Diff:* k = 10 → 50. *Predict:* raw m_NN goes up, down or stays, plus the new chance level (0.049).
- Q2 (Easy). *Diff:* shuffle the captions against the images. *Predict:* an interval around chance. (key run)
- Q3 (Medium). *Diff:* captions cut to their first 5 words. *Predict:* the change in calibrated m_NN, as a bin. (key run)
- Q4 (Medium). *Diff:* last layer only, instead of the max over layer pairs. *Rank:* raw and calibrated scores, both ways. (key run)
- Q5 (Hard). *Diff:* the untrained Pythia-1.4B. *Predict:* calibrated m_NN, as an interval, with a one-line reason. (key run)

**19.3**
- Q1 (Medium). *Diff:* gap 0.011, ε = 0.003. *Answer bins:* [0, .25) [.25, .5) [.5, 1) [1, ∞). *Answer:* 1.32.
- Q2 (Medium). *Diff:* merge state 0 with a state in group B, on the same side of the kept split. *Predict:* the residual, as a bin. (key run)
- Q3 (Hard). *Diff:* a 2-D embedding of the tied world. *Predict:* is the subspace unique? One word, plus a reason. (key run)

---

## Compute

- 19.1 and 19.3: CPU. The probe takes about 2 minutes on one core, mostly the 200-permutation search; in torch on the GPU it is seconds.
- 19.2 core: fits the laptop (estimates above). Quiz key runs re-extract nothing; they reuse the cached features, so every fresh-kernel run regenerates them in minutes.
- Julia worker: only the optional 19.3 Bonus arm, and only with the user's permission when its batch runs.
