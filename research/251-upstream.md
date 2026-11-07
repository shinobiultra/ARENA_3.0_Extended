# What should the extension take from upstream ARENA? (Issue #251)

Status: research findings, 2026-11-07. Facts come from git (`f9f034bd..6a46100c`, already merged into `integration/extension`), from upstream's open pull requests on GitHub, from the live learn.arena.education site, and from the extension's pages on `integration/extension` (`09b2557a`). No lesson code was executed. Treatments below are proposals for the plan tickets and sync batches to accept or change, not decisions. Throwaway branch `research/251-upstream`, no PR.

Treatment words, used in every table:

- **TAKE**: arrives through the sync batch. The extension does nothing beyond the sync's own checks.
- **LINK**: a route-block or Prerequisites line names it, or an existing line is fixed.
- **ADAPT**: a named lesson plan changes because of it.
- **IGNORE**: nothing to do.

## Recommendations at a glance

1. **Take chapters 0–4 as they are** (§1). Upstream restructured four things: 0.5 became 0.5.1/0.5.2, 2.2 became 2.2.1/2.2.2, 2.4 became an "RL + LLMs" group with a rewritten 2.4.1, and 2.3.2 is new. Everything else in chapters 0, 1, 3 and 4 is audit, errata and test work. The one extension line that named a moved section (13's Prerequisites) is already fixed.
2. **Link rather than re-teach, and cut where an exercise is duplicated** (§2): 6.1 (Gated and JumpReLU SAEs, already in 1.5.4), 7.2 §1 (1.3.3's dashboard exercises), 7.4 (two GRPO exercises, now in 2.4.1), 9.2 (probe estimators, already in 1.3.1), 15.2 (four LoRA wrapper exercises, now in 2.4.1). 7.3, 9.1, 9.3, 9.4 and chapter 10 stay as they are, with links.
3. **Do not build on 2.3.2, 2.4.2 or 2.4.3 yet.** All three are "Work in progress" upstream and 2.4.3 is a stub (§1.2, §4.5).
4. **Link to learn.arena.education pages, not to `.ipynb` paths** (§4.4): upstream's open PRs #731 and #740 stop committing those notebooks, and five extension pages link to them today.
5. **Taste bar:** one amendment for the user (item 8's "at least one hint" is stricter than the original's 19%, and 41 of 86 lessons miss it), additions for the authoring file, and a list of per-lesson page defects (§3).
6. **Infrastructure:** take nothing from upstream's workflows. The first release push will wake `build-master`, which should fail red on the release commit; disable the three in Actions settings (§4.2).
7. **The next sync is already visible** (§5): four open content audits, generated notebooks leaving `main`, upstream's own `uv.lock`, 21 errata PRs, a new section 4.6.

## 1. What changed upstream, and what the extension does with each group

### 1.1 How to read the numbers

The range is 22 June to 6 November 2026: 1,144 commits, 437 files (96,247 added, 76,694 removed lines). Upstream `main` is still `6a46100c` today (`gh api …/compare/6a46100c...main`: ahead by 0).

Who wrote the commits (857 non-merge commits; the other 287 are merges):

| Author | Commits | What |
|---|---|---|
| ARENA-workflow[bot] | 154 | "Auto-generate outputs from master files [skip ci]": the build-master workflow's output |
| arena-errata-bot[bot] | 142 | errata fixes reported in the #errata Slack channel and approved in an errata audit tool, one PR each (§4.6) |
| "David" (424) and "David Quarel" (67), which appear to be the same maintainer under two git identities | 491 | content, audits, tests, infrastructure |
| other contributors | 70 | typos, hints, library fixes (styme3279 32, Elliotte Rusty Harold 7, "Arena Autocommit" 6, a dozen more) |

Where the files are: chapter 2 has 113, infrastructure 111, chapter 1 80, chapter 0 53, chapter 3 33, chapter 4 27, `.github` 7.

The unit of change is the **master file** `infrastructure/chapters/chapter*/**/master_x_y.py`. Everything under `chapter*/instructions/`, `chapter*/exercises/*/solutions.py` and the notebooks is generated from it. Commit titles use `[x.y]` prefixes, but 323 of the 703 commits other than the regeneration bot's are untagged, so the per-section counts below come from the commits that touch each master file, not from titles alone.

"Commits" counts non-bot commits that touch the section's master file; "errata" counts errata-bot commits among them; "master +/−" is the master file's diff. A new file shows only `+`.

### 1.2 Inventory by chapter and section

| Group | What changed (fact, with source) | Size | Proposal | Routed to |
|---|---|---|---|---|
| **[0.0]–[0.4]** | Chapter 0 content audit #755 (2026-11-06) plus errata passes. 0.3 `ring_all_reduce` rewritten as a real ring all-reduce and an NCCL deadlock fixed with `dist.batch_isend_irecv` (commit body of `71173ab8`). 0.2 BatchNorm running statistics no longer track gradients (`93a425ec`) and use the unbiased variance (`8a06d278`). 0.4's note on subtracting the max before `exp` no longer calls it a bonus (`b4bc4f48`, `git diff` of `master_0_4.py`). Tests strengthened everywhere; hosted figures replaced by `-v2` files. | 0.0 +55/−146, 8 commits; 0.1 +290/−242, 16; 0.2 +194/−156, 24 (7 errata); 0.3 +259/−210, 18; 0.4 +264/−201, 25 | **TAKE**. One **LINK** candidate: the 0.7 Foundations Toolkit's stable log-softmax row (FND02) can cite 0.4's now-required stable `cross_entropy` instead of re-teaching it. | Spec "Integration, upstream sync and releases" (sync checks); Spec "Intuition at first use and the 0.7 Foundations Toolkit"; plan #264 (0.7) |
| **[0.5.1] VAEs, [0.5.2] GANs** (was [0.5]) | Split on 2026-09-29 (`76c9765c`, `426b13bc`, `1aed63af`). New group page `0_5_vaes_gans.md`; ids `51_vaes`, `52_gans`. 0.5.1: autoencoders and VAEs on MNIST, bonus transposed convolutions. 0.5.2: DCGAN on celebrity faces, FID demo (`9c7f98e5`). GPU-side data loading (`fd8c7b43`) and a new shared `fundamentals_utils.py`. Open upstream PR #766 will replace the hand-built DCGAN modules with PyTorch's and drop the Tanh/LeakyReLU exercise. | 0.5.1 renamed from 0.5 (42 commits of history); 0.5.2 +2,184 new | **LINK (already done)**: 13's Home line "Prerequisites: [0.5] VAEs & GANs" now reads "[0.5.1] VAEs and [0.5.2] GANs" on `integration/extension`; it came in with the catch-up merge `06a600b1`, and a grep of the extension finds no remaining `[0.5]` or "VAEs & GANs" reference. 13.2, 13.6 and 18.2 mention VAEs only in passing. The route block (#30) uses the new numbers. | Spec "Integration…" (P16: route and Prerequisites lines); plan #255, #14 for 13.x's own Prerequisites lines |
| **[1.1] [1.2]** | Errata, dead links, tokenise the first 200k TinyStories (`7637bceb`), BPE intro wording. | 1.1 +171/−119, 23 commits (6 errata); 1.2 +115/−77, 17 (3 errata) | **TAKE** | sync checks |
| **[1.3.1] Linear Probes** | The stale "Probing for Deception" page was removed (`4a667504`, errata-668); the section stays 1.3.1 and keeps its exercises. `test_extract_activations` and the steering-hook test were rewritten to report one named mistake at a time. Page word count halves (40.8k → 20.8k) because of the removed page. | +47/−32, 19 commits | **TAKE**. Overlap with 0.6 and chapter 11: see §2. | plans #252 (0.6), #260 (11.3–11.5) |
| **[1.3.2] Function Vectors & Model Steering** | nnsight fixes (`tracer.iter`, GPT-J block output), ActAdd uses the block input `resid_pre` (`90a522ce`, errata-697), steering tests that fail when steering is a no-op (`91705f2b`). Open #702 moves it to nnsight 0.8. | +171/−102, 34 commits | **TAKE**. 9.1's `lesson.md` already names this section as a prerequisite (relative link): keep it, see §2. | plan #9 (9.1) |
| **[1.3.3] SAEs, [1.4.2] SAE circuits** | sae-lens 6 support through the ARENA fork of sae-vis (`ad33aa2b`), Gemma-2-2B in bf16, `map_through_mlp` no longer applies `ln2_post` on top of `W_skip` (`b92b1f5f`). | 1.3.3 +146/−110, 23; 1.4.2 +50/−52, 12 | **TAKE**. Overlap with chapter 6, 8.6, 8.7: see §2. | plans #252, #253, #11, #257, #17 |
| **[1.3.4] Activation Oracles** | Hook hints added to `collect_activations_multiple_layers` (errata v8-31), exact-id test for `create_oracle_input`. | +66/−15, 9 commits | **TAKE**. Overlap with 7.3: see §2. | plan #258 (7.3) |
| **[1.4.1] IOI, [1.5.1]–[1.5.4]** | Errata and test fixes: noising/denoising mix-up in the activation-patching intro (errata-43/633/711), `embed_SVD` and `fourier_embed` fixed in 1.5.2, OthelloGPT probe training sped up. | 1.4.1 +122/−101, 18; 1.5.1 +72/−65, 16; 1.5.2 +68/−66, 11; 1.5.3 +112/−94, 20; 1.5.4 +75/−41, 14 | **TAKE**. The six original `met` requirement rows still match: 1.2, 1.3.1, 1.4.1, 1.5.3 keep their numbers. | requirement registry (#8, #37 P19) |
| **[2.1] Intro to RL** | Errata pass (10 errata-bot commits), agent gets its own RNG stream (#760). | +374/−142, 26 commits | **TAKE** | sync checks |
| **[2.2.1] DQN, [2.2.2] VPG** (was [2.2]) | Split; group page `2_2_dqn_vpg.md`. The one 10.8k-word page becomes 13.5k (DQN) plus 7.5k (VPG) words. Preallocated circular `ReplayBuffer` (#765). | old 2.2 deleted (3,082 lines); 2.2.1 +2,068, 2.2.2 +1,475 new | **TAKE**. No extension page names [2.2]. | route block only |
| **[2.3] PPO** | Content audit #757, std-normalisation note, swing-up text honest about not balancing. | +396/−308, 48 commits | **TAKE** | sync checks |
| **[2.3.2] Goal Misgeneralisation** (new) | Gridworld "pottery shop": reward hacking, potential shaping, then goal misgeneralisation under distribution shift. `status: "Work in progress"` in `config.yaml`; 13.3k words, 11 exercises, 4 hint dropdowns. | +2,609 new, 4 commits | **LINK** as an optional original section; no extension lesson depends on it yet. Pattern notes in §3. | route block (#30); plan #19 (9.x) as a "see also" |
| **[2.4] RL + LLMs** (group), **[2.4.1] RLHF** | New group (#743, the pin's last commit). The old PPO-based 2.4 was archived under `infrastructure/archive/2.4.1-ppo/`; 2.4.1 is now GRPO plus LoRA, with a shared vLLM sampler (`samplers.py`). The LoRA part was rewritten on 2026-10-21 (`88ef2b6b`). | 2.4.1 +4,456 new, 58 commits | **LINK** + **ADAPT**: see §2 (15.1, 7.4). | plans #255 (15.1), #258 (7.4) |
| **[2.4.2] RLVR, [2.4.3] RLAIF** (new) | Both `status: "Work in progress"` in `config.yaml`. 2.4.2: Dr. GRPO on multi-digit multiplication with a verifiable reward, 13.7k words. **2.4.3 is a 104-word stub at the pin**; the live site shows "Coming soon". Open PRs #706 and #737 fill it. | 2.4.2 +2,200, 42 commits; 2.4.3 +45, 1 commit | **IGNORE until a later sync** for lesson plans. **LINK** nothing from extension pages yet. Re-read at the next sync. | next sync's list; plan #19 |
| **[2.5] MCTS & AlphaZero** | Rewrite and a Gumbel bonus (#734), tests audit (#730). 28.9k words (was 23.4k). | +2,014/−1,159, 38 commits | **TAKE** | sync checks |
| **[3.1]–[3.5]** | Tests, errata, delisted OpenRouter models replaced in 3.1 (`d349b4e9`). Small diffs. Open content audit #758 (12 files) and 13 open errata-bot PRs. | each ±10–40 lines; 3.1 has 7 errata-bot commits | **TAKE**. The extension hard-codes no API model IDs in its pages or exercises (grep of `integration/extension` found none), so ID churn does not reach it. | sync checks |
| **[4.1]–[4.5]** | Tests, hooks for transformers 5 (`0102989c`), petri pinned to v2.0.0 (`d3100202`), repo-root lookup by chapter directory. Structure unchanged. Open content audit #759 (14 files, +2,075/−1,210). | 4.1 +38/−16; 4.3 +31/−29; 4.4 +55/−57; 4.5 +64/−33 | **TAKE**. Overlaps: see §2. | plans #19, #9, #254 |
| **[infra]** | See §4. | 111 files | see §4 | Spec "Runtime and performance", "Integration…" |

What did *not* change: chapters 1, 3 and 4 have no structural change and no new exercises. Hint counts are identical at both pins: 80 hint dropdowns and 102 exercises with a hint of either style (dropdown or inline "Hint -"), out of 543 exercises at the old pin and 533 at the new (counted as "Difficulty:" blocks in the instruction pages of chapters 0–4).

### 1.3 What the sync must still do by hand (from this inventory)

1. Nothing for 13's Prerequisites line: it already says 0.5.1/0.5.2 (merge `06a600b1`). It was the only extension line naming a moved section, as the #37 facts said. One other spot says "[2.4]" for RLHF (7.4's master, line 2262, "RLHF ([2.4])"); it is not broken, since 2.4 is now the group, but it should say 2.4.1.
2. Replace relative links into upstream's generated notebooks (§4.4).
3. Nothing for the 13 extension lessons whose pages name `HookedTransformer` (7.1–7.5, 8.1–8.5, 8.8, 12.3, 16.6): the env-upgrade batch already ran them on TransformerLens 3.9 (§4.3).

## 2. Overlaps: link, build on, or cut

Method: for each pair, the instruction pages at `6a46100c` and on `integration/extension` were read side by side (introductions, objectives, exercise lists, prerequisite lines and the cited passages), and the upstream diff for the section was checked. Exercise bodies were not read end to end. A "concrete" overlap means the same exercise, pipeline, model, dataset or estimator; "partial" means the same idea at a different scale or with a different purpose.

**Upstream did not add, remove or restructure an exercise anywhere in chapters 1, 3 or 4** between the pins: exercise counts per page are identical for all 24 pages. So nothing in this table was invalidated by the sync; the pairs below describe the situation the lesson plans start from.

### 2.1 The pairs named in the ticket

| Pair | Overlap | Recommendation | Plan ticket |
|---|---|---|---|
| 1.3.4 Activation Oracles ↔ **7.3** Mini Activation Oracles | Partial. Same injection idea (a `?` placeholder, norm-matched addition at an early layer, a "Layer N:" prompt), different everything else: 1.3.4 queries a pretrained Qwen3-8B oracle and reads about training; 7.3 trains a GPT-2 oracle on a synthetic four-attribute world, with five controls and a probe bank. 7.3 already names 1.3.4 seven times, in the introduction, the prerequisites, the Bonus and the reading list, accurately. | **Leave alone.** It is the model case of "builds on, does not re-teach". 7.4 does not name 1.3.4; an optional see-also. | #258 |
| 1.3.3 SAEs and 1.4.2 SAE circuits ↔ **chapter 6**, 8.6, 8.7, 7.2 | Mixed, see §2.2 | See §2.2 | #252, #253, #11, #257, #17, #258 |
| 1.3.1 Linear Probes ↔ **0.6** and chapter 11 | 0.6: complementary. 1.3.1 has no shuffled-label or random-direction controls (a grep finds only a regularisation `C` and a CSV shuffle); 0.6 supplies exactly those. 11.5 implements the CCS that 1.3.1 declines to (its CCS part is prose plus a bonus note). 11.3 and 11.4 have no upstream counterpart. | **0.6: leave alone**, optional see-also to 1.3.1. **11.5: link** to 1.3.1's CCS discussion. **11.3, 11.4: leave alone.** Also see 9.2 in §2.2. | #252 (0.6), #260 (11.3–11.5) |
| 4.1 Emergent Misalignment ↔ **9.4** and chapter 15 | Partial. 4.1 §4 finds the transition from real LoRA checkpoints on Qwen2.5-14B (about 64 GB of VRAM); 9.4 and 15.4 use authored toy trajectories with controls and say so (15.4 states it cannot show a real transition). 15.1–15.3 and 15.5 have no EM content. Neither Prerequisites line names 4.1. | **Link.** 9.4 and 15.4 each get a line: "4.1 does this on real checkpoints; this is the toy with controls". No cuts. Separately, 9.4 and 15.4 duplicate *each other* (first-crossing onset detection); #19 and #261 should decide which one owns it. | #19 (9.4), #261 (15.4) |
| 4.3 Interpreting Reasoning Models ↔ **9.3** CoT Faithfulness | Superficial. Both ask whether visible reasoning causes the answer, with different instruments: 4.3 resamples sentences of real R1-distill rollouts; 9.3 probes and patches hidden state in an exact toy. | **Leave alone**; an optional see-also to 4.3 §2 and §4. | #19 |
| 4.4 LLM Psychology & Persona Vectors ↔ **9.1** Refusal Directions | Superficial. Shared primitives only (difference of means, residual add by hook, PCA). 4.4 is persona-level, 27B-scale and API-judged; 9.1 is one refusal behaviour at Qwen2.5-0.5B with rate-based scoring and controls. 9.1 already names 1.2, 1.3.1 and 1.3.2 correctly and has a 1.3.2 prerequisite link in its `lesson.md`. | **Leave alone.** One optional line in 9.1 §5 pointing at 4.4 §3–4. | #9 |
| 4.5 Investigator Agents ↔ **chapter 10** Research Practice | None. 4.5 is API-driven red-teaming of an LLM patient (about $25 of API cost); 10.1 is CPU-only toy research practice with pre-registration, logs and intervals; 7.6's agent is a finite hypothesis controller with no LLM. 10.1 correctly names 1.4.1, 8.1 and 0.6. | **Leave alone** (all three). | #254, #259 |
| 2.4.1 RLHF (LoRA part rewritten) ↔ **15.1** LoRA and DoRA | Partial for 15.1, **concrete for 15.2**. 2.4.1 §2 builds `LoRALinear`, `merge`/`unmerge`, `apply_lora` and a LoRA GRPO trainer on Qwen3-1.7B-Base. 15.1 computes the scaled update and checks merge/unmerge parity on a planted-direction classifier, then adds DoRA, spectra and controls (none upstream). 15.2's first four exercises (exact update, transparent LoRA wrapper, inject by module name and freeze, merge/unmerge parity) match 2.4.1's `LoRALinear`, `apply_lora` and `merge`/`unmerge`; 15.3 and 15.5 restate the wrapper again. **The conventions differ**: 15.1 and 15.2 write `ΔW = (α/r)·B@A` with `A:(rank,in)`, `B:(out,rank)`; 2.4.1 writes `ΔW = AB` with row-vectors, `A:(in,r)`, `B` zero-initialised, float32 adapters. | **15.1: link** to 2.4.1 §2 and add a one-line convention warning; the opener can say what 2.4.1 already built. **15.2: build on, and shrink** the four wrapper exercises to a pointer; keep matched training, spectra, ablation and the random-adapter control. **15.3, 15.5: link.** One condition: 2.4.1 wants a GPU (about 24 GB; below 22 GiB it drops to GPT-2 124M and Qwen3-0.6B), while 15.1 and 15.2 run on CPU. If #23's no-GPU tier is to *build* the wrapper rather than read about it, 15.2 keeps its toy exercises and only links. | #255 (15.1), #22 (15.2, 15.3), #261 (15.5) |
| 2.4.1 §1 (GRPO) ↔ **7.4** Mini Natural Language Autoencoders | **Concrete in two exercises.** 7.4's "group-normalised advantages" and "KL penalty and policy-gradient loss" implement what 2.4.1's `compute_grpo_advantages` and `calc_kl_penalty` implement; 7.4's own text says "you've seen both in [2.3] PPO and [2.4.1] RLHF, but we re-derive everything we need". New in 7.4: the NLA training loop, the controls and the Gumbel-softmax alternative. One stale "[2.4]" (line 2262). | **Build on, shrink.** Name 2.4.1 §1 as a prerequisite for part 3; give the advantage and loss functions as "use yours or ours"; keep the NLA loop and the controls. Fix "[2.4]" to 2.4.1. | #258 (7.4) |

### 2.2 Other overlaps found, and the real duplications

| Pair | Overlap | Recommendation | Plan ticket |
|---|---|---|---|
| 1.5.4 §5 ↔ **6.1** SAE Variants | **Concrete.** Gated SAEs (with the auxiliary loss, "up to 60 minutes") and JumpReLU SAEs (custom threshold gradient, "up to 40 minutes") are existing exercises in 1.5.4; they are two of 6.1's four sparsity rules. 6.1 names 1.5.4 as "the prerequisite arc" but not that these two are already implemented there. New in 6.1: TopK, exact-dictionary recovery, train-frozen feature selection, orthogonal steering controls. | **Build on, and cut the duplicated derivation.** The plan keeps the four-rule comparison on a planted dictionary (that is the signature result) and gives the Gated auxiliary loss and the JumpReLU pseudo-gradient as "use yours from 1.5.4 or ours" with a link; the minutes go to TopK, recovery and controls. | #252 (6.1) |
| 1.3.3 ↔ **7.2** Feature Verbalizers | **Concrete.** Same SAE and hook (`gpt2-small-res-jb`, layer 7), same highlighted-token format. 1.3.3 already has "implement autointerp scoring" and an explainer. 7.2 §1 (`sae_latent_acts`, `top_activating_examples`, `format_example`) repeats 1.3.3's dashboard exercises; §3–5 (simulation score, fuzzing, permutation null, steering score) are new. | **Build on.** 7.2 §1 shrinks to a short recap with a pointer to 1.3.3; the new scoring sections carry the lesson. | #258 (7.2) |
| 1.3.3 patch scoping ↔ **7.1** Patchscopes | Partial. 1.3.3 has "implement patch scoping" (Patchscopes applied to SAE latents, Gemma-2B-it). No extension page mentions it. | **Link** from 7.1's Patchscopes part ("you met this applied to SAE latents in 1.3.3"). | #9 (7.1) |
| 1.3.3 ↔ **6.2** Gemma Scope and Feature Steering | **Concrete idea, different model.** Gemma Scope loading, JumpReLU, feature steering and feature DLA appear in both; 6.2 runs Gemma 3 1B on CPU, 1.3.3 uses Gemma-2-2B. 6.2's exact-organism ground truth and controls are new. 6.2 has no body link to 1.3.3, and its only pointer is a dead Streamlit URL. | **Link** to 1.3.3 and fix the URL (§4.4). | #253 (6.2) |
| 1.3.3 §2 ↔ **6.3** Splitting, Absorption | Partial, complementary. 1.3.3 teaches feature splitting with UMAP on real widths and lists absorption as "not implemented"; 6.3 supplies a toy absorption exercise. | **Link and build on**: say 1.3.3 left absorption open. | #11 (6.3) |
| 1.4.2 ↔ **6.4** and **8.7** | Partial, same goal at different scale. 8.7's own scope paragraph lists as missing "released-transcoder integration, the circuit-tracer workflow, frozen-attention replacement-model validation, explicit error nodes, and supernode construction". 1.4.2 teaches every one of them on Gemma 3-1B IT. 6.4's and 8.7's planted ground truth, edge conservation and shuffled-edge controls have no upstream counterpart. | **Link and build on.** Turn the "missing" list into "do the real one first in 1.4.2 §2–4". Keep the toy. | #257 (6.4), #17 (8.7) |
| 1.3.1 ↔ **9.2** Truthfulness Directions | **Concrete.** Same source paper (Marks and Tegmark), same estimators (mass-mean direction, covariance-adjusted readout, logistic probe, cross-dataset transfer matrix, equal-norm addition). 9.2's Prerequisites line reads "original ARENA 1.2, 0.3, and 9.1" and **omits 1.3.1**. New in 9.2: the tag-versus-truth organism with a label-permutation control, the recognition-versus-correctness gate and the ITI head shift. | **Build on, and cut.** Add 1.3.1 to Prerequisites; shrink `mass_mean`, `class_covariance`, `iid_direction` and the logistic fit; keep the organism, `permute_labels`, `selective_rates` and `head_shift`. | #12 (9.2) |
| 1.4.1 ↔ **8.1** Activation Patching Refresher | Concrete but intended: 8.1 repeats 1.4.1 §3's patching sweeps on the same model and prompt, adds four corruptions, noising versus denoising, frozen-downstream effects and bootstrap intervals, and says it unifies 1.4.1's separate hooks. 25-odd correct links to 1.4.1 across 8.x. | **Leave alone**; the plan may skim 8.1 §1 for material that is only a repeat. | #253 |
| 1.2 ↔ **17.2**, 1.5.2 ↔ **17.1** | Partial, complementary (construct versus reverse-engineer). Both link correctly. | **Leave alone.** | #16, #22 |
| 0.5.1/0.5.2 ↔ **13.6**, **13.x** | Weak. No chapter 13 lesson teaches a VAE, GAN or FID. 13.6's VQ loss and straight-through estimator build on the autoencoder idea 0.5.1 teaches, but its Prerequisites line names only causal attention. | **Link** 0.5.1 from 13.6 as an optional prerequisite. Do not copy FID. | #14 (13.6) |
| 2.1/2.2.1 ↔ **14.7**, 2.5 ↔ **14.3** | Weak. 14.7 lists "temporal-difference targets" as a prerequisite with no link; 14.3 contrasts an exact simulator with a fitted predictor, as 2.5's exact Connect-4 simulator does. | **Link** 2.1 and 2.2.1 from 14.7, 2.5 from 14.3. | #15 (14.3, 14.7) |
| 0.3 ↔ **17.1** | Weak. 17.1 trains with `AdamW(weight_decay=1e-3)` and never says why; 0.3 explains AdamW against L2. | **Link.** | #22 (17.1) |
| 2.3.2, 2.4.2 ↔ 15.4, 9.4, 9.5 | None. No extension lesson trains with RL or rewards, apart from 7.4 above. 2.4.2's intro readings and 2.3.2's opening are fair background. | **Link as background** only once their `status` is no longer "Work in progress"; 2.4.3 is unusable. | #19, #261 |
| 4.2, chapter 3, 1.6 ↔ extension | None. 4.2 and chapter 3 are API-based and behavioural. | **Ignore.** | |

### 2.3 Intra-extension duplication, found on the way

A mean-difference direction is implemented as an exercise in 9.1 §1, 9.2, 9.3, 9.4, 9.5 and 11.5, besides 1.3.1's `MMProbe` (exercise names read from the pages; not every body checked); AUROC in 9.1 and 9.5; `first_crossing` in 15.4 and 17.2; PCA in 1.3.1, 4.4, 9.1 and 11.1. 10.1 itself notes that the 9.1 and 9.2 constructions are "the same". This is the Spec's "intuition at first use" rule applied across lessons: **9.1 builds the difference of means once; 9.2–9.5 and 11.5 link back ("you built this in 9.1") and may hand it over as a given function** instead of asking for it again. Route: #9 (9.1 as the home), #12, #19, #260.

### 2.4 Stale or missing links the overlap read turned up

- **9.2** omits 1.3.1 as a prerequisite (above).
- **6.4, 6.5, 8.6, 8.7, 11.x, 15.4, 9.3, 9.4, 9.5, 11.2** never name the original section they build on, or say "original ARENA …" without a number. The route slice (#30) and each lesson plan's Prerequisites line fix this.
- **17.3** says "[1.2] activation patching" as a prerequisite; 1.2 has no activation patching at either pin (it is taught in 1.4.1 §3). Predates the sync. #16.
- **Dead Streamlit links** (upstream retired its per-chapter Streamlit apps; `c85ddd73` repoints its own): 6.2 line 2926, 9.5 line 1156 (the chapter 3 app), 17.2 line 857, plus `master_6_2.py:3468`, `master_9_5.py:2281` and `master_17_2.py:1114`. Proposal: replace with learn.arena.education links (§4.4) and extend the repo-link test of #37 P9 to flag `streamlit.app` hosts.
- **Chapter-level `requirements.txt` files** in the extension (for example `chapter7_activation_to_language/instructions/requirements.txt`) still say `transformers>=4.51,<5`, against upstream's `>=5.9,<6` and the single lock. They are leftovers of the per-chapter requirements era; no upstream change forces anything, but the lock is the only runtime #23 supports.

## 3. Lesson patterns: taste-bar amendments and per-lesson findings

What upstream's new and rewritten sections (0.5.x, 2.2.x, 2.3.2, 2.4.1, 2.4.2, 2.5) do that the extension could adopt, and where they conflict with the taste bar. The new sections follow the original format closely: the same difficulty/importance/time yaml block, the same "Learning Objectives" quote blocks, the same Hint/Help/Solution dropdowns and `=== NEW CHAPTER ===` markers. The differences worth recording are these.

### 3.1 Candidate amendments

Each candidate says what the evidence is, where it would land, and who decides. None changes the taste bar by itself: #243's checklist was accepted by the user, so these go to the user, either as an amendment to #243's items or as a per-lesson finding in the plan tickets.

**A. Item 8 asks for "at least one hint" per exercise; original ARENA does not.** Counted as `Difficulty:` blocks in the instruction pages:

| | exercises | with any hint | with a Solution dropdown |
|---|---|---|---|
| Upstream chapters 0–4 at `6a46100c` | 533 | 102 (19%): 80 `<summary>Hint…` dropdowns plus 22 inline "Hint -" lines | 488 (92%) |
| Upstream at `f9f034bd` | 543 | 102 (19%) | 498 |
| Extension chapters 5–18, 0.6, 1.6 on `integration/extension` | 1,110 | 517 (47%) | 1,058 (95%) |

In the extension, 41 of the 86 lessons have no hint text at all, so they already miss item 8 as written (5.1–5.5, 5.8, 5.9, 6.1, 6.2, 6.4, 6.5, 8.6, 8.7, 9.3–9.6, 11.1–11.5, 12.5, 13.1, 13.2, 14.1, 14.2, 14.5, 15.1–15.3, 16.1–16.5, 16.7, 16.8, 16.10, 16.11, 17.1; a regex count, not a reading). Upstream puts hints where a learner is likely to stumble, and makes them **graduated and named**: the errata fix `e636a7f3` for 1.3.4 adds "Hint - how PyTorch forward hooks work" (the concept) and then "Hint - code skeleton" (the shape of the answer). Two ways out, for the user: **(i)** amend item 8 to "a hint where learners are likely to stumble; every exercise has a Solution dropdown", which matches the original's practice and the 41 lessons as they stand; or **(ii)** keep item 8 and have each lesson plan list the hints it adds, which costs writing time on about 590 exercises. Recommendation: (i), plus the named, graduated form as advice in the authoring file. Lands: Spec "Taste bar" item 8; `AUTHORING.md`; every plan ticket (item 3, "parts").

**B. Tests that name the mistake.** Upstream's audit method is the one part of it the extension should copy: *every new or strengthened test passes the reference solution and at least one legitimate alternative implementation, and fails targeted buggy versions with a message that names the bug* (#746's description; commits `793cbd45`, `aeb40ae2` and `170fee86` rework 1.3.1's `test_extract_activations` to "report one mistake at a time" and "only name a mistake when the activations match it"). `AUTHORING.md` already says "prefer independent oracles… give informative assertion messages"; add the alternative-implementation check and the named-bug message. Do **not** copy the audit's independent hunk-by-hunk reviewers (it re-checked every hunk and corrected 27), which the lightweight-verification rule excludes. Lands: `AUTHORING.md` (tests.py section); Spec "Testing Decisions" (supporting CPU checks); plan item 3 ("what each test checks").

**C. Test fixtures that need no download.** Upstream's 2.4.1 trainer tests use a float64 CPU toy language model (`_ToyLM`), 2.4.2's use a fake causal LM (`_FakeCausalLM`), and `test_vae` injects a `FixedEncoder` so an initialisation bug cannot hide behind a lucky draw (`part51_vaes/tests.py`). One shared `_close(actual, expected, name, what, hint)` helper keeps messages uniform. This is what the focused-test rule ("one CPU command per lesson, no model downloads") needs for lessons that train on a language model. Lands: `AUTHORING.md` tests section; plans for lessons that train on a model (7.3, 7.4, 15.x).

**D. Smaller things upstream's new sections do well** (each a candidate for `AUTHORING.md` or a lesson plan's items 2 and 3; no taste-bar item changes):

- **"You build / We provide" list.** 2.4.2's intro says which pieces the learner writes and which are given (2.4.2 page, lines 90–104), and 0.5.x says "Don't copy this code in" before a reference block. It makes Setup honest about how much is the learner's. Fits plan item 2 and the AUTHORING rule that the learner implements the core method while utils hold plumbing.
- **Optional exercises are titled `Exercise (optional) - …`** and carry no yaml block (2.3.2). The extension's optional work lives in the Bonus; the tag is a cheap way to mark one inside the core.
- **"Why …?" and "Worked example" dropdowns** (2.4.2): the reading of a result or a derivation sits in a dropdown after the exercise, like the "Help - how should I interpret this?" dropdown item 9 already asks for.
- **Credit for ported material.** 2.3.2 carries a CC BY attribution dropdown for its Iliad Intensive and `reward-lab` port. If any extension lesson ports CC-licensed material, that form is the model, and item 6's ban on "provenance" on learner pages should say that credit and licence notices stay. No extension lesson is known to need it today.
- **Cheap environment lines:** a thread cap (`t.set_num_threads(min(4, …))`) and `tqdm.auto`. The `LOW_GPU_MEM` switch to smaller models conflicts with the one-reference-machine runtime (#23) and is not worth copying.

### 3.2 Where upstream's patterns conflict, and what to ignore

- **Openings.** 2.3.2 opens with two definitions and 2.4.2 with a contrast and a caveat, against taste-bar item 2 ("open with something concrete from this lesson"). The taste bar's own references are original 0.1, 1.2, 1.4.1 and 1.5.4, not the new sections, so nothing changes. One piece is worth borrowing: 0.5.2 opens with "connects to earlier days" bullets that cite earlier sections, a fair form for item 3's prerequisite list.
- **Status banners.** 2.4.1 and 2.4.2 carry a "Work in progress" banner and the config `status` string prints on every page. Item 6 forbids both on learner pages. Ignore. The same string drives the route (§4.5).
- **Hosted figures.** Upstream keeps figures, GIFs and playable pages in a separate repository (`info-arena/ARENA_img`, served by GitHub Pages), regenerated by a section-local `fig.py` marked "not used by the notebook", and 2.5 pins its images by commit SHA. That needs a pull request to another repository and does not work offline; the extension's in-repo `media/NAME.png`, inlined into the exercises notebook, is the better design for a learner on the no-GPU tier. Ignore the hosting. 2.3.2's browser-playable pottery shop, placed before any exercise, is nice but needs the same hosting; the extension's equivalent is its "Try it yourself" cell.
- **Expected outputs.** Upstream's new sections have almost none (0 in 2.3.2 and 2.4.2, 1 in 2.4.1, 6 in 0.5.1). The extension's "real executed output in a dropdown" is the stricter bar. Keep it.
- **Pipeline guards.** Upstream's build now fails on unbalanced `<details>` and requires a blank line after `</summary>` (`cell.py`, `30583112`, `03671660`). A scan of the 84 chapter 5–18 instruction pages found no `</summary>` without its blank line and no unbalanced `<details>`.

### 3.3 Per-lesson findings that came out of the comparison

These are defects in extension pages, found because upstream's pages have the right order. A script compared, for each `### Exercise` block with both a stub and a Solution dropdown, the position of the dropdown against the first `raise NotImplementedError`:

- **Solution before the stub: 100 of 1,046 exercises, in 10 lessons: 6.1, 6.2, 12.5, 13.1, 14.2, 14.5, 15.1, 16.1, 16.7, 16.8.** The learner meets the answer before the exercise. Upstream has 0 of 370. Spot-checked by reading 15.1 (first exercise, lines 196–275), 6.1 (line 148) and 14.5 (line 277).
- **Two difficulty blocks on one exercise: 40 exercises, in 6.2, 12.5, 13.1, 15.2 and 16.7.** The page carries the 🔴/🔵 yaml block and, below it, an older "Difficulty: easy / Importance: high / Time: 6 minutes" block (15.2, lines 203–215).

The early cleanup (#244) did not catch either. Proposal: each lesson's plan lists the fix, and the focused tests gain one structural CPU check ("each exercise has one yaml block, and the stub precedes the Solution dropdown"). Lands: Spec "Taste bar" item 8 (it lists the order: task, test, hint, Solution dropdown); Spec "Testing Decisions" (supporting CPU checks); plans #252 (6.1), #253 (6.2), #20 (12.5, 14.2, 14.5), #255 (13.1, 15.1), #21 (16.1), #262 (16.7, 16.8), #22 (15.2).

## 4. Infrastructure and tooling

### 4.1 Upstream's workflows and scripts

All three upstream workflows (`build-master.yml`, `check-master.yml`, `pr-preview.yml`), `CODEOWNERS`, `PR-PREVIEWS.md` and the preview scripts are already on `integration/extension` unchanged: `git diff 6a46100c origin/integration/extension -- .github` shows only the extension's own `extension-quality.yml` added. `infrastructure/core` differs only in the config files (`config.yaml` merged, `config_original.yaml` the frozen upstream copy, `config_extension.yaml`). Chapters 0–4 differ from the pin by 68 added files (0.6 and 1.6) and nothing else.

| Item | What it does (read from the file) | Treatment |
|---|---|---|
| `build-master.yml` | On a push to `main` that touches `infrastructure/chapters/**/master_*.py`, runs `generate_master.sh --changed` and commits the regenerated pages and notebooks back as `ARENA-workflow[bot]`, using a token minted from a GitHub App (`vars.APP_CLIENT_ID`, `secrets.APP_PRIVATE_KEY`, ruleset bypass). All or nothing. | **Disable in the repository's Actions settings once it first appears; do not edit the file.** See §4.2. |
| `check-master.yml` | On pull requests touching masters: regenerates the changed sections read-only, pass/fail, and uploads the changed pages as an artifact. | **Ignore.** The fork takes no pull requests (#37), and the extension's own checks cover its masters. |
| `pr-preview.yml`, `publish-pr-preview.sh`, `preview-notebook.py` | After a passing check, force-pushes an orphan `pr-preview` branch and comments a `learn.arena.education/pr-preview/pr-N/` link; a separate website repository renders it. | **Ignore.** Without that website it only makes a stray branch and a dead link. Inert while there are no pull requests. |
| `generate_master.sh`, `core/scripts/README.md` | Selects sections by `git diff` and runs `main.py` per id. Its `CHAPTER_DIRS` lists chapters 0–4, but `--changed` and `--all` find any `master_<digit>*.py` under `infrastructure/chapters`, extension masters included. | **Link.** The README is the best document of upstream's pipeline. |
| `main.py`, `master_file.py`, `cell.py` | Master `.py` is the source of truth; the `.ipynb` is generated and may be missing; exit non-zero on failure; fail on unbalanced `<details>`. | Already merged; the extension's `build.py` does the same (`find_master` by `rglob`, stub notebook). **Ignore.** |
| `config.yaml` | Adds groups (`is_group`, `children`, `parent`, `local_path`) for 0.5, 2.2 and 2.4, a per-section `status`, and the section `id` that forms the site URL. | **Adapt at each sync:** copy upstream's `config.yaml` over `config_original.yaml`, then regenerate the merged file (`scripts/build_merged_config.py`). A plain git merge of `config.yaml` conflicts every time. |
| `install_uv.sh` | uv, Python 3.13, a `.venv`, a kernel named `arena-env`, and `~/.vscode/settings.json` `extraPaths` for chapters 0–4. | Already adapted (Python 3.14, repo root). **Small adapt:** `extraPaths` still lists only chapters 0–4. |
| `.gitattributes` | `* text=auto eol=lf`. It explains most of the whole-file churn in the range: the chapter 0 and 2 diffstat is 43.5k/36.9k lines, versus 37.7k/31.1k with `--ignore-space-at-eol`. | Already adopted. |
| `requirements.txt`, `pyproject.toml` | The ranges in §4.3. | Already merged (the extension's `pyproject.toml` states its `torch>=2.15` override of the cap). |
| `.streamlit`, `style.css`, `st_chat.py`, `test.py`, `django_run/*`, per-chapter `requirements.txt` | End-of-line changes, the website app, a stray heading lister. `config.yaml` points the 2.4 group at `2_4_rl_llms.md`, which is not in the repository. | **Ignore.** |

### 4.2 `main` will move onto upstream's `build-master`

GitHub lists one workflow on the fork today, "Extension Quality", `disabled_manually`; the fork's `main` is still `f9f034bd`, which lacks the other three. `gh variable list` and `gh secret list` on the fork return nothing.

When a release moves `main` (#37 P2) onto a commit that contains `build-master.yml`, the push changes `infrastructure/chapters/**/master_*.py`, which is the workflow's path filter. So **the first release push starts it**, and its first step mints the App token from variables the fork does not have. The expected result is a red X on the release commit with nothing pushed, and the same on every later push that touches a master. If the secrets were ever set, `generate_master.sh` would regenerate sections, extension ones included, with the stock converter and push a bot commit over the extension's own `build.py` output. That would also make the next release push non-fast-forward, which #37 P2 relies on never happening.

Proposal: after the first push, disable `build-master`, `check-master` and `pr-preview` (`gh workflow disable`, or Settings → Actions), and add "upstream's three workflows are disabled" to the documentation and CI slices (Spec batch 4: P4 and P6). Do not edit the files: they are upstream's, so any edit conflicts at every sync, and P6 only switches on the extension's own workflow. This is a repository-settings change; the user decides, and nothing was changed here.

### 4.3 Runtime: upstream's ranges, and what the env-upgrade batch already settled

Read from `requirements.txt` at both pins (primary source):

| Package | `f9f034bd` | `6a46100c` | In the fork's lock today |
|---|---|---|---|
| torch | unpinned | `>=2.11,<2.15` (comment: CUDA 13 build; the cap stops a silent move to a new CUDA major) | 2.15.0+cu134; the cap was dropped because its reason does not apply (#23 comment) |
| transformer_lens | `>=2.16.1,<3` | `>=3.9,<4` ("still has HookedTransformer, deprecated in favour of TransformerBridge") | 3.9.0 |
| transformers | `>=4.51,<5` | `>=5.9,<6` | 5.22.2 |
| sae-lens / sae-vis | `>=6.37.4,<7` / `==0.3.7` | `>=6.51.3,<7` / ARENA fork `v0.3.7+arena.2` | 6.55.0 / the fork |
| nnsight | `>=0.3.5` | `>=0.7,<0.8` | 0.7.0 (cap kept) |
| numpy | forced `<2` by TL<3 and sae-vis | `>=2` allowed | 2.5.4 |
| gymnasium / jax | `==0.29.0` / `[cuda12]==0.10.1` | `>=1.2,<2` / `[cuda13]==0.11.2` | 1.4.0 / 0.11.2 |
| Python | (no `install_uv.sh`) | `install_uv.sh` defaults to 3.13 | 3.14.10 |

The env-upgrade batch folded the first catch-up in and posted its result on #23 (comment of 2026-11-07, `integration/extension` at `2925d8ed`): **82 of 86 extension lessons and original 0.1, 1.2, 2.1, 2.2.1 and 2.3 run end to end in fresh kernels on this lock**; 13.1, 16.3, 16.5 and 16.8 stop on lesson bugs that fail the same way in the old environment. So two things the #37 notes left unverified are now answered for `6a46100c`: upstream's dependency set installs on Python 3.14 next to torch 2.15, and the 13 extension lessons whose pages name `HookedTransformer` (7.1–7.5, 8.1–8.5, 8.8, 12.3, 16.6) are not among the four that stop. TL 3.9 keeps `HookedTransformer`, and upstream's own chapter 1 pages still use only `HookedTransformer` (no page mentions `TransformerBridge`), so **no extension lesson has to move to `TransformerBridge` for this pin**. Nothing in this ticket's remit remains open here; the live risks are draft #439 (TL 4) and #738 (upstream's own lock), both in §5.

### 4.4 Links from extension pages into upstream's generated notebooks

The extension already links, by relative path, to upstream's generated exercise notebooks. In the instruction pages there are five, all to `*_exercises.ipynb`:

| Extension page | Links to |
|---|---|
| 5.5 State Tracking | `chapter1_transformer_interp/exercises/part53_othellogpt/1.5.3_OthelloGPT_exercises.ipynb` |
| 6.2 Gemma Scope and Feature Steering | `…/part33_interp_with_saes/1.3.3_Interpretability_with_SAEs_exercises.ipynb` |
| 16.1 Exact Shapley | `…/part54_toy_models_of_superposition_and_saes/1.5.4_Toy_Models_of_Superposition_&_SAEs_exercises.ipynb` |
| 16.10 Influence Functions, 16.11 Datamodels | `chapter0_fundamentals/exercises/part3_optimization/0.3_Optimization_exercises.ipynb` |

The `lesson.md` source files of 9.1 (to 1.3.2 Function Vectors) and of the influence, datamodels and cartography parts of chapter 16 (to 0.3) carry the same kind of link.

Upstream's open PRs #731 and #740 (§5) stop committing generated notebooks to `main`: they would exist only after `./build_notebooks`, and Colab would read an orphan `colab` branch. If they land, each of these links is dead on a fresh clone, and #37 P9's repo-link test ("each link is relative, or points at a `blob/main/<path>` that exists") would fail on them.

Proposal: **new prerequisite and "see also" links to original sections go to the section's learn.arena.education page** (`https://learn.arena.education/<chapter dir>/<section id>/`, ids from `infrastructure/core/config.yaml`: `13_saes`, `14_activation_oracles`, `12_function_vectors`, `04_rlhf`, `51_vaes`, …), not to `.ipynb` paths. All 45 section and group URLs listed in upstream's `config.yaml` answer HTTP 200 and a made-up URL answers 404, so the check is meaningful (probed 2026-11-07, status codes only). Convert the existing relative links in the same slice that adds the route's links. Links into upstream stay "upstream's" under #37 P9, so the route walk covers them.

### 4.5 No core or optional marking exists to follow (affects #37 P16)

P16 says a new upstream section joins its chapter's route block "following upstream's own core or optional marking". At the pin there is almost none to follow:

- upstream's `README.md` still says chapter 4 is "Coming soon!" and names none of 0.5.x, 2.2.x, 2.3.2 or 2.4.x; it points contributors at a removed `infrastructure/master_files` path;
- the only core/optional statements are chapter 1's own: it opens by saying chapter 0's material was all compulsory, then says only the first two exercise sets of chapter 1 are, the rest optional, with IOI (1.4.1), superposition (1.5.4) and function vectors (1.3.2) weakly recommended. Chapters 2, 3 and 4 have none;
- `config.yaml` has a per-section `status` string instead: 27 "All exercises complete and verified", 1 "Exercises mostly verified, but may need some polishing and testing" (2.5), 3 "Work in progress" (2.3.2, 2.4.2, 2.4.3).

So the route slice (#30) chooses core/optional for the new sections itself, with `status` as the one upstream signal: **2.3.2, 2.4.2 and 2.4.3 enter the route as optional and are not named as anyone's prerequisite until their `status` changes**. The "what's new upstream" list should print each new section's `status` string.

### 4.6 The errata bot, and how the sync's "what's new upstream" list should treat it

- `arena-errata-bot[bot]` is a GitHub App (id 337331655) that authored 142 non-merge commits in the range. 56 carry the title prefix `[errata-bot][<section>]`; the other 86 use plain titles such as `[2.2.1] … (v8-58)` or `… (errata-497)`. By section tag, all 142: chapter 2 has 53, chapter 1 34, chapter 0 25, chapter 3 20, chapter 4 8, other 2.
- The mechanism is inferred from the commits only, since no bot code is in the repository: it opens same-repo pull requests from branches named `errata-bot/<epoch>-<rand>`, and a maintainer merges them (37 such merges in the first-parent history, plus 13 squash-merged ones). Commit messages cite an errata id, a Slack permalink and "Approved in the errata audit tool" (`72897cbb`). It edits master files and `tests.py`, never generated outputs (33 distinct masters touched).
- 21 more errata-bot pull requests are open (§5).

**Treatment: take the output, ignore the bot.** The fork has no learner-report channel and, under the lightweight-verification rule, no use for an approval tool. What the sync batches should borrow is the commit shape: errata ids and one-line titles make a sync's "what's new upstream" list easy to compress.

### 4.7 A recipe for each sync's "what's new upstream" list

The #37 sync batches post this list for the user's sample walk, and the list should be short enough to read. From this note's own method, in order:

1. **Collapse the noise.** In this range, 296 of 857 non-merge commits are the `ARENA-workflow[bot]` regeneration (154) and errata-bot (142) commits. Report them as counts per section, not as lines.
2. **Group the rest by master file**, not by title prefix: 323 of the 703 commits other than the regeneration bot's are untagged. `git diff --numstat OLD NEW -- infrastructure/chapters` gives the per-section `+/−`; `git log --no-merges OLD..NEW -- <master>` gives the commits.
3. **Name structural changes first:** sections added, split, renamed or deleted (`git diff --name-status -M OLD NEW`), and the `config.yaml` diff, which also shows each new section's `status`.
4. **Name the extension lessons touched:** grep the extension's pages and `lesson.md` files for the changed section numbers (`[x.y]` and `x.y`) and for relative links into the changed `exercises/` folders.
5. **List the open upstream pull requests** that would change layout or requirements (§5 shows the kinds), with their size.

This is the same procedure that produced §1 and §5.

## 5. Looking ahead: open upstream pull requests on 2026-11-07

These are not in the pin. They decide how big the *next* sync is, and the first three change the layout the extension reads.

| PR | What | Size | Why it matters |
|---|---|---|---|
| #756, #758, #759 (chapters 1, 3, 4) and #754 (infra) | Content audits split per chapter from the closed #746. #754 "merge first". All opened 2026-11-06; chapters 0 and 2 already merged (#755, #757). | 32 / 12 / 14 / 17 files | The next sync's bulk. #746's description lists real bugs: `main`-tagged cells corrupted multi-line strings in `solutions.py` (1.2 loss 4.30 instead of 4.2514), 1.4.2 `W_skip` applied transposed, 4.3 blackmail loaders matching the wrong rollouts. Expect fresh-kernel reruns for lessons that import this code. |
| #731, #740 | Commit only `master_*.py`; generated notebooks stop being committed to `main`. They go to an orphan `colab` branch, and local students run `./build_notebooks`. #740 is a draft stacked on #731. | 50 / 174 files | **Breaks the extension's relative links into upstream's exercise notebooks** (§4.4) and changes what "upstream's files match the pin exactly" means in the preservation audit. |
| #738 | Locks student dependencies with `pyproject.toml` + `uv.lock`; `install_uv.sh` runs `uv sync --frozen`; a `lock-refresh.yml` is inactive until a repo variable is set. | 16 files | Same file names as the fork's own `pyproject.toml`/`uv.lock` (P15: "ours"). Conflicts will recur at every sync until upstream's file is renamed the way `requirements-original.txt` was. |
| #739 | Replaces the vendored website copy with an `arena-heroku` submodule. | 56 files, −35,757 lines | Removes upstream's vendored website copy (most likely under `infrastructure/core/django_run/`, which the extension does not use; not checked against the PR diff). TAKE. |
| #683 | New [4.6] Finetuning and Midtraining (and [4.6.1] hackathon edition). Draft, "please don't merge yet". | 25 files, +17,288 | New original section. Overlaps chapter 15 (PEFT) and 17.3 (fine-tuning diffs) by topic. Re-read at the sync that includes it. |
| #706, #737, #745, #766 | 2.4.3 RLAIF content, 2.4.3 learning-rate fix, 2.3.2 bonus (keys and chests), 0.5.2 DCGAN simplification. | small | Move 2.3.2, 2.4.3 and 0.5.2 under the plans' feet. |
| #439 | Port chapters 1, 2.4 and 4 to TransformerLens 4 (`TransformerBridge`). Draft since 2026-09-25, last updated 2026-10-18. | 33 files | If it lands, upstream's range moves past `<4` and the 13 `HookedTransformer` lessons need a real check. |
| #702 | [1.3.2] function-vector solutions on nnsight 0.8 remote, "bump pin". | small | The fork's lock keeps `nnsight<0.8` (upstream's cap at the pin, kept in #23). The sync that includes #702 needs a lock decision, not just a merge. |
| 21 open errata-bot PRs | 13 are chapter 3 (3.1 ×4, 3.2 ×2, 3.4 ×2, 3.5 ×4, one tagged 3.2 and 3.4), 2 are 4.5, and one each is 1.3.4, 2.3 and 2.5, with two for 2.4.1. | one-line fixes | TAKE. They arrive as the "errata" part of the next sync. |
| #680, #761, #741 | `install_uv.sh` for Linux, macOS and Windows; setup cells look in the cwd for the repo root; a no-copy marker for reference-only code blocks. | small | TAKE. |

## 6. Where each recommendation lands

Spec sections are those of [Spec #39](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/39); plan tickets are its "Plan tickets → Owners" list. Plan item 8 ("cuts, merges, retitles and additions from #237, #248, #251") is where each lesson plan records what it takes from this note.

| # | Recommendation | Spec #39 section | Plan ticket(s) or slice |
|---|---|---|---|
| R1 | Take chapters 0–4 as they are. The sync's own checks cover them; no extension lesson depends on anything upstream changed, apart from the items below. | Integration, upstream sync and releases (P13); Testing Decisions item 1 (preservation audit) | the sync batches |
| R2 | The route's original blocks use the new numbers (0.5.1/0.5.2, 2.2.1/2.2.2, 2.3.2, 2.4.1–2.4.3). 2.3.2, 2.4.2 and 2.4.3 enter as optional and are named as no one's prerequisite while their upstream `status` is "Work in progress". The "what's new upstream" list prints each new section's `status`. | Study route; Integration… (P16) | route slice (#30) |
| R3 | New links to original sections go to learn.arena.education pages, not to relative `.ipynb` paths. Convert the five existing page links (5.5, 6.2, 16.1, 16.10, 16.11) and the `lesson.md` ones with the route's links, and fix the three dead Streamlit links (6.2, 9.5, 17.2). Extend the repo-link test to flag `*.streamlit.app`. | Integration… (P9 repo-link test); Study route | route slice (#30); the link-test slice |
| R4 | 6.1: cut the Gated and JumpReLU re-implementation to "use yours or ours", link 1.5.4 §5. | Plan tickets (item 8) | #252 |
| R5 | 7.2: shrink §1 to a recap with a pointer to 1.3.3's dashboard exercises. 7.4: link 2.4.1 §1 for GRPO and the KL penalty, and shrink the two exercises that re-implement them. 7.1: one link to 1.3.3's patch scoping. | Plan tickets (item 8) | #258 (7.2, 7.4), #9 (7.1) |
| R6 | 9.2: add 1.3.1 as a prerequisite and shrink the mean-difference, covariance and logistic-fit exercises. 9.4 and 15.4: link 4.1 §4. | Plan tickets (item 8) | #12 (9.2), #19 (9.4), #261 (15.4) |
| R7 | 6.2, 6.3, 6.4, 8.7: add the missing links to 1.3.3 and 1.4.2; 8.7 turns its "missing" list into "do the real one first in 1.4.2". | Plan tickets (item 3 Prerequisites; item 8) | #253 (6.2), #11 (6.3), #257 (6.4), #17 (8.7) |
| R8 | The mean-difference direction is built once, in 9.1; 9.2–9.5 and 11.5 link back and may take it as a given. | Intuition at first use and the 0.7 Foundations Toolkit; Plan tickets (item 8) | #9, #12, #19, #260 |
| R9 | 17.3: fix the "[1.2] activation patching" prerequisite (it is 1.4.1 §3). | Plan tickets (item 3) | #16 |
| R10 | LoRA: 15.1 links 2.4.1 §2 with a convention warning (`B@A` against `x A B`); 15.2 shrinks its four wrapper exercises to a pointer unless #23's no-GPU tier needs them; 15.3 and 15.5 link. | Plan tickets (item 8); Runtime and performance (hardware tiers) | #255 (15.1), #22 (15.2, 15.3), #261 (15.5) |
| R11 | Fix the page-structure defects of §3.3 in each lesson's plan (Solution before the stub: 6.1, 6.2, 12.5, 13.1, 14.2, 14.5, 15.1, 16.1, 16.7, 16.8; doubled difficulty blocks: 6.2, 12.5, 13.1, 15.2, 16.7) and add one structural CPU check to the focused tests. | Taste bar (item 8's order); Testing Decisions (supporting CPU checks) | #252, #253, #20, #255, #21, #262, #22 |
| R12 | **User decision:** amend taste-bar item 8 from "at least one hint" to "a hint where learners are likely to stumble" (§3.1 A), or keep it and plan the hints. | Taste bar (item 8) | #243's checklist; every plan ticket (item 3) |
| R13 | `AUTHORING.md`: tests pass an alternative correct implementation and fail named bugs; no-download toy fixtures; "You build / We provide"; `Exercise (optional)`; named, graduated hints (§3.1 B–D). | Taste bar (the authoring file is the first slice); Testing Decisions | the authoring-file slice |
| R14 | After the first release push, disable upstream's three workflows in Actions settings; do not edit the files (§4.2). | Integration, upstream sync and releases (P4, P6) | documentation and CI slices (batch 4) |
| R15 | Each sync's list follows the recipe of §4.7; the next sync's watch list is §5. | Integration, upstream sync and releases (P13) | Batch 0's walk-stop list; every later sync batch |
| R16 | Lock decisions the next sync forces: nnsight's `<0.8` cap against #702, upstream's own `pyproject.toml`/`uv.lock` (#738), generated notebooks leaving `main` (#731, #740). | Integration… (P15 shared files, P17); Runtime and performance | the next sync batch |
| R17 | The chapter-level `requirements.txt` files (still `transformers>=4.51,<5`) go with the lesson-folder cleanup; the lock is the only runtime. | Lesson files; Runtime and performance | batch 3 (lesson-folder cleanup) |

## 7. What this note does not establish

- **Exercise bodies were not read end to end.** Overlap calls rest on introductions, objectives, exercise names, time lines, prerequisite lines and the passages cited. In particular, whether 9.2's `mass_mean` and 1.3.1's `MMProbe` are implemented the same way was not checked, and 1.3.3's feature-splitting section was characterised from headings and opening paragraphs.
- **No lesson code was run.** No code, test or notebook from either course was executed (only counting scripts, `git`, `gh` and `curl` status checks); the runtime facts come from `requirements.txt`, commit messages and the env-upgrade batch's posted results (#23).
- **Counts are regex counts** over the generated instruction pages (`Difficulty:` blocks, hint dropdowns, inline "Hint -" lines). They are good for comparing the two pins and the two courses; they are not an audit of any one lesson.
- **Open upstream pull requests change daily.** §5 is a 2026-11-07 snapshot read with `gh`; upstream `main` was still `6a46100c` when it was taken.
- **learn.arena.education was probed for HTTP status only** (45 section and group URLs, plus one made-up URL for a 404 control), not read lesson by lesson. A status of 200 shows the page exists, not that its content matches the pin.
- **The upstream licence question** stays as #37 left it: none at either commit. This note links to upstream pages and quotes only short phrases (section titles, hint titles).

## Sources

- Upstream: `ARENA-education/ARENA_materials`, `f9f034bd..6a46100c` (`git log`, `git diff`, `git show` on the fork's clone, where both commits exist), `infrastructure/core/config.yaml` and `requirements.txt` at both commits, `.github/` and `install_uv.sh` at `6a46100c`; open pull requests #439, #680, #683, #706, #731, #738, #739, #740, #741, #745, #746, #754, #756, #758, #759, #761, #766 and merged #743, #755, #757 (`gh pr view`).
- The extension: `integration/extension` at `09b2557a` (instruction pages, `lesson.md` files, `infrastructure/extension/AUTHORING.md`, `syllabus.yaml`, master files).
- Fork planning: [Spec #39](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/39) ("Integration, upstream sync and releases", "Taste bar", "Plan tickets"), the [#37 resolution](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/37), [#243](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/243), the [#23 env-upgrade comment](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/23#issuecomment-6705376357), [Batch 0](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/265) (no "what's new upstream" list in its comments yet).
- The live site: https://learn.arena.education/ (section URLs, status codes only).
