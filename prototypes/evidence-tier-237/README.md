# PROTOTYPE (throwaway, never merged): evidence tier for the LatentLens and weight-space case studies, ticket #237

Question: which evidence tier should the LatentLens bridge (7.1 → 12.4) and the weight-space case (17.3 with 15.4) target first?

- `tier_outlines.md` is the outline to react to: the tiers, and each of the four bridge pieces (LL01 7.1, LL02 12.4, WSP02 15.4, WSP01 17.3) at each tier. It covers learner functions, signature, controls, prediction point, compute and limits.
- `evidence_tier_toy.py` is the T0 toy (numpy, about 5 s on CPU): A contextual retrieval and its traps; B a rank-1 weight edit that rescues completion without adding capability. Outputs are `run_output.txt` and `evidence_tier_toy.png`.
- `gpt2_contextual_lens_probe.py` is a T1 preview for 7.1: real GPT-2 small over TinyStories, the model and corpus 7.1 already uses. It runs on CPU in 33 s, offline from the local HF cache with nothing downloaded. Outputs are `gpt2_probe_output.txt` and `gpt2_contextual_lens_probe.png`.
- No VLM, Qwen checkpoint, Sparks artefact or GPU was touched. HF file sizes were read from the public API only. Every number not from a run is marked *estimate*.

Run (from this folder):

```bash
<checkout>/.venv/bin/python -I evidence_tier_toy.py > run_output.txt
HF_HUB_OFFLINE=1 <checkout>/.venv/bin/python -I gpt2_contextual_lens_probe.py > gpt2_probe_output.txt
```

## What the runs say

- **The T0 toy teaches the traps, not the claim.** With a phrase-level "any related word in the top 5" rubric, norm-matched random vectors pass 61% of the time (96% at k = 20). Background patches pass the broad rubric 90–97% of the time with no local grounding. In the toy the static embedding lens beats the contextual one (1.00 vs 0.86), because the toy is planted, so a toy can't rank lenses.
- **T1 in 7.1 is nearly free and gives a real, surprising result.** On GPT-2 the contextual lens recovers the current token 0.84–0.94 at every layer, where the embedding lens collapses to ≤ 0.05 after layer 0. Its "what comes next" signal (0.18–0.33) is mostly a bigram table (floor 0.24). Any two real states have cosine 0.53–0.72, so a raw cosine is meaningless without the null.
- **The weight-edit toy gives the Sparks lesson in one table.** A rank-1 edit touching 5.1% of the parameters takes accuracy from 0.11 to 0.49, while accuracy among finished answers falls from 0.61 to 0.49. Banning the loop at decode time, with no weight changed, gives the same 0.49. In the toy the gain is completion rescue, not capability.

## Decisions (recommendation first)

| ID | Decision | Recommendation | Alternatives |
| --- | --- | --- | --- |
| ET1 | Tier vocabulary | **T0 illustration, T1 core method on real caches, T2 faithful replication, plus a declared extension.** Lesson plans name a tier per piece; learner pages never print tier labels (taste bar item 6). | — |
| ET2 | LL01 in 7.1 (pilot, #9) | **T1 on 7.1's own GPT-2 and TinyStories**, as a short bridge after Part 2. It has its own k-NN functions, the lens-vs-layer figure and heatmap, shuffled-label and random-query controls, and the bigram floor. Hand fixtures are the immediate tests, with no separate planted toy part. +33 s CPU. | (B) A T0 toy part (planted, so a stand-in inside a real lesson). (C) Bonus only. |
| ET3 | LL02 in 12.4 (#13) | **T1 on a small real VLM if #13 gives 12.4 a real organism**: a learner-built reduced phrase bank, judge-free local and broad scores, the same controls, the object-token ablation as the causal step. **If #13 keeps 12.4 a toy, LL02 is T0** on the toy's caches. | (B) T0 regardless of #13. (C) Use the authors' Qwen2-VL bank (17.4 GB, package corpus, not the paper's). |
| ET4 | WSP02 in 15.4 (#261) | **T0 only**: the projector-sign identity test and the signed counterfactual on the existing `remove_output_direction`, plus one line forward to 17.3. | — |
| ET5 | WSP01 in 17.3 (#16) | **T0 toy B (core), then T1a (core, CPU)**: the learner's `parameter_diff` on the released base and edited Qwen3.5-0.8B (3.5 GB), plus the released MATH-500 samples' finished-vs-truncated breakdown. **T1b**, a paired regeneration with a decoding-only comparator on GPU, is a Try it with saved outputs. | (B) T0 only. (C) T1b in the core (17.3 then needs a GPU). |
| ET6 | T2 replications | **Neither is in a lesson or on the 10.2 menu.** Both are Bonus reading with an author-numbers claim card (LatentLens 68 / 32 / 24%; Sparks MATH-500 7.9 → 48.1%, token-limit hits 93 → 33%) and the Gemma null. | LatentLens T2 as a 10.2 menu item on the Julia worker. |
| ET7 | Claims carried at every tier | Retrieval is description, not causal use. Completion rescue is not capability. u → −u is an identity test and −ΔW a counterfactual. The failed Gemma follow-up stays. Paper corpus and package corpus are named apart. | — |

Changes: Spec #39 gains a short "Case-study bridges (#237)" section. Plan tickets #9 (7.1), #13 (12.4), #261 (15.4) and #16 (17.3) each take the matching row as an addition (lesson-plan item 7).
