# The two case studies at each evidence tier (PROTOTYPE, ticket #237)

Every toy number is from `run_output.txt` and every GPT-2 number from `gpt2_probe_output.txt`. Anything marked *estimate* was not run. Source facts are from #233 and `docs/mastery/research/inbox_20261103/weights_latentlens.md` (on `integration/extension`).

## The tiers

| Tier | What the learner runs | What it can support | What it can't |
| --- | --- | --- | --- |
| **T0 Illustration** | A planted CPU toy (numpy) | The mechanism, its traps and the controls, all visible | Anything about a real model. Results are true by construction. |
| **T1 Core method on real caches** | Their own functions on hidden states or weights of a real model at a pinned revision (or on released artefacts), with a reduced corpus or subset, judge-free metrics and controls | "The operation works, or fails, on a real model, with these controls" | The paper's headline numbers |
| **T2 Faithful replication** | The paper's models, corpus or index, judge or grader, and protocol | The headline numbers | Feasible only far beyond 24 GB / 30 min / one GPU run (see below) |
| *Extension* | A changed corpus, added interventions, new seeds | A new claim, labelled as an extension | Parity with the paper |

## A. LatentLens: contextual retrieval, bridge 7.1 → 12.4

Source: LatentLens v5 (arXiv 2602.00462v5, ICML 2026). The paper describes a VLM's visual tokens by their nearest *contextual* text vectors and shows the token in its phrase. Reported: 68% of tokens judged interpretable, against EmbeddingLens 32% and LogitLens 24%. That's across 15 VLMs, with a 2.99M-phrase Visual Genome index and an LLM judge (κ = 0.68 on 1,020 human labels). A token counts as interpretable if *any* of its top-5 is judged related.

### LL01 in 7.1 (pilot, batch 7, plan #9)

7.1 already loads GPT-2 small and caches `resid_pre` over TinyStories (`get_resid_and_logprobs`). The bridge sits after Part 2's `evaluate_lenses`, before Patchscopes.

| | T0 | T1 (previewed in `gpt2_contextual_lens_probe.py`) |
| --- | --- | --- |
| Data | Planted phrases and "patches" (toy A) | 7.1's own GPT-2 small and TinyStories: a 36,000-vector bank per layer (600 stories × 60 positions), 3,600 held-out query tokens |
| Learner writes | `normalize_rows`, `contextual_topk`, the rubric scorers | `normalize_rows`, `contextual_topk` (chunked), `bank_from_cache`, `layer_alignment_grid`; hand fixtures as immediate tests (stable ties, scale invariance, chunked = brute force) |
| Signature | Pass rates for object, background, random and shuffled queries | Match rate against layer for three lenses, plus the query-layer × bank-layer heatmap |
| What it shows | Random vectors pass a phrase-level "any related word in the top 5" rubric 61% of the time (27% at k = 1, 96% at k = 20). Background patches pass the broad rubric 90–97% of the time with 0% local grounding. | The contextual lens recovers the current token 0.84–0.94 at every layer; the embedding lens does 1.00 at layer 0 and ≤ 0.05 from layer 1 on. Whether the neighbour's next token is the query's next token: 0.18–0.33, against a **bigram floor of 0.24**, so it is mostly bigram lookup. The logit lens's next-token rate climbs 0.00 → 0.36. |
| Controls that fail | Shuffled metadata, norm-matched random queries | Shuffled bank labels give 0.011–0.017. Norm-matched random queries have top-1 cosine 0.08–0.10, against **0.53–0.72 for a random pair of real states** and 0.93–0.98 for the real top-1, so a raw cosine means nothing without the null. Mean-centring changes nothing (0.84–0.94). |
| Prediction point | "Will a random vector get a 'related' description?" | "Will the contextual neighbour know what comes *next* better than a bigram table?" (it barely does) |
| First-use intuition | Cosine k-NN → 0.7 Part 3 (the mutual k-NN block) | The same link, plus an anisotropy line: "any two GPT-2 states point the same way (cosine ≈ 0.6)" |
| Compute | ~0 s | 33 s on CPU for all 12 layers, the heatmap and the controls; seconds on GPU; no new download |
| Limit (Bonus) | Planted. In the toy the static lens wins (1.00 vs 0.86), so the toy can't rank lenses. | Text only. It reproduces the *operation*; the VLM claim lives in 12.4. Retrieval is description, not causal use. |

### LL02 in 12.4 (batch 26, plan #13)

12.4 *Visual Token Flow in Real VLMs* is a CPU toy today: a four-patch image, a hand-specified attention route, and `decode_visual_tokens` as a fixed 2-column readout. Its title promises real VLMs. **#13 decides 12.4's organism**; the tier here follows from that.

| | T0 | T1 (*estimates*, not run) | T1 via the authors' cache |
| --- | --- | --- | --- |
| Data | `clean_cache` / `corrupt_cache` from the toy, plus a toy text bank | A small real VLM at a pinned revision (Qwen3-VL-2B-Instruct is already in the local HF cache; SmolVLM or Qwen2-VL-2B are alternatives); a learner-built bank of a few thousand caption phrases from its own LM at 3–4 layers; 20–50 images with object boxes | `McGill-NLP/latentlens-qwen2vl-embeddings`: 8 layers × 2.18 GB = **17.4 GB**, the *package* corpus, not the paper's |
| Learner writes | A nearest-neighbour variant of `decode_visual_tokens` | `contextual_topk`, `patch_descriptions`, `local_hit` / `broad_hit` (judge-free: a top-k token names the box's object; any word of the image), `layer_alignment_grid` | The same, on a downloaded bank |
| Signature | A decoded grid: fixed readout against contextual readout | Patch overlay with context snippets for object and background patches; logit vs embedding vs contextual lens on the same patches; image-layer × text-layer heatmap | Same |
| Controls | Random, shuffled | Norm-matched random queries, shuffled metadata, background patches, same-layer-only, a k-sweep with an equal candidate count per lens | Same |
| Causal step | none | Optional: ablate the described object tokens and measure a predeclared answer change (from #233). Only if #13 adopts hooks. | none |
| Compute | ~0 s | ~5 GB bf16, minutes on the 5090 Laptop (*estimate*) | 17.4 GB download, which breaks the ≥ 30 GB-free rule today (81 GB free) |

### T2 (not in any lesson)

15 VLMs; a 2.99M-phrase VG index at about 2 GPU-hours of encoding and 13 h including index build per backbone; 26 GB of storage; an LLM judge. That fails 24 GB / 30 min and two of #248's replication questions (judge-free; a falsifying run under an hour).

## B. Sparks of Weight-Space Programming: 17.3 with a 15.4 bridge

Source: author blog (Takehi, Teoh, Zhang, Khattab, 2026-11-02) with released logs. Reported: an LLM editor, with no gradients but up to 300 Python runs and 50 batched forward requests (158 turns, 20.3 h), edits Qwen3.5-0.8B. MATH-500 goes 7.9 → 48.1%, GSM8K 13.8 → 67.1%, and token-limit hits fall 93 → 33%. A less-looping Gemma follow-up finds no established math gain.

### WSP02 in 15.4 (batch 25, plan #261)

15.4 is a planted toy and already has the edit: `remove_output_direction` (`update - outer(unit, unit @ update)`). **T0 is the only sensible tier.** Add two checks after Part 3:

- **Projector-sign identity:** editing along −u gives the identical matrix, because u uᵀ is unchanged. This is a test, not a negative control. (Toy B: `PASS`.)
- **Signed counterfactual:** −ΔW is not a mirror image. In toy B, +ΔW takes accuracy 0.11 → 0.49 and −ΔW takes it 0.11 → 0.00.

Then add one line forward to 17.3. This is about 15 minutes of learner time and no compute.

### WSP01 in 17.3 (batch 34, plan #16)

17.3 is a CPU toy (y = ab, `parameter_diff`, `relative_change`, `signed_margins`, `capability_accuracy`); it has no SVD of ΔW and no projection edit. The bridge goes after Part 1, or in the "Native fine-tuning" bridge section.

| | T0 (toy B, run) | T1a: released artefacts, CPU | T1b: paired regeneration, GPU (*estimate*) |
| --- | --- | --- | --- |
| Data | A tiny MLP solver (8 → 20 → 10 answers + 1 "loop") that usually loops | Base `Qwen/Qwen3.5-0.8B` @ `2fc06364` and edited `RikiyaT/Qwen3.5-0.8B-weight-space-programming` @ `d1671a17`: 1.75 GB each, Apache-2.0, ungated; released `eval/samples_{base,edited}_math500.jsonl` (1.6 + 7.8 MB) | The same two models, plus 20–50 MATH-500 problems, one sample each, a ≤ 2k-token budget |
| Learner writes | `output_direction_edit`, `weight_diff_summary` (support, SVD rank), `behaviour_breakdown`, `matched_random_edit`, `permute_hidden_units` | `parameter_diff` on real tensors (which matrices, rows and columns; SVD rank of each ΔW); `behaviour_breakdown` on released samples (finished / truncated, length, acc \| finished) | `behaviour_breakdown` on fresh paired generations for base, edited and a decoding-only comparator (larger budget or an EOS nudge) |
| Signature | Answered / accuracy / acc \| answered by condition, beside the ΔW heatmap | Real ΔW support and spectrum, beside the released finished-vs-truncated breakdown | The paired rescue/regression matrix |
| What it shows | Base answers 0.17 of the time (acc 0.11, acc \| answered 0.61). The rank-1 edit (21 of 411 parameters, 5.1%) gives answered 1.00, acc 0.49 and acc \| answered **0.49**: completion rescue, not capability. **Banning the loop at decode time, with no weight changed, gives the same 0.49.** | Whether the real edit looks like the toy's (one output direction, low rank), and how much of the headline is finished answers. The base samples are reportedly a 25-problem smoke test (#233 note; not verified here). | Completion rescue against capability on the real model, with a decoding-only comparator |
| Controls | Same-norm random edits (same support: answered 0.39 ± 0.23, acc 0.19 ± 0.11; random rank-1: 0.39 ± 0.19, 0.14 ± 0.08; 20 seeds); wrong row (acc 0.08); signed counterfactual; a hidden-unit permutation that leaves outputs identical while moving all 220 entries of W2 | Unedited matrices as the zero diff; the toy's predictions stated before the reveal | Decoding-only comparator; matched token budget |
| Compute | ~5 s CPU | 3.5 GB download; diff on CPU in about a minute (*estimate*) | Minutes on the 5090 Laptop (*estimate*); saved outputs for no-GPU learners |
| Limit (Bonus) | Planted | The editor itself is not run: replaying logs tests neither the editor nor generalisation. The Gemma null goes in the claim card. | Small n; one model family |

### T2 (not in any lesson)

T2 needs the editor runs: a paid frontier model, an external sandbox, about 20 h per trajectory. It also needs the full 500 × 4 evaluation, which the card reports as 43.4M tokens and 51 min on an H200. It fails the course's compute rules, and the editor can't be pinned.
