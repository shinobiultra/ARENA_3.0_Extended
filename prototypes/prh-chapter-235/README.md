# PROTOTYPE (throwaway, never merged): PRH chapter placement and claims, ticket #235

Question: where does the Platonic Representation Hypothesis chapter sit, and which convergence claims does a learner actually test?

- `chapter19_outline.md` is the outline to react to: placement, prerequisites, three sections, the claims table with honest nulls, signature results and quiz questions.
- `prh_toy_probe.py` is a CPU toy (numpy/scipy, random data, about 2 minutes) that produces every number in the outline: `run_output.txt` and `prh_toy_probe.png`.
- No model, dataset or GPU was touched. Real-model sizes and times are estimates.

Run: `<checkout>/.venv/bin/python -I prh_toy_probe.py > run_output.txt`

## Decisions (recommendation first)

| ID | Decision | Recommendation | Alternatives |
| --- | --- | --- | --- |
| PRH-D1 | Placement | **New chapter 19**, appended after 18, with three sections 19.1–19.3. Every prerequisite has a lower number, so the JEPA revisit needs no cycle. Sections go from 87 to 90. | (B) 11.7 plus 14.8, the ticket's earlier "geometry entry, JEPA revisit" (no new chapter, but the story is split across two chapters). (C) A Research Practice rung, 10.4, plus 14.8. |
| PRH-D2 | Section split | **19.1 Measuring Convergence** (CPU, PRH01–03), **19.2 Do Real Models Converge? Replicate and Critique the PRH** (GPU), **19.3 Proving and Improving the PRH with JEPAs** (CPU, PRH04–06; exact paper title kept). | Two sections, with 19.1 folded into 19.2's first part (a 6+ hour lesson). |
| PRH-D3 | Real-model tier | **Make 19.2 a core real-model replication.** This is new: the Nov 3 amendment's PRH01–06 are toy and finite-world only, with real models left as a future tier. | Keep the chapter toy-only and put real models in a Bonus. |
| PRH-D4 | Which claims are core | **Core:** C1 (cross-modal alignment rises with LM quality), C3 (caption density), C4 (what survives calibration, local vs global). **Try it:** C5 (S04 structure). **Bonus:** C2 (vision–vision). **Author-reported only:** C8 (S02's 42× samples, 109M vs 72B). PRH06 behaviour transfer goes in 19.3's finite world, with real-model transfer left to 11.2 Part 3. | Make C8 core via the authors' Flickr arm (1–2 GPU-h each, author estimate; Julia). |
| PRH-D5 | Organisms and data | **The Pythia ladder 70M–1.4B** (same data, same order) for captions × **DINOv2-B** for images, on S01's **1,024 WIT pairs** loaded at runtime (link, don't vendor; check the licence when the batch opens). CLIP's two towers are the positive control; an untrained Pythia-1.4B and shuffled pairs are the negative controls. 2.8B/6.9B are a Try it. | A Qwen3 ladder (modern, but its small models are distilled from large ones, which confounds scale; kept as a Bonus contrast). COCO captions. |
| PRH-D6 | Metrics and verdict rule | **m_NN** (k = 10, cosine, max over layer pairs, as S01 does) plus **debiased CKA**. A **whole-pipeline permutation null** (one row permutation shared by every layer pair, 1,000 permutations), bootstrap intervals over items, and #246's four verdicts through 10.1's `claim_verdict`, fixed before running. | m_NN only (simpler, but it loses the local-vs-global axis). |
| PRH-D7 | Signature results | **19.1:** CKA of unrelated noise against width (0.94 at d = 16,384), beside the local/global 2×2. **19.2:** raw against calibrated alignment across the ladder, m_NN and CKA, with null bands and both controls. **19.3:** residual against perturbation for wide, narrow and tied eigengaps. | 19.2's signature as the raw S01 curve alone, with calibration as a follow-up panel. |
| PRH-D8 | Critical framing | **Keep the chapter title** *The Platonic Representation Hypothesis: What Converges, Why, and When?* Each section opens with a claim card: who claimed what, in their words and linked, with the S03/S04 responses and S02's conditions. Predictions and the verdict rule come before any run. A faithful null is a finished result (RP5). Learner pages carry no status text. | — |
| PRH-D9 | Prerequisites and 0.7 | CKA and mutual k-NN are **shared tools** (12.7 uses m_NN, 17.4 uses CKA), so under #234's rule each gets a short intuition block in **0.7** (CKA in Part 2, m_NN in Part 3), with no new part. 19.1 implements and breaks both. Prerequisites: 19.1 ← 0.7 Parts 2/3/5, 11.1, 11.2 Parts 1–2 (0.6 recommended); 19.2 ← 19.1, 12.1, 10.2; 19.3 ← 19.1, 14.1, 14.4, 0.7 Part 4. | 19.1 is their home, and 12.7/17.4 get inline blocks linking forward to it. |
| PRH-D10 | Quizzes and compute | **Three quizzes** shaped like the sketch (3, 5 and 3 questions), whose key runs reuse cached features. The core fits the laptop. The **Julia worker** is used only for the optional 19.3 Bonus arm, with the user's permission. | — |

The Spec #39 section this changes is "PRH and JEPA chapter", which replaces the Nov 3 amendment's "Named PRH research chapter (PRH01–PRH06)" paragraph and table.
