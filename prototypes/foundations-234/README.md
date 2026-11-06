# Prototype: where the foundations route lives, and what counts as enough (#234)

This is a throwaway prototype on the branch `prototype/foundations-234`. It will never be merged.

It answers the question in [#234](https://github.com/shinobiultra/ARENA_3.0_Extended/issues/234): where should learners get the intuition for the course's shared tools (KL divergence, PCA, t-SNE, least squares, bootstrap and so on), and what explanation is enough? The gaps come from the tool inventory in #241. The rule they must meet is taste bar item 7 in #243: intuition, or a link, at each tool's first use.

## What to look at

1. **[`first_use_in_host_lessons.md`](first_use_in_host_lessons.md)**: two real first uses, 7.1 (KL) and 11.1 Part 5️⃣ (t-SNE), shown as they are today and in three variants:
   - A: link only;
   - B: inline only;
   - C: both.

   It ends with a first-use table covering every gap in the inventory.
2. **[`0.7_foundations_toolkit_draft.md`](0.7_foundations_toolkit_draft.md)**: a draft of the foundations home as one ARENA-style section. It has:
   - a hook, Content & Learning Objectives and Setup;
   - **Part 1️⃣ KL divergence** and **Part 3️⃣ t-SNE**, written out in full, each with a skip check, intuition before the formula, an exercise (🔴/🔵 block, time line, test, hints, solution), "Make a prediction first:" and a summary;
   - stubs for Parts 2️⃣, 4️⃣ and 5️⃣;
   - two sample held-out quiz questions in the #245 format;
   - a Bonus.
3. **[`foundations_toolkit.py`](foundations_toolkit.py)**: the solutions, tests and figures behind both parts. It runs on a CPU in about 10 seconds. Every number in the drafts is in [`run_output.txt`](run_output.txt), and the figures are in [`figures/`](figures).

Nothing here touched the main checkout or the extension pages. No GPU or cluster job was run.

## Open decisions, with recommendations

| ID | Question | Options | Recommendation |
|---|---|---|---|
| **FR1** | Where does the remediation live? | **A** a foundations home, linked from lessons; **B** inline blocks in lessons only; **C** both | **C.** A short inline block at first use, ending in a "Go deeper" link to the foundations part that holds the exercise, controls and quiz. A alone sends someone who half-remembers KL away from 7.1. B alone re-teaches KL in eight lessons and gives the practice no home. |
| **FR2** | What shape is the foundations home? | **a** one new section **0.7 Foundations Toolkit** in chapter 0, with five parts; **b** five separate sections 0.7–0.11; **c** mastery-hub companions outside the syllabus | **a.** One section with the parts (1) distributions and KL, (2) geometry: covariance, whitening, PCA, effective rank, (3) t-SNE and UMAP, (4) least squares, ridge and logistic probes, (5) bootstrap, MC standard errors and permutation nulls. It is appended after 0.6, as 0.6 was, and 0.0 stays untouched and is linked. That makes 87 sections. The proposal IDs FND01–06 fold into these parts (FND02 → 1; FND01 and FND03 → 2; FND04 → 3; FND05 → 3 and 4; FND06 → 5), and #241's gaps fill the rest. 0.7 is not a required prefix: the study route (#30) links into it rather than running through it. Split it into 0.7 and 0.8 only if a walkthrough finds it longer than a day. |
| **FR3** | What must an inline first-use block contain? | (see right) | At most about 150 words plus one picture or a few toy numbers, **before** the formula. It says what the tool keeps or measures and its main trap, and ends with the "Go deeper" link. Every lesson gets the block or a one-line link at the tool's first use in that lesson. The **full block** is required where the course first uses the tool, and wherever the tool carries the lesson's signature result. A block may reuse a 0.7 toy figure. Any numbers it quotes must come from a real run, in 0.7 or in the lesson. |
| **FR4** | What makes a foundations part "enough"? | keep the #39 amendment's six diagnostic nodes that route mistakes to varied retries; or an ARENA part shape | **ARENA part shape**, accepted in the user's walkthrough: a hook with the part's own numbers; intuition worked by hand on a toy before the formula; one small exercise with an immediate test, hints and a solution; one "Make a prediction first:" on a changed case, with an interpretation dropdown; one control the learner sees fail (such as t-SNE on noise); a summary; and questions in 0.7's held-out quiz chapter (#245). There is no routing engine, no mastery threshold and no badge. |
| **FR5** | Should each part open with a "Can you skip this part?" check? | yes, two self-checked questions with answer dropdowns; no | **Yes.** It is what makes 0.7 remediation rather than a compulsory chapter. Nothing is recorded or scored; the learner checks their own answers. |
| **FR6** | How are single-home tools and ordering problems handled? | reorder the syllabus; or inline blocks plus links | **No reordering.** PCA (7.2/7.4 before 11.1) and the permutation null (7.2 before 11.1) link back to 0.7, and 13.3 and 15.4's "SVD from chapter 0" become links to 0.7 Part 2️⃣. Tools used in one place only get no 0.7 part: 13.1 builds diffusion forward noising itself; 16.9 gets an inline HVP block with a forward link to 16.10; LoRA gets an inline block at its first use (7.3) with a forward link to 15.1. |

## What this changes downstream

- **Spec #39:** a new "Foundations route" section replaces the November 3 amendment's "Foundations companion (FND01–FND06)" paragraph and table.
- **Foundations:** writing 0.7 becomes one implementation batch after the Spec rewrite.
- **Learner repairs (#10):** adding the inline blocks and links to host lessons is part of the per-lesson repairs.
- **Unblocked:** #235 (PRH chapter), #30 (study route) and #237 (case-study evidence tier) no longer wait on this ticket.
