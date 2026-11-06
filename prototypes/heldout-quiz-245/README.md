# Prototype: a held-out prediction quiz at the end of 12.1 (issue 245)

**Throwaway branch `prototype/heldout-quiz-245`, never merged.** It answers one question: what does a CS 312-style held-out
prediction quiz look like inside an ARENA lesson?

**Plan, in one line:** mock the quiz chapter at the end of 12.1 *CLIP and SigLIP from Scratch* as a page section and as an
executed notebook. It has four questions on the tiny CLIP: one Easy, two Medium and one Hard.

## Files

| File | What it is |
| --- | --- |
| [`12.1_heldout_quiz.md`](12.1_heldout_quiz.md) | The quiz chapter as it would appear on the 12.1 page. Read this first; it renders on GitHub, dropdowns included. |
| [`12.1_heldout_quiz.ipynb`](12.1_heldout_quiz.ipynb) | The same chapter as a notebook, executed with an example learner's answers so that the score table and reveals show. |
| [`quiz_12_1.py`](quiz_12_1.py) | The key, the run table and a local scorer (about 130 lines). |
| [`build_notebook.py`](build_notebook.py) | Builds the notebook from the markdown. |
| `media/quiz_reveal.png` | The reveal figure. |

## Where the numbers come from

Nothing new was run. The four questions are:

- **Q1** uses fixed logits, and its reveal runs the learner's own `zero_shot_probabilities`. This question is the old blind CLIP
  companion (`mastery_hub/units/clip_prediction`) rewritten as one question. It uses the same logits and the same metric.
- **Q2–Q4** reveal numbers copied from 12.1 as it is today:
  - the Part 2 expected output (batch 64, learning rate 5e-4, one seed);
  - the Bonus section "Batch size, temperature and the learning rate", which reports the lesson author's exploratory runs:
    batch 16 with three seeds, batch 32 with one seed, and learning rates 2e-3 and 5e-3 with two seeds each.

Those Bonus runs are not code in the course today, and the Bonus doesn't say which learning rate the batch-size sweep used. The
final design closes this gap (HQ3): the quiz's key runs become the last cell of the chapter.

## Decisions (open until the user answers)

| ID | Decision | Recommendation |
| --- | --- | --- |
| HQ1 | Where the quiz sits | Its own `=== NEW CHAPTER ===` page, "🎯 Held-out Quiz", after the last numbered part and before ☆ Bonus, with its own entry in Content & Learning Objectives. Taste-bar item 15 then reads "the core lesson ends with the quiz", and the Bonus still follows it. |
| HQ2 | Format, answer types, scoring | 4–6 questions and 20–30 minutes, with at least one Easy and one Hard. Each question has the CS 312 shape: *From the lesson*, *This question changes*, a diff against a call in the lesson, then the question. Answers are five exhaustive intervals by default, a ranking with `=` inside a stated ε when three or more runs are compared, or a one-word choice. Each answer gets one sentence of reasoning that names a result from the lesson. Scoring: full points for the right interval and half for a neighbouring one; for rankings, one point per correctly ordered pair. |
| HQ3 | Where results come from, and how they stay honest | Key runs are code: the last cell of the quiz chapter. Every fresh-kernel run regenerates them. That cell ends with a check that the written key still matches what the runs produced, so drift fails the completion check. Reveals quote those saved outputs. Bin edges sit clear of the seed spread. Cheap or analytic questions reveal live (like Q1). There is no sealing: a reveal is as open as an ARENA Solution dropdown. Key runs add at most about 15 minutes on the laptop GPU. Numbers that would spoil the quiz move out of the Bonus. |
| HQ4 | How answers are recorded without a server | One code cell, `my_answers = {...}` plus `my_reasons`, saved in the learner's own notebook, then `quiz.score(my_answers)` prints a table locally. Nothing is stored or sent. The scorer is written once (shared, about 60 lines). Each lesson declares only its key. No widgets. |
| HQ5 | How difficulty is tagged | **Easy / Medium / Hard** in words in each question heading. Easy is something you built, used on its design case. Medium is one knob turned beyond the range you ran. Hard is an unseen regime, or a number rather than a direction. 🔴 dots stay for coding effort. The quiz gets one "You should spend up to N minutes" line. |
| HQ6 | What the debrief looks like | Each Reveal dropdown holds the real numbers (seeds shown), the answer, a "mental model" paragraph naming the lesson result that predicts it, and an "if you chose X" line. Then come one figure of all key runs with a "Help - how should I read this figure?" dropdown and a short "What this quiz tested" list. Reproducing a result is optional: one "Try it yourself - reproduce one result" cell for the cheapest variant. |
| HQ7 | The old prediction layer: the workbook, the blind companions, #63, #67–#73 | Retire the 437-cell `extension_predictions` workbook. Delete the blind companions, with git history as the record. Each companion's question becomes a seed question for its host lesson's quiz: CLIP to 12.1 (done here as Q1), readout-gain circuit to 8.8, graph/path to 8.7, fixed-forward SAE to 6.1, SSD state-carry to 5.4, Data Shapley to 16.9, SAGE to 16.4. #63 and #67–#73 lose their receipt and exposure acceptance criteria. The Spec #39 rewrite (#250) recuts them as "write the quiz for lesson X" slices. |
| HQ8 | How the Alchemy companions (#32–#34) reuse it | Same chapter shape and same scorer. A companion's quiz asks about held-out variants of the companion's *own* runs in its own modality. CS 312's quizzes are linked as further practice and never copied. u02's commit/fingerprint machinery is replaced by this format when #32 designs the companions. |

## How to look at it

Open `12.1_heldout_quiz.md` on GitHub, or open the notebook in VS Code or Jupyter. To rebuild the notebook:

```bash
uv run --no-project --with nbformat --with nbclient --with ipykernel --with numpy --with matplotlib python build_notebook.py
```
