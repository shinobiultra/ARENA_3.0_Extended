# Writing an ARENA extension section

This is the authoring standard for every lesson in chapters 5–19, the extension pages of chapters 0 and 1
(0.6, 0.7, 1.6) and the Alchemy companions. It has two halves: the **Taste bar**, which says what a finished lesson
feels like (it is the course's standard, taken word for word from the spec), and the mechanics of building one.
`AGENTS.md` points here. Whatever the two halves leave open, original ARENA (chapters 0–4) is the reference.

Every extension section is written the way the original ARENA sections are: as **one master file**, from which
the exercises notebook, solutions notebook, `solutions.py` and Streamlit page are generated. The reference
implementation is section 7.1:

- master: `infrastructure/chapters/chapter7_activation_to_language/master_7_1.py`
- section dir: `chapter7_activation_to_language/exercises/part1_lenses_patchscopes/` (`tests.py`, `utils.py`, generated files, `media/`)

Read the 7.1 master before writing anything, and skim one original master, e.g.
`infrastructure/chapters/chapter1_transformer_interp/master_1_2.py`, to absorb the voice. The Taste bar's references
are original 0.1, 1.2, 1.4.1 and 1.5.4; extension 12.1, 12.3, 9.1, 8.3, 7.2 and 8.4; companion u02.

The syllabus (`infrastructure/extension/syllabus.yaml`) says which section you're writing, its title and directory,
which prototype material to mine (`sources`), which roadmap requirements it must teach, and a brief. A batch works
only from the spec, its batch issue and the lesson plans it points to; a part, claim or requirement line no plan names
is not added by the author, it becomes a finding at the walk.

## Preservation

- Chapters 0–4 belong to upstream ARENA. The fork never edits upstream's files, only adds its own, and upstream's files match the *upstream pin* exactly (`6a46100c` since the first sync in Batch 0, merged 2026-11-07, `f9f034bd` before; publication path, #37). Upstream's config entries go in unchanged, with the extension's after them; `requirements-original.txt` is upstream's requirements file, refreshed at each sync.
- Extension pages (chapters 5–19, 0.6, 0.7, 1.6) are freely editable. Git history is the preservation; stale duplicates are deleted.
- CS 312 materials, blog-post repositories and paper code are linked at pinned revisions, never copied or vendored. CS 312 states no licence.

## The Taste bar

From "What taste bar must every extension lesson meet to feel like original ARENA?" (#243), item 15 from the quiz (#245). It applies to every extension lesson and companion; "original" means ARENA chapters 0–4. References: original 0.1, 1.2, 1.4.1, 1.5.4; extension 12.1, 12.3, 9.1, 8.3, 7.2, 8.4; companion u02. The repository's authoring file carries these words, and `AGENTS.md` points to it.

**The first screen: title, introduction, objectives, first exercise**

1. The title names only what the lesson builds. If a part uses a stand-in (a toy encoder instead of CLIP, hand-built checkpoints instead of training), the title says so or that part moves to the Bonus. A lesson that builds its named method on a toy may keep the name, and its first paragraph says it's a toy.
2. The introduction opens with something concrete from this lesson: a prompt, a picture or a result the learner will reproduce, with real numbers where the lesson has them.
3. It then says which model or toy we study and why it matters. It gives an "In this section you will:" list that names the signature result and its controls, and lists prerequisites with section links.
4. Content & Learning Objectives: every part has one plain paragraph, then two to five objectives that start with a verb (implement, understand, compare, measure). Jargon waits until it is taught.
5. Setup says in a sentence or two what hardware and time the lesson needs, when that matters.
6. Nothing is there for maintainers: no banners, provenance, receipts, "historical", "migration", status or acceptance language, requirement or question IDs, `GT_TIER`, or paths into `docs/`.

**Every part**

7. Intuition comes before formulas. The first time a lesson uses a tool, it gets a picture, a toy example or a plain-words explanation before its equation, or a one-line link to the place in the course that builds it.
8. Every exercise has the block with Difficulty 🔴, Importance 🔵 and "You should spend up to N minutes on this exercise.". Then come a short task, a test called with the function, a hint where learners are likely to stumble, and a Solution dropdown.
9. Predictions are asked where the surprise is, about the plot or number that comes next: "**Make a prediction first:**" plus one specific question. The reading of the result follows in a "Help - how should I interpret this?" dropdown. There are no generic forecast or reflect forms.
10. Results come with controls that should fail, on the same axes, named in plain words.
11. Each part ends with a short summary.

**The whole lesson**

12. It has one signature result: the figure or table that is the lesson's payoff, built by the learner.
13. It has at least one place to play: a "Try it yourself" cell where the learner changes an input and sees what happens.
14. It has a Bonus with open questions, limitations and further reading. The limits of what the lesson shows live here, written as teaching.
15. The core lesson ends with the 🎯 Held-out Quiz (see "Prediction quizzes"). The Bonus follows it.
16. It is deep enough: every part has its intuition, exercises, a result and a control. There is no word target.
17. The voice uses "you" and "we'll", is plain and friendly, and explains why as well as what. Results that didn't work are reported as results.

**Openers.** The default is a concrete prompt plus two or three of the lesson's own surprising numbers, then "In this section you will:". Build-from-scratch lessons may use a short goal plus a route list, as 0.1 and 1.4.1 do. One real figure may open a lesson only if it poses the question without spoiling the first prediction. **Not a taste problem:** `=== NEW CHAPTER ===` markers (the build's page splitter) and the dark-mode line stay. Colab links leave the extension (#37 P5); learners run locally.

## Intuition at first use

- **Block (FR1, FR3).** At a shared tool's first use, an intuition block of at most ~150 words plus one picture or a few toy numbers comes before the formula: what the tool keeps or measures, and its main trap, ending with a "Go deeper" link into 0.7. Every lesson gets the block or a one-line link at first use; the full block is required where the course first uses the tool and wherever it carries the signature result. Quoted numbers come from real runs. The foundations prototype's first-use table lists where each block goes.

## Held-out Quiz

The quiz is part of the Taste bar (item 15), so it is part of every lesson. The decisions below are the spec's, word
for word.

From "What does an Alchemy-style held-out prediction quiz look like inside an ARENA lesson?" (#245, HQ1–HQ8); the 12.1 prototype on `prototype/heldout-quiz-245` is the model.

- **Placement (HQ1).** Its own `=== NEW CHAPTER ===` page, "🎯 Held-out Quiz", before ☆ Bonus, with its own objectives entry.
- **Format (HQ2).** 4–6 questions, 20–30 minutes, at least one Easy and one Hard. Each has the CS 312 shape: *From the lesson*, *This question changes*, a diff against a call the learner ran, then the question. Answers are five exhaustive intervals by default, a ranking (`=` within a stated ε) when three or more runs are compared, or sometimes one word; no free numbers. Each answer gets one sentence of reasoning naming a lesson result. Right interval: full points; neighbour: half; rankings: one point per correctly ordered pair.
- **Key runs (HQ3)** are the quiz chapter's last cell, so every fresh-kernel run regenerates them, and the cell ends by checking the written key: a drifting key fails the completion check. Reveals quote saved outputs; interval edges sit clear of seed spread; cheap or analytic questions reveal live. Nothing is sealed: a reveal is as open as a Solution dropdown. Key runs add at most ~15 minutes. Numbers that would spoil the quiz move out of the Bonus.
- **Answers and difficulty (HQ4, HQ5).** One cell holds `my_answers` and `my_reasons`; the shared `quiz.score(my_answers)` (~60 lines) prints a table locally; nothing is stored or sent; each lesson declares only its key; no widgets. Easy: something you built, on its designed case. Medium: one setting beyond the range you ran. Hard: a regime you never saw, or a number rather than a direction. The quiz gets one time line.
- **Debrief (HQ6).** Each Reveal gives the real numbers (seeds shown), the answer, a "mental model" paragraph naming the lesson result that predicts it, and an "if you chose X" line; then one figure of all the quiz's runs with a "Help - how should I read this figure?" dropdown, a "What this quiz tested" list, and one optional "Try it yourself - reproduce one result" cell.
- **Old prediction layer (HQ7).** The `extension_predictions` workbook and blind companions retire; their questions seed host quizzes: CLIP → 12.1 (in the prototype), readout-gain circuit → 8.8, graph/path → 8.7, fixed-forward SAE → 6.1, SSD state-carry → 5.4, Data Shapley → 16.9, SAGE → 16.4.
- **Companions (HQ8)** use the same layout and scorer on variants of their own runs; CS 312's quizzes are linked, never copied. There are no prediction records.

### Declaring a quiz key

The scorer is `infrastructure/extension/learner/quiz.py`. A lesson declares only its key; the scorer prints the table.
The quiz chapter's setup puts that folder on the path (the lesson's setup cell already finds `root_dir`) and the key
cell reads like this:

```python
sys.path.append(str(root_dir / "infrastructure" / "extension" / "learner"))
from quiz import Interval, Quiz, Ranking, Word

quiz = Quiz({
    "Q1": Word("falls", points=1, choices=("falls", "rises", "unchanged")),   # one word
    "Q2": Interval("c", points=2),                                            # five intervals, a to e
    "Q3": Ranking.from_values(RUNS, eps=0.02),                                # RUNS = {"A": 0.96, "B": 0.97, ...}
})
```

and the learner's cell is `my_answers = {"Q1": "falls", "Q2": "d", "Q3": "D < C < A = B"}` followed by
`quiz.score(my_answers)`. The interval edges and the ranking come from the key runs (the quiz chapter's last cell),
which check the written key before the notebook finishes. `tests/fixtures/quiz_example.py` is a complete example;
`tests/test_quiz_scorer.py` pins down the scoring rules (full credit, half credit for a neighbour, one point per
correctly ordered pair, ties within ε, malformed answers score zero with a reason).

## Retired machinery: never

Earlier agents drifted into verification-shaped lessons. These were retired on purpose (completion check #249, taste
cleanup #244, quizzes #245). Don't build or restore them, and keep every trace of them off learner pages:

- **Receipts** and **hash ledgers**: no run receipts, no `artifacts.lock.yml`, no `verification_report.json`, no
  content hashes recorded to prove a lesson was run or reviewed.
- **Custody**: no `GT_TIER`/`EXERCISE_ID` constants, no `run_gpu_test`/`run_full_experiment` freeze checks, no state
  kept outside the repository to prove who ran what.
- **Sealed targets**: no hidden answers, commitment records or "blind" companions. A quiz reveal is as open as a
  Solution dropdown.
- **Independent reviews** and **acceptance packets**: no review bundles, reviewer sign-offs or evidence archives.
  The completion check and the user's walk are the acceptance; nothing else is recorded.
- **Status or provenance on learner pages**: no banners, "historical", "migration" or acceptance language, requirement
  or question IDs, or paths into `docs/` (Taste bar item 6).
- **Prediction records**: no `extension_predictions` workbook, no confidence or reasoning forms, no stored forecasts.
  Predictions are one specific question where the surprise is (item 9), and the quiz is scored locally.

Verification stays light: a fresh-kernel run, focused tests, the preservation audit and the walk.

## Section anatomy

Concretely, in the master file:

1. **Title, header image, `# Introduction`.** Taste bar items 1–3: a title that names only what's built, then a
   concrete opener from this lesson and the "In this section you will:" list. Mention prerequisites (which earlier
   sections, linked).
2. **`## Content & Learning Objectives`.** One `### 1️⃣ ...` subsection per part, each with a paragraph and a
   `> ##### Learning Objectives` block (exact syntax, see 7.1). Parts, then the 🎯 Held-out Quiz entry, then `☆ Bonus`.
3. **`## Setup code`.** The main setup cell (imports, `device`, path setup, `import partN_xxx.tests as tests`,
   `import partN_xxx.utils as utils`, `MAIN = __name__ == "__main__"`), plus a sentence on the hardware and time the
   lesson needs. There is no Colab cell: learners run locally.
4. **Numbered parts** (`# 1️⃣ Title` as a markdown cell on its own). Each part teaches the concept the next exercise
   needs (intuition first, then the equation and how the tensor shapes map onto it), builds up through exercises to a
   visible result generated live in the notebook, interprets that result in a `Help - how should I interpret this?`
   dropdown written from the *actual* output, and ends with a short `## Summary of N️⃣`.
5. **Controls next to every headline result**: random/shuffled baselines, norm-matched random vectors, wrong-layer /
   wrong-position / text-only conditions, held-out data, plotted on the same axes as the result.
6. **`## Try it yourself`**, where the student changes an input and sees what happens, with an example output.
7. **The 🎯 Held-out Quiz** on its own `=== NEW CHAPTER ===` page, then **`# ☆ Bonus`** with open questions,
   limitations and further reading.
8. **Honesty.** Results are what they are. If a method fails, a control doesn't fail, or something is unexplained,
   say so in the text and turn it into a question for the student (7.1 does this with the tuned lens on capitals and
   the Patchscope target-layer-2 anomaly). Never write expected outputs or interpretations you haven't checked against
   the executed notebook.

## Exercise format

Copy the format from 7.1:

````
### Exercise - implement `foo`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Plain-English description of what `foo` takes and returns, shapes, edge cases.
````

then a code cell:

```python
def foo(x: Float[Tensor, "batch d"]) -> Float[Tensor, "batch"]:
    """Docstring."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    return x.sum(-1)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_foo(foo)
# END HIDE
```

then a markdown cell with `<details>` dropdowns: `Hint`, sometimes `Help - ...`, and `Expected output` where the
output is text. The pipeline adds the `Solution` dropdown automatically. Aim for 10-25 exercises per section. The
student must implement the core method (the main computation, metric, intervention, and control); utils may only
hold plumbing (data loading, plotting, model-download helpers).

Training loops, long experiments and anything that loads a model go in cells tagged `# ! TAGS: [main]`, so that
`import partN_xxx.solutions` (which the tests do) doesn't run them.

### Expected outputs

- **Text** (printed numbers, tables): put the real output, copied from the executed solutions notebook, in an
  `Expected output` dropdown, inside the `<pre style=...>` block used in 7.1.
- **Figures**: put a markdown cell tagged `# ! TAGS: [html]` *immediately after* the code cell that draws the figure,
  containing `<img src="media/NAME.png" width="...">`. When the section is built with `--execute`, the figure the cell
  produces is saved to `media/NAME.png` and pre-filled under the cell in the exercises notebook. The cell must produce
  exactly as many images as `media/` names it lists. Use matplotlib (or anything that renders `image/png`) for
  these figures; plotly figures don't produce PNGs here (no kaleido), so use plotly only for extra interactive plots.
- Every figure needs a title, axis labels, a legend when there's more than one series, and the control on the same axes.

### tests.py and utils.py

- `tests.py` follows the original ARENA style: `test_xxx(student_fn, ...)` functions that print
  `All tests in `test_xxx` passed!`. Prefer **independent oracles** (a library function, a brute-force computation,
  a hand-worked small example, a mathematical identity) over comparing against `solutions`; when you do compare with
  solutions, import them lazily inside the test (`import partN_xxx.solutions as solutions`). Test edge cases and give
  informative assertion messages. Tests should run in seconds.
- `utils.py` holds plotting and loading helpers with docstrings. No method logic.
- Don't import `arena_ext` (the old prototype library); it will be deleted. Copy what you need into the notebook (if
  it's the method) or `utils.py` (if it's plumbing), rewritten readably.


## Building and checking

```bash
uv run python infrastructure/extension/build.py 7.1            # generate files only
uv run python infrastructure/extension/build.py 7.1 --execute  # generate + run solutions notebook + save figures
```

(`.venv/bin/python` works in place of `uv run python`.) `--execute` runs the solutions notebook top to bottom in a
fresh kernel; on failure it saves `*_solutions.FAILED.ipynb` next to it with the partial outputs. A section is not
done until `--execute` passes. Without `--execute`, the build keeps the previous execution's outputs as long as the
code cells haven't changed, so markdown-only edits don't need re-execution.

To execute on the Würzburg cluster instead of the local GPU (while the worker job is running; needs the VPN):

```bash
infrastructure/extension/cluster/remote_build.sh 7.1   # sync sources, execute on an L40S, copy notebooks + media back
```

For text expected outputs, write a placeholder `<pre style=...>FILL_SOMETHING</pre>` after the code cell, and after
an executed build run `uv run python infrastructure/extension/fill_expected.py 7.1 --write` to copy the real printed
output in (then rebuild without `--execute`). Then:

1. Read every text output in the executed solutions notebook and make sure every number, table and claim in the
   markdown matches it. Rebuild after editing.
2. Look at every PNG in `media/` (use the Read tool on the image). Check it shows what the text says it shows.
3. Check the exercises notebook: stubs raise `NotImplementedError`, tests are called, solutions are hidden.
4. Keep the fresh-kernel run within 30 minutes of wall-clock time, quiz key runs included (they add at most about 15);
   a lesson over the limit is shrunk, not excused. Say in Setup what hardware and time the lesson needs. Long
   trainings should show progress (tqdm).

## Environment notes (important)

- The supported runtime is whatever `uv.lock` holds (Python, torch, CUDA, transformers, TransformerLens, SAELens, nnsight,
  diffusers); read it rather than trusting a version written here. The two TransformerLens notes below were written
  against 2.16 and are re-checked when the lock is upgraded.
- **`model.to_str_tokens(...)` is broken** in this environment (TransformerLens 2.16 + transformers 5 decode the whole
  sequence as one string). Use `[model.to_string(tok) for tok in tokens[0]]` instead.
- `cache.accumulated_resid(apply_ln=True)` normalises by the final layer's scale (the DLA convention), not per-layer.
- Tensors created under `t.inference_mode()` can't be saved for backward; use `t.no_grad()` for anything that feeds
  a training graph.
- A lesson needs at most 24 GB of VRAM (the reference machine's GPU) and 30 minutes; if it needs more, shrink it.
  Prefer small models (GPT-2 small, Pythia-70m/160m, Gemma-3-270m, SmolLM2-360M, gemma-3-1b, siglip-base). Models and
  datasets download at run time from the Hub with `revision=` pinned. If you need something bigger, use bf16, batch
  carefully, and free memory (`del model; t.cuda.empty_cache()`) between parts.
- Use `tqdm.auto`. Set seeds so the executed outputs are reproducible.
- Several sections are written in parallel. Keep scratch files in a subdirectory named after your section
  (e.g. `<scratchpad>/s7_2/`), never directly in a shared temp directory.

## Files you own

When writing section N.M:

- **Create/overwrite**: `infrastructure/chapters/<chapter>/master_N_M.py`, and in the section dir: `tests.py`,
  `utils.py`, optional small data files (< a few MB), plus everything the build generates.
- **Clean up**: if the section dir already existed and contains old prototype files (`artifacts.lock.yml`,
  `verification_report*.json`, `*.schema.json`, `expected_outputs/`, `build_notebooks.py`, `README.md`, `lesson.md`,
  `*_API.md`, `historical/`, `results/`, `reports/`, `scripts/`, sockets, other old `.py` modules), delete them so the
  directory only contains the generated notebooks, `solutions.py`, `tests.py`, `utils.py`, `media/` and data files.
- **Don't touch**: original ARENA chapters 0-4 (except the extension sections 0.6, 0.7 and 1.6), other sections, shared
  config, `syllabus.yaml`, `arena_ext/`, `tests/` at the repo root, docs. If the section moved to a new directory,
  leave the old directory alone (it'll be removed centrally). Report the one-line description for the chapter page
  in your final message instead of editing config.
