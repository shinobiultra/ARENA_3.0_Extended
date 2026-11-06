# Writing an ARENA extension section

Every extension section is written the way the original ARENA sections are: as **one master file**, from which
the exercises notebook, solutions notebook, `solutions.py` and Streamlit page are generated. The reference
implementation is section 7.1:

- master: `infrastructure/chapters/chapter7_activation_to_language/master_7_1.py`
- section dir: `chapter7_activation_to_language/exercises/part1_lenses_patchscopes/` (`tests.py`, `utils.py`, generated files, `media/`)

Read the 7.1 master before writing anything. Also skim one original master, e.g.
`infrastructure/chapters/chapter1_transformer_interp/master_1_2.py`, to absorb the voice.

The syllabus (`infrastructure/extension/syllabus.yaml`) says which section you're writing, its title and directory,
which prototype material to mine (`sources`), which roadmap requirements it must teach, and a brief.

## What a section must feel like

The bar is the original ARENA notebooks: 100-250 cells, roughly 8k-25k words of prose, a guided path where the
student builds the method themselves, sees a convincing result, and understands it. Concretely:

1. **Title, header image, `# Introduction`.** Open with a concrete example or question (a prompt and the model's
   answer, an image, a puzzle), not a definition. Say why the topic matters and what the student will have built and
   seen by the end. Mention prerequisites (which earlier ARENA sections).
2. **`## Content & Learning Objectives`.** One `### 1️⃣ ...` subsection per part, each with a paragraph and a
   `> ##### Learning Objectives` block (exact syntax, see 7.1). 3-5 parts plus `☆ Bonus`.
3. **`## Setup code`.** The Colab cell (copy from 7.1, change the chapter and pip installs) and the main setup cell
   (imports, `device`, path setup, `import partN_xxx.tests as tests`, `import partN_xxx.utils as utils`, `MAIN = __name__ == "__main__"`).
4. **Numbered parts** (`# 1️⃣ Title` as a markdown cell on its own). Each part:
   - teaches the concept the next exercise needs ("here is the mental model, here is the equation, here is how the
     tensor shapes map onto it, now implement it"); equations in LaTeX;
   - has exercises (see format below) that build up to a result;
   - produces at least one **visible result** (plot, table, generated text/images) generated live in the notebook;
   - interprets that result in a `Help - how should I interpret this?` dropdown, written from the *actual* output;
   - ends with a short `## Summary of N️⃣`.
5. **Controls next to every headline result**: random/shuffled baselines, norm-matched random vectors, wrong-layer /
   wrong-position / text-only conditions, held-out data. Plot the control on the same axes as the result.
6. **At least one `## Try it yourself` play cell**, where the student changes an input (prompt, layer, image,
   hyperparameter) and sees what happens. Give an example output.
7. **`# ☆ Bonus`**: extensions, anomaly hunts, open questions, and a further-reading list with links.
8. **Honesty.** Results are what they are. If a method fails, a control doesn't fail, or something is unexplained,
   say so in the text and turn it into a question for the student (7.1 does this with the tuned lens on capitals and
   the Patchscope target-layer-2 anomaly). Never write expected outputs or interpretations you haven't checked against
   the executed notebook.

### Exercise format

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

## Prediction learning

When writing or revising a lesson, add its method-specific case to
`mastery_hub/extension_checkpoints.json` and rebuild the guided workbook with
`scripts/build_extension_prediction_workbook.py`. Read the
[integration design](../../docs/mastery/designs/2026-10-15-integrated-prediction-learning.md)
when authoring a case or a blind challenge variant.

Each case needs a specific forecast before the result, confidence and causal reasoning,
the question-defining inputs, a controlled experiment using the student's visible method,
observed controls, a discrepancy explanation, and a forecast for a changed scientific
condition. Match the question to the actual method and metric; a generic "predict what
happens" paragraph is insufficient. Keep canonical sources intact and linked with provenance.
Original ARENA chapters 0–4 remain unchanged; any prediction companions for them are additive.

The standard notebooks deliberately show reference outputs. Their workbook route is
**guided practice**: seen or unknown exposure stays explicit and earns no unseen score.
Blind variants must omit the target output and answer-bearing interpretation until commitment;
an expected-output dropdown or prefilled HTML image still exposes the answer. Preserve
immediate semantic tests and nonleaking worked examples. Scientific data role, learner exposure,
evidence class and execution kind are separate fields. Repeatedly monitored evaluation data
cannot become untouched TEST by changing its label. A new seed alone is replication, not transfer.

Before calling the case engineering-checked, verify its source/reference paths and configuration
fingerprints, immutable commitment and stale-input rejection, observation provenance, portable
record round-trip, and changed-case instructions. Learner acceptance remains a separate gate.

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
4. Keep total execution time reasonable: ideally under 10 minutes on a GPU, and runnable on CPU with documented
   smaller settings where possible. Long trainings should show progress (tqdm).

## Environment notes (important)

- Python 3.14, torch 2.12 (CUDA available), transformers 5.x, TransformerLens 2.16.1, SAELens 6, nnsight, diffusers.
- **`model.to_str_tokens(...)` is broken** in this environment (TransformerLens 2.16 + transformers 5 decode the whole
  sequence as one string). Use `[model.to_string(tok) for tok in tokens[0]]` instead.
- `cache.accumulated_resid(apply_ln=True)` normalises by the final layer's scale (the DLA convention), not per-layer.
- Tensors created under `t.inference_mode()` can't be saved for backward; use `t.no_grad()` for anything that feeds
  a training graph.
- The GPU is shared with other jobs: assume **at most ~6 GB of VRAM** is free. Prefer small models (GPT-2 small,
  Pythia-70m/160m, Gemma-3-270m, SmolLM2-360M, gemma-3-1b, siglip-base). Many are cached in `~/.cache/huggingface`;
  downloading others is fine if they're small. If you need something bigger, use bf16, batch carefully, and free memory
  (`del model; t.cuda.empty_cache()`) between parts.
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
- **Don't touch**: original ARENA chapters 0-4 (except the extension sections 0.6 and 1.6), other sections, shared
  config, `syllabus.yaml`, `arena_ext/`, `tests/` at the repo root, docs. If the section moved to a new directory,
  leave the old directory alone (it'll be removed centrally). Report the one-line description for the chapter page
  in your final message instead of editing config.
