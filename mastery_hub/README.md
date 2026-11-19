# ARENA Mastery Hub

A local companion to the original ARENA course and
[Stanford Deep Learning Alchemy](https://deep-learning-alchemy.github.io/).
Use the study reader to choose among all 86 configured extension lessons, open
their explanations and download a working notebook. The separate optimizer lab
explores when curvature, step size, momentum and gradient noise turn convergence
into oscillation or divergence.

The 26 new CPU study editions fill the former lesson-file gaps, leaving zero missing
configured lessons. They teach visible methods and controlled small organisms; their
native/model bridges and learner acceptance remain pending. Existing historical reference
outputs keep their original evidence scope.

## Launch

For a new checkout without an environment, follow the [separate study installation](../docs/mastery/install_study.md).
It does not install CUDA or change the existing scientific environment.

From the repository root, use the existing environment:

```bash
uv run --no-sync --offline streamlit run mastery_hub/read.py --server.address=127.0.0.1 --server.port=18505 --server.headless=true --browser.gatherUsageStats=false
```

Open [the study reader](http://127.0.0.1:18505/). Reading needs no account, API key,
model download or GPU. The reader shows the selected notebook's exact save
directory, local Jupyter instructions and a bookmarkable lesson/part URL.
Start with [16.1 Exact Shapley](http://127.0.0.1:18505/?section=16.1&part=1)
for CPU mathematical exercises, or choose a model-based branch in the
[study guide](STUDY-GUIDE.md). Each lesson keeps its own execution requirements.

If uv's cache is not writable, set `UV_CACHE_DIR=/tmp/arena-mastery-uv`.
The direct equivalent uses `.venv/bin/python -m streamlit run` with the same
file and options. To work on a downloaded notebook, start Jupyter from the
repository root using `uv run --no-sync --offline jupyter lab`; keep your personal
copy in the lesson directory displayed by the reader. Check that the selected
kernel uses this repository's `.venv/bin/python` before running setup cells.

## Optional optimizer lab

Launch the existing offline CPU lab separately:

```bash
uv run --no-sync --offline streamlit run mastery_hub/app.py --server.address=127.0.0.1 --server.port=18506 --server.headless=true --browser.gatherUsageStats=false
```

This is the analytic optimizer companion, rather than the course reader. Its
learner walkthrough remains deferred; the steps below describe that lab.

## A first optimizer session

1. Open the worked example. Verify the first update `(1,1) → (.9,0)` and loss `.405`.
2. Explore one setting at a time. Inspect a step, compare baseline overlays, and
   read the table as well as the plots. Try zero learning rate, a noiseless optimum,
   an unstable learning rate and gradient noise.
3. Switch to Challenge, select an unseen transfer regime, and commit your answer,
   confidence and explanation before revealing. A setting change invalidates the
   pending prediction. Previously exposed dynamics remain practice even if only
   the seed, rotation, starting point or step horizon changes.
4. Implement the core method in [the exercises notebook](units/u02_optimization/exercises.ipynb).
   Run each immediate test and compare with the expandable hints and solutions.
   The [executed solutions](units/u02_optimization/solutions.ipynb) include the figures.
5. Explain a failed assumption, save one next action, then reload and replay.
   Export JSON and Markdown only when you want a portable copy.

The plotted experiment uses unnormalized heavy-ball momentum and seeded Gaussian
gradient noise. The challenge oracle concerns strict **deterministic** stability
in every eigendirection. A finite successful simulation is not proof of asymptotic
convergence; a noisy run need not converge to zero. This is an **analytic toy**,
not fresh language-model training or a transformer loss landscape.

## Lessons in the reader

`mastery_hub/read.py` serves each extension lesson with its exercises notebook and
executed reference. Sections 0.6, 1.6 and 16.3 also link their earlier notebooks.

The [original-materials shelf](../docs/mastery/original_materials.md) preserves
canonical upstream links and provenance. Original ARENA chapters 0–4 are unchanged.

## Private progress

The default store is `$XDG_DATA_HOME/arena-mastery/session.json`, or
`~/.local/share/arena-mastery/session.json`. It is outside the checkout. Override
with an absolute `ARENA_MASTERY_DATA_DIR`, for example a temporary directory for
a demo. Records are local; the app performs no automatic upload or analytics.
Keep one active local writer for a store. Completed history and case exposures
survive resets. Session files and imports have a 1 MiB limit.

Import accepts validated JSON only; it does not run code. Explicit local exports
refuse overwriting an existing file. An unsupported simulator version can be
inspected as a record but cannot silently replay. Private reflections are separate
from immutable prediction history. See [the record schema](../docs/mastery/record_schema.md).
The app is a self-study tool, not tamper-proof exam software.

## Build and verification

```bash
uv run --no-sync --offline python scripts/build_mastery_notebooks.py --check
uv run --no-sync --offline python scripts/build_mastery_notebooks.py --execute
uv run --no-sync --offline pytest -q tests/mastery
uv run --no-sync --offline python scripts/audit_original_arena_preservation.py
uv run --no-sync --offline python scripts/build_merged_config.py --check
```

Fresh notebook kernels require permission to open local sockets. The generator
writes only this pilot's two notebooks; `--check` is read-only and preserves saved
execution outputs. Keep original ARENA unchanged.

Only the optimizer pilot is implemented as an interactive Alchemy companion. The learner walkthrough (Q001) remains a deferred release
gate. The [implementation record](../docs/mastery/implementation_record.md) lists
the unfinished extension work and later milestones. Published upstream material
does not make its companion ready; no recurring study schedule is created here.

Start with the [study guide](STUDY-GUIDE.md) to choose one question and follow the exact exercise, reference and reading links across all 86 covered lessons. It includes CPU/model route guidance, the pilot reading notes and the 26 new CPU study editions with native evidence still pending.
