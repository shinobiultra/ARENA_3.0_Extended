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

## Prediction practice in existing extension lessons

The [guided workbook](units/extension_predictions/exercises.ipynb) covers 86
current master-backed extension lessons. Each lesson also links to its checkpoint.
Prepare and test your student method, freeze its exact inputs and source, commit a
forecast with confidence and reasoning, then run the frozen experiment. Record
the actual controls, explain any mismatch, and predict a changed scientific case.
These public reference-rich lessons remain guided practice; completing a record
does not establish unseen mastery or independent scientific evaluation.

Sections 0.6, 1.6 and 16.3 offer separate blank Practice copies with direct links
to the intact original notebooks. Saved original outputs remain historical;
blank exercises and preserved outputs serve different points in the lesson.

For a local forecast before the pilot lessons, open [7.1 blank public Practice](units/pilot_practice/7_1_exercises.ipynb) or [9.1 blank public Practice](units/pilot_practice/9_1_exercises.ipynb). Each copy preserves the original method, controls and play, links the intact references, and adds manual reflection after controls. These copies do not add blind companions or checkpoints; published text/hints still expose results. See [Practice instructions](units/pilot_practice/README.md).

Workbook observations are explicitly learner reported. Its separate
`arena-extension-guided-practice` JSON format restores the worksheet, without
rerunning the experiment, and is not accepted by the optimizer app importer.
Transfers remain forecasts until separately run. Intentional stubs stop Run All;
the workbook launches no models or training. Keep private exports outside Git.

The additive [blind circuit challenge](units/circuit_prediction/exercises.ipynb)
asks how changing readout gain affects circuit attribution. Implement the visible
compiler, patching, EAP, integrated gradients and AUC methods; pass the separate
practice tests; freeze inputs and commit your forecast before running. The reveal
compares paired AUC bars, exact-effect heatmaps, task behavior and a noise-free
control. No target answers or saved result plots are prefilled. Run the notebook
from this checkout on CPU; no model download is needed. Its private JSON receipt
has its own schema and replay does not run a model. Prior exposure remains an
explicit learner declaration, never an automatic claim of unseen mastery.

The [blind Data Shapley challenge](units/shapley_prediction/exercises.ipynb)
asks how coalition-training normalization changes one example's credit. Implement
gradient descent, validation utility, exact Shapley values and marginal contributions;
pass independent practice fixtures and commit before the eight-example experiment.
Paired credit bars and a marginal-contribution plot expose the difference, with
duplicate-example, label-relocation, zero-learning-rate and efficiency controls.
Play with the stored results, then save a reflection and transfer forecast. The
notebook ships without target answers. Its separate private receipt also retains
failed measurements, and replay performs no training.

The [blind SAGE challenge](units/sage_prediction/exercises.ipynb) asks how
conditional information and marginal model use allocate global credit. Implement
the restricted predictor, population loss game, exact Shapley allocation and
marginal-contribution summaries; commit before comparing the two allocations.
The reveal shows credit bars, coalition-risk curves and controls. A separately
committed covariance change tests transfer. Its notebook contains no target
answers or saved outputs; strict private receipts support replay without recomputation.

The [blind SSD challenge](units/ssd_prediction/exercises.ipynb) asks how weakening
state carry changes a signed final readout. Implement the recurrent, quadratic
and chunked forms, pass distinct practice fixtures, and commit before seeing
state heatmaps and carry-effect curves. A separately committed decay change tests
transfer. Scalar-oracle, chunk-size, causal and linearity controls check the three
equivalent computations. The notebook ships blank; private receipts support replay
without rerunning methods. This fixed filter does not establish trained-model
behavior, CUDA performance or unseen learner mastery.

The fifth blank companion, [8.7 graph/path prediction](units/graph_prediction/exercises.ipynb),
asks how graph attribution and path recovery respond to a fixed public synthetic setup.
It is separate from the 8.8 readout-gain circuit challenge. Implement the visible
methods, pass the distinct DEV mathematics fixtures, and commit a forecast before
an explicitly authorized CPU target experiment. Source and DEV checks passed;
the target remains unexecuted, with no empirical result or prefilled answer.
Its own strict private receipt retains measurements and failures, supports exclusive
file export, and replays stored observations without running methods. The notebook
does not automatically train, download or run the target. Public setup and DEV
success do not establish unseen mastery or independent scientific evaluation.

The sixth blank companion, [6.1 fixed-forward SAE prediction](units/sae_prediction/exercises.ipynb),
asks how four fixed forward rules change codes and reconstructions under the same
inspectable weights. Implement seven visible methods, pass distinct DEV fixtures,
and commit a signed forecast before any separately authorized CPU target attempt.
Source and DEV checks passed; the public target remains unexecuted, with no target
result or prefilled answer. Code/residual heatmaps and row MSE/L0 retain zero-code
and mismatched-decoder controls, alongside a separate known-failing DEV alignment
panel. Private receipts retain partial failures and replay stored observations
without running methods. The notebook does not automatically run, train or download
models/data; fixed-rule mechanics do not rank learned SAEs or establish unseen mastery.

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
uv run --no-sync --offline python scripts/build_extension_prediction_workbook.py --check
uv run --no-sync --offline python scripts/build_circuit_prediction_notebook.py --check
uv run --no-sync --offline python scripts/build_shapley_prediction_notebook.py --check
uv run --no-sync --offline python scripts/build_sage_prediction_notebook.py --check
uv run --no-sync --offline python scripts/build_ssd_prediction_notebook.py --check
uv run --no-sync --offline python scripts/build_graph_prediction_notebook.py --check
uv run --no-sync --offline python scripts/build_sae_prediction_notebook.py --check
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
