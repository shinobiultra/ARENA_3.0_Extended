# [16.7] Data Shapley in One Training Run

Which training images helped a model learn—and does that tell us which images to
remove? Start with exact subset retraining, then distinguish that game from
contribution accounting along one nonlinear SGD trajectory.

The notebook contains twelve exercise groups and 26 student-implemented functions.
It develops exact and permutation Data Shapley, checkpoint-gradient scores,
influence and leave-one-out baselines, affine ghost gradient products, exact
minibatch games, Taylor allocations, and a complete training accountant. Every
exercise has immediate tests and visible help and solutions.

Nine figures are generated from the learner's implementations: an exact-retraining
comparison, gradient-similarity check, Taylor-error sweep, cumulative accounting,
per-image credit history, actual digit examples, held-out deletion controls,
measured runtime costs, and a retraining play experiment. The real digit study
uses three fixed seeds, 128 training images, 48 ranking images and 64 untouched
test images. All 49 neural training runs take 128 SGD steps. The exact local
accountant telescopes; the first-order approximation overallocates, and removing
low-credit examples worsens test loss for one seed. These failures are part of
the lesson.

Open the exercise notebook in the project Python environment. No GPU or gated
model is required. The data loader reuses the pinned public UCI digit data and
splits from 16.9; its existing data-access instructions apply. The first run may
fetch those small public data files if they are absent from the local cache.

To reconstruct the published solutions from the exercise notebook's own solution
dropdowns and execute everything into a new directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python3 chapter16_shapley_attribution_baselines/exercises/part7_data_shapley_in_one_training_run/build_inrun_notebooks.py \
  --output /tmp/arena-inrun-primary
```

Use a second new directory for a separate complete execution, then compare both:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python3 chapter16_shapley_attribution_baselines/exercises/part7_data_shapley_in_one_training_run/build_inrun_notebooks.py \
  --output /tmp/arena-inrun-fresh
.venv/bin/python3 chapter16_shapley_attribution_baselines/exercises/part7_data_shapley_in_one_training_run/inrun_report.py \
  --primary /tmp/arena-inrun-primary --fresh /tmp/arena-inrun-fresh \
  --output /tmp/arena-inrun-comparison
.venv/bin/python3 -m pytest -q tests/test_inrun_data_shapley.py tests/test_data_shapley.py
```

The comparison retains every deterministic numeric field, including all fitted
weights, loss trajectories, per-step credits and deletion choices. It excludes
only 23 explicitly named wall-clock measurements; both measured timing sets
remain in the raw payloads. The report assembler fits no models. Old CUDA
one-step preflight evidence is historical, not certification of this learner.

Ghost products here assume independent examples and separate affine parameters.
The tiny explicit Hessian illustrates second-order allocation; it is not the
paper's scalable second-order implementation or a GhostSuite package-parity
claim. The CPU timing experiment does not establish foundation-model throughput.
