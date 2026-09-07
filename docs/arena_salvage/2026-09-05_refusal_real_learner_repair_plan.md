# Put real refusal-direction experiments in the 9.1 learner

## Current learner failure

Inspected both 37-cell 9.1 notebooks, the instruction page and section reference
code, alongside the original 104-cell ARENA 1.3.2 Function Vectors and Model
Steering notebook, particularly its multi-token generation, caching, padding
and student-written intervention exercises (cells 53, 66, 76 and 89).

The present learner mainly generates a six-layer planted activation fixture.
The real-model method lives in `solutions.py`; its outputs are read from a
saved report. Most solution dropdowns describe an algorithm without executable
answers. The play cell maps a prompt to a toy row rather than generating text.
There are also substantive evidence problems:

- Classification thresholds are recomputed from evaluation class means.
- The claimed label shuffle reverses balanced labels; it is an anti-direction,
  not a random permutation null. Its fitting also includes evaluation examples.
- One random direction cannot characterize the null distribution.
- The real hook edits one prompt position once. It is not the paper's
  all-position addition or all-layer directional ablation during generation.
- First-token controls can be identical chat-template tokens. A degenerate
  direction needs an explicit undefined marker, not invented zero accuracy.
- A 16-token substring rubric is called paper-equivalent behavioral evidence;
  it does not establish harmfulness or broad capability preservation.

No original ARENA file will change. This repairs an existing priority pilot;
no additional paper, topic or model family is introduced.

## Target section claim

Students extract and evaluate real Qwen2.5-0.5B-Instruct residual directions,
then intervene during autoregressive generation. They distinguish correlational
separation, induced refusal on benign prompts, direction removal, and capability
side effects, with genuine held-out tests and multiple matched nulls. The claim
will follow measured results, including failed candidates and classifier errors.

## Signature result to create

A layer-by-post-instruction-position separation heatmap with null bands;
actual benign baseline/steered generations; addition and projection curves;
a distribution of every random/permuted control; and an independently scored
benign capability table. The learner generates these from its own methods.
The planted example becomes a short numerical oracle, not the headline result.

## Student-implemented functions

Fit a difference vector, unit direction and midpoint threshold on training
activations; apply a frozen threshold and compute pairwise AUROC; refit true
permutation controls; collect residuals at unambiguous layer/position sites;
compute centered SVD and compare it with the mean direction; implement scoped
addition and projection hooks; generate with cache-aware persistent hooks;
score refusal markers and expose their limitations; aggregate paired effects
and select/freeze candidates using development data only. Provide working
answers, immediate semantic tests, expected output, help and interpretation.

## Data and frozen choices

Retain the existing pinned model revision
`7ae557604adf67be50417f59c2c2f167def9a775` and public dataset revision
`b6ed3432f1d4a695e13be1c373bf7fb5af43f376`. Inspect duplicate/overlap and token
lengths before fixing splits. Start with class-wise source rows 0:32 for fitting,
32:64 for development, and 64:100 for final evaluation (36 per class). The
previous report used only the first 32 rows per class; its old evaluation must
not be presented as a new untouched test. Preserve source row IDs and hashes.
Any deduplication or split changes must be documented before final access.

Initially investigate layers 4, 8, 12, 16, 20 and the last three chat-template
suffix positions. Evaluate candidates on development only; write an explicit
selection rule and frozen candidate record before final generation. Do not
search for a passing final result. Pin generation length and decoding settings
before final evaluation. Report every tested amplitude, not only its optimum.

## Controls and baselines

Unedited generation; zero-strength identity and post-hook cleanup parity;
32 fixed random directions and 32 training-label permutations for separation;
at least eight of each fixed control for causal evaluation; matched addition
norms; a first-template-position negative control with undefined cases visible;
and capability scores on fixed benign questions. Retain unexpectedly effective
nulls and absolute outcomes. Refusal-marker absence is not successful answering
or safety evidence. Show readable benign generations; store sensitive benchmark
rows as IDs, hashes and aggregate labels, continuing the existing artifact scope.

## Play cells

Change a benign prompt and addition strength and inspect generated words.
Choose a layer/position to compare true and shuffled projection distributions.
Compare cached and uncached intervention behavior on a short benign example.

## Method connection and scope

The primary reference is [Arditi et al.](https://arxiv.org/html/2406.11717v3),
sections 2.3–2.5, and [their code](https://github.com/andyrdt/refusal_direction).
It distinguishes single-layer all-position addition from all-layer/all-position
ablation. This course uses a smaller existing model and public course dataset,
with a shorter declared generation budget. It is a method study, not an exact
replication of the paper's model suite or safety evaluations. No permanent
model-weight modification is needed for this learner.

## Files to modify

The 9.1 paired notebooks, instruction page, lesson source/builder/executor,
section-local method/data/test modules, plots and expected evidence; later its
README, lock/report and registry descriptions once the actual learner passes.
Leave the active 8.3 and 12.3 source sets untouched while their kernels run.

## Tests to run

Analytic direction/threshold/AUROC oracles including tied scores and shifted
evaluation distributions; permutation fitting restricted to training; exact
projection geometry and gradients; independent tiny-Qwen native forward and
cached-generation parity; padding and residual-site checks; hook removal on
exceptions and preservation of model state; fixed split/hash checks; complete
fresh own-answer learner execution; inspect actual figures; then independent
CUDA evidence, source/report contracts and original-ARENA preservation.
