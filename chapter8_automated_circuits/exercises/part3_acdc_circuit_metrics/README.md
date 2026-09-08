# [8.3] ACDC and Circuit Metrics

Build an automated circuit search whose retained edges carry the current
hybrid computation. Begin with an exact nonlinear circuit, then implement the
full GPT-2 additive residual graph, sequential ACDC, EAP and input-interpolated
EAP-IG. Inspect real prompts, named edges, threshold curves and causal controls.

The paired notebooks and instruction page are generated from `lesson.md`.
Core methods appear as student exercises with immediate independent tests,
expected outputs, help and complete answer dropdowns. Loading, checkpoint
storage and figure layout are supplied plumbing. Original ARENA is unchanged.

The bounded experiments deliberately retain their different outcomes:

- Greater-than: development freezes ACDC 78, EAP 300, task EAP-IG 100 and
  KL EAP-IG 300 edges. Changed-wording recovery is about 0.832, 0.895, 0.859
  and 0.909. ACDC drops below the development floor; it is not retuned.
- IOI: all four fresh thresholds fail at least one fixed development criterion.
  The holdout stays unopened. The 737-edge candidate is inspected on development
  because it has the highest recovery, with its accuracy failure still explicit.
- Each inspected circuit has all 32 uniform and 32 receiver-matched random nulls.
  Every retained ACDC edge is deleted separately, and sampled full-versus-circuit
  subset removals expose limitations of minimality and completeness claims.

Play cells change year endings, inspect the 100-edge graph, remove named edges,
compare sparse/full/random interventions and inspect actual next-token names.
All-name and repeated-subject IOI corruption also give visibly different
attribution curves. Author-code parity supports the EAP-IG implementation;
these small datasets do not reproduce the papers' complete benchmarks.

Allow two to three study days. The complete fresh GPU searches and diagnostics
take about three hours on the reference machine. CUDA and up to the course's
24 GiB budget are required for the full path; the explicit numerical-only
scope skips real experiments and never counts as full validation.

```bash
uv run python chapter8_automated_circuits/exercises/part3_acdc_circuit_metrics/build_notebooks.py --check
uv run python chapter8_automated_circuits/exercises/part3_acdc_circuit_metrics/build_notebooks.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -u chapter8_automated_circuits/exercises/part3_acdc_circuit_metrics/execute_learner.py --scope full --output-directory /tmp/arena-acdc-new-full-run --publish
```

Choose a new output directory for each run. Interactive checkpoints bind the
actual method bodies, model revision and data; full validation always performs
fresh searches. `solutions.run_gpu_test()` checks full learner provenance and
remeasures five circuits and all 320 random nulls on CUDA.

**Publication status:** the full 94-cell learner completed all 44 code cells,
five owned ACDC searches, fourteen figures and all play experiments with
unchanged sources. The successful run took 5,884.11 seconds after an explicitly
recorded continuation of its own interrupted checkpoint chain; it is not an
uninterrupted new search. All 56 section tests pass. Separate fresh CUDA
verification remeasures five circuits and all 320 nulls with zero difference
from the learner's recorded behavior. Peak learner allocation is 9.43 GiB;
the separate recheck uses 3.81 GiB. `expected_outputs/own_full_release_v2`
archives the exact sources, execution and checkpoint chain. Earlier independent
and partial candidates remain separately labeled in `edge_candidates`.

An interrupted full learner can continue its own checkpoint chain with
`execute_learner.py --scope full --output-directory NEW_DIRECTORY --publish
--resume-interrupted-run PREVIOUS_DIRECTORY`. The previous execution must be an
unchanged own-answer full run ending in a kernel interruption. Its lesson and
method sources, actual function ASTs, data/model context and checkpoint hashes
must match. The new kernel reruns the learner and all diagnostics, rechecks
checkpoint losses live, and continues unfinished search decisions. Completed
searches from that same interrupted run retain their full traces. Candidate
archives and failed method assertions cannot be substituted. The execution
manifest records the continuation and exact infrastructure changes explicitly.
