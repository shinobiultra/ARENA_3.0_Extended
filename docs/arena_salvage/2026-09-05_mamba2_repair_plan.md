# Mamba-2 learner and precision repair

Current learner failure: eight solution dropdowns cannot be parsed, several
class stubs omit constructors and helpers, and the CUDA comparison mixes
precision settings without recording them. The original ARENA 1.1 notebook
and page were inspected as the comparator: each small component must be
usable in the next exercise, with visible intermediate results.

Target section claim: recurrent, quadratic, and chunked SSD compute the same
operator; recurrent memory solves delayed recall beyond the convolution window.

Signature result to create: preserve and re-execute the live operator matrix,
parity, recall-control, and scaling panels; show released-model comparison with
explicit precision and real token predictions.

Student-implemented functions: segment sums, grouped parameters, semiseparable
matrix, recurrence, quadratic and chunked schedules, block and cache, recall
training, scaling, stacked LM, checkpoint mapping and parity.

Controls/baselines: exact recurrence, official Listing 1, state ablation,
local convolution, shuffled training labels, longer sequences, full-vs-cache,
and official CUDA kernels with a documented arithmetic setting.

Play cells: change chunk size, state width, decay, and sequence length and
compare equivalence, memory use, and task behavior.

Files to modify: section 5.7 builder, notebook/page surfaces, solutions, tests,
artifact lock, expected outputs, verification report, and status/roadmap rows.

Tests to run: parse all dropdowns; execute the completed learner notebook in
a clean kernel; exact output/state/gradient/cache tests; CUDA precision
diagnostics and pinned-checkpoint parity without raising tolerances; inspect
figures; regenerate the bound report and run preservation and surface audits.

Release status stays prototype until scientific and editorial review passes.
