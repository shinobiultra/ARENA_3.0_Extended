# MoE candidate review

Current learner failure: the implementation was partially corrected to ordinary
Switch top-1 probability gates and causal left-to-right per-sequence capacity,
but the notebooks still teach straight-through unit gates and affinity-ranked
overflow. Their test calls and dropdowns are stale. The fresh three-seed CPU
experiment fails the previous preferred-expert ablation claim: preferred
ablation drop is zero. The section remains outside the released config.

The roadmap's architecture matrix, paired notebooks, page, implementation,
tests, and builder were inspected. Original ARENA 1.1's derive/implement/test/
compose progression supplies the comparator.

Target section claim: implement exact sparse routing and show that task accuracy,
healthy router use, context-only decodability, and causal expert dependence are
different properties. Keep the failed ablation hypothesis visible; do not
retroactively relax its acceptance bound or promote this run as a positive
specialization result. A subsequent training redesign must have a new declared
claim and controls before its results are interpreted.

Signature result to create: live load/overflow and aux-by-z comparisons plus
all-expert ablation/forcing tables, with the old specialization hypothesis
explicitly evaluated and rejected if its preregistered inequality still fails.

Student-implemented functions: router, causal capacity ledger, differentiable
dispatch/combine, expert bank, router losses, sparse layer, decoder and cache,
task objective, context-only probe, coherent expert permutation, and exhaustive
interventions. Split the giant supplied experiment cell into readable stages.

Controls/baselines: exact routing/gradient/capacity oracles, identical cloned
initializations for the paired aux/z design, aligned dense baseline, independent
seeds, held-out streams, shuffled probe labels, all nonpreferred interventions,
and coherent expert permutation. Preserve the longer-stream failure result.

Play cells: change capacity and loss coefficients, then inspect the ledger and
intervention table rather than task accuracy alone.

Files to modify: candidate 5.8 builder, notebooks/page, tests, reference notes,
and evidence metadata as required. No course registration or positive report
until the scientific and learner contracts actually pass.

Tests to run: all semantic CPU tests, all-dropdown parsing and source parity,
clean-kernel learner completion, live CPU signature/controls, visual inspection.
Strict CUDA follows a resolved scientific claim and review.
