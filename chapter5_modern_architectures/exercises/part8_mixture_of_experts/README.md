# [5.8] Mixture of Experts from Scratch — candidate under repair

This twelve-exercise lab implements Switch-style top-1 probability gates,
normalized top-k gates, per-sequence left-to-right capacity, differentiable
sparse dispatch, experts, router losses, cached decoding, probes, and exhaustive
interventions. The complete experiment is visible in small stages and uses
three independently trained seeds with paired aux-by-z conditions.

The current CPU result is a useful falsification, **not an accepted positive
expert-specialization result**. It remains outside the released course config.
The original positive release predicate and its thresholds are retained; strict
CUDA hooks still reject a run that fails that predicate. No CUDA run or accepted
report has been produced for this candidate.

Current three-seed means after the routing/capacity correction:

- Fully regularized final accuracy: 0.9942; dense: 1.0000.
- No regularizers: 0.9968 accuracy despite zero hard-load entropy and 0.625 drops.
- With both regularizers: 0.9994 hard-load entropy but 0.1815 drops. Global balance
  does not remove per-sequence capacity overflow.
- Context-only router probe: 0.3403 versus 0.3333 shuffled labels.
- Preferred ablation drop: 0.0000 versus 0.00023 for nonpreferred experts.
- Preferred/nonpreferred forcing drops: 0.0352/0.2888. Forcing damage does not
  establish normal-use necessity; accuracy also misses sub-argmax logit changes.
- Longer-stream accuracy: 0.6567 versus dense 0.6646.
- Sparse/dense padded FFN slots: 1,620/1,440. Equal width is not equal work.
  The router additionally evaluates 3,060 positions (440,640 matmul FLOPs).

The notebooks explicitly evaluate the failed specialization hypothesis and keep
its failure visible. Do not recover the old positive claim by reverting to
noncausal affinity-ranked overflow, surrogate unit gates, or relaxed bounds.

`build_notebooks.py` now owns the complete narrative, generates every solution
dropdown from the current implementation, validates Python syntax, and breaks
the former giant experiment cell into five explained stages. Its paired outputs
are `5.8_Mixture_of_Experts_from_Scratch_{exercises,solutions}.ipynb` and the
chapter instruction page. After rebuilding, execute a learner copy completed
from the twelve answers; stored output is deliberately cleared during builds.

The exact hand oracles and function tests precede the trained result. The play
cell varies capacity and regularizers and shows route and intervention tables.
The synthetic task, biased initialization, only three seeds, unmatched padded
work, and chunk-dependent finite capacity remain explicit limitations. Frontier
MoE checkpoint inspection and an accepted real-model path remain unfinished.
