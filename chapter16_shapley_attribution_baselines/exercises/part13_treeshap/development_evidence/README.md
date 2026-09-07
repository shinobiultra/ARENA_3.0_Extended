# TreeSHAP development evidence

These archives preserve intermediate runs and failures. They do not identify
the current published learner; use `../expected_outputs/learner_execution.json`
for that boundary. Each archive has a read-back-verified manifest.

- `native_dev_v1`: 18 models fit, then the NumPy scalar-threshold rounding bug
  failed forest additivity on two explanation rows. The failure was retained.
- `native_dev_v2`: fresh corrected study passed in 12.68 seconds, including all
  native models, both games, exact subset checks, backgrounds and deletion.
- `learner_v1`: own-answer execution failed because an aggregate diagnostic
  requested Saabas before the student had reached that exercise.
- `learner_v2`: all native work ran, but duplicate embedded figures and source
  changes correctly prevented acceptance.
- `learner_v3` and `reproduction_v3`: complete scientific execution and exact
  fresh reproduction passed before the final pedagogical review.
- `learner_v4` and `reproduction_v4`: 87-cell revised lesson and fresh comparison
  passed; browser review found one performance-plot note covering data, so this
  version was kept as development evidence.

Later numbered development archives, when present, retain final layout
iterations. Timing changes are descriptive. No failed or stale-source run is
silently relabeled as current publication evidence.
