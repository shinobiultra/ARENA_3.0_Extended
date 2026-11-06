# Exact Shapley on Ground-Truth Games

Open `16.1_Exact_Shapley_on_Ground-Truth_Games_exercises.ipynb` to build the exact
finite-game method. Its completed companion retains the historical reference
outputs and figure. They are **not a fresh execution** of this canonical notebook.
The authoring source is `infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_1.py`.

The CPU lesson implements the full coalition table, dividend game, weighted
marginals, analytic oracle, permutation parity, efficiency/symmetry/dummy controls,
leave-one-out comparison and interaction sweep. Local `tests.py` uses hand-worked
finite-game answers; the taught calculations are visible in the notebook.

The signature has three panels: credit against the dividend oracle and
leave-one-out; surplus versus interaction strength with the additive control; and
predecessor-context marginals. The original CPU play changes
`PLAY_INTERACTION_SCALE` and `PLAY_PLAYER`, keeping the oracle check active.
Forecast/reflection prompts ask for confidence, causal reasoning and exposure.
Saved answers make this guided practice rather than unseen mastery.

`verification_report.json`, its schema, `artifacts.lock.yml` and `expected_outputs/`
are byte-identical historical supporting receipts. They retain their original
source paths and acceptance fields; those fields do not approve this migration.
The optional CUDA wrapper delegates to the unchanged old section's separate
neural-game implementation. It is not called by the CPU lesson and was not run
during migration.

All old lesson files, instruction assets and the old page remain physically
unchanged. Migration scope, hashes, exact adaptations and open acceptance gates
are recorded in `docs/mastery/migrations/exact_shapley.md` and the adjacent evidence
directory named `exact_shapley_master_migration`.
