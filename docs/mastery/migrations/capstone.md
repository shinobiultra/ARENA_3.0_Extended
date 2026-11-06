# Capstone test migration: current visible lesson APIs

Date: 2026-10-15. Scope: the legacy root test imported deleted `sprint.py` and
`sprint_tests.py`; the current canonical lesson defines the method in
`infrastructure/chapters/chapter10_capstone_research_sprint/master_10_1.py`,
its generated `solutions.py`, and the exercise/solution notebooks. The migration
uses those current APIs and the current section's immediate learner tests.
It does not restore the old prototype, add a tests-only adapter, or mark missing
contracts skipped/xfail.

## Old-to-current guarantees

| Legacy test/API | Current check and interpretation |
| --- | --- |
| `split_examples` | Current immediate contract checks seeded stratification, exhaustive/disjoint indices and changed-seed behavior. Both real digit choices retain unique row identities and at least 40 independent confirmation pairs. |
| `budget_counts` / registered readout ceiling | Restored as visible lightweight lesson plumbing. `budget_counts` freezes intervention names/draw counts and expected DEV/confirmation pair counts. `evaluate_registered` counts actual rows passed to the head, refuses overspend before a stage, refuses repeated stages, and requires DEV before confirmation. A separate counting head verifies accounting against actual calls. |
| `mean_direction` | `mean_difference_direction`, with current normalization and class-mean oracle. |
| `rank_coordinates`, coordinate/direction patching | Current learner tests check the taught ranking and exact interchange behavior, including input preservation and direction scaling. Additional assembly coverage retains every same-size random baseline and all 16 draws. |
| `select_on_development` | Current tests preserve registered-candidate membership, deterministic tie order and rejection of missing/unregistered scores. Integration selects on DEV before accessing confirmation interventions. |
| `paired_group_interval` | `paired_bootstrap` receives one paired method-minus-control observation per independent digit pair. Current tests verify the seeded bootstrap recipe, pairing and sign reversal. |
| `bounded_pair_interval` for all-zero outcomes | The primary absolute swap-rate interval is now `clopper_pearson(0, 44)`, with upper limit about 0.0804 rather than an exact-zero population interval. The former grouped Hoeffding bound of about 0.205 is not preserved numerically because the current binary-pair estimand and interval are different. The uncertainty guarantee is preserved and checked against SciPy's exact binomial interval. |
| `train_digit_model` | Current `test_train_model` verifies both MLP and linear training, decreasing loss, trivial-task accuracy, reproducibility, eval mode and input preservation. The root suite now invokes it. |
| `register_plan` / `verify_registration` | Restored in the live lesson: the receipt binds a deep-copied JSON plan, exact split assignments, data/label bytes, array shapes/dtypes and original dataset row IDs with SHA256. Tests reject nested plan tampering, data/label mutation, row-identity changes, valid-but-reassigned split indices, modified receipt budgets and overlapping splits. Both live DEV/confirmation entry points verify the receipt. |
| Two model/task choices and a valid negative result | Current visible functions compose two bounded CPU runs: MLP on 3/8 and linear on 1/7. Both rule out the registered single-unit threshold while retaining positive/negative instrument checks, registered selection and the actual readout ceiling. Methodological acceptance stays distinct from scientific support. |
| Legacy 88 directional score rows / pair IDs | Current `evaluate_patch` requires both directional swaps and returns one binary observation per independent digit pair. The canonical 3/8 confirmation analysis uses 44 pairs; the two directions are not counted as 88 independent trials. Row-disjoint pairing is checked directly. |
| Unexpected successful shuffled-label control | The MLP regression explicitly retains shuffled-direction swap rate 1.0 and top-unit swap rate 0.0. The anomaly is not removed to make the specificity claim pass. |
| Visible notebook answers and teaching markers | Current notebooks retain visible definitions plus immediate tests for all core methods, pre-registration, negative results, shuffled-label discussion, exact intervals, write-up and editable `PLAY_` cells. Solution code implements the method visibly and does not import deleted sprint modules. |

The receipt and ledger detect accidental changes in this local teaching workflow;
they are not an externally timestamped, tamper-proof registry. Training and initial
instrument checks precede the lesson's registration, as the lesson explicitly
states. The intervention readout budget covers only the registered DEV and
confirmation sweeps; it excludes fitting, accuracy checks, initial training-pair
instrument checks and labelled post-hoc/play analyses. No guarantee of universal
compute accounting is claimed.

## Historical evidence and fresh execution

Before changing any lesson code, both current notebooks were copied losslessly as
compressed historical snapshots under
`historical/capstone-before-20261015/`. Its `manifest.json` records original byte
lengths and SHA256 hashes. These snapshots preserve previous outputs as historical
evidence; they are not evidence for the changed code.

The generator first cleared outputs because code changed. Fresh full solution
notebook execution then completed in **6 seconds** on CPU, with **46 executed code
cells** and **10 generated figures**. The registered intervention budget was
**45,936 head rows**: `(43 DEV + 44 confirmation pairs) × 88 intervention draws ×
6 head calls per pair`. The root integration tests independently count actual
head rows and require equality to this budget for both model/task choices.

Fresh canonical output retained:

- H1: 0/44 top-unit swaps; exact 95% interval [0, 0.080]; **ruled out** versus 0.9.
- H2: selected direction swaps 44/44, and the shuffled-label direction also swaps
  44/44; specificity **not supported**.
- H3: top-unit recovered fraction exceeds the same-size random-unit baseline by
  0.0284, with registered paired-bootstrap interval [0.0272, 0.0296].
- All registered methodology checks pass, including exact stage order and budget.

A subsequent expected-output prose correction regenerated only markdown; the
builder verified code-cell sources were identical before retaining this fresh
execution's outputs. Historical outputs were never carried under changed code.

## Validation

Executed with the existing offline environment, one Torch CPU thread, no GPU,
download, cluster job or research sweep:

```bash
MPLCONFIGDIR=/tmp/arena-mastery-mpl UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --offline --no-sync pytest -q tests/test_capstone_research_sprint.py
MPLCONFIGDIR=/tmp/arena-mastery-mpl UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --offline --no-sync python infrastructure/extension/build.py 10.1 --execute --timeout 90
```

Focused root tests: **27 passed**. The notebook's learner tests also passed during
fresh execution. The first full-notebook attempt could not open the local Jupyter
IPC socket in the sandbox; a tool-approved local execution succeeded. This was an
environment restriction, not a lesson-code failure.

Original ARENA chapters 0–4 were not edited by this migration. This is a bounded
regression repair and current-lesson execution check, not acceptance of all ARENA
extension sections or an independent scientific discovery.
