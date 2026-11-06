# Local mastery records, schema 1

This CPU pilot stores `analytic_toy` evidence. A 2-D quadratic is not a transformer
loss landscape or a real-model training run. Local self-study is not tamper-proof
exam software; editing files manually can falsify personal progress.

## Public interfaces

`arena_ext.mastery.challenges` exposes these pure transitions. Each returns a new
JSON-native dictionary; callers retain earlier dictionaries as immutable history.

- `new_attempt(challenge_id, config_fingerprint, *, exposed_cases=())`
- `commit_prediction(attempt, prediction, confidence, reasoning)`
- `invalidate(attempt, new_config_fingerprint)`
- `reveal(attempt, config_fingerprint, observed)`
- `expose_case(exposed_cases, challenge_id, config_fingerprint)`
- `validate_attempt(attempt)` and `brier(p, y)`

States are `uncommitted`, `committed`, `revealed`, and `invalidated`. Reveal requires
a committed prediction and the exact original configuration fingerprint. A changed
configuration invalidates a pending attempt; start a new attempt for those inputs.
Completed attempts remain unchanged. Repeating reveal with the same fingerprint
and observation returns the same attempt, including its ID and single score.
A conflicting repeated observation is rejected.

Attempt dictionaries have exactly these fields:

| Field | Meaning |
| --- | --- |
| `attempt_id` | Generated UUID hex, unchanged across transitions |
| `challenge_id` | Safe local challenge identifier |
| `config_fingerprint` | Lowercase SHA-256 digest of full scientific config |
| `status` | One of the four states above |
| `assessment` | `unseen` or `practice`, fixed when the attempt starts |
| `prediction` | Finite JSON value, null before commit |
| `confidence` | Finite probability in `[0,1]`, null before commit |
| `reasoning` | Learner explanation, required at commit |
| `observed` | Finite JSON result, null before reveal |
| `score` | Binary Brier score, or null for unscored tasks |
| `score_type` | `brier` or `unscored` |

For a boolean prediction, confidence means probability of the selected answer.
The positive-event probability is `confidence` for `True`, and `1-confidence` for
`False`. Boolean observations receive `(p-y)**2`. Categorical, ranking and numeric
answers remain unscored; their values are retained without inventing a binary
metric. A single correct guess does not award a mastery badge.

`exposed_cases` is a list of objects with exactly `challenge_id` and
`config_fingerprint`. Record exposure whenever exploration displays the answer,
when a challenge reveals it, and when an imported result exposes it. A previously
exposed pair starts as practice, including after reload. The application must keep
these exposures alongside completed history. The declared `HELD_OUT_TEMPLATES`
separate interpolation from extrapolation through curvature, initial direction,
noise regime, or an explicitly changed recurrence in a future lesson; these are
question templates, not an assertion of independent scientific evaluation.

`arena_ext.mastery.records` exposes:

- `fingerprint(config)` → digest of sorted, compact UTF-8 JSON, with finite values.
- `validate_record(record)` → validated independent copy.
- `loads_attempt(bytes_or_text)` and `load_attempt(path)` → strict validated record.
- `export_attempt(record, destination_directory)` → generated local JSON path.
- `replay(record)` → a fresh reference `QuadraticTrace` from the recorded seed/config.
- `empty_session()`, `validate_session(session)`, `save_session(session, directory=None)`,
  `load_session(directory=None)`, and `default_data_dir()`.

## Portable record envelope

Portable envelopes require `attempt.status == "revealed"`. Uncommitted, committed
and invalidated attempts belong only in `session.current_attempt`; validation,
export, import and replay reject them as portable completed evidence. All fields
below are required; unknown fields are rejected. Example construction:

```python
from dataclasses import asdict
from arena_ext.mastery.quadratic import SIMULATOR_VERSION
from arena_ext.mastery.records import fingerprint, validate_record

scientific_config = asdict(config)
scientific_config['x0'] = list(scientific_config['x0'])
scientific_config['simulator_version'] = SIMULATOR_VERSION
# Create and transition attempt using fingerprint(scientific_config).
record = validate_record({
    'schema_version': 1,
    'unit_id': 'u02_optimization',
    'challenge_id': attempt['challenge_id'],
    'attempt_id': attempt['attempt_id'],
    'config': scientific_config,
    'config_fingerprint': fingerprint(scientific_config),
    'source_revision': 'local-analytic-recurrence-v1',
    'code_revision': code_revision,  # Actual revision, with dirty status if applicable.
    'seeds': [config.seed],
    'evidence_class': 'analytic_toy',
    'attempt': attempt,
    'control_outcomes': control_outcomes,  # Actual measured control outputs.
    'next_action': 'Predict an unseen curvature regime',
    'reflection': '',
})
```

`config` has exactly `lambda_min`, `lambda_max`, `angle_deg`, `x0`, `eta`, `beta`,
`noise_std`, `seed`, `steps`, and `simulator_version`. `x0` is a JSON list containing
two finite coordinates. Numerical validation delegates to `QuadraticConfig`.
The fingerprint covers every scientific field, including the simulator version;
source/code provenance is stored explicitly in the envelope. `seeds` must equal
the single config seed. Duplicate identity/fingerprint fields must agree with the
nested attempt. Text fields are strings; revisions must be nonempty.
`control_outcomes` is a finite JSON object; its contents describe actual controls.

Only `analytic_toy` is accepted in this pilot. Future `recorded_real_run` and
`live_real_run` require their own validated evidence contracts. A mismatched
simulator version can be inspected as a record but `replay` refuses it explicitly.
Replay recreates the analytic toy and does not establish fresh training evidence.
Same-environment tests compare seeded noisy states and losses at absolute `1e-12`
tolerance; portable JSON does not promise bitwise cross-platform linear algebra.

## Private session and reload

A session has exactly these fields:

```json
{
  "schema_version": 1,
  "config": null,
  "current_attempt": null,
  "history": [],
  "exposed_cases": [],
  "next_action": "",
  "mode": "Explore",
  "reflections": {}
}
```

A nonnull `config` is the complete scientific configuration above.
`current_attempt` is a validated attempt whose fingerprint matches that config.
`history` contains completed record envelopes only, with unique attempt IDs.
A revealed `current_attempt` must exactly match the attempt in its completed
history envelope; saving reveal therefore also saves the immutable history entry.
A later save cannot rewrite that same revealed attempt ID.
`exposed_cases` includes every historical revealed pair and any free exploration.
`mode` is `Explore` or `Challenge`; restoring Challenge prevents a reload from
silently exposing an uncommitted answer through Explore mode. `reflections` maps
completed attempt IDs to editable text so reflection does not rewrite historical
predictions. Explicit exports may merge a reflection into a new export copy.

Saving an existing session retains every completed envelope unchanged and every
previous case exposure. Once committed, the pending prediction, confidence,
reasoning, challenge, fingerprint and attempt ID are fixed. A same-config rerun
cannot delete that commitment, replace it with a new ID, or return it to an
uncommitted state. It may reveal the fixed answer, or explicitly persist that
same attempt as invalidated before starting a same-config retry with a new ID.
Changing the scientific config may directly replace the old commitment with a
new attempt ID; that change invalidates the old question. Switching into Explore
may change `unseen` to `practice` only with a retained case exposure; `practice`
never becomes `unseen`. Widget reruns can update mode, reflection and next action
without changing the committed answer. Baseline reset
retains completed history and exposure. Multiple competing processes are not a
supported concurrent editing workflow; stale saves are rejected when they would
remove historical records or exposure. There is a 1 MiB bound for the entire
session as well as each portable record; overflow is an explicit error.

Default storage is `ARENA_MASTERY_DATA_DIR` when explicitly set, otherwise
`$XDG_DATA_HOME/arena-mastery`, otherwise `~/.local/share/arena-mastery`.
Directories must be absolute. Merely resolving a path or loading an absent
session creates nothing. Tests pass temporary directories and never write the
user's home. A corrupt session raises an explicit error instead of silently
resetting progress.

## File safety and privacy

Import accepts UTF-8 JSON only, up to 1 MiB, with known schema versions, required
fields, finite values and no duplicate object keys. Pickles, Python expressions,
NaN/Infinity, unknown evidence classes and extra fields are rejected. Imported
strings are data and are never evaluated. Identifiers contain only ASCII letters,
digits, underscores or hyphens, begin with a letter/digit, and have at most 128
characters. Slashes, dots and path traversal are rejected.

Export requires an explicit action and a chosen directory. Its filename is
`<challenge_id>-<attempt_id>.json`; record content cannot specify an output path.
The API writes a mode-0600 temporary file, flushes/fsyncs it, then atomically
publishes without overwriting an existing export, including a racing writer.
Private `session.json` updates use atomic replacement. Interrupted writes remove
temporary files and preserve previously published sessions. Direct symlink
record paths and destination directories are rejected. Exporting the same attempt
twice to the same directory raises `FileExistsError`; choose a different directory
for an intentionally revised export.

No function uploads, sends telemetry, opens a network connection, downloads data,
or launches GPU work. Private notes remain local unless the learner explicitly
exports or shares them. Learners should avoid entering credentials in free text;
strict JSON validation cannot determine whether arbitrary prose contains a secret.
The app may offer explicit JSON and Markdown downloads after validation; those
are user-requested local exports and include only the displayed selected record.
