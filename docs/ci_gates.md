# CI Gates

CI runs cheap checks that need no model downloads. Lessons themselves are
checked by the completion check (shinobiultra/ARENA_3.0_Extended issue 249),
outside CI.

## Hosted CPU

Runs on every push and pull request:

```bash
.venv/bin/python scripts/ci_quality_gate.py --mode hosted-cpu
```

It checks whitespace, that original ARENA chapters 0-4 are byte-identical to the
pinned base (`scripts/audit_original_arena_preservation.py`), the merged config,
the course surface (chapter scaffold, config entries, page shape), artifact
hygiene (no raw weights or archives, nothing over 5 MB, prompt data contracts,
`.gitignore`), unreadable long lines in chapter files, and the batch runner's
exit codes.

## Study reader

```bash
.venv/bin/python scripts/ci_quality_gate.py --mode study-delivery
```

Checks the mastery-hub reader and prediction notebooks without running lesson code.

## Lessons: the completion check

A lesson is finished after a fresh-kernel run, its focused tests, the
preservation audit above and the user's walkthrough:

```bash
uv run python infrastructure/extension/build.py 7.1 --execute    # fresh-kernel run
uv run python infrastructure/extension/focused_tests.py 7.1      # focused tests, CPU
```

Single lessons run on the laptop GPU; batches run on the Julia worker
(`infrastructure/extension/cluster/`) at normal QOS. Fresh-kernel runs stay out of CI.
