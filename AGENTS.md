# AGENTS.md — ARENA 3.0 Extended

The extension's authoring standard lives in one place: [infrastructure/extension/AUTHORING.md](infrastructure/extension/AUTHORING.md).
Read it before writing or changing any lesson or Alchemy companion. It carries the **Taste bar** (what every extension
lesson must feel like), the Held-out Quiz rules and the quiz scorer, the master-file mechanics, and the list of
machinery that was retired and must not come back.

## Sacred rule

Chapters 0–4 are upstream ARENA and belong to upstream. Never edit their files, only add your own. The preservation
audit (`scripts/audit_original_arena_preservation.py`) checks them against the upstream pin. Extension pages
(chapters 5–19, 0.6, 0.7, 1.6) are freely editable.

## Finishing a task

Report what changed, which checks you ran (fresh-kernel run, focused tests, preservation audit) and what you left
undone. Work you find outside your task's scope becomes a finding, not an edit.
