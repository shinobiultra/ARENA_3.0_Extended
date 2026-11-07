# Original ARENA Preservation Contract

The extension is append-only with respect to the original ARENA course.
Original ARENA is pinned at `6a46100c4f219b2764d3a30fe501e810ee500d86` (upstream ARENA-education/ARENA_materials main,
first catch-up under #37; `docs/artifact_registry.lock.yml`). Each upstream catch-up moves this pin.

`scripts/audit_original_arena_preservation.py` checks, on every CI run:

- every file the pinned commit ships under `chapter0_fundamentals/` …
  `chapter4_alignment_science/`, and under their master folders
  `infrastructure/chapters/chapter0_fundamentals/` … `chapter4_alignment_science/`,
  still exists with identical bytes. New files beside them are fine: the 0.6
  and 1.6 extension lessons live in chapters 0 and 1;
- `infrastructure/core/config.yaml` keeps every original entry unchanged (new
  chapters and sections may be added), `config_original.yaml` equals the
  original config, and `config.yaml` equals `config_original.yaml` merged with
  `config_extension.yaml`;
- `requirements-original.txt` equals the original `requirements.txt`.

Extension lessons (chapters 5-18, 0.6 and 1.6) are freely editable; git
history is their preservation. Until 2026-11-06 this audit also enforced a strict
allowlist of changed paths; that allowlist was retired with the move to
lightweight verification (shinobiultra/ARENA_3.0_Extended issue 249).
