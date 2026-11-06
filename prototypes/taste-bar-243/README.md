# Taste bar prototype (throwaway)

Prototype for the ticket "What taste bar must every extension lesson meet to feel like original ARENA?" (shinobiultra/ARENA_3.0_Extended issue 243). Never merged; the decisions live in that ticket's resolution comment.

- `taste-bar-prototype.html`: open in a browser. 12.1's first screen beside 11.1's, today and rewritten three ways (A numbers first, B "Today we'll", C picture first), a draft checklist and seven decision cards. The user accepted all seven recommendations on 2026-11-06 (opener rule with A as default).
- `taste_metrics.py`, `metrics.csv`: per-lesson text-matching counts over the 86 syllabus lessons, read from the extension checkout on 2026-11-06. Run: `uv run --no-project --with pyyaml python -I taste_metrics.py <extension checkout> metrics.csv`.
