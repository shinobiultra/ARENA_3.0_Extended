# Narrow audit compatibility repair — 2026-10-16

## Plan and boundary

1. Validate actual image-only HTML data URLs in notebook output cells, retaining
   source, malformed-image, mixed-content and arbitrary HTML readability checks.
2. Recognize the current shared chapter-home renderer through its actual Python
   import and direct call contract, with positive and negative fixture tests.
3. Inspect and seal the existing geometry/alignment historical publication
   archives using `scripts.evidence_bundles`, retaining their exact bytes.
4. Run only relevant offline CPU validator tests and record failures as well as
   passes. Preserve the root-audit snapshot and all unrelated concurrent work.

This repairs validator compatibility and archive packaging only. It does not
recompute lesson results, establish learner acceptance, approve historical
scientific claims, or clear real missing sources/assets and stale-report gates.
Original ARENA content remains untouched. No model/download/cluster execution,
staging, commit, or push is authorized in this lane.

## Implemented compatibility contracts

- `scripts/audit_no_minified_files.py` permits long HTML only inside an actual
  code-cell `display_data`/`execute_result` output. The complete HTML bundle must
  contain only inert `<img>` tags and whitespace. Every strict-base64 data URL
  must have a supported raster-image MIME, matching signatures and a verified,
  decodable image. Source strings, HTML attachments, malformed/mismatched images,
  script/event/style content and mixed text still receive the original limit.
  JSON value paths prevent an identical HTML string in source from inheriting
  an output exemption. Ordinary binary MIME handling is unchanged.
- `scripts/audit_course_surface.py` recognizes the actual
  `infrastructure.extension.home.render_chapter_home` import and direct top-level
  call with `__file__` and nonempty introductory Markdown. The inspected renderer
  resolves chapter identity from that path and delegates configured chapter and
  section selection/styling to `st_dependencies`; its output includes the chapter
  introduction, section links/descriptions and exercise guidance. Comments, a
  similarly named wrapper, hidden calls, wrong arguments and shadowed bindings
  are insufficient. This is a static course-surface contract, not browser proof.
- Added adjacent `.review-bundle.json` contracts for the existing geometry and
  alignment pre-master publication archives, after safe bounded member inspection.
  Geometry has eight regular members; alignment has seven. Neither archive was
  extracted, rewritten or deleted. Contracts explicitly confer no scientific
  validation or human acceptance.

Archive SHA-256 values retained exactly:

| Archive migration directory | Bytes | SHA-256 |
| --- | ---: | --- |
| `geometry_master_migration` | 3251230 | `0e8f7b73df2ffcc9c76713854d7a02331a564a0a3351d72530e6d77b1e7d477c` |
| `alignment_master_migration` | 1198476 | `ed90bd17d83c81ae39e966c3e8afb67964f420f13e5ebaf1415725b601bb5996` |

## Reusable sealing command

From the repository root, set `archive_path` to the exact existing repository-relative
archive owned by that migration, then run this command. Inspection rejects unsafe,
unsupported, linked, raw or oversized members before writing a sidecar. This is
appropriate only for an unchanged historical archive whose complete contents meet
the review-bundle contract. If inspection fails, preserve the failure and archive;
do not weaken the contract or silently omit members.

```bash
archive_path='docs/mastery/evidence/geometry_master_migration/published_before_master.tar.gz'
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/arena-audit-compat-uv \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
uv run --no-sync --offline python - "$archive_path" <<'PY'
import sys
from pathlib import Path
from scripts.evidence_bundles import (
    file_record, inspect_archive, seal_review_bundle, validate_review_bundle,
)

root = Path.cwd().resolve()
archive = (root / sys.argv[1]).resolve()
relative = archive.relative_to(root).as_posix()
before = file_record(archive)
members = inspect_archive(archive)
provenance = {
    "kind": "historical_archive_packaging",
    "measurement_claim": (
        "Packaging contract for unchanged historical publication bytes only. "
        "No new measurement, scientific validation, learner acceptance, "
        "or human approval is established."
    ),
    "source_archive": {
        "absolute_path": str(archive), "repository_path": relative, **before,
    },
    "source_archive_members": {
        name: {key: item[key] for key in ("sha256", "size_bytes")}
        for name, item in members.items()
    },
}
seal_review_bundle(archive, provenance=provenance)
validate_review_bundle(archive)
assert file_record(archive) == before
print(relative, before, "safe members:", len(members))
PY
```

## Evidence and remaining limits

Scoped records live in
`docs/mastery/evidence/2026-10-16-audit-compatibility/`: the targeted CPU test
transcript, before-sealing member inventory, after-sealing archive validation and
current blocker snapshot. The original root-audit evidence is retained unchanged.
The shared checkout is changing concurrently, so later whole-tree counts are
separate observations rather than a reconciled before/after scientific result.

No learner notebook content, signatures, controls/baselines or play cells were
changed by this lane. Existing visible results remain embedded in the notebooks.
Original ARENA files were not modified. Real missing lessons/assets, historical
report freshness, source/prose readability, and learner acceptance remain open.
