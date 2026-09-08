# Historical evidence packaging

This work changes packaging and its audit contract. It makes no native execution,
scientific revalidation, fresh-review, publication or course-completion claim.

Current learner failure: the hygiene gate treats readable review archives and raw
model/tensor caches alike, while some archives actually contain raw serialization.
Target section claim: preserve the evidence and make the distinction enforceable.
Signature result to create: none; existing figures and scientific values are preserved.
Student-implemented functions: unchanged.
Controls/baselines: unchanged, including historical failed runs and negative results.
Play cells: unchanged.
Files to modify: hygiene audit/tests, focused packaging helpers/contracts, ten
historical archives with raw payloads, the packaging functions in the GradientSHAP
publisher, and oversized JSON evidence packaging.
Tests to run: archive member/path/type/hash/bounds mutation tests, external-original
resolution, bounded JSON validation, full artifact hygiene and original preservation.

The section 16.15 student/solution notebooks and instruction page, original 0.4
exercise notebook, repository AGENTS.md, salvage skill and HANDOFF.md were inspected.
The section 16.15 publisher's AST is unchanged outside its two archive functions
and inline-payload limit. Numerical arrays, models, scientific methods, protocols,
thresholds, seeds, receipts and native input hashes are not changed.

Each `.tar.gz.review-bundle.json` contract binds the archive and every member by
path, regular-file type, format, byte size and SHA256. The validator rejects unsafe
paths, duplicate paths, links/special files, unsupported types, raw serialization,
disguised zip/pickle/NumPy payloads, unlisted members, changed hashes and changed
bounds. Gzip members must decode to JSON and receive a separate decoded hash.

Bounds are 64 MiB per archive, 16 MiB per encoded member, 128 MiB total members,
64 MiB per decoded JSON, 512 MiB total decoded evidence, and 4,096 members. These
are bounded review/data contracts; raw archive suffixes alone never pass the audit.
The whole expanded tar container, including metadata and padding, is separately
bounded at 160 MiB; nonzero bytes hidden after its final member are rejected.

All original archive bytes are first copied under their SHA256 filenames in
`~/.cache/arena_extended/evidence_packaging/original_archives/`. The packaging
inventory records the original repository path, exact digest/size, external path,
Git commit/blob recovery provenance, derived digest, and retained/external member
counts. Ten bundles have raw `.pt`, `.joblib` or `.npz` members removed from the
review copy; the exact original contains every removed payload. Other bundle bytes
remain unchanged. Section 16.17 is handled by its separate revalidation worker.

Historical manifests and embedded receipts keep their old hashes and dates.
Where an original archive digest differs from its current review copy, resolve the
old receipt with `scripts.evidence_bundles.resolve_historical_archive(path, sha256)`.
It verifies the old bytes from the recorded cache. If the cache is unavailable,
restore the recorded Git blob (`git cat-file blob <git_blob>`) into the recorded
external path and verify its SHA256. The original member inventory must partition
exactly into retained review members and external payloads; retained bytes match.

The audit validates committed review bytes and provenance structure without requiring
another machine to already have this local cache. Migration verifies all originals
and external bytes physically; the resolver verifies them again whenever requested.

Reproduce packaging with the project Python:

```bash
.venv/bin/python scripts/package_historical_evidence.py \
  --cache ~/.cache/arena_extended/evidence_packaging/original_archives \
  --inventory docs/maintenance/2026-09-08_evidence_packaging/archive_inventory.json \
  --exclude-part part17_shapiq_estimators
```

The command validates existing sealed bundles without repackaging them. This
maintenance directory and its contracts record packaging time separately from
immutable scientific measurement time.

The completed migration covers 168 historical archives: 609,552,846 original
compressed bytes become 386,014,804 review bytes. All 41 removed raw members
(420,119,628 uncompressed bytes) remain in their exact originals. The 168 Git
recovery blob identities and cache SHA256 values are verified in
`equivalence_verification.json`; `archive_inventory.json` lists every affected path.

The 12.3 report changed from 11,492,470 to 3,730,522 bytes by preserving object
indentation while keeping primitive array records on one line. Every JSON value
and key/record ordering is retained. Its `.packaging.json` sidecar records the exact
original and canonical value hash. The complete 5.5 diffusion text outcome file
remains byte-identical at 7,304,827 compressed bytes / 44,218,340 decoded bytes;
its `.data-contract.json` binds both forms. Both originals are additionally saved
under `~/.cache/arena_extended/evidence_packaging/original_json/`.

The separate 16.17 worker subsequently supplied its newly validated review archive.
Only its adjacent strict contract was added here: 92 members, six external numerical
payloads physically rehashed, and unchanged archive SHA256
`470407a4d90c3083efed4247b4be1ae32180162cd13120ca01e892fb7863fce4`.
Five `learner/html_review/browser_*.png` files actually contain JPEG bytes:
`browser_budget_curves`, `browser_native_play`, `browser_nullspace_derivation`,
`browser_pair_derivation`, and `browser_signed_pairs`. Their reviewed bytes and
filenames remain unchanged; detected JPEG format and filename mismatch are explicit
in the contract. No browser review is newly claimed by this packaging operation.
