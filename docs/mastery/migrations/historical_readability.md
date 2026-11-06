# Historical readability proof repair — 2026-10-16

## Plan and boundary

The two historical proof tests read the live publication files for all eleven
entries in the pinned 2026-09-08 compatibility record. Two original 8.3 files were
retired: `part3_acdc_circuit_metrics/edge_evidence.py` and `lesson.md`. Their exact
recorded after-bytes remain recoverable from Git HEAD; this does not describe the
current 8.4 lesson or its execution.

1. Retain the original failing test transcript and pin the unchanged eleven-entry
   record and production helper hashes.
2. Recover only those two historical files from the exact Git commit, verify each
   `after_sha256`, and seal a two-member historical bundle with explicit Git/blob
   provenance. Keep retired paths absent from the runtime tree.
3. Add a test-only historical byte resolver with a pinned retired-path mapping.
   A present file must match its exact after-hash; only the two missing retired
   identities may use the sealed archive. Unknown paths/mappings and tampering fail.
4. Keep the full AST/CommonMark proofs and all reviewed guard reversals under
   their original identities. Add rejection checks for changed current sources,
   missing/unknown mappings, archive tampering and any transfer to 8.4.

`arena_ext/readability_compatibility.py`, including production
`source_matches_execution`, remains byte-identical. Missing current execution
sources still fail closed; archived historical proof establishes neither current
execution compatibility nor scientific/learner acceptance. The original manifest
is not edited, shortened or redirected. No active pilot, registry, runtime route,
model, report or scientific result is changed.

## Implemented proof boundary

Only `tests/test_readability_compatibility.py` changes executable validation code.
Its private-purpose historical resolver first checks the unchanged manifest pin
and exact original entry identity. A present publication must match its named
after-hash; a changed present file never falls back to the archive. A missing
publication can resolve only if it is one of the two explicitly retired paths,
the mapping pin and original-record identity match, the mapping preserves the
original member name, the archive/contract hashes and bounded bundle validation
pass, and the recovered member still matches its original after-hash.

The eleven-entry proof test still checks all eleven source pairs and modes.
Nine use present exact publication bytes and production execution matching. Two
use archived exact publication bytes for historical equivalence only, while
explicitly requiring production execution matching to reject the absent paths.
The four reviewed validation-guard cases still reverse only the original hardcoded
snippets and reject a scientific tolerance addition. CommonMark comparison still
protects claims, code fences and all non-whitespace rendered content.

No code in `arena_ext/readability_compatibility.py` changes. In particular,
`source_matches_execution` has no archive import, locator or fallback. Its existing
arithmetic/literal/tolerance, unknown-path, snapshot, record, duplicate and arbitrary
guard-replacement rejection tests remain intact.

## Exact recovered provenance

The source Git HEAD was `dec028ec8ed8accb40bf0d0a7315bb72f0c1fcc6`.
Each file was read with `git show COMMIT:original/path`, then checked against the
original manifest's after-hash before being placed in a two-member regular-file
archive. No checkout/restore command or filesystem recreation of the retired
runtime route occurred. `git-capture.json` records the commit, individual Git blob
IDs, exact retrieval argv, sizes, hashes and original helper/manifest fingerprints.

| Artifact | SHA-256 |
| --- | --- |
| Unchanged original eleven-entry manifest | `f6d6fc026d139f46697ee3626ac96869ab49401f600fe5d3c9ddc5e36c941708` |
| Sealed recovered historical archive | `68095ab1dc3ba8585daa83483945e3cfbf563cd34d9bb8f14830cb71dea551f6` |
| Pinned retired-path mapping | `4d84a8186612065a7ba043aa214fc4fdc73b099231d2cdd1aba876974f2654da` |
| Original 8.3 `edge_evidence.py` after-bytes | `8ca9a2d28ebbec75f45daa0436dbfe9ad9a3db6e3eb72b58d1ea5cdcd5e6c802` |
| Original 8.3 `lesson.md` after-bytes | `d69e0aad2b5d498411e9acabf643214420ec672451a31e3945951f3aa8b982db` |

Artifacts live in `docs/mastery/evidence/2026-10-16-historical-readability/`.
The adjacent bundle contract labels exact Git-blob recovery as historical cosmetic
proof, not new runtime source, execution, measurement or acceptance. The mapping
has no replacement route; identical archived bytes placed under an 8.4 identity
are explicitly rejected by the tests.

## Verification and limits

The exact original failure was reproduced: **2 failed, 12 passed in 0.21s**.
Both failures were `FileNotFoundError` while the historical tests read the retired
`edge_evidence.py`. `before-tests.txt` retains this result. The final focused
run passed **28 tests in 0.41s**, including the unchanged original checks and new
archive, identity, present-file, missing-map, pinned-map, contract, manifest and
8.4-transfer rejection checks. `git diff --check` passed for the test change.

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/arena-audit-compat-uv \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
uv run --no-sync --offline python -m pytest -q -p no:cacheprovider \
  tests/test_readability_compatibility.py --tb=short
```

The original manifest and production helper were rehashed unchanged, the bundle
was revalidated, and both retired runtime paths remain absent. No original ARENA
file, learner method, figure, control, baseline or play cell changed. This narrow
historical validation does not establish current 8.3/8.4 execution, fresh model
results, full-course completion or learner acceptance. Independent root review
remains the next boundary; no staging, commit or push was performed.
