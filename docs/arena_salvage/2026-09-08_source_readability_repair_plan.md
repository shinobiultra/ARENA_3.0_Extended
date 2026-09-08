# Targeted source readability repair

Current learner failure: the initial audit found 87 long source/prose lines,
including 80 findings across the 26 non-chapter-16 files assigned to this pass.
Long paragraphs and string literals obscured review of otherwise unchanged
teaching material. The audit limits remain Python 240, Markdown 400 and notebook
serialization 5000 characters.

Target section claim: preserve every existing claim, result, limitation, date,
control, threshold and tolerance. This is a cosmetic publication change with a
separately reviewed compatibility check; it does not represent a new native
experiment or an upgrade in course readiness.

Signature result to create: none. Existing figures and concrete outputs remain
unchanged. Student-implemented functions, controls/baselines and play cells are
unchanged. Eleven inspected notebooks, including the original ARENA IOI
comparator, remain byte-identical.

Files modified: all 26 assigned flagged source/prose files, three additional
validator files, one narrow compatibility helper and its mutation tests, four
publication input manifests, and the three dated plan/proof records. The exact
inventory and before/after hashes are in
`2026-09-08_readability_equivalence.json`. Before-files are retained in
`/home/bitwise/.cache/arena-readability-repair-2026-09-08/before/`.

The accepted 2026-09-05 cosmetic procedure applies to ordinary source and prose:
compare complete Python ASTs including constants and CommonMark rendering with
only non-code whitespace normalized. The initial safe pass repaired 48 flags
across 18 files. Eight remaining files were first left untouched because native
validators required exact execution-source bytes. That dependency was reported
before editing; changing publication hashes alone would have broken validation.

The coordinating root agent then explicitly authorized and reviewed the narrow
compatibility route. `arena_ext/readability_compatibility.py` pins the digest of
`2026-09-08_readability_compatibility.json`. Its eleven named entries retain exact
historical source bytes and permit only exact current publication hashes after
complete AST/constants or CommonMark comparison. Four validator guard changes
are separately classified: their exact reviewed snippets must be reversed before
whole-AST equality holds. They are not represented as AST-identical formatting.
All scientific code remains identical under that comparison. No record can
supply arbitrary accepted patches or admit a future source mismatch.

The helper pins the compatibility record, and affected publication manifests
bind both. The compatibility record does not hash the helper or reports, so
there is no circular hash dependency. Historical execution receipts, numerical
measurements, dates, source snapshots and archive manifests remain unchanged by
this pass. Only `report_inputs` changed in the 5.5, 8.3, 9.1 and 11.1 reports.
The 5.5 historical published-page hash remains intact; the current wrapped page
is explicitly recorded as a derived cosmetic publication.

Tests run: 28 initial source/audit tests pass; the final compatibility mutation
and affected-section run passes 58 tests in 3.39 seconds. Mutation checks reject
changed arithmetic, literals, tolerances, claims, code fences, snapshots,
unlisted files and altered compatibility records. The actual 8.3 published
evidence validator passes. The actual 11.1 learner and archived native-cache
validators pass, preserving all 120,649 learner outcomes and all 6,656 native
comparison rows across 208 conditions, with zero logit error or token mismatches.
This validates retained evidence; no model was executed.

All sixteen original 5.5 archive source snapshots match their execution hashes
and the permitted current publications. Its full native validator remains
untested because the historical `/tmp/diffusion-text-full-own-answer-release-v1`
and `/tmp/diffusion-text-data-candidate-v3` directories, including required data
and weights, are absent. Unit checks and source equivalence do not recreate that
native execution. No data was rebuilt and no historical receipt was relabeled.

All 80 assigned readability findings are resolved. At final proof generation the
global readability audit also has zero findings after other agents' chapter-16
repairs. Original ARENA preservation passes. The last report evidence audit has
three remaining chapter-16 publication-source freshness findings, owned by the
coordinator. No source-readability threshold changed. This pass performs no
notebook regeneration, archive packaging, staging, commit or release transition.
