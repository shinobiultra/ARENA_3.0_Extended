# 16.19 — Explain the Proxy, Check the Model

**Status: native learner integration prepared; actual portable replay awaits shared data.**
The draft has 10 exercise groups and 13 visible algorithm functions, with
adjacent tests, expected outputs, help and solutions. It does not establish
native-model quality, section completion or course readiness.

The current local draft is `draft_execution_v6/` under
`/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/`.
Its own-answer execution passed all 24 analytic code cells and generated nine
analytic figures in 17.29 seconds, with zero language-model queries and unchanged
sources. Three native cells are deliberately unexecuted in this analytic-only
receipt; the notebook contains 27 code cells in total. All nine PNG files are
byte-identical to v3. The focused method/learner suite has 26 passing tests.
Frozen `proxy_methods.py` remains unchanged.

The native cells now load the shared hash-checked portable query fixture and
refit real and shuffled responses using the notebook's own functions. They show
the complete recorded population comparison, fresh stage predictions and word
spectra, and an editable prompt/budget refit. The author and maintained package
results remain explicitly recorded comparators. No extra model query path is
invoked. Synthetic interface tests are labeled as tests and are not native results.

- [Current analytic receipt](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v6/execution.json)
- [Current exercise HTML](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v6/html_review/exercises.html) and
  [solution HTML](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v6/html_review/solutions.html)
- [Current exported page and assets](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v6/publication_inputs/)
- [Analytic PNG parity](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v6/analytic_png_parity.json)

The v6 execution receipt SHA-256 is:

`68287512c76a7181b0dc2c7bfd5854efec68a8edc8a6a31a736e76930731c58a`

The prior v3 draft and all earlier evidence remain unchanged:

- [Execution receipt](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v3/execution.json)
- [Completed learner notebook](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v3/completed_learner.ipynb)
- [Exercise HTML](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v3/html_review/exercises.html) and
  [solution HTML](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v3/html_review/solutions.html)
- [PNG/PDF figures](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v3/figures/)
- [Exact output comparison with v2](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v3/output_parity.json)
- [Browser visual review of v2](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_v2/html_review/visual_review.json)
- [Relocation manifest and file inventories](/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/relocation_manifest.json)

The v3 execution receipt SHA-256 is:

`a5ac6fe91acef54671b36a5e98321be8aa958138b897c35dd4a623bdd3c640ef`

Version 3 only wraps overlong prose lines. Its 26 code cells, recorded text and
display payloads, and all nine PNG files are identical to v2. The earlier drafts
remain in the cache, with every file's SHA-256 verified during relocation.
Historical receipts and snapshots retain their original embedded paths and
unchanged bytes; the relocation manifest maps those paths to the current copies.
`/draft_execution_*/` remains ignored as protection against future local draft
outputs. Later publication must bind reviewed evidence in the planned sealed
archive.

To reproduce into a new empty directory, run from the repository root:

```bash
.venv/bin/python3 chapter16_shapley_attribution_baselines/exercises/part19_proxyspex/build_notebooks.py \
  --draft-output /home/bitwise/.cache/arena_extended/sparse_interaction_discovery/part19_learner_drafts/draft_execution_fresh \
  --execute-analytic --render
```

After the shared completed run, cold reproduction and sealed portable fixture
are available, use `--execute-native --native-fixture /absolute/path/to/native_query_fixture.json.gz`
instead of `--execute-analytic`, into a new empty output directory. That mode
requires all 27 code cells and 13 figures, verifies saved-result metrics and
24 pointwise prediction checks across the two declared real/shuffled replay pairs,
and records both upstream execution hashes and the fixture hash. The prediction
checks include every proxy stage and both baselines at `atol=1e-8, rtol=1e-7`.
It never calls the language model.

The actual native replay, native visual review and independent publication review
remain outstanding. Original ARENA files and canonical publication metadata were
not edited by this draft work.
