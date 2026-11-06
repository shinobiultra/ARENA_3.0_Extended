# Lesson identity crosswalk (2026-11-06)

Fact-gathering for the map ticket **"Reconcile current lesson identities and the complete requirement inventory"**
(shinobiultra/ARENA_3.0_Extended issue 6). It is a throwaway branch, `task/lesson-identity-crosswalk`, and is never merged.
Nothing in the main checkout, the teaching registry or the status registries was edited, staged or committed.

## What is here

| File | What it is |
|---|---|
| [`lesson_crosswalk.md`](lesson_crosswalk.md) / [`.tsv`](lesson_crosswalk.tsv) | **Table 1.** One row per syllabus section (86): number, title, master path, page `.md`/`.py`, exercises/solutions notebooks, presence, word count, requirement IDs and their registry statuses, checkpoint, legacy style row, duplicate-page count, git state. |
| [`requirement_crosswalk.tsv`](requirement_crosswalk.tsv) | All 205 requirements: status, the syllabus section that teaches it (blank for the 7 original/cross-course rows), the registry's own historical section label, whether the two agree, evidence paths missing, blocker. |
| [`page_identity.md`](page_identity.md) / [`.tsv`](page_identity.tsv) | **Table 2.** Every section number with more than one instruction page: which page the syllabus and build use, each stale duplicate, and the evidence that it is unused, plus flags. |
| [`legacy_style_crosswalk.tsv`](legacy_style_crosswalk.tsv) | The 59 legacy style-registry rows against the current canonical pages (supporting). |
| [`orphan_exercise_dirs.tsv`](orphan_exercise_dirs.tsv) | The 42 exercise directories no section uses, and which sections' `sources:` still name them (supporting). |
| [`snapshot.json`](snapshot.json), [`summary.json`](summary.json) | Snapshot (time, commit, source hashes) and every count quoted below. |
| [`build_crosswalk.py`](build_crosswalk.py) | The generator. Read-only against the checkout. |

Reproduce: `python3 -I build_crosswalk.py /home/bitwise/Documents/ARENA_3.0_Extended <out dir>` (16 s).

## Snapshot

Measured 2026-11-06 14:44 JST in `/home/bitwise/Documents/ARENA_3.0_Extended`, branch `codex/arena-frontier-lab-extension`.
Local HEAD is `dec028ec` ("Rebuild extension sections on ARENA master files"), one commit ahead of the pushed branch tip `458a7b3a`.
The working tree is far ahead of HEAD (149 modified, 7,881 untracked, 255 staged deletions, 60 unstaged deletions), and nothing in it had changed for two hours, so the measurement is stable.
**All presence, word and duplicate counts are from the working tree on disk**; HEAD columns are given where they differ.

## Re-measured counts (ticket's old figures in brackets)

- **86 sections** configured. Master, page `.md`, page `.py`, exercises notebook, solutions notebook and guided checkpoint all exist for **86 of 86** [60 masters, 26 absent]. Every section has exactly one `master_N_M.py`. `config_extension.yaml`, the merged `config.yaml` and the syllabus-derived page names agree for 86 of 86.
- **Presence is not teaching completeness.** Canonical page length runs from 2,900 to 100,175 words (median 8,959); 26 sections are under 6,000 words, which matches the taste-bar audit.
- **205 requirements**, unchanged: 54 TAUGHT, 20 TOY_ONLY, 35 PREFLIGHT_ONLY, 83 UNIMPLEMENTED, 13 DEFERRED. The registry file hash equals the one pinned in `quality_20261103/audit_index.json`, and all 205 statuses equal that earlier crosswalk's. No evidence path in the registry is missing.
- **198 mapped by the syllabus, 7 not**: `foundations.original_arena_preservation`, `foundations.original_mechinterp_spine`, `foundations.hard_exercise_test_ladders`, `representation.linear_and_nonlinear_probes`, `circuits.path_patching`, `world.othello_gpt`, `organisms.greater_than`. No requirement is assigned to two sections. Requirements per section: 1 (31 sections), 2 (31), 3 (12), 4 (4), 5 (2), 6 (3), 7 (2), 11 (1).
- **Registry labels versus syllabus**: the registry's own `section` / `course_sections` agree with the syllabus for 58 requirements, disagree (renumbered) for 45, and say `unassigned` for 95; the syllabus supplies the identity for all of those. Match on requirement ID, not on the number.
- **Legacy style registry**: 59 rows (55 course_ready, 4 prototype). 53 keys are syllabus sections and 6 (16.12–16.17) are not. For only 17 of the 53 does the registry's `page:` equal the canonical page; for 36 it names a different page. Two legacy pages no longer exist on disk (8.3 ACDC, 16.7 Data Shapley).
- **Canonical 8.3 is AtP from Scratch and canonical 8.4 is ACDC and Circuit Metrics**, as the ticket says; the old 8.3 ACDC page is a stale HEAD-only file.

## Page identity (Table 2) in numbers

A *page* is one filename stem; an `.md` and its `.py` twin are one page.

- **38 of the 86 sections have more than one page on disk** (41 counting three stale pages that exist at HEAD but are already staged for deletion on disk: 8.3, 8.6, 16.7). By chapter: 16 (10 sections), 5 (6), 9 (5), 14 (4), 6 (3), 8 (2), 15 (2), and one each in 0, 1, 11, 12, 13, 17.
- **42 stale-duplicate pages** in extension sections: 39 on disk and 3 HEAD-only. Every one is **`.md` only**; no stale page has a `.py` twin. 41 are tracked at HEAD and one is untracked (the 16.3 `TreeSHAP_DeepSHAP_and_GradientSHAP` page).
- **6 orphan pages for numbers outside the syllabus** (16.12–16.17), all `.md` only and tracked at HEAD.
- The audits' "16.3 four times" is 3 page identities and 4 files (canonical `.md` + `.py`, plus two stale `.md`). The audits' "14.1–14.5 twice" measures here as 14.1–14.4 twice and 14.5–14.7 once; the old 14.4 *Maze Transformers and Latent Planning* page is the pre-renumbering 14.5.
- **Canonical pages are mostly uncommitted**: 65 of the 86 canonical `.md` pages are untracked and 21 are tracked and modified; none is tracked clean. HEAD therefore still contains the stale pages and, for those 65 sections, not the canonical ones.

### What "the page the syllabus and build use" means

For each number the canonical page is `{page_prefix or index}_[N.M]_{slug(title)}`, derived exactly as `infrastructure/extension/sync_config.py` does. It is the `page_file` in `config_extension.yaml` and the merged `config.yaml`, and the `streamlit_page` that `infrastructure/extension/build.py` hands to the converter. Consumers all go through that config name: the Django chapter loader (`page_file`), `infrastructure/extension/home.py`, `mastery_hub/read.py`, `scripts/audit_arena_style_depth.py`, `scripts/audit_course_surface.py`, `scripts/run_extension_verification_reports.py`. Evidence that a page is unused, per stale row in `page_identity.tsv`:

1. its name differs from the syllabus-derived name;
2. it is not a `page_file` in `config_extension.yaml` or `config.yaml`;
3. no code, test or config outside docs and evidence names it, except the cases flagged below (references are exact stem matches, so `..._Practice` does not count as a reference to its shorter prefix);
4. it has no `.py` twin, so Streamlit's classic multipage discovery (every `pages/*.py` is a route) never serves it.

## Flags: not safe to treat as simply unused

1. **Three stale pages are hash-pinned.** `05_[11.5]_Contrast_Consistent_Search.md` is checked by `run_gpu_test` in `infrastructure/chapters/chapter11_representation_geometry/master_11_5.py:3299`; `09_[16.9]_Influence_Functions_and_TracIn.md` by `tests/mastery/test_influence_tracin_master_migration.py:86`; `12_[16.12]_SAGE_Global_Importance.md` by `tests/mastery/test_sage_master_migration.py:80`. Each page's current bytes equal the receipt's `published_page_sha256`, while the canonical page differs, so deleting them breaks those checks unless they are retired or repointed in the same batch. Those two tests also pin the legacy exercise directories and archive tarballs, which the page cleanup does not touch.
2. **0.6 and 1.6 stale pages are named in the preservation allowlist** (`scripts/audit_original_arena_preservation.py:98-104`, `scripts/audit_extension_artifact_hygiene.py:103-109`, `tests/mastery/test_preservation_scope.py:76-80`, `tests/test_original_arena_preservation_audit.py:26`). These are allowances (a file need not exist), and the audit also lists the new `_Practice` pages, so deletion is safe for the audit; the old strings are tidy-up only.
3. **The legacy style registry still names 42 of the 48 stale and orphan pages** in its `page:` fields. Deleting them leaves dangling pointers, but 36 of its 53 in-syllabus rows already disagree with the config, and `style_status_blockers` (`scripts/audit_arena_style_depth.py:121-131`, read but not run) is written to flag exactly that mismatch, plus a missing entry for each of the 33 sections with no legacy row. The ticket says not to alter the registry here; whoever runs the cleanup needs a rule for it.
4. **Stale pages whose content went to several successors** (title match alone is not enough): stale 12.1 CLIP/SigLIP controls feeds 12.1 and 12.4; 13.1 diffusion controls feeds 13.1 and 13.3; 14.1 JEPA controls feeds 14.2 and 14.3; Stale 5.4, 9.4, 13.1, 16.5, 16.8 and 16.10 match no current title, and their successor comes only from syllabus `sources:` lineage.
5. **16.12–16.17 tail.** There is no syllabus or config entry for these numbers. Syllabus `sources:` lineage (the stale page's `part<N>_*` exercise directory; weak, inferred from the directory name) puts 16.12 SAGE into 16.4, 16.13–16.15 TreeSHAP, DeepSHAP and Expected Gradients into 16.3, and 16.16–16.17 into 16.5. Only 16.12 matches a current title exactly. Whether any content in the 16.13–16.17 pages is missing from 16.3 and 16.5 was not checked; that is for the cleanup to decide. 16.12 is also hash-pinned (flag 1).
6. **Name variants of the same lesson, with stale notebooks beside the canonical ones.** For 0.6 and 1.6 (`..._Practice` suffix), 11.5 (underscore versus hyphen), 15.2 (`Finetuning` versus `Fine-Tuning`), 16.3 (`..._Practice` suffix) and 17.1 (retitled from *Mechanism Emergence* to *Grokking*) the stale page and the canonical page share one exercise directory, and that directory also holds a stale-named exercises/solutions notebook pair next to the canonical pair (`extra_notebooks_in_dir` in `lesson_crosswalk.tsv`). A page-only cleanup leaves those six stale notebook pairs behind. 16.1 and 16.2 are name variants too, with no stale notebooks. The canonical page is whatever the config names; 11.5's variant is hash-pinned (flag 1).
7. **Latent hazard that cleanup removes**: `arena_ext/streamlit_home.py::_section_page_link` picks the first sorted `*.md` containing `[N.M]`, which for duplicate numbers can be a stale page (for example 5.2). No chapter `Home.py` uses it (all 14 extension homes use `render_chapter_home`), so it is not live.
8. **Exercise directories are not stale pages.** 42 exercise directories no section uses are each named in some section's syllabus `sources:` (rewrite input). Deleting stale pages must not take them along.
9. **Already partly cleaned in the main checkout's index**: 315 deleted files in all (255 staged: chapter 8 has 184, chapter 16 has 71; 60 more unstaged in chapter 12), which include three stale pages (8.3 ACDC, 8.6 AtP-Star, 16.7 Data Shapley); the rest is mostly exercise material. That is someone else's work; this crosswalk lists those three pages as HEAD-only.
10. **Original ARENA duplicates, out of scope (append-only)**: 1.3.1 (`Linear_Probes`, `Probing_for_Deception`), 2.2.2 (`Policy_Gradient`, `VPG`) and the non-numeric `39_[1.5.X]_Monthly_Algorithmic_Problems`. They are in `page_identity.tsv` under the role `original-arena` and were not touched.

## Method notes

- Sources read: the five ticket sources plus `config_extension.yaml`, `config.yaml`, the 2026-11-03 `requirement_crosswalk.tsv` and `audit_index.json`, the build scripts, and a text scan of the 11,273 tracked and untracked files (binary extensions and files over 8 MB skipped) for exact page-stem references. Git was run with `GIT_OPTIONAL_LOCKS=0`, so the index was not refreshed.
- Word count is whitespace-separated tokens of the canonical page `.md` (`master_words`, the master `.py`, is alongside).
- Reference categories in the scan: live code or config (`infrastructure/`, `scripts/`, `tests/`, `arena_ext/`, `mastery_hub/`), the legacy style registry, other docs, historical evidence (`docs/mastery/evidence`, `expected_outputs`, review bundles) and other pages. Historical evidence references are counted but never treated as dependencies.
- The three hash-pin checks (flag 1) were identified by reading the listed references, then verified by recomputing sha256 (`summary.json`, `hard_pin_checks`).
- Lineage by exercise directory comes from the stale page's own Colab links where it has them, otherwise from an unused directory named `part<k>_*` closest in title, marked "inferred". It is a hint, not proof.
- Not done: no notebook execution, no test run, no model or cluster use, and no judgement of whether any lesson is taught well.
