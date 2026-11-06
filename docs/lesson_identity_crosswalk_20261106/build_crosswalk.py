#!/usr/bin/env python3
"""Lesson-identity crosswalk for the ARENA extension (read-only against a checkout).

    python3 build_crosswalk.py CHECKOUT OUT_DIR

Reads, never writes, CHECKOUT (git is run with GIT_OPTIONAL_LOCKS=0 so even the index is untouched):
  infrastructure/extension/syllabus.yaml        the 86 configured sections (single source of truth)
  infrastructure/core/config_extension.yaml     generated from the syllabus (page_file, exercise_dir)
  infrastructure/core/config.yaml               merged original + extension config (what the build loads)
  docs/roadmap_requirement_registry.yml         the 205 requirements and their statuses
  docs/arena_style_rewrite_status.yml           legacy style registry (59 rows, historical keys)
  mastery_hub/extension_checkpoints.json        guided prediction checkpoints
  docs/mastery/evidence/quality_20261103/{requirement_crosswalk.tsv,audit_index.json}  earlier crosswalk, re-measured
  chapter*/instructions/pages/*.md|*.py         instruction pages on disk
  chapter*/exercises/*                          exercise directories and notebooks
  infrastructure/chapters/**/master_*.py        section masters

Writes into OUT_DIR: lesson_crosswalk.tsv/.md, requirement_crosswalk.tsv, page_identity.tsv/.md,
orphan_exercise_dirs.tsv, snapshot.json, summary.json.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}

SECTION_KEY = lambda s: [int(x) for x in s.split(".")]  # noqa: E731


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, env=ENV, check=True).stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(path: Path) -> int | None:
    return len(path.read_text(encoding="utf-8", errors="replace").split()) if path.is_file() else None


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def slug(title: str) -> str:  # copied from infrastructure/extension/sync_config.py
    title = title.replace("&", "and").replace(" - ", " ")
    title = re.sub(r"[^A-Za-z0-9\-\s]", "", title)
    return re.sub(r"\s+", "_", title.strip())


# ---------------------------------------------------------------- git state (read-only)
head_files = set(git("ls-tree", "-r", "--name-only", "HEAD").splitlines())
status: dict[str, str] = {}
raw = git("status", "--porcelain=v1", "-z", "--untracked-files=all").split("\0")
i = 0
while i < len(raw):
    entry = raw[i]
    i += 1
    if len(entry) < 4:
        continue
    code, path = entry[:2], entry[3:]
    if code[0] in "RC":
        i += 1  # skip the "from" path
    status[path] = code


def gstate(relpath: str) -> str:
    exists = (ROOT / relpath).exists()
    code = status.get(relpath)
    if exists:
        if code == "??":
            return "untracked"
        if relpath in head_files:
            return "modified" if code and code.strip() else "tracked"
        return "added" if code else "ignored"
    return "deleted-in-worktree" if relpath in head_files else "absent"


# ---------------------------------------------------------------- sources
syllabus = yaml.safe_load((ROOT / "infrastructure/extension/syllabus.yaml").read_text())["chapters"]
cfg_ext = yaml.safe_load((ROOT / "infrastructure/core/config_extension.yaml").read_text())
cfg = yaml.safe_load((ROOT / "infrastructure/core/config.yaml").read_text())
registry = yaml.safe_load((ROOT / "docs/roadmap_requirement_registry.yml").read_text())["requirements"]
style = yaml.safe_load((ROOT / "docs/arena_style_rewrite_status.yml").read_text())["sections"]
checkpoints = {c["section"]: c for c in json.loads((ROOT / "mastery_hub/extension_checkpoints.json").read_text())["checkpoints"]}
quality_dir = ROOT / "docs/mastery/evidence/quality_20261103"
old_crosswalk = {r["id"]: r for r in csv.DictReader((quality_dir / "requirement_crosswalk.tsv").open(), delimiter="\t")}
audit_index = json.loads((quality_dir / "audit_index.json").read_text())
reg_by_id = {r["id"]: r for r in registry}

# config page_file by section number (extension-only and merged)
def page_files(config: dict) -> dict[str, tuple[str, str]]:
    return {
        s["number"]: (s["page_file"], s.get("exercise_dir", ""))
        for ch in config["chapters"].values()
        for s in ch["sections"]
        if s.get("number") and s.get("page_file")
    }

ext_pages = page_files(cfg_ext)
merged_pages = page_files(cfg)

# master files
masters: dict[str, list[str]] = defaultdict(list)
for p in (ROOT / "infrastructure/chapters").rglob("master_*.py"):
    masters[p.name].append(rel(p))

sections = []  # ordered
for chapter_key, chapter in syllabus.items():
    for index, s in enumerate(chapter["sections"], start=1):
        prefix = s.get("page_prefix", f"{index:02d}")
        stem = f"{prefix}_[{s['number']}]_{slug(s['title'])}"
        sections.append({**s, "chapter_key": chapter_key, "index": index, "stem": stem})
assert len(sections) == 86, len(sections)
by_number = {s["number"]: s for s in sections}
req_to_section: dict[str, str] = {}
for s in sections:
    for r in s.get("requirements", []):
        assert r not in req_to_section, r
        req_to_section[r] = s["number"]

# ---------------------------------------------------------------- page scan (disk and HEAD)
PAGE_RE = re.compile(r"^(\d+)_\[(\d+(?:\.\d+)+)\]_(.+)\.(md|py)$")
PAGE_DIR_RE = re.compile(r"^(chapter\d+_[^/]+)/instructions/pages/([^/]+)$")
pages: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))  # number -> stem -> {ext: relpath}   (on disk)
head_pages: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))  # same, from the HEAD tree
anomalies = []
for pages_dir in sorted(ROOT.glob("chapter*/instructions/pages")):
    for f in sorted(pages_dir.iterdir()):
        if f.is_dir() or f.suffix not in (".md", ".py"):
            continue
        m = PAGE_RE.match(f.name)
        if not m:
            anomalies.append(rel(f))
            continue
        pages[m.group(2)][f.stem][f.suffix[1:]] = rel(f)
for path in sorted(head_files):
    d = PAGE_DIR_RE.match(path)
    if not d or not path.endswith((".md", ".py")):
        continue
    m = PAGE_RE.match(d.group(2))
    if m:
        head_pages[m.group(2)][Path(path).stem][path[-2:] if path.endswith("md") else "py"] = path

# ---------------------------------------------------------------- text index for reference search
SKIP_EXT = {".png", ".gz", ".pt", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".whl", ".npz", ".safetensors", ".pyc",
            ".ico", ".mp4", ".webp", ".svg", ".bin", ".parquet", ".tar", ".zst", ".lock", ".csv"}
all_files = [p for p in git("ls-files", "-co", "--exclude-standard", "-z").split("\0") if p]


def category(path: str) -> str:
    if re.match(r"chapter\d+_[^/]+/instructions/pages/", path):
        return "other-page"
    if path == "docs/arena_style_rewrite_status.yml":
        return "legacy-style-registry"
    if (
        path.startswith(("docs/mastery/evidence/", ".scratch/", "docs/mastery/receipts", "FOR_REVIEW/"))
        or "/expected_outputs/" in path
        or "review-bundle" in path
        or "/evidence/" in path
    ):
        return "historical-evidence"
    if path.startswith(("docs/",)) or (("/" not in path) and path.endswith((".md", ".tex"))):
        return "docs"
    if path.endswith(".py") or path.endswith((".yml", ".yaml", ".toml", ".json", ".sh")):
        if path.startswith(("infrastructure/", "scripts/", "tests/", "arena_ext/", "mastery_hub/", ".github/")) or "/" not in path:
            return "live-code-or-config"
    return "other"


text_hits: dict[str, list[tuple[str, str]]] = defaultdict(list)  # stem -> [(path, category)]
all_stems = {stem for source in (pages, head_pages) for number_pages in source.values() for stem in number_pages}
# A stem-shaped string, plain or URL-quoted; the character class is greedy so a longer stem
# (e.g. "..._Practice") never counts as a reference to its shorter prefix.
STEM_RE = re.compile(r"\d+_(?:\[|%5[Bb])\d+(?:\.\d+)+(?:\]|%5[Dd])_[A-Za-z0-9_\-]+")
prefilter = ("_[", "_%5B", "_%5b")
for path in all_files:
    p = ROOT / path
    if p.suffix in SKIP_EXT or not p.is_file():
        continue
    try:
        if p.stat().st_size > 8_000_000:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    if not any(x in text for x in prefilter):
        continue
    seen: set[tuple[str, int]] = set()
    for m in STEM_RE.finditer(text):
        found = unquote(m.group(0))
        if found in all_stems:
            own = re.match(r"chapter\d+_[^/]+/instructions/pages/", path) and Path(path).stem == found
            if not own:
                line = text.count("\n", 0, m.start()) + 1
                cat = category(path)
                # one entry per (stem, file); keep the first line number
                if (found, hash(path)) in seen:
                    continue
                seen.add((found, hash(path)))
                text_hits[found].append((f"{path}:{line}", cat))

# lineage: exercise dirs a page links to (Colab links), mapped back to syllabus `sources`
EX_RE = re.compile(r"/exercises/([A-Za-z0-9_\-]+)/")
sources_index: dict[str, list[str]] = defaultdict(list)  # "chapterN/exercises/dir" -> [section numbers]
for s in sections:
    for src in s.get("sources", []):
        sources_index[src.rstrip("/")].append(s["number"])
    sources_index[f"{s['chapter_key']}/exercises/{s['dir']}"].append(s["number"])

section_dirs = {f"{s['chapter_key']}/exercises/{s['dir']}" for s in sections}


# ---------------------------------------------------------------- lesson crosswalk
lesson_rows = []
cfg_conv = cfg["conversion_map"]
for s in sections:
    n, ck = s["number"], s["chapter_key"]
    colab = s["stem"].split("_[")[-1].replace("]", "")
    master_name = f"master_{n.replace('.', '_')}.py"
    master_paths = masters.get(master_name, [])
    master = master_paths[0] if len(master_paths) == 1 else (master_paths[0] if master_paths else f"infrastructure/chapters/{ck}/{master_name}")
    ex_dir = f"{ck}/exercises/{s['dir']}"
    paths = {
        "master": master,
        "page_md": f"{ck}/instructions/pages/{s['stem']}.md",
        "page_py": f"{ck}/instructions/pages/{s['stem']}.py",
        "exercises": f"{ex_dir}/{colab}_exercises.ipynb",
        "solutions": f"{ex_dir}/{colab}_solutions.ipynb",
        "solutions_py": f"{ex_dir}/solutions.py",
    }
    exist = {k: (ROOT / v).is_file() for k, v in paths.items()}
    extra_nbs = sorted(
        x.name for x in (ROOT / ex_dir).glob("*.ipynb") if x.name not in {Path(paths['exercises']).name, Path(paths['solutions']).name}
    ) if (ROOT / ex_dir).is_dir() else []
    req_ids = list(s.get("requirements", []))
    stat = Counter(reg_by_id[r]["status"] for r in req_ids)
    legacy = style.get(n)
    cfg_ok = (
        cfg_conv.get(n, {}).get("streamlit_page") == s["stem"]
        and cfg_conv.get(n, {}).get("exercise_dir") == s["dir"]
        and ext_pages.get(n, ("", ""))[0] == s["stem"] + ".md"
        and merged_pages.get(n, ("", ""))[0] == s["stem"] + ".md"
    )
    n_pages = len(pages.get(n, {}))
    n_head = len(head_pages.get(n, {}))
    n_union = len(set(pages.get(n, {})) | set(head_pages.get(n, {})))
    cp = checkpoints.get(n)
    lesson_rows.append({
        "section": n,
        "chapter_dir": ck,
        "title": s["title"],
        "syllabus_status": s.get("status", ""),
        "exercise_dir": s["dir"],
        "master_path": paths["master"], "master_exists": exist["master"],
        "master_matches": len(master_paths),
        "page_md_path": paths["page_md"], "page_md_exists": exist["page_md"],
        "page_py_path": paths["page_py"], "page_py_exists": exist["page_py"],
        "exercises_path": paths["exercises"], "exercises_exists": exist["exercises"],
        "solutions_path": paths["solutions"], "solutions_exists": exist["solutions"],
        "solutions_py_exists": exist["solutions_py"],
        "extra_notebooks_in_dir": ";".join(extra_nbs),
        "all_six_present": all(exist.values()),
        "page_words": words(ROOT / paths["page_md"]),
        "master_words": words(ROOT / paths["master"]),
        "config_matches_syllabus": cfg_ok,
        "page_stems_on_disk": n_pages,
        "page_stems_at_head": n_head,
        "has_duplicate_pages": n_pages > 1,
        "has_duplicate_pages_disk_or_head": n_union > 1,
        "checkpoint_exists": cp is not None,
        "checkpoint_master_matches": (cp or {}).get("master_path") == paths["master"] if cp else "",
        "requirement_count": len(req_ids),
        "requirement_ids": ";".join(req_ids),
        "requirement_statuses": ";".join(f"{r}={reg_by_id[r]['status']}" for r in req_ids),
        "status_mix": ";".join(f"{k}:{stat[k]}" for k in ("TAUGHT", "TOY_ONLY", "PREFLIGHT_ONLY", "UNIMPLEMENTED", "DEFERRED") if stat[k]),
        "legacy_style_status": legacy["status"] if legacy else "",
        "legacy_style_page": legacy["page"] if legacy else "",
        "legacy_page_matches_current": (legacy["page"] == paths["page_md"]) if legacy else "",
        "legacy_page_exists": (ROOT / legacy["page"]).is_file() if legacy else "",
        "git_page_md": gstate(paths["page_md"]),
        "git_page_py": gstate(paths["page_py"]),
        "git_master": gstate(paths["master"]),
        "git_exercises": gstate(paths["exercises"]),
        "git_solutions": gstate(paths["solutions"]),
    })

# ---------------------------------------------------------------- requirement crosswalk
NUM_RE = re.compile(r"\d+\.\d+(?:\.\d+)?")


def path_like(entry: str) -> bool:
    return "/" in entry or entry.endswith((".py", ".md", ".yml", ".yaml", ".json", ".ipynb", ".txt", ".sh"))


req_rows = []
for r in registry:
    rid = r["id"]
    mapped = req_to_section.get(rid, "")
    hist_section = str(r.get("section", ""))
    hist_course = [str(x) for x in (r.get("course_sections") or [])]
    hist_numbers = set(NUM_RE.findall(hist_section)) | set(hist_course)
    if not mapped:
        identity = "unmapped:original-or-cross-course"
    elif mapped in hist_numbers:
        identity = "agrees"
    elif hist_section == "unassigned" and not hist_course:
        identity = "registry-unassigned"
    else:
        identity = "renumbered"
    evidence = [str(e) for e in (r.get("evidence") or [])]
    missing = []
    for e in evidence:
        bare = re.sub(r":\d+(-\d+)?$", "", e)
        if path_like(bare) and not (ROOT / bare).exists():
            missing.append(e)
    old = old_crosswalk.get(rid)
    sec = by_number.get(mapped)
    req_rows.append({
        "id": rid,
        "status": r["status"],
        "topic": r.get("topic", ""),
        "syllabus_section": mapped,
        "syllabus_title": sec["title"] if sec else "",
        "registry_chapter": r.get("chapter", ""),
        "registry_section_label": hist_section,
        "registry_course_sections": ";".join(hist_course),
        "identity_check": identity,
        "evidence_total": len(evidence),
        "evidence_missing": ";".join(missing),
        "blocker": (r.get("blocker") or "").replace("\n", " "),
        "quality_20261103_status_same": (old["status"] == r["status"]) if old else "absent-from-old-tsv",
    })

# ---------------------------------------------------------------- page identity
def norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower().replace("_", " ")).strip()


section_titles = {n: norm_title(slug(s["title"])) for n, s in by_number.items()}


def title_match(stem: str) -> str:
    """Where does a page with this title live now? Hint only: exact or fuzzy match against current syllabus titles."""
    t = norm_title(stem.split("]_", 1)[1])
    exact = [n for n, v in section_titles.items() if v == t]
    if exact:
        return "exact:" + ",".join(exact)
    ratio, best = max(((difflib.SequenceMatcher(None, t, v).ratio(), n) for n, v in section_titles.items()), default=(0.0, ""))
    return f"fuzzy:{best} ({ratio:.2f})" if ratio >= 0.6 else ""


def head_text(path: str) -> str:
    return git("show", f"HEAD:{path}")


def live_ref_kind(ref: str) -> str:
    """Classify one `path:line` reference to a page stem from live code, tests or config."""
    path, line_no = ref.rsplit(":", 1)
    try:
        lines = (ROOT / path).read_text(encoding="utf-8", errors="ignore").split("\n")
    except OSError:
        return "unreadable"
    window = "\n".join(lines[max(0, int(line_no) - 4): int(line_no) + 3])
    if path.startswith("infrastructure/core/config"):
        return "config page_file (expected for the canonical page)"
    if re.search(r"preservation|hygiene", path):
        return "preservation-allowlist string (file need not exist)"
    if path.startswith("infrastructure/extension/") and not re.search(r"\bsha\(|digest\(|sha256", window):
        return "extension page registry or contract (names the canonical page)"
    if re.search(r"\bsha\(|digest\(|sha256|published_page", window):
        return "HARD: hash-pins the page bytes"
    if "tmp_path" in window:
        return "synthetic tmp_path fixture"
    return "review"


def inferred_dir(ck: str, number: str, stem: str) -> str:
    k = number.split(".")[-1]
    ex = ROOT / ck / "exercises"
    cands = [d.name for d in ex.glob(f"part{k}_*") if d.is_dir() and f"{ck}/exercises/{d.name}" not in section_dirs]
    if not cands:
        return ""
    t = norm_title(stem.split("]_", 1)[1])
    return max(cands, key=lambda c: difflib.SequenceMatcher(None, t, norm_title(c.split("_", 1)[1])).ratio())


def role_hint(number, in_syllabus, original, canonical_stem, stem):
    if original:
        return "original"
    if canonical_stem is None:
        return "orphan"
    return "canonical" if stem == canonical_stem else "stale"


ORIGINAL_CHAPTERS = {f"chapter{i}_" for i in range(5)}
page_rows = []
group_rows = []
for number in sorted(set(pages) | set(head_pages), key=SECTION_KEY):
    disk, head = pages.get(number, {}), head_pages.get(number, {})
    stems = sorted(set(disk) | set(head))
    canonical_stem = merged_pages[number][0][:-3] if number in merged_pages else None
    in_syllabus = number in by_number
    syllabus_stem = by_number[number]["stem"] if in_syllabus else None
    first_path = next(iter({**head.get(stems[0], {}), **disk.get(stems[0], {})}.values()))
    ck = first_path.split("/")[0]
    original = ck.startswith(tuple(ORIGINAL_CHAPTERS)) and not in_syllabus
    if len(stems) < 2 and canonical_stem is not None and stems[0] == canonical_stem:
        continue  # one page, and it is the configured one: nothing to reconcile
    if original and len(stems) < 2:
        continue
    for stem in stems:
        d_files, h_files = disk.get(stem, {}), head.get(stem, {})
        hits = text_hits.get(stem, [])
        by_cat = Counter(c for _, c in hits)
        md_disk = d_files.get("md")
        body = ""
        if md_disk:
            body = (ROOT / md_disk).read_text(encoding="utf-8", errors="ignore")
        elif "md" in h_files:
            body = head_text(h_files["md"])
        # 1) the page's own Colab links (notebook name starts with this page's section number), not cross-links
        lineage_dirs = sorted(set(re.findall(rf"/exercises/([A-Za-z0-9_\-]+)/{re.escape(number)}_", body)))
        lineage_basis = "own Colab link" if lineage_dirs else ""
        # 2) fallback: an exercise directory no section uses, named part<last number component>_*, closest in title
        if not lineage_dirs and role_hint(number, in_syllabus, original, canonical_stem, stem) != "canonical":
            guess = inferred_dir(ck, number, stem)
            if guess:
                lineage_dirs, lineage_basis = [guess], "inferred from part number (weak)"
        lineage_sections = sorted(
            {sec for dd in lineage_dirs for sec in sources_index.get(f"{ck}/exercises/{dd}", [])}, key=SECTION_KEY
        )
        if original:
            role = "original-arena (out of scope, append-only)"
        elif canonical_stem is None:
            role = "orphan-number-not-in-syllabus"
        elif stem == canonical_stem:
            role = "canonical"
        else:
            role = "stale-duplicate"
        live_refs = sorted({p for p, c in hits if c == "live-code-or-config"})
        live_ref_text = [f"{r} [{live_ref_kind(r)}]" for r in live_refs]
        hard = any("HARD" in t for t in live_ref_text)
        legacy_ref = any(c == "legacy-style-registry" for _, c in hits)
        flags = []
        if role in ("stale-duplicate", "orphan-number-not-in-syllabus"):
            if hard:
                flags.append("BLOCKER-FOR-DELETION: a test or runner hash-pins this page")
            if any("review" in t for t in live_ref_text):
                flags.append("live reference needs a human look")
            if not (d_files.get("md") or d_files.get("py")):
                flags.append("already absent on disk (staged deletion in the main index); still in HEAD")
            tm = title_match(stem)
            if role == "stale-duplicate" and tm.startswith("exact:") and number not in tm:
                flags.append(f"same title as current {tm[6:]} (content renumbered)")
            elif role == "stale-duplicate" and not tm:
                flags.append("no current title matches: confirm the content is superseded before deleting")
            if role == "orphan-number-not-in-syllabus":
                flags.append("number is not in the syllabus; no canonical page for this number")
            if legacy_ref:
                flags.append("named by a legacy style-registry row")
        if role == "canonical" and not ((d_files.get("md") and d_files.get("py"))):
            flags.append("canonical page incomplete on disk")
        page_rows.append({
            "section": number,
            "chapter_dir": ck,
            "page_stem": stem,
            "role": role,
            "title_match_to_current_section": title_match(stem) if role != "canonical" else "",
            "on_disk_md": "md" in d_files, "on_disk_py": "py" in d_files,
            "in_head_md": "md" in h_files, "in_head_py": "py" in h_files,
            "git_md": gstate((d_files.get("md") or h_files.get("md"))) if (d_files.get("md") or h_files.get("md")) else "",
            "git_py": gstate((d_files.get("py") or h_files.get("py"))) if (d_files.get("py") or h_files.get("py")) else "",
            "md_words_on_disk": words(ROOT / md_disk) if md_disk else "",
            "equals_syllabus_derived_name": (stem == syllabus_stem) if in_syllabus else "n/a-not-in-syllabus",
            "equals_config_extension_page_file": (stem + ".md" == ext_pages[number][0]) if number in ext_pages else "n/a-not-in-config",
            "equals_merged_config_page_file": (stem + ".md" == merged_pages[number][0]) if number in merged_pages else "n/a-not-in-config",
            "referenced_by_live_code_or_config": ";".join(live_ref_text),
            "hard_dependency": hard,
            "referenced_by_legacy_style_registry": legacy_ref,
            "refs_docs": by_cat.get("docs", 0),
            "refs_historical_evidence": by_cat.get("historical-evidence", 0),
            "refs_other_pages": by_cat.get("other-page", 0),
            "exercise_dirs_linked_from_page": ";".join(lineage_dirs),
            "lineage_basis": lineage_basis,
            "lineage_sections_via_syllabus_sources": ";".join(lineage_sections),
            "streamlit_autodiscovers_py": "yes (classic multipage: every pages/*.py is a route)" if "py" in d_files else "",
            "flags": "; ".join(flags),
        })
    group_rows.append({
        "section": number, "stems_on_disk": len(disk), "stems_at_head": len(head), "stems_union": len(stems),
        "canonical": canonical_stem, "in_syllabus": in_syllabus, "original": original,
    })

# orphan exercise dirs (not any section's dir) and whether the syllabus `sources` still points at them
orphan_dir_rows = []
for ck in sorted({s["chapter_key"] for s in sections}):
    ex = ROOT / ck / "exercises"
    if not ex.is_dir():
        continue
    for d in sorted(ex.iterdir()):
        if not d.is_dir() or d.name.startswith((".", "__")):
            continue
        key = f"{ck}/exercises/{d.name}"
        if key in section_dirs:
            continue
        if ck in ("chapter0_fundamentals", "chapter1_transformer_interp") and key not in sources_index:
            continue  # original ARENA exercise dirs are not extension orphans
        refs = sources_index.get(key, [])
        orphan_dir_rows.append({
            "exercise_dir": key,
            "referenced_by_syllabus_sources_of": ";".join(sorted(set(refs), key=SECTION_KEY)),
            "git_notebooks": ";".join(sorted({gstate(rel(n)) for n in d.glob("*.ipynb")})),
            "files": sum(1 for _ in d.rglob("*") if _.is_file()),
        })

# legacy style registry (59 historical keys): is its page the canonical page for that key, or a different lesson?
legacy_rows = []
for key in sorted(style, key=SECTION_KEY):
    entry = style[key]
    page = entry["page"]
    stem = Path(page).stem
    current = by_number.get(key)
    legacy_rows.append({
        "legacy_key": key,
        "legacy_status": entry["status"],
        "legacy_page": page,
        "legacy_page_exists_on_disk": (ROOT / page).is_file(),
        "git_legacy_page": gstate(page),
        "key_in_syllabus": current is not None,
        "current_canonical_page_for_key": (f"{current['chapter_key']}/instructions/pages/{current['stem']}.md" if current else ""),
        "legacy_page_is_canonical_for_key": (page == f"{current['chapter_key']}/instructions/pages/{current['stem']}.md") if current else "",
        "legacy_page_title_now_lives_at": title_match(stem) if stem[0].isdigit() and "]_" in stem else "",
        "current_title_for_key": current["title"] if current else "",
    })

# The three stale/orphan pages that a test or runner hash-pins. Identified by reading the references the page
# table lists (tests/mastery/test_influence_tracin_master_migration.py:86, tests/mastery/test_sage_master_migration.py:80,
# infrastructure/chapters/chapter11_representation_geometry/master_11_5.py:3299); this just re-checks the bytes.
HARD_PINS = {
    "chapter11_representation_geometry/instructions/pages/05_[11.5]_Contrast_Consistent_Search.md": "chapter11_representation_geometry/exercises/part5_contrast_consistent_search",
    "chapter16_shapley_attribution_baselines/instructions/pages/09_[16.9]_Influence_Functions_and_TracIn.md": "chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin",
    "chapter16_shapley_attribution_baselines/instructions/pages/12_[16.12]_SAGE_Global_Importance.md": "chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance",
}
hard_pin_checks = []
for page_path, ex_dir in HARD_PINS.items():
    receipt = ROOT / ex_dir / "expected_outputs/learner_execution.json"
    data = json.loads(receipt.read_text())
    hard_pin_checks.append({
        "page": page_path,
        "receipt": rel(receipt),
        "page_sha256_equals_receipt_published_page_sha256": sha256(ROOT / page_path) == data["published_page_sha256"],
    })

# ---------------------------------------------------------------- summaries
status_counts = Counter(r["status"] for r in req_rows)
syl_reqs = {r for r in req_to_section}
summary = {
    "sections": len(lesson_rows),
    "sections_with_master": sum(r["master_exists"] for r in lesson_rows),
    "sections_with_page_md": sum(r["page_md_exists"] for r in lesson_rows),
    "sections_with_page_py": sum(r["page_py_exists"] for r in lesson_rows),
    "sections_with_exercises_nb": sum(r["exercises_exists"] for r in lesson_rows),
    "sections_with_solutions_nb": sum(r["solutions_exists"] for r in lesson_rows),
    "sections_with_all_six": sum(r["all_six_present"] for r in lesson_rows),
    "sections_with_checkpoint": sum(r["checkpoint_exists"] for r in lesson_rows),
    "sections_config_matches_syllabus": sum(r["config_matches_syllabus"] for r in lesson_rows),
    "sections_with_duplicate_pages": sum(r["has_duplicate_pages"] for r in lesson_rows),
    "sections_in_legacy_style_registry": sum(1 for r in lesson_rows if r["legacy_style_status"]),
    "legacy_style_rows_total": len(style),
    "legacy_style_status": dict(Counter(v["status"] for v in style.values())),
    "legacy_style_rows_not_in_syllabus": sorted((k for k in style if k not in by_number), key=SECTION_KEY),
    "legacy_page_matches_current": sum(1 for r in lesson_rows if r["legacy_page_matches_current"] is True),
    "legacy_page_differs_from_current": [r["section"] for r in lesson_rows if r["legacy_page_matches_current"] is False],
    "requirements_total": len(req_rows),
    "requirement_status_counts": dict(status_counts),
    "requirements_mapped_by_syllabus": len(syl_reqs),
    "requirements_unmapped_original_or_cross_course": [r["id"] for r in req_rows if not r["syllabus_section"]],
    "requirement_identity_check": dict(Counter(r["identity_check"] for r in req_rows)),
    "requirements_with_missing_evidence_paths": sum(1 for r in req_rows if r["evidence_missing"]),
    "requirements_status_matches_quality_20261103_tsv": sum(1 for r in req_rows if r["quality_20261103_status_same"] is True),
    "requirements_per_section_distribution": dict(sorted(Counter(r["requirement_count"] for r in lesson_rows).items())),
    "requirement_status_by_section_count": {k: sum(1 for r in lesson_rows if k in r["status_mix"]) for k in ("TAUGHT", "TOY_ONLY", "PREFLIGHT_ONLY", "UNIMPLEMENTED", "DEFERRED")},
    "quality_20261103_audit_index_registry_sha256": (audit_index.get("requirement_registry") or {}).get("sha256") if isinstance(audit_index.get("requirement_registry"), dict) else audit_index.get("requirement_registry"),
    "registry_sha256_now": sha256(ROOT / "docs/roadmap_requirement_registry.yml"),
    "audit_index_rows": len(audit_index.get("rows", [])),
    "sections_with_duplicate_pages_on_disk": sum(r["has_duplicate_pages"] for r in lesson_rows),
    "sections_with_duplicate_pages_disk_or_head": sum(r["has_duplicate_pages_disk_or_head"] for r in lesson_rows),
    "page_stems_on_disk_total": sum(len(v) for v in pages.values()),
    "page_stems_at_head_total": sum(len(v) for v in head_pages.values()),
    "extension_stale_duplicate_stems": sum(1 for r in page_rows if r["role"] == "stale-duplicate"),
    "extension_stale_duplicate_stems_on_disk": sum(1 for r in page_rows if r["role"] == "stale-duplicate" and (r["on_disk_md"] or r["on_disk_py"])),
    "extension_stale_duplicate_stems_head_only": sum(1 for r in page_rows if r["role"] == "stale-duplicate" and not (r["on_disk_md"] or r["on_disk_py"])),
    "extension_stale_files_md_on_disk": sum(1 for r in page_rows if r["role"] == "stale-duplicate" and r["on_disk_md"]),
    "extension_stale_files_py_on_disk": sum(1 for r in page_rows if r["role"] == "stale-duplicate" and r["on_disk_py"]),
    "extension_stale_files_tracked_at_head": sum(int(r["in_head_md"]) + int(r["in_head_py"]) for r in page_rows if r["role"] == "stale-duplicate"),
    "extension_stale_files_untracked_on_disk": sum(int(r["git_md"] == "untracked") + int(r["git_py"] == "untracked") for r in page_rows if r["role"] == "stale-duplicate"),
    "orphan_number_stems": [r["page_stem"] for r in page_rows if r["role"] == "orphan-number-not-in-syllabus"],
    "canonical_stems_in_page_table": sum(1 for r in page_rows if r["role"] == "canonical"),
    "canonical_pages_untracked_on_disk": sum(1 for r in lesson_rows if r["git_page_md"] == "untracked"),
    "canonical_pages_tracked_clean_or_modified": sum(1 for r in lesson_rows if r["git_page_md"] in ("tracked", "modified")),
    "original_arena_groups_with_more_than_one_stem": sorted({r["section"] for r in page_rows if r["role"].startswith("original")}, key=SECTION_KEY),
    "stale_stems_referenced_by_live_code_or_config": {r["page_stem"]: r["referenced_by_live_code_or_config"] for r in page_rows if r["role"] in ("stale-duplicate", "orphan-number-not-in-syllabus") and r["referenced_by_live_code_or_config"]},
    "stale_or_orphan_stems_hash_pinned_by_tests_or_runners": [r["page_stem"] for r in page_rows if r["hard_dependency"] and r["role"] in ("stale-duplicate", "orphan-number-not-in-syllabus")],
    "stale_files_on_disk_with_py_twin": sum(1 for r in page_rows if r["role"] in ("stale-duplicate", "orphan-number-not-in-syllabus") and r["on_disk_py"]),
    "stale_or_orphan_stems_referenced_by_legacy_style_registry": [r["page_stem"] for r in page_rows if r["role"] in ("stale-duplicate", "orphan-number-not-in-syllabus") and r["referenced_by_legacy_style_registry"]],
    "stale_or_orphan_stems_with_no_reference_anywhere_outside_evidence": [r["page_stem"] for r in page_rows if r["role"] in ("stale-duplicate", "orphan-number-not-in-syllabus") and not r["referenced_by_live_code_or_config"] and not r["referenced_by_legacy_style_registry"] and not r["refs_docs"] and not r["refs_other_pages"]],
    "hard_pin_checks": hard_pin_checks,
    "legacy_keys_whose_page_is_the_canonical_page": sum(1 for r in legacy_rows if r["legacy_page_is_canonical_for_key"] is True),
    "legacy_keys_whose_page_is_not_the_canonical_page": sum(1 for r in legacy_rows if r["legacy_page_is_canonical_for_key"] is False),
    "legacy_pages_missing_on_disk": [r["legacy_key"] for r in legacy_rows if not r["legacy_page_exists_on_disk"]],
    "sections_under_6000_page_words": sum(1 for r in lesson_rows if (r["page_words"] or 0) < 6000),
    "page_words_min_median_max": [min(r["page_words"] for r in lesson_rows), sorted(r["page_words"] for r in lesson_rows)[len(lesson_rows)//2], max(r["page_words"] for r in lesson_rows)],
    "page_name_anomalies": anomalies,
    "orphan_exercise_dirs": len(orphan_dir_rows),
    "orphan_exercise_dirs_referenced_by_syllabus_sources": sum(1 for r in orphan_dir_rows if r["referenced_by_syllabus_sources_of"]),
}

snapshot = {
    "taken_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "checkout": str(ROOT),
    "branch": git("branch", "--show-current").strip(),
    "head": git("rev-parse", "HEAD").strip(),
    "head_subject": git("log", "-1", "--format=%s").strip(),
    "worktree_status_entry_counts": dict(Counter(v for v in status.values())),
    "source_sha256": {
        p: sha256(ROOT / p)
        for p in (
            "infrastructure/extension/syllabus.yaml",
            "infrastructure/core/config_extension.yaml",
            "infrastructure/core/config.yaml",
            "docs/roadmap_requirement_registry.yml",
            "docs/arena_style_rewrite_status.yml",
            "mastery_hub/extension_checkpoints.json",
            "docs/mastery/migrations/roadmap_evidence_repair.md",
            "docs/mastery/evidence/quality_20261103/requirement_crosswalk.tsv",
            "docs/mastery/evidence/quality_20261103/audit_index.json",
        )
    },
}


# ---------------------------------------------------------------- write
def write_tsv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("yes" if v is True else "no" if v is False else ("" if v is None else v)) for k, v in r.items()})


write_tsv("lesson_crosswalk.tsv", lesson_rows)
write_tsv("requirement_crosswalk.tsv", req_rows)
write_tsv("page_identity.tsv", page_rows)
write_tsv("orphan_exercise_dirs.tsv", orphan_dir_rows)
write_tsv("legacy_style_crosswalk.tsv", legacy_rows)
(OUT / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

mark = lambda b: "✓" if b is True else ("✗" if b is False else "")  # noqa: E731
with (OUT / "lesson_crosswalk.md").open("w", encoding="utf-8") as fh:
    fh.write("# Lesson crosswalk: 86 configured sections\n\n")
    fh.write("Generated by `build_crosswalk.py`; full paths and every column are in `lesson_crosswalk.tsv`. ")
    fh.write("`M` master, `Pg` instruction page `.md`, `Py` page `.py`, `Ex`/`So` exercise/solution notebook, `CP` guided checkpoint. ")
    fh.write("`Words` counts the canonical page `.md`. Status mix is the registry status of the requirement IDs the syllabus assigns to the section.\n\n")
    fh.write("| § | Title | Dir | M | Pg | Py | Ex | So | CP | Words | Reqs | Status mix | Legacy style | Dup pages |\n")
    fh.write("|---|---|---|---|---|---|---|---|---|---:|---:|---|---|---|\n")
    for r in lesson_rows:
        legacy = r["legacy_style_status"]
        if legacy:
            legacy += "" if r["legacy_page_matches_current"] is True else " (page≠)"
        fh.write(
            f"| {r['section']} | {r['title']} | `{r['exercise_dir']}` | {mark(r['master_exists'])} | {mark(r['page_md_exists'])} | "
            f"{mark(r['page_py_exists'])} | {mark(r['exercises_exists'])} | {mark(r['solutions_exists'])} | {mark(r['checkpoint_exists'])} | "
            f"{r['page_words'] if r['page_words'] is not None else ''} | {r['requirement_count']} | {r['status_mix']} | {legacy} | "
            f"{r['page_stems_on_disk'] if r['has_duplicate_pages'] else ''} |\n"
        )
    fh.write("\n## Requirement IDs by section\n\n")
    for r in lesson_rows:
        fh.write(f"- **{r['section']}** ({r['requirement_count']}): " + ", ".join(f"`{x.split('=')[0]}` {x.split('=')[1]}" for x in r["requirement_statuses"].split(";") if x) + "\n")
    fh.write("\n## Original / cross-course requirements (no syllabus section)\n\n")
    for r in req_rows:
        if not r["syllabus_section"]:
            fh.write(f"- `{r['id']}` {r['status']} (registry label: {r['registry_section_label']})\n")

with (OUT / "page_identity.md").open("w", encoding="utf-8") as fh:
    fh.write("# Page identity: section numbers with more than one instruction page\n\n")
    fh.write("One *page* is one filename stem (a `.md` plus its `.py` twin). `canonical` is the stem that the syllabus derives and the generated `config_extension.yaml`, the merged `config.yaml` and the build all agree on. ")
    fh.write("Everything else is `stale-duplicate`, or `orphan-number-not-in-syllabus` when the number is not in the syllabus at all; original ARENA chapters 0-4 are listed but out of scope. ")
    fh.write("Disk = the main checkout's working tree; HEAD = its committed tree. Columns: `disk md/py`, `HEAD md/py`, `git` state, words, ")
    fh.write("`live refs` (code, tests, config outside docs/evidence that name the stem), `legacy` (named by `docs/arena_style_rewrite_status.yml`), `title now` (current section whose title matches), `lineage` (current sections whose syllabus `sources` include an exercise directory this page links to).\n\n")
    for number in sorted({r["section"] for r in page_rows}, key=SECTION_KEY):
        rows = [r for r in page_rows if r["section"] == number]
        fh.write(f"### {number}" + (f" — {by_number[number]['title']}" if number in by_number else " — not in the syllabus") + "\n\n")
        fh.write("| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |\n|---|---|---|---|---|---:|---|---|---|---|---|\n")
        for r in sorted(rows, key=lambda r: (r["role"] != "canonical", r["page_stem"])):
            fh.write(
                f"| {r['role']} | `{r['page_stem']}` | {mark(r['on_disk_md'])}/{mark(r['on_disk_py']) or '·'} | {mark(r['in_head_md'])}/{mark(r['in_head_py']) or '·'} | "
                f"{r['git_md']}/{r['git_py'] or '·'} | {r['md_words_on_disk']} | {r['referenced_by_live_code_or_config'].replace(';', '<br>') or '—'} | "
                f"{mark(r['referenced_by_legacy_style_registry']) or '—'} | {r['title_match_to_current_section'] or '—'} | {(r['lineage_sections_via_syllabus_sources'] + (' (inferred)' if 'inferred' in r['lineage_basis'] else '')) if r['lineage_sections_via_syllabus_sources'] else '—'} | {r['flags'] or '—'} |\n"
            )
        fh.write("\n")

print(json.dumps(summary, indent=2))
print(json.dumps(snapshot, indent=2))
