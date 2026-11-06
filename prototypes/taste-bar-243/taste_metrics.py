"""Per-page taste metrics for the authoritative extension pages (read-only)."""
import csv, re, sys, difflib
from pathlib import Path
import yaml

M = Path(sys.argv[1]); out = Path(sys.argv[2])
syl = yaml.safe_load((M / "infrastructure/extension/syllabus.yaml").read_text())
rows = []
for chap, cd in syl["chapters"].items():
    pages = list((M / chap / "instructions/pages").glob("*.md"))
    for s in cd["sections"]:
        num, title = str(s["number"]), s["title"]
        cands = [p for p in pages if f"[{num}]" in p.name]
        if not cands:
            rows.append(dict(num=num, title=title, page="MISSING")); continue
        norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
        best = max(cands, key=lambda p: difflib.SequenceMatcher(None, norm(p.stem.split("]_",1)[-1]), norm(title)).ratio())
        txt = best.read_text(errors="ignore")
        intro = txt.split("## Content & Learning Objectives")[0]
        ex = re.findall(r"^#{2,4} Exercise\b.*$", txt, re.M)
        # exercise blocks: text between an exercise heading and next heading
        blocks = re.split(r"^#{2,4} Exercise\b", txt, flags=re.M)[1:]
        tagged = sum(1 for b in blocks if "Difficulty:" in b[:800] and "Importance:" in b[:800])
        timed = sum(1 for b in blocks if re.search(r"You should spend", b[:800]))
        rows.append(dict(
            num=num, title=title, page=best.name, dupes=len(cands)-1,
            status=s.get("status",""),
            words=len(txt.split()),
            intro_words=len(intro.split("# Introduction")[-1].split()),
            opener_shown=int(bool(re.search(r"By the end of this (notebook|section)[^\n]{0,40}shown", txt))),
            by_the_end=int("By the end of this" in intro),
            banner=int("Guided prediction practice" in txt),
            provenance_before_clo=int(bool(re.search(r"(Historical reference|receipt|historical reference|execution boundary|provenance|Status)", intro))),
            gt_tier=int("GT_TIER" in txt),
            new_chapter=txt.count("=== NEW CHAPTER ==="),
            internal_ids=len(re.findall(r"(docs/mastery/|\bBV\d\d\b|\bQ00\d\b|requirement id|_execution\.json)", txt)),
            historical=len(re.findall(r"\bhistorical\b", txt, re.I)),
            exercises=len(ex), tagged=tagged, timed=timed,
            predict_first=len(re.findall(r"(Make a prediction first|Predict first|before you run|Before running)", txt, re.I)),
            generic_forecast=len(re.findall(r"^## (Forecast before the result|Reflect after the result)", txt, re.M)),
            details=txt.count("<details"), imgs=len(re.findall(r"<img |!\[", txt)),
            bonus=int(bool(re.search(r"^#+ .*Bonus", txt, re.M))),
            signature=int(bool(re.search(r"signature result", txt, re.I))),
        ))
keys = list(max(rows, key=len).keys())
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
print(len(rows), "rows")
