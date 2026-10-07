#!/usr/bin/env python
"""Print a compact review summary of a built extension section.

    uv run python infrastructure/extension/review.py 7.1
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "infrastructure" / "core" / "config.yaml").read_text())


def main(number: str) -> None:
    mapping = CONFIG["conversion_map"][number]
    chapter = CONFIG["chapter_names"][int(number.split(".")[0])]
    exercise_dir = ROOT / chapter / "exercises" / mapping["exercise_dir"]
    colab = mapping["streamlit_page"].split("_[")[-1].replace("]", "")
    ex = nbformat.read(exercise_dir / f"{colab}_exercises.ipynb", as_version=4)
    so = nbformat.read(exercise_dir / f"{colab}_solutions.ipynb", as_version=4)

    md = "\n".join(c.source for c in ex.cells if c.cell_type == "markdown")
    print(f"== {number} {mapping['streamlit_page']}")
    print(f"cells={len(ex.cells)} code={sum(c.cell_type == 'code' for c in ex.cells)} words={len(md.split()):,}")
    print("files:", sorted(p.name for p in exercise_dir.iterdir()))
    media = sorted((exercise_dir / "media").glob("*.png")) if (exercise_dir / "media").exists() else []
    print(f"figures ({len(media)}):", [p.stem for p in media])

    print("\n-- outline")
    for line in md.splitlines():
        if re.match(r"^#{1,3} ", line) and not line.startswith("### Exercise"):
            print("  " + line[:100])
    exercises = re.findall(r"^### Exercise - (.*)$", md, flags=re.M)
    print(f"\n-- exercises ({len(exercises)})")
    for e in exercises:
        print("  - " + e[:90])
    dropdowns = re.findall(r"<summary>(.*?)</summary>", md)
    kinds = {}
    for d in dropdowns:
        key = d.split(" - ")[0].split("(")[0].strip()[:30]
        kinds[key] = kinds.get(key, 0) + 1
    print("\n-- dropdowns:", dict(sorted(kinds.items(), key=lambda kv: -kv[1])[:8]))
    print("-- try-it-yourself:", len(re.findall(r"Try it yourself", md, flags=re.I)),
          "| controls mentioned:", len(re.findall(r"control", md, flags=re.I)),
          "| bonus:", "# ☆ Bonus" in md)

    errors = [o for c in so.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    executed = sum(bool(c.get("execution_count")) for c in so.cells if c.cell_type == "code")
    print(f"-- solutions: {executed} executed code cells, {len(errors)} errors")
    sizes = {p.name: f"{p.stat().st_size // 1024}KB" for p in exercise_dir.glob("*.ipynb")}
    print("-- sizes:", sizes)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        main(arg)
        print()
