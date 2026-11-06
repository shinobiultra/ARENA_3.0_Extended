#!/usr/bin/env python
"""Fill `Expected output` placeholders in a master file from the executed solutions notebook.

    uv run python infrastructure/extension/fill_expected.py 8.4            # show what would be filled
    uv run python infrastructure/extension/fill_expected.py 8.4 --write    # write it into the master

In the master, write an expected-output block whose contents is a placeholder starting with `FILL_`, e.g.

    <pre style="...">FILL_SWEEP</pre>

directly after the code cell whose output it should show (an html-tagged image cell may sit in between). After
`build.py N --execute`, this script replaces each placeholder with the text that the preceding code cell actually
printed (stdout, plus plain-text displays such as DataFrames), dropping progress bars and model-loading chatter.
Interpretations are not filled: write those yourself, from the real output.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "infrastructure" / "core" / "config.yaml").read_text())
PLACEHOLDER = re.compile(r"(<pre[^>]*>)(FILL_[A-Z0-9_]+)(</pre>)")
NOISE = re.compile(r"(it/s\]|s/it\]|%\|.*\||^Loading weights|^Loaded pretrained model|^Moving model to device|Warning|warn\()")


def cell_text(cell) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream" and out.get("name") == "stdout":
            parts.append(out["text"])
        elif out.get("output_type") in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "text/plain" in data and "image/png" not in data and "text/html" not in data:
                parts.append(data["text/plain"] + "\n")
            elif "text/html" in data and "text/plain" in data and "<table" in "".join(data["text/html"]):
                parts.append(data["text/plain"] + "\n")
    text = "".join(parts)
    lines = [line.split("\r")[-1] for line in text.split("\n")]
    lines = [line for line in lines if not NOISE.search(line)]
    return "\n".join(lines).strip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("section")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    mapping = CONFIG["conversion_map"][args.section]
    chapter = CONFIG["chapter_names"][int(args.section.split(".")[0])]
    colab = mapping["streamlit_page"].split("_[")[-1].replace("]", "")
    nb = nbformat.read(ROOT / chapter / "exercises" / mapping["exercise_dir"] / f"{colab}_solutions.ipynb", as_version=4)
    master = next((ROOT / "infrastructure" / "chapters").rglob(f"master_{args.section.replace('.', '_')}.py"))

    fills = {}
    last_code = None
    for cell in nb.cells:
        if cell.cell_type == "code":
            last_code = cell
            continue
        for _, name, _ in PLACEHOLDER.findall(cell.source):
            if last_code is None or not last_code.get("outputs"):
                print(f"{name}: no executed code cell before it (run build.py --execute first)")
                continue
            fills[name] = cell_text(last_code)

    text = master.read_text()
    for name, value in fills.items():
        print(f"===== {name}\n{value}\n")
        text = text.replace(f">{name}</pre>", f">{html.escape(value, quote=False)}</pre>")
    if args.write:
        master.write_text(text)
        print(f"Filled {len(fills)} placeholder(s) in {master.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
