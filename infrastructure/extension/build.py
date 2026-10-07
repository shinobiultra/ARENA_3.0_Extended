#!/usr/bin/env python
"""Build extension sections from ARENA master files.

This wraps the original ARENA converter in `infrastructure/core` so that extension
sections are authored exactly like the original ones: one readable master file per
section, from which the exercises notebook, solutions notebook, `solutions.py` and
Streamlit page are generated.

On top of the original pipeline it adds three things the extension needs:

1. Page links point at this fork rather than the upstream ARENA repo.
2. `--execute` runs the generated solutions notebook top-to-bottom in a fresh kernel,
   and saves it with its outputs (so the solutions notebook doubles as a full set of
   expected outputs).
3. Expected-output figures. In the master file, a markdown cell tagged `html` placed
   directly after a code cell is shown as that cell's output (this is how the original
   ARENA notebooks show expected plots). If that cell contains `<img src="media/NAME.png">`,
   the image produced by the code cell during `--execute` is saved to the section's
   `media/NAME.png`, and inlined into the exercises notebook so it renders anywhere.

Usage (from the repo root):

    uv run python infrastructure/extension/build.py 7.1 --execute
    uv run python infrastructure/extension/build.py 7.* --execute
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import re
import sys
import time
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "infrastructure" / "core"
MASTERS = ROOT / "infrastructure" / "chapters"
sys.path.insert(0, str(CORE))

from conversion import MasterFileData  # noqa: E402

FORK = "shinobiultra/ARENA_3.0_Extended"
BRANCH = "main"
COLAB_ROOT = f"https://colab.research.google.com/github/{FORK}/blob/{BRANCH}/"
RAW_ROOT = f"https://raw.githubusercontent.com/{FORK}/{BRANCH}/"
FIRST_EXTENSION_CHAPTER = 5
MEDIA_PATTERN = re.compile(r"""src=["']media/([\w\-]+)\.png["']""")


class ExtensionMasterFile(MasterFileData):
    """The original converter, with links pointing at this fork."""

    def get_page_links(self) -> tuple[list[str], list[str]]:
        rel = f"{self.chapter_dir.name}/exercises/{self.exercise_dir_name}/{self.colab_name}"
        exercises_link = f"{COLAB_ROOT}{rel}_exercises.ipynb"
        solutions_link = f"{COLAB_ROOT}{rel}_solutions.ipynb"
        colab_lines = [
            f"> **Colab: [exercises]({exercises_link}) | [solutions]({solutions_link})**",
            "",
            "This section is part of the ARENA extension. The solutions notebook is saved with all of its "
            "outputs, so you can use it as a reference for what every cell should produce.",
            "",
            "You can collapse each section so only the headers are visible, by clicking the arrow symbol on "
            "the left hand side of the markdown header cells.",
        ]
        streamlit_lines = [
            f"> **Colab: [exercises]({exercises_link}) | [solutions]({solutions_link})**",
            "",
            "If you want to change to dark mode, you can do this by clicking the three horizontal lines in the "
            "top-right, then navigating to Settings → Theme.",
        ]
        return streamlit_lines, colab_lines


def load_config() -> dict:
    return yaml.safe_load((CORE / "config.yaml").read_text(encoding="utf-8"))


def find_master(section: str) -> Path:
    name = f"master_{section.replace('.', '_')}.py"
    matches = sorted(MASTERS.rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {name} under {MASTERS}, found {matches}")
    return matches[0]


def generate(section: str, config: dict) -> ExtensionMasterFile:
    chapter_num = int(section.split(".")[0])
    if chapter_num < FIRST_EXTENSION_CHAPTER and section not in ("0.6", "1.6"):
        raise ValueError(f"{section} is an original ARENA section; build it with infrastructure/core/main.py")
    mapping = config["conversion_map"][section]
    master_py = find_master(section)
    master_ipynb = master_py.with_suffix(".ipynb")
    if not master_ipynb.exists():
        master_ipynb.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}')
    master = ExtensionMasterFile(
        master_path=master_ipynb,
        chapter_dir=ROOT / config["chapter_names"][chapter_num],
        chapter_name_long=config["chapter_names_long"][chapter_num],
        exercise_dir_name=mapping["exercise_dir"],
        streamlit_page_name=mapping["streamlit_page"],
    )
    master.exercises_dir.mkdir(parents=True, exist_ok=True)
    master.instructions_dir.mkdir(parents=True, exist_ok=True)
    master.master_py_to_ipynb(overwrite=True)
    master.generate_files(overwrite=True, verbose=False)
    return master


def media_cells(nb: nbformat.NotebookNode) -> dict[int, list[str]]:
    """Map code-cell index -> media names referenced by its pre-filled html output."""
    found = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        names = []
        for out in cell.get("outputs", []):
            html = out.get("data", {}).get("text/html", "")
            html = "".join(html) if isinstance(html, list) else html
            names += MEDIA_PATTERN.findall(html)
        if names:
            found[i] = names
    return found


def execute(path: Path, timeout: int) -> tuple[nbformat.NotebookNode, float]:
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    start = time.time()
    try:
        client.execute()
    except CellExecutionError:
        failed = path.with_name(path.stem + ".FAILED.ipynb")
        nbformat.write(nb, failed)
        print(f"\nExecution failed. Partially executed notebook saved to {failed}")
        raise
    return nb, time.time() - start


NOISE_PATTERNS = [
    r"^W\d{4} .*\] .*$",  # torch/glog-style warnings emitted at import time
    r"^.*is an Enum subclass and is now natively supported.*$",
    r"^\[transformers\] `torch_dtype` is deprecated.*$",
    r"^Unable to import `torchao` Tensor objects.*$",
]


def strip_environment_noise(nb: nbformat.NotebookNode) -> None:
    """Remove library warnings that say nothing about the exercises from saved stderr outputs."""
    pattern = re.compile("|".join(NOISE_PATTERNS), flags=re.M)
    for cell in nb.cells:
        kept = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                text = pattern.sub("", out["text"])
                text = re.sub(r"\n{2,}", "\n", text).strip("\n")
                if not text:
                    continue
                out["text"] = text + "\n"
            kept.append(out)
        cell["outputs"] = kept


def save_media(executed: nbformat.NotebookNode, wanted: dict[int, list[str]], media_dir: Path) -> list[str]:
    saved, problems = [], []
    for idx, names in wanted.items():
        images = [
            out["data"]["image/png"]
            for out in executed.cells[idx].get("outputs", [])
            if "image/png" in out.get("data", {})
        ]
        if len(images) < len(names):
            problems.append(f"cell {idx} produced {len(images)} image(s) but the master expects {names}")
            continue
        media_dir.mkdir(exist_ok=True)
        for name, data in zip(names, images[-len(names) :]):
            (media_dir / f"{name}.png").write_bytes(base64.b64decode(data))
            saved.append(name)
    if problems:
        raise RuntimeError("Expected-output figures missing:\n  " + "\n  ".join(problems))
    return saved


def inline_media(path: Path, media_dir: Path) -> None:
    """Replace `media/NAME.png` references in pre-filled outputs with inline images."""
    nb = nbformat.read(path, as_version=4)

    def to_data_uri(match: re.Match) -> str:
        file = media_dir / f"{match.group(1)}.png"
        if not file.exists():
            return match.group(0)
        return f'src="data:image/png;base64,{base64.b64encode(file.read_bytes()).decode()}"'

    for cell in nb.cells:
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "text/html" in data:
                html = data["text/html"]
                html = "".join(html) if isinstance(html, list) else html
                data["text/html"] = MEDIA_PATTERN.sub(to_data_uri, html)
    nbformat.write(nb, path)


def rewrite_page_media(master: ExtensionMasterFile) -> None:
    page = master.instructions_dir / f"{master.streamlit_page_name}.md"
    rel = f"{master.chapter_dir.name}/exercises/{master.exercise_dir_name}/media/"
    text = page.read_text(encoding="utf-8")
    page.write_text(re.sub(r"""src=(["'])media/""", rf"src=\1{RAW_ROOT}{rel}", text), encoding="utf-8")


def summarise(master: ExtensionMasterFile) -> str:
    ex = nbformat.read(master.exercises_dir / f"{master.colab_name}_exercises.ipynb", as_version=4)
    md = "\n".join(c.source for c in ex.cells if c.cell_type == "markdown")
    n_ex = len(re.findall(r"^#+ Exercise", md, flags=re.M))
    n_code = sum(c.cell_type == "code" for c in ex.cells)
    return f"{len(ex.cells)} cells ({n_code} code), {len(md.split()):,} words of prose, {n_ex} exercises"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sections", nargs="+", help="Section numbers or patterns, e.g. 7.1 or '7.*'")
    parser.add_argument("--execute", action="store_true", help="Execute the solutions notebook and save outputs")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-cell execution timeout in seconds")
    args = parser.parse_args()

    config = load_config()
    available = list(config["conversion_map"])
    sections = []
    for pattern in args.sections:
        matches = [s for s in available if fnmatch.fnmatch(s, pattern)]
        matches = [s for s in matches if any(MASTERS.rglob(f"master_{s.replace('.', '_')}.py"))]
        if not matches:
            raise SystemExit(f"No extension master file matches {pattern!r}")
        sections += [s for s in matches if s not in sections]

    for section in sections:
        print(f"\n=== {section} ===")
        master = generate(section, config)
        solutions = master.exercises_dir / f"{master.colab_name}_solutions.ipynb"
        exercises = master.exercises_dir / f"{master.colab_name}_exercises.ipynb"
        media_dir = master.exercises_dir / "media"
        if args.execute:
            wanted = media_cells(nbformat.read(solutions, as_version=4))
            executed, seconds = execute(solutions, args.timeout)
            saved = save_media(executed, wanted, media_dir)
            strip_environment_noise(executed)
            nbformat.write(executed, solutions)
            solutions.with_name(solutions.stem + ".FAILED.ipynb").unlink(missing_ok=True)
            print(f"Executed solutions notebook in {seconds:.0f}s; saved {len(saved)} figure(s) to media/")
        inline_media(exercises, media_dir)
        rewrite_page_media(master)
        print(summarise(master))


if __name__ == "__main__":
    main()
