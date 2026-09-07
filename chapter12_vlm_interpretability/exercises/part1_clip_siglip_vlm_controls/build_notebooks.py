"""Build 12.1 from its readable lesson, and complete only its own exercise stubs."""

from __future__ import annotations

import ast
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "12.1_CLIP_SigLIP_and_VLM_Controls"
MARKER = re.compile(r"^<!-- ARENA:cell (markdown|code) -->\s*$", re.M)


def lesson_cells() -> list[dict]:
    pieces = MARKER.split((HERE / "lesson.md").read_text())
    cells = []
    for kind, text in zip(pieces[1::2], pieces[2::2], strict=True):
        text = text.strip() + "\n"
        if kind == "code":
            if not text.startswith("```python\n") or not text.rstrip().endswith("```"):
                raise ValueError("Each code cell needs one Python fence")
            text = text[len("```python\n") :].rstrip()[:-3].rstrip() + "\n"
            ast.parse(text)
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return cells


def complete_from_answer(source: str, answer: str) -> str:
    """Replace stub definitions by name; retain all supplied setup and tests."""
    lines, answer_lines = source.splitlines(True), answer.splitlines(True)
    answers = {}
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.parse(answer).body:
        if isinstance(node, definitions):
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
            answers[node.name] = "".join(answer_lines[start : node.end_lineno])
    for node in reversed(ast.parse(source).body):
        if isinstance(node, definitions):
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
            if "raise NotImplementedError" in "".join(lines[start : node.end_lineno]):
                if node.name not in answers:
                    raise ValueError(f"Solution is missing {node.name}")
                lines[start : node.end_lineno] = [answers[node.name] + "\n"]
    result = "".join(lines)
    if "raise NotImplementedError" in result:
        raise ValueError("An exercise was not completed")
    return result


def complete_learner(notebook: dict) -> dict:
    notebook = copy.deepcopy(notebook)
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or "raise NotImplementedError" not in source:
            continue
        answer = None
        for following in notebook["cells"][index + 1 :]:
            text = "".join(following["source"])
            if following["cell_type"] == "markdown" and "<summary>Solution</summary>" in text:
                block = re.search(r"<summary>Solution</summary>(.*?)</details>", text, re.S).group(
                    1
                )
                answer = "\n\n".join(re.findall(r"```python\n(.*?)```", block, re.S))
                break
            if following["cell_type"] == "code" and "raise NotImplementedError" in text:
                raise ValueError(f"No answer before next exercise at cell {index}")
        if answer is None:
            raise ValueError(f"No answer for cell {index}")
        cell["source"] = complete_from_answer(source, answer).splitlines(True)
    return notebook


def export_instruction_page() -> None:
    page = MARKER.sub("", (HERE / "lesson.md").read_text())
    page = page.replace("../../instructions/assets/", "../assets/")
    figures = {
        "### Inspect actual predictions": (
            "clip_pixel_encoder_comparison.png",
            "Pixel encoder training, retrieval and controls",
        ),
        "### Try It Yourself — Change the Pixels or a Caption": (
            "clip_pixel_encoder_examples.png",
            "Fixed examples and actual classification mistakes",
        ),
        "## Zero-Shot Classification and Typographic Conflict in Pretrained CLIP": (
            "clip_siglip_live_patch_results.png",
            "Actual CLIP and SigLIP patch effects",
        ),
        "### Try It Yourself — Inspect One Word Conflict": (
            "clip_typographic_controls.png",
            "Pretrained CLIP word-overlay controls",
        ),
    }
    for heading, (filename, label) in figures.items():
        if page.count(heading) != 1:
            raise ValueError(f"Figure heading missing or ambiguous: {heading}")
        page = page.replace(heading, f"![{label}](../assets/{filename})\n\n{heading}")
    target = (
        HERE.parents[1] / "instructions" / "pages" / "01_[12.1]_CLIP_SigLIP_and_VLM_Controls.md"
    )
    target.write_text(page.strip() + "\n")


def main() -> None:
    if "--page-only" in sys.argv:
        export_instruction_page()
        return
    previous = json.loads((HERE / f"{STEM}_exercises.ipynb").read_text())
    learner = {**previous, "cells": lesson_cells()}
    solution = complete_learner(learner)
    for suffix, notebook in (("exercises", learner), ("solutions", solution)):
        (HERE / f"{STEM}_{suffix}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    export_instruction_page()
    print(f"Built {len(learner['cells'])} cells and the complete instruction page")


if __name__ == "__main__":
    main()
