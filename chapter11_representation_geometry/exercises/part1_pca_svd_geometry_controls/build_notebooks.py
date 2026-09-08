"""Build 11.1 from its readable lesson; complete stubs only from their own answers.

Use --check while the experiment/interpretation cells are under construction.
It validates the draft without replacing the published notebooks or page.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "11.1_PCA_SVD_and_Geometry_Controls"
MARKER = re.compile(r"^<!-- ARENA:cell (markdown|code) -->\s*$", re.M)


def lesson_cells() -> list[dict]:
    source = (HERE / "lesson.md").read_text()
    if any(ord(character) < 32 and character not in "\n\t\r" for character in source):
        raise ValueError("lesson contains control characters; check LaTeX escaping")
    pieces = MARKER.split(source)
    cells = []
    for kind, text in zip(pieces[1::2], pieces[2::2], strict=True):
        text = text.strip() + "\n"
        if kind == "code":
            if not text.startswith("```python\n") or not text.rstrip().endswith("```"):
                raise ValueError("each code cell needs one Python fence")
            text = text[len("```python\n"):].rstrip()[:-3].rstrip() + "\n"
            ast.parse(text)
        cell = {"id": hashlib.sha256(f"{len(cells)}:{kind}:{text}".encode()).hexdigest()[:12],
                "cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return cells


def complete_from_answer(source: str, answer: str) -> str:
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    lines, answer_lines = source.splitlines(True), answer.splitlines(True)
    answers = {}
    for node in ast.parse(answer).body:
        if isinstance(node, definitions):
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
            answers[node.name] = "".join(answer_lines[start:node.end_lineno])
    for node in reversed(ast.parse(source).body):
        if isinstance(node, definitions):
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
            if "raise NotImplementedError" in "".join(lines[start:node.end_lineno]):
                if node.name not in answers:
                    raise ValueError(f"this exercise's solution is missing {node.name}")
                lines[start:node.end_lineno] = [answers[node.name] + "\n"]
    completed = "".join(lines)
    if "raise NotImplementedError" in completed:
        raise ValueError("an exercise was not completed from its own answer")
    ast.parse(completed)
    return completed


def complete_learner(notebook: dict) -> dict:
    result = copy.deepcopy(notebook)
    preceding_markdown = []
    for index, cell in enumerate(result["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            preceding_markdown.append(source)
            continue
        if "raise NotImplementedError" in source:
            blocks = re.findall(r"<summary>Solution</summary>(.*?)</details>",
                                "\n".join(preceding_markdown), re.S)
            if len(blocks) != 1:
                raise ValueError(f"cell {index} must have exactly one preceding exercise answer")
            answer = "\n\n".join(re.findall(r"```python\n(.*?)```", blocks[0], re.S))
            cell["source"] = complete_from_answer(source, answer).splitlines(True)
        preceding_markdown = []
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--draft-output", type=Path)
    args = parser.parse_args()
    previous = json.loads((HERE / f"{STEM}_exercises.ipynb").read_text())
    learner = {**previous, "cells": lesson_cells()}
    solved = complete_learner(learner)
    if args.draft_output:
        args.draft_output.mkdir(parents=True, exist_ok=False)
        for suffix, notebook in (("exercises",learner),("solutions",solved)):
            (args.draft_output/f"{STEM}_{suffix}.ipynb").write_text(json.dumps(notebook,indent=1)+"\n")
    print(f"Draft parses and completes from its own answers: {len(learner['cells'])} cells")


if __name__ == "__main__":main()
