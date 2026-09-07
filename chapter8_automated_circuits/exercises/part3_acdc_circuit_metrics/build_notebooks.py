"""Build 8.3 from its readable lesson; complete stubs only from their own answers.

Use --check while the experiment/interpretation cells are under construction.
It validates the draft without replacing the published notebooks or page.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = "8.3_ACDC_and_Circuit_Metrics"
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
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
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


def export_instruction_page():
    page = MARKER.sub("", (HERE / "lesson.md").read_text())
    page = page.replace("../../instructions/assets/", "../assets/")
    figures = {
        "<details><summary>Interpretation - why the result is convincing</summary>": [
            ("acdc_toy_ground_truth_signature.png", "Known circuit, threshold sweep and exhaustive same-size null"),
        ],
        "<details><summary>Interpretation - what the real result supports</summary>": [
            ("acdc_transformerlens_component_signature.png", "GELU-1L head-component warm-up and controls"),
        ],
        "## Try It Yourself — inspect the year-ending distribution": [
            ("acdc_greater_than_method_curves.png", "Greater-than attribution curves with all random samples"),
            ("acdc_greater_than_100_edge_graph.png", "All 100 retained EAP-IG edges, labeled by receiver port"),
        ],
        "### Freeze the choices before changing the wording": [
            ("acdc_greater_than_acdc_graph.png", "Greater-than circuit found by sequential deletion"),
        ],
        "## Is every retained edge necessary? What did we leave out?": [
            ("acdc_greater_than_frozen_transfer.png", "Four frozen circuits on new wording with both matched random nulls"),
        ],
        "## Try It Yourself — delete a named real edge": [
            ("acdc_greater_than_edge_diagnostics.png", "Retained-edge deletions and full-versus-circuit backup probes"),
        ],
        "### Respect a failed selection, then inspect its mechanism": [
            ("acdc_full_edge_thresholds.png", "Every tested IOI threshold, including failed candidates"),
        ],
        "### See every retained IOI edge": [
            ("acdc_ioi_candidate_controls.png", "IOI diagnostic candidate and every matched random control, labeled by evaluation split"),
        ],
        "## Try It Yourself — explain a name prediction": [
            ("acdc_ioi_edge_matrix.png", "Every retained IOI edge beside a same-size random circuit"),
            ("acdc_ioi_edge_diagnostics.png", "IOI edge-deletion and sampled completeness diagnostics"),
        ],
        "## What can you now defend?": [
            ("acdc_ioi_all-names_topk_attribution.png", "IOI with global top-score selection"),
            ("acdc_ioi_all-names_greedy_attribution.png", "IOI with output-frontier selection"),
            ("acdc_ioi_second-subject_greedy_attribution.png", "Repeated-subject corruption control"),
        ],
    }
    for anchor, entries in figures.items():
        if page.count(anchor) != 1:
            raise ValueError(f"Figure anchor is missing or ambiguous: {anchor}")
        images = "\n\n".join(f"![{label}](../assets/{filename})" for filename, label in entries)
        page = page.replace(anchor, images + "\n\n" + anchor)
    target = HERE.parents[1] / "instructions" / "pages" / "03_[8.3]_ACDC_and_Circuit_Metrics.md"
    target.write_text(page.strip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--page-only", action="store_true")
    args = parser.parse_args()
    previous = json.loads((HERE / f"{STEM}_exercises.ipynb").read_text())
    learner = {**previous, "cells": lesson_cells()}
    solved = complete_learner(learner)
    if args.check:
        print(f"Draft parses and completes from its own answers: {len(learner['cells'])} cells")
        return
    if not args.page_only:
        for suffix, notebook in (("exercises", learner), ("solutions", solved)):
            (HERE / f"{STEM}_{suffix}.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    export_instruction_page()
    print(f"Built {len(learner['cells'])} cells and the full instruction page")


if __name__ == "__main__":
    main()
