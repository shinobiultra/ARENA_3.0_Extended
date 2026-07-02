#!/usr/bin/env python3
"""
Lightweight ARENA learner-surface audit.

This is intentionally heuristic. It catches obvious failures such as notebooks
whose only signature result is a JSON verification report, notebooks with no
play cells, and notebooks whose main outputs are dictionaries rather than
learner-facing plots/tables/examples.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_text(path: Path) -> str:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        parts: list[str] = []
        for cell in nb.get("cells", []):
            src = "".join(cell.get("source", []))
            parts.append(f"\n<!-- {cell.get('cell_type')} -->\n{src}")
        return "\n".join(parts)
    return path.read_text(encoding="utf-8")


def audit(path: Path) -> tuple[list[str], list[str]]:
    text = read_text(path)
    lower = text.lower()
    failures: list[str] = []
    warnings: list[str] = []

    required = {
        "one-sentence claim / core question": ["by the end of this notebook", "core question"],
        "learning objectives": ["learning objectives"],
        "exercise blocks": ["exercise -", "### exercise"],
        "expected outputs": ["expected output"],
        "solution dropdown/sketch": ["<summary>solution", "solution sketch"],
        "help / interpretation": ["<summary>help", "interpreting the result", "interpretation"],
        "controls/baselines": ["control", "baseline"],
        "limitations": ["limitations"],
        "try it yourself / play": ["try it yourself", "play with", "change the prompt", "change the layer", "change this"],
    }
    for label, needles in required.items():
        if not any(n in lower for n in needles):
            failures.append(f"missing {label}")

    if "signature result" not in lower:
        failures.append("missing Signature Result section")

    # Suspicious report-only pattern.
    sig_index = lower.find("signature result")
    if sig_index != -1:
        sig_text = lower[sig_index : sig_index + 3000]
        if "verification_report.json" in sig_text and not any(
            marker in sig_text for marker in ["plt.", "imshow", "plotly", "heatmap", "table", "image", "graph"]
        ):
            failures.append("signature result appears to be verification_report-only")

    if "verification_report.json" in lower and lower.count("verification_report.json") >= 1:
        warnings.append("uses verification_report.json; ensure this is supporting evidence, not the lesson")

    if re.search(r"return\s*\{[^\n]*\}", text):
        warnings.append("contains dict-return pattern; ensure main result is not just a dict")

    if "raise notimplementederror" in lower and "tests.test_" not in lower:
        failures.append("has stubs but no visible tests")

    code_core_hidden = any(s in lower for s in ["from part", "import solutions", "import utils"])
    if code_core_hidden:
        warnings.append("imports external solutions/utils/tests; ensure core implementation remains in notebook")

    visual_markers = ["plt.", "imshow", "plotly", "px.", "display(", "image", "heatmap", "graph", "bar("]
    if not any(v in lower for v in visual_markers):
        warnings.append("no obvious visual/table output marker")

    return failures, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    any_fail = False
    for path in args.paths:
        failures, warnings = audit(path)
        print(f"\n== {path} ==")
        for w in warnings:
            print(f"WARN: {w}")
        for f in failures:
            print(f"FAIL: {f}")
        if not failures:
            print("PASS: no blocking learner-surface failures found by heuristic audit")
        any_fail = any_fail or bool(failures)
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
