"""Create only missing notebooks; never overwrite filled or recorded work."""
import json
from pathlib import Path

try:
    from .lesson import notebook_cells
except ImportError:
    from lesson import notebook_cells


def build_notebook(worked=False):
    cells = []
    for index, (kind, source) in enumerate(notebook_cells(worked)):
        cell = {"cell_type": kind, "id": f"metric-{index:02d}", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "arena_companion": {"scope": "6.5 CPU metric contracts only",
                            "historical_results_recomputed": False}},
        "nbformat": 4, "nbformat_minor": 5}


def generate(destination=None):
    destination = Path(destination) if destination else Path(__file__).parent
    destination.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, worked in [("exercises.ipynb", False), ("solutions.ipynb", True)]:
        path = destination / name
        try:
            with path.open("x") as handle:
                json.dump(build_notebook(worked), handle, indent=1, ensure_ascii=False)
                handle.write("\n")
            result[name] = "created"
        except FileExistsError:
            result[name] = "preserved existing file (including answers and outputs)"
    return result


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
