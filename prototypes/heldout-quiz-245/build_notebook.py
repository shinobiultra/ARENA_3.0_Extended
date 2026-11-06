"""PROTOTYPE for issue 245: turn 12.1_heldout_quiz.md into an executed notebook.

    uv run --no-project --with nbformat --with nbclient --with ipykernel --with numpy --with matplotlib python build_notebook.py

```python blocks become code cells and everything else becomes markdown. Cells that need the real 12.1
lesson (training runs) are kept but not executed. The answer cell is filled with an example learner's
answers so that the score table has something to show.
"""

import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).parent
NEEDS_LESSON = ("train_tiny_clip(", "fresh_model(")

STAND_IN = '''# 🛠 Prototype only: a NumPy stand-in for the zero_shot_probabilities you wrote in Part 3.
import numpy as np

def zero_shot_probabilities(logits, kind):
    logits = np.asarray(logits, dtype=float)
    if kind == "softmax":
        e = np.exp(logits - logits.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    return 1 / (1 + np.exp(-logits))'''

EXAMPLE = {
    '"Q1a_softmax": "",': '"Q1a_softmax": "falls",',
    '"Q1a_sigmoid": "",': '"Q1a_sigmoid": "unchanged",',
    '"Q1b": "",': '"Q1b": "b",',
    '"Q2": "",': '"Q2": "D = C < A = B",',
    '"Q3": "",': '"Q3": "d",',
    '"Q4": "",': '"Q4": "e",',
}


def main() -> None:
    text = (HERE / "12.1_heldout_quiz.md").read_text()
    text = re.sub(r'<img src="media/quiz_reveal.png"[^>]*>\n', "", text)  # the plot cell draws it
    cells = [nbformat.v4.new_code_cell(STAND_IN)]
    for i, part in enumerate(re.split(r"```python\n(.*?)\n```", text, flags=re.S)):
        part = part.strip("\n")
        if not part.strip():
            continue
        if i % 2 == 0:
            cells.append(nbformat.v4.new_markdown_cell(part))
            continue
        if part.startswith("# Fill in every answer"):
            part = "# 🛠 Prototype: filled with an example learner's answers.\n" + part
            for blank, filled in EXAMPLE.items():
                part = part.replace(blank, filled)
        cell = nbformat.v4.new_code_cell(part)
        if any(marker in part for marker in NEEDS_LESSON):
            cell.metadata["tags"] = ["needs-lesson-not-run"]
        cells.append(cell)

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    client = NotebookClient(nb, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and "needs-lesson-not-run" not in cell.metadata.get("tags", []):
                client.execute_cell(cell, index)
    nbformat.write(nb, HERE / "12.1_heldout_quiz.ipynb")


if __name__ == "__main__":
    main()
