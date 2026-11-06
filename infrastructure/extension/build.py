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
   expected outputs). Without `--execute`, if the code cells are unchanged since the last execution (e.g. only
   markdown was edited), the previous execution's outputs are kept.
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
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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

    def master_py_to_ipynb(self, overwrite: bool = False, ignore_errors: bool = False) -> None:
        """Validate extension master compatibility in memory before its only write."""
        from scripts.practice16_execution_checkpoint import preflight_practice16_publication

        preflight_practice16_publication(
            root=ROOT, master_path=self.master_py_path,
            identity={"section": self.colab_name.split("_", 1)[0], "chapter": self.chapter_dir.name,
                      "exercise_dir": self.exercise_dir_name, "page": self.streamlit_page_name},
        )
        from conversion.converters import _cells_to_notebook_data

        from infrastructure.extension.master_notebook_schema import canonical_master_notebook

        if self.master_path.exists() and not overwrite:
            raise FileExistsError(f"File {self.master_path} already exists")
        # Retain the upstream signature; ignore_errors never bypasses schema checks.
        content = canonical_master_notebook(
            _cells_to_notebook_data(self.cells), master_basename=self.master_py_path.name
        )
        self.master_path.write_text(content, encoding="utf-8")

    def create_arena_files(self, verbose: bool = False) -> None:
        """Apply only an exact reviewed preservation recipe before any file write."""
        from scripts.practice16_execution_checkpoint import restore_practice16_notebooks as restore_notebooks

        super().create_arena_files(verbose)
        restored = restore_notebooks(
            master_path=self.master_py_path,
            identity={"section": self.colab_name.split("_", 1)[0],
                      "chapter": self.chapter_dir.name,
                      "exercise_dir": self.exercise_dir_name,
                      "page": self.streamlit_page_name},
            raw_notebooks={"exercises": self.colab_ex_cells, "solutions": self.colab_soln_cells},
            raw_modules={name: value if isinstance(value, str) else "".join(value)
                         for name, value in self.solutions_py_files.items()},
            existing_notebooks={kind: path.read_text() if path.exists() else None
                                for kind in ("exercises", "solutions")
                                for path in [self.exercises_dir / f"{self.colab_name}_{kind}.ipynb"]},
            root=ROOT,
        )
        self.source_preservation_applied = restored is not None
        if restored is not None:
            self.colab_ex_cells = restored["exercises"]
            self.colab_soln_cells = restored["solutions"]

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
        checkpoint_path = ROOT / "mastery_hub" / "extension_checkpoints.json"
        if checkpoint_path.exists():
            checkpoints = json.loads(checkpoint_path.read_text(encoding="utf-8"))["checkpoints"]
            section = self.colab_name.split("_", 1)[0]
            if any(checkpoint["section"] == section for checkpoint in checkpoints):
                anchor = "section-" + section.replace(".", "-")
                workbook = "../../../mastery_hub/units/extension_predictions/exercises.ipynb"
                colab_lines[2:2] = [
                    f"> **Prediction practice: [start the guided workbook]({workbook}#{anchor}) before viewing this lesson's results.**",
                    "> Record a forecast, confidence and reasoning, then compare controls and predict a changed case. These reference-rich lessons support guided practice; prior exposure is not scored as unseen mastery.",
                    "",
                ]
                streamlit_lines[2:2] = [
                    "> **Guided prediction practice:** open `mastery_hub/units/extension_predictions/exercises.ipynb` in the checkout before viewing the results. Record confidence and reasoning, compare controls, then predict a changed case. Reference results are practice, not an unseen test.",
                    "",
                ]
        # Keep long paired links and guidance readable without splitting URLs.
        def readable_quotes(lines):
            result = []
            for line in lines:
                if line.startswith("> ") and len(line) > 240:
                    result.extend(textwrap.wrap(
                        line[2:], width=240, initial_indent="> ", subsequent_indent="> ",
                        break_long_words=False, break_on_hyphens=False,
                    ))
                else:
                    result.append(line)
            return result

        return readable_quotes(streamlit_lines), readable_quotes(colab_lines)


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
    from scripts.practice16_execution_checkpoint import preflight_practice16_publication

    preflight_practice16_publication(
        root=ROOT, master_path=master_py,
        identity={"section": section, "chapter": config["chapter_names"][chapter_num],
                  "exercise_dir": mapping["exercise_dir"], "page": mapping["streamlit_page"]},
    )
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
    prepare_instruction_page(master.instructions_dir / f"{master.streamlit_page_name}.py")
    for suffix in ("exercises", "solutions"):
        normalize_generated_notebook(master.exercises_dir / f"{master.colab_name}_{suffix}.ipynb")
    return master


def normalize_generated_notebook(path: Path) -> None:
    """Remove empty code-only fields emitted on markdown by the upstream converter.

    Apply only to generated extension notebooks; retain code outputs and reject
    unexpected nonempty fields instead of discarding possible learner evidence.
    """
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            if cell.get("execution_count") is not None or cell.get("outputs", []):
                raise ValueError(f"Non-code cell contains execution evidence: {path}")
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


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
    # A non-interactive MPLBACKEND (e.g. Agg, set on clusters) would stop the kernel from rendering figures inline
    os.environ.pop("MPLBACKEND", None)
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
        if cell.cell_type != "code":
            continue
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
    rel = f"../../exercises/{master.exercise_dir_name}/media/"
    text = page.read_text(encoding="utf-8")
    page.write_text(re.sub(r"""src=(["'])media/""", rf"src=\1{rel}", text), encoding="utf-8")


def prepare_instruction_page(page: Path) -> None:
    """Keep the generated display shell readable and resolve its local figures."""
    from scripts.format_rendered_python import format_rendered_python

    source = page.read_text(encoding="utf-8")
    original = 'markdown_content_file.read_text(encoding="utf-8").split('
    replacement = (
        'inline_page_media(markdown_content_file.read_text(encoding="utf-8"), '
        'markdown_content_file).split('
    )
    if original not in source:
        raise ValueError("Generated instruction loader changed; review its media routing")
    source = source.replace(original, replacement, 1)
    source = source.replace(
        "from st_dependencies import generate_toc, styling\n",
        "from st_dependencies import generate_toc, styling\n"
        "from infrastructure.extension.render_media import inline_page_media\n",
        1,
    )
    page.write_text(format_rendered_python(source), encoding="utf-8")


def code_sources(nb: nbformat.NotebookNode) -> list[str]:
    return [cell.source for cell in nb.cells if cell.cell_type == "code"]


def carry_over_outputs(previous: nbformat.NotebookNode | None, solutions: Path) -> bool:
    """
    If the freshly generated solutions notebook has exactly the same code cells as the previously executed one,
    copy the previous outputs across (so a markdown-only edit doesn't need re-execution). Returns True if it did.
    """
    if previous is None or not any(cell.get("outputs") for cell in previous.cells if cell.cell_type == "code"):
        return False
    fresh = nbformat.read(solutions, as_version=4)
    if code_sources(fresh) != code_sources(previous):
        return False
    # Widget views reference a notebook-level state graph, and retained execution
    # receipts describe these exact code cells. Keep that state with the outputs.
    # Generated teaching metadata wins conflicts except for runtime identity.
    from copy import deepcopy

    fresh.metadata = nbformat.from_dict({**deepcopy(previous.metadata), **fresh.metadata})
    for key in ("widgets", "kernelspec", "language_info"):
        if key in previous.metadata:
            fresh.metadata[key] = deepcopy(previous.metadata[key])
    old_code = iter(cell for cell in previous.cells if cell.cell_type == "code")
    for cell in fresh.cells:
        if cell.cell_type == "code":
            old = next(old_code)
            cell["outputs"], cell["execution_count"] = old.get("outputs", []), old.get("execution_count")
            cell.metadata = nbformat.from_dict({**deepcopy(old.metadata), **cell.metadata})
    nbformat.write(fresh, solutions)
    return True


def carry_over_exercise_provenance(previous: nbformat.NotebookNode | None, exercises: Path) -> bool:
    """Keep allowlisted teaching/provenance metadata for identical student sources.

    Generated values win every conflict, including nested metadata. Only explicit
    reference/migration provenance, arena teaching markers, Jupyter visibility and
    tags may carry. Learner answers, trust, timing, execution metadata, cell source,
    outputs and execution counts never carry. Migration's nested source snapshots
    use the same metadata allowlist rather than retaining arbitrary old state.
    """
    if previous is None:
        return False
    fresh = nbformat.read(exercises, as_version=4)
    if code_sources(previous) != code_sources(fresh):
        return False
    from copy import deepcopy

    def select(values, keys):
        return {key: deepcopy(values[key]) for key in keys if key in values}

    def arena(values):
        return select(values, ('section', 'status', 'original_chapters_modified',
                               'exercise_functions', 'calling_cell', 'provided_code', 'solution_for'))

    def cell_metadata(values):
        selected = {}
        if 'arena' in values:
            selected['arena'] = arena(values['arena'])
        if 'jupyter' in values:
            selected['jupyter'] = select(values['jupyter'], ('source_hidden', 'outputs_hidden'))
        if 'tags' in values:
            selected['tags'] = deepcopy(values['tags'])
        return selected

    def source_metadata(values):
        selected = cell_metadata(values)
        for key, fields in (
            ('kernelspec', ('display_name', 'language', 'name')),
            ('language_info', ('name', 'version', 'pygments_lexer')),
        ):
            if key in values:
                selected[key] = select(values[key], fields)
        return selected

    def notebook_metadata(values):
        selected = source_metadata(values)
        if 'fresh_reference' in values:
            selected['fresh_reference'] = select(values['fresh_reference'], (
                'executed_solution_result', 'kind', 'publication_receipt', 'section'))
        if 'migration' in values:
            migration = values['migration']
            selected['migration'] = select(migration, ('execution_kind', 'fresh_execution', 'status'))
            if 'source_notebook_metadata' in migration:
                selected['migration']['source_notebook_metadata'] = {
                    name: source_metadata(metadata)
                    for name, metadata in migration['source_notebook_metadata'].items()}
            if 'original_all_cell_metadata' in migration:
                selected['migration']['original_all_cell_metadata'] = [
                    {**select(row, ('legacy_cell', 'id', 'cell_type')),
                     'metadata': cell_metadata(row.get('metadata', {}))}
                    for row in migration['original_all_cell_metadata']]
            if 'source_cell_metadata' in migration:
                selected['migration']['source_cell_metadata'] = {
                    name: [{**select(row, ('id', 'cell_type')),
                            'metadata': cell_metadata(row.get('metadata', {}))} for row in cells]
                    for name, cells in migration['source_cell_metadata'].items()}
        return selected

    def merge_generated(old, generated):
        merged = deepcopy(old)
        for key, value in generated.items():
            merged[key] = (merge_generated(merged[key], value)
                           if isinstance(merged.get(key), dict) and isinstance(value, dict)
                           else deepcopy(value))
        return merged

    fresh.metadata = nbformat.from_dict(merge_generated(notebook_metadata(previous.metadata), fresh.metadata))
    old_code = [cell for cell in previous.cells if cell.cell_type == 'code']
    new_code = [cell for cell in fresh.cells if cell.cell_type == 'code']
    for old, new in zip(old_code, new_code, strict=True):
        new.metadata = nbformat.from_dict(merge_generated(cell_metadata(old.metadata), new.metadata))
    nbformat.write(fresh, exercises)
    return True


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
        mapping = config["conversion_map"][section]
        chapter_dir = ROOT / config["chapter_names"][int(section.split(".")[0])]
        colab_name = mapping["streamlit_page"].split("_[")[-1].replace("]", "")
        previous_path = chapter_dir / "exercises" / mapping["exercise_dir"] / f"{colab_name}_solutions.ipynb"
        previous = nbformat.read(previous_path, as_version=4) if previous_path.exists() else None
        previous_exercises_path = previous_path.with_name(previous_path.name.replace("_solutions.ipynb", "_exercises.ipynb"))
        previous_exercises = nbformat.read(previous_exercises_path, as_version=4) if previous_exercises_path.exists() else None
        master = generate(section, config)
        solutions = master.exercises_dir / f"{master.colab_name}_solutions.ipynb"
        exercises = master.exercises_dir / f"{master.colab_name}_exercises.ipynb"
        media_dir = master.exercises_dir / "media"
        if not args.execute and not master.source_preservation_applied:
            carry_over_exercise_provenance(previous_exercises, exercises)
            if carry_over_outputs(previous, solutions):
                print("Code cells unchanged: kept the previous execution's outputs.")
            elif previous is not None:
                print("WARNING: code cells changed, so the solutions notebook now has no outputs. Run with --execute.")
        if args.execute:
            wanted = media_cells(nbformat.read(solutions, as_version=4))
            executed, seconds = execute(solutions, args.timeout)
            saved = save_media(executed, wanted, media_dir)
            strip_environment_noise(executed)
            nbformat.write(executed, solutions)
            solutions.with_name(solutions.stem + ".FAILED.ipynb").unlink(missing_ok=True)
            print(f"Executed solutions notebook in {seconds:.0f}s; saved {len(saved)} figure(s) to media/")
        if args.execute or not master.source_preservation_applied:
            inline_media(exercises, media_dir)
        rewrite_page_media(master)
        print(summarise(master))


if __name__ == "__main__":
    main()
