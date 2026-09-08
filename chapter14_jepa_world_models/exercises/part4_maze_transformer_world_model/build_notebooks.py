"""Build both maze notebooks and the instruction page from readable lesson source."""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import re

import nbformat

SECTION = Path(__file__).resolve().parent
ROOT = SECTION.parents[2]
PAGE = SECTION.parents[1] / 'instructions/pages/04_[14.4]_Maze_Transformers_and_Latent_Planning.md'
SOURCE = (SECTION / 'solutions.py').read_text()
SOURCE_LINES = SOURCE.splitlines(keepends=True)
DEFINITIONS = {
    node.name: node for node in ast.parse(SOURCE).body
    if isinstance(node, (ast.FunctionDef, ast.ClassDef))
}


def definition(name: str, *, stub: bool = False) -> str:
    node = DEFINITIONS[name]
    if stub:
        node = deepcopy(node)
        methods = [item for item in node.body if isinstance(item, ast.FunctionDef)] if isinstance(node, ast.ClassDef) else [node]
        for method in methods:
            doc = method.body[:1] if ast.get_docstring(method) is not None else []
            method.body = doc + [ast.Raise(
                exc=ast.Call(func=ast.Name(id='NotImplementedError', ctx=ast.Load()),
                             args=[ast.Constant(value='Implement this method, then run its tests.')], keywords=[]),
                cause=None,
            )]
        return ast.unparse(ast.fix_missing_locations(node))
    start = min([node.lineno, *[item.lineno for item in node.decorator_list]]) - 1
    return ''.join(SOURCE_LINES[start:node.end_lineno]).rstrip()


def definitions(names: str, *, stub: bool = False) -> str:
    return '\n\n\n'.join(definition(name.strip(), stub=stub) for name in names.split(','))


def build_cells(*, solved: bool):
    chunks = re.split(r'<!-- CELL: (markdown|code) -->\n', (SECTION / 'lesson.md').read_text())
    cells = []
    for kind, source in zip(chunks[1::2], chunks[2::2], strict=True):
        source = source.strip()
        if kind == 'markdown':
            source = re.sub(r'\{\{solution:([^}]+)\}\}',
                            lambda match: '```python\n' + definitions(match[1]) + '\n```', source)
            for block in re.findall(r'```python\n(.*?)```', source, re.S):
                ast.parse(block)
            cell = nbformat.v4.new_markdown_cell(source)
        else:
            source = re.sub(r'^# (EXERCISE|PROVIDED): (.+)$',
                            lambda match: definitions(match[2], stub=match[1] == 'EXERCISE' and not solved),
                            source, flags=re.M)
            ast.parse(source)
            cell = nbformat.v4.new_code_cell(source)
        cell.id = f'maze-14-4-{len(cells):03d}'
        cells.append(cell)
    return cells


def main():
    for solved, suffix in ((False, 'exercises'), (True, 'solutions')):
        path = SECTION / f'14.4_Maze_Transformers_and_Latent_Planning_{suffix}.ipynb'
        old = nbformat.read(path, as_version=4) if path.exists() else None
        notebook = nbformat.v4.new_notebook(cells=build_cells(solved=solved), metadata={
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
        })
        if old is not None:
            prior = {cell.id: cell for cell in old.cells if cell.cell_type == 'code'}
            for cell in notebook.cells:
                previous = prior.get(cell.id)
                if previous is not None and previous.source == cell.source:
                    cell.outputs, cell.execution_count = previous.outputs, previous.execution_count
        nbformat.write(notebook, path)
    rendered = [cell.source if cell.cell_type == 'markdown' else '```python\n' + cell.source + '\n```'
                for cell in build_cells(solved=False)]
    PAGE.write_text('\n\n'.join(rendered).replace('../../instructions/assets/', '../assets/') + '\n')


if __name__ == '__main__':
    main()
