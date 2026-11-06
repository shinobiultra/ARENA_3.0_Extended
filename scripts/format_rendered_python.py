"""Format generated display shells without changing their Python AST.

This helper never executes source. Callers must scope it to generated instruction
pages, not scientific implementations. String contents (including markup and
whitespace) stay byte-for-byte equal as Python values.
"""
from __future__ import annotations

import ast
import io
import tokenize


def format_rendered_python(source: str, *, width: int = 180) -> str:
    """Split display literals and metadata lists, requiring full AST identity."""
    if not 40 <= width <= 180:
        raise ValueError("width must be between 40 and 180")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and any(
            len(line.rstrip("\r\n")) > width
            for line in lines[node.lineno - 1:node.end_lineno]
        ):
            raise ValueError("Formatted strings require manual review")
    replacements = []
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "metadata"
                        for target in node.targets)
                and isinstance(node.value, ast.List)
                and any(len(line.rstrip("\r\n")) > width
                        for line in lines[node.lineno - 1:node.end_lineno])):
            replacement = "metadata = [\n" + "".join(
                "    " + ast.unparse(item) + ",\n" for item in node.value.elts
            ) + "]\n"
            replacements.append((node.lineno - 1, node.end_lineno, replacement))
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.STRING or not any(
            len(line) > width for line in token.string.splitlines()
        ):
            continue
        # Formatted strings are deliberately excluded: rewriting them could
        # change interpolation semantics. Bytes are outside the display scope.
        value = ast.literal_eval(token.string)
        if not isinstance(value, str):
            raise ValueError("Only plain display strings can be formatted")
        parts = []
        # Budget encoded representation, so quotes, escapes and Unicode are
        # counted exactly as they appear in the generated source.
        for paragraph in value.splitlines(keepends=True):
            while paragraph:
                cut = min(len(paragraph), width - 8)
                while len(repr(paragraph[:cut])) + 4 > width and cut > 1:
                    cut -= 1
                boundary = paragraph.rfind(" ", 0, cut) + 1
                if boundary and cut < len(paragraph):
                    cut = boundary
                parts.append(paragraph[:cut])
                paragraph = paragraph[cut:]
        prefix = lines[token.start[0] - 1][:token.start[1]]
        suffix = lines[token.end[0] - 1][token.end[1]:]
        replacement = prefix + "(\n" + "".join(
            "    " + repr(part) + "\n" for part in parts
        ) + ")" + suffix
        start, end = token.start[0] - 1, token.end[0]
        if any(start < other_end and other_start < end
               for other_start, other_end, _ in replacements):
            raise ValueError("Overlapping formatting edits require manual review")
        replacements.append((start, end, replacement))
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines)
    if ast.dump(tree) != ast.dump(ast.parse(result)):
        raise ValueError("Formatting changed the Python AST")
    return result
