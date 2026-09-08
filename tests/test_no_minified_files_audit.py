import scripts.audit_no_minified_files as audit
import json


def test_no_minified_audit_flags_long_python_line(tmp_path):
    path = tmp_path / "generated.py"
    path.write_text("x = '" + ("a" * 241) + "'\n")

    blockers = audit.minified_file_blockers([path])

    assert blockers == [f"{path}:1: line length 247 exceeds 240"]


def test_no_minified_audit_allows_markdown_tables_and_urls(tmp_path):
    table_path = tmp_path / "table.md"
    table_path.write_text("| " + ("column | " * 80) + "\n")
    url_path = tmp_path / "url.md"
    url_path.write_text("See https://example.com/" + ("a" * 500) + "\n")

    blockers = audit.minified_file_blockers([table_path, url_path])

    assert blockers == []


def test_no_minified_audit_flags_one_line_notebook_json(tmp_path):
    path = tmp_path / "collapsed.ipynb"
    path.write_text('{"cells":[' + (" " * 6000) + "]}")

    blockers = audit.minified_file_blockers([path])

    assert blockers == [f"{path}: notebook JSON is collapsed into too few lines"]


def test_no_minified_audit_flags_notebook_without_cells(tmp_path):
    path = tmp_path / "empty.ipynb"
    path.write_text('{\n "cells": []\n}\n')

    blockers = audit.minified_file_blockers([path])

    assert blockers == [f"{path}: notebook has no readable cells"]


def test_current_extension_sources_are_not_minified():
    assert audit.minified_file_blockers() == []


def test_rendered_images_are_not_minified_source(tmp_path):
    path = tmp_path / "visible_result.ipynb"
    path.write_text(json.dumps({"cells": [{
        "source": ["display(figure)\n"],
        "outputs": [{"data": {"image/png": "a" * 6000}}],
        "attachments": {"photo": {"image/jpeg": "b" * 6000}},
    }]}, indent=1))
    assert audit.minified_file_blockers([path]) == []


def test_image_output_does_not_exempt_long_source(tmp_path):
    path = tmp_path / "minified_source.ipynb"
    path.write_text(json.dumps({"cells": [{
        "source": ["x = '" + "a" * 6000 + "'"],
        "outputs": [{"data": {"image/png": "a" * 6000}}],
    }]}, indent=1))
    blockers = audit.minified_file_blockers([path])
    assert len(blockers) == 1 and "notebook line length" in blockers[0]


def test_frozen_protocol_requires_identical_readable_mirror(tmp_path):
    import hashlib
    import json
    import textwrap
    from scripts.audit_no_minified_files import minified_file_blockers

    path = tmp_path / 'protocol.md'
    original = 'Frozen experimental decision. ' * 30
    path.write_text(original)
    (tmp_path / 'protocol_freeze.json').write_text(json.dumps({'sha256': {'protocol.md': hashlib.sha256(path.read_bytes()).hexdigest()}}))
    assert minified_file_blockers([path])
    mirror = tmp_path / 'protocol_readable.md'
    mirror.write_text(textwrap.fill(original, width=80))
    assert not minified_file_blockers([path, mirror])
    mirror.write_text(textwrap.fill(original.replace('decision', 'changed'), width=80))
    assert minified_file_blockers([path, mirror])
    mirror.write_text(textwrap.fill(original, width=80))
    path.write_text(original + 'Different protocol.')
    assert minified_file_blockers([path, mirror])
