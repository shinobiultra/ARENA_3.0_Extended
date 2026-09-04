import scripts.audit_no_minified_files as audit


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
