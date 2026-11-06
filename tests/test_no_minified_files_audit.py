import scripts.audit_no_minified_files as audit
import json
import base64
import io

import pytest
from PIL import Image


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


def _image_html():
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(image, format="PNG", compress_level=0)
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" width="750">'


def _html_notebook(path, html, *, source=None, cell_type="code", output_type="display_data"):
    path.write_text(json.dumps({"cells": [{
        "cell_type": cell_type,
        "source": source or ["display(figure)\n"],
        "outputs": [{"output_type": output_type, "data": {"text/html": html}}],
    }]}, indent=1))


@pytest.mark.parametrize("as_list", [False, True])
def test_html_output_allows_verified_image_data_url(tmp_path, as_list):
    html = _image_html()
    path = tmp_path / "image.ipynb"
    _html_notebook(path, [html] if as_list else html)
    assert audit.minified_file_blockers([path]) == []


@pytest.mark.parametrize("mutation", [
    lambda html: html.replace("base64,", "base64,!"),
    lambda html: html.replace("image/png", "image/jpeg"),
    lambda html: html.replace("iVBOR", "AAAAA"),
    lambda html: html.replace('width="750"', 'onerror="alert(1)"'),
    lambda html: html.replace('width="750"', 'style="background:url(javascript:alert(1))"'),
    lambda html: html + "<script>alert(1)</script>",
    lambda html: "Some minified result text " + html,
    lambda html: "<div>" + html + "</div>",
    lambda html: html + "<!-- hidden content -->",
    lambda html: html[:-1],
    lambda html: html.replace("image/png;base64,", "image/svg+xml;base64,"),
])
def test_html_image_exception_rejects_malformed_or_mixed_content(tmp_path, mutation):
    path = tmp_path / "invalid.ipynb"
    _html_notebook(path, [mutation(_image_html())])
    assert audit.minified_file_blockers([path])


def test_base64_text_is_not_an_image_even_with_correct_mime(tmp_path):
    encoded = base64.b64encode(b"not an image" * 1000).decode("ascii")
    path = tmp_path / "text.ipynb"
    _html_notebook(path, [f'<img src="data:image/png;base64,{encoded}">'])
    assert audit.minified_file_blockers([path])


def test_corrupt_png_checksum_is_a_blocker_instead_of_an_audit_crash(tmp_path):
    html = _image_html()
    encoded = html.split("base64,", 1)[1].split('"', 1)[0]
    image = bytearray(base64.b64decode(encoded))
    chunk = image.index(b"IDAT")
    length = int.from_bytes(image[chunk - 4:chunk], "big")
    image[chunk + 4 + length] ^= 1  # damage CRC, retaining signature and IEND
    html = html.replace(encoded, base64.b64encode(image).decode("ascii"))
    path = tmp_path / "corrupt_png.ipynb"
    _html_notebook(path, [html])
    assert audit.minified_file_blockers([path])


def test_valid_html_output_does_not_exempt_identical_source(tmp_path):
    html = _image_html()
    path = tmp_path / "source.ipynb"
    _html_notebook(path, [html], source=[html])
    blockers = audit.minified_file_blockers([path])
    assert len(blockers) == 1 and "notebook line length" in blockers[0]


def test_valid_image_in_mixed_html_list_remains_blocked(tmp_path):
    path = tmp_path / "mixed.ipynb"
    _html_notebook(path, [_image_html(), "<script>alert(1)</script>"])
    assert audit.minified_file_blockers([path])


def test_html_output_allows_multiple_verified_images_only(tmp_path):
    path = tmp_path / "multiple_images.ipynb"
    _html_notebook(path, [_image_html() + "\n", _image_html()])
    assert audit.minified_file_blockers([path]) == []


def test_duplicate_html_key_does_not_exempt_different_unreadable_content(tmp_path):
    path = tmp_path / "duplicate.ipynb"
    _html_notebook(path, [_image_html()])
    text = path.read_text().replace(
        '"text/html": [',
        '"text/html": [\n' + json.dumps("minified text " * 1000) + '\n],\n"text/html": [',
        1,
    )
    path.write_text(text)
    assert len(audit.minified_file_blockers([path])) == 1


@pytest.mark.parametrize("cell_type,output_type", [("markdown", "display_data"), ("code", "stream")])
def test_image_html_requires_actual_code_cell_display_output(tmp_path, cell_type, output_type):
    path = tmp_path / "wrong_location.ipynb"
    _html_notebook(path, [_image_html()], cell_type=cell_type, output_type=output_type)
    assert audit.minified_file_blockers([path])


def test_image_html_attachment_is_not_an_output_exception(tmp_path):
    path = tmp_path / "attachment.ipynb"
    path.write_text(json.dumps({"cells": [{
        "cell_type": "markdown", "source": ["Attachment"],
        "attachments": {"photo": {"text/html": [_image_html()]}},
    }]}, indent=1))
    assert audit.minified_file_blockers([path])


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
