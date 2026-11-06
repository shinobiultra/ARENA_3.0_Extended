"""Local teaching figures must work before an extension branch is published."""
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from infrastructure.extension.render_media import inline_page_media
from infrastructure.extension.build import prepare_instruction_page, rewrite_page_media


def setup_media(tmp_path):
    chapter = tmp_path / "chapter5_modern_architectures"
    pages = chapter / "instructions/pages"
    media = chapter / "exercises/part1_gemma_from_scratch/media"
    pages.mkdir(parents=True)
    media.mkdir(parents=True)
    return chapter, pages, media


def test_generated_markdown_uses_local_png_and_svg_routes(tmp_path):
    chapter, pages, media = setup_media(tmp_path)
    path = pages / "01_[5.1]_Gemma_from_Scratch.md"
    path.write_text('<img src="media/curve.png"><img src=\'media/decoder.svg\'>')
    master = SimpleNamespace(instructions_dir=pages, streamlit_page_name=path.stem,
        chapter_dir=chapter, exercise_dir_name=media.parent.name)
    rewrite_page_media(master)
    assert path.read_text() == (
        '<img src="../../exercises/part1_gemma_from_scratch/media/curve.png">'
        "<img src='../../exercises/part1_gemma_from_scratch/media/decoder.svg'>")


def test_streamlit_embeds_exact_local_image_bytes_without_remote_fetch(tmp_path):
    _, pages, media = setup_media(tmp_path)
    png, svg = b"\x89PNG\r\n\x1a\nfixture", b'<svg xmlns="http://www.w3.org/2000/svg"/>'
    (media / "curve.png").write_bytes(png)
    (media / "decoder.svg").write_bytes(svg)
    path = pages / "lesson.md"
    text = ('<img src="../../exercises/part1_gemma_from_scratch/media/curve.png" width="400">'
            "<img src='../../exercises/part1_gemma_from_scratch/media/decoder.svg'>"
            '<img src="https://example.test/reference.png">')
    rendered = inline_page_media(text, path)
    assert 'data:image/png;base64,' + base64.b64encode(png).decode() in rendered
    assert 'data:image/svg+xml;base64,' + base64.b64encode(svg).decode() in rendered
    assert 'width="400"' in rendered
    assert '<img src="https://example.test/reference.png">' in rendered
    assert inline_page_media(rendered, path) == rendered


def test_missing_images_and_symlink_escape_fail_explicitly(tmp_path):
    _, pages, media = setup_media(tmp_path)
    path = pages / "lesson.md"
    text = '<img src="../../exercises/part1_gemma_from_scratch/media/missing.png">'
    with pytest.raises(FileNotFoundError):
        inline_page_media(text, path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (media / "missing.png").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        inline_page_media(text, path)


def test_display_helper_import_follows_project_path_setup(tmp_path):
    page = tmp_path / "page.py"
    page.write_text(
        'import sys\nimport streamlit as st\n'
        'sys.path.append(str(arena_root_dir))\n'
        'from st_dependencies import generate_toc, styling\n'
        'st.session_state["content"] = markdown_content_file.read_text(encoding="utf-8").split("=== NEW CHAPTER ===")\n'
    )
    prepare_instruction_page(page)
    source = page.read_text()
    assert source.index("sys.path.append") < source.index("from infrastructure.extension.render_media")
    assert 'inline_page_media(markdown_content_file.read_text(encoding="utf-8"), markdown_content_file)' in source
    compile(source, str(page), "exec")
