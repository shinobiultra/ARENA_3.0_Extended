"""Shared Streamlit home page renderer for extension chapters."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

import streamlit as st


def _repo_root_from_home(home_file: str | Path) -> Path:
    instructions_dir = Path(home_file).resolve().parent
    return instructions_dir.parent.parent


def _section_page_link(chapter_dir: Path, section: object) -> str:
    pages = sorted((chapter_dir / "instructions" / "pages").glob("*.md"))
    for page in pages:
        if section.number and f"[{section.number}]" in page.name:
            return f"pages/{quote(page.name)}"
    raise RuntimeError(f"Could not find instruction page for section {section.name}")


def render_extension_home(home_file: str | Path) -> None:
    """Render a compact ARENA-style home page for an extension chapter."""

    root = _repo_root_from_home(home_file)
    if str(root) not in sys.path:
        sys.path.append(str(root))

    from st_dependencies import (  # noqa: PLC0415 - root path is set above.
        generate_toc,
        get_chapter_content,
        get_displayable_sections,
        styling,
    )

    instructions_dir = Path(home_file).resolve().parent
    chapter_dir = instructions_dir.parent
    chapter_key = chapter_dir.name
    chapter, _ = get_chapter_content(chapter_key)
    sections = get_displayable_sections(chapter_key)
    title = chapter["title"]
    short_description = chapter.get("short_description") or chapter.get("description", "")
    description = chapter.get("description", short_description)
    header_image = chapter.get("header_image", "")

    styling(title.replace(":", " - "))

    section_lines = []
    for section in sections:
        link = _section_page_link(chapter_dir, section)
        section_lines.append(f"- [{section.name}]({link})")

    page_markdown = f"""
<img src="{header_image}" width="600">

# {title}

{description}

## Chapter Sections

{chr(10).join(section_lines)}

## Verification Contract

Every extension section keeps the ARENA rhythm: motivation, implementation
tasks, visible tests, expected outputs, solution code, baselines, negative
controls, VRAM-aware reports, and a narrow claim scope.

The local release checks are:

```bash
uv run pytest -q
BNB_CUDA_VERSION=130 uv run python scripts/run_extension_verification_reports.py --max-vram-gb 24
uv run python scripts/audit_report_evidence_contracts.py
uv run python scripts/audit_hard_exercise_ladders.py
```
"""

    st.sidebar.markdown(generate_toc(page_markdown), unsafe_allow_html=True)
    st.markdown(page_markdown, unsafe_allow_html=True)

    selected = st.selectbox(
        "Section summary",
        [section.name for section in sections],
        index=0,
    )
    for section in sections:
        if section.name == selected:
            st.info(f"**{section.title}**\n\n{section.description}")
            break

    st.caption(short_description)
