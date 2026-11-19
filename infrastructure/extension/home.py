"""Streamlit home page for extension chapters, laid out like the original ARENA chapter home pages.

Each chapter's `instructions/Home.py` calls `render_chapter_home(__file__, INTRO)`, where INTRO is that chapter's own
description. The TLDR above it (before, during and after the chapter on the study route) comes from the syllabus
route, and section summaries come from `infrastructure/core/config.yaml`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


def chapter_tldr(chapter: str) -> str:
    """The chapter's place on the study route, from the syllabus `route:` block (Spec #39, Study route).

    Original ARENA lessons link to learn.arena.education; this chapter's own lessons are listed below the TLDR
    with their page links, and lessons elsewhere in the extension are named.
    """
    from infrastructure.extension import route as study_route

    route = study_route.load()

    def ref(stop: study_route.Stop) -> str:
        url = route.original_url(stop)
        return f"[{stop.label}]({url})" if url else stop.label

    return study_route.chapter_tldr(route, chapter, ref)


def render_chapter_home(home_file: str, intro_markdown: str) -> None:
    chapter_dir = Path(home_file).resolve().parent.parent
    root = chapter_dir.parent
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from st_dependencies import get_chapter_content, get_displayable_sections, styling  # noqa: PLC0415

    chapter, _ = get_chapter_content(chapter_dir.name)
    sections = get_displayable_sections(chapter_dir.name)
    styling(chapter["title"].replace(":", " -"))

    tldr = chapter_tldr(chapter_dir.name)
    st.markdown(
        f'<img src="{chapter["header_image"]}" width="600">\n\n# {chapter["title"]}\n\n{tldr}\n{intro_markdown}',
        unsafe_allow_html=True,
    )

    st.markdown("## Sections in this chapter")
    section_pages = {
        section.get("number"): section.get("page_file") for section in chapter.get("sections", [])
    }
    for section in sections:
        page_file = section_pages.get(section.number)
        page = chapter_dir / "instructions" / "pages" / page_file if page_file else None
        script = page.with_suffix(".py") if page is not None else None
        if page is not None and page.is_file() and script.is_file():
            # Streamlit owns the registered route; a filesystem path is not its URL.
            st.page_link(script, label=section.name)
            st.markdown(section.description)
        else:
            st.markdown(f"- **{section.name} (page unavailable)** - {section.description}")

    st.markdown(
        """
## How to do the exercises

Each section has an **exercises** notebook (with the functions you'll implement left blank) and a **solutions**
notebook (saved with all of its outputs, so you can always see what a cell should produce). Clone the repo, install
the environment, then open the exercises notebook from its directory:

```bash
git clone https://github.com/shinobiultra/ARENA_3.0_Extended.git
cd ARENA_3.0_Extended
uv sync
uv run jupyter lab
```

Most sections run comfortably on a single consumer GPU, and many run on a CPU with the smaller settings described in
the notebook. Each section's setup code tells you which models it downloads.
""",
        unsafe_allow_html=True,
    )
