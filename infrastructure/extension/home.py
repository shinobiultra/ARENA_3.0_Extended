"""Streamlit home page for extension chapters, laid out like the original ARENA chapter home pages.

Each chapter's `instructions/Home.py` calls `render_chapter_home(__file__, INTRO)`, where INTRO is that chapter's own
markdown (TLDR / recommended order / description). Section summaries come from `infrastructure/core/config.yaml`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


def render_chapter_home(home_file: str, intro_markdown: str) -> None:
    chapter_dir = Path(home_file).resolve().parent.parent
    root = chapter_dir.parent
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from st_dependencies import get_chapter_content, get_displayable_sections, styling  # noqa: PLC0415

    chapter, _ = get_chapter_content(chapter_dir.name)
    sections = get_displayable_sections(chapter_dir.name)
    styling(chapter["title"].replace(":", " -"))

    st.markdown(
        f'<img src="{chapter["header_image"]}" width="600">\n\n# {chapter["title"]}\n\n{intro_markdown}',
        unsafe_allow_html=True,
    )

    st.markdown("## Sections in this chapter")
    pages = sorted((chapter_dir / "instructions" / "pages").glob("*.md"))
    for section in sections:
        page = next((p for p in pages if f"[{section.number}]" in p.name), None)
        link = f"[{section.name}](pages/{page.stem})" if page else section.name
        st.markdown(f"- **{link}** - {section.description}")

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
