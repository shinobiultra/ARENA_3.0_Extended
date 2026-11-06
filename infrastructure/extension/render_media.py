"""Display extension-local teaching figures without requiring a published branch."""
from __future__ import annotations

import base64
from pathlib import Path
import re

from infrastructure.extension.page_links import rewrite_page_links


_LOCAL_IMAGE = re.compile(
    r'''src=(?P<quote>["'])(?P<path>\.\./\.\./exercises/[\w./-]+/media/[\w.-]+\.(?:png|svg))(?P=quote)'''
)


def inline_page_media(text: str, markdown_path: Path) -> str:
    """Embed only local PNG/SVG references confined to this chapter's exercises."""
    markdown_path = Path(markdown_path).resolve()
    text = rewrite_page_links(text, markdown_path)
    allowed = markdown_path.parents[2] / "exercises"

    def replace(match):
        path = (markdown_path.parent / match["path"]).resolve()
        if not path.is_relative_to(allowed):
            raise ValueError("Teaching figure resolves outside the chapter exercises")
        # A missing figure is a broken lesson, not permission to fetch a different one.
        payload = path.read_bytes()
        mime = "image/png" if path.suffix == ".png" else "image/svg+xml"
        encoded = base64.b64encode(payload).decode("ascii")
        return f'src={match["quote"]}data:{mime};base64,{encoded}{match["quote"]}'

    return _LOCAL_IMAGE.sub(replace, text)
