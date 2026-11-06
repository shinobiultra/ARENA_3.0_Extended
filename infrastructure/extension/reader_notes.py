"""Compose dated reading notes without changing the preserved lesson source."""
from __future__ import annotations

from pathlib import Path
import re


_NOTES_LINK = "../../../docs/mastery/pilot_reader_notes_20261023.md"
_ADDON_LINK = "../../../mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb"
# Exact chapter/path/filename matches; no chapter-wide or section-number fallback.
_TARGETS = {
    "chapter7_activation_to_language/instructions/pages/01_[7.1]_Logit_Lens_Tuned_Lens_and_Patchscopes.md": (
        ("7-1-interpretation", "<summary>Help - how should I interpret this grid?</summary>", "after"),
        ("7-1-controls-reflection", "### Exercise - controls for the Patchscope", "after"),
        ("7-1-summary", "## Summary of 3️⃣", "after"),
    ),
    "chapter12_vlm_interpretability/instructions/pages/03_[12.3]_Mini_VLM_from_Scratch.md": (
        ("12-3-recovery", "### Exercise - the recovery metric", "before"),
    ),
}


def _key(markdown_path: Path) -> str:
    return Path(*Path(markdown_path).parts[-4:]).as_posix()


def has_reader_notes(markdown_path: Path) -> bool:
    return _key(markdown_path) in _TARGETS


def add_reader_notes(text: str, markdown_path: Path, notes_text: str) -> str:
    """Return annotated text; file reads and writes belong to the caller, not this helper."""
    targets = _TARGETS.get(_key(markdown_path))
    if targets is None:
        return text

    blocks = []
    for note_id, anchor, side in targets:
        # Whole lines prevent a similar heading or quoted sentence from being an anchor.
        anchors = list(re.finditer(r"(?m)^" + re.escape(anchor) + r"$", text))
        if len(anchors) != 1:
            raise ValueError(f"Reader-note anchor {anchor!r} expected once in {markdown_path}; found {len(anchors)}; review the dated note placement")
        start = f"<!-- reader-note:{note_id}:start -->"
        end = f"<!-- reader-note:{note_id}:end -->"
        if notes_text.count(start) != 1 or notes_text.count(end) != 1:
            raise ValueError(f"Reader-note note section {note_id!r} expected once; review the dated notes source")
        begin = notes_text.index(start) + len(start)
        finish = notes_text.index(end)
        if finish < begin:
            raise ValueError(f"Reader-note note section {note_id!r} expected once in order; review the dated notes source")
        body = notes_text[begin:finish].strip()
        body += f"\n\n[Download these reading notes]({_NOTES_LINK}). Checkout: `docs/mastery/pilot_reader_notes_20261023.md`."
        if note_id == "12-3-recovery":
            body += f"\n\n[Download the checked recovery practice notebook]({_ADDON_LINK}). Checkout: `mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb`."
        block = f"\n<!-- pilot-reader-note:{note_id}:start -->\n\n{body}\n\n<!-- pilot-reader-note:{note_id}:end -->\n"
        blocks.append((anchors[0].start() if side == "before" else anchors[0].end(), block))

    # Repeated page rendering is harmless; partial or changed annotations need review.
    if "<!-- pilot-reader-note:" in text:
        if all(text.count(block) == 1 for _, block in blocks) and text.count("<!-- pilot-reader-note:") == 2 * len(blocks):
            return text
        raise ValueError(f"Reader-note partial or changed annotations in {markdown_path}; review the source before rendering")
    result = text
    for offset, block in sorted(blocks, reverse=True):
        result = result[:offset] + block + result[offset:]
    return result
