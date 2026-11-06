"""Add four dated cautions to exact reading routes; preserve the lesson text."""
from __future__ import annotations

from pathlib import Path
import re


_TARGETS = {
    'chapter8_automated_circuits/instructions/pages/06_[8.6]_Sparse_Feature_Circuits_and_SHIFT.md': (
        '8-6-node-metrics', '### Exercise - implement `faithfulness_minimality_completeness_report`',
    ),
    'chapter8_automated_circuits/instructions/pages/07_[8.7]_Circuit_Tracing_with_Attribution_Graphs.md': (
        '8-7-primary-reading', '## Reading Links',
    ),
    'chapter9_alignment_interpretability/instructions/pages/03_[9.3]_Chain-of-Thought_Faithfulness.md': (
        '9-3-primary-reading', '## Reading links',
    ),
    'chapter9_alignment_interpretability/instructions/pages/04_[9.4]_Emergent_Misalignment_Detection.md': (
        '9-4-primary-reading', '## Limitations',
    ),
    'chapter9_alignment_interpretability/instructions/pages/06_[9.6]_Model_Editing_with_ROME_and_MEMIT.md': (
        '9-6-reading-boundary', '## Reading',
    ),
    "chapter13_image_generation_interpretability/instructions/pages/02_[13.2]_Denoising-Time_Causal_Patching.md": (
        "13-2-reference-check", "### Exercise 7: Sweep the Known Circuit",
    ),
    "chapter14_jepa_world_models/instructions/pages/01_[14.1]_I-JEPA_from_Scratch.md": (
        "14-1-conflicting-signature", "## Signature Result: Does Visible Context Determine the Masked Latent?",
    ),
    "chapter15_peft_misalignment/instructions/pages/03_[15.3]_QLoRA_from_Scratch.md": (
        "15-3-failed-parity", "## 10. Train the Marker Rule on TinyStories",
    ),
    "chapter17_training_dynamics/instructions/pages/01_[17.1]_Checkpoint_Archaeology_and_Grokking.md": (
        "17-1-finite-domain", "## Core Question",
    ),
}
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _key(markdown_path: Path) -> str:
    return Path(*Path(markdown_path).parts[-4:]).as_posix()


def has_study_cautions(markdown_path: Path) -> bool:
    return _key(markdown_path) in _TARGETS


def _anchor_end(text: str, anchor: str) -> int:
    """Find one exact unquoted heading line outside backtick/tilde code fences."""
    offsets = []
    offset = 0
    fence = None
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        match = _FENCE.fullmatch(content)
        if fence is not None:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
        elif match:
            fence = match[1]
        elif content == anchor:
            offsets.append(offset + len(anchor))
        offset += len(line)
    # Lessons are editable: a moved or removed anchor drops the caution instead of breaking the page.
    return offsets[0] if len(offsets) == 1 else None


def add_study_cautions(text: str, markdown_path: Path, notes_text: str) -> str:
    """Pure additive projection; unknown paths return unchanged without parsing notes."""
    target = _TARGETS.get(_key(markdown_path))
    if target is None:
        return text
    note_id, anchor = target
    offset = _anchor_end(text, anchor)
    if offset is None:
        return text
    start = f"<!-- study-caution-source:{note_id}:start -->"
    end = f"<!-- study-caution-source:{note_id}:end -->"
    if notes_text.count(start) != 1 or notes_text.count(end) != 1:
        raise ValueError(f"Study-caution note {note_id!r} requires one start and end")
    begin = notes_text.index(start) + len(start)
    finish = notes_text.index(end)
    if finish < begin:
        raise ValueError(f"Study-caution note {note_id!r} markers are out of order")
    body = notes_text[begin:finish].strip()
    if not body or "<!-- study-caution" in body or any(_FENCE.fullmatch(line) for line in body.splitlines()):
        raise ValueError(f"Study-caution note {note_id!r} must contain plain reading prose")
    block = f"\n\n<!-- study-caution:{note_id}:start -->\n\n{body}\n\n<!-- study-caution:{note_id}:end -->\n"
    if "<!-- study-caution:" in text:
        if text.count("<!-- study-caution:") == 2 and text.count(block) == 1 and text.startswith(block, offset):
            return text
        raise ValueError(f"Study-caution annotation in {markdown_path} is partial, changed or misplaced")
    return text[:offset] + block + text[offset:]
