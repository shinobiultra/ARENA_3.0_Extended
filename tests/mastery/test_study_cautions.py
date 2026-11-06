"""Pure reading guidance; no lesson imports or numerical execution."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import re
import sys

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
PROJECTOR_ROOT = Path(os.environ.get("ARENA_STUDY_CAUTIONS_CANDIDATE", ROOT))
CASES = {
    '8.6': ('chapter8_automated_circuits/instructions/pages/06_[8.6]_Sparse_Feature_Circuits_and_SHIFT.md', '### Exercise - implement `faithfulness_minimality_completeness_report`', '8-6-node-metrics'),
    '8.7': ('chapter8_automated_circuits/instructions/pages/07_[8.7]_Circuit_Tracing_with_Attribution_Graphs.md', '## Reading Links', '8-7-primary-reading'),
    '9.3': ('chapter9_alignment_interpretability/instructions/pages/03_[9.3]_Chain-of-Thought_Faithfulness.md', '## Reading links', '9-3-primary-reading'),
    '9.4': ('chapter9_alignment_interpretability/instructions/pages/04_[9.4]_Emergent_Misalignment_Detection.md', '## Limitations', '9-4-primary-reading'),
    '9.6': ('chapter9_alignment_interpretability/instructions/pages/06_[9.6]_Model_Editing_with_ROME_and_MEMIT.md', '## Reading', '9-6-reading-boundary'),
    "13.2": ("chapter13_image_generation_interpretability/instructions/pages/02_[13.2]_Denoising-Time_Causal_Patching.md", "### Exercise 7: Sweep the Known Circuit", "13-2-reference-check"),
    "14.1": ("chapter14_jepa_world_models/instructions/pages/01_[14.1]_I-JEPA_from_Scratch.md", "## Signature Result: Does Visible Context Determine the Masked Latent?", "14-1-conflicting-signature"),
    "15.3": ("chapter15_peft_misalignment/instructions/pages/03_[15.3]_QLoRA_from_Scratch.md", "## 10. Train the Marker Rule on TinyStories", "15-3-failed-parity"),
    "17.1": ("chapter17_training_dynamics/instructions/pages/01_[17.1]_Checkpoint_Archaeology_and_Grokking.md", "## Core Question", "17-1-finite-domain"),
}
BLOCK = re.compile(r"\n\n<!-- study-caution:([\w-]+):start -->\n\n.*?\n\n<!-- study-caution:\1:end -->\n", re.S)


@pytest.fixture
def helper():
    path = PROJECTOR_ROOT / "infrastructure/extension/study_cautions.py"
    assert path.is_file(), "The finite study-caution composer is missing"
    spec = importlib.util.spec_from_file_location("candidate_study_cautions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def notes():
    path = PROJECTOR_ROOT / "docs/mastery/study_cautions_20261023.md"
    assert path.is_file(), "The dated study cautions are missing"
    return path.read_text(encoding="utf-8")


def actual(section):
    path = ROOT / CASES[section][0]
    return path, path.read_text(encoding="utf-8")


@pytest.mark.parametrize("section", CASES)
def test_whole_actual_page_exact_one_annotation_and_lossless_inverse(helper, notes, section):
    path, source = actual(section)
    rendered = helper.add_study_cautions(source, path, notes)
    assert BLOCK.findall(rendered) == [CASES[section][2]]
    assert BLOCK.sub("", rendered).encode() == path.read_bytes()
    assert helper.add_study_cautions(rendered, path, notes) == rendered
    anchor_end = rendered.index(CASES[section][1]) + len(CASES[section][1])
    assert BLOCK.match(rendered, anchor_end)
    assert source == path.read_text(encoding="utf-8")


@pytest.mark.parametrize("section,fragments", [
    ("13.2", ("no-argument", "reference implementation only", "causal_patch_sweep(model, clean_labels, corrupt_labels, initial_noise, schedule)", "toy_result = causal_patch_sweep(", "does not validate your sweep", "random-channel")),
    ("14.1", ("third category", "zero height", "fourth category", "full height", "seed 0", "historical axis-reading mistake", "not a fresh scientific validation", "| 2 | 1.0 | 0.0 | 1.0 |")),
    ("15.3", ("exact blockwise NF4", "0.03125 > 0.01", "unchanged", "not accepted", "random labels", "same-norm")),
    ("17.1", ("169", "training and evaluation", "held-out grokking is not established", "reload", "random-label", "random-mode")),
])
def test_specific_caution_and_concrete_next_step(helper, notes, section, fragments):
    path, source = actual(section)
    rendered = helper.add_study_cautions(source, path, notes)
    block = BLOCK.search(rendered)[0]
    for fragment in fragments:
        assert fragment in block
    assert "**Study caution — 23 October 2026.**" in block


@pytest.mark.parametrize("section", CASES)
@pytest.mark.parametrize("change", ["missing", "renamed", "duplicate", "only-fenced"])
def test_changed_or_ambiguous_actual_anchor_refuses(helper, notes, section, change):
    path, source = actual(section)
    anchor = CASES[section][1]
    if change == "missing":
        source = source.replace(anchor, "")
    elif change == "renamed":
        source = source.replace(anchor, anchor + " revised")
    elif change == "duplicate":
        source += "\n" + anchor + "\n"
    else:
        source = source.replace(anchor, "```text\n" + anchor + "\n```")
    with pytest.raises(ValueError, match="anchor"):
        helper.add_study_cautions(source, path, notes)


@pytest.mark.parametrize("fence", ["```", "~~~~", "   ```"])
def test_matching_heading_inside_fence_is_never_annotated(helper, notes, fence):
    path, source = actual("13.2")
    anchor = CASES["13.2"][1]
    code = f"\n{fence}text\n{anchor}\n{fence}\n"
    rendered = helper.add_study_cautions(code + source, path, notes)
    assert rendered.startswith(code)
    assert len(BLOCK.findall(rendered)) == 1
    assert BLOCK.sub("", rendered) == code + source


@pytest.mark.parametrize("change", ["partial", "changed", "moved", "extra"])
def test_annotation_tampering_fails_closed(helper, notes, change):
    path, source = actual("13.2")
    rendered = helper.add_study_cautions(source, path, notes)
    block = BLOCK.search(rendered)[0]
    if change == "partial":
        rendered = rendered.replace("<!-- study-caution:13-2-reference-check:end -->", "")
    elif change == "changed":
        rendered = rendered.replace("reference implementation only", "your implementation")
    elif change == "moved":
        rendered = rendered.replace(block, "") + block
    else:
        rendered += "\n<!-- study-caution:unexpected:start -->"
    with pytest.raises(ValueError, match="annotation"):
        helper.add_study_cautions(rendered, path, notes)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reversed", "empty", "fenced-body"])
def test_malformed_selected_note_refuses(helper, notes, change):
    path, source = actual("13.2")
    start = "<!-- study-caution-source:13-2-reference-check:start -->"
    end = "<!-- study-caution-source:13-2-reference-check:end -->"
    if change == "missing":
        notes = notes.replace(start, "")
    elif change == "duplicate":
        notes += start
    elif change == "reversed":
        notes = notes.replace(start, "SWAP").replace(end, start).replace("SWAP", end)
    else:
        left, right = notes.index(start) + len(start), notes.index(end)
        notes = notes[:left] + ("" if change == "empty" else "\n```\nbody\n```\n") + notes[right:]
    with pytest.raises(ValueError, match="note"):
        helper.add_study_cautions(source, path, notes)


@pytest.mark.parametrize("relative", [
    "chapter7_activation_to_language/instructions/pages/01_[7.1]_Logit_Lens_Tuned_Lens_and_Patchscopes.md",
    "chapter12_vlm_interpretability/instructions/pages/03_[12.3]_Mini_VLM_from_Scratch.md",
    "chapter1_transformer_interp/instructions/pages/01_[1.1]_Transformer_from_Scratch.md",
    "chapter13_image_generation_interpretability/instructions/pages/02_[13.2]_Different.md",
    "other/instructions/pages/02_[13.2]_Denoising-Time_Causal_Patching.md",
])
def test_non_allowlisted_pages_are_exact_identity_without_reading_notes(helper, relative):
    path = ROOT / relative
    source = path.read_text() if path.is_file() else "## Core Question\n```\nraw code\n```\n"
    assert not helper.has_study_cautions(path)
    assert helper.add_study_cautions(source, path, "malformed irrelevant notes") == source


@pytest.mark.parametrize("section", CASES)
def test_actual_renderer_composes_before_links_and_media(helper, monkeypatch, section):
    monkeypatch.setitem(sys.modules, "infrastructure.extension.study_cautions", helper)
    path = PROJECTOR_ROOT / "infrastructure/extension/render_media.py"
    spec = importlib.util.spec_from_file_location("candidate_render_cautions", path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    events = []
    original = helper.add_study_cautions
    def compose(*args):
        events.append("cautions")
        return original(*args)
    def links(text, path):
        assert BLOCK.search(text)
        events.append("links")
        return text
    monkeypatch.setattr(renderer, "add_study_cautions", compose)
    monkeypatch.setattr(renderer, "rewrite_page_links", links)
    markdown, source = actual(section)
    rendered = renderer.inline_page_media(source, markdown)
    assert events == ["cautions", "links"]
    assert BLOCK.findall(rendered) == [CASES[section][2]]


def test_notes_do_not_change_code_fences_or_raw_source(helper, notes):
    # The inverse assertion on complete actual pages preserves every code character,
    # including solution dropdowns, HTML figures, failed thresholds and old prose.
    for section in CASES:
        path, source = actual(section)
        assert hashlib.sha256(BLOCK.sub("", helper.add_study_cautions(source, path, notes)).encode()).digest() == hashlib.sha256(path.read_bytes()).digest()


@pytest.mark.parametrize("section", CASES)
def test_whole_actual_renderer_changes_only_the_note(helper, monkeypatch, section):
    monkeypatch.setitem(sys.modules, "infrastructure.extension.study_cautions", helper)
    path = PROJECTOR_ROOT / "infrastructure/extension/render_media.py"
    spec = importlib.util.spec_from_file_location("candidate_render_full_cautions", path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    markdown, source = actual(section)
    rendered = renderer.inline_page_media(source, markdown)
    monkeypatch.setattr(renderer, "add_study_cautions", lambda text, *args: text)
    baseline = renderer.inline_page_media(source, markdown)
    assert BLOCK.sub("", rendered) == baseline
    assert BLOCK.findall(rendered) == [CASES[section][2]]


@pytest.mark.parametrize("section", CASES)
def test_added_prose_renders_as_markdown_at_actual_heading(helper, notes, section):
    from markdown_it import MarkdownIt
    path, source = actual(section)
    rendered = helper.add_study_cautions(source, path, notes)
    block = BLOCK.search(rendered)[0]
    parser = MarkdownIt("commonmark", {"html": True})
    if section == "14.1":
        parser.enable("table")
    html = parser.render(block)
    assert "<strong>Study caution — 23 October 2026.</strong>" in html
    assert "**Study caution" not in html
    assert "<pre>" not in html
    assert html.count("<p>") == 2
    if section == "14.1":
        assert html.count("<table>") == 1
        assert re.findall(r"<td>([^<]*)</td>", html) == ["0", "1.0", "0.0", "1.0", "1", "1.0", "0.0", "1.0", "2", "1.0", "0.0", "1.0"]


PRIMARY_READINGS = {
    "8.6": {
        "https://learn.arena.education/chapter1_transformer_interp/21_ioi/",
        "https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/",
        "https://arxiv.org/abs/2403.19647",
    },
    "8.7": {
        "https://learn.arena.education/chapter1_transformer_interp/21_ioi/",
        "https://transformer-circuits.pub/2025/attribution-graphs/methods.html",
    },
    "9.3": {
        "https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/",
        "https://arxiv.org/abs/2305.04388",
    },
    "9.4": {
        "https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/",
        "https://arxiv.org/abs/2502.17424",
    },
    "9.6": {"https://rome.baulab.info/", "https://memit.baulab.info/"},
}


def test_exact_nine_page_allowlist_only(helper):
    assert set(helper._TARGETS) == {case[0] for case in CASES.values()}
    assert len(helper._TARGETS) == 9


@pytest.mark.parametrize("section", PRIMARY_READINGS)
def test_new_notes_contain_only_exact_verified_clickable_primary_routes(helper, notes, section):
    from markdown_it import MarkdownIt
    path, source = actual(section)
    block = BLOCK.search(helper.add_study_cautions(source, path, notes))[0]
    urls = re.findall(r"\]\((https://[^)]+)\)", block)
    assert set(urls) == PRIMARY_READINGS[section]
    assert len(urls) == len(PRIMARY_READINGS[section])
    html = MarkdownIt("commonmark", {"html": True}).render(block)
    assert set(re.findall(r'<a href="(https://[^"]+)"', html)) == PRIMARY_READINGS[section]
    assert "2403.00745" not in block  # rejected AtP* lookup is not Sparse Feature Circuits


def test_8_6_caution_precedes_preserved_metrics_and_matches_literal_node_code(helper, notes):
    path, source = actual("8.6")
    rendered = helper.add_study_cautions(source, path, notes)
    block = BLOCK.search(rendered)[0]
    assert rendered.index(block) < rendered.index("Compute the validation curves")
    assert rendered.index(block) < rendered.index("If it overlaps the target graph, the baseline has leaked the answer.")
    formulas = [
        "full_score = contributions.sum().item()",
        "selected_score = contributions[selected].sum().item()",
        "faithfulness = selected_score / full_score",
        "completeness = 1.0 - abs(full_score - selected_score) / abs(full_score)",
        "minimality = (contributions[selected].abs().min() / abs(full_score)).item()",
    ]
    for formula in formulas:
        assert f"`{formula}`" in block
        assert formula in source
    for phrase in [
        'planted node contributions',
        'general edge-circuit',
        'conditioned on disjoint nuisance nodes',
        'not an unconditional random-graph null',
        'Overlap is not inherently answer leakage',
        'empty selection',
        'what the comparison establishes',
    ]:
        assert phrase in block
    # The retained helper guard is still present; no scientific boundary is changed.
    assert 'raise ValueError("random graph must be disjoint from selected features.")' in source
    assert BLOCK.sub("", rendered) == source


def test_9_6_reader_boundary_does_not_advertise_runall_or_memit_implementation(helper, notes):
    path, source = actual("9.6")
    block = BLOCK.search(helper.add_study_cautions(source, path, notes))[0]
    for phrase in [
        'isolated rank-one affine-write algebra',
        'Run All path loads a pinned model and writes legacy expected outputs',
        'do not use it merely to inspect',
        'same fixed matrix',
        'it is not MEMIT',
        'does not train a new multi-token edit',
    ]:
        assert phrase in block
    assert "explicit writes to legacy expected outputs" in source
    assert "does not implement MEMIT mass editing" in source
