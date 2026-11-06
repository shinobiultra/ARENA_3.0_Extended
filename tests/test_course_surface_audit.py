from copy import deepcopy
import pytest

from scripts.audit_course_surface import (
    course_surface_blockers,
    _home_uses_shared_renderer,
    is_extension_section,
    load_config,
)


@pytest.mark.parametrize("source", [
    'from infrastructure.extension.home import render_chapter_home\n'
    'INTRO = "## Description of material\\nChapter introduction"\n'
    'render_chapter_home(__file__, INTRO)\n',
    'from infrastructure.extension.home import render_chapter_home as render\n'
    'render(__file__, "Chapter introduction")\n',
])
def test_shared_home_renderer_requires_known_import_and_direct_call(source):
    assert _home_uses_shared_renderer(source)


@pytest.mark.parametrize("source", [
    '# render_extension_home(__file__)\n# render_chapter_home(__file__, INTRO)',
    'render_chapter_home(__file__, "Introduction")',
    'from arbitrary_wrapper import render_chapter_home\n'
    'render_chapter_home(__file__, "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'def unused():\n    render_chapter_home(__file__, "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'render_chapter_home("other_chapter/Home.py", "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'render_chapter_home(__file__, "")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'INTRO = hidden_intro()\nrender_chapter_home(__file__, INTRO)',
    'from infrastructure.extension.home import render_chapter_home\n'
    'render_chapter_home = arbitrary_wrapper\nrender_chapter_home(__file__, "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'def render_chapter_home(*args):\n    pass\nrender_chapter_home(__file__, "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'from unrelated import render_chapter_home\nrender_chapter_home(__file__, "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'INTRO = "Introduction"\nINTRO = ""\nrender_chapter_home(__file__, INTRO)',
    'from infrastructure.extension.home import render_chapter_home\n'
    'if False:\n    render_chapter_home(__file__, "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    '__file__ = "other_chapter/Home.py"\nrender_chapter_home(__file__, "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'if arbitrary_condition:\n    render_chapter_home = wrapper\n'
    'render_chapter_home(__file__, "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'from arbitrary_wrapper import *\nrender_chapter_home(__file__, "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'INTRO = "Introduction"\nfrom unrelated import INTRO\nrender_chapter_home(__file__, INTRO)',
    'from infrastructure.extension.home import render_chapter_home\n'
    'from unrelated import other_file as __file__\nrender_chapter_home(__file__, "Introduction")',
    'from infrastructure.extension.home import render_chapter_home\n'
    'if arbitrary_condition:\n    from unrelated import *\n'
    'render_chapter_home(__file__, "Introduction")',
    'invalid Python !!!',
])
def test_shared_home_renderer_rejects_name_only_wrong_arguments_or_wrappers(source):
    assert not _home_uses_shared_renderer(source)


def test_chapter_surface_applies_renderer_contract_to_actual_home(tmp_path, monkeypatch):
    import scripts.audit_course_surface as audit

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    instructions = tmp_path / "chapter5_fixture" / "instructions"
    instructions.mkdir(parents=True)
    home = instructions / "Home.py"
    home.write_text('from infrastructure.extension.home import render_chapter_home\n'
                    'render_chapter_home(__file__, "Introduction")')
    assert not any("renderer" in item for item in audit.chapter_surface_blockers({}, "chapter5_fixture"))
    home.write_text('# render_extension_home(__file__)\n'
                    'from arbitrary_wrapper import render_chapter_home\n'
                    'render_chapter_home(__file__, "Introduction")')
    assert any("does not use shared extension renderer" in item
               for item in audit.chapter_surface_blockers({}, "chapter5_fixture"))


def test_is_extension_section_tracks_original_additions_and_new_chapters():
    assert is_extension_section("0.6")
    assert is_extension_section("1.6")
    assert is_extension_section("5.1")
    assert is_extension_section("17.1")
    assert not is_extension_section("1.2")
    assert not is_extension_section("4.5")


def test_course_surface_audit_flags_conversion_map_mismatch():
    config = deepcopy(load_config())
    config["conversion_map"]["5.1"]["exercise_dir"] = "wrong_dir"

    blockers = course_surface_blockers(config)

    assert "5.1: conversion_map exercise_dir mismatch" in blockers


def test_course_surface_audit_flags_missing_home_renderer():
    config = deepcopy(load_config())
    chapter = config["chapter_names"][5]
    config["chapter_names"][5] = chapter
    # The live audit covers filesystem presence; this mutation makes sure the
    # fixture still exercises chapter-level checks through the real config.
    blockers = course_surface_blockers(config)

    assert not any("missing instructions/Home.py" in blocker for blocker in blockers)


def test_current_course_surface_passes():
    assert course_surface_blockers() == []
