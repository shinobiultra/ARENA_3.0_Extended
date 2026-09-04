import pytest

from scripts.build_merged_config import merge_config


def test_merge_config_adds_extension_without_editing_original():
    original = {
        "conversion_map": {"1.1": {"streamlit_page": "one", "exercise_dir": "part1"}},
        "chapter_names": {1: "chapter1"},
        "chapter_names_long": {1: "Chapter 1"},
        "chapters": {
            "chapter1": {
                "title": "Chapter 1",
                "sections": [{"number": "1.1", "title": "Original"}],
            }
        },
    }
    extension = {
        "conversion_map": {"1.6": {"streamlit_page": "six", "exercise_dir": "part6"}},
        "chapters": {
            "chapter1": {
                "sections": [{"number": "1.6", "title": "Extension"}],
            }
        },
    }

    merged = merge_config(original, extension)

    assert merged["conversion_map"]["1.1"] == original["conversion_map"]["1.1"]
    assert merged["conversion_map"]["1.6"] == extension["conversion_map"]["1.6"]
    assert [section["number"] for section in merged["chapters"]["chapter1"]["sections"]] == [
        "1.1",
        "1.6",
    ]


def test_merge_config_rejects_original_section_overwrite():
    original = {
        "conversion_map": {"1.1": {"streamlit_page": "one", "exercise_dir": "part1"}},
        "chapter_names": {},
        "chapter_names_long": {},
        "chapters": {},
    }
    extension = {
        "conversion_map": {"1.1": {"streamlit_page": "changed", "exercise_dir": "partX"}},
    }

    with pytest.raises(ValueError, match="duplicates original section 1.1"):
        merge_config(original, extension)


def test_merge_config_rejects_original_chapter_metadata_overlay():
    original = {
        "conversion_map": {},
        "chapter_names": {},
        "chapter_names_long": {},
        "chapters": {"chapter1": {"title": "Chapter 1", "sections": []}},
    }
    extension = {"chapters": {"chapter1": {"title": "Changed", "sections": []}}}

    with pytest.raises(ValueError, match="may only add sections"):
        merge_config(original, extension)
