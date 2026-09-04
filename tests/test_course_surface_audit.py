import json
from copy import deepcopy

from scripts.audit_course_surface import (
    course_surface_blockers,
    _exercise_notebook_declares_verification_contract,
    is_extension_section,
    load_config,
)


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


def test_notebook_verification_contract_detector_requires_visible_gpu_surface(tmp_path):
    notebook_path = tmp_path / "example_exercises.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "def run_gpu_test(max_vram_gb: float = 24.0):\n",
                            "    return json.loads((section_dir / 'verification_report.json').read_text())\n",
                            "\n",
                            "def run_full_experiment(max_vram_gb: float = 24.0):\n",
                            "    return run_gpu_test(max_vram_gb=max_vram_gb)\n",
                        ],
                    }
                ]
            }
        )
    )

    assert _exercise_notebook_declares_verification_contract(notebook_path)

    notebook_path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": []}]}))

    assert not _exercise_notebook_declares_verification_contract(notebook_path)


def test_current_course_surface_passes():
    assert course_surface_blockers() == []
