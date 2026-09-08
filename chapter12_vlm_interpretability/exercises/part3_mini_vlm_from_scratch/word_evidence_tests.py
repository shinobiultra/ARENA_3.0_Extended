"""Reject misleading report changes using actual independently executed word outputs."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from . import word_evidence as evidence


@pytest.fixture(scope="module")
def measured_study():
    path = Path(__file__).parent / "expected_outputs/word_candidates/learner_validation/completed_word_study.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def test_raw_generated_words_reproduce_all_measured_metrics(measured_study):
    result = evidence.validate_word_measurements(measured_study)
    assert result["final_evaluations"] == 36 and result["stage_region_experiments"] == 285
    assert result["final_means"]["compositional_muted"]["joint"]["caption_exact_match"] == 0.
    assert result["final_means"]["familiar_muted"]["joint"]["exact_match"] > .85


def test_word_evidence_rejects_omitted_seeds_and_changed_checkpoint_selection(measured_study):
    for change in ("seed", "final_condition", "checkpoint", "history"):
        altered = copy.deepcopy(measured_study)
        if change == "seed":
            altered["training"].pop()
        elif change == "final_condition":
            altered["final_evaluation"]["rows"].pop()
        elif change == "checkpoint":
            altered["frozen_choices"]["choices"][0]["selected_step"] = 100
        else:
            altered["training"][0]["history"].pop()
        with pytest.raises(ValueError):
            evidence.validate_word_measurements(altered)


def test_word_evidence_rejects_better_metrics_without_better_generated_words(measured_study):
    for change in ("metric", "per_example", "caption_tokens"):
        altered = copy.deepcopy(measured_study)
        row = altered["final_evaluation"]["rows"][2]  # Joint seed 0, excluded combinations.
        if change == "metric":
            row["caption_exact_match"] = 1.
        elif change == "per_example":
            row["per_example_exact_match"][2] = True
        else:
            row["generated_tokens"][2] = [0] * 8
            row["per_example_exact_match"][2] = True
        with pytest.raises(ValueError, match="word|exact|metric"):
            evidence.validate_word_measurements(altered)


def test_word_evidence_rejects_filtered_controls_and_wrong_patch_denominators(measured_study):
    for change in ("missing_draw", "denominator", "region", "endpoint", "finished"):
        altered = copy.deepcopy(measured_study)
        study = altered["stage_patching"]
        if change == "missing_draw":
            study["rows"].pop()
        elif change == "denominator":
            study["rows"][0]["eligible"] = 72
        elif change == "region":
            study["rows"][1]["selected_visual_rows"] = study["object_rows"]
        elif change == "endpoint":
            study["endpoints"][0]["donor_correct"] = [True] * 72
        else:
            study["rows"][0]["generated_finished"][0] = False
        with pytest.raises(ValueError):
            evidence.validate_word_measurements(altered)


@pytest.fixture
def provenance_record(tmp_path, monkeypatch):
    """Isolate file provenance from the separate actual-word semantic checks above."""
    here = tmp_path / "course" / "exercises" / "section"
    here.mkdir(parents=True)
    monkeypatch.setattr(evidence, "HERE", here)
    monkeypatch.setattr(evidence, "validate_word_measurements", lambda payload: {"semantic_check_separately_tested": True})
    (here / "lesson.md").write_text("The exact executed lesson.\n")
    (here / "method.py").write_text("def method():\n    return 1\n")
    directory = here / "expected_outputs"
    directory.mkdir()
    raw = directory / "completed_word_study.json.gz"
    raw.write_bytes(gzip.compress(b"{}", mtime=0))
    for suffix in ("exercises", "solutions"):
        (here / f"{evidence.STEM}_{suffix}.ipynb").write_text("{}")
    assets = here.parents[1] / "instructions/assets"
    assets.mkdir(parents=True)
    for name in evidence.REQUIRED_FIGURES:
        (assets / name).write_bytes(b"fixed figure bytes")
    manifest = {"status": "PASS", "scope": "full", "completed_from_own_answers": True,
        "fresh_training_required": True, "sources_changed_during_run": [],
        "input_files": {name: evidence.sha256(here / name) for name in ("lesson.md", "method.py")},
        "published_evidence_sha256": evidence.sha256(raw),
        "published_exercises_sha256": evidence.sha256(here / f"{evidence.STEM}_exercises.ipynb"),
        "published_notebook_sha256": evidence.sha256(here / f"{evidence.STEM}_solutions.ipynb"),
        "figures": {name: evidence.sha256(assets / name) for name in evidence.REQUIRED_FIGURES}}
    record = directory / "learner_execution.json"
    record.write_text(json.dumps(manifest))
    return directory, record, manifest, here, assets


def test_word_provenance_rejects_partial_or_mixed_source_execution(provenance_record):
    directory, record, manifest, here, _ = provenance_record
    evidence.read_completed_word_evidence(directory)
    for change in ({"scope": "numerical"}, {"fresh_training_required": False},
                   {"sources_changed_during_run": ["method.py"]}, {"status": "FAIL"}):
        record.write_text(json.dumps({**manifest, **change}))
        with pytest.raises(ValueError, match="successful full"):
            evidence.read_completed_word_evidence(directory)
    record.write_text(json.dumps(manifest))
    (here / "method.py").write_text("def method():\n    return 2\n")
    with pytest.raises(ValueError, match="stale"):
        evidence.read_completed_word_evidence(directory)


def test_word_provenance_rejects_missing_or_changed_published_figures(provenance_record):
    directory, record, manifest, _, assets = provenance_record
    reduced = copy.deepcopy(manifest)
    reduced["figures"].pop("word_compositional_gallery.png")
    record.write_text(json.dumps(reduced))
    with pytest.raises(ValueError, match="every learner"):
        evidence.read_completed_word_evidence(directory)
    record.write_text(json.dumps(manifest))
    (assets / "word_compositional_gallery.png").write_bytes(b"different figure")
    with pytest.raises(ValueError, match="published figure"):
        evidence.read_completed_word_evidence(directory)
