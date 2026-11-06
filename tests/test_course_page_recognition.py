"""Bounded page-title aliases and explicit runtime-prose recognition."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_course_surface import _page_has_arena_shape


def page(title="Example", runtime="The examples run on CPU.", number="11.4"):
    return (f"# [{number}] {title}\n# Introduction\n"
            f"## Content & Learning Objectives\nExercise\ntests.check()\n{runtime}")


@pytest.mark.parametrize("runtime", [
    "Everything runs on a CPU in about a minute.",
    "The whole thing runs in a few seconds on a CPU.",
    "The example runs on CPU.",
    "Use CPU for these exercises.",
    "The defaults take about a minute on a GPU.",
    "This takes well under a minute on a GPU.",
    "The whole notebook runs in about 7 minutes on one GPU.",
    "Exact patching takes about ten seconds on a GPU (a minute or two on a CPU).",
    "The defaults take roughly 10-40 seconds per run on a GPU.",
    "Training all of them takes about 30–90 seconds on a GPU.",
    "It takes a couple of minutes on a GPU.",
])
def test_explicit_cpu_or_timed_gpu_runtime_prose(runtime):
    assert _page_has_arena_shape(page(runtime=runtime), "11.4", "Example")


@pytest.mark.parametrize("runtime", [
    "", "This lesson mentions GPU memory.", "Move this tensor to the CPU.",
    "This method appeared a minute ago; the plot uses a GPU.",
    "This takes a few minutes; use the plot below.",
    "Here are seconds on a GPU.", "The dataset has ten seconds of GPU telemetry.",
])
def test_missing_or_incidental_runtime_is_rejected(runtime):
    assert not _page_has_arena_shape(page(runtime=runtime), "11.4", "Example")


@pytest.mark.parametrize("number,configured,heading", [
    ("7.1", "Logit Lens, Tuned Lens and Patchscopes", "Logit Lens, Tuned Lens & Patchscopes"),
    ("11.1", "PCA, SVD and Geometry Controls", "PCA, SVD, and Geometry Controls"),
    ("11.2", "CCA, SVCCA and Representation Alignment", "CCA, SVCCA, and Representation Alignment"),
    ("8.3", "AtP* from Scratch", "AtP*: Fixing Attribution Patching"),
    ("15.2", "LoRA vs Full Fine-Tuning", "LoRA vs Full Finetuning on a Real Transformer"),
])
def test_reviewed_title_identity_variants(number, configured, heading):
    assert _page_has_arena_shape(page(title=heading, number=number), number, configured)


@pytest.mark.parametrize("number,configured,heading", [
    ("8.3", "AtP* from Scratch", "Attribution Patching and EAP"),
    ("8.3", "AtP* from Scratch", "AtP*: Arbitrary Untested Extension"),
    ("8.3", "Unrelated Method", "AtP*: Fixing Attribution Patching"),
    ("8.2", "AtP* from Scratch", "AtP*: Fixing Attribution Patching"),
    ("15.2", "LoRA vs Full Fine-Tuning", "LoRA vs Full Finetuning on a Real Transformer Extra"),
    ("15.1", "LoRA vs Full Fine-Tuning", "LoRA vs Full Finetuning on a Real Transformer"),
    ("11.4", "Example", "Example Extra"),
    ("11.4", "Example", "Wrong Method"),
])
def test_unreviewed_wrong_or_prefix_title_is_rejected(number, configured, heading):
    assert not _page_has_arena_shape(page(title=heading, number=number), number, configured)


def test_numbered_h1_identity_cannot_be_satisfied_by_prose_or_other_number():
    assert not _page_has_arena_shape(page(number="11.3"), "11.4", "Example")
    assert not _page_has_arena_shape(page().replace("# [11.4] Example", "See # [11.4] Example"), "11.4", "Example")


@pytest.mark.parametrize("missing", ["# Introduction", "## Content & Learning Objectives", "Exercise", "tests.check()"])
def test_recognition_does_not_bypass_other_surface_requirements(missing):
    assert not _page_has_arena_shape(page(runtime="This takes a minute on a GPU.").replace(missing, ""), "11.4", "Example")


def test_frozen_twenty_page_inventory_retains_real_missing_runtime_failures():
    evidence = (Path(__file__).resolve().parents[1] / "docs/mastery/evidence"
                / "2026-10-16-page-surface-recognition")
    before = json.loads((evidence / "before.json").read_text())
    with gzip.open(evidence / "before_pages.json.gz", "rt") as source:
        pages = json.load(source)
    failures = []
    assert len(before["page_failures"]) == len(pages) == 20
    for row in before["page_failures"]:
        text = pages[row["path"]]
        assert hashlib.sha256(text.encode()).hexdigest() == row["sha256"]
        if not _page_has_arena_shape(text, row["number"], row["title"]):
            failures.append(row["number"])
    assert failures == ["8.2", "8.5"]
