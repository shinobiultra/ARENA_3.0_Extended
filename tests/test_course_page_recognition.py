"""Page-title recognition in the course-surface audit, including reviewed aliases."""

import pytest

from scripts.audit_course_surface import _page_has_arena_shape


def page(title="Example", number="11.4"):
    return f"# [{number}] {title}\n# Introduction\n## Content & Learning Objectives\nExercise\ntests.check()\n"


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
    assert not _page_has_arena_shape(page().replace(missing, ""), "11.4", "Example")
