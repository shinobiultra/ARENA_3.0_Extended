import scripts.audit_arena_style_depth as audit


def test_course_ready_page_rejects_keyword_stuffing(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(
        """
# Fake page

expected output solution common bug difficulty importance signature result limitations
This page says all the old keywords but has no real dropdowns.
"""
    )

    blockers = audit._page_course_ready_blockers("5.1", page)

    assert any("expected output dropdown" in blocker for blocker in blockers)
    assert any("solution dropdown" in blocker for blocker in blockers)
    assert any("help or interpretation dropdown" in blocker for blocker in blockers)


def test_course_ready_page_accepts_real_dropdowns_and_result_table(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(
        """
# Real page

<details><summary>Expected output</summary>
ok
</details>

<details><summary>Solution</summary>
hidden
</details>

<details><summary>Help - why did this fail?</summary>
interpretation
</details>

## Signature result

| Check | Observed |
| --- | --- |
| parity | pass |

## Limitations
This proves only the scoped claim.
"""
    )

    assert audit._page_course_ready_blockers("5.1", page) == []


def test_page_metadata_consistency_blockers_flag_page_lock_drift(tmp_path, monkeypatch):
    page_dir = tmp_path / "chapter5_modern_architectures/instructions/pages"
    page_dir.mkdir(parents=True)
    page = page_dir / "01_[5.1]_Gemma_from_Scratch.md"
    page.write_text(
        '\n'.join(
            [
                '```python',
                'GT_TIER = "GT-1"',
                'EXERCISE_ID = "stale.id"',
                'EXPECTED_RUNTIME = "seconds"',
                'REQUIRES_GPU = True',
                '```',
            ]
        )
    )

    lock_dir = tmp_path / "chapter5_modern_architectures/exercises/part1_gemma_from_scratch"
    lock_dir.mkdir(parents=True)
    (lock_dir / "artifacts.lock.yml").write_text(
        "\n".join(
            [
                "section: '5.1'",
                "exercise_metadata:",
                "  GT_TIER: GT-0",
                "  EXERCISE_ID: 5_1_gemma_from_scratch",
            ]
        )
    )

    config = {
        "chapters": {
            "chapter5_modern_architectures": {
                "sections": [
                    {
                        "number": "5.1",
                        "page_file": "01_[5.1]_Gemma_from_Scratch.md",
                    }
                ]
            }
        }
    }

    monkeypatch.setattr(audit, "ROOT", tmp_path)

    blockers = audit.page_metadata_consistency_blockers(config)

    assert any("page GT_TIER='GT-1'" in blocker for blocker in blockers), (
        "The style audit should reject GT-tier drift between learner page and lockfile."
    )
    assert any("page EXERCISE_ID='stale.id'" in blocker for blocker in blockers), (
        "The style audit should reject exercise-id drift between learner page and lockfile."
    )


def test_notebook_setup_contract_flags_undefined_section_dir(tmp_path, monkeypatch):
    notebook_dir = tmp_path / "chapter5_modern_architectures/exercises/part1_fake"
    notebook_dir.mkdir(parents=True)
    (notebook_dir / "fake_exercises.ipynb").write_text(
        """
{
 "cells": [
  {
   "cell_type": "code",
   "metadata": {},
   "outputs": [],
   "source": ["report = (section_dir / \\"verification_report.json\\").read_text()\\n"]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
"""
    )

    monkeypatch.setattr(audit, "ROOT", tmp_path)

    blockers = audit.notebook_setup_contract_blockers()

    assert any("references section_dir without defining it" in blocker for blocker in blockers)


def test_notebook_setup_contract_accepts_defined_section_dir(tmp_path, monkeypatch):
    notebook_dir = tmp_path / "chapter5_modern_architectures/exercises/part1_fake"
    notebook_dir.mkdir(parents=True)
    (notebook_dir / "fake_exercises.ipynb").write_text(
        """
{
 "cells": [
  {
   "cell_type": "code",
   "metadata": {},
   "outputs": [],
   "source": [
    "section = \\"part1_fake\\"\\n",
    "exercises_dir = root_dir / chapter / \\"exercises\\"\\n",
    "section_dir = exercises_dir / section\\n",
    "report = (section_dir / \\"verification_report.json\\").read_text()\\n"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
"""
    )

    monkeypatch.setattr(audit, "ROOT", tmp_path)

    assert audit.notebook_setup_contract_blockers() == []
