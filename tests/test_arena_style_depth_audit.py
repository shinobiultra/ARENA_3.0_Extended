import scripts.audit_arena_style_depth as audit


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
