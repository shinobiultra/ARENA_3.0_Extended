from scripts.audit_original_arena_preservation import (
    COMPATIBILITY_PATCH_RATIONALES,
    compatibility_contract_blockers,
    changed_paths,
    is_allowed_preservation_change,
    original_preservation_blockers,
    preservation_blockers,
)


def test_preservation_audit_rejects_original_exercise_edits():
    assert not is_allowed_preservation_change(
        "chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/solutions.py"
    )
    assert not is_allowed_preservation_change(
        "chapter2_rl/exercises/part1_intro_to_rl/solutions.py"
    )


def test_preservation_audit_allows_explicit_additive_original_sections():
    assert is_allowed_preservation_change(
        "chapter0_fundamentals/exercises/part6_fake_interpretability_results/solutions.py"
    )
    assert is_allowed_preservation_change(
        "chapter1_transformer_interp/instructions/pages/"
        "40_[1.6]_Local_Frontier_ML_Infrastructure.md"
    )


def test_preservation_audit_allows_extension_chapters_and_compatibility_files():
    assert is_allowed_preservation_change("chapter16_shapley_attribution_baselines/README.md")
    assert is_allowed_preservation_change("arena_ext/gated_artifacts.py")
    assert is_allowed_preservation_change(".github/workflows/extension-quality.yml")
    assert is_allowed_preservation_change("requirements-ci-cpu.txt")
    assert is_allowed_preservation_change("requirements.txt")
    assert is_allowed_preservation_change("uv.lock")


def test_compatibility_patch_rationales_are_documented():
    assert all(reason.strip() for reason in COMPATIBILITY_PATCH_RATIONALES.values())
    assert compatibility_contract_blockers() == []


def test_preservation_blockers_include_source_and_path():
    blockers = preservation_blockers(
        {
            "tracked": ["chapter0_fundamentals/exercises/part0_prereqs/solutions.py"],
            "untracked": ["chapter5_modern_architectures/exercises/part1_gemma/solutions.py"],
        }
    )

    assert blockers == ["tracked: chapter0_fundamentals/exercises/part0_prereqs/solutions.py"]


def test_current_worktree_preserves_original_arena_surface():
    blockers = original_preservation_blockers()

    assert blockers == []
