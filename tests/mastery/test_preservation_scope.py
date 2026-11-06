"""Extension migration paths must not widen access to original ARENA sources."""

import pytest

from scripts.audit_original_arena_preservation import (
    is_allowed_preservation_change,
    preservation_blockers,
)


@pytest.mark.parametrize(
    "path",
    [
        "MASTERY-HUB.md",
        "mastery_hub/README.md",
        "mastery_hub/notebooks/01_optimizer.ipynb",
        "infrastructure/extension/build.py",
        "infrastructure/extension/cluster/README.md",
        # Q008 permits these two exact additive Practice master sources.
        "infrastructure/chapters/chapter0_fundamentals/master_0_6.py",
        "infrastructure/chapters/chapter1_transformer_interp/master_1_6.py",
        "infrastructure/chapters/chapter5_modern_architectures/master_5_1.py",
        "infrastructure/chapters/chapter6_sparse_feature_methods/master_6_1.py",
        "infrastructure/chapters/chapter7_activation_to_language/master_7_1.py",
        "infrastructure/chapters/chapter8_automated_circuits/master_8_4.py",
        "infrastructure/chapters/chapter9_alignment_interpretability/master_9_1.py",
        "infrastructure/chapters/chapter10_capstone_research_sprint/master_10_1.py",
        "infrastructure/chapters/chapter11_representation_geometry/master_11_1.py",
        "infrastructure/chapters/chapter12_vlm_interpretability/master_12_3.py",
        "infrastructure/chapters/chapter13_image_generation_interpretability/master_13_1.py",
        "infrastructure/chapters/chapter14_jepa_world_models/master_14_1.py",
        "infrastructure/chapters/chapter15_peft_misalignment/master_15_1.py",
        "infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_9.py",
        "infrastructure/chapters/chapter17_training_dynamics/master_17_1.py",
        "infrastructure/chapters/chapter18_intrinsic_interpretability/master_18_1.py",
        "chapter18_intrinsic_interpretability/exercises/part1_sparse_models/solutions.py",
    ],
)
def test_accepts_additive_mastery_and_extension_migration_paths(path):
    assert preservation_blockers({"tracked": [path], "untracked": [path]}) == []


@pytest.mark.parametrize(
    "path",
    [
        "infrastructure/chapters/chapter0_fundamentals/master_0_0.py",
        "infrastructure/chapters/chapter1_transformer_interp/master_1_1.ipynb",
        "infrastructure/chapters/chapter2_rl/master_2_2.py",
        "infrastructure/chapters/chapter3_llm_evals/master_3_1.py",
        "infrastructure/chapters/chapter4_alignment_science/master_4_1.py",
        "infrastructure/chapters/chapter0_fundamentals/master_0_5.py",
        "infrastructure/chapters/chapter1_transformer_interp/master_1_2.py",
        "infrastructure/chapters/chapter0_fundamentals/master_0_6.py.bak",
        "infrastructure/chapters/chapter1_transformer_interp/master_1_6.py.bak",
        "infrastructure/chapters/chapter0_fundamentals/master_0_6.py/child.py",
        "infrastructure/chapters/chapter1_transformer_interp/master_1_6.py/child.py",
        "infrastructure/chapters/chapter0_fundamentals/master_0_6.py/../master_0_5.py",
        "infrastructure/chapters/chapter1_transformer_interp/master_1_6.py/../master_1_2.py",
        "infrastructure/chapters/chapter5_unapproved/master_5_1.py",
        "infrastructure/chapters/chapter19_unapproved/master_19_1.py",
        "infrastructure/chapters/README.md",
        "infrastructure/extension_backup/build.py",
        "mastery_hub_backup/README.md",
        "MASTERY-HUB.md.backup",
        "chapter1_transformer_interp/MASTERY-HUB.md",
    ],
)
def test_rejects_original_masters_and_nearby_unapproved_paths(path):
    assert preservation_blockers({"tracked": [path]}) == [f"tracked: {path}"]


@pytest.mark.parametrize(
    "path",
    [
        "chapter0_fundamentals/exercises/part6_fake_interpretability_results/solutions.py",
        "chapter0_fundamentals/instructions/pages/"
        "06_[0.6]_How_to_Know_When_an_Interpretability_Result_Is_Fake.md",
        "chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/solutions.py",
        "chapter1_transformer_interp/instructions/pages/"
        "40_[1.6]_Local_Frontier_ML_Infrastructure.md",
    ],
)
def test_preserves_existing_explicit_additive_original_section_exceptions(path):
    assert is_allowed_preservation_change(path)
