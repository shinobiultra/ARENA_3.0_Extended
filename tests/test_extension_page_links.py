"""Finite local page downloads and figures: source-only rendering tests."""

import base64
import importlib.util
import os
import re
import shutil
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "infrastructure/extension/page_links.py"
_SPEC = importlib.util.spec_from_file_location("candidate_page_links", _MODULE_PATH)
page_links = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(page_links)
ROOT = Path.cwd()

# Independently supplied breadth-audit target identities; no target discovery.
CASES = [
    (
        "chapter0_fundamentals/instructions/pages/06_[0.6]_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice.md",
        "expected_outputs/fake_interpretability_signature_result.svg",
        "chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/media/fake_interpretability_signature_result.svg",
    ),
    (
        "chapter5_modern_architectures/instructions/pages/05_[5.5]_State_Tracking_in_SSMs_and_Transformers.md",
        "../part4_mamba_state_tracking/5.4_Mamba_State_Tracking_exercises.ipynb",
        "chapter5_modern_architectures/exercises/part4_mamba_state_tracking/5.4_Mamba_State_Tracking_exercises.ipynb",
    ),
    (
        "chapter5_modern_architectures/instructions/pages/05_[5.5]_State_Tracking_in_SSMs_and_Transformers.md",
        "../part3_mamba_from_scratch/5.3_Mamba_from_Scratch_exercises.ipynb",
        "chapter5_modern_architectures/exercises/part3_mamba_from_scratch/5.3_Mamba_from_Scratch_exercises.ipynb",
    ),
    (
        "chapter13_image_generation_interpretability/instructions/pages/02_[13.2]_Denoising-Time_Causal_Patching.md",
        "13.2_Denoising-Time_Causal_Patching_exercises.ipynb",
        "chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching/13.2_Denoising-Time_Causal_Patching_exercises.ipynb",
    ),
    (
        "chapter13_image_generation_interpretability/instructions/pages/02_[13.2]_Denoising-Time_Causal_Patching.md",
        "13.2_Denoising-Time_Causal_Patching_solutions.ipynb",
        "chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching/13.2_Denoising-Time_Causal_Patching_solutions.ipynb",
    ),
    (
        "chapter14_jepa_world_models/instructions/pages/02_[14.2]_V-JEPA_and_Video_World_Models.md",
        "14.1_JEPA_and_World_Model_Controls_exercises.ipynb",
        "chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls/14.1_JEPA_and_World_Model_Controls_exercises.ipynb",
    ),
    (
        "chapter14_jepa_world_models/instructions/pages/02_[14.2]_V-JEPA_and_Video_World_Models.md",
        "14.1_JEPA_and_World_Model_Controls_solutions.ipynb",
        "chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls/14.1_JEPA_and_World_Model_Controls_solutions.ipynb",
    ),
    (
        "chapter15_peft_misalignment/instructions/pages/01_[15.1]_LoRA_and_DoRA_from_Scratch.md",
        "media/lora_dora_adapter_controls_validation_loop.svg",
        "chapter15_peft_misalignment/exercises/part1_lora_dora/media/lora_dora_adapter_controls_validation_loop.svg",
    ),
    (
        "chapter15_peft_misalignment/instructions/pages/01_[15.1]_LoRA_and_DoRA_from_Scratch.md",
        "media/lora_dora_adapter_controls_signature_panel.png",
        "chapter15_peft_misalignment/exercises/part1_lora_dora/media/lora_dora_adapter_controls_signature_panel.png",
    ),
    (
        "chapter15_peft_misalignment/instructions/pages/01_[15.1]_LoRA_and_DoRA_from_Scratch.md",
        "media/lora_dora_adapter_controls_behavior_table.png",
        "chapter15_peft_misalignment/exercises/part1_lora_dora/media/lora_dora_adapter_controls_behavior_table.png",
    ),
    (
        "chapter16_shapley_attribution_baselines/instructions/pages/02_[16.2]_KernelSHAP_and_PartitionSHAP.md",
        "../part1_exact_shapley_ground_truth_games/16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb",
        "chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley_ground_truth_games/16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb",
    ),
    (
        "chapter16_shapley_attribution_baselines/instructions/pages/04_[16.4]_SAGE_Global_Importance.md",
        "../part1_exact_shapley_ground_truth_games/16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb",
        "chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley_ground_truth_games/16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb",
    ),
    (
        "chapter16_shapley_attribution_baselines/instructions/pages/05_[16.5]_Shapley_Interactions.md",
        "protocol.md",
        "chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap/protocol.md",
    ),
    (
        "chapter16_shapley_attribution_baselines/instructions/pages/10_[16.10]_Influence_Functions_and_TracIn.md",
        "../part9_data_shapley/16.9_Data_Shapley_in_One_Training_Run_exercises.ipynb",
        "chapter16_shapley_attribution_baselines/exercises/part9_data_shapley/16.9_Data_Shapley_in_One_Training_Run_exercises.ipynb",
    ),
    (
        "chapter16_shapley_attribution_baselines/instructions/pages/11_[16.11]_Datamodels_and_Dataset_Cartography.md",
        "../part9_influence_and_tracin/README.md",
        "chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin/README.md",
    ),
    (
        "chapter16_shapley_attribution_baselines/instructions/pages/11_[16.11]_Datamodels_and_Dataset_Cartography.md",
        "../part10_datamodels/README.md",
        "chapter16_shapley_attribution_baselines/exercises/part10_datamodels/README.md",
    ),
]


@pytest.mark.parametrize("page,old,target", CASES)
def test_audited_destinations_download_or_embed_exact_canonical_bytes(page, old, target):
    source = f"![Visible control]({old})" if Path(target).suffix in {".png", ".svg"} else f"[Original label]({old})"
    rendered = page_links.rewrite_page_links(source, ROOT / page)
    payload = (ROOT / target).read_bytes()
    match = re.search(r"data:([^;]+)(?:;charset=utf-8)?;base64,([A-Za-z0-9+/=]+)", rendered)
    assert match is not None
    assert base64.b64decode(match[2]) == payload
    if source.startswith("!"):
        assert rendered.startswith("![Visible control](data:image/")
    else:
        assert f'download="{Path(target).name}"' in rendered
        assert "Original label" in rendered
        if target.endswith(".ipynb"):
            assert "historical raw notebook" in rendered
    assert page_links.rewrite_page_links(rendered, ROOT / page) == rendered


@pytest.fixture
def local_figure(tmp_path):
    page, old, target = CASES[7]
    destination = tmp_path / target
    destination.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / target, destination)
    markdown = tmp_path / page
    markdown.parent.mkdir(parents=True)
    markdown.write_text("Untouched raw lesson\n")
    return markdown, old, destination


def test_only_exact_allowlisted_page_and_link_destination_are_rewritten(local_figure):
    page, old, _ = local_figure
    text = f"![control]({old})"
    assert page_links.rewrite_page_links(text, page.with_name("unregistered.md")) == text
    other_chapter = page.parents[3] / "other_chapter/instructions/pages" / page.name
    assert page_links.rewrite_page_links(text, other_chapter) == text
    unknown = "[external](https://example.test/file.svg) [unknown](media/unknown.svg)"
    assert page_links.rewrite_page_links(unknown, page) == unknown


def test_code_fences_inline_code_and_plain_paths_remain_unchanged(local_figure):
    page, old, _ = local_figure
    link = f"![control]({old})"
    protected = f"````python\n{link}\n````\n~~~text\n{link}\n~~~\n`{link}`\n{old}\n"
    assert page_links.rewrite_page_links(protected, page) == protected
    assert "data:image/svg+xml;base64," in page_links.rewrite_page_links(link, page)


def test_missing_exact_target_fails_without_selecting_a_nearby_alternative(local_figure):
    page, old, destination = local_figure
    destination.rename(destination.with_name("similar.svg"))
    with pytest.raises(FileNotFoundError):
        page_links.rewrite_page_links(f"![control]({old})", page)



def test_symlink_target_cannot_escape_chapter_or_change_approved_identity(local_figure, tmp_path):
    page, old, destination = local_figure
    outside = tmp_path / "outside.svg"
    destination.rename(outside)
    destination.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        page_links.rewrite_page_links(f"![control]({old})", page)


def test_html_download_labels_escape_markup_and_keep_exact_filename():
    page, old, target = CASES[5]
    rendered = page_links.rewrite_page_links(f'[unsafe <script> & "label"]({old})', ROOT / page)
    assert '<script>' not in rendered
    assert '&lt;script&gt; &amp; &quot;label&quot;' in rendered
    assert f'download="{Path(target).name}"' in rendered


def test_rendering_does_not_modify_raw_page_or_target(local_figure):
    page, old, target = local_figure
    before = page.read_bytes(), target.read_bytes()
    page_links.rewrite_page_links(f"![control]({old})", page)
    assert (page.read_bytes(), target.read_bytes()) == before


@pytest.mark.parametrize("page", sorted({case[0] for case in CASES}))
def test_actual_complete_pages_replace_every_audited_link_without_changing_code_fences(page):
    path = ROOT / page
    source = path.read_text(encoding="utf-8")
    rendered = page_links.rewrite_page_links(source, path)
    for case_page, old, _ in CASES:
        if case_page == page:
            assert f"]({old})" in source
            assert f"]({old})" not in rendered
    assert re.findall(r"```.*?```", rendered, re.S) == re.findall(r"```.*?```", source, re.S)
    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize("page,target", [
    (
        "chapter7_activation_to_language/instructions/pages/01_[7.1]_Logit_Lens_Tuned_Lens_and_Patchscopes.md",
        "docs/mastery/pilot_reader_notes_20261023.md",
    ),
    (
        "chapter12_vlm_interpretability/instructions/pages/03_[12.3]_Mini_VLM_from_Scratch.md",
        "docs/mastery/pilot_reader_notes_20261023.md",
    ),
    (
        "chapter12_vlm_interpretability/instructions/pages/03_[12.3]_Mini_VLM_from_Scratch.md",
        "mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb",
    ),
])
def test_reader_notes_and_checked_practice_download_exact_bytes(tmp_path, page, target):
    source = ROOT / target
    if target.startswith("docs/"):
        # Candidate-only runs point at the reviewed, not-yet-adopted note payload.
        source = Path(os.environ.get("ARENA_TEST_READER_NOTES", str(source)))
    target_path = tmp_path / target
    target_path.parent.mkdir(parents=True)
    shutil.copyfile(source, target_path)
    page_path = tmp_path / page
    page_path.parent.mkdir(parents=True)
    link = f"[Download checked material](../../../{target})"
    rendered = page_links.rewrite_page_links(link, page_path)
    match = re.search(r";base64,([A-Za-z0-9+/=]+)", rendered)
    assert match is not None
    assert base64.b64decode(match[1]) == source.read_bytes()
    assert f'download="{source.name}"' in rendered
    assert "historical" not in rendered


def test_addon_download_is_not_enabled_for_other_pilot_pages():
    page = ROOT / "chapter7_activation_to_language/instructions/pages/01_[7.1]_Logit_Lens_Tuned_Lens_and_Patchscopes.md"
    link = "[practice](../../../mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb)"
    assert page_links.rewrite_page_links(link, page) == link
