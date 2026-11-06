"""Source-only reading contracts: routes, full parts and untouched notebook downloads."""
from pathlib import Path
import builtins
import importlib.util
import os

import pytest
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "infrastructure/extension/syllabus.yaml").is_file())
SOURCE = Path(os.environ.get("ARENA_READER_SOURCE", ROOT / "mastery_hub/read.py"))


@pytest.fixture
def reader(monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "st_chat" or name.split(".")[0] in {"torch", "transformers", "transformer_lens", "arena_ext", "requests", "socket"} or "solutions" in name:
            raise AssertionError(f"Reader imported teaching/network/chat module {name}")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert SOURCE.is_file(), "The standalone lesson reader is not implemented"
    spec = importlib.util.spec_from_file_location("arena_reading_candidate", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_uses_exact_current_config_and_excludes_historical_collisions(reader):
    rows = reader.inventory(ROOT)
    available = [r for r in rows if r["available"]]
    assert len(rows) == len(available) == 86
    assert len(rows) - len(available) == 0
    config = yaml.safe_load((ROOT / "infrastructure/core/config_extension.yaml").read_text())
    for row in available:
        mapping = config["conversion_map"][row["number"]]
        assert row["page"].name == mapping["streamlit_page"] + ".md"
        assert row["exercises"].parent.name == mapping["exercise_dir"]
        assert row["reference"].parent == row["exercises"].parent
    actual = {r["number"]: r["page"].name for r in available}
    assert actual["5.2"] == "02_[5.2]_Mixture_of_Experts_from_Scratch.md"
    assert actual["8.3"] == "03_[8.3]_AtP_from_Scratch.md"
    assert actual["8.4"] == "04_[8.4]_ACDC_and_Circuit_Metrics.md"
    assert actual["9.1"] == "01_[9.1]_Refusal_Directions_and_Representation_Engineering.md"
    assert actual["12.1"] == "01_[12.1]_CLIP_and_SigLIP_from_Scratch.md"
    assert actual["16.3"] == "03_[16.3]_TreeSHAP_DeepSHAP_and_GradientSHAP_Practice.md"


def test_missing_page_is_actionable_without_historical_fallback(reader, tmp_path):
    for rel in ["infrastructure/extension/syllabus.yaml", "infrastructure/core/config_extension.yaml"]:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
    master = tmp_path / "infrastructure/chapters/chapter5_modern_architectures/master_5_2.py"
    master.parent.mkdir(parents=True)
    master.write_text("# authoring source, never imported\n")
    old = tmp_path / "chapter5_modern_architectures/instructions/pages/02_[5.2]_Gemma_Scope_and_Feature_Steering.md"
    old.parent.mkdir(parents=True)
    old.write_text("historical collision")
    row = next(r for r in reader.inventory(tmp_path) if r["number"] == "5.2")
    assert not row["available"]
    assert "02_[5.2]_Mixture_of_Experts_from_Scratch.md" in row["status"]
    assert "restore" in row["status"].lower()
    assert row["page"] != old


def test_raw_parts_preserve_the_complete_original_order(reader):
    for row in reader.inventory(ROOT):
        if row["available"]:
            raw = row["page"].read_text(encoding="utf-8")
            assert "=== NEW CHAPTER ===".join(reader.parts(row["page"])) == raw
    assert reader.part_title("\n# Introduction\ntext", 0) == "Introduction"
    assert reader.part_title("\n## 2️⃣ Train the model\n", 2) == "2️⃣ Train the model"
    assert reader.part_title("plain text", 3) == "Part 4"


class ReadingUI:
    """Collect only the public UI boundary; any secret access is an immediate failure."""
    def __init__(self, chapter="chapter7_activation_to_language", section="7.1", part=0):
        self.chapter, self.section, self.part = chapter, section, part
        self.sidebar = self
        self.text, self.downloads = [], []

    @property
    def secrets(self):
        raise AssertionError("Reading must not access credentials")

    def set_page_config(self, **kwargs):
        pass

    def title(self, text):
        self.text.append(text)

    def markdown(self, text, **kwargs):
        self.text.append(text)

    def info(self, text):
        self.text.append(text)

    def selectbox(self, label, options, **kwargs):
        if label == "Chapter":
            return self.chapter
        if label == "Lesson":
            return next(r for r in options if r["number"] == self.section)
        return self.part

    def download_button(self, label, data, file_name, **kwargs):
        self.downloads.append((label, data, file_name))


def test_render_displays_adopted_content_and_exact_download_bytes_without_methods(reader):
    ui = ReadingUI(part=3)
    reader.render(ui, ROOT)
    row = next(r for r in reader.inventory(ROOT) if r["number"] == "7.1")
    from infrastructure.extension.render_media import inline_page_media
    text = row["page"].read_bytes().decode("utf-8")
    expected = inline_page_media(text, row["page"]).split("=== NEW CHAPTER ===")[3]
    assert expected in ui.text
    assert any("[7.1] Logit Lens, Tuned Lens and Patchscopes" in text for text in ui.text)
    downloads = {name: data for _, data, name in ui.downloads}
    assert downloads[row["exercises"].name] == row["exercises"].read_bytes()
    assert downloads[row["reference"].name] == row["reference"].read_bytes()
    for url in (
        "https://learn.arena.education/chapter0_fundamentals/",
        "https://learn.arena.education/chapter1_transformer_interp/",
        "https://learn.arena.education/chapter2_rl/",
        "https://learn.arena.education/chapter3_llm_evals/",
        "https://learn.arena.education/chapter4_alignment_science/",
    ):
        assert any(url in text for text in ui.text), url


def test_absent_chapter_has_explicit_status_and_no_download(reader, monkeypatch):
    rows = [dict(row) for row in reader.inventory(ROOT)]
    for row in rows:
        if row["chapter"] == "chapter18_intrinsic_interpretability":
            row.update(available=False, status="Lesson unavailable. Restore the configured fixture page.")
    monkeypatch.setattr(reader, "inventory", lambda root: rows)
    ui = ReadingUI(chapter="chapter18_intrinsic_interpretability", section="18.1")
    reader.render(ui, ROOT)
    assert not ui.downloads
    assert any("not available" in text.lower() for text in ui.text)


SUPPLEMENT_PATHS = {
    "6.5": ["mastery_hub/practice_addons/crosscoder_metric_repair/exercises.ipynb", "mastery_hub/practice_addons/crosscoder_metric_repair/solutions.ipynb", "mastery_hub/practice_addons/crosscoder_metric_repair/README.md"],
    "7.1": ["docs/mastery/pilot_reader_notes_20261023.md", "mastery_hub/units/pilot_practice/7_1_exercises.ipynb"],
    "9.1": ["mastery_hub/units/pilot_practice/9_1_exercises.ipynb"],
    "12.3": ["docs/mastery/pilot_reader_notes_20261023.md", "mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb", "mastery_hub/practice_addons/mini_vlm_memory_handoff/README.md"],
    "8.4": ["mastery_hub/units/acdc_own_method/exercises.ipynb", "mastery_hub/units/acdc_own_method/README.md"],
    "0.6": [
        "chapter0_fundamentals/exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb",
        "chapter0_fundamentals/exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb",
    ],
    "1.6": [
        "chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb",
        "chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb",
    ],
    "16.3": [
        "chapter16_shapley_attribution_baselines/exercises/part13_treeshap/16.13_TreeSHAP_from_First_Principles_exercises.ipynb",
        "chapter16_shapley_attribution_baselines/exercises/part13_treeshap/16.13_TreeSHAP_from_First_Principles_solutions.ipynb",
        "chapter16_shapley_attribution_baselines/exercises/part14_deepshap/16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP_exercises.ipynb",
        "chapter16_shapley_attribution_baselines/exercises/part14_deepshap/16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP_solutions.ipynb",
        "chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/16.15_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate_exercises.ipynb",
        "chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/16.15_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate_solutions.ipynb",
    ],
}


def test_supplements_have_only_finite_section_identities_without_path_fallback(reader):
    for section, paths in SUPPLEMENT_PATHS.items():
        expected = ["mastery_hub/STUDY-GUIDE.md", *paths]
        assert [str(path.relative_to(ROOT)) for _, path in reader.supplements(section, ROOT)] == expected
    for unlisted in ("5.2", "../../private.txt", "/etc/passwd", "mastery_hub/units/acdc_own_method/protocol.json"):
        assert [str(path.relative_to(ROOT)) for _, path in reader.supplements(unlisted, ROOT)] == ["mastery_hub/STUDY-GUIDE.md"]


@pytest.mark.parametrize("section", list(SUPPLEMENT_PATHS))
def test_native_supplements_and_main_notebooks_are_exact_original_bytes(reader, section):
    row = next(r for r in reader.inventory(ROOT) if r["number"] == section)
    ui = ReadingUI(chapter=row["chapter"], section=section)
    reader.render(ui, ROOT)
    expected_paths = [row["exercises"], row["reference"], ROOT / "mastery_hub/STUDY-GUIDE.md"]
    expected_paths.extend(ROOT / rel for rel in SUPPLEMENT_PATHS[section])
    downloads = {name: data for _, data, name in ui.downloads}
    revised = [label for label, _, _ in ui.downloads if label.startswith("Download revised ")]
    assert revised in ([], ["Download revised Study Practice", "Download revised worked reference"])
    assert len(downloads) == len(expected_paths) + 1 + len(revised)
    assert sum(label == "Download blank Practice exercises" for label, _, _ in ui.downloads) == 1
    for path in expected_paths:
        assert downloads[path.name] == path.read_bytes()
    if section in {"7.1", "9.1", "8.4"}:
        assert any("guided public Practice" in label for label, _, _ in ui.downloads)
