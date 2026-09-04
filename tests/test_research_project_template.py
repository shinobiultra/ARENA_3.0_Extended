import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "research_projects/00_project_template"
REQUIRED_FILES = {
    "README.md",
    "question.md",
    "literature_map.md",
    "experiment_registry.csv",
    "hypothesis_tracker.md",
    "baselines.md",
    "baselines_checklist.md",
    "falsification_tests.md",
    "failure_modes.md",
    "paper_skeleton.md",
    "method_cards/README.md",
    "method_cards/template.md",
    "research_log.ipynb",
    "results.ipynb",
}
EXPERIMENT_REGISTRY_COLUMNS = {
    "run_id",
    "date",
    "status",
    "question",
    "benchmark",
    "seed",
    "model",
    "dataset",
    "baselines",
    "negative_controls",
    "ood_split",
    "causal_interventions",
    "metrics_path",
    "artifact_dir",
    "known_failures",
    "next_action",
}


def test_research_project_template_has_required_files():
    missing = [
        relative
        for relative in sorted(REQUIRED_FILES)
        if not (TEMPLATE / relative).exists()
    ]

    assert missing == []


def test_experiment_registry_schema_matches_capstone_contract():
    with (TEMPLATE / "experiment_registry.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert set(rows[0]) == EXPERIMENT_REGISTRY_COLUMNS
    assert rows[0]["status"] == "planned"
    assert "random_control" in rows[0]["baselines"]
    assert "random_activation" in rows[0]["negative_controls"]
    assert "patching" in rows[0]["causal_interventions"]


def test_research_notebook_stubs_are_valid_and_output_free():
    for filename in ("research_log.ipynb", "results.ipynb"):
        notebook = json.loads((TEMPLATE / filename).read_text())

        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        for cell in notebook["cells"]:
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") in {None, 0}
