"""Own-answer, visible-dependency and artifact contracts for the analytic draft."""
import hashlib
import inspect
import json

import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup

from . import build_notebooks as builder, spex_methods, spex_plots as plots


def test_own_answers_are_visible_and_complete():
    student, solved = builder.build_notebooks()
    completed = builder.complete_from_own_answers(student)
    assert completed["cells"] == solved["cells"]
    groups = [c for c in student["cells"] if c["metadata"].get("arena", {}).get("exercise_key")]
    assert len(groups) == 12
    functions = [name for cell in groups for name in cell["metadata"]["arena"]["exercise_functions"]]
    assert len(functions) == 22 and len(set(functions)) == 22
    assert {"berlekamp_massey", "chien_search", "recover_spectrum", "masked_word_inputs"} <= set(functions)
    for cell in student["cells"]:
        if cell["metadata"].get("arena", {}).get("solution_for"):
            soup = BeautifulSoup(cell["source"], "html.parser")
            details = soup.find("details")
            assert details.find("code") and "open" not in details.attrs
            assert "def " in details.find("code").get_text()
        if cell["cell_type"] == "markdown":
            for details in BeautifulSoup(cell["source"], "html.parser").find_all("details"):
                assert details.find("div"), "Collapsed labels must contain their actual help/solution body"


def test_answer_functions_use_learner_globals(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_SPEX_OUTPUT_DIRECTORY", str(tmp_path))
    _, solved = builder.build_notebooks()
    namespace = {"__name__": "learner_injected"}
    for cell in solved["cells"]:
        info = cell["metadata"].get("arena", {})
        if cell["cell_type"] == "code" and (info.get("exercise_key") or info.get("calling_cell") == "setup"):
            exec(compile(cell["source"], f"<learner:{cell['id']}>", "exec"), namespace)
    for group in builder.EXERCISES.values():
        for name in group["functions"]:
            function = inspect.unwrap(namespace[name])
            assert function.__globals__ is namespace
            assert function.__module__ == "learner_injected"


def test_saved_analytic_failure_receipt(tmp_path):
    games = plots.analytic_games(31)
    payload = plots.run_analytic_suite(spex_methods, n=31, seeds=(0,), budgets=(3,),
                                       games={"sparse": games["sparse"], "shuffled": games["shuffled"]})
    plots.save_analytic_payload(payload, tmp_path)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert len(saved["records"]) == 2
    assert saved["records"][1]["nrmse"] > .5
    for name, digest in json.loads((tmp_path / "manifest.json").read_text()).items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    for record in saved["records"]:
        with np.load(tmp_path / record["arrays"], allow_pickle=False) as arrays:
            assert arrays["masks"].shape == (record["requested"], 31)
            assert len(np.unique(arrays["masks"], axis=0)) == record["unique"]
            prediction = sum(row["coefficient"] * spex_methods.boolean_character(payload["audit_masks"], tuple(row["support"])) for row in record["spectrum"])
            np.testing.assert_allclose(prediction, arrays["predictions"], atol=1e-12)


def test_native_plot_contract_without_model_queries():
    # Synthetic schema fixture only. It is not saved as a lesson native result.
    actual = np.linspace(-1, 1, 20)
    base = {"prompt_index": 0, "prompt_id": "contract_fixture", "method": "spex", "seed": 0,
            "budget": 456, "audit_values": actual,
            "baseline_predictions": {"constant": np.zeros(20), "additive": .5 * actual},
            "spectrum": {(0, 2): -.4}, "query_receipt": {"raw_requested": 456, "unique_model_masks": 418}}
    payload = {"prompts": [{"prompt_index": 0, "feature_names": [f"word{i}" for i in range(28)]}],
               "results": [dict(base, condition="real", predictions={"final": .9 * actual}),
                           dict(base, condition="shuffled", predictions={"final": actual[::-1]})]}
    figure = plots.plot_native_comparison(payload)
    assert len(figure.axes) == 3
    assert "456 requests, 418 unique masks" in figure._suptitle.get_text()
    assert any("word0" in label.get_text() and "word2" in label.get_text() for label in figure.axes[1].get_yticklabels())
    plt.close(figure)


def test_native_cells_refit_student_namespace_and_have_no_live_run_switch():
    student, _ = builder.build_notebooks()
    cells = {c.get("metadata", {}).get("arena", {}).get("calling_cell"): c
             for c in student["cells"] if c["cell_type"] == "code"}
    native = "\n".join(cells[key]["source"] for key in cells if key and key.startswith("native_"))
    assert "run_study(" not in native and "RUN_NATIVE_STUDY" not in native
    assert "load_native_evidence()" in native
    assert native.count("native_learner.replay_condition(") == 2
    assert native.count('method="spex", methods=methods') == 2
    assert "prompt_index=0, seed=0, budget=1824" in native
    assert "recorded completed-study metrics" in native
    assert "if native_payload" not in native


def test_generated_page_uses_canonical_asset_names(tmp_path):
    student, solved = builder.build_notebooks()
    path = builder.render_instruction_markdown(student, solved, tmp_path / "page.md")
    text = path.read_text()
    assert "<!-- code:" not in text and "<!-- exercise:" not in text
    assert text.count("](../assets/1618_") == 12
    assert "1618_09_student_native_recovery.png" in text
    assert "1618_12_student_native_prompt_play.png" in text
    assert "<details><summary>Solution" in text
