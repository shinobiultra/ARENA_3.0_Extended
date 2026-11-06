"""Learner-facing graph feedback and copied-input play: CPU DEV arithmetic only."""
from pathlib import Path
from types import SimpleNamespace
import ast
import importlib.util
import json

import pytest

REPO = Path(__file__).resolve().parents[1]
FOLDER = REPO / "mastery_hub/units/acdc_own_method"


@pytest.fixture
def lesson():
    spec = importlib.util.spec_from_file_location("acdc_practice_semantics", FOLDER / "generate.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    ns = {}
    exec(generator.GRAPH, ns)
    exec(generator.GRAPH_CHECK, ns)
    return generator, ns


def test_reference_graph_passes_learner_semantic_checks(lesson, capsys):
    _, ns = lesson
    ns["check_graph_dev"](ns["own_graph"], ns["own_gpt2_receiver_order"], ns["own_gpt2_receiver_owner"])
    assert "reverse topology and owners" in capsys.readouterr().out


@pytest.mark.parametrize("bad_order", [lambda g: list(range(len(g.RECEIVER_NAMES))),
                                      lambda g: list(reversed(range(len(g.RECEIVER_NAMES))))])
def test_permutations_are_insufficient(lesson, bad_order):
    _, ns = lesson
    with pytest.raises(AssertionError, match="reverse receiver order"):
        ns["check_graph_dev"](ns["own_graph"], bad_order, ns["own_gpt2_receiver_owner"])


def test_all_none_owners_are_rejected(lesson):
    _, ns = lesson
    with pytest.raises(AssertionError, match="receiver owner"):
        ns["check_graph_dev"](ns["own_graph"], ns["own_gpt2_receiver_order"],
                              lambda g: [None] * len(g.RECEIVER_NAMES))


def test_wrong_head_owner_rejected_even_with_valid_logits_owner(lesson):
    _, ns = lesson
    def bad_owner(g):
        owners = ns["own_gpt2_receiver_owner"](g)
        owners[g.head_receiver(0, "k", 0)] = g.mlp_sender(0)
        return owners
    with pytest.raises(AssertionError, match="receiver owner"):
        ns["check_graph_dev"](ns["own_graph"], ns["own_gpt2_receiver_order"], bad_owner)


@pytest.mark.parametrize("case", ["qkv_index", "sender_index", "edge_relation", "names"])
def test_graph_relations_are_independent_of_learner_metadata(lesson, case):
    _, ns = lesson
    def wrong_graph(layers, heads):
        g = ns["own_graph"](layers, heads)
        if case == "qkv_index":
            actual = g.head_receiver
            g.head_receiver = lambda layer, kind, head: actual(layer, {"q": "k", "k": "q", "v": "v"}[kind], head)
        elif case == "sender_index":
            g.head_sender = lambda layer, head: g.mlp_sender(layer)
        elif case == "edge_relation":
            # Preserve size and total number of valid edges: the old checks passed this.
            valid = [list(row) for row in g.valid]
            valid[1][0], valid[1][g.LOGITS_RECEIVER] = valid[1][g.LOGITS_RECEIVER], valid[1][0]
            assert sum(map(sum, valid)) == sum(map(sum, g.valid))
            g.valid = tuple(map(tuple, valid))
        else:
            g.RECEIVER_NAMES = tuple(reversed(g.RECEIVER_NAMES))
        return g
    with pytest.raises(AssertionError):
        ns["check_graph_dev"](wrong_graph, ns["own_gpt2_receiver_order"], ns["own_gpt2_receiver_owner"])


def test_control_stubs_expose_each_callable_and_record_schema(lesson):
    generator, _ = lesson
    factory = ast.parse(generator.CONTROLS_STUB).body[0]
    nested = {node.name: node for node in factory.body if isinstance(node, ast.FunctionDef)}
    assert set(nested) == {"raw_one_edge_oracle", "greater_than_rows", "summarize_rows"}
    assert [arg.arg for arg in nested["raw_one_edge_oracle"].args.args] == [
        "model", "clean_tokens", "corrupt_tokens", "sender", "receiver", "graph"]
    for node in nested.values():
        assert ast.get_docstring(node)
        assert any(isinstance(child, ast.Raise) for child in ast.walk(node))
    for key in ("row_ids", "raw_means", "denominator", "recovery", "random_19", "[batch, vocab]"):
        assert key in generator.CONTROLS_STUB
    ns = {}
    exec(generator.CONTROLS_STUB, ns)
    functions = ns["make_controls"](SimpleNamespace())
    assert [fn.__name__ for fn in functions] == ["raw_one_edge_oracle", "greater_than_rows", "summarize_rows"]


@pytest.mark.parametrize("edited_input", [3.0, 4.0])
def test_play_uses_copies_and_compares_forecasts_without_changing_dev_baseline(lesson, edited_input, capsys):
    import torch as t
    generator, ns = lesson
    for source in (generator.METHOD, generator.CONTROLS, generator.FIXTURE):
        exec(source, ns)
    ns["t"] = t
    ns["own_sender_outputs"], ns["own_circuit_logits"] = ns["make_own_methods"](t)
    ns["raw_one_edge_oracle"], _, _ = ns["make_controls"](t)
    play_source = generator.PLAY.replace("= 3.0  # EDIT", f"= {edited_input}  # EDIT")
    exec(play_source, ns)
    assert ns["baseline_clean"][0, -1, 0].item() == 2.0
    assert ns["play_clean"][0, -1, 0].item() == edited_input
    assert ns["baseline_clean"].data_ptr() != ns["play_clean"].data_ptr()
    assert ns["baseline_corrupt"].data_ptr() != ns["play_corrupt"].data_ptr()
    assert ns["play_oracle"] == ns["play_logits_actual"]
    assert (ns["play_logits_prediction"] == ns["play_logits_actual"]) == (edited_input == 3.0)
    # The original literal check remains correct after the editable play.
    ns["check_sender_dev"](t, ns["own_sender_outputs"])
    output = capsys.readouterr().out
    assert "prediction" in output and "actual" in output and "Independent raw-hook oracle" in output


def test_new_notebook_remains_blank_with_semantic_checks_and_play(lesson):
    generator, _ = lesson
    protocol = json.loads((FOLDER / "protocol.json").read_text())
    book = generator.build_notebook(protocol)
    cells = {c["id"]: c for c in book["cells"]}
    assert "check_graph_dev" in "".join(cells["acdc-own-graph-test"]["source"])
    assert "play_clean" in "".join(cells["acdc-own-dev-play"]["source"])
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))


def test_reference_completed_notebook_runs_visible_cpu_dev_path(lesson, capsys):
    """Reference-derived QA of cell order; never learner mastery or GPT-2 evidence."""
    generator, _ = lesson
    protocol = json.loads((FOLDER / "protocol.json").read_text())
    replacements = {
        "acdc-own-graph-stub": generator.GRAPH,
        "acdc-own-method-stub": (
            "import torch as t\n" + generator.METHOD +
            "\nown_sender_outputs, own_circuit_logits = make_own_methods(t)\n"
        ),
        "acdc-own-controls-stub": generator.CONTROLS,
    }
    namespace = {}
    for cell in generator.build_notebook(protocol)["cells"]:
        if cell["cell_type"] == "code":
            source = replacements.get(cell["id"], "".join(cell["source"]))
            exec(compile(source, cell["id"], "exec"), namespace)
    assert namespace["RUN_PREPARED_GPT2"] is False
    assert namespace["prepared"] is None
    assert namespace["GPT2_PARITY_TOLERANCES"] is None
    assert namespace["play_logits_actual"] == [72., 84., 96.]
    output = capsys.readouterr().out
    assert "cross-position" in output and "Independent raw-hook oracle" in output
