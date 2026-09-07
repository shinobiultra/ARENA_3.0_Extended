"""Ensure long notebook runs cannot silently reuse stale or unchecked results."""
from __future__ import annotations

import json

import pytest

from . import real_edges as edge
from .edge_experiments import checkpointed_learner_search


def test_notebook_checkpoint_reuse_and_fresh_execution(tmp_path):
    graph = edge.build_residual_graph(1, 1)
    calls = []
    def loss(mask):
        calls.append(mask.clone())
        primary = float(mask[0, 0] and mask[1, 4])
        backup = float(mask[0, 3] and mask[2, 4])
        return (1.2 - primary - .2 * backup) ** 2
    kwargs = dict(threshold=.01, search_fn=edge.greedy_edge_acdc,
                  core_functions={"graph": edge.build_residual_graph},
                  data={"task": "two_path_oracle", "seed": 1},
                  model_revision="exact_oracle_v1", output_directory=tmp_path)
    first = checkpointed_learner_search(graph, loss, **kwargs)
    first_calls = len(calls)
    assert first_calls > 1
    cached = checkpointed_learner_search(graph, loss, **kwargs)
    assert cached["trace"] == first["trace"]
    assert len(calls) == first_calls + 1, "A cached final mask must be evaluated live."
    calls.clear()
    fresh = checkpointed_learner_search(graph, loss, **kwargs, fresh=True)
    assert len(calls) == first_calls, "Fresh learner validation must actually execute every search decision."
    assert fresh["active_edges"] == first["active_edges"]
    calls.clear()
    changed = {**kwargs, "data": {"task": "two_path_oracle", "seed": 2}}
    checkpointed_learner_search(graph, loss, **changed)
    assert len(calls) == first_calls, "Different task data must start a separate search."

    # Corrupt a cached result's recorded final loss without changing source/data.
    context_dirs = [path for path in tmp_path.iterdir() if not path.name.startswith("fresh-")]
    assert len(context_dirs) == 2
    for directory in context_dirs:
        path = directory / "checkpoint.json"
        record = json.loads(path.read_text())
        record["final_loss"] += 1.
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="live recomputation"):
        checkpointed_learner_search(graph, loss, **kwargs)


def test_owned_partial_search_continues_same_decisions(tmp_path, monkeypatch):
    import hashlib
    graph = edge.build_residual_graph(1, 1)
    def loss(mask):
        return (1.2 - float(mask[0, 0] and mask[1, 4]) - .2 * float(mask[0, 3] and mask[2, 4])) ** 2
    kwargs = dict(threshold=.01, search_fn=edge.greedy_edge_acdc,
                  core_functions={"graph": edge.build_residual_graph}, data={"task": "oracle"},
                  model_revision="oracle", fresh=True)
    full = checkpointed_learner_search(graph, loss, output_directory=tmp_path / "first", **kwargs)
    directory = next((tmp_path / "first").iterdir())
    partial = {**full, "complete": False, "trace": full["trace"][:3], "skipped": []}
    active = graph.legal_mask.clone()
    for row in partial["trace"]:
        if row["removed"]:
            active[row["source"], row["receiver"]] = False
    partial["active_edges"] = active.nonzero().tolist()
    checkpoint = tmp_path / "owned.json"
    checkpoint.write_text(json.dumps(partial))
    inventory = {"prior_execution_sha256": "a" * 64, "checkpoints": [{
        "context": json.loads((directory / "context.json").read_text()),
        "snapshot_path": str(checkpoint), "prior_checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}]}
    inventory_path = tmp_path / "inventory.json";inventory_path.write_text(json.dumps(inventory))
    monkeypatch.setenv("ARENA_ACDC_OWNED_RESUME_INVENTORY", str(inventory_path))
    continued = checkpointed_learner_search(graph, loss, output_directory=tmp_path / "continued", **kwargs)
    assert continued["trace"] == full["trace"]
    assert continued["active_edges"] == full["active_edges"]
    assert len({(row["source"], row["receiver"]) for row in continued["trace"]}) == len(continued["trace"])
    checkpoint.write_text(json.dumps({**partial, "initial_loss": 100.}))
    with pytest.raises(ValueError, match="changed after executor validation"):
        checkpointed_learner_search(graph, loss, output_directory=tmp_path / "tampered", **kwargs)


def test_owned_resume_rejects_method_changes_and_noninterrupts(tmp_path):
    from .execute_learner import prepare_owned_resume, file_hash
    prior = tmp_path / "prior";prior.mkdir()
    lesson = prior / "lesson_snapshot.md";lesson.write_text("unchanged lesson")
    notebook = prior / "completed_learner.ipynb"
    notebook.write_text(json.dumps({"cells": [{"outputs": [{"output_type": "error", "ename": "KeyboardInterrupt"}]}]}))
    source_hashes = {"lesson.md": file_hash(lesson), "real_edges.py": "a" * 64,
                     "execute_learner.py": "b" * 64, "edge_experiments.py": "c" * 64}
    record = {"status": "FAIL", "scope": "full", "completed_from_own_answers": True,
              "fresh_searches_required": True, "sources_changed_during_run": [],
              "output_directory_override": str(prior.resolve()), "input_files": source_hashes}
    (prior / "execution.json").write_text(json.dumps(record))
    checkpoint = prior / "searches" / "fresh-test";checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.json").write_text('{"complete": false}')
    (checkpoint / "context.json").write_text('{"threshold": 0.01}')
    output = tmp_path / "valid";output.mkdir()
    result = prepare_owned_resume(prior, output, {**source_hashes, "execute_learner.py": "d" * 64})
    assert result["infrastructure_changes"] == ["execute_learner.py"] and len(result["checkpoints"]) == 1
    with pytest.raises(ValueError, match="method source changed"):
        prepare_owned_resume(prior, tmp_path / "invalid", {**source_hashes, "real_edges.py": "e" * 64})
    notebook.write_text(json.dumps({"cells": [{"outputs": [{"output_type": "error", "ename": "AssertionError"}]}]}))
    with pytest.raises(ValueError, match="Only an interrupted"):
        prepare_owned_resume(prior, tmp_path / "failed-test", source_hashes)
