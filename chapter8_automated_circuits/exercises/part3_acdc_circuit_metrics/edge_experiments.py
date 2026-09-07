"""Provided notebook checkpoint plumbing; all discovery methods are passed in.

Completed-learner validation sets fresh=True. Interactive reruns may reuse a
completed search only after matching model/data/function context and checking
its final loss again with the live learner implementation.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import tempfile
import textwrap
from pathlib import Path

import torch as t


def checkpointed_learner_search(graph, evaluate_loss, *, threshold, search_fn,
                               core_functions: dict, data: dict, model_revision: str,
                               output_directory: Path, fresh: bool = False):
    """Wrap the passed student search with a source/data-specific durable checkpoint."""
    implementations = {}
    for name, function in {**core_functions, "search_fn": search_fn, "evaluate_loss": evaluate_loss}.items():
        source = textwrap.dedent(inspect.getsource(inspect.unwrap(function)))
        semantic = ast.dump(ast.parse(source), include_attributes=False)
        implementations[name] = hashlib.sha256(semantic.encode()).hexdigest()
    serializable_data = {key: value.tolist() if isinstance(value, t.Tensor) else value
                         for key, value in data.items()}
    context = {
        "threshold": threshold, "model_revision": model_revision,
        "function_semantic_sha256": implementations,
        "data_sha256": hashlib.sha256(json.dumps(serializable_data, sort_keys=True).encode()).hexdigest(),
        "graph_sources": list(graph.sources), "graph_receivers": [r.name for r in graph.receivers],
        "legal_mask_sha256": hashlib.sha256(graph.legal_mask.cpu().numpy().tobytes()).hexdigest(),
        "torch_version": t.__version__,
    }
    context_key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:16]
    output_directory.mkdir(parents=True, exist_ok=True)
    directory = (Path(tempfile.mkdtemp(prefix=f"fresh-{context_key}-", dir=output_directory)) if fresh
                 else output_directory / context_key)
    directory.mkdir(parents=True, exist_ok=True)
    context_path, checkpoint = directory / "context.json", directory / "checkpoint.json"
    # A fresh output may continue this same learner's interrupted search chain.
    # The executor validates ownership and unchanged method sources first; each
    # copied checkpoint still must match the actual student functions and data.
    inventory_path = os.environ.get("ARENA_ACDC_OWNED_RESUME_INVENTORY") if fresh else None
    if inventory_path:
        inventory = json.loads(Path(inventory_path).read_text())
        matches = [entry for entry in inventory["checkpoints"] if entry["context"] == context]
        if len(matches) > 1:
            raise ValueError("ambiguous owned checkpoint for this learner context")
        if matches:
            entry = matches[0]
            path = Path(entry["snapshot_path"])
            contents = path.read_bytes()
            if hashlib.sha256(contents).hexdigest() != entry["checkpoint_sha256"]:
                raise ValueError("owned checkpoint changed after executor validation")
            checkpoint.write_bytes(contents)
            (directory / "owned_continuation.json").write_text(json.dumps({
                "prior_execution_sha256": inventory["prior_execution_sha256"],
                "prior_checkpoint_sha256": entry["checkpoint_sha256"],
                "prior_checkpoint": entry["prior_checkpoint"],
            }, indent=2) + "\n")
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("checkpoint context differs from the current learner implementation")
    context_path.write_text(json.dumps(context, indent=2) + "\n")
    previous = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    if previous is not None and not previous.get("complete"):
        mask = t.zeros_like(graph.legal_mask)
        if previous["active_edges"]:
            indices = t.tensor(previous["active_edges"])
            mask[indices[:, 0], indices[:, 1]] = True
        last = previous["trace"][-1] if previous["trace"] else None
        recorded = (last["trial_loss"] if last["removed"] else last["before_loss"]) if last else previous["initial_loss"]
        if abs(float(evaluate_loss(mask)) - recorded) > 1e-4:
            raise ValueError("partial checkpoint loss differs from its live recomputation")
        print(f"Continuing {len(previous['trace'])} owned edge trials at {directory}; loss rechecked live.")
    if previous is not None and previous.get("complete"):
        mask = t.zeros_like(graph.legal_mask)
        if previous["active_edges"]:
            indices = t.tensor(previous["active_edges"])
            mask[indices[:, 0], indices[:, 1]] = True
        measured = float(evaluate_loss(mask))
        if abs(measured - previous["final_loss"]) > 1e-4:
            raise ValueError("cached circuit loss differs from its live recomputation")
        print(f"Reused matching completed learner search at {directory}; final loss rechecked live.")
        return previous
    return search_fn(graph, evaluate_loss, threshold=threshold,
                     checkpoint_path=checkpoint, resume=previous)
