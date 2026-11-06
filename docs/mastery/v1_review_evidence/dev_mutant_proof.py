"""Reviewer-only distinct DEV mutation check; never access or construct the target."""
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
started = time.process_time()
path = Path("/tmp/arena-graph-blind-build-20261020/mastery_hub/units/graph_prediction/lesson.py")
spec = importlib.util.spec_from_file_location("review_graph_lesson", path)
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)
import torch as t

t.set_num_threads(1)
t.set_default_dtype(t.float32)
t.set_default_device("cpu")
reference = lesson.REFERENCE_METHODS
mutations = {
    "IG_replaced_with_half_exact": (
        "    if type(steps) is not int or steps <= 0:",
        "    return exact_edge_effects(graph) / 2\n    if type(steps) is not int or steps <= 0:",
    ),
    "IG_ignores_requested_steps": (
        "    if type(steps) is not int or steps <= 0:",
        "    steps = 1\n    if type(steps) is not int or steps <= 0:",
    ),
    "denominator_omits_corrupt": (
        "denominator = metrics['full']-metrics['corrupt']",
        "denominator = metrics['full']",
    ),
    "minimality_uses_maximum": (
        "'minimality':min(path_result['drops'])/denominator",
        "'minimality':max(path_result['drops'])/denominator",
    ),
}

# This exposed anonymous three-edge chain is not the seven-node target.
chain = {
    "node_order": ["dev-a", "dev-b", "dev-c", "dev-d"],
    "edge_weights": [[0, 1, 2.], [1, 2, .5], [2, 3, 3.]],
    "clean_inputs": [1., 0., 0., 0.],
    "corrupt_inputs": [.25, 0., 0., 0.],
    "source_index": 0, "target_index": 3, "max_depth": 3,
}
# M(alpha) = 3 alpha^3, each gate derivative is 3 alpha^2.
# Four midpoints have squares (1,9,25,49)/64, mean 21/64.
# Thus each IG edge is 63/64=.984375, independently derived.
expected_ig = 63 / 64
parallel = {**chain, "edge_weights": [*chain["edge_weights"], [1, 3, .5], [0, 3, 1.]]}
# Paths contribute 3 (0,1,2,3), 1 (0,1,3), 1 (0,3).
# Clean full=5, quarter-input corrupt=1.25; kept chain=3; removed chain=1.
# Full-background drops along the chain are 4,3,3, not equal.
expected_path = {
    "metrics": {"full": 5., "corrupt": 1.25, "path_only": 3., "path_removed": 1.},
    "drops": [4., 3., 3.],
}
fixture_controls = {
    "metrics": {"selected": 3., "random": 1., "reversed": 0.},
    "random_edges": [], "reversed_edges": [],
}
# Controls are scalar assembly-only placeholders, not a complete measured receipt.
expected_ratios = {"faithfulness": 7 / 15, "completeness": 16 / 15, "minimality": 4 / 5}
rows = []
for name, mutation in {"reference": None, **mutations}.items():
    source = reference if mutation is None else reference.replace(*mutation)
    assert mutation is None or source != reference
    ns = lesson.practice_namespace(source)
    make = ns["make_graph"]
    def distinct_only(case, _make=make):
        assert len(case["node_order"]) == 4
        assert case["node_order"][0] in {"practice-input", "dev-a"}
        return _make(case)
    ns["make_graph"] = distinct_only
    passes = [lesson.practice_test(ns, stage) for stage in (
        "graph", "scores", "selection", "interventions", "assembly"
    )]
    graph = ns["make_graph"](chain)
    scores = ns["integrated_scores"](graph, 4)
    actual_ig = [scores[u, v].item() for u, v in ((0, 1), (1, 2), (2, 3))]
    parallel_graph = ns["make_graph"](parallel)
    parallel_exact = ns["exact_edge_effects"](parallel_graph)
    fixture_path = ns["path_report"](parallel_graph, [0, 1, 2, 3], parallel_exact)
    assert fixture_path == expected_path
    ratios = ns["assemble_result"](
        t.zeros((4, 4)), t.zeros((4, 4)), [], [], fixture_path, fixture_controls
    )["ratios"]
    rows.append({
        "name": name, "method_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "all_five_current_practice_stages_pass": len(passes) == 5,
        "three_edge_chain_IG_expected": [expected_ig] * 3,
        "three_edge_chain_IG_observed": actual_ig,
        "IG_matches_independent_oracle": actual_ig == [expected_ig] * 3,
        "nonzero_corrupt_unequal_drops_expected_ratios": expected_ratios,
        "nonzero_corrupt_unequal_drops_observed_ratios": ratios,
        "ratios_match_independent_oracle": ratios == expected_ratios,
    })
assert rows[0]["IG_matches_independent_oracle"] and rows[0]["ratios_match_independent_oracle"]
assert all(row["all_five_current_practice_stages_pass"] for row in rows)
assert all(not (row["IG_matches_independent_oracle"] and row["ratios_match_independent_oracle"]) for row in rows[1:])
output = {
    "scope": "distinct exposed DEV only; no target access, construction, evaluation, or result",
    "source_lesson_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "proof_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cpu_seconds": time.process_time() - started,
    "results": rows,
}
Path(__file__).with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps({"all_mutants_evade_current_DEV_tests": True, "mutants": len(mutations),
                  "cpu_seconds": output["cpu_seconds"]}))
