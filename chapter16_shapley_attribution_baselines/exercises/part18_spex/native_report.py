"""Assemble external SPEX/ProxySPEX reports from completed, checked evidence.

Report assembly never queries a model or fits an estimator. Completion concerns
coverage and reproduction; failed references and poor prediction fidelity remain
measured outcomes. Independent publication approval belongs to the publisher.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chapter16_shapley_attribution_baselines.exercises.part18_spex import publish_study as publish

STUDY = "16.18_16.19_spex_proxyspex"
SUMMARY_SCHEMA = "arena.sparse-interaction.native-summary.v1"
IDS = {
    "16.18": "16_18_find_the_interactions_without_trying_every_subset",
    "16.19": "16_19_explain_the_proxy_check_the_model",
}
TITLES = {"16.18": "Find the Interactions Without Trying Every Subset", "16.19": "Explain the Proxy, Check the Model"}
REFERENCE_IDS = [f"{variant}_b{bits}" for bits in (3, 4, 5)
                 for variant in ("author_spex", "author_proxy", "maintained_proxy")]
GRID = {(prompt, method, seed, budget, label) for prompt in range(32)
        for method in ("spex", "proxyspex") for seed in range(3)
        for budget in (456, 912, 1824) for label in ("real", "shuffled")}
INVENTORY = {"prompt_count": 32, "condition_count": 1152, "audit_rows_per_prompt": 512, "reference_count": 9}
MEASURED_INVENTORY = {
    "spex_recoveries": 576, "proxy_pipelines": 576, "proxy_ensemble_fits": 7488,
    "ridge_candidate_fits": 9216, "ridge_matrix_decompositions": 2304,
    "baseline_fits": 2304, "physical_evaluations": 481999,
    "completed_condition_records": 1152, "audit_records": 32, "reference_stage_records": 9,
}
RECEIPT_FIELDS = {
    "learner_evidence_sha256": "learner_execution.json",
    "fresh_evidence_sha256": "fresh_reproduction.json",
    "native_summary_sha256": "native_summary.json",
    "native_execution_sha256": "native_execution.json",
    "fresh_execution_sha256": "fresh_native_execution.json",
    "comparison_summary_sha256": "native_comparison_summary.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _positive(value):
    return type(value) is int and value > 0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _prefix(section):
    require(section in IDS, "Unknown sparse-interaction section")
    return f"{publish.CHAPTER}/exercises/{publish.SECTIONS[section][0]}/"


def _identity(row):
    return tuple(row[key] for key in ("prompt_index", "method", "seed", "budget", "condition"))


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _validate_outcomes(summary):
    require(summary.get("inventory", {}).items() >= INVENTORY.items(), "Incomplete prompt/condition/reference inventory")
    observed = summary["inventory"]
    require(all(observed.get(key) == value for key, value in MEASURED_INVENTORY.items()), "Incomplete measured native inventory")
    require(type(observed.get("proxy_residual_tree_fits")) is int
            and 2285568 <= observed["proxy_residual_tree_fits"] <= 2506752, "Proxy tree-fit inventory is outside the fixed grid")
    rows = summary.get("condition_outcomes", [])
    require(len(rows) == len(GRID) and {_identity(row) for row in rows} == GRID, "All 1152 distinct conditions must be retained")
    for row in rows:
        require(row.get("record_origin") == "recorded_complete_native_study", "Recorded outcomes must retain their origin")
        metrics = row.get("metrics", {})
        required = {"constant", "additive", "final"} if row["method"] == "spex" else {
            "constant", "additive", "proxy", "extracted", "truncated", "refined"}
        require(required <= set(metrics), "Condition omitted its estimator or query-matched baseline")
        for values in metrics.values():
            require(set(values) == {"rmse", "nrmse", "r2"} and _finite(values["rmse"])
                    and values["rmse"] >= 0, "Invalid measured fidelity")
            require(all(value is None or _finite(value) for value in values.values()), "Nonfinite measured fidelity")
            require(values["nrmse"] is None or values["nrmse"] >= 0, "Negative normalized error")
        require(row["query_receipt"]["raw_requested"] == row["budget"], "Query budget changed")
        require(_positive(row["query_receipt"]["unique_model_masks"])
                and row["query_receipt"]["unique_model_masks"] <= row["budget"], "Invalid unique query count")
        if row["method"] == "proxyspex":
            require(len(row.get("stage_errors", {})) == 7
                    and all(_finite(value) and value >= 0 for value in row["stage_errors"].values()), "Proxy error stages omitted")
    references = summary.get("reference_outcomes", [])
    require([row.get("stage_id") for row in references] == REFERENCE_IDS, "All nine ordered reference outcomes are required")
    for row in references:
        require(isinstance(row.get("status"), str) and row["status"]
                and isinstance(row.get("process_status"), str) and row["process_status"], "Reference status missing")
        require("scientific_status" in row and isinstance(row.get("metrics"), dict), "Reference scientific outcome missing")
        require(row.get("record_origin") == "recorded_prompt0_seed0_reference_stage", "Reference scope changed")
    prompts = summary.get("prompt_outcomes", [])
    require(len(prompts) == 32 and {row["prompt_index"] for row in prompts} == set(range(32)), "Prompt outcomes incomplete")
    for row in prompts:
        require(len(row["audit_values"]) == 512 and all(_finite(value) for value in row["audit_values"]), "Incomplete prompt audit scores")
        require(len(row["unmasked"]) == len(row["counterfactual"]) == 1
                and all(_finite(value) for value in row["unmasked"] + row["counterfactual"]), "Primary/counterfactual outcomes incomplete")


def _validate_comparison(comparison):
    require(comparison.get("status") == "PASS" and comparison.get("mismatch_count") == 0, "Cold reproduction did not pass")
    counts = comparison.get("counts", {})
    require(counts.get("original") == counts.get("fresh") and isinstance(counts.get("original"), dict), "Reproduction inventory differs")
    require(all(_positive(counts["original"].get(key)) for key in ("arrays", "array_elements", "scalars", "containers")), "Reproduction counts missing")
    difference_count = comparison.get("numerical_difference_leaves")
    require(type(difference_count) is int and difference_count >= 0, "Numerical differences must be counted")
    require(comparison.get("bitwise_exact") is (difference_count == 0), "Toleranced differences must not be reported as bitwise exact")
    require(_hash(comparison.get("comparison_sha256")), "Full comparison hash missing")


def _validate_learner(learner, bindings):
    require(learner.get("status") in {"passed", "FULL_LEARNER_PASS"}
            and learner.get("sources_unchanged") is True, "Complete unchanged-source own-answer execution required")
    require(_positive(learner.get("executed_code_cells"))
            and learner["executed_code_cells"] == learner.get("total_code_cells"), "Learner cell coverage incomplete")
    require(isinstance(learner.get("figures"), dict) and learner["figures"]
            and all(_hash(value) for value in learner["figures"].values()), "Actual learner figures must be hash-bound")
    for key in ("native_execution_sha256", "fresh_reproduction_sha256"):
        require(learner.get(key) == bindings[key], "Learner belongs to different native/reproduction evidence")
    replay = learner.get("portable_replay", {})
    require(replay.get("status") == "PASS" and replay.get("no_full_study_cache_required") is True
            and replay.get("fixture_sha256") == bindings["fixture_sha256"], "Portable replay did not pass with the shared fixture")
    require(replay.get("new_model_queries", learner.get("model_queries")) == 0, "Actual zero-query replay required")
    _replay_paths(learner)


def _replay_paths(learner):
    """The two authored notebooks retain their different original receipt formats."""
    paths = learner["portable_replay"].get("replay_records")
    if paths is not None:
        require(isinstance(paths, list) and len(paths) == 2 and len(set(paths)) == 2
                and all(isinstance(path, str) and Path(path).is_absolute() for path in paths), "Require two actual replay directories")
        return paths
    receipts = learner.get("replay_receipts", {})
    require(set(receipts) == {"native_default_replay.json", "native_play_replay.json"}
            and all(_hash(value) for value in receipts.values()), "Require both hash-bound ProxySPEX replay receipts")
    return ["native_default_replay.json", "native_play_replay.json"]


def _validate_replays(outcomes, learner, section):
    require([row["record_path"] for row in outcomes] == _replay_paths(learner), "Replay inventory changed")
    expected_method = "spex" if section == "16.18" else "proxyspex"
    for outcome in outcomes:
        expected_files = ({outcome["record_path"]} if outcome["record_path"].endswith(".json") else {"record.json", "arrays.npz"})
        require(set(outcome["hashes"]) == expected_files
                and all(_hash(value) for value in outcome["hashes"].values()), "Replay physical hashes missing")
        if outcome["record_path"].endswith(".json"):
            require(outcome["hashes"][outcome["record_path"]] == learner["replay_receipts"][outcome["record_path"]], "Replay hash differs from learner")
        row = outcome["replay"]
        require(row.get("status") == "COMPLETE" and row.get("new_model_queries") == 0
                and row.get("new_physical_model_rows") == 0, "Replay incorrectly claims model execution")
        require(row.get("audit_labels_received_by_fit") is False
                and row.get("estimator_refits") == 2 and row.get("baseline_fits") == 4, "Replay fit boundary or coverage changed")
        require((row["prompt_index"], row["method"], row["seed"], row["budget"], "real") in GRID, "Undeclared replay condition")
        require(row["method"] == expected_method, "Learner replayed the other section's method")
        stages = {"final", "constant", "additive"} if expected_method == "spex" else {
            "proxy", "extracted", "truncated", "refined", "constant", "additive"}
        comparisons = row.get("comparisons", [])
        require(len(comparisons) == 2 and {item.get("condition") for item in comparisons} == {"real", "shuffled"},
                "Both actual replay controls require pointwise comparisons")
        for comparison in comparisons:
            require(set(comparison.get("prediction_allclose", {})) == stages
                    and all(value is True for value in comparison["prediction_allclose"].values())
                    and comparison.get("prediction_tolerance") == {"atol": 1e-8, "rtol": 1e-7},
                    "Student replay predictions differ from the recorded calculation")
            deltas = comparison.get("maximum_prediction_absolute_delta", {})
            require(set(deltas) == stages and all(_finite(value) and value >= 0 for value in deltas.values()),
                    "Replay numerical differences must remain recorded")


def _decode_selected(value, arrays):
    if isinstance(value, dict) and set(value) == {"__array__"}:
        return arrays[value["__array__"]]
    if isinstance(value, dict):
        return {key: _decode_selected(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_decode_selected(item, arrays) for item in value)
    return value


def _verify_portable_outcomes(native_run, portable, native_summary):
    """Stream only prediction arrays, never fitted states, to verify exported metrics."""
    from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_learner

    rows = {_identity(row): row for row in portable["results"]}
    audit = {row["prompt_index"]: row for row in portable["audit"]}
    study = Path(native_run) / "study"
    for entry in native_summary["audit_records"]:
        meta = publish._study_record(study / entry["path"], entry["hashes"])
        with np.load(study / entry["path"] / "arrays.npz", allow_pickle=False) as arrays:
            actual = _plain(_decode_selected(meta, arrays))
        require(actual == audit[actual["prompt_index"]], "Portable audit values differ from physical native records")
    for entry in native_summary["conditions"]:
        meta = publish._study_record(study / entry["path"], entry["hashes"])
        selected = {key: meta[key] for key in ("prompt_index", "prompt_id", "method", "seed", "budget", "condition",
                                              "query_receipt", "predictions", "baseline_predictions", "spectrum")}
        with np.load(study / entry["path"] / "arrays.npz", allow_pickle=False) as arrays:
            actual = _decode_selected(selected, arrays)
        actual["audit_values"] = np.asarray(audit[actual["prompt_index"]]["audit_values"], dtype=np.float64)
        compact = _plain(native_learner._compact_result(actual))
        require(compact == rows[_identity(compact)], "Portable condition differs from physical native predictions")
    original_refs = native_summary["references"]
    compact_refs = [native_learner._compact_reference(row, np.asarray(audit[0]["audit_values"])) for row in original_refs]
    require(_plain(compact_refs) == portable["references"], "Portable references omitted or changed an outcome")


def _validated_inputs(section, native_run, fresh_run, learner_path, fixture_path):
    native = publish.validate_native_pair(native_run, fresh_run)
    learner = publish.read_json(learner_path)
    publish.bundles.validate_json_data(Path(fixture_path))
    from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_learner

    native_learner.load_native_evidence(fixture_path)
    # Preserve the original compact JSON, before the loader adds numerical views.
    import gzip
    portable = json.loads(gzip.decompress(Path(fixture_path).read_bytes()))
    bindings = {
        "native_execution_sha256": native["native_execution"]["sha256"],
        "fresh_execution_sha256": native["fresh_execution"]["sha256"],
        "fresh_reproduction_sha256": native["reproduction"]["sha256"],
        "learner_execution_sha256": publish.bundles.sha256(Path(learner_path)),
        "fixture_sha256": publish.bundles.sha256(Path(fixture_path)),
    }
    require(all(portable["binding"].get(key) == bindings[key]
                for key in ("native_execution_sha256", "fresh_reproduction_sha256")), "Portable fixture has different native lineage")
    _validate_learner(learner, bindings)
    source_hashes = learner.get("source_sha256", learner.get("source_inputs", {}))
    require(isinstance(source_hashes, dict) and source_hashes, "Actual learner source inventory is required")
    if "source_sha256" in learner:
        require(source_hashes == learner.get("source_sha256_after"), "Learner source hashes changed during execution")
    else:
        require(learner.get("changed_sources") == [], "Learner sources changed during execution")
    for name, checksum in source_hashes.items():
        require(_hash(checksum), "Invalid learner source hash")
        relative = name if "/" in name else _prefix(section) + name
        publish.safe_relative(relative)
        require(publish.bundles.sha256(ROOT / relative) == checksum, f"Current learner source differs from execution: {relative}")
    require(publish.bundles.sha256(Path(learner["executed_notebook"])) == learner["notebook_sha256"], "Own-answer notebook changed")
    replay_outcomes = []
    for locator in _replay_paths(learner):
        path = (Path(learner_path).resolve().parent / locator).resolve()
        require(path.is_relative_to(Path(learner_path).resolve().parent), "Replay record belongs to another learner directory")
        if path.is_file():
            require(publish.bundles.sha256(path) == learner["replay_receipts"][locator], "Proxy replay bytes changed")
            replay, hashes = publish.read_json(path), {locator: publish.bundles.sha256(path)}
        else:
            replay, hashes = publish._study_record(path)["replay"], publish.read_json(path / "hashes.json")
        replay_outcomes.append({"record_path": locator, "hashes": hashes, "replay": replay})
    _, original = publish.validate_native_run(native_run)
    require(original["source_inputs"] == portable["source_inputs"], "Portable fixture changed frozen source identities")
    _verify_portable_outcomes(native_run, portable, original)
    comparison = publish.read_json(native["comparison_summary"]["absolute_path"])
    _validate_comparison(comparison)
    return {"native": native, "learner": learner, "portable": portable, "bindings": bindings,
            "comparison": comparison, "replay_outcomes": replay_outcomes,
            "native_receipt": publish.read_json(native["native_execution"]["absolute_path"]),
            "fresh_receipt": publish.read_json(native["fresh_execution"]["absolute_path"])}


def _summary(section, context):
    portable, native, learner = (context[key] for key in ("portable", "native", "learner"))
    summary = {
        "schema": SUMMARY_SCHEMA, "study": STUDY, "section": section, "status": "COMPLETE_VERIFIED_STUDY",
        "inventory": {**INVENTORY, **native["observed_inventory"]}, "bindings": context["bindings"],
        "condition_outcomes": deepcopy(portable["results"]), "reference_outcomes": deepcopy(portable["references"]),
        "prompt_outcomes": deepcopy(portable["audit"]), "comparison": context["comparison"],
        "portable_replay": deepcopy(learner["portable_replay"]),
        "replay_outcomes": context["replay_outcomes"],
        "interpretation": {
            "acceptance": "Complete recorded experiment, physical evidence verification and strict cold reproduction; not predictive superiority.",
            "reference_failures": "Every reference status, exception and available metric is retained; completion does not require source success.",
            "learner_scope": "Only portable_replay.replay_records were recomputed in this notebook. Complete study curves are recorded comparisons.",
            "full_spectra": "The portable fixture displays up to twelve terms per recorded condition. Complete spectra remain in the native study.",
            "publication": "Independent publication review is separate from report assembly.",
        },
    }
    _validate_outcomes(summary)
    _validate_replays(summary["replay_outcomes"], learner, section)
    return summary


def _metric_summary(summary, learner):
    counts = summary["comparison"]["counts"]["original"]
    values = {
        "study": STUDY, "section": summary["section"], "device": "cpu", "actual_learner_passed": True,
        "native_execution_complete": True, "fresh_execution_complete": True, "fresh_reproduction_passed": True,
        "cached_learner_models_or_values_used_for_native_execution": False,
        "native_model_queries_in_report": 0, "native_estimator_fits_in_report": 0,
        "recorded_study_reexecuted_in_notebook": False, "method_superiority_is_acceptance_gate": False,
        "reference_success_is_acceptance_gate": False, "publication_approval_claimed": False,
        **summary["inventory"], "fresh_compared_arrays": counts["arrays"], "fresh_compared_elements": counts["array_elements"],
        "fresh_bitwise_exact": summary["comparison"]["bitwise_exact"],
        "fresh_numerical_difference_leaves": summary["comparison"]["numerical_difference_leaves"],
        "fresh_comparison": "structural_exact_with_disclosed_float_tolerances",
        "own_answer_code_cells": learner["executed_code_cells"], "embedded_figures": len(learner["figures"]),
        "learner_evidence_sha256": summary["bindings"]["learner_execution_sha256"],
        "fresh_evidence_sha256": summary["bindings"]["fresh_reproduction_sha256"],
        **{key: summary["bindings"][key] for key in ("native_execution_sha256", "fresh_execution_sha256", "fixture_sha256")},
        "reference_status_counts": dict(Counter(row["status"] for row in summary["reference_outcomes"])),
        "portable_replay": summary["portable_replay"], "peak_vram_gb": 0.0,
    }
    return values


def has_native_evidence(report):
    """Read-only structural predicate; the fixture gate checks receipt contents/bytes."""
    try:
        values = report["metrics"]["gpu_test"]
        section = values["section"]
        require(report["notebook_id"] == IDS[section], "Section/report identity mismatch")
        expected = {"study": STUDY, "device": "cpu", "actual_learner_passed": True,
                    "native_execution_complete": True, "fresh_execution_complete": True, "fresh_reproduction_passed": True,
                    "cached_learner_models_or_values_used_for_native_execution": False,
                    "native_model_queries_in_report": 0, "native_estimator_fits_in_report": 0,
                    "recorded_study_reexecuted_in_notebook": False, "method_superiority_is_acceptance_gate": False,
                    "reference_success_is_acceptance_gate": False, "publication_approval_claimed": False,
                    "fresh_comparison": "structural_exact_with_disclosed_float_tolerances", **INVENTORY, **MEASURED_INVENTORY}
        require(all(values.get(key) == value and type(values.get(key)) is type(value) for key, value in expected.items()), "Native completion contract changed")
        require(all(_positive(values.get(key)) for key in ("fresh_compared_arrays", "fresh_compared_elements", "own_answer_code_cells", "embedded_figures")), "Measured coverage missing")
        differences = values.get("fresh_numerical_difference_leaves")
        require(type(differences) is int and differences >= 0
                and values.get("fresh_bitwise_exact") is (differences == 0), "False exact-reproduction claim")
        entries = report["report_inputs"]["files"]
        for field, name in RECEIPT_FIELDS.items():
            checksum = values.get(field)
            matches = [row for row in entries if row.get("path") == _prefix(section) + "expected_outputs/" + name]
            require(_hash(checksum) and len(matches) == 1 and matches[0].get("sha256") == checksum, "Receipt binding missing")
        fixture = [row for row in entries if row.get("path") == publish.FIXTURE]
        require(len(fixture) == 1 and fixture[0].get("sha256") == values.get("fixture_sha256"), "Portable fixture binding missing")
        require(sum(values["reference_status_counts"].values()) == 9, "Reference outcomes incomplete")
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def receipt_blockers(section, summary, learner, fresh, report):
    """Section-specific triplet validation after common identity/hash/size checks."""
    try:
        require(has_native_evidence(report), "Sparse native report contract failed")
        require(summary.get("schema") == SUMMARY_SCHEMA and summary.get("section") == section
                and summary.get("study") == STUDY and summary.get("status") == "COMPLETE_VERIFIED_STUDY", "Native summary schema/identity mismatch")
        _validate_outcomes(summary)
        _validate_comparison(summary["comparison"])
        _validate_learner(learner, summary["bindings"])
        _validate_replays(summary["replay_outcomes"], learner, section)
        require(summary["portable_replay"] == learner["portable_replay"], "Summary replay differs from actual learner")
        require(fresh.get("status") == "PASS" and fresh.get("failures") == []
                and fresh.get("native_child_received_learner_payloads") is False
                and fresh.get("cached_learner_models_or_values_used_for_native_execution") is False, "Cold reproduction used incomplete or cached execution")
        for field, binding in (("learner_execution_sha256", "native_execution_sha256"),
                               ("fresh_execution_sha256", "fresh_execution_sha256")):
            require(fresh.get(field) == summary["bindings"][binding], "Cold reproduction uses different execution receipts")
        comparison = summary["comparison"]
        require(fresh.get("comparison_sha256") == comparison["comparison_sha256"]
                and fresh.get("comparison_counts") == comparison["counts"]
                and fresh.get("bitwise_exact") is comparison["bitwise_exact"]
                and fresh.get("numerical_difference_leaves") == comparison["numerical_difference_leaves"]
                and fresh.get("mismatch_count") == comparison["mismatch_count"], "Summary comparison differs from actual reproduction")
        expected = _metric_summary(summary, learner)
        measured = report["metrics"]["gpu_test"]
        require(all(measured.get(key) == value for key, value in expected.items()), "Report metrics differ from native/learner receipts")
        require(report["metrics"].get("native_outcomes") == {key: summary[key] for key in (
            "condition_outcomes", "reference_outcomes", "prompt_outcomes")}, "Report omitted or changed observed native outcomes")
        return []
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return [f"{section}: {error}"]


def _manifest(inputs):
    entries = []
    for relative, path in sorted(inputs.items()):
        publish.safe_relative(relative)
        record = publish.physical_record(path)
        entries.append({"path": relative, "sha256": record["sha256"], "size_bytes": record["size_bytes"]})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"schema_version": 1, "algorithm": "sha256", "files": entries,
            "combined_sha256": hashlib.sha256(payload).hexdigest()}


def build_report(section, native_run, fresh_run, learner_execution, fixture, artifact_inputs, output):
    """Create exact report/lock/schema/summary bytes in a new external directory.

    artifact_inputs maps final canonical repo-relative paths to existing physical
    source files. Include the actual pair, page, figures, tests and regeneration
    instructions. The downstream publisher separately binds the review archive.
    """
    require(section in IDS, "Unknown section")
    output = Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(ROOT), "Report output must be a new external directory")
    context = _validated_inputs(section, native_run, fresh_run, learner_execution, fixture)
    summary = _summary(section, context)
    values = _metric_summary(summary, context["learner"])
    controls = ["Shuffled-response refits on the same query masks", "Query-matched constant and additive baselines",
                "Subject-only counterfactual prompts and incorrect unmasked model answers retained",
                "All declared author/maintained reference outcomes including source failures"]
    claim = (
        f"Student-implemented {'SPEX recovery' if section == '16.18' else 'ProxySPEX fitting, extraction and refinement'} "
        "on fixed-position 28-word GPT-2 masking games. The recorded joint study retains all 32 prompts, 1152 real/shuffled "
        "conditions and nine reference outcomes, including failures and poor held-out fidelity. Portable notebook cells "
        "recompute only their explicitly recorded selected conditions; full-study curves are recorded comparisons. "
        "Cold source-only reproduction verifies the complete study under frozen disclosed tolerances. No predictive "
        "superiority, complete-study notebook rerun, reference success or independent publication approval is inferred."
    )
    lock = {"notebook_id": IDS[section], "section": section, "title": TITLES[section], "execution_device": "cpu",
            "gt_tier": "GT-0", "evidence_level": "complete_fixed_query_native_study_with_portable_student_replay",
            "claim_scope": claim, "required_gpu_gb": 0, "max_allowed_gpu_gb": 0,
            "exercise_metadata": {"EXERCISE_ID": IDS[section], "GT_TIER": "GT-0", "DIFFICULTY": 5,
                                  "IMPORTANCE": 5, "EXPECTED_RUNTIME": "Portable selected replay; optional full study under its approved resource cap",
                                  "REQUIRES_GPU": False},
            "models": [{"id": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
                        "precision": "float32", "source": "pinned local pretrained weights; new physical query values"}],
            "datasets": [{"id": "fixed_32_prompts_28_word_masks", "source": "native_data.py and frozen query plan",
                          "primary_prompts": 32, "subject_counterfactuals": 32, "audit_masks_per_prompt": 512}],
            "controls": controls, "safety_notes": ["Synthetic grammar prompts; fixed attention-mask intervention preserves token positions."],
            "expected_metrics": {**INVENTORY, "native_execution_complete": True, "fresh_reproduction_passed": True,
                                 "method_superiority_is_acceptance_gate": False, "reference_success_is_acceptance_gate": False}}
    from scripts.generate_extension_verification_assets import verification_schema
    schema = verification_schema()
    prefix = _prefix(section)
    inputs = {str(key): Path(path) for key, path in artifact_inputs.items()}
    for role in ("exercises", "solutions", "page"):
        require(publish.artifact_target(section, role) in inputs, f"Report input mapping lacks {role}")
    figure_prefix = publish.SECTIONS[section][3]
    figures = [path for key, path in inputs.items() if key.startswith(f"{publish.CHAPTER}/instructions/assets/{figure_prefix}")
               and key.endswith(".png")]
    require(Counter(publish.bundles.sha256(path) for path in figures) == Counter(context["learner"]["figures"].values()),
            "Report must bind every actual learner figure exactly once")
    for path in [*HERE.glob("*.py"), *HERE.glob("*.md"), *HERE.with_name("part19_proxyspex").glob("*.py"),
                 *HERE.with_name("part19_proxyspex").glob("*.md"), HERE / "native_freeze.json",
                 ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "scripts/evidence_bundles.py",
                 ROOT / "scripts/generate_extension_verification_assets.py"]:
        relative = path.relative_to(ROOT).as_posix()
        require(relative not in inputs or publish.bundles.sha256(inputs[relative]) == publish.bundles.sha256(path),
                f"Supplied input substitutes another source: {relative}")
        inputs[relative] = path
    for name, record in context["native_receipt"]["source_inputs"].items():
        path = Path(name).resolve()
        if path.is_relative_to(ROOT):
            relative = path.relative_to(ROOT).as_posix()
            require(relative not in inputs or publish.bundles.sha256(inputs[relative]) == publish.bundles.sha256(path),
                    f"Supplied input substitutes another native source: {relative}")
            inputs[relative] = path
    receipt_paths = {"learner_execution.json": Path(learner_execution), "fresh_reproduction.json": Path(fresh_run) / "reproduction.json",
                     "native_execution.json": Path(native_run) / "execution.json", "fresh_native_execution.json": Path(fresh_run) / "execution.json",
                     "native_comparison_summary.json": Path(fresh_run) / "comparison_process/comparison_summary.json"}
    for name, path in receipt_paths.items():
        inputs[prefix + "expected_outputs/" + name] = path
    inputs[publish.FIXTURE] = Path(fixture)
    inputs[publish.FIXTURE + ".data-contract.json"] = Path(str(fixture) + ".data-contract.json")
    # Check every upstream file before creating an output directory.
    _manifest(inputs)
    output.mkdir(parents=True)
    publish.write_json(output / "native_summary.json", summary)
    (output / "artifacts.lock.yml").write_text(yaml.safe_dump(lock, sort_keys=False))
    publish.write_json(output / "verification_report.schema.json", schema)
    inputs[prefix + "expected_outputs/native_summary.json"] = output / "native_summary.json"
    inputs[prefix + "artifacts.lock.yml"] = output / "artifacts.lock.yml"
    inputs[prefix + "verification_report.schema.json"] = output / "verification_report.schema.json"
    values["native_summary_sha256"] = publish.bundles.sha256(output / "native_summary.json")
    values["comparison_summary_sha256"] = publish.bundles.sha256(receipt_paths["native_comparison_summary.json"])
    report = {
        "notebook_id": IDS[section], "date_run": context["fresh_receipt"].get("finished_at_utc", context["fresh_receipt"].get("started_at_utc")),
        "git_commit": context["native_receipt"].get("git_commit", "not_recorded_in_native_receipt"),
        "report_inputs": _manifest(inputs), "gt_tier": lock["gt_tier"], "evidence_level": lock["evidence_level"],
        "claim_scope": claim, "gpu_name": "not_used_cpu_execution", "peak_vram_gb": 0.0,
        "wall_clock_seconds": context["native_receipt"]["elapsed_seconds"] + context["fresh_receipt"]["elapsed_seconds"],
        "models": lock["models"], "datasets": lock["datasets"], "tests_passed": True, "accepted": True,
        "metrics": {"gpu_test": values, "gpu_evidence": {"category": "cpu_or_budget_metric", "uses_cuda": False,
                    "placeholder_only": False, "section_specific_metric_keys": sorted(values)},
                    "native_outcomes": {"reference_outcomes": summary["reference_outcomes"],
                                        "condition_outcomes": summary["condition_outcomes"], "prompt_outcomes": summary["prompt_outcomes"]},
                    "assembly": {"created_at_utc": datetime.now(timezone.utc).isoformat(), "new_model_queries": 0, "new_estimator_fits": 0,
                                 "known_failures_semantics": "Report-contract failures only. Observed reference failures remain in native_outcomes.",
                                 "native_execution": context["native"]["native_execution"], "fresh_execution": context["native"]["fresh_execution"]}},
        "baselines": {"declared_controls": controls, "expected_metrics": lock["expected_metrics"],
                      "measured_location": "metrics.native_outcomes.condition_outcomes[*].metrics.constant/additive"},
        "negative_controls": {"declared_controls": [controls[0]], "measured_location": "All condition='shuffled' outcome rows; no required failure threshold"},
        "ood_tests": {"scope": "Fixed subject-only counterfactuals, not broad out-of-distribution generalization",
                      "measured_location": "metrics.native_outcomes.prompt_outcomes[*].counterfactual"},
        "known_failures": [], "safety_notes": lock["safety_notes"],
    }
    require(isinstance(report["date_run"], str) and report["date_run"], "Native measurement timestamp missing")
    require(_finite(report["wall_clock_seconds"]) and report["wall_clock_seconds"] > 0, "Native measurement time missing")
    require(not receipt_blockers(section, summary, context["learner"], publish.read_json(Path(fresh_run) / "reproduction.json"), report), "Generated receipt/report contract failed")
    import jsonschema
    jsonschema.validate(report, schema)
    publish.write_json(output / "verification_report.json", report)
    artifacts = [{"role": role, **publish.physical_record(output / name)} for role, name in (
        ("verification_report", "verification_report.json"), ("native_summary", "native_summary.json"),
        ("artifacts_lock", "artifacts.lock.yml"), ("verification_schema", "verification_report.schema.json"))]
    publish.write_json(output / "report_artifacts.json", {"section": section, "status": "EVIDENCE_REPORT_ASSEMBLED_PENDING_PUBLICATION_REVIEW",
                       "artifacts": artifacts, "physical_report_inputs": {key: publish.physical_record(path) for key, path in inputs.items()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=IDS, required=True)
    for name in ("native-run", "fresh-run", "learner-execution", "fixture", "artifact-inputs", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    build_report(args.section, args.native_run, args.fresh_run, args.learner_execution, args.fixture,
                 publish.read_json(args.artifact_inputs), args.output)


if __name__ == "__main__":
    main()
