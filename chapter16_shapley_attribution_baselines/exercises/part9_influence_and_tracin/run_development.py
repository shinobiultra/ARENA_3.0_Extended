"""Fit the predeclared digit study while keeping final queries unevaluated."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin import attribution_methods as m
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_data import load_study_data, data_manifest
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_reference import scipy_logistic_optimum
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_study import prepare_study, L2


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def execute(directory):
    directory.mkdir(parents=True, exist_ok=False)
    names = ["attribution_methods.py", "attribution_methods_tests.py", "attribution_data.py",
             "attribution_data_tests.py", "attribution_study.py", "attribution_reference.py",
             "run_development.py", "native_protocol.md", "data_manifest.json"]
    sources = {}
    for name in names:
        path = HERE / name
        relative = str(path.relative_to(ROOT))
        snapshot = directory / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        sources[relative] = digest(path)
    started = time.perf_counter()
    state = {"status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
             "source_inputs": sources, "final_evaluated": False, "device": "cpu", "dtype": "float64",
             "progress": [], "conditions": {}}

    def save():
        state["elapsed_seconds"] = time.perf_counter() - started
        (directory / "development.json").write_text(json.dumps(state, indent=2) + "\n")

    def progress(message):
        state["progress"].append({"seconds": time.perf_counter() - started, "message": message})
        save()
        print(message, flush=True)

    save()
    try:
        torch.set_num_threads(4)
        data = load_study_data()
        manifest = data_manifest(data)
        if manifest != json.loads((HERE / "data_manifest.json").read_text()):
            raise ValueError("The input data differ from the prospective manifest.")
        (directory / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        fitted, measured = prepare_study(data, m, directory, progress)
        train = data["train"]
        for name, fits in fitted.items():
            baseline = measured[name]["baseline"]
            checks = []
            n = len(train["y"])
            cases = [("base", torch.ones(n, dtype=torch.float64), fits["base"]["theta"][0])]
            for index in (0, 1, 2):
                weights = torch.ones(n, dtype=torch.float64)
                weights[index] = 0
                cases.append((f"loo_{index}", weights, fits["loo"]["theta"][index]))
            weights = torch.ones(n, dtype=torch.float64)
            weights[data["upweight_probe_indices"][0]] = 0.5
            cases.append(("perturbation_first_minus_half", weights, fits["perturbations"]["theta"][1]))
            for label, weights, actual in cases:
                reference, residual = scipy_logistic_optimum(train["x"], fits["train_y"], weights, L2)
                error = float((actual - reference).abs().max())
                if error > 1e-8:
                    raise ValueError(f"Independent refit differs in {name}/{label}: {error}.")
                checks.append({"case": label, "max_parameter_error": error, "reference_gradient_max": residual})
            max_gradient = max(float(fits[key]["gradient_norms"][-1].max()) for key in ("base", "loo", "perturbations", "deletions"))
            state["conditions"][name] = {
                "ranking_correct": int(baseline["correct"].sum()), "ranking_count": baseline["correct"].numel(),
                "ranking_mean_loss": float(baseline["losses"].mean()), "refits": 465, "sgd_paths": len(fits["paths"]),
                "sgd_updates": sum(len(trace["used_ids"]) for trace in fits["paths"]),
                "rankers": len(fits["rankers"]), "deletion_policies": len(fits["policies"]),
                "max_final_refit_gradient": max_gradient, "independent_reference_checks": checks,
                "max_individual_refit_error": float(fits["individual_refit_errors"].max()),
                "max_ghost_product_error": max(path["checkpoints"]["ghost_max_error"] for path in measured[name]["paths"]),
                "all_queries_used_for_selection": "ranking_only",
            }
            progress(f"{name}: independent SciPy reference and separate-fit checks passed")
        state["artifacts"] = {
            str(path.relative_to(directory)): {"sha256": digest(path), "size_bytes": path.stat().st_size}
            for path in sorted(directory.rglob("*.pt"))
        }
        state["status"] = "PASS"
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["changed_sources"] = [name for name, checksum in sources.items() if digest(ROOT / name) != checksum]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        save()
    if state["status"] != "PASS":
        raise ValueError("A source changed during the native development run.")
    print(json.dumps({k: v for k, v in state.items() if k not in ("source_inputs", "artifacts", "progress")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    execute(parser.parse_args().output_directory.resolve())
