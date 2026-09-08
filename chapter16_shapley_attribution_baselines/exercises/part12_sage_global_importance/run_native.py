"""Fresh CPU fit → explanation ranking freeze → audit → independent parity."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import torch as t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance import (
    sage_data,
    sage_methods,
    sage_study,
    sage_reference,
)


def source_paths():
    return [
        HERE / name
        for name in [
            "sage_data.py",
            "sage_methods.py",
            "sage_methods_tests.py",
            "sage_study.py",
            "sage_reference.py",
            "author_reference.json",
            "protocol.md",
            "protocol_freeze.json",
            "data_manifest.json",
            "run_native.py",
        ]
    ]


def validate_freeze():
    freeze = json.loads((HERE / "protocol_freeze.json").read_text())
    for name, checksum in freeze["sha256"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Frozen source changed: {name}")


def summarize(prepared, audit, diagnostic, reference):
    model_mse = [
        float((r["predictions"][:, -1] - audit["y"]).square().mean()) for r in audit["models"]
    ]
    return {
        "training_runs": 7,
        "neural_training_runs": 6,
        "optimizer_updates": 4800,
        "train_rows": len(prepared["train_y"]),
        "explanation_rows": len(prepared["explain_y"]),
        "audit_rows": len(audit["y"]),
        "coalitions": 256,
        "background_rows": 64,
        "direct_sampled_paths": len(diagnostic["direct_scores"]),
        "repeated_sampling_runs": len(diagnostic["repeated_scores"]),
        "audit_used_for_selection": False,
        "audit_mse": model_mse,
        "training_mean_audit_mse": float(audit["training_mean_mse"]),
        "explanation_sage": prepared["explanations"][0]["sage"].tolist(),
        "audit_sage": audit["models"][0]["sage"].tolist(),
        "outer_path_table_max_error": float(
            (diagnostic["direct_scores"] - diagnostic["table_scores"]).abs().max()
        ),
        "dummy_importance": float(diagnostic["dummy_sage"][-1]),
        "local_global_max_error": max(
            float((r["local_loss"].mean(0) - r["sage"]).abs().max())
            for r in prepared["explanations"]
        ),
        "reference": reference,
    }


def run(directory):
    validate_freeze()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()
    }
    for name in sources:
        path = directory / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    state = {
        "status": "RUNNING",
        "source_inputs": sources,
        "new_training": True,
        "cached_model_reuse": False,
    }
    start = time.perf_counter()
    try:
        train = sage_data.load_split("train")
        explain = sage_data.load_split("explain")
        prepared = sage_study.prepare_study(train, explain, directory / "study", sage_methods)
        boundary = {
            "phase": prepared["phase"],
            "audit_loaded": False,
            "prepared_sha256": hashlib.sha256(
                (directory / "study/prepared.pt").read_bytes()
            ).hexdigest(),
        }
        (directory / "before_audit.json").write_text(json.dumps(boundary, indent=2) + "\n")
        audit = sage_study.evaluate_study(prepared, sage_data.load_split("audit"), sage_methods)
        t.save(audit, directory / "study/evaluated.pt")
        diagnostic = sage_study.diagnostics(prepared, sage_methods)
        t.save(diagnostic, directory / "study/diagnostics.pt")
        play = sage_study.native_play(prepared, audit, sage_methods)
        t.save(play, directory / "study/play.pt")
        reference = sage_reference.checks(
            prepared, sage_methods, sage_study.make_predictor(prepared["models"][0])
        )
        summary = summarize(prepared, audit, diagnostic, reference)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        state.update(status="PASS", summary=summary)
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["elapsed_seconds"] = time.perf_counter() - start
        state["changed_sources"] = [
            n
            for n, h in sources.items()
            if hashlib.sha256((ROOT / n).read_bytes()).hexdigest() != h
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        (directory / "run.json").write_text(json.dumps(state, indent=2) + "\n")
    if state["status"] != "PASS":
        raise ValueError("Sources changed during native execution.")
    print(
        json.dumps(
            {
                "status": state["status"],
                "seconds": state["elapsed_seconds"],
                "summary": state["summary"],
            },
            indent=2,
        )
    )
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    run(parser.parse_args().output_directory.resolve())
