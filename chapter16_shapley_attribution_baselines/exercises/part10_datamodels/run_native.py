"""Execute the frozen real-data study; never accept cached model fits as training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import torch
import numpy as np
from sklearn.linear_model import Lasso

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chapter16_shapley_attribution_baselines.exercises.part10_datamodels import (
    datamodel_methods as methods,
)
from chapter16_shapley_attribution_baselines.exercises.part10_datamodels import (
    datamodel_study as study,
)
from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.datamodel_data import (
    load_data,
    data_manifest,
)
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_reference import (
    scipy_logistic_optimum,
)

HERE = Path(__file__).resolve().parent


def reference_checks(data: dict, prepared: dict, evaluated: dict | None) -> dict:
    checks = []
    # Six independently scheduled fits establish batch scheduling parity.
    for fraction in study.FRACTIONS:
        masks = prepared["designs"][fraction]["train"]
        theta = prepared["subset_fits"][fraction]["train"]["theta"]
        for row in (0, 1):
            separate = study.refit_masks(data["train"], masks[row : row + 1])["theta"][0]
            difference = float((theta[row] - separate).abs().max())
            if difference > 1e-8:
                raise ValueError("Batch versus separate classifier refits disagree.")
            checks.append(
                dict(kind="separate_refit", fraction=fraction, row=row, max_error=difference)
            )
    candidates = []
    for fraction in study.FRACTIONS:
        for row in (0, 1, 2):
            candidates.append(
                (
                    f"alpha={fraction}/train/{row}",
                    prepared["designs"][fraction]["train"][row],
                    prepared["subset_fits"][fraction]["train"]["theta"][row],
                )
            )
    # The tenth independent optimum checks another training model in a
    # preparation-only run, or the full-pool base in a complete run.
    if evaluated is None:
        candidates.append(
            (
                "alpha=0.5/validation/0",
                prepared["designs"][0.5]["validation"][0],
                prepared["subset_fits"][0.5]["validation"]["theta"][0],
            )
        )
    else:
        candidates.append(
            ("full_pool", torch.ones(128, dtype=torch.bool), evaluated["base_fit"]["theta"][0])
        )
    for name, mask, theta in candidates:
        expected, residual = scipy_logistic_optimum(
            data["train"]["x"], data["train"]["y"], mask.double(), 0.02
        )
        difference = float((theta - expected).abs().max())
        if difference > 1e-8:
            raise ValueError("The classifier disagrees with the independent SciPy optimum.")
        checks.append(
            dict(kind="scipy", model=name, max_error=difference, gradient_residual=residual)
        )
    # The four gallery queries are fixed by source order, not by solver agreement.
    gallery = (
        torch.cat([torch.nonzero(data["query"]["y"] == label).flatten()[:2] for label in (0, 1)])
        .sort()
        .values.tolist()
    )
    for fraction in study.FRACTIONS:
        x = prepared["designs"][fraction]["train"].double().numpy()
        y = prepared["margins"][fraction]["train"].numpy()
        fit = prepared["responses"][fraction]["lasso"]
        for q in gallery:
            index = fit["selected"][q]
            penalty = fit["penalty"][index, q].item()
            independent = Lasso(alpha=penalty, tol=1e-13, max_iter=100000).fit(x, y[:, q])
            prediction = x @ fit["selected_weights"][:, q].numpy() + fit["selected_bias"][q].item()
            expected = independent.predict(x)
            difference = float(np.abs(prediction - expected).max())
            objective = (
                np.mean((prediction - y[:, q]) ** 2) / 2
                + penalty * fit["selected_weights"][:, q].abs().sum().item()
            )
            expected_objective = (
                np.mean((expected - y[:, q]) ** 2) / 2 + penalty * np.abs(independent.coef_).sum()
            )
            if difference > 1e-5 or abs(objective - expected_objective) > 1e-7:
                raise ValueError(
                    "Native Lasso predictions/objective disagree with independent coordinate descent."
                )
            checks.append(
                dict(
                    kind="sklearn_lasso",
                    fraction=fraction,
                    query=q,
                    max_prediction_error=difference,
                    objective_error=abs(objective - expected_objective),
                )
            )
    return dict(
        checks=checks, separate_refits=6, independent_scipy_optima=10, independent_lasso_fits=12
    )


def source_paths() -> list[Path]:
    local = [
        HERE / name
        for name in (
            "protocol.md",
            "protocol_freeze.json",
            "data_manifest.json",
            "datamodel_methods.py",
            "datamodel_methods_tests.py",
            "datamodel_data.py",
            "datamodel_study.py",
            "run_native.py",
        )
    ]
    previous = HERE.parent / "part9_influence_and_tracin"
    return local + [
        previous / name
        for name in (
            "attribution_data.py",
            "attribution_methods.py",
            "attribution_reference.py",
        )
    ]


def run(output: Path, prepare_only: bool = False) -> dict:
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    paths = source_paths()
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in paths:
        destination = snapshots / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    (output / "source_inputs.json").write_text(json.dumps(sources, indent=2) + "\n")
    start = time.perf_counter()
    try:
        freeze = json.loads((HERE / "protocol_freeze.json").read_text())
        for name, digest in freeze["inputs"].items():
            if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Frozen scientific input changed: {name}")
        data = load_data()
        if data_manifest(data) != json.loads((HERE / "data_manifest.json").read_text()):
            raise ValueError("Fresh source data does not match the frozen manifest.")
        designs = study.make_designs(methods)
        print(
            "Starting 3456 independent train/validation classifier fits and regression paths.",
            flush=True,
        )
        prepared = study.prepare_study(
            data["train"], data["query"], designs, methods, output / "native"
        )
        prepared_digest = hashlib.sha256((output / "native/prepared.pt").read_bytes()).hexdigest()
        boundary = dict(
            phase=prepared["phase"],
            prepared_sha256=prepared_digest,
            test_subset_responses_seen=False,
            counterfactual_responses_seen=False,
            elapsed_seconds=time.perf_counter() - start,
        )
        (output / "selection_boundary.json").write_text(json.dumps(boundary, indent=2) + "\n")
        print(
            "Every path and all 6528 deletion masks are fixed; evaluation responses are still unseen.",
            flush=True,
        )
        evaluated = None
        if not prepare_only:
            evaluated = study.evaluate_study(
                prepared, data["train"], data["query"], methods, output / "native"
            )
            play = study.deletion_play(prepared, data["train"], data["query"], methods)
            torch.save(play, output / "native/play.pt")
        references = reference_checks(data, prepared, evaluated)
        (output / "reference_checks.json").write_text(json.dumps(references, indent=2) + "\n")
        unchanged = all(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
            for name, digest in sources.items()
        )
        if not unchanged:
            raise ValueError("A source input changed while the experiment was running.")
        report = dict(
            status="PASS",
            stage="PREPARATION_ONLY" if prepare_only else "COMPLETE_NATIVE",
            elapsed_seconds=time.perf_counter() - start,
            source_inputs_unchanged=unchanged,
            source_input_count=len(sources),
            classifier_refits=3456 if prepare_only else 10753,
            separate_refits=6,
            independent_scipy_optima=10,
            independent_lasso_fits=12,
            play_refits=0 if prepare_only else 18,
            test_subset_responses_evaluated=not prepare_only,
            selection_used_test_responses=False,
        )
    except Exception as error:
        report = dict(
            status="FAIL",
            elapsed_seconds=time.perf_counter() - start,
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        (output / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (output / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.prepare_only)
