"""Short visible calling cells for the lesson; estimator bodies live in the exercises."""

CELLS = {
'setup': '''from __future__ import annotations
import itertools
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as t
from torch import Tensor
from IPython.display import display

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators import (
    estimator_methods_tests as tests, estimator_study as study, estimator_plots as plots,
)
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.estimator_methods import (
    _players, _masks, _values, _queried_inputs, interaction_masks, boundary_masks,
)
t.set_num_threads(4)
OUTPUT_DIR = Path(os.environ.get("ARENA_ESTIMATOR_OUTPUT_DIRECTORY", "./estimator_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def show_figure(figure, name):
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    display(figure)
    plt.close(figure)
''',
'cold_open': '''# This explicit sixteen-row polynomial is our known exact teaching game.
game_rows = []
for mask in range(16):
    a, b, c, d = [float(bool(mask & (1 << i))) for i in range(4)]
    value = .2*a + .2*b + 5*c + 4*d + 3*a*b + 2*a*b*c
    game_rows.append({"mask": f"{mask:04b}", "A": a, "B": b, "C": c, "D": d, "value": value})
display(pd.DataFrame(game_rows).set_index("mask"))
print("Query {A,B}:", game_rows[3]["value"])
''',
'shared_query_figure': '''games = plots.analytic_games(coalition_design)
figure = plots.plot_shared_query(games, sii_query_weights, query_mask=3)
show_figure(figure, "01_shared_query")
''',
'query_play': '''QUERY_MASK = 3           # integer mask 0..15
GAME_NAME = "mixed"      # "mixed", "additive", "constant", or "pure_four"
figure = plots.plot_shared_query(games, sii_query_weights, query_mask=QUERY_MASK, game_name=GAME_NAME)
show_figure(figure, "02_query_play")
''',
'sampling_figure': '''figure = plots.plot_sampling(sample_coalitions)
show_figure(figure, "03_sampling_and_cost")
''',
'boundary_figure': '''figure = plots.plot_boundary_limit(kernelshap_iq)
show_figure(figure, "04_boundary_limit")
''',
'methods_namespace': '''# Pass your actual completed functions into the fixed native study.
methods = SimpleNamespace(
    coalition_design=coalition_design,
    sii_query_weights=sii_query_weights,
    sample_coalitions=sample_coalitions,
    shapiq_estimate=shapiq_estimate,
    query_sampled_game=query_sampled_game,
    constrained_wls=constrained_wls,
    pair_residual_design=pair_residual_design,
    kernel_weights=kernel_weights,
    kernelshap_iq=kernelshap_iq,
)
''',
'native_study': '''# Refit the known models, then run every fixed query plan with your estimators.
# A new output directory preserves each complete run. No fitted cache is supplied.
payload = study.run_study(methods, OUTPUT_DIR / "native")
model_games = payload["games"]
print("All 21 known model-state checks passed.")
print("Complete reference:", model_games["exact_reference_coalition_queries"], "coalition queries;")
print(model_games["exact_reference_donor_predictions"], "underlying donor predictions.")
source_zero = int(t.nonzero(model_games["source_ids"] == 0).item())
display(pd.DataFrame({
    "feature": model_games["feature_names"],
    "source 0 recipe": model_games["queries"][source_zero].numpy(),
    "first intact donor": model_games["donors"][0].numpy(),
}).set_index("feature"))
''',
'model_controls': '''figure = plots.plot_model_controls(payload)
show_figure(figure, "05_model_controls")
''',
'convergence': '''figure = plots.plot_convergence(payload)
show_figure(figure, "06_query_budget_errors")
''',
'signed_comparison': '''figure = plots.plot_signed_comparison(payload)
show_figure(figure, "07_signed_pair_comparison")
''',
'variance_comparison': '''figure = plots.plot_variance(payload)
show_figure(figure, "08_sampling_variance")
''',
'query_costs': '''figure = plots.plot_query_costs(payload)
show_figure(figure, "09_unique_queries_and_rank")
''',
'native_play': '''MODEL_KEY = "clean_0"
SOURCE_ID = 0
DRAWS = 128
SEED = 161750
PROPOSAL = "kernel"
play = plots.make_native_play(
    payload, methods, model_key=MODEL_KEY, source_id=SOURCE_ID,
    draws=DRAWS, seed=SEED, proposal=PROPOSAL,
)
print("Unique queried masks:", play["receipt"]["queried_masks"].tolist())
print("Pair free rank:", play["kernelshap_iq"]["pair_stage"]["free_rank"], "/ 20")
if play["exploratory_audit_reuse"]:
    print("Exploratory audit reuse: this play does not change the fixed study.")
figure = plots.plot_native_play(play)
show_figure(figure, "10_native_query_play")
''',
}
