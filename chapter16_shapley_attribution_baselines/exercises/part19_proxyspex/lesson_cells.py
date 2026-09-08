"""Visible experiment cells: all fits and transforms call the student's functions."""

import inspect


def run_native_bridge(output_dir, *, proxy_namespace, spex_namespace):
    """Execute the fixed common study with completed notebook methods, when ready."""
    from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_study
    required = ("character_matrix", "predict_spectrum", "path_indicator_fourier", "tree_to_fourier",
                "fit_boosted_trees", "predict_boosted_trees", "ensemble_to_fourier",
                "select_energy_support", "grouped_mask_folds", "ridge_path", "refine_coefficients",
                "fit_proxy", "stage_errors")
    for name in required:
        function = getattr(proxy_namespace, name)
        if not callable(function) or proxy_namespace.fit_proxy.__globals__.get(name) is not function:
            raise ValueError(f"{name} must come from the same completed learner namespace as fit_proxy")
    return native_study.run_study(output_dir, spex_methods=spex_namespace, proxy_methods=proxy_namespace)


CELLS = {
"setup": '''from __future__ import annotations
from itertools import product
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from IPython.display import display, Markdown
%matplotlib inline

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import (
    proxy_methods_tests as tests, proxy_learner_tests as learner_tests, proxy_plots as plots,
)
# Only validation helpers are supplied; the algorithm functions are exercises below.
from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex.proxy_methods import (
    binary_masks, training_arrays, canonical_spectrum,
)

PROXY_CANDIDATES = tuple(
    {"max_depth": depth, "n_estimators": count, "learning_rate": .1, "min_samples_leaf": 2}
    for depth, count in product((1, 3), (128, 512))
)
RIDGE_ALPHAS = (1e-6, 1e-3, .1, 1., 10.)
thread_limit = threadpool_limits(limits=4)
OUTPUT_DIR = Path(os.environ.get("ARENA_PROXYSPEX_OUTPUT_DIRECTORY", "./proxyspex_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})

def show_figure(figure, name):
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=140, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    display(figure)
    plt.close(figure)
''',
"cold_open": '''small_masks = np.asarray(list(product((0, 1), repeat=4)), dtype=np.uint8)
small_values = .3 + small_masks[:, 0] + 2. * small_masks[:, 0] * small_masks[:, 1] * small_masks[:, 2]
show_figure(plots.plot_cold_open(small_masks, small_values), "01_exact_cold_open")
''',
"boosting_figure": '''# Training and audit masks are deliberately disjoint subsets of an eight-bit cube.
boost_cube = np.asarray(list(product((0, 1), repeat=8)), dtype=np.uint8)
permutation = np.random.default_rng(25).permutation(len(boost_cube))
boost_train, boost_audit = boost_cube[permutation[:128]], boost_cube[permutation[128:]]
def hierarchical_score(x):
    return .3 + x[:, 0] + 2. * x[:, 0] * x[:, 1] * x[:, 2]
boost_train_y, boost_audit_y = hierarchical_score(boost_train), hierarchical_score(boost_audit)
boost_labels = {"depth 3": boost_train_y, "depth 1": boost_train_y,
                "shuffled responses": np.random.default_rng(84).permutation(boost_train_y)}
boost_models, boost_audit_mse = {}, {}
for label, responses in boost_labels.items():
    model = fit_boosted_trees(boost_train, responses, max_depth=1 if label == "depth 1" else 3,
                             n_estimators=40, learning_rate=.1, min_samples_leaf=1, seed=5)
    boost_models[label] = model
    prediction = np.full(len(boost_audit), model["initial_value"])
    losses = [np.mean((prediction - boost_audit_y)**2)]
    for tree in model["trees"]:
        prediction += model["learning_rate"] * tree.predict(boost_audit)
        losses.append(np.mean((prediction - boost_audit_y)**2))
    boost_audit_mse[label] = np.asarray(losses)
show_figure(plots.plot_boosting(boost_models, boost_audit_mse,
                               np.mean((boost_train_y.mean() - boost_audit_y)**2)), "02_residual_boosting")
''',
"path_figure": '''path_masks = np.asarray(list(product((0, 1), repeat=3)), dtype=np.uint8)
worked_path = [(0, 0), (1, 1)]
path_spectrum = path_indicator_fourier(worked_path)
direct_indicator = np.asarray([all(row[i] == bit for i, bit in worked_path) for row in path_masks], float)
contributions = character_matrix(path_masks, path_spectrum) * np.asarray(list(path_spectrum.values()))
path_prediction = predict_spectrum(path_masks, path_spectrum)
show_figure(plots.plot_path(path_masks, path_spectrum, contributions, direct_indicator, path_prediction,
                            title="(x₀ = 0) AND (x₁ = 1): four terms make one indicator"), "03_worked_path")
''',
"tree_figure": '''# One CART primitive fits the complete small analytic function; your code extracts it.
worked_tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=1, random_state=2).fit(small_masks, small_values)
worked_tree_spectrum = tree_to_fourier(worked_tree)
worked_tree_prediction = predict_spectrum(small_masks, worked_tree_spectrum)
worked_tree_error = np.max(np.abs(worked_tree_prediction - worked_tree.predict(small_masks)))
show_figure(plots.plot_tree_spectrum(worked_tree, worked_tree_spectrum, worked_tree_error), "04_tree_to_spectrum")
''',
"parity_figure": '''parity_masks = np.asarray(list(product((0, 1), repeat=10)), dtype=np.uint8)
parity_values = -1.25 * np.prod(1. - 2. * parity_masks[:, [1, 3, 5]], axis=1)
# This intentionally small analytic control has enough depth but unhelpful greedy splits.
parity_proxy = fit_boosted_trees(parity_masks, parity_values, max_depth=3, n_estimators=3,
                                min_samples_leaf=1, seed=1)
parity_proxy_values = predict_boosted_trees(parity_proxy, parity_masks)
parity_extracted_values = predict_spectrum(parity_masks, ensemble_to_fourier(parity_proxy))

# Independently construct an exact parity tree by specifying its branch decisions.
def known_parity_tree(remaining, bits):
    if not remaining:
        return -1.25 * (-1) ** sum(bits)
    return (remaining[0], .5, known_parity_tree(remaining[1:], bits + [0]),
            known_parity_tree(remaining[1:], bits + [1]))
known_tree = tests.explicit_tree(known_parity_tree([2, 0, 1], []), n_features=4)
known_spectrum = tree_to_fourier(known_tree)
known_values = -1.25 * np.prod(1. - 2. * small_masks[:, [0, 1, 2]], axis=1)
known_prediction = predict_spectrum(small_masks, known_spectrum)
show_figure(plots.plot_parity(parity_values, parity_proxy_values, parity_extracted_values,
                             known_values, known_prediction, known_spectrum), "05_parity_fit_and_extraction")
''',
"author_diagnostic": '''# This is the author's unchanged selection algebra only, translated to index-tuple keys.
# It is not a LightGBM fit or an execution of the whole author estimator.
def author_strict_energy_selection(spectrum):
    keys = list(spectrum)
    coefficients = np.asarray(list(spectrum.values()), dtype=float)
    if () in keys:
        coefficients[keys.index(())] = 0
    squared = coefficients**2
    sorted_squared = np.sort(squared)[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        boundary_index = np.argmin(np.cumsum(sorted_squared / np.sum(squared)) < .95)
    threshold = np.sqrt(sorted_squared[boundary_index])
    return {support: value for support, value in spectrum.items() if abs(value) > threshold}

tests.test_author_strict_energy_failure(author_strict_energy_selection)
print("Untouched author parity execution, preserved in the source report:")
print("ValueError: Dot product shape mismatch, (1024, 12) vs (0,)")
print("The cell above reproduces selection behavior only; it does not rerun that saved failure.")
''',
"energy_figure": '''energy_masks = np.asarray(list(product((0, 1), repeat=4)), dtype=np.uint8)
energy_games = {"additive with a small term": {(): .3, (0,): 1., (1,): -.7, (2,): .2},
                "single signed parity": {(0, 1, 2): -1.25}}
energy_runs = {}
for name, spectrum in energy_games.items():
    # Include both sides of the exact prefix boundaries, including just above zero.
    squared = sorted((value**2 for support,value in spectrum.items() if support), reverse=True)
    boundaries = np.cumsum(squared) / sum(squared)
    fractions = np.unique(np.clip(np.r_[np.linspace(0, 1, 101), .95, np.nextafter(0., 1.),
                                         boundaries, np.nextafter(boundaries, 1.)], 0., 1.))
    true_values = predict_spectrum(energy_masks, spectrum)
    retained, errors = [], []
    for fraction in fractions:
        selection = select_energy_support(spectrum, fraction)
        reconstruction = predict_spectrum(energy_masks, selection["spectrum"])
        retained.append(selection["retained_energy_fraction"])
        errors.append(np.sqrt(np.mean((reconstruction - true_values)**2)))
    author_selected = author_strict_energy_selection(spectrum)
    total_energy = sum(value**2 for support,value in spectrum.items() if support)
    author_energy = sum(value**2 for support,value in author_selected.items() if support)
    energy_runs[name] = {"fractions": fractions, "retained": np.asarray(retained), "rmse": np.asarray(errors),
        "author_retained": author_energy / total_energy,
        "author_rmse": np.sqrt(np.mean((predict_spectrum(energy_masks, author_selected)-true_values)**2)),
        "constant_rmse": np.sqrt(np.mean((spectrum.get((), 0.)-true_values)**2))}
show_figure(plots.plot_energy(energy_runs), "06_energy_boundary")
''',
"student_namespace": '''# Every algorithm entry below is your completed notebook function.
methods = SimpleNamespace(
    binary_masks=binary_masks, training_arrays=training_arrays, canonical_spectrum=canonical_spectrum,
    character_matrix=character_matrix, predict_spectrum=predict_spectrum,
    path_indicator_fourier=path_indicator_fourier, tree_to_fourier=tree_to_fourier,
    fit_boosted_trees=fit_boosted_trees, predict_boosted_trees=predict_boosted_trees,
    ensemble_to_fourier=ensemble_to_fourier, select_energy_support=select_energy_support,
    grouped_mask_folds=grouped_mask_folds, ridge_path=ridge_path,
    refine_coefficients=refine_coefficients, fit_proxy=fit_proxy, stage_errors=stage_errors,
    PROXY_CANDIDATES=PROXY_CANDIDATES, RIDGE_ALPHAS=RIDGE_ALPHAS,
)
''',
"analytic_study": '''analytic_train = np.random.default_rng(100).integers(0, 2, size=(456, 28), dtype=np.uint8)
analytic_audit = np.random.default_rng(991).integers(0, 2, size=(2048, 28), dtype=np.uint8)
assert not ({tuple(row) for row in analytic_train} & {tuple(row) for row in analytic_audit})

def analytic_scores(x):
    z = 1. - 2. * x.astype(float)
    return {
        "additive": .3 + z[:, 0] - .7*z[:, 3] + .2*z[:, 7],
        "signed_parity": -1.25*z[:, 2]*z[:, 4]*z[:, 9],
        "sparse_polynomial": .3 + x[:, 0] - .7*x[:, 3] + 1.2*x[:, 1]*x[:, 7]
                              - 2.4*x[:, 2]*x[:, 4]*x[:, 9],
    }

analytic_runs = {}
for name, training_values in analytic_scores(analytic_train).items():
    # The fitting call has no audit inputs. Only these supplied training rows select the model.
    fitted = fit_proxy(analytic_train, training_values, fold_seed=0)
    shuffled = fit_proxy(analytic_train, np.random.default_rng(840).permutation(training_values), fold_seed=0)
    audit_values = analytic_scores(analytic_audit)[name]
    predictions = {"proxy": predict_boosted_trees(fitted["model"], analytic_audit)}
    for label, key in (("extracted", "full_spectrum"), ("truncated", "truncated_spectrum"), ("refined", "refined_spectrum")):
        predictions[label] = predict_spectrum(analytic_audit, fitted[key])
    # Same training rows, independent simple baselines; neither receives audit values.
    train_additive = np.column_stack([np.ones(len(analytic_train)), analytic_train])
    audit_additive = np.column_stack([np.ones(len(analytic_audit)), analytic_audit])
    additive_coefficients = np.linalg.lstsq(train_additive, training_values, rcond=None)[0]
    baselines = {"constant": np.full(len(analytic_audit), training_values.mean()),
                 "additive": audit_additive @ additive_coefficients}
    analytic_runs[name] = {"fit": fitted, "audit_values": audit_values, "predictions": predictions,
        "baseline_predictions": baselines,
        "shuffled_predictions": predict_spectrum(analytic_audit, shuffled["refined_spectrum"]),
        "stage_errors": stage_errors(audit_values, predictions["proxy"], predictions["extracted"],
                                      predictions["truncated"], predictions["refined"])}
    print(f"Fitted {name}: {fitted['accounting']['ensemble_fits']} ensembles, "
          f"{fitted['accounting']['residual_tree_fits']} individual trees; zero language-model calls.")
''',
"analytic_signature": '''show_figure(plots.plot_stage_comparison(analytic_runs), "07_analytic_stage_comparison")
''',
"cv_figure": '''show_figure(plots.plot_cv(analytic_runs["sparse_polynomial"]["fit"]), "08_training_only_selection")
''',
"analytic_play": '''PLAY_PATH = [(0, 0), (1, 1), (0, 0)]  # try adding (0,1), or changing a required bit
PLAY_ENERGY = .95                       # try .5, .9, .95 or 1.0
play_spectrum = path_indicator_fourier(PLAY_PATH)
play_selection = select_energy_support(play_spectrum, PLAY_ENERGY)
play_truth = np.asarray([all(row[i] == bit for i, bit in PLAY_PATH) for row in path_masks], float)
play_contributions = character_matrix(path_masks, play_spectrum) * np.asarray(list(play_spectrum.values()))
play_full = predict_spectrum(path_masks, play_spectrum)
play_selected = predict_spectrum(path_masks, play_selection["spectrum"])
show_figure(plots.plot_path(path_masks, play_spectrum, play_contributions, play_truth, play_full,
                            selected_values=play_selected,
                            title=f"Path play: {PLAY_PATH}; requested energy {PLAY_ENERGY:.0%}"), "09_path_energy_play")
print(f"Actual retained nonconstant energy: {play_selection['retained_energy_fraction']:.1%}; extra model queries: 0")
''',
"native_bridge": inspect.getsource(run_native_bridge),
"native_study": '''# UNEXECUTED_NATIVE_CELL: the common GPT-2 study has not run in this development draft.
payload = None
NATIVE_EXECUTED = False
display(Markdown("**UNEXECUTED:** the common GPT-2 word-mask experiment has no results in this draft."))
# Once the shared study is ready, the complete learner run calls the bridge with
# the completed 16.18 functions and this notebook's `methods` namespace:
# payload = run_native_bridge(OUTPUT_DIR / "native", proxy_namespace=methods, spex_namespace=spex_methods)
# NATIVE_EXECUTED = True
''',
"native_play": '''PROMPT_INDEX = 0
QUERY_BUDGET = 456
SEED = 0
if payload is None or not NATIVE_EXECUTED:
    display(Markdown("**UNEXECUTED:** run the common model experiment before inspecting a prompt. No extra queries made."))
else:
    show_figure(plots.plot_native_stages(payload, prompt_index=PROMPT_INDEX, seed=SEED, budget=QUERY_BUDGET),
                "10_native_error_play")
    show_figure(plots.plot_native_spectrum(payload, prompt_index=PROMPT_INDEX, seed=SEED, budget=QUERY_BUDGET),
                "11_native_spectrum_play")
    print("Exploratory selection from the fixed completed study; extra model queries: 0.")
''',
}


TEST_CALLS = {
    "character_matrix": "tests.test_character_matrix(character_matrix)",
    "predict_spectrum": "learner_tests.test_predict_spectrum(predict_spectrum)",
    "fit_boosted_trees": "tests.test_fit_boosted_trees(fit_boosted_trees)",
    "predict_boosted_trees": "learner_tests.test_predict_boosted_trees(predict_boosted_trees)",
    "path_indicator_fourier": "tests.test_path_indicator_fourier(path_indicator_fourier)",
    "tree_to_fourier": "tests.test_tree_to_fourier(tree_to_fourier)\ntests.test_known_signed_parity_tree(tree_to_fourier)",
    "ensemble_to_fourier": "tests.test_ensemble_to_fourier(ensemble_to_fourier)",
    "select_energy_support": "tests.test_select_energy_support(select_energy_support)",
    "grouped_mask_folds": "tests.test_grouped_mask_folds(grouped_mask_folds)",
    "ridge_path": "tests.test_ridge_path(ridge_path)",
    "refine_coefficients": "tests.test_refine_coefficients(refine_coefficients)",
    "fit_proxy": "learner_tests.test_fit_proxy(fit_proxy)",
    "stage_errors": "tests.test_stage_errors(stage_errors)",
}
