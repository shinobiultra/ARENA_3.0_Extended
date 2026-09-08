"""Visible calling cells and exercise tests for the unpublished 16.18 draft."""

EXERCISES = {
    "characters": {"functions": ("boolean_character", "predict_spectrum"),
                   "tests": "lesson_tests.test_characters(boolean_character, predict_spectrum)"},
    "walsh": {"functions": ("normalized_fwht",),
              "tests": "tests.test_boolean_basis(character=boolean_character, transform=normalized_fwht)"},
    "shifts": {"functions": ("shifted_subsample_queries", "bin_index"),
               "tests": "lesson_tests.test_shifts(shifted_subsample_queries, bin_index)"},
    "construction": {"functions": ("bch_generator_polynomial", "make_bch_code", "bch_encode"),
                     "tests": "lesson_tests.test_code_construction(make_bch_code)\nlesson_tests.test_encoding(bch_encode, make_bch_code)"},
    "query_design": {"functions": ("make_query_design", "transform_query_values"),
                     "tests": "tests.test_query_order_and_shift_identity(make_design=make_query_design, transform_values=transform_query_values)"},
    "syndromes": {"functions": ("bch_syndromes", "berlekamp_massey"),
                  "tests": "lesson_tests.test_syndromes_and_locator(bch_syndromes, berlekamp_massey)"},
    "hard_decode": {"functions": ("chien_search", "decode_bch_parity"),
                    "tests": "tests.test_bch_guaranteed_radius(encode=bch_encode, decoder=decode_bch_parity)\ntests.test_bch_scaling_and_decoder_steps(bm=berlekamp_massey, chien=chien_search)"},
    "chase": {"functions": ("chase_patterns", "soft_decode_parity"),
              "tests": "lesson_tests.test_chase(chase_patterns, soft_decode_parity)"},
    "signed_bin": {"functions": ("signed_bin_check", "decode_bin"),
                   "tests": "lesson_tests.test_bin_decode(signed_bin_check, decode_bin)"},
    "peeling": {"functions": ("recover_spectrum",),
                "tests": "tests.test_small_cube_recovery(recover=recover_spectrum)\ntests.test_iterative_collisions_and_accounting(recover=recover_spectrum)\ntests.test_wrong_degree_dense_and_iteration_controls(recover=recover_spectrum)"},
    "attribution": {"functions": ("fourier_to_sii", "fourier_to_shapley"),
                    "tests": "tests.test_fourier_to_sii_and_shapley(convert=fourier_to_sii, shapley=fourier_to_shapley)"},
    "word_masking": {"functions": ("masked_word_inputs",),
                     "tests": "native_tests.test_word_groups_hide_all_subtokens_and_keep_absolute_positions(fn=masked_word_inputs)"},
}


CELLS = {
"setup": '''from __future__ import annotations
import os
import sys
import heapq
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

chapter = "chapter16_shapley_attribution_baselines"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
from chapter16_shapley_attribution_baselines.exercises.part18_spex import (
    spex_methods_tests as tests, lesson_tests, spex_plots as plots, native_data_tests as native_tests,
)
from chapter16_shapley_attribution_baselines.exercises.part18_spex.native_data import N_FEATURES
from chapter16_shapley_attribution_baselines.exercises.part18_spex.spex_coding import BinaryField, binary_field
# Containers, validation and small-mask enumeration are supplied plumbing.
# Every transform, encoder, decoder and recovery operation is implemented below.
from chapter16_shapley_attribution_baselines.exercises.part18_spex.spex_methods import (
    BCHCode, BCHDecode, QueryDesign, RecoveryResult, Spectrum,
    _binary_array, binary_masks, mask_ids,
)

OUTPUT_DIR = Path(os.environ.get("ARENA_SPEX_OUTPUT_DIRECTORY", "./spex_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_NATIVE_STUDY = False  # Frozen analytic draft: no model is queried by default.
native_payload = None

def show_figure(figure, name):
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=140, bbox_inches="tight")
    display(figure)
    plt.close(figure)
''',
"cold_open": '''# Only three of 31 switches affect this exact teaching score.
active_masks = binary_masks(3)
parity_score = -1.25 * (-1.0) ** active_masks.sum(axis=1)
additive_control = .4 * (3 - 2 * active_masks.sum(axis=1).astype(float))
show_figure(plots.plot_cold_open(active_masks, parity_score, additive_control), "01_three_switch_question")
print("Full 31-switch cube:", 2**31, "masks. We will not enumerate it.")
''',
"walsh_figure": '''small_masks = binary_masks(3)
small_truth = {(): .3, (0,): .5, (1, 2): -.8}
small_values = predict_spectrum(small_masks, small_truth)
small_fourier = normalized_fwht(small_values)
small_inverse = normalized_fwht(small_fourier) * len(small_fourier)
show_figure(plots.plot_transform(small_masks, small_values, small_fourier, small_inverse), "02_walsh_signed_coordinates")
''',
"shift_example": '''# A tiny, inspectable collision: supports (0,) and (1,2) hash to the same bin.
M_demo = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
D_demo = np.vstack((np.zeros((1, 3), dtype=np.uint8), np.eye(3, dtype=np.uint8)))
demo_queries = shifted_subsample_queries(M_demo, D_demo)
demo_truth = {(0,): .8, (1, 2): -.5}
demo_values = predict_spectrum(demo_queries.reshape(-1, 3), demo_truth).reshape(4, 4)
demo_bins = normalized_fwht(demo_values)
print("Both supports reach bin:", bin_index(M_demo, (0,)), bin_index(M_demo, (1, 2)))
display(pd.DataFrame(demo_bins, index=["zero shift", "shift x0", "shift x1", "shift x2"], columns=["bin 00", "bin 01", "bin 10", "bin 11"]))
np.testing.assert_allclose(demo_bins[:, 2], [.3, -1.3, 1.3, 1.3])
''',
"code_example": '''code = make_bch_code(n=5, t=2)
message = np.array([0, 1, 0, 0, 1], dtype=np.uint8)
codeword = bch_encode(message, code)
print(f"Full BCH({code.nc},{code.kc}); remove {code.shortened} leading message positions.")
print("Message:", message, "parity:", codeword[5:])
display(pd.DataFrame(code.parity, columns=[f"x{i}" for i in range(5)], index=[f"parity {j}" for j in range(8)]))
''',
"design_receipt": '''design = make_query_design(n=31, t=3, b=3, C=3, seed=1)
print("Query array:", design.masks.shape)
print("Requested:", design.requested_count, "distinct:", design.unique_count)
assert design.requested_count == 3 * len(design.delays) * 2**design.b
assert design.masks.shape[1] == 31
print("The oracle will see binary rows; feature 0 is the first column.")
''',
"decode_example": '''received = np.concatenate((np.zeros(5, dtype=np.uint8), codeword[5:]))
syndrome = bch_syndromes(received, code)
locator = berlekamp_massey(syndrome, code.field)
decoded = decode_bch_parity(received[5:], code)
print("Syndromes:", syndrome)
print("Locator coefficients (ascending powers):", locator)
print("Recovered feature positions:", decoded.support)
print("Corrected shortened-word positions:", decoded.corrected_positions)
assert decoded.success and decoded.support == (1, 4)
''',
"confidence_play": '''# Try It Yourself: change the support, corruption position and confidence.
SUPPORT = (1, 3, 7)
BAD_PARITY_BIT = 0
REPORTED_CONFIDENCE = .01  # Small means uncertain; try 10.0 for a misranked error.
DEGREE_BUDGET = 3
decoder_methods = SimpleNamespace(
    make_bch_code=make_bch_code, boolean_character=boolean_character,
    decode_bch_parity=decode_bch_parity, soft_decode_parity=soft_decode_parity,
)
show_figure(plots.plot_code_rescue(decoder_methods, support=SUPPORT,
    bad_bit=BAD_PARITY_BIT, confidence=REPORTED_CONFIDENCE, t=DEGREE_BUDGET), "03_confidence_play")
print("This coding play makes zero oracle queries.")
''',
"methods_namespace": '''# These are the actual functions you just defined in this notebook.
methods = SimpleNamespace(
    boolean_character=boolean_character, normalized_fwht=normalized_fwht,
    predict_spectrum=predict_spectrum, shifted_subsample_queries=shifted_subsample_queries,
    bin_index=bin_index, bch_generator_polynomial=bch_generator_polynomial,
    make_bch_code=make_bch_code, bch_encode=bch_encode,
    make_query_design=make_query_design, transform_query_values=transform_query_values,
    bch_syndromes=bch_syndromes, berlekamp_massey=berlekamp_massey,
    chien_search=chien_search, decode_bch_parity=decode_bch_parity,
    chase_patterns=chase_patterns, soft_decode_parity=soft_decode_parity,
    signed_bin_check=signed_bin_check, decode_bin=decode_bin,
    recover_spectrum=recover_spectrum,
)
''',
"signature": '''# A signed polynomial has a independently expandable 13-term Fourier truth.
# This is an analytic oracle, not a language-model result.
truth = plots.analytic_games(n=31)["sparse"]
values = predict_spectrum(design.masks, truth)
recovery = recover_spectrum(design, values)
show_figure(plots.plot_recovery_trace(methods, design, recovery, truth), "04_collision_to_signed_spectrum")
print("Recovered", len(recovery.spectrum), "supports across", len(recovery.trace), "recorded sweeps.")
print("Stop reason:", recovery.stop_reason)
''',
"support_play": '''# Try It Yourself: edit signs/supports, hash seed, degree or bin budget.
PLAY_TRUTH = {(): .3, (1, 3, 7): -1.25, (0,): .6, (2, 4): -.4}
PLAY_N = 31
PLAY_T = 3
PLAY_B = 3
PLAY_SEED = 2
play_design = make_query_design(PLAY_N, PLAY_T, PLAY_B, seed=PLAY_SEED)
play_values = predict_spectrum(play_design.masks, PLAY_TRUTH)
play_recovery = recover_spectrum(play_design, play_values)
show_figure(plots.plot_recovery_trace(methods, play_design, play_recovery, PLAY_TRUTH), "05_support_and_budget_play")
print("This play requested", play_design.requested_count, "analytic masks;", play_design.unique_count, "were distinct.")
print("Additional native-model evaluations: 0.")
''',
"analytic_comparison": '''# Predeclared seeds and budgets: audit labels never choose an algorithm setting.
analytic_payload = plots.run_analytic_suite(methods, n=31, seeds=(0, 1, 2), budgets=(3, 4, 5))
plots.save_analytic_payload(analytic_payload, OUTPUT_DIR / "analytic_runs")
show_figure(plots.plot_budget_curves(analytic_payload), "06_sparse_dense_and_failed_controls")
show_figure(plots.plot_query_costs(analytic_payload), "07_requested_and_unique_cost")
display(pd.DataFrame([
    {"game": game, "truth terms": len(analytic_payload["truths"][game]),
     "worst audit NRMSE": max(r["nrmse"] for r in analytic_payload["records"] if r["game"] == game),
     "largest false-positive count": max(r["false_positives"] for r in analytic_payload["records"] if r["game"] == game)}
    for game in analytic_payload["truths"]
]))
''',
"attribution_figure": '''methods.fourier_to_sii = fourier_to_sii
methods.fourier_to_shapley = fourier_to_shapley
show_figure(plots.plot_attribution_conversion(methods), "08_fourier_is_not_shapley")
''',
"word_masking_example": '''# Toy subtoken IDs expose the intervention without loading or querying a model.
toy_ids = np.array([101, 102, 103, 104, 105])
toy_groups = np.array([0, 0, 1, 2, -1])  # First word has TWO subtokens; -1 is fixed suffix.
toy_masks = np.zeros((2, 28), dtype=np.uint8)
toy_masks[1] = 1
masked = masked_word_inputs(toy_ids, toy_groups, toy_masks, replacement_id=999)
display(pd.DataFrame({
    "word feature": toy_groups,
    "all hidden: token ID": masked["input_ids"][0],
    "all hidden: attention key": masked["attention_mask"][0],
    "all retained: token ID": masked["input_ids"][1],
    "position in either row": masked["position_ids"][0],
}))
methods.masked_word_inputs = masked_word_inputs
''',
"native_study": '''# This cell needs the shared, frozen native protocol and BOTH completed methods.
# Set RUN_NATIVE_STUDY=True only in a run that is meant to execute that protocol.
if RUN_NATIVE_STUDY:
    if "proxy_methods" not in globals():
        raise RuntimeError("Complete 16.19 and supply its student proxy_methods namespace first.")
    from chapter16_shapley_attribution_baselines.exercises.part18_spex import native_study
    native_payload = native_study.run_study(
        OUTPUT_DIR / "native", spex_methods=methods, proxy_methods=proxy_methods,
    )
    show_figure(plots.plot_native_comparison(native_payload), "09_native_model_comparison")
else:
    print("UNEXECUTED — native GPT-2 study. This draft has no native result.")
''',
"native_play": '''# Try It Yourself after a completed native study: inspect existing held-out results.
PROMPT_INDEX = 0
NATIVE_BUDGET = None  # None selects the largest completed budget; or use an existing value.
NATIVE_SEED = 0
if native_payload is not None:
    show_figure(plots.plot_native_comparison(native_payload, prompt_index=PROMPT_INDEX,
        budget=NATIVE_BUDGET, seed=NATIVE_SEED), "10_native_prompt_play")
    print("This inspection reuses completed predictions: 0 extra model evaluations.")
    print("Audit reuse is exploratory; it does not select a new fixed-study setting.")
else:
    print("UNEXECUTED — native prompt play requires a completed native_payload.")
''',
}
