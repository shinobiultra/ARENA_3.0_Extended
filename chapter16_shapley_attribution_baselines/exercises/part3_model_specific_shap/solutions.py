# %%


from __future__ import annotations

MAIN = __name__ == "__main__"

if MAIN:
    from pathlib import Path
    canonical_dir = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / "chapter16_shapley_attribution_baselines").exists()) / "chapter16_shapley_attribution_baselines" / "exercises" / "part3_model_specific_shap"

# %%

def tree_curriculum_namespace():
    """Snapshot every currently bound curriculum name; future keys remain absent."""
    namespace = globals()
    names = {'FEATURE_NAMES': 'tree_FEATURE_NAMES',
     'FIGURE_DIR': 'tree_FIGURE_DIR',
     'NATIVE_DIR': 'NATIVE_DIR',
     'OUTPUT_ROOT': 'OUTPUT_ROOT',
     'RUN_OUTPUT': 'tree_RUN_OUTPUT',
     'SAVE_FIGURES': 'tree_SAVE_FIGURES',
     '_as_query': '_as_query',
     '_benchmark_axis': '_benchmark_axis',
     '_binary_additive_tree': '_binary_additive_tree',
     '_conditional_binary_game': '_conditional_binary_game',
     '_enumerate_cardinality_weights': '_enumerate_cardinality_weights',
     '_fixed_queries': '_fixed_queries',
     '_interventional_binary_game': '_interventional_binary_game',
     '_method': 'tree__method',
     '_repeated_threshold_tree': '_repeated_threshold_tree',
     'adapt_ensemble': 'adapt_ensemble',
     'adapt_tree': 'adapt_tree',
     'audit': 'tree_audit',
     'background_order': 'tree_background_order',
     'benchmark_runtime': 'benchmark_runtime',
     'build_balanced_tree': 'build_balanced_tree',
     'chapter': 'tree_chapter',
     'check_reference': 'check_reference',
     'coalition_masks': 'tree_coalition_masks',
     'correlated_games': 'correlated_games',
     'default_play': 'default_play',
     'diagnostics': 'tree_diagnostics',
     'ensemble_path_game': 'ensemble_path_game',
     'ensemble_predict': 'ensemble_predict',
     'ensemble_saabas': 'ensemble_saabas',
     'ensemble_shap': 'ensemble_shap',
     'error': 'error',
     'evaluate_study': 'tree_evaluate_study',
     'evaluated': 'tree_evaluated',
     'exact_shapley': 'tree_exact_shapley',
     'exercises_dir': 'tree_exercises_dir',
     'explain': 'tree_explain',
     'explain_block': 'tree_explain_block',
     'extend_path': 'extend_path',
     'fit_models': 'tree_fit_models',
     'fitted': 'tree_fitted',
     'game': 'game',
     'interventional_game': 'interventional_game',
     'joint': 'joint',
     'load_split': 'tree_load_split',
     'model_key': 'model_key',
     'name': 'name',
     'native_play': 'tree_native_play',
     'path_cardinality_diagnostic': 'path_cardinality_diagnostic',
     'persist_completed': 'tree_persist_completed',
     'plots': 'tree_plots',
     'prepare_study': 'tree_prepare_study',
     'prepared': 'tree_prepared',
     'query': 'query',
     'real_play': 'tree_real_play',
     'reference': 'tree_reference',
     'repeated_feature_diagnostic': 'repeated_feature_diagnostic',
     'root_dir': 'tree_root_dir',
     'root_feature': 'root_feature',
     'row': 'row',
     'run_diagnostics': 'tree_run_diagnostics',
     'saabas': 'saabas',
     'save_payload': 'tree_save_payload',
     'section': 'tree_section',
     'section_dir': 'tree_section_dir',
     'show_figure': 'tree_show_figure',
     'size': 'size',
     'summary': 'tree_summary',
     'tests': 'tree_tests',
     'tiny_game': 'tiny_game',
     'tiny_masks': 'tiny_masks',
     'tiny_phi': 'tiny_phi',
     'tiny_play': 'tree_tiny_play',
     'tiny_tree': 'tiny_tree',
     'tiny_tree_play': 'tiny_tree_play',
     'tiny_x': 'tiny_x',
     'train': 'tree_train',
     'tree_path_game': 'tree_path_game',
     'tree_predict': 'tree_predict',
     'tree_shap': 'tree_shap',
     'unwind_path': 'unwind_path',
     'write_before_audit': 'tree_write_before_audit',
     'write_notebook_summary': 'tree_write_notebook_summary',
     '_DEFAULT_METHODS': 'tree__DEFAULT_METHODS'}
    shared = ['Any',
     'DecisionTreeRegressor',
     'GradientBoostingRegressor',
     'Mapping',
     'Path',
     'RandomForestRegressor',
     'Sequence',
     'SimpleNamespace',
     'Tensor',
     'display',
     'hashlib',
     'itertools',
     'math',
     'nn',
     'np',
     'os',
     'plt',
     'sys',
     't',
     'time',
     'torch']
    result = {name: namespace[name] for name in shared if name in namespace}
    result.update({name: namespace[target] for name, target in names.items() if target in namespace})
    return result

# %%

def deep_curriculum_namespace():
    """Snapshot every currently bound curriculum name; future keys remain absent."""
    namespace = globals()
    names = {'DTYPE': 'deep_DTYPE',
     'FEATURES': 'deep_FEATURES',
     'FEATURE_NAMES': 'deep_FEATURE_NAMES',
     'FIGURE_DIR': 'deep_FIGURE_DIR',
     'HIDDEN': 'deep_HIDDEN',
     'INDEPENDENT_BACKGROUND_SEEDS': 'INDEPENDENT_BACKGROUND_SEEDS',
     'LEARNING_RATE': 'deep_LEARNING_RATE',
     'METHOD_NAMES': 'deep_METHOD_NAMES',
     'MODEL_KEYS': 'deep_MODEL_KEYS',
     'MODEL_LABELS': 'deep_MODEL_LABELS',
     'MODEL_SEEDS': 'deep_MODEL_SEEDS',
     'PRIMARY_BACKGROUND_SEED': 'PRIMARY_BACKGROUND_SEED',
     'RANDOM_ORDER_SEED': 'deep_RANDOM_ORDER_SEED',
     'RIDGE_PENALTY': 'deep_RIDGE_PENALTY',
     'RUN_OUTPUT': 'deep_RUN_OUTPUT',
     'SAVE_FIGURES': 'deep_SAVE_FIGURES',
     'SHUFFLE_SEEDS': 'deep_SHUFFLE_SEEDS',
     'TRAINING_STEPS': 'deep_TRAINING_STEPS',
     'UNITS': 'deep_UNITS',
     'WEIGHT_DECAY': 'deep_WEIGHT_DECAY',
     '_background_indices': 'deep__background_indices',
     '_check_widths': '_check_widths',
     '_clone_state': 'deep__clone_state',
     '_clone_tree': '_clone_tree',
     '_combine_explanations': '_combine_explanations',
     '_combine_row_blocks': '_combine_row_blocks',
     '_epsilon': '_epsilon',
     '_evaluate_background_variants': '_evaluate_background_variants',
     '_evaluate_replacements': '_evaluate_replacements',
     '_explain_model': '_explain_model',
     '_fixed_random_orders': 'deep__fixed_random_orders',
     '_float_tensor': 'deep__float_tensor',
     '_fold_mlp_result': '_fold_mlp_result',
     '_fold_mlp_state': 'deep__fold_mlp_state',
     '_fold_ridge_state': 'deep__fold_ridge_state',
     '_folding_error': 'deep__folding_error',
     '_ids_tensor': 'deep__ids_tensor',
     '_layers': '_layers',
     '_linear': 'deep__linear',
     '_max_abs': 'deep__max_abs',
     '_method': 'deep__method',
     '_method_tensor': '_method_tensor',
     '_model_from_state': 'deep__model_from_state',
     '_model_output': '_model_output',
     '_normalization_tensors': 'deep__normalization_tensors',
     '_output_tensor': 'deep__output_tensor',
     '_performance_table': 'deep__performance_table',
     '_prediction': 'deep__prediction',
     '_random_curves': '_random_curves',
     '_ranked_curves': '_ranked_curves',
     '_require_disjoint_ids': 'deep__require_disjoint_ids',
     '_require_methods': '_require_methods',
     '_rows': 'deep__rows',
     '_state_shapes': 'deep__state_shapes',
     '_train_mlp': 'deep__train_mlp',
     '_training_normalization': 'deep__training_normalization',
     '_validate_evaluation_inputs': 'deep__validate_evaluation_inputs',
     '_validate_fitted': 'deep__validate_fitted',
     '_validate_rankings': '_validate_rankings',
     'affine_controls': 'affine_controls',
     'affine_fixture': 'affine_fixture',
     'audit': 'deep_audit',
     'background_average': 'background_average',
     'background_average_fixture': 'background_average_fixture',
     'background_deepshap': 'background_deepshap',
     'background_order': 'deep_background_order',
     'cancellation': 'cancellation',
     'chapter': 'deep_chapter',
     'coalition_masks': 'deep_coalition_masks',
     'coalition_values': 'coalition_values',
     'diagnose_affine_controls': 'diagnose_affine_controls',
     'diagnose_background_average': 'deep_diagnose_background_average',
     'diagnose_cancellation': 'diagnose_cancellation',
     'diagnose_near_zero': 'diagnose_near_zero',
     'diagnose_replacement_control': 'diagnose_replacement_control',
     'diagnose_rescale_limits': 'diagnose_rescale_limits',
     'diagnose_threshold': 'diagnose_threshold',
     'diagnostics': 'deep_diagnostics',
     'evaluate_study': 'deep_evaluate_study',
     'evaluated': 'deep_evaluated',
     'exact_shapley': 'deep_exact_shapley',
     'exercises_dir': 'deep_exercises_dir',
     'explain': 'deep_explain',
     'explain_block': 'deep_explain_block',
     'fit_models': 'deep_fit_models',
     'fit_one_mlp_control': 'deep_fit_one_mlp_control',
     'fit_ridge_control': 'deep_fit_ridge_control',
     'fitted': 'deep_fitted',
     'fixed_play': 'deep_fixed_play',
     'gradient_delta': 'gradient_delta',
     'index': 'index',
     'load_split': 'deep_load_split',
     'make_model': 'make_model',
     'methods': 'deep_curriculum_methods',
     'min_fixture': 'min_fixture',
     'native_dir': 'native_dir',
     'native_play': 'deep_native_play',
     'near_zero': 'near_zero',
     'near_zero_fixture': 'near_zero_fixture',
     'paired_forward': 'paired_forward',
     'persist_completed': 'deep_persist_completed',
     'play': 'play',
     'plots': 'deep_plots',
     'prepare_study': 'deep_prepare_study',
     'prepared': 'deep_prepared',
     'ranked_replacement_curves': 'ranked_replacement_curves',
     'reference': 'deep_reference',
     'replacement_control': 'replacement_control',
     'replacement_fixture': 'deep_replacement_fixture',
     'rescale_limits': 'rescale_limits',
     'rescale_multipliers': 'rescale_multipliers',
     'rescale_slope': 'rescale_slope',
     'reveal_cancel_groups': 'reveal_cancel_groups',
     'root_dir': 'deep_root_dir',
     'run_diagnostics': 'deep_run_diagnostics',
     'save_payload': 'deep_save_payload',
     'section': 'deep_section',
     'section_dir': 'deep_section_dir',
     'seed': 'deep_seed',
     'show_figure': 'deep_show_figure',
     'single_reference': 'single_reference',
     'summary': 'deep_summary',
     'tests': 'deep_tests',
     'threshold': 'threshold',
     'threshold_fixture': 'deep_threshold_fixture',
     'threshold_play': 'deep_threshold_play',
     'tiny_play': 'deep_tiny_play',
     'train': 'deep_train',
     'validate_reference': 'deep_validate_reference',
     'write_before_audit': 'deep_write_before_audit',
     'write_notebook_summary': 'deep_write_notebook_summary'}
    shared = ['Any',
     'DecisionTreeRegressor',
     'GradientBoostingRegressor',
     'Mapping',
     'Path',
     'RandomForestRegressor',
     'Sequence',
     'SimpleNamespace',
     'Tensor',
     'display',
     'hashlib',
     'itertools',
     'math',
     'nn',
     'np',
     'os',
     'plt',
     'sys',
     't',
     'time',
     'torch']
    result = {name: namespace[name] for name in shared if name in namespace}
    result.update({name: namespace[target] for name, target in names.items() if target in namespace})
    return result

# %%

def gradient_curriculum_namespace():
    """Snapshot every currently bound curriculum name; future keys remain absent."""
    namespace = globals()
    names = {'BACKGROUND_SPECS': 'BACKGROUND_SPECS',
     'BUDGETS': 'BUDGETS',
     'COALITION_DISTANCE_DONORS': 'COALITION_DISTANCE_DONORS',
     'DEEP_METHOD_NAMES': 'DEEP_METHOD_NAMES',
     'DISTANCE_BATCH': 'DISTANCE_BATCH',
     'DTYPE': 'gradient_DTYPE',
     'FEATURES': 'gradient_FEATURES',
     'FEATURE_NAMES': 'gradient_FEATURE_NAMES',
     'FIGURE_DIR': 'gradient_FIGURE_DIR',
     'FORWARD_BATCH': 'FORWARD_BATCH',
     'HIDDEN': 'gradient_HIDDEN',
     'LEARNING_RATE': 'gradient_LEARNING_RATE',
     'METHOD_NAMES': 'gradient_METHOD_NAMES',
     'MODEL_KEYS': 'gradient_MODEL_KEYS',
     'MODEL_LABELS': 'gradient_MODEL_LABELS',
     'MODEL_SEEDS': 'gradient_MODEL_SEEDS',
     'NOISE_SCHEDULE_SEED': 'NOISE_SCHEDULE_SEED',
     'NOISE_SIGMAS': 'NOISE_SIGMAS',
     'PATH_DISTANCE_SAMPLES': 'PATH_DISTANCE_SAMPLES',
     'PRIMARY_SCHEDULE_SEED': 'PRIMARY_SCHEDULE_SEED',
     'QUADRATURE_NODES': 'QUADRATURE_NODES',
     'RANDOM_ORDER_SEED': 'gradient_RANDOM_ORDER_SEED',
     'REFERENCE_SAMPLES': 'REFERENCE_SAMPLES',
     'REFERENCE_SEEDS': 'REFERENCE_SEEDS',
     'REPLICATES': 'REPLICATES',
     'REPLICATE_MODELS': 'REPLICATE_MODELS',
     'REPLICATE_SCHEDULE_SEED': 'REPLICATE_SCHEDULE_SEED',
     'RIDGE_PENALTY': 'gradient_RIDGE_PENALTY',
     'RUN_OUTPUT': 'gradient_RUN_OUTPUT',
     'SAVE_FIGURES': 'gradient_SAVE_FIGURES',
     'SHUFFLE_SEEDS': 'gradient_SHUFFLE_SEEDS',
     'TRAINING_STEPS': 'gradient_TRAINING_STEPS',
     'UNITS': 'gradient_UNITS',
     'WEIGHT_DECAY': 'gradient_WEIGHT_DECAY',
     '_background_block': '_background_block',
     '_background_comparison': '_background_comparison',
     '_background_indices': 'gradient__background_indices',
     '_batched_prediction': '_batched_prediction',
     '_checked_curves': '_checked_curves',
     '_clone_state': 'gradient__clone_state',
     '_coalition_hybrid_predictions': '_coalition_hybrid_predictions',
     '_combine_baselines': '_combine_baselines',
     '_combine_mc': '_combine_mc',
     '_combine_primary': '_combine_primary',
     '_combine_quadrature': '_combine_quadrature',
     '_comparison_baselines': '_comparison_baselines',
     '_distance_comparison': '_distance_comparison',
     '_eg_float_tensor': '_eg_float_tensor',
     '_eg_index_tensor': '_eg_index_tensor',
     '_eg_model': '_eg_model',
     '_eg_model_output': '_eg_model_output',
     '_eg_positive_int': '_eg_positive_int',
     '_eg_preserve_execution_state': '_eg_preserve_execution_state',
     '_eg_seed_words': '_eg_seed_words',
     '_eg_validate_paths': '_eg_validate_paths',
     '_fixed_random_orders': 'gradient__fixed_random_orders',
     '_float_tensor': 'gradient__float_tensor',
     '_fold_mlp_state': 'gradient__fold_mlp_state',
     '_fold_ridge_state': 'gradient__fold_ridge_state',
     '_folding_error': 'gradient__folding_error',
     '_global_rankings': '_global_rankings',
     '_identity_hash': '_identity_hash',
     '_ids_tensor': 'gradient__ids_tensor',
     '_linear': 'gradient__linear',
     '_max_abs': 'gradient__max_abs',
     '_method': 'gradient__method',
     '_model_from_state': 'gradient__model_from_state',
     '_monte_carlo_block': '_monte_carlo_block',
     '_nearest_train_distance': '_nearest_train_distance',
     '_noise_comparison': '_noise_comparison',
     '_normalization_tensors': 'gradient__normalization_tensors',
     '_output_tensor': 'gradient__output_tensor',
     '_performance_table': 'gradient__performance_table',
     '_prediction': 'gradient__prediction',
     '_primary_schedule': '_primary_schedule',
     '_quadratic_model': '_quadratic_model',
     '_quadrature_block': '_quadrature_block',
     '_relu_model': '_relu_model',
     '_replacement_evaluation': '_replacement_evaluation',
     '_replicate_comparison': '_replicate_comparison',
     '_replicate_seed': '_replicate_seed',
     '_require_disjoint_ids': 'gradient__require_disjoint_ids',
     '_require_namespace': '_require_namespace',
     '_rows': 'gradient__rows',
     '_seed_words': '_seed_words',
     '_standard_normal_schedule': '_standard_normal_schedule',
     '_state_shapes': 'gradient__state_shapes',
     '_threshold_model': '_threshold_model',
     '_train_mlp': 'gradient__train_mlp',
     '_training_normalization': 'gradient__training_normalization',
     '_two_feature_shapley': '_two_feature_shapley',
     '_validate_evaluation_inputs': 'gradient__validate_evaluation_inputs',
     '_validate_fitted': 'gradient__validate_fitted',
     'audit': 'gradient_audit',
     'background_check': 'background_check',
     'background_order': 'gradient_background_order',
     'before_audit': 'before_audit',
     'chapter': 'gradient_chapter',
     'convergence_check': 'convergence_check',
     'deep_methods': 'deep_methods',
     'diagnose_background_average': 'gradient_diagnose_background_average',
     'diagnose_convergence': 'diagnose_convergence',
     'diagnose_expected_gradient_samples': 'diagnose_expected_gradient_samples',
     'diagnose_expected_gradients': 'diagnose_expected_gradients',
     'diagnose_gradients': 'diagnose_gradients',
     'diagnose_interpolation': 'diagnose_interpolation',
     'diagnose_noise_target': 'diagnose_noise_target',
     'diagnose_path_vs_coalition': 'diagnose_path_vs_coalition',
     'diagnose_quadrature': 'diagnose_quadrature',
     'diagnose_schedule': 'diagnose_schedule',
     'diagnostics': 'gradient_diagnostics',
     'estimator_check': 'estimator_check',
     'estimator_fixture': 'estimator_fixture',
     'evaluate_study': 'gradient_evaluate_study',
     'evaluated': 'gradient_evaluated',
     'exercises_dir': 'gradient_exercises_dir',
     'expected_gradient_samples': 'expected_gradient_samples',
     'expected_gradients': 'expected_gradients',
     'explain': 'gradient_explain',
     'fit_models': 'gradient_fit_models',
     'fit_one_mlp_control': 'gradient_fit_one_mlp_control',
     'fit_ridge_control': 'gradient_fit_ridge_control',
     'fitted': 'gradient_fitted',
     'fixed_play': 'gradient_fixed_play',
     'game_check': 'game_check',
     'gauss_legendre_pairs': 'gauss_legendre_pairs',
     'gradient_check': 'gradient_check',
     'gradient_fixture': 'gradient_fixture',
     'i': 'i',
     'input_gradients': 'input_gradients',
     'interpolate_paths': 'interpolate_paths',
     'key': 'key',
     'kink_fixture': 'kink_fixture',
     'load_manifest': 'load_manifest',
     'load_split': 'gradient_load_split',
     'methods': 'gradient_methods',
     'models': 'models',
     'native_directory': 'native_directory',
     'native_play': 'gradient_native_play',
     'noise_check': 'noise_check',
     'noise_fixture': 'noise_fixture',
     'noisy_expected_gradient_samples': 'noisy_expected_gradient_samples',
     'path_check': 'path_check',
     'path_fixture': 'path_fixture',
     'persist_completed': 'gradient_persist_completed',
     'persist_prepared': 'persist_prepared',
     'plots': 'gradient_plots',
     'prepare_study': 'gradient_prepare_study',
     'prepared': 'gradient_prepared',
     'quadrature_check': 'quadrature_check',
     'quadrature_fixture': 'quadrature_fixture',
     'ranked_coalition_curves': 'ranked_coalition_curves',
     'real_play': 'gradient_real_play',
     'record': 'record',
     'reference': 'gradient_reference',
     'reference_check': 'reference_check',
     'replacement_fixture': 'gradient_replacement_fixture',
     'reproduction': 'reproduction',
     'root_dir': 'gradient_root_dir',
     'run_diagnostics': 'gradient_run_diagnostics',
     'sample_check': 'sample_check',
     'sample_fixture': 'sample_fixture',
     'sample_path_schedule': 'sample_path_schedule',
     'schedule_check': 'schedule_check',
     'schedule_fixture': 'schedule_fixture',
     'section': 'gradient_section',
     'section_dir': 'gradient_section_dir',
     'seed': 'gradient_seed',
     'show_figure': 'gradient_show_figure',
     'summarize_expected_gradients': 'summarize_expected_gradients',
     'summary': 'gradient_summary',
     'tensor_mapping_sha256': 'tensor_mapping_sha256',
     'tests': 'gradient_tests',
     'threshold_fixture': 'gradient_threshold_fixture',
     'threshold_play': 'gradient_threshold_play',
     'tiny_play': 'gradient_tiny_play',
     'train': 'gradient_train',
     'validate_reference': 'gradient_validate_reference',
     'verify_16_14_reproduction': 'verify_16_14_reproduction',
     'write_notebook_summary': 'gradient_write_notebook_summary'}
    shared = ['Any',
     'DecisionTreeRegressor',
     'GradientBoostingRegressor',
     'Mapping',
     'Path',
     'RandomForestRegressor',
     'Sequence',
     'SimpleNamespace',
     'Tensor',
     'display',
     'hashlib',
     'itertools',
     'math',
     'nn',
     'np',
     'os',
     'plt',
     'sys',
     't',
     'time',
     'torch']
    result = {name: namespace[name] for name in shared if name in namespace}
    result.update({name: namespace[target] for name, target in names.items() if target in namespace})
    return result

# %%

def refresh_tree_default_methods():
    """Explicitly restore the omitted legacy fallback from current visible bindings."""
    global tree__DEFAULT_METHODS
    from types import SimpleNamespace
    tree__DEFAULT_METHODS = SimpleNamespace(**tree_curriculum_namespace())

# %%

def deep_module_deepshap_methods_namespace():
    namespace = globals()
    names = {'_method_tensor': 'deep_helper_deepshap_methods__method_tensor',
     '_layers': 'deep_helper_deepshap_methods__layers',
     '_model_output': 'deep_helper_deepshap_methods__model_output',
     '_check_widths': 'deep_helper_deepshap_methods__check_widths',
     '_epsilon': 'deep_helper_deepshap_methods__epsilon',
     'paired_forward': 'paired_forward',
     'rescale_slope': 'rescale_slope',
     'rescale_multipliers': 'rescale_multipliers',
     'single_reference': 'single_reference',
     'background_deepshap': 'background_deepshap',
     'coalition_masks': 'deep_coalition_masks',
     'coalition_values': 'coalition_values',
     'exact_shapley': 'deep_exact_shapley',
     'reveal_cancel_groups': 'reveal_cancel_groups',
     'ranked_replacement_curves': 'ranked_replacement_curves',
     'gradient_delta': 'gradient_delta'}
    return {name: namespace[target] for name, target in names.items() if target in namespace}

# %%

def gradient_module_gradientshap_methods_namespace():
    namespace = globals()
    names = {'_eg_float_tensor': 'gradient_helper_gradientshap_methods__eg_float_tensor',
     '_eg_index_tensor': 'gradient_helper_gradientshap_methods__eg_index_tensor',
     '_eg_positive_int': 'gradient_helper_gradientshap_methods__eg_positive_int',
     '_eg_seed_words': 'gradient_helper_gradientshap_methods__eg_seed_words',
     '_eg_model': 'gradient_helper_gradientshap_methods__eg_model',
     '_eg_model_output': 'gradient_helper_gradientshap_methods__eg_model_output',
     '_eg_preserve_execution_state': 'gradient_helper_gradientshap_methods__eg_preserve_execution_state',
     '_eg_validate_paths': 'gradient_helper_gradientshap_methods__eg_validate_paths',
     'sample_path_schedule': 'sample_path_schedule',
     'interpolate_paths': 'interpolate_paths',
     'input_gradients': 'input_gradients',
     'expected_gradient_samples': 'expected_gradient_samples',
     'summarize_expected_gradients': 'summarize_expected_gradients',
     'expected_gradients': 'expected_gradients',
     'gauss_legendre_pairs': 'gauss_legendre_pairs',
     'noisy_expected_gradient_samples': 'noisy_expected_gradient_samples',
     'ranked_coalition_curves': 'ranked_coalition_curves'}
    return {name: namespace[target] for name, target in names.items() if target in namespace}

# %%

if MAIN:
    refresh_tree_default_methods()
    import itertools
if MAIN:
    refresh_tree_default_methods()
    import math
if MAIN:
    refresh_tree_default_methods()
    import os
if MAIN:
    refresh_tree_default_methods()
    import sys
if MAIN:
    refresh_tree_default_methods()
    from pathlib import Path

if MAIN:
    refresh_tree_default_methods()
    import matplotlib.pyplot as plt
if MAIN:
    refresh_tree_default_methods()
    import numpy as np
if MAIN:
    refresh_tree_default_methods()
    from IPython.display import display
if MAIN:
    refresh_tree_default_methods()
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
if MAIN:
    refresh_tree_default_methods()
    from sklearn.tree import DecisionTreeRegressor

tree_chapter = "chapter16_shapley_attribution_baselines"
tree_section = "part13_treeshap"
if MAIN:
    refresh_tree_default_methods()
    tree_root_dir = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / tree_chapter).exists())
if MAIN:
    refresh_tree_default_methods()
    tree_exercises_dir = tree_root_dir / tree_chapter / "exercises"
if MAIN:
    refresh_tree_default_methods()
    tree_section_dir = tree_exercises_dir / tree_section
if MAIN:
    refresh_tree_default_methods()
    if str(tree_root_dir) not in sys.path:
        sys.path.append(str(tree_root_dir))
    if str(tree_exercises_dir) not in sys.path:
        sys.path.append(str(tree_exercises_dir))

if MAIN:
    refresh_tree_default_methods()
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import tree_treeshap_methods_tests as tree_tests
if MAIN:
    refresh_tree_default_methods()
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.tree_treeshap_data import background_order as tree_background_order, load_split as tree_load_split

if MAIN:
    refresh_tree_default_methods()
    tree_FEATURE_NAMES = np.array([
        "Cement", "Blast-furnace slag", "Fly ash", "Water",
        "Superplasticizer", "Coarse aggregate", "Fine aggregate", "Age",
    ])
if MAIN:
    refresh_tree_default_methods()
    tree_RUN_OUTPUT = os.environ.get("ARENA_TREESHAP_OUTPUT_DIRECTORY")
if MAIN:
    refresh_tree_default_methods()
    tree_FIGURE_DIR = Path(
        os.environ.get("ARENA_TREESHAP_FIGURE_DIR", tree_RUN_OUTPUT or canonical_dir / "media" / 'tree')
    )
if MAIN:
    refresh_tree_default_methods()
    tree_SAVE_FIGURES = os.environ.get(
        "ARENA_TREESHAP_SAVE_FIGURES", "1" if tree_RUN_OUTPUT else "0"
    ) == "1"


def tree_show_figure(fig, stem):
    """Display a figure and optionally save the exact notebook rendering."""
    display(fig)
    if tree_SAVE_FIGURES:
        tree_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(tree_FIGURE_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

# %%

if MAIN:
    refresh_tree_default_methods()
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import tree_treeshap_plots as tree_plots

if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(tree_plots.plot_missing_feature_games(), "01_missing_feature_games")

# %%

def tree_predict(tree: dict, x: np.ndarray) -> np.ndarray:
    """Predict N rows with one scalar-output tree; match sklearn float32 routing."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    nodes = np.zeros(len(x), dtype=int)
    while True:
        active = np.flatnonzero(tree["left"][nodes] != -1)
        if not len(active):
            return tree["value"][nodes].copy()
        current = nodes[active]
        goes_left = x[active, tree["feature"][current]] <= tree["threshold"][current]
        nodes[active] = np.where(goes_left, tree["left"][current], tree["right"][current])


if MAIN:
    refresh_tree_default_methods()
    tree_tests.test_tree_predict(tree_predict)

# %%

def tree_path_game(tree: dict, x: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return (N, K) coalition values using local training-cover branch weights."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    masks = np.asarray(masks, dtype=bool)

    def visit(node):
        left, right = tree["left"][node], tree["right"][node]
        if left == -1:
            return np.full((len(x), len(masks)), tree["value"][node])
        feature = tree["feature"][node]
        lvalue, rvalue = visit(left), visit(right)
        weight = tree["cover"][left] / tree["cover"][node]
        absent = weight * lvalue + (1 - weight) * rvalue
        present = np.where(
            (x[:, feature] <= tree["threshold"][node])[:, None], lvalue, rvalue
        )
        return np.where(masks[None, :, feature], present, absent)

    return visit(0)


if MAIN:
    refresh_tree_default_methods()
    tree_tests.test_tree_path_game(tree_path_game)

# %%

def tree_exact_shapley(values: np.ndarray) -> np.ndarray:
    """Contract a (..., 2**D) game with the exact Shapley subset weights."""
    values = np.asarray(values, dtype=np.float64)
    d = int(math.log2(values.shape[-1]))
    if d < 1 or values.shape[-1] != 2**d:
        raise ValueError("The final axis must contain every subset of D >= 1 features.")
    phi = np.zeros((*values.shape[:-1], d))
    for feature in range(d):
        absent = [s for s in range(2**d) if not s & (1 << feature)]
        present = [s | (1 << feature) for s in absent]
        weights = np.array([1 / (d * math.comb(d - 1, s.bit_count())) for s in absent])
        phi[..., feature] = ((values[..., present] - values[..., absent]) * weights).sum(-1)
    return phi


if MAIN:
    refresh_tree_default_methods()
    tree_tests.test_exact_shapley(tree_exact_shapley)

# %%

def tree_coalition_masks(d: int) -> np.ndarray:
    """All subsets, with feature zero in the least-significant bit."""
    return np.array([[bool(s & (1 << j)) for j in range(d)] for s in range(2**d)])


if MAIN:
    refresh_tree_default_methods()
    tiny_masks = tree_coalition_masks(2)
if MAIN:
    refresh_tree_default_methods()
    tiny_tree = tree_tests.binary_tree()
if MAIN:
    refresh_tree_default_methods()
    tiny_x = np.array([[1.0, 1.0]])
if MAIN:
    refresh_tree_default_methods()
    tiny_game = tree_path_game(tiny_tree, tiny_x, tiny_masks)
if MAIN:
    refresh_tree_default_methods()
    tiny_phi = tree_exact_shapley(tiny_game)
if MAIN:
    refresh_tree_default_methods()
    print("coalition values:", tiny_game[0].tolist())
if MAIN:
    refresh_tree_default_methods()
    print("exact SHAP:      ", tiny_phi[0].tolist())

# %%

def extend_path(weights: np.ndarray, zero: np.ndarray, one: np.ndarray) -> np.ndarray:
    """Add a unique feature to batched cardinality weights: (N, k+1) -> (N, k+2)."""
    weights = np.asarray(weights, dtype=np.float64)
    zero, one = np.asarray(zero), np.asarray(one)
    k = weights.shape[1] - 1
    extended = np.zeros((len(weights), k + 2))
    for size in range(k + 1):
        extended[:, size] += zero * weights[:, size] * (k + 1 - size) / (k + 2)
        extended[:, size + 1] += one * weights[:, size] * (size + 1) / (k + 2)
    return extended


if MAIN:
    refresh_tree_default_methods()
    tree_tests.test_extend_path(extend_path)

# %%

def unwind_path(weights: np.ndarray, zero: np.ndarray, one: np.ndarray) -> np.ndarray:
    """Remove one feature's extension, without mutating its input or dividing by zero."""
    weights = np.asarray(weights, dtype=np.float64)
    zero, one = np.broadcast_arrays(np.asarray(zero), np.asarray(one))
    if np.any((zero == 0) & (one == 0)):
        raise ValueError("A dead path has no reversible extension; prune it first.")
    n = weights.shape[1] - 1
    if n < 1:
        raise ValueError("Cannot remove a feature from the empty path.")
    result = np.zeros((len(weights), n))
    next_one = weights[:, n].copy()
    nonzero = one != 0
    safe_one = np.where(nonzero, one, 1.0)
    safe_zero = np.where(nonzero, 1.0, zero)
    for size in range(n - 1, -1, -1):
        from_one = next_one * (n + 1) / ((size + 1) * safe_one)
        from_zero = weights[:, size] * (n + 1) / ((n - size) * safe_zero)
        result[:, size] = np.where(nonzero, from_one, from_zero)
        next_one = weights[:, size] - result[:, size] * zero * (n - size) / (n + 1)
    return result


if MAIN:
    refresh_tree_default_methods()
    tree_tests.test_unwind_path(unwind_path)

# %%

def tree_shap(tree: dict, x: np.ndarray, donor: np.ndarray | None = None) -> np.ndarray:
    """Exact path-game SHAP, or paired-query/donor SHAP when donor has shape (N, D)."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    if donor is not None:
        donor = np.asarray(donor, dtype=np.float32).astype(np.float64)
        if donor.shape != x.shape:
            raise ValueError("Each query needs exactly one paired donor row.")
    phi = np.zeros(x.shape, dtype=np.float64)

    def visit(node, rows, features, zeros, ones, weights):
        left, right = tree["left"][node], tree["right"][node]
        if left == -1:
            for index, feature in enumerate(features):
                other_weights = unwind_path(weights, zeros[index], ones[index])
                phi[rows, feature] += (
                    other_weights.sum(1)
                    * (ones[index] - zeros[index])
                    * tree["value"][node]
                )
            return

        feature = tree["feature"][node]
        features, zeros, ones = list(features), list(zeros), list(ones)
        incoming_zero = incoming_one = np.ones(len(rows))
        if feature in features:
            index = features.index(feature)
            incoming_zero, incoming_one = zeros.pop(index), ones.pop(index)
            weights = unwind_path(weights, incoming_zero, incoming_one)
            features.pop(index)

        query_left = x[rows, feature] <= tree["threshold"][node]
        if donor is not None:
            donor_left = donor[rows, feature] <= tree["threshold"][node]
        for child, is_left in ((left, True), (right, False)):
            present = (query_left == is_left).astype(float)
            absent = (
                (donor_left == is_left).astype(float)
                if donor is not None
                else np.full(len(rows), tree["cover"][child] / tree["cover"][node])
            )
            zero, one = incoming_zero * absent, incoming_one * present
            keep = (zero != 0) | (one != 0)
            if not keep.any():
                continue
            visit(
                child,
                rows[keep],
                features + [feature],
                [z[keep] for z in zeros] + [zero[keep]],
                [o[keep] for o in ones] + [one[keep]],
                extend_path(weights[keep], zero[keep], one[keep]),
            )

    visit(0, np.arange(len(x)), [], [], [], np.ones((len(x), 1)))
    return phi


if MAIN:
    refresh_tree_default_methods()
    tree_tests.test_tree_shap(tree_shap)

# %%

if MAIN:
    refresh_tree_default_methods()
    import itertools

if MAIN:
    refresh_tree_default_methods()
    import math

if MAIN:
    refresh_tree_default_methods()
    import time

if MAIN:
    refresh_tree_default_methods()
    import numpy as np

def tree__method(methods, name):
    """Resolve a taught method in both the package and AST-extracted notebook."""
    if methods is not None:
        return getattr(methods, name) if hasattr(methods, name) else methods[name]
    local = tree_curriculum_namespace().get(name)
    if callable(local):
        return local
    return getattr(tree__DEFAULT_METHODS, name)

def _as_query(query):
    query = np.asarray(query, dtype=float)
    if query.shape != (2,) or not np.isin(query, [0.0, 1.0]).all():
        raise ValueError("The binary fixture query must contain exactly two zeros/ones.")
    return query

# %%

def _binary_additive_tree(order=(0, 1), joint=((9.0, 1.0), (1.0, 9.0))):
    """Build an exact tree for f(x)=x0+2*x1 with population node covers."""
    order = tuple(int(feature) for feature in order)
    if sorted(order) != [0, 1]:
        raise ValueError("order must be a permutation of (0, 1).")
    joint = np.asarray(joint, dtype=float)
    if joint.shape != (2, 2) or np.any(joint <= 0):
        raise ValueError("joint must be a positive 2x2 table in (x0, x1) order.")
    examples = np.array(list(itertools.product([0, 1], repeat=2)))
    counts = joint.ravel()
    fields = {key: [] for key in ("left", "right", "feature", "threshold", "value", "cover")}

    def add(depth, keep):
        node = len(fields["left"])
        for key in fields:
            fields[key].append(-1)
        fields["cover"][node] = counts[keep].sum()
        fields["value"][node] = np.average(examples[keep] @ [1.0, 2.0], weights=counts[keep])
        fields["threshold"][node] = 0.5
        if depth < 2:
            feature = order[depth]
            fields["feature"][node] = feature
            fields["left"][node] = add(depth + 1, keep[examples[keep, feature] == 0])
            fields["right"][node] = add(depth + 1, keep[examples[keep, feature] == 1])
        return node

    add(0, np.arange(4))
    return {
        key: np.asarray(value, dtype=int if key in ("left", "right", "feature") else float)
        for key, value in fields.items()
    }

def _repeated_threshold_tree():
    """A hand-computable tree where x0 splits twice but remains one player."""
    return {
        "left": np.array([1, -1, 3, -1, -1]),
        "right": np.array([2, -1, 4, -1, -1]),
        "feature": np.array([0, -1, 0, -1, -1]),
        "threshold": np.array([1.0, 0.0, 2.0, 0.0, 0.0]),
        "value": np.array([20.0, 10.0, 80 / 3, 20.0, 30.0]),
        "cover": np.array([100.0, 40.0, 60.0, 20.0, 40.0]),
    }

# %%

def _conditional_binary_game(query, joint):
    """Population-conditional game E[f(X) | X_S=query_S] by direct enumeration."""
    examples = np.array(list(itertools.product([0, 1], repeat=2)), dtype=float)
    probabilities = np.asarray(joint, dtype=float).ravel()
    probabilities /= probabilities.sum()
    targets = examples @ np.array([1.0, 2.0])
    masks = np.array(list(itertools.product([False, True], repeat=2)), dtype=bool)
    # Reorder lexicographic masks to the least-significant-feature convention.
    masks = masks[[0, 2, 1, 3]]
    values = []
    for mask in masks:
        matches = np.all(examples[:, mask] == query[mask], axis=1)
        values.append(np.average(targets[matches], weights=probabilities[matches]))
    return masks, np.asarray(values)

def _interventional_binary_game(query, joint):
    """Marginal/interventional game E[f(x_S, X_-S)] by direct enumeration."""
    examples = np.array(list(itertools.product([0, 1], repeat=2)), dtype=float)
    probabilities = np.asarray(joint, dtype=float).ravel()
    probabilities /= probabilities.sum()
    masks = np.array([[False, False], [True, False], [False, True], [True, True]])
    values = []
    for mask in masks:
        hybrids = np.where(mask, query, examples)
        values.append(np.average(hybrids @ np.array([1.0, 2.0]), weights=probabilities))
    return masks, np.asarray(values)

# %%

def correlated_games(query=(1.0, 1.0), joint=((0.45, 0.05), (0.05, 0.45)), methods=None):
    """Compare conditional, interventional and two cover-weighted path games.

    The two trees compute the same f(x)=x0+2*x1 on all binary inputs.  Only the
    root split changes, which makes any difference in their path attributions a
    property of the cover-weighted game rather than the predictive function.
    """
    query = _as_query(query)
    joint = np.asarray(joint, dtype=float)
    masks, conditional = _conditional_binary_game(query, joint)
    _, interventional = _interventional_binary_game(query, joint)
    trees = [_binary_additive_tree((0, 1), joint), _binary_additive_tree((1, 0), joint)]
    path_game = tree__method(methods, "tree_path_game")
    exact = tree__method(methods, "exact_shapley")
    saabas_fn = tree__method(methods, "saabas")
    games = np.vstack(
        [
            conditional,
            interventional,
            path_game(trees[0], query[None], masks)[0],
            path_game(trees[1], query[None], masks)[0],
        ]
    )
    shapley = exact(games)
    saabas_values = np.vstack([saabas_fn(tree, query[None])[1][0] for tree in trees])
    predictions = np.array(
        [tree__method(methods, "tree_predict")(tree, query[None])[0] for tree in trees]
    )
    return {
        "query": query,
        "joint": joint,
        "masks": masks,
        "game_names": [
            "Population conditional",
            "Interventional",
            "Path: x0 root",
            "Path: x1 root",
        ],
        "games": games,
        "shapley": shapley,
        "trees": trees,
        "saabas": saabas_values,
        "predictions": predictions,
        "empty_value": float(games[0, 0]),
    }

# %%

def _enumerate_cardinality_weights(zeros, ones):
    """Definition-level subset enumeration for one unique-feature path."""
    zeros, ones = np.asarray(zeros, dtype=float), np.asarray(ones, dtype=float)
    count = len(zeros)
    answer = np.zeros(count + 1)
    for flags in itertools.product([False, True], repeat=count):
        size = sum(flags)
        product = math.prod(ones[j] if flag else zeros[j] for j, flag in enumerate(flags))
        answer[size] += (
            product
            * math.factorial(size)
            * math.factorial(count - size)
            / math.factorial(count + 1)
        )
    return answer

def path_cardinality_diagnostic(zeros=(0.2, 0.7, 0.5), ones=(1.0, 0.0, 1.0), methods=None):
    """Record each extend-path update and verify it against subset enumeration."""
    zeros, ones = np.asarray(zeros, dtype=float), np.asarray(ones, dtype=float)
    if zeros.shape != ones.shape or zeros.ndim != 1 or not len(zeros):
        raise ValueError(
            "zeros and ones must be non-empty one-dimensional arrays of equal length."
        )
    extend = tree__method(methods, "extend_path")
    weights = np.ones((1, 1))
    stages, exact_stages = [weights[0].copy()], [np.ones(1)]
    for index in range(len(zeros)):
        weights = extend(weights, zeros[[index]], ones[[index]])
        stages.append(weights[0].copy())
        exact_stages.append(
            _enumerate_cardinality_weights(zeros[: index + 1], ones[: index + 1])
        )
    return {
        "zeros": zeros,
        "ones": ones,
        "stages": stages,
        "enumerated_stages": exact_stages,
        "max_abs_error": float(
            max(np.max(np.abs(a - b)) for a, b in zip(stages, exact_stages))
        ),
    }

# %%

def repeated_feature_diagnostic(methods=None):
    """Show that repeated x0 splits contract to one player and a dummy stays zero."""
    tree = _repeated_threshold_tree()
    query = np.array([[0.0, 99.0], [1.5, -7.0], [3.0, 4.0]])
    masks = tree__method(methods, "coalition_masks")(2)
    games = tree__method(methods, "tree_path_game")(tree, query, masks)
    phi = tree__method(methods, "tree_shap")(tree, query)
    base, saabas_values = tree__method(methods, "saabas")(tree, query)
    predictions = tree__method(methods, "tree_predict")(tree, query)
    return {
        "tree": tree,
        "query": query,
        "masks": masks,
        "games": games,
        "base": base,
        "phi": phi,
        "saabas": saabas_values,
        "prediction": predictions,
        "dummy_max_abs": float(np.max(np.abs(phi[:, 1]))),
        "additivity_max_abs_error": float(np.max(np.abs(base + phi.sum(1) - predictions))),
    }

# %%

def build_balanced_tree(depth, n_features=None, seed=161390):
    """Build the frozen balanced binary timing tree with one feature per level."""
    depth = int(depth)
    n_features = depth if n_features is None else int(n_features)
    if depth < 1 or n_features < depth:
        raise ValueError("depth must be positive and n_features must supply every tree level.")
    node_count = 2 ** (depth + 1) - 1
    leaf_start = 2**depth - 1
    left = np.full(node_count, -1, dtype=int)
    right = np.full(node_count, -1, dtype=int)
    feature = np.full(node_count, -1, dtype=int)
    threshold = np.zeros(node_count, dtype=float)
    value = np.zeros(node_count, dtype=float)
    cover = np.ones(node_count, dtype=float)
    for node in range(leaf_start):
        level = int(math.floor(math.log2(node + 1)))
        left[node], right[node], feature[node] = 2 * node + 1, 2 * node + 2, level
        threshold[node] = 0.5
    value[leaf_start:] = np.random.default_rng(seed).normal(loc=10.0, scale=3.0, size=2**depth)
    for node in range(leaf_start - 1, -1, -1):
        cover[node] = cover[left[node]] + cover[right[node]]
        value[node] = (
            cover[left[node]] * value[left[node]] + cover[right[node]] * value[right[node]]
        ) / cover[node]
    return {
        "left": left,
        "right": right,
        "feature": feature,
        "threshold": threshold,
        "value": value,
        "cover": cover,
        "n_features": np.array(n_features, dtype=int),
    }

def _fixed_queries(n_features):
    """Eight deterministic binary queries, with a visible 0..7 bit pattern."""
    integers = np.arange(8, dtype=int)
    return np.array(
        [
            [(number >> (feature % 3)) & 1 for feature in range(n_features)]
            for number in integers
        ],
        dtype=float,
    )

# %%

def _benchmark_axis(specifications, methods, repeats):
    masks_fn = tree__method(methods, "coalition_masks")
    game_fn = tree__method(methods, "tree_path_game")
    exact_fn = tree__method(methods, "exact_shapley")
    dp_fn = tree__method(methods, "tree_shap")
    exact_times, dp_times, errors = [], [], []
    for depth, supplied_features in specifications:
        tree = build_balanced_tree(depth, supplied_features, seed=161390)
        query = _fixed_queries(supplied_features)
        masks = masks_fn(supplied_features)
        expected = exact_fn(game_fn(tree, query, masks))
        actual = dp_fn(tree, query)
        errors.append(float(np.max(np.abs(expected - actual))))
        current_exact, current_dp = [], []
        for _ in range(repeats):
            started = time.perf_counter()
            exact_fn(game_fn(tree, query, masks))
            current_exact.append(time.perf_counter() - started)
            started = time.perf_counter()
            dp_fn(tree, query)
            current_dp.append(time.perf_counter() - started)
        exact_times.append(current_exact)
        dp_times.append(current_dp)
    exact_times, dp_times = np.asarray(exact_times), np.asarray(dp_times)
    return {
        "exhaustive_seconds": exact_times,
        "dynamic_programming_seconds": dp_times,
        "exhaustive_median_seconds": np.median(exact_times, axis=1),
        "dynamic_programming_median_seconds": np.median(dp_times, axis=1),
        "max_abs_error": np.asarray(errors),
        "queries_per_timing": np.array(8, dtype=int),
        "repeats": np.array(repeats, dtype=int),
    }

def benchmark_runtime(methods=None, repeats=3):
    """Run both frozen timing sweeps and retain every wall-clock measurement."""
    repeats = int(repeats)
    if repeats != 3:
        raise ValueError("The frozen protocol requires exactly three timing repetitions.")
    dummy_features = np.array([3, 6, 9, 12, 15], dtype=int)
    depths = np.arange(2, 9, dtype=int)
    dummy = _benchmark_axis([(3, int(count)) for count in dummy_features], methods, repeats)
    depth = _benchmark_axis([(int(value), int(value)) for value in depths], methods, repeats)
    dummy["supplied_features"] = dummy_features
    dummy["tree_depth"] = np.array(3, dtype=int)
    depth["tree_depth"] = depths
    depth["supplied_features"] = depths.copy()
    return {"dummy_features": dummy, "depth": depth}

# %%

def tiny_tree_play(
    query=(1.0, 1.0),
    joint=((9.0, 1.0), (1.0, 9.0)),
    root_feature=0,
    methods=None,
):
    """Editable tiny-tree play: change the query, root feature, or four covers."""
    query = _as_query(query)
    root_feature = int(root_feature)
    if root_feature not in (0, 1):
        raise ValueError("root_feature must be 0 or 1.")
    tree = _binary_additive_tree((root_feature, 1 - root_feature), joint)
    masks = tree__method(methods, "coalition_masks")(2)
    game = tree__method(methods, "tree_path_game")(tree, query[None], masks)[0]
    phi = tree__method(methods, "tree_shap")(tree, query[None])[0]
    base, saabas_values = tree__method(methods, "saabas")(tree, query[None])
    prediction = tree__method(methods, "tree_predict")(tree, query[None])[0]
    return {
        "query": query,
        "joint": np.asarray(joint, dtype=float),
        "root_feature": np.array(root_feature, dtype=int),
        "tree": tree,
        "masks": masks,
        "game": game,
        "base": np.array(base[0]),
        "phi": phi,
        "saabas": saabas_values[0],
        "prediction": np.array(prediction),
    }

# %%

def tree_run_diagnostics(methods=None, include_timing=True):
    """Return exact mechanism evidence and a separately persisted timing branch."""
    mechanism = {
        "correlated": correlated_games(methods=methods),
        "cardinality": path_cardinality_diagnostic(methods=methods),
        "repeated_feature": repeated_feature_diagnostic(methods=methods),
        "tiny_play": tiny_tree_play(methods=methods),
    }
    timing = benchmark_runtime(methods=methods) if include_timing else {}
    if timing:
        mechanism["runtime_parity"] = {
            axis: {
                "coordinate": (
                    values["supplied_features"]
                    if axis == "dummy_features"
                    else values["tree_depth"]
                ),
                "max_abs_error": values["max_abs_error"],
            }
            for axis, values in timing.items()
        }
        mechanism["runtime_parity"]["dummy_feature_max_abs"] = np.array(
            np.max(timing["dummy_features"]["max_abs_error"])
        )
        mechanism["runtime_parity"]["depth_max_abs"] = np.array(
            np.max(timing["depth"]["max_abs_error"])
        )
        timing = {
            axis: {key: value for key, value in values.items() if key != "max_abs_error"}
            for axis, values in timing.items()
        }
    return {"mechanism": mechanism, "timing": timing}

# %%

def ensemble_shap(
    ensemble: dict, x: np.ndarray, background: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Use tree and donor linearity; return base (N,) and contributions (N, D)."""
    x = np.asarray(x)
    phi = np.zeros(x.shape, dtype=float)
    base = np.full(len(x), float(ensemble["offset"]))
    for tree, weight in zip(ensemble["trees"], ensemble["weights"], strict=True):
        if background is None:
            contribution = tree_shap(tree, x)
            empty = tree_path_game(tree, x[:1], np.zeros((1, x.shape[1]), dtype=bool))[0, 0]
        else:
            background = np.asarray(background)
            if not len(background):
                raise ValueError("The empirical background cannot be empty.")
            queries = np.repeat(x, len(background), axis=0)
            donors = np.tile(background, (len(x), 1))
            contribution = tree_shap(tree, queries, donors).reshape(
                len(x), len(background), x.shape[1]
            ).mean(1)
            empty = tree_predict(tree, background).mean()
        phi += weight * contribution
        base += weight * empty
    return base, phi


if MAIN:
    refresh_tree_default_methods()
    tree_tests.test_ensemble_shap(ensemble_shap)

# %%

def interventional_game(
    predict, x: np.ndarray, background: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    """Average predictions of intact-donor hybrids, retaining all supplied donors."""
    if not len(background):
        raise ValueError("The empirical background cannot be empty.")
    result = np.empty((len(x), len(masks)))
    for start in range(0, len(x), 8):
        query = x[start : start + 8]
        hybrids = np.where(
            masks[None, :, None, :], query[:, None, None, :], background[None, None, :, :]
        )
        predictions = predict(hybrids.reshape(-1, x.shape[1]))
        result[start : start + len(query)] = predictions.reshape(
            len(query), len(masks), len(background)
        ).mean(2)
    return result


if MAIN:
    refresh_tree_default_methods()
    tree_tests.test_interventional_game(interventional_game)

# %%

def saabas(tree: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each visited parent-to-child expectation change to its split feature."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    expected = np.zeros(len(tree["value"]))

    def expectation(node):
        left, right = tree["left"][node], tree["right"][node]
        if left == -1:
            expected[node] = tree["value"][node]
        else:
            expected[node] = (
                tree["cover"][left] * expectation(left)
                + tree["cover"][right] * expectation(right)
            ) / tree["cover"][node]
        return expected[node]

    expectation(0)
    phi = np.zeros(x.shape, dtype=float)
    for row, query in enumerate(x):
        node = 0
        while tree["left"][node] != -1:
            feature = tree["feature"][node]
            child = (
                tree["left"][node]
                if query[feature] <= tree["threshold"][node]
                else tree["right"][node]
            )
            phi[row, feature] += expected[child] - expected[node]
            node = child
    return np.full(len(x), expected[0]), phi


if MAIN:
    refresh_tree_default_methods()
    tree_tests.test_saabas(saabas)

# %%

if MAIN:
    refresh_tree_default_methods()
    tree_diagnostics = tree_run_diagnostics(methods=tree_curriculum_namespace(), include_timing=True)
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(tree_plots.plot_tiny_tree(tree_diagnostics), "02_tiny_tree")
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(tree_plots.plot_cardinality_weights(tree_diagnostics), "03_cardinality_weights")
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(tree_plots.plot_correlated_games(tree_diagnostics), "04_correlated_games")
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(tree_plots.plot_repeated_feature(tree_diagnostics), "05_repeated_feature")
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(tree_plots.plot_saabas_control(tree_diagnostics), "06_saabas_control")

# %%

if MAIN:
    refresh_tree_default_methods()
    query = np.array([1.0, 1.0])
if MAIN:
    refresh_tree_default_methods()
    joint = np.array([[9.0, 1.0], [1.0, 9.0]])
root_feature = 0
if MAIN:
    refresh_tree_default_methods()
    tree_tiny_play = tiny_tree_play(query=query, joint=joint, root_feature=root_feature)
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(tree_plots.plot_tiny_tree_play(tree_tiny_play), "15_tiny_tree_play")

# %%

if MAIN:
    refresh_tree_default_methods()
    import numpy as np

if MAIN:
    refresh_tree_default_methods()
    from sklearn.tree import DecisionTreeRegressor

if MAIN:
    refresh_tree_default_methods()
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def adapt_tree(estimator):
    raw = estimator.tree_
    return {
        "left": raw.children_left.copy(),
        "right": raw.children_right.copy(),
        "feature": raw.feature.copy(),
        "threshold": raw.threshold.copy(),
        "value": raw.value[:, 0, 0].copy(),
        "cover": raw.weighted_n_node_samples.copy(),
    }

def adapt_ensemble(model, family):
    if family == "tree":
        estimators, weights, offset = [model], np.ones(1), 0.0
    elif family == "forest":
        estimators = model.estimators_
        weights, offset = np.full(len(estimators), 1 / len(estimators)), 0.0
    elif family == "boost":
        estimators = model.estimators_.ravel()
        weights = np.full(len(estimators), model.learning_rate)
        offset = float(model.init_.constant_[0, 0])
    else:
        raise ValueError(f"Unknown family {family}.")
    return {
        "trees": [adapt_tree(tree) for tree in estimators],
        "weights": weights,
        "offset": offset,
    }

def ensemble_predict(ensemble, x):
    return float(ensemble["offset"]) + sum(
        weight * tree_predict(tree, x)
        for tree, weight in zip(ensemble["trees"], ensemble["weights"], strict=True)
    )

def ensemble_path_game(ensemble, x, masks):
    return float(ensemble["offset"]) + sum(
        weight * tree_path_game(tree, x, masks)
        for tree, weight in zip(ensemble["trees"], ensemble["weights"], strict=True)
    )

def ensemble_saabas(ensemble, x):
    return sum(
        weight * saabas(tree, x)[1]
        for tree, weight in zip(ensemble["trees"], ensemble["weights"], strict=True)
    )

# %%

def tree_fit_models(train):
    models = {}
    for family in ("tree", "forest", "boost"):
        for seed_index in range(3):
            for target_kind in ("clean", "shuffled"):
                seed = 161310 + seed_index
                y = train["y"].copy()
                if target_kind == "shuffled":
                    y = y[np.random.default_rng(161320 + seed_index).permutation(len(y))]
                if family == "tree":
                    model = DecisionTreeRegressor(
                        max_depth=5, min_samples_leaf=8, random_state=seed
                    )
                elif family == "forest":
                    model = RandomForestRegressor(
                        n_estimators=32,
                        max_depth=5,
                        min_samples_leaf=8,
                        max_features=1.0,
                        bootstrap=True,
                        n_jobs=1,
                        random_state=seed,
                    )
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=48,
                        learning_rate=0.05,
                        max_depth=3,
                        min_samples_leaf=8,
                        loss="squared_error",
                        subsample=1.0,
                        random_state=seed,
                    )
                model.fit(train["x"], y)
                key = f"{family}_{target_kind}_{seed_index}"
                models[key] = {
                    "sklearn": model,
                    "ensemble": adapt_ensemble(model, family),
                    "training_targets": y,
                    "parameters": model.get_params(),
                    "train_prediction": model.predict(train["x"]),
                }
    return {
        "models": models,
        "train_ids": train["ids"].copy(),
        "training_mean": float(train["y"].mean()),
    }

# %%

def tree_explain_block(fitted, x, background):
    result = {}
    masks = tree_coalition_masks(x.shape[1])
    exact_keys = {f"{family}_clean_0" for family in ("tree", "forest", "boost")}
    for key, item in fitted["models"].items():
        ensemble, model = item["ensemble"], item["sklearn"]
        prediction = ensemble_predict(ensemble, x)
        np.testing.assert_allclose(prediction, model.predict(x), atol=1e-10, rtol=1e-10)
        base_path, phi_path = ensemble_shap(ensemble, x)
        base_interventional, phi_interventional = ensemble_shap(ensemble, x, background)
        phi_saabas = ensemble_saabas(ensemble, x)
        for base, phi in (
            (base_path, phi_path),
            (base_interventional, phi_interventional),
            (base_path, phi_saabas),
        ):
            np.testing.assert_allclose(base + phi.sum(1), prediction, atol=1e-8, rtol=1e-8)
        result[key] = {
            "prediction": prediction,
            "base_path": base_path,
            "phi_path": phi_path,
            "base_interventional": base_interventional,
            "phi_interventional": phi_interventional,
            "phi_saabas": phi_saabas,
        }
        if key in exact_keys:
            game_path = ensemble_path_game(ensemble, x, masks)
            # Independent prediction route: exact hybrid games use the fitted sklearn model.
            game_interventional = interventional_game(model.predict, x, background, masks)
            for game, phi in ((game_path, phi_path), (game_interventional, phi_interventional)):
                np.testing.assert_allclose(tree_exact_shapley(game), phi, atol=1e-8, rtol=1e-8)
            result[key].update(game_path=game_path, game_interventional=game_interventional)
    return result

def tree_prepare_study(fitted, train, explain):
    order = tree_background_order(train["ids"], 161360)
    background = train["x"][order[:64]]
    block = tree_explain_block(fitted, explain["x"], background)
    rankings = {
        key: {
            name: np.argsort(-np.abs(item[f"phi_{name}"]).mean(0), kind="stable")
            for name in ("path", "interventional", "saabas")
        }
        for key, item in block.items()
    }
    return {
        "explanation": block,
        "rankings": rankings,
        "explain_ids": explain["ids"].copy(),
        "background_ids": train["ids"][order[:64]],
        "background": background,
        "audit_used_for_selection": False,
    }

# %%

def tree_evaluate_study(fitted, prepared, train, explain, audit):
    block = tree_explain_block(fitted, audit["x"], prepared["background"])
    rng = np.random.default_rng(161380)
    random_orders = np.array([rng.permutation(8) for _ in range(100)])
    deletion = {}
    for key, item in block.items():
        if "game_interventional" not in item:
            continue
        orders = np.vstack(
            [prepared["rankings"][key][name] for name in ("path", "interventional", "saabas")]
            + [random_orders]
        )
        masks_by_order = []
        for order in orders:
            mask, sequence = 255, [255]
            for feature in order:
                mask &= ~(1 << int(feature))
                sequence.append(mask)
            masks_by_order.append(sequence)
        predictions = item["game_interventional"][:, np.array(masks_by_order)]
        deletion[key] = {
            "orders": orders,
            "remaining_masks": np.array(masks_by_order),
            "mse": ((predictions - audit["y"][:, None, None]) ** 2).mean(0),
        }
    primary = fitted["models"]["forest_clean_0"]["ensemble"]
    all_x = np.vstack([explain["x"], audit["x"]])
    backgrounds = {}
    for size, seed in ((16, 161360), (128, 161360), (64, 161361), (64, 161362)):
        order = tree_background_order(train["ids"], seed)[:size]
        base, phi = ensemble_shap(primary, all_x, train["x"][order])
        backgrounds[f"{size}_{seed}"] = {"ids": train["ids"][order], "base": base, "phi": phi}
    return {
        "audit": block,
        "audit_ids": audit["ids"].copy(),
        "audit_targets": audit["y"].copy(),
        "deletion": deletion,
        "backgrounds": backgrounds,
        "performance": {
            key: float(np.mean((item["prediction"] - audit["y"]) ** 2))
            for key, item in block.items()
        },
        "mean_baseline_mse": float(np.mean((fitted["training_mean"] - audit["y"]) ** 2)),
    }

# %%

def tree_native_play(
    fitted, train, audit, model_key="forest_clean_0", row=0, size=64, game="interventional"
):
    query = audit["x"][[row]]
    indices = tree_background_order(train["ids"], 161360)[:size]
    background = train["x"][indices] if game == "interventional" else None
    if game not in ("interventional", "path"):
        raise ValueError("Choose interventional or path.")
    ensemble = fitted["models"][model_key]["ensemble"]
    base, phi = ensemble_shap(ensemble, query, background)
    return {
        "query": query,
        "source_id": audit["ids"][[row]],
        "base": base,
        "phi": phi,
        "prediction": ensemble_predict(ensemble, query),
        "target": audit["y"][[row]],
    }

# %%

tree_helper_run_native_DETERMINISTIC_ARTIFACTS = (
    "fitted.pt",
    "prepared.pt",
    "evaluated.pt",
    "diagnostics.pt",
    "reference.pt",
    "play.pt",
)

# %%

tree_helper_run_native_TIMING_ARTIFACT = "timings.pt"

# %%

if MAIN:
    from pathlib import Path
if MAIN:
    import hashlib

def tree_helper_run_native_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np

def tree_helper_run_native__without_sklearn(value: Any) -> Any:
    """Copy a nested fitted payload while removing live sklearn estimators."""
    if isinstance(value, dict):
        return {key: tree_helper_run_native__without_sklearn(item) for key, item in value.items() if key != "sklearn"}
    if isinstance(value, list):
        return [tree_helper_run_native__without_sklearn(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_helper_run_native__without_sklearn(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value

# %%

def tree_helper_run_native_serializable_fitted(fitted: dict) -> dict:
    """Return fitted numerical/tree evidence without live sklearn estimators."""
    return tree_helper_run_native__without_sklearn(fitted)

# %%

if MAIN:
    from typing import Any
if MAIN:
    from pathlib import Path
if MAIN:
    import torch

def tree_helper_run_native_save_payload(directory: Path, name: str, value: Any) -> Path:
    """Persist a named course-owned payload; fitted.pt is stripped of sklearn objects."""
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("Payload names must be simple .pt filenames.")
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "fitted.pt":
        value = tree_helper_run_native_serializable_fitted(value)
    torch.save(value, path)
    return path

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np

def tree_helper_run_native__reference_scalars(value: Any, prefix: str = "") -> dict[str, int | float | bool | str]:
    """Flatten scalar reference outcomes for the JSON summary; arrays remain in reference.pt."""
    result: dict[str, int | float | bool | str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(tree_helper_run_native__reference_scalars(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (bool, str, int, float, np.bool_, np.integer, np.floating)):
        result[prefix] = value.item() if isinstance(value, np.generic) else value
    return result

# %%

if MAIN:
    from typing import Any

def tree_helper_run_native_summarize(
    fitted: dict,
    prepared: dict,
    evaluated: dict,
    diagnostics: dict,
    reference: dict,
) -> dict[str, Any]:
    models = fitted["models"]
    component_trees = sum(len(item["ensemble"]["trees"]) for item in models.values())
    exact_keys = [key for key, item in prepared["explanation"].items() if "game_path" in item]
    clean = {key: value for key, value in evaluated["performance"].items() if "_clean_" in key}
    shuffled = {
        key: value for key, value in evaluated["performance"].items() if "_shuffled_" in key
    }
    return {
        "model_fits": len(models),
        "clean_fits": len(clean),
        "shuffled_fits": len(shuffled),
        "component_trees": component_trees,
        "model_families": ["tree", "forest", "boost"],
        "training_seeds": [161310, 161311, 161312],
        "train_rows": len(fitted["train_ids"]),
        "explanation_row_count": len(prepared["explain_ids"]),
        "heldout_row_count": len(evaluated["audit_ids"]),
        "rows_explained_per_model": len(prepared["explain_ids"]) + len(evaluated["audit_ids"]),
        "model_row_explanations": len(models)
        * (len(prepared["explain_ids"]) + len(evaluated["audit_ids"])),
        "coalitions": 256,
        "exact_models": len(exact_keys),
        "exact_games": 2,
        "background_rows": len(prepared["background_ids"]),
        "background_variants": len(evaluated["backgrounds"]),
        "deletion_models": len(evaluated["deletion"]),
        "saabas_models": len(models),
        "random_order_controls": 100,
        "audit_used_for_selection": prepared["audit_used_for_selection"],
        "audit_mse_clean": clean,
        "audit_mse_shuffled": shuffled,
        "training_mean_audit_mse": evaluated["mean_baseline_mse"],
        "diagnostic_groups": sorted(diagnostics),
        "reference_scalars": tree_helper_run_native__reference_scalars(reference),
    }

# %%

if MAIN:
    from typing import Any
if MAIN:
    from pathlib import Path
if MAIN:
    import json

def tree_helper_run_native_persist_completed(
    directory: Path,
    fitted: dict,
    prepared: dict,
    evaluated: dict,
    diagnostics: dict,
    reference: dict,
    play: dict,
) -> dict[str, Any]:
    """Persist post-audit evidence, retaining timing in its own nondeterministic artifact."""
    directory = Path(directory)
    boundary_path = directory / "before_audit.json"
    if not boundary_path.exists():
        raise ValueError("Prepared evidence must be persisted before audit evaluation.")
    boundary = json.loads(boundary_path.read_text())
    if boundary.get("audit_loaded") is not False:
        raise ValueError("The recorded preparation boundary is invalid.")
    if tree_helper_run_native_digest(directory / "prepared.pt") != boundary.get("prepared_sha256"):
        raise ValueError("Prepared evidence changed after the audit boundary.")
    if diagnostics.keys() != {"mechanism", "timing"}:
        raise ValueError("Diagnostics must retain separate mechanism and timing branches.")
    tree_helper_run_native_save_payload(directory, "evaluated.pt", evaluated)
    tree_helper_run_native_save_payload(directory, "diagnostics.pt", diagnostics["mechanism"])
    tree_helper_run_native_save_payload(directory, tree_helper_run_native_TIMING_ARTIFACT, diagnostics["timing"])
    tree_helper_run_native_save_payload(directory, "reference.pt", reference)
    tree_helper_run_native_save_payload(directory, "play.pt", play)
    summary = tree_helper_run_native_summarize(fitted, prepared, evaluated, diagnostics["mechanism"], reference)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    expected = set(tree_helper_run_native_DETERMINISTIC_ARTIFACTS) | {tree_helper_run_native_TIMING_ARTIFACT}
    present = {path.name for path in directory.glob("*.pt")}
    if present != expected:
        raise ValueError(f"Native artifact set mismatch: {sorted(present)}")
    return summary

# %%

if MAIN:
    from typing import Any
if MAIN:
    from pathlib import Path
if MAIN:
    import hashlib
if MAIN:
    import json
if MAIN:
    import numpy as np

def tree_helper_run_native_write_before_audit(directory: Path, prepared: dict) -> dict[str, Any]:
    """Seal already-saved fitted/prepared payloads before the audit split is loaded."""
    directory = Path(directory)
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("Prepared rankings must explicitly exclude the audit split.")
    fitted_path = directory / "fitted.pt"
    prepared_path = directory / "prepared.pt"
    if not fitted_path.is_file() or not prepared_path.is_file():
        raise ValueError("Save fitted.pt and prepared.pt before sealing the audit boundary.")
    boundary = {
        "phase": "prepared_before_audit",
        "audit_loaded": False,
        "fitted_sha256": tree_helper_run_native_digest(fitted_path),
        "prepared_sha256": tree_helper_run_native_digest(prepared_path),
        "explain_ids_sha256": hashlib.sha256(
            np.asarray(prepared["explain_ids"]).tobytes()
        ).hexdigest(),
        "background_ids_sha256": hashlib.sha256(
            np.asarray(prepared["background_ids"]).tobytes()
        ).hexdigest(),
    }
    (directory / "before_audit.json").write_text(json.dumps(boundary, indent=2) + "\n")
    return boundary

# %%

if MAIN:
    from pathlib import Path
if MAIN:
    import json

def tree_helper_run_native_write_notebook_summary(output_root: Path, native_directory: Path, summary: dict) -> Path:
    """Write the small JSON index consumed by the fresh-kernel executor."""
    output_root = Path(output_root).resolve()
    native_directory = Path(native_directory).resolve()
    native_directory.relative_to(output_root)
    payload = dict(summary, native_directory=str(native_directory))
    path = output_root / "native_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path

# %%

if MAIN:
    refresh_tree_default_methods()
    tree_persist_completed = tree_helper_run_native_persist_completed
if MAIN:
    refresh_tree_default_methods()
    tree_save_payload = tree_helper_run_native_save_payload
if MAIN:
    refresh_tree_default_methods()
    tree_write_before_audit = tree_helper_run_native_write_before_audit
if MAIN:
    refresh_tree_default_methods()
    tree_write_notebook_summary = tree_helper_run_native_write_notebook_summary

if MAIN:
    refresh_tree_default_methods()
    OUTPUT_ROOT = Path(
        os.environ.get("ARENA_TREESHAP_OUTPUT_DIRECTORY", canonical_dir / "artifacts" / "tree")
    )
if MAIN:
    refresh_tree_default_methods()
    NATIVE_DIR = OUTPUT_ROOT / "native"
if MAIN:
    refresh_tree_default_methods()
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)

if MAIN:
    refresh_tree_default_methods()
    tree_train = tree_load_split("train")
if MAIN:
    refresh_tree_default_methods()
    tree_explain = tree_load_split("explain")
if MAIN:
    refresh_tree_default_methods()
    tree_fitted = tree_fit_models(tree_train)
if MAIN:
    refresh_tree_default_methods()
    tree_prepared = tree_prepare_study(tree_fitted, tree_train, tree_explain)
if MAIN:
    refresh_tree_default_methods()
    tree_save_payload(NATIVE_DIR, "fitted.pt", tree_fitted)
if MAIN:
    refresh_tree_default_methods()
    tree_save_payload(NATIVE_DIR, "prepared.pt", tree_prepared)
if MAIN:
    refresh_tree_default_methods()
    tree_write_before_audit(NATIVE_DIR, tree_prepared)
if MAIN:
    refresh_tree_default_methods()
    print("Prepared rankings saved before audit load:", not tree_prepared["audit_used_for_selection"])

# %%

if MAIN:
    refresh_tree_default_methods()
    tree_audit = tree_load_split("audit")
if MAIN:
    refresh_tree_default_methods()
    tree_evaluated = tree_evaluate_study(tree_fitted, tree_prepared, tree_train, tree_explain, tree_audit)
if MAIN:
    refresh_tree_default_methods()
    print("Models evaluated:", len(tree_evaluated["performance"]))
if MAIN:
    refresh_tree_default_methods()
    print("Audit rows:", len(tree_evaluated["audit_ids"]))

# %%

if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(tree_plots.plot_performance(tree_evaluated), "07_audit_performance")

# %%

tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL = 2e-5

# %%

tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL = 2e-6

# %%

if MAIN:
    tree_helper_treeshap_reference_EXACT_KEYS = tuple(f"{family}_clean_0" for family in ("tree", "forest", "boost"))

# %%

if MAIN:
    tree_helper_treeshap_reference_MODEL_KEYS = tuple(
        f"{family}_{target}_{seed_index}"
        for family in ("tree", "forest", "boost")
        for seed_index in range(3)
        for target in ("clean", "shuffled")
    )

# %%

if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__interventional_game_reference(predict, x, background):
    """Enumerate all intact-donor games for the rounded-threshold proof."""
    masks = np.array(
        [[bool(mask & (1 << feature)) for feature in range(8)] for mask in range(256)]
    )
    values = np.empty((len(x), len(masks)), dtype=np.float64)
    for start in range(0, len(x), 8):
        query = x[start : start + 8]
        hybrids = np.where(
            masks[None, :, None, :],
            query[:, None, None, :],
            background[None, None, :, :],
        )
        prediction = predict(hybrids.reshape(-1, 8))
        values[start : start + len(query)] = prediction.reshape(
            len(query), 256, 64
        ).mean(axis=2)
    return values

# %%

if MAIN:
    import hashlib
if MAIN:
    import pickle

def tree_helper_treeshap_reference__model_hash(model) -> str:
    return hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest()

# %%

tree_helper_treeshap_reference_ATOL = tree_helper_treeshap_reference_RTOL = 1e-8

# %%

if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    return float(np.max(np.abs(actual - expected), initial=0.0))

# %%

if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__require_close(name, actual, expected, *, atol=tree_helper_treeshap_reference_ATOL, rtol=tree_helper_treeshap_reference_RTOL) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    try:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(f"{name} failed full-value parity.\n{error}") from error
    return tree_helper_treeshap_reference__max_abs(actual, expected)

# %%

if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__floor_float32_thresholds(original: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nearest = np.asarray(original, dtype=np.float32)
    rounded_up = nearest.astype(np.float64) > np.asarray(original, dtype=np.float64)
    corrected = nearest.copy()
    corrected[rounded_up] = np.nextafter(
        corrected[rounded_up], np.float32(-np.inf)
    )
    return nearest, corrected, rounded_up

# %%

if MAIN:
    import copy
if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__threshold_copies(raw_explainer, sklearn_model, all_source_x):
    original = raw_explainer.model.thresholds.copy()
    internal = (raw_explainer.model.children_left >= 0) & (raw_explainer.model.features >= 0)
    nearest, corrected_values, rounded_up = tree_helper_treeshap_reference__floor_float32_thresholds(original)
    change = internal & rounded_up

    corrected = copy.deepcopy(raw_explainer)
    corrected.model.thresholds[internal] = corrected_values[internal].astype(np.float64)
    corrected.expected_value = corrected.model.predict(corrected.data).mean(0)

    nearest_model = copy.deepcopy(raw_explainer)
    nearest_model.model.thresholds[internal] = nearest[internal].astype(np.float64)
    nearest_model.expected_value = nearest_model.model.predict(nearest_model.data).mean(0)

    positions = np.argwhere(change)
    trees = positions[:, 0] if len(positions) else np.empty(0, dtype=np.int64)
    nodes = positions[:, 1] if len(positions) else np.empty(0, dtype=np.int64)
    features = raw_explainer.model.features[change].astype(np.int64)
    changed_original = original[change]
    changed_nearest = nearest[change]
    changed_corrected = corrected_values[change]
    lower = np.nextafter(changed_corrected, np.float32(-np.inf))
    upper = np.nextafter(changed_corrected, np.float32(np.inf))
    neighbors = np.column_stack([lower, changed_corrected, upper]).astype(np.float32)
    original_routes = neighbors.astype(np.float64) <= changed_original[:, None]
    corrected_routes = neighbors <= changed_corrected[:, None]
    raw_routes = neighbors <= changed_nearest[:, None]
    expected_routes = np.tile(np.array([True, True, False]), (len(positions), 1))
    if not np.array_equal(original_routes, expected_routes):
        raise AssertionError("Float32 boundary neighbors do not bracket original thresholds.")
    if not np.array_equal(corrected_routes, expected_routes):
        raise AssertionError("Corrected threshold boundary proof failed.")

    source_mismatches_raw = np.zeros(len(positions), dtype=np.int64)
    for index, (feature, threshold, raw_threshold, fixed_threshold) in enumerate(
        zip(features, changed_original, changed_nearest, changed_corrected, strict=True)
    ):
        values = all_source_x[:, feature]
        original_route = values.astype(np.float64) <= threshold
        corrected_route = values <= fixed_threshold
        if not np.array_equal(original_route, corrected_route):
            raise AssertionError("Correction changed an original sklearn route on source data.")
        source_mismatches_raw[index] = np.count_nonzero(original_route != (values <= raw_threshold))

    corrected_source_route_disagreements = 0
    for tree_index, node_index in np.argwhere(internal):
        feature = raw_explainer.model.features[tree_index, node_index]
        values = all_source_x[:, feature]
        original_route = values.astype(np.float64) <= original[tree_index, node_index]
        corrected_route = values <= corrected_values[tree_index, node_index]
        corrected_source_route_disagreements += int(
            np.count_nonzero(original_route != corrected_route)
        )
    if corrected_source_route_disagreements:
        raise AssertionError("Floored thresholds changed original routes on source data.")

    corrected_prediction = corrected.model.predict(all_source_x)
    sklearn_prediction = sklearn_model.predict(all_source_x)
    corrected_prediction_error = tree_helper_treeshap_reference__require_close(
        "corrected explainer routes on all 1,030 source rows",
        corrected_prediction,
        sklearn_prediction,
    )
    nearest_prediction = nearest_model.model.predict(all_source_x)
    return corrected, nearest_model, {
        "component_tree_count": int(len(original)),
        "internal_threshold_count": int(np.count_nonzero(internal)),
        "rounded_up_threshold_count": int(np.count_nonzero(change)),
        "tree_index": trees,
        "node_index": nodes,
        "feature": features,
        "original": changed_original,
        "raw_nearest_float32": changed_nearest,
        "corrected_floor_float32": changed_corrected,
        "neighbor_values": neighbors,
        "original_neighbor_routes": original_routes,
        "raw_neighbor_routes": raw_routes,
        "corrected_neighbor_routes": corrected_routes,
        "source_raw_route_mismatch_count": source_mismatches_raw,
        "all_source_corrected_route_disagreement_count": corrected_source_route_disagreements,
        "all_source_corrected_prediction_max_abs_error": corrected_prediction_error,
        "all_source_raw_rounded_prediction_max_abs_error": tree_helper_treeshap_reference__max_abs(
            nearest_prediction, sklearn_prediction
        ),
        "all_source_raw_rounded_prediction_mismatch_count": int(
            np.count_nonzero(~np.isclose(nearest_prediction, sklearn_prediction, atol=tree_helper_treeshap_reference_ATOL, rtol=tree_helper_treeshap_reference_RTOL))
        ),
    }

# %%

if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__compiled_reference(model, all_x, background, all_source_x, *, exact_raw_game=False):
    import shap

    model_hash_before = tree_helper_treeshap_reference__model_hash(model)
    path_explainer = shap.TreeExplainer(
        model, model_output="raw", feature_perturbation="tree_path_dependent"
    )
    if path_explainer.feature_perturbation != "tree_path_dependent":
        raise AssertionError("Path reference did not retain the explicit perturbation mode.")
    phi_path = np.asarray(
        path_explainer.shap_values(all_x, approximate=False, check_additivity=True),
        dtype=np.float64,
    )
    base_path = np.full(len(all_x), np.asarray(path_explainer.expected_value, dtype=np.float64).item())

    raw = shap.TreeExplainer(
        model,
        data=background,
        model_output="raw",
        feature_perturbation="interventional",
    )
    if raw.feature_perturbation != "interventional" or raw.model_output != "raw":
        raise AssertionError("Interventional reference changed the explicit mode/output.")
    if raw.data.shape != (64, 8) or raw.data.dtype != np.float32:
        raise AssertionError("TreeExplainer did not retain the complete float32 background.")
    if not np.array_equal(raw.data, background):
        raise AssertionError("TreeExplainer altered or subsampled the 64-row background.")
    phi_raw = np.asarray(
        # Preserve the unmodified library output even where SHAP's own loose
        # additivity guard raises; the strict residual is checked and returned below.
        raw.shap_values(all_x, approximate=False, check_additivity=False), dtype=np.float64
    )
    base_raw = np.full(len(all_x), np.asarray(raw.expected_value, dtype=np.float64).item())

    corrected, nearest_model, threshold_audit = tree_helper_treeshap_reference__threshold_copies(
        raw, model, all_source_x
    )
    if not np.array_equal(corrected.data, background) or corrected.data.shape != (64, 8):
        raise AssertionError("Corrected explainer must still use every frozen donor.")
    phi_corrected = np.asarray(
        corrected.shap_values(all_x, approximate=False, check_additivity=True),
        dtype=np.float64,
    )
    base_corrected = np.full(len(all_x), np.asarray(corrected.expected_value, dtype=np.float64).item())
    nearest_prediction = nearest_model.model.predict(all_x)
    nearest_base = np.full(len(all_x), np.asarray(nearest_model.expected_value, dtype=np.float64).item())
    raw_rounding_error = tree_helper_treeshap_reference__require_close(
        "unmodified C++ totals against nearest-float32 threshold model",
        nearest_base + phi_raw.sum(axis=1),
        nearest_prediction,
        atol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL,
        rtol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL,
    )
    model_hash_after = tree_helper_treeshap_reference__model_hash(model)
    if model_hash_before != model_hash_after:
        raise AssertionError("Reference construction mutated the fitted sklearn model.")
    result = {
        "prediction": np.asarray(model.predict(all_x), dtype=np.float64),
        "base_path": base_path,
        "phi_path": phi_path,
        "base_interventional_raw": base_raw,
        "phi_interventional_raw": phi_raw,
        "raw_nearest_threshold_prediction": nearest_prediction,
        "raw_nearest_threshold_base": nearest_base,
        "base_interventional_corrected": base_corrected,
        "phi_interventional_corrected": phi_corrected,
        "raw_nearest_threshold_additivity_max_abs_error": raw_rounding_error,
        "threshold_audit": threshold_audit,
        "sklearn_model_sha256_before": model_hash_before,
        "sklearn_model_sha256_after": model_hash_after,
    }
    if exact_raw_game:
        result["game_interventional_raw_nearest"] = tree_helper_treeshap_reference__interventional_game_reference(
            nearest_model.model.predict, all_x, background
        )
    return result

# %%

if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__isclose_rows(actual, expected, *, atol, rtol) -> np.ndarray:
    close = np.isclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol)
    return close if close.ndim == 1 else close.reshape(len(close), -1).all(axis=1)

# %%

tree_helper_treeshap_reference_ADDENDUM_SHA256 = "2e639c56060d45131a093caeb838a8896399a2d6e89cba9fe477bcd9b816023e"

# %%

if MAIN:
    from pathlib import Path

if MAIN:
    tree_helper_treeshap_reference_HERE = tree_root_dir / 'chapter16_shapley_attribution_baselines' / "exercises" / 'part13_treeshap'

# %%

if MAIN:
    from pathlib import Path
if MAIN:
    import hashlib

def tree_helper_treeshap_reference__sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

# %%

if MAIN:
    from pathlib import Path
if MAIN:
    import json
if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__reference_environment() -> dict:
    import shap
    import sklearn
    from shap.explainers import _tree

    manifest = json.loads((tree_helper_treeshap_reference_HERE / "author_reference.json").read_text())
    # The author reference is pinned by its SHAP version and the source hashes below. numpy and
    # scikit-learn are recorded, not required, so environment upgrades keep this check running.
    expected_versions = {"shap": "0.49.1"}
    actual_versions = {
        "numpy": np.__version__, "scikit_learn": sklearn.__version__, "shap": shap.__version__
    }
    if actual_versions["shap"] != expected_versions["shap"]:
        raise ValueError(f"Reference version mismatch: {actual_versions} != {expected_versions}")
    if getattr(_tree, "_cext", None) is None or not hasattr(_tree._cext, "dense_tree_shap"):
        raise RuntimeError("Installed SHAP C++ dense_tree_shap is unavailable.")

    # The installed SHAP source is pinned below. The author's cached source copies and paper PDF
    # are not needed to run the lesson, so they are not checked here.
    installed_tree_hash = tree_helper_treeshap_reference__sha256(Path(_tree.__file__))
    if installed_tree_hash != manifest["sources"]["explainers/_tree.py"]:
        raise ValueError("Installed SHAP TreeExplainer Python source is not the pinned source.")
    addendum_hash = tree_helper_treeshap_reference__sha256(tree_helper_treeshap_reference_HERE / "reference_boundary_addendum.md")
    if addendum_hash != tree_helper_treeshap_reference_ADDENDUM_SHA256:
        raise ValueError("The post-freeze threshold addendum changed.")
    return {
        "versions": actual_versions,
        "installed_tree_source_sha256": installed_tree_hash,
        "boundary_addendum_sha256": addendum_hash,
        "cext_module": str(Path(_tree._cext.__file__).resolve()),
    }

# %%

if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__validate_block(name, block, rows):
    if set(block) != set(tree_helper_treeshap_reference_MODEL_KEYS):
        raise ValueError(f"{name} must contain exactly the 18 frozen model keys.")
    ordinary = {
        "prediction": (rows,),
        "base_path": (rows,),
        "phi_path": (rows, 8),
        "base_interventional": (rows,),
        "phi_interventional": (rows, 8),
        "phi_saabas": (rows, 8),
    }
    for key in tree_helper_treeshap_reference_MODEL_KEYS:
        item = block[key]
        for field, shape in ordinary.items():
            if field not in item or np.asarray(item[field]).shape != shape:
                raise ValueError(f"{name}[{key!r}][{field!r}] must have shape {shape}.")
        for field in ("game_path", "game_interventional"):
            present = field in item
            if key in tree_helper_treeshap_reference_EXACT_KEYS and (not present or np.asarray(item[field]).shape != (rows, 256)):
                raise ValueError(f"{name}[{key!r}][{field!r}] must have shape {(rows, 256)}.")
            if key not in tree_helper_treeshap_reference_EXACT_KEYS and present:
                raise ValueError(f"{name}[{key!r}] unexpectedly contains {field}.")

# %%

if MAIN:
    import hashlib
if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__background_indices(train_ids: np.ndarray) -> np.ndarray:
    return np.array(
        sorted(
            range(len(train_ids)),
            key=lambda i: hashlib.sha256(
                f"arena16.13/background/161360/{int(train_ids[i])}".encode()
            ).digest(),
        ),
        dtype=np.int64,
    )[:64]

# %%

if MAIN:
    import json
if MAIN:
    import numpy as np
if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.tree_treeshap_data import source_data as tree_plumbing_treeshap_data_source_data

def tree_helper_treeshap_reference__frozen_identity():
    manifest = json.loads((tree_helper_treeshap_reference_HERE / "data_manifest.json").read_text())
    data = tree_plumbing_treeshap_data_source_data()
    ids = {
        name: np.asarray(manifest["splits"][name], dtype=np.int64)
        for name in ("train", "explain", "audit")
    }
    return data, ids

# %%

if MAIN:
    import numpy as np

def tree_helper_treeshap_reference__validate_identity(fitted, prepared, evaluated, explain, audit):
    data, frozen_ids = tree_helper_treeshap_reference__frozen_identity()
    supplied = {
        "train": np.asarray(fitted["train_ids"]),
        "explain": np.asarray(explain["ids"]),
        "audit": np.asarray(audit["ids"]),
    }
    for name in ("train", "explain", "audit"):
        if not np.array_equal(supplied[name], frozen_ids[name]):
            raise ValueError(f"{name} row identity/order does not match the frozen manifest.")
    if not np.array_equal(np.asarray(prepared["explain_ids"]), frozen_ids["explain"]):
        raise ValueError("prepared explanation row order does not match the frozen manifest.")
    if not np.array_equal(np.asarray(evaluated["audit_ids"]), frozen_ids["audit"]):
        raise ValueError("evaluated audit row order does not match the frozen manifest.")

    for name, block in (("explain", explain), ("audit", audit)):
        expected = data[frozen_ids[name]]
        if not np.array_equal(np.asarray(block["x"]), expected[:, :8]):
            raise ValueError(f"{name} inputs do not match their frozen source-row order.")
        if not np.array_equal(np.asarray(block["y"]), expected[:, 8]):
            raise ValueError(f"{name} targets do not match their frozen source-row order.")
    if not np.array_equal(np.asarray(evaluated["audit_targets"]), data[frozen_ids["audit"], 8]):
        raise ValueError("evaluated audit targets do not match the frozen source rows.")

    order = tree_helper_treeshap_reference__background_indices(frozen_ids["train"])
    expected_background_ids = frozen_ids["train"][order]
    if not np.array_equal(np.asarray(prepared["background_ids"]), expected_background_ids):
        raise ValueError("Primary background IDs/order are not the frozen 64 donors.")
    expected_background = data[expected_background_ids, :8].astype(np.float32)
    actual_background = np.asarray(prepared["background"], dtype=np.float32)
    if actual_background.shape != (64, 8) or not np.array_equal(
        actual_background, expected_background
    ):
        raise ValueError("Primary background values are not all 64 frozen float32 donors.")
    return (
        data[:, :8].astype(np.float32),
        np.asarray(explain["x"], dtype=np.float32),
        np.asarray(audit["x"], dtype=np.float32),
        expected_background,
        frozen_ids,
    )

# %%

if MAIN:
    import math
if MAIN:
    import numpy as np

def tree_helper_treeshap_reference_subset_reference(game: np.ndarray) -> np.ndarray:
    """Exact all-subset contraction, independently written for actual games."""
    game = np.asarray(game, dtype=np.float64)
    coalition_count = game.shape[-1]
    d = coalition_count.bit_length() - 1
    if d < 1 or coalition_count != 2**d:
        raise ValueError("The final game axis must contain all 2**D coalitions.")
    result = np.zeros((*game.shape[:-1], d), dtype=np.float64)
    for feature in range(d):
        for mask in range(coalition_count):
            if mask & (1 << feature):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(d - size - 1) / math.factorial(d)
            result[..., feature] += weight * (
                game[..., mask | (1 << feature)] - game[..., mask]
            )
    return result

# %%

if MAIN:
    import numpy as np

def tree_helper_treeshap_reference_check_reference(fitted, prepared, evaluated, explain, audit):
    """Check every fitted model and row, returning all numeric reference arrays.

    The unmodified interventional arrays are evidence of a pinned library
    failure.  Only the separately copied threshold-corrected explainer is
    required to match the actual finite-background game.
    """
    environment = tree_helper_treeshap_reference__reference_environment()
    if set(fitted["models"]) != set(tree_helper_treeshap_reference_MODEL_KEYS):
        raise ValueError("fitted['models'] must contain exactly the 18 frozen models.")
    tree_helper_treeshap_reference__validate_block("prepared explanation", prepared["explanation"], 153)
    tree_helper_treeshap_reference__validate_block("evaluated audit", evaluated["audit"], 119)
    all_source_x, explain_x, audit_x, background, frozen_ids = tree_helper_treeshap_reference__validate_identity(
        fitted, prepared, evaluated, explain, audit
    )
    all_x = np.vstack([explain_x, audit_x])
    split_slices = {"explanation": slice(0, 153), "audit": slice(153, 272)}
    stored_blocks = {
        "explanation": prepared["explanation"],
        "audit": evaluated["audit"],
    }

    references = {"explanation": {}, "audit": {}}
    threshold_audit = {}
    model_hashes = {}
    per_model_max_errors = {}
    max_errors = {
        "prediction": 0.0,
        "path_base": 0.0,
        "path_phi": 0.0,
        "interventional_raw_base": 0.0,
        "interventional_raw_phi": 0.0,
        "interventional_raw_prediction_residual": 0.0,
        "interventional_raw_nearest_threshold_additivity": 0.0,
        "interventional_corrected_base": 0.0,
        "interventional_corrected_phi": 0.0,
        "interventional_corrected_prediction_residual": 0.0,
        "exact_path_phi": 0.0,
        "exact_interventional_phi": 0.0,
        "exact_interventional_raw_phi": 0.0,
        "exact_raw_rounded_interventional_phi": 0.0,
    }
    affected = {name: [] for name in ("split", "model_index", "row_index", "source_id")}
    affected.update(
        prediction_residual=[], phi_max_abs_error=[],
        prediction_failed=[], phi_failed=[], base_failed=[],
    )

    for model_index, key in enumerate(tree_helper_treeshap_reference_MODEL_KEYS):
        item = fitted["models"][key]
        if "sklearn" not in item:
            raise ValueError(f"Live fitted model {key} lacks its sklearn estimator.")
        compiled = tree_helper_treeshap_reference__compiled_reference(
            item["sklearn"], all_x, background, all_source_x,
            exact_raw_game=key in tree_helper_treeshap_reference_EXACT_KEYS,
        )
        threshold_audit[key] = compiled.pop("threshold_audit")
        model_hashes[key] = {
            "before": compiled.pop("sklearn_model_sha256_before"),
            "after": compiled.pop("sklearn_model_sha256_after"),
        }
        max_errors["interventional_raw_nearest_threshold_additivity"] = max(
            max_errors["interventional_raw_nearest_threshold_additivity"],
            compiled.pop("raw_nearest_threshold_additivity_max_abs_error"),
        )
        per_model_max_errors[key] = {}

        for split, section in split_slices.items():
            stored = stored_blocks[split][key]
            ref = {
                field: np.asarray(values[section]).copy()
                for field, values in compiled.items()
            }
            ref["raw_prediction_residual"] = (
                ref["base_interventional_raw"]
                + ref["phi_interventional_raw"].sum(axis=1)
                - ref["prediction"]
            )
            ref["corrected_prediction_residual"] = (
                ref["base_interventional_corrected"]
                + ref["phi_interventional_corrected"].sum(axis=1)
                - ref["prediction"]
            )
            references[split][key] = ref

            errors = {
                "prediction": tree_helper_treeshap_reference__require_close(
                    f"{split} {key} prediction", stored["prediction"], ref["prediction"]
                ),
                "path_base": tree_helper_treeshap_reference__require_close(
                    f"{split} {key} path base", stored["base_path"], ref["base_path"]
                ),
                "path_phi": tree_helper_treeshap_reference__require_close(
                    f"{split} {key} path local values", stored["phi_path"], ref["phi_path"]
                ),
                "interventional_corrected_base": tree_helper_treeshap_reference__require_close(
                    f"{split} {key} corrected interventional base",
                    stored["base_interventional"],
                    ref["base_interventional_corrected"],
                    atol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL,
                    rtol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL,
                ),
                "interventional_corrected_phi": tree_helper_treeshap_reference__require_close(
                    f"{split} {key} corrected interventional local values",
                    stored["phi_interventional"],
                    ref["phi_interventional_corrected"],
                    atol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL,
                    rtol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL,
                ),
                "interventional_raw_base": tree_helper_treeshap_reference__max_abs(
                    stored["base_interventional"], ref["base_interventional_raw"]
                ),
                "interventional_raw_phi": tree_helper_treeshap_reference__max_abs(
                    stored["phi_interventional"], ref["phi_interventional_raw"]
                ),
                "interventional_raw_prediction_residual": float(
                    np.max(np.abs(ref["raw_prediction_residual"]), initial=0.0)
                ),
                "interventional_corrected_prediction_residual": tree_helper_treeshap_reference__require_close(
                    f"{split} {key} corrected interventional additivity",
                    ref["base_interventional_corrected"]
                    + ref["phi_interventional_corrected"].sum(axis=1),
                    ref["prediction"],
                    atol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL,
                    rtol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL,
                ),
            }
            tree_helper_treeshap_reference__require_close(
                f"{split} {key} path additivity",
                ref["base_path"] + ref["phi_path"].sum(axis=1),
                ref["prediction"],
            )
            tree_helper_treeshap_reference__require_close(
                f"{split} {key} Saabas additivity",
                np.asarray(stored["base_path"]) + np.asarray(stored["phi_saabas"]).sum(axis=1),
                ref["prediction"],
            )
            per_model_max_errors[key][split] = errors
            for name, value in errors.items():
                max_errors[name] = max(max_errors[name], value)

            raw_phi_close = tree_helper_treeshap_reference__isclose_rows(
                ref["phi_interventional_raw"], stored["phi_interventional"],
                atol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL,
                rtol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL,
            )
            raw_base_close = tree_helper_treeshap_reference__isclose_rows(
                ref["base_interventional_raw"], stored["base_interventional"],
                atol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL,
                rtol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL,
            )
            raw_prediction_close = tree_helper_treeshap_reference__isclose_rows(
                ref["base_interventional_raw"] + ref["phi_interventional_raw"].sum(axis=1),
                ref["prediction"],
                atol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL,
                rtol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL,
            )
            raw_rows_close = raw_phi_close & raw_base_close & raw_prediction_close
            ids = frozen_ids["explain" if split == "explanation" else "audit"]
            for row in np.flatnonzero(~raw_rows_close):
                affected["split"].append(0 if split == "explanation" else 1)
                affected["model_index"].append(model_index)
                affected["row_index"].append(int(row))
                affected["source_id"].append(int(ids[row]))
                affected["prediction_residual"].append(float(ref["raw_prediction_residual"][row]))
                affected["phi_max_abs_error"].append(
                    float(np.max(np.abs(
                        ref["phi_interventional_raw"][row]
                        - np.asarray(stored["phi_interventional"])[row]
                    )))
                )
                affected["prediction_failed"].append(not raw_prediction_close[row])
                affected["phi_failed"].append(not raw_phi_close[row])
                affected["base_failed"].append(not raw_base_close[row])

    exact_max = {
        "path_phi": 0.0,
        "interventional_phi": 0.0,
        "interventional_raw_phi": 0.0,
        "raw_rounded_interventional_phi": 0.0,
        "path_base": 0.0,
        "interventional_base": 0.0,
        "prediction": 0.0,
    }
    for split in ("explanation", "audit"):
        for key in tree_helper_treeshap_reference_EXACT_KEYS:
            stored = stored_blocks[split][key]
            ref = references[split][key]
            exact_path = tree_helper_treeshap_reference_subset_reference(stored["game_path"])
            exact_interventional = tree_helper_treeshap_reference_subset_reference(stored["game_interventional"])
            exact_max["path_base"] = max(
                exact_max["path_base"],
                tree_helper_treeshap_reference__require_close(
                    f"{split} {key} exact path empty coalition",
                    stored["game_path"][:, 0], stored["base_path"],
                ),
            )
            exact_max["interventional_base"] = max(
                exact_max["interventional_base"],
                tree_helper_treeshap_reference__require_close(
                    f"{split} {key} exact interventional empty coalition",
                    stored["game_interventional"][:, 0], stored["base_interventional"],
                ),
            )
            for game_name in ("game_path", "game_interventional"):
                exact_max["prediction"] = max(
                    exact_max["prediction"],
                    tree_helper_treeshap_reference__require_close(
                        f"{split} {key} {game_name} full coalition",
                        stored[game_name][:, -1], stored["prediction"],
                    ),
                )
            exact_max["path_phi"] = max(
                exact_max["path_phi"],
                tree_helper_treeshap_reference__require_close(
                    f"{split} {key} exhaustive path values", exact_path, stored["phi_path"]
                ),
                tree_helper_treeshap_reference__require_close(
                    f"{split} {key} compiled path versus exhaustive", ref["phi_path"], exact_path
                ),
            )
            exact_max["interventional_phi"] = max(
                exact_max["interventional_phi"],
                tree_helper_treeshap_reference__require_close(
                    f"{split} {key} exhaustive interventional values",
                    exact_interventional, stored["phi_interventional"],
                ),
                tree_helper_treeshap_reference__require_close(
                    f"{split} {key} corrected compiled versus exhaustive",
                    ref["phi_interventional_corrected"], exact_interventional,
                    atol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL,
                    rtol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL,
                ),
            )
            exact_max["interventional_raw_phi"] = max(
                exact_max["interventional_raw_phi"],
                tree_helper_treeshap_reference__max_abs(ref["phi_interventional_raw"], exact_interventional),
            )
            raw_rounded_exact = tree_helper_treeshap_reference_subset_reference(ref["game_interventional_raw_nearest"])
            exact_max["raw_rounded_interventional_phi"] = max(
                exact_max["raw_rounded_interventional_phi"],
                tree_helper_treeshap_reference__require_close(
                    f"{split} {key} raw C++ versus its rounded-threshold exact game",
                    ref["phi_interventional_raw"], raw_rounded_exact,
                    atol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL,
                    rtol=tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL,
                ),
            )
            tree_helper_treeshap_reference__require_close(
                f"{split} {key} raw rounded game empty coalition",
                ref["game_interventional_raw_nearest"][:, 0],
                ref["raw_nearest_threshold_base"],
            )
            tree_helper_treeshap_reference__require_close(
                f"{split} {key} raw rounded game full coalition",
                ref["game_interventional_raw_nearest"][:, -1],
                ref["raw_nearest_threshold_prediction"],
            )
    max_errors["exact_path_phi"] = exact_max["path_phi"]
    max_errors["exact_interventional_phi"] = exact_max["interventional_phi"]
    max_errors["exact_interventional_raw_phi"] = exact_max["interventional_raw_phi"]
    max_errors["exact_raw_rounded_interventional_phi"] = exact_max[
        "raw_rounded_interventional_phi"
    ]

    affected_arrays = {
        name: np.asarray(
            values,
            dtype=(
                np.float64
                if name in ("prediction_residual", "phi_max_abs_error")
                else np.bool_
                if name.endswith("_failed")
                else np.int64
            ),
        )
        for name, values in affected.items()
    }
    unmodified_passed = len(affected_arrays["row_index"]) == 0
    if unmodified_passed:
        raise AssertionError(
            "The pinned SHAP 0.49.1 raw interventional threshold failure was not reproduced."
        )
    threshold_counts = {
        "component_trees": int(
            sum(v["component_tree_count"] for v in threshold_audit.values())
        ),
        "internal_thresholds": int(
            sum(v["internal_threshold_count"] for v in threshold_audit.values())
        ),
        "corrected_thresholds": int(
            sum(v["rounded_up_threshold_count"] for v in threshold_audit.values())
        ),
        "raw_source_route_disagreements": int(
            sum(v["source_raw_route_mismatch_count"].sum() for v in threshold_audit.values())
        ),
        "corrected_source_route_disagreements": int(
            sum(
                v["all_source_corrected_route_disagreement_count"]
                for v in threshold_audit.values()
            )
        ),
    }
    if tuple(threshold_counts[name] for name in (
        "component_trees", "internal_thresholds", "corrected_thresholds",
        "raw_source_route_disagreements", "corrected_source_route_disagreements",
    )) != (486, 5480, 941, 217, 0):
        raise AssertionError(f"Frozen threshold audit counts changed: {threshold_counts}")
    return {
        "status": "PASS_WITH_EXPECTED_RAW_REFERENCE_FAILURE",
        "passes": {
            "row_identity": True,
            "background_identity_all_64": True,
            "prediction": True,
            "path_compiled": True,
            "interventional_unmodified": False,
            "interventional_threshold_corrected": True,
            "exhaustive_path": True,
            "exhaustive_interventional": True,
            "sklearn_models_unchanged": True,
        },
        "unmodified_interventional_passed": False,
        "threshold_corrected_interventional_passed": True,
        "model_keys": tree_helper_treeshap_reference_MODEL_KEYS,
        "counts": {
            "models": 18,
            "explanation_rows": 153,
            "audit_rows": 119,
            "model_rows": 18 * (153 + 119),
            "local_values_per_game": 18 * (153 + 119) * 8,
            "exact_models": 3,
            "exact_model_rows": 3 * (153 + 119),
            "background_rows": 64,
            "raw_affected_model_rows": len(affected_arrays["row_index"]),
            "raw_prediction_affected_model_rows": int(
                affected_arrays["prediction_failed"].sum()
            ),
            "raw_phi_affected_model_rows": int(affected_arrays["phi_failed"].sum()),
            **threshold_counts,
        },
        "explain_ids": frozen_ids["explain"].copy(),
        "audit_ids": frozen_ids["audit"].copy(),
        "background_ids": np.asarray(prepared["background_ids"]).copy(),
        "background_float32": background.copy(),
        "explanation": references["explanation"],
        "audit": references["audit"],
        "raw_affected_rows": affected_arrays,
        "threshold_audit": threshold_audit,
        "sklearn_model_hashes": model_hashes,
        "max_errors": max_errors,
        "exact_max_errors": exact_max,
        "per_model_max_errors": per_model_max_errors,
        "reference_environment": environment,
    }

# %%

if MAIN:
    refresh_tree_default_methods()
    check_reference = tree_helper_treeshap_reference_check_reference

if MAIN:
    refresh_tree_default_methods()
    tree_reference = check_reference(tree_fitted, tree_prepared, tree_evaluated, tree_explain, tree_audit)
if MAIN:
    refresh_tree_default_methods()
    print("Reference checks:", tree_reference["passes"])
if MAIN:
    refresh_tree_default_methods()
    print("Raw compiled parity passed:", tree_reference["unmodified_interventional_passed"])
if MAIN:
    refresh_tree_default_methods()
    print(
        "Threshold-compatible parity passed:",
        tree_reference["threshold_corrected_interventional_passed"],
    )
if MAIN:
    refresh_tree_default_methods()
    print("Maximum absolute errors by check:")
if MAIN:
    refresh_tree_default_methods()
    for name, error in tree_reference["max_errors"].items():
        print(f"  {name:28s} {error:.3e}")
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(tree_plots.plot_reference_parity(tree_reference), "17_reference_parity")

# %%

if MAIN:
    refresh_tree_default_methods()
    default_play = tree_native_play(
        tree_fitted, tree_train, tree_audit,
        model_key="forest_clean_0", row=0, size=64, game="interventional",
    )
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(
        tree_plots.plot_waterfall(
            default_play,
            tree_FEATURE_NAMES,
            title="Audit recipe 0: interventional TreeSHAP",
        ),
        "08_prediction_waterfall",
    )
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(
        tree_plots.plot_attribution_heatmaps(
            tree_prepared["explanation"]["forest_clean_0"],
            tree_FEATURE_NAMES,
            row_labels=tree_prepared["explain_ids"],
        ),
        "09_attribution_heatmaps",
    )

# %%

if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(
        tree_plots.plot_model_variation(tree_prepared["explanation"], tree_FEATURE_NAMES),
        "10_model_seed_variation",
    )
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(
        tree_plots.plot_background_sensitivity(tree_prepared, tree_evaluated, tree_FEATURE_NAMES),
        "11_background_sensitivity",
    )
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(
        tree_plots.plot_deletion_curves(tree_evaluated),
        "12_deletion_curves",
    )

# %%

if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(
        tree_plots.plot_runtime_dummy_features(tree_diagnostics["timing"]),
        "13_runtime_dummy_features",
    )
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(
        tree_plots.plot_runtime_depth(tree_diagnostics["timing"]),
        "14_runtime_depth",
    )

# %%

row = 0
model_key = "forest_clean_0"
game = "interventional"  # choose "interventional" or "path"
size = 64                 # used only by the interventional game

if MAIN:
    refresh_tree_default_methods()
    tree_real_play = tree_native_play(
        tree_fitted, tree_train, tree_audit,
        model_key=model_key, row=row, size=size, game=game,
    )
if MAIN:
    refresh_tree_default_methods()
    tree_show_figure(tree_plots.plot_real_recipe_play(tree_real_play, tree_FEATURE_NAMES), "16_real_recipe_play")

# %%

# Persist the complete visible run. All scientific computations occurred above.
if MAIN:
    refresh_tree_default_methods()
    tree_summary = tree_persist_completed(
        NATIVE_DIR, tree_fitted, tree_prepared, tree_evaluated, tree_diagnostics, tree_reference, default_play
    )
if MAIN:
    refresh_tree_default_methods()
    tree_write_notebook_summary(OUTPUT_ROOT, NATIVE_DIR, tree_summary)
if MAIN:
    refresh_tree_default_methods()
    print("Saved notebook-native payloads to", NATIVE_DIR)

# %%

if MAIN:
    import itertools
if MAIN:
    import math
if MAIN:
    import os
if MAIN:
    import sys
if MAIN:
    from pathlib import Path
if MAIN:
    from types import SimpleNamespace

if MAIN:
    import matplotlib.pyplot as plt
if MAIN:
    import numpy as np
if MAIN:
    import torch
if MAIN:
    import torch as t
if MAIN:
    from IPython.display import display
if MAIN:
    from torch import Tensor, nn

if MAIN:
    torch.set_default_dtype(torch.float64)

deep_chapter = "chapter16_shapley_attribution_baselines"
deep_section = "part14_deepshap"
if MAIN:
    deep_root_dir = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / deep_chapter).exists())
if MAIN:
    deep_exercises_dir = deep_root_dir / deep_chapter / "exercises"
if MAIN:
    deep_section_dir = deep_exercises_dir / deep_section
if MAIN:
    if str(deep_root_dir) not in sys.path:
        sys.path.append(str(deep_root_dir))
    if str(deep_exercises_dir) not in sys.path:
        sys.path.append(str(deep_exercises_dir))

if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import deep_deepshap_methods_tests as deep_tests
if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import deep_deepshap_plots as deep_plots
if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.deep_deepshap_data import load_split as deep_load_split

if MAIN:
    deep_FEATURE_NAMES = np.array([
        "Cement", "Slag", "Fly ash", "Water", "Superplasticizer",
        "Coarse aggregate", "Fine aggregate", "Age",
    ])
if MAIN:
    deep_RUN_OUTPUT = os.environ.get("ARENA_DEEPSHAP_OUTPUT_DIRECTORY")
if MAIN:
    deep_FIGURE_DIR = Path(
        os.environ.get("ARENA_DEEPSHAP_FIGURE_DIR", deep_RUN_OUTPUT or canonical_dir / "media" / 'deep')
    )
if MAIN:
    deep_SAVE_FIGURES = os.environ.get(
        "ARENA_DEEPSHAP_SAVE_FIGURES", "1" if deep_RUN_OUTPUT else "0"
    ) == "1"


def deep_show_figure(fig, stem):
    """Display a figure and optionally save the exact notebook rendering."""
    display(fig)
    if deep_SAVE_FIGURES:
        deep_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(deep_FIGURE_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

# %%

if MAIN:
    from torch import Tensor, nn

def deep_helper_deepshap_methods__check_widths(layers: list[nn.Module], n_features: int) -> None:
    width = n_features
    for index, layer in enumerate(layers):
        if type(layer) is nn.Linear:
            if layer.in_features != width:
                raise ValueError(
                    f"Layer {index} expects {layer.in_features} values, but the previous "
                    f"state has width {width}."
                )
            width = layer.out_features

# %%

if MAIN:
    import math

def deep_helper_deepshap_methods__epsilon(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("epsilon must be a finite nonnegative number.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("epsilon must be a finite nonnegative number.")
    return value

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def deep_helper_deepshap_methods__layers(model: nn.Sequential) -> list[nn.Module]:
    """Validate the deliberately narrow graph supported by the taught rules."""
    if type(model) is not nn.Sequential:
        raise TypeError("model must be an exact nn.Sequential instance.")
    layers = list(model)
    if not layers:
        raise ValueError("model cannot be empty.")
    if len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("Every layer must be a distinct module; shared modules are unsupported.")
    for layer in layers:
        if type(layer) not in (nn.Linear, nn.ReLU):
            raise TypeError(
                f"Unsupported module {type(layer).__name__}; "
                "use exact nn.Linear and nn.ReLU modules."
            )
        if type(layer) is nn.ReLU and layer.inplace:
            raise ValueError(
                "In-place ReLU is unsupported because paired states must be preserved."
            )
        for parameter in layer.parameters(recurse=False):
            if parameter.device.type != "cpu" or parameter.dtype != t.float64:
                raise TypeError("Every model parameter must be a CPU float64 tensor.")
            if not t.isfinite(parameter).all():
                raise ValueError("Every model parameter must be finite.")
    if type(layers[-1]) is not nn.Linear or layers[-1].out_features != 1:
        raise ValueError("model must end with an exact scalar-output nn.Linear layer.")
    return layers

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def deep_helper_deepshap_methods__method_tensor(name: str, value: Tensor, ndim: int | None = None) -> Tensor:
    """Check the lesson's CPU-float64 tensor contract without changing the value."""
    if type(value) is not Tensor:
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cpu" or value.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if value.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not t.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def deep_helper_deepshap_methods__model_output(model: nn.Sequential, x: Tensor) -> Tensor:
    output = model(x)
    if output.shape != (len(x), 1):
        raise ValueError("The model must return one scalar output per input row.")
    if not t.isfinite(output).all():
        raise ValueError("The model returned a nonfinite value.")
    return output[:, 0]

# %%

if MAIN:
    _check_widths = deep_helper_deepshap_methods__check_widths
if MAIN:
    _epsilon = deep_helper_deepshap_methods__epsilon
if MAIN:
    _layers = deep_helper_deepshap_methods__layers
if MAIN:
    _method_tensor = deep_helper_deepshap_methods__method_tensor
if MAIN:
    _model_output = deep_helper_deepshap_methods__model_output

# %%

if MAIN:
    import torch

if MAIN:
    deep_helper_deepshap_diagnostics_DTYPE = torch.float64

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch

def deep_helper_deepshap_diagnostics__linear(weight, bias):
    """Construct one deterministic float64 Linear layer from literal arrays."""
    weight = torch.as_tensor(weight, dtype=deep_helper_deepshap_diagnostics_DTYPE)
    bias = torch.as_tensor(bias, dtype=deep_helper_deepshap_diagnostics_DTYPE).reshape(-1)
    layer = nn.Linear(weight.shape[1], weight.shape[0], dtype=deep_helper_deepshap_diagnostics_DTYPE)
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return layer

# %%

if MAIN:
    import torch

def deep_helper_deepshap_diagnostics__max_abs(left, right):
    difference = torch.as_tensor(left).detach() - torch.as_tensor(right).detach()
    return float(difference.abs().max())

# %%

def deep_helper_deepshap_diagnostics__method(methods, name):
    """Resolve a taught function from a module, namespace, or mapping."""
    if methods is None:
        local = globals().get(name)
        if callable(local):
            return local
        deepshap_methods = SimpleNamespace(**deep_module_deepshap_methods_namespace())

        return getattr(deepshap_methods, name)
    if hasattr(methods, name):
        return getattr(methods, name)
    return methods[name]

# %%

if MAIN:
    import torch

def deep_helper_deepshap_diagnostics__prediction(model, rows):
    with torch.no_grad():
        return model(rows).reshape(-1)

# %%

if MAIN:
    deep_DTYPE = deep_helper_deepshap_diagnostics_DTYPE
if MAIN:
    deep__linear = deep_helper_deepshap_diagnostics__linear
if MAIN:
    deep__max_abs = deep_helper_deepshap_diagnostics__max_abs
if MAIN:
    deep__method = deep_helper_deepshap_diagnostics__method
if MAIN:
    deep__prediction = deep_helper_deepshap_diagnostics__prediction

# %%

def paired_forward(
    model: nn.Sequential, x: Tensor, background: Tensor
) -> tuple[list[Tensor], list[Tensor]]:
    """Save the input and every layer output for queries and references."""
    layers = _layers(model)
    _method_tensor("x", x, 2)
    _method_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    _check_widths(layers, x.shape[1])

    with t.no_grad():
        x_value, b_value = x.clone(), background.clone()
        x_states, b_states = [x_value], [b_value]
        for layer in layers:
            x_value, b_value = layer(x_value), layer(b_value)
            x_states.append(x_value)
            b_states.append(b_value)
        if x_states[-1].shape != (len(x), 1) or b_states[-1].shape != (len(background), 1):
            raise ValueError("The model must return one scalar output per input row.")
        if not t.isfinite(x_states[-1]).all() or not t.isfinite(b_states[-1]).all():
            raise ValueError("The model returned a nonfinite value.")
    return x_states, b_states


if MAIN:
    deep_tests.test_paired_forward(paired_forward)

# %%

def rescale_slope(zx: Tensor, zb: Tensor, epsilon: float = 1e-6) -> Tensor:
    """ReLU secants, using the reference derivative for small denominators."""
    _method_tensor("zx", zx)
    _method_tensor("zb", zb)
    epsilon = _epsilon(epsilon)
    try:
        zx, zb = t.broadcast_tensors(zx, zb)
    except RuntimeError as error:
        raise ValueError("zx and zb must be broadcast-compatible.") from error

    delta = zx - zb
    use_derivative = delta.eq(0) | delta.abs().lt(epsilon)
    safe_delta = t.where(use_derivative, t.ones_like(delta), delta)
    secant = (t.relu(zx) - t.relu(zb)) / safe_delta
    return t.where(use_derivative, (zb > 0).to(zb.dtype), secant)


if MAIN:
    deep_tests.test_rescale_slope(rescale_slope)

# %%

def diagnose_rescale_limits(methods):
    """Exercise the secant, exact-zero and named small-delta branches."""
    rescale_slope = deep__method(methods, "rescale_slope")
    reference = torch.tensor([-1.0, 0.5, -4e-7], dtype=deep_DTYPE)
    actual = torch.tensor([1.0, 0.5, 4e-7], dtype=deep_DTYPE)
    primary = rescale_slope(actual, reference, epsilon=1e-6)
    exact_nonzero = rescale_slope(actual, reference, epsilon=0.0)
    expected_primary = torch.tensor([0.5, 1.0, 0.0], dtype=deep_DTYPE)
    expected_exact = torch.tensor([0.5, 1.0, 0.5], dtype=deep_DTYPE)
    grid = torch.linspace(-2.0, 2.0, 161, dtype=deep_DTYPE)
    grid_reference = torch.full_like(grid, -1.0)
    return {
        "reference": reference,
        "actual": actual,
        "epsilon": 1e-6,
        "primary": primary,
        "exact_nonzero": exact_nonzero,
        "expected_primary": expected_primary,
        "expected_exact_nonzero": expected_exact,
        "curve": {
            "actual": grid,
            "reference": grid_reference,
            "slope": rescale_slope(grid, grid_reference, epsilon=1e-6),
        },
        "primary_max_abs_error": deep__max_abs(primary, expected_primary),
        "exact_nonzero_max_abs_error": deep__max_abs(exact_nonzero, expected_exact),
    }

# %%

if MAIN:
    rescale_limits = diagnose_rescale_limits(deep_curriculum_namespace())
if MAIN:
    deep_show_figure(deep_plots.plot_rescale_limits(rescale_limits), "01_rescale_limits")

# %%

def initialize_deep_16_rescale_multipliers():
    global rescale_multipliers
    @t.no_grad()
    def rescale_multipliers(
        model: nn.Sequential,
        x: Tensor,
        background: Tensor,
        epsilon: float = 1e-6,
    ) -> Tensor:
        """Propagate scalar-output Rescale multipliers to every input feature."""
        layers = _layers(model)
        epsilon = _epsilon(epsilon)
        x_states, b_states = paired_forward(model, x, background)
        multiplier = t.ones((len(x), len(background), 1), dtype=t.float64)

        for index in range(len(layers) - 1, -1, -1):
            layer = layers[index]
            if type(layer) is nn.Linear:
                multiplier = t.einsum("nro,oi->nri", multiplier, layer.weight)
            else:
                slope = rescale_slope(
                    x_states[index][:, None, :],
                    b_states[index][None, :, :],
                    epsilon,
                )
                multiplier = multiplier * slope
        return multiplier


if MAIN:
    initialize_deep_16_rescale_multipliers()


if MAIN:
    deep_tests.test_rescale_multipliers(rescale_multipliers)

# %%

def initialize_deep_19_single_reference():
    global single_reference
    @t.no_grad()
    def single_reference(
        model: nn.Sequential,
        x: Tensor,
        reference: Tensor,
        epsilon: float = 1e-6,
    ) -> Tensor:
        """Explain each query relative to one complete reference vector."""
        _method_tensor("x", x, 2)
        _method_tensor("reference", reference, 1)
        if x.shape[1] != reference.shape[0]:
            raise ValueError("reference must contain one value for every input feature.")
        multiplier = rescale_multipliers(model, x, reference[None, :], epsilon)[:, 0]
        return multiplier * (x - reference)


if MAIN:
    initialize_deep_19_single_reference()


if MAIN:
    deep_tests.test_single_reference(single_reference)

# %%

def near_zero_fixture():
    """Return the named ReLU crossing inside the 1e-6 fallback branch."""
    model = nn.Sequential(
        deep__linear([[1.0]], [0.0]),
        nn.ReLU(),
        deep__linear([[1.0]], [0.0]),
    ).eval()
    return {
        "model": model,
        "query": torch.tensor([[4e-7]], dtype=deep_DTYPE),
        "reference": torch.tensor([-4e-7], dtype=deep_DTYPE),
        "epsilon": 1e-6,
        "expected_primary_phi": torch.tensor([[0.0]], dtype=deep_DTYPE),
        "expected_exact_secant_phi": torch.tensor([[4e-7]], dtype=deep_DTYPE),
    }

# %%

def diagnose_near_zero(methods):
    """Expose the expected local-accuracy loss from the named epsilon branch."""
    single_reference = deep__method(methods, "single_reference")
    fixture = near_zero_fixture()
    model, query, reference = (
        fixture["model"],
        fixture["query"],
        fixture["reference"],
    )
    primary = single_reference(model, query, reference, epsilon=fixture["epsilon"])
    exact_secant = single_reference(model, query, reference, epsilon=0.0)
    prediction = deep__prediction(model, query)[0]
    base = deep__prediction(model, reference[None])[0]
    delta = prediction - base
    primary_residual = delta - primary.sum()
    exact_residual = delta - exact_secant.sum()
    return {
        "query": query,
        "reference": reference,
        "epsilon": fixture["epsilon"],
        "prediction": prediction,
        "base": base,
        "prediction_delta": delta,
        "primary_phi": primary,
        "exact_secant_phi": exact_secant,
        "expected_primary_phi": fixture["expected_primary_phi"],
        "expected_exact_secant_phi": fixture["expected_exact_secant_phi"],
        "primary_completeness_residual": primary_residual,
        "exact_secant_completeness_residual": exact_residual,
        "expected_primary_residual": torch.tensor(4e-7, dtype=deep_DTYPE),
        "primary_phi_max_abs_error": deep__max_abs(primary, fixture["expected_primary_phi"]),
        "exact_secant_phi_max_abs_error": deep__max_abs(
            exact_secant, fixture["expected_exact_secant_phi"]
        ),
    }

# %%

if MAIN:
    near_zero = diagnose_near_zero(deep_curriculum_namespace())
if MAIN:
    deep_show_figure(deep_plots.plot_near_zero(near_zero), "02_near_zero_epsilon")

# %%

def affine_fixture():
    """Return affine, dummy-input and constant controls with exact answers."""
    query = torch.tensor([[4.0, 2.0]], dtype=deep_DTYPE)
    reference = torch.tensor([1.0, -1.0], dtype=deep_DTYPE)
    affine = nn.Sequential(deep__linear([[2.0, -3.0]], [5.0])).eval()

    dummy_query = torch.tensor([[4.0, 2.0, 9.0]], dtype=deep_DTYPE)
    dummy_reference = torch.tensor([1.0, -1.0, -4.0], dtype=deep_DTYPE)
    dummy = nn.Sequential(deep__linear([[2.0, -3.0, 0.0]], [5.0])).eval()
    constant = nn.Sequential(deep__linear([[0.0, 0.0]], [7.0])).eval()
    return {
        "affine": {
            "model": affine,
            "query": query,
            "reference": reference,
            "expected_phi": torch.tensor([[6.0, -9.0]], dtype=deep_DTYPE),
            "expected_base": torch.tensor(10.0, dtype=deep_DTYPE),
            "expected_prediction": torch.tensor(7.0, dtype=deep_DTYPE),
        },
        "dummy": {
            "model": dummy,
            "query": dummy_query,
            "reference": dummy_reference,
            "expected_phi": torch.tensor([[6.0, -9.0, 0.0]], dtype=deep_DTYPE),
            "expected_base": torch.tensor(10.0, dtype=deep_DTYPE),
            "expected_prediction": torch.tensor(7.0, dtype=deep_DTYPE),
        },
        "constant": {
            "model": constant,
            "query": query,
            "reference": reference,
            "expected_phi": torch.zeros((1, 2), dtype=deep_DTYPE),
            "expected_base": torch.tensor(7.0, dtype=deep_DTYPE),
            "expected_prediction": torch.tensor(7.0, dtype=deep_DTYPE),
        },
    }

# %%

def diagnose_affine_controls(methods):
    """Check exact affine credit, an unused input, and a constant network."""
    single_reference = deep__method(methods, "single_reference")
    answer = {}
    for name, fixture in affine_fixture().items():
        phi = single_reference(fixture["model"], fixture["query"], fixture["reference"])
        prediction = deep__prediction(fixture["model"], fixture["query"])[0]
        base = deep__prediction(fixture["model"], fixture["reference"][None])[0]
        answer[name] = {
            "query": fixture["query"],
            "reference": fixture["reference"],
            "phi": phi,
            "expected_phi": fixture["expected_phi"],
            "prediction": prediction,
            "expected_prediction": fixture["expected_prediction"],
            "base": base,
            "expected_base": fixture["expected_base"],
            "completeness_residual": prediction - base - phi.sum(),
            "phi_max_abs_error": deep__max_abs(phi, fixture["expected_phi"]),
        }
    answer["dummy_feature_max_abs"] = float(answer["dummy"]["phi"][0, 2].detach().abs())
    return answer

# %%

if MAIN:
    affine_controls = diagnose_affine_controls(deep_curriculum_namespace())
if MAIN:
    deep_show_figure(deep_plots.plot_affine_controls(affine_controls), "03_affine_controls")

# %%

def initialize_deep_30_background_deepshap():
    global background_deepshap
    @t.no_grad()
    def background_deepshap(
        model: nn.Sequential,
        x: Tensor,
        background: Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[Tensor, Tensor]:
        """Average complete reference explanations and return E[f(background)]."""
        multiplier = rescale_multipliers(model, x, background, epsilon)
        pair_values = multiplier * (x[:, None, :] - background[None, :, :])
        with t.no_grad():
            base = _model_output(model, background).mean()
        return base, pair_values.mean(dim=1)


if MAIN:
    initialize_deep_30_background_deepshap()


if MAIN:
    deep_tests.test_background_deepshap(background_deepshap)

# %%

def background_average_fixture():
    """Return a nonlinear case where averaging references and inputs differ."""
    fixture = near_zero_fixture()
    fixture.update(
        {
            "query": torch.tensor([[2.0]], dtype=deep_DTYPE),
            "background": torch.tensor([[-2.0], [1.0]], dtype=deep_DTYPE),
            "expected_pair_phi": torch.tensor([[2.0], [1.0]], dtype=deep_DTYPE),
            "expected_mean_phi": torch.tensor([[1.5]], dtype=deep_DTYPE),
            "expected_base": torch.tensor(0.5, dtype=deep_DTYPE),
            "expected_mean_reference_phi": torch.tensor([[2.0]], dtype=deep_DTYPE),
            "expected_mean_reference_base": torch.tensor(0.0, dtype=deep_DTYPE),
        }
    )
    fixture.pop("reference")
    fixture.pop("epsilon")
    fixture.pop("expected_primary_phi")
    fixture.pop("expected_exact_secant_phi")
    return fixture

# %%

def deep_diagnose_background_average(methods):
    """Compare averaging attributions with explaining the mean reference."""
    single_reference = deep__method(methods, "single_reference")
    background_deepshap = deep__method(methods, "background_deepshap")
    fixture = background_average_fixture()
    model, query, background = fixture["model"], fixture["query"], fixture["background"]
    base, phi = background_deepshap(model, query, background)
    pair_phi = torch.stack(
        [single_reference(model, query, reference) for reference in background], dim=1
    ).squeeze(0)
    mean_reference = background.mean(dim=0)
    mean_reference_phi = single_reference(model, query, mean_reference)
    mean_reference_base = deep__prediction(model, mean_reference[None])[0]
    prediction = deep__prediction(model, query)[0]
    return {
        "query": query,
        "background": background,
        "mean_reference": mean_reference,
        "pair_phi": pair_phi,
        "expected_pair_phi": fixture["expected_pair_phi"],
        "phi": phi,
        "expected_phi": fixture["expected_mean_phi"],
        "base": base,
        "expected_base": fixture["expected_base"],
        "mean_reference_phi": mean_reference_phi,
        "expected_mean_reference_phi": fixture["expected_mean_reference_phi"],
        "mean_reference_base": mean_reference_base,
        "expected_mean_reference_base": fixture["expected_mean_reference_base"],
        "prediction": prediction,
        "averaging_gap": mean_reference_phi - phi,
        "phi_max_abs_error": deep__max_abs(phi, fixture["expected_mean_phi"]),
        "pair_phi_max_abs_error": deep__max_abs(pair_phi, fixture["expected_pair_phi"]),
    }

# %%

if MAIN:
    background_average = deep_diagnose_background_average(deep_curriculum_namespace())
if MAIN:
    deep_show_figure(deep_plots.plot_background_average(background_average), "04_background_average")

# %%

def deep_coalition_masks(d: int) -> Tensor:
    """Return all Boolean coalitions in integer order, with feature zero as bit zero."""
    if type(d) is not int or d < 1:
        raise ValueError("d must be a positive integer.")
    subset = t.arange(2**d, dtype=t.int64)
    feature = t.arange(d, dtype=t.int64)
    return subset[:, None].bitwise_and(1 << feature).ne(0)

# %%

def coalition_values(
    model: nn.Sequential,
    x: Tensor,
    background: Tensor,
    masks: Tensor,
    row_batch: int = 8,
    forward_batch: int = 65_536,
) -> Tensor:
    """Evaluate every supplied intact-donor hybrid, then average over donors."""
    layers = _layers(model)
    _method_tensor("x", x, 2)
    _method_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    _check_widths(layers, x.shape[1])
    if type(masks) is not Tensor or masks.device.type != "cpu" or masks.dtype != t.bool:
        raise TypeError("masks must be a CPU Boolean tensor.")
    if masks.ndim != 2 or masks.shape[1] != x.shape[1] or len(masks) == 0:
        raise ValueError("masks must have shape (nonempty coalitions, input features).")
    for name, size in (("row_batch", row_batch), ("forward_batch", forward_batch)):
        if type(size) is not int or size < 1:
            raise ValueError(f"{name} must be a positive integer.")

    result = t.empty((len(x), len(masks)), dtype=t.float64)
    with t.no_grad():
        for start in range(0, len(x), row_batch):
            query = x[start : start + row_batch]
            hybrids = t.where(
                masks[None, :, None, :],
                query[:, None, None, :],
                background[None, None, :, :],
            )
            flat = hybrids.reshape(-1, x.shape[1])
            predictions = t.cat(
                [
                    _model_output(model, flat[i : i + forward_batch])
                    for i in range(0, len(flat), forward_batch)
                ]
            )
            result[start : start + len(query)] = predictions.reshape(
                len(query), len(masks), len(background)
            ).mean(dim=2)
    return result


if MAIN:
    deep_tests.test_coalition_values(coalition_values)

# %%

def deep_exact_shapley(values: Tensor) -> Tensor:
    """Contract a complete (..., 2**D) game with exact Shapley weights."""
    _method_tensor("values", values)
    if values.ndim < 1:
        raise ValueError("values must have a final coalition axis.")
    size = values.shape[-1]
    d = size.bit_length() - 1
    if d < 1 or size != 2**d:
        raise ValueError("The final values axis must contain all 2**D coalitions for D >= 1.")

    result = t.zeros((*values.shape[:-1], d), dtype=t.float64)
    subset_ids = t.arange(size, dtype=t.int64)
    for feature in range(d):
        absent = subset_ids[subset_ids.bitwise_and(1 << feature).eq(0)]
        weights = t.tensor(
            [1 / (d * math.comb(d - 1, int(subset).bit_count())) for subset in absent],
            dtype=t.float64,
        )
        marginal = values[..., absent | (1 << feature)] - values[..., absent]
        result[..., feature] = (marginal * weights).sum(dim=-1)
    return result


if MAIN:
    deep_tests.test_exact_shapley(deep_exact_shapley)

# %%

def gradient_delta(model: nn.Sequential, x: Tensor, background: Tensor) -> Tensor:
    """Point gradient at x times x minus the mean background; not GradientSHAP."""
    layers = _layers(model)
    _method_tensor("x", x, 2)
    _method_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    _check_widths(layers, x.shape[1])

    query = x.detach().clone().requires_grad_(True)
    with t.enable_grad():
        output = _model_output(model, query)
        gradient = t.autograd.grad(output.sum(), query)[0]
    if not t.isfinite(gradient).all():
        raise ValueError("The model produced a nonfinite input gradient.")
    return (gradient * (query - background.mean(dim=0))).detach()

# %%

if MAIN:
    deep_tests.test_gradient_delta(gradient_delta)

# %%

def deep_threshold_fixture(
    weights: Sequence[float] = (2.0, 1.0),
    bias: float = -2.0,
    query: Sequence[float] = (1.0, 1.0),
    reference: Sequence[float] = (0.0, 0.0),
):
    """Build ``ReLU(w @ x + bias)`` as a supported explicit Sequential."""
    weights = torch.as_tensor(weights, dtype=deep_DTYPE).reshape(1, -1)
    query = torch.as_tensor(query, dtype=deep_DTYPE).reshape(1, -1)
    reference = torch.as_tensor(reference, dtype=deep_DTYPE).reshape(-1)
    if query.shape[1] != weights.shape[1] or reference.shape != query.shape[1:]:
        raise ValueError("weights, query and reference must have one common feature size.")
    model = nn.Sequential(
        deep__linear(weights, [bias]),
        nn.ReLU(),
        deep__linear([[1.0]], [0.0]),
    ).eval()
    return {"model": model, "query": query, "reference": reference}

# %%

def diagnose_threshold(methods):
    """Show that Rescale can be complete while differing from exact Shapley."""
    single_reference = deep__method(methods, "single_reference")
    coalition_values = deep__method(methods, "coalition_values")
    exact_shapley = deep__method(methods, "exact_shapley")
    gradient_delta = deep__method(methods, "gradient_delta")
    fixture = deep_threshold_fixture()
    model, query, reference = (
        fixture["model"],
        fixture["query"],
        fixture["reference"],
    )
    masks = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.bool)
    game = coalition_values(model, query, reference[None], masks)
    deep = single_reference(model, query, reference)
    exact = exact_shapley(game)
    gradient = gradient_delta(model, query, reference[None])
    prediction, base = game[0, -1], game[0, 0]
    expected_game = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=deep_DTYPE)
    expected_exact = torch.tensor([[0.5, 0.5]], dtype=deep_DTYPE)
    expected_deep = torch.tensor([[2.0 / 3.0, 1.0 / 3.0]], dtype=deep_DTYPE)
    expected_gradient = torch.tensor([[2.0, 1.0]], dtype=deep_DTYPE)
    full_minus_feature = torch.stack([prediction - game[0, 2], prediction - game[0, 1]])
    return {
        "query": query,
        "reference": reference,
        "masks": masks,
        "game": game,
        "expected_game": expected_game,
        "deep": deep,
        "expected_deep": expected_deep,
        "exact": exact,
        "expected_exact": expected_exact,
        "gradient": gradient,
        "expected_gradient": expected_gradient,
        "full_minus_feature": full_minus_feature,
        "expected_full_minus_feature": torch.ones(2, dtype=deep_DTYPE),
        "prediction": prediction,
        "base": base,
        "prediction_delta": prediction - base,
        "deep_completeness_residual": prediction - base - deep.sum(),
        "exact_completeness_residual": prediction - base - exact.sum(),
        "gradient_completeness_residual": prediction - base - gradient.sum(),
        "deep_exact_max_abs_error": deep__max_abs(deep, exact),
        "fixture_max_abs_error": max(
            deep__max_abs(game, expected_game),
            deep__max_abs(deep, expected_deep),
            deep__max_abs(exact, expected_exact),
            deep__max_abs(gradient, expected_gradient),
        ),
    }

# %%

if MAIN:
    threshold = diagnose_threshold(deep_curriculum_namespace())
if MAIN:
    deep_show_figure(deep_plots.plot_threshold(threshold), "05_threshold_complete_not_exact")

# %%

def reveal_cancel_groups(
    activation,
    reference: Tensor,
    positive: Tensor,
    negative: Tensor,
) -> tuple[Tensor, Tensor]:
    """Two-player Shapley allocation of positive/negative groups through one activation."""
    if not callable(activation):
        raise TypeError("activation must be callable.")
    for name, value in (("reference", reference), ("positive", positive), ("negative", negative)):
        _method_tensor(name, value)
    try:
        reference, positive, negative = t.broadcast_tensors(reference, positive, negative)
    except RuntimeError as error:
        raise ValueError(
            "reference, positive and negative must be broadcast-compatible."
        ) from error
    if (positive < 0).any() or (negative > 0).any():
        raise ValueError("positive must be nonnegative and negative must be nonpositive.")

    def activate(value: Tensor) -> Tensor:
        output = activation(value)
        if type(output) is not Tensor or output.shape != value.shape:
            raise TypeError("activation must return a tensor with the same shape as its input.")
        if output.device.type != "cpu" or output.dtype != t.float64 or not t.isfinite(output).all():
            raise ValueError("activation must return finite CPU float64 values.")
        return output

    base = activate(reference)
    positive_first = activate(reference + positive)
    negative_first = activate(reference + negative)
    together = activate(reference + positive + negative)
    positive_value = 0.5 * (positive_first - base + together - negative_first)
    negative_value = 0.5 * (negative_first - base + together - positive_first)
    return positive_value, negative_value


if MAIN:
    deep_tests.test_reveal_cancel_groups(reveal_cancel_groups)

# %%

def min_fixture():
    """Return two graph forms of min(x0, x1) on nonnegative inputs."""
    first = nn.Sequential(
        deep__linear([[0.0, 1.0], [-1.0, 1.0]], [0.0, 0.0]),
        nn.ReLU(),
        deep__linear([[1.0, -1.0]], [0.0]),
    ).eval()
    mirrored = nn.Sequential(
        deep__linear([[1.0, 0.0], [1.0, -1.0]], [0.0, 0.0]),
        nn.ReLU(),
        deep__linear([[1.0, -1.0]], [0.0]),
    ).eval()
    return {
        "models": {"x1 route": first, "x0 route": mirrored},
        "query": torch.tensor([[1.0, 1.0]], dtype=deep_DTYPE),
        "reference": torch.tensor([0.0, 0.0], dtype=deep_DTYPE),
        "binary_inputs": torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=deep_DTYPE
        ),
        "expected_binary": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=deep_DTYPE),
        "expected_exact": torch.tensor([[0.5, 0.5]], dtype=deep_DTYPE),
        "expected_deep": {
            "x1 route": torch.tensor([[0.0, 1.0]], dtype=deep_DTYPE),
            "x0 route": torch.tensor([[1.0, 0.0]], dtype=deep_DTYPE),
        },
    }

# %%

def diagnose_cancellation(methods):
    """Compare two equivalent min graphs and the bounded RevealCancel repair."""
    single_reference = deep__method(methods, "single_reference")
    coalition_values = deep__method(methods, "coalition_values")
    exact_shapley = deep__method(methods, "exact_shapley")
    reveal_cancel = deep__method(methods, "reveal_cancel_groups")
    fixture = min_fixture()
    query, reference = fixture["query"], fixture["reference"]
    masks = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.bool)
    positive, negative = reveal_cancel(
        torch.relu,
        torch.tensor(0.0, dtype=deep_DTYPE),
        torch.tensor(1.0, dtype=deep_DTYPE),
        torch.tensor(-1.0, dtype=deep_DTYPE),
    )
    reveal_phi = {
        "x1 route": torch.stack([-negative, 1.0 - positive])[None],
        "x0 route": torch.stack([1.0 - positive, -negative])[None],
    }
    models = {}
    for name, model in fixture["models"].items():
        game = coalition_values(model, query, reference[None], masks)
        deep = single_reference(model, query, reference)
        exact = exact_shapley(game)
        binary_prediction = deep__prediction(model, fixture["binary_inputs"])
        models[name] = {
            "binary_prediction": binary_prediction,
            "expected_binary": fixture["expected_binary"],
            "game": game,
            "deep": deep,
            "expected_deep": fixture["expected_deep"][name],
            "exact": exact,
            "expected_exact": fixture["expected_exact"],
            "reveal_cancel": reveal_phi[name],
            "binary_max_abs_error": deep__max_abs(binary_prediction, fixture["expected_binary"]),
            "deep_max_abs_error": deep__max_abs(deep, fixture["expected_deep"][name]),
            "exact_max_abs_error": deep__max_abs(exact, fixture["expected_exact"]),
            "reveal_cancel_max_abs_error": deep__max_abs(
                reveal_phi[name], fixture["expected_exact"]
            ),
        }
    return {
        "query": query,
        "reference": reference,
        "masks": masks,
        "models": models,
        "group_allocation": {"positive": positive, "negative": negative},
        "computational_domain": "nonnegative inputs",
        "deep_implementation_gap": models["x0 route"]["deep"] - models["x1 route"]["deep"],
    }

# %%

if MAIN:
    cancellation = diagnose_cancellation(deep_curriculum_namespace())
if MAIN:
    deep_show_figure(
        deep_plots.plot_cancellation(cancellation),
        "06_cancellation_implementation_dependence",
    )

# %%

def ranked_replacement_curves(
    values: Tensor,
    phi: Tensor,
    orders: Tensor | None = None,
) -> dict[str, Tensor]:
    """Read insertion/deletion prefixes from a complete per-row coalition table."""
    _method_tensor("values", values, 2)
    _method_tensor("phi", phi, 2)
    n, size = values.shape
    d = size.bit_length() - 1
    if d < 1 or size != 2**d or phi.shape != (n, d):
        raise ValueError("values must be (N,2**D) and phi must be (N,D), with D >= 1.")

    if orders is None:
        direction = t.where(values[:, -1] - values[:, 0] >= 0, 1.0, -1.0)[:, None]
        orders = t.argsort(direction * phi, dim=1, descending=True, stable=True)
    else:
        if type(orders) is not Tensor or orders.device.type != "cpu" or orders.dtype != t.int64:
            raise TypeError("orders must be a CPU int64 tensor.")
        if orders.shape != (n, d):
            raise ValueError("orders must have shape (rows, features).")
        expected = t.arange(d, dtype=t.int64).expand(n, d)
        if not t.equal(orders.sort(dim=1).values, expected):
            raise ValueError("Every order row must contain each feature exactly once.")
        orders = orders.clone()

    insertion_masks = t.zeros((n, d + 1), dtype=t.int64)
    deletion_masks = t.full((n, d + 1), size - 1, dtype=t.int64)
    for step in range(d):
        bit = 1 << orders[:, step]
        insertion_masks[:, step + 1] = insertion_masks[:, step] | bit
        deletion_masks[:, step + 1] = deletion_masks[:, step] ^ bit
    return {
        "orders": orders,
        "insertion_masks": insertion_masks,
        "deletion_masks": deletion_masks,
        "insertion": values.gather(1, insertion_masks),
        "deletion": values.gather(1, deletion_masks),
    }


if MAIN:
    deep_tests.test_ranked_replacement_curves(ranked_replacement_curves)

# %%

def deep_replacement_fixture():
    """Return an additive three-player game with one exact dummy feature."""
    contributions = torch.tensor([3.0, -2.0, 0.0], dtype=deep_DTYPE)
    masks = torch.tensor(
        [[bool(mask & (1 << feature)) for feature in range(3)] for mask in range(8)]
    )
    values = 10.0 + (masks.to(deep_DTYPE) * contributions).sum(dim=1)
    return {
        "masks": masks,
        "values": values[None],
        "phi": contributions[None],
        "expected_order": torch.tensor([[0, 2, 1]], dtype=torch.long),
        "expected_insertion": torch.tensor([[10.0, 13.0, 13.0, 11.0]], dtype=deep_DTYPE),
        "expected_deletion": torch.tensor([[11.0, 8.0, 8.0, 10.0]], dtype=deep_DTYPE),
    }

# %%

def diagnose_replacement_control(methods):
    """Check ranked insertion/deletion endpoints and a zero-credit feature."""
    ranked_curves = deep__method(methods, "ranked_replacement_curves")
    fixture = deep_replacement_fixture()
    result = ranked_curves(fixture["values"], fixture["phi"])
    return {
        **fixture,
        "result": result,
        "order_max_abs_error": deep__max_abs(result["orders"], fixture["expected_order"]),
        "insertion_max_abs_error": deep__max_abs(result["insertion"], fixture["expected_insertion"]),
        "deletion_max_abs_error": deep__max_abs(result["deletion"], fixture["expected_deletion"]),
        "dummy_insertion_change": result["insertion"][0, 2] - result["insertion"][0, 1],
        "dummy_deletion_change": result["deletion"][0, 2] - result["deletion"][0, 1],
    }

# %%

if MAIN:
    replacement_control = diagnose_replacement_control(deep_curriculum_namespace())
if MAIN:
    deep_show_figure(deep_plots.plot_replacement_control(replacement_control), "07_replacement_fixture")

# %%

if MAIN:
    deep_curriculum_methods = SimpleNamespace(
        paired_forward=paired_forward,
        rescale_slope=rescale_slope,
        rescale_multipliers=rescale_multipliers,
        single_reference=single_reference,
        background_deepshap=background_deepshap,
        coalition_values=coalition_values,
        exact_shapley=deep_exact_shapley,
        reveal_cancel_groups=reveal_cancel_groups,
        ranked_replacement_curves=ranked_replacement_curves,
        coalition_masks=deep_coalition_masks,
        gradient_delta=gradient_delta,
    )

# %%

def deep_threshold_play(
    methods,
    weights: Sequence[float] = (2.0, 1.0),
    bias: float = -2.0,
    query: Sequence[float] = (1.0, 1.0),
    reference: Sequence[float] = (0.0, 0.0),
    epsilon: float = 1e-6,
):
    """Recompute the editable gate with the learner's implemented methods."""
    single_reference = deep__method(methods, "single_reference")
    coalition_values = deep__method(methods, "coalition_values")
    exact_shapley = deep__method(methods, "exact_shapley")
    gradient_delta = deep__method(methods, "gradient_delta")
    fixture = deep_threshold_fixture(weights, bias, query, reference)
    model, query_tensor, reference_tensor = (
        fixture["model"],
        fixture["query"],
        fixture["reference"],
    )
    dimension = query_tensor.shape[1]
    masks = torch.tensor(
        [
            [bool(mask & (1 << feature)) for feature in range(dimension)]
            for mask in range(2**dimension)
        ]
    )
    game = coalition_values(model, query_tensor, reference_tensor[None], masks)
    deep = single_reference(model, query_tensor, reference_tensor, epsilon=epsilon)
    exact = exact_shapley(game)
    gradient = gradient_delta(model, query_tensor, reference_tensor[None])
    return {
        "weights": torch.as_tensor(weights, dtype=deep_DTYPE),
        "bias": float(bias),
        "query": query_tensor,
        "reference": reference_tensor,
        "epsilon": float(epsilon),
        "masks": masks,
        "game": game,
        "prediction": game[0, -1],
        "base": game[0, 0],
        "deep": deep,
        "exact": exact,
        "gradient": gradient,
        "deep_completeness_residual": game[0, -1] - game[0, 0] - deep.sum(),
        "deep_exact_max_abs_error": deep__max_abs(deep, exact),
    }

# %%

def deep_run_diagnostics(methods=None, include_timing=True):
    """Collect every checkpoint after all nine methods have been defined.

    The lesson calls each diagnostic separately at its own checkpoint.  This
    convenience runner is reserved for the final native/reproduction harness,
    where the complete student-method namespace is already available.  There
    is no timing study in this section, so the separately persisted timing
    branch is intentionally empty.
    """

    def detach_tree(value):
        if isinstance(value, Tensor):
            return value.detach()
        if isinstance(value, dict):
            return {key: detach_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [detach_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(detach_tree(item) for item in value)
        return value

    mechanism = {
        "rescale_limits": diagnose_rescale_limits(methods),
        "near_zero": diagnose_near_zero(methods),
        "affine_controls": diagnose_affine_controls(methods),
        "background_average": deep_diagnose_background_average(methods),
        "threshold": diagnose_threshold(methods),
        "cancellation": diagnose_cancellation(methods),
        "replacement_control": diagnose_replacement_control(methods),
        "threshold_play": deep_threshold_play(methods),
    }
    return {"mechanism": detach_tree(mechanism), "timing": {} if include_timing else {}}

# %%

if MAIN:
    deep_tiny_play = deep_threshold_play(
        deep_curriculum_methods,
        weights=(2.0, 1.0),
        bias=-2.0,
        query=(1.0, 1.0),
        reference=(0.0, 0.0),
        epsilon=1e-6,
    )
if MAIN:
    deep_show_figure(deep_plots.plot_threshold_play(deep_tiny_play), "16_threshold_play")

# %%

if MAIN:
    from collections.abc import Mapping

if MAIN:
    from typing import Any

if MAIN:
    import numpy as np

if MAIN:
    import torch as t

if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.deep_deepshap_data import (
        background_order as deep_background_order,
    )

deep_FEATURES = 8

deep_HIDDEN = 32

deep_TRAINING_STEPS = 800

deep_LEARNING_RATE = 0.003

deep_WEIGHT_DECAY = 0.0001

deep_RIDGE_PENALTY = 0.01

deep_MODEL_SEEDS = (161410, 161411, 161412)

deep_SHUFFLE_SEEDS = (161420, 161421, 161422)

PRIMARY_BACKGROUND_SEED = 161460

INDEPENDENT_BACKGROUND_SEEDS = (161461, 161462)

deep_RANDOM_ORDER_SEED = 161480

# %%

if MAIN:
    deep_MODEL_KEYS = tuple(
        [f"clean_{index}" for index in range(3)]
        + [f"shuffled_{index}" for index in range(3)]
        + ["ridge"]
    )

deep_FEATURE_NAMES = (
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "Superplasticizer",
    "Coarse aggregate",
    "Fine aggregate",
    "Age",
)

if MAIN:
    deep_MODEL_LABELS = {
        **{f"clean_{index}": f"clean MLP seed {seed}" for index, seed in enumerate(deep_MODEL_SEEDS)},
        **{
            f"shuffled_{index}": f"shuffled MLP seed {seed}"
            for index, seed in enumerate(deep_MODEL_SEEDS)
        },
        "ridge": "ridge control",
    }

deep_METHOD_NAMES = (
    "rescale_multipliers",
    "background_deepshap",
    "coalition_values",
    "exact_shapley",
    "gradient_delta",
    "coalition_masks",
    "ranked_replacement_curves",
)

deep_UNITS = {
    "input": "source feature units",
    "target": "MPa",
    "prediction": "MPa",
    "base": "MPa",
    "attribution": "MPa",
    "game": "MPa",
    "squared_error": "MPa squared",
}

# %%

if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def deep_helper_deepshap_study__clone_state(state: Mapping[str, t.Tensor]) -> dict[str, t.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError("A model state must be a tensor mapping.")
    cloned: dict[str, t.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, t.Tensor):
            raise TypeError("State keys must be strings and values must be tensors.")
        if value.device.type != "cpu" or value.dtype != t.float64:
            raise TypeError("Every model-state tensor must be CPU float64.")
        if not bool(t.isfinite(value).all()):
            raise ValueError("Model-state tensors must be finite.")
        cloned[name] = value.detach().clone()
    return cloned

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def deep_helper_deepshap_study__clone_tree(value: Any) -> Any:
    if isinstance(value, t.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {name: deep_helper_deepshap_study__clone_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [deep_helper_deepshap_study__clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(deep_helper_deepshap_study__clone_tree(item) for item in value)
    return value

# %%

if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def deep_helper_deepshap_study__combine_explanations(
    explanation: Mapping[str, t.Tensor], audit: Mapping[str, t.Tensor]
) -> dict[str, t.Tensor]:
    if not t.allclose(explanation["base"], audit["base"], atol=0.0, rtol=0.0):
        raise ValueError("Primary background base changed between row blocks.")
    return {
        "base": explanation["base"].clone(),
        **{
            name: t.cat([explanation[name], audit[name]], dim=0)
            for name in (
                "prediction",
                "phi_deep",
                "phi_exact",
                "phi_gradient",
                "game",
                "pair_deep",
            )
        },
    }

# %%

if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def deep_helper_deepshap_study__combine_row_blocks(rows: Mapping[str, t.Tensor]) -> dict[str, t.Tensor]:
    """Concatenate explanation then audit rows and retain a numeric split label."""
    return {
        "x": t.cat([rows["explain_x"], rows["audit_x"]], dim=0),
        "y": t.cat([rows["explain_y"], rows["audit_y"]], dim=0),
        "ids": t.cat([rows["explain_ids"], rows["audit_ids"]], dim=0),
        "split": t.cat(
            [
                t.zeros(len(rows["explain_ids"]), dtype=t.int64),
                t.ones(len(rows["audit_ids"]), dtype=t.int64),
            ]
        ),
    }

# %%

if MAIN:
    from typing import Any
if MAIN:
    import torch as t

def deep_helper_deepshap_study__float_tensor(value: Any, name: str, dimensions: int) -> t.Tensor:
    tensor = value if isinstance(value, t.Tensor) else t.as_tensor(value)
    if tensor.device.type != "cpu" or tensor.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor or array.")
    if tensor.ndim != dimensions or not bool(t.isfinite(tensor).all()):
        raise ValueError(f"{name} has the wrong shape or contains nonfinite values.")
    return tensor.detach().clone()

# %%

if MAIN:
    from typing import Any
if MAIN:
    import torch as t

def deep_helper_deepshap_study__ids_tensor(value: Any, rows: int, name: str) -> t.Tensor:
    tensor = value if isinstance(value, t.Tensor) else t.as_tensor(value)
    if tensor.device.type != "cpu" or tensor.dtype != t.int64 or tensor.shape != (rows,):
        raise TypeError(f"{name} must be a one-dimensional CPU int64 array.")
    if len(t.unique(tensor)) != rows:
        raise ValueError(f"{name} must contain unique source-row identities.")
    return tensor.detach().clone()

# %%

deep_helper_deepshap_study_FEATURES = 8

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def deep_helper_deepshap_study__normalization_tensors(normalization: Mapping[str, Any]) -> tuple[t.Tensor, ...]:
    if not isinstance(normalization, Mapping):
        raise TypeError("normalization must be a mapping.")
    x_mean = deep_helper_deepshap_study__float_tensor(normalization.get("x_mean"), "x_mean", 1)
    x_std = deep_helper_deepshap_study__float_tensor(normalization.get("x_std"), "x_std", 1)
    y_mean = deep_helper_deepshap_study__float_tensor(normalization.get("y_mean"), "y_mean", 0)
    y_std = deep_helper_deepshap_study__float_tensor(normalization.get("y_std"), "y_std", 0)
    if x_mean.shape != (deep_helper_deepshap_study_FEATURES,) or x_std.shape != (deep_helper_deepshap_study_FEATURES,):
        raise ValueError("Normalization must describe exactly eight input features.")
    if bool((x_std <= 0).any()) or not bool(y_std > 0):
        raise ValueError("Training standard deviations must be positive.")
    return x_mean, x_std, y_mean, y_std

# %%

if MAIN:
    from typing import Any
if MAIN:
    import torch as t

def deep_helper_deepshap_study__output_tensor(value: Any, shape: tuple[int, ...], name: str) -> t.Tensor:
    if not isinstance(value, t.Tensor):
        raise TypeError(f"{name} must be a Torch tensor.")
    if value.device.type != "cpu" or value.dtype != t.float64 or tuple(value.shape) != shape:
        raise ValueError(f"{name} has the wrong dtype, device or shape.")
    if not bool(t.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values.")
    return value.detach().clone()

# %%

if MAIN:
    import torch as t

def deep_helper_deepshap_study__require_disjoint_ids(**splits: t.Tensor) -> None:
    """Reject source-row reuse across named train, explanation and audit splits."""
    names = tuple(splits)
    for left_index, left_name in enumerate(names):
        left = splits[left_name]
        for right_name in names[left_index + 1 :]:
            right = splits[right_name]
            if bool(t.isin(left, right).any()):
                raise ValueError(
                    f"{left_name} and {right_name} source-row identities must be disjoint."
                )

# %%

deep_helper_deepshap_study_METHOD_NAMES = (
    "rescale_multipliers",
    "background_deepshap",
    "coalition_values",
    "exact_shapley",
    "gradient_delta",
    "coalition_masks",
    "ranked_replacement_curves",
)

# %%

if MAIN:
    from typing import Any

def deep_helper_deepshap_study__require_methods(methods: Any) -> None:
    missing = [name for name in deep_helper_deepshap_study_METHOD_NAMES if not callable(getattr(methods, name, None))]
    if missing:
        raise TypeError(f"Student methods namespace is missing {missing}.")

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def deep_helper_deepshap_study__rows(rows: Mapping[str, Any], name: str) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    if not isinstance(rows, Mapping) or not {"x", "y", "ids"} <= set(rows):
        raise TypeError(f"{name} must contain x, y and ids.")
    x = deep_helper_deepshap_study__float_tensor(rows["x"], f"{name}.x", 2)
    y = deep_helper_deepshap_study__float_tensor(rows["y"], f"{name}.y", 1)
    if x.shape != (len(y), deep_helper_deepshap_study_FEATURES) or len(y) == 0:
        raise ValueError(f"{name} must contain nonempty eight-feature rows.")
    ids = deep_helper_deepshap_study__ids_tensor(rows["ids"], len(y), f"{name}.ids")
    return x, y, ids

# %%

deep_helper_deepshap_study_HIDDEN = 32

# %%

def deep_helper_deepshap_study__state_shapes(kind: str) -> dict[str, tuple[int, ...]]:
    if kind == "mlp":
        return {
            "0.weight": (deep_helper_deepshap_study_HIDDEN, deep_helper_deepshap_study_FEATURES),
            "0.bias": (deep_helper_deepshap_study_HIDDEN,),
            "2.weight": (deep_helper_deepshap_study_HIDDEN, deep_helper_deepshap_study_HIDDEN),
            "2.bias": (deep_helper_deepshap_study_HIDDEN,),
            "4.weight": (1, deep_helper_deepshap_study_HIDDEN),
            "4.bias": (1,),
        }
    if kind == "ridge":
        return {"0.weight": (1, deep_helper_deepshap_study_FEATURES), "0.bias": (1,)}
    raise ValueError(f"Unsupported fitted-model kind: {kind!r}.")

# %%

if MAIN:
    deep_helper_deepshap_study_MODEL_KEYS = tuple(
        [f"clean_{index}" for index in range(3)]
        + [f"shuffled_{index}" for index in range(3)]
        + ["ridge"]
    )

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping

def deep_helper_deepshap_study__validate_fitted(fitted: Mapping[str, Any]) -> None:
    if not isinstance(fitted, Mapping) or fitted.get("phase") != "models_fitted_on_train_only":
        raise ValueError("Use the complete train-only fitted payload.")
    if not isinstance(fitted.get("models"), Mapping) or tuple(fitted["models"]) != deep_helper_deepshap_study_MODEL_KEYS:
        raise ValueError("The fitted payload must retain all seven models in frozen order.")
    deep_helper_deepshap_study__normalization_tensors(fitted.get("normalization"))
    x = deep_helper_deepshap_study__float_tensor(fitted.get("train_x_raw"), "train_x_raw", 2)
    y = deep_helper_deepshap_study__float_tensor(fitted.get("train_y_raw"), "train_y_raw", 1)
    if x.shape != (len(y), deep_helper_deepshap_study_FEATURES):
        raise ValueError("Fitted training arrays have incompatible shapes.")
    deep_helper_deepshap_study__ids_tensor(fitted.get("train_ids"), len(y), "train_ids")
    for key in deep_helper_deepshap_study_MODEL_KEYS:
        model = fitted["models"][key]
        expected_kind = "ridge" if key == "ridge" else "mlp"
        if model.get("kind") != expected_kind:
            raise ValueError(f"{key} has the wrong model kind.")
        make_model(model)

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def deep_helper_deepshap_study__validate_rankings(rankings: Any) -> None:
    if not isinstance(rankings, Mapping) or tuple(rankings) != deep_helper_deepshap_study_MODEL_KEYS:
        raise ValueError("Prepared rankings must retain all seven models in frozen order.")
    expected = t.arange(deep_helper_deepshap_study_FEATURES, dtype=t.int64)
    for key in deep_helper_deepshap_study_MODEL_KEYS:
        item = rankings[key]
        if not isinstance(item, Mapping) or set(item) != {"exact", "deep", "gradient"}:
            raise ValueError(f"{key} must retain exact, DeepSHAP and gradient rankings.")
        for name, order in item.items():
            if not isinstance(order, t.Tensor) or order.device.type != "cpu":
                raise TypeError(f"{key}/{name} ranking must be a CPU tensor.")
            if order.dtype != t.int64 or order.shape != (deep_helper_deepshap_study_FEATURES,):
                raise ValueError(f"{key}/{name} ranking must be one int64 feature order.")
            if not t.equal(order.sort().values, expected):
                raise ValueError(f"{key}/{name} ranking must contain each feature once.")

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def deep_helper_deepshap_study__validate_evaluation_inputs(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, t.Tensor]:
    """Recheck the persisted preparation boundary against all named row blocks."""
    deep_helper_deepshap_study__validate_fitted(fitted)
    if prepared.get("phase") != "rankings_frozen_before_audit":
        raise ValueError("Persist explanation rankings before audit evaluation.")
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("Audit rows cannot select models, backgrounds or rankings.")
    train_x, train_y, train_ids = deep_helper_deepshap_study__rows(train, "train")
    explain_x, explain_y, explain_ids = deep_helper_deepshap_study__rows(explain, "explain")
    audit_x, audit_y, audit_ids = deep_helper_deepshap_study__rows(audit, "audit")
    deep_helper_deepshap_study__require_disjoint_ids(train=train_ids, explanation=explain_ids, audit=audit_ids)
    if not t.equal(train_ids, fitted["train_ids"]) or not t.equal(
        train_x, fitted["train_x_raw"]
    ) or not t.equal(train_y, fitted["train_y_raw"]):
        raise ValueError("Training rows changed after fitting.")
    if not t.equal(prepared["explain_ids"], explain_ids):
        raise ValueError("Explanation identities changed after rankings were frozen.")
    if not t.equal(prepared["explain_x_raw"], explain_x) or not t.equal(
        prepared["explain_targets"], explain_y
    ):
        raise ValueError("Explanation values changed after rankings were frozen.")
    expected_indices = deep__background_indices(train_ids)
    indices = prepared.get("background_indices")
    if not isinstance(indices, Mapping) or tuple(indices) != tuple(expected_indices):
        raise ValueError("Prepared background designs changed before audit evaluation.")
    if any(not t.equal(indices[name], expected_indices[name]) for name in expected_indices):
        raise ValueError("Prepared background indices changed before audit evaluation.")
    primary = expected_indices["primary64"]
    if not t.equal(prepared["background_ids"], train_ids[primary]) or not t.equal(
        prepared["background"], train_x[primary]
    ):
        raise ValueError("Primary background identities changed before audit evaluation.")
    deep_helper_deepshap_study__validate_rankings(prepared.get("rankings"))
    return {
        "train_x": train_x,
        "train_ids": train_ids,
        "explain_x": explain_x,
        "explain_y": explain_y,
        "explain_ids": explain_ids,
        "audit_x": audit_x,
        "audit_y": audit_y,
        "audit_ids": audit_ids,
    }

# %%

if MAIN:
    deep__clone_state = deep_helper_deepshap_study__clone_state
if MAIN:
    _clone_tree = deep_helper_deepshap_study__clone_tree
if MAIN:
    _combine_explanations = deep_helper_deepshap_study__combine_explanations
if MAIN:
    _combine_row_blocks = deep_helper_deepshap_study__combine_row_blocks
if MAIN:
    deep__float_tensor = deep_helper_deepshap_study__float_tensor
if MAIN:
    deep__ids_tensor = deep_helper_deepshap_study__ids_tensor
if MAIN:
    deep__normalization_tensors = deep_helper_deepshap_study__normalization_tensors
if MAIN:
    deep__output_tensor = deep_helper_deepshap_study__output_tensor
if MAIN:
    deep__require_disjoint_ids = deep_helper_deepshap_study__require_disjoint_ids
if MAIN:
    _require_methods = deep_helper_deepshap_study__require_methods
if MAIN:
    deep__rows = deep_helper_deepshap_study__rows
if MAIN:
    deep__state_shapes = deep_helper_deepshap_study__state_shapes
if MAIN:
    deep__validate_evaluation_inputs = deep_helper_deepshap_study__validate_evaluation_inputs
if MAIN:
    deep__validate_fitted = deep_helper_deepshap_study__validate_fitted
if MAIN:
    _validate_rankings = deep_helper_deepshap_study__validate_rankings

# %%

def deep__model_from_state(kind: str, state: Mapping[str, t.Tensor]) -> t.nn.Sequential:
    copied = deep__clone_state(state)
    shapes = deep__state_shapes(kind)
    if set(copied) != set(shapes):
        raise ValueError(f"{kind} state has unexpected parameter names.")
    if any(tuple(copied[name].shape) != shape for name, shape in shapes.items()):
        raise ValueError(f"{kind} state has an unexpected parameter shape.")
    if kind == "mlp":
        model = t.nn.Sequential(
            t.nn.Linear(deep_FEATURES, deep_HIDDEN, dtype=t.float64),
            t.nn.ReLU(),
            t.nn.Linear(deep_HIDDEN, deep_HIDDEN, dtype=t.float64),
            t.nn.ReLU(),
            t.nn.Linear(deep_HIDDEN, 1, dtype=t.float64),
        )
    else:
        model = t.nn.Sequential(t.nn.Linear(deep_FEATURES, 1, dtype=t.float64))
    model.load_state_dict(copied, strict=True)
    model.eval()
    return model

def make_model(item: Mapping[str, Any]) -> t.nn.Sequential:
    """Reconstruct a raw-input, raw-output Sequential solely from saved state."""
    if not isinstance(item, Mapping) or "kind" not in item or "state" not in item:
        raise TypeError("A fitted model item must contain kind and raw state.")
    return deep__model_from_state(str(item["kind"]), item["state"])

# %%

def deep__fold_mlp_state(
    standardized_state: Mapping[str, t.Tensor], normalization: Mapping[str, Any]
) -> dict[str, t.Tensor]:
    """Fold input and output standardization into the first and last affine layers."""
    state = deep__clone_state(standardized_state)
    if set(state) != set(deep__state_shapes("mlp")):
        raise ValueError("Cannot fold an unexpected MLP state.")
    x_mean, x_std, y_mean, y_std = deep__normalization_tensors(normalization)
    first_weight = state["0.weight"].clone()
    state["0.weight"] = first_weight / x_std[None, :]
    state["0.bias"] = state["0.bias"] - first_weight @ (x_mean / x_std)
    state["4.weight"] = y_std * state["4.weight"]
    state["4.bias"] = y_std * state["4.bias"] + y_mean
    return state

def deep__fold_ridge_state(
    standardized_state: Mapping[str, t.Tensor], normalization: Mapping[str, Any]
) -> dict[str, t.Tensor]:
    """Fold standardized linear regression back into raw feature and MPa units."""
    state = deep__clone_state(standardized_state)
    if set(state) != set(deep__state_shapes("ridge")):
        raise ValueError("Cannot fold an unexpected ridge state.")
    x_mean, x_std, y_mean, y_std = deep__normalization_tensors(normalization)
    weight = state["0.weight"].clone()
    state["0.weight"] = y_std * weight / x_std[None, :]
    state["0.bias"] = y_std * (
        state["0.bias"] - (weight * (x_mean / x_std)[None, :]).sum(1)
    ) + y_mean
    return state

def deep__folding_error(
    kind: str,
    standardized_state: Mapping[str, t.Tensor],
    raw_state: Mapping[str, t.Tensor],
    normalization: Mapping[str, Any],
    x_raw: t.Tensor,
) -> float:
    x_mean, x_std, y_mean, y_std = deep__normalization_tensors(normalization)
    standard_model = deep__model_from_state(kind, standardized_state)
    raw_model = deep__model_from_state(kind, raw_state)
    with t.no_grad():
        expected = standard_model((x_raw - x_mean) / x_std).squeeze(-1) * y_std + y_mean
        actual = raw_model(x_raw).squeeze(-1)
    return float((actual - expected).abs().max())

# %%

def deep__training_normalization(
    x_raw: t.Tensor, y_raw: t.Tensor
) -> tuple[dict[str, t.Tensor], t.Tensor, t.Tensor]:
    """Compute train-only moments and return standardized training coordinates."""
    normalization = {
        "x_mean": x_raw.mean(0),
        "x_std": x_raw.std(0, correction=0),
        "y_mean": y_raw.mean(),
        "y_std": y_raw.std(correction=0),
    }
    x_mean, x_std, y_mean, y_std = deep__normalization_tensors(normalization)
    x = (x_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std
    return normalization, x, y

# %%

def deep__train_mlp(network: t.nn.Sequential, x: t.Tensor, labels: t.Tensor) -> t.Tensor:
    """Run the fixed full-batch Adam loop and retain every pre-update loss."""
    optimizer = t.optim.Adam(
        network.parameters(), lr=deep_LEARNING_RATE, weight_decay=deep_WEIGHT_DECAY
    )
    losses: list[t.Tensor] = []
    network.train()
    with t.enable_grad():
        for _ in range(deep_TRAINING_STEPS):
            optimizer.zero_grad()
            loss = (network(x).squeeze(-1) - labels).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().clone())
    return t.stack(losses)

# %%

def _fold_mlp_result(
    network: t.nn.Sequential,
    normalization: Mapping[str, Any],
    x_raw: t.Tensor,
) -> tuple[dict[str, t.Tensor], dict[str, t.Tensor], t.Tensor, float]:
    """Freeze both MLP coordinate systems and verify raw-unit prediction parity."""
    standardized_state = deep__clone_state(network.state_dict())
    raw_state = deep__fold_mlp_state(standardized_state, normalization)
    folding_error = deep__folding_error(
        "mlp", standardized_state, raw_state, normalization, x_raw
    )
    if folding_error > 1e-10:
        raise AssertionError("Raw-unit MLP folding changed its training predictions.")
    with t.no_grad():
        prediction = deep__model_from_state("mlp", raw_state)(x_raw).squeeze(-1)
    return standardized_state, raw_state, prediction, folding_error

# %%

def deep_fit_one_mlp_control(
    control: str,
    index: int,
    x: t.Tensor,
    y: t.Tensor,
    x_raw: t.Tensor,
    y_raw: t.Tensor,
    normalization: Mapping[str, Any],
) -> dict[str, Any]:
    """Train one predeclared clean or label-shuffled MLP and fold it to raw units."""
    if control not in {"clean", "shuffled"} or index not in range(3):
        raise ValueError("Choose a clean or shuffled control and seed index 0, 1 or 2.")
    seed = deep_MODEL_SEEDS[index]
    t.manual_seed(seed)
    network = t.nn.Sequential(
        t.nn.Linear(deep_FEATURES, deep_HIDDEN, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(deep_HIDDEN, deep_HIDDEN, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(deep_HIDDEN, 1, dtype=t.float64),
    )
    initial_state = deep__clone_state(network.state_dict())
    permutation = t.arange(len(y), dtype=t.int64)
    shuffle_seed = None
    if control == "shuffled":
        shuffle_seed = deep_SHUFFLE_SEEDS[index]
        permutation = t.from_numpy(
            np.random.default_rng(shuffle_seed).permutation(len(y))
        )
    labels = y[permutation]
    loss_trace = deep__train_mlp(network, x, labels)
    standardized_state, raw_state, prediction, folding_error = _fold_mlp_result(
        network, normalization, x_raw
    )
    return {
        "kind": "mlp",
        "control": control,
        "seed": seed,
        "initial_state": initial_state,
        "standardized_state": standardized_state,
        "state": raw_state,
        "loss_trace": loss_trace,
        "target_permutation": permutation.clone(),
        "training_targets_raw": y_raw[permutation].clone(),
        "training_prediction": prediction.clone(),
        "folding_max_abs_error": folding_error,
        "parameters": {
            "architecture": [deep_FEATURES, deep_HIDDEN, deep_HIDDEN, 1],
            "steps": deep_TRAINING_STEPS,
            "learning_rate": deep_LEARNING_RATE,
            "weight_decay": deep_WEIGHT_DECAY,
            "optimizer": "Adam",
            "risk": "mean_squared_error",
            "batch": "full_training_split",
            "shuffle_seed": shuffle_seed,
            "dtype": "float64",
            "parameter_count": sum(value.numel() for value in raw_state.values()),
        },
    }

# %%

def deep_fit_ridge_control(
    x: t.Tensor,
    y: t.Tensor,
    x_raw: t.Tensor,
    y_raw: t.Tensor,
    normalization: Mapping[str, Any],
) -> dict[str, Any]:
    """Solve the fixed standardized ridge control once, excluding its intercept."""
    design = t.cat([x, t.ones((len(x), 1), dtype=t.float64)], dim=1)
    penalty = t.eye(deep_FEATURES + 1, dtype=t.float64) * deep_RIDGE_PENALTY
    penalty[-1, -1] = 0.0
    coefficients = t.linalg.solve(
        design.T @ design / len(x) + penalty,
        design.T @ y / len(x),
    )
    ridge_standardized_state = {
        "0.weight": coefficients[:deep_FEATURES][None, :].clone(),
        "0.bias": coefficients[deep_FEATURES:].clone(),
    }
    ridge_state = deep__fold_ridge_state(ridge_standardized_state, normalization)
    ridge_error = deep__folding_error(
        "ridge", ridge_standardized_state, ridge_state, normalization, x_raw
    )
    if ridge_error > 1e-10:
        raise AssertionError("Raw-unit ridge folding changed its training predictions.")
    with t.no_grad():
        ridge_prediction = deep__model_from_state("ridge", ridge_state)(x_raw).squeeze(-1)
    ridge_loss = (
        (deep__model_from_state("ridge", ridge_standardized_state)(x).squeeze(-1) - y)
        .square()
        .mean()
        .detach()
    )
    return {
        "kind": "ridge",
        "control": "ridge",
        "seed": 0,
        "initial_state": {},
        "standardized_state": deep__clone_state(ridge_standardized_state),
        "state": ridge_state,
        "loss_trace": ridge_loss[None].clone(),
        "target_permutation": t.arange(len(y_raw), dtype=t.int64),
        "training_targets_raw": y_raw.clone(),
        "training_prediction": ridge_prediction.clone(),
        "folding_max_abs_error": ridge_error,
        "parameters": {
            "penalty": deep_RIDGE_PENALTY,
            "penalize_intercept": False,
            "risk": "mean_squared_error",
            "dtype": "float64",
            "parameter_count": deep_FEATURES + 1,
        },
    }

# %%

def deep_fit_models(train: Mapping[str, Any]) -> dict[str, Any]:
    """Fit six fixed MLPs and one ridge using training rows only."""
    x_raw, y_raw, train_ids = deep__rows(train, "train")
    t.set_num_threads(4)
    t.use_deterministic_algorithms(True)
    normalization, x, y = deep__training_normalization(x_raw, y_raw)
    models = {
        f"{control}_{index}": deep_fit_one_mlp_control(
            control, index, x, y, x_raw, y_raw, normalization
        )
        for control in ("clean", "shuffled")
        for index in range(3)
    }
    models["ridge"] = deep_fit_ridge_control(x, y, x_raw, y_raw, normalization)
    return {
        "models": models,
        "normalization": {name: value.detach().clone() for name, value in normalization.items()},
        "train_ids": train_ids,
        "train_x_raw": x_raw,
        "train_y_raw": y_raw,
        "units": dict(deep_UNITS),
        "feature_names": deep_FEATURE_NAMES,
        "model_labels": dict(deep_MODEL_LABELS),
        "phase": "models_fitted_on_train_only",
    }

# %%

if MAIN:
    deep_train = deep_load_split("train")
if MAIN:
    deep_explain = deep_load_split("explain")
if MAIN:
    deep_fitted = deep_fit_models(deep_train)

# %%

def _explain_model(
    item: Mapping[str, Any],
    x: t.Tensor,
    background: t.Tensor,
    methods: Any,
) -> dict[str, t.Tensor]:
    model = make_model(item)
    rows, donors = len(x), len(background)
    with t.no_grad():
        prediction = model(x).squeeze(-1)
        expected_base = model(background).squeeze(-1).mean()
    base, phi_deep = methods.background_deepshap(model, x, background)
    multipliers = methods.rescale_multipliers(model, x, background)
    pair_deep = multipliers * (x[:, None, :] - background[None, :, :])
    masks = methods.coalition_masks(deep_FEATURES)
    game = methods.coalition_values(model, x, background, masks)
    phi_exact = methods.exact_shapley(game)
    phi_gradient = methods.gradient_delta(model, x, background)

    prediction = deep__output_tensor(prediction, (rows,), "prediction")
    base = deep__output_tensor(base, (), "base")
    phi_deep = deep__output_tensor(phi_deep, (rows, deep_FEATURES), "phi_deep")
    pair_deep = deep__output_tensor(pair_deep, (rows, donors, deep_FEATURES), "pair_deep")
    game = deep__output_tensor(game, (rows, 2**deep_FEATURES), "game")
    phi_exact = deep__output_tensor(phi_exact, (rows, deep_FEATURES), "phi_exact")
    phi_gradient = deep__output_tensor(phi_gradient, (rows, deep_FEATURES), "phi_gradient")
    if not t.allclose(base, expected_base, atol=1e-10, rtol=1e-10):
        raise AssertionError("DeepSHAP base is not the mean background prediction.")
    if not t.allclose(phi_deep, pair_deep.mean(1), atol=1e-10, rtol=1e-10):
        raise AssertionError("DeepSHAP must average complete per-reference explanations.")
    if not t.allclose(game[:, 0], base.expand(rows), atol=1e-10, rtol=1e-10):
        raise AssertionError("The empty coalition does not equal the empirical background base.")
    if not t.allclose(game[:, -1], prediction, atol=1e-10, rtol=1e-10):
        raise AssertionError("The full coalition does not equal the raw model prediction.")
    return {
        "prediction": prediction,
        "base": base,
        "phi_deep": phi_deep,
        "phi_exact": phi_exact,
        "phi_gradient": phi_gradient,
        "game": game,
        "pair_deep": pair_deep,
    }

def deep_explain_block(
    fitted: Mapping[str, Any],
    rows: Mapping[str, Any],
    background: Any,
    methods: Any,
) -> dict[str, dict[str, t.Tensor]]:
    """Explain every retained model on one named row block in raw units."""
    deep__validate_fitted(fitted)
    _require_methods(methods)
    x, _, _ = deep__rows(rows, "rows")
    background_tensor = deep__float_tensor(background, "background", 2)
    if background_tensor.shape[1:] != (deep_FEATURES,) or len(background_tensor) == 0:
        raise ValueError("background must contain nonempty eight-feature donor rows.")
    return {
        key: _explain_model(fitted["models"][key], x, background_tensor, methods)
        for key in deep_MODEL_KEYS
    }

# %%

def deep__background_indices(train_ids: t.Tensor) -> dict[str, t.Tensor]:
    primary = t.from_numpy(deep_background_order(train_ids.numpy(), PRIMARY_BACKGROUND_SEED))
    independent_a = t.from_numpy(
        deep_background_order(train_ids.numpy(), INDEPENDENT_BACKGROUND_SEEDS[0])
    )
    independent_b = t.from_numpy(
        deep_background_order(train_ids.numpy(), INDEPENDENT_BACKGROUND_SEEDS[1])
    )
    if len(primary) < 128:
        raise ValueError("The frozen study requires at least 128 training donors.")
    return {
        "primary64": primary[:64].clone(),
        "nested16": primary[:16].clone(),
        "nested128": primary[:128].clone(),
        "independent64_a": independent_a[:64].clone(),
        "independent64_b": independent_b[:64].clone(),
    }

# %%

def deep_prepare_study(
    fitted: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    methods: Any,
) -> dict[str, Any]:
    """Explain and rank on the explanation split without receiving audit rows."""
    deep__validate_fitted(fitted)
    train_x, train_y, train_ids = deep__rows(train, "train")
    explain_x, explain_y, explain_ids = deep__rows(explain, "explain")
    deep__require_disjoint_ids(train=train_ids, explanation=explain_ids)
    if not t.equal(train_ids, fitted["train_ids"]):
        raise ValueError("Training identities changed after fitting.")
    if not t.equal(train_x, fitted["train_x_raw"]) or not t.equal(
        train_y, fitted["train_y_raw"]
    ):
        raise ValueError("Training values changed after fitting.")
    indices = deep__background_indices(train_ids)
    primary = train_x[indices["primary64"]]
    models = deep_explain_block(fitted, explain, primary, methods)
    rankings = {
        key: {
            "exact": t.argsort(
                models[key]["phi_exact"].abs().mean(0), descending=True, stable=True
            ),
            "deep": t.argsort(models[key]["phi_deep"].abs().mean(0), descending=True, stable=True),
            "gradient": t.argsort(
                models[key]["phi_gradient"].abs().mean(0), descending=True, stable=True
            ),
        }
        for key in deep_MODEL_KEYS
    }
    return {
        "models": models,
        "explain_ids": explain_ids,
        "explain_x_raw": explain_x,
        "explain_targets": explain_y,
        "background_ids": train_ids[indices["primary64"]],
        "background": primary,
        "background_indices": indices,
        "rankings": rankings,
        "audit_used_for_selection": False,
        "units": dict(deep_UNITS),
        "feature_names": deep_FEATURE_NAMES,
        "model_labels": dict(deep_MODEL_LABELS),
        "phase": "rankings_frozen_before_audit",
    }

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np
if MAIN:
    import torch

def deep_helper_run_native__serializable(value: Any) -> Any:
    """Copy numerical fit evidence while rejecting opaque live model objects."""
    if isinstance(value, torch.nn.Module):
        raise TypeError("Fitted evidence must store states, not live torch.nn.Module objects.")
    if isinstance(value, dict):
        return {key: deep_helper_run_native__serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deep_helper_run_native__serializable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(deep_helper_run_native__serializable(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return value

# %%

def deep_helper_run_native_serializable_fitted(fitted: dict) -> dict:
    """Return the full numerical fit payload without opaque PyTorch modules."""
    return deep_helper_run_native__serializable(fitted)

# %%

if MAIN:
    from typing import Any
if MAIN:
    from pathlib import Path
if MAIN:
    import torch

def deep_helper_run_native_save_payload(directory: Path, name: str, value: Any) -> Path:
    """Persist a named course-owned payload; fitted.pt contains states, not model objects."""
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("Payload names must be simple .pt filenames.")
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "fitted.pt":
        value = deep_helper_run_native_serializable_fitted(value)
    torch.save(value, path)
    return path

# %%

if MAIN:
    from pathlib import Path
if MAIN:
    import hashlib

def deep_helper_run_native_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

# %%

if MAIN:
    from typing import Any
if MAIN:
    from pathlib import Path
if MAIN:
    import hashlib
if MAIN:
    import json
if MAIN:
    import numpy as np

def deep_helper_run_native_write_before_audit(directory: Path, prepared: dict) -> dict[str, Any]:
    """Seal already-saved fitted/prepared payloads before the audit split is loaded."""
    directory = Path(directory)
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("Prepared rankings must explicitly exclude the audit split.")
    fitted_path = directory / "fitted.pt"
    prepared_path = directory / "prepared.pt"
    if not fitted_path.is_file() or not prepared_path.is_file():
        raise ValueError("Save fitted.pt and prepared.pt before sealing the audit boundary.")
    boundary = {
        "phase": "prepared_before_audit",
        "audit_loaded": False,
        "fitted_sha256": deep_helper_run_native_digest(fitted_path),
        "prepared_sha256": deep_helper_run_native_digest(prepared_path),
        "explain_ids_sha256": hashlib.sha256(
            np.asarray(prepared["explain_ids"]).tobytes()
        ).hexdigest(),
        "background_ids_sha256": hashlib.sha256(
            np.asarray(prepared["background_ids"]).tobytes()
        ).hexdigest(),
    }
    (directory / "before_audit.json").write_text(json.dumps(boundary, indent=2) + "\n")
    return boundary

# %%

if MAIN:
    deep_save_payload = deep_helper_run_native_save_payload
if MAIN:
    deep_write_before_audit = deep_helper_run_native_write_before_audit

if MAIN:
    native_dir = Path(deep_RUN_OUTPUT or canonical_dir / "artifacts" / "deep") / "native"
if MAIN:
    deep_prepared = deep_prepare_study(deep_fitted, deep_train, deep_explain, deep_curriculum_methods)
if MAIN:
    deep_save_payload(native_dir, "fitted.pt", deep_fitted)
if MAIN:
    deep_save_payload(native_dir, "prepared.pt", deep_prepared)
if MAIN:
    deep_write_before_audit(native_dir, deep_prepared)

# %%

if MAIN:
    deep_audit = deep_load_split("audit")

# %%

def _ranked_curves(
    game: t.Tensor, phi: t.Tensor, order: t.Tensor, methods: Any
) -> dict[str, t.Tensor]:
    expanded = order[None, :].expand(len(game), deep_FEATURES).clone()
    curve = methods.ranked_replacement_curves(game, phi, expanded)
    expected = {
        "orders": (len(game), deep_FEATURES),
        "insertion_masks": (len(game), deep_FEATURES + 1),
        "deletion_masks": (len(game), deep_FEATURES + 1),
        "insertion": (len(game), deep_FEATURES + 1),
        "deletion": (len(game), deep_FEATURES + 1),
    }
    if not isinstance(curve, Mapping) or set(curve) != set(expected):
        raise ValueError("Replacement curves returned an unexpected payload.")
    result: dict[str, t.Tensor] = {}
    for name, shape in expected.items():
        value = curve[name]
        if name.endswith("masks") or name == "orders":
            if not isinstance(value, t.Tensor) or value.dtype != t.int64:
                raise TypeError(f"{name} must be a long tensor.")
            if value.device.type != "cpu" or tuple(value.shape) != shape:
                raise ValueError(f"{name} has the wrong shape or device.")
            result[name] = value.detach().clone()
        else:
            result[name] = deep__output_tensor(value, shape, name)
    return result

# %%

def _random_curves(game: t.Tensor, orders: t.Tensor, methods: Any) -> dict[str, t.Tensor]:
    curves = [
        _ranked_curves(game, t.zeros((len(game), deep_FEATURES), dtype=t.float64), order, methods)
        for order in orders
    ]
    return {
        "orders": orders.clone(),
        **{
            name: t.stack([curve[name] for curve in curves])
            for name in ("insertion_masks", "deletion_masks", "insertion", "deletion")
        },
    }

# %%

def deep__performance_table(
    fitted: Mapping[str, Any],
    models: Mapping[str, Mapping[str, t.Tensor]],
    audit_y: t.Tensor,
) -> dict[str, dict[str, Any]]:
    """Retain audit MSE for every model and the train-only mean baseline."""
    performance = {
        key: {
            "kind": fitted["models"][key]["kind"],
            "control": fitted["models"][key]["control"],
            "mse": (models[key]["prediction"] - audit_y).square().mean(),
        }
        for key in deep_MODEL_KEYS
    }
    performance["training_mean"] = {
        "kind": "constant",
        "control": "training_mean",
        "mse": (audit_y - fitted["train_y_raw"].mean()).square().mean(),
    }
    return performance

def deep__fixed_random_orders() -> t.Tensor:
    """Draw the one frozen stream of 100 random feature-order controls."""
    rng = np.random.default_rng(deep_RANDOM_ORDER_SEED)
    return t.from_numpy(np.stack([rng.permutation(deep_FEATURES) for _ in range(100)]))

# %%

def _evaluate_replacements(
    models: Mapping[str, Mapping[str, t.Tensor]],
    rankings: Mapping[str, Mapping[str, t.Tensor]],
    random_orders: t.Tensor,
    methods: Any,
) -> dict[str, Any]:
    """Evaluate fixed global and random orders on every audit model and row."""
    replacement: dict[str, Any] = {}
    for key in deep_MODEL_KEYS:
        replacement[key] = {
            name: _ranked_curves(
                models[key]["game"], models[key][f"phi_{name}"], rankings[key][name], methods
            )
            for name in ("exact", "deep", "gradient")
        }
        replacement[key]["random"] = _random_curves(
            models[key]["game"], random_orders, methods
        )
    return replacement

# %%

def _evaluate_background_variants(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    audit_models: Mapping[str, Mapping[str, t.Tensor]],
    rows: Mapping[str, t.Tensor],
    combined: Mapping[str, t.Tensor],
    methods: Any,
) -> dict[str, Any]:
    """Compare clean_0 on all held-out rows under five prespecified backgrounds."""
    primary = _combine_explanations(
        prepared["models"]["clean_0"], audit_models["clean_0"]
    )
    variants: dict[str, Any] = {
        "primary64": {
            **primary,
            "background": prepared["background"].clone(),
            "background_ids": prepared["background_ids"].clone(),
        }
    }
    for name in ("nested16", "nested128", "independent64_a", "independent64_b"):
        indices = prepared["background_indices"][name]
        background = rows["train_x"][indices]
        variants[name] = {
            **_explain_model(fitted["models"]["clean_0"], combined["x"], background, methods),
            "background": background.clone(),
            "background_ids": rows["train_ids"][indices].clone(),
        }
    for result in variants.values():
        result.update(
            row_ids=combined["ids"].clone(),
            row_split=combined["split"].clone(),
            targets=combined["y"].clone(),
        )
    return variants

# %%

def deep_evaluate_study(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
    methods: Any,
) -> dict[str, Any]:
    """Evaluate frozen rankings, controls and backgrounds on the audit split."""
    _require_methods(methods)
    rows = deep__validate_evaluation_inputs(fitted, prepared, train, explain, audit)
    models = deep_explain_block(fitted, audit, prepared["background"], methods)
    performance = deep__performance_table(fitted, models, rows["audit_y"])
    random_orders = deep__fixed_random_orders()
    replacement = _evaluate_replacements(
        models, prepared["rankings"], random_orders, methods
    )
    combined = _combine_row_blocks(rows)
    background_variants = _evaluate_background_variants(
        fitted, prepared, models, rows, combined, methods
    )

    return {
        "models": models,
        "audit_ids": rows["audit_ids"],
        "audit_targets": rows["audit_y"],
        "performance": performance,
        "replacement_curves": replacement,
        "random_orders": random_orders,
        "background_variants": background_variants,
        "background_variant_order": (
            "primary64",
            "nested16",
            "nested128",
            "independent64_a",
            "independent64_b",
        ),
        "combined_explanation_audit_ids": combined["ids"],
        "combined_split": combined["split"],
        "combined_split_labels": ("explanation", "audit"),
        "model_order": deep_MODEL_KEYS,
        "model_labels": dict(deep_MODEL_LABELS),
        "feature_names": deep_FEATURE_NAMES,
        "units": dict(deep_UNITS),
        "audit_used_for_selection": False,
        "phase": "audit_evaluated_after_frozen_rankings",
    }

# %%

if MAIN:
    deep_evaluated = deep_evaluate_study(deep_fitted, deep_prepared, deep_train, deep_explain, deep_audit, deep_curriculum_methods)

# %%

def deep_native_play(
    fitted: Mapping[str, Any],
    train: Mapping[str, Any],
    audit: Mapping[str, Any],
    methods: Any,
    model_key: str = "clean_0",
    row: int = 0,
    size: int = 64,
) -> dict[str, Any]:
    """Recompute one audit recipe under an editable model and nested background."""
    deep__validate_fitted(fitted)
    _require_methods(methods)
    train_x, _, train_ids = deep__rows(train, "train")
    audit_x, audit_y, audit_ids = deep__rows(audit, "audit")
    if model_key not in deep_MODEL_KEYS:
        raise ValueError("Choose one of the seven retained model keys.")
    if not 0 <= row < len(audit_x):
        raise ValueError("Choose an existing audit row.")
    if size not in (16, 64, 128):
        raise ValueError("Choose a nested background size of 16, 64 or 128.")
    order = t.from_numpy(deep_background_order(train_ids.numpy(), PRIMARY_BACKGROUND_SEED))
    indices = order[:size]
    background = train_x[indices]
    x = audit_x[row : row + 1]
    result = _explain_model(fitted["models"][model_key], x, background, methods)
    curves = {
        name: _clone_tree(
            methods.ranked_replacement_curves(
                result["game"], result[f"phi_{name}"]
            )
        )
        for name in ("exact", "deep", "gradient")
    }
    return {
        "model_key": model_key,
        "source_id": int(audit_ids[row]),
        "query": x[0].clone(),
        "target": audit_y[row].clone(),
        "prediction": result["prediction"][0].clone(),
        "base": result["base"].clone(),
        "deep": result["phi_deep"][0].clone(),
        "exact": result["phi_exact"][0].clone(),
        "gradient": result["phi_gradient"][0].clone(),
        "game": result["game"][0].clone(),
        "pair_deep": result["pair_deep"][0].clone(),
        "curves": curves,
        "background_size": size,
        "background_ids": train_ids[indices].clone(),
        "background": background.clone(),
        "units": dict(deep_UNITS),
        "feature_names": deep_FEATURE_NAMES,
        "model_label": deep_MODEL_LABELS[model_key],
        "audit_reuse": "exploratory",
    }

# %%

deep_helper_run_native_DETERMINISTIC_ARTIFACTS = (
    "fitted.pt",
    "prepared.pt",
    "evaluated.pt",
    "diagnostics.pt",
    "reference.pt",
    "play.pt",
)

# %%

deep_helper_run_native_TIMING_ARTIFACT = "timings.pt"

# %%

if MAIN:
    from typing import Any

def deep_helper_run_native__loss_trace(item: dict) -> Any:
    """Return a model's retained optimizer trace under the canonical field names."""
    for name in ("loss_trace", "training_loss", "losses"):
        if name in item:
            return item[name]
    return ()

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np

def deep_helper_run_native__reference_scalars(value: Any, prefix: str = "") -> dict[str, int | float | bool | str]:
    """Flatten scalar reference outcomes for the JSON summary; arrays remain in reference.pt."""
    result: dict[str, int | float | bool | str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(deep_helper_run_native__reference_scalars(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (bool, str, int, float, np.bool_, np.integer, np.floating)):
        result[prefix] = value.item() if isinstance(value, np.generic) else value
    return result

# %%

if MAIN:
    from typing import Any

def deep_helper_run_native_summarize(
    fitted: dict,
    prepared: dict,
    evaluated: dict,
    diagnostics: dict,
    reference: dict,
) -> dict[str, Any]:
    """Build a compact index; complete arrays remain in the six deterministic payloads."""
    models = fitted["models"]
    keys = list(models)
    clean_keys = [key for key in keys if key.startswith("clean_")]
    shuffled_keys = [key for key in keys if key.startswith("shuffled_")]
    ridge_keys = [key for key in keys if key == "ridge" or models[key].get("kind") == "ridge"]
    mlp_keys = [key for key in keys if key not in ridge_keys]
    rows_per_model = len(prepared["explain_ids"]) + len(evaluated["audit_ids"])
    optimizer_updates = sum(len(deep_helper_run_native__loss_trace(models[key])) for key in mlp_keys)
    background_variants = evaluated.get("backgrounds", evaluated.get("background_variants", {}))
    replacement = evaluated.get("replacement", evaluated.get("replacement_curves", {}))
    return {
        "model_fits": len(models),
        "mlp_fits": len(mlp_keys),
        "ridge_fits": len(ridge_keys),
        "clean_fits": len(clean_keys),
        "shuffled_fits": len(shuffled_keys),
        "optimizer_updates": optimizer_updates,
        "model_keys": keys,
        "training_seeds": [161410, 161411, 161412],
        "train_rows": len(fitted["train_ids"]),
        "explanation_row_count": len(prepared["explain_ids"]),
        "heldout_row_count": len(evaluated["audit_ids"]),
        "rows_explained_per_model": rows_per_model,
        "model_row_explanations": len(models) * rows_per_model,
        "coalitions": 256,
        "exact_models": len(models),
        "exact_model_rows": len(models) * rows_per_model,
        "coalition_values": len(models) * rows_per_model * 256,
        "primary_hybrid_predictions": len(models)
        * rows_per_model
        * 256
        * len(prepared["background_ids"]),
        "attribution_coordinates_per_method": len(models) * rows_per_model * 8,
        "background_rows": len(prepared["background_ids"]),
        "background_variants": len(background_variants),
        "replacement_models": len(replacement),
        "random_order_controls": 100,
        "pair_model_rows": len(models) * rows_per_model * len(prepared["background_ids"]),
        "pair_local_values": len(models)
        * rows_per_model
        * len(prepared["background_ids"])
        * 8,
        "audit_used_for_selection": prepared["audit_used_for_selection"],
        "diagnostic_groups": sorted(diagnostics),
        "reference_scalars": deep_helper_run_native__reference_scalars(reference),
    }

# %%

if MAIN:
    from typing import Any
if MAIN:
    from pathlib import Path
if MAIN:
    import json

def deep_helper_run_native_persist_completed(
    directory: Path,
    fitted: dict,
    prepared: dict,
    evaluated: dict,
    diagnostics: dict,
    reference: dict,
    play: dict,
) -> dict[str, Any]:
    """Persist post-audit evidence, retaining timing in its own nondeterministic artifact."""
    directory = Path(directory)
    boundary_path = directory / "before_audit.json"
    if not boundary_path.exists():
        raise ValueError("Prepared evidence must be persisted before audit evaluation.")
    boundary = json.loads(boundary_path.read_text())
    if boundary.get("audit_loaded") is not False:
        raise ValueError("The recorded preparation boundary is invalid.")
    if deep_helper_run_native_digest(directory / "prepared.pt") != boundary.get("prepared_sha256"):
        raise ValueError("Prepared evidence changed after the audit boundary.")
    if diagnostics.keys() != {"mechanism", "timing"}:
        raise ValueError("Diagnostics must retain separate mechanism and timing branches.")
    deep_helper_run_native_save_payload(directory, "evaluated.pt", evaluated)
    deep_helper_run_native_save_payload(directory, "diagnostics.pt", diagnostics["mechanism"])
    deep_helper_run_native_save_payload(directory, deep_helper_run_native_TIMING_ARTIFACT, diagnostics["timing"])
    deep_helper_run_native_save_payload(directory, "reference.pt", reference)
    deep_helper_run_native_save_payload(directory, "play.pt", play)
    summary = deep_helper_run_native_summarize(fitted, prepared, evaluated, diagnostics["mechanism"], reference)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    expected = set(deep_helper_run_native_DETERMINISTIC_ARTIFACTS) | {deep_helper_run_native_TIMING_ARTIFACT}
    present = {path.name for path in directory.glob("*.pt")}
    if present != expected:
        raise ValueError(f"Native artifact set mismatch: {sorted(present)}")
    return summary

# %%

if MAIN:
    from pathlib import Path
if MAIN:
    import json

def deep_helper_run_native_write_notebook_summary(output_root: Path, native_directory: Path, summary: dict) -> Path:
    """Write the small JSON index consumed by the fresh-kernel executor."""
    output_root = Path(output_root).resolve()
    native_directory = Path(native_directory).resolve()
    native_directory.relative_to(output_root)
    payload = dict(summary, native_directory=str(native_directory))
    path = output_root / "native_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path

# %%

deep_helper_deepshap_reference_AUTHOR_ATOL = 1e-7

# %%

deep_helper_deepshap_reference_AUTHOR_RTOL = 1e-8

# %%

deep_helper_deepshap_reference_BACKGROUND_ROWS = 64

# %%

deep_helper_deepshap_reference_EXACT_ATOL = 1e-10

# %%

deep_helper_deepshap_reference_EXACT_RTOL = 1e-10

# %%

deep_helper_deepshap_reference_MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def deep_helper_deepshap_reference__array(value: Any) -> np.ndarray:
    if isinstance(value, t.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import hashlib
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def deep_helper_deepshap_reference__state_hash(state: Mapping[str, Any]) -> str:
    if not isinstance(state, Mapping):
        raise TypeError("A model state must be a tensor mapping.")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, t.Tensor):
            raise TypeError("State keys must be strings and values must be tensors.")
        if value.device.type != "cpu" or value.dtype != t.float64:
            raise TypeError("Every raw model-state tensor must be CPU float64.")
        array = value.detach().numpy()
        if not np.isfinite(array).all():
            raise ValueError("Raw model-state tensors must be finite.")
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).astype("<f8", copy=False).tobytes())
    return digest.hexdigest()

# %%

if MAIN:
    import torch as t

def deep_helper_deepshap_reference__model_hash(model: t.nn.Module) -> str:
    return deep_helper_deepshap_reference__state_hash(model.state_dict())

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np

def deep_helper_deepshap_reference__max_abs(actual: Any, expected: Any) -> float:
    difference = np.abs(deep_helper_deepshap_reference__array(actual).astype(np.float64) - deep_helper_deepshap_reference__array(expected).astype(np.float64))
    return float(difference.max(initial=0.0))

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np

def deep_helper_deepshap_reference__require_close(
    name: str,
    actual: Any,
    expected: Any,
    *,
    atol: float = deep_helper_deepshap_reference_EXACT_ATOL,
    rtol: float = deep_helper_deepshap_reference_EXACT_RTOL,
) -> float:
    actual_array = deep_helper_deepshap_reference__array(actual)
    expected_array = deep_helper_deepshap_reference__array(expected)
    if actual_array.shape != expected_array.shape or not np.allclose(
        actual_array, expected_array, atol=atol, rtol=rtol
    ):
        raise AssertionError(
            f"{name} failed full-value parity: shape {actual_array.shape} versus "
            f"{expected_array.shape}, max absolute error {deep_helper_deepshap_reference__max_abs(actual_array, expected_array):.12g}."
        )
    return deep_helper_deepshap_reference__max_abs(actual_array, expected_array)

# %%

if MAIN:
    import torch as t

def deep_helper_deepshap_reference__validate_reference_model(model: t.nn.Module) -> list[t.nn.Module]:
    """Enforce the exact, deliberately narrow author-parity model boundary."""
    if type(model) is not t.nn.Sequential:
        raise TypeError("model must be an exact nn.Sequential instance.")
    layers = list(model)
    if not layers or type(layers[-1]) is not t.nn.Linear or layers[-1].out_features != 1:
        raise ValueError("model must end with an exact scalar-output nn.Linear layer.")
    if len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("Shared modules are unsupported by the reference boundary.")
    previous_width: int | None = None
    for index, layer in enumerate(layers):
        if type(layer) not in (t.nn.Linear, t.nn.ReLU):
            raise TypeError("Only exact nn.Linear and nn.ReLU modules are supported.")
        if type(layer) is t.nn.ReLU:
            if layer.inplace:
                raise ValueError("In-place ReLU modules are unsupported.")
            if index == 0 or index == len(layers) - 1:
                raise ValueError("A ReLU must occur between affine layers.")
        else:
            if previous_width is not None and layer.in_features != previous_width:
                raise ValueError("Adjacent layer dimensions are incompatible.")
            previous_width = layer.out_features
        for parameter in layer.parameters(recurse=False):
            if parameter.device.type != "cpu" or parameter.dtype != t.float64:
                raise TypeError("Every model parameter must be CPU float64.")
            if not bool(t.isfinite(parameter).all()):
                raise ValueError("Every model parameter must be finite.")
    return layers

# %%

if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def deep_helper_deepshap_reference__author_deep(
    model: t.nn.Sequential, x: np.ndarray, background: np.ndarray
) -> dict[str, np.ndarray | float | str]:
    """Call source-pinned SHAP explicitly, retaining complete local vectors."""
    import shap

    deep_helper_deepshap_reference__validate_reference_model(model)
    x_tensor = t.as_tensor(np.array(x, copy=True), dtype=t.float64)
    background_tensor = t.as_tensor(np.array(background, copy=True), dtype=t.float64)
    state_before = deep_helper_deepshap_reference__model_hash(model)
    training_before = bool(model.training)
    explainer = shap.DeepExplainer(model, background_tensor)
    backend_data = getattr(explainer.explainer, "data", None)
    if not isinstance(backend_data, list) or len(backend_data) != 1:
        raise AssertionError("DeepExplainer did not retain one explicit background tensor.")
    if backend_data[0].shape != background_tensor.shape or not t.equal(
        backend_data[0], background_tensor
    ):
        raise AssertionError("DeepExplainer changed, summarized or reordered the background.")
    raw_phi = explainer.shap_values(x_tensor, check_additivity=False)
    if isinstance(raw_phi, list):
        raise AssertionError("The scalar-output reference unexpectedly returned multiple outputs.")
    phi = np.asarray(raw_phi, dtype=np.float64)
    if phi.shape == (len(x), x.shape[1], 1):
        phi = phi[..., 0]
    if phi.shape != x.shape or not np.isfinite(phi).all():
        raise AssertionError("DeepExplainer returned an unexpected local-value array.")
    expected = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
    if expected.shape != (1,) or not np.isfinite(expected).all():
        raise AssertionError("DeepExplainer returned an unexpected expected value.")
    with t.no_grad():
        prediction = model(x_tensor).squeeze(-1).detach().numpy().copy()
        direct_base = float(model(background_tensor).squeeze(-1).mean())
    state_after = deep_helper_deepshap_reference__model_hash(model)
    if state_after != state_before or bool(model.training) != training_before:
        raise AssertionError("SHAP DeepExplainer changed model parameters or training state.")
    base = float(expected[0])
    deep_helper_deepshap_reference__require_close("DeepExplainer expected base", np.array(base), np.array(direct_base))
    return {
        "prediction": prediction,
        "base": base,
        "phi": phi.copy(),
        "local_accuracy_residual": prediction - base - phi.sum(axis=1),
        "model_state_sha256": state_after,
    }

# %%

if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def deep_helper_deepshap_reference__bounded_pair_author(
    model: t.nn.Sequential, x: np.ndarray, background: np.ndarray
) -> dict[str, np.ndarray | float]:
    if x.shape != (1, background.shape[1]):
        raise ValueError("The bounded donor check requires exactly one query row.")
    pair_phi = np.empty((len(background), background.shape[1]), dtype=np.float64)
    pair_base = np.empty(len(background), dtype=np.float64)
    pair_residual = np.empty(len(background), dtype=np.float64)
    prediction: float | None = None
    for donor_index in range(len(background)):
        result = deep_helper_deepshap_reference__author_deep(model, x, background[donor_index : donor_index + 1])
        pair_phi[donor_index] = np.asarray(result["phi"])[0]
        pair_base[donor_index] = float(result["base"])
        pair_residual[donor_index] = np.asarray(result["local_accuracy_residual"])[0]
        current_prediction = float(np.asarray(result["prediction"])[0])
        if prediction is None:
            prediction = current_prediction
        elif current_prediction != prediction:
            raise AssertionError("The query prediction changed across single-donor explainers.")
    return {
        "prediction": float(prediction),
        "base": pair_base,
        "phi": pair_phi,
        "local_accuracy_residual": pair_residual,
    }

# %%

deep_helper_deepshap_reference_FEATURES = 8

# %%

deep_helper_deepshap_reference_HIDDEN = 32

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def deep_helper_deepshap_reference__model_from_item(item: Mapping[str, Any]) -> t.nn.Sequential:
    if not isinstance(item, Mapping) or item.get("kind") not in ("mlp", "ridge"):
        raise ValueError("Each fitted item must declare kind 'mlp' or 'ridge'.")
    state = item.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("Each fitted item must contain a raw tensor state.")
    if item["kind"] == "mlp":
        expected = {
            "0.weight": (deep_helper_deepshap_reference_HIDDEN, deep_helper_deepshap_reference_FEATURES),
            "0.bias": (deep_helper_deepshap_reference_HIDDEN,),
            "2.weight": (deep_helper_deepshap_reference_HIDDEN, deep_helper_deepshap_reference_HIDDEN),
            "2.bias": (deep_helper_deepshap_reference_HIDDEN,),
            "4.weight": (1, deep_helper_deepshap_reference_HIDDEN),
            "4.bias": (1,),
        }
        model = t.nn.Sequential(
            t.nn.Linear(deep_helper_deepshap_reference_FEATURES, deep_helper_deepshap_reference_HIDDEN, dtype=t.float64),
            t.nn.ReLU(),
            t.nn.Linear(deep_helper_deepshap_reference_HIDDEN, deep_helper_deepshap_reference_HIDDEN, dtype=t.float64),
            t.nn.ReLU(),
            t.nn.Linear(deep_helper_deepshap_reference_HIDDEN, 1, dtype=t.float64),
        )
    else:
        expected = {"0.weight": (1, deep_helper_deepshap_reference_FEATURES), "0.bias": (1,)}
        model = t.nn.Sequential(t.nn.Linear(deep_helper_deepshap_reference_FEATURES, 1, dtype=t.float64))
    if set(state) != set(expected):
        raise ValueError("Raw model state has unexpected parameter names.")
    copied: dict[str, t.Tensor] = {}
    for name, shape in expected.items():
        value = state[name]
        if not isinstance(value, t.Tensor) or value.device.type != "cpu" or value.dtype != t.float64:
            raise TypeError("Raw state values must be CPU float64 tensors.")
        if tuple(value.shape) != shape or not bool(t.isfinite(value).all()):
            raise ValueError(f"Raw state tensor {name!r} has the wrong shape or values.")
        copied[name] = value.detach().clone()
    model.load_state_dict(copied, strict=True)
    model.eval()
    deep_helper_deepshap_reference__validate_reference_model(model)
    return model

# %%

if MAIN:
    from pathlib import Path

if MAIN:
    deep_helper_deepshap_reference_HERE = deep_root_dir / 'chapter16_shapley_attribution_baselines' / "exercises" / 'part14_deepshap'

# %%

if MAIN:
    from pathlib import Path
if MAIN:
    import hashlib

def deep_helper_deepshap_reference__sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

# %%

if MAIN:
    from typing import Any
if MAIN:
    from pathlib import Path
if MAIN:
    import json
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def deep_helper_deepshap_reference__reference_environment() -> dict[str, Any]:
    """Verify the installed SHAP files against the author manifest by hash."""
    import shap
    import sklearn
    from shap.explainers import _deep

    manifest_path = deep_helper_deepshap_reference_HERE / "author_reference.json"
    manifest = json.loads(manifest_path.read_text())
    actual_versions = {
        "numpy": np.__version__,
        "torch": t.__version__,
        "scikit_learn": sklearn.__version__,
        "shap": shap.__version__,
    }
    # The author reference is pinned by its SHAP version and the source hashes below. numpy, torch
    # and scikit-learn are recorded, not required, so environment upgrades keep this check running.
    expected_versions = {"shap": "0.49.1"}
    if actual_versions["shap"] != expected_versions["shap"] or manifest.get("version") != "0.49.1":
        raise ValueError(f"Reference version mismatch: {actual_versions}.")
    expected_source_keys = {
        "shap/explainers/_deep/__init__.py",
        "shap/explainers/_deep/deep_pytorch.py",
        "shap/explainers/_deep/deep_utils.py",
        "LICENSE",
    }
    if set(manifest.get("sources", {})) != expected_source_keys:
        raise ValueError("The SHAP author-source manifest has unexpected entries.")
    # The installed SHAP sources are pinned below. The author's cached source copies, paper PDFs
    # and historical DeepLIFT sources are not needed to run the lesson, so they are not checked here.
    package_root = Path(shap.__file__).resolve().parent
    installed_hashes: dict[str, str] = {}
    for name, record in manifest["sources"].items():
        if name != "LICENSE":
            installed = package_root / record["installed_relative_path"]
            installed_digest = deep_helper_deepshap_reference__sha256(installed)
            if installed_digest != record["installed_sha256"] or installed_digest != record["sha256"]:
                raise ValueError(f"Installed SHAP source is not the pinned v0.49.1 file: {name}.")
            installed_hashes[name] = installed_digest
    return {
        "versions": actual_versions,
        "author_manifest_sha256": deep_helper_deepshap_reference__sha256(manifest_path),
        "installed_source_sha256": installed_hashes,
        "deep_module": str(Path(_deep.__file__).resolve()),
    }

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def deep_helper_deepshap_reference__float64_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if isinstance(value, t.Tensor):
        if value.device.type != "cpu" or value.dtype != t.float64:
            raise TypeError(f"{name} must be a CPU float64 tensor or float64 array.")
    array = deep_helper_deepshap_reference__array(value)
    if array.dtype != np.float64 or array.shape != shape:
        raise ValueError(f"{name} must have float64 shape {shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values.")
    return np.array(array, copy=True)

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import numpy as np

def deep_helper_deepshap_reference__validate_block(name: str, block: Any, rows: int) -> dict[str, dict[str, np.ndarray | float]]:
    if not isinstance(block, Mapping) or tuple(block) != deep_helper_deepshap_reference_MODEL_KEYS:
        raise ValueError(f"{name} must contain the seven models in frozen order.")
    shapes = {
        "prediction": (rows,),
        "base": (),
        "phi_deep": (rows, deep_helper_deepshap_reference_FEATURES),
        "phi_exact": (rows, deep_helper_deepshap_reference_FEATURES),
        "phi_gradient": (rows, deep_helper_deepshap_reference_FEATURES),
        "game": (rows, 2**deep_helper_deepshap_reference_FEATURES),
        "pair_deep": (rows, deep_helper_deepshap_reference_BACKGROUND_ROWS, deep_helper_deepshap_reference_FEATURES),
    }
    result: dict[str, dict[str, np.ndarray | float]] = {}
    for key in deep_helper_deepshap_reference_MODEL_KEYS:
        item = block[key]
        if not isinstance(item, Mapping):
            raise TypeError(f"{name}[{key!r}] must be a mapping.")
        converted: dict[str, np.ndarray | float] = {}
        for field, shape in shapes.items():
            array = deep_helper_deepshap_reference__float64_array(item.get(field), shape, f"{name}[{key}].{field}")
            converted[field] = float(array) if shape == () else array
        result[key] = converted
    return result

# %%

deep_helper_deepshap_reference_PRIMARY_BACKGROUND_SEED = 161460

# %%

if MAIN:
    import hashlib
if MAIN:
    import numpy as np

def deep_helper_deepshap_reference__background_indices(train_ids: np.ndarray) -> np.ndarray:
    return np.array(
        sorted(
            range(len(train_ids)),
            key=lambda index: hashlib.sha256(
                f"arena16.14/background/{deep_helper_deepshap_reference_PRIMARY_BACKGROUND_SEED}/{int(train_ids[index])}".encode()
            ).digest(),
        ),
        dtype=np.int64,
    )[:deep_helper_deepshap_reference_BACKGROUND_ROWS]

# %%

if MAIN:
    import json
if MAIN:
    import numpy as np
if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.deep_deepshap_data import source_data as deep_plumbing_deepshap_data_source_data

def deep_helper_deepshap_reference__frozen_identity() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    manifest = json.loads((deep_helper_deepshap_reference_HERE / "data_manifest.json").read_text())
    data = deep_plumbing_deepshap_data_source_data()
    ids = {
        name: np.asarray(manifest["splits"][name], dtype=np.int64)
        for name in ("train", "explain", "audit")
    }
    if data.shape != (1030, 9) or {name: len(value) for name, value in ids.items()} != {
        "train": 758,
        "explain": 153,
        "audit": 119,
    }:
        raise ValueError("The frozen Concrete source or split sizes changed.")
    return data, ids

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def deep_helper_deepshap_reference__int64_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if isinstance(value, t.Tensor):
        if value.device.type != "cpu" or value.dtype != t.int64:
            raise TypeError(f"{name} must be a CPU int64 tensor or int64 array.")
    array = deep_helper_deepshap_reference__array(value)
    if array.dtype != np.int64 or array.shape != shape:
        raise ValueError(f"{name} must have int64 shape {shape}.")
    return np.array(array, copy=True)

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import numpy as np

def deep_helper_deepshap_reference__validate_rows(name: str, supplied: Mapping[str, Any], data: np.ndarray, ids: np.ndarray) -> None:
    if not isinstance(supplied, Mapping):
        raise TypeError(f"{name} must be a row mapping.")
    actual_ids = deep_helper_deepshap_reference__int64_array(supplied.get("ids"), (len(ids),), f"{name}.ids")
    x = deep_helper_deepshap_reference__float64_array(supplied.get("x"), (len(ids), deep_helper_deepshap_reference_FEATURES), f"{name}.x")
    y = deep_helper_deepshap_reference__float64_array(supplied.get("y"), (len(ids),), f"{name}.y")
    if not np.array_equal(actual_ids, ids):
        raise ValueError(f"{name} row identity/order does not match the frozen manifest.")
    if not np.array_equal(x, data[ids, :deep_helper_deepshap_reference_FEATURES]) or not np.array_equal(y, data[ids, deep_helper_deepshap_reference_FEATURES]):
        raise ValueError(f"{name} values do not match their frozen source rows.")

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import numpy as np

def deep_helper_deepshap_reference__validate_identity(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    data, ids = deep_helper_deepshap_reference__frozen_identity()
    for name, supplied in (("train", train), ("explain", explain), ("audit", audit)):
        deep_helper_deepshap_reference__validate_rows(name, supplied, data, ids[name])
    fitted_ids = deep_helper_deepshap_reference__int64_array(fitted.get("train_ids"), (758,), "fitted.train_ids")
    fitted_x = deep_helper_deepshap_reference__float64_array(fitted.get("train_x_raw"), (758, deep_helper_deepshap_reference_FEATURES), "fitted.train_x_raw")
    fitted_y = deep_helper_deepshap_reference__float64_array(fitted.get("train_y_raw"), (758,), "fitted.train_y_raw")
    if not np.array_equal(fitted_ids, ids["train"]):
        raise ValueError("Fitted training row identity/order changed.")
    if not np.array_equal(fitted_x, data[ids["train"], :deep_helper_deepshap_reference_FEATURES]) or not np.array_equal(
        fitted_y, data[ids["train"], deep_helper_deepshap_reference_FEATURES]
    ):
        raise ValueError("Fitted training values changed.")
    if not np.array_equal(
        deep_helper_deepshap_reference__int64_array(prepared.get("explain_ids"), (153,), "prepared.explain_ids"), ids["explain"]
    ):
        raise ValueError("Prepared explanation row identity/order changed.")
    if not np.array_equal(
        deep_helper_deepshap_reference__float64_array(prepared.get("explain_x_raw"), (153, deep_helper_deepshap_reference_FEATURES), "prepared.explain_x_raw"),
        data[ids["explain"], :deep_helper_deepshap_reference_FEATURES],
    ):
        raise ValueError("Prepared explanation values changed.")
    if not np.array_equal(
        deep_helper_deepshap_reference__float64_array(prepared.get("explain_targets"), (153,), "prepared.explain_targets"),
        data[ids["explain"], deep_helper_deepshap_reference_FEATURES],
    ):
        raise ValueError("Prepared explanation targets changed.")
    if not np.array_equal(
        deep_helper_deepshap_reference__int64_array(evaluated.get("audit_ids"), (119,), "evaluated.audit_ids"), ids["audit"]
    ):
        raise ValueError("Evaluated audit row identity/order changed.")
    if not np.array_equal(
        deep_helper_deepshap_reference__float64_array(evaluated.get("audit_targets"), (119,), "evaluated.audit_targets"),
        data[ids["audit"], deep_helper_deepshap_reference_FEATURES],
    ):
        raise ValueError("Evaluated audit targets changed.")
    background_indices = deep_helper_deepshap_reference__background_indices(ids["train"])
    background_ids = ids["train"][background_indices]
    background = data[background_ids, :deep_helper_deepshap_reference_FEATURES]
    actual_ids = deep_helper_deepshap_reference__int64_array(prepared.get("background_ids"), (deep_helper_deepshap_reference_BACKGROUND_ROWS,), "background_ids")
    actual_background = deep_helper_deepshap_reference__float64_array(
        prepared.get("background"), (deep_helper_deepshap_reference_BACKGROUND_ROWS, deep_helper_deepshap_reference_FEATURES), "background"
    )
    if not np.array_equal(actual_ids, background_ids):
        raise ValueError("Primary background IDs/order are not the frozen 64 donors.")
    if not np.array_equal(actual_background, background):
        raise ValueError("Primary background values are not all 64 frozen float64 donors.")
    supplied_indices = prepared.get("background_indices", {}).get("primary64")
    if not np.array_equal(
        deep_helper_deepshap_reference__int64_array(supplied_indices, (deep_helper_deepshap_reference_BACKGROUND_ROWS,), "background_indices.primary64"),
        background_indices,
    ):
        raise ValueError("Primary background positions changed.")
    primary = evaluated.get("background_variants", {}).get("primary64", {})
    if not np.array_equal(
        deep_helper_deepshap_reference__int64_array(primary.get("background_ids"), (deep_helper_deepshap_reference_BACKGROUND_ROWS,), "primary64.background_ids"),
        background_ids,
    ) or not np.array_equal(
        deep_helper_deepshap_reference__float64_array(primary.get("background"), (deep_helper_deepshap_reference_BACKGROUND_ROWS, deep_helper_deepshap_reference_FEATURES), "primary64.background"),
        background,
    ):
        raise ValueError("Evaluated primary background changed.")
    combined_ids = np.concatenate([ids["explain"], ids["audit"]])
    combined_split = np.concatenate(
        [np.zeros(153, dtype=np.int64), np.ones(119, dtype=np.int64)]
    )
    for owner, supplied_ids, supplied_split in (
        (
            "evaluated",
            evaluated.get("combined_explanation_audit_ids"),
            evaluated.get("combined_split"),
        ),
        ("primary64", primary.get("row_ids"), primary.get("row_split")),
    ):
        if not np.array_equal(
            deep_helper_deepshap_reference__int64_array(supplied_ids, (272,), f"{owner}.combined_ids"), combined_ids
        ) or not np.array_equal(
            deep_helper_deepshap_reference__int64_array(supplied_split, (272,), f"{owner}.combined_split"), combined_split
        ):
            raise ValueError(f"{owner} combined row identity/order changed.")
    return {
        "train_ids": ids["train"].copy(),
        "explain_ids": ids["explain"].copy(),
        "audit_ids": ids["audit"].copy(),
        "background_indices": background_indices,
        "background_ids": background_ids,
        "background": background.copy(),
        "explain_x": data[ids["explain"], :deep_helper_deepshap_reference_FEATURES].copy(),
        "audit_x": data[ids["audit"], :deep_helper_deepshap_reference_FEATURES].copy(),
    }

# %%

if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def deep_helper_deepshap_reference__coalition_game(model: t.nn.Sequential, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    features = x.shape[1]
    masks = np.array(
        [[bool(mask & (1 << feature)) for feature in range(features)] for mask in range(2**features)]
    )
    values = np.empty((len(x), len(masks)), dtype=np.float64)
    with t.no_grad():
        for row_index, row in enumerate(x):
            hybrids = np.where(masks[:, None, :], row[None, None, :], background[None, :, :])
            prediction = model(t.as_tensor(hybrids.reshape(-1, features), dtype=t.float64))
            values[row_index] = prediction.reshape(len(masks), len(background)).mean(1).numpy()
    return values

# %%

if MAIN:
    import torch as t

def deep_helper_deepshap_reference__linear_model(weight: list[list[float]], bias: list[float]) -> t.nn.Sequential:
    model = t.nn.Sequential(t.nn.Linear(len(weight[0]), len(weight), dtype=t.float64))
    with t.no_grad():
        model[0].weight.copy_(t.tensor(weight, dtype=t.float64))
        model[0].bias.copy_(t.tensor(bias, dtype=t.float64))
    model.eval()
    return model

# %%

if MAIN:
    import torch as t

def deep_helper_deepshap_reference__relu_model(
    first_weight: list[list[float]], first_bias: list[float], final_weight: list[float]
) -> t.nn.Sequential:
    model = t.nn.Sequential(
        t.nn.Linear(len(first_weight[0]), len(first_weight), dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(len(first_weight), 1, dtype=t.float64),
    )
    with t.no_grad():
        model[0].weight.copy_(t.tensor(first_weight, dtype=t.float64))
        model[0].bias.copy_(t.tensor(first_bias, dtype=t.float64))
        model[2].weight.copy_(t.tensor([final_weight], dtype=t.float64))
        model[2].bias.zero_()
    model.eval()
    return model

# %%

if MAIN:
    from typing import Any
if MAIN:
    import itertools
if MAIN:
    import numpy as np

def deep_helper_deepshap_reference_permutation_reference(values: Any) -> tuple[np.ndarray, int]:
    """Average marginal contributions over every feature permutation.

    This deliberately slow definition is used only for small hand games.  It is
    algorithmically separate from the subset-weight contraction used for the
    real eight-feature games.
    """
    game = np.asarray(values, dtype=np.float64)
    if game.ndim < 1 or not np.isfinite(game).all():
        raise ValueError("values must be a finite array with a coalition axis.")
    size = game.shape[-1]
    features = size.bit_length() - 1
    if features < 1 or size != 2**features:
        raise ValueError("The final axis must contain all 2**D coalitions for D >= 1.")
    permutations = tuple(itertools.permutations(range(features)))
    result = np.zeros((*game.shape[:-1], features), dtype=np.float64)
    for order in permutations:
        before = 0
        for feature in order:
            after = before | (1 << feature)
            result[..., feature] += game[..., after] - game[..., before]
            before = after
    result /= len(permutations)
    return result, len(permutations)

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np

def deep_helper_deepshap_reference_ridge_closed_form(
    weight: Any, bias: Any, x: Any, background: Any
) -> dict[str, np.ndarray | float]:
    """Analytic empirical-background SHAP values for one affine scalar model."""
    x_array = np.asarray(x, dtype=np.float64)
    background_array = np.asarray(background, dtype=np.float64)
    weight_array = np.asarray(weight, dtype=np.float64).reshape(-1)
    bias_value = float(np.asarray(bias, dtype=np.float64).reshape(-1)[0])
    if x_array.ndim != 2 or background_array.ndim != 2:
        raise ValueError("x and background must be two-dimensional.")
    if x_array.shape[1] != background_array.shape[1] or x_array.shape[1] != len(weight_array):
        raise ValueError("Affine inputs, donors and weights have incompatible shapes.")
    if not all(np.isfinite(value).all() for value in (x_array, background_array, weight_array)):
        raise ValueError("Affine reference inputs must be finite.")
    mean_background = background_array.mean(axis=0)
    base = float(mean_background @ weight_array + bias_value)
    prediction = x_array @ weight_array + bias_value
    phi = (x_array - mean_background) * weight_array
    return {"base": base, "prediction": prediction, "phi": phi}

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def deep_helper_deepshap_reference_known_diagnostics() -> dict[str, Any]:
    """Run the predeclared hand cases through independent algebra and SHAP."""
    threshold = deep_helper_deepshap_reference__relu_model([[2.0, 1.0]], [-2.0], [1.0])
    one = np.array([[1.0, 1.0]], dtype=np.float64)
    zero = np.array([[0.0, 0.0]], dtype=np.float64)
    threshold_game = deep_helper_deepshap_reference__coalition_game(threshold, one, zero)
    threshold_exact, threshold_permutations = deep_helper_deepshap_reference_permutation_reference(threshold_game)
    threshold_author = deep_helper_deepshap_reference__author_deep(threshold, one, zero)
    deep_helper_deepshap_reference__require_close("threshold coalition table", threshold_game, [[0.0, 0.0, 0.0, 1.0]])
    deep_helper_deepshap_reference__require_close("threshold exact Shapley", threshold_exact, [[0.5, 0.5]])
    deep_helper_deepshap_reference__require_close("threshold author DeepSHAP", threshold_author["phi"], [[2 / 3, 1 / 3]])

    minimum = deep_helper_deepshap_reference__relu_model([[0.0, 1.0], [-1.0, 1.0]], [0.0, 0.0], [1.0, -1.0])
    mirrored = deep_helper_deepshap_reference__relu_model([[1.0, 0.0], [1.0, -1.0]], [0.0, 0.0], [1.0, -1.0])
    min_game = deep_helper_deepshap_reference__coalition_game(minimum, one, zero)
    mirrored_game = deep_helper_deepshap_reference__coalition_game(mirrored, one, zero)
    min_exact, min_permutations = deep_helper_deepshap_reference_permutation_reference(min_game)
    mirrored_exact, mirrored_permutations = deep_helper_deepshap_reference_permutation_reference(mirrored_game)
    min_author = deep_helper_deepshap_reference__author_deep(minimum, one, zero)
    mirrored_author = deep_helper_deepshap_reference__author_deep(mirrored, one, zero)
    binary_inputs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    with t.no_grad():
        min_table = minimum(t.as_tensor(binary_inputs, dtype=t.float64)).squeeze(-1).numpy()
        mirrored_table = mirrored(t.as_tensor(binary_inputs, dtype=t.float64)).squeeze(-1).numpy()
    expected_min_table = binary_inputs.min(axis=1)
    deep_helper_deepshap_reference__require_close("minimum binary domain", min_table, expected_min_table)
    deep_helper_deepshap_reference__require_close("mirrored minimum binary domain", mirrored_table, expected_min_table)
    deep_helper_deepshap_reference__require_close("minimum exact Shapley", min_exact, [[0.5, 0.5]])
    deep_helper_deepshap_reference__require_close("mirrored exact Shapley", mirrored_exact, [[0.5, 0.5]])
    deep_helper_deepshap_reference__require_close("minimum author DeepSHAP", min_author["phi"], [[0.0, 1.0]])
    deep_helper_deepshap_reference__require_close("mirrored author DeepSHAP", mirrored_author["phi"], [[1.0, 0.0]])

    nearzero = deep_helper_deepshap_reference__relu_model([[1.0]], [0.0], [1.0])
    nearzero_x = np.array([[4e-7]], dtype=np.float64)
    nearzero_background = np.array([[-4e-7]], dtype=np.float64)
    nearzero_author = deep_helper_deepshap_reference__author_deep(nearzero, nearzero_x, nearzero_background)
    nearzero_delta = np.array([4e-7], dtype=np.float64)
    deep_helper_deepshap_reference__require_close("near-zero guarded author value", nearzero_author["phi"], [[0.0]])
    nearzero_passed = bool(
        np.allclose(
            np.asarray(nearzero_author["base"]) + np.asarray(nearzero_author["phi"]).sum(1),
            np.asarray(nearzero_author["prediction"]),
            atol=deep_helper_deepshap_reference_AUTHOR_ATOL,
            rtol=deep_helper_deepshap_reference_AUTHOR_RTOL,
        )
    )
    if nearzero_passed:
        raise AssertionError("The predeclared near-zero conservation failure disappeared.")

    affine = deep_helper_deepshap_reference__linear_model([[2.0, -3.0]], [5.0])
    affine_x = np.array([[4.0, 2.0]], dtype=np.float64)
    affine_background = np.array([[1.0, -1.0], [2.0, 0.0]], dtype=np.float64)
    affine_analytic = deep_helper_deepshap_reference_ridge_closed_form([[2.0, -3.0]], [5.0], affine_x, affine_background)
    affine_author = deep_helper_deepshap_reference__author_deep(affine, affine_x, affine_background)
    affine_game = deep_helper_deepshap_reference__coalition_game(affine, affine_x, affine_background)
    affine_exact, affine_permutations = deep_helper_deepshap_reference_permutation_reference(affine_game)
    deep_helper_deepshap_reference__require_close("affine base", np.array(affine_analytic["base"]), np.array(9.5))
    deep_helper_deepshap_reference__require_close("affine contribution", affine_analytic["phi"], [[5.0, -7.5]])
    deep_helper_deepshap_reference__require_close("affine author", affine_author["phi"], affine_analytic["phi"])
    deep_helper_deepshap_reference__require_close("affine exact", affine_exact, affine_analytic["phi"])

    background_relu = deep_helper_deepshap_reference__relu_model([[1.0, 1.0]], [0.0], [1.0])
    background_x = np.array([[2.0, 1.0]], dtype=np.float64)
    background_two = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float64)
    background_author = deep_helper_deepshap_reference__author_deep(background_relu, background_x, background_two)
    background_game = deep_helper_deepshap_reference__coalition_game(background_relu, background_x, background_two)
    background_exact, background_permutations = deep_helper_deepshap_reference_permutation_reference(background_game)
    deep_helper_deepshap_reference__require_close("two-background author", background_author["phi"], [[1.5, 1.5]])
    deep_helper_deepshap_reference__require_close("two-background exact", background_exact, [[1.5, 1.5]])

    constant = deep_helper_deepshap_reference__linear_model([[0.0, 0.0]], [3.0])
    constant_author = deep_helper_deepshap_reference__author_deep(constant, affine_x, affine_background)
    deep_helper_deepshap_reference__require_close("constant author", constant_author["phi"], [[0.0, 0.0]])
    dummy = deep_helper_deepshap_reference__linear_model([[2.0, -3.0, 0.0]], [5.0])
    dummy_x = np.array([[4.0, 2.0, 99.0]], dtype=np.float64)
    dummy_background = np.array([[1.0, -1.0, -10.0], [2.0, 0.0, 30.0]], dtype=np.float64)
    dummy_author = deep_helper_deepshap_reference__author_deep(dummy, dummy_x, dummy_background)
    dummy_game = deep_helper_deepshap_reference__coalition_game(dummy, dummy_x, dummy_background)
    dummy_exact, dummy_permutations = deep_helper_deepshap_reference_permutation_reference(dummy_game)
    deep_helper_deepshap_reference__require_close("dummy author", dummy_author["phi"][:, 2], [0.0])
    deep_helper_deepshap_reference__require_close("dummy exact", dummy_exact[:, 2], [0.0])

    return {
        "threshold": {
            "game": threshold_game,
            "exact": threshold_exact,
            "deep_explainer": np.asarray(threshold_author["phi"]),
            "single_feature_necessity": np.array([[1.0, 1.0]]),
            "permutation_count": threshold_permutations,
        },
        "minimum": {
            "binary_inputs": binary_inputs,
            "minimum_outputs": min_table,
            "mirrored_outputs": mirrored_table,
            "game": min_game,
            "mirrored_game": mirrored_game,
            "exact": min_exact,
            "mirrored_exact": mirrored_exact,
            "deep_explainer": np.asarray(min_author["phi"]),
            "mirrored_deep_explainer": np.asarray(mirrored_author["phi"]),
            "representations_equal_on_binary_domain": True,
            "permutation_counts": np.array([min_permutations, mirrored_permutations]),
        },
        "nearzero": {
            "x": nearzero_x,
            "background": nearzero_background,
            "prediction_delta": nearzero_delta,
            "deep_explainer": np.asarray(nearzero_author["phi"]),
            "epsilon_zero_secant": nearzero_delta[:, None],
            "local_accuracy_residual": np.asarray(nearzero_author["local_accuracy_residual"]),
            "conservation_passed_at_primary_tolerance": nearzero_passed,
            "expected_failure_disclosed": not nearzero_passed,
        },
        "affine": {
            "base": float(affine_analytic["base"]),
            "prediction": np.asarray(affine_analytic["prediction"]),
            "analytic": np.asarray(affine_analytic["phi"]),
            "deep_explainer": np.asarray(affine_author["phi"]),
            "exact": affine_exact,
            "game": affine_game,
            "permutation_count": affine_permutations,
        },
        "two_background_relu": {
            "game": background_game,
            "deep_explainer": np.asarray(background_author["phi"]),
            "exact": background_exact,
            "permutation_count": background_permutations,
        },
        "constant": {
            "deep_explainer": np.asarray(constant_author["phi"]),
            "local_accuracy_residual": np.asarray(constant_author["local_accuracy_residual"]),
        },
        "dummy": {
            "game": dummy_game,
            "deep_explainer": np.asarray(dummy_author["phi"]),
            "exact": dummy_exact,
            "permutation_count": dummy_permutations,
        },
    }

# %%

if MAIN:
    from typing import Any
if MAIN:
    import math
if MAIN:
    import numpy as np

def deep_helper_deepshap_reference_subset_reference(values: Any) -> np.ndarray:
    """Contract complete games with independently constructed Shapley weights."""
    game = np.asarray(values, dtype=np.float64)
    if game.ndim < 1 or not np.isfinite(game).all():
        raise ValueError("values must be a finite array with a coalition axis.")
    size = game.shape[-1]
    features = size.bit_length() - 1
    if features < 1 or size != 2**features:
        raise ValueError("The final axis must contain all 2**D coalitions for D >= 1.")
    result = np.zeros((*game.shape[:-1], features), dtype=np.float64)
    subset_ids = np.arange(size, dtype=np.int64)
    for feature in range(features):
        absent = subset_ids[(subset_ids & (1 << feature)) == 0]
        cardinality = np.fromiter((int(mask).bit_count() for mask in absent), dtype=np.int64)
        weights = np.array(
            [1.0 / (features * math.comb(features - 1, int(k))) for k in cardinality],
            dtype=np.float64,
        )
        result[..., feature] = (
            (game[..., absent | (1 << feature)] - game[..., absent]) * weights
        ).sum(axis=-1)
    return result

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def deep_helper_deepshap_reference_validate_reference(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return complete independent author/algebraic references.

    Author parity covers every averaged local vector.  Donor-level author
    parity is separately and accurately labelled as the two predeclared rows
    (first explanation and first audit row) for every model and all 64 donors.
    Neural DeepSHAP-versus-exact differences are returned as measurements and
    are never used as an acceptance threshold.
    """
    environment = deep_helper_deepshap_reference__reference_environment()
    if not isinstance(fitted, Mapping) or fitted.get("phase") != "models_fitted_on_train_only":
        raise ValueError("Use the complete train-only fitted payload.")
    if not isinstance(fitted.get("models"), Mapping) or tuple(fitted["models"]) != deep_helper_deepshap_reference_MODEL_KEYS:
        raise ValueError("The fitted payload must retain all seven models in frozen order.")
    if prepared.get("phase") != "rankings_frozen_before_audit" or evaluated.get(
        "phase"
    ) != "audit_evaluated_after_frozen_rankings":
        raise ValueError("Use the prepared and evaluated payloads from the frozen study phases.")
    if prepared.get("audit_used_for_selection") is not False or evaluated.get(
        "audit_used_for_selection"
    ) is not False:
        raise ValueError("Audit rows must not select models or rankings.")
    identity = deep_helper_deepshap_reference__validate_identity(fitted, prepared, evaluated, train, explain, audit)
    explanation = deep_helper_deepshap_reference__validate_block("prepared.models", prepared.get("models"), 153)
    audit_block = deep_helper_deepshap_reference__validate_block("evaluated.models", evaluated.get("models"), 119)
    background = identity["background"]
    combined_x = np.concatenate([identity["explain_x"], identity["audit_x"]], axis=0)
    split_slices = {"explanation": slice(0, 153), "audit": slice(153, 272)}
    stored_blocks = {"explanation": explanation, "audit": audit_block}
    row_ids = {"explanation": identity["explain_ids"], "audit": identity["audit_ids"]}

    references: dict[str, dict[str, Any]] = {"explanation": {}, "audit": {}}
    bounded_pairs: dict[str, dict[str, Any]] = {"explanation": {}, "audit": {}}
    descriptive: dict[str, dict[str, Any]] = {"explanation": {}, "audit": {}}
    per_model_errors: dict[str, dict[str, float]] = {}
    ridge_reference: dict[str, Any] = {}
    state_hashes: dict[str, str] = {}
    maxima: dict[str, float] = {
        "prediction": 0.0,
        "base": 0.0,
        "author_phi_deep": 0.0,
        "author_local_accuracy": 0.0,
        "own_local_accuracy": 0.0,
        "own_pair_mean": 0.0,
        "bounded_pair_phi": 0.0,
        "bounded_pair_base": 0.0,
        "bounded_pair_local_accuracy": 0.0,
        "bounded_pair_prediction": 0.0,
        "bounded_pair_average": 0.0,
        "exact_phi": 0.0,
        "exact_efficiency": 0.0,
        "ridge_analytic_prediction": 0.0,
        "ridge_analytic_base": 0.0,
        "ridge_analytic_phi": 0.0,
    }

    for key in deep_helper_deepshap_reference_MODEL_KEYS:
        item = fitted["models"][key]
        model = deep_helper_deepshap_reference__model_from_item(item)
        item_state_before = deep_helper_deepshap_reference__state_hash(item["state"])
        state_hashes[key] = item_state_before
        author = deep_helper_deepshap_reference__author_deep(model, combined_x, background)
        if author["model_state_sha256"] != deep_helper_deepshap_reference__model_hash(model):
            raise AssertionError("Author reference changed a reconstructed model state.")
        model_errors = {name: 0.0 for name in maxima}

        for split, row_slice in split_slices.items():
            stored = stored_blocks[split][key]
            author_prediction = np.asarray(author["prediction"])[row_slice].copy()
            author_phi = np.asarray(author["phi"])[row_slice].copy()
            author_residual = np.asarray(author["local_accuracy_residual"])[row_slice].copy()
            stored_prediction = np.asarray(stored["prediction"])
            stored_base = float(stored["base"])
            stored_phi = np.asarray(stored["phi_deep"])
            stored_pair = np.asarray(stored["pair_deep"])
            game = np.asarray(stored["game"])
            exact = deep_helper_deepshap_reference_subset_reference(game)
            exact_residual = game[:, -1] - game[:, 0] - exact.sum(axis=1)
            own_residual = stored_prediction - stored_base - stored_phi.sum(axis=1)

            errors = {
                "prediction": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} raw prediction", author_prediction, stored_prediction
                ),
                "base": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} expected base",
                    np.array(author["base"]),
                    np.array(stored_base),
                ),
                "author_phi_deep": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} author DeepSHAP",
                    author_phi,
                    stored_phi,
                    atol=deep_helper_deepshap_reference_AUTHOR_ATOL,
                    rtol=deep_helper_deepshap_reference_AUTHOR_RTOL,
                ),
                "author_local_accuracy": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} author local accuracy",
                    float(author["base"]) + author_phi.sum(axis=1),
                    author_prediction,
                    atol=deep_helper_deepshap_reference_AUTHOR_ATOL,
                    rtol=deep_helper_deepshap_reference_AUTHOR_RTOL,
                ),
                "own_local_accuracy": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} learner local accuracy",
                    stored_base + stored_phi.sum(axis=1),
                    stored_prediction,
                    atol=deep_helper_deepshap_reference_AUTHOR_ATOL,
                    rtol=deep_helper_deepshap_reference_AUTHOR_RTOL,
                ),
                "own_pair_mean": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} complete pair mean", stored_pair.mean(axis=1), stored_phi
                ),
                "exact_phi": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} exact coalition contraction", exact, stored["phi_exact"]
                ),
                "exact_efficiency": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} exact-game efficiency",
                    game[:, 0] + exact.sum(axis=1),
                    game[:, -1],
                ),
            }
            deep_helper_deepshap_reference__require_close(f"{split} {key} empty coalition", game[:, 0], np.full(len(game), stored_base))
            deep_helper_deepshap_reference__require_close(f"{split} {key} full coalition", game[:, -1], stored_prediction)

            reference_item = {
                "prediction": author_prediction,
                "base_deep_explainer": float(author["base"]),
                "phi_deep_explainer": author_phi,
                "local_accuracy_residual": author_residual,
                "phi_exact_independent": exact,
                "exact_efficiency_residual": exact_residual,
                "deep_minus_exact": author_phi - exact,
            }
            references[split][key] = reference_item
            difference = author_phi - exact
            descriptive[split][key] = {
                "deep_minus_exact": difference.copy(),
                "max_abs": float(np.abs(difference).max(initial=0.0)),
                "mean_abs": float(np.abs(difference).mean()),
                "rmse": float(np.sqrt(np.square(difference).mean())),
                "is_acceptance_gate": False,
            }

            pair = deep_helper_deepshap_reference__bounded_pair_author(
                model,
                (identity["explain_x"] if split == "explanation" else identity["audit_x"])[0:1],
                background,
            )
            pair_errors = {
                "bounded_pair_phi": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} bounded donor-level DeepSHAP",
                    pair["phi"],
                    stored_pair[0],
                    atol=deep_helper_deepshap_reference_AUTHOR_ATOL,
                    rtol=deep_helper_deepshap_reference_AUTHOR_RTOL,
                ),
                "bounded_pair_base": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} bounded donor bases",
                    pair["base"],
                    model(t.as_tensor(background, dtype=t.float64)).detach().squeeze(-1).numpy(),
                ),
                "bounded_pair_local_accuracy": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} bounded donor local accuracy",
                    np.asarray(pair["base"]) + np.asarray(pair["phi"]).sum(axis=1),
                    np.full(deep_helper_deepshap_reference_BACKGROUND_ROWS, float(pair["prediction"])),
                    atol=deep_helper_deepshap_reference_AUTHOR_ATOL,
                    rtol=deep_helper_deepshap_reference_AUTHOR_RTOL,
                ),
                "bounded_pair_prediction": deep_helper_deepshap_reference__require_close(
                    f"{split} {key} bounded query prediction",
                    np.array(pair["prediction"]),
                    np.array(stored_prediction[0]),
                ),
                "bounded_pair_average": max(
                    deep_helper_deepshap_reference__require_close(
                        f"{split} {key} bounded-pair attribution average",
                        np.asarray(pair["phi"]).mean(axis=0),
                        author_phi[0],
                        atol=deep_helper_deepshap_reference_AUTHOR_ATOL,
                        rtol=deep_helper_deepshap_reference_AUTHOR_RTOL,
                    ),
                    deep_helper_deepshap_reference__require_close(
                        f"{split} {key} bounded-pair base average",
                        np.array(np.asarray(pair["base"]).mean()),
                        np.array(author["base"]),
                    ),
                ),
            }
            bounded_pairs[split][key] = {
                "row_index": 0,
                "source_id": int(row_ids[split][0]),
                "background_ids": identity["background_ids"].copy(),
                "prediction": float(pair["prediction"]),
                "base_per_reference": np.asarray(pair["base"]).copy(),
                "phi_deep_explainer_per_reference": np.asarray(pair["phi"]).copy(),
                "local_accuracy_residual_per_reference": np.asarray(
                    pair["local_accuracy_residual"]
                ).copy(),
                "scope": "first row only; all 64 donors in frozen source-ID order",
            }
            errors.update(pair_errors)
            for name, error in errors.items():
                model_errors[name] = max(model_errors[name], error)
                maxima[name] = max(maxima[name], error)

            if key == "ridge":
                state = item["state"]
                analytic = deep_helper_deepshap_reference_ridge_closed_form(
                    deep_helper_deepshap_reference__array(state["0.weight"]),
                    deep_helper_deepshap_reference__array(state["0.bias"]),
                    identity["explain_x"] if split == "explanation" else identity["audit_x"],
                    background,
                )
                ridge_errors = {
                    "ridge_analytic_prediction": deep_helper_deepshap_reference__require_close(
                        f"{split} ridge analytic prediction", analytic["prediction"], stored_prediction
                    ),
                    "ridge_analytic_base": deep_helper_deepshap_reference__require_close(
                        f"{split} ridge analytic base",
                        np.array(analytic["base"]),
                        np.array(stored_base),
                    ),
                    "ridge_analytic_phi": max(
                        deep_helper_deepshap_reference__require_close(
                            f"{split} ridge analytic learner DeepSHAP", analytic["phi"], stored_phi
                        ),
                        deep_helper_deepshap_reference__require_close(
                            f"{split} ridge analytic exact Shapley", analytic["phi"], exact
                        ),
                        deep_helper_deepshap_reference__require_close(
                            f"{split} ridge analytic author DeepSHAP", analytic["phi"], author_phi
                        ),
                    ),
                }
                ridge_reference[split] = {
                    "base": float(analytic["base"]),
                    "prediction": np.asarray(analytic["prediction"]).copy(),
                    "phi": np.asarray(analytic["phi"]).copy(),
                }
                for name, error in ridge_errors.items():
                    model_errors[name] = max(model_errors[name], error)
                    maxima[name] = max(maxima[name], error)

        if deep_helper_deepshap_reference__state_hash(item["state"]) != item_state_before or deep_helper_deepshap_reference__model_hash(model) != item_state_before:
            raise AssertionError(f"Reference evaluation changed the raw state for {key}.")
        per_model_errors[key] = model_errors

    diagnostics = deep_helper_deepshap_reference_known_diagnostics()
    passes = {
        "row_identity": True,
        "background_identity_all_64": True,
        "author_deep_explainer_all_model_rows": True,
        "expected_base": True,
        "bounded_pair_author": True,
        "own_full_pair_shape_finite_and_mean": True,
        "exact_game_contraction": True,
        "ridge_analytic": True,
        "model_states_unchanged": True,
        "tiny_threshold": True,
        "tiny_min_equivalent_representations": True,
        "nearzero_expected_failure_disclosed": bool(
            diagnostics["nearzero"]["expected_failure_disclosed"]
        ),
        "reference_model_boundaries": True,
    }
    if not all(passes.values()):
        raise AssertionError("An independent reference acceptance check failed.")
    return {
        "status": "PASS",
        "summary": {
            "author_averaged_vector_parity_passed": True,
            "author_bounded_pair_parity_passed": True,
            "exact_game_contraction_passed": True,
            "ridge_analytic_parity_passed": True,
            "neural_deep_vs_exact_is_descriptive": True,
            "nearzero_expected_failure_disclosed": True,
        },
        "passes": passes,
        "tolerances": {
            "author_deep_explainer": {"atol": deep_helper_deepshap_reference_AUTHOR_ATOL, "rtol": deep_helper_deepshap_reference_AUTHOR_RTOL},
            "exact_algebraic": {"atol": deep_helper_deepshap_reference_EXACT_ATOL, "rtol": deep_helper_deepshap_reference_EXACT_RTOL},
        },
        "counts": {
            "models": 7,
            "explanation_rows": 153,
            "audit_rows": 119,
            "model_rows": 1904,
            "background_rows": 64,
            "local_values": 15232,
            "author_attribution_coordinates": 15232,
            "own_pair_model_row_backgrounds": 121856,
            "own_pair_coordinates": 974848,
            "bounded_author_pair_checks": 896,
            "bounded_author_pair_coordinates": 7168,
            "pair_model_rows": 896,
            "pair_local_values": 7168,
            "exact_game_model_rows": 1904,
            "coalition_values": 487424,
        },
        "model_keys": deep_helper_deepshap_reference_MODEL_KEYS,
        "units": {"prediction": "MPa", "base": "MPa", "attribution": "MPa"},
        "identity": identity,
        "explanation": references["explanation"],
        "audit": references["audit"],
        "bounded_pair_reference": bounded_pairs,
        "max_errors": maxima,
        "per_model_max_errors": per_model_errors,
        "deep_vs_exact": {
            "role": "descriptive approximation comparison; not an author-parity gate",
            "is_acceptance_gate": False,
            **descriptive,
        },
        "ridge_analytic": ridge_reference,
        "diagnostics": diagnostics,
        "model_state_sha256": state_hashes,
        "reference_environment": environment,
    }

# %%

if MAIN:
    deep_validate_reference = deep_helper_deepshap_reference_validate_reference
if MAIN:
    deep_persist_completed = deep_helper_run_native_persist_completed
if MAIN:
    deep_write_notebook_summary = deep_helper_run_native_write_notebook_summary

if MAIN:
    deep_reference = deep_validate_reference(deep_fitted, deep_prepared, deep_evaluated, deep_train, deep_explain, deep_audit)
if MAIN:
    deep_diagnostics = deep_run_diagnostics(deep_curriculum_methods, include_timing=True)
if MAIN:
    deep_fixed_play = deep_native_play(deep_fitted, deep_train, deep_audit, deep_curriculum_methods)
if MAIN:
    deep_summary = deep_persist_completed(
        native_dir,
        deep_fitted,
        deep_prepared,
        deep_evaluated,
        deep_diagnostics,
        deep_reference,
        deep_fixed_play,
    )
if MAIN:
    deep_write_notebook_summary(Path(deep_RUN_OUTPUT or canonical_dir / "artifacts" / "deep"), native_dir, deep_summary)

if MAIN:
    print("Independent DeepExplainer averaged-vector parity:", deep_reference["status"])
if MAIN:
    print("Averaged coordinates checked:", deep_reference["counts"]["author_attribution_coordinates"])
if MAIN:
    print("Bounded donor coordinates checked:", deep_reference["counts"]["bounded_author_pair_coordinates"])
if MAIN:
    print("Exact game rows checked:", deep_reference["counts"]["exact_game_model_rows"])

# %%

if MAIN:
    deep_show_figure(deep_plots.plot_performance(deep_prepared, deep_evaluated), "08_performance_controls")
if MAIN:
    deep_show_figure(deep_plots.plot_signed_parity(deep_prepared, deep_evaluated), "09_signed_parity_all_rows")

# %%

if MAIN:
    deep_show_figure(deep_plots.plot_error_heatmaps(deep_prepared, deep_evaluated), "10_error_heatmaps_all_rows")
if MAIN:
    deep_show_figure(
        deep_plots.plot_fixed_waterfall(deep_fixed_play, feature_names=deep_FEATURE_NAMES),
        "11_fixed_waterfall",
    )

# %%

if MAIN:
    deep_show_figure(
        deep_plots.plot_fixed_attribution_comparison(deep_fixed_play, feature_names=deep_FEATURE_NAMES),
        "12_fixed_attribution_comparison",
    )
if MAIN:
    deep_show_figure(
        deep_plots.plot_completeness_vs_exact(deep_prepared, deep_evaluated),
        "13_completeness_vs_exact",
    )

# %%

if MAIN:
    deep_show_figure(deep_plots.plot_background_sensitivity(deep_evaluated), "14_background_sensitivity")
if MAIN:
    deep_show_figure(deep_plots.plot_replacement_curves(deep_evaluated), "15_replacement_curves")

# %%

if MAIN:
    play = deep_native_play(
        deep_fitted,
        deep_train,
        deep_audit,
        deep_curriculum_methods,
        model_key="clean_0",
        row=0,
        size=64,
    )
if MAIN:
    deep_show_figure(
        deep_plots.plot_real_recipe_play(play, feature_names=deep_FEATURE_NAMES),
        "17_real_recipe_play",
    )

# %%

if MAIN:
    import hashlib
if MAIN:
    import math
if MAIN:
    import os
if MAIN:
    import sys
if MAIN:
    from collections.abc import Mapping, Sequence
if MAIN:
    from pathlib import Path
if MAIN:
    from types import SimpleNamespace
if MAIN:
    from typing import Any

if MAIN:
    import matplotlib.pyplot as plt
if MAIN:
    import numpy as np
if MAIN:
    import torch
if MAIN:
    import torch as t
if MAIN:
    from IPython.display import display
if MAIN:
    from torch import Tensor, nn

if MAIN:
    torch.set_default_dtype(torch.float64)

gradient_chapter = "chapter16_shapley_attribution_baselines"
gradient_section = "part15_gradientshap"
if MAIN:
    gradient_root_dir = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / gradient_chapter).exists())
if MAIN:
    gradient_exercises_dir = gradient_root_dir / gradient_chapter / "exercises"
if MAIN:
    gradient_section_dir = gradient_exercises_dir / gradient_section
if MAIN:
    if str(gradient_root_dir) not in sys.path:
        sys.path.append(str(gradient_root_dir))
    if str(gradient_exercises_dir) not in sys.path:
        sys.path.append(str(gradient_exercises_dir))

if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import gradient_gradientshap_methods_tests as gradient_tests
if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import gradient_gradientshap_plots as gradient_plots
if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.gradient_gradientshap_data import (
        background_order as gradient_background_order,
        load_manifest as load_manifest,
        load_split as gradient_load_split,
        tensor_mapping_sha256 as tensor_mapping_sha256,
    )

if MAIN:
    gradient_FEATURE_NAMES = np.array([
        "Cement", "Slag", "Fly ash", "Water", "Superplasticizer",
        "Coarse aggregate", "Fine aggregate", "Age",
    ])
if MAIN:
    gradient_RUN_OUTPUT = os.environ.get("ARENA_GRADIENTSHAP_OUTPUT_DIRECTORY")
if MAIN:
    gradient_FIGURE_DIR = Path(
        os.environ.get("ARENA_GRADIENTSHAP_FIGURE_DIR", gradient_RUN_OUTPUT or canonical_dir / "media" / 'gradient')
    )
if MAIN:
    gradient_SAVE_FIGURES = os.environ.get(
        "ARENA_GRADIENTSHAP_SAVE_FIGURES", "1" if gradient_RUN_OUTPUT else "0"
    ) == "1"


def gradient_show_figure(fig, stem):
    """Display a figure and optionally save the exact notebook rendering."""
    display(fig)
    if gradient_SAVE_FIGURES:
        gradient_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(gradient_FIGURE_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def gradient_helper_gradientshap_methods__eg_float_tensor(name: str, value: Tensor, ndim: int | None = None) -> Tensor:
    """Check the lesson's CPU-float64 tensor contract without changing the value."""
    if type(value) is not Tensor:
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cpu" or value.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if value.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not t.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def gradient_helper_gradientshap_methods__eg_index_tensor(name: str, value: Tensor, ndim: int | None = None) -> Tensor:
    """Check a CPU int64 index tensor without coercing or aliasing it."""
    if type(value) is not Tensor:
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cpu" or value.dtype != t.int64:
        raise TypeError(f"{name} must be a CPU int64 tensor.")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if value.numel() == 0:
        raise ValueError(f"{name} cannot be empty.")
    return value

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def gradient_helper_gradientshap_methods__eg_model(model: nn.Module) -> nn.Module:
    """Validate the CPU-float64 state used by the lesson's scalar models."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != t.float64:
            raise TypeError("Every model parameter must be a CPU float64 tensor.")
        if not t.isfinite(parameter).all():
            raise ValueError("Every model parameter must be finite.")
    for buffer in model.buffers():
        if buffer.device.type != "cpu":
            raise TypeError("Every model buffer must be on CPU.")
        if buffer.is_floating_point():
            if buffer.dtype != t.float64:
                raise TypeError("Every floating model buffer must be float64.")
            if not t.isfinite(buffer).all():
                raise ValueError("Every floating model buffer must be finite.")
    return model

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def gradient_helper_gradientshap_methods__eg_model_output(model: nn.Module, inputs: Tensor) -> Tensor:
    """Return one scalar per row; callers control whether gradients are enabled."""
    output = model(inputs)
    if type(output) is not Tensor:
        raise TypeError("The model must return a torch.Tensor.")
    if output.shape == (len(inputs), 1):
        output = output[:, 0]
    elif output.shape != (len(inputs),):
        raise ValueError("The model must return one scalar output per input row.")
    if output.device.type != "cpu" or output.dtype != t.float64:
        raise TypeError("The model must return CPU float64 outputs.")
    if not t.isfinite(output).all():
        raise ValueError("The model returned a nonfinite value.")
    return output

# %%

def gradient_helper_gradientshap_methods__eg_positive_int(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value

# %%

if MAIN:
    from collections.abc import Iterator
if MAIN:
    import contextlib
if MAIN:
    from torch import Tensor, nn
if MAIN:
    import numpy as np
if MAIN:
    import random
if MAIN:
    import torch as t

def initialize_helper_gradient_3_5_gradient_helper_gradientshap_methods__eg_preserve_execution_state():
    global gradient_helper_gradientshap_methods__eg_preserve_execution_state
    @contextlib.contextmanager
    def gradient_helper_gradientshap_methods__eg_preserve_execution_state(model: nn.Module) -> Iterator[None]:
        """Restore global RNG, default dtype, module modes, buffers, and parameter grads."""
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = t.random.get_rng_state()
        default_dtype = t.get_default_dtype()
        module_modes = [(module, module.training) for module in model.modules()]
        buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
        gradients = [
            (
                parameter,
                parameter.grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for parameter in model.parameters()
        ]
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            t.random.set_rng_state(torch_state)
            t.set_default_dtype(default_dtype)
            for module, training in module_modes:
                module.training = training
            with t.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
                for parameter, original, saved in gradients:
                    if original is None:
                        parameter.grad = None
                    else:
                        parameter.grad = original
                        original.copy_(saved)


if MAIN:
    initialize_helper_gradient_3_5_gradient_helper_gradientshap_methods__eg_preserve_execution_state()

# %%

if MAIN:
    import hashlib

def gradient_helper_gradientshap_methods__eg_seed_words(seed: int, source_id: int, stream_tag: str) -> int:
    """Derive one local 128-bit PCG64 seed from the declared lesson namespace."""
    name = f"arena16.15/gradientshap/v1/{seed}/{source_id}/{stream_tag}"
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:16], "little")

# %%

if MAIN:
    from torch import Tensor, nn

def gradient_helper_gradientshap_methods__eg_validate_paths(
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    noise: Tensor | None = None,
) -> tuple[int, int, int]:
    gradient_helper_gradientshap_methods__eg_float_tensor("x", x, 2)
    gradient_helper_gradientshap_methods__eg_float_tensor("background", background, 2)
    gradient_helper_gradientshap_methods__eg_index_tensor("indices", indices, 2)
    gradient_helper_gradientshap_methods__eg_float_tensor("alpha", alpha, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    if indices.shape != alpha.shape or indices.shape[0] != len(x):
        raise ValueError("indices and alpha must both have shape (rows of x, samples).")
    if indices.min() < 0 or indices.max() >= len(background):
        raise ValueError("Every donor index must select a supplied background row.")
    if alpha.min() < 0 or alpha.max() > 1:
        raise ValueError("Every alpha must lie in the closed interval [0, 1].")
    n_rows, samples = indices.shape
    n_features = x.shape[1]
    if noise is not None:
        gradient_helper_gradientshap_methods__eg_float_tensor("noise", noise, 3)
        if noise.shape != (n_rows, samples, n_features):
            raise ValueError("noise must have shape (rows of x, samples, input features).")
    return n_rows, samples, n_features

# %%

if MAIN:
    _eg_float_tensor = gradient_helper_gradientshap_methods__eg_float_tensor
if MAIN:
    _eg_index_tensor = gradient_helper_gradientshap_methods__eg_index_tensor
if MAIN:
    _eg_model = gradient_helper_gradientshap_methods__eg_model
if MAIN:
    _eg_model_output = gradient_helper_gradientshap_methods__eg_model_output
if MAIN:
    _eg_positive_int = gradient_helper_gradientshap_methods__eg_positive_int
if MAIN:
    _eg_preserve_execution_state = gradient_helper_gradientshap_methods__eg_preserve_execution_state
if MAIN:
    _eg_seed_words = gradient_helper_gradientshap_methods__eg_seed_words
if MAIN:
    _eg_validate_paths = gradient_helper_gradientshap_methods__eg_validate_paths

# %%

if MAIN:
    gradient_DTYPE = t.float64

def gradient__method(methods, name):
    if methods is None:
        gradientshap_methods = SimpleNamespace(**gradient_module_gradientshap_methods_namespace())

        candidate = getattr(gradientshap_methods, name)
    elif isinstance(methods, Mapping):
        candidate = methods.get(name)
    else:
        candidate = getattr(methods, name, None)
    if not callable(candidate):
        raise TypeError(
            f"Learner method namespace must provide a callable {name!r}; "
            "pass globals(), a mapping, or a SimpleNamespace built from the taught methods."
        )
    return candidate

def gradient__linear(weight, bias):
    weight = t.as_tensor(weight, dtype=gradient_DTYPE)
    bias = t.as_tensor(bias, dtype=gradient_DTYPE).reshape(-1)
    layer = nn.Linear(weight.shape[1], weight.shape[0], dtype=gradient_DTYPE)
    with t.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return layer

def gradient__prediction(model, rows):
    with t.no_grad():
        return model(rows).reshape(-1)

def gradient__max_abs(left, right):
    left = t.as_tensor(left, dtype=gradient_DTYPE)
    right = t.as_tensor(right, dtype=gradient_DTYPE)
    return float((left - right).abs().max())

def _relu_model():
    return nn.Sequential(gradient__linear([[1.0]], [0.0]), nn.ReLU()).eval()

def _threshold_model(weights=(2.0, 1.0), bias=-2.0):
    weights = t.as_tensor(weights, dtype=gradient_DTYPE).reshape(1, -1)
    return nn.Sequential(gradient__linear(weights, [bias]), nn.ReLU()).eval()

def _quadratic_model():
    class Quadratic(nn.Module):
        def forward(self, rows):
            return rows[:, 0].square()

    return Quadratic().eval()

def _two_feature_shapley(values):
    values = t.as_tensor(values, dtype=gradient_DTYPE).reshape(4)
    phi_0 = ((values[1] - values[0]) + (values[3] - values[2])) / 2
    phi_1 = ((values[2] - values[0]) + (values[3] - values[1])) / 2
    return t.stack((phi_0, phi_1))

# %%

def sample_path_schedule(
    source_ids: Tensor,
    background_rows: int,
    samples: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Draw donor indices and alphas from independent local streams for each source ID."""
    _eg_index_tensor("source_ids", source_ids, 1)
    background_rows = _eg_positive_int("background_rows", background_rows)
    samples = _eg_positive_int("samples", samples)
    if type(seed) is not int or seed < 0 or seed >= 2**64:
        raise ValueError("seed must be an integer in [0, 2**64).")
    if (source_ids < 0).any():
        raise ValueError("source_ids must be nonnegative integers.")
    if len(t.unique(source_ids)) != len(source_ids):
        raise ValueError("source_ids must be unique so every row names one sampling stream.")

    indices = np.empty((len(source_ids), samples), dtype=np.int64)
    alpha = np.empty((len(source_ids), samples), dtype=np.float64)
    for row, source_id in enumerate(source_ids.tolist()):
        donor_rng = np.random.Generator(
            np.random.PCG64(_eg_seed_words(seed, source_id, "baseline"))
        )
        alpha_rng = np.random.Generator(
            np.random.PCG64(_eg_seed_words(seed, source_id, "alpha"))
        )
        indices[row] = donor_rng.integers(0, background_rows, size=samples, dtype=np.int64)
        alpha[row] = alpha_rng.random(samples, dtype=np.float64)
    return t.tensor(indices, dtype=t.int64), t.tensor(alpha, dtype=t.float64)


if MAIN:
    gradient_tests.test_sample_path_schedule(sample_path_schedule)

# %%

def schedule_fixture():
    return {
        "source_ids": t.tensor([101, 305, 909], dtype=t.int64),
        "background_rows": 5,
        "short_samples": 8,
        "long_samples": 32,
        "seed": 161510,
    }

# %%

def diagnose_schedule(sample_path_schedule_fn):
    fixture = schedule_fixture()
    source_ids = fixture["source_ids"]
    args = (fixture["background_rows"], fixture["seed"])
    indices_8, alpha_8 = sample_path_schedule_fn(
        source_ids, args[0], fixture["short_samples"], args[1]
    )
    indices_32, alpha_32 = sample_path_schedule_fn(
        source_ids, args[0], fixture["long_samples"], args[1]
    )
    reordered = source_ids[t.tensor([2, 0], dtype=t.int64)]
    reorder_indices, reorder_alpha = sample_path_schedule_fn(
        reordered, args[0], fixture["short_samples"], args[1]
    )
    single_indices, single_alpha = sample_path_schedule_fn(
        source_ids[:1], args[0], fixture["short_samples"], args[1]
    )
    return {
        **fixture,
        "indices_8": indices_8,
        "alpha_8": alpha_8,
        "indices_32": indices_32,
        "alpha_32": alpha_32,
        "prefix_match": bool(
            t.equal(indices_8, indices_32[:, :8]) and t.equal(alpha_8, alpha_32[:, :8])
        ),
        "reorder_match": bool(
            t.equal(reorder_indices, indices_8[[2, 0]])
            and t.equal(reorder_alpha, alpha_8[[2, 0]])
        ),
        "single_row_match": bool(
            t.equal(single_indices, indices_8[:1]) and t.equal(single_alpha, alpha_8[:1])
        ),
    }

# %%

if MAIN:
    schedule_check = diagnose_schedule(sample_path_schedule)
if MAIN:
    gradient_show_figure(gradient_plots.plot_schedule_prefix(schedule_check), "01_schedule_prefix")

# %%

def interpolate_paths(
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return sampled path points and query-minus-donor deltas, both shaped N,K,D."""
    _eg_validate_paths(x, background, indices, alpha, noise)
    donors = background[indices]
    query = x[:, None, :] if noise is None else x[:, None, :] + noise
    delta = query - donors
    points = donors + alpha[:, :, None] * delta
    return points, delta


if MAIN:
    gradient_tests.test_interpolate_paths(interpolate_paths)

# %%

def path_fixture():
    return {
        "query": t.tensor([[2.0, 1.0]], dtype=gradient_DTYPE),
        "background": t.tensor([[-1.0, 0.0], [0.0, 2.0]], dtype=gradient_DTYPE),
        "indices": t.tensor([[0, 1, 0, 1]], dtype=t.int64),
        "alpha": t.tensor([[0.20, 0.35, 0.75, 0.90]], dtype=gradient_DTYPE),
    }

# %%

def diagnose_interpolation(interpolate_paths_fn):
    fixture = path_fixture()
    points, deltas = interpolate_paths_fn(
        fixture["query"],
        fixture["background"],
        fixture["indices"],
        fixture["alpha"],
    )
    donors = fixture["background"][fixture["indices"]]
    expected = donors + fixture["alpha"][..., None] * (fixture["query"][:, None, :] - donors)
    return {
        **fixture,
        "donors": donors,
        "points": points,
        "deltas": deltas,
        "expected_points": expected,
        "max_abs_error": gradient__max_abs(points, expected),
    }

# %%

if MAIN:
    path_check = diagnose_interpolation(interpolate_paths)
if MAIN:
    gradient_show_figure(gradient_plots.plot_path_interpolation(path_check), "02_path_interpolation")

# %%

def input_gradients(
    model: nn.Module,
    points: Tensor,
    forward_batch: int = 32_768,
) -> Tensor:
    """Differentiate a deterministic scalar function applied independently to each row.

    The sum-of-outputs gradient is valid only under that batch-independence assumption.
    This function preserves the caller's tensors, RNG states, model modes, buffers, and
    parameter gradients. A scalar function constant with respect to its input gets zero
    input gradients.
    """
    model = _eg_model(model)
    _eg_float_tensor("points", points, 3)
    forward_batch = _eg_positive_int("forward_batch", forward_batch)
    flat_points = points.reshape(-1, points.shape[-1])
    flat_gradients = t.empty_like(flat_points)

    with _eg_preserve_execution_state(model):
        for start in range(0, len(flat_points), forward_batch):
            chunk = flat_points[start : start + forward_batch].detach().clone().requires_grad_(True)
            with t.enable_grad():
                output = _eg_model_output(model, chunk)
                if output.requires_grad:
                    gradient = t.autograd.grad(output.sum(), chunk, allow_unused=True)[0]
                else:
                    gradient = None
            if gradient is None:
                gradient = t.zeros_like(chunk)
            if not t.isfinite(gradient).all():
                raise ValueError("The model produced a nonfinite input gradient.")
            flat_gradients[start : start + len(chunk)] = gradient.detach()
    return flat_gradients.reshape(points.shape)


if MAIN:
    gradient_tests.test_input_gradients(input_gradients)

# %%

def gradient_fixture():
    affine = nn.Sequential(gradient__linear([[2.0, -3.0]], [0.5])).eval()
    affine_points = t.tensor(
        [[[-1.0, 0.0], [0.5, 2.0], [2.0, -1.0], [3.0, 3.0]]],
        dtype=gradient_DTYPE,
    )
    z = t.linspace(-1.0, 1.5, 21, dtype=gradient_DTYPE)
    return {
        "affine_model": affine,
        "affine_points": affine_points,
        "affine_expected": t.tensor([2.0, -3.0], dtype=gradient_DTYPE),
        "quadratic_model": _quadratic_model(),
        "quadratic_points": z.reshape(1, -1, 1),
        "quadratic_expected": (2 * z).reshape(1, -1, 1),
    }

# %%

def diagnose_gradients(input_gradients_fn):
    fixture = gradient_fixture()
    affine = input_gradients_fn(fixture["affine_model"], fixture["affine_points"])
    quadratic = input_gradients_fn(fixture["quadratic_model"], fixture["quadratic_points"])
    return {
        "affine_points": fixture["affine_points"],
        "affine_gradients": affine,
        "affine_expected": fixture["affine_expected"],
        "affine_max_abs_error": gradient__max_abs(affine, fixture["affine_expected"]),
        "quadratic_points": fixture["quadratic_points"],
        "quadratic_gradients": quadratic,
        "quadratic_expected": fixture["quadratic_expected"],
        "quadratic_max_abs_error": gradient__max_abs(quadratic, fixture["quadratic_expected"]),
    }

# %%

if MAIN:
    gradient_check = diagnose_gradients(input_gradients)
if MAIN:
    gradient_show_figure(gradient_plots.plot_gradient_controls(gradient_check), "03_gradient_controls")

# %%

def expected_gradient_samples(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    forward_batch: int = 32_768,
) -> Tensor:
    """Return clean Expected Gradients Monte Carlo contributions, shape N,K,D."""
    points, delta = interpolate_paths(x, background, indices, alpha)
    gradients = input_gradients(model, points, forward_batch)
    return gradients * delta


if MAIN:
    gradient_tests.test_expected_gradient_samples(expected_gradient_samples)

# %%

def sample_fixture():
    return {
        "relu_model": _relu_model(),
        "relu_query": t.tensor([[1.0]], dtype=gradient_DTYPE),
        "relu_background": t.tensor([[-1.0]], dtype=gradient_DTYPE),
        "source_ids": t.tensor([17], dtype=t.int64),
        "samples": 128,
        "seed": 161510,
        "quadratic_model": _quadratic_model(),
        "quadratic_query": t.tensor([[1.0]], dtype=gradient_DTYPE),
        "quadratic_background": t.tensor([[0.0]], dtype=gradient_DTYPE),
        "quadratic_indices": t.tensor([[0, 0]], dtype=t.int64),
        "quadratic_alpha": t.tensor([[0.25, 0.75]], dtype=gradient_DTYPE),
    }

# %%

def diagnose_expected_gradient_samples(
    expected_gradient_samples_fn,
    sample_path_schedule_fn,
):
    fixture = sample_fixture()
    indices, alpha = sample_path_schedule_fn(
        fixture["source_ids"], 1, fixture["samples"], fixture["seed"]
    )
    relu_samples = expected_gradient_samples_fn(
        fixture["relu_model"],
        fixture["relu_query"],
        fixture["relu_background"],
        indices,
        alpha,
    )
    quadratic_samples = expected_gradient_samples_fn(
        fixture["quadratic_model"],
        fixture["quadratic_query"],
        fixture["quadratic_background"],
        fixture["quadratic_indices"],
        fixture["quadratic_alpha"],
    )
    relu_expected = (alpha > 0.5).to(gradient_DTYPE)[..., None] * 2.0
    quadratic_expected = (2 * fixture["quadratic_alpha"])[..., None]
    return {
        "alpha": alpha,
        "relu_samples": relu_samples,
        "relu_expected": relu_expected,
        "relu_max_abs_error": gradient__max_abs(relu_samples, relu_expected),
        "quadratic_alpha": fixture["quadratic_alpha"],
        "quadratic_samples": quadratic_samples,
        "quadratic_expected": quadratic_expected,
        "quadratic_max_abs_error": gradient__max_abs(quadratic_samples, quadratic_expected),
        "population_mean": t.tensor(1.0, dtype=gradient_DTYPE),
        "population_variance": t.tensor(1.0, dtype=gradient_DTYPE),
    }

# %%

if MAIN:
    sample_check = diagnose_expected_gradient_samples(
        expected_gradient_samples,
        sample_path_schedule,
    )
if MAIN:
    gradient_show_figure(gradient_plots.plot_relu_samples(sample_check), "04_relu_samples")

# %%

def summarize_expected_gradients(
    samples: Tensor,
    budgets: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    """Return ordinary sample SE, not SHAP 0.49.1's variance/sqrt(K) proxy."""
    _eg_float_tensor("samples", samples, 3)
    if type(budgets) is not tuple or not budgets:
        raise TypeError("budgets must be a nonempty tuple of sample counts.")
    if any(type(budget) is not int for budget in budgets):
        raise TypeError("Every budget must be an integer sample count.")
    if any(budget < 2 or budget > samples.shape[1] for budget in budgets):
        raise ValueError("Every budget must be between 2 and the available sample count.")
    if any(left >= right for left, right in zip(budgets, budgets[1:])):
        raise ValueError("budgets must be strictly increasing nested prefixes.")

    means, standard_errors = [], []
    for budget in budgets:
        prefix = samples[:, :budget]
        means.append(prefix.mean(dim=1))
        standard_errors.append(prefix.std(dim=1, correction=1) / math.sqrt(budget))
    return t.stack(means), t.stack(standard_errors)


if MAIN:
    gradient_tests.test_summarize_expected_gradients(summarize_expected_gradients)

# %%

def diagnose_convergence(
    sample_path_schedule_fn,
    expected_gradient_samples_fn,
    summarize_expected_gradients_fn,
):
    fixture = sample_fixture()
    indices, alpha = sample_path_schedule_fn(
        fixture["source_ids"], 1, fixture["samples"], fixture["seed"]
    )
    samples = expected_gradient_samples_fn(
        fixture["relu_model"],
        fixture["relu_query"],
        fixture["relu_background"],
        indices,
        alpha,
    )
    budgets = (2, 4, 8, 16, 32, 64, 128)
    means, standard_errors = summarize_expected_gradients_fn(samples, budgets)
    budget_tensor = t.tensor(budgets, dtype=gradient_DTYPE)
    return {
        "budgets": t.tensor(budgets, dtype=t.int64),
        "means": means[:, 0, 0],
        "standard_errors": standard_errors[:, 0, 0],
        "population_mean": t.tensor(1.0, dtype=gradient_DTYPE),
        "analytic_standard_error": budget_tensor.rsqrt(),
        "absolute_error": (means[:, 0, 0] - 1.0).abs(),
        "samples": samples,
    }

# %%

if MAIN:
    convergence_check = diagnose_convergence(
        sample_path_schedule,
        expected_gradient_samples,
        summarize_expected_gradients,
    )
if MAIN:
    gradient_show_figure(gradient_plots.plot_mc_convergence(convergence_check), "05_mc_convergence")

# %%

def expected_gradients(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    forward_batch: int = 32_768,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return full-background base, clean Expected Gradients, and ordinary estimator SE."""
    _, samples, _ = _eg_validate_paths(x, background, indices, alpha)
    if samples < 2:
        raise ValueError(
            "Expected Gradients needs at least two samples to estimate standard error."
        )
    contributions = expected_gradient_samples(
        model, x, background, indices, alpha, forward_batch
    )
    means, standard_errors = summarize_expected_gradients(contributions, (samples,))
    model = _eg_model(model)
    with _eg_preserve_execution_state(model), t.no_grad():
        base = _eg_model_output(model, background).mean().detach().clone()
    return base, means[0], standard_errors[0]


if MAIN:
    gradient_tests.test_expected_gradients(expected_gradients)

# %%

def estimator_fixture():
    affine = nn.Sequential(gradient__linear([[2.0, -3.0]], [0.5])).eval()
    dummy = nn.Sequential(gradient__linear([[2.0, -3.0, 0.0]], [0.5])).eval()
    constant = nn.Sequential(gradient__linear([[0.0, 0.0]], [2.0])).eval()
    return {
        "affine_model": affine,
        "affine_query": t.tensor([[2.0, 1.0]], dtype=gradient_DTYPE),
        "affine_background": t.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=gradient_DTYPE),
        "balanced_indices": t.tensor([[0, 1, 0, 1]], dtype=t.int64),
        "balanced_alpha": t.tensor([[0.15, 0.35, 0.65, 0.85]], dtype=gradient_DTYPE),
        "affine_expected_phi": t.tensor([[3.0, 0.0]], dtype=gradient_DTYPE),
        "dummy_model": dummy,
        "dummy_query": t.tensor([[2.0, 1.0, 9.0]], dtype=gradient_DTYPE),
        "dummy_background": t.tensor([[0.0, 0.0, -5.0], [1.0, 2.0, 7.0]], dtype=gradient_DTYPE),
        "constant_model": constant,
        "relu_model": _relu_model(),
        "relu_query": t.tensor([[1.0]], dtype=gradient_DTYPE),
        "relu_background": t.tensor([[-1.0]], dtype=gradient_DTYPE),
        "relu_indices": t.zeros((1, 5), dtype=t.int64),
        "relu_alpha": t.tensor([[0.10, 0.20, 0.30, 0.40, 0.60]], dtype=gradient_DTYPE),
    }

# %%

def diagnose_expected_gradients(expected_gradients_fn, sample_path_schedule_fn):
    fixture = estimator_fixture()
    affine_base, affine_phi, affine_se = expected_gradients_fn(
        fixture["affine_model"],
        fixture["affine_query"],
        fixture["affine_background"],
        fixture["balanced_indices"],
        fixture["balanced_alpha"],
    )
    scheduled_indices, scheduled_alpha = sample_path_schedule_fn(
        t.tensor([41], dtype=t.int64), 2, 32, 161510
    )
    dummy_base, dummy_phi, dummy_se = expected_gradients_fn(
        fixture["dummy_model"],
        fixture["dummy_query"],
        fixture["dummy_background"],
        scheduled_indices,
        scheduled_alpha,
    )
    constant_base, constant_phi, constant_se = expected_gradients_fn(
        fixture["constant_model"],
        fixture["affine_query"],
        fixture["affine_background"],
        scheduled_indices,
        scheduled_alpha,
    )
    relu_base, relu_phi, relu_se = expected_gradients_fn(
        fixture["relu_model"],
        fixture["relu_query"],
        fixture["relu_background"],
        fixture["relu_indices"],
        fixture["relu_alpha"],
    )

    def result(model, query, base, phi, se):
        prediction = gradient__prediction(model, query)[0]
        return {
            "base": base,
            "phi": phi,
            "standard_error": se,
            "prediction": prediction,
            "reconstruction": base + phi.sum(),
            "residual": base + phi.sum() - prediction,
        }

    return {
        "affine": result(
            fixture["affine_model"],
            fixture["affine_query"],
            affine_base,
            affine_phi,
            affine_se,
        ),
        "dummy": result(
            fixture["dummy_model"],
            fixture["dummy_query"],
            dummy_base,
            dummy_phi,
            dummy_se,
        ),
        "constant": result(
            fixture["constant_model"],
            fixture["affine_query"],
            constant_base,
            constant_phi,
            constant_se,
        ),
        "finite_relu": result(
            fixture["relu_model"],
            fixture["relu_query"],
            relu_base,
            relu_phi,
            relu_se,
        ),
        "affine_expected_phi": fixture["affine_expected_phi"],
        "affine_max_abs_error": gradient__max_abs(affine_phi, fixture["affine_expected_phi"]),
        "dummy_max_abs": float(dummy_phi[:, 2].abs().max()),
        "constant_max_abs": float(constant_phi.abs().max()),
    }

# %%

if MAIN:
    estimator_check = diagnose_expected_gradients(
        expected_gradients,
        sample_path_schedule,
    )
if MAIN:
    gradient_show_figure(gradient_plots.plot_estimator_controls(estimator_check), "06_estimator_controls")

# %%

def gauss_legendre_pairs(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    n_nodes: int,
    row_batch: int = 4,
    forward_batch: int = 32_768,
) -> Tensor:
    """Approximate every query-donor Integrated Gradients path by finite quadrature.

    ReLU path gradients jump, so finite-node quadrature remains a numerical control
    rather than exact path truth.
    """
    model = _eg_model(model)
    _eg_float_tensor("x", x, 2)
    _eg_float_tensor("background", background, 2)
    if x.shape[1] != background.shape[1]:
        raise ValueError("x and background must have the same feature dimension.")
    n_nodes = _eg_positive_int("n_nodes", n_nodes)
    row_batch = _eg_positive_int("row_batch", row_batch)
    forward_batch = _eg_positive_int("forward_batch", forward_batch)

    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(n_nodes)
    nodes = t.tensor((raw_nodes + 1) / 2, dtype=t.float64)
    weights = t.tensor(raw_weights / 2, dtype=t.float64)
    pairs = t.empty((len(x), len(background), x.shape[1]), dtype=t.float64)

    for start in range(0, len(x), row_batch):
        query = x[start : start + row_batch]
        delta = query[:, None, :] - background[None, :, :]
        points = background[None, :, None, :] + nodes[None, None, :, None] * delta[:, :, None, :]
        packed = points.reshape(len(query), len(background) * n_nodes, x.shape[1])
        gradients = input_gradients(model, packed, forward_batch).reshape(points.shape)
        mean_gradient = (gradients * weights[None, None, :, None]).sum(dim=2)
        pairs[start : start + len(query)] = mean_gradient * delta
    return pairs


if MAIN:
    gradient_tests.test_gauss_legendre_pairs(gauss_legendre_pairs)

# %%

def quadrature_fixture():
    return {
        "relu_model": _relu_model(),
        "query": t.tensor([[1.0]], dtype=gradient_DTYPE),
        "background": t.tensor([[-2.0]], dtype=gradient_DTYPE),
        "nodes": (4, 8, 16, 32, 64, 128, 256, 512),
        "analytic": t.tensor(1.0, dtype=gradient_DTYPE),
        "quadratic_model": _quadratic_model(),
        "quadratic_query": t.tensor([[1.0]], dtype=gradient_DTYPE),
        "quadratic_background": t.tensor([[0.0]], dtype=gradient_DTYPE),
    }

# %%

def diagnose_quadrature(gauss_legendre_pairs_fn, input_gradients_fn):
    fixture = quadrature_fixture()
    estimates = []
    for nodes in fixture["nodes"]:
        pair = gauss_legendre_pairs_fn(
            fixture["relu_model"], fixture["query"], fixture["background"], nodes
        )
        estimates.append(pair[0, 0, 0])
    estimates = t.stack(estimates)
    quadratic = gauss_legendre_pairs_fn(
        fixture["quadratic_model"],
        fixture["quadratic_query"],
        fixture["quadratic_background"],
        2,
    )[0, 0, 0]
    probe_points = t.tensor([[[-0.5], [0.25], [1.0]]], dtype=gradient_DTYPE)
    probe_gradients = input_gradients_fn(fixture["quadratic_model"], probe_points)
    return {
        "nodes": t.tensor(fixture["nodes"], dtype=t.int64),
        "relu_estimates": estimates,
        "analytic": fixture["analytic"],
        "absolute_error": (estimates - fixture["analytic"]).abs(),
        "successive_drift": t.cat((t.zeros(1, dtype=gradient_DTYPE), estimates.diff().abs())),
        "quadratic_n2": quadratic,
        "quadratic_error": float(abs(quadratic - 1.0)),
        "quadratic_probe_gradients": probe_gradients,
        "quadratic_probe_expected": 2 * probe_points,
    }

# %%

if MAIN:
    quadrature_check = diagnose_quadrature(
        gauss_legendre_pairs,
        input_gradients,
    )
if MAIN:
    gradient_show_figure(gradient_plots.plot_quadrature_doubling(quadrature_check), "07_quadrature_doubling")

# %%

def gradient_diagnose_background_average(gauss_legendre_pairs_fn):
    model = _relu_model()
    query = t.tensor([[2.0]], dtype=gradient_DTYPE)
    background = t.tensor([[-2.0], [1.0]], dtype=gradient_DTYPE)
    mean_background = background.mean(dim=0, keepdim=True)
    pair_256 = gauss_legendre_pairs_fn(model, query, background, 256)[0, :, 0]
    pair_512 = gauss_legendre_pairs_fn(model, query, background, 512)[0, :, 0]
    mean_reference_512 = gauss_legendre_pairs_fn(model, query, mean_background, 512)[0, 0, 0]
    return {
        "background": background[:, 0],
        "pair_256": pair_256,
        "pair_512": pair_512,
        "pair_expected": t.tensor([2.0, 1.0], dtype=gradient_DTYPE),
        "mean_pair": pair_512.mean(),
        "mean_reference": mean_reference_512,
        "expected_mean_pair": t.tensor(1.5, dtype=gradient_DTYPE),
        "expected_mean_reference": t.tensor(2.0, dtype=gradient_DTYPE),
        "base": gradient__prediction(model, background).mean(),
        "mean_reference_base": gradient__prediction(model, mean_background)[0],
        "quadrature_drift": (pair_512 - pair_256).abs(),
    }

# %%

if MAIN:
    background_check = gradient_diagnose_background_average(gauss_legendre_pairs)
if MAIN:
    gradient_show_figure(gradient_plots.plot_background_average(background_check), "08_background_average")

# %%

def noisy_expected_gradient_samples(
    model: nn.Module,
    x: Tensor,
    background: Tensor,
    indices: Tensor,
    alpha: Tensor,
    noise: Tensor,
    forward_batch: int = 32_768,
) -> Tensor:
    """Use x+noise in point and delta, targeting the noise-smoothed query output."""
    points, delta = interpolate_paths(x, background, indices, alpha, noise)
    gradients = input_gradients(model, points, forward_batch)
    return gradients * delta


if MAIN:
    gradient_tests.test_noisy_expected_gradient_samples(noisy_expected_gradient_samples)

# %%

def noise_fixture(samples=65_536):
    rng = np.random.default_rng(161512)
    return {
        "model": _relu_model(),
        "query": t.tensor([[0.0]], dtype=gradient_DTYPE),
        "background": t.tensor([[-1.0]], dtype=gradient_DTYPE),
        "indices": t.zeros((1, samples), dtype=t.int64),
        "alpha": t.tensor(rng.random((1, samples)), dtype=gradient_DTYPE),
        "standard_normal": t.tensor(rng.standard_normal((1, samples, 1)), dtype=gradient_DTYPE),
        "sigmas": t.tensor([0.0, 0.25, 0.50, 1.0], dtype=gradient_DTYPE),
    }

# %%

def diagnose_noise_target(
    noisy_expected_gradient_samples_fn,
    expected_gradient_samples_fn,
    summarize_expected_gradients_fn,
):
    fixture = noise_fixture()
    clean = expected_gradient_samples_fn(
        fixture["model"],
        fixture["query"],
        fixture["background"],
        fixture["indices"],
        fixture["alpha"],
    )
    estimates, standard_errors = [], []
    zero_samples = None
    for sigma in fixture["sigmas"]:
        noise = sigma * fixture["standard_normal"]
        samples = noisy_expected_gradient_samples_fn(
            fixture["model"],
            fixture["query"],
            fixture["background"],
            fixture["indices"],
            fixture["alpha"],
            noise,
        )
        mean, standard_error = summarize_expected_gradients_fn(samples, (samples.shape[1],))
        estimates.append(mean[0, 0, 0])
        standard_errors.append(standard_error[0, 0, 0])
        if float(sigma) == 0.0:
            zero_samples = samples
    estimates = t.stack(estimates)
    standard_errors = t.stack(standard_errors)
    analytic = fixture["sigmas"] / math.sqrt(2 * math.pi)
    return {
        "sigmas": fixture["sigmas"],
        "estimates": estimates,
        "standard_errors": standard_errors,
        "analytic_smoothed_target": analytic,
        "clean_target": t.zeros_like(analytic),
        "clean_residual": estimates,
        "smoothed_residual": estimates - analytic,
        "zero_noise_equal": bool(t.equal(clean, zero_samples)),
        "samples": int(fixture["alpha"].shape[1]),
    }

# %%

if MAIN:
    noise_check = diagnose_noise_target(
        noisy_expected_gradient_samples,
        expected_gradient_samples,
        summarize_expected_gradients,
    )
if MAIN:
    gradient_show_figure(gradient_plots.plot_noise_target(noise_check), "09_gaussian_target_shift")

# %%

def ranked_coalition_curves(
    values: Tensor,
    phi: Tensor,
    orders: Tensor | None = None,
) -> dict[str, Tensor]:
    """Gather raw insertion and deletion prefixes from complete coalition tables."""
    _eg_float_tensor("values", values, 2)
    _eg_float_tensor("phi", phi, 2)
    n_rows, size = values.shape
    n_features = size.bit_length() - 1
    if n_features < 1 or size != 2**n_features or phi.shape != (n_rows, n_features):
        raise ValueError("values must be (N,2**D) and phi must be (N,D), with D >= 1.")

    if orders is None:
        direction = t.where(values[:, -1] - values[:, 0] >= 0, 1.0, -1.0)[:, None]
        orders = t.argsort(direction * phi, dim=1, descending=True, stable=True)
    else:
        _eg_index_tensor("orders", orders, 2)
        if orders.shape != (n_rows, n_features):
            raise ValueError("orders must have shape (rows, features).")
        expected = t.arange(n_features, dtype=t.int64).expand(n_rows, n_features)
        if not t.equal(orders.sort(dim=1).values, expected):
            raise ValueError("Every order row must contain each feature exactly once.")
        orders = orders.clone()

    insertion_masks = t.zeros((n_rows, n_features + 1), dtype=t.int64)
    deletion_masks = t.full((n_rows, n_features + 1), size - 1, dtype=t.int64)
    for step in range(n_features):
        bit = 1 << orders[:, step]
        insertion_masks[:, step + 1] = insertion_masks[:, step] | bit
        deletion_masks[:, step + 1] = deletion_masks[:, step] ^ bit
    return {
        "orders": orders,
        "insertion_masks": insertion_masks,
        "deletion_masks": deletion_masks,
        "insertion": values.gather(1, insertion_masks),
        "deletion": values.gather(1, deletion_masks),
    }


if MAIN:
    gradient_tests.test_ranked_coalition_curves(ranked_coalition_curves)

# %%

def gradient_threshold_fixture():
    return {
        "model": _threshold_model(),
        "query": t.tensor([[1.0, 1.0]], dtype=gradient_DTYPE),
        "background": t.tensor([[0.0, 0.0]], dtype=gradient_DTYPE),
        "coalition_values": t.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=gradient_DTYPE),
        "exact_shapley": t.tensor([[0.5, 0.5]], dtype=gradient_DTYPE),
        "analytic_path": t.tensor([[2 / 3, 1 / 3]], dtype=gradient_DTYPE),
        "necessity_drop": t.tensor([1.0, 1.0], dtype=gradient_DTYPE),
    }

# %%

def kink_fixture():
    first = nn.Sequential(
        gradient__linear([[0.0, 1.0], [-1.0, 1.0]], [0.0, 0.0]),
        nn.ReLU(),
        gradient__linear([[1.0, -1.0]], [0.0]),
    ).eval()
    mirrored = nn.Sequential(
        gradient__linear([[1.0, 0.0], [1.0, -1.0]], [0.0, 0.0]),
        nn.ReLU(),
        gradient__linear([[1.0, -1.0]], [0.0]),
    ).eval()
    return {
        "models": {"x1 route": first, "x0 route": mirrored},
        "query": t.tensor([[1.0, 1.0]], dtype=gradient_DTYPE),
        "background": t.tensor([[0.0, 0.0]], dtype=gradient_DTYPE),
        "binary_inputs": t.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=gradient_DTYPE
        ),
        "expected_binary": t.tensor([0.0, 0.0, 0.0, 1.0], dtype=gradient_DTYPE),
        "exact_shapley": t.tensor([[0.5, 0.5]], dtype=gradient_DTYPE),
    }

# %%

def gradient_replacement_fixture():
    contributions = t.tensor([3.0, -2.0, 0.0], dtype=gradient_DTYPE)
    masks = t.arange(8, dtype=t.int64)
    present = (masks[:, None].bitwise_and(1 << t.arange(3)) != 0).to(gradient_DTYPE)
    return {
        "values": (10.0 + present @ contributions).reshape(1, -1),
        "phi": contributions.reshape(1, -1),
        "all_orders": t.tensor(
            [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]],
            dtype=t.int64,
        ),
    }

# %%

def diagnose_path_vs_coalition(gauss_legendre_pairs_fn, ranked_coalition_curves_fn):
    threshold = gradient_threshold_fixture()
    path_256 = gauss_legendre_pairs_fn(
        threshold["model"], threshold["query"], threshold["background"], 256
    ).mean(dim=1)
    path_512 = gauss_legendre_pairs_fn(
        threshold["model"], threshold["query"], threshold["background"], 512
    ).mean(dim=1)
    threshold_orders = t.tensor([[0, 1], [1, 0]], dtype=t.int64)
    threshold_curves = [
        ranked_coalition_curves_fn(
            threshold["coalition_values"], path_512, order.reshape(1, -1)
        )
        for order in threshold_orders
    ]

    kink = kink_fixture()
    kink_phi, kink_outputs = {}, {}
    for name, model in kink["models"].items():
        kink_phi[name] = gauss_legendre_pairs_fn(
            model, kink["query"], kink["background"], 32
        ).mean(dim=1)
        kink_outputs[name] = gradient__prediction(model, kink["binary_inputs"])

    replacement = gradient_replacement_fixture()
    local_curve = ranked_coalition_curves_fn(replacement["values"], replacement["phi"])
    random_curves = [
        ranked_coalition_curves_fn(
            replacement["values"], replacement["phi"], order.reshape(1, -1)
        )
        for order in replacement["all_orders"]
    ]
    return {
        "threshold": {
            "path_256": path_256,
            "path_512": path_512,
            "quadrature_drift": (path_512 - path_256).abs(),
            "analytic_path": threshold["analytic_path"],
            "exact_shapley": threshold["exact_shapley"],
            "necessity_drop": threshold["necessity_drop"],
            "orders": threshold_orders,
            "curves": threshold_curves,
        },
        "kink": {
            "binary_inputs": kink["binary_inputs"],
            "expected_binary": kink["expected_binary"],
            "outputs": kink_outputs,
            "path_phi": kink_phi,
            "exact_shapley": kink["exact_shapley"],
        },
        "replacement": {
            "values": replacement["values"],
            "phi": replacement["phi"],
            "local_curve": local_curve,
            "random_curves": random_curves,
        },
    }

# %%

if MAIN:
    game_check = diagnose_path_vs_coalition(
        gauss_legendre_pairs,
        ranked_coalition_curves,
    )
if MAIN:
    gradient_show_figure(
        gradient_plots.plot_path_vs_coalition(game_check),
        "10_path_vs_coalition",
    )
if MAIN:
    gradient_show_figure(
        gradient_plots.plot_kink_representation(game_check),
        "11_kink_representation",
    )
if MAIN:
    gradient_show_figure(
        gradient_plots.plot_replacement_fixture(game_check),
        "12_replacement_fixture",
    )

# %%

if MAIN:
    gradient_methods = SimpleNamespace(
        sample_path_schedule=sample_path_schedule,
        interpolate_paths=interpolate_paths,
        input_gradients=input_gradients,
        expected_gradient_samples=expected_gradient_samples,
        summarize_expected_gradients=summarize_expected_gradients,
        expected_gradients=expected_gradients,
        gauss_legendre_pairs=gauss_legendre_pairs,
        noisy_expected_gradient_samples=noisy_expected_gradient_samples,
        ranked_coalition_curves=ranked_coalition_curves,
    )

# %%

def gradient_threshold_play(
    methods,
    weights: Sequence[float] = (2.0, 1.0),
    bias: float = -2.0,
    query: Sequence[float] = (1.0, 1.0),
    reference: Sequence[float] = (0.0, 0.0),
    samples: int = 512,
    seed: int = 161510,
):
    schedule_fn = gradient__method(methods, "sample_path_schedule")
    sample_fn = gradient__method(methods, "expected_gradient_samples")
    summarize_fn = gradient__method(methods, "summarize_expected_gradients")
    quadrature_fn = gradient__method(methods, "gauss_legendre_pairs")
    curves_fn = gradient__method(methods, "ranked_coalition_curves")
    model = _threshold_model(weights, bias)
    query_tensor = t.as_tensor(query, dtype=gradient_DTYPE).reshape(1, -1)
    reference_tensor = t.as_tensor(reference, dtype=gradient_DTYPE).reshape(1, -1)
    if query_tensor.shape != reference_tensor.shape:
        raise ValueError("query and reference must have the same feature count.")
    if query_tensor.shape[1] != len(weights):
        raise ValueError("weights must contain one value per query feature.")
    indices, alpha = schedule_fn(t.tensor([0], dtype=t.int64), 1, samples, seed)
    draws = sample_fn(model, query_tensor, reference_tensor, indices, alpha)
    mean, standard_error = summarize_fn(draws, (samples,))
    quadrature = quadrature_fn(model, query_tensor, reference_tensor, 512).mean(dim=1)

    feature_count = query_tensor.shape[1]
    if feature_count != 2:
        raise ValueError("The editable teaching play currently requires exactly two features.")
    masks = t.arange(4, dtype=t.int64)
    present = masks[:, None].bitwise_and(1 << t.arange(2)) != 0
    hybrids = t.where(present, query_tensor, reference_tensor)
    values = gradient__prediction(model, hybrids).reshape(1, -1)
    exact = _two_feature_shapley(values).reshape(1, -1)
    curves = {
        "expected_gradients": curves_fn(values, mean[0]),
        "exact_shapley": curves_fn(values, exact),
    }
    return {
        "weights": t.as_tensor(weights, dtype=gradient_DTYPE),
        "bias": float(bias),
        "query": query_tensor,
        "reference": reference_tensor,
        "alpha": alpha,
        "draws": draws,
        "mean": mean[0],
        "standard_error": standard_error[0],
        "quadrature": quadrature,
        "coalition_values": values,
        "exact_shapley": exact,
        "prediction": gradient__prediction(model, query_tensor)[0],
        "base": gradient__prediction(model, reference_tensor)[0],
        "curves": curves,
        "samples": samples,
        "seed": seed,
    }

def gradient_run_diagnostics(methods=None):
    schedule = diagnose_schedule(gradient__method(methods, "sample_path_schedule"))
    interpolation = diagnose_interpolation(gradient__method(methods, "interpolate_paths"))
    gradients = diagnose_gradients(gradient__method(methods, "input_gradients"))
    samples = diagnose_expected_gradient_samples(
        gradient__method(methods, "expected_gradient_samples"),
        gradient__method(methods, "sample_path_schedule"),
    )
    convergence = diagnose_convergence(
        gradient__method(methods, "sample_path_schedule"),
        gradient__method(methods, "expected_gradient_samples"),
        gradient__method(methods, "summarize_expected_gradients"),
    )
    estimator = diagnose_expected_gradients(
        gradient__method(methods, "expected_gradients"),
        gradient__method(methods, "sample_path_schedule"),
    )
    quadrature = diagnose_quadrature(
        gradient__method(methods, "gauss_legendre_pairs"),
        gradient__method(methods, "input_gradients"),
    )
    background = gradient_diagnose_background_average(gradient__method(methods, "gauss_legendre_pairs"))
    noise = diagnose_noise_target(
        gradient__method(methods, "noisy_expected_gradient_samples"),
        gradient__method(methods, "expected_gradient_samples"),
        gradient__method(methods, "summarize_expected_gradients"),
    )
    game = diagnose_path_vs_coalition(
        gradient__method(methods, "gauss_legendre_pairs"),
        gradient__method(methods, "ranked_coalition_curves"),
    )
    namespace = methods
    if namespace is None:
        gradientshap_methods = SimpleNamespace(**gradient_module_gradientshap_methods_namespace())

        namespace = gradientshap_methods
    play = gradient_threshold_play(namespace)
    return {
        "schedule": schedule,
        "interpolation": interpolation,
        "gradients": gradients,
        "samples": samples,
        "convergence": convergence,
        "estimator": estimator,
        "quadrature": quadrature,
        "background": background,
        "noise": noise,
        "game": game,
        "threshold_play": play,
    }

# %%

if MAIN:
    gradient_tiny_play = gradient_threshold_play(
        gradient_methods,
        weights=(2.0, 1.0),
        bias=-2.0,
        query=(1.0, 1.0),
        reference=(0.0, 0.0),
        samples=512,
        seed=161510,
    )
if MAIN:
    gradient_show_figure(gradient_plots.plot_threshold_play(gradient_tiny_play), "13_threshold_play")

# %%

gradient_helper_gradientshap_study_FEATURES = 8

# %%

gradient_helper_gradientshap_study_FORWARD_BATCH = 32_768

# %%

if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__batched_prediction(model: t.nn.Module, values: t.Tensor) -> t.Tensor:
    flat = values.reshape(-1, gradient_helper_gradientshap_study_FEATURES)
    with t.no_grad():
        prediction = t.cat(
            [
                model(flat[start : start + gradient_helper_gradientshap_study_FORWARD_BATCH]).squeeze(-1)
                for start in range(0, len(flat), gradient_helper_gradientshap_study_FORWARD_BATCH)
            ]
        )
    return prediction.reshape(values.shape[:-1])

# %%

if MAIN:
    from typing import Any
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__output_tensor(value: Any, shape: tuple[int, ...], name: str) -> t.Tensor:
    if not isinstance(value, t.Tensor):
        raise TypeError(f"{name} must be a Torch tensor.")
    if value.device.type != "cpu" or value.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if tuple(value.shape) != shape or not bool(t.isfinite(value).all()):
        raise ValueError(f"{name} has the wrong shape or contains nonfinite values.")
    return value

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__checked_curves(
    values: t.Tensor, phi: t.Tensor, order: t.Tensor, methods: Any
) -> dict[str, t.Tensor]:
    orders = order[None, :].expand(len(values), gradient_helper_gradientshap_study_FEATURES).clone()
    result = methods.ranked_coalition_curves(values, phi, orders)
    expected = {
        "orders": (len(values), gradient_helper_gradientshap_study_FEATURES),
        "insertion_masks": (len(values), gradient_helper_gradientshap_study_FEATURES + 1),
        "deletion_masks": (len(values), gradient_helper_gradientshap_study_FEATURES + 1),
        "insertion": (len(values), gradient_helper_gradientshap_study_FEATURES + 1),
        "deletion": (len(values), gradient_helper_gradientshap_study_FEATURES + 1),
    }
    if not isinstance(result, Mapping) or set(result) != set(expected):
        raise ValueError("Replacement curves returned an unexpected payload.")
    for name, shape in expected.items():
        value = result[name]
        if not isinstance(value, t.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Replacement curve {name} has the wrong shape.")
        if name in {"orders", "insertion_masks", "deletion_masks"}:
            if value.dtype != t.int64:
                raise TypeError(f"Replacement curve {name} must be int64.")
        else:
            gradient_helper_gradientshap_study__output_tensor(value, shape, name)
    return dict(result)

# %%

if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__clone_state(state: Mapping[str, t.Tensor]) -> dict[str, t.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError("A model state must be a tensor mapping.")
    result: dict[str, t.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, t.Tensor):
            raise TypeError("State keys must be strings and values must be tensors.")
        if value.device.type != "cpu" or value.dtype != t.float64:
            raise TypeError("Every model-state tensor must be CPU float64.")
        if not bool(t.isfinite(value).all()):
            raise ValueError("Model-state tensors must be finite.")
        result[name] = value.detach().clone()
    return result

# %%

if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__combine_baselines(
    explanation: Mapping[str, t.Tensor], audit: Mapping[str, t.Tensor]
) -> dict[str, t.Tensor]:
    result: dict[str, t.Tensor] = {}
    for name in explanation:
        if name == "base_deep":
            if not t.equal(explanation[name], audit[name]):
                raise ValueError("Primary DeepSHAP base changed across row blocks.")
            result[name] = explanation[name]
        else:
            result[name] = t.cat([explanation[name], audit[name]], dim=0)
    return result

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__combine_mc(
    explanation: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
    if not t.equal(explanation["budgets"], audit["budgets"]):
        raise ValueError("Nested budgets changed between explanation and audit blocks.")
    if not t.equal(explanation["base_full"], audit["base_full"]):
        raise ValueError("Primary background base changed across the audit boundary.")
    return {
        "source_ids": t.cat([explanation["source_ids"], audit["source_ids"]]),
        "baseline_index": t.cat(
            [explanation["baseline_index"], audit["baseline_index"]]
        ),
        "alpha": t.cat([explanation["alpha"], audit["alpha"]]),
        "samples": t.cat([explanation["samples"], audit["samples"]]),
        "budgets": explanation["budgets"],
        "phi_prefix": t.cat(
            [explanation["phi_prefix"], audit["phi_prefix"]], dim=1
        ),
        "se_prefix": t.cat(
            [explanation["se_prefix"], audit["se_prefix"]], dim=1
        ),
        "base_full": explanation["base_full"],
        "prediction": t.cat([explanation["prediction"], audit["prediction"]]),
        "completeness_residual_prefix": t.cat(
            [
                explanation["completeness_residual_prefix"],
                audit["completeness_residual_prefix"],
            ],
            dim=1,
        ),
    }

# %%

gradient_helper_gradientshap_study_MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)

# %%

if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__combine_quadrature(
    explanation: Mapping[str, t.Tensor], audit: Mapping[str, t.Tensor]
) -> dict[str, t.Tensor]:
    result: dict[str, t.Tensor] = {}
    for name in explanation:
        result[name] = t.cat([explanation[name], audit[name]], dim=0)
    return result

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping

def gradient_helper_gradientshap_study__combine_primary(
    prepared: Mapping[str, Any], audit_primary: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        key: {
            "mc": gradient_helper_gradientshap_study__combine_mc(prepared["primary"][key]["mc"], audit_primary[key]["mc"]),
            "quadrature": gradient_helper_gradientshap_study__combine_quadrature(
                prepared["primary"][key]["quadrature"],
                audit_primary[key]["quadrature"],
            ),
            "baselines": gradient_helper_gradientshap_study__combine_baselines(
                prepared["primary"][key]["baselines"],
                audit_primary[key]["baselines"],
            ),
        }
        for key in gradient_helper_gradientshap_study_MODEL_KEYS
    }

# %%

gradient_helper_gradientshap_study_RANDOM_ORDER_SEED = 161480

# %%

if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__fixed_random_orders() -> t.Tensor:
    rng = np.random.default_rng(gradient_helper_gradientshap_study_RANDOM_ORDER_SEED)
    return t.from_numpy(np.stack([rng.permutation(gradient_helper_gradientshap_study_FEATURES) for _ in range(100)]))

# %%

if MAIN:
    from typing import Any
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__float_tensor(value: Any, name: str, dimensions: int) -> t.Tensor:
    tensor = value if isinstance(value, t.Tensor) else t.as_tensor(value)
    if tensor.device.type != "cpu" or tensor.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor or array.")
    if tensor.ndim != dimensions or not bool(t.isfinite(tensor).all()):
        raise ValueError(f"{name} has the wrong shape or contains nonfinite values.")
    return tensor.detach().clone()

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__normalization_tensors(normalization: Mapping[str, Any]) -> tuple[t.Tensor, ...]:
    if not isinstance(normalization, Mapping):
        raise TypeError("normalization must be a mapping.")
    x_mean = gradient_helper_gradientshap_study__float_tensor(normalization.get("x_mean"), "x_mean", 1)
    x_std = gradient_helper_gradientshap_study__float_tensor(normalization.get("x_std"), "x_std", 1)
    y_mean = gradient_helper_gradientshap_study__float_tensor(normalization.get("y_mean"), "y_mean", 0)
    y_std = gradient_helper_gradientshap_study__float_tensor(normalization.get("y_std"), "y_std", 0)
    if x_mean.shape != (gradient_helper_gradientshap_study_FEATURES,) or x_std.shape != (gradient_helper_gradientshap_study_FEATURES,):
        raise ValueError("Normalization must describe exactly eight input features.")
    if bool((x_std <= 0).any()) or not bool(y_std > 0):
        raise ValueError("Training standard deviations must be positive.")
    return x_mean, x_std, y_mean, y_std

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__folding_error(
    kind: str,
    standardized_state: Mapping[str, t.Tensor],
    raw_state: Mapping[str, t.Tensor],
    normalization: Mapping[str, Any],
    x_raw: t.Tensor,
) -> tuple[t.Tensor, float]:
    x_mean, x_std, y_mean, y_std = gradient_helper_gradientshap_study__normalization_tensors(normalization)
    standard = gradient__model_from_state({"kind": kind, "state": standardized_state})
    raw = gradient__model_from_state({"kind": kind, "state": raw_state})
    with t.no_grad():
        expected = standard((x_raw - x_mean) / x_std).squeeze(-1) * y_std + y_mean
        prediction = raw(x_raw).squeeze(-1)
    return prediction, float((prediction - expected).abs().max())

# %%

if MAIN:
    from typing import Any
if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.gradient_gradientshap_data import tensor_mapping_sha256 as gradient_plumbing_gradientshap_data_tensor_mapping_sha256

def gradient_helper_gradientshap_study__identity_hash(**arrays: Any) -> str:
    return gradient_plumbing_gradientshap_data_tensor_mapping_sha256(arrays)

# %%

if MAIN:
    from typing import Any
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__ids_tensor(value: Any, rows: int, name: str) -> t.Tensor:
    tensor = value if isinstance(value, t.Tensor) else t.as_tensor(value)
    if tensor.device.type != "cpu" or tensor.dtype != t.int64:
        raise TypeError(f"{name} must be a CPU int64 tensor or array.")
    if tensor.shape != (rows,) or len(t.unique(tensor)) != rows:
        raise ValueError(f"{name} must contain one unique identity per row.")
    return tensor.detach().clone()

# %%

if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__require_disjoint_ids(**splits: t.Tensor) -> None:
    names = tuple(splits)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            if bool(t.isin(splits[left_name], splits[right_name]).any()):
                raise ValueError(
                    f"{left_name} and {right_name} source-row identities must be disjoint."
                )

# %%

if MAIN:
    from typing import Any

def gradient_helper_gradientshap_study__require_namespace(namespace: Any, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not callable(getattr(namespace, name, None))]
    if missing:
        raise TypeError(f"{label} namespace is missing {missing}.")

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__rows(rows: Mapping[str, Any], name: str) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    if not isinstance(rows, Mapping) or not {"x", "y", "ids"} <= set(rows):
        raise TypeError(f"{name} must contain x, y and ids.")
    x = gradient_helper_gradientshap_study__float_tensor(rows["x"], f"{name}.x", 2)
    y = gradient_helper_gradientshap_study__float_tensor(rows["y"], f"{name}.y", 1)
    if x.shape != (len(y), gradient_helper_gradientshap_study_FEATURES) or len(y) == 0:
        raise ValueError(f"{name} must contain nonempty eight-feature rows.")
    ids = gradient_helper_gradientshap_study__ids_tensor(rows["ids"], len(y), f"{name}.ids")
    return x, y, ids

# %%

if MAIN:
    import hashlib

def gradient_helper_gradientshap_study__seed_words(seed: int, source_id: int, stream_tag: str) -> int:
    name = f"arena16.15/gradientshap/v1/{seed}/{source_id}/{stream_tag}"
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:16], "little")

# %%

gradient_helper_gradientshap_study_HIDDEN = 32

# %%

def gradient_helper_gradientshap_study__state_shapes(kind: str) -> dict[str, tuple[int, ...]]:
    if kind == "mlp":
        return {
            "0.weight": (gradient_helper_gradientshap_study_HIDDEN, gradient_helper_gradientshap_study_FEATURES),
            "0.bias": (gradient_helper_gradientshap_study_HIDDEN,),
            "2.weight": (gradient_helper_gradientshap_study_HIDDEN, gradient_helper_gradientshap_study_HIDDEN),
            "2.bias": (gradient_helper_gradientshap_study_HIDDEN,),
            "4.weight": (1, gradient_helper_gradientshap_study_HIDDEN),
            "4.bias": (1,),
        }
    if kind == "ridge":
        return {"0.weight": (1, gradient_helper_gradientshap_study_FEATURES), "0.bias": (1,)}
    raise ValueError(f"Unsupported fitted-model kind: {kind!r}.")

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping

def gradient_helper_gradientshap_study__validate_fitted(fitted: Mapping[str, Any]) -> None:
    if not isinstance(fitted, Mapping) or fitted.get("phase") != "fitted_before_attribution":
        raise ValueError("Use the complete train-only fitted payload.")
    if not isinstance(fitted.get("models"), Mapping):
        raise ValueError("The fitted payload must contain all model records.")
    if tuple(fitted["models"]) != gradient_helper_gradientshap_study_MODEL_KEYS:
        raise ValueError("The fitted payload must retain all seven models in frozen order.")
    gradient_helper_gradientshap_study__normalization_tensors(fitted.get("normalization"))
    x = gradient_helper_gradientshap_study__float_tensor(fitted.get("train_x_raw"), "train_x_raw", 2)
    y = gradient_helper_gradientshap_study__float_tensor(fitted.get("train_y_raw"), "train_y_raw", 1)
    ids = gradient_helper_gradientshap_study__ids_tensor(fitted.get("train_ids"), len(y), "train_ids")
    if x.shape != (len(y), gradient_helper_gradientshap_study_FEATURES) or len(ids) != len(y):
        raise ValueError("Fitted training arrays have incompatible shapes.")
    for key in gradient_helper_gradientshap_study_MODEL_KEYS:
        expected_kind = "ridge" if key == "ridge" else "mlp"
        if fitted["models"][key].get("kind") != expected_kind:
            raise ValueError(f"{key} has the wrong model kind.")
        gradient__model_from_state(fitted["models"][key])

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Mapping
if MAIN:
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.gradient_gradientshap_data import load_manifest as gradient_plumbing_gradientshap_data_load_manifest
if MAIN:
    import torch as t

def gradient_helper_gradientshap_study__validate_evaluation_inputs(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, t.Tensor]:
    gradient_helper_gradientshap_study__validate_fitted(fitted)
    if prepared.get("phase") != "prepared_before_audit":
        raise ValueError("Persist the prepared payload before audit evaluation.")
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("Audit rows cannot select models, schedules, backgrounds or rankings.")
    if prepared.get("reproduction", {}).get("status") != "PASS":
        raise ValueError("The 21-state reproduction gate must pass before evaluation.")
    train_x, train_y, train_ids = gradient_helper_gradientshap_study__rows(train, "train")
    explain_x, explain_y, explain_ids = gradient_helper_gradientshap_study__rows(explain, "explain")
    audit_x, audit_y, audit_ids = gradient_helper_gradientshap_study__rows(audit, "audit")
    gradient_helper_gradientshap_study__require_disjoint_ids(train=train_ids, explanation=explain_ids, audit=audit_ids)
    if not t.equal(train_ids, fitted["train_ids"]):
        raise ValueError("Training identities changed after fitting.")
    if not t.equal(train_x, fitted["train_x_raw"]) or not t.equal(
        train_y, fitted["train_y_raw"]
    ):
        raise ValueError("Training values changed after fitting.")
    if not t.equal(explain_ids, prepared["explain_ids"]):
        raise ValueError("Explanation identities changed after preparation.")
    if not t.equal(explain_x, prepared["explain_x_raw"]) or not t.equal(
        explain_y, prepared["explain_targets"]
    ):
        raise ValueError("Explanation values changed after preparation.")
    manifest = gradient_plumbing_gradientshap_data_load_manifest()
    expected_audit = t.as_tensor(manifest["splits"]["audit"], dtype=t.int64)
    if not t.equal(audit_ids, expected_audit):
        raise ValueError("Audit identities differ from the literal pinned split.")
    identities = prepared.get("payload_identities", {})
    if identities.get("train") != gradient_helper_gradientshap_study__identity_hash(ids=train_ids, x=train_x, y=train_y):
        raise ValueError("Prepared training identity seal changed.")
    if identities.get("explain") != gradient_helper_gradientshap_study__identity_hash(
        ids=explain_ids, x=explain_x, y=explain_y
    ):
        raise ValueError("Prepared explanation identity seal changed.")
    expected_backgrounds = gradient__background_indices(train_ids, manifest)
    for name, indices in expected_backgrounds.items():
        item = prepared.get("backgrounds", {}).get(name, {})
        if not t.equal(item.get("indices"), indices):
            raise ValueError(f"Prepared {name} background indices changed.")
        if not t.equal(item.get("source_ids"), train_ids[indices]):
            raise ValueError(f"Prepared {name} background identities changed.")
        if not t.equal(item.get("x_raw"), train_x[indices]):
            raise ValueError(f"Prepared {name} background values changed.")
        expected_hash = gradient_helper_gradientshap_study__identity_hash(
            indices=indices, ids=train_ids[indices], x=train_x[indices]
        )
        if identities.get("backgrounds", {}).get(name) != expected_hash:
            raise ValueError(f"Prepared {name} background seal changed.")
    primary_mc = prepared.get("primary", {}).get(gradient_helper_gradientshap_study_MODEL_KEYS[0], {}).get("mc", {})
    schedule_manifest = prepared.get("schedule_manifest", {})
    if schedule_manifest.get("baseline_index_sha256") != gradient_helper_gradientshap_study__identity_hash(
        value=primary_mc.get("baseline_index")
    ):
        raise ValueError("Prepared donor schedule seal changed.")
    if schedule_manifest.get("alpha_sha256") != gradient_helper_gradientshap_study__identity_hash(
        value=primary_mc.get("alpha")
    ):
        raise ValueError("Prepared alpha schedule seal changed.")
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_ids": train_ids,
        "explain_x": explain_x,
        "explain_y": explain_y,
        "explain_ids": explain_ids,
        "audit_x": audit_x,
        "audit_y": audit_y,
        "audit_ids": audit_ids,
    }

# %%

if MAIN:
    deep_methods = SimpleNamespace(**deep_module_deepshap_methods_namespace())
if MAIN:
    _batched_prediction = gradient_helper_gradientshap_study__batched_prediction
if MAIN:
    _checked_curves = gradient_helper_gradientshap_study__checked_curves
if MAIN:
    gradient__clone_state = gradient_helper_gradientshap_study__clone_state
if MAIN:
    _combine_baselines = gradient_helper_gradientshap_study__combine_baselines
if MAIN:
    _combine_mc = gradient_helper_gradientshap_study__combine_mc
if MAIN:
    _combine_primary = gradient_helper_gradientshap_study__combine_primary
if MAIN:
    _combine_quadrature = gradient_helper_gradientshap_study__combine_quadrature
if MAIN:
    gradient__fixed_random_orders = gradient_helper_gradientshap_study__fixed_random_orders
if MAIN:
    gradient__float_tensor = gradient_helper_gradientshap_study__float_tensor
if MAIN:
    gradient__folding_error = gradient_helper_gradientshap_study__folding_error
if MAIN:
    _identity_hash = gradient_helper_gradientshap_study__identity_hash
if MAIN:
    gradient__ids_tensor = gradient_helper_gradientshap_study__ids_tensor
if MAIN:
    gradient__normalization_tensors = gradient_helper_gradientshap_study__normalization_tensors
if MAIN:
    gradient__output_tensor = gradient_helper_gradientshap_study__output_tensor
if MAIN:
    gradient__require_disjoint_ids = gradient_helper_gradientshap_study__require_disjoint_ids
if MAIN:
    _require_namespace = gradient_helper_gradientshap_study__require_namespace
if MAIN:
    gradient__rows = gradient_helper_gradientshap_study__rows
if MAIN:
    _seed_words = gradient_helper_gradientshap_study__seed_words
if MAIN:
    gradient__state_shapes = gradient_helper_gradientshap_study__state_shapes
if MAIN:
    gradient__validate_evaluation_inputs = gradient_helper_gradientshap_study__validate_evaluation_inputs
if MAIN:
    gradient__validate_fitted = gradient_helper_gradientshap_study__validate_fitted

# %%

gradient_FEATURES = 8

gradient_HIDDEN = 32

gradient_TRAINING_STEPS = 800

gradient_LEARNING_RATE = 0.003

gradient_WEIGHT_DECAY = 0.0001

gradient_RIDGE_PENALTY = 0.01

gradient_MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)

gradient_MODEL_SEEDS = (161410, 161411, 161412)

gradient_SHUFFLE_SEEDS = (161420, 161421, 161422)

BACKGROUND_SPECS = {
    "primary64": (161460, 64),
    "nested16": (161460, 16),
    "nested128": (161460, 128),
    "independent64_a": (161461, 64),
    "independent64_b": (161462, 64),
}

BUDGETS = (8, 32, 128, 512, 2048)

QUADRATURE_NODES = (256, 512)

NOISE_SIGMAS = (0.0, 0.02, 0.10)

PRIMARY_SCHEDULE_SEED = 161510

REPLICATE_SCHEDULE_SEED = 161511

NOISE_SCHEDULE_SEED = 161512

REFERENCE_SEEDS = {"shap": 161513, "captum_numpy": 161514, "captum_torch": 161515}

gradient_RANDOM_ORDER_SEED = 161480

REPLICATES = 64

REPLICATE_MODELS = ("clean_0", "shuffled_0", "ridge")

PATH_DISTANCE_SAMPLES = 128

COALITION_DISTANCE_DONORS = 8

FORWARD_BATCH = 32_768

DISTANCE_BATCH = 1_024

# %%

gradient_FEATURE_NAMES = (
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "Superplasticizer",
    "Coarse aggregate",
    "Fine aggregate",
    "Age",
)

if MAIN:
    gradient_MODEL_LABELS = {
        **{f"clean_{i}": f"clean MLP seed {seed}" for i, seed in enumerate(gradient_MODEL_SEEDS)},
        **{
            f"shuffled_{i}": f"shuffled MLP seed {seed}"
            for i, seed in enumerate(gradient_MODEL_SEEDS)
        },
        "ridge": "ridge control",
    }

gradient_UNITS = {
    "input": "source feature units",
    "target": "MPa",
    "prediction": "MPa",
    "base": "MPa",
    "attribution": "MPa",
    "game": "MPa",
    "squared_error": "MPa squared",
    "distance": "train-standardized Euclidean distance",
}

gradient_METHOD_NAMES = (
    "sample_path_schedule",
    "interpolate_paths",
    "input_gradients",
    "expected_gradient_samples",
    "summarize_expected_gradients",
    "expected_gradients",
    "gauss_legendre_pairs",
    "noisy_expected_gradient_samples",
    "ranked_coalition_curves",
)

DEEP_METHOD_NAMES = (
    "rescale_multipliers",
    "background_deepshap",
    "coalition_masks",
    "coalition_values",
    "exact_shapley",
    "gradient_delta",
)

# %%

def gradient__model_from_state(record: Mapping[str, Any]) -> t.nn.Sequential:
    """Rebuild one supported raw-input/raw-MPa model from its retained state."""
    if not isinstance(record, Mapping) or not {"kind", "state"} <= set(record):
        raise TypeError("A fitted model record must contain kind and state.")
    kind = str(record["kind"])
    state = gradient__clone_state(record["state"])
    shapes = gradient__state_shapes(kind)
    if set(state) != set(shapes):
        raise ValueError(f"{kind} state has unexpected parameter names.")
    if any(tuple(state[name].shape) != shape for name, shape in shapes.items()):
        raise ValueError(f"{kind} state has an unexpected parameter shape.")
    if kind == "mlp":
        model = t.nn.Sequential(
            t.nn.Linear(gradient_FEATURES, gradient_HIDDEN, dtype=t.float64),
            t.nn.ReLU(),
            t.nn.Linear(gradient_HIDDEN, gradient_HIDDEN, dtype=t.float64),
            t.nn.ReLU(),
            t.nn.Linear(gradient_HIDDEN, 1, dtype=t.float64),
        )
    else:
        model = t.nn.Sequential(t.nn.Linear(gradient_FEATURES, 1, dtype=t.float64))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# %%

def gradient__training_normalization(train: Mapping[str, Any]) -> dict[str, t.Tensor]:
    """Compute the four train-only population moments used by every fit."""
    x_raw, y_raw, _ = gradient__rows(train, "train")
    normalization = {
        "x_mean": x_raw.mean(0),
        "x_std": x_raw.std(0, correction=0),
        "y_mean": y_raw.mean(),
        "y_std": y_raw.std(correction=0),
    }
    gradient__normalization_tensors(normalization)
    return normalization

# %%

def gradient__train_mlp(
    model: t.nn.Sequential,
    x_standardized: t.Tensor,
    target_standardized: t.Tensor,
) -> tuple[dict[str, t.Tensor], t.Tensor]:
    """Run exactly 800 full-batch float64 Adam updates and retain every loss."""
    optimizer = t.optim.Adam(
        model.parameters(), lr=gradient_LEARNING_RATE, weight_decay=gradient_WEIGHT_DECAY
    )
    losses: list[t.Tensor] = []
    model.train()
    with t.enable_grad():
        for _ in range(gradient_TRAINING_STEPS):
            optimizer.zero_grad()
            loss = (model(x_standardized).squeeze(-1) - target_standardized).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().clone())
    return gradient__clone_state(model.state_dict()), t.stack(losses)

def gradient__fold_mlp_state(
    state: Mapping[str, t.Tensor], normalization: Mapping[str, Any]
) -> dict[str, t.Tensor]:
    """Fold train normalization into the first and last MLP affine layers."""
    folded = gradient__clone_state(state)
    if set(folded) != set(gradient__state_shapes("mlp")):
        raise ValueError("Cannot fold an unexpected MLP state.")
    x_mean, x_std, y_mean, y_std = gradient__normalization_tensors(normalization)
    first_weight = folded["0.weight"].clone()
    folded["0.weight"] = first_weight / x_std[None, :]
    folded["0.bias"] = folded["0.bias"] - first_weight @ (x_mean / x_std)
    folded["4.weight"] = y_std * folded["4.weight"]
    folded["4.bias"] = y_std * folded["4.bias"] + y_mean
    return folded

def gradient__fold_ridge_state(
    state: Mapping[str, t.Tensor], normalization: Mapping[str, Any]
) -> dict[str, t.Tensor]:
    """Fold standardized ridge coefficients into raw feature and MPa units."""
    folded = gradient__clone_state(state)
    if set(folded) != set(gradient__state_shapes("ridge")):
        raise ValueError("Cannot fold an unexpected ridge state.")
    x_mean, x_std, y_mean, y_std = gradient__normalization_tensors(normalization)
    weight = folded["0.weight"].clone()
    folded["0.weight"] = y_std * weight / x_std[None, :]
    folded["0.bias"] = y_std * (
        folded["0.bias"] - (weight * (x_mean / x_std)[None, :]).sum(1)
    ) + y_mean
    return folded

# %%

def gradient_fit_one_mlp_control(
    train: Mapping[str, Any],
    normalization: Mapping[str, Any],
    seed: int,
    shuffle_seed: int | None,
) -> dict[str, Any]:
    """Fit and fully retain one clean or shuffled 8-32-32-1 control."""
    if seed not in gradient_MODEL_SEEDS:
        raise ValueError("seed must be one of the three frozen model seeds.")
    if shuffle_seed is not None and shuffle_seed not in gradient_SHUFFLE_SEEDS:
        raise ValueError("shuffle_seed must be a frozen paired shuffle seed or None.")
    x_raw, y_raw, _ = gradient__rows(train, "train")
    x_mean, x_std, y_mean, y_std = gradient__normalization_tensors(normalization)
    x = (x_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std
    t.manual_seed(seed)
    model = t.nn.Sequential(
        t.nn.Linear(gradient_FEATURES, gradient_HIDDEN, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(gradient_HIDDEN, gradient_HIDDEN, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(gradient_HIDDEN, 1, dtype=t.float64),
    )
    initial_state = gradient__clone_state(model.state_dict())
    permutation = t.arange(len(y), dtype=t.int64)
    if shuffle_seed is not None:
        permutation = t.from_numpy(np.random.default_rng(shuffle_seed).permutation(len(y)))
    standardized_state, loss_trace = gradient__train_mlp(model, x, y[permutation])
    raw_state = gradient__fold_mlp_state(standardized_state, normalization)
    prediction, folding_error = gradient__folding_error(
        "mlp", standardized_state, raw_state, normalization, x_raw
    )
    if folding_error > 1e-10:
        raise AssertionError("Raw-unit MLP folding changed its training predictions.")
    return {
        "kind": "mlp",
        "control": "clean" if shuffle_seed is None else "shuffled",
        "seed": seed,
        "initial_state": initial_state,
        "standardized_state": standardized_state,
        "state": raw_state,
        "loss_trace": loss_trace,
        "target_permutation": permutation,
        "training_targets_raw": y_raw[permutation].clone(),
        "training_prediction": prediction,
        "folding_max_abs_error": folding_error,
        "parameters": {
            "architecture": [gradient_FEATURES, gradient_HIDDEN, gradient_HIDDEN, 1],
            "steps": gradient_TRAINING_STEPS,
            "learning_rate": gradient_LEARNING_RATE,
            "weight_decay": gradient_WEIGHT_DECAY,
            "optimizer": "Adam",
            "risk": "mean_squared_error",
            "batch": "full_training_split",
            "shuffle_seed": shuffle_seed,
            "dtype": "float64",
            "parameter_count": sum(value.numel() for value in raw_state.values()),
        },
    }

# %%

def gradient_fit_ridge_control(
    train: Mapping[str, Any], normalization: Mapping[str, Any]
) -> dict[str, Any]:
    """Solve the fixed penalty-0.01 standardized ridge normal equations once."""
    x_raw, y_raw, _ = gradient__rows(train, "train")
    x_mean, x_std, y_mean, y_std = gradient__normalization_tensors(normalization)
    x = (x_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std
    design = t.cat([x, t.ones((len(x), 1), dtype=t.float64)], dim=1)
    penalty = t.eye(gradient_FEATURES + 1, dtype=t.float64) * gradient_RIDGE_PENALTY
    penalty[-1, -1] = 0.0
    coefficient = t.linalg.solve(
        design.T @ design / len(x) + penalty,
        design.T @ y / len(x),
    )
    standardized_state = {
        "0.weight": coefficient[:gradient_FEATURES][None, :].clone(),
        "0.bias": coefficient[gradient_FEATURES:].clone(),
    }
    raw_state = gradient__fold_ridge_state(standardized_state, normalization)
    prediction, folding_error = gradient__folding_error(
        "ridge", standardized_state, raw_state, normalization, x_raw
    )
    if folding_error > 1e-10:
        raise AssertionError("Raw-unit ridge folding changed its training predictions.")
    standard_model = gradient__model_from_state({"kind": "ridge", "state": standardized_state})
    with t.no_grad():
        ridge_loss = (standard_model(x).squeeze(-1) - y).square().mean()
    return {
        "kind": "ridge",
        "control": "ridge",
        "seed": 0,
        "initial_state": {},
        "standardized_state": standardized_state,
        "state": raw_state,
        "loss_trace": ridge_loss[None].clone(),
        "target_permutation": t.arange(len(y), dtype=t.int64),
        "training_targets_raw": y_raw.clone(),
        "training_prediction": prediction,
        "folding_max_abs_error": folding_error,
        "parameters": {
            "penalty": gradient_RIDGE_PENALTY,
            "penalize_intercept": False,
            "risk": "mean_squared_error",
            "dtype": "float64",
            "parameter_count": gradient_FEATURES + 1,
        },
    }

# %%

def gradient_fit_models(train: Mapping[str, Any]) -> dict[str, Any]:
    """Refit all seven known predictors from the pinned training rows."""
    x_raw, y_raw, train_ids = gradient__rows(train, "train")
    t.set_num_threads(4)
    t.use_deterministic_algorithms(True)
    normalization = gradient__training_normalization(train)
    models: dict[str, Any] = {}
    for index, seed in enumerate(gradient_MODEL_SEEDS):
        models[f"clean_{index}"] = gradient_fit_one_mlp_control(
            train, normalization, seed, None
        )
    for index, seed in enumerate(gradient_MODEL_SEEDS):
        models[f"shuffled_{index}"] = gradient_fit_one_mlp_control(
            train, normalization, seed, gradient_SHUFFLE_SEEDS[index]
        )
    models["ridge"] = gradient_fit_ridge_control(train, normalization)
    for record in models.values():
        record["state_hashes"] = {
            "initial_state": tensor_mapping_sha256(record["initial_state"]),
            "standardized_state": tensor_mapping_sha256(record["standardized_state"]),
            "raw_folded_state": tensor_mapping_sha256(record["state"]),
        }
    return {
        "models": models,
        "normalization": normalization,
        "train_ids": train_ids,
        "train_x_raw": x_raw,
        "train_y_raw": y_raw,
        "units": dict(gradient_UNITS),
        "feature_names": gradient_FEATURE_NAMES,
        "model_labels": dict(gradient_MODEL_LABELS),
        "phase": "fitted_before_attribution",
    }

# %%

def verify_16_14_reproduction(
    fitted: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Hard-fail unless all 21 states and every deterministic fit identity agree."""
    gradient__validate_fitted(fitted)
    evidence = manifest.get("known_16_14_evidence", {})
    expected_hashes = evidence.get("model_state_hashes", {})
    if tuple(expected_hashes) != gradient_MODEL_KEYS:
        raise ValueError("Manifest must contain all seven known 16.14 state records.")
    observed: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    state_fields = {
        "initial_state": "initial_state",
        "standardized_state": "standardized_state",
        "raw_folded_state": "state",
    }
    for key in gradient_MODEL_KEYS:
        record = fitted["models"][key]
        observed[key] = {
            label: tensor_mapping_sha256(record[field])
            for label, field in state_fields.items()
        }
        for label in state_fields:
            if observed[key][label] != expected_hashes[key][label]:
                mismatches.append(f"{key}.{label}")
        if record.get("state_hashes") != observed[key]:
            mismatches.append(f"{key}.retained_state_hashes")

    expected_train_ids = t.as_tensor(manifest.get("splits", {}).get("train"), dtype=t.int64)
    train_ids_match = t.equal(fitted["train_ids"], expected_train_ids)
    if not train_ids_match:
        mismatches.append("train_ids")
    normalization = gradient__training_normalization(
        {
            "x": fitted["train_x_raw"],
            "y": fitted["train_y_raw"],
            "ids": fitted["train_ids"],
        }
    )
    if any(
        not t.equal(normalization[name], fitted["normalization"][name])
        for name in normalization
    ):
        mismatches.append("normalization")

    recipe = manifest.get("model_recipe", {})
    training_recipe_match = (
        recipe.get("model_keys") == list(gradient_MODEL_KEYS)
        and recipe.get("model_seeds") == list(gradient_MODEL_SEEDS)
        and recipe.get("shuffle_seeds") == list(gradient_SHUFFLE_SEEDS)
        and recipe.get("training_steps") == gradient_TRAINING_STEPS
        and recipe.get("learning_rate") == gradient_LEARNING_RATE
        and recipe.get("weight_decay") == gradient_WEIGHT_DECAY
        and recipe.get("ridge_penalty") == gradient_RIDGE_PENALTY
    )
    for index, key in enumerate(gradient_MODEL_KEYS):
        record = fitted["models"][key]
        expected_permutation = t.arange(len(fitted["train_ids"]), dtype=t.int64)
        if key.startswith("shuffled_"):
            expected_permutation = t.from_numpy(
                np.random.default_rng(gradient_SHUFFLE_SEEDS[index - 3]).permutation(
                    len(fitted["train_ids"])
                )
            )
        if not t.equal(record.get("target_permutation"), expected_permutation):
            mismatches.append(f"{key}.target_permutation")
        expected_targets = fitted["train_y_raw"][expected_permutation]
        if not t.equal(record.get("training_targets_raw"), expected_targets):
            mismatches.append(f"{key}.training_targets_raw")
        expected_steps = 1 if key == "ridge" else gradient_TRAINING_STEPS
        trace = record.get("loss_trace")
        if not isinstance(trace, t.Tensor) or trace.shape != (expected_steps,):
            mismatches.append(f"{key}.loss_trace")
        with t.no_grad():
            prediction = gradient__model_from_state(record)(fitted["train_x_raw"]).squeeze(-1)
        if not t.equal(record.get("training_prediction"), prediction):
            mismatches.append(f"{key}.training_prediction")
        if float(record.get("folding_max_abs_error", float("inf"))) > 1e-10:
            mismatches.append(f"{key}.folding_max_abs_error")
    if not training_recipe_match:
        mismatches.append("training_recipe")
    if mismatches:
        raise AssertionError(
            "16.14 reproduction mismatch before attribution: " + ", ".join(mismatches)
        )
    return {
        "status": "PASS",
        "known_section": "16.14",
        "fitted_artifact_sha256": evidence.get("fitted_artifact_sha256"),
        "model_state_hashes": observed,
        "expected_model_state_hashes": expected_hashes,
        "mismatches": (),
        "train_ids_match": train_ids_match,
        "training_recipe_match": training_recipe_match,
    }

# %%

def gradient__background_indices(
    train_ids: Any, manifest: Mapping[str, Any]
) -> dict[str, t.Tensor]:
    """Recreate all five background index arrays and verify every donor identity."""
    ids = gradient__ids_tensor(train_ids, len(train_ids), "train_ids")
    declared = manifest.get("backgrounds", {})
    if tuple(declared) != tuple(BACKGROUND_SPECS):
        raise ValueError("Manifest background names or order changed.")
    result: dict[str, t.Tensor] = {}
    for name, (seed, size) in BACKGROUND_SPECS.items():
        order = t.from_numpy(gradient_background_order(ids.numpy(), seed))
        indices = order[:size].clone()
        item = declared[name]
        if item.get("seed") != seed or item.get("rows") != size:
            raise ValueError(f"{name} changed its seed or donor count.")
        expected_indices = t.as_tensor(item.get("train_indices"), dtype=t.int64)
        expected_ids = t.as_tensor(item.get("source_ids"), dtype=t.int64)
        if not t.equal(indices, expected_indices) or not t.equal(ids[indices], expected_ids):
            raise ValueError(f"{name} donor indices or source identities changed.")
        result[name] = indices
    return result

def _primary_schedule(
    rows: Mapping[str, Any], background_rows: int, methods: Any
) -> dict[str, t.Tensor]:
    """Create the one shared K=2048 primary schedule for a named row block."""
    _require_namespace(methods, gradient_METHOD_NAMES, "Student methods")
    _, _, source_ids = gradient__rows(rows, "rows")
    indices, alpha = methods.sample_path_schedule(
        source_ids, background_rows, max(BUDGETS), PRIMARY_SCHEDULE_SEED
    )
    return {"source_ids": source_ids, "baseline_index": indices, "alpha": alpha}

# %%

def _monte_carlo_block(
    model: t.nn.Module,
    rows: Mapping[str, Any],
    background: t.Tensor,
    schedule: Mapping[str, t.Tensor],
    methods: Any,
) -> dict[str, Any]:
    """Retain all per-draw contributions and every declared nested prefix."""
    x, _, source_ids = gradient__rows(rows, "rows")
    if not t.equal(source_ids, schedule["source_ids"]):
        raise ValueError("Schedule source identities do not match the explained rows.")
    samples = methods.expected_gradient_samples(
        model, x, background, schedule["baseline_index"], schedule["alpha"], FORWARD_BATCH
    )
    samples = gradient__output_tensor(samples, (len(x), max(BUDGETS), gradient_FEATURES), "samples")
    phi_prefix, se_prefix = methods.summarize_expected_gradients(samples, BUDGETS)
    phi_prefix = gradient__output_tensor(phi_prefix, (len(BUDGETS), len(x), gradient_FEATURES), "phi_prefix")
    se_prefix = gradient__output_tensor(se_prefix, (len(BUDGETS), len(x), gradient_FEATURES), "se_prefix")
    with t.no_grad():
        base = model(background).squeeze(-1).mean()
        prediction = model(x).squeeze(-1)
    residual = phi_prefix.sum(-1) - (prediction[None, :] - base)
    return {
        "source_ids": source_ids,
        "baseline_index": schedule["baseline_index"],
        "alpha": schedule["alpha"],
        "samples": samples,
        "budgets": t.tensor(BUDGETS, dtype=t.int64),
        "phi_prefix": phi_prefix,
        "se_prefix": se_prefix,
        "base_full": base.detach(),
        "prediction": prediction.detach(),
        "completeness_residual_prefix": residual.detach(),
    }

# %%

def _quadrature_block(
    model: t.nn.Module,
    rows: Mapping[str, Any],
    background: t.Tensor,
    methods: Any,
) -> dict[str, t.Tensor]:
    """Retain donor-wise Q=256/Q=512 numerical path controls and drift."""
    x, _, _ = gradient__rows(rows, "rows")
    low, high = QUADRATURE_NODES
    pair_low = methods.gauss_legendre_pairs(
        model, x, background, low, row_batch=4, forward_batch=FORWARD_BATCH
    )
    pair_high = methods.gauss_legendre_pairs(
        model, x, background, high, row_batch=4, forward_batch=FORWARD_BATCH
    )
    shape = (len(x), len(background), gradient_FEATURES)
    pair_low = gradient__output_tensor(pair_low, shape, f"pair_{low}")
    pair_high = gradient__output_tensor(pair_high, shape, f"pair_{high}")
    mean_low, mean_high = pair_low.mean(1), pair_high.mean(1)
    return {
        f"pair_{low}": pair_low,
        f"pair_{high}": pair_high,
        f"mean_{low}": mean_low,
        f"mean_{high}": mean_high,
        "drift": mean_high - mean_low,
    }

# %%

def _coalition_hybrid_predictions(
    model: t.nn.Module,
    x: t.Tensor,
    background: t.Tensor,
    masks: t.Tensor,
    *,
    row_batch: int = 4,
    forward_batch: int = FORWARD_BATCH,
) -> t.Tensor:
    """Retain every query/coalition/donor prediction in bounded forward batches."""
    result = t.empty((len(x), len(masks), len(background)), dtype=t.float64)
    with t.no_grad():
        for start in range(0, len(x), row_batch):
            query = x[start : start + row_batch]
            hybrids = t.where(
                masks[None, :, None, :],
                query[:, None, None, :],
                background[None, None, :, :],
            )
            flat = hybrids.reshape(-1, gradient_FEATURES)
            output = t.cat(
                [
                    model(flat[index : index + forward_batch]).squeeze(-1)
                    for index in range(0, len(flat), forward_batch)
                ]
            )
            result[start : start + len(query)] = output.reshape(
                len(query), len(masks), len(background)
            )
    return result

# %%

def _comparison_baselines(
    model: t.nn.Module,
    rows: Mapping[str, Any],
    background: t.Tensor,
    deep_methods: Any,
) -> dict[str, t.Tensor]:
    """Retain exact-game hybrids, DeepSHAP pairs, and endpoint gradients."""
    _require_namespace(deep_methods, DEEP_METHOD_NAMES, "DeepSHAP control")
    x, _, _ = gradient__rows(rows, "rows")
    with t.no_grad():
        prediction = model(x).squeeze(-1)
    masks = deep_methods.coalition_masks(gradient_FEATURES)
    hybrid_prediction = _coalition_hybrid_predictions(model, x, background, masks)
    coalition_values = hybrid_prediction.mean(-1)
    phi_exact = deep_methods.exact_shapley(coalition_values)
    base_deep, phi_deep = deep_methods.background_deepshap(model, x, background)
    multipliers = deep_methods.rescale_multipliers(model, x, background)
    deep_pair = multipliers * (x[:, None, :] - background[None, :, :])
    phi_gradient = deep_methods.gradient_delta(model, x, background)
    return {
        "prediction": gradient__output_tensor(prediction, (len(x),), "prediction"),
        "hybrid_prediction": gradient__output_tensor(
            hybrid_prediction,
            (len(x), 2**gradient_FEATURES, len(background)),
            "hybrid_prediction",
        ),
        "coalition_values": gradient__output_tensor(
            coalition_values, (len(x), 2**gradient_FEATURES), "coalition_values"
        ),
        "phi_exact": gradient__output_tensor(phi_exact, (len(x), gradient_FEATURES), "phi_exact"),
        "base_deep": gradient__output_tensor(base_deep, (), "base_deep"),
        "phi_deep": gradient__output_tensor(phi_deep, (len(x), gradient_FEATURES), "phi_deep"),
        "deep_pair": gradient__output_tensor(
            deep_pair, (len(x), len(background), gradient_FEATURES), "deep_pair"
        ),
        "phi_gradient": gradient__output_tensor(
            phi_gradient, (len(x), gradient_FEATURES), "phi_gradient"
        ),
    }

# %%

def _global_rankings(primary: Mapping[str, Any]) -> dict[str, dict[str, t.Tensor]]:
    """Freeze stable mean-absolute feature orders from explanation rows only."""
    fields = {
        "exact": lambda item: item["baselines"]["phi_exact"],
        "deep": lambda item: item["baselines"]["phi_deep"],
        "gradient": lambda item: item["baselines"]["phi_gradient"],
        "eg_k2048": lambda item: item["mc"]["phi_prefix"][-1],
        "quadrature_512": lambda item: item["quadrature"][f"mean_{QUADRATURE_NODES[1]}"],
    }
    return {
        method: {
            key: t.argsort(getter(primary[key]).abs().mean(0), descending=True, stable=True)
            for key in gradient_MODEL_KEYS
        }
        for method, getter in fields.items()
    }

# %%

def gradient_prepare_study(
    fitted: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    *,
    methods: Any,
    deep_methods: Any,
) -> dict[str, Any]:
    """Prepare and seal all explanation-side choices before audit is supplied."""
    _require_namespace(methods, gradient_METHOD_NAMES, "Student methods")
    _require_namespace(deep_methods, DEEP_METHOD_NAMES, "DeepSHAP control")
    manifest = load_manifest()
    reproduction = verify_16_14_reproduction(fitted, manifest)
    train_x, train_y, train_ids = gradient__rows(train, "train")
    explain_x, explain_y, explain_ids = gradient__rows(explain, "explain")
    gradient__require_disjoint_ids(train=train_ids, explanation=explain_ids)
    if not t.equal(train_ids, fitted["train_ids"]):
        raise ValueError("Training identities changed after fitting.")
    if not t.equal(train_x, fitted["train_x_raw"]) or not t.equal(
        train_y, fitted["train_y_raw"]
    ):
        raise ValueError("Training values changed after fitting.")
    expected_explain = t.as_tensor(manifest["splits"]["explain"], dtype=t.int64)
    if not t.equal(explain_ids, expected_explain):
        raise ValueError("Explanation identities differ from the pinned split.")

    indices = gradient__background_indices(train_ids, manifest)
    backgrounds = {
        name: {
            "seed": BACKGROUND_SPECS[name][0],
            "indices": index,
            "source_ids": train_ids[index],
            "x_raw": train_x[index],
        }
        for name, index in indices.items()
    }
    primary_background = backgrounds["primary64"]["x_raw"]
    schedule = _primary_schedule(explain, len(primary_background), methods)
    primary: dict[str, Any] = {}
    for key in gradient_MODEL_KEYS:
        model = gradient__model_from_state(fitted["models"][key])
        primary[key] = {
            "mc": _monte_carlo_block(model, explain, primary_background, schedule, methods),
            "quadrature": _quadrature_block(model, explain, primary_background, methods),
            "baselines": _comparison_baselines(
                model, explain, primary_background, deep_methods
            ),
        }
    random_orders = gradient__fixed_random_orders()
    schedule_manifest = {
        "namespace": "arena16.15/gradientshap/v1",
        "seed": PRIMARY_SCHEDULE_SEED,
        "budgets": t.tensor(BUDGETS, dtype=t.int64),
        "source_ids": explain_ids,
        "baseline_index_sha256": _identity_hash(value=schedule["baseline_index"]),
        "alpha_sha256": _identity_hash(value=schedule["alpha"]),
    }
    payload_identities = {
        "train": _identity_hash(ids=train_ids, x=train_x, y=train_y),
        "explain": _identity_hash(ids=explain_ids, x=explain_x, y=explain_y),
        "backgrounds": {
            name: _identity_hash(
                indices=item["indices"], ids=item["source_ids"], x=item["x_raw"]
            )
            for name, item in backgrounds.items()
        },
    }
    return {
        "phase": "prepared_before_audit",
        "audit_used_for_selection": False,
        "reproduction": reproduction,
        "explain_ids": explain_ids,
        "explain_x_raw": explain_x,
        "explain_targets": explain_y,
        "backgrounds": backgrounds,
        "primary": primary,
        "rankings": _global_rankings(primary),
        "random_orders": random_orders,
        "schedule_manifest": schedule_manifest,
        "payload_identities": payload_identities,
        "units": dict(gradient_UNITS),
        "feature_names": gradient_FEATURE_NAMES,
        "model_labels": dict(gradient_MODEL_LABELS),
    }

# %%

gradient_helper_run_native_EXTERNAL_DETERMINISTIC_ARTIFACT = "replicate_samples.pt"

# %%

gradient_helper_run_native_SMALL_DETERMINISTIC_ARTIFACTS = (
    "fitted.pt",
    "prepared.pt",
    "evaluated.pt",
    "reference.pt",
    "diagnostics.pt",
    "play.pt",
)

# %%

if MAIN:
    gradient_helper_run_native_DETERMINISTIC_ARTIFACTS = gradient_helper_run_native_SMALL_DETERMINISTIC_ARTIFACTS + (
        gradient_helper_run_native_EXTERNAL_DETERMINISTIC_ARTIFACT,
    )

# %%

gradient_helper_run_native_REPLICATE_SHAPE = (64, 3, 64, 2048, 8)

# %%

if MAIN:
    import math

if MAIN:
    gradient_helper_run_native_REPLICATE_ELEMENTS = math.prod(gradient_helper_run_native_REPLICATE_SHAPE)

# %%

gradient_helper_run_native_TIMING_ARTIFACT = "timings.pt"

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    import numpy as np
if MAIN:
    import torch

def gradient_helper_run_native__array(value: Any) -> tuple[np.ndarray, str]:
    if isinstance(value, np.ndarray):
        array = value
        kind = "numpy.ndarray"
    elif isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Scientific payload tensors must be on CPU before persistence.")
        array = value.detach().numpy()
        kind = "torch.Tensor"
    else:
        raise TypeError(f"Expected NumPy array or Torch tensor, found {type(value).__name__}.")
    if array.dtype.hasobject:
        raise TypeError("Object arrays cannot enter deterministic evidence.")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array, kind

# %%

if MAIN:
    gradient_helper_run_native_HASH_CHUNK_BYTES = 64 * 1024 * 1024

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    import hashlib

def gradient_helper_run_native__array_value_sha256(value: Any) -> str:
    """Hash contiguous array bytes in bounded views, including multi-gigabyte arrays."""
    array, _ = gradient_helper_run_native__array(value)
    byte_view = memoryview(array).cast("B")
    result = hashlib.sha256()
    for start in range(0, len(byte_view), gradient_helper_run_native_HASH_CHUNK_BYTES):
        result.update(byte_view[start : start + gradient_helper_run_native_HASH_CHUNK_BYTES])
    return result.hexdigest()

# %%

if MAIN:
    from typing import Any, Iterator

def gradient_helper_run_native__array_identity(value: Any) -> dict[str, Any]:
    array, kind = gradient_helper_run_native__array(value)
    return {
        "kind": kind,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "value_sha256": gradient_helper_run_native__array_value_sha256(value),
    }

# %%

if MAIN:
    from pathlib import Path

def gradient_helper_run_native_manifest_path(artifact: Path) -> Path:
    return Path(artifact).with_suffix(".manifest.json")

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    from pathlib import Path

def gradient_helper_run_native__external_reference(samples: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    identity = gradient_helper_run_native__array_identity(samples)
    return {
        "storage": "external_pt",
        "artifact": gradient_helper_run_native_EXTERNAL_DETERMINISTIC_ARTIFACT,
        "manifest": gradient_helper_run_native_manifest_path(Path(gradient_helper_run_native_EXTERNAL_DETERMINISTIC_ARTIFACT)).name,
        "payload_path": "/sampling_replicates/samples",
        **identity,
        "artifact_sha256": manifest["artifact_sha256"],
        "artifact_size_bytes": manifest["size_bytes"],
    }

# %%

if MAIN:
    from typing import Any, Iterator

def gradient_helper_run_native__persisted_evaluated(evaluated: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy only the route to the external sample leaf; do not copy arrays."""
    result = dict(evaluated)
    replicates = dict(result["sampling_replicates"])
    replicates["samples"] = reference
    result["sampling_replicates"] = replicates
    return result

# %%

gradient_helper_run_native_BUDGETS = (8, 32, 128, 512, 2048)

# %%

gradient_helper_run_native_REPLICATE_MODELS = ("clean_0", "shuffled_0", "ridge")

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    import numpy as np

def gradient_helper_run_native__require_array(value: Any, shape: tuple[int, ...], dtype: np.dtype, name: str) -> None:
    array, _ = gradient_helper_run_native__array(value)
    if array.shape != shape or array.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}.")

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    import numpy as np

def gradient_helper_run_native__validate_replicate_branch(replicates: dict[str, Any]) -> None:
    """Require every declared schedule, split, estimate, and diagnostic array before save."""
    required = {
        "source_ids",
        "row_split",
        "split_labels",
        "models",
        "model_labels",
        "budgets",
        "replicate_seeds",
        "baseline_index",
        "alpha",
        "samples",
        "phi_prefix",
        "se_prefix",
        "q512",
        "q512_error",
        "coverage_95",
        "coverage_rate",
    }
    if not isinstance(replicates, dict) or not required <= set(replicates):
        raise ValueError("Sampling-replicate evidence is missing required complete branches.")
    r, m, n, k, d = gradient_helper_run_native_REPLICATE_SHAPE
    p = len(gradient_helper_run_native_BUDGETS)
    if tuple(replicates["models"]) != gradient_helper_run_native_REPLICATE_MODELS:
        raise ValueError("Sampling-replicate model order changed.")
    if tuple(replicates["split_labels"]) != ("explanation", "audit"):
        raise ValueError("Sampling-replicate split labels changed.")
    gradient_helper_run_native__require_array(replicates["source_ids"], (n,), np.dtype("int64"), "replicate source_ids")
    gradient_helper_run_native__require_array(replicates["row_split"], (n,), np.dtype("int64"), "replicate row_split")
    split = gradient_helper_run_native__array(replicates["row_split"])[0]
    if not np.array_equal(split, np.r_[np.zeros(n // 2), np.ones(n - n // 2)]):
        raise ValueError("Replicate rows must retain explanation then audit split identity.")
    gradient_helper_run_native__require_array(replicates["budgets"], (p,), np.dtype("int64"), "replicate budgets")
    if not np.array_equal(gradient_helper_run_native__array(replicates["budgets"])[0], np.asarray(gradient_helper_run_native_BUDGETS)):
        raise ValueError("Replicate budgets changed from the frozen nested prefixes.")
    gradient_helper_run_native__require_array(replicates["replicate_seeds"], (r,), np.dtype("uint64"), "replicate seeds")
    gradient_helper_run_native__require_array(replicates["baseline_index"], (r, n, k), np.dtype("int64"), "replicate donors")
    gradient_helper_run_native__require_array(replicates["alpha"], (r, n, k), np.dtype("float64"), "replicate alphas")
    gradient_helper_run_native__require_array(replicates["samples"], gradient_helper_run_native_REPLICATE_SHAPE, np.dtype("float64"), "replicate samples")
    prefix_shape = (r, m, p, n, d)
    gradient_helper_run_native__require_array(replicates["phi_prefix"], prefix_shape, np.dtype("float64"), "replicate means")
    gradient_helper_run_native__require_array(replicates["se_prefix"], prefix_shape, np.dtype("float64"), "replicate SE")
    gradient_helper_run_native__require_array(replicates["q512"], (m, n, d), np.dtype("float64"), "replicate q512")
    gradient_helper_run_native__require_array(replicates["q512_error"], prefix_shape, np.dtype("float64"), "replicate errors")
    gradient_helper_run_native__require_array(replicates["coverage_95"], prefix_shape, np.dtype("bool"), "replicate coverage")
    gradient_helper_run_native__require_array(replicates["coverage_rate"], (m, p), np.dtype("float64"), "coverage rates")

# %%

if MAIN:
    from pathlib import Path
if MAIN:
    import hashlib

def gradient_helper_run_native_digest(path: Path) -> str:
    """Stream a file SHA256 without materializing a large payload as bytes."""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(gradient_helper_run_native_HASH_CHUNK_BYTES):
            result.update(chunk)
    return result.hexdigest()

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    from pathlib import Path
if MAIN:
    import json

def gradient_helper_run_native__verify_boundary(directory: Path) -> dict[str, Any]:
    boundary_path = directory / "before_audit.json"
    if not boundary_path.is_file():
        raise ValueError("Prepared evidence must be sealed before audit evaluation.")
    boundary = json.loads(boundary_path.read_text())
    if boundary.get("audit_loaded") is not False:
        raise ValueError("The recorded preparation boundary is invalid.")
    for name in ("fitted", "prepared"):
        artifact = directory / f"{name}.pt"
        manifest = gradient_helper_run_native_manifest_path(artifact)
        if gradient_helper_run_native_digest(artifact) != boundary.get(f"{name}_sha256"):
            raise ValueError(f"{name.title()} evidence changed after the audit boundary.")
        if gradient_helper_run_native_digest(manifest) != boundary.get(f"{name}_manifest_sha256"):
            raise ValueError(f"{name.title()} manifest changed after the audit boundary.")
    return boundary

# %%

if MAIN:
    from typing import Any, Iterator

def gradient_helper_run_native__path(parent: str, item: Any) -> str:
    token = str(item).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{token}" if parent else f"/{token}"

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    import torch

def gradient_helper_run_native__reject_live_modules(value: Any, path: str = "/") -> None:
    if isinstance(value, torch.nn.Module):
        raise TypeError(f"Payload {path} contains a live module; persist complete states instead.")
    if isinstance(value, dict):
        for key, item in value.items():
            gradient_helper_run_native__reject_live_modules(item, gradient_helper_run_native__path(path.rstrip("/"), key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            gradient_helper_run_native__reject_live_modules(item, gradient_helper_run_native__path(path.rstrip("/"), index))

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    from pathlib import Path
if MAIN:
    import json

def gradient_helper_run_native__write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    import hashlib
if MAIN:
    import numpy as np

def gradient_helper_run_native__scalar_record(value: Any) -> dict[str, Any]:
    if isinstance(value, np.generic):
        return {
            "kind": "numpy.scalar",
            "dtype": str(value.dtype),
            "value": gradient_helper_run_native__scalar_record(value.item()),
        }
    if value is None:
        return {"kind": "none", "value": None}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": str(value)}
    if isinstance(value, float):
        return {"kind": "float", "value_hex": value.hex()}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "size_bytes": len(value),
            "value_sha256": hashlib.sha256(value).hexdigest(),
        }
    raise TypeError(f"Unsupported deterministic payload leaf: {type(value).__name__}")

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    import numpy as np
if MAIN:
    import torch

def gradient_helper_run_native__manifest_records(value: Any, path: str = "") -> Iterator[dict[str, Any]]:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        array, kind = gradient_helper_run_native__array(value)
        yield {
            "path": path or "/",
            "kind": kind,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "elements": int(array.size),
            "value_sha256": gradient_helper_run_native__array_value_sha256(value),
        }
        return
    if isinstance(value, dict):
        keys = list(value)
        if not all(isinstance(key, (str, int)) and not isinstance(key, bool) for key in keys):
            raise TypeError("Deterministic payload dictionaries require string or integer keys.")
        yield {
            "path": path or "/",
            "kind": "dict",
            "keys": [
                {"kind": "str", "value": key}
                if isinstance(key, str)
                else {"kind": "int", "value": str(key)}
                for key in keys
            ],
        }
        for key in keys:
            yield from gradient_helper_run_native__manifest_records(value[key], gradient_helper_run_native__path(path, key))
        return
    if isinstance(value, (list, tuple)):
        yield {
            "path": path or "/",
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "length": len(value),
        }
        for index, item in enumerate(value):
            yield from gradient_helper_run_native__manifest_records(item, gradient_helper_run_native__path(path, index))
        return
    yield {"path": path or "/", **gradient_helper_run_native__scalar_record(value)}

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    from pathlib import Path

def gradient_helper_run_native_payload_manifest(value: Any, artifact: Path) -> dict[str, Any]:
    """Describe every payload leaf and bind it to the serialized artifact bytes."""
    records = list(gradient_helper_run_native__manifest_records(value))
    array_records = [record for record in records if "elements" in record]
    return {
        "schema_version": 1,
        "artifact": Path(artifact).name,
        "artifact_sha256": gradient_helper_run_native_digest(artifact),
        "size_bytes": Path(artifact).stat().st_size,
        "records": records,
        "counts": {
            "arrays": len(array_records),
            "array_elements": sum(record["elements"] for record in array_records),
            "records": len(records),
        },
    }

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    from pathlib import Path
if MAIN:
    import torch

def gradient_helper_run_native_save_payload(directory: Path, name: str, value: Any) -> tuple[Path, dict[str, Any]]:
    """Direct-save a course-owned payload, then write its recursive identity."""
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("Payload names must be simple .pt filenames.")
    gradient_helper_run_native__reject_live_modules(value)
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    manifest = gradient_helper_run_native_payload_manifest(value, path)
    gradient_helper_run_native__write_manifest(gradient_helper_run_native_manifest_path(path), manifest)
    return path, manifest

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    import numpy as np
if MAIN:
    import torch

def gradient_helper_run_native__sample_count(blocks: dict[str, Any]) -> int:
    total = 0
    for item in blocks.values():
        mc = item.get("mc", item).get("samples") if isinstance(item, dict) else None
        if isinstance(mc, (np.ndarray, torch.Tensor)):
            total += int(mc.numel() if isinstance(mc, torch.Tensor) else mc.size)
    return total

# %%

if MAIN:
    from typing import Any, Iterator

def gradient_helper_run_native_summarize(
    fitted: dict[str, Any],
    prepared: dict[str, Any],
    evaluated: dict[str, Any],
    reference: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build an index from actual 16.15 payload shapes, without outcome claims."""
    models = fitted["models"]
    replicate = evaluated["sampling_replicates"]["samples"]
    identity = gradient_helper_run_native__array_identity(replicate)
    return {
        "study": "16.15_gradientshap_expected_gradients",
        "phase": evaluated.get("phase"),
        "model_keys": list(models),
        "model_fits": len(models),
        "optimizer_updates": sum(
            len(item.get("loss_trace", ()))
            for item in models.values()
            if item.get("kind") != "ridge"
        ),
        "train_rows": len(fitted["train_ids"]),
        "explanation_rows": len(prepared["explain_ids"]),
        "audit_rows": len(evaluated["audit_ids"]),
        "all_model_rows": len(models)
        * (len(prepared["explain_ids"]) + len(evaluated["audit_ids"])),
        "nested_budgets": list(gradient_helper_run_native_BUDGETS),
        "quadrature_nodes": [256, 512],
        "background_variants": len(evaluated.get("background_variants", {})),
        "noise_variants": sum(
            isinstance(name, str) and name.startswith("sigma_")
            for name in evaluated.get("noise_variants", {})
        ),
        "sampling_replicates": int(identity["shape"][0]),
        "replicate_sample_shape": identity["shape"],
        "replicate_sample_dtype": identity["dtype"],
        "replicate_sample_coordinates": identity["elements"],
        "replicate_sample_sha256": identity["value_sha256"],
        "primary_sample_coordinates": gradient_helper_run_native__sample_count(prepared.get("primary", {}))
        + gradient_helper_run_native__sample_count(evaluated.get("audit_primary", {})),
        "audit_used_for_selection": prepared["audit_used_for_selection"],
        "reference_status": reference.get("status"),
        "deterministic_artifacts": list(gradient_helper_run_native_DETERMINISTIC_ARTIFACTS),
        "manifest_array_count": sum(item["counts"]["arrays"] for item in manifests.values()),
        "manifest_array_elements": sum(
            item["counts"]["array_elements"] for item in manifests.values()
        ),
    }

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    from pathlib import Path
if MAIN:
    import json
if MAIN:
    import numpy as np

def gradient_helper_run_native_persist_completed(
    directory: Path,
    fitted: dict[str, Any],
    prepared: dict[str, Any],
    evaluated: dict[str, Any],
    diagnostics: dict[str, Any],
    reference: dict[str, Any],
    play: dict[str, Any],
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist all results and externalize only the complete replicate sample tensor."""
    directory = Path(directory)
    gradient_helper_run_native__verify_boundary(directory)
    if evaluated.get("phase") != "evaluated_after_audit":
        raise ValueError("Evaluated payload must declare phase='evaluated_after_audit'.")
    if evaluated.get("audit_used_for_selection") is not False:
        raise ValueError("Evaluation must preserve audit_used_for_selection=False.")
    replicates = evaluated.get("sampling_replicates", {})
    gradient_helper_run_native__validate_replicate_branch(replicates)
    samples = replicates.get("samples")
    array, _ = gradient_helper_run_native__array(samples)
    if array.shape != gradient_helper_run_native_REPLICATE_SHAPE or array.dtype != np.dtype("float64"):
        raise ValueError(
            f"Replicate samples must be CPU float64 with exact shape {gradient_helper_run_native_REPLICATE_SHAPE}."
        )

    _, replicate_manifest = gradient_helper_run_native_save_payload(directory, gradient_helper_run_native_EXTERNAL_DETERMINISTIC_ARTIFACT, samples)
    external = gradient_helper_run_native__external_reference(samples, replicate_manifest)
    persisted_evaluation = gradient_helper_run_native__persisted_evaluated(evaluated, external)
    manifests = {
        "fitted.pt": json.loads(gradient_helper_run_native_manifest_path(directory / "fitted.pt").read_text()),
        "prepared.pt": json.loads(gradient_helper_run_native_manifest_path(directory / "prepared.pt").read_text()),
        gradient_helper_run_native_EXTERNAL_DETERMINISTIC_ARTIFACT: replicate_manifest,
    }
    for name, value in (
        ("evaluated.pt", persisted_evaluation),
        ("reference.pt", reference),
        ("diagnostics.pt", diagnostics),
        ("play.pt", play),
    ):
        _, manifests[name] = gradient_helper_run_native_save_payload(directory, name, value)
    gradient_helper_run_native_save_payload(directory, gradient_helper_run_native_TIMING_ARTIFACT, timings or {})
    summary = gradient_helper_run_native_summarize(fitted, prepared, evaluated, reference, manifests)
    if summary["replicate_sample_coordinates"] != gradient_helper_run_native_REPLICATE_ELEMENTS:
        raise ValueError("The full declared sampling-replicate array was not retained.")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    inventory = {
        "schema_version": 1,
        "deterministic": {
            name: {
                "artifact_sha256": manifests[name]["artifact_sha256"],
                "size_bytes": manifests[name]["size_bytes"],
                "manifest": gradient_helper_run_native_manifest_path(Path(name)).name,
                "manifest_sha256": gradient_helper_run_native_digest(gradient_helper_run_native_manifest_path(directory / name)),
                "external_large_payload": name == gradient_helper_run_native_EXTERNAL_DETERMINISTIC_ARTIFACT,
            }
            for name in gradient_helper_run_native_DETERMINISTIC_ARTIFACTS
        },
        "timing": {
            "artifact": gradient_helper_run_native_TIMING_ARTIFACT,
            "artifact_sha256": gradient_helper_run_native_digest(directory / gradient_helper_run_native_TIMING_ARTIFACT),
            "manifest": gradient_helper_run_native_manifest_path(Path(gradient_helper_run_native_TIMING_ARTIFACT)).name,
            "manifest_sha256": gradient_helper_run_native_digest(gradient_helper_run_native_manifest_path(directory / gradient_helper_run_native_TIMING_ARTIFACT)),
        },
    }
    (directory / "artifact_manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
    present = {path.name for path in directory.glob("*.pt")}
    expected = set(gradient_helper_run_native_DETERMINISTIC_ARTIFACTS) | {gradient_helper_run_native_TIMING_ARTIFACT}
    if present != expected:
        raise ValueError(f"Native artifact set mismatch: {sorted(present)}")
    return summary

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    import numpy as np

def gradient_helper_run_native__primary_background_ids(prepared: dict[str, Any]) -> np.ndarray:
    backgrounds = prepared.get("backgrounds", {})
    primary = backgrounds.get("primary64", {})
    for key in ("source_ids", "ids"):
        if key in primary:
            return np.asarray(primary[key], dtype=np.int64)
    if "background_ids" in prepared:
        return np.asarray(prepared["background_ids"], dtype=np.int64)
    raise ValueError("Prepared payload must retain primary64 source IDs.")

# %%

if MAIN:
    from typing import Any, Iterator

def gradient_helper_run_native__require_reproduction_gate(prepared: dict[str, Any]) -> None:
    reproduction = prepared.get("reproduction")
    if not isinstance(reproduction, dict) or reproduction.get("status") != "PASS":
        raise ValueError("Prepared attribution requires a passing 16.14 state reproduction gate.")
    expected = reproduction.get("expected_model_state_hashes", {})
    actual = reproduction.get("model_state_hashes", {})
    state_hash_count = sum(
        len(states) for states in expected.values() if isinstance(states, dict)
    ) if isinstance(expected, dict) else 0
    if state_hash_count != 21 or expected != actual:
        raise ValueError("All 21 initial, standardized, and raw model hashes must match exactly.")
    mismatches = reproduction.get("mismatches")
    if not isinstance(mismatches, (list, tuple, dict)) or len(mismatches) != 0:
        raise ValueError("A model-state mismatch must stop before GradientSHAP attribution.")

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    from pathlib import Path
if MAIN:
    import json
if MAIN:
    import numpy as np

def gradient_helper_run_native_write_before_audit(directory: Path, prepared: dict[str, Any]) -> dict[str, Any]:
    """Seal fitted and prepared bytes before the literal audit load."""
    directory = Path(directory)
    if prepared.get("phase") != "prepared_before_audit":
        raise ValueError("Prepared payload must declare phase='prepared_before_audit'.")
    if prepared.get("audit_used_for_selection") is not False:
        raise ValueError("Prepared rankings must explicitly exclude the audit split.")
    gradient_helper_run_native__require_reproduction_gate(prepared)
    files = [directory / "fitted.pt", directory / "prepared.pt"]
    manifests = [gradient_helper_run_native_manifest_path(path) for path in files]
    if not all(path.is_file() for path in files + manifests):
        raise ValueError("Save fitted/prepared payloads and manifests before sealing audit.")
    explain_ids = np.asarray(prepared["explain_ids"], dtype=np.int64)
    background_ids = gradient_helper_run_native__primary_background_ids(prepared)
    boundary = {
        "schema_version": 1,
        "phase": "prepared_before_audit",
        "audit_loaded": False,
        "fitted_sha256": gradient_helper_run_native_digest(files[0]),
        "fitted_manifest_sha256": gradient_helper_run_native_digest(manifests[0]),
        "prepared_sha256": gradient_helper_run_native_digest(files[1]),
        "prepared_manifest_sha256": gradient_helper_run_native_digest(manifests[1]),
        "explain_ids": gradient_helper_run_native__array_identity(explain_ids),
        "background_ids": gradient_helper_run_native__array_identity(background_ids),
        "state_hash_count": 21,
    }
    (directory / "before_audit.json").write_text(json.dumps(boundary, indent=2) + "\n")
    return boundary

# %%

if MAIN:
    from typing import Any, Iterator
if MAIN:
    from pathlib import Path

def gradient_helper_run_native_persist_prepared(directory: Path, fitted: dict, prepared: dict) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    gradient_helper_run_native_save_payload(directory, "fitted.pt", fitted)
    gradient_helper_run_native_save_payload(directory, "prepared.pt", prepared)
    return gradient_helper_run_native_write_before_audit(directory, prepared)

# %%

if MAIN:
    from pathlib import Path
if MAIN:
    import json

def gradient_helper_run_native_write_notebook_summary(output_root: Path, native_directory: Path, summary: dict) -> Path:
    output_root = Path(output_root).resolve()
    native_directory = Path(native_directory).resolve()
    native_directory.relative_to(output_root)
    payload = dict(summary, native_directory=str(native_directory))
    path = output_root / "native_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path

# %%

if MAIN:
    gradient_persist_completed = gradient_helper_run_native_persist_completed
if MAIN:
    persist_prepared = gradient_helper_run_native_persist_prepared
if MAIN:
    gradient_write_notebook_summary = gradient_helper_run_native_write_notebook_summary

if MAIN:
    native_directory = canonical_dir / "artifacts" / "gradient" / "native"
if MAIN:
    gradient_train = gradient_load_split("train")
if MAIN:
    gradient_explain = gradient_load_split("explain")
if MAIN:
    gradient_fitted = gradient_fit_models(gradient_train)
if MAIN:
    reproduction = verify_16_14_reproduction(gradient_fitted, load_manifest())
if MAIN:
    gradient_fitted = dict(gradient_fitted)
if MAIN:
    gradient_fitted["reproduction"] = reproduction
if MAIN:
    gradient_prepared = gradient_prepare_study(
        gradient_fitted,
        gradient_train,
        gradient_explain,
        methods=gradient_methods,
        deep_methods=deep_methods,
    )
if MAIN:
    before_audit = persist_prepared(native_directory, gradient_fitted, gradient_prepared)

if MAIN:
    print("16.14 state reproduction:", reproduction["status"])
if MAIN:
    print(
        "State hashes checked:",
        sum(len(item) for item in reproduction["model_state_hashes"].values()),
    )
if MAIN:
    print("Prepared phase:", gradient_prepared["phase"])
if MAIN:
    print("Audit loaded at this boundary:", before_audit["audit_loaded"])

# %%

if MAIN:
    gradient_audit = gradient_load_split("audit")

# %%

def gradient__performance_table(
    fitted: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
    """Return audit MSE for every model and the train-mean predictor."""
    audit_x, audit_y, _ = gradient__rows(audit, "audit")
    result: dict[str, Any] = {}
    with t.no_grad():
        for key in gradient_MODEL_KEYS:
            prediction = gradient__model_from_state(fitted["models"][key])(audit_x).squeeze(-1)
            result[key] = {
                "kind": fitted["models"][key]["kind"],
                "control": fitted["models"][key]["control"],
                "prediction": prediction,
                "mse": (prediction - audit_y).square().mean(),
            }
    result["training_mean_mse"] = (
        audit_y - fitted["train_y_raw"].mean()
    ).square().mean()
    return result

# %%

def _background_block(
    fitted: Mapping[str, Any],
    rows: Mapping[str, Any],
    background: Mapping[str, Any],
    methods: Any,
    deep_methods: Any,
) -> dict[str, Any]:
    model = gradient__model_from_state(fitted["models"]["clean_0"])
    schedule = _primary_schedule(rows, len(background["x_raw"]), methods)
    return {
        "source_ids": gradient__rows(rows, "rows")[2],
        "targets": gradient__rows(rows, "rows")[1],
        "background_seed": background["seed"],
        "background_indices": background["indices"],
        "background_source_ids": background["source_ids"],
        "background_x_raw": background["x_raw"],
        "mc": _monte_carlo_block(
            model, rows, background["x_raw"], schedule, methods
        ),
        "quadrature": _quadrature_block(
            model, rows, background["x_raw"], methods
        ),
        "baselines": _comparison_baselines(
            model, rows, background["x_raw"], deep_methods
        ),
    }

def _background_comparison(
    fitted: Mapping[str, Any],
    rows: Mapping[str, Any],
    backgrounds: Mapping[str, Any],
    methods: Any,
    deep_methods: Any,
) -> dict[str, Any]:
    """Compute complete clean_0 blocks for every supplied finite background."""
    return {
        name: _background_block(
            fitted, rows, background, methods, deep_methods
        )
        for name, background in backgrounds.items()
    }

# %%

def _standard_normal_schedule(source_ids: t.Tensor, samples: int) -> t.Tensor:
    arrays = []
    for source_id in source_ids.tolist():
        rng = np.random.Generator(
            np.random.PCG64(
                _seed_words(NOISE_SCHEDULE_SEED, int(source_id), "standard_normal")
            )
        )
        arrays.append(rng.standard_normal((samples, gradient_FEATURES)))
    return t.tensor(np.stack(arrays), dtype=t.float64)

def _noise_comparison(
    fitted: Mapping[str, Any],
    rows: Mapping[str, Any],
    primary: Mapping[str, Any],
    methods: Any,
) -> dict[str, Any]:
    """Retain clean and positive-noise estimators with both completeness targets."""
    x, _, source_ids = gradient__rows(rows, "rows")
    model = gradient__model_from_state(fitted["models"]["clean_0"])
    clean = primary["clean_0"]["mc"]
    samples = clean["samples"].shape[1]
    standard_normal = _standard_normal_schedule(source_ids, samples)
    _, x_std, _, _ = gradient__normalization_tensors(fitted["normalization"])
    base = clean["base_full"]
    clean_prediction = clean["prediction"]
    result: dict[str, Any] = {
        "standard_normal": standard_normal,
        "source_ids": source_ids,
    }
    for sigma in NOISE_SIGMAS:
        name = "sigma_0" if sigma == 0 else f"sigma_{sigma:.2f}".replace(".", "_")
        if sigma == 0:
            noise_raw = standard_normal * 0.0
            noisy_prediction = clean_prediction[:, None].expand(-1, samples)
            contribution = clean["samples"]
        else:
            noise_raw = standard_normal * (sigma * x_std)[None, None, :]
            contribution = methods.noisy_expected_gradient_samples(
                model,
                x,
                primary["_background_x_raw"],
                clean["baseline_index"],
                clean["alpha"],
                noise_raw,
                FORWARD_BATCH,
            )
            noisy_prediction = _batched_prediction(model, x[:, None, :] + noise_raw)
        phi, se = methods.summarize_expected_gradients(contribution, (samples,))
        smoothed_prediction = noisy_prediction.mean(1)
        smoothed_prediction_se = noisy_prediction.std(1, correction=1) / samples**0.5
        result[name] = {
            "sigma_standardized": sigma,
            "noise_raw": noise_raw,
            "samples": contribution,
            "noisy_prediction_samples": noisy_prediction,
            "phi": phi[0],
            "se": se[0],
            "clean_target_residual": phi[0].sum(-1)
            - (clean_prediction - base),
            "smoothed_target_residual": phi[0].sum(-1)
            - (smoothed_prediction - base),
            "smoothed_target_se": smoothed_prediction_se,
        }
    return result

# %%

def _replicate_seed(replicate: int) -> int:
    name = (
        "arena16.15/gradientshap/v1/"
        f"{REPLICATE_SCHEDULE_SEED}/replicate/{replicate}"
    )
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")

def _replicate_comparison(
    fitted: Mapping[str, Any],
    rows: Mapping[str, Any],
    primary: Mapping[str, Any],
    methods: Any,
) -> dict[str, Any]:
    """Retain 64 shared schedules and all three-model per-draw contributions."""
    x, _, source_ids = gradient__rows(rows, "replicate rows")
    background = primary["_background_x_raw"]
    design = load_manifest()["replicate_diagnostic_candidate"]
    explain_ids = t.as_tensor(design["explanation_source_ids"], dtype=t.int64)
    audit_ids = t.as_tensor(design["audit_source_ids"], dtype=t.int64)
    if not t.equal(source_ids, t.cat([explain_ids, audit_ids])):
        raise ValueError("Replicate rows changed their frozen split boundary or order.")
    row_split = t.cat(
        [
            t.zeros(len(explain_ids), dtype=t.int64),
            t.ones(len(audit_ids), dtype=t.int64),
        ]
    )
    samples_per_replicate = max(BUDGETS)
    baseline_index = t.empty(
        (REPLICATES, len(x), samples_per_replicate), dtype=t.int64
    )
    alpha = t.empty(
        (REPLICATES, len(x), samples_per_replicate), dtype=t.float64
    )
    samples = t.empty(
        (
            REPLICATES,
            len(REPLICATE_MODELS),
            len(x),
            samples_per_replicate,
            gradient_FEATURES,
        ),
        dtype=t.float64,
    )
    seed_values = [_replicate_seed(index) for index in range(REPLICATES)]
    declared = load_manifest()["sampling_design_candidate"].get("replicate_seeds")
    if declared is not None and seed_values != declared[:REPLICATES]:
        raise ValueError("Derived replicate seeds differ from the frozen manifest.")
    seeds = t.tensor(seed_values, dtype=t.uint64)
    for replicate in range(REPLICATES):
        indices, draws = methods.sample_path_schedule(
            source_ids,
            len(background),
            samples_per_replicate,
            int(seeds[replicate].item()),
        )
        baseline_index[replicate] = indices
        alpha[replicate] = draws
        for model_index, key in enumerate(REPLICATE_MODELS):
            model = gradient__model_from_state(fitted["models"][key])
            samples[replicate, model_index] = methods.expected_gradient_samples(
                model, x, background, indices, draws, FORWARD_BATCH
            )
    phi_prefix = t.empty(
        (REPLICATES, len(REPLICATE_MODELS), len(BUDGETS), len(x), gradient_FEATURES),
        dtype=t.float64,
    )
    se_prefix = t.empty_like(phi_prefix)
    for replicate in range(REPLICATES):
        for model_index in range(len(REPLICATE_MODELS)):
            phi, se = methods.summarize_expected_gradients(
                samples[replicate, model_index], BUDGETS
            )
            phi_prefix[replicate, model_index] = phi
            se_prefix[replicate, model_index] = se
    q512 = t.stack(
        [
            primary[key]["quadrature"][f"mean_{QUADRATURE_NODES[1]}"]
            for key in REPLICATE_MODELS
        ]
    )
    error = phi_prefix - q512[None, :, None, :, :]
    coverage = error.abs() <= 1.96 * se_prefix
    return {
        "source_ids": source_ids,
        "row_split": row_split,
        "split_labels": ("explanation", "audit"),
        "models": REPLICATE_MODELS,
        "model_labels": tuple(gradient_MODEL_LABELS[key] for key in REPLICATE_MODELS),
        "budgets": t.tensor(BUDGETS, dtype=t.int64),
        "replicate_seeds": seeds,
        "baseline_index": baseline_index,
        "alpha": alpha,
        "samples": samples,
        "phi_prefix": phi_prefix,
        "se_prefix": se_prefix,
        "q512": q512,
        "q512_error": error,
        "coverage_95": coverage,
        "coverage_rate": coverage.to(t.float64).mean(dim=(0, 3, 4)),
    }

# %%

def _nearest_train_distance(
    points: t.Tensor, train_standardized: t.Tensor
) -> t.Tensor:
    result = t.empty(len(points), dtype=t.float64)
    for start in range(0, len(points), DISTANCE_BATCH):
        block = points[start : start + DISTANCE_BATCH]
        result[start : start + len(block)] = t.cdist(block, train_standardized).min(1).values
    return result

def _distance_comparison(
    train: Mapping[str, Any],
    rows: Mapping[str, Any],
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain every declared nearest-training distance and its exact identity."""
    train_x, _, train_ids = gradient__rows(train, "train")
    x, _, source_ids = gradient__rows(rows, "rows")
    x_mean = train_x.mean(0)
    x_std = train_x.std(0, correction=0)
    train_z = (train_x - x_mean) / x_std
    query_z = (x - x_mean) / x_std
    background = primary["_background_x_raw"]
    background_ids = primary["_background_source_ids"]
    schedule = primary["clean_0"]["mc"]

    path_indices = schedule["baseline_index"][:, :PATH_DISTANCE_SAMPLES]
    path_alpha = schedule["alpha"][:, :PATH_DISTANCE_SAMPLES]
    donors = background[path_indices]
    path_points = donors + path_alpha[:, :, None] * (x[:, None, :] - donors)
    path_distance = _nearest_train_distance(
        ((path_points - x_mean) / x_std).reshape(-1, gradient_FEATURES), train_z
    )

    masks = (
        t.arange(2**gradient_FEATURES, dtype=t.int64)[:, None]
        .bitwise_and(1 << t.arange(gradient_FEATURES, dtype=t.int64)[None, :])
        .ne(0)
    )
    donor_count = min(COALITION_DISTANCE_DONORS, len(background))
    coalition_distances: list[t.Tensor] = []
    for query in x:
        donor = background[:donor_count]
        hybrid = t.where(
            masks[None, :, :], query[None, None, :], donor[:, None, :]
        )
        coalition_distances.append(
            _nearest_train_distance(
                ((hybrid - x_mean) / x_std).reshape(-1, gradient_FEATURES), train_z
            )
        )
    coalition_distance = t.cat(coalition_distances)

    path_source = source_ids[:, None].expand(-1, PATH_DISTANCE_SAMPLES).reshape(-1)
    path_donor = background_ids[path_indices].reshape(-1)
    coalition_source = source_ids[:, None, None].expand(
        -1, donor_count, 2**gradient_FEATURES
    ).reshape(-1)
    coalition_donor = background_ids[:donor_count][None, :, None].expand(
        len(x), -1, 2**gradient_FEATURES
    ).reshape(-1)
    coalition_mask = t.arange(2**gradient_FEATURES, dtype=t.int64)[None, None, :].expand(
        len(x), donor_count, -1
    ).reshape(-1)
    return {
        "normalization": {"x_mean": x_mean, "x_std": x_std},
        "path": {
            "source_id": path_source,
            "donor_id": path_donor,
            "alpha": path_alpha.reshape(-1),
            "distance": path_distance,
        },
        "coalition": {
            "source_id": coalition_source,
            "donor_id": coalition_donor,
            "mask": coalition_mask,
            "distance": coalition_distance,
        },
        "query_distance": _nearest_train_distance(query_z, train_z),
        "donor_distance": _nearest_train_distance(
            (background[:donor_count] - x_mean) / x_std, train_z
        ),
        "donor_source_ids": background_ids[:donor_count],
        "scope": {
            "training_rows": len(train_ids),
            "query_rows": len(source_ids),
            "path_samples_per_row": PATH_DISTANCE_SAMPLES,
            "coalition_donors_per_row": donor_count,
            "coalition_masks": 2**gradient_FEATURES,
            "meaning": "geometry_not_density_or_ood_probability",
        },
    }

# %%

def _replacement_evaluation(
    prepared: Mapping[str, Any],
    audit_primary: Mapping[str, Any],
    methods: Any,
) -> dict[str, Any]:
    """Apply frozen global and all random orders to every audit coalition table."""
    method_fields = {
        "exact": ("baselines", "phi_exact"),
        "deep": ("baselines", "phi_deep"),
        "gradient": ("baselines", "phi_gradient"),
        "eg_k2048": ("mc", "phi_prefix"),
        "quadrature_512": ("quadrature", f"mean_{QUADRATURE_NODES[1]}"),
    }
    result: dict[str, Any] = {}
    random_insertion = []
    random_deletion = []
    random_insertion_masks = None
    random_deletion_masks = None
    for key in gradient_MODEL_KEYS:
        values = audit_primary[key]["baselines"]["coalition_values"]
        result[key] = {}
        for method, (branch, field) in method_fields.items():
            phi = audit_primary[key][branch][field]
            if method == "eg_k2048":
                phi = phi[-1]
            result[key][method] = _checked_curves(
                values, phi, prepared["rankings"][method][key], methods
            )
        model_insertion = []
        model_deletion = []
        for order in prepared["random_orders"]:
            curve = _checked_curves(
                values, t.zeros((len(values), gradient_FEATURES), dtype=t.float64), order, methods
            )
            model_insertion.append(curve["insertion"])
            model_deletion.append(curve["deletion"])
            if random_insertion_masks is None:
                random_insertion_masks = []
                random_deletion_masks = []
            if key == gradient_MODEL_KEYS[0]:
                random_insertion_masks.append(curve["insertion_masks"])
                random_deletion_masks.append(curve["deletion_masks"])
        random_insertion.append(t.stack(model_insertion, dim=1))
        random_deletion.append(t.stack(model_deletion, dim=1))
    result["random"] = {
        "models": gradient_MODEL_KEYS,
        "orders": prepared["random_orders"],
        "insertion_masks": t.stack(random_insertion_masks, dim=1),
        "deletion_masks": t.stack(random_deletion_masks, dim=1),
        "insertion": t.stack(random_insertion),
        "deletion": t.stack(random_deletion),
    }
    return result

# %%

def gradient_evaluate_study(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    methods: Any,
    deep_methods: Any,
) -> dict[str, Any]:
    """Evaluate every frozen control after receiving the literal audit split."""
    _require_namespace(methods, gradient_METHOD_NAMES, "Student methods")
    _require_namespace(deep_methods, DEEP_METHOD_NAMES, "DeepSHAP control")
    row_blocks = gradient__validate_evaluation_inputs(fitted, prepared, train, explain, audit)
    primary_background = prepared["backgrounds"]["primary64"]["x_raw"]
    audit_schedule = _primary_schedule(audit, len(primary_background), methods)
    audit_primary: dict[str, Any] = {}
    for key in gradient_MODEL_KEYS:
        model = gradient__model_from_state(fitted["models"][key])
        audit_primary[key] = {
            "mc": _monte_carlo_block(
                model, audit, primary_background, audit_schedule, methods
            ),
            "quadrature": _quadrature_block(model, audit, primary_background, methods),
            "baselines": _comparison_baselines(
                model, audit, primary_background, deep_methods
            ),
        }
    all_rows = _combine_primary(prepared, audit_primary)
    all_row_data = {
        "x": t.cat([row_blocks["explain_x"], row_blocks["audit_x"]]),
        "y": t.cat([row_blocks["explain_y"], row_blocks["audit_y"]]),
        "ids": t.cat([row_blocks["explain_ids"], row_blocks["audit_ids"]]),
    }
    row_split = t.cat(
        [
            t.zeros(len(row_blocks["explain_ids"]), dtype=t.int64),
            t.ones(len(row_blocks["audit_ids"]), dtype=t.int64),
        ]
    )

    primary_variant = {
        "source_ids": all_row_data["ids"],
        "targets": all_row_data["y"],
        "row_split": row_split,
        "background_seed": prepared["backgrounds"]["primary64"]["seed"],
        "background_indices": prepared["backgrounds"]["primary64"]["indices"],
        "background_source_ids": prepared["backgrounds"]["primary64"]["source_ids"],
        "background_x_raw": primary_background,
        **all_rows["clean_0"],
    }
    alternatives = {
        name: value
        for name, value in prepared["backgrounds"].items()
        if name != "primary64"
    }
    background_variants = {"primary64": primary_variant}
    background_variants.update(
        _background_comparison(
            fitted, all_row_data, alternatives, methods, deep_methods
        )
    )
    for value in background_variants.values():
        value.setdefault("row_split", row_split)

    primary_context = {
        **all_rows,
        "_background_x_raw": primary_background,
        "_background_source_ids": prepared["backgrounds"]["primary64"]["source_ids"],
    }
    manifest = load_manifest()
    replicate_ids = t.as_tensor(
        manifest["replicate_diagnostic_candidate"]["explanation_source_ids"]
        + manifest["replicate_diagnostic_candidate"]["audit_source_ids"],
        dtype=t.int64,
    )
    locations = {int(source): index for index, source in enumerate(all_row_data["ids"].tolist())}
    if any(int(source) not in locations for source in replicate_ids):
        raise ValueError("Replicate source IDs are absent from the held-out row order.")
    replicate_positions = t.tensor(
        [locations[int(source)] for source in replicate_ids], dtype=t.int64
    )
    replicate_rows = {
        "x": all_row_data["x"][replicate_positions],
        "y": all_row_data["y"][replicate_positions],
        "ids": replicate_ids,
    }
    replicate_primary = {
        key: {
            "quadrature": {
                name: value[replicate_positions]
                for name, value in all_rows[key]["quadrature"].items()
            }
        }
        for key in REPLICATE_MODELS
    }
    replicate_primary["_background_x_raw"] = primary_background

    return {
        "phase": "evaluated_after_audit",
        "audit_used_for_selection": False,
        "reproduction": prepared["reproduction"],
        "performance": gradient__performance_table(fitted, audit),
        "audit_ids": row_blocks["audit_ids"],
        "audit_targets": row_blocks["audit_y"],
        "audit_primary": audit_primary,
        "all_rows": all_rows,
        "all_row_x_raw": all_row_data["x"],
        "all_row_targets": all_row_data["y"],
        "all_row_split": row_split,
        "background_variants": background_variants,
        "noise_variants": _noise_comparison(
            fitted, all_row_data, primary_context, methods
        ),
        "sampling_replicates": _replicate_comparison(
            fitted, replicate_rows, replicate_primary, methods
        ),
        "distance_diagnostic": _distance_comparison(
            train, all_row_data, primary_context
        ),
        "replacement_curves": _replacement_evaluation(
            prepared, audit_primary, methods
        ),
        "random_orders": prepared["random_orders"],
        "units": dict(gradient_UNITS),
        "feature_names": gradient_FEATURE_NAMES,
        "model_labels": dict(gradient_MODEL_LABELS),
        "query_source_ids": all_row_data["ids"],
    }

# %%

def gradient_native_play(
    fitted: Mapping[str, Any],
    prepared: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    train: Mapping[str, Any],
    explain: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    methods: Any,
    deep_methods: Any,
    model_key: str = "clean_0",
    split: str = "audit",
    row_index: int = 0,
    background_key: str = "primary64",
    samples: int = 2048,
    sigma: float = 0.0,
) -> dict[str, Any]:
    """Recompute one editable real recipe without changing aggregate evidence."""
    _require_namespace(methods, gradient_METHOD_NAMES, "Student methods")
    _require_namespace(deep_methods, DEEP_METHOD_NAMES, "DeepSHAP control")
    gradient__validate_evaluation_inputs(fitted, prepared, train, explain, audit)
    if evaluated.get("phase") != "evaluated_after_audit":
        raise ValueError("Run the complete frozen evaluation before the real play.")
    if model_key not in gradient_MODEL_KEYS:
        raise ValueError("Choose one of the seven retained model keys.")
    if split not in {"explain", "audit"}:
        raise ValueError("split must be 'explain' or 'audit'.")
    selected_rows = explain if split == "explain" else audit
    x_all, y_all, ids_all = gradient__rows(selected_rows, split)
    if not 0 <= row_index < len(x_all):
        raise ValueError("row_index must select an existing held-out row.")
    if background_key not in prepared["backgrounds"]:
        raise ValueError("Choose one of the five frozen background keys.")
    if type(samples) is not int or samples < 2 or samples > max(BUDGETS):
        raise ValueError("samples must be an integer from 2 through 2048.")
    if sigma not in NOISE_SIGMAS:
        raise ValueError("sigma must be one of the three predeclared scales.")
    background_record = prepared["backgrounds"][background_key]
    background = background_record["x_raw"]
    x = x_all[row_index : row_index + 1]
    selected = {"x": x, "y": y_all[row_index : row_index + 1], "ids": ids_all[row_index : row_index + 1]}
    indices, alpha = methods.sample_path_schedule(
        selected["ids"], len(background), samples, PRIMARY_SCHEDULE_SEED
    )
    model = gradient__model_from_state(fitted["models"][model_key])
    if sigma == 0:
        mc_samples = methods.expected_gradient_samples(
            model, x, background, indices, alpha, FORWARD_BATCH
        )
    else:
        standard_normal = _standard_normal_schedule(selected["ids"], samples)
        _, x_std, _, _ = gradient__normalization_tensors(fitted["normalization"])
        noise = standard_normal * (sigma * x_std)[None, None, :]
        mc_samples = methods.noisy_expected_gradient_samples(
            model, x, background, indices, alpha, noise, FORWARD_BATCH
        )
    phi, se = methods.summarize_expected_gradients(mc_samples, (samples,))
    quadrature = _quadrature_block(model, selected, background, methods)
    baselines = _comparison_baselines(model, selected, background, deep_methods)
    local_values = baselines["coalition_values"]
    local_phi = {
        "eg": phi[0],
        "quadrature_512": quadrature[f"mean_{QUADRATURE_NODES[1]}"],
        "exact": baselines["phi_exact"],
        "deep": baselines["phi_deep"],
        "gradient": baselines["phi_gradient"],
    }
    curves = {
        name: methods.ranked_coalition_curves(local_values, value)
        for name, value in local_phi.items()
    }
    with t.no_grad():
        prediction = model(x).squeeze(-1)[0]
        base = model(background).squeeze(-1).mean()
    return {
        "selection": {
            "model_key": model_key,
            "model_label": gradient_MODEL_LABELS[model_key],
            "split": split,
            "row_index": row_index,
            "source_id": int(ids_all[row_index]),
            "query": x[0],
            "target": y_all[row_index],
            "background_key": background_key,
            "background_source_ids": background_record["source_ids"],
            "background_x_raw": background,
            "samples": samples,
            "sigma_standardized": sigma,
            "audit_reuse": "exploratory" if split == "audit" else "not_audit",
        },
        "prediction": prediction,
        "base": base,
        "mc_samples": mc_samples[0],
        "phi_eg": phi[0, 0],
        "se_eg": se[0, 0],
        "phi_q512": quadrature[f"mean_{QUADRATURE_NODES[1]}"][0],
        "phi_exact": baselines["phi_exact"][0],
        "phi_deep": baselines["phi_deep"][0],
        "phi_gradient": baselines["phi_gradient"][0],
        "coalition_values": local_values[0],
        "local_replacement_curves": {
            name: {field: item[0] for field, item in curve.items()}
            for name, curve in curves.items()
        },
        "units": dict(gradient_UNITS),
        "feature_names": gradient_FEATURE_NAMES,
    }

# %%

gradient_helper_gradientshap_reference_REFERENCE_SAMPLES = 128

# %%

gradient_helper_gradientshap_reference_ALL_ROWS = 272

# %%

gradient_helper_gradientshap_reference_AUTHOR_ATOL = 1e-10

# %%

gradient_helper_gradientshap_reference_AUTHOR_RTOL = 1e-10

# %%

gradient_helper_gradientshap_reference_BACKGROUND_ROWS = 64

# %%

gradient_helper_gradientshap_reference_CAPTUM_NUMPY_SEED = 161514

# %%

gradient_helper_gradientshap_reference_CAPTUM_TORCH_SEED = 161515

# %%

gradient_helper_gradientshap_reference_CAPTUM_WARNINGS = (
    "Captum 0.9.0 draws all donors, then Torch noise, then all alphas in one full-query call.",
    "Splitting the 272 rows into public Captum calls would change the draw table.",
    "author_convergence_delta contains one delta per query/sample pair; it is neither estimator uncertainty nor the averaged-vector completeness residual.",
    "Captum's positive-noise stream is generated from a float32 stdev buffer under the pinned ordinary default-dtype convention.",
)

# %%

gradient_helper_gradientshap_reference_EXPECTED_VERSIONS = {
    "shap": "0.49.1",
    "captum": "0.9.0",
}

# %%

gradient_helper_gradientshap_reference_FEATURES = 8

# %%

gradient_helper_gradientshap_reference_MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)

# %%

gradient_helper_gradientshap_reference_SHAP_BATCH_SIZE = 128

# %%

gradient_helper_gradientshap_reference_SHAP_NUMPY_SEED = 161513

# %%

gradient_helper_gradientshap_reference_SHAP_TORCH_SEED = 161515

# %%

gradient_helper_gradientshap_reference_SHAP_WARNINGS = (
    "SHAP 0.49.1 draws one donor then one alpha for each sample and query row; row order is part of the draw identity.",
    "SHAP rseed controls its legacy NumPy stream but does not seed Torch local-smoothing noise.",
    "The scalar-output raw attribution shape is (N,D,1); author_phi is the normalized (N,D) view.",
    "author_shap_variance_proxy is samples.var(ddof=0)/sqrt(K), neither ordinary variance-of-mean nor standard error, and is excluded from parity acceptance.",
    "The adapter temporarily uses the input dtype as Torch's default to repair SHAP's implicit sample-buffer dtype and restores it afterward.",
)

# %%

if MAIN:
    from typing import Any
if MAIN:
    from torch import Tensor, nn
if MAIN:
    import numpy as np

def gradient_helper_gradientshap_reference__numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

# %%

if MAIN:
    from typing import Any
if MAIN:
    import hashlib
if MAIN:
    import json
if MAIN:
    import numpy as np

def gradient_helper_gradientshap_reference__array_sha256(value: Any) -> str:
    """Hash array dtype, shape, and complete C-order bytes."""
    array = np.ascontiguousarray(gradient_helper_gradientshap_reference__numpy(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Callable, Iterator, Mapping

def gradient_helper_gradientshap_reference__compact_model_result(result: Mapping[str, Any], backend: str) -> dict[str, Any]:
    common = {
        "author_phi": result["author_phi"],
        "student_phi": result["student_phi"],
        "student_contributions": result["student_contributions"],
        "ordinary_standard_error": result["ordinary_standard_error"],
        "prediction": result["prediction"],
        "full_background_base": result["full_background_base"],
        "sampled_background_base": result["sampled_background_base"],
        "parity_max_abs_error": result["parity_max_abs_error"],
        "parity_passed": result["parity_passed"],
        "identity_sha256": result["identity_sha256"],
    }
    if backend == "shap":
        common.update(
            author_phi_raw=result["author_phi_raw"],
            author_shap_variance_proxy=result["author_shap_variance_proxy"],
            student_shap_variance_proxy=result["student_shap_variance_proxy"],
            variance_proxy_max_abs_error=result["variance_proxy_max_abs_error"],
        )
    else:
        common.update(
            author_convergence_delta=result["author_convergence_delta"],
            student_repeated_convergence_delta=result[
                "student_repeated_convergence_delta"
            ],
            convergence_delta_max_abs_error=result["convergence_delta_max_abs_error"],
        )
    return common

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def gradient_helper_gradientshap_reference__float64_tensor(name: str, value: Tensor, ndim: int) -> Tensor:
    if type(value) is not Tensor:
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cpu" or value.dtype != t.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor.")
    if value.ndim != ndim or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional tensor.")
    if not bool(t.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values.")
    return value

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import numpy as np

def gradient_helper_gradientshap_reference__source_ids(value: Tensor | np.ndarray | None, rows: int) -> np.ndarray:
    if value is None:
        return np.arange(rows, dtype=np.int64)
    array = gradient_helper_gradientshap_reference__numpy(value)
    if array.dtype != np.int64 or array.shape != (rows,):
        raise ValueError(f"source_ids must have int64 shape ({rows},).")
    if np.any(array < 0):
        raise ValueError("source_ids must be nonnegative.")
    if len(np.unique(array)) != rows:
        raise ValueError("source_ids must be unique so row order has an unambiguous identity.")
    return np.array(array, copy=True)

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def gradient_helper_gradientshap_reference__validate_model(model: nn.Module, features: int) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != t.float64:
            raise TypeError("Every model parameter must be a CPU float64 tensor.")
        if not bool(t.isfinite(parameter).all()):
            raise ValueError("Every model parameter must be finite.")
    for buffer in model.buffers():
        if buffer.device.type != "cpu":
            raise TypeError("Every model buffer must be on CPU.")
        if buffer.is_floating_point() and buffer.dtype != t.float64:
            raise TypeError("Every floating model buffer must be float64.")
        if buffer.is_floating_point() and not bool(t.isfinite(buffer).all()):
            raise ValueError("Every floating model buffer must be finite.")
    if features < 1:
        raise ValueError("features must be positive.")
    return model

# %%

gradient_helper_gradientshap_reference_FORWARD_BATCH = 32_768

# %%

if MAIN:
    from typing import Any
if MAIN:
    import numpy as np

def gradient_helper_gradientshap_reference__close(actual: Any, expected: Any) -> bool:
    actual_array, expected_array = gradient_helper_gradientshap_reference__numpy(actual), gradient_helper_gradientshap_reference__numpy(expected)
    return bool(
        actual_array.shape == expected_array.shape
        and np.isfinite(actual_array).all()
        and np.isfinite(expected_array).all()
        and np.allclose(
            actual_array,
            expected_array,
            atol=gradient_helper_gradientshap_reference_AUTHOR_ATOL,
            rtol=gradient_helper_gradientshap_reference_AUTHOR_RTOL,
        )
    )

# %%

if MAIN:
    import hashlib
if MAIN:
    from torch import Tensor, nn

def gradient_helper_gradientshap_reference__model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(gradient_helper_gradientshap_reference__array_sha256(value).encode())
    return digest.hexdigest()

# %%

if MAIN:
    import hashlib
if MAIN:
    import json
if MAIN:
    from torch import Tensor, nn

def gradient_helper_gradientshap_reference__model_structure_sha256(model: nn.Module) -> str:
    description: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        item: dict[str, Any] = {"name": name, "type": type(module).__qualname__}
        if isinstance(module, nn.Linear):
            item.update(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
        if isinstance(module, nn.ReLU):
            item["inplace"] = module.inplace
        description.append(item)
    return hashlib.sha256(
        json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Callable, Iterator, Mapping
if MAIN:
    from torch import Tensor, nn
if MAIN:
    import numpy as np

def gradient_helper_gradientshap_reference__identity(
    model: nn.Module,
    queries: Tensor,
    background: Tensor,
    ids: np.ndarray,
    draws: Mapping[str, np.ndarray],
    author_phi: np.ndarray,
    replay: Mapping[str, Any],
    **extra: Any,
) -> dict[str, str]:
    result = {
        "model_state": gradient_helper_gradientshap_reference__model_state_sha256(model),
        "model_structure": gradient_helper_gradientshap_reference__model_structure_sha256(model),
        "source_ids": gradient_helper_gradientshap_reference__array_sha256(ids),
        "queries": gradient_helper_gradientshap_reference__array_sha256(queries),
        "background": gradient_helper_gradientshap_reference__array_sha256(background),
        "donor_indices": gradient_helper_gradientshap_reference__array_sha256(draws["donor_indices"]),
        "alphas": gradient_helper_gradientshap_reference__array_sha256(draws["alphas"]),
        "noise": gradient_helper_gradientshap_reference__array_sha256(draws["noise"]),
        "author_phi": gradient_helper_gradientshap_reference__array_sha256(author_phi),
        "student_phi": gradient_helper_gradientshap_reference__array_sha256(replay["student_phi"]),
        "student_contributions": gradient_helper_gradientshap_reference__array_sha256(replay["student_contributions"]),
    }
    result.update({name: gradient_helper_gradientshap_reference__array_sha256(value) for name, value in extra.items()})
    return result

# %%

if MAIN:
    from typing import Any
if MAIN:
    import math
if MAIN:
    import numpy as np

def gradient_helper_gradientshap_reference__max_abs(actual: Any, expected: Any) -> float:
    actual_array = gradient_helper_gradientshap_reference__numpy(actual).astype(np.float64, copy=False)
    expected_array = gradient_helper_gradientshap_reference__numpy(expected).astype(np.float64, copy=False)
    if actual_array.shape != expected_array.shape:
        return math.inf
    return float(np.abs(actual_array - expected_array).max(initial=0.0))

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def gradient_helper_gradientshap_reference__model_output(model: nn.Module, inputs: Tensor) -> Tensor:
    output = model(inputs)
    if type(output) is not Tensor or output.shape != (len(inputs), 1):
        raise ValueError("Package reference models must return float64 shape (rows, 1).")
    if output.device.type != "cpu" or output.dtype != t.float64:
        raise TypeError("Package reference models must return CPU float64 values.")
    if not bool(t.isfinite(output).all()):
        raise ValueError("The model returned a nonfinite value.")
    return output[:, 0]

# %%

def gradient_helper_gradientshap_reference__positive_int(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value

# %%

if MAIN:
    from collections.abc import Callable, Iterator, Mapping
if MAIN:
    import contextlib
if MAIN:
    import copy
if MAIN:
    from torch import Tensor, nn
if MAIN:
    import numpy as np
if MAIN:
    import random
if MAIN:
    import torch as t

def initialize_helper_gradient_115_29_gradient_helper_gradientshap_reference__preserve_execution_state():
    global gradient_helper_gradientshap_reference__preserve_execution_state
    @contextlib.contextmanager
    def gradient_helper_gradientshap_reference__preserve_execution_state(model: nn.Module) -> Iterator[None]:
        """Restore all caller-visible execution state on success and exceptions."""
        python_state = random.getstate()
        numpy_state = copy.deepcopy(np.random.get_state())
        torch_state = t.random.get_rng_state().clone()
        default_dtype = t.get_default_dtype()
        modes = [(module, module.training) for module in model.modules()]
        parameters = [
            (
                parameter,
                parameter.detach().clone(),
                parameter.requires_grad,
                parameter.grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for parameter in model.parameters()
        ]
        buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            t.random.set_rng_state(torch_state)
            t.set_default_dtype(default_dtype)
            for module, training in modes:
                module.training = training
            with t.no_grad():
                for parameter, value, requires_grad, original_grad, gradient in parameters:
                    parameter.copy_(value)
                    parameter.requires_grad_(requires_grad)
                    if original_grad is None:
                        parameter.grad = None
                    else:
                        parameter.grad = original_grad
                        original_grad.copy_(gradient)
                for buffer, value in buffers:
                    buffer.copy_(value)


if MAIN:
    initialize_helper_gradient_115_29_gradient_helper_gradientshap_reference__preserve_execution_state()

# %%

def gradient_helper_gradientshap_reference__seed(name: str, value: int) -> int:
    if type(value) is not int or value < 0 or value >= 2**32:
        raise ValueError(f"{name} must be an integer in [0, 2**32).")
    return value

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Callable, Iterator, Mapping
if MAIN:
    import functools
if MAIN:
    from torch import Tensor, nn

def gradient_helper_gradientshap_reference__state_restoring_adapter(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Wrap a model-first adapter so even validation and import failures restore state."""

    @functools.wraps(function)
    def wrapped(model: nn.Module, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not isinstance(model, nn.Module):
            return function(model, *args, **kwargs)
        with gradient_helper_gradientshap_reference__preserve_execution_state(model):
            return function(model, *args, **kwargs)

    return wrapped

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import numpy as np

def gradient_helper_gradientshap_reference__validate_inputs(
    model: nn.Module,
    queries: Tensor,
    background: Tensor,
    source_ids: Tensor | np.ndarray | None,
) -> tuple[nn.Module, np.ndarray]:
    gradient_helper_gradientshap_reference__float64_tensor("queries", queries, 2)
    gradient_helper_gradientshap_reference__float64_tensor("background", background, 2)
    if queries.shape[1] != background.shape[1]:
        raise ValueError("queries and background must have the same feature dimension.")
    return gradient_helper_gradientshap_reference__validate_model(model, queries.shape[1]), gradient_helper_gradientshap_reference__source_ids(source_ids, len(queries))

# %%

if MAIN:
    import math
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def gradient_helper_gradientshap_reference_captum_draw_table(
    rows: int,
    features: int,
    background_rows: int,
    samples: int,
    *,
    numpy_seed: int,
    torch_seed: int,
    stdev: float,
) -> dict[str, np.ndarray]:
    """Recreate Captum 0.9.0's all-donor, noise, then all-alpha sequence."""
    rows = gradient_helper_gradientshap_reference__positive_int("rows", rows)
    features = gradient_helper_gradientshap_reference__positive_int("features", features)
    background_rows = gradient_helper_gradientshap_reference__positive_int("background_rows", background_rows)
    samples = gradient_helper_gradientshap_reference__positive_int("samples", samples)
    numpy_seed = gradient_helper_gradientshap_reference__seed("numpy_seed", numpy_seed)
    torch_seed = gradient_helper_gradientshap_reference__seed("torch_seed", torch_seed)
    if not isinstance(stdev, (int, float)) or not math.isfinite(stdev):
        raise ValueError("stdev must be a finite nonnegative number.")
    stdev = float(stdev)
    if stdev < 0:
        raise ValueError("stdev must be nonnegative.")

    legacy = np.random.RandomState(numpy_seed)
    donor_indices = legacy.choice(background_rows, rows * samples).reshape(rows, samples)
    generator = t.Generator(device="cpu").manual_seed(torch_seed)
    # Captum 0.9.0 NoiseTunnel constructs this buffer from Torch's default
    # dtype.  The canonical adapter below pins that default to float32.
    stdevs = t.tensor(stdev, dtype=t.float32).repeat(rows * samples, features)
    noise = t.normal(0.0, stdevs, generator=generator).reshape(rows, samples, features)
    alphas = legacy.uniform(0.0, 1.0, rows * samples).reshape(rows, samples)
    return {
        "donor_indices": donor_indices.astype(np.int64, copy=False),
        "alphas": alphas.astype(np.float64, copy=False),
        "noise": noise.numpy().astype(np.float64),
    }

# %%

gradient_helper_gradientshap_reference_EXPECTED_SOURCE_SHA256 = {
    "shap/explainers/_gradient.py": "ae1d0fe9405e62c9400c655e6d5af792ccc989a7632fd05171a731776551840a",
    "captum/attr/_core/gradient_shap.py": "b2333e8819b82eb18693ae5a8a78c7a9ff1e30986179de2ee97b2820dcbff00c",
    "captum/attr/_core/noise_tunnel.py": "d013c38cdbb4a10d4717a4e38ac063903cacc0b54389a6591a9bcbc4a9fa8606",
    "captum/_utils/common.py": "a1fc8b6752c48a76ffd3305983fed7972c7013ae6764ddb837c873c8eaa4198e",
}

# %%

if MAIN:
    from pathlib import Path
if MAIN:
    import hashlib

def gradient_helper_gradientshap_reference__sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

# %%

if MAIN:
    from typing import Any
if MAIN:
    from pathlib import Path
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def gradient_helper_gradientshap_reference_reference_environment() -> dict[str, Any]:
    """Require the exact installed package implementations."""
    import captum
    import shap

    versions = {
        "numpy": np.__version__,
        "torch": t.__version__,
        "shap": shap.__version__,
        "captum": captum.__version__,
    }
    # Only the reference implementations are pinned (also by source hash below); numpy and torch
    # are recorded, not required, so environment upgrades keep this check running.
    pinned = {key: versions[key] for key in gradient_helper_gradientshap_reference_EXPECTED_VERSIONS}
    if pinned != gradient_helper_gradientshap_reference_EXPECTED_VERSIONS:
        raise ValueError(f"GradientSHAP reference version mismatch: {versions}.")

    shap_root = Path(shap.__file__).resolve().parent
    captum_root = Path(captum.__file__).resolve().parent
    installed = {
        "shap/explainers/_gradient.py": shap_root / "explainers/_gradient.py",
        "captum/attr/_core/gradient_shap.py": captum_root / "attr/_core/gradient_shap.py",
        "captum/attr/_core/noise_tunnel.py": captum_root / "attr/_core/noise_tunnel.py",
        "captum/_utils/common.py": captum_root / "_utils/common.py",
    }
    # The installed sources are pinned below. The author's cached source copies and evidence file
    # are not needed to run the lesson, so they are not checked here.
    installed_hashes = {}
    for name, expected in gradient_helper_gradientshap_reference_EXPECTED_SOURCE_SHA256.items():
        installed_hashes[name] = gradient_helper_gradientshap_reference__sha256(installed[name])
        if installed_hashes[name] != expected:
            raise ValueError(f"Pinned package source changed: {name}.")
    return {
        "versions": versions,
        "installed_source_sha256": installed_hashes,
    }

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Callable, Iterator, Mapping
if MAIN:
    from torch import Tensor, nn

def gradient_helper_gradientshap_reference__method(methods: Any, name: str) -> Callable[..., Tensor]:
    function = methods.get(name) if isinstance(methods, Mapping) else getattr(methods, name, None)
    if not callable(function):
        raise TypeError(f"methods must expose callable {name}.")
    return function

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def gradient_helper_gradientshap_reference__model_values(model: nn.Module, inputs: Tensor, batch: int = gradient_helper_gradientshap_reference_FORWARD_BATCH) -> Tensor:
    values: list[Tensor] = []
    with t.no_grad():
        for start in range(0, len(inputs), batch):
            values.append(gradient_helper_gradientshap_reference__model_output(model, inputs[start : start + batch]).detach())
    return t.cat(values)

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Callable, Iterator, Mapping
if MAIN:
    import numpy as np

def gradient_helper_gradientshap_reference__validate_draw_table(
    draws: Mapping[str, Any], rows: int, samples: int, features: int, donors: int
) -> dict[str, np.ndarray]:
    if not isinstance(draws, Mapping) or set(draws) != {"donor_indices", "alphas", "noise"}:
        raise ValueError("draws must contain exactly donor_indices, alphas, and noise.")
    indices = gradient_helper_gradientshap_reference__numpy(draws["donor_indices"])
    alphas = gradient_helper_gradientshap_reference__numpy(draws["alphas"])
    noise = gradient_helper_gradientshap_reference__numpy(draws["noise"])
    if indices.dtype != np.int64 or indices.shape != (rows, samples):
        raise ValueError("donor_indices has the wrong shape or dtype.")
    if alphas.dtype != np.float64 or alphas.shape != (rows, samples):
        raise ValueError("alphas has the wrong shape or dtype.")
    if noise.dtype != np.float64 or noise.shape != (rows, samples, features):
        raise ValueError("noise has the wrong shape or dtype.")
    if indices.min() < 0 or indices.max() >= donors:
        raise ValueError("donor_indices selects a row outside the supplied background.")
    if not np.isfinite(alphas).all() or np.any((alphas < 0) | (alphas > 1)):
        raise ValueError("alphas must be finite values in [0,1].")
    if not np.isfinite(noise).all():
        raise ValueError("noise must contain only finite values.")
    return {
        "donor_indices": np.array(indices, copy=True),
        "alphas": np.array(alphas, copy=True),
        "noise": np.array(noise, copy=True),
    }

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Callable, Iterator, Mapping
if MAIN:
    from torch import Tensor, nn
if MAIN:
    import math
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def initialize_helper_gradient_115_41_gradient_helper_gradientshap_reference_replay_draw_table():
    global gradient_helper_gradientshap_reference_replay_draw_table
    @gradient_helper_gradientshap_reference__state_restoring_adapter
    def gradient_helper_gradientshap_reference_replay_draw_table(
        model: nn.Module,
        queries: Tensor,
        background: Tensor,
        draws: Mapping[str, Any],
        methods: Any,
        *,
        forward_batch: int = gradient_helper_gradientshap_reference_FORWARD_BATCH,
    ) -> dict[str, np.ndarray | float]:
        """Replay one explicit package draw table through the visible student methods."""
        model, _ = gradient_helper_gradientshap_reference__validate_inputs(model, queries, background, None)
        forward_batch = gradient_helper_gradientshap_reference__positive_int("forward_batch", forward_batch)
        if not isinstance(draws, Mapping) or "donor_indices" not in draws:
            raise ValueError("draws must be a mapping with a donor_indices table.")
        index_shape = gradient_helper_gradientshap_reference__numpy(draws["donor_indices"]).shape
        if len(index_shape) != 2:
            raise ValueError("donor_indices must be a two-dimensional table.")
        samples = index_shape[1]
        table = gradient_helper_gradientshap_reference__validate_draw_table(
            draws, len(queries), samples, queries.shape[1], len(background)
        )
        indices = t.from_numpy(table["donor_indices"].copy())
        alphas = t.from_numpy(table["alphas"].copy())
        noise = t.from_numpy(table["noise"].copy())
        query_copy, background_copy = queries.detach().clone(), background.detach().clone()

        with gradient_helper_gradientshap_reference__preserve_execution_state(model):
            model.eval()
            if np.count_nonzero(table["noise"]) == 0:
                function = gradient_helper_gradientshap_reference__method(methods, "expected_gradient_samples")
                contributions = function(
                    model,
                    query_copy,
                    background_copy,
                    indices,
                    alphas,
                    forward_batch=forward_batch,
                )
            else:
                function = gradient_helper_gradientshap_reference__method(methods, "noisy_expected_gradient_samples")
                contributions = function(
                    model,
                    query_copy,
                    background_copy,
                    indices,
                    alphas,
                    noise,
                    forward_batch=forward_batch,
                )
            gradient_helper_gradientshap_reference__float64_tensor("student contributions", contributions, 3)
            expected_shape = (len(queries), samples, queries.shape[1])
            if tuple(contributions.shape) != expected_shape:
                raise ValueError(f"Student contributions must have shape {expected_shape}.")
            donors = background_copy[indices]
            noisy_queries = query_copy[:, None, :] + noise
            flat_donors = donors.reshape(-1, queries.shape[1])
            flat_noisy = noisy_queries.reshape(-1, queries.shape[1])
            donor_values = gradient_helper_gradientshap_reference__model_values(model, flat_donors, forward_batch).reshape(
                len(queries), samples
            )
            noisy_values = gradient_helper_gradientshap_reference__model_values(model, flat_noisy, forward_batch).reshape(
                len(queries), samples
            )
            prediction = gradient_helper_gradientshap_reference__model_values(model, query_copy, forward_batch)
            background_values = gradient_helper_gradientshap_reference__model_values(model, background_copy, forward_batch)

        phi = contributions.mean(dim=1)
        standard_error = (
            contributions.std(dim=1, correction=1) / math.sqrt(samples)
            if samples >= 2
            else t.full_like(phi, math.nan)
        )
        variance_proxy = contributions.var(dim=1, correction=0) / math.sqrt(samples)
        repeated_delta = contributions.sum(dim=2) - (noisy_values - donor_values)
        return {
            "student_phi": phi.detach().numpy().copy(),
            "student_contributions": contributions.detach().numpy().copy(),
            "ordinary_standard_error": standard_error.detach().numpy().copy(),
            "shap_variance_proxy": variance_proxy.detach().numpy().copy(),
            "repeated_convergence_delta": repeated_delta.detach().numpy().reshape(-1).copy(),
            "prediction": prediction.detach().numpy().copy(),
            "full_background_base": float(background_values.mean()),
            "sampled_background_base": donor_values.mean(dim=1).detach().numpy().copy(),
            "noisy_query_values": noisy_values.detach().numpy().copy(),
        }


if MAIN:
    initialize_helper_gradient_115_41_gradient_helper_gradientshap_reference_replay_draw_table()

# %%

if MAIN:
    from typing import Any
if MAIN:
    from torch import Tensor, nn
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def initialize_helper_gradient_115_42_gradient_helper_gradientshap_reference_captum_same_draw_reference():
    global gradient_helper_gradientshap_reference_captum_same_draw_reference
    @gradient_helper_gradientshap_reference__state_restoring_adapter
    def gradient_helper_gradientshap_reference_captum_same_draw_reference(
        model: nn.Module,
        queries: Tensor,
        background: Tensor,
        methods: Any,
        *,
        samples: int,
        numpy_seed: int,
        torch_seed: int,
        stdev: float = 0.0,
        source_ids: Tensor | np.ndarray | None = None,
        forward_batch: int = gradient_helper_gradientshap_reference_FORWARD_BATCH,
    ) -> dict[str, Any]:
        """Call Captum 0.9.0 once and replay its exact full-query draw table."""
        from captum.attr import GradientShap
        import captum

        model, ids = gradient_helper_gradientshap_reference__validate_inputs(model, queries, background, source_ids)
        samples = gradient_helper_gradientshap_reference__positive_int("samples", samples)
        gradient_helper_gradientshap_reference__seed("numpy_seed", numpy_seed)
        gradient_helper_gradientshap_reference__seed("torch_seed", torch_seed)
        gradient_helper_gradientshap_reference_reference_environment()
        draws = gradient_helper_gradientshap_reference_captum_draw_table(
            len(queries),
            queries.shape[1],
            len(background),
            samples,
            numpy_seed=numpy_seed,
            torch_seed=torch_seed,
            stdev=stdev,
        )
        query_copy, background_copy = queries.detach().clone(), background.detach().clone()
        state_before = gradient_helper_gradientshap_reference__model_state_sha256(model)
        with gradient_helper_gradientshap_reference__preserve_execution_state(model):
            # Pin the source-observed NoiseTunnel convention.  This is restored even
            # if construction, attribution, or result validation raises.
            t.set_default_dtype(t.float32)
            np.random.seed(numpy_seed)
            t.manual_seed(torch_seed)
            model.eval()
            gradient_helper_gradientshap_reference__model_output(model, query_copy[:1])
            raw_phi, raw_delta = GradientShap(model).attribute(
                query_copy,
                background_copy,
                n_samples=samples,
                stdevs=float(stdev),
                return_convergence_delta=True,
            )
            if type(raw_phi) is not Tensor or raw_phi.shape != queries.shape:
                raise AssertionError("Captum returned an unexpected attribution tensor.")
            if raw_phi.device.type != "cpu" or raw_phi.dtype != t.float64:
                raise AssertionError("Captum attribution must remain CPU float64.")
            if type(raw_delta) is not Tensor or raw_delta.shape != (len(queries) * samples,):
                raise AssertionError("Captum returned an unexpected repeated-delta tensor.")
            author_phi = raw_phi.detach().numpy().copy()
            author_delta = raw_delta.detach().numpy().copy()

        replay = gradient_helper_gradientshap_reference_replay_draw_table(
            model,
            queries,
            background,
            draws,
            methods,
            forward_batch=forward_batch,
        )
        state_after = gradient_helper_gradientshap_reference__model_state_sha256(model)
        if state_after != state_before:
            raise AssertionError("Captum reference adapter did not restore the model state.")
        parity_error = gradient_helper_gradientshap_reference__max_abs(author_phi, replay["student_phi"])
        delta_error = gradient_helper_gradientshap_reference__max_abs(author_delta, replay["repeated_convergence_delta"])
        return {
            "backend": "captum",
            "version": captum.__version__,
            "samples": samples,
            "numpy_seed": numpy_seed,
            "torch_seed": torch_seed,
            "stdev": float(stdev),
            "source_ids": ids,
            "draws": draws,
            "author_phi": author_phi,
            "student_phi": replay["student_phi"],
            "student_contributions": replay["student_contributions"],
            "ordinary_standard_error": replay["ordinary_standard_error"],
            "author_convergence_delta": author_delta,
            "student_repeated_convergence_delta": replay["repeated_convergence_delta"],
            "prediction": replay["prediction"],
            "full_background_base": replay["full_background_base"],
            "sampled_background_base": replay["sampled_background_base"],
            "parity_max_abs_error": parity_error,
            "convergence_delta_max_abs_error": delta_error,
            "parity_passed": gradient_helper_gradientshap_reference__close(author_phi, replay["student_phi"]),
            "warnings": gradient_helper_gradientshap_reference_CAPTUM_WARNINGS,
            "identity_sha256": gradient_helper_gradientshap_reference__identity(
                model,
                queries,
                background,
                ids,
                draws,
                author_phi,
                replay,
                author_convergence_delta=author_delta,
                student_repeated_convergence_delta=replay["repeated_convergence_delta"],
            ),
        }


if MAIN:
    initialize_helper_gradient_115_42_gradient_helper_gradientshap_reference_captum_same_draw_reference()

# %%

if MAIN:
    from torch import Tensor, nn
if MAIN:
    import torch as t

def gradient_helper_gradientshap_reference__smoothed_relu() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(1, 1, bias=False, dtype=t.float64),
        nn.ReLU(),
        nn.Linear(1, 1, bias=False, dtype=t.float64),
    )
    with t.no_grad():
        model[0].weight.fill_(1.0)
        model[2].weight.fill_(1.0)
    return model.eval()

# %%

if MAIN:
    import math
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def gradient_helper_gradientshap_reference_shap_draw_table(
    rows: int,
    features: int,
    background_rows: int,
    samples: int,
    *,
    numpy_seed: int,
    torch_seed: int,
    local_smoothing: float,
) -> dict[str, np.ndarray]:
    """Recreate SHAP 0.49.1's row-major donor/alpha/noise sequence locally."""
    rows = gradient_helper_gradientshap_reference__positive_int("rows", rows)
    features = gradient_helper_gradientshap_reference__positive_int("features", features)
    background_rows = gradient_helper_gradientshap_reference__positive_int("background_rows", background_rows)
    samples = gradient_helper_gradientshap_reference__positive_int("samples", samples)
    numpy_seed = gradient_helper_gradientshap_reference__seed("numpy_seed", numpy_seed)
    torch_seed = gradient_helper_gradientshap_reference__seed("torch_seed", torch_seed)
    if not isinstance(local_smoothing, (int, float)) or not math.isfinite(local_smoothing):
        raise ValueError("local_smoothing must be a finite nonnegative number.")
    local_smoothing = float(local_smoothing)
    if local_smoothing < 0:
        raise ValueError("local_smoothing must be nonnegative.")

    legacy = np.random.RandomState(numpy_seed)
    donor_indices = np.empty((rows, samples), dtype=np.int64)
    alphas = np.empty((rows, samples), dtype=np.float64)
    for row in range(rows):
        for sample in range(samples):
            donor_indices[row, sample] = legacy.choice(background_rows)
            alphas[row, sample] = legacy.uniform()

    generator = t.Generator(device="cpu").manual_seed(torch_seed)
    noise = np.zeros((rows, samples, features), dtype=np.float64)
    if local_smoothing > 0:
        for row in range(rows):
            for sample in range(samples):
                noise[row, sample] = (
                    t.empty(features, dtype=t.float64)
                    .normal_(generator=generator)
                    .mul(local_smoothing)
                    .numpy()
                )
    return {
        "donor_indices": donor_indices,
        "alphas": alphas,
        "noise": noise,
    }

# %%

if MAIN:
    from typing import Any
if MAIN:
    from torch import Tensor, nn
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def initialize_helper_gradient_115_45_gradient_helper_gradientshap_reference_shap_same_draw_reference():
    global gradient_helper_gradientshap_reference_shap_same_draw_reference
    @gradient_helper_gradientshap_reference__state_restoring_adapter
    def gradient_helper_gradientshap_reference_shap_same_draw_reference(
        model: nn.Module,
        queries: Tensor,
        background: Tensor,
        methods: Any,
        *,
        samples: int,
        numpy_seed: int,
        torch_seed: int,
        batch_size: int = gradient_helper_gradientshap_reference_SHAP_BATCH_SIZE,
        local_smoothing: float = 0.0,
        source_ids: Tensor | np.ndarray | None = None,
        forward_batch: int = gradient_helper_gradientshap_reference_FORWARD_BATCH,
    ) -> dict[str, Any]:
        """Call SHAP 0.49.1 and replay its exact draws through student methods."""
        import shap

        model, ids = gradient_helper_gradientshap_reference__validate_inputs(model, queries, background, source_ids)
        samples = gradient_helper_gradientshap_reference__positive_int("samples", samples)
        batch_size = gradient_helper_gradientshap_reference__positive_int("batch_size", batch_size)
        gradient_helper_gradientshap_reference__seed("numpy_seed", numpy_seed)
        gradient_helper_gradientshap_reference__seed("torch_seed", torch_seed)
        gradient_helper_gradientshap_reference_reference_environment()
        draws = gradient_helper_gradientshap_reference_shap_draw_table(
            len(queries),
            queries.shape[1],
            len(background),
            samples,
            numpy_seed=numpy_seed,
            torch_seed=torch_seed,
            local_smoothing=local_smoothing,
        )
        query_copy, background_copy = queries.detach().clone(), background.detach().clone()
        state_before = gradient_helper_gradientshap_reference__model_state_sha256(model)
        with gradient_helper_gradientshap_reference__preserve_execution_state(model):
            t.set_default_dtype(t.float64)
            np.random.seed(numpy_seed)
            t.manual_seed(torch_seed)
            model.eval()
            gradient_helper_gradientshap_reference__model_output(model, query_copy[:1])
            explainer = shap.GradientExplainer(
                model,
                background_copy,
                batch_size=batch_size,
                local_smoothing=float(local_smoothing),
            )
            raw_phi, raw_variance = explainer.shap_values(
                query_copy,
                nsamples=samples,
                rseed=numpy_seed,
                return_variances=True,
            )
            raw_phi_array = np.asarray(raw_phi, dtype=np.float64)
            if raw_phi_array.shape != (*queries.shape, 1):
                raise AssertionError(
                    f"SHAP scalar-output attribution had unexpected shape {raw_phi_array.shape}."
                )
            author_phi = raw_phi_array[..., 0].copy()
            if not isinstance(raw_variance, list) or len(raw_variance) != 1:
                raise AssertionError("SHAP scalar-output variance must be list[one ndarray].")
            variance_proxy = np.asarray(raw_variance[0], dtype=np.float64)
            if variance_proxy.shape != tuple(queries.shape) or not np.isfinite(variance_proxy).all():
                raise AssertionError("SHAP returned an invalid variance-proxy array.")
            prediction = gradient_helper_gradientshap_reference__model_values(model, query_copy).numpy().copy()
            full_background_base = float(gradient_helper_gradientshap_reference__model_values(model, background_copy).mean())

        replay = gradient_helper_gradientshap_reference_replay_draw_table(
            model,
            queries,
            background,
            draws,
            methods,
            forward_batch=forward_batch,
        )
        state_after = gradient_helper_gradientshap_reference__model_state_sha256(model)
        if state_after != state_before:
            raise AssertionError("SHAP reference adapter did not restore the model state.")
        parity_error = gradient_helper_gradientshap_reference__max_abs(author_phi, replay["student_phi"])
        variance_error = gradient_helper_gradientshap_reference__max_abs(variance_proxy, replay["shap_variance_proxy"])
        prediction_error = gradient_helper_gradientshap_reference__max_abs(prediction, replay["prediction"])
        base_error = abs(full_background_base - float(replay["full_background_base"]))
        return {
            "backend": "shap",
            "version": shap.__version__,
            "samples": samples,
            "numpy_seed": numpy_seed,
            "torch_seed": torch_seed,
            "batch_size": batch_size,
            "local_smoothing": float(local_smoothing),
            "source_ids": ids,
            "draws": draws,
            "author_phi_raw": raw_phi_array.copy(),
            "author_phi": author_phi,
            "student_phi": replay["student_phi"],
            "student_contributions": replay["student_contributions"],
            "ordinary_standard_error": replay["ordinary_standard_error"],
            "author_shap_variance_proxy": variance_proxy.copy(),
            "student_shap_variance_proxy": replay["shap_variance_proxy"],
            "prediction": prediction,
            "full_background_base": full_background_base,
            "sampled_background_base": replay["sampled_background_base"],
            "parity_max_abs_error": parity_error,
            "variance_proxy_max_abs_error": variance_error,
            "prediction_max_abs_error": prediction_error,
            "base_max_abs_error": base_error,
            "parity_passed": gradient_helper_gradientshap_reference__close(author_phi, replay["student_phi"]),
            "warnings": gradient_helper_gradientshap_reference_SHAP_WARNINGS,
            "identity_sha256": gradient_helper_gradientshap_reference__identity(
                model,
                queries,
                background,
                ids,
                draws,
                author_phi,
                replay,
                author_phi_raw=raw_phi_array,
                author_shap_variance_proxy=variance_proxy,
                student_shap_variance_proxy=replay["shap_variance_proxy"],
            ),
        }


if MAIN:
    initialize_helper_gradient_115_45_gradient_helper_gradientshap_reference_shap_same_draw_reference()

# %%

if MAIN:
    from typing import Any
if MAIN:
    import math
if MAIN:
    import numpy as np
if MAIN:
    import torch as t

def gradient_helper_gradientshap_reference_positive_noise_reference(methods: Any) -> dict[str, Any]:
    """Check both positive-noise streams on an analytic one-dimensional ReLU.

    With a zero donor, every non-kink draw contributes ``ReLU(x + noise)``.
    This gives an exact finite-draw oracle for each package while the Gaussian
    population target remains ``sigma*pdf(x/sigma) + x*cdf(x/sigma)``.
    """
    model = gradient_helper_gradientshap_reference__smoothed_relu()
    query = t.tensor([[0.15]], dtype=t.float64)
    background = t.tensor([[0.0]], dtype=t.float64)
    sigma = 0.2
    samples = 257
    ids = np.array([915001], dtype=np.int64)
    shap_result = gradient_helper_gradientshap_reference_shap_same_draw_reference(
        model,
        query,
        background,
        methods,
        samples=samples,
        numpy_seed=gradient_helper_gradientshap_reference_SHAP_NUMPY_SEED,
        torch_seed=gradient_helper_gradientshap_reference_SHAP_TORCH_SEED,
        batch_size=257,
        local_smoothing=sigma,
        source_ids=ids,
    )
    captum_result = gradient_helper_gradientshap_reference_captum_same_draw_reference(
        model,
        query,
        background,
        methods,
        samples=samples,
        numpy_seed=gradient_helper_gradientshap_reference_CAPTUM_NUMPY_SEED,
        torch_seed=gradient_helper_gradientshap_reference_CAPTUM_TORCH_SEED,
        stdev=sigma,
        source_ids=ids,
    )
    ratio = float(query[0, 0] / sigma)
    normal_pdf = math.exp(-0.5 * ratio**2) / math.sqrt(2 * math.pi)
    normal_cdf = 0.5 * (1.0 + math.erf(ratio / math.sqrt(2.0)))
    population_target = sigma * normal_pdf + float(query[0, 0]) * normal_cdf

    package_results = {}
    all_passed = True
    for name, result in (("shap", shap_result), ("captum", captum_result)):
        finite_draw_oracle = np.maximum(
            float(query[0, 0]) + result["draws"]["noise"][0, :, 0], 0.0
        ).mean()
        finite_draw_error = abs(float(result["author_phi"][0, 0]) - finite_draw_oracle)
        passed = bool(result["parity_passed"] and finite_draw_error <= gradient_helper_gradientshap_reference_AUTHOR_ATOL)
        all_passed &= passed
        package_results[name] = {
            "same_draw": result,
            "finite_draw_analytic_phi": finite_draw_oracle,
            "finite_draw_analytic_max_abs_error": finite_draw_error,
            "population_smoothed_relu_phi": population_target,
            "finite_draw_minus_population": finite_draw_oracle - population_target,
            "positive_noise_confirmed": bool(np.any(result["draws"]["noise"] != 0)),
            "passed": passed,
        }
    return {
        "status": "PASS" if all_passed else "FAIL",
        "fixture": "ReLU(x), zero donor, x=0.15, sigma=0.2",
        "samples": samples,
        "packages": package_results,
        "warning": "SHAP and Captum use different draw tables and noise dtypes; their finite Monte Carlo vectors are not cross-package equality targets.",
    }

# %%

if MAIN:
    from typing import Any
if MAIN:
    from collections.abc import Callable, Iterator, Mapping
if MAIN:
    from torch import Tensor, nn
if MAIN:
    import numpy as np

def gradient_helper_gradientshap_reference_validate_reference(
    models: Mapping[str, nn.Module],
    queries: Tensor,
    background: Tensor,
    source_ids: Tensor | np.ndarray,
    methods: Any,
    *,
    samples: int = gradient_helper_gradientshap_reference_REFERENCE_SAMPLES,
) -> dict[str, Any]:
    """Run complete same-draw package parity over all seven models and 272 rows."""
    if not isinstance(models, Mapping) or tuple(models) != gradient_helper_gradientshap_reference_MODEL_KEYS:
        raise ValueError("models must contain the seven frozen model keys in protocol order.")
    gradient_helper_gradientshap_reference__float64_tensor("queries", queries, 2)
    gradient_helper_gradientshap_reference__float64_tensor("background", background, 2)
    ids = gradient_helper_gradientshap_reference__source_ids(source_ids, len(queries))
    if tuple(queries.shape) != (gradient_helper_gradientshap_reference_ALL_ROWS, gradient_helper_gradientshap_reference_FEATURES):
        raise ValueError(f"queries must have frozen shape ({gradient_helper_gradientshap_reference_ALL_ROWS},{gradient_helper_gradientshap_reference_FEATURES}).")
    if tuple(background.shape) != (gradient_helper_gradientshap_reference_BACKGROUND_ROWS, gradient_helper_gradientshap_reference_FEATURES):
        raise ValueError(
            f"background must contain all {gradient_helper_gradientshap_reference_BACKGROUND_ROWS} primary donors and {gradient_helper_gradientshap_reference_FEATURES} features."
        )
    if samples != gradient_helper_gradientshap_reference_REFERENCE_SAMPLES:
        raise ValueError(f"Full package reference coverage is frozen at K={gradient_helper_gradientshap_reference_REFERENCE_SAMPLES}.")
    environment = gradient_helper_gradientshap_reference_reference_environment()

    package_results: dict[str, dict[str, Any]] = {
        "shap": {"models": {}},
        "captum": {"models": {}},
    }
    maxima = {"shap": 0.0, "captum": 0.0}
    all_passed = True
    shared_draw_hashes: dict[str, dict[str, str]] = {}
    for key in gradient_helper_gradientshap_reference_MODEL_KEYS:
        model = gradient_helper_gradientshap_reference__validate_model(models[key], gradient_helper_gradientshap_reference_FEATURES)
        shap_result = gradient_helper_gradientshap_reference_shap_same_draw_reference(
            model,
            queries,
            background,
            methods,
            samples=samples,
            numpy_seed=gradient_helper_gradientshap_reference_SHAP_NUMPY_SEED,
            torch_seed=gradient_helper_gradientshap_reference_SHAP_TORCH_SEED,
            batch_size=gradient_helper_gradientshap_reference_SHAP_BATCH_SIZE,
            local_smoothing=0.0,
            source_ids=ids,
        )
        captum_result = gradient_helper_gradientshap_reference_captum_same_draw_reference(
            model,
            queries,
            background,
            methods,
            samples=samples,
            numpy_seed=gradient_helper_gradientshap_reference_CAPTUM_NUMPY_SEED,
            torch_seed=gradient_helper_gradientshap_reference_CAPTUM_TORCH_SEED,
            stdev=0.0,
            source_ids=ids,
        )
        for backend, result in (("shap", shap_result), ("captum", captum_result)):
            package_results[backend]["models"][key] = gradient_helper_gradientshap_reference__compact_model_result(result, backend)
            maxima[backend] = max(maxima[backend], float(result["parity_max_abs_error"]))
            all_passed &= bool(result["parity_passed"])
            hashes = {
                name: result["identity_sha256"][name]
                for name in ("source_ids", "queries", "background", "donor_indices", "alphas", "noise")
            }
            if backend not in shared_draw_hashes:
                shared_draw_hashes[backend] = hashes
                package_results[backend]["draws"] = result["draws"]
            elif hashes != shared_draw_hashes[backend]:
                raise AssertionError(f"{backend} draw identity changed between models.")

    package_results["shap"].update(
        version=gradient_helper_gradientshap_reference_EXPECTED_VERSIONS["shap"],
        warnings=gradient_helper_gradientshap_reference_SHAP_WARNINGS,
        max_parity_error=maxima["shap"],
        all_models_passed=all(
            item["parity_passed"] for item in package_results["shap"]["models"].values()
        ),
        draw_identity_sha256=shared_draw_hashes["shap"],
    )
    package_results["captum"].update(
        version=gradient_helper_gradientshap_reference_EXPECTED_VERSIONS["captum"],
        warnings=gradient_helper_gradientshap_reference_CAPTUM_WARNINGS,
        max_parity_error=maxima["captum"],
        all_models_passed=all(
            item["parity_passed"] for item in package_results["captum"]["models"].values()
        ),
        draw_identity_sha256=shared_draw_hashes["captum"],
    )

    positive_noise = gradient_helper_gradientshap_reference_positive_noise_reference(methods)
    package_draws_distinct = any(
        shared_draw_hashes["shap"][name] != shared_draw_hashes["captum"][name]
        for name in ("donor_indices", "alphas")
    )
    passes = {
        "sources_pinned": True,
        "shap_same_draw_all_models": bool(package_results["shap"]["all_models_passed"]),
        "captum_same_draw_all_models": bool(package_results["captum"]["all_models_passed"]),
        "positive_noise_same_draw": positive_noise["status"] == "PASS",
        "package_draw_tables_distinct": package_draws_distinct,
    }
    all_passed &= all(passes.values())
    counts = {
        "models": 7,
        "model_rows": 1904,
        "background_rows": 64,
        "samples_per_package": 128,
        "shap_vectors": 1904,
        "shap_coordinates": 15232,
        "captum_vectors": 1904,
        "captum_coordinates": 15232,
        "package_vectors": 3808,
        "package_coordinates": 30464,
    }
    return {
        "status": "PASS" if all_passed else "FAIL",
        "environment": environment,
        "versions": {
            "shap": gradient_helper_gradientshap_reference_EXPECTED_VERSIONS["shap"],
            "captum": gradient_helper_gradientshap_reference_EXPECTED_VERSIONS["captum"],
        },
        "counts": counts,
        "tolerances": {
            "same_draw_atol": gradient_helper_gradientshap_reference_AUTHOR_ATOL,
            "same_draw_rtol": gradient_helper_gradientshap_reference_AUTHOR_RTOL,
        },
        "passes": passes,
        "scope": {
            "models": len(gradient_helper_gradientshap_reference_MODEL_KEYS),
            "model_keys": gradient_helper_gradientshap_reference_MODEL_KEYS,
            "rows_per_model": gradient_helper_gradientshap_reference_ALL_ROWS,
            "background_rows": gradient_helper_gradientshap_reference_BACKGROUND_ROWS,
            "features": gradient_helper_gradientshap_reference_FEATURES,
            "samples_per_query": samples,
            "model_rows_per_package": len(gradient_helper_gradientshap_reference_MODEL_KEYS) * gradient_helper_gradientshap_reference_ALL_ROWS,
            "local_coordinates_per_package": len(gradient_helper_gradientshap_reference_MODEL_KEYS) * gradient_helper_gradientshap_reference_ALL_ROWS * gradient_helper_gradientshap_reference_FEATURES,
            "draws_per_package": len(gradient_helper_gradientshap_reference_MODEL_KEYS) * gradient_helper_gradientshap_reference_ALL_ROWS * samples,
            "contribution_coordinates_per_package": (
                len(gradient_helper_gradientshap_reference_MODEL_KEYS) * gradient_helper_gradientshap_reference_ALL_ROWS * samples * gradient_helper_gradientshap_reference_FEATURES
            ),
        },
        "identity_sha256": {
            "source_ids": gradient_helper_gradientshap_reference__array_sha256(ids),
            "queries": gradient_helper_gradientshap_reference__array_sha256(queries),
            "background": gradient_helper_gradientshap_reference__array_sha256(background),
        },
        "packages": package_results,
        "positive_noise": positive_noise,
        "acceptance": {
            "atol": gradient_helper_gradientshap_reference_AUTHOR_ATOL,
            "rtol": gradient_helper_gradientshap_reference_AUTHOR_RTOL,
            "requires_same_draws": True,
            "shap_variance_proxy_is_acceptance_metric": False,
            "captum_convergence_delta_is_acceptance_metric": False,
            "cross_package_same_seed_parity_is_claimed": False,
        },
    }

# %%

if MAIN:
    REFERENCE_SAMPLES = gradient_helper_gradientshap_reference_REFERENCE_SAMPLES
if MAIN:
    gradient_validate_reference = gradient_helper_gradientshap_reference_validate_reference

# %%

def reference_check(
    models: Mapping[str, nn.Module],
    queries: Tensor,
    background: Tensor,
    source_ids: Tensor | np.ndarray,
    methods: Any,
) -> dict[str, Any]:
    """Visible learner call for the frozen all-model, all-row package check."""
    return gradient_validate_reference(
        models,
        queries,
        background,
        source_ids,
        methods,
        samples=REFERENCE_SAMPLES,
    )

# %%

if MAIN:
    gradient_evaluated = gradient_evaluate_study(
        gradient_fitted,
        gradient_prepared,
        gradient_train,
        gradient_explain,
        gradient_audit,
        methods=gradient_methods,
        deep_methods=deep_methods,
    )
if MAIN:
    models = {
        key: gradient__model_from_state(record)
        for key, record in gradient_fitted["models"].items()
    }
if MAIN:
    gradient_reference = reference_check(
        models,
        gradient_evaluated["all_row_x_raw"],
        gradient_prepared["backgrounds"]["primary64"]["x_raw"],
        gradient_evaluated["query_source_ids"],
        gradient_methods,
    )
if MAIN:
    gradient_diagnostics = gradient_run_diagnostics(methods=gradient_methods)
if MAIN:
    gradient_fixed_play = gradient_native_play(
        gradient_fitted,
        gradient_prepared,
        gradient_evaluated,
        gradient_train,
        gradient_explain,
        gradient_audit,
        methods=gradient_methods,
        deep_methods=deep_methods,
    )
if MAIN:
    gradient_summary = gradient_persist_completed(
        native_directory,
        gradient_fitted,
        gradient_prepared,
        gradient_evaluated,
        gradient_diagnostics,
        gradient_reference,
        gradient_fixed_play,
        timings={},
    )
if MAIN:
    gradient_write_notebook_summary(canonical_dir / "artifacts" / "gradient", native_directory, gradient_summary)

if MAIN:
    print("Reference parity:", gradient_reference["status"])
if MAIN:
    print("Package vectors checked:", gradient_reference["counts"]["package_vectors"])
if MAIN:
    print("All model-row explanations:", gradient_summary["all_model_rows"])
if MAIN:
    print("Audit used for selection:", gradient_summary["audit_used_for_selection"])

# %%

if MAIN:
    gradient_show_figure(
        gradient_plots.plot_native_overview(gradient_prepared, gradient_evaluated),
        "14_native_overview",
    )
if MAIN:
    gradient_show_figure(
        gradient_plots.plot_allrow_method_map(gradient_evaluated),
        "15_allrow_method_map",
    )

# %%

if MAIN:
    gradient_show_figure(
        gradient_plots.plot_fixed_waterfall(gradient_fixed_play, feature_names=gradient_FEATURE_NAMES),
        "16_fixed_waterfall",
    )
if MAIN:
    gradient_show_figure(
        gradient_plots.plot_background_and_noise(gradient_evaluated),
        "17_background_and_noise",
    )

# %%

if MAIN:
    gradient_show_figure(
        gradient_plots.plot_replicates_and_distance(gradient_evaluated),
        "18_replicates_and_distance",
    )
if MAIN:
    gradient_show_figure(
        gradient_plots.plot_replacement_curves(gradient_evaluated, model_key="clean_0"),
        "19_replacement_curves",
    )

# %%

if MAIN:
    gradient_real_play = gradient_native_play(
        gradient_fitted,
        gradient_prepared,
        gradient_evaluated,
        gradient_train,
        gradient_explain,
        gradient_audit,
        methods=gradient_methods,
        deep_methods=deep_methods,
        model_key="clean_0",
        split="audit",
        row_index=0,
        background_key="primary64",
        samples=2048,
        sigma=0.0,
    )
if MAIN:
    gradient_show_figure(
        gradient_plots.plot_real_recipe_play(gradient_real_play, feature_names=gradient_FEATURE_NAMES),
        "20_real_recipe_play",
    )

# %%

def initialize_lesson_plumbing():
    """Explicit imports, source paths, metadata and original decorated definitions.

    No data loader, scientific callback, model, experiment or plot is called.
    This explicit operation can initialize external libraries and path metadata.
    Repeated calls intentionally rebind decorated callbacks as original definitions do.
    """
    global Any
    global BACKGROUND_SPECS
    global BUDGETS
    global COALITION_DISTANCE_DONORS
    global Callable
    global DEEP_METHOD_NAMES
    global DISTANCE_BATCH
    global DecisionTreeRegressor
    global FORWARD_BATCH
    global GradientBoostingRegressor
    global INDEPENDENT_BACKGROUND_SEEDS
    global Iterator
    global Mapping
    global NATIVE_DIR
    global NOISE_SCHEDULE_SEED
    global NOISE_SIGMAS
    global OUTPUT_ROOT
    global PATH_DISTANCE_SAMPLES
    global PRIMARY_BACKGROUND_SEED
    global PRIMARY_SCHEDULE_SEED
    global Path
    global QUADRATURE_NODES
    global REFERENCE_SAMPLES
    global REFERENCE_SEEDS
    global REPLICATES
    global REPLICATE_MODELS
    global REPLICATE_SCHEDULE_SEED
    global RandomForestRegressor
    global Sequence
    global SimpleNamespace
    global Tensor
    global _batched_prediction
    global _check_widths
    global _checked_curves
    global _clone_tree
    global _combine_baselines
    global _combine_explanations
    global _combine_mc
    global _combine_primary
    global _combine_quadrature
    global _combine_row_blocks
    global _eg_float_tensor
    global _eg_index_tensor
    global _eg_model
    global _eg_model_output
    global _eg_positive_int
    global _eg_preserve_execution_state
    global _eg_seed_words
    global _eg_validate_paths
    global _epsilon
    global _identity_hash
    global _layers
    global _method_tensor
    global _model_output
    global _require_methods
    global _require_namespace
    global _seed_words
    global _validate_rankings
    global canonical_dir
    global check_reference
    global contextlib
    global copy
    global deep_DTYPE
    global deep_FEATURES
    global deep_FEATURE_NAMES
    global deep_FIGURE_DIR
    global deep_HIDDEN
    global deep_LEARNING_RATE
    global deep_METHOD_NAMES
    global deep_MODEL_KEYS
    global deep_MODEL_LABELS
    global deep_MODEL_SEEDS
    global deep_RANDOM_ORDER_SEED
    global deep_RIDGE_PENALTY
    global deep_RUN_OUTPUT
    global deep_SAVE_FIGURES
    global deep_SHUFFLE_SEEDS
    global deep_TRAINING_STEPS
    global deep_UNITS
    global deep_WEIGHT_DECAY
    global deep__clone_state
    global deep__float_tensor
    global deep__ids_tensor
    global deep__linear
    global deep__max_abs
    global deep__method
    global deep__normalization_tensors
    global deep__output_tensor
    global deep__prediction
    global deep__require_disjoint_ids
    global deep__rows
    global deep__state_shapes
    global deep__validate_evaluation_inputs
    global deep__validate_fitted
    global deep_background_order
    global deep_curriculum_methods
    global deep_exercises_dir
    global deep_helper_deepshap_diagnostics_DTYPE
    global deep_helper_deepshap_reference_AUTHOR_ATOL
    global deep_helper_deepshap_reference_AUTHOR_RTOL
    global deep_helper_deepshap_reference_BACKGROUND_ROWS
    global deep_helper_deepshap_reference_EXACT_ATOL
    global deep_helper_deepshap_reference_EXACT_RTOL
    global deep_helper_deepshap_reference_FEATURES
    global deep_helper_deepshap_reference_HERE
    global deep_helper_deepshap_reference_HIDDEN
    global deep_helper_deepshap_reference_MODEL_KEYS
    global deep_helper_deepshap_reference_PRIMARY_BACKGROUND_SEED
    global deep_helper_deepshap_study_FEATURES
    global deep_helper_deepshap_study_HIDDEN
    global deep_helper_deepshap_study_METHOD_NAMES
    global deep_helper_deepshap_study_MODEL_KEYS
    global deep_helper_run_native_DETERMINISTIC_ARTIFACTS
    global deep_helper_run_native_TIMING_ARTIFACT
    global deep_load_split
    global deep_methods
    global deep_persist_completed
    global deep_plots
    global deep_plumbing_deepshap_data_source_data
    global deep_root_dir
    global deep_save_payload
    global deep_section_dir
    global deep_tests
    global deep_validate_reference
    global deep_write_before_audit
    global deep_write_notebook_summary
    global display
    global functools
    global gradient_DTYPE
    global gradient_FEATURES
    global gradient_FEATURE_NAMES
    global gradient_FIGURE_DIR
    global gradient_HIDDEN
    global gradient_LEARNING_RATE
    global gradient_METHOD_NAMES
    global gradient_MODEL_KEYS
    global gradient_MODEL_LABELS
    global gradient_MODEL_SEEDS
    global gradient_RANDOM_ORDER_SEED
    global gradient_RIDGE_PENALTY
    global gradient_RUN_OUTPUT
    global gradient_SAVE_FIGURES
    global gradient_SHUFFLE_SEEDS
    global gradient_TRAINING_STEPS
    global gradient_UNITS
    global gradient_WEIGHT_DECAY
    global gradient__clone_state
    global gradient__fixed_random_orders
    global gradient__float_tensor
    global gradient__folding_error
    global gradient__ids_tensor
    global gradient__normalization_tensors
    global gradient__output_tensor
    global gradient__require_disjoint_ids
    global gradient__rows
    global gradient__state_shapes
    global gradient__validate_evaluation_inputs
    global gradient__validate_fitted
    global gradient_background_order
    global gradient_exercises_dir
    global gradient_helper_gradientshap_reference_ALL_ROWS
    global gradient_helper_gradientshap_reference_AUTHOR_ATOL
    global gradient_helper_gradientshap_reference_AUTHOR_RTOL
    global gradient_helper_gradientshap_reference_BACKGROUND_ROWS
    global gradient_helper_gradientshap_reference_CAPTUM_NUMPY_SEED
    global gradient_helper_gradientshap_reference_CAPTUM_TORCH_SEED
    global gradient_helper_gradientshap_reference_CAPTUM_WARNINGS
    global gradient_helper_gradientshap_reference_EXPECTED_SOURCE_SHA256
    global gradient_helper_gradientshap_reference_EXPECTED_VERSIONS
    global gradient_helper_gradientshap_reference_FEATURES
    global gradient_helper_gradientshap_reference_FORWARD_BATCH
    global gradient_helper_gradientshap_reference_MODEL_KEYS
    global gradient_helper_gradientshap_reference_REFERENCE_SAMPLES
    global gradient_helper_gradientshap_reference_SHAP_BATCH_SIZE
    global gradient_helper_gradientshap_reference_SHAP_NUMPY_SEED
    global gradient_helper_gradientshap_reference_SHAP_TORCH_SEED
    global gradient_helper_gradientshap_reference_SHAP_WARNINGS
    global gradient_helper_gradientshap_study_FEATURES
    global gradient_helper_gradientshap_study_FORWARD_BATCH
    global gradient_helper_gradientshap_study_HIDDEN
    global gradient_helper_gradientshap_study_MODEL_KEYS
    global gradient_helper_gradientshap_study_RANDOM_ORDER_SEED
    global gradient_helper_run_native_BUDGETS
    global gradient_helper_run_native_DETERMINISTIC_ARTIFACTS
    global gradient_helper_run_native_EXTERNAL_DETERMINISTIC_ARTIFACT
    global gradient_helper_run_native_HASH_CHUNK_BYTES
    global gradient_helper_run_native_REPLICATE_ELEMENTS
    global gradient_helper_run_native_REPLICATE_MODELS
    global gradient_helper_run_native_REPLICATE_SHAPE
    global gradient_helper_run_native_SMALL_DETERMINISTIC_ARTIFACTS
    global gradient_helper_run_native_TIMING_ARTIFACT
    global gradient_load_split
    global gradient_methods
    global gradient_persist_completed
    global gradient_plots
    global gradient_plumbing_gradientshap_data_load_manifest
    global gradient_plumbing_gradientshap_data_tensor_mapping_sha256
    global gradient_root_dir
    global gradient_section_dir
    global gradient_tests
    global gradient_validate_reference
    global gradient_write_notebook_summary
    global hashlib
    global itertools
    global json
    global load_manifest
    global math
    global nn
    global np
    global os
    global persist_prepared
    global pickle
    global plt
    global random
    global sys
    global t
    global tensor_mapping_sha256
    global time
    global torch
    global tree_FEATURE_NAMES
    global tree_FIGURE_DIR
    global tree_RUN_OUTPUT
    global tree_SAVE_FIGURES
    global tree_background_order
    global tree_exercises_dir
    global tree_helper_run_native_DETERMINISTIC_ARTIFACTS
    global tree_helper_run_native_TIMING_ARTIFACT
    global tree_helper_treeshap_reference_ADDENDUM_SHA256
    global tree_helper_treeshap_reference_ATOL
    global tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL
    global tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL
    global tree_helper_treeshap_reference_EXACT_KEYS
    global tree_helper_treeshap_reference_HERE
    global tree_helper_treeshap_reference_MODEL_KEYS
    global tree_helper_treeshap_reference_RTOL
    global tree_load_split
    global tree_persist_completed
    global tree_plots
    global tree_plumbing_treeshap_data_source_data
    global tree_root_dir
    global tree_save_payload
    global tree_section_dir
    global tree_tests
    global tree_write_before_audit
    global tree_write_notebook_summary

    from pathlib import Path
    import itertools
    import math
    import os
    import sys
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import tree_treeshap_methods_tests as tree_tests
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.tree_treeshap_data import background_order as tree_background_order, load_split as tree_load_split
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import tree_treeshap_plots as tree_plots
    import time
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    import hashlib
    from typing import Any
    import torch
    import json
    import pickle
    import copy
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.tree_treeshap_data import source_data as tree_plumbing_treeshap_data_source_data
    from types import SimpleNamespace
    import torch as t
    from torch import Tensor, nn
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import deep_deepshap_methods_tests as deep_tests
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import deep_deepshap_plots as deep_plots
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.deep_deepshap_data import load_split as deep_load_split
    from collections.abc import Mapping
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.deep_deepshap_data import (
            background_order as deep_background_order,
        )
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.deep_deepshap_data import source_data as deep_plumbing_deepshap_data_source_data
    from collections.abc import Mapping, Sequence
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import gradient_gradientshap_methods_tests as gradient_tests
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap import gradient_gradientshap_plots as gradient_plots
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.gradient_gradientshap_data import (
            background_order as gradient_background_order,
            load_manifest as load_manifest,
            load_split as gradient_load_split,
            tensor_mapping_sha256 as tensor_mapping_sha256,
        )
    from collections.abc import Iterator
    import contextlib
    import random
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.gradient_gradientshap_data import tensor_mapping_sha256 as gradient_plumbing_gradientshap_data_tensor_mapping_sha256
    from chapter16_shapley_attribution_baselines.exercises.part3_model_specific_shap.gradient_gradientshap_data import load_manifest as gradient_plumbing_gradientshap_data_load_manifest
    from typing import Any, Iterator
    from collections.abc import Callable, Iterator, Mapping
    import functools
    canonical_dir = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / "chapter16_shapley_attribution_baselines").exists()) / "chapter16_shapley_attribution_baselines" / "exercises" / "part3_model_specific_shap"
    tree_root_dir = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / tree_chapter).exists())
    tree_exercises_dir = tree_root_dir / tree_chapter / "exercises"
    tree_section_dir = tree_exercises_dir / tree_section
    tree_FEATURE_NAMES = np.array([
            "Cement", "Blast-furnace slag", "Fly ash", "Water",
            "Superplasticizer", "Coarse aggregate", "Fine aggregate", "Age",
        ])
    tree_RUN_OUTPUT = os.environ.get("ARENA_TREESHAP_OUTPUT_DIRECTORY")
    tree_FIGURE_DIR = Path(
            os.environ.get("ARENA_TREESHAP_FIGURE_DIR", tree_RUN_OUTPUT or canonical_dir / "media" / 'tree')
        )
    tree_SAVE_FIGURES = os.environ.get(
            "ARENA_TREESHAP_SAVE_FIGURES", "1" if tree_RUN_OUTPUT else "0"
        ) == "1"
    tree_helper_run_native_DETERMINISTIC_ARTIFACTS = (
        "fitted.pt",
        "prepared.pt",
        "evaluated.pt",
        "diagnostics.pt",
        "reference.pt",
        "play.pt",
    )
    tree_helper_run_native_TIMING_ARTIFACT = "timings.pt"
    OUTPUT_ROOT = Path(
            os.environ.get("ARENA_TREESHAP_OUTPUT_DIRECTORY", canonical_dir / "artifacts" / "tree")
        )
    NATIVE_DIR = OUTPUT_ROOT / "native"
    tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_ATOL = 2e-5
    tree_helper_treeshap_reference_COMPILED_INTERVENTIONAL_RTOL = 2e-6
    tree_helper_treeshap_reference_EXACT_KEYS = tuple(f"{family}_clean_0" for family in ("tree", "forest", "boost"))
    tree_helper_treeshap_reference_MODEL_KEYS = tuple(
            f"{family}_{target}_{seed_index}"
            for family in ("tree", "forest", "boost")
            for seed_index in range(3)
            for target in ("clean", "shuffled")
        )
    tree_helper_treeshap_reference_ATOL = tree_helper_treeshap_reference_RTOL = 1e-8
    tree_helper_treeshap_reference_ADDENDUM_SHA256 = "2e639c56060d45131a093caeb838a8896399a2d6e89cba9fe477bcd9b816023e"
    tree_helper_treeshap_reference_HERE = tree_root_dir / 'chapter16_shapley_attribution_baselines' / "exercises" / 'part13_treeshap'
    deep_root_dir = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / deep_chapter).exists())
    deep_exercises_dir = deep_root_dir / deep_chapter / "exercises"
    deep_section_dir = deep_exercises_dir / deep_section
    deep_FEATURE_NAMES = np.array([
            "Cement", "Slag", "Fly ash", "Water", "Superplasticizer",
            "Coarse aggregate", "Fine aggregate", "Age",
        ])
    deep_RUN_OUTPUT = os.environ.get("ARENA_DEEPSHAP_OUTPUT_DIRECTORY")
    deep_FIGURE_DIR = Path(
            os.environ.get("ARENA_DEEPSHAP_FIGURE_DIR", deep_RUN_OUTPUT or canonical_dir / "media" / 'deep')
        )
    deep_SAVE_FIGURES = os.environ.get(
            "ARENA_DEEPSHAP_SAVE_FIGURES", "1" if deep_RUN_OUTPUT else "0"
        ) == "1"
    deep_helper_deepshap_diagnostics_DTYPE = torch.float64
    deep_FEATURES = 8
    deep_HIDDEN = 32
    deep_TRAINING_STEPS = 800
    deep_LEARNING_RATE = 0.003
    deep_WEIGHT_DECAY = 0.0001
    deep_RIDGE_PENALTY = 0.01
    deep_MODEL_SEEDS = (161410, 161411, 161412)
    deep_SHUFFLE_SEEDS = (161420, 161421, 161422)
    PRIMARY_BACKGROUND_SEED = 161460
    INDEPENDENT_BACKGROUND_SEEDS = (161461, 161462)
    deep_RANDOM_ORDER_SEED = 161480
    deep_MODEL_KEYS = tuple(
            [f"clean_{index}" for index in range(3)]
            + [f"shuffled_{index}" for index in range(3)]
            + ["ridge"]
        )
    deep_FEATURE_NAMES = (
        "Cement",
        "Slag",
        "Fly ash",
        "Water",
        "Superplasticizer",
        "Coarse aggregate",
        "Fine aggregate",
        "Age",
    )
    deep_MODEL_LABELS = {
            **{f"clean_{index}": f"clean MLP seed {seed}" for index, seed in enumerate(deep_MODEL_SEEDS)},
            **{
                f"shuffled_{index}": f"shuffled MLP seed {seed}"
                for index, seed in enumerate(deep_MODEL_SEEDS)
            },
            "ridge": "ridge control",
        }
    deep_METHOD_NAMES = (
        "rescale_multipliers",
        "background_deepshap",
        "coalition_values",
        "exact_shapley",
        "gradient_delta",
        "coalition_masks",
        "ranked_replacement_curves",
    )
    deep_UNITS = {
        "input": "source feature units",
        "target": "MPa",
        "prediction": "MPa",
        "base": "MPa",
        "attribution": "MPa",
        "game": "MPa",
        "squared_error": "MPa squared",
    }
    deep_helper_deepshap_study_FEATURES = 8
    deep_helper_deepshap_study_METHOD_NAMES = (
        "rescale_multipliers",
        "background_deepshap",
        "coalition_values",
        "exact_shapley",
        "gradient_delta",
        "coalition_masks",
        "ranked_replacement_curves",
    )
    deep_helper_deepshap_study_HIDDEN = 32
    deep_helper_deepshap_study_MODEL_KEYS = tuple(
            [f"clean_{index}" for index in range(3)]
            + [f"shuffled_{index}" for index in range(3)]
            + ["ridge"]
        )
    deep_helper_run_native_DETERMINISTIC_ARTIFACTS = (
        "fitted.pt",
        "prepared.pt",
        "evaluated.pt",
        "diagnostics.pt",
        "reference.pt",
        "play.pt",
    )
    deep_helper_run_native_TIMING_ARTIFACT = "timings.pt"
    deep_helper_deepshap_reference_AUTHOR_ATOL = 1e-7
    deep_helper_deepshap_reference_AUTHOR_RTOL = 1e-8
    deep_helper_deepshap_reference_BACKGROUND_ROWS = 64
    deep_helper_deepshap_reference_EXACT_ATOL = 1e-10
    deep_helper_deepshap_reference_EXACT_RTOL = 1e-10
    deep_helper_deepshap_reference_MODEL_KEYS = (
        "clean_0",
        "clean_1",
        "clean_2",
        "shuffled_0",
        "shuffled_1",
        "shuffled_2",
        "ridge",
    )
    deep_helper_deepshap_reference_FEATURES = 8
    deep_helper_deepshap_reference_HIDDEN = 32
    deep_helper_deepshap_reference_HERE = deep_root_dir / 'chapter16_shapley_attribution_baselines' / "exercises" / 'part14_deepshap'
    deep_helper_deepshap_reference_PRIMARY_BACKGROUND_SEED = 161460
    gradient_root_dir = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / gradient_chapter).exists())
    gradient_exercises_dir = gradient_root_dir / gradient_chapter / "exercises"
    gradient_section_dir = gradient_exercises_dir / gradient_section
    gradient_FEATURE_NAMES = np.array([
            "Cement", "Slag", "Fly ash", "Water", "Superplasticizer",
            "Coarse aggregate", "Fine aggregate", "Age",
        ])
    gradient_RUN_OUTPUT = os.environ.get("ARENA_GRADIENTSHAP_OUTPUT_DIRECTORY")
    gradient_FIGURE_DIR = Path(
            os.environ.get("ARENA_GRADIENTSHAP_FIGURE_DIR", gradient_RUN_OUTPUT or canonical_dir / "media" / 'gradient')
        )
    gradient_SAVE_FIGURES = os.environ.get(
            "ARENA_GRADIENTSHAP_SAVE_FIGURES", "1" if gradient_RUN_OUTPUT else "0"
        ) == "1"
    gradient_DTYPE = t.float64
    gradient_helper_gradientshap_study_FEATURES = 8
    gradient_helper_gradientshap_study_FORWARD_BATCH = 32_768
    gradient_helper_gradientshap_study_MODEL_KEYS = (
        "clean_0",
        "clean_1",
        "clean_2",
        "shuffled_0",
        "shuffled_1",
        "shuffled_2",
        "ridge",
    )
    gradient_helper_gradientshap_study_RANDOM_ORDER_SEED = 161480
    gradient_helper_gradientshap_study_HIDDEN = 32
    gradient_FEATURES = 8
    gradient_HIDDEN = 32
    gradient_TRAINING_STEPS = 800
    gradient_LEARNING_RATE = 0.003
    gradient_WEIGHT_DECAY = 0.0001
    gradient_RIDGE_PENALTY = 0.01
    gradient_MODEL_KEYS = (
        "clean_0",
        "clean_1",
        "clean_2",
        "shuffled_0",
        "shuffled_1",
        "shuffled_2",
        "ridge",
    )
    gradient_MODEL_SEEDS = (161410, 161411, 161412)
    gradient_SHUFFLE_SEEDS = (161420, 161421, 161422)
    BACKGROUND_SPECS = {
        "primary64": (161460, 64),
        "nested16": (161460, 16),
        "nested128": (161460, 128),
        "independent64_a": (161461, 64),
        "independent64_b": (161462, 64),
    }
    BUDGETS = (8, 32, 128, 512, 2048)
    QUADRATURE_NODES = (256, 512)
    NOISE_SIGMAS = (0.0, 0.02, 0.10)
    PRIMARY_SCHEDULE_SEED = 161510
    REPLICATE_SCHEDULE_SEED = 161511
    NOISE_SCHEDULE_SEED = 161512
    REFERENCE_SEEDS = {"shap": 161513, "captum_numpy": 161514, "captum_torch": 161515}
    gradient_RANDOM_ORDER_SEED = 161480
    REPLICATES = 64
    REPLICATE_MODELS = ("clean_0", "shuffled_0", "ridge")
    PATH_DISTANCE_SAMPLES = 128
    COALITION_DISTANCE_DONORS = 8
    FORWARD_BATCH = 32_768
    DISTANCE_BATCH = 1_024
    gradient_FEATURE_NAMES = (
        "Cement",
        "Slag",
        "Fly ash",
        "Water",
        "Superplasticizer",
        "Coarse aggregate",
        "Fine aggregate",
        "Age",
    )
    gradient_MODEL_LABELS = {
            **{f"clean_{i}": f"clean MLP seed {seed}" for i, seed in enumerate(gradient_MODEL_SEEDS)},
            **{
                f"shuffled_{i}": f"shuffled MLP seed {seed}"
                for i, seed in enumerate(gradient_MODEL_SEEDS)
            },
            "ridge": "ridge control",
        }
    gradient_UNITS = {
        "input": "source feature units",
        "target": "MPa",
        "prediction": "MPa",
        "base": "MPa",
        "attribution": "MPa",
        "game": "MPa",
        "squared_error": "MPa squared",
        "distance": "train-standardized Euclidean distance",
    }
    gradient_METHOD_NAMES = (
        "sample_path_schedule",
        "interpolate_paths",
        "input_gradients",
        "expected_gradient_samples",
        "summarize_expected_gradients",
        "expected_gradients",
        "gauss_legendre_pairs",
        "noisy_expected_gradient_samples",
        "ranked_coalition_curves",
    )
    DEEP_METHOD_NAMES = (
        "rescale_multipliers",
        "background_deepshap",
        "coalition_masks",
        "coalition_values",
        "exact_shapley",
        "gradient_delta",
    )
    gradient_helper_run_native_EXTERNAL_DETERMINISTIC_ARTIFACT = "replicate_samples.pt"
    gradient_helper_run_native_SMALL_DETERMINISTIC_ARTIFACTS = (
        "fitted.pt",
        "prepared.pt",
        "evaluated.pt",
        "reference.pt",
        "diagnostics.pt",
        "play.pt",
    )
    gradient_helper_run_native_DETERMINISTIC_ARTIFACTS = gradient_helper_run_native_SMALL_DETERMINISTIC_ARTIFACTS + (
            gradient_helper_run_native_EXTERNAL_DETERMINISTIC_ARTIFACT,
        )
    gradient_helper_run_native_REPLICATE_SHAPE = (64, 3, 64, 2048, 8)
    gradient_helper_run_native_REPLICATE_ELEMENTS = math.prod(gradient_helper_run_native_REPLICATE_SHAPE)
    gradient_helper_run_native_TIMING_ARTIFACT = "timings.pt"
    gradient_helper_run_native_HASH_CHUNK_BYTES = 64 * 1024 * 1024
    gradient_helper_run_native_BUDGETS = (8, 32, 128, 512, 2048)
    gradient_helper_run_native_REPLICATE_MODELS = ("clean_0", "shuffled_0", "ridge")
    gradient_helper_gradientshap_reference_REFERENCE_SAMPLES = 128
    gradient_helper_gradientshap_reference_ALL_ROWS = 272
    gradient_helper_gradientshap_reference_AUTHOR_ATOL = 1e-10
    gradient_helper_gradientshap_reference_AUTHOR_RTOL = 1e-10
    gradient_helper_gradientshap_reference_BACKGROUND_ROWS = 64
    gradient_helper_gradientshap_reference_CAPTUM_NUMPY_SEED = 161514
    gradient_helper_gradientshap_reference_CAPTUM_TORCH_SEED = 161515
    gradient_helper_gradientshap_reference_CAPTUM_WARNINGS = (
        "Captum 0.9.0 draws all donors, then Torch noise, then all alphas in one full-query call.",
        "Splitting the 272 rows into public Captum calls would change the draw table.",
        "author_convergence_delta contains one delta per query/sample pair; it is neither estimator uncertainty nor the averaged-vector completeness residual.",
        "Captum's positive-noise stream is generated from a float32 stdev buffer under the pinned ordinary default-dtype convention.",
    )
    gradient_helper_gradientshap_reference_EXPECTED_VERSIONS = {
        "shap": "0.49.1",
        "captum": "0.9.0",
    }
    gradient_helper_gradientshap_reference_FEATURES = 8
    gradient_helper_gradientshap_reference_MODEL_KEYS = (
        "clean_0",
        "clean_1",
        "clean_2",
        "shuffled_0",
        "shuffled_1",
        "shuffled_2",
        "ridge",
    )
    gradient_helper_gradientshap_reference_SHAP_BATCH_SIZE = 128
    gradient_helper_gradientshap_reference_SHAP_NUMPY_SEED = 161513
    gradient_helper_gradientshap_reference_SHAP_TORCH_SEED = 161515
    gradient_helper_gradientshap_reference_SHAP_WARNINGS = (
        "SHAP 0.49.1 draws one donor then one alpha for each sample and query row; row order is part of the draw identity.",
        "SHAP rseed controls its legacy NumPy stream but does not seed Torch local-smoothing noise.",
        "The scalar-output raw attribution shape is (N,D,1); author_phi is the normalized (N,D) view.",
        "author_shap_variance_proxy is samples.var(ddof=0)/sqrt(K), neither ordinary variance-of-mean nor standard error, and is excluded from parity acceptance.",
        "The adapter temporarily uses the input dtype as Torch's default to repair SHAP's implicit sample-buffer dtype and restores it afterward.",
    )
    gradient_helper_gradientshap_reference_FORWARD_BATCH = 32_768
    gradient_helper_gradientshap_reference_EXPECTED_SOURCE_SHA256 = {
        "shap/explainers/_gradient.py": "ae1d0fe9405e62c9400c655e6d5af792ccc989a7632fd05171a731776551840a",
        "captum/attr/_core/gradient_shap.py": "b2333e8819b82eb18693ae5a8a78c7a9ff1e30986179de2ee97b2820dcbff00c",
        "captum/attr/_core/noise_tunnel.py": "d013c38cdbb4a10d4717a4e38ac063903cacc0b54389a6591a9bcbc4a9fa8606",
        "captum/_utils/common.py": "a1fc8b6752c48a76ffd3305983fed7972c7013ae6764ddb837c873c8eaa4198e",
    }
    initialize_deep_16_rescale_multipliers()
    initialize_deep_19_single_reference()
    initialize_deep_30_background_deepshap()
    initialize_helper_gradient_3_5_gradient_helper_gradientshap_methods__eg_preserve_execution_state()
    initialize_helper_gradient_115_29_gradient_helper_gradientshap_reference__preserve_execution_state()
    initialize_helper_gradient_115_41_gradient_helper_gradientshap_reference_replay_draw_table()
    initialize_helper_gradient_115_42_gradient_helper_gradientshap_reference_captum_same_draw_reference()
    initialize_helper_gradient_115_45_gradient_helper_gradientshap_reference_shap_same_draw_reference()
    tree_persist_completed = tree_helper_run_native_persist_completed
    tree_save_payload = tree_helper_run_native_save_payload
    tree_write_before_audit = tree_helper_run_native_write_before_audit
    tree_write_notebook_summary = tree_helper_run_native_write_notebook_summary
    check_reference = tree_helper_treeshap_reference_check_reference
    _check_widths = deep_helper_deepshap_methods__check_widths
    _epsilon = deep_helper_deepshap_methods__epsilon
    _layers = deep_helper_deepshap_methods__layers
    _method_tensor = deep_helper_deepshap_methods__method_tensor
    _model_output = deep_helper_deepshap_methods__model_output
    deep_DTYPE = deep_helper_deepshap_diagnostics_DTYPE
    deep__linear = deep_helper_deepshap_diagnostics__linear
    deep__max_abs = deep_helper_deepshap_diagnostics__max_abs
    deep__method = deep_helper_deepshap_diagnostics__method
    deep__prediction = deep_helper_deepshap_diagnostics__prediction
    deep__clone_state = deep_helper_deepshap_study__clone_state
    _clone_tree = deep_helper_deepshap_study__clone_tree
    _combine_explanations = deep_helper_deepshap_study__combine_explanations
    _combine_row_blocks = deep_helper_deepshap_study__combine_row_blocks
    deep__float_tensor = deep_helper_deepshap_study__float_tensor
    deep__ids_tensor = deep_helper_deepshap_study__ids_tensor
    deep__normalization_tensors = deep_helper_deepshap_study__normalization_tensors
    deep__output_tensor = deep_helper_deepshap_study__output_tensor
    deep__require_disjoint_ids = deep_helper_deepshap_study__require_disjoint_ids
    _require_methods = deep_helper_deepshap_study__require_methods
    deep__rows = deep_helper_deepshap_study__rows
    deep__state_shapes = deep_helper_deepshap_study__state_shapes
    deep__validate_evaluation_inputs = deep_helper_deepshap_study__validate_evaluation_inputs
    deep__validate_fitted = deep_helper_deepshap_study__validate_fitted
    _validate_rankings = deep_helper_deepshap_study__validate_rankings
    deep_save_payload = deep_helper_run_native_save_payload
    deep_write_before_audit = deep_helper_run_native_write_before_audit
    deep_validate_reference = deep_helper_deepshap_reference_validate_reference
    deep_persist_completed = deep_helper_run_native_persist_completed
    deep_write_notebook_summary = deep_helper_run_native_write_notebook_summary
    _eg_float_tensor = gradient_helper_gradientshap_methods__eg_float_tensor
    _eg_index_tensor = gradient_helper_gradientshap_methods__eg_index_tensor
    _eg_model = gradient_helper_gradientshap_methods__eg_model
    _eg_model_output = gradient_helper_gradientshap_methods__eg_model_output
    _eg_positive_int = gradient_helper_gradientshap_methods__eg_positive_int
    _eg_preserve_execution_state = gradient_helper_gradientshap_methods__eg_preserve_execution_state
    _eg_seed_words = gradient_helper_gradientshap_methods__eg_seed_words
    _eg_validate_paths = gradient_helper_gradientshap_methods__eg_validate_paths
    _batched_prediction = gradient_helper_gradientshap_study__batched_prediction
    _checked_curves = gradient_helper_gradientshap_study__checked_curves
    gradient__clone_state = gradient_helper_gradientshap_study__clone_state
    _combine_baselines = gradient_helper_gradientshap_study__combine_baselines
    _combine_mc = gradient_helper_gradientshap_study__combine_mc
    _combine_primary = gradient_helper_gradientshap_study__combine_primary
    _combine_quadrature = gradient_helper_gradientshap_study__combine_quadrature
    gradient__fixed_random_orders = gradient_helper_gradientshap_study__fixed_random_orders
    gradient__float_tensor = gradient_helper_gradientshap_study__float_tensor
    gradient__folding_error = gradient_helper_gradientshap_study__folding_error
    _identity_hash = gradient_helper_gradientshap_study__identity_hash
    gradient__ids_tensor = gradient_helper_gradientshap_study__ids_tensor
    gradient__normalization_tensors = gradient_helper_gradientshap_study__normalization_tensors
    gradient__output_tensor = gradient_helper_gradientshap_study__output_tensor
    gradient__require_disjoint_ids = gradient_helper_gradientshap_study__require_disjoint_ids
    _require_namespace = gradient_helper_gradientshap_study__require_namespace
    gradient__rows = gradient_helper_gradientshap_study__rows
    _seed_words = gradient_helper_gradientshap_study__seed_words
    gradient__state_shapes = gradient_helper_gradientshap_study__state_shapes
    gradient__validate_evaluation_inputs = gradient_helper_gradientshap_study__validate_evaluation_inputs
    gradient__validate_fitted = gradient_helper_gradientshap_study__validate_fitted
    gradient_persist_completed = gradient_helper_run_native_persist_completed
    persist_prepared = gradient_helper_run_native_persist_prepared
    gradient_write_notebook_summary = gradient_helper_run_native_write_notebook_summary
    REFERENCE_SAMPLES = gradient_helper_gradientshap_reference_REFERENCE_SAMPLES
    gradient_validate_reference = gradient_helper_gradientshap_reference_validate_reference
    deep_curriculum_methods = SimpleNamespace(
            paired_forward=paired_forward,
            rescale_slope=rescale_slope,
            rescale_multipliers=rescale_multipliers,
            single_reference=single_reference,
            background_deepshap=background_deepshap,
            coalition_values=coalition_values,
            exact_shapley=deep_exact_shapley,
            reveal_cancel_groups=reveal_cancel_groups,
            ranked_replacement_curves=ranked_replacement_curves,
            coalition_masks=deep_coalition_masks,
            gradient_delta=gradient_delta,
        )
    gradient_methods = SimpleNamespace(
            sample_path_schedule=sample_path_schedule,
            interpolate_paths=interpolate_paths,
            input_gradients=input_gradients,
            expected_gradient_samples=expected_gradient_samples,
            summarize_expected_gradients=summarize_expected_gradients,
            expected_gradients=expected_gradients,
            gauss_legendre_pairs=gauss_legendre_pairs,
            noisy_expected_gradient_samples=noisy_expected_gradient_samples,
            ranked_coalition_curves=ranked_coalition_curves,
        )
    deep_methods = SimpleNamespace(**deep_module_deepshap_methods_namespace())
    refresh_tree_default_methods()

# %%
