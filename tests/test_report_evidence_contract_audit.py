from scripts.audit_report_evidence_contracts import (
    report_freshness_blockers,
    report_contract_blockers,
    report_evidence_blockers,
    verification_report_records,
)
from arena_ext.verification import build_report_input_manifest, verify_report_input_manifest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _valid_lock_report():
    lock = {
        "notebook_id": "section_x",
        "gt_tier": "GT-1",
        "evidence_level": "real_cuda_preflight",
        "claim_scope": "A concrete CUDA preflight claim with baselines, controls, models, datasets, and narrow scope.",
        "controls": ["baseline_metric", "random_direction_control"],
        "expected_metrics": {"accuracy_min": 0.9},
    }
    report = {
        "notebook_id": "section_x",
        "gt_tier": "GT-1",
        "evidence_level": "real_cuda_preflight",
        "claim_scope": lock["claim_scope"],
        "tests_passed": True,
        "accepted": True,
        "known_failures": [],
        "report_inputs": build_report_input_manifest(ROOT, [ROOT / "pyproject.toml"]),
        "metrics": {
            "gpu_evidence": {
                "category": "cuda_section_metric",
                "uses_cuda": True,
                "placeholder_only": False,
                "section_specific_metric_keys": ["accuracy"],
            },
            "gpu_test": {"accuracy": 0.95},
        },
        "baselines": {
            "declared_controls": ["baseline_metric", "random_direction_control"],
            "expected_metrics": {"accuracy_min": 0.9},
        },
        "negative_controls": {"declared_controls": ["random_direction_control"]},
        "ood_tests": {"declared": "required_where_applicable"},
        "models": [{"id": "tiny-model"}],
        "datasets": [{"id": "generated"}],
        "safety_notes": ["No unsafe prompts or raw gated weights."],
    }
    return lock, report


def test_report_contract_accepts_complete_report():
    lock, report = _valid_lock_report()

    assert report_contract_blockers(lock, report) == []


def test_report_contract_flags_placeholder_report():
    lock, report = _valid_lock_report()
    lock["evidence_level"] = "notebook_contract"
    report["evidence_level"] = "notebook_contract"
    report["metrics"]["gpu_evidence"] = {
        "category": "placeholder_only",
        "uses_cuda": False,
        "placeholder_only": True,
        "section_specific_metric_keys": [],
    }
    report["negative_controls"] = {}
    report["known_failures"] = ["metric failed"]

    blockers = report_contract_blockers(lock, report)

    assert "section_x: placeholder evidence_level=notebook_contract" in blockers
    assert "section_x: gpu_evidence.category='placeholder_only'" in blockers
    assert "section_x: gpu_evidence.uses_cuda is not true" in blockers
    assert "section_x: gpu_evidence is placeholder_only" in blockers
    assert "section_x: negative_controls are empty" in blockers
    assert "section_x: accepted report has known_failures" in blockers


def test_report_contract_flags_missing_report_inputs():
    lock, report = _valid_lock_report()
    report.pop("report_inputs")

    blockers = report_contract_blockers(lock, report)

    assert "section_x: report_inputs.files must be a nonempty list" in blockers


def test_report_input_manifest_detects_changed_file(tmp_path):
    source = tmp_path / "input.py"
    source.write_text("value = 1\n")
    manifest = build_report_input_manifest(tmp_path, [source])
    source.write_text("value = 2\n")

    blockers = verify_report_input_manifest(tmp_path, manifest)

    assert "report input hash changed: input.py" in blockers
    assert "report_inputs.combined_sha256 does not match current inputs" in blockers


def test_report_freshness_blockers_prefixes_notebook_id():
    blockers = report_freshness_blockers({"notebook_id": "section_x"})

    assert "section_x: report_inputs.files must be a nonempty list" in blockers


def test_report_contract_flags_expected_metric_regression():
    lock, report = _valid_lock_report()
    report["metrics"]["gpu_test"]["accuracy"] = 0.5

    blockers = report_contract_blockers(lock, report)

    assert any("expected gpu_test.accuracy >= 0.9" in blocker for blocker in blockers)


def test_current_report_evidence_contracts_pass():
    assert report_evidence_blockers() == []


def test_report_record_discovery_includes_docs_evidence_reports():
    report_paths = {record["report_path"].as_posix() for record in verification_report_records()}

    assert any(path.endswith("docs/evidence/ioi_path_patching/verification_report.json") for path in report_paths)
    assert any(path.endswith("docs/evidence/othello_gpt/verification_report.json") for path in report_paths)


def test_cpu_native_model_contract_requires_explicit_device_and_actual_fresh_results():
    import copy
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence
    lock, report = _valid_lock_report()
    lock.update(execution_device='cpu',exercise_metadata={'REQUIRES_GPU':False})
    report['metrics']['gpu_evidence'].update(category='cpu_or_budget_metric',uses_cuda=False)
    report['metrics']['gpu_test'].update(device='cpu',native_model_reloaded=True,actual_learner_passed=True,
        fresh_reproduction_passed=True,own_answer_code_cells=20,fresh_compared_examples=5184,
        fresh_condition_batches=108,fresh_top_token_mismatches=0,fresh_max_logit_error=0.,
        learner_evidence_sha256='bound-by-input-manifest',fresh_evidence_sha256='also-bound-by-input-manifest')
    assert has_native_cpu_model_evidence(lock,report)
    assert report_contract_blockers(lock,report)==[]
    # CPU budget, host information, or a completion flag alone is insufficient.
    mutations={'native_model_reloaded':False,'actual_learner_passed':False,'fresh_reproduction_passed':False,
               'fresh_compared_examples':0,'fresh_condition_batches':0,'fresh_top_token_mismatches':1,
               'fresh_max_logit_error':float('nan'),'learner_evidence_sha256':'','fresh_evidence_sha256':''}
    for key,value in mutations.items():
        broken=copy.deepcopy(report);broken['metrics']['gpu_test'][key]=value
        assert not has_native_cpu_model_evidence(lock,broken), key
        assert report_contract_blockers(lock,broken), key
    for bad_lock in [{**lock,'execution_device':'cuda'},{**lock,'exercise_metadata':{'REQUIRES_GPU':True}}]:
        assert not has_native_cpu_model_evidence(bad_lock,report)
def test_cpu_scratch_model_requires_held_out_refits_and_full_numerical_evidence():
    from copy import deepcopy
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

    lock = {'execution_device': 'cpu', 'exercise_metadata': {'REQUIRES_GPU': False}}
    measured = dict(
        device='cpu', native_model_retrained=True, native_model_retrained_in_this_report=True,
        native_evidence_reused=False, actual_learner_passed=True, fresh_reproduction_passed=True,
        own_answer_code_cells=40, final_query_count=64, final_used_for_selection=False,
        reference_refits=930, independent_scipy_checks=10, fresh_compared_tensors=1048,
        fresh_compared_elements=16726855, fresh_max_error=0.,
        learner_evidence_sha256='learner-evidence', fresh_evidence_sha256='fresh-evidence',
    )
    report = {'metrics': {'gpu_evidence': {'category': 'cpu_or_budget_metric', 'uses_cuda': False},
                          'gpu_test': measured}}
    assert has_native_cpu_model_evidence(lock, report)
    for key, value in [
        ('native_model_retrained', False), ('native_model_retrained_in_this_report', False),
        ('native_evidence_reused', True), ('actual_learner_passed', False),
        ('fresh_reproduction_passed', False), ('final_query_count', 1),
        ('final_used_for_selection', True), ('reference_refits', 0),
        ('independent_scipy_checks', 0), ('fresh_compared_tensors', 0),
        ('fresh_compared_elements', 0), ('fresh_max_error', 1e-3),
        ('fresh_max_error', float('nan')), ('learner_evidence_sha256', ''),
        ('fresh_evidence_sha256', ''),
    ]:
        broken = deepcopy(report)
        broken['metrics']['gpu_test'][key] = value
        assert not has_native_cpu_model_evidence(lock, broken), key
    assert not has_native_cpu_model_evidence(
        {**lock, 'exercise_metadata': {'REQUIRES_GPU': True}}, report,
    )


def test_datamodel_evidence_distinguishes_heldout_subsets_from_query_targets():
    from copy import deepcopy
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

    lock = {'execution_device': 'cpu', 'exercise_metadata': {'REQUIRES_GPU': False}}
    measured = dict(
        device='cpu', native_model_retrained=True, native_model_retrained_in_this_report=True,
        native_evidence_reused=False, actual_learner_passed=True, fresh_reproduction_passed=True,
        own_answer_code_cells=31, reference_refits=10753, independent_scipy_checks=10,
        independent_lasso_checks=12, fresh_compared_tensors=263, fresh_compared_elements=5546159,
        fresh_max_error=0., learner_evidence_sha256='bound-input', fresh_evidence_sha256='bound-input',
        heldout_unit='training_subset_model', heldout_model_count=768,
        test_responses_used_for_selection=False, query_image_count=64,
        query_images_used_as_regression_targets=True, query_images_excluded_from_classifier_training=True,
    )
    report = {'metrics': {'gpu_evidence': {'category': 'cpu_or_budget_metric', 'uses_cuda': False},
                          'gpu_test': measured}}
    assert has_native_cpu_model_evidence(lock, report)
    assert 'final_used_for_selection' not in measured
    for key, value in [
        ('native_model_retrained', False), ('native_model_retrained_in_this_report', False),
        ('native_evidence_reused', True), ('heldout_unit', 'image'), ('heldout_model_count', 0),
        ('test_responses_used_for_selection', True), ('query_image_count', 1),
        ('query_images_used_as_regression_targets', False),
        ('query_images_excluded_from_classifier_training', False),
        ('independent_lasso_checks', 0), ('reference_refits', 64), ('fresh_compared_elements', 64),
        ('fresh_max_error', float('nan')), ('fresh_max_error', 1e-3),
    ]:
        broken = deepcopy(report)
        broken['metrics']['gpu_test'][key] = value
        assert not has_native_cpu_model_evidence(lock, broken), key


def test_cartography_evidence_requires_independent_sgd_paths_and_real_selection_controls():
    from copy import deepcopy
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

    lock = {'execution_device': 'cpu', 'exercise_metadata': {'REQUIRES_GPU': False}}
    measured = dict(
        device='cpu', native_model_retrained=True, native_model_retrained_in_this_report=True,
        native_evidence_reused=False, actual_learner_passed=True, fresh_reproduction_passed=True,
        own_answer_code_cells=30, reference_refits=140, mapping_runs=20, selection_refits=120,
        independent_optimizer_checks=4, independent_numpy_checks=24, separate_training_checks=2,
        optimizer_updates=53550, fresh_compared_tensors=675, fresh_compared_elements=41903198,
        fresh_max_error=0., learner_evidence_sha256='bound-input', fresh_evidence_sha256='bound-input',
        heldout_unit='evaluation_image', heldout_image_count=2620, heldout_split_count=2,
        final_used_for_selection=False, heldout_images_excluded_from_training=True,
        both_logging_conventions_recorded=True, all_retraining_seeds_retained=True,
        class_matched_controls=True, equal_update_control=True,
    )
    report = {'metrics': {'gpu_evidence': {'category': 'cpu_or_budget_metric', 'uses_cuda': False},
                          'gpu_test': measured}}
    assert has_native_cpu_model_evidence(lock, report)
    assert 'independent_scipy_checks' not in measured
    for key, value in [
        ('native_model_retrained', False), ('native_evidence_reused', True),
        ('heldout_unit', 'training_example'), ('heldout_image_count', 0),
        ('heldout_split_count', 1), ('final_used_for_selection', True),
        ('heldout_images_excluded_from_training', False), ('independent_optimizer_checks', 0),
        ('independent_numpy_checks', 0), ('separate_training_checks', 0),
        ('mapping_runs', 0), ('selection_refits', 0), ('reference_refits', 1),
        ('optimizer_updates', 0), ('both_logging_conventions_recorded', False),
        ('all_retraining_seeds_retained', False), ('class_matched_controls', False),
        ('equal_update_control', False), ('fresh_compared_elements', 3000),
        ('fresh_max_error', float('nan')), ('fresh_reproduction_passed', False),
    ]:
        broken = deepcopy(report)
        broken['metrics']['gpu_test'][key] = value
        assert not has_native_cpu_model_evidence(lock, broken), key


def test_sage_native_evidence_requires_recipe_audit_and_actual_game_references():
    import json
    from copy import deepcopy
    from pathlib import Path
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

    here = Path(__file__).resolve().parents[1] / 'chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance'
    report = json.loads((here / 'verification_report.json').read_text())
    lock = {'execution_device': 'cpu', 'exercise_metadata': {'REQUIRES_GPU': False}}
    assert has_native_cpu_model_evidence(lock, report)
    for key, value in [
        ('native_model_retrained_in_this_report',False), ('native_evidence_reused',True),
        ('heldout_unit','row'), ('heldout_group_count',0), ('heldout_row_count',0),
        ('explanation_row_count',0), ('final_used_for_selection',True), ('cross_split_recipe_overlap',1),
        ('all_training_seeds_retained',False), ('shuffled_training_targets',False),
        ('neural_training_runs',0), ('optimizer_updates',0), ('random_order_controls',0),
        ('exact_coalitions',0), ('direct_sampled_paths',0), ('independent_permutations',0),
        ('independent_weighted_regression_coalitions',0), ('independent_numpy_predictions',0),
        ('author_imputation_comparisons',0), ('local_global_max_error',1.),
        ('direct_path_max_error',float('nan')), ('ignored_feature_importance',1.),
    ]:
        broken = deepcopy(report)
        broken['metrics']['gpu_test'][key] = value
        assert not has_native_cpu_model_evidence(lock, broken), key


def test_treeshap_evidence_requires_fresh_fitted_trees_full_parity_and_controls():
    from copy import deepcopy
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

    lock, report = _valid_lock_report()
    lock.update(execution_device='cpu', exercise_metadata={'REQUIRES_GPU': False})
    measured = dict(
        device='cpu', native_model_retrained=True, native_model_retrained_in_this_report=True,
        native_evidence_reused=False, actual_learner_passed=True, fresh_reproduction_passed=True,
        own_answer_code_cells=30, fresh_compared_arrays=100,
        fresh_compared_elements=1055546, fresh_max_error=0.,
        learner_evidence_sha256='bound-input', fresh_evidence_sha256='bound-input',
        model_fits=18, fresh_model_fits=18, component_trees=486,
        model_families=['tree', 'forest', 'boost'], training_seeds=[161310, 161311, 161312],
        clean_fits=9, shuffled_fits=9, all_model_fits_retained=True,
        all_training_seeds_retained=True, shuffled_training_targets=True,
        shuffled_models_evaluated=9, heldout_unit='recipe_group', train_row_count=758,
        explanation_row_count=153, heldout_row_count=119, train_group_count=298,
        explanation_group_count=64, heldout_group_count=65, cross_split_recipe_overlap=0,
        final_used_for_selection=False, rows_explained_per_model=272,
        model_row_explanations=4896, shap_parity_rows_per_model=272,
        shap_parity_model_rows=4896, exact_models=3, exact_games=2,
        exact_coalitions=256, exact_model_rows=816, exact_game_model_rows=1632,
        path_reference_passed=True, path_reference_model_rows=4896,
        path_reference_max_error=0., raw_interventional_reference_passed=False,
        raw_interventional_reference_disclosed=True,
        raw_interventional_reference_model_rows=4896,
        raw_interventional_reference_affected_rows=2,
        raw_interventional_reference_max_error=0.49,
        corrected_interventional_reference_passed=True,
        corrected_interventional_reference_model_rows=4896,
        corrected_interventional_reference_max_error=2e-6,
        corrected_thresholds=941, raw_source_route_disagreements=217,
        corrected_source_route_disagreements=0,
        exhaustive_path_max_error=0., exhaustive_interventional_max_error=0.,
        prediction_parity_max_error=0., base_value_parity_max_error=2e-6,
        deletion_models=3, deletion_audit_rows=119,
        deletion_uses_exact_interventional_game=True, random_order_controls=100,
        background_variants=4, background_rows_compared=272,
        alternate_backgrounds_compared=True, saabas_model_rows=4896,
        saabas_deletion_models=3, saabas_differs_from_shap=True,
        saabas_additivity_max_error=0., saabas_attribution_gap_max=0.1,
        training_mean_baseline=True, dummy_feature_max_abs=0., embedded_figures=16,
    )
    report['metrics']['gpu_evidence'].update(category='cpu_or_budget_metric', uses_cuda=False)
    report['metrics']['gpu_test'].update(measured)
    assert has_native_cpu_model_evidence(lock, report)
    assert report_contract_blockers(lock, report) == []
    # TreeSHAP must not inherit SAGE's neural optimizer or sampled-path fields.
    assert not {'neural_training_runs', 'optimizer_updates', 'direct_sampled_paths'} & measured.keys()

    mutations = [
        ('native_model_retrained_in_this_report', False), ('native_evidence_reused', True),
        ('actual_learner_passed', False), ('fresh_reproduction_passed', False),
        ('fresh_compared_arrays', 0), ('fresh_max_error', float('nan')),
        ('model_fits', 17), ('fresh_model_fits', 17), ('component_trees', 485),
        ('model_families', ['tree', 'forest']), ('training_seeds', [161310, 161311]),
        ('clean_fits', 8), ('shuffled_fits', 8), ('all_model_fits_retained', False),
        ('all_training_seeds_retained', False), ('shuffled_training_targets', False),
        ('shuffled_models_evaluated', 8), ('heldout_unit', 'row'), ('train_row_count', 757),
        ('explanation_row_count', 152), ('heldout_row_count', 118), ('train_group_count', 297),
        ('explanation_group_count', 63), ('heldout_group_count', 64),
        ('cross_split_recipe_overlap', 1), ('final_used_for_selection', True),
        ('rows_explained_per_model', 271), ('model_row_explanations', 4895),
        ('shap_parity_rows_per_model', 271), ('shap_parity_model_rows', 4895),
        ('exact_models', 2), ('exact_games', 1), ('exact_coalitions', 255),
        ('exact_model_rows', 815), ('exact_game_model_rows', 1631),
        ('path_reference_passed', False), ('path_reference_model_rows', 4895),
        ('path_reference_max_error', 1e-5), ('raw_interventional_reference_passed', True),
        ('raw_interventional_reference_disclosed', False),
        ('raw_interventional_reference_model_rows', 4895),
        ('raw_interventional_reference_affected_rows', 0),
        ('raw_interventional_reference_max_error', 0.),
        ('corrected_interventional_reference_passed', False),
        ('corrected_interventional_reference_model_rows', 4895),
        ('corrected_interventional_reference_max_error', 1e-3),
        ('corrected_thresholds', 940), ('raw_source_route_disagreements', 216),
        ('corrected_source_route_disagreements', 1),
        ('exhaustive_path_max_error', 1e-5), ('exhaustive_interventional_max_error', float('nan')),
        ('prediction_parity_max_error', 1e-5), ('base_value_parity_max_error', 1e-3),
        ('deletion_models', 2), ('deletion_audit_rows', 118),
        ('deletion_uses_exact_interventional_game', False), ('random_order_controls', 99),
        ('background_variants', 3), ('background_rows_compared', 271),
        ('alternate_backgrounds_compared', False), ('saabas_model_rows', 4895),
        ('saabas_deletion_models', 2), ('saabas_differs_from_shap', False),
        ('saabas_additivity_max_error', 1e-5), ('saabas_attribution_gap_max', 0.),
        ('training_mean_baseline', False), ('dummy_feature_max_abs', 1e-4),
        ('embedded_figures', 15), ('fresh_compared_elements', 1055545),
        ('learner_evidence_sha256', ''), ('fresh_evidence_sha256', ''),
    ]
    for key, value in mutations:
        broken = deepcopy(report)
        broken['metrics']['gpu_test'][key] = value
        assert not has_native_cpu_model_evidence(lock, broken), key
        assert report_contract_blockers(lock, broken), key


def test_deepshap_evidence_requires_full_neural_games_references_and_failures():
    from copy import deepcopy

    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

    lock, report = _valid_lock_report()
    lock.update(execution_device="cpu", exercise_metadata={"REQUIRES_GPU": False})
    measured = dict(
        device="cpu",
        native_model_retrained=True,
        native_model_retrained_in_this_report=True,
        native_evidence_reused=False,
        actual_learner_passed=True,
        fresh_reproduction_passed=True,
        own_answer_code_cells=50,
        fresh_compared_arrays=842,
        fresh_compared_elements=5_887_125,
        fresh_max_error=0.0,
        learner_evidence_sha256="bound-input",
        fresh_evidence_sha256="bound-input",
        model_fits=7,
        fresh_model_fits=7,
        all_model_fits_retained=True,
        mlp_fits=6,
        neural_training_runs=6,
        ridge_fits=1,
        clean_fits=3,
        shuffled_fits=3,
        shuffled_models_evaluated=3,
        optimizer_updates=4800,
        model_keys=[
            "clean_0",
            "clean_1",
            "clean_2",
            "shuffled_0",
            "shuffled_1",
            "shuffled_2",
            "ridge",
        ],
        training_seeds=[161410, 161411, 161412],
        all_training_seeds_retained=True,
        shuffled_training_targets=True,
        train_row_count=758,
        train_group_count=298,
        explanation_group_count=64,
        heldout_unit="recipe_group",
        heldout_group_count=65,
        heldout_row_count=119,
        explanation_row_count=153,
        rows_explained_per_model=272,
        model_row_explanations=1904,
        final_used_for_selection=False,
        cross_split_recipe_overlap=0,
        exact_models=7,
        exact_coalitions=256,
        exact_model_rows=1904,
        model_coalition_values=487424,
        primary_hybrid_predictions=31195136,
        attribution_coordinates_per_method=15232,
        pair_model_rows=121856,
        pair_local_values=974848,
        background_rows=64,
        folding_max_error=1e-13,
        author_reference_passed=True,
        author_reference_model_rows=1904,
        author_reference_local_values=15232,
        author_pair_rows_checked=896,
        author_pair_local_values_checked=7168,
        author_reference_atol=1e-7,
        author_reference_rtol=1e-8,
        author_reference_max_error=7e-13,
        real_local_accuracy_passed=True,
        real_local_accuracy_atol=1e-7,
        real_local_accuracy_rtol=1e-8,
        real_local_accuracy_max_error=3e-13,
        exact_efficiency_max_error=1e-13,
        deepshap_is_exact_shap=False,
        deepshap_approximation_disclosed=True,
        deepshap_exact_gap_max=51.0,
        ridge_exact_max_error=1e-13,
        nearzero_primary_local_accuracy_passed=False,
        nearzero_primary_failure_disclosed=True,
        nearzero_primary_residual=4e-7,
        nearzero_epsilon_zero_max_error=0.0,
        threshold_exact_max_error=0.0,
        min_exact_max_error=0.0,
        background_variants=5,
        background_rows_compared=272,
        alternate_backgrounds_compared=True,
        replacement_models=7,
        replacement_audit_rows=119,
        random_order_controls=100,
        gradient_baseline_model_rows=1904,
        training_mean_baseline=True,
        dummy_feature_max_abs=0.0,
        unsupported_operator_rejected=True,
        whole_paper_replication=False,
        embedded_figures=17,
    )
    report["metrics"]["gpu_evidence"].update(
        category="cpu_or_budget_metric", uses_cuda=False
    )
    report["metrics"]["gpu_test"].update(measured)
    assert has_native_cpu_model_evidence(lock, report)
    assert report_contract_blockers(lock, report) == []

    mutations = [
        ("native_model_retrained_in_this_report", False),
        ("native_evidence_reused", True),
        ("actual_learner_passed", False),
        ("fresh_reproduction_passed", False),
        ("fresh_compared_arrays", 841),
        ("fresh_compared_elements", 5_887_124),
        ("fresh_max_error", float("nan")),
        ("model_fits", 6),
        ("fresh_model_fits", 6),
        ("all_model_fits_retained", False),
        ("mlp_fits", 5),
        ("neural_training_runs", 5),
        ("ridge_fits", 0),
        ("clean_fits", 2),
        ("shuffled_fits", 2),
        ("shuffled_models_evaluated", 2),
        ("optimizer_updates", 4799),
        ("model_keys", ["clean_0", "ridge"]),
        ("training_seeds", [161410, 161411]),
        ("all_training_seeds_retained", False),
        ("shuffled_training_targets", False),
        ("heldout_unit", "row"),
        ("train_row_count", 757),
        ("explanation_row_count", 152),
        ("heldout_row_count", 118),
        ("train_group_count", 297),
        ("explanation_group_count", 63),
        ("heldout_group_count", 64),
        ("cross_split_recipe_overlap", 1),
        ("final_used_for_selection", True),
        ("rows_explained_per_model", 271),
        ("model_row_explanations", 1903),
        ("exact_models", 6),
        ("exact_coalitions", 255),
        ("exact_model_rows", 1903),
        ("model_coalition_values", 487423),
        ("primary_hybrid_predictions", 31195135),
        ("attribution_coordinates_per_method", 15231),
        ("background_rows", 63),
        ("pair_model_rows", 121855),
        ("pair_local_values", 974847),
        ("folding_max_error", 1e-5),
        ("author_reference_passed", False),
        ("author_reference_model_rows", 1903),
        ("author_reference_local_values", 15231),
        ("author_pair_rows_checked", 895),
        ("author_pair_local_values_checked", 7167),
        ("author_reference_max_error", 1e-3),
        ("real_local_accuracy_passed", False),
        ("real_local_accuracy_max_error", 1e-3),
        ("exact_efficiency_max_error", 1e-3),
        ("deepshap_is_exact_shap", True),
        ("deepshap_approximation_disclosed", False),
        ("deepshap_exact_gap_max", 0.0),
        ("ridge_exact_max_error", 1e-3),
        ("nearzero_primary_local_accuracy_passed", True),
        ("nearzero_primary_failure_disclosed", False),
        ("nearzero_primary_residual", 0.0),
        ("nearzero_epsilon_zero_max_error", 1e-3),
        ("threshold_exact_max_error", 1e-3),
        ("min_exact_max_error", 1e-3),
        ("background_variants", 4),
        ("background_rows_compared", 271),
        ("alternate_backgrounds_compared", False),
        ("replacement_models", 6),
        ("replacement_audit_rows", 118),
        ("random_order_controls", 99),
        ("gradient_baseline_model_rows", 1903),
        ("training_mean_baseline", False),
        ("dummy_feature_max_abs", 1e-3),
        ("unsupported_operator_rejected", False),
        ("whole_paper_replication", True),
        ("embedded_figures", 16),
        ("learner_evidence_sha256", ""),
        ("fresh_evidence_sha256", ""),
    ]
    for key, value in mutations:
        broken = deepcopy(report)
        broken["metrics"]["gpu_test"][key] = value
        assert not has_native_cpu_model_evidence(lock, broken), key
        assert report_contract_blockers(lock, broken), key


def _gradientshap_scope_fixture():
    """The prospective lock describes required scope, never a native test result."""
    import yaml

    lock, report = _valid_lock_report()
    part = ROOT / "chapter16_shapley_attribution_baselines/exercises/part15_gradientshap"
    expected = yaml.safe_load((part / "artifacts.lock.yml").read_text())["expected_metrics"]
    measured = {key.removesuffix("_min").removesuffix("_max"): value for key, value in expected.items()}
    measured.update(
        device="cpu",
        own_answer_code_cells=80,
        learner_evidence_sha256="a" * 64,
        fresh_evidence_sha256="b" * 64,
        replicate_sample_sha256="c" * 64,
        replicate_file_sha256="d" * 64,
        author_shap_max_error=2e-12,
        author_captum_max_error=0.0,
        author_shap_scaled_error=0.02,
        author_captum_scaled_error=0.0,
        author_shap_tolerance_bound=1e-8,
        author_captum_tolerance_bound=1e-8,
    )
    lock.update(execution_device="cpu", exercise_metadata={"REQUIRES_GPU": False})
    report["metrics"]["gpu_evidence"] = {"category": "cpu_or_budget_metric", "uses_cuda": False}
    report["metrics"]["gpu_test"] = measured
    prefix = "chapter16_shapley_attribution_baselines/exercises/part15_gradientshap/expected_outputs/"
    report["report_inputs"]["files"] = [
        {"path": prefix + "learner_execution.json", "sha256": "a" * 64, "size_bytes": 10},
        {"path": prefix + "fresh_reproduction.json", "sha256": "b" * 64, "size_bytes": 10},
    ]
    return lock, report


def test_gradientshap_evidence_requires_complete_draws_and_independent_reproduction():
    import copy
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

    lock, report = _gradientshap_scope_fixture()
    assert has_native_cpu_model_evidence(lock, report), "Complete declared scope must be supported."
    defects = {
        "known_model_state_hashes_matched": 18,
        "model_row_explanations": 119,
        "primary_hybrid_predictions": 487424,
        "replicate_sample_shape": [64, 3, 32, 2048, 8],
        "replicate_sample_coordinates": 100663296,
        "replicate_file_sha256": "missing",
        "replicate_file_size_bytes": 0,
        "author_total_coordinates": 3808,
        "author_reference_passed": False,
        "positive_noise_reference_passed": False,
        "author_reference_rtol": 1e-3,
        "random_order_controls": 0,
        "background_variants": 1,
        "coalition_distance_points": 0,
        "native_evidence_reused": True,
        "fresh_comparison": "aggregate_only",
        "fresh_max_error": 1e-12,
        "author_shap_scaled_error": 1.1,
        "author_captum_max_error": 1000.0,
        "learner_evidence_sha256": "unverified",
        "fresh_evidence_sha256": "e" * 64,
    }
    for key, value in defects.items():
        broken = copy.deepcopy(report)
        broken["metrics"]["gpu_test"][key] = value
        assert not has_native_cpu_model_evidence(lock, broken), f"Incomplete {key} must be rejected."


def test_gradientshap_evidence_does_not_reward_small_scientific_disagreements():
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

    lock, report = _gradientshap_scope_fixture()
    measured = report["metrics"]["gpu_test"]
    for outcome in (0.0, 1.0, 1000.0):
        measured.update(eg_exact_gap_max=outcome, quadrature_drift_max=outcome, mc_residual_max=outcome)
        assert has_native_cpu_model_evidence(lock, report), "Scientific gaps are results, not gates."
    measured["eg_is_exact_feature_shapley"] = True
    assert not has_native_cpu_model_evidence(lock, report), "The report must preserve the estimand distinction."


def _shapley_interactions_scope_fixture():
    """Synthetic report metadata for gate tests; not evidence of a native execution."""
    import yaml

    lock, report = _valid_lock_report()
    part = ROOT / "chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap"
    expected = yaml.safe_load((part / "artifacts.lock.yml").read_text())["expected_metrics"]
    measured = {key.removesuffix("_min").removesuffix("_max"): value for key, value in expected.items()}
    measured.update(
        device="cpu", own_answer_code_cells=80,
        learner_evidence_sha256="a" * 64, fresh_evidence_sha256="b" * 64,
        physical_hybrid_file_sha256={
            f"{background}_{split}_hybrid.pt": "c" * 64
            for split in ("explain", "audit")
            for background in ("primary", "nested16", "nested128", "independent64_a", "independent64_b")
        },
    )
    lock.update(execution_device="cpu", exercise_metadata={"REQUIRES_GPU": False})
    report["metrics"]["gpu_evidence"] = {"category": "cpu_or_budget_metric", "uses_cuda": False}
    report["metrics"]["gpu_test"] = measured
    prefix = "chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap/expected_outputs/"
    report["report_inputs"]["files"] = [
        {"path": prefix + "learner_execution.json", "sha256": "a" * 64, "size_bytes": 10},
        {"path": prefix + "fresh_reproduction.json", "sha256": "b" * 64, "size_bytes": 10},
    ]
    return lock, report


def test_shapley_interaction_evidence_requires_all_games_contexts_and_raw_references():
    import copy
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

    lock, report = _shapley_interactions_scope_fixture()
    assert has_native_cpu_model_evidence(lock, report)
    defects = {
        "known_model_state_hashes_matched": 18,
        "unique_empirical_games": 1904,
        "physical_hybrid_shards": 5,
        "full_hybrid_predictions": 31195136,
        "primary_context_values": 28,
        "direct_four_forward_values": 0,
        "within_size_null_replicates": 1,
        "within_size_permutation_elements": 119 * 256,
        "within_size_seeds_retained": 0,
        "reference_student_coordinates": 1615680 - 256,
        "reference_raw_package_coordinates": 305184,
        "reference_package_index_calls": 3,
        "reference_max_error_to_bound_ratio": 1.001,
        "reference_full_direct_package_scope": False,
        "exact_random_game_order_cases": 15,
        "exact_boundary_order_n_cases": 0,
        "exact_package_index_calls": 159,
        "physical_hybrid_values_rehashed": False,
        "physical_hybrid_file_sha256": {},
        "fresh_evidence_sha256": "d" * 64,
        "native_evidence_reused": True,
        "fresh_comparison": "aggregate_only",
        "fresh_max_error": 1e-12,
        "support_distance_points": 0,
        "embedded_figures": 0,
    }
    for key, value in defects.items():
        broken = copy.deepcopy(report)
        broken["metrics"]["gpu_test"][key] = value
        assert not has_native_cpu_model_evidence(lock, broken), f"Incomplete {key} must fail."


def test_shapley_interaction_evidence_does_not_grade_method_superiority():
    from scripts.audit_report_evidence_contracts import has_native_cpu_model_evidence

    lock, report = _shapley_interactions_scope_fixture()
    measured = report["metrics"]["gpu_test"]
    for value in (0.0, 1.0, 1000.0):
        measured.update(
            stii_faith_coordinate_difference_max=value,
            faith_kernel_mse_max=value, stii_kernel_mse_max=value,
        )
        assert has_native_cpu_model_evidence(lock, report)
    for key in ("sii_joint_efficiency_claimed", "method_superiority_is_acceptance_gate", "observational_interactions_are_causal"):
        measured[key] = True
        assert not has_native_cpu_model_evidence(lock, report), f"The report must preserve {key}."
        measured[key] = False
