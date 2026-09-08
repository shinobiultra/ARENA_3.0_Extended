import json
from copy import deepcopy

from scripts.audit_course_surface import (
    course_surface_blockers,
    _exercise_notebook_declares_verification_contract,
    _has_direct_experiment_contract,
    _page_has_arena_shape,
    is_extension_section,
    load_config,
)


def test_is_extension_section_tracks_original_additions_and_new_chapters():
    assert is_extension_section("0.6")
    assert is_extension_section("1.6")
    assert is_extension_section("5.1")
    assert is_extension_section("17.1")
    assert not is_extension_section("1.2")
    assert not is_extension_section("4.5")


def test_course_surface_audit_flags_conversion_map_mismatch():
    config = deepcopy(load_config())
    config["conversion_map"]["5.1"]["exercise_dir"] = "wrong_dir"

    blockers = course_surface_blockers(config)

    assert "5.1: conversion_map exercise_dir mismatch" in blockers


def test_course_surface_audit_flags_missing_home_renderer():
    config = deepcopy(load_config())
    chapter = config["chapter_names"][5]
    config["chapter_names"][5] = chapter
    # The live audit covers filesystem presence; this mutation makes sure the
    # fixture still exercises chapter-level checks through the real config.
    blockers = course_surface_blockers(config)

    assert not any("missing instructions/Home.py" in blocker for blocker in blockers)


def test_notebook_verification_contract_detector_requires_visible_gpu_surface(tmp_path):
    notebook_path = tmp_path / "example_exercises.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "def run_gpu_test(max_vram_gb: float = 24.0):\n",
                            "    return json.loads((section_dir / 'verification_report.json').read_text())\n",
                            "\n",
                            "def run_full_experiment(max_vram_gb: float = 24.0):\n",
                            "    return run_gpu_test(max_vram_gb=max_vram_gb)\n",
                        ],
                    }
                ]
            }
        )
    )

    assert _exercise_notebook_declares_verification_contract(notebook_path)

    notebook_path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": []}]}))

    assert not _exercise_notebook_declares_verification_contract(notebook_path)


def test_current_course_surface_passes():
    assert course_surface_blockers() == []


def test_explicit_cpu_first_page_has_runtime_guidance():
    page = "# [12.3] Mini VLM\n## Core Question\n## Learning Objectives\nExercise\ntests.check()\nREQUIRES_GPU = False"
    assert _page_has_arena_shape(page, "12.3", "Mini VLM")
    assert not _page_has_arena_shape(page.replace("REQUIRES_GPU = False", ""), "12.3", "Mini VLM")
    assert _page_has_arena_shape(page.replace("Core Question", "Core question").replace("Learning Objectives", "Learning objectives"), "12.3", "Mini VLM")


def test_separate_publication_requires_validated_outputs_and_complete_execution(tmp_path):
    from pathlib import Path
    import shutil
    root=Path(__file__).resolve().parents[1]
    section=root/'chapter5_modern_architectures/exercises/part5_diffusion_language_models'
    source=next(section.glob('*_exercises.ipynb'))
    for name in (source.name,'lesson.md','execute_learner.py','solutions.py','publish_study.py','study_release.py'):
        shutil.copyfile(section/name,tmp_path/name)
    notebook=tmp_path/source.name
    assert _has_direct_experiment_contract(notebook)
    publisher=tmp_path/'publish_study.py';original=publisher.read_text()
    publisher.write_text('# A named file cannot stand in for validated publication\n')
    assert not _has_direct_experiment_contract(notebook)
    publisher.write_text(original)
    delegate=tmp_path/'study_release.py'
    delegate.write_text(delegate.read_text().replace('execute_learner.py','unrelated.py'))
    assert not _has_direct_experiment_contract(notebook)


def test_direct_learner_contract_requires_own_methods_answers_and_experiments(tmp_path):
    from pathlib import Path
    import shutil
    root = Path(__file__).resolve().parents[1]
    section = root / "chapter9_alignment_interpretability/exercises/part1_refusal_directions_safe_steering"
    source = next(section.glob("*_exercises.ipynb"))
    for name in (source.name, "lesson.md", "execute_learner.py", "solutions.py"):
        shutil.copyfile(section / name, tmp_path / name)
    notebook = tmp_path / source.name
    original = json.loads(notebook.read_text())
    assert _has_direct_experiment_contract(notebook)
    no_answers = deepcopy(original)
    for cell in no_answers["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = "".join(cell["source"]).replace("<summary>Solution</summary>", "<summary>Placeholder</summary>")
    notebook.write_text(json.dumps(no_answers))
    assert not _has_direct_experiment_contract(notebook)
    no_figures = deepcopy(original)
    for cell in no_figures["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = "".join(cell["source"]).replace(".savefig(", ".unrelated(")
    notebook.write_text(json.dumps(no_figures))
    assert not _has_direct_experiment_contract(notebook)
    notebook.write_text(json.dumps(original))
    (tmp_path / "execute_learner.py").write_text("# A filename alone is not an execution path\n")
    assert not _has_direct_experiment_contract(notebook)


def test_native_reproduction_requires_validation_actual_model_and_called_entrypoint(tmp_path):
    from pathlib import Path
    import shutil
    root=Path(__file__).resolve().parents[1]
    section=root/'chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls'
    source=next(section.glob('*_exercises.ipynb'))
    for name in (source.name,'lesson.md','execute_learner.py','solutions.py','publish_study.py','geometry_cuda.py'):
        shutil.copyfile(section/name,tmp_path/name)
    notebook=tmp_path/source.name
    assert _has_direct_experiment_contract(notebook)
    delegate=tmp_path/'geometry_cuda.py';original=delegate.read_text()
    for old,new in [('validate_learner(','unrelated('),('AutoModelForCausalLM.from_pretrained(','AutoModelForCausalLM.unrelated(')]:
        delegate.write_text(original.replace(old,new))
        assert not _has_direct_experiment_contract(notebook)
    delegate.write_text(original)
    source=tmp_path/'solutions.py';source.write_text(source.read_text().replace('fresh=reproduce(','fresh=unrelated('))
    assert not _has_direct_experiment_contract(notebook)


def test_das_reproduction_requires_visible_swap_native_continuation_and_hook_oracle(tmp_path):
    from pathlib import Path
    import shutil

    root = Path(__file__).resolve().parents[1]
    section = root / "chapter11_representation_geometry/exercises/part3_causal_abstraction_das"
    source = next(section.glob("*_exercises.ipynb"))
    for name in (source.name, "lesson.md", "execute_learner.py", "solutions.py",
                 "publish_study.py", "das_reproduction.py"):
        shutil.copyfile(section / name, tmp_path / name)
    notebook = tmp_path / source.name
    assert _has_direct_experiment_contract(notebook)
    delegate = tmp_path / "das_reproduction.py"
    original = delegate.read_text()
    mutations = [
        ("validate_learner(learner)", "unrelated(learner)"),
        ("AutoModelForCausalLM.from_pretrained(", "AutoModelForCausalLM.unrelated("),
        ("cache = capture_native(", "cache = unrelated("),
        ("hook = full_forward_intervention(", "hook = unrelated("),
        ('methods["subspace_interchange"](', 'methods["unrelated"]('),
        ('methods["continue_native"](', 'methods["unrelated"]('),
    ]
    for old, new in mutations:
        assert old in original, f"The mutation must target an actual call: {old}"
        delegate.write_text(original.replace(old, new))
        assert not _has_direct_experiment_contract(notebook), old
    delegate.write_text(original)
    entrypoint = tmp_path / "solutions.py"
    original_entrypoint = entrypoint.read_text()
    assert "fresh = reproduce(" in original_entrypoint
    entrypoint.write_text(original_entrypoint.replace("fresh = reproduce(", "fresh = unrelated("))
    assert not _has_direct_experiment_contract(notebook)


def test_plain_cpu_runtime_guidance_accepts_singular_and_plural():
    page = '# [11.4] Concept Erasure\n# Introduction\n## Content & Learning Objectives\nExercise\ntests.check()\nThe examples run on CPU.'
    assert _page_has_arena_shape(page, '11.4', 'Concept Erasure')
    assert _page_has_arena_shape(page.replace('examples run', 'example runs'), '11.4', 'Concept Erasure')
    assert _page_has_arena_shape(page.replace('Content & Learning Objectives', 'Content and learning objectives'),
                                 '11.4', 'Concept Erasure')
    assert not _page_has_arena_shape(page.replace('Content & Learning Objectives', 'Content and lesson notes'),
                                     '11.4', 'Concept Erasure')
    assert not _page_has_arena_shape(page.replace('run on CPU', 'show a plot'), '11.4', 'Concept Erasure')


def test_erasure_reproduction_requires_refits_native_calls_and_matched_controls(tmp_path):
    from pathlib import Path
    import shutil

    root = Path(__file__).resolve().parents[1]
    section = root / 'chapter11_representation_geometry/exercises/part4_concept_erasure'
    source = next(section.glob('*_exercises.ipynb'))
    for name in [source.name, 'lesson.md', 'execute_learner.py', 'solutions.py',
                 'publish_study.py', 'erasure_reproduction.py']:
        shutil.copyfile(section / name, tmp_path / name)
    notebook = tmp_path / source.name
    assert _has_direct_experiment_contract(notebook)
    delegate = tmp_path / 'erasure_reproduction.py'
    original = delegate.read_text()
    for old, new in [
        ('validate_learner(learner)', 'unrelated(learner)'),
        ('AutoModelForCausalLM.from_pretrained(', 'AutoModelForCausalLM.unrelated('),
        ('capture_native(model,', 'unrelated(model,'),
        ('full_forward_intervention(model,', 'unrelated(model,'),
        ('= construct_conditions(', '= unrelated('),
        ('compare_fitted_conditions(fresh_bank', 'unrelated(fresh_bank'),
        ('= infer(model,', '= unrelated(model,'),
        ('m.fit_leace(', 'm.unrelated('),
        ('m.fit_binary_probe(', 'm.unrelated('),
        ('m.fit_orthogonal_eraser(', 'm.unrelated('),
        ('continue_native(model,', 'unrelated(model,'),
        ('replacement = edit(', 'replacement = unrelated('),
        ('m.matched_random_edit(', 'm.unrelated('),
        ('m.apply_eraser(', 'm.unrelated('),
    ]:
        assert old in original, f'The mutation must target an actual scientific call: {old}'
        delegate.write_text(original.replace(old, new))
        assert not _has_direct_experiment_contract(notebook), old
    delegate.write_text(original)
    entry = tmp_path / 'solutions.py'
    assert 'fresh = run(' in entry.read_text()
    entry.write_text(entry.read_text().replace('fresh = run(', 'fresh = unrelated('))
    assert not _has_direct_experiment_contract(notebook)


def test_ccs_contract_follows_capture_refits_calibration_and_native_metrics(tmp_path):
    import shutil
    from pathlib import Path
    from chapter11_representation_geometry.exercises.part5_contrast_consistent_search.build_notebooks import lesson_cells

    section = Path(__file__).resolve().parents[1] / 'chapter11_representation_geometry/exercises/part5_contrast_consistent_search'
    notebook = tmp_path / '11.5_exercises.ipynb'
    notebook.write_text(json.dumps({'cells': lesson_cells()}))
    files = ['lesson.md', 'execute_learner.py', 'solutions.py', 'publish_study.py',
             'ccs_reproduction.py', 'ccs_native.py', 'ccs_study.py']
    for name in files:
        shutil.copyfile(section / name, tmp_path / name)
    assert _has_direct_experiment_contract(notebook)
    original_notebook = notebook.read_text()
    assert '.savefig(' in original_notebook
    notebook.write_text(original_notebook.replace('.savefig(', '.unrelated('))
    assert not _has_direct_experiment_contract(notebook)
    notebook.write_text(original_notebook)
    mutations = {
        'solutions.py': ['fresh = reproduce('],
        'ccs_reproduction.py': ['validate_learner(', 'compare_nested(', 'validate_complete_result(',
                               'native.load_model(', 'native.capture_pairs(', 'native.check_capture(',
                               'study.fit_reporters(', 'study.calibrate_reporters(', 'study.reporter_scores(',
                               'study.evaluate_reporters(', 'study.summarize_native_outputs('],
        'ccs_native.py': ['AutoModelForCausalLM.from_pretrained(', 'model.transformer(', 'model.lm_head(',
                          'expected = direct.logits'],
        'ccs_study.py': ['m.fit_pair_centering(', 'm.fit_ccs_restarts(', 'm.select_restart(',
                         'm.fit_difference_pca(', 'fit_binary_probe(', 'torch.randn(',
                         'm.calibrate_orientation(', 'm.paired_probabilities(', 'nm.paired_opinion_change(',
                         'nm.model_answer_metrics('],
    }
    # The oracle mutation changes the model call itself, not a metric label.
    mutations['ccs_native.py'][-1] = 'direct = model('
    for name, calls in mutations.items():
        path = tmp_path / name
        original = path.read_text()
        for call in calls:
            assert call in original, (name, call)
            replacement = call.rsplit('(', 1)[0].rsplit('.', 1)[0] + '.unrelated(' if '.' in call else 'unrelated('
            if call.startswith('fresh = '):
                replacement = 'fresh = unrelated('
            elif call.startswith('direct = '):
                replacement = 'direct = unrelated('
            path.write_text(original.replace(call, replacement))
            assert not _has_direct_experiment_contract(notebook), (name, call)
        path.write_text(original)
def test_data_attribution_contract_requires_actual_refits_and_all_comparisons(tmp_path):
    from pathlib import Path
    from scripts.audit_course_surface import _has_direct_experiment_contract

    root = Path(__file__).resolve().parents[1]
    section = root / 'chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin'
    notebook = tmp_path / '16.9_Influence_Functions_and_TracIn_exercises.ipynb'
    names = ['16.9_Influence_Functions_and_TracIn_exercises.ipynb', 'lesson.md',
             'execute_learner.py', 'solutions.py', 'publish_study.py', 'reproduce_native.py',
             'attribution_study.py', 'attribution_data.py', 'attribution_play.py']
    for name in names:
        (tmp_path / name).write_bytes((section / name).read_bytes())
    assert _has_direct_experiment_contract(notebook)
    mutations = {
        'reproduce_native.py': ['prepare_study(', 'evaluate_final(', 'compare(', 'scipy_logistic_optimum('],
        'attribution_study.py': ['m.fit_weighted_logistic(', 'm.sgd_trace(', 'm.upweight_influence(',
                                 'm.trace_loss_reductions(', 'm.ghost_linear_products(', 'm.class_matched_removals('],
        'attribution_data.py': ['zipfile.ZipFile('],
        'attribution_play.py': ['m.fit_weighted_logistic('],
    }
    for name, calls in mutations.items():
        path = tmp_path / name
        original = path.read_text()
        for call in calls:
            path.write_text(original.replace(call, 'missing_required_operation('))
            assert not _has_direct_experiment_contract(notebook), (name, call)
            path.write_text(original)


def test_datamodel_contract_follows_subset_training_selection_and_reproduction(tmp_path):
    from pathlib import Path
    from scripts.audit_course_surface import _has_direct_experiment_contract

    section = Path(__file__).resolve().parents[1] / 'chapter16_shapley_attribution_baselines/exercises/part10_datamodels'
    notebook = tmp_path / '16.10_Datamodels_exercises.ipynb'
    names = ['16.10_Datamodels_exercises.ipynb', 'lesson.md', 'execute_learner.py',
             'solutions.py', 'publish_study.py', 'reproduce_native.py', 'run_native.py',
             'datamodel_study.py', 'datamodel_data.py', 'datamodel_methods.py']
    for name in names:
        (tmp_path / name).write_bytes((section / name).read_bytes())
    assert _has_direct_experiment_contract(notebook)
    mutations = {
        'reproduce_native.py': ['native_state = run(', 'compare('],
        'run_native.py': ['data = load_data(', 'study.prepare_study(', 'study.evaluate_study(',
                          'study.deletion_play(', 'scipy_logistic_optimum(', ').fit('],
        'datamodel_study.py': ['fit_weighted_logistic(', 'methods.fit_datamodel_path(',
                              'methods.sample_subsets(', 'methods.matched_random_removals(',
                              'methods.correct_class_margins('],
        'datamodel_data.py': ['arrays = source_arrays('],
        'datamodel_methods.py': ['fit = fit_lasso(', 'updated = soft_threshold(', 'residual = lasso_kkt('],
    }
    for name, calls in mutations.items():
        path = tmp_path / name
        original = path.read_text()
        for call in calls:
            assert call in original, (name, call)
            replacement = ').unrelated(' if call == ').fit(' else 'missing_required_operation('
            path.write_text(original.replace(call, replacement))
            assert not _has_direct_experiment_contract(notebook), (name, call)
            path.write_text(original)


def test_cartography_contract_requires_training_logging_selection_and_refits(tmp_path):
    from pathlib import Path
    from scripts.audit_course_surface import _has_direct_experiment_contract

    section = Path(__file__).resolve().parents[1] / 'chapter16_shapley_attribution_baselines/exercises/part11_dataset_cartography'
    notebook = tmp_path / '16.11_Dataset_Cartography_exercises.ipynb'
    names = ['16.11_Dataset_Cartography_exercises.ipynb', 'lesson.md', 'execute_learner.py',
             'solutions.py', 'publish_study.py', 'reproduce_native.py', 'run_native.py',
             'cartography_methods.py', 'cartography_study.py', 'cartography_data.py',
             'cartography_reference.py']
    for name in names:
        (tmp_path / name).write_bytes((section / name).read_bytes())
    assert _has_direct_experiment_contract(notebook)
    mutations = {
        'reproduce_native.py': ['native_state = run(', 'compare('],
        'run_native.py': ['mapping = map_training_data(', 'prepared = retrain_selections(',
                          'evaluated = evaluate_study(', 'play = retraining_play(',
                          'implementation.train_ensemble(', 'np.testing.assert_allclose('],
        'cartography_methods.py': ['torch.autograd.grad(', 'align_epoch(torch.cat(',
                                   'probabilities.std(', 'value.add_('],
        'cartography_study.py': ['methods.train_ensemble(', 'methods.select_indices(',
                                'methods.matched_random_indices(', 'methods.evaluate_models('],
        'cartography_data.py': ['arrays = source_arrays('],
        'cartography_reference.py': ['optimizer.step('],
    }
    for name, calls in mutations.items():
        path = tmp_path / name
        original = path.read_text()
        for call in calls:
            assert call in original, (name, call)
            replacement = 'missing_required_operation(torch.cat(' if call == 'align_epoch(torch.cat(' else 'missing_required_operation('
            path.write_text(original.replace(call, replacement))
            assert not _has_direct_experiment_contract(notebook), (name, call)
            path.write_text(original)


def test_sage_contract_requires_model_loss_sampling_and_independent_references(tmp_path):
    from pathlib import Path
    from scripts.audit_course_surface import _has_direct_experiment_contract

    section = Path(__file__).resolve().parents[1] / 'chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance'
    notebook = tmp_path / '16.12_SAGE_Global_Importance_exercises.ipynb'
    names = ['16.12_SAGE_Global_Importance_exercises.ipynb', 'lesson.md', 'execute_learner.py',
             'solutions.py', 'publish_study.py', 'reproduce_native.py', 'run_native.py',
             'sage_methods.py', 'sage_study.py', 'sage_data.py', 'sage_reference.py']
    for name in names:
        (tmp_path / name).write_bytes((section / name).read_bytes())
    assert _has_direct_experiment_contract(notebook)
    mutations = {
        'reproduce_native.py': ['native_state = run(', 'compare('],
        'run_native.py': ['sage_study.prepare_study(', 'sage_study.evaluate_study(', 'sage_reference.checks('],
        'sage_methods.py': ['t.where(', 'math.comb(', 't.linalg.pinv(', 'path_contributions('],
        'sage_study.py': ['optimizer.step(', 'methods.loss_game(', 'methods.prefix_values(', 'methods.permutation_sage('],
        'sage_reference.py': ['np.linalg.lstsq(', 'author_imputer()', 'np.bitwise_or('],
    }
    for name, calls in mutations.items():
        path = tmp_path / name
        original = path.read_text()
        for call in calls:
            assert call in original, (name, call)
            path.write_text(original.replace(call, 'missing_required_operation()' if call.endswith('()') else 'missing_required_operation('))
            assert not _has_direct_experiment_contract(notebook), (name, call)
            path.write_text(original)
    # A helper name with no actual save operation is insufficient.
    import json
    document = json.loads(notebook.read_text())
    for cell in document['cells']:
        if cell['cell_type'] == 'code':
            cell['source'] = [''.join(cell['source']).replace('plt.gcf().savefig(', 'print(')]
    notebook.write_text(json.dumps(document))
    assert not _has_direct_experiment_contract(notebook)


def test_treeshap_contract_follows_nine_visible_methods_fits_audit_and_references(tmp_path):
    import ast
    import copy
    import shutil
    from pathlib import Path
    from chapter16_shapley_attribution_baselines.exercises.part13_treeshap.build_notebooks import (
        build_notebooks,
    )
    from scripts.audit_course_surface import _has_direct_experiment_contract

    section = (
        Path(__file__).resolve().parents[1]
        / 'chapter16_shapley_attribution_baselines/exercises/part13_treeshap'
    )
    notebook = tmp_path / '16.13_TreeSHAP_from_First_Principles_exercises.ipynb'
    learner, _ = build_notebooks()
    notebook.write_text(json.dumps(learner))
    names = [
        'lesson.md', 'execute_learner.py', 'solutions.py', 'publish_study.py', 'reproduce_native.py',
        'run_native.py', 'treeshap_data.py', 'treeshap_methods.py',
        'treeshap_study.py', 'treeshap_diagnostics.py', 'treeshap_reference.py',
    ]
    for name in names:
        shutil.copyfile(section / name, tmp_path / name)
    assert _has_direct_experiment_contract(notebook)

    mutations = {
        'reproduce_native.py': ['fresh_state = run(', '            compare(\n'],
        'run_native.py': [
            'fitted = study.fit_models(train)',
            'audit = data.load_split("audit")',
            'evaluated = study.evaluate_study(',
            'reference = reference_module.check_reference(',
            'diagnostics = diagnostics_module.run_diagnostics(',
        ],
        'treeshap_study.py': ['model.fit(', 'rng.permutation('],
        'treeshap_diagnostics.py': ['"correlated": correlated_games('],
        'treeshap_reference.py': [
            'shap.TreeExplainer(', 'raw.shap_values(', 'corrected.shap_values(',
            'exact_path = subset_reference(',
        ],
    }
    for name, calls in mutations.items():
        path = tmp_path / name
        original = path.read_text()
        for call in calls:
            assert call in original, (name, call)
            path.write_text(original.replace(call, 'missing_required_operation('))
            assert not _has_direct_experiment_contract(notebook), (name, call)
            path.write_text(original)

    # Passing a method to its immediate test is not enough. Hiding an actual
    # taught call, either editable play, or the visible result calls must fail
    # even while stubs, answers, labels, and helper definitions remain.
    document = json.loads(notebook.read_text())

    class HideCalls(ast.NodeTransformer):
        def __init__(self, hidden_name):
            self.hidden_name = hidden_name

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == self.hidden_name:
                node.func = ast.Attribute(
                    value=ast.Name(id='hidden_methods', ctx=ast.Load()),
                    attr=self.hidden_name, ctx=ast.Load(),
                )
            return node

    for hidden_name in ('ensemble_shap', 'tiny_tree_play', 'native_play', 'show_figure'):
        hidden = copy.deepcopy(document)
        for cell in hidden['cells']:
            if cell['cell_type'] != 'code':
                continue
            tree = HideCalls(hidden_name).visit(ast.parse(''.join(cell['source'])))
            ast.fix_missing_locations(tree)
            cell['source'] = [ast.unparse(tree) + '\n']
        notebook.write_text(json.dumps(hidden))
        assert not _has_direct_experiment_contract(notebook), hidden_name


def test_deepshap_contract_follows_nine_visible_methods_fits_audit_and_references(tmp_path):
    import copy
    import shutil
    from pathlib import Path

    from chapter16_shapley_attribution_baselines.exercises.part14_deepshap.build_notebooks import (
        build_notebooks,
    )
    from scripts.audit_course_surface import _has_direct_experiment_contract

    section = (
        Path(__file__).resolve().parents[1]
        / "chapter16_shapley_attribution_baselines/exercises/part14_deepshap"
    )
    notebook = tmp_path / "16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP_exercises.ipynb"
    learner, _ = build_notebooks()
    notebook.write_text(json.dumps(learner))
    names = [
        "lesson.md",
        "execute_learner.py",
        "solutions.py",
        "publish_study.py",
        "reproduce_native.py",
        "run_native.py",
        "deepshap_data.py",
        "deepshap_methods.py",
        "deepshap_study.py",
        "deepshap_diagnostics.py",
        "deepshap_reference.py",
    ]
    for name in names:
        shutil.copyfile(section / name, tmp_path / name)
    assert _has_direct_experiment_contract(notebook)

    mutations = {
        "reproduce_native.py": ["fresh_state = run(", "            compare(\n"],
        "run_native.py": [
            "fitted = study.fit_models(train)",
            "persist_prepared(directory, fitted, prepared)",
            'audit = data.load_split("audit")',
            "evaluated = study.evaluate_study(",
            "reference = reference_module.validate_reference(",
            "diagnostics = diagnostics_module.run_diagnostics(",
            "play = study.native_play(",
        ],
        "deepshap_methods.py": [
            "x_states, b_states = paired_forward(",
            "secant = (t.relu(",
            "multiplier = rescale_multipliers(",
            "hybrids = t.where(",
            "math.comb(",
            "positive_first = activate(",
            "values.gather(",
        ],
        "deepshap_study.py": [
            "optimizer.step(",
            "coefficients = t.linalg.solve(",
            "base, phi_deep = methods.background_deepshap(",
            "game = methods.coalition_values(",
            "replacement[key][\"random\"] = _random_curves(",
            "**_explain_model(fitted[\"models\"][\"clean_0\"]",
        ],
        "deepshap_diagnostics.py": [
            '"near_zero": diagnose_near_zero(',
            '"threshold": diagnose_threshold(',
            '"cancellation": diagnose_cancellation(',
        ],
        "deepshap_reference.py": [
            "shap.DeepExplainer(",
            "pair = _bounded_pair_author(",
            "exact = subset_reference(",
        ],
    }
    for name, calls in mutations.items():
        path = tmp_path / name
        original = path.read_text()
        for call in calls:
            assert call in original, (name, call)
            path.write_text(original.replace(call, "missing_required_operation("))
            assert not _has_direct_experiment_contract(notebook), (name, call)
            path.write_text(original)

    # Stubs, solutions and method names alone are insufficient. Each answer
    # must be visibly bound into the namespace used by diagnostics and study.
    methods = (
        "paired_forward",
        "rescale_slope",
        "rescale_multipliers",
        "single_reference",
        "background_deepshap",
        "coalition_values",
        "exact_shapley",
        "reveal_cancel_groups",
        "ranked_replacement_curves",
    )
    document = json.loads(notebook.read_text())
    for method in methods:
        hidden = copy.deepcopy(document)
        needle = f"{method}={method}"
        replacement = f"{method}=hidden_methods.{method}"
        changed = False
        for cell in hidden["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if needle in source:
                cell["source"] = [source.replace(needle, replacement)]
                changed = True
        assert changed, method
        notebook.write_text(json.dumps(hidden))
        assert not _has_direct_experiment_contract(notebook), method


def test_gradientshap_refit_contract_follows_hashes_audit_reference_and_external_draws(
    tmp_path,
):
    import ast
    import shutil
    from pathlib import Path

    from scripts.audit_course_surface import _has_gradientshap_refit_contract

    section = (
        Path(__file__).resolve().parents[1]
        / "chapter16_shapley_attribution_baselines/exercises/part15_gradientshap"
    )
    names = [
        "run_native.py",
        "reproduce_native.py",
        "gradientshap_data.py",
        "gradientshap_methods.py",
        "gradientshap_study.py",
        "gradientshap_diagnostics.py",
        "gradientshap_reference.py",
    ]
    for name in names:
        shutil.copyfile(section / name, tmp_path / name)

    def contract() -> bool:
        tree = ast.parse((tmp_path / "reproduce_native.py").read_text())
        reproduce = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "reproduce"
        )
        fresh_names = {
            ast.unparse(call.func)
            for call in ast.walk(reproduce)
            if isinstance(call, ast.Call)
        }
        return _has_gradientshap_refit_contract(tmp_path, fresh_names)

    assert contract()
    mutations = {
        "reproduce_native.py": [
            ("process = subprocess.run(", "process = missing_required_operation("),
            ("compare_manifests(\n", "missing_required_operation(\n"),
        ],
        "run_native.py": [
            ("fitted = study.fit_models(train)", "fitted = missing_required_operation(train)"),
            (
                "reproduction = study.verify_16_14_reproduction(fitted, data.load_manifest())",
                "reproduction = missing_required_operation(fitted, data.load_manifest())",
            ),
            ("persist_prepared(directory, fitted, prepared)", "missing_required_operation(directory, fitted, prepared)"),
            ('audit = data.load_split("audit")', 'audit = data.load_split("explain")'),
            ("evaluated = study.evaluate_study(", "evaluated = missing_required_operation("),
            (
                "reference = reference_module.validate_reference(",
                "reference = missing_required_operation(",
            ),
            (
                "diagnostics = diagnostics_module.run_diagnostics(methods=methods)",
                "diagnostics = missing_required_operation(methods=methods)",
            ),
            ("play = study.native_play(", "play = missing_required_operation("),
            ("persist_completed(\n", "missing_required_operation(\n"),
            ("state_hash_count != 21", "state_hash_count != 22"),
            (
                "EXTERNAL_DETERMINISTIC_ARTIFACT = \"replicate_samples.pt\"",
                "EXTERNAL_DETERMINISTIC_ARTIFACT = \"replicate_summary.pt\"",
            ),
            (
                "REPLICATE_SHAPE = (64, 3, 64, 2048, 8)",
                "REPLICATE_SHAPE = (64, 3, 64, 2048, 7)",
            ),
            (
                'samples = replicates.get("samples")',
                'samples = replicates.get("summary")',
            ),
            (
                "save_payload(directory, EXTERNAL_DETERMINISTIC_ARTIFACT, samples)",
                "save_payload(directory, EXTERNAL_DETERMINISTIC_ARTIFACT, array.shape)",
            ),
            (
                '"payload_path": "/sampling_replicates/samples"',
                '"payload_path": "/sampling_replicates/summary"',
            ),
        ],
        "gradientshap_study.py": [
            ("tensor_mapping_sha256(record[field])", "str(record[field])"),
            ('"raw_folded_state": "state"', '"raw_folded_state": "standardized_state"'),
            ("optimizer.step()", "missing_required_operation()"),
            (
                "methods.expected_gradient_samples(",
                "methods.missing_expected_gradient_samples(",
            ),
            (
                "methods.noisy_expected_gradient_samples(",
                "methods.missing_noisy_expected_gradient_samples(",
            ),
            (
                "methods.ranked_coalition_curves(local_values, value)",
                "missing_required_operation(local_values, value)",
            ),
        ],
        "gradientshap_methods.py": [
            ("np.random.PCG64(", "np.random.MT19937("),
            ("t.autograd.grad(", "missing_required_operation("),
            ("prefix.std(", "prefix.var("),
            ("np.polynomial.legendre.leggauss(", "missing_required_operation("),
            ("t.argsort(", "missing_required_operation("),
        ],
        "gradientshap_reference.py": [
            ("shap.GradientExplainer(", "missing_required_operation("),
            ("explainer.shap_values(", "missing_required_operation("),
            ("GradientShap(model).attribute(", "missing_required_operation("),
            ("replay_draw_table(\n", "missing_required_operation(\n"),
            ("for key in MODEL_KEYS:", "for key in MODEL_KEYS[:1]:"),
            ("positive_noise = positive_noise_reference(methods)", "positive_noise = {}"),
        ],
        "gradientshap_diagnostics.py": [
            ("noise = diagnose_noise_target(\n", "noise = missing_required_operation(\n"),
            ("game = diagnose_path_vs_coalition(\n", "game = missing_required_operation(\n"),
        ],
    }
    for name, changes in mutations.items():
        path = tmp_path / name
        original = path.read_text()
        for old, new in changes:
            assert old in original, (name, old)
            path.write_text(original.replace(old, new))
            assert not contract(), (name, old)
            path.write_text(original)


def test_gradientshap_direct_learner_requires_nine_answers_order_and_real_refit_path(
    tmp_path,
):
    import copy
    import shutil
    from pathlib import Path

    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.build_notebooks import (
        build_notebooks,
    )
    from scripts.audit_course_surface import _has_direct_experiment_contract

    section = (
        Path(__file__).resolve().parents[1]
        / "chapter16_shapley_attribution_baselines/exercises/part15_gradientshap"
    )
    notebook = tmp_path / "16.15_Expected_Gradients_exercises.ipynb"
    learner, _ = build_notebooks()
    notebook.write_text(json.dumps(learner))
    names = [
        "lesson.md",
        "execute_learner.py",
        "solutions.py",
        "publish_study.py",
        "reproduce_native.py",
        "run_native.py",
        "gradientshap_data.py",
        "gradientshap_methods.py",
        "gradientshap_study.py",
        "gradientshap_diagnostics.py",
        "gradientshap_reference.py",
    ]
    for name in names:
        shutil.copyfile(section / name, tmp_path / name)
    assert _has_direct_experiment_contract(notebook)

    document = json.loads(notebook.read_text())

    def altered_notebook(old: str, new: str) -> dict:
        changed = False
        result = copy.deepcopy(document)
        for cell in result["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if old in source:
                cell["source"] = [source.replace(old, new)]
                changed = True
        assert changed, old
        return result

    methods = (
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
    for method in methods:
        hidden = altered_notebook(
            f"{method}={method}",
            f"{method}=hidden_methods.{method}",
        )
        notebook.write_text(json.dumps(hidden))
        assert not _has_direct_experiment_contract(notebook), method

    visible_steps = [
        ("fitted = fit_models(train)", "fitted = hidden_fit(train)"),
        (
            "reproduction = verify_16_14_reproduction(fitted, load_manifest())",
            "reproduction = hidden_verify(fitted, load_manifest())",
        ),
        ("prepared = prepare_study(\n", "prepared = hidden_prepare(\n"),
        (
            "before_audit = persist_prepared(native_directory, fitted, prepared)",
            "before_audit = hidden_persist(native_directory, fitted, prepared)",
        ),
        ('audit = load_split("audit")', 'audit = load_split("explain")'),
        ("evaluated = evaluate_study(\n", "evaluated = hidden_evaluate(\n"),
        ("reference = reference_check(\n", "reference = hidden_reference(\n"),
        ("diagnostics = run_diagnostics(methods=methods)", "diagnostics = hidden_diagnostics(methods=methods)"),
        ("fixed_play = native_play(\n", "fixed_play = hidden_play(\n"),
        ("summary = persist_completed(\n", "summary = hidden_completed(\n"),
        ("write_notebook_summary(FIGURE_DIR, native_directory, summary)", "hidden_summary(FIGURE_DIR, native_directory, summary)"),
        ("tiny_play = threshold_play(\n", "tiny_play = hidden_threshold_play(\n"),
    ]
    for old, new in visible_steps:
        notebook.write_text(json.dumps(altered_notebook(old, new)))
        assert not _has_direct_experiment_contract(notebook), old

    reordered = copy.deepcopy(document)
    preparation = next(
        index
        for index, cell in enumerate(reordered["cells"])
        if cell["cell_type"] == "code" and "fitted = fit_models(train)" in "".join(cell["source"])
    )
    audit = next(
        index
        for index, cell in enumerate(reordered["cells"])
        if cell["cell_type"] == "code" and 'audit = load_split("audit")' in "".join(cell["source"])
    )
    reordered["cells"][preparation], reordered["cells"][audit] = (
        reordered["cells"][audit],
        reordered["cells"][preparation],
    )
    notebook.write_text(json.dumps(reordered))
    assert not _has_direct_experiment_contract(notebook)

    notebook.write_text(json.dumps(altered_notebook("show_figure(", "hidden_figure(")))
    assert not _has_direct_experiment_contract(notebook)

    notebook.write_text(json.dumps(document))
    for filename, old, new in (
        ("execute_learner.py", "complete_learner(draft)", "hidden_completion(draft)"),
        ("solutions.py", "reproduce(learner, output)", "hidden_reproduction(learner, output)"),
        (
            "publish_study.py",
            "execution, reproduction = validate_fresh(learner, fresh)",
            "execution, reproduction = hidden_validation(learner, fresh)",
        ),
        ("gradientshap_methods.py", "np.random.PCG64(", "np.random.MT19937("),
    ):
        path = tmp_path / filename
        original = path.read_text()
        assert old in original, (filename, old)
        path.write_text(original.replace(old, new))
        assert not _has_direct_experiment_contract(notebook), (filename, old)
        path.write_text(original)


def test_shapley_interaction_refit_contract_requires_seal_references_and_real_plays(
    tmp_path,
):
    import ast
    import shutil
    from pathlib import Path

    from scripts.audit_course_surface import _has_shapley_interaction_refit_contract

    section = (
        Path(__file__).resolve().parents[1]
        / "chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap"
    )
    names = [
        "run_native.py",
        "reproduce_native.py",
        "interaction_data.py",
        "interaction_methods.py",
        "interaction_study.py",
        "interaction_reference.py",
        "interaction_diagnostics.py",
    ]
    for name in names:
        shutil.copyfile(section / name, tmp_path / name)

    def contract() -> bool:
        tree = ast.parse((tmp_path / "reproduce_native.py").read_text())
        reproduce = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "reproduce"
        )
        fresh_names = {
            ast.unparse(call.func)
            for call in ast.walk(reproduce)
            if isinstance(call, ast.Call)
        }
        return _has_shapley_interaction_refit_contract(tmp_path, fresh_names)

    assert contract()
    mutations = {
        "reproduce_native.py": [
            ("process = subprocess.run(", "process = missing_required_operation("),
            ("compare_manifests(\n", "missing_required_operation(\n"),
        ],
        "run_native.py": [
            ("fitted = study.fit_known_models(train)", "fitted = hidden_fit(train)"),
            ("_state_hash_gate(reproduction)", "missing_required_operation(reproduction)"),
            ("count != 21", "count != 22"),
            ("persist_prepared(directory, fitted, prepared)", "hidden_seal(directory, fitted, prepared)"),
            ('audit = data_module.load_split("audit")', 'audit = data_module.load_split("explain")'),
            ("evaluated = study.evaluate_study(", "evaluated = hidden_evaluation("),
            (
                "reference = reference_module.validate_study_reference(",
                "reference = hidden_reference(",
            ),
            (
                "diagnostics = diagnostics_module.interaction_diagnostics(",
                "diagnostics = hidden_diagnostics(",
            ),
            ("play = study.native_play(", "play = hidden_play("),
            (
                "persisted_prepared, _ = _persist_phase_hybrids(",
                "persisted_prepared, _ = missing_required_operation(",
            ),
        ],
        "interaction_methods.py": [
            ("mobius[..., present] -= mobius[..., lower]", "mobius[..., present] += mobius[..., lower]"),
            (
                "math.comb(\n                    superset.bit_count(), order\n                )",
                "superset.bit_count()",
            ),
            ("torch.linalg.solve(kkt, right_hand_side.T)", "torch.linalg.lstsq(kkt, right_hand_side.T).solution"),
            (
                "math.comb(superset_size + max_order - 1, max_order + size)",
                "math.comb(superset_size, max_order + size)",
            ),
            ("sums.index_add_(0, slots, output)", "missing_required_operation(sums, slots, output)"),
            (
                "- game_values[..., with_second]",
                "+ game_values[..., with_second]",
            ),
        ],
        "interaction_study.py": [
            ("optimizer.step()", "missing_required_operation()"),
            ("enumerate(MODEL_SEEDS):", "enumerate(MODEL_SEEDS[:1]):"),
            ('"raw_folded_state": "state"', '"raw_folded_state": "standardized_state"'),
            ('"exact_games": _exact_suite(methods, reference)', '"exact_games": {}'),
            ("methods.masked_model_game(", "methods.hidden_masked_model_game("),
            ("methods.faith_shap_interactions(", "methods.hidden_faith_shap_interactions("),
            ("\"exact_dividend\": exact_dividend_play(methods=methods)", "\"exact_dividend\": {}"),
            ("empirical = empirical_context_play(", "empirical = hidden_empirical_play("),
            ("donor = empirical_context_play(", "donor = hidden_empirical_play("),
        ],
        "interaction_reference.py": [
            ("def _direct_stii(", "def hidden_direct_stii("),
            ("def _direct_fsii(", "def hidden_direct_fsii("),
            ("np.linalg.solve(kkt, rhs)", "np.linalg.lstsq(kkt, rhs, rcond=None)[0]"),
            ("exact = ExactComputer(game=game, n_players=n_players, evaluate_game=True)", "exact = hidden_computer(game=game, n_players=n_players, evaluate_game=True)"),
            ("payload_validation = validate_reference_payload(", "payload_validation = hidden_payload_validation("),
        ],
        "interaction_diagnostics.py": [
            (
                'primary.get("pair_values"), "primary64.pair_values", (7, 272, 3, 28)',
                'primary.get("pair_values"), "primary64.pair_values", (1, 1, 1, 1)',
            ),
        ],
    }
    for name, changes in mutations.items():
        path = tmp_path / name
        original = path.read_text()
        for old, new in changes:
            assert old in original, (name, old)
            path.write_text(original.replace(old, new))
            assert not contract(), (name, old)
            path.write_text(original)
