"""Source-only fictional metadata; no corpus, model, target, torch, or GPU access."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json

import pytest

from mastery_hub.empirical_protocol import (
    Budget, CacheIdentity, Ledger, Protocol, RunPlan, SplitItem,
    canonical_settings, require_fair_pair, validate_splits,
)


def digest(text):
    return sha256(text.encode()).hexdigest()


def settings(**changes):
    return canonical_settings(dict(width=8, depth=1, optimizer='adamw', **changes))


def run(name='a', **changes):
    return RunPlan(run_id=name, phase='fixed', seed=11, settings_json=settings(),
                   budget=Budget(10, 5, 20, 3, 4, 100, 50, 6), **changes)


def split_items():
    return tuple(SplitItem(split, f'group{i}', f'source{i}', digest(f'document{i}'))
                 for i, split in enumerate(('TRAIN', 'DEV', 'TEST')))


def protocol():
    return Protocol('fictional-only', digest('source'), digest('model'), digest('env'),
                    validate_splits(split_items()), (run(),), Budget(30, 15, 40, 8, 10, 500, 500, 20))


def test_protocol_is_immutable_and_round_trips_strictly():
    p = protocol()
    assert Protocol.from_json(p.to_json()) == p
    with pytest.raises(FrozenInstanceError):
        p.protocol_id = 'changed'
    payload = json.loads(p.to_json())
    payload['unknown'] = 1
    with pytest.raises(ValueError, match='fields'):
        Protocol.from_json(json.dumps(payload))
    payload.pop('unknown')
    payload['runs'][0]['mystery'] = 'ignored?'
    with pytest.raises(ValueError, match='fields'):
        Protocol.from_json(json.dumps(payload))


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}',
                                '{"a":1e999}', '__import__("os")'])
def test_nonfinite_duplicate_and_executable_json_rejected(raw):
    with pytest.raises(ValueError):
        canonical_settings(raw)


def test_cache_identity_changes_every_scientific_axis_and_rejects_stale():
    p = protocol()
    cache = CacheIdentity.for_run(p, 'a')
    assert cache.require_match(p, 'a') is None
    for field in ('source_sha256', 'model_sha256', 'environment_sha256', 'split_sha256'):
        changed = replace(p, **{field: digest(field)})
        with pytest.raises(ValueError, match='stale'):
            cache.require_match(changed, 'a')
    for change in (dict(seed=12), dict(settings_json=settings(lr=.1)),
                   dict(budget=replace(p.runs[0].budget, steps=5))):
        changed = replace(p, runs=(replace(p.runs[0], **change),))
        with pytest.raises(ValueError, match='stale'):
            cache.require_match(changed, 'a')
    with pytest.raises(ValueError, match='unknown'):
        CacheIdentity.for_run(p, 'other')


def test_split_guard_checks_entire_source_group_and_exact_content_not_only_row_ids():
    items = split_items()
    assert validate_splits(tuple(reversed(items))) == validate_splits(items)
    for field in ('group_id', 'source_id', 'document_sha256'):
        independent = SplitItem('TEST', 'newgroup', 'newsource', digest('new-document'))
        bad = items + (replace(independent, **{field: getattr(items[1], field)}),)
        with pytest.raises(ValueError, match='leakage'):
            validate_splits(bad)
    with pytest.raises(ValueError, match='duplicate'):
        validate_splits(items + (items[0],))
    with pytest.raises(ValueError, match='TRAIN.*DEV.*TEST'):
        validate_splits(items[:2])
    with pytest.raises(ValueError):
        SplitItem('validation', 'g', 's', digest('d'))


def test_ledger_reserves_whole_runs_including_failure_and_never_recycles():
    p = protocol()
    ledger = Ledger(p.fingerprint, ())
    reserved = ledger.reserve(p, 'a')
    assert ledger.reserved_run_ids == ()
    assert reserved.reserved_run_ids == ('a',)
    assert reserved.consumed(p) == p.runs[0].budget
    with pytest.raises(ValueError, match='already reserved'):
        reserved.reserve(p, 'a')
    with pytest.raises(ValueError, match='unknown'):
        ledger.reserve(p, 'missing')
    with pytest.raises(ValueError, match='stale'):
        ledger.reserve(replace(p, source_sha256=digest('new')), 'a')


@pytest.mark.parametrize('axis', tuple(Budget.__dataclass_fields__))
def test_every_resource_axis_is_enforced_without_partial_reservation(axis):
    p = protocol()
    tight = replace(p, cap=replace(p.cap, **{axis: getattr(p.runs[0].budget, axis) - 1}))
    ledger = Ledger(tight.fingerprint, ())
    with pytest.raises(ValueError, match='budget'):
        ledger.reserve(tight, 'a')
    assert ledger.reserved_run_ids == ()


def test_aggregate_caps_include_all_runs_not_just_each_run():
    p = protocol()
    p = replace(p, runs=(run('a'), run('b')), cap=run().budget)
    ledger = Ledger(p.fingerprint, ()).reserve(p, 'a')
    with pytest.raises(ValueError, match='budget'):
        ledger.reserve(p, 'b')
    assert ledger.reserved_run_ids == ('a',)


def test_strict_resource_types_and_known_phases():
    for invalid in (-1, True, 1.1, float('nan')):
        with pytest.raises(ValueError):
            replace(run().budget, steps=invalid)
    with pytest.raises(ValueError):
        replace(run(), phase='retry')
    with pytest.raises(ValueError):
        replace(protocol(), runs=(run(), run()))
    with pytest.raises(ValueError):
        replace(protocol(), runs=[run()])


def test_fixed_comparison_rejects_hidden_changes_and_mixed_budget():
    a = run()
    b = replace(a, run_id='b', settings_json=canonical_settings(dict(width=16, depth=1, optimizer='adamw')))
    require_fair_pair(a, b, factor='width', regime='fixed')
    with pytest.raises(ValueError, match='settings'):
        require_fair_pair(a, replace(b, settings_json=canonical_settings(dict(width=16, depth=2, optimizer='adamw'))),
                          factor='width', regime='fixed')
    with pytest.raises(ValueError, match='budget'):
        require_fair_pair(a, replace(b, budget=replace(b.budget, train_tokens=101)), factor='width', regime='fixed')
    with pytest.raises(ValueError, match='seed'):
        require_fair_pair(a, replace(b, seed=99), factor='width', regime='fixed')
    with pytest.raises(ValueError):
        require_fair_pair(a, b, factor='width', regime='best-looking')


def test_retuned_pair_needs_matching_grid_and_selection_rule():
    a = replace(run(), phase='retuned', settings_json=canonical_settings(dict(
        width=8, lr=.01, tuning_grid=[.01,.02], selection_rule='mean-dev-final', depth=1)))
    b = replace(a, run_id='b', settings_json=canonical_settings(dict(
        width=16, lr=.02, tuning_grid=[.01,.02], selection_rule='mean-dev-final', depth=1)))
    require_fair_pair(a, b, factor='width', regime='retuned')
    with pytest.raises(ValueError, match='grid'):
        require_fair_pair(a, replace(b, settings_json=canonical_settings(dict(
            width=16, lr=.02, tuning_grid=[.01,.02,.03], selection_rule='mean-dev-final', depth=1))),
            factor='width', regime='retuned')


def test_no_torch_model_network_or_data_imports_in_pure_guard_module():
    # Inspect this module's own AST, never any corpus/cache/target.
    import ast
    from pathlib import Path
    from mastery_hub import empirical_protocol

    tree = ast.parse(Path(empirical_protocol.__file__).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module.split('.')[0])
    assert set(names) <= {'__future__', 'dataclasses', 'hashlib', 'json', 'math', 're', 'typing'}


def test_import_rejects_unknown_nested_budget_and_nonfinite_settings():
    payload = json.loads(protocol().to_json())
    payload['runs'][0]['budget']['imaginary_resource'] = 1
    with pytest.raises(ValueError, match='fields'):
        Protocol.from_json(json.dumps(payload))
    payload['runs'][0]['budget'].pop('imaginary_resource')
    payload['runs'][0]['settings_json'] = '{"learning_rate":NaN}'
    with pytest.raises(ValueError):
        Protocol.from_json(json.dumps(payload))


@pytest.mark.parametrize('grid', ['not-a-grid', [], [True], [0.0], [-.01], [.01, .01]])
def test_retuning_grid_is_a_finite_positive_unique_candidate_list(grid):
    config = canonical_settings(dict(width=8, lr=.01, tuning_grid=grid, selection_rule='mean-dev-final'))
    a = replace(run(), phase='retuned', settings_json=config)
    b = replace(a, run_id='b', settings_json=canonical_settings(dict(
        width=16, lr=.01, tuning_grid=grid, selection_rule='mean-dev-final')))
    with pytest.raises(ValueError, match='grid'):
        require_fair_pair(a, b, factor='width', regime='retuned')


@pytest.mark.parametrize('side,field,value', [
    ('left', 'lr', True), ('right', 'lr', True),
    ('left', 'tuning_grid', [True, 2]), ('right', 'tuning_grid', [True, 2]),
])
def test_retuned_comparisons_reject_boolean_numeric_aliases_on_both_sides(side, field, value):
    a_settings = dict(width=8, lr=1, tuning_grid=[1, 2], selection_rule='mean-dev-final')
    b_settings = dict(a_settings, width=16)
    (a_settings if side == 'left' else b_settings)[field] = value
    a = replace(run(), phase='retuned', settings_json=canonical_settings(a_settings))
    b = replace(a, run_id='b', settings_json=canonical_settings(b_settings))
    with pytest.raises(ValueError):
        require_fair_pair(a, b, factor='width', regime='retuned')


@pytest.mark.parametrize('before,after', [(False, 0), (True, 1), ({'nested': [False]}, {'nested': [0]})])
def test_fixed_comparison_rejects_hidden_json_type_change(before, after):
    a = replace(run(), settings_json=canonical_settings(dict(width=8, use_norm=before)))
    b = replace(a, run_id='b', settings_json=canonical_settings(dict(width=16, use_norm=after)))
    with pytest.raises(ValueError, match='settings'):
        require_fair_pair(a, b, factor='width', regime='fixed')


def test_declared_boolean_factor_can_change_without_hidden_changes():
    a = replace(run(), settings_json=canonical_settings(dict(width=8, use_norm=True)))
    b = replace(a, run_id='b', settings_json=canonical_settings(dict(width=8, use_norm=False)))
    require_fair_pair(a, b, factor='use_norm', regime='fixed')


def test_recommended_grid_metadata_has_exact_fits_steps_and_checkpoint_eval_caps():
    # Protocol design metadata only: no data, forward pass, model, or training.
    groups = (('A', 9, 512), ('B', 3, 384), ('C', 6, 256), ('D', 12, 256),
              ('E', 15, 512), ('F', 30, 256), ('G', 6, 256), ('P', 1, 8))
    fit_steps = {f'{group}{index}': steps for group, count, steps in groups for index in range(count)}
    assert len(fit_steps) == 82
    assert sum(fit_steps.values()) == 27_272
    assert sum(fit_steps.values()) * 2048 == 55_853_056
    scientific = {name: steps for name, steps in fit_steps.items() if not name.startswith('P')}
    curves = [(name, checkpoint, split, 8192) for name, steps in scientific.items()
              for checkpoint in range(0, steps + 1, 128) for split in ('TRAIN', 'DEV')]
    final_test = [(name, steps, 'TEST', 8192) for name, steps in scientific.items()]
    # A indices3..5 designate width128's three seeds. All15 E fits need prefix256.
    retuning_test = [(f'A{index}', 256, 'TEST', 8192) for index in (3, 4, 5)]
    retuning_test += [(f'E{index}', 256, 'TEST', 8192) for index in range(15)]
    test_targets = final_test + retuning_test
    assert len(test_targets) == len(set(test_targets)) == 99
    assert all(checkpoint <= fit_steps[name] for name, checkpoint, _, _ in test_targets)
    assert len(curves) == 588
    # Landscapes:162grid+10directional evaluation calls for each of3base seeds.
    landscape_targets = 516 * 4096
    assert len(curves) + 1 + len(test_targets) + 516 == 1204
    assert sum(row[3] for row in curves + test_targets) + 8192 + landscape_targets == 7_749_632


def test_checkpoint_declaration_is_part_of_cache_identity_without_claiming_file_validation():
    p = protocol()
    p = replace(p, runs=(replace(p.runs[0], settings_json=canonical_settings(dict(
        width=8, eval_checkpoints={'DEV': [0, 3], 'TEST': [3]}))),))
    cache = CacheIdentity.for_run(p, 'a')
    changed = replace(p, runs=(replace(p.runs[0], settings_json=canonical_settings(dict(
        width=8, eval_checkpoints={'DEV': [0, 3], 'TEST': [2, 3]}))),))
    with pytest.raises(ValueError, match='stale'):
        cache.require_match(changed, 'a')
