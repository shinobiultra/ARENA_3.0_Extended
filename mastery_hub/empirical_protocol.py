"""Draft, pure metadata guards, not an executor or an authorization mechanism.

No model, corpus, cache file, network, tensor library, or GPU is opened here.
Frozen reservations conservatively charge full runs, including failed attempts.
A reviewed scientific protocol and human launch/resource authority remain external.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _json(raw: str) -> Any:
    try:
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        # Reject overflow such as 1e999 as well as explicit NaN/Infinity.
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, json.JSONDecodeError, OverflowError) as error:
        raise ValueError('invalid finite JSON') from error


def canonical_settings(value: dict[str, Any] | str) -> str:
    """Canonical immutable settings, strict JSON objects with no duplicate keys."""
    try:
        parsed = _json(value) if isinstance(value, str) else _json(json.dumps(value, allow_nan=False))
        _require(type(parsed) is dict, 'settings must be a JSON object')
        return json.dumps(parsed, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    except (TypeError, OverflowError) as error:
        raise ValueError('invalid settings') from error


def _digest(value: str) -> None:
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'invalid SHA-256')


def _id(value: str) -> None:
    _require(type(value) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', value) is not None,
             'invalid identifier')


def _construct(cls: Any, payload: Any) -> Any:
    _require(type(payload) is dict and set(payload) == {f.name for f in fields(cls)},
             f'unknown or missing {cls.__name__} fields')
    return cls(**payload)


@dataclass(frozen=True)
class Budget:
    """Integer hard reservations; storage sums bytes written, including failures.

    CPU core-seconds count all workers. GPU seconds include compilation/evaluation.
    Peak memory is guarded by a future executor, not inferred from these counters.
    """
    wall_seconds: int
    gpu_seconds: int
    cpu_core_seconds: int
    steps: int
    eval_calls: int
    train_tokens: int
    eval_tokens: int
    storage_bytes: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            _require(type(value) is int and value >= 0, f'invalid budget {field.name}')

    def plus(self, other: Budget) -> Budget:
        return Budget(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def require_within(self, cap: Budget) -> None:
        for field in fields(self):
            _require(getattr(self, field.name) <= getattr(cap, field.name),
                     f'budget exceeded: {field.name}')


@dataclass(frozen=True)
class SplitItem:
    split: str
    group_id: str
    source_id: str
    document_sha256: str

    def __post_init__(self) -> None:
        _require(self.split in ('TRAIN', 'DEV', 'TEST'), 'unknown split')
        _id(self.group_id)
        _id(self.source_id)
        _digest(self.document_sha256)


def validate_splits(items: tuple[SplitItem, ...]) -> str:
    """Check metadata only, not semantic/near-duplicate leakage or actual bytes."""
    _require(type(items) is tuple and all(type(item) is SplitItem for item in items), 'invalid split items')
    _require({item.split for item in items} == {'TRAIN', 'DEV', 'TEST'}, 'require TRAIN, DEV, TEST')
    _require(len(set(items)) == len(items), 'duplicate split item')
    for axis in ('group_id', 'source_id', 'document_sha256'):
        seen: dict[str, str] = {}
        for item in items:
            identity = getattr(item, axis)
            _require(identity not in seen or seen[identity] == item.split, f'split leakage: {axis}')
            seen[identity] = item.split
    manifest = [asdict(item) for item in sorted(items, key=lambda item: (
        item.split, item.group_id, item.source_id, item.document_sha256))]
    return sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    phase: str
    seed: int
    settings_json: str
    budget: Budget

    def __post_init__(self) -> None:
        _id(self.run_id)
        _require(self.phase in ('fixed', 'retuned', 'control', 'landscape', 'evaluation', 'preflight'),
                 'unknown phase')
        _require(type(self.seed) is int and self.seed >= 0, 'invalid seed')
        _require(type(self.settings_json) is str and self.settings_json == canonical_settings(self.settings_json),
                 'settings must be canonical JSON')
        _require(type(self.budget) is Budget, 'invalid budget')


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    source_sha256: str
    model_sha256: str
    environment_sha256: str
    split_sha256: str
    runs: tuple[RunPlan, ...]
    cap: Budget

    def __post_init__(self) -> None:
        _id(self.protocol_id)
        for name in ('source_sha256', 'model_sha256', 'environment_sha256', 'split_sha256'):
            _digest(getattr(self, name))
        _require(type(self.runs) is tuple and bool(self.runs)
                 and all(type(run) is RunPlan for run in self.runs), 'runs must be immutable RunPlans')
        _require(len({run.run_id for run in self.runs}) == len(self.runs), 'duplicate run id')
        _require(type(self.cap) is Budget, 'invalid cap')

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'), allow_nan=False)

    @classmethod
    def from_json(cls, raw: str) -> Protocol:
        value = _json(raw)
        _require(type(value) is dict and set(value) == {f.name for f in fields(cls)},
                 'unknown or missing protocol fields')
        _require(type(value['runs']) is list, 'invalid runs')
        runs = []
        for item in value['runs']:
            _require(type(item) is dict and set(item) == {f.name for f in fields(RunPlan)},
                     'unknown or missing run fields')
            item['budget'] = _construct(Budget, item['budget'])
            runs.append(_construct(RunPlan, item))
        value['runs'] = tuple(runs)
        value['cap'] = _construct(Budget, value['cap'])
        return cls(**value)

    @property
    def fingerprint(self) -> str:
        return sha256(self.to_json().encode()).hexdigest()

    def run(self, run_id: str) -> RunPlan:
        for run in self.runs:
            if run.run_id == run_id:
                return run
        raise ValueError('unknown run id')


@dataclass(frozen=True)
class Ledger:
    protocol_sha256: str
    reserved_run_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.protocol_sha256)
        _require(type(self.reserved_run_ids) is tuple, 'immutable reservations required')
        for run_id in self.reserved_run_ids:
            _id(run_id)
        _require(len(set(self.reserved_run_ids)) == len(self.reserved_run_ids), 'duplicate reservations')

    def consumed(self, protocol: Protocol) -> Budget:
        _require(self.protocol_sha256 == protocol.fingerprint, 'stale ledger')
        total = Budget(0, 0, 0, 0, 0, 0, 0, 0)
        for run_id in self.reserved_run_ids:
            total = total.plus(protocol.run(run_id).budget)
        return total

    def reserve(self, protocol: Protocol, run_id: str) -> Ledger:
        total = self.consumed(protocol)
        run = protocol.run(run_id)
        _require(run_id not in self.reserved_run_ids, 'run already reserved; failures do not release budget')
        total.plus(run.budget).require_within(protocol.cap)
        return Ledger(self.protocol_sha256, self.reserved_run_ids + (run_id,))


@dataclass(frozen=True)
class CacheIdentity:
    protocol_sha256: str
    run_id: str

    def __post_init__(self) -> None:
        _digest(self.protocol_sha256)
        _id(self.run_id)

    @classmethod
    def for_run(cls, protocol: Protocol, run_id: str) -> CacheIdentity:
        protocol.run(run_id)
        return cls(protocol.fingerprint, run_id)

    def require_match(self, protocol: Protocol, run_id: str) -> None:
        _require(self == self.for_run(protocol, run_id), 'stale cache identity')


def require_fair_pair(left: RunPlan, right: RunPlan, *, factor: str, regime: str) -> None:
    """Guard declared matched-token/step comparisons; never claim matched FLOPs.

    Retuning permits selected LR to differ only with identical declared grid and
    selection rule. A complete equal-effort search still needs all candidate runs
    and failures in the reviewed ledger; this pair check cannot establish that.
    """
    _require(regime in ('fixed', 'retuned'), 'unknown fairness regime')
    _require(left.phase == right.phase == regime, 'mismatched phases')
    _require(left.seed == right.seed, 'mismatched seed')
    for axis in ('steps', 'train_tokens', 'eval_calls', 'eval_tokens'):
        _require(getattr(left.budget, axis) == getattr(right.budget, axis), f'mismatched budget: {axis}')
    a, b = _json(left.settings_json), _json(right.settings_json)
    _require(factor in a and factor in b
             and canonical_settings({'factor': a[factor]}) != canonical_settings({'factor': b[factor]}),
             'factor must change')
    a.pop(factor)
    b.pop(factor)
    if regime == 'retuned':
        def positive_number(value: Any) -> bool:
            return (type(value) is int and value > 0) or (
                type(value) is float and isfinite(value) and value > 0)

        for settings in (a, b):
            grid = settings.get('tuning_grid')
            _require(type(grid) is list and bool(grid)
                     and all(positive_number(candidate) for candidate in grid), 'invalid tuning grid')
            _require(len(set(grid)) == len(grid), 'duplicate tuning grid')
            selected = settings.get('lr')
            _require(positive_number(selected) and selected in grid, 'invalid selected lr or outside tuning grid')
        _require(canonical_settings({'grid': a['tuning_grid']})
                 == canonical_settings({'grid': b['tuning_grid']}), 'mismatched tuning grid')
        _require(type(a.get('selection_rule')) is str and bool(a['selection_rule'].strip())
                 and a.get('selection_rule') == b.get('selection_rule'),
                 'mismatched selection rule')
        a.pop('lr')
        b.pop('lr')
    _require(canonical_settings(a) == canonical_settings(b), 'undeclared settings changes')
