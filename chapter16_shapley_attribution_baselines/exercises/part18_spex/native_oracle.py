"""Audited, request-only storage and CPU GPT-2 inference for the visible mask game.

Construction hashes inputs but never loads weights. Native loading and every query
require a current source-bound APPROVED freeze. Attribution and recovery methods
do not belong here; the learner supplies the word-masking implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import dis
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import types
from typing import Any

import numpy as np

from . import native_data


ROOT = Path(__file__).resolve().parents[3]
MAX_BATCH = 256
VOCAB_SIZE = 50257
REPLACEMENT_ID = 50256
ATOL = 1e-5
RTOL = 1e-6
MAX_OUTPUT_BYTES = 8 * 1024**3
SCHEMA = "arena.spex.native_oracle.v1"
INVARIANCE_PLAN = {
    "prompt_index": 0,
    "mask_order": ["empty", "full", "alternating_feature_zero_retained"],
    "calls": [
        {"name": "reduced_batch", "rows": 3, "head": "two_target", "replacement_id": 50256},
        {"name": "identical_repeat", "rows": 3, "head": "two_target", "replacement_id": 50256},
        {"name": "singleton_0", "rows": 1, "head": "two_target", "replacement_id": 50256},
        {"name": "singleton_1", "rows": 1, "head": "two_target", "replacement_id": 50256},
        {"name": "singleton_2", "rows": 1, "head": "two_target", "replacement_id": 50256},
        {"name": "replacement_zero", "rows": 3, "head": "two_target", "replacement_id": 0},
        {"name": "full_head", "rows": 3, "head": "full", "replacement_id": 50256},
    ],
    "physical_evaluations": 15,
    "atol": ATOL,
    "rtol": RTOL,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, slice):
        return {"slice": [_jsonable(value.start), _jsonable(value.stop), _jsonable(value.step)]}
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Cannot bind identity value {type(value).__name__}.")


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _code_identity(code: types.CodeType) -> dict:
    """Normalize compiler call scaffolding and locations, retaining the control flow.

    Python 3.14 emits LOAD_ATTR+PUSH_NULL for a known module import but an
    equivalent method-call LOAD_ATTR inside notebook compilation. Attribute and
    global arg values identify the operation; stack-call scaffolding and branch
    byte offsets do not identify scientific semantics.
    """
    constants = [
        {"code": _code_identity(value)} if isinstance(value, types.CodeType)
        else {"bytes": value.hex()} if isinstance(value, bytes)
        else {"slice": [value.start, value.stop, value.step]} if isinstance(value, slice)
        else {"ellipsis": True} if value is Ellipsis
        else _jsonable(value)
        for value in code.co_consts
    ]
    instructions = [item for item in dis.get_instructions(code)
                    if item.opname not in {"CACHE", "RESUME", "PUSH_NULL", "NOP", "NOT_TAKEN"}]

    def target_index(offset):
        return next((index for index, item in enumerate(instructions) if item.offset >= offset), len(instructions))

    operations = []
    for item in instructions:
        argument = item.argval
        if item.opcode in dis.hasjump:
            argument = {"instruction_target": target_index(argument)}
        elif isinstance(argument, types.CodeType):
            argument = {"code": _code_identity(argument)}
        operations.append([item.opname, _jsonable(argument)])
    exceptions = [[target_index(item.start), target_index(item.end), target_index(item.target), item.depth, item.lasti]
                  for item in dis.Bytecode(code).exception_entries]
    return {
        "instructions": operations, "constants": constants,
        "names": code.co_names, "varnames": code.co_varnames,
        "arguments": [code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount],
        "flags": code.co_flags & 0xFFFF, "freevars": code.co_freevars, "cellvars": code.co_cellvars,
        "exception_table": exceptions,
    }


def masking_identity(function) -> dict:
    """Bind Python code, defaults and its referenced non-builtin globals."""
    if not inspect.isfunction(function) or function.__closure__:
        raise TypeError("The student masker must be a Python function without a closure.")
    globals_used = {}
    for name in function.__code__.co_names:
        if name not in function.__globals__:
            continue
        value = function.__globals__[name]
        if isinstance(value, types.ModuleType):
            globals_used[name] = {"module": value.__name__, "version": getattr(value, "__version__", None)}
        else:
            globals_used[name] = _jsonable(value)
    return {"code": _code_identity(function.__code__), "defaults": function.__defaults__,
            "kwdefaults": function.__kwdefaults__, "globals": globals_used}


def _record(value: Mapping) -> dict:
    result = json.loads(_canonical(value))
    for name in ("input_ids", "token_feature", "target_ids"):
        array = np.asarray(value[name])
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be an integer vector.")
        result[name] = array.astype(np.int64, copy=True)
    ids, groups, targets = (result[name] for name in ("input_ids", "token_feature", "target_ids"))
    if not len(ids) or len(ids) > 1024 or groups.shape != ids.shape:
        raise ValueError("Require aligned nonempty token vectors within GPT-2 positions.")
    if np.any(ids < 0) or np.any(ids >= VOCAB_SIZE):
        raise ValueError("Input token outside the GPT-2 vocabulary.")
    if targets.shape != (2,) or targets[0] == targets[1] or np.any(targets < 0) or np.any(targets >= VOCAB_SIZE):
        raise ValueError("Require two distinct valid target tokens.")
    if set(groups.tolist()) != set(range(native_data.N_FEATURES)) | {-1}:
        raise ValueError("Every declared word and the fixed suffix must be represented.")
    ordered = np.where(groups == -1, native_data.N_FEATURES, groups)
    if groups[-1] != -1 or np.any(np.diff(ordered) < 0):
        raise ValueError("Word token groups must be ordered with the suffix last.")
    if "offset_mapping" in value:
        offsets = np.asarray(value["offset_mapping"])
        if offsets.shape != (len(ids), 2) or not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError("Offsets must contain an integer span for each token.")
        result["offset_mapping"] = offsets.astype(np.int64, copy=True)
    if "counterfactual" in value:
        result["counterfactual"] = _record(value["counterfactual"])
    return result


class _GPT2Backend:
    """Loaded only after Oracle's approval guard; no constructor-level model import."""

    kind = "native_gpt2_cpu_float32"

    def __init__(self, snapshot: Path):
        import torch
        from transformers import GPT2LMHeadModel

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Native execution requires CUDA_VISIBLE_DEVICES=''.")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            if os.environ.get(name) != "4":
                raise RuntimeError(f"Native execution requires {name}=4.")
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.model = GPT2LMHeadModel.from_pretrained(
            str(snapshot), local_files_only=True, dtype=torch.float32,
            attn_implementation="eager",
        ).cpu().float().eval()
        config = self.model.config
        if (config.model_type, config.n_layer, config.n_embd, config.vocab_size) != ("gpt2", 12, 768, VOCAB_SIZE):
            raise ValueError("The approved native architecture must be the declared GPT-2 checkpoint.")
        if config._attn_implementation != "eager":
            raise ValueError("Explicit eager causal attention is required.")
        self.model.config.use_cache = False
        self.model.requires_grad_(False)

    def predict(self, inputs: dict, target_ids: np.ndarray, *, full_head: bool = False) -> np.ndarray:
        torch = self.torch
        tensors = {name: torch.from_numpy(np.array(value, copy=True)).long() for name, value in inputs.items()}
        targets = torch.from_numpy(target_ids.copy()).long()
        with torch.inference_mode():
            if full_head:
                logits = self.model(**tensors, use_cache=False, return_dict=True).logits[:, -1, :]
                result = logits.index_select(-1, targets)
            else:
                hidden = self.model.transformer(**tensors, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
                weights = self.model.lm_head.weight.index_select(0, targets)
                result = torch.nn.functional.linear(hidden, weights)
        return result.detach().cpu().numpy()


class Oracle:
    """Request-only disk cache with separate logical and physical cost inventories."""

    def __init__(self, output_dir, tokenized_records, snapshot_path, *,
                 student_mask_inputs=native_data.masked_word_inputs, freeze_path=None):
        self.output_dir = Path(output_dir).resolve()
        self.snapshot_path = Path(snapshot_path).resolve()
        self.records = [_record(value) for value in tokenized_records]
        if not self.records:
            raise ValueError("At least one prospectively bound prompt is required.")
        names = [record.get("id", str(index)) for index, record in enumerate(self.records)]
        if len(set(names)) != len(names):
            raise ValueError("Prompt identifiers must be unique.")
        self.student_mask_inputs = student_mask_inputs
        self.identity = {
            "schema": SCHEMA, "model_id": native_data.MODEL_ID,
            "model_revision": native_data.MODEL_REVISION,
            "snapshot": native_data.snapshot_records(self.snapshot_path),
            "tokenized_records": [_jsonable(record) for record in self.records],
            "masking": masking_identity(student_mask_inputs),
            "engine": {"dtype": "float32", "difference_dtype": "float64", "device": "cpu",
                       "attention": "eager_causal", "eval": True, "threads": 8,
                       "max_batch": MAX_BATCH, "new_mask_order": "first_occurrence",
                       "replacement_id": REPLACEMENT_ID},
            "invariance_plan": INVARIANCE_PLAN,
        }
        self.identity_sha256 = _sha(self.identity)
        self._snapshot_stats = self._stats()
        self._backend = None
        self._freeze_path = self._freeze_sha256 = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        identity_path = self.output_dir / "oracle_identity.json"
        database_path = self.output_dir / "oracle.sqlite3"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != _jsonable(self.identity):
                raise ValueError("Existing oracle cache belongs to a different exact identity.")
            if not database_path.is_file():
                raise ValueError("An existing cache identity must retain its database.")
        elif database_path.exists():
            raise ValueError("Refuse a cache database without its exact identity.")
        else:
            identity_path.write_bytes(_canonical(self.identity) + b"\n")
        self._db = sqlite3.connect(database_path)
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute(f"PRAGMA max_page_count={MAX_OUTPUT_BYTES // 4096 - 4096}")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY, receipt TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY, call_id INTEGER NOT NULL REFERENCES calls(id),
                status TEXT NOT NULL, identity TEXT NOT NULL, error TEXT);
            CREATE TABLE IF NOT EXISTS physical_rows (
                id INTEGER PRIMARY KEY, batch_id INTEGER NOT NULL REFERENCES batches(id),
                prompt_index INTEGER NOT NULL, game TEXT NOT NULL, mask_id INTEGER NOT NULL,
                row_in_batch INTEGER NOT NULL, logits BLOB NOT NULL, value REAL NOT NULL,
                checksum TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cache (
                prompt_index INTEGER NOT NULL, game TEXT NOT NULL, mask_id INTEGER NOT NULL,
                physical_id INTEGER NOT NULL REFERENCES physical_rows(id),
                PRIMARY KEY(prompt_index, game, mask_id));
        """)
        self._db.commit()
        self._check_database()
        if freeze_path is not None:
            self.authorize(freeze_path)

    def _stats(self):
        return {name: (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                for name in self.identity["snapshot"] for path in [self.snapshot_path / name]}

    def _check_database(self):
        if self._db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Oracle database integrity check failed.")
        if (self.output_dir / "execution_blocked.json").exists():
            raise RuntimeError("A failed numerical check is retained; do not silently resume.")
        if self._db.execute("SELECT COUNT(*) FROM batches WHERE status!='PASS'").fetchone()[0]:
            raise RuntimeError("A partial/failed inference batch is retained; do not silently resume.")
        if self._db.execute("SELECT COUNT(*) FROM calls WHERE json_extract(receipt,'$.status')!='PASS'").fetchone()[0]:
            raise RuntimeError("A partial/failed oracle request is retained; do not silently resume.")

    def authorize(self, freeze_path):
        """Bind an immutable approval; later source or freeze changes revoke execution."""
        self._freeze_path = Path(freeze_path).resolve()
        self._freeze_sha256 = _file_sha(self._freeze_path)
        self._approval()
        return self

    def _approval(self):
        if self._freeze_path is None or _file_sha(self._freeze_path) != self._freeze_sha256:
            raise PermissionError("Native queries require the unchanged source-bound approved freeze.")
        freeze = json.loads(self._freeze_path.read_text())
        if freeze.get("status") != "APPROVED" or freeze.get("oracle_identity_sha256") != self.identity_sha256:
            raise PermissionError("The freeze does not approve this exact oracle identity.")
        if freeze.get("native_queries_before_freeze") != 0 or freeze.get("native_fits_before_freeze") != 0:
            raise PermissionError("The freeze must precede native queries and fits.")
        sources = freeze.get("source_inputs", {})
        required = {str(Path(__file__).resolve().relative_to(ROOT)), str(Path(native_data.__file__).resolve().relative_to(ROOT))}
        if not isinstance(sources, dict) or not required <= set(sources):
            raise PermissionError("The freeze must bind the oracle and masking source files.")
        for name, checksum in sources.items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or _file_sha(path) != checksum:
                raise PermissionError(f"Approved source changed: {name}.")
        if _sha(masking_identity(self.student_mask_inputs)) != _sha(self.identity["masking"]):
            raise PermissionError("The called student masking implementation changed.")
        if self._stats() != self._snapshot_stats:
            if native_data.snapshot_records(self.snapshot_path) != self.identity["snapshot"]:
                raise PermissionError("Approved physical model/tokenizer files changed.")
            self._snapshot_stats = self._stats()
        if _sha(self.identity) != self.identity_sha256:
            raise PermissionError("In-memory oracle identity changed.")
        if _sha([_jsonable(record) for record in self.records]) != _sha(self.identity["tokenized_records"]):
            raise PermissionError("Bound tokenized records changed.")

    def _make_backend(self):
        return _GPT2Backend(self.snapshot_path)

    def _get_backend(self):
        self._approval()
        if self._backend is None:
            if native_data.snapshot_records(self.snapshot_path) != self.identity["snapshot"]:
                raise PermissionError("Physical model files changed before weight loading.")
            self._backend = self._make_backend()
        return self._backend

    def _prompt(self, index):
        if type(index) is not int or not 0 <= index < len(self.records):
            raise ValueError("prompt_index must select a bound prompt.")
        return self.records[index]

    @staticmethod
    def _masks(masks):
        masks = np.asarray(masks)
        if masks.ndim != 2 or masks.shape[1] != native_data.N_FEATURES or not np.isin(masks, (0, 1)).all():
            raise ValueError("masks must have 28 binary feature columns.")
        return masks.astype(np.int64, copy=True)

    def _inputs(self, record, masks, replacement_id):
        result = self.student_mask_inputs(record["input_ids"].copy(), record["token_feature"].copy(), masks.copy(), replacement_id=replacement_id)
        expected = native_data.masked_word_inputs(record["input_ids"], record["token_feature"], masks, replacement_id)
        if not isinstance(result, Mapping) or set(result) != set(expected):
            raise ValueError("The student masker must return only input_ids, attention_mask and position_ids.")
        for name, value in expected.items():
            actual = np.asarray(result[name])
            if not np.issubdtype(actual.dtype, np.integer) or actual.shape != value.shape or not np.array_equal(actual, value):
                raise ValueError(f"Student masking violates fixed-position semantics: {name}.")
        return {name: np.array(value, dtype=np.int64, copy=True) for name, value in result.items()}

    @staticmethod
    def _metadata(stage, method, seed, budget):
        if not isinstance(stage, str) or not stage.strip() or not isinstance(method, str) or not method.strip():
            raise ValueError("Every call needs explicit stage and method labels.")
        if type(seed) is not int or seed < 0 or type(budget) is not int or budget < 0:
            raise ValueError("seed and budget must be nonnegative integers.")
        return {"stage": stage, "method": method, "seed": seed, "budget": budget}

    def _physical(self, call_id, prompt_index, game, record, masks, *, head="two_target", replacement_id=REPLACEMENT_ID):
        inputs = self._inputs(record, masks, replacement_id)
        backend = self._get_backend()
        mask_ids = native_data.mask_ids(masks).astype(np.int64)
        identity = {"oracle_identity_sha256": self.identity_sha256, "game": game,
                    "prompt_index": prompt_index, "mask_ids": mask_ids.tolist(),
                    "head": head, "replacement_id": replacement_id, "backend": backend.kind,
                    "inputs_sha256": _sha(inputs), "target_ids": record["target_ids"].tolist(),
                    "rows": len(masks), "batch_size_limit": MAX_BATCH}
        with self._db:
            cursor = self._db.execute("INSERT INTO batches(call_id,status,identity) VALUES (?,'RUNNING',?)", (call_id, _canonical(identity).decode()))
            batch_id = cursor.lastrowid
        try:
            logits = np.asarray(backend.predict(inputs, record["target_ids"], full_head=head == "full"))
            if logits.shape != (len(masks), 2) or logits.dtype != np.float32 or not np.isfinite(logits).all():
                raise ValueError("Native backend must return exactly two finite float32 logits per requested row.")
            values = logits[:, 0].astype(np.float64) - logits[:, 1].astype(np.float64)
            physical_ids = []
            with self._db:
                for row, (mask_id, pair, value) in enumerate(zip(mask_ids, logits, values, strict=True)):
                    packed = np.asarray(pair, dtype="<f4").tobytes()
                    checksum = _sha([batch_id, prompt_index, game, int(mask_id), row, packed.hex(), float(value)])
                    cursor = self._db.execute(
                        "INSERT INTO physical_rows(batch_id,prompt_index,game,mask_id,"
                        "row_in_batch,logits,value,checksum) VALUES (?,?,?,?,?,?,?,?)",
                        (batch_id, prompt_index, game, int(mask_id), row, packed, float(value), checksum),
                    )
                    physical_ids.append(cursor.lastrowid)
                self._db.execute("UPDATE batches SET status='PASS' WHERE id=?", (batch_id,))
            return logits.copy(), values, physical_ids
        except BaseException as error:
            with self._db:
                self._db.execute("UPDATE batches SET status='FAIL',error=? WHERE id=?", (f"{type(error).__name__}: {error}", batch_id))
            raise

    def _cached(self, prompt_index, game, mask_id):
        result = self._db.execute(
            "SELECT p.id,p.batch_id,p.prompt_index,p.game,p.mask_id,p.row_in_batch,"
            "p.logits,p.value,p.checksum FROM cache c JOIN physical_rows p ON p.id=c.physical_id "
            "WHERE c.prompt_index=? AND c.game=? AND c.mask_id=?",
            (prompt_index, game, mask_id),
        ).fetchone()
        if result is None:
            return None
        _, batch, prompt, stored_game, stored_mask, row, packed, value, checksum = result
        if (prompt, stored_game, stored_mask) != (prompt_index, game, mask_id):
            raise ValueError("A cached row points to another prompt/game/mask.")
        if _sha([batch, prompt, stored_game, stored_mask, row, packed.hex(), value]) != checksum:
            raise ValueError("Cached physical row checksum mismatch.")
        pair = np.frombuffer(packed, dtype="<f4")
        if pair.shape != (2,) or not np.isfinite(pair).all() or value != float(np.float64(pair[0]) - np.float64(pair[1])):
            raise ValueError("Cached target logits and promoted difference disagree.")
        return value

    def _query(self, prompt_index, record, game, masks, metadata, *, force=False, head="two_target", replacement_id=REPLACEMENT_ID):
        self._approval()
        self._check_database()
        ids = [int(value) for value in native_data.mask_ids(masks)]
        unique = list(dict.fromkeys(ids))
        before = {mask_id: self._cached(prompt_index, game, mask_id) for mask_id in unique} if not force else dict.fromkeys(unique)
        new = [mask_id for mask_id in unique if before[mask_id] is None]
        receipt = {**metadata, "status": "RUNNING", "prompt_index": prompt_index, "game": game,
                   "ordered_mask_ids": ids, "raw_count": len(ids), "unique_count": len(unique),
                   "cached_unique_before_call": sum(value is not None for value in before.values()),
                   "cache_hits_before_call": sum(before[mask_id] is not None for mask_id in ids),
                   "within_call_duplicate_count": len(ids) - len(unique), "new_physical_count": 0,
                   "forced_probe": force, "head": head, "replacement_id": replacement_id,
                   "oracle_identity_sha256": self.identity_sha256, "freeze_sha256": self._freeze_sha256,
                   "started_at_utc": datetime.now(timezone.utc).isoformat()}
        with self._db:
            cursor = self._db.execute("INSERT INTO calls(receipt) VALUES (?)", (_canonical(receipt).decode(),))
            call_id = cursor.lastrowid
        receipt["call_id"] = call_id
        positions = {}
        for index, mask_id in enumerate(ids):
            positions.setdefault(mask_id, index)
        requested = list(range(len(ids))) if force else [positions[mask_id] for mask_id in new]
        returned = []
        try:
            for start in range(0, len(requested), MAX_BATCH):
                selected = requested[start:start + MAX_BATCH]
                logits, values, physical_ids = self._physical(call_id, prompt_index, game, record, masks[selected], head=head, replacement_id=replacement_id)
                receipt["new_physical_count"] += len(selected)
                returned.extend(logits.tolist())
                if not force:
                    with self._db:
                        for position, value, physical_id in zip(selected, values, physical_ids, strict=True):
                            self._db.execute("INSERT INTO cache VALUES (?,?,?,?)", (prompt_index, game, ids[position], physical_id))
                            before[ids[position]] = float(value)
            receipt.update(status="PASS", cache_hit_count=len(ids) - receipt["new_physical_count"],
                           finished_at_utc=datetime.now(timezone.utc).isoformat())
            if sum(path.stat().st_size for path in self.output_dir.rglob('*') if path.is_file()) > MAX_OUTPUT_BYTES:
                raise RuntimeError("Oracle retained output exceeded the frozen 8 GiB cap.")
        except BaseException as error:
            receipt.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            with self._db:
                self._db.execute("UPDATE calls SET receipt=? WHERE id=?", (_canonical(receipt).decode(), call_id))
        if force:
            return np.asarray(returned, dtype=np.float32).reshape(-1, 2)
        return np.asarray([before[mask_id] for mask_id in ids], dtype=np.float64)

    def query(self, prompt_index, masks, *, stage, method, seed, budget):
        """Return values in exact requested order, preserving logical duplicates."""
        record = self._prompt(prompt_index)
        masks = self._masks(masks)
        metadata = self._metadata(stage, method, seed, budget)
        return self._query(prompt_index, record, "primary", masks, metadata)

    def counterfactual(self, prompt_index, tokenized_record=None, *, stage="counterfactual", method="control", seed=0, budget=1):
        """Evaluate the bound subject-only counterfactual, keeping target orientation."""
        primary = self._prompt(prompt_index)
        if "counterfactual" not in primary:
            raise ValueError("The counterfactual must be tokenized and bound before the freeze.")
        record = primary["counterfactual"]
        if tokenized_record is not None and _sha(_record(tokenized_record)) != _sha(record):
            raise ValueError("The supplied counterfactual differs from the frozen record.")
        subject = primary.get("subject_word_index")
        if type(subject) is not int or not 0 <= subject < native_data.N_FEATURES:
            raise ValueError("The primary record must identify its prospective subject word.")
        if not np.array_equal(primary["token_feature"], record["token_feature"]) or not np.array_equal(primary["target_ids"], record["target_ids"]):
            raise ValueError("Counterfactual positions, groups and target orientation must stay fixed.")
        changed = primary["input_ids"] != record["input_ids"]
        if not np.any(changed) or np.any(changed & (primary["token_feature"] != subject)):
            raise ValueError("Only the final subject word may change in the counterfactual.")
        subject_tokens = primary["token_feature"] == subject
        if subject_tokens.sum() != 1 or primary["input_ids"][subject_tokens][0] != primary["target_ids"][1] or record["input_ids"][subject_tokens][0] != primary["target_ids"][0]:
            raise ValueError("The single-token final subject must change from the second target name to the first.")
        masks = np.ones((1, native_data.N_FEATURES), dtype=np.int64)
        return self._query(prompt_index, record, "counterfactual", masks, self._metadata(stage, method, seed, budget))

    def validate_invariance(self, prompt_index=0, masks=None):
        """Run the exact fifteen-row prospective probe table; retain failed outcomes."""
        expected = np.stack([
            np.zeros(native_data.N_FEATURES, dtype=np.int64),
            np.ones(native_data.N_FEATURES, dtype=np.int64),
            (np.arange(native_data.N_FEATURES) % 2 == 0).astype(np.int64),
        ])
        if prompt_index != 0 or (masks is not None and not np.array_equal(self._masks(masks), expected)):
            raise ValueError("Invariance probes are fixed prospectively to prompt zero and three masks.")
        record = self._prompt(0)
        observations = {}
        for spec in INVARIANCE_PLAN["calls"]:
            name = spec["name"]
            selected = expected[int(name[-1]):int(name[-1]) + 1] if name.startswith("singleton_") else expected
            observations[name] = self._query(0, record, "invariance/" + name, selected,
                self._metadata("invariance", name, 0, len(selected)), force=True,
                head=spec["head"], replacement_id=spec["replacement_id"])
        baseline = observations["reduced_batch"].astype(np.float64)
        comparisons = {}
        for name, actual in observations.items():
            reference = baseline[int(name[-1]):int(name[-1]) + 1] if name.startswith("singleton_") else baseline
            comparisons[name] = {"passed": bool(np.allclose(actual, reference, atol=ATOL, rtol=RTOL)),
                                 "max_absolute_error": float(np.max(np.abs(actual.astype(np.float64) - reference)))}
        result = {"status": "PASS" if all(item["passed"] for item in comparisons.values()) else "FAIL",
                  "plan": INVARIANCE_PLAN, "comparisons": comparisons,
                  "two_target_logits": {name: values.tolist() for name, values in observations.items()},
                  "physical_evaluations": 15, "oracle_identity_sha256": self.identity_sha256,
                  "freeze_sha256": self._freeze_sha256}
        path = self.output_dir / f'invariance_{len(list(self.output_dir.glob("invariance_*.json"))):03d}.json'
        path.write_bytes(_canonical(result) + b"\n")
        if result["status"] != "PASS":
            (self.output_dir / "execution_blocked.json").write_bytes(_canonical({
                "reason": "frozen_invariance_failure", "receipt": path.name,
                "receipt_sha256": _file_sha(path), "oracle_identity_sha256": self.identity_sha256,
            }) + b"\n")
            raise ValueError(f"Frozen native invariance check failed; results retained at {path}.")
        return result

    @property
    def receipts(self):
        return [json.loads(row[0]) for row in self._db.execute("SELECT receipt FROM calls ORDER BY id")]

    def physical_rows(self, prompt_index=None):
        """Stream every physical row, including repeats, full-head and counterfactuals."""
        where, parameters = ("", ()) if prompt_index is None else (" WHERE prompt_index=?", (prompt_index,))
        for row in self._db.execute("SELECT id,batch_id,prompt_index,game,mask_id,row_in_batch,logits,value,checksum FROM physical_rows" + where + " ORDER BY id", parameters):
            physical_id, batch_id, prompt, game, mask_id, position, packed, value, checksum = row
            if _sha([batch_id, prompt, game, mask_id, position, packed.hex(), value]) != checksum:
                raise ValueError("Physical row inventory checksum mismatch.")
            yield {"physical_id": physical_id, "batch_id": batch_id, "prompt_index": prompt,
                   "game": game, "mask_id": mask_id, "row_in_batch": position,
                   "target_logits_float32": np.frombuffer(packed, dtype="<f4").tolist(),
                   "difference_float64": value, "checksum": checksum}

    def inventory(self):
        """Snapshot durable row counts and byte hashes without materializing all masks."""
        self._db.commit()
        batches = [{"batch_id": row[0], "call_id": row[1], "status": row[2],
                    "identity": json.loads(row[3]), "error": row[4]}
                   for row in self._db.execute("SELECT id,call_id,status,identity,error FROM batches ORDER BY id")]
        per_prompt = [{"prompt_index": row[0], "game": row[1], "physical_rows": row[2]}
                      for row in self._db.execute("SELECT prompt_index,game,COUNT(*) FROM physical_rows GROUP BY prompt_index,game ORDER BY prompt_index,game")]
        database = self.output_dir / "oracle.sqlite3"
        result = {"schema": SCHEMA, "oracle_identity_sha256": self.identity_sha256,
                  "physical_evaluations": self._db.execute("SELECT COUNT(*) FROM physical_rows").fetchone()[0],
                  "attempted_physical_evaluations": sum(item["identity"]["rows"] for item in batches),
                  "cached_unique_masks": self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0],
                  "calls": self.receipts, "batches": batches, "per_prompt": per_prompt,
                  "database": {"path": str(database), "sha256": _file_sha(database), "size_bytes": database.stat().st_size}}
        (self.output_dir / "oracle_inventory.json").write_bytes(_canonical(result) + b"\n")
        return result

    def export_values(self):
        """Stable comparison view; raw inventories retain paths, times and freeze hashes.

        Physical rows include every probe and counterfactual, in producing order.
        Requests retain logical duplicates. Compare IDs/order/metadata exactly and
        native float32 logits using the prospectively frozen native tolerances.
        """
        rows = list(self.physical_rows())
        calls = self.receipts
        call_by_id = {call["call_id"]: call for call in calls}
        batches = [{"batch_id": row[0], "call_id": row[1], "status": row[2],
                    "identity": json.loads(row[3]), "error": row[4]}
                   for row in self._db.execute("SELECT id,call_id,status,identity,error FROM batches ORDER BY id")]
        batch_by_id = {batch["batch_id"]: batch for batch in batches}
        physical = {
            name: np.asarray([row[name] for row in rows], dtype=np.int64)
            for name in ("physical_id", "batch_id", "prompt_index", "mask_id", "row_in_batch")
        }
        physical.update(
            target_logits_float32=np.asarray([row["target_logits_float32"] for row in rows], dtype=np.float32).reshape(-1, 2),
            difference_float64=np.asarray([row["difference_float64"] for row in rows], dtype=np.float64),
            game=np.asarray([row["game"] for row in rows], dtype=str),
            stage=np.asarray([call_by_id[batch_by_id[row["batch_id"]]["call_id"]]["stage"] for row in rows], dtype=str),
            method=np.asarray([call_by_id[batch_by_id[row["batch_id"]]["call_id"]]["method"] for row in rows], dtype=str),
        )
        omitted = {"started_at_utc", "finished_at_utc", "freeze_sha256"}
        return {"schema": SCHEMA, "oracle_identity_sha256": self.identity_sha256,
                "physical": physical, "batches": batches,
                "requests": [{key: value for key, value in call.items() if key not in omitted} for call in calls],
                "comparison_omissions": sorted(omitted),
                "raw_receipts_retained_in": "oracle.sqlite3 and oracle_inventory.json"}

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
