"""Fresh pinned-model verification, separate from the learner's stored results."""
from __future__ import annotations

from pathlib import Path
import tempfile
import time

import torch as t

from . import direction_data as data, direction_methods as method
from .direction_evidence import read_completed_refusal_evidence, sha256
from .direction_final import final_capability_examples
from .run_direction_study import generate_rows, rates, write_json


def compare_generated_rows(fresh, stored):
    """Require exact generated outcomes, with sensitive strings reduced to hashes."""
    expected = {row["id"]: row for row in stored}
    if len(fresh) != len(expected) or {row["id"] for row in fresh} != set(expected):
        raise ValueError("Fresh generation must retain every recorded example")
    keys = ("refusal_label", "refusal_marker", "completion_sha256", "generated_length", "finished")
    for row in fresh:
        old = expected[row["id"]]
        for key in keys:
            if row[key] != old[key]:
                raise ValueError(f"Fresh generated outcome differs for {row['id']} field {key}")
        if row["refusal_label"] is not True:
            if row["completion"] != old["completion"] or row["generated_tokens"].tolist() != old["generated_tokens"]:
                raise ValueError("Fresh benign words/tokens differ from the learner")
        if "exact_correct" in row and row["exact_correct"] != old["exact_correct"]:
            raise ValueError("Fresh exact capability scoring differs from the learner")


def run_fresh_refusal_cuda_check(max_vram_gb=24., *, output_directory=None):
    if not t.cuda.is_available():
        return {"cuda_available": False, "accepted": False,
                "reason": "The real refusal-direction learner requires CUDA"}
    manifest, evidence, audit = read_completed_refusal_evidence()
    if not audit["selection_passed"]:
        raise ValueError("This frozen-final recheck requires a successful development selection")
    output = Path(output_directory) if output_directory is not None else Path(tempfile.mkdtemp(prefix="arena-refusal-cuda-"))
    output.mkdir(parents=True, exist_ok=True)
    selected = evidence["selection"]["selected"]
    write_json(output / "protocol.json", {"scope": "fresh_frozen_final_cuda_recheck",
        "learner_evidence_sha256": manifest["published_evidence_sha256"],
        "selection_sha256": evidence["selection_sha256"], "selected_layer": selected["layer"],
        "selected_offset": selected["offset"], "conditions": 34, "strength": 1.,
        "directions": "target and all first eight random/permuted training draws",
        "generation": "FP32 SDPA, greedy 48 tokens, batch size 8, EOS and KV cache",
        "no_final_retuning": True})
    started = time.perf_counter()
    t.cuda.reset_peak_memory_stats()
    model, tokenizer = data.load_qwen_model(dtype=t.float32)
    train, development = data.load_prompt_split("train"), data.load_prompt_split("development")
    # Match the learner's actual batch padding and cache inputs exactly.
    encoded = data.encode_chat_prompts(tokenizer, [row.text for row in train + development])
    cache = method.cache_qwen_residuals(model, encoded)
    position = (-3,-2,-1).index(selected["offset"])
    nulls = method.direction_null_ensemble(cache[selected["layer"], :64, position], data.prompt_labels(train), samples=32, seed=901)
    error = float((nulls["target"]["difference"] - t.tensor(selected["vector"])).abs().max())
    if error > 1e-4:
        raise ValueError(f"Fresh training-only direction differs from frozen learner: {error}")
    directions = [("target",0,t.tensor(selected["vector"]))]
    directions += [("random",row["sample"],row["difference"]) for row in nulls["random"][:8]]
    for row in nulls["permuted"][:8]:
        if row["fitted"] is None:
            raise ValueError("A prespecified permutation is undefined; do not silently resample")
        directions.append(("permuted",row["sample"],row["fitted"]["difference"]))
    final_records = data.load_prompt_split("final")
    final_encoded = data.encode_chat_prompts(tokenizer,[row.text for row in final_records])
    final_cache = method.cache_qwen_residuals(model,final_encoded)
    values = final_cache[selected["layer"], :, position]
    labels = data.prompt_labels(final_records)
    separation = {"target":method.evaluate_frozen_direction(values,labels,nulls["target"]),
        "random":[method.evaluate_frozen_direction(values,labels,row) for row in nulls["random"]],
        "permuted":[None if row["fitted"] is None else method.evaluate_frozen_direction(values,labels,row["fitted"])
                    for row in nulls["permuted"]]}
    largest_score_error = 0.
    for family in ("target","random","permuted"):
        fresh_rows = [separation[family]] if family == "target" else separation[family]
        old_rows = [evidence["final"]["separation"][family]] if family == "target" else evidence["final"]["separation"][family]
        for fresh,old in zip(fresh_rows,old_rows,strict=True):
            if fresh is None or old is None:
                if fresh is not old:
                    raise ValueError("An undefined permutation changed between executions")
                continue
            score_error = float((fresh["scores"]-t.tensor(old["scores"])).abs().max())
            largest_score_error = max(largest_score_error,score_error)
            if score_error > 1e-4 or fresh["predictions"].tolist() != old["predictions"]:
                raise ValueError("Fresh frozen classification differs from recorded final examples")
    write_json(output / "separation.json", separation)
    prompts = [{"id":f"source-{row.row}","prompt":row.text,"refusal_label":row.refusal} for row in final_records]
    capability = final_capability_examples()
    baseline = generate_rows(model,tokenizer,prompts)
    cap_base = generate_rows(model,tokenizer,capability)
    compare_generated_rows(baseline,evidence["final"]["baseline"])
    compare_generated_rows(cap_base,evidence["final"]["capability_baseline"])
    write_json(output / "baseline.json", {"benchmark":baseline,"capability":cap_base})
    conditions = []
    for kind,sample,vector in directions:
        for mode in ("add","ablate"):
            benchmark = generate_rows(model,tokenizer,prompts,vector=vector,mode=mode,layer=selected["layer"],strength=1.)
            cap = generate_rows(model,tokenizer,capability,vector=vector,mode=mode,layer=selected["layer"],strength=1.)
            old = next(row for row in evidence["final"]["conditions"]
                       if (row["kind"],row["sample"],row["mode"],row["strength"]) == (kind,sample,mode,1.))
            compare_generated_rows(benchmark,old["benchmark"])
            compare_generated_rows(cap,old["capability"])
            row = {"kind":kind,"sample":sample,"mode":mode,"strength":1.,"benchmark":benchmark,"capability":cap,
                   "rates":rates(benchmark),"capability_accuracy":sum(item["exact_correct"] for item in cap)/32}
            write_json(output / f"{kind}-{sample}-{mode}.json",row)
            conditions.append({key:row[key] for key in ("kind","sample","mode","strength","rates","capability_accuracy")})
        print(f"Fresh CUDA outcomes match for {kind} draw {sample}; {len(conditions)}/34 conditions",flush=True)
    t.cuda.synchronize()
    peak = t.cuda.max_memory_allocated()/1024**3
    result = {"cuda_available":True,"accepted":peak<=max_vram_gb,"device":t.cuda.get_device_name(0),
        "claim_scope":"one pinned model, frozen development direction, marker behavior and exact benign tasks",
        "stored_learner_audit":audit,"completed_learner_code_cells":manifest["code_cells_executed"],
        "fresh_cuda_conditions":34,"fresh_generation_examples":35*104,"fresh_separation_directions":65,
        "all_fresh_outcomes_match_learner":True,"fresh_training_vector_max_error":error,
        "fresh_score_max_error":largest_score_error,"fresh_conditions":conditions,
        "model_id":data.MODEL_ID,"model_revision":data.MODEL_REVISION,
        "dataset_id":data.DATASET_ID,"dataset_revision":data.DATASET_REVISION,
        "learner_evidence_sha256":manifest["published_evidence_sha256"],
        "raw_directory":str(output),"raw_files_sha256":{path.name:sha256(path) for path in output.glob("*.json")},
        "peak_vram_gb":peak,"within_vram_budget":peak<=max_vram_gb,"elapsed_seconds":time.perf_counter()-started,
        "paper_equivalent_full_replication":False,"marker_absence_is_safety_or_correctness":False}
    write_json(output / "execution.json",result)
    del model,cache,final_cache
    t.cuda.empty_cache()
    return result
