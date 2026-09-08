"""Mutation tests for consequential measurement and provenance mistakes."""
from copy import deepcopy
import hashlib

import pytest
import torch as t

from . import direction_evidence as evidence
from .direction_cuda import compare_generated_rows


def _benign_rows():
    prompts = [{"id":"a","prompt":"What is 2 + 3?","task":"addition","answer":"5"},
               {"id":"b","prompt":"Copy pear","task":"copy","answer":"pear"}]
    rows = []
    for record,completion,tokens in zip(prompts,("5.","I cannot help with that."),([20,2],[30,31,32,2]),strict=True):
        rows.append({"id":record["id"],"refusal_label":None,"refusal_marker":record["id"]=="b",
            "completion":completion,"completion_sha256":hashlib.sha256(completion.encode()).hexdigest(),
            "generated_tokens":tokens,"generated_length":len(tokens),"finished":True,"hook_calls":0,
            "prompt":record["prompt"],"task":record["task"],"expected_answer":record["answer"],
            "exact_correct":record["id"]=="a"})
    return prompts,rows


def test_generation_audit_uses_actual_words_and_keeps_denominators():
    prompts,rows = _benign_rows()
    assert evidence.validate_generation_rows(list(reversed(rows)),prompts)==.5
    for mutate in (lambda x:x.pop(), lambda x:x.append(x[0]),
                   lambda x:x[0].update(exact_correct=False),
                   lambda x:x[1].update(refusal_marker=False),
                   lambda x:x[0].update(completion="6."),
                   lambda x:x[0].update(expected_answer="6"),
                   lambda x:x[0].update(hook_calls=1)):
        changed=deepcopy(rows);mutate(changed)
        with pytest.raises(ValueError):evidence.validate_generation_rows(changed,prompts)


def test_score_audit_rejects_eval_threshold_and_flipped_auc():
    labels=t.tensor([True,True,False,False])
    row={"scores":[14.,16.,10.,12.],"predictions":[True]*4,"accuracy":.5,"auroc":1.,"margin":4.}
    assert evidence.validate_score_measurement(row,labels,threshold=3.)["accuracy"]==.5
    changed={**row,"predictions":labels.tolist(),"accuracy":1.}
    with pytest.raises(ValueError,match="threshold"):
        evidence.validate_score_measurement(changed,labels,threshold=3.)
    changed={**row,"scores":[-14.,-16.,-10.,-12.],"margin":-4.}
    with pytest.raises(ValueError,match="metrics"):
        evidence.validate_score_measurement(changed,labels)


def test_fresh_generation_comparison_checks_individual_sensitive_hashes():
    _,rows=_benign_rows()
    sensitive={"id":"source-0","refusal_label":True,"refusal_marker":True,
               "completion_sha256":"a"*64,"generated_length":8,"finished":True}
    stored=rows+[sensitive]
    fresh=deepcopy(stored)
    for row in fresh:
        if "generated_tokens" in row:row["generated_tokens"]=t.tensor(row["generated_tokens"])
    compare_generated_rows(list(reversed(fresh)),stored)
    changed=deepcopy(fresh);changed[-1]["completion_sha256"]="b"*64
    with pytest.raises(ValueError,match="completion_sha256"):
        compare_generated_rows(changed,stored)


def test_published_refusal_measurements_reject_missing_nulls_and_retuning():
    """Run after full publication; raw-file tampering must fail independently of hashes."""
    import gzip,json
    path=evidence.HERE/"expected_outputs"/"completed_refusal_study.json.gz"
    if not path.exists():
        pytest.skip("Full own-answer learner has not yet published its measured evidence")
    raw=json.loads(gzip.decompress(path.read_bytes()))
    assert evidence.validate_refusal_measurements(raw)["final_conditions"]==85
    for mutate in (lambda x:x["final"]["conditions"].pop(),
                   lambda x:x["final"]["separation"]["permuted"].pop(),
                   lambda x:x["selection"]["selected"].update(layer=20),
                   lambda x:x["development"]["candidates"][0].update(eligible=True),
                   lambda x:x["final"]["conditions"][0]["rates"].update(benign_category_marker_rate=1.)):
        changed=deepcopy(raw);mutate(changed)
        with pytest.raises(ValueError):evidence.validate_refusal_measurements(changed)


def test_publication_requires_full_current_own_answer_sources(tmp_path):
    import json
    (tmp_path/"learner_execution.json").write_text(json.dumps({"status":"PASS","scope":"numerical"}))
    with pytest.raises(ValueError,match="26 cells"):
        evidence.read_completed_refusal_evidence(tmp_path)
