"""Scoring counterexamples, missing-row rejection and controlled timing checks."""
from copy import deepcopy
import pytest
import torch as t

from . import text_evaluation as evaluation
from .text_data import CONTEXT_SLOTS, TARGET_SLOTS, PAD_ID, MASK_ID, EOS_ID


class TinyTokenizer:
    words={0:'[PAD]',1:'[MASK]',2:'[BOS]',3:'[SEP]',5:'(',6:')',7:'red',8:'blue',9:'x'}
    def decode(self,ids,skip_special_tokens=False):return ''.join(self.words[i] for i in ids)


def test_valid_alternate_grammar_is_not_reference_match(fn=evaluation.score_generated_text):
    row={'id':'p','task':'parentheses','parenthesis_prefix':'(()(','pairs':4,'answer':'))()',
         'target_ids':[6,6,5,6,4]}
    score=fn(row,[5,6,6,6,4],TinyTokenizer())
    assert score['grammar_valid'], 'The alternate completion is a balanced eight-character string.'
    assert not score['exact_answer'] and not score['exact_reference_tokens'], 'Grammar correctness is distinct from one reference continuation.'
    assert score['reference_token_edit_distance']>0, 'A correct alternative can have nonzero edit distance.'
    print('Valid alternate strings stay distinct from reference-token matching!')


def test_closed_answers_require_eos_and_keep_special_errors(score_fn=evaluation.score_generated_text):
    row={'id':'q','task':'qa','answer':'red','target_ids':[7,4,0,0]}
    fn=lambda tokens:score_fn(row,tokens,TinyTokenizer())
    assert fn([7,4,8,8])['exact_answer'], 'Only tokens before EOS form the answer.'
    assert not fn([7,0,4,0])['exact_answer'], 'PAD before EOS is an actual generation error.'
    assert not fn([7])['exact_answer'], 'The fixed answer protocol requires a generated EOS.'
    assert fn([1,7,4,0])['text']=='[MASK]red', 'Do not silently drop an unreplaced MASK.'
    print('EOS termination and visible special-token mistakes passed!')


def test_scoring_requires_all_examples_and_no_story_correctness(fn=evaluation.summarize_generated_rows):
    rows=[{'id':'a','task':'story','target_ids':[7,8]},{'id':'b','task':'story','target_ids':[8,7]}]
    measured=[evaluation.score_generated_text(row,[7,8],TinyTokenizer()) for row in rows]
    result=fn(measured,rows)['story']
    assert result['n']==2 and 'exact_answer_rate' not in result, 'An open-ended story does not have one correct continuation.'
    with pytest.raises(ValueError,match='exactly once'):fn(measured[:1],rows)
    with pytest.raises(ValueError,match='exactly once'):fn([measured[0]]*2,rows)
    print('Complete denominators and bounded open-ended text metrics passed!')


def test_generation_inputs_ignore_answers_and_target_padding(fn=evaluation.model_inputs_from_context):
    row={'context_ids':[2]+[0]*(CONTEXT_SLOTS-2)+[3],
         'valid':[True]+[False]*(CONTEXT_SLOTS-2)+[True]+[True]*TARGET_SLOTS,
         'eligible':[False]*CONTEXT_SLOTS+[True]*TARGET_SLOTS,'answer':'red','target_ids':[7,4]+[0]*(TARGET_SLOTS-2)}
    other=deepcopy(row);other['answer']='blue blue';other['target_ids']=[8,8,4]+[0]*(TARGET_SLOTS-3)
    a=fn([row],'cpu');b=fn([other],'cpu')
    for left,right in zip(a,b):assert t.equal(left,right), 'Only the question and fixed slot protocol may enter generation.'
    assert a[1][:,CONTEXT_SLOTS:].eq(MASK_ID).all(), 'Every target starts masked, regardless of its answer length.'
    print('Counterfactual answers leave all generation inputs identical!')


def test_latency_repeats_same_tokens_and_rejects_trace_overhead(fn=evaluation.measure_generation_latency):
    def generate():return t.tensor([[5,6]]),[{'input_tokens':4},{'input_tokens':1}]
    result=fn(generate,device='cpu',warmups=1,repeats=3)
    assert result['forward_calls']==2 and result['input_tokens']==5, 'Report real forward calls and input work.'
    assert len(result['seconds'])==3 and result['median_seconds']>=0, 'Keep every timing replicate.'
    def traced():return t.tensor([[5,6]]),[{'input_tokens':4,'entropy':[0.,1.]}]
    with pytest.raises(ValueError,match='storage'):fn(traced,device='cpu')
    count=0
    def changing():
        nonlocal count
        count+=1
        return t.tensor([[count]]),[{'input_tokens':1}]
    with pytest.raises(ValueError,match='RNG'):fn(changing,device='cpu')
    print('Fixed-output timing, workload counts and trace-overhead rejection passed!')
