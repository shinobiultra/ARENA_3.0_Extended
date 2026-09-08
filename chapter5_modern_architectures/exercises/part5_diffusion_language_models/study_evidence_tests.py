"""Mutation checks for the audit itself; these fixtures are never model evidence."""
from copy import deepcopy

import pytest

from .study_evidence import unique_rows,expected_configs,validate_generated_condition,validate_training_records
from .text_evaluation import score_generated_text,summarize_generated_rows
from .study_release import summarize_primary_tasks


class FixtureTokenizer:
    def decode(self,tokens,skip_special_tokens=False):
        return ''.join({0:'[PAD]',1:'[MASK]',2:'[BOS]',3:'[SEP]',5:'red',6:'blue'}[x] for x in tokens)


def test_report_does_not_treat_story_reference_match_as_answer_correctness():
    tasks={'parentheses':[{'valid_grammar_rate':.5,'exact_answer_rate':0.},
                          {'valid_grammar_rate':1.,'exact_answer_rate':.1}],
           'arithmetic':[{'exact_answer_rate':.25},{'exact_answer_rate':.75}],
           'qa':[{'exact_answer_rate':.5},{'exact_answer_rate':0.}],
           'story':[{'eos_rate':0.,'mean_reference_token_edit_distance':20.,'reference_token_match_rate':0.},
                    {'eos_rate':.5,'mean_reference_token_edit_distance':22.,'reference_token_match_rate':.25}]}
    means,stories=summarize_primary_tasks({'trained':tasks,'shuffled':deepcopy(tasks)})
    assert means=={label:{'parentheses':.75,'arithmetic':.5,'qa':.25} for label in ('trained','shuffled')}, 'Use grammar validity, not equality with one valid bracket reference.'
    assert stories=={label:{'eos_rate':.25,'mean_reference_token_edit_distance':21.,'reference_token_match_rate':.125} for label in ('trained','shuffled')}, 'Story-reference diagnostics retain every control without inventing correctness.'


def generation_fixture():
    rows=[{'id':'audit-fixture-qa','task':'qa','prompt':'Which color?','answer':'red','target_ids':[5,4]+[0]*22}]
    tokenizer=FixtureTokenizer()
    measured=[score_generated_text(rows[0],[5,4]+[0]*22,tokenizer)]
    result={'kind':'autoregressive','seed':1000,'batch_size':32,'temperatures':[0.,0.],'block_size':None,
            'rows':measured,'summary':summarize_generated_rows(measured,rows),'traces':[]}
    return {'kind':'autoregressive','config':expected_configs('autoregressive')[0],'result':result},rows,tokenizer


def test_audit_rejects_missing_or_duplicated_failures():
    record,rows,tokenizer=generation_fixture()
    validate_generated_condition(record,rows,tokenizer,[])
    for replacement in ([],record['result']['rows']*2):
        changed=deepcopy(record);changed['result']['rows']=replacement
        with pytest.raises(ValueError,match='exactly once'):validate_generated_condition(changed,rows,tokenizer,[])
    print('The audit rejects omitted or duplicated generation rows!')


def test_audit_recomputes_words_and_scores_from_tokens():
    record,rows,tokenizer=generation_fixture()
    for update in ({'exact_answer':False},{'text':'blue'},{'generated_ids':[5,0,4]+[0]*21}):
        changed=deepcopy(record);changed['result']['rows'][0].update(update)
        with pytest.raises(ValueError,match='actual token IDs'):validate_generated_condition(changed,rows,tokenizer,[])
    changed=deepcopy(record);changed['result']['summary']['qa']['exact_answer_rate']=0.
    with pytest.raises(ValueError,match='Summary metrics'):validate_generated_condition(changed,rows,tokenizer,[])
    print('Token changes, erased PAD errors, relabeled words and false rates fail the audit!')


def test_audit_rejects_missing_runs_and_relabelled_sampling():
    with pytest.raises(ValueError,match='twelve training'):validate_training_records([],[])
    with pytest.raises(ValueError,match='all controls'):unique_rows([{'id':'main'},{'id':'main'}],lambda x:x['id'],{'main','control'},'all controls')
    record,rows,tokenizer=generation_fixture();record['result']['temperatures']=[1.,1.]
    with pytest.raises(ValueError,match='frozen RNG'):validate_generated_condition(record,rows,tokenizer,[])
    print('Missing control runs and changed sampling settings fail the audit!')
