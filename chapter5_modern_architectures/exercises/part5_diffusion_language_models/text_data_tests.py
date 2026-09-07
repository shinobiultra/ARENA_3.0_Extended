"""Data-split, context dependence and target-length leakage checks."""
from copy import deepcopy
import pytest

from . import text_data as data
from .text_diffusion_methods import balanced_parentheses


def test_synthetic_split_semantics():
    splits=data.synthetic_examples({'train':30,'development':12,'final':12})
    assert splits==data.synthetic_examples({'train':30,'development':12,'final':12}), "Keep questions disjoint and check each task against independently computed grammar, arithmetic or contextual facts. Required condition: splits == data.synthetic_examples({'train': 30, 'development': 12, 'final': 12})"
    seen=set()
    for split,rows in splits.items():
        for row in rows:
            assert row['prompt'] not in seen, "Keep questions disjoint and check each task against independently computed grammar, arithmetic or contextual facts. Required condition: row['prompt'] not in seen";seen.add(row['prompt'])
            if row['task']=='parentheses':
                assert balanced_parentheses(row['parenthesis_prefix']+row['answer']), "Keep questions disjoint and check each task against independently computed grammar, arithmetic or contextual facts. Required condition: balanced_parentheses(row['parenthesis_prefix'] + row['answer'])"
                assert len(row['answer'])==row['pairs'], "Keep questions disjoint and check each task against independently computed grammar, arithmetic or contextual facts. Required condition: len(row['answer']) == row['pairs']"
            elif row['task']=='arithmetic':assert int(row['answer'])==sum(row['operands']), "Keep questions disjoint and check each task against independently computed grammar, arithmetic or contextual facts. Required condition: int(row['answer']) == sum(row['operands'])"
            else:assert row['facts'][row['query']]==row['answer'], "Keep questions disjoint and check each task against independently computed grammar, arithmetic or contextual facts. Required condition: row['facts'][row['query']] == row['answer']"
    print('Independent grammar/arithmetic/QA scoring and disjoint questions passed!')


def test_train_only_tokenizer():
    synthetic=data.synthetic_examples({'train':4,'development':2,'final':2})
    with pytest.raises(ValueError,match='training'):
        data.fit_training_tokenizer(synthetic['development'],[])
    tokenizer=data.fit_training_tokenizer(synthetic['train'],[],vocabulary_size=300)
    for value in ['Ada has red.','23 + 19 = 42','(()())','A new snowman!']:
        assert tokenizer.decode(tokenizer.encode(value).ids)==value, 'The byte-level tokenizer must round-trip literal input text and fit only training examples. Required condition: tokenizer.decode(tokenizer.encode(value).ids) == value'
    print('Training-only tokenizer and byte-level exact roundtrip passed!')


def test_answer_length_never_changes_input_validity():
    synthetic=data.synthetic_examples({'train':3,'development':1,'final':1})
    tokenizer=data.fit_training_tokenizer(synthetic['train'],[],vocabulary_size=300)
    empty={split:[] for split in synthetic}
    original=data.encode_examples(synthetic,empty,tokenizer)
    modified=deepcopy(synthetic)
    for rows in modified.values():
        for row in rows:row['answer']='1' if len(row['answer'])>1 else '12345'
    altered=data.encode_examples(modified,empty,tokenizer)
    for split in original:
        for left,right in zip(original[split],altered[split]):
            assert left['context_ids']==right['context_ids'], "Changing the answer length must not alter conditioning, attention validity, target eligibility or output capacity. Required condition: left['context_ids'] == right['context_ids']"
            assert left['eligible']==right['eligible'] and left['valid']==right['valid'], "Changing the answer length must not alter conditioning, attention validity, target eligibility or output capacity. Required condition: left['eligible'] == right['eligible'] and left['valid'] == right['valid']"
            assert all(left['valid'][data.CONTEXT_SLOTS:]), "Changing the answer length must not alter conditioning, attention validity, target eligibility or output capacity. Required condition: all(left['valid'][data.CONTEXT_SLOTS:])"
            assert len(left['target_ids'])==len(right['target_ids'])==data.TARGET_SLOTS, "Changing the answer length must not alter conditioning, attention validity, target eligibility or output capacity. Required condition: len(left['target_ids']) == len(right['target_ids']) == data.TARGET_SLOTS"
    print('Counterfactual answer lengths leave every input validity/eligibility bit unchanged!')


def test_duplicate_conditioning_is_rejected():
    synthetic=data.synthetic_examples({'train':2,'development':1,'final':1})
    tokenizer=data.fit_training_tokenizer(synthetic['train'],[],vocabulary_size=300)
    synthetic['development'][0]['prompt']=synthetic['train'][0]['prompt']
    with pytest.raises(ValueError,match='conditioning context'):
        data.encode_examples(synthetic,{split:[] for split in synthetic},tokenizer)
    print('Identical conditioning cannot leak across the train/evaluation boundary!')
