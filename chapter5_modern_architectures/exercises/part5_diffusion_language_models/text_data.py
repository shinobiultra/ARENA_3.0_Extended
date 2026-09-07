"""Pinned I/O and deterministic data recipes for the selected text progression.

Dataset I/O is plumbing. Model, noising, training, sampling and scoring remain
student-visible methods. These candidate data are not yet published evidence.
"""
from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from pathlib import Path

DATASET_ID = 'roneneldan/TinyStories'
DATASET_REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
SOURCE_FILES = {
    'train': ('data/train-00000-of-00004-2d5a1467fff1081b.parquet',
              '77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b'),
    'validation': ('data/validation-00000-of-00001-869c898b519ad725.parquet',
                   '33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab'),
}
SPECIALS = ['[PAD]', '[MASK]', '[BOS]', '[SEP]', '[EOS]']
PAD_ID, MASK_ID, BOS_ID, SEP_ID, EOS_ID = range(5)
CONTEXT_SLOTS = 40
TARGET_SLOTS = 24
STORY_PREFIX_TOKENS = 30
SYNTHETIC_COUNTS = {'train': 512, 'development': 128, 'final': 128}
STORY_COUNTS = {'train': 4096, 'development': 128, 'final': 128}


def normalized_text(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _parentheses(rng):
    # A constrained random walk, explicitly not uniform over Catalan strings.
    pairs = rng.randrange(6,13)
    chars = []; opened = closed = 0
    while closed < pairs:
        if opened == pairs:
            choose_open = False
        elif opened == closed:
            choose_open = True
        else:
            choose_open = rng.random() < .5
        chars.append('(' if choose_open else ')')
        opened += choose_open
        closed += not choose_open
    sequence = ''.join(chars)
    prefix, answer = sequence[:pairs], sequence[pairs:]
    return {'task': 'parentheses', 'prompt': 'balance '+prefix, 'answer': answer,
            'parenthesis_prefix': prefix, 'pairs': pairs}


def _arithmetic(rng):
    a,b = rng.randrange(50),rng.randrange(50)
    return {'task': 'arithmetic', 'prompt': f'add {a} + {b} =', 'answer': str(a+b), 'operands': [a,b]}


def _qa(rng):
    people = rng.sample(['Ada','Ben','Cleo','Dara','Eli','Fay'],3)
    colors = rng.sample(['red','blue','green','yellow','black','white','pink','gold'],3)
    facts = dict(zip(people,colors)); query = rng.choice(people)
    prompt = ' '.join(f'{person} has {color}.' for person,color in facts.items())
    return {'task': 'qa', 'prompt': prompt+f' What color does {query} have?',
            'answer': facts[query], 'facts': facts, 'query': query}


def synthetic_examples(counts=None):
    counts = SYNTHETIC_COUNTS if counts is None else counts
    result = {split: [] for split in counts}
    for task, maker, seed in [('parentheses',_parentheses,101),('arithmetic',_arithmetic,102),('qa',_qa,103)]:
        rng = random.Random(seed); seen = set()
        for split,count in counts.items():
            attempts = 0
            while sum(row['task']==task for row in result[split]) < count:
                attempts += 1
                if attempts > 1000000:
                    raise ValueError('requested too many unique synthetic contexts')
                row = maker(rng)
                if row['prompt'] in seen:
                    continue
                seen.add(row['prompt'])
                row['id'] = f'{task}-{text_sha256(row["prompt"])[:20]}'
                row['split'] = split
                result[split].append(row)
    return result


def load_story_subset(counts=None):
    """Use source-order unique stories of >=512 normalized characters.

    Training uses only the first pinned training shard. Development and final
    use successive disjoint records in the pinned validation shard. The first
    30 tokens condition one fixed 24-token continuation; this is not a complete
    evaluation of TinyStories or a claim of reference-continuation correctness.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    counts = STORY_COUNTS if counts is None else counts
    seen = set(); results = {key: [] for key in counts}
    for source_split, partitions in [('train',['train']),('validation',['development','final'])]:
        filename, digest = SOURCE_FILES[source_split]
        source = Path(hf_hub_download(DATASET_ID,repo_type='dataset',revision=DATASET_REVISION,filename=filename))
        with source.open('rb') as handle:
            actual = hashlib.file_digest(handle,'sha256').hexdigest()
        if actual != digest:
            raise ValueError('pinned TinyStories source hash mismatch')
        index = 0
        for batch in pq.ParquetFile(source).iter_batches(batch_size=1024,columns=['text']):
            for text in batch.column('text').to_pylist():
                row_index = index; index += 1
                normalized = normalized_text(text)
                sha = text_sha256(normalized)
                if len(normalized) < 512 or sha in seen:
                    continue
                split = next((part for part in partitions if len(results[part]) < counts[part]),None)
                if split is None:
                    break
                seen.add(sha)
                results[split].append({'task':'story','split':split,'id':f'{source_split}-{row_index}',
                    'source_file':filename,'source_row':row_index,'normalized_sha256':sha,'text':normalized})
            if all(len(results[part])==counts[part] for part in partitions):
                break
        if any(len(results[part])!=counts[part] for part in partitions):
            raise ValueError('source exhausted before declared unique subset was selected')
    return results


def fit_training_tokenizer(training_synthetic, training_stories, *, vocabulary_size=768):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    if any(row['split'] != 'train' for row in [*training_synthetic,*training_stories]):
        raise ValueError('tokenizer fitting can receive training rows only')
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocabulary_size,min_frequency=2,special_tokens=SPECIALS,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False)
    strings = [row['prompt']+' '+row['answer'] for row in training_synthetic]
    strings += [row['text'] for row in training_stories]
    tokenizer.train_from_iterator(strings,trainer)
    assert [tokenizer.token_to_id(x) for x in SPECIALS] == list(range(5))
    return tokenizer


def encode_examples(synthetic, stories, tokenizer, *, story_limits=None, selection_log=None):
    result = {key: [] for key in synthetic}
    seen_blocks = set()
    seen_contexts = {}
    for split in result:
        story_count = 0
        for row in [*synthetic[split],*stories[split]]:
            if row['task']=='story':
                if story_limits is not None and story_count == story_limits[split]:
                    continue
                tokens = tokenizer.encode(row['text']).ids
                if len(tokens) < STORY_PREFIX_TOKENS+TARGET_SLOTS:
                    raise ValueError('selected story is too short for the declared fixed block')
                prefix = tokens[:STORY_PREFIX_TOKENS]
                target = tokens[STORY_PREFIX_TOKENS:STORY_PREFIX_TOKENS+TARGET_SLOTS]
                metadata = {k:v for k,v in row.items() if k!='text'}
                metadata['prompt'] = tokenizer.decode(prefix)
                metadata['reference_continuation'] = tokenizer.decode(target)
            else:
                prefix = tokenizer.encode(row['prompt']).ids
                target = tokenizer.encode(row['answer']).ids+[EOS_ID]
                metadata = dict(row)
            if len(prefix)>CONTEXT_SLOTS-2 or len(target)>TARGET_SLOTS:
                raise ValueError('do not silently truncate a question or answer')
            context = [BOS_ID]+prefix+[PAD_ID]*(CONTEXT_SLOTS-len(prefix)-2)+[SEP_ID]
            context_key = tuple(context)
            if context_key in seen_contexts:
                if row['task'] != 'story' or story_limits is None:
                    raise ValueError('duplicate conditioning context across the experiment')
                if selection_log is not None:
                    selection_log.append({'id':row['id'],'split':split,'reason':'duplicate tokenized prefix',
                                          'matches':seen_contexts[context_key]})
                continue
            seen_contexts[context_key] = row['id']
            story_count += row['task']=='story'
            target = target+[PAD_ID]*(TARGET_SLOTS-len(target))
            # This validity pattern depends on the question, never the answer.
            valid = [token!=PAD_ID for token in context]+[True]*TARGET_SLOTS
            eligible = [False]*CONTEXT_SLOTS+[True]*TARGET_SLOTS
            block = tuple(context+target)
            if block in seen_blocks:
                raise ValueError('duplicate token block across the experiment')
            seen_blocks.add(block)
            result[split].append({**metadata,'context_ids':context,'target_ids':target,
                                  'valid':valid,'eligible':eligible})
        if story_limits is not None and story_count != story_limits[split]:
            raise ValueError('candidate pool exhausted before unique-context story count')
    return result


def build_text_data(output_directory):
    directory = Path(output_directory); directory.mkdir(parents=True,exist_ok=True)
    synthetic = synthetic_examples()
    # Reserve candidates replace duplicated tokenized prefixes without changing
    # the fit/evaluation sizes. The tokenizer sees only the training pool.
    pool_counts = {split:count+(128 if split=='train' else 32) for split,count in STORY_COUNTS.items()}
    stories = load_story_subset(pool_counts)
    tokenizer = fit_training_tokenizer(synthetic['train'],stories['train'])
    selection_log = []
    examples = encode_examples(synthetic,stories,tokenizer,story_limits=STORY_COUNTS,selection_log=selection_log)
    tokenizer.save(str(directory/'tokenizer.json'))
    inventory = {}
    for split,rows in examples.items():
        path = directory/f'{split}.json'
        path.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
        inventory[split] = {'count':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'task_counts':{task:sum(row['task']==task for row in rows) for task in ('parentheses','arithmetic','qa','story')}}
    provenance = {'dataset_id':DATASET_ID,'revision':DATASET_REVISION,'source_files':SOURCE_FILES,
        'source_url':f'https://huggingface.co/datasets/{DATASET_ID}/tree/{DATASET_REVISION}',
        'source_license':'cdla-sharing-1.0','license_url':'https://cdla.dev/sharing-1-0/',
        'story_selection':'source-order unique NFKC/whitespace-normalized stories, at least 512 characters',
        'story_split_policy':'first training shard; disjoint successive development/final rows from validation',
        'synthetic_split_policy':'one seeded stream per task, globally unique questions, train/development/final order',
        'tokenizer_fit_split':'training candidate pool only','story_candidate_pool_counts':pool_counts,
        'tokenizer_training_stories':[{'id':row['id'],'normalized_sha256':row['normalized_sha256']} for row in stories['train']],
        'duplicate_context_rejections':selection_log,
        'tokenizer_sha256':hashlib.sha256((directory/'tokenizer.json').read_bytes()).hexdigest(),
        'vocabulary':tokenizer.get_vocab_size(),'context_slots':CONTEXT_SLOTS,'target_slots':TARGET_SLOTS,
        'story_prefix_tokens':STORY_PREFIX_TOKENS,'answer_length_supplied_to_sampler':False,
        'answer_padding_supplied_as_attention_mask':False,
        'all_target_slots_valid_including_eos_and_fill':True,'inventory':inventory}
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance
