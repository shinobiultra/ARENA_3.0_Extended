"""Student-visible generation scoring, trajectory measurements and fair timing.

Reference edit distance describes one continuation; it is not a correctness
metric for open-ended stories. A valid alternate parenthesis completion can
also differ from the chosen reference string.
"""
from __future__ import annotations

import statistics
import time

import torch as t

from .text_data import CONTEXT_SLOTS, TARGET_SLOTS, PAD_ID, MASK_ID, EOS_ID
from .text_diffusion_methods import (sample_text_diffusion, stable_token_commitment,
                                    levenshtein_distance, balanced_parentheses)
from .text_model import sample_autoregressive


def score_generated_text(row, generated_ids, tokenizer):
    """Keep special-token mistakes visible and require EOS on closed-answer tasks."""
    tokens = list(generated_ids)
    ended = EOS_ID in tokens
    end = tokens.index(EOS_ID) if ended else len(tokens)
    content = tokens[:end]
    # Do not erase invalid PAD, MASK, BOS or SEP predictions before EOS.
    decoded = tokenizer.decode(content,skip_special_tokens=False)
    reference = row['target_ids']
    reference_end = reference.index(EOS_ID) if EOS_ID in reference else len(reference)
    reference_content = reference[:reference_end]
    out = {'id':row['id'],'task':row['task'],'generated_ids':tokens,'text':decoded,
        'eos_seen':ended,'generated_length':end+int(ended),
        'reference_token_edit_distance':levenshtein_distance(content,reference_content),
        'exact_reference_tokens':content==reference_content and (ended or row['task']=='story')}
    if row['task'] in {'qa','arithmetic'}:
        out['exact_answer'] = ended and decoded==row['answer']
    elif row['task']=='parentheses':
        combined = row['parenthesis_prefix']+decoded
        out['grammar_valid'] = ended and len(combined)==2*row['pairs'] and balanced_parentheses(combined)
        out['exact_answer'] = ended and decoded==row['answer']
    return out


def summarize_generated_rows(measured, source_rows):
    """Every prespecified example contributes once; no success-only denominator."""
    expected = {row['id']:row['task'] for row in source_rows}
    if len(measured)!=len(expected) or {row['id'] for row in measured}!=set(expected):
        raise ValueError('retain every generation example exactly once')
    if any(row['task']!=expected[row['id']] for row in measured):
        raise ValueError('score by the task attached to the input ID')
    result = {}
    for task in sorted(set(expected.values())):
        rows = [row for row in measured if row['task']==task]
        values = {'n':len(rows),'eos_rate':sum(row['eos_seen'] for row in rows)/len(rows),
            'mean_reference_token_edit_distance':sum(row['reference_token_edit_distance'] for row in rows)/len(rows),
            'reference_token_match_rate':sum(row['exact_reference_tokens'] for row in rows)/len(rows)}
        if task!='story':
            values['exact_answer_rate'] = sum(row['exact_answer'] for row in rows)/len(rows)
        if task=='parentheses':
            values['valid_grammar_rate'] = sum(row['grammar_valid'] for row in rows)/len(rows)
        result[task] = values
    return result


def model_inputs_from_context(rows, device):
    context = t.tensor([row['context_ids'] for row in rows],device=device)
    valid = t.tensor([row['valid'] for row in rows],device=device,dtype=t.bool)
    eligible = t.tensor([row['eligible'] for row in rows],device=device,dtype=t.bool)
    if context.shape[1]!=CONTEXT_SLOTS or not valid[:,CONTEXT_SLOTS:].all():
        raise ValueError('generation uses fixed context and fully valid target slots')
    if eligible[:,:CONTEXT_SLOTS].any() or not eligible[:,CONTEXT_SLOTS:].all():
        raise ValueError('only the fixed target region can be generated')
    conditioning = t.cat((context,t.full((len(rows),TARGET_SLOTS),MASK_ID,device=device,dtype=t.long)),dim=1)
    return context,conditioning,valid,eligible


@t.no_grad()
def generate_text_examples(model, rows, tokenizer, *, kind, seed: int = 0,
                           sampler='confidence', steps: int = 16, temperatures=(0.,0.),
                           block_size=None, batch_size: int = 32, trace_ids=()):
    """Generate from questions alone and score the actual decoded output afterwards."""
    device = next(model.parameters()).device
    generator = t.Generator(device=device).manual_seed(seed)
    measured = []; traces = []; calls = 0
    was_training = model.training;model.eval()
    try:
        for start in range(0,len(rows),batch_size):
            batch = rows[start:start+batch_size]
            context,conditioning,valid,eligible = model_inputs_from_context(batch,device)
            capture = any(row['id'] in trace_ids for row in batch)
            if kind=='autoregressive':
                generated,lengths,records = sample_autoregressive(model,context,valid[:,:CONTEXT_SLOTS],
                    output_length=TARGET_SLOTS,mask_id=MASK_ID,eos_id=EOS_ID,pad_id=PAD_ID,
                    temperature=temperatures[0],generator=generator,record_trace=capture)
                states = None
            elif kind=='diffusion':
                def model_fn(tokens,p_t):return model(tokens,valid,causal=False)[0]
                full,states,records = sample_text_diffusion(model_fn,conditioning,eligible,MASK_ID,
                    steps=steps,sampler=sampler,temperatures=temperatures,block_size=block_size,
                    generator=generator,record_trace=capture)
                generated = full[:,CONTEXT_SLOTS:]
            else:
                raise ValueError('choose autoregressive or diffusion')
            calls += len(records)
            for index,row in enumerate(batch):
                outcome = score_generated_text(row,generated[index].tolist(),tokenizer)
                measured.append(outcome)
                if row['id'] in trace_ids:
                    trace = {'id':row['id'],'kind':kind,'outcome':outcome,'prompt':row['prompt']}
                    if states is not None:
                        trace['states'] = states[:,index,CONTEXT_SLOTS:].tolist()
                        trace['stable_commitment'] = stable_token_commitment(states,MASK_ID)[index,CONTEXT_SLOTS:].tolist()
                    trace['steps'] = [{key:(value[index].tolist() if key in {'entropy','masked_before','active'}
                        else value.tolist() if isinstance(value,t.Tensor) else value)
                        for key,value in record.items()} for record in records]
                    traces.append(trace)
    finally:
        model.train(was_training)
    return {'rows':measured,'summary':summarize_generated_rows(measured,rows),'traces':traces,
            'total_batch_forward_calls':calls,'batch_size':batch_size,'seed':seed,'kind':kind,
            'sampler':sampler if kind=='diffusion' else 'cached AR','steps':steps if kind=='diffusion' else None,
            'temperatures':list(temperatures),'block_size':block_size}


def measure_generation_latency(generate_fn, *, device, repeats: int = 5, warmups: int = 2):
    """Time identical fixed batches with GPU synchronization and trace storage off.

    The supplied zero-argument function must recreate its seeded generator for
    every call. It returns (generated_tokens, minimal_records). Model loading,
    training and decoding to strings are outside this timer. Actual forward
    calls and input-token counts accompany latency; parallel token generation
    alone is not a speed claim.
    """
    device = t.device(device)
    if repeats < 1 or warmups < 1:
        raise ValueError('at least one warm-up and one measured repeat are required')
    def sync():
        if device.type=='cuda':t.cuda.synchronize(device)
    reference = None; timings=[]; counts=[]
    for iteration in range(warmups+repeats):
        sync();started=time.perf_counter()
        tokens,records = generate_fn()
        sync();elapsed=time.perf_counter()-started
        if reference is None:reference=tokens.detach().clone()
        elif not t.equal(tokens,reference):raise ValueError('timing repeats changed generated tokens or RNG state')
        if any({'entropy','masked_before','active'} & row.keys() for row in records):
            raise ValueError('disable trajectory/entropy storage in both timing paths')
        if iteration>=warmups:
            timings.append(elapsed)
            counts.append({'forward_calls':len(records),'input_tokens':sum(row['input_tokens'] for row in records)})
    if any(row!=counts[0] for row in counts):
        raise ValueError('timing repeats must perform the same workload')
    return {'seconds':timings,'median_seconds':statistics.median(timings),'warmups':warmups,
            'repeats':repeats,**counts[0],'batch_size':len(reference),'output_slots':reference.shape[-1],
            'cuda_synchronized':device.type=='cuda','trajectory_storage':False}
