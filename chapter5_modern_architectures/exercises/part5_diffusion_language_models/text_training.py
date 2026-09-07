"""Visible matched-training and development-only checkpoint selection methods."""
from __future__ import annotations

import hashlib
import time

import torch as t

from .text_diffusion_methods import mask_targets, finite_diffusion_loss
from .text_model import autoregressive_target_loss


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare_text_tensors(rows, *, device='cpu'):
    """Boring row-to-tensor conversion; never infer target eligibility from padding."""
    tasks = sorted({row['task'] for row in rows})
    splits = {row['split'] for row in rows}
    if len(splits) != 1:
        raise ValueError('one data split per tensor bundle is required')
    return {'sequence':t.tensor([row['context_ids']+row['target_ids'] for row in rows],device=device),
            'valid':t.tensor([row['valid'] for row in rows],dtype=t.bool,device=device),
            'eligible':t.tensor([row['eligible'] for row in rows],dtype=t.bool,device=device),
            'task_ids':t.tensor([tasks.index(row['task']) for row in rows],device=device),
            'row_ids':[row['id'] for row in rows], 'tasks':tasks,'split':next(iter(splits))}


def permute_targets_within_task(sequence, eligible, task_ids, *, generator):
    """One full permutation per task, retaining every draw and any fixed points."""
    if not t.equal(eligible,eligible[:1].expand_as(eligible)):
        raise ValueError('matched controls require identical fixed target slots')
    permutation = t.arange(len(sequence),device=sequence.device)
    for task in task_ids.unique(sorted=True):
        indices = task_ids.eq(task).nonzero().flatten()
        permutation[indices] = indices[t.randperm(len(indices),device=sequence.device,generator=generator)]
    result = t.where(eligible,sequence[permutation],sequence)
    return result,permutation


def fixed_development_noise(development, *, total_steps: int, mask_id: int, repeats: int = 3):
    """Common deterministic Monte Carlo times/masks, independent of training seeds."""
    sequence = development['sequence'].repeat(repeats,1)
    eligible = development['eligible'].repeat(repeats,1)
    generator = t.Generator(device=sequence.device).manual_seed(1729)
    steps = t.randint(1,total_steps+1,(len(sequence),),device=sequence.device,generator=generator)
    noisy,_ = mask_targets(sequence,eligible,steps/total_steps,mask_id,generator=generator)
    return {'sequence':sequence,'eligible':eligible,'valid':development['valid'].repeat(repeats,1),
            'noisy':noisy,'steps':steps}


def development_objective(model, development, noise, *, kind: str, total_steps: int,
                          mask_id: int, batch_size: int = 64):
    """Compare checkpoints within one objective; diffusion bound is not AR NLL."""
    source = development if kind=='autoregressive' else noise
    was_training = model.training; model.eval()
    values = []
    try:
        with t.no_grad():
            for start in range(0,len(source['sequence']),batch_size):
                sl = slice(start,start+batch_size)
                if kind=='autoregressive':
                    logits,_ = model(source['sequence'][sl],source['valid'][sl],causal=True)
                    loss = autoregressive_target_loss(logits,source['sequence'][sl],source['eligible'][sl],mask_id)
                else:
                    logits,_ = model(source['noisy'][sl],source['valid'][sl],causal=False)
                    loss = finite_diffusion_loss(logits,source['sequence'][sl],source['noisy'][sl],
                        source['eligible'][sl],source['steps'][sl],total_steps,mask_id)
                values.append((float(loss),len(source['sequence'][sl])))
    finally:
        model.train(was_training)
    return sum(value*count for value,count in values)/sum(count for _,count in values)


def fit_text_model(factory, training, development, *, kind: str, shuffled: bool,
                   seed: int, mask_id: int, total_steps: int = 16, updates: int = 1500,
                   batch_size: int = 32, learning_rate: float = .002, weight_decay: float = .01,
                   development_every: int = 300, checkpoint_callback=None):
    """Matched batches/initial weights, separate RNG streams and earliest best dev loss.

    No final data is accepted by this API. Every tested checkpoint is retained
    in history; a callback can save each weight snapshot outside the notebook.
    Each task contributes equally to each batch, regardless of corpus size.
    """
    if kind not in {'diffusion','autoregressive'} or updates < 1 or development_every < 1:
        raise ValueError('choose a model objective and positive training budgets')
    if set(training['row_ids']) & set(development['row_ids']):
        raise ValueError('training and development IDs must be disjoint')
    if training['split'] != 'train' or development['split'] != 'development':
        raise ValueError('fitting accepts train and development splits only')
    task_count = len(training['tasks'])
    if batch_size % task_count:
        raise ValueError('batch size must permit equal task contributions')
    device = training['sequence'].device
    gpu_devices = [device.index or 0] if device.type=='cuda' else []
    with t.random.fork_rng(devices=gpu_devices):
        t.manual_seed(seed)
        model = factory().to(device)
    initial_digest = tensor_digest(t.cat([value.detach().flatten() for value in model.parameters()]))
    batch_rng = t.Generator(device=device).manual_seed(seed+100)
    noise_rng = t.Generator(device=device).manual_seed(seed+200)
    shuffle_rng = t.Generator(device=device).manual_seed(seed+300)
    sequence = training['sequence']
    if shuffled:
        sequence,permutation = permute_targets_within_task(sequence,training['eligible'],training['task_ids'],generator=shuffle_rng)
    else:
        permutation = t.arange(len(sequence),device=device)
    groups = [training['task_ids'].eq(index).nonzero().flatten() for index in range(task_count)]
    optimizer = t.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    noise = fixed_development_noise(development,total_steps=total_steps,mask_id=mask_id)
    histories = []; preview = []; batch_hash = hashlib.sha256()
    best_score = float('inf'); best_state = None; best_step = None
    started = time.perf_counter()
    for update in range(1,updates+1):
        model.train()
        indices = t.cat([group[t.randint(len(group),(batch_size//task_count,),device=device,generator=batch_rng)] for group in groups])
        batch_hash.update(indices.detach().cpu().numpy().tobytes())
        clean = sequence[indices]; eligible = training['eligible'][indices]; valid = training['valid'][indices]
        if kind=='diffusion':
            steps = t.randint(1,total_steps+1,(batch_size,),device=device,generator=noise_rng)
            noisy,_ = mask_targets(clean,eligible,steps/total_steps,mask_id,generator=noise_rng)
            logits,_ = model(noisy,valid,causal=False)
            loss = finite_diffusion_loss(logits,clean,noisy,eligible,steps,total_steps,mask_id)
        else:
            steps = None; noisy = clean
            logits,_ = model(clean,valid,causal=True)
            loss = autoregressive_target_loss(logits,clean,eligible,mask_id)
        if not t.isfinite(loss):
            raise ValueError('nonfinite training loss; retain the failed run')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = t.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if update <= 2:
            preview.append({'indices':indices.detach().cpu().tolist(),'noisy':noisy.detach().cpu().tolist(),
                'steps':None if steps is None else steps.detach().cpu().tolist()})
        histories.append({'step':update,'training_loss':float(loss.detach()),'gradient_norm':float(norm)})
        if update % development_every==0 or update==updates:
            score = development_objective(model,development,noise,kind=kind,total_steps=total_steps,mask_id=mask_id)
            histories[-1]['development_objective'] = score
            if score < best_score:
                best_score = score; best_step = update
                best_state = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            if checkpoint_callback is not None:
                checkpoint_callback(update,model,histories[-1])
    model.load_state_dict(best_state); model.eval()
    record = {'kind':kind,'shuffled':shuffled,'seed':seed,'updates':updates,'batch_size':batch_size,
        'learning_rate':learning_rate,'weight_decay':weight_decay,'gradient_clip':1.,'total_diffusion_steps':total_steps,
        'development_every':development_every,'development_noise_seed':1729,'development_noise_repeats':3,
        'development_steps_sha256':tensor_digest(noise['steps']),'development_noisy_sha256':tensor_digest(noise['noisy']),
        'initial_parameters_sha256':initial_digest,'batch_indices_sha256':batch_hash.hexdigest(),
        'target_permutation':permutation.detach().cpu().tolist(),'preview_batches':preview,'history':histories,
        'selected_step':best_step,'selected_development_objective':best_score,
        'selection_rule':'earliest minimum of the fixed development objective within this model kind',
        'parameter_count':sum(value.numel() for value in model.parameters()),'elapsed_seconds':time.perf_counter()-started}
    return model,record
