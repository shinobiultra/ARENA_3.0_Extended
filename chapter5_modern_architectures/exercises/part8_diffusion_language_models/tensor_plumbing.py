"""Import-inert row/tensor plumbing; no method, data reads or model imports."""
import hashlib
import torch as t

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
