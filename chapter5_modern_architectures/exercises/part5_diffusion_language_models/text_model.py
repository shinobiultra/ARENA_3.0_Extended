"""Same inspectable Transformer for bidirectional denoising and causal prediction.

These classes are being prepared as student implementations. No pretrained
weights or model implementation are hidden behind the training comparison.
"""
from __future__ import annotations

import torch as t
from torch import nn
from torch.nn import functional as F

from .text_diffusion_methods import clean_prediction_probs, categorical_tokens, entropy_from_probs


class TextAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by heads')
        self.heads = heads
        self.qkv = nn.Linear(width, 3*width)
        self.output = nn.Linear(width, width)

    def forward(self, hidden, valid, *, causal: bool, cache=None):
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).reshape(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4)
        past = 0 if cache is None else cache[0].shape[-2]
        if cache is not None:
            if not causal:
                raise ValueError('a bidirectional pass cannot append to a stale KV cache')
            k = t.cat((cache[0],k),dim=-2)
            v = t.cat((cache[1],v),dim=-2)
        if valid.dtype != t.bool or valid.shape != (batch,past+length) or not valid.any(-1).all():
            raise ValueError('valid must describe every existing and current key')
        mask = valid[:,None,None,:]
        if causal:
            query_position = past+t.arange(length,device=hidden.device)
            key_position = t.arange(past+length,device=hidden.device)
            mask = mask & (key_position[None,:] <= query_position[:,None])[None,None]
        attended = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
        out = attended.transpose(1,2).reshape(batch,length,width)
        return self.output(out), (k,v)


class TextBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = TextAttention(width,heads)
        self.norm_mlp = nn.LayerNorm(width)
        self.up = nn.Linear(width,4*width)
        self.down = nn.Linear(4*width,width)

    def forward(self, hidden, valid, *, causal: bool, cache=None):
        attention, new_cache = self.attention(self.norm_attention(hidden),valid,causal=causal,cache=cache)
        hidden = hidden+attention
        hidden = hidden+self.down(F.gelu(self.up(self.norm_mlp(hidden))))
        return hidden,new_cache


class TextTransformer(nn.Module):
    def __init__(self, vocabulary: int, *, width: int = 96, heads: int = 4,
                 layers: int = 3, max_length: int = 64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary,width)
        self.position_embedding = nn.Embedding(max_length,width)
        self.blocks = nn.ModuleList([TextBlock(width,heads) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        self.unembed = nn.Linear(width,vocabulary)

    def forward(self, tokens, valid, *, causal: bool, cache=None):
        if cache is not None and (not causal or len(cache) != len(self.blocks)):
            raise ValueError('only a complete causal cache can be extended')
        past = 0 if cache is None else cache[0][0].shape[-2]
        if past+tokens.shape[1] > self.position_embedding.num_embeddings:
            raise ValueError('context exceeds the declared positional budget')
        positions = t.arange(past,past+tokens.shape[1],device=tokens.device)
        hidden = self.token_embedding(tokens)+self.position_embedding(positions)
        new_cache = []
        for i,block in enumerate(self.blocks):
            hidden,entry = block(hidden,valid,causal=causal,cache=None if cache is None else cache[i])
            new_cache.append(entry)
        return self.unembed(self.final_norm(hidden)), new_cache if causal else None


def autoregressive_target_loss(logits, sequence, eligible, mask_id: int):
    """Predict token j from logits at j-1 and average only the fixed target slots."""
    if logits.shape[:-1] != sequence.shape or eligible.shape != sequence.shape or eligible.dtype != t.bool:
        raise ValueError('logits, sequence and boolean target mask shapes disagree')
    if eligible[:,0].any() or not eligible.any(-1).all():
        raise ValueError('each example needs predicted targets after a context token')
    prediction = logits[:,:-1].clone()
    prediction[...,mask_id] = -t.inf
    selected = eligible[:,1:]
    targets = sequence[:,1:].masked_fill(~selected,(mask_id+1) % logits.shape[-1])
    loss = -prediction.log_softmax(-1).gather(-1,targets[...,None]).squeeze(-1)
    return (t.where(selected,loss,0.).sum(-1)/selected.sum(-1)).mean()


@t.no_grad()
def sample_autoregressive(model, context, valid, *, output_length: int,
                          mask_id: int, eos_id: int, pad_id: int, temperature: float = 1.,
                          use_cache: bool = True, generator: t.Generator | None = None,
                          record_trace: bool = True):
    """Cached/uncached generation with individual EOS, retaining actual cost and tokens."""
    if output_length < 1 or context.shape != valid.shape:
        raise ValueError('need positive output length and context validity')
    tokens = context.clone()
    keys_valid = valid.clone()
    generated = t.full((len(tokens),output_length),pad_id,device=tokens.device,dtype=tokens.dtype)
    finished = t.zeros(len(tokens),device=tokens.device,dtype=t.bool)
    lengths = t.full((len(tokens),),output_length,device=tokens.device,dtype=t.long)
    cache = None
    records = []
    was_training = model.training
    model.eval()
    try:
        for step in range(output_length):
            current = tokens[:,-1:] if use_cache and cache is not None else tokens
            logits,new_cache = model(current,keys_valid,causal=True,cache=cache if use_cache else None)
            if use_cache:
                cache = new_cache
            probabilities = clean_prediction_probs(logits[:,-1],mask_id,temperature)
            sampled = categorical_tokens(probabilities,generator=generator)
            sampled = t.where(finished,pad_id,sampled)
            generated[:,step] = sampled
            newly_finished = ~finished & sampled.eq(eos_id)
            lengths[newly_finished] = step+1
            record = {'model_calls':step+1,'input_tokens':current.numel()}
            if record_trace:
                record.update(entropy=entropy_from_probs(clean_prediction_probs(logits[:,-1],mask_id,1.)).detach(),active=(~finished).clone())
            records.append(record)
            finished |= newly_finished
            tokens = t.cat((tokens,sampled[:,None]),dim=1)
            keys_valid = t.cat((keys_valid,t.ones_like(sampled[:,None],dtype=t.bool)),dim=1)
            if finished.all():
                break
    finally:
        model.train(was_training)
    return generated,lengths,records
