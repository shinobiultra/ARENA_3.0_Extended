"""Independent ordinary torch.nn/torch.optim SGD implementation for parity checks."""

import math

import torch
from torch import nn
import torch.nn.functional as F


def standard_training(x, labels, seed, width=64, epochs=30, batch_size=128, classes=10):
    generator = torch.Generator().manual_seed(seed)
    if width:
        model = nn.Sequential(nn.Linear(x.shape[1], width), nn.ReLU(), nn.Linear(width, classes))
        with torch.no_grad():
            model[0].weight.copy_(
                (torch.randn(x.shape[1], width, generator=generator) * math.sqrt(2 / x.shape[1])).T
            )
            model[0].bias.zero_()
            model[2].weight.copy_(
                (torch.randn(width, classes, generator=generator) / math.sqrt(width)).T
            )
            model[2].bias.zero_()
    else:
        model = nn.Linear(x.shape[1], classes)
        with torch.no_grad():
            model.weight.copy_(
                (torch.randn(x.shape[1], classes, generator=generator) / math.sqrt(x.shape[1])).T
            )
            model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)
    online, end, losses = [], [], []
    for _ in range(epochs):
        order = torch.randperm(len(labels), generator=generator)
        visited = torch.empty(len(labels), classes)
        for batch in order.split(batch_size):
            optimizer.zero_grad()
            logits = model(x[batch])
            visited[batch] = logits.detach()
            F.cross_entropy(logits, labels[batch]).backward()
            optimizer.step()
        with torch.no_grad():
            final = model(x)
            online.append(visited)
            end.append(final.clone())
            losses.append(F.cross_entropy(final, labels))
    if width:
        parameters = dict(
            w1=model[0].weight.T,
            b1=model[0].bias[None],
            w2=model[2].weight.T,
            b2=model[2].bias[None],
        )
    else:
        parameters = dict(w=model.weight.T, b=model.bias[None])
    return dict(
        parameters={name: value.detach().clone() for name, value in parameters.items()},
        online_logits=torch.stack(online),
        end_logits=torch.stack(end),
        loss=torch.stack(losses),
    )
