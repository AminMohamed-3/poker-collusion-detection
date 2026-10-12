"""Permutation-invariant pair detector and optional noisy annotation preferences."""
import torch
from torch import nn
from torch.nn import functional as F


class PairMIL(nn.Module):
    def __init__(self, hand_features, pair_features):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(hand_features, 128), nn.GELU(), nn.Dropout(.15),
                                     nn.Linear(128, 64), nn.GELU())
        self.attention = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.events = nn.Linear(64, 3)
        self.pair_encoder = nn.Sequential(nn.Linear(pair_features, 64), nn.GELU(), nn.Dropout(.15))
        self.head = nn.Sequential(nn.Linear(64*4+3, 64), nn.GELU(), nn.Dropout(.15), nn.Linear(64, 4))

    def forward(self, x, mask, pair):
        h = self.encoder(x)
        weights = self.attention(h).squeeze(-1).masked_fill(~mask, -1e9).softmax(1)
        avg = (h*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True)
        maximum = h.masked_fill(~mask.unsqueeze(-1), -1e9).amax(1)
        attended = (h*weights.unsqueeze(-1)).sum(1)
        events = self.events(h)
        event_pool = torch.logsumexp(events.masked_fill(~mask.unsqueeze(-1), -1e9), 1) - mask.sum(1, keepdim=True).log()
        logits = self.head(torch.cat([avg, maximum, attended, self.pair_encoder(pair), event_pool], 1))
        return logits, events


def objective(logits, events, mask, y, relevance, auxiliary):
    loss = F.cross_entropy(logits, y)
    if auxiliary:
        valid = (y > 0) & ((relevance*mask).sum(1) > 0)
        if valid.any():
            rows = torch.arange(len(y), device=y.device)[valid]
            score = events[rows, :, y[valid]-1].masked_fill(~mask[valid], -1e9)
            target = relevance[valid]*mask[valid]
            target = target/target.sum(1, keepdim=True)
            loss = loss + .25*(-(target*score.log_softmax(1)).sum(1)).mean()
    return loss
