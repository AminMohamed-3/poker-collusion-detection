"""Small causal encoder for observed action histories.

This module models disclosed observed behavior.  It does not supply ``normal``
labels, poker-data I/O, optimization, or any evaluation routine.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


HISTORY_STEPS = 20
ACTION_CLASSES = 4
TARGET_DIM = 6


class HistoryEncoder(nn.Module):
    """Encode oldest-to-newest right-padded observed-history tokens.

    ``x`` is a float32 tensor of shape ``[B, 20, input_dim]`` and ``lengths``
    supplies the valid prefix length for each row.  Tokens after a length are
    never selected as that row's representation.  Empty histories map to an
    exact zero representation rather than a learned identity token.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16) -> None:
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0.0)
        self.head = nn.Linear(hidden_dim, TARGET_DIM)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.dtype != torch.float32:
            raise TypeError("x must have dtype torch.float32")
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] != HISTORY_STEPS or x.shape[2] != self.input_dim:
            raise ValueError(f"x must have shape [B, {HISTORY_STEPS}, {self.input_dim}] with B > 0")
        if not isinstance(lengths, torch.Tensor):
            raise TypeError("lengths must be a torch.Tensor")
        if lengths.dtype not in (torch.int32, torch.int64):
            raise TypeError("lengths must have an integer dtype")
        if lengths.ndim != 1 or lengths.shape[0] != x.shape[0]:
            raise ValueError("lengths must have shape [B]")
        if lengths.device != x.device:
            raise ValueError("x and lengths must be on the same device")
        if bool(((lengths < 0) | (lengths > HISTORY_STEPS)).any()):
            raise ValueError(f"lengths must lie in [0, {HISTORY_STEPS}]")

        outputs, _ = self.gru(x)
        last = (lengths.to(dtype=torch.long) - 1).clamp_min(0)
        h = outputs[torch.arange(x.shape[0], device=x.device), last]
        h = torch.where((lengths > 0).unsqueeze(1), h, torch.zeros_like(h))
        logits = self.head(h)
        return h, logits


def behavioral_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean soft-action cross-entropy plus two binary behavior losses.

    The first four target values are observed action fractions and must sum to
    one.  The final two are observed fold/showdown binary targets.  No target
    is a normal-play or collusion label.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("logits and targets must be torch.Tensors")
    if logits.dtype != torch.float32 or targets.dtype != torch.float32:
        raise TypeError("logits and targets must have dtype torch.float32")
    if logits.ndim != 2 or logits.shape[1] != TARGET_DIM or logits.shape[0] == 0:
        raise ValueError("logits must have shape [B, 6] with B > 0")
    if targets.shape != logits.shape or targets.device != logits.device:
        raise ValueError("targets must match logits shape and device")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("logits and targets must be finite")
    action_targets = targets[:, :ACTION_CLASSES]
    binary_targets = targets[:, ACTION_CLASSES:]
    if bool((action_targets < 0).any()) or not bool(torch.allclose(action_targets.sum(1), torch.ones_like(action_targets[:, 0]), rtol=0.0, atol=1e-6)):
        raise ValueError("the first four target values must be nonnegative fractions summing to one")
    if bool(((binary_targets < 0) | (binary_targets > 1)).any()):
        raise ValueError("the final two target values must lie in [0, 1]")
    action_loss = -(action_targets * F.log_softmax(logits[:, :ACTION_CLASSES], dim=1)).sum(1).mean()
    binary_loss = F.binary_cross_entropy_with_logits(logits[:, ACTION_CLASSES:], binary_targets, reduction="mean")
    return action_loss + binary_loss
