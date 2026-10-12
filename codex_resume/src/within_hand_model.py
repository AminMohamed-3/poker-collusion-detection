"""Small deterministic model for bounded public within-hand action prefixes.

This module consumes already extracted public tokens and supplied observed action
classes.  It contains no poker-data I/O, labels beyond those action classes, or
fold/disk orchestration.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PREFIX_STEPS = 24
TOKEN_DIM = 18
QUERY_DIM = 37
HIDDEN_DIM = 16
ACTION_CLASSES = 4
BATCH_SIZE = 512
DEFAULT_SEED = 71931
Arm = Literal["ordered", "shuffled", "query_only"]


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def configure_deterministic_cpu(seed: int = DEFAULT_SEED) -> None:
    """Set the fixed CPU execution state used by the frozen protocol.

    The NumPy minibatch generator is deliberately local to ``train_final`` so
    every model begins its batch ordering from the same seed.
    """
    _positive_int(seed, "seed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def _validate_tensor_inputs(
    query: torch.Tensor, x: torch.Tensor, lengths: torch.Tensor
) -> None:
    if not isinstance(query, torch.Tensor) or not isinstance(x, torch.Tensor):
        raise TypeError("query and x must be torch.Tensors")
    if query.dtype != torch.float32 or x.dtype != torch.float32:
        raise TypeError("query and x must have dtype torch.float32")
    if query.ndim != 2 or query.shape[0] == 0 or query.shape[1] != QUERY_DIM:
        raise ValueError(f"query must have shape [B, {QUERY_DIM}] with B > 0")
    if x.ndim != 3 or x.shape != (query.shape[0], PREFIX_STEPS, TOKEN_DIM):
        raise ValueError(f"x must have shape [B, {PREFIX_STEPS}, {TOKEN_DIM}]")
    if not isinstance(lengths, torch.Tensor):
        raise TypeError("lengths must be a torch.Tensor")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("lengths must have dtype int32 or int64")
    if lengths.ndim != 1 or lengths.shape[0] != query.shape[0]:
        raise ValueError("lengths must have shape [B]")
    if query.device != x.device or query.device != lengths.device:
        raise ValueError("query, x, and lengths must share a device")
    if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(x).all()):
        raise ValueError("query and x must be finite")
    if bool(((lengths < 0) | (lengths > PREFIX_STEPS)).any()):
        raise ValueError(f"lengths must lie in [0, {PREFIX_STEPS}]")


def _validate_arm(arm: str) -> Arm:
    if arm not in ("ordered", "shuffled", "query_only"):
        raise ValueError("arm must be ordered, shuffled, or query_only")
    return arm  # type: ignore[return-value]


class QueryPrefixModel(nn.Module):
    """Predict the observed four-class action from a query and public prefix.

    ``query`` must already be normalized with training-row-only statistics.
    For ``ordered`` and ``shuffled`` the supplied token order is used directly;
    callers perform the fixed external shuffle.  ``query_only`` does not invoke
    the GRU and uses an exact all-zero hidden vector for every row.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(TOKEN_DIM, HIDDEN_DIM, num_layers=1, batch_first=True, dropout=0.0)
        self.fc1 = nn.Linear(QUERY_DIM + HIDDEN_DIM, 64)
        self.fc2 = nn.Linear(64, ACTION_CLASSES)

    def forward(
        self, query: torch.Tensor, x: torch.Tensor, lengths: torch.Tensor, arm: str
    ) -> torch.Tensor:
        _validate_tensor_inputs(query, x, lengths)
        selected_arm = _validate_arm(arm)
        if selected_arm == "query_only":
            hidden = torch.zeros((query.shape[0], HIDDEN_DIM), dtype=query.dtype, device=query.device)
        else:
            outputs, _ = self.gru(x)
            last = (lengths.to(dtype=torch.long) - 1).clamp_min(0)
            hidden = outputs[torch.arange(query.shape[0], device=query.device), last]
            hidden = torch.where((lengths > 0).unsqueeze(1), hidden, torch.zeros_like(hidden))
        return self.fc2(F.silu(self.fc1(torch.cat((query, hidden), dim=1))) )


def _float32_array(value: np.ndarray, name: str, ndim: int, shape_tail: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if value.ndim != ndim or value.shape[0] == 0 or value.shape[1:] != shape_tail:
        dims = "B, " + ", ".join(str(dim) for dim in shape_tail)
        raise ValueError(f"{name} must have shape [{dims}] with B > 0")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _length_array(value: np.ndarray, rows: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("lengths must be a numpy array")
    if value.dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise TypeError("lengths must have dtype int32 or int64")
    if value.ndim != 1 or value.shape[0] != rows:
        raise ValueError("lengths must have shape [B]")
    if np.any(value < 0) or np.any(value > PREFIX_STEPS):
        raise ValueError(f"lengths must lie in [0, {PREFIX_STEPS}]")
    return value


def _target_array(value: np.ndarray, rows: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("targets must be a numpy array")
    if value.dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise TypeError("targets must have dtype int32 or int64")
    if value.ndim != 1 or value.shape[0] != rows:
        raise ValueError("targets must have shape [B]")
    if np.any(value < 0) or np.any(value >= ACTION_CLASSES):
        raise ValueError("targets must lie in [0, 3]")
    return value


def _validate_arrays(
    query: np.ndarray, x: np.ndarray, lengths: np.ndarray, targets: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    query = _float32_array(query, "query", 2, (QUERY_DIM,))
    x = _float32_array(x, "x", 3, (PREFIX_STEPS, TOKEN_DIM))
    if x.shape[0] != query.shape[0]:
        raise ValueError("query and x must have equal row counts")
    lengths = _length_array(lengths, query.shape[0])
    if targets is not None:
        targets = _target_array(targets, query.shape[0])
    return query, x, lengths, targets


def query_normalization(training_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 mean/std using only the supplied training query rows."""
    query = _float32_array(training_query, "training_query", 2, (QUERY_DIM,))
    mean = np.mean(query, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(query, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    return mean, std


def normalize_query(query: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    query = _float32_array(query, "query", 2, (QUERY_DIM,))
    mean = _float32_array(mean, "mean", 1, ())
    std = _float32_array(std, "std", 1, ())
    if mean.shape != (QUERY_DIM,) or std.shape != (QUERY_DIM,):
        raise ValueError(f"mean and std must have shape [{QUERY_DIM}]")
    if np.any(std < np.float32(1e-6)):
        raise ValueError("std must be at least 1e-6")
    normalized = ((query - mean) / std).astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized query must be finite")
    return normalized


def epoch_minibatches(row_count: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Return one complete fixed-size permutation partition for a training epoch."""
    _positive_int(row_count, "row_count")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    order = rng.permutation(row_count)
    return tuple(order[start:start + BATCH_SIZE] for start in range(0, row_count, BATCH_SIZE))

def train_final(
    model: QueryPrefixModel,
    query: np.ndarray,
    x: np.ndarray,
    lengths: np.ndarray,
    targets: np.ndarray,
    arm: str,
    *,
    epochs: int = 6,
    numpy_seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a supplied CPU model for final fixed-epoch action prediction.

    Statistics are calculated solely from these supplied training rows.  The
    local NumPy generator is reset on every call, fixing minibatch order for
    each fold and arm.  No checkpoint selection or disk I/O occurs here.
    """
    if not isinstance(model, QueryPrefixModel):
        raise TypeError("model must be a QueryPrefixModel")
    _positive_int(epochs, "epochs")
    _positive_int(numpy_seed, "numpy_seed")
    _validate_arm(arm)
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("train_final requires a CPU model")
    query, x, lengths, targets = _validate_arrays(query, x, lengths, targets)
    assert targets is not None
    mean, std = query_normalization(query)
    normalized = normalize_query(query, mean, std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    query_tensor = torch.from_numpy(normalized)
    x_tensor = torch.from_numpy(x)
    length_tensor = torch.from_numpy(lengths)
    target_tensor = torch.from_numpy(targets.astype(np.int64, copy=False))
    rng = np.random.default_rng(numpy_seed)
    model.train()
    for _ in range(epochs):
        for indices in epoch_minibatches(query.shape[0], rng):
            index_tensor = torch.from_numpy(indices.astype(np.int64, copy=False))
            optimizer.zero_grad(set_to_none=True)
            logits = model(query_tensor[index_tensor], x_tensor[index_tensor], length_tensor[index_tensor], arm)
            loss = F.cross_entropy(logits, target_tensor[index_tensor])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return mean, std


def predict_probabilities(
    model: QueryPrefixModel,
    query: np.ndarray,
    x: np.ndarray,
    lengths: np.ndarray,
    arm: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Return fixed-512-batch float32 softmax probabilities from a CPU model."""
    if not isinstance(model, QueryPrefixModel):
        raise TypeError("model must be a QueryPrefixModel")
    _validate_arm(arm)
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("predict_probabilities requires a CPU model")
    query, x, lengths, _ = _validate_arrays(query, x, lengths)
    normalized = normalize_query(query, mean, std)
    query_tensor = torch.from_numpy(normalized)
    x_tensor = torch.from_numpy(x)
    length_tensor = torch.from_numpy(lengths)
    was_training = model.training
    model.eval()
    batches: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, query.shape[0], BATCH_SIZE):
                stop = min(start + BATCH_SIZE, query.shape[0])
                probabilities = torch.softmax(
                    model(query_tensor[start:stop], x_tensor[start:stop], length_tensor[start:stop], arm), dim=1
                )
                batches.append(probabilities.cpu().numpy().astype(np.float32, copy=True))
    finally:
        model.train(was_training)
    return np.concatenate(batches, axis=0)


def load_plain_state_dict(model: QueryPrefixModel, path: str | Path) -> QueryPrefixModel:
    """Load a plain state dict with PyTorch's safe ``weights_only`` path."""
    if not isinstance(model, QueryPrefixModel):
        raise TypeError("model must be a QueryPrefixModel")
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint must contain a plain state dict")
    model.load_state_dict(state)
    return model

