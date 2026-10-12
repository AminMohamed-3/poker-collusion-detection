from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from within_hand_model import (  # noqa: E402
    ACTION_CLASSES,
    PREFIX_STEPS,
    QUERY_DIM,
    TOKEN_DIM,
    QueryPrefixModel,
    configure_deterministic_cpu,
    epoch_minibatches,
    load_plain_state_dict,
    predict_probabilities,
    train_final,
)


class QueryPrefixModelTests(unittest.TestCase):
    def setUp(self):
        configure_deterministic_cpu()
        rng = np.random.default_rng(90210)
        self.query = rng.normal(size=(9, QUERY_DIM)).astype(np.float32)
        self.x = rng.normal(size=(9, PREFIX_STEPS, TOKEN_DIM)).astype(np.float32)
        self.lengths = np.array([0, 1, 2, 3, 7, 9, 12, 20, 24], dtype=np.int64)
        self.targets = np.arange(9, dtype=np.int64) % ACTION_CLASSES

    def tensors(self):
        return torch.from_numpy(self.query), torch.from_numpy(self.x), torch.from_numpy(self.lengths)

    def test_padding_after_valid_length_is_ignored(self):
        model = QueryPrefixModel().eval()
        query, x, lengths = self.tensors()
        changed = x.clone()
        for row, length in enumerate(self.lengths):
            changed[row, length:, :] = 123.0 + row
        with torch.no_grad():
            expected = model(query, x, lengths, "ordered")
            actual = model(query, changed, lengths, "ordered")
        torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)

    def test_empty_nonempty_and_backward_are_finite(self):
        model = QueryPrefixModel()
        query, x, lengths = self.tensors()
        logits = model(query, x, lengths, "ordered")
        self.assertEqual(tuple(logits.shape), (9, ACTION_CLASSES))
        self.assertTrue(torch.isfinite(logits).all())
        zero_logits = model(query[:1], x[:1], lengths[:1], "ordered")
        manual = model.fc2(torch.nn.functional.silu(model.fc1(torch.cat((query[:1], torch.zeros((1, 16))), dim=1))))
        torch.testing.assert_close(zero_logits, manual, rtol=0.0, atol=0.0)
        torch.nn.functional.cross_entropy(logits, torch.from_numpy(self.targets)).backward()
        self.assertTrue(any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_query_only_does_not_use_prefix_or_lengths(self):
        model = QueryPrefixModel().eval()
        query, x, lengths = self.tensors()
        alternative = torch.full_like(x, 91.0)
        alternative_lengths = torch.full_like(lengths, PREFIX_STEPS)
        with torch.no_grad():
            expected = model(query, x, lengths, "query_only")
            actual = model(query, alternative, alternative_lengths, "query_only")
        torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)

    def test_epoch_minibatches_cover_each_row_once_with_partial_final_batch(self):
        first = epoch_minibatches(513, np.random.default_rng(71931))
        second = epoch_minibatches(513, np.random.default_rng(71931))
        self.assertEqual([len(batch) for batch in first], [512, 1])
        combined = np.concatenate(first)
        np.testing.assert_array_equal(np.sort(combined), np.arange(513, dtype=combined.dtype))
        self.assertEqual(np.unique(combined).size, 513)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
    def test_initialization_and_minibatch_training_are_reproducible(self):
        configure_deterministic_cpu(71931)
        first = QueryPrefixModel()
        initial = copy.deepcopy(first.state_dict())
        configure_deterministic_cpu(71931)
        second = QueryPrefixModel()
        for name, value in initial.items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0.0, atol=0.0)
        first.load_state_dict(initial)
        second.load_state_dict(initial)
        first_norm = train_final(first, self.query, self.x, self.lengths, self.targets, "ordered", epochs=1)
        second_norm = train_final(second, self.query, self.x, self.lengths, self.targets, "ordered", epochs=1)
        for left, right in zip(first_norm, second_norm):
            np.testing.assert_array_equal(left, right)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0.0, atol=0.0)

    def test_fixed_batch_probability_reload_is_exact(self):
        model = QueryPrefixModel()
        mean, std = train_final(model, self.query, self.x, self.lengths, self.targets, "shuffled", epochs=1)
        expected = predict_probabilities(model, self.query, self.x, self.lengths, "shuffled", mean, std)
        self.assertEqual(expected.dtype, np.float32)
        np.testing.assert_allclose(expected.sum(axis=1), np.ones(expected.shape[0], dtype=np.float32), rtol=0.0, atol=1e-7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save(model.state_dict(), path)
            restored = load_plain_state_dict(QueryPrefixModel(), path)
            actual = predict_probabilities(restored, self.query, self.x, self.lengths, "shuffled", mean, std)
        np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)


