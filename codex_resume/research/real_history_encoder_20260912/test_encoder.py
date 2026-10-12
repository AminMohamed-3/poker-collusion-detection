"""Synthetic mechanical tests for the pure observed-history encoder."""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from real_history_encoder import HistoryEncoder, behavioral_loss  # noqa: E402


torch.set_num_threads(2)
torch.set_num_interop_threads(2)


class HistoryEncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(48271)
        self.model = HistoryEncoder(input_dim=10, hidden_dim=16)

    def test_shape_finite_and_backward(self) -> None:
        x = torch.randn(3, 20, 10, dtype=torch.float32, requires_grad=True)
        lengths = torch.tensor([20, 7, 1], dtype=torch.int64)
        h, logits = self.model(x, lengths)
        targets = torch.tensor([[.25, .25, .25, .25, 0., 1.], [.5, .25, .25, 0., 1., 0.], [0., 0., 1., 0., 1., 1.]], dtype=torch.float32)
        loss = behavioral_loss(logits, targets)
        loss.backward()
        self.assertEqual(h.shape, (3, 16))
        self.assertEqual(logits.shape, (3, 6))
        self.assertTrue(torch.isfinite(h).all())
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()))

    def test_empty_history_is_exact_zero_before_head(self) -> None:
        x = torch.randn(2, 20, 10, dtype=torch.float32)
        h, logits = self.model(x, torch.tensor([0, 0], dtype=torch.int64))
        self.assertTrue(torch.equal(h, torch.zeros_like(h)))
        self.assertTrue(torch.equal(logits, self.model.head(torch.zeros_like(h))))

    def test_padding_and_future_suffix_do_not_change_selected_prefix(self) -> None:
        x = torch.randn(3, 20, 10, dtype=torch.float32)
        lengths = torch.tensor([3, 11, 0], dtype=torch.int64)
        changed = x.clone()
        for row, length in enumerate(lengths.tolist()):
            changed[row, length:] = torch.randn_like(changed[row, length:]) * 1000
        h_a, logits_a = self.model(x, lengths)
        h_b, logits_b = self.model(changed, lengths)
        self.assertTrue(torch.equal(h_a, h_b))
        self.assertTrue(torch.equal(logits_a, logits_b))

    def test_soft_cross_entropy_and_binary_loss_match_manual_formula(self) -> None:
        logits = torch.tensor([[1.2, -.4, .3, .9, -1.1, .7], [-.2, .8, .1, -.5, .4, -2.]], dtype=torch.float32)
        targets = torch.tensor([[.1, .2, .3, .4, 0., 1.], [.25, .25, .25, .25, 1., 0.]], dtype=torch.float32)
        expected = (-(targets[:, :4] * F.log_softmax(logits[:, :4], dim=1)).sum(1).mean()
                    + F.binary_cross_entropy_with_logits(logits[:, 4:], targets[:, 4:], reduction="mean"))
        self.assertTrue(torch.equal(behavioral_loss(logits, targets), expected))

    def test_cpu_state_dict_reload_is_exact(self) -> None:
        x = torch.randn(4, 20, 10, dtype=torch.float32)
        lengths = torch.tensor([0, 4, 12, 20], dtype=torch.int64)
        before = self.model(x, lengths)
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        replay = HistoryEncoder(input_dim=10, hidden_dim=16)
        replay.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
        after = replay(x, lengths)
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1], after[1]))

    def test_strict_input_validation(self) -> None:
        good = torch.zeros(1, 20, 10, dtype=torch.float32)
        with self.assertRaises(TypeError):
            self.model(good.to(torch.float64), torch.tensor([1], dtype=torch.int64))
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 19, 10, dtype=torch.float32), torch.tensor([1], dtype=torch.int64))
        with self.assertRaises(TypeError):
            self.model(good, torch.tensor([1.], dtype=torch.float32))
        with self.assertRaises(ValueError):
            self.model(good, torch.tensor([21], dtype=torch.int64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
