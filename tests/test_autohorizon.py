import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from deploy.autohorizon_runtime import action_attention, AutoHorizonCollector, select_horizon


class AutoHorizonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)

    def test_full_softmax_denominator_and_grouped_heads(self):
        q = torch.randn(2, 7, 4, 8)
        k = torch.randn(2, 12, 2, 8)
        mask = torch.ones(2, 7, 12, dtype=torch.bool)
        mask[:, :, 1] = False
        full = torch.einsum('bqhd,bkhd->bhqk', q, k.repeat_interleave(2, dim=2)) / 8**0.5
        probs = full.masked_fill(~mask[:, None], -2.3819763e38).softmax(-1)
        expected = probs[..., -5:, -5:].mean(1)
        torch.testing.assert_close(action_attention(q, k, mask, 5), expected)
        self.assertTrue((expected.sum(-1) < 1).all())

    def test_known_attention_patterns(self):
        self.assertEqual(select_horizon(torch.eye(50))[0], 50)
        pinned = torch.zeros(50, 50)
        pinned[:, 0] = 1
        self.assertEqual(select_horizon(pinned, method='forward')[0], 1)
        for method in ['bidir', 'forward']:
            k, _ = select_horizon(torch.rand(50, 50), method)
            self.assertTrue(1 <= k <= 50)
        with self.assertRaises(ValueError):
            select_horizon(torch.zeros(50, 50))

    def test_collection_step_reset_and_unchanged_velocity(self):
        q = torch.randn(1, 6, 4, 8)
        k = torch.randn(1, 10, 2, 8)
        mask = torch.ones(1, 6, 10, dtype=torch.bool)
        def attention(q, k, v, mask):
            return q * 2
        expert = SimpleNamespace(attention_interface=attention)
        def velocity(x):
            for _ in range(3):
                out = expert.attention_interface(q, k, k, mask)
            return x + out.sum()
        model = SimpleNamespace(config=SimpleNamespace(num_steps=5),
                                qwenvl_with_expert=expert, predict_velocity=velocity)
        expected = model.predict_velocity(torch.tensor(2.))
        collector = AutoHorizonCollector(model, horizon=5)
        expert.attention_interface(q, k, k, mask)  # Prefix forward must not count.
        for _ in range(5):
            torch.testing.assert_close(model.predict_velocity(torch.tensor(2.)), expected)
        self.assertEqual(collector.layers, 3)
        torch.testing.assert_close(collector.finish(), action_attention(q, k, mask, 5) / 5)
        collector.reset()
        with self.assertRaises(RuntimeError):
            collector.finish()

    def test_matches_upstream_selectors(self):
        source = Path(os.environ.get('AUTOHORIZON_ROOT', '/root/autodl-tmp/projects/AutoHorizon')) / 'src/openpi/models_pytorch/pi0_pytorch.py'
        if not source.exists():
            self.skipTest('Upstream checkout unavailable')
        tree = ast.parse(source.read_text())
        names = {'pick_horizon_softpointer', '_soft_pointer_prefix', 'bidir_soft_pointer'}
        module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        import math
        import torch.nn.functional as F
        namespace = {'torch': torch, 'math': math, 'F': F}
        exec(compile(module, str(source), 'exec'), namespace)
        for _ in range(10):
            attention = torch.rand(50, 50)
            for method, name in [('bidir', 'bidir_soft_pointer'), ('forward', 'pick_horizon_softpointer')]:
                expected, _ = namespace[name](attention)
                self.assertEqual(select_horizon(attention, method)[0], int(expected))


if __name__ == '__main__':
    unittest.main()
