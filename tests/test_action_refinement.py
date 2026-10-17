import unittest
from types import SimpleNamespace

import torch
from torch import nn

from lingbotvla.models.vla.lingbot_vla.action_refinement import (
    ActionFlowRefiner, ActionRefinerConfig, FrozenLingbotProposal,
    action_valid_mask, masked_error,
)


class RefinerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = ActionRefinerConfig(action_dim=3, state_dim=3, horizon=4,
                                          context_dim=6, hidden_dim=16, num_layers=2, num_heads=2)
        self.model = ActionFlowRefiner(self.config)
        self.proposal = torch.randn(2, 4, 3)
        self.state = torch.randn(2, 3)
        self.context = torch.randn(2, 6)

    def test_initial_refinement_is_identity_and_does_not_mutate_proposal(self):
        original = self.proposal.clone()
        actual = self.model.refine(self.proposal, self.state, self.context)
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
        torch.testing.assert_close(self.proposal, original, rtol=0, atol=0)

    def test_euler_direction_reaches_known_target(self):
        delta = torch.ones_like(self.proposal) * 0.75
        self.model.forward = lambda *args, **kwargs: delta
        actual = self.model.refine(self.proposal, self.state, self.context, num_steps=3)
        torch.testing.assert_close(actual, self.proposal + delta)

    def test_loss_matches_analytic_flow_and_masks_padding(self):
        target = self.proposal + 2
        joints = torch.tensor([[True, False, True], [True, False, True]])
        padding = torch.tensor([[False, False, True, True], [False, False, False, True]])
        mask = action_valid_mask(target, joints, padding)
        target = torch.where(mask, target, float("nan"))
        loss, metrics = self.model.flow_loss(self.proposal, target, self.state, self.context,
                                             joints, padding, time=torch.tensor([0.2, 0.8]))
        self.assertEqual(metrics["valid_count"].item(), 10)
        self.assertAlmostEqual(loss.item(), 2.0, places=6)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.assertGreater(self.model.output[-1].weight.grad.abs().sum().item(), 0)

    def test_padding_values_cannot_affect_real_tokens(self):
        nn.init.normal_(self.model.output[-1].weight, std=0.1)
        joints = torch.tensor([[True, False, True], [True, False, True]])
        changed = self.proposal.clone()
        changed[:, :, 1] = 1e6
        first = self.model.refine(self.proposal, self.state, self.context, joints)
        second = self.model.refine(changed, self.state, self.context, joints)
        torch.testing.assert_close(first[:, :, [0, 2]], second[:, :, [0, 2]])
        torch.testing.assert_close(second[:, :, 1], changed[:, :, 1])

    def test_all_padded_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no valid"):
            self.model.flow_loss(self.proposal, self.proposal, self.state, self.context,
                                 action_is_pad=torch.ones(2, 4, dtype=torch.bool))

    def test_masked_metric_counts_real_joints_and_frames(self):
        mask = torch.zeros_like(self.proposal, dtype=torch.bool)
        mask[:, :2, :1] = True
        total, count = masked_error(self.proposal + 3, self.proposal, mask, "mse")
        self.assertEqual(count.item(), 4)
        self.assertAlmostEqual(total.item(), 36, places=5)

    def test_base_sampling_has_no_gt_and_captures_last_valid_language_feature(self):
        class Core(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))
                self.norm = nn.Identity()
                self.qwenvl_with_expert = SimpleNamespace(
                    qwenvl=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(norm=self.norm))))

            def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                               noise=None, image_grid_thw=None):
                features = torch.arange(2 * 7 * 6).reshape(2, 7, 6).float()
                self.norm(features)
                return torch.ones(2, 4, 3) * self.weight

        policy = nn.Module()
        policy.model = Core()
        base = FrozenLingbotProposal(policy)
        batch = dict(images=torch.zeros(2, 3), img_masks=torch.ones(2, 3, dtype=torch.bool),
                     lang_tokens=torch.zeros(2, 4, dtype=torch.long),
                     lang_masks=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
                     state=self.state, actions=torch.full_like(self.proposal, float("nan")))
        proposal, context = base(batch)
        expected = torch.arange(2 * 7 * 6).reshape(2, 7, 6).float()[[0, 1], [4, 5]]
        torch.testing.assert_close(context, expected)
        self.assertFalse(proposal.requires_grad)
        self.assertFalse(context.requires_grad)
        self.assertFalse(policy.training)
        self.assertTrue(all(not p.requires_grad for p in policy.parameters()))
        self.assertEqual(len(policy.model.norm._forward_hooks), 0)
        base.train()
        self.assertFalse(policy.training)


if __name__ == "__main__":
    unittest.main()
