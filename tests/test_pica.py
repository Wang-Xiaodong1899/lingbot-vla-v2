"""CPU checks of actual V2 forward methods, with small substitute attention blocks.

The production module imports CUDA-only fused kernels even at import time, so
we compile its forward methods from the AST (as in test_b_augmentation.py).
This checks layer selection, loss integration and gradients without CUDA.
"""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from lingbotvla.utils.pica import (
    action_alignment_loss, attach_pica_projector, update_pica_teacher, validate_pica_config,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(student_layer=0, teacher_layer=2, weight=0.05, ema_decay=0.9999, projector_hidden_dim=5)


def load_augmentation():
    path = ROOT / 'lingbotvla/data/vla_data/pica_augmentation.py'
    spec = importlib.util.spec_from_file_location('pica_augmentation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def production_forward(class_name):
    path = ROOT / 'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    env = dict(torch=torch, F=F, make_att_2d_masks=lambda pad, att: pad[:, :, None] & pad[:, None, :])
    exec(compile(module, str(path), 'exec'), env)
    return env['forward']


class Block(nn.Module):
    def __init__(self, action):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4) * 0.1)
        self.action = action
        self.calls = 0

    def forward(self, x, attention=None, start=None, end=None, compute_kqv=False, **kwargs):
        if compute_kqv:
            self.calls += 1
            q = x.unsqueeze(2)
            return q, q, q
        out = x + torch.tanh(x @ self.weight) + attention[:, start:end].squeeze(2)
        return (out, None) if self.action else out


class Stream(nn.Module):
    def __init__(self, action):
        super().__init__()
        self.layers = nn.ModuleList(Block(action) for _ in range(4))
        self.norm = nn.Identity()


class Backbone(nn.Module):
    forward = production_forward('QwenvlWithExpertV2Model')

    def __init__(self):
        super().__init__()
        self.qwenvl = nn.Module()
        self.qwenvl.model = nn.Module()
        self.qwenvl.model.language_model = Stream(False)
        self.qwenvl.config = NS(text_config=NS(num_hidden_layers=4))
        self.qwen_expert = nn.Module()
        self.qwen_expert.model = Stream(True)
        self.config = NS(qwen_expert_config=NS(num_hidden_layers=4),
                         attention_implementation='test', final_norm_adanorm=False)

    def apply_mrope(self, q, k, pos):
        return q, k

    def handle_kv_cache(self, k, v, *args, **kwargs):
        return k, v, None

    def attention_interface(self, q, k, v, mask):
        # Let action tokens depend on the visual prefix.
        return v.mean(dim=1, keepdim=True).expand_as(q) * 0.1

    def _apply_deepstack(self, x, *args):
        return x


class Flow(nn.Module):
    forward = production_forward('FlowMatchingV2')

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qwenvl_with_expert = Backbone()
        self.action_out_proj = nn.Linear(4, 4)
        self.block_future_depth_to_action = False

    def embed_prefix(self, images, *args, **kwargs):
        masks = torch.ones(images.shape[:2], dtype=torch.bool)
        return images, masks, masks, None, None, None

    def embed_suffix(self, state, x, time):
        self.last_noisy_action = x.detach().clone()
        self.last_time = time.detach().clone()
        x = torch.cat([state[:, None], x], dim=1)
        masks = torch.ones(x.shape[:2], dtype=torch.bool)
        return None, x, masks, masks

    def _block_suffix_to_future_video_if_enabled_(self, mask, **kwargs):
        return mask

    def _build_full_position_ids(self, *args):
        return None

    def _moe_losses_and_metrics(self, *args):
        return 0, 0, {}


class Policy(nn.Module):
    forward = production_forward('LingbotVlaV2Policy')

    def __init__(self):
        super().__init__()
        self.config = NS(proj_width=4, n_action_steps=3, action_dim=4, loss_type='fm',
                         action_fp32=False, use_cache=False, align_params={})
        self.model = Flow(self.config)
        # This fixture has no depth/video losses: final visual block output
        # is unused. Freeze it so strict optimizer restore has no unused slots.
        self.model.qwenvl_with_expert.qwenvl.model.language_model.layers[-1].requires_grad_(False)
        self.register_buffer('routing_bias', torch.tensor([1.0]))


def batch():
    return dict(images=torch.randn(2, 2, 4), img_masks=torch.ones(2, 2, dtype=torch.bool),
                state=torch.randn(2, 4), lang_tokens=None, lang_masks=None,
                actions=torch.randn(2, 3, 4), noise=torch.randn(2, 3, 4), time=torch.tensor([0.2, 0.8]))


class PICATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def test_temporal_sharing_camera_independence_and_labels(self):
        aug = load_augmentation()
        image = torch.randint(256, (3, 24, 24), dtype=torch.uint8)
        observation = dict(image={'left': image, 'right': image},
                           future_image={'left': image.clone(), 'right': image.clone()},
                           actions=torch.randn(3, 4), state=torch.randn(4))
        original = copy.deepcopy(observation)
        for seed in range(16):  # exercise all mixture branches and geometry
            torch.manual_seed(seed)
            result = aug.augment_observation_pair(observation)
            for camera in observation['image']:
                self.assertTrue(torch.equal(result['image'][camera], result['future_image'][camera]))
                self.assertTrue(torch.equal(observation['image'][camera], original['image'][camera]))
            self.assertFalse(torch.equal(result['image']['left'], result['image']['right']))
            self.assertIs(result['actions'], observation['actions'])
            self.assertIs(result['state'], observation['state'])

    def test_negative_cosine_padding_and_stop_gradient(self):
        projected = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]], requires_grad=True)
        target = torch.tensor([[[1., 1.], [-1., 0.], [0., -1.]]], requires_grad=True)
        pad = torch.tensor([[False, True, False]])
        joints = torch.tensor([[[True], [True], [False]]])
        loss = action_alignment_loss(projected, target, pad, joints)
        torch.testing.assert_close(loss, torch.tensor(-2 ** -0.5))
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertEqual(projected.grad[:, 1:].count_nonzero(), 0)
        self.assertGreater(projected.grad[:, :1].abs().sum(), 0)
        zero = action_alignment_loss(projected, target, torch.ones_like(pad))
        self.assertEqual(zero.item(), 0)

    def test_actual_forward_alignment_early_exit_and_optimizer(self):
        student = Policy()
        teacher = copy.deepcopy(student).requires_grad_(False).eval()
        attach_pica_projector(student, CFG)
        inputs = batch()
        base = student(**inputs)
        with torch.no_grad():
            target = teacher(**inputs, action_feature_layer=2, action_features_only=True)
        self.assertEqual(tuple(target.shape), (2, 3, 4))  # excludes state token
        self.assertEqual([b.calls for b in teacher.model.qwenvl_with_expert.qwen_expert.model.layers], [1, 1, 1, 0])
        augmented = dict(inputs, images=inputs['images'] + 0.1)
        plain = student(**augmented)
        outputs = student(**augmented, pica_target=target)
        torch.testing.assert_close(student.model.last_noisy_action, teacher.model.last_noisy_action)
        torch.testing.assert_close(student.model.last_time, teacher.model.last_time)
        expected_noisy = inputs['time'][:, None, None] * inputs['noise'] + (1-inputs['time'][:, None, None])*inputs['actions']
        torch.testing.assert_close(student.model.last_noisy_action, expected_noisy)
        self.assertEqual(len(base), len(outputs))
        torch.testing.assert_close(outputs[1], plain[1])
        torch.testing.assert_close(outputs[0], plain[0] + outputs[6]['pica/alignment_loss_weighted'])
        outputs[0].backward()
        self.assertGreater(student.pica_projector[0].weight.grad.abs().sum(), 0)
        self.assertGreater(student.model.qwenvl_with_expert.qwen_expert.model.layers[0].weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        old = {k: v.clone() for k, v in teacher.named_parameters()}
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
        optimizer.step()
        student.routing_bias.fill_(2)
        update_pica_teacher(teacher, student, CFG['ema_decay'])
        for name, parameter in teacher.named_parameters():
            expected = old[name] * 0.9999 + dict(student.named_parameters())[name] * 0.0001
            torch.testing.assert_close(parameter, expected)
        torch.testing.assert_close(teacher.routing_bias, student.routing_bias)

    def test_layer_feature_is_block_output_and_retains_graph(self):
        backbone = Backbone()
        inputs = [torch.randn(1, 2, 4), torch.randn(1, 4, 4)]
        reference = backbone(inputs_embeds=list(inputs), action_feature_layer=0, action_features_only=True)
        full = backbone(inputs_embeds=list(inputs), action_feature_layer=0)
        torch.testing.assert_close(reference, full[3])
        self.assertIsNotNone(full[3].grad_fn)
        self.assertEqual(len(backbone(inputs_embeds=list(inputs))), 3)
        with self.assertRaises(ValueError):
            backbone(inputs_embeds=list(inputs), action_feature_layer=4)

    def test_config_and_ema_precision_guards(self):
        validate_pica_config(CFG, 4)
        for update in [dict(student_layer=3), dict(teacher_layer=4), dict(weight=-1), dict(ema_decay=1)]:
            with self.assertRaises(ValueError):
                validate_pica_config({**CFG, **update}, 4)
        student = Policy()
        teacher = copy.deepcopy(student).bfloat16()
        with self.assertRaises(ValueError):
            update_pica_teacher(teacher, student, 0.9)

    def test_dcp_roundtrip_restores_student_projector_teacher_optimizer(self):
        import torch.distributed as dist
        from lingbotvla.checkpoint.checkpointer import DistributedCheckpointer
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group('gloo', init_method=f'file://{tmp}/init', rank=0, world_size=1)
            try:
                student = Policy()
                teacher = copy.deepcopy(student).requires_grad_(False)
                attach_pica_projector(student, CFG)
                optimizer = torch.optim.AdamW(student.parameters())
                inputs = batch()
                target = teacher(**inputs, action_feature_layer=2, action_features_only=True)
                student(**inputs, pica_target=target)[0].backward()
                optimizer.step()
                update_pica_teacher(teacher, student, 0.5)
                expected_s, expected_t = copy.deepcopy(student.state_dict()), copy.deepcopy(teacher.state_dict())
                state = dict(model=student, ema=teacher, optimizer=optimizer,
                             extra_state=dict(global_step=1, pica_params=CFG))
                DistributedCheckpointer.save(tmp, state, global_steps=1)
                with torch.no_grad():
                    for p in list(student.parameters()) + list(teacher.parameters()):
                        p.zero_()
                student.zero_grad(set_to_none=True)
                optimizer = torch.optim.AdamW(student.parameters())
                state['optimizer'] = optimizer
                state['extra_state'] = {}
                DistributedCheckpointer.load(f'{tmp}/global_step_1', state)
                for key, value in student.state_dict().items():
                    torch.testing.assert_close(value, expected_s[key])
                for key, value in teacher.state_dict().items():
                    torch.testing.assert_close(value, expected_t[key])
                self.assertEqual(state['extra_state']['pica_params'], CFG)
                self.assertTrue(optimizer.state)
            finally:
                dist.destroy_process_group()

    def test_hf_export_drops_projector_without_mutating_dcp(self):
        from lingbotvla.utils.async_hf_checkpoint import AsyncHFCheckpointSaver
        weights = {'model.action_out_proj.weight': torch.ones(4, 4),
                   'pica_projector.0.weight': torch.ones(5, 4)}
        saved = {}

        def save_weights(path, state, **kwargs):
            saved.update(state)
            Path(path).mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp, patch(
            'lingbotvla.utils.async_hf_checkpoint.ckpt_to_state_dict', return_value=dict(weights),
        ), patch('lingbotvla.utils.async_hf_checkpoint.save_model_weights', side_effect=save_weights):
            saver = AsyncHFCheckpointSaver(enabled=False)
            saver._save_one_hf_checkpoint(
                final_dir=f'{tmp}/hf_ckpt', checkpoint_path=tmp, output_dir=tmp,
                ckpt_manager='dcp', enable_fp32=False, model_assets=None, ema=False, global_step=1,
            )
        self.assertEqual(set(saved), {'model.action_out_proj.weight'})
        self.assertIn('pica_projector.0.weight', weights)


if __name__ == '__main__':
    unittest.main()
