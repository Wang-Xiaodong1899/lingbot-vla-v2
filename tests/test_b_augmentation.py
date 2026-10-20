"""CPU checks for B augmentation, without importing model/CUDA dependencies."""
import ast
import math
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import zlib

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative_path, names):
    tree = ast.parse((ROOT / relative_path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(torch=torch, Tensor=torch.Tensor, F=F, math=math, os=os, zlib=zlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative_path, 'exec'), namespace)
    return namespace


class BAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        for key in ('LINGBOT_B_AUG', 'LINGBOT_AUG_LEVEL', 'LINGBOT_STATE_NOISE_STD'):
            os.environ.pop(key, None)
        self.tf = load_functions('lingbotvla/data/vla_data/transform.py', {
            '_visual_hw', 'get_visual_aug_level', '_view_generator', '_urand',
            '_hue_matrix', 'extended_visual_augmentation',
            'sample_visual_augmentation_params', 'apply_visual_augmentation', 'prepare_images',
        })
        # Read the actual parameter tables too, without importing optional dependencies.
        tree = ast.parse((ROOT / 'lingbotvla/data/vla_data/transform.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ('_AUG_LEVELS', '_HEAD_HINTS', '_YIQ')
                for t in node.targets
            ):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<constants>', 'exec'), self.tf)
        torch.manual_seed(42)
        self.image = torch.randint(0, 256, (3, 48, 64), dtype=torch.uint8)

    def test_default_matches_original_colors_and_random_sequence(self):
        params = self.tf['sample_visual_augmentation_params'](self.image)
        self.assertNotIn('seed', params)
        actual = self.tf['apply_visual_augmentation'](self.image, params)
        expected = self.image.float() / 255
        expected = expected * params['brightness']
        mean = expected.mean(dim=[0, 1, 2], keepdim=True)
        expected = (expected - mean) * params['contrast'] + mean
        gray = expected.mean(dim=0, keepdim=True)
        expected = gray + (expected - gray) * params['saturation']
        expected = (expected.clamp(0, 1) * 255).round().to(torch.uint8)
        self.assertTrue(torch.equal(actual, expected))
        os.environ['LINGBOT_B_AUG'] = '1'
        os.environ['LINGBOT_AUG_LEVEL'] = '0'
        self.assertEqual(self.tf['get_visual_aug_level'](), 0)

    def test_enabled_shapes_ranges_and_temporal_replay(self):
        os.environ['LINGBOT_B_AUG'] = '1'
        self.assertEqual(self.tf['get_visual_aug_level'](), 1)
        params = self.tf['sample_visual_augmentation_params'](self.image)
        self.assertIn('seed', params)
        for image in (self.image, self.image.float() / 255, self.image.repeat(3, 1, 1, 1)):
            result = self.tf['apply_visual_augmentation'](image, params, 'camera_top')
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, image.dtype)
            self.assertTrue(torch.isfinite(result).all())
            self.assertGreaterEqual(result.min().item(), 0)
            self.assertLessEqual(result.max().item(), 255 if image.dtype == torch.uint8 else 1)
        # Disable only fresh pixel noise to verify exact geometry/color/occlusion replay.
        self.tf['_AUG_LEVELS'][1]['noise'] = 0
        clip = self.image.repeat(3, 1, 1, 1)
        current = self.tf['apply_visual_augmentation'](self.image, params, 'camera_top')
        future = self.tf['apply_visual_augmentation'](clip, params, 'camera_top')
        for frame in future:
            self.assertTrue(torch.equal(current, frame))
        wrist = self.tf['apply_visual_augmentation'](self.image, params, 'camera_wrist_left')
        self.assertFalse(torch.equal(current, wrist))

    def test_b_enables_images_even_when_config_disabled(self):
        tree = ast.parse((ROOT / 'lingbotvla/data/vla_data/utils.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FeatureTransform')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == 'image_augment' for t in n.targets
        ))
        obj = types.SimpleNamespace()
        namespace = dict(self=obj, image_augment=False, get_visual_aug_level=self.tf['get_visual_aug_level'])
        code = compile(ast.Module(body=[assignment], type_ignores=[]), '<image_augment>', 'exec')
        exec(code, namespace)
        self.assertFalse(obj.image_augment)
        os.environ['LINGBOT_B_AUG'] = '1'
        exec(code, namespace)
        self.assertTrue(obj.image_augment)

    def test_image_preprocessing_inference_bypasses_augmentation(self):
        os.environ['LINGBOT_B_AUG'] = '1'
        observation = {'state': torch.zeros(3), 'image': {'camera_top': self.image}}
        run = lambda train: self.tf['prepare_images'](
            None, observation, image_keys=['camera_top'], train=train,
            return_augment_params=True,
        )
        images, _, _, _, params = run(False)
        self.assertIsNone(params)
        self.assertTrue(torch.equal(images[0], self.image.float() / 127.5 - 1))
        augmented, _, _, _, params = run(True)
        self.assertIn('seed', params)
        self.assertFalse(torch.equal(augmented, images))

    def test_state_noise_training_only_actions_unchanged(self):
        tree = ast.parse((ROOT / 'lingbotvla/data/vla_data/utils.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FeatureTransform')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'apply')
        namespace = dict(torch=torch, os=os)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<FeatureTransform.apply>', 'exec'), namespace)
        obj = types.SimpleNamespace(
            org_features={'actions': ['action']}, actions_convert_from_state=[], chunk_size=2,
            convert_features=lambda item, **kw: dict(item), actions=['action'],
            action_subtract_state={'action': False}, states=['state'],
            normalizer=types.SimpleNamespace(normalize=lambda item: dict(item)),
            return_item_befor_padding=True,
        )
        item = {'state': torch.zeros(20000), 'action': torch.ones(2, 3), 'action_is_pad': torch.zeros(2)}
        run = lambda eval=False: namespace['apply'](obj, item, policy_eval=eval)
        self.assertTrue(torch.equal(run()['state'], item['state']))
        os.environ['LINGBOT_B_AUG'] = '1'
        result = run()
        self.assertAlmostEqual(result['state'].std().item(), 0.01, delta=0.0003)
        self.assertTrue(torch.equal(result['action'], item['action']))
        self.assertTrue(torch.equal(item['state'], torch.zeros(20000)))
        self.assertTrue(torch.equal(run(True)['state'], item['state']))
        os.environ['LINGBOT_STATE_NOISE_STD'] = '0'
        self.assertTrue(torch.equal(run()['state'], item['state']))


if __name__ == '__main__':
    unittest.main()
