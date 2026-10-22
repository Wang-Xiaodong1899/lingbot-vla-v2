"""CPU tests for Z4 probabilities, clean identity, temporal replay and foreground safety."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('z4_under_test', ROOT / 'lingbotvla/data/vla_data/z4_augmentation.py')
z4 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = z4
spec.loader.exec_module(z4)


def plan(**updates):
    p = dict(color=False, affine=False, erase=False, noise=None, blur=None, jpeg=False)
    return {**p, **updates}


class Z4Test(unittest.TestCase):
    def setUp(self):
        self.config = z4.Z4Config()
        g = torch.Generator().manual_seed(10)
        self.frame = torch.randint(0, 256, (3, 64, 96), dtype=torch.uint8, generator=g)
        self.observation = {
            'image': {'camera_top': self.frame, 'camera_wrist_left': self.frame.clone()},
            'future_image': {'camera_top': self.frame.clone(), 'camera_wrist_left': self.frame.clone()},
            'state': torch.tensor([0.5, -0.5]), 'actions': torch.ones(50, 14),
            'task': 'put the cup on the table',
        }

    def test_yaml_matches_defaults_and_rejects_invalid_settings(self):
        loaded = z4.load_config(str(ROOT / 'configs/augmentation/z4.yaml'))
        self.assertEqual(tuple(loaded.scale), self.config.scale)
        self.assertEqual(loaded.clean_probability, 0.1)
        for updates in [dict(clean_probability=1.1), dict(salt_pepper_probability=0.6),
                        dict(motion_blur_probability=0.9), dict(erase_area=(0.2, 0.1)),
                        dict(jpeg_quality=(0, 90)), dict(motion_blur_sizes=(4,)),
                        dict(scale=(0, 1)), dict(erase_probability=0.1)]:
            with self.assertRaises(ValueError):
                replace(self.config, **updates)

    def test_clean_observation_is_bit_identical_without_rng_side_effects(self):
        before = self.frame.clone()
        result, trace = z4.augment_observation(self.observation, seed=9,
                                              config=replace(self.config, clean_probability=1), return_trace=True)
        self.assertTrue(trace['clean'])
        self.assertIs(result, self.observation)
        self.assertTrue(torch.equal(before, self.frame))
        for camera in result['image']:
            self.assertTrue(torch.equal(result['image'][camera], self.observation['image'][camera]))
            self.assertTrue(torch.equal(result['future_image'][camera], self.observation['future_image'][camera]))

    def test_probabilities_and_nonempty_combinations(self):
        rng = np.random.default_rng(42)
        count = dict(color=0, affine=0, erase=0, gaussian_noise=0, salt_pepper=0,
                     gaussian_blur=0, motion_blur=0, jpeg=0)
        n = 40000
        for _ in range(n):
            p = z4.sample_plan(rng, self.config)
            self.assertTrue(any(p.values()))
            for key in ('color', 'affine', 'erase', 'jpeg'):
                count[key] += p[key]
            count['gaussian_noise'] += p['noise'] == 'gaussian'
            count['salt_pepper'] += p['noise'] == 'salt_pepper'
            count['gaussian_blur'] += p['blur'] == 'gaussian'
            count['motion_blur'] += p['blur'] == 'motion'
        expected = dict(color=.8, affine=.5, erase=.95, gaussian_noise=.5, salt_pepper=.15,
                        gaussian_blur=.25, motion_blur=.1, jpeg=.3)
        for key, probability in expected.items():
            self.assertAlmostEqual(count[key] / n, probability, delta=.008, msg=key)
        # This is the exact per-observation seed/draw used in augment_observation.
        clean = sum(np.random.default_rng(seed).random() < self.config.clean_probability for seed in range(10000))
        self.assertAlmostEqual(clean / 10000, .1, delta=.01)

    def test_all_operations_replay_over_time_and_do_not_mutate_inputs(self):
        frames = torch.stack([self.frame, self.frame])
        original = frames.clone()
        masks = np.zeros((2, 64, 96), bool)
        forced = plan(color=True, affine=True, erase=True, noise='gaussian', blur='motion', jpeg=True)
        output, trace = z4.augment_camera_clip(frames, np.random.default_rng(8), self.config, masks, forced)
        repeated, _ = z4.augment_camera_clip(frames, np.random.default_rng(8), self.config, masks, forced)
        self.assertTrue(torch.equal(output[0], output[1]))
        self.assertTrue(torch.equal(output, repeated))
        self.assertTrue(torch.equal(frames, original))
        self.assertEqual(output.dtype, torch.uint8)
        self.assertEqual(output.shape, frames.shape)
        self.assertTrue(trace['erase_applied'])
        self.assertTrue(35 <= trace['jpeg_quality'] <= 90)
        for key, limits in [('brightness', (.25, 1.1)), ('contrast', (.5, 1.5)), ('saturation', (.4, 1.6))]:
            self.assertTrue(limits[0] <= trace['color_params'][key] <= limits[1])

    def test_original_branch_is_shared_across_cameras_and_time(self):
        masks = {k: np.zeros((2, 64, 96), bool) for k in self.observation['image']}
        found = set()
        for seed in range(30):
            result, trace = z4.augment_observation(self.observation, seed=seed, foreground_masks=masks, return_trace=True)
            found.add(trace['clean'])
            self.assertIs(result['state'], self.observation['state'])
            self.assertIs(result['actions'], self.observation['actions'])
            self.assertEqual(result['task'], self.observation['task'])
            for camera in result['image']:
                self.assertTrue(torch.equal(result['image'][camera], result['future_image'][camera]))
                if trace['clean']:
                    self.assertTrue(torch.equal(result['image'][camera], self.observation['image'][camera]))
        self.assertEqual(found, {True, False})

    def test_occlusion_avoids_union_of_current_and_future_foreground(self):
        config = replace(self.config, foreground_margin=0)
        masks = np.zeros((2, 64, 96), bool)
        masks[0, 20:45, 10:30] = True
        masks[1, 20:45, 60:80] = True
        union = torch.from_numpy(masks.any(axis=0))
        frames = torch.stack([self.frame, self.frame])
        seen_modes = set()
        for seed in range(40):
            output, trace = z4.augment_camera_clip(frames, np.random.default_rng(seed), config, masks, plan(erase=True))
            self.assertTrue(torch.equal(output[:, :, union], frames[:, :, union]))
            if trace['erase_applied']:
                x, y, w, h = trace['box']
                self.assertFalse(masks[:, y:y+h, x:x+w].any())
                seen_modes.add(trace['erase_mode'])
                self.assertTrue(torch.equal(output[0], output[1]))
        self.assertEqual(seen_modes, {'solid', 'mosaic', 'pixel_noise'})

    def test_no_safe_location_skips_occlusion(self):
        frames = self.frame.unsqueeze(0)
        output, trace = z4.augment_camera_clip(frames, np.random.default_rng(2), self.config,
                                               np.ones((1, 64, 96), bool), plan(erase=True))
        self.assertFalse(trace['erase_applied'])
        self.assertIsNone(trace['box'])
        self.assertTrue(torch.equal(frames, output))

    def test_detector_protects_colored_and_low_saturation_objects(self):
        image = np.full((128, 160, 3), 150, dtype=np.uint8)
        image[30:50, 10:30] = [240, 40, 20]
        image[70:90, 130:150] = 230
        mask = z4.detect_protected_regions(image, replace(self.config, protected_center=0))
        self.assertTrue(mask[30:50, 10:30].all())
        self.assertTrue(mask[70:90, 130:150].all())
        self.assertFalse(mask[:5, :5].any())
        # A textured/uncertain image should leave little or no eligible background.
        random = self.frame.permute(1, 2, 0).numpy()
        self.assertGreater(z4.detect_protected_regions(random, self.config).mean(), .95)

    def test_affine_transforms_foreground_mask_and_keeps_black_border(self):
        config = replace(self.config, scale=(.9, .9), translation=.06, foreground_margin=0)
        frames = torch.full((2, 3, 64, 96), 160, dtype=torch.uint8)
        mask = np.zeros((2, 64, 96), bool)
        mask[:, 20:40, 30:60] = True
        output, trace = z4.augment_camera_clip(frames, np.random.default_rng(99), config, mask,
            plan(color=True, affine=True, erase=True, noise='salt_pepper', blur='gaussian', jpeg=True))
        params = trace['affine_params']
        matrix = z4.affine_matrix(64, 96, **params)
        valid = z4.cv2.warpAffine(np.ones((64, 96), np.float32), matrix, (96, 64), flags=z4.cv2.INTER_LINEAR) >= 1 - 1e-6
        self.assertTrue((output[:, :, torch.from_numpy(~valid)] == 0).all())
        transformed = z4.cv2.warpAffine(mask.any(axis=0).astype(np.float32), matrix, (96, 64),
                                       flags=z4.cv2.INTER_LINEAR, borderValue=1) > 0
        if trace['erase_applied']:
            x, y, w, h = trace['box']
            self.assertFalse(transformed[y:y+h, x:x+w].any())
        self.assertTrue(torch.equal(output[0], output[1]))

    def test_camera_order_does_not_change_results(self):
        config = replace(self.config, clean_probability=0)
        reverse = {**self.observation, 'image': dict(reversed(list(self.observation['image'].items())))}
        a = z4.augment_observation(self.observation, config=config, seed=10)
        b = z4.augment_observation(reverse, config=config, seed=10)
        for key in a['image']:
            self.assertTrue(torch.equal(a['image'][key], b['image'][key]))


if __name__ == '__main__':
    unittest.main()
