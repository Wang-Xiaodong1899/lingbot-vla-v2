"""Z6b whitelist behavior, opt-in integration and instruction-pool checks."""
import copy
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import torch
from lingbotvla.data.vla_data import transform
from lingbotvla.data.vla_data import z6b_augmentation as aug
from lingbotvla.data.vla_data.instruction_pool import load_seen_pool, sample_seen_instruction

ROOT = Path(__file__).resolve().parents[1]


def synth():
    image = torch.tensor([.55, .45, .35]).view(1, 3, 1, 1).expand(1, 3, 64, 80).clone()
    image[:, :, 24:38, 33:47] = torch.tensor([.9, .1, .1]).view(1, 3, 1, 1)
    return image


class Z6bTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ten_percent_clean_is_shared_across_views_and_bypasses_jitter(self):
        image = synth()
        kept = 0
        for seed in range(300):
            params = {'seed': torch.tensor(seed), 'brightness': torch.tensor(.3),
                      'contrast': torch.tensor(.5), 'saturation': torch.tensor(.4)}
            g = aug._view_generator(params, '__sample__')
            clean = aug._urand(g, 0, 1) < aug._AUG5['keep_p']
            if clean:
                kept += 1
                for view in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
                    with patch.dict(os.environ, LINGBOT_AUG_LEVEL='5'):
                        result = transform.apply_visual_augmentation(image.clone()[0], params, view)
                    torch.testing.assert_close(result, image[0], rtol=0, atol=0)
        self.assertTrue(12 <= kept <= 50, kept)

    def test_reference_foreground_mask_excludes_object_and_margin(self):
        image = synth()
        mask = aug._table_mask(image[0], aug._AUG5['table_thr'])
        self.assertFalse(mask[:, :, 20:42, 29:51].any())
        self.assertTrue(mask[0, 0, -4:, :10].all())

    def test_occlusion_only_preserves_protected_pixels(self):
        cfg = {**aug._AUG5, 'keep_p': 0., 'geo_p': 0., 'color_p': 0.,
               'occ_p': 1., 'gauss_p': 0., 'sp_p': 0., 'gblur_p': 0., 'mblur_p': 0., 'jpeg_p': 0.}
        image = synth()
        protected = ~aug._table_mask(image[0], cfg['table_thr']).bool().expand_as(image)
        changed = 0
        with patch.object(aug, '_AUG5', cfg):
            for seed in range(30):
                result = aug.whitelist_visual_augmentation(image.clone(), {'seed': seed}, 'cam_high')
                torch.testing.assert_close(result[protected], image[protected], rtol=0, atol=0)
                changed += not torch.equal(result, image)
        self.assertGreater(changed, 10)

    def test_geometry_shared_over_time_and_clean_pair(self):
        cfg = {**aug._AUG5, 'keep_p': 0., 'geo_p': 1., 'color_p': 0., 'occ_p': 0.,
               'gauss_p': 0., 'sp_p': 0., 'gblur_p': 0., 'mblur_p': 0., 'jpeg_p': 0.}
        image = synth().expand(3, -1, -1, -1).clone()
        with patch.object(aug, '_AUG5', cfg):
            result, clean = aug.whitelist_visual_augmentation(image.clone(), {'seed': 7}, 'cam_high', clean=image.clone())
        torch.testing.assert_close(result, clean, rtol=0, atol=0)
        torch.testing.assert_close(result[0], result[2], rtol=0, atol=0)
        self.assertTrue((result == 0).any())

    def test_full_recipe_bounds_dtype_and_replay_with_rng_state(self):
        image = (synth()[0] * 255).round().to(torch.uint8)
        with patch.dict(os.environ, LINGBOT_AUG_LEVEL='5'):
            params = transform.sample_visual_augmentation_params(image)
            for seed in range(20):
                params['seed'] = seed
                torch.manual_seed(123)
                a = transform.apply_visual_augmentation(image.clone(), params, 'cam_high')
                torch.manual_seed(123)
                b = transform.apply_visual_augmentation(image.clone(), params, 'cam_high')
                self.assertTrue(torch.equal(a, b))
                self.assertEqual(a.dtype, torch.uint8)
                self.assertEqual(a.shape, image.shape)

    def test_v1_pool_matches_official_episode_ids_and_seeded_sampling(self):
        path = str(ROOT / 'assets/training_data/official_instruction_pool_v1.json.gz')
        pool = load_seen_pool(path)
        self.assertEqual(set(pool), {str(i) for i in range(2500)})
        random.seed(7)
        a = sample_seen_instruction(path, 500)
        random.seed(7)
        self.assertEqual(a, sample_seen_instruction(path, 500))
        self.assertIn(a, pool['500'])
        with self.assertRaises(KeyError):
            sample_seen_instruction(path, 2500)


if __name__ == '__main__':
    unittest.main()
