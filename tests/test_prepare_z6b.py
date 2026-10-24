"""Z6b config generation and isolation from other continuation recipes."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_z6b', ROOT / 'tools/prepare_z3a_official.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PrepareZ6bTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'pica/global_step_1300'
        self.checkpoint.mkdir(parents=True)
        align = yaml.safe_load((ROOT / 'configs/vla/robotwin/robotwin.yaml').read_text())['train']['align_params']
        (self.checkpoint / 'config.json').write_text(json.dumps({'align_params': align, 'pica_params': {'weight': .05}}))
        self.output = self.root / 'output'
        self.env = dict(OUTPUT_DIR=str(self.output), INIT_MODEL=str(self.checkpoint),
                        LINGBOT_TRAIN_RECIPE='z6b', LINGBOT_STATE_NOISE_STD='.01',
                        LINGBOT_INSTRUCTION_POOL=str(ROOT / 'assets/training_data/official_instruction_pool_v1.json.gz'),
                        ROBOT_CONFIG=str(ROOT / 'configs/robot_configs/robotwin.yaml'),
                        NORM_FILE=str(self.root / 'norm.json'), MICRO_BATCH_SIZE='8',
                        GLOBAL_BATCH_SIZE='2048', NGPU='2', GRAD_ACCUM='128',
                        LR='1e-4', LR_MIN='5e-5', MAX_STEPS='500', SAVE_STEPS='500',
                        LR_DECAY_STYLE='cosine', LR_DECAY_RATIO='2.0', DRY_RUN='1')

    def run_prepare(self, **overrides):
        with patch.dict(os.environ, {**self.env, **overrides}), contextlib.redirect_stdout(io.StringIO()):
            prepare.main()
        return yaml.safe_load((self.output / 'launch/config.yaml').read_text())

    def test_reference_recipe_uses_pica_weights_without_pica_loss(self):
        cfg = self.run_prepare()
        self.assertEqual(cfg['model']['model_path'], str(self.checkpoint))
        self.assertEqual(cfg['train']['pica_params'], {})
        self.assertEqual(cfg['train']['optimizer'], 'muon')
        self.assertFalse(cfg['train']['use_moe_expert_lr'])
        self.assertTrue(cfg['data']['image_augment'])
        self.assertEqual(cfg['train']['global_batch_size'], 2048)
        self.assertEqual(cfg['train']['gradient_accumulation_steps'], 128)
        self.assertEqual(cfg['train']['max_steps'], 500)
        self.assertEqual(cfg['train']['lr_decay_ratio'], 2.)
        self.assertEqual(cfg['train']['lr'], 1e-4)
        self.assertEqual(cfg['train']['lr_min'], 5e-5)
        self.assertEqual(cfg['train']['lr_warmup_ratio'], 0.)
        recipe = json.loads((self.output / 'launch/recipe.json').read_text())
        self.assertEqual(recipe['augmentation_level'], 5)
        self.assertEqual(recipe['state_noise_std'], .01)
        self.assertEqual(recipe['z6b_config']['temp'], .10)
        self.assertEqual((self.output / 'launch/instruction_pool_v1.json.gz').read_bytes(),
                         Path(self.env['LINGBOT_INSTRUCTION_POOL']).read_bytes())
        self.assertEqual(json.loads((self.checkpoint / 'config.json').read_text())['pica_params'], {'weight': .05})

    def test_existing_run_cannot_switch_augmentation_or_state_noise(self):
        self.run_prepare()
        (self.output / 'checkpoints/global_step_500').mkdir(parents=True)
        recipe = (self.output / 'launch/recipe.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'different Z6b recipe'):
            self.run_prepare(LINGBOT_STATE_NOISE_STD='.02')
        with self.assertRaisesRegex(ValueError, 'different augmentation'):
            self.run_prepare(LINGBOT_TRAIN_RECIPE='z3a')
        self.assertEqual((self.output / 'launch/recipe.json').read_bytes(), recipe)
        self.run_prepare()

    def test_invalid_scheduler_is_rejected(self):
        for kwargs in ({'LR_DECAY_RATIO': 'nan'}, {'LR_DECAY_RATIO': '.5'}, {'LR_DECAY_STYLE': 'constant'}):
            with self.assertRaisesRegex(ValueError, 'Z6b requires cosine'):
                self.run_prepare(**kwargs)


if __name__ == '__main__':
    unittest.main()
