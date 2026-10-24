"""Lossless STM observation and deferred-render behavior contracts."""
import ast
import copy
from collections import deque
from pathlib import Path
import unittest
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'experiment/robotwin/eval_policy_client_lingbotvla.py'
NAMES = {'stm_observation_fastpath_allowed', 'capture_stm_observation', 'take_action_with_deferred_render'}
nodes = [n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
namespace = {'np': np}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
allowed = namespace['stm_observation_fastpath_allowed']
capture = namespace['capture_stm_observation']
take_action = namespace['take_action_with_deferred_render']


class FakeEnv:
    eval_mode = True
    render_freq = 0
    eval_video_path = None
    crazy_random_light = False
    save_data = False

    def __init__(self, success_at=None, fail_at=None):
        self.physics_steps = 0
        self.success_checks = 0
        self.render_calls = 0
        self.rendered_pose = 0
        self.picture_calls = 0
        self.success_at, self.fail_at = success_at, fail_at
        self.success_image = None
        self.cameras = self.robot = self

    def _update_render(self):
        self.render_calls += 1
        self.rendered_pose = self.physics_steps

    def update_picture(self):
        self.picture_calls += 1

    def get_rgb(self):
        return {name: {'rgb': np.full((4, 4, 3), self.rendered_pose, np.uint8)}
                for name in ('head_camera', 'left_camera', 'right_camera')}

    def get_left_arm_jointState(self):
        return [float(self.physics_steps)] * 7

    def get_right_arm_jointState(self):
        return [float(self.physics_steps + 1)] * 7

    def get_obs(self):
        result = capture(self)
        self.now_obs = copy.deepcopy(result)
        result['unused_metadata'] = {'intrinsics': np.eye(3)}
        return result

    def take_action(self, action):
        self._update_render()
        for _ in range(action):
            self.physics_steps += 1
            self._update_render()
            self.success_checks += 1
            if self.physics_steps == self.fail_at:
                raise RuntimeError('planner failure')
            if self.physics_steps == self.success_at:
                self.success_image = self.get_obs()
                return
        self._update_render()


class ObservationTests(unittest.TestCase):
    def assert_inputs_equal(self, left, right):
        np.testing.assert_array_equal(left['joint_action']['vector'], right['joint_action']['vector'])
        for cam in ('head_camera', 'left_camera', 'right_camera'):
            np.testing.assert_array_equal(left['observation'][cam]['rgb'], right['observation'][cam]['rgb'])

    def test_deferred_sync_preserves_physics_success_checks_and_pixels(self):
        baseline, optimized = FakeEnv(), FakeEnv()
        for action in (3, 1, 8, 15, 2):
            baseline.take_action(action)
            take_action(optimized, action, True)
            self.assert_inputs_equal(baseline.get_obs(), capture(optimized))
            self.assertEqual(baseline.physics_steps, optimized.physics_steps)
            self.assertEqual(baseline.success_checks, optimized.success_checks)
        self.assertEqual(optimized.render_calls, 5)
        self.assertGreater(baseline.render_calls, optimized.render_calls)
        self.assertNotIn('_update_render', vars(optimized))
        self.assertNotIn('get_obs', vars(optimized))

    def test_early_success_captures_fresh_image_inside_action(self):
        baseline, optimized = FakeEnv(success_at=4), FakeEnv(success_at=4)
        baseline.take_action(10)
        take_action(optimized, 10, True)
        self.assertEqual(optimized.physics_steps, 4)
        self.assert_inputs_equal(baseline.success_image, optimized.success_image)
        self.assertEqual(optimized.render_calls, 1)

    def test_exception_restores_original_methods_and_existing_overrides(self):
        env = FakeEnv(fail_at=2)
        env._update_render = env._update_render
        original = env._update_render
        with self.assertRaisesRegex(RuntimeError, 'planner failure'):
            take_action(env, 10, True)
        self.assertIs(env._update_render, original)
        self.assertNotIn('get_obs', vars(env))
        env._update_render()
        self.assertEqual(env.rendered_pose, 2)

    def test_get_obs_exception_restores_methods(self):
        env = FakeEnv(success_at=2)
        def fail():
            raise RuntimeError('camera failure')
        env.get_obs = fail
        with self.assertRaisesRegex(RuntimeError, 'camera failure'):
            take_action(env, 10, True)
        self.assertIs(env.get_obs, fail)
        self.assertNotIn('_update_render', vars(env))

    def test_side_effect_modes_keep_original_render_calls(self):
        for attr, value in (('eval_mode', False), ('render_freq', 1),
                            ('eval_video_path', '/video'), ('crazy_random_light', True),
                            ('save_data', True)):
            env = FakeEnv()
            setattr(env, attr, value)
            self.assertFalse(allowed(env))
            take_action(env, 3, True)
            self.assertEqual(env.render_calls, 5)
        env = FakeEnv()
        take_action(env, 3, False)
        self.assertEqual(env.render_calls, 5)

    def test_boundary_reuse_preserves_independent_history_snapshot(self):
        module = ast.parse(SOURCE.read_text())
        record = next(n for n in ast.walk(module)
                      if isinstance(n, ast.FunctionDef) and n.name == 'record_temporal_frame')
        factory = ast.parse("""def build(TASK_ENV):
    temporal_frames = deque(maxlen=32)
    last_recorded_observation = None
    now_seed = 100000
    pass
    return record_temporal_frame, temporal_frames
""")
        factory.body[0].body[3] = record
        scope = {'deque': deque, 'copy': copy}
        exec(compile(ast.fix_missing_locations(factory), str(SOURCE), 'exec'), scope)
        env = FakeEnv()
        env.take_action_cnt = 0
        env.get_instruction = lambda: 'test'
        record_frame, frames = scope['build'](env)
        obs = capture(env)
        snapshot = record_frame(obs)
        obs['observation']['head_camera']['rgb'][:] = 255
        self.assertIs(record_frame(obs), snapshot)
        self.assertEqual(len(frames), 1)
        self.assertFalse(snapshot['observation.images.cam_high'].any())
        env.take_action_cnt = 1
        take_action(env, 1, True)
        next_frame = record_frame(capture(env))
        self.assertEqual(len(frames), 2)
        self.assertEqual(next_frame['timestamp'], 1 / 15)
        self.assertFalse(snapshot['observation.images.cam_high'].any())

    def test_snapshot_is_not_overwritten_by_future_capture(self):
        env = FakeEnv()
        first = capture(env)
        snapshot = copy.deepcopy(first)
        take_action(env, 4, True)
        later = capture(env)
        self.assert_inputs_equal(first, snapshot)
        self.assertFalse(np.array_equal(first['observation']['head_camera']['rgb'],
                                       later['observation']['head_camera']['rgb']))


if __name__ == '__main__':
    unittest.main()
