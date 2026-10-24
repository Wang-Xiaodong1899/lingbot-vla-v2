"""Compare real SAPIEN physics/RGB before and after deferred render sync.

Run with the RoboTwin Python from the repository root. Uses a small synthetic
scene, the pinned RoboTwin get_obs/RGB code, and unchanged RT32/depth8/OIDN.
"""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import sapien.core as sapien
from sapien.physx import PhysxRigidDynamicComponent

ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin', type=Path, default=ROOT / 'evaluation/RoboTwin')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation/stm_render_sync_validation.json')
    args = parser.parse_args()
    runtime = {'np': np}
    load_functions(ROOT / 'experiment/robotwin/eval_policy_client_lingbotvla.py',
                   {'stm_observation_fastpath_allowed', 'capture_stm_observation',
                    'take_action_with_deferred_render'}, runtime)
    base = {'np': np, 'deepcopy': deepcopy}
    load_functions(args.robotwin / 'envs/_base_task.py', {'get_obs'}, base)
    camera = {}
    load_functions(args.robotwin / 'envs/camera/camera.py', {'get_rgba', 'get_rgb'}, camera)

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    sapien.render.set_camera_shader_dir('rt')
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser('oidn')
    config = sapien.SceneConfig()
    config.gravity = [0., 0., 0.]
    scene = engine.create_scene(config)
    scene.set_timestep(1 / 240)
    scene.set_ambient_light([.5, .5, .5])
    scene.add_directional_light([1, 0, -1], [.5, .5, .5], shadow=True)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.1, .1, .1])
    material = renderer.create_material()
    material.set_base_color([.8, .2, .1, 1.])
    builder.add_box_visual(half_size=[.1, .1, .1], material=material)
    actor = builder.build(name='moving_box')
    body = actor.find_component_by_type(PhysxRigidDynamicComponent)

    class Cameras:
        collect_wrist_camera = collect_head_camera = True
        get_rgba, get_rgb = camera['get_rgba'], camera['get_rgb']

        def __init__(self):
            cams = []
            for i in range(3):
                c = scene.add_camera(f'camera{i}', 128, 128, 1., .01, 10.)
                c.entity.set_pose(sapien.Pose([-1., (i-1)*.15, 0.]))
                cams.append(c)
            self.left_camera, self.right_camera, head = cams
            self.static_camera_list, self.static_camera_name = [head], ['head_camera']

        def update_picture(self):
            for c in [self.left_camera, self.right_camera] + self.static_camera_list:
                c.take_picture()

        def get_config(self):
            return {name: {} for name in ('head_camera', 'left_camera', 'right_camera')}

    class Env:
        eval_mode, render_freq, eval_video_path = True, 0, None
        crazy_random_light, save_data = False, False
        data_type = {'rgb': True, 'qpos': True}
        get_obs = base['get_obs']

        def __init__(self):
            self.cameras, self.robot = Cameras(), self
            self.render_calls = self.physics_steps = self.success_checks = 0
            self.success_image = None

        def _update_render(self):
            self.render_calls += 1
            for c, offset in ((self.cameras.left_camera, -.15), (self.cameras.right_camera, .15)):
                c.entity.set_pose(sapien.Pose([-1., float(actor.get_pose().p[1]) + offset, 0.]))
            scene.update_render()

        def get_left_arm_jointState(self):
            return actor.get_pose().p.tolist() + actor.get_pose().q.tolist()

        get_right_arm_jointState = get_left_arm_jointState

        def take_action(self, count):
            self._update_render()
            for _ in range(count):
                scene.step()
                self.physics_steps += 1
                self._update_render()
                self.success_checks += 1
                if self.physics_steps == 35:
                    self.success_image = self.get_obs()
                    return
            self._update_render()

    env = Env()
    images, states, measurements = {}, {}, {}
    for mode in ('baseline', 'optimized'):
        actor.set_pose(sapien.Pose([0., 0., 0.]))
        body.set_linear_velocity([0., .1, 0.])
        body.set_angular_velocity([0., 0., 0.])
        env.get_obs()  # Warm up shader compilation/allocations outside timing.
        env.render_calls = env.physics_steps = env.success_checks = 0
        env.success_image = None
        frames, positions = [], []
        t = time.perf_counter()
        for count in (5, 10, 20):
            runtime['take_action_with_deferred_render'](env, count, mode == 'optimized')
            obs = (runtime['capture_stm_observation'](env) if mode == 'optimized' else env.get_obs())
            frames.append([obs['observation'][c]['rgb'].copy()
                           for c in ('head_camera', 'left_camera', 'right_camera')])
            positions.append(obs['joint_action']['vector'].copy())
        frames.append([env.success_image['observation'][c]['rgb'].copy()
                       for c in ('head_camera', 'left_camera', 'right_camera')])
        images[mode], states[mode] = np.array(frames), np.array(positions)
        measurements[mode] = {'elapsed_s': time.perf_counter()-t,
                              'render_sync_calls': env.render_calls,
                              'physics_steps': env.physics_steps,
                              'success_checks': env.success_checks}
    exact_pixels = np.array_equal(images['baseline'], images['optimized'])
    exact_states = np.array_equal(states['baseline'], states['optimized'])
    report = {'measurements': measurements, 'pixels_exact': exact_pixels,
              'states_exact': exact_states,
              'image_range': [int(images['baseline'].min()), int(images['baseline'].max())],
              'moving_pixels': int(np.count_nonzero(images['baseline'][0] != images['baseline'][2])),
              'max_pixel_difference': int(np.abs(images['baseline'].astype(int)-images['optimized'].astype(int)).max()),
              'rt_samples': 32, 'path_depth': 8, 'denoiser': 'oidn',
              'scope': 'synthetic moving rigid body and three 128x128 cameras; includes capture on early success'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    assert np.ptp(images['baseline']) > 0, 'Camera validation image is blank'
    assert np.any(images['baseline'][0] != images['baseline'][2]), 'Moving scene must change image'
    assert exact_pixels and exact_states, 'RGB/state parity failed; do not deploy deferred sync'
    assert measurements['baseline']['physics_steps'] == measurements['optimized']['physics_steps'] == 35
    assert measurements['baseline']['success_checks'] == measurements['optimized']['success_checks'] == 35


if __name__ == '__main__':
    main()
