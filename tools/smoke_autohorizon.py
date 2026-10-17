"""Check real Lingbot attention capture and exact fixed-seed action parity."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from deploy.lingbot_vla_autohorizon_policy import AutoHorizonServer
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=str(ROOT / 'weights/lingbotvla-robotwin-clean-sft/global_step_10000'))
    args = parser.parse_args()
    server = AutoHorizonServer(args.model_path,
                              robot_norm_path=str(Path(args.model_path).parent / 'robotwin_clean_norm.json'),
                              use_length=50, chunk_ret=True, use_bf16=False,
                              use_fp32=True, use_compile=False)
    server.infer({'reset': True, 'robo_name': 'robotwin'})
    rng = np.random.default_rng(3)
    observation = {key: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
                   for key in ['observation.images.cam_high', 'observation.images.cam_left_wrist',
                               'observation.images.cam_right_wrist']}
    observation.update({'observation.state': np.zeros(14, dtype=np.float32), 'task': 'Lift the pot.'})
    flow = server.vla.model
    expert = flow.qwenvl_with_expert
    wrapped_velocity, wrapped_attention = flow.predict_velocity, expert.attention_interface
    flow.predict_velocity = server.collector.original_velocity
    expert.attention_interface = server.collector.original_attention
    torch.manual_seed(123)
    baseline = server._infer_batch([observation])
    flow.predict_velocity, expert.attention_interface = wrapped_velocity, wrapped_attention
    server.collector.reset()
    torch.manual_seed(123)
    captured = server._infer_batch([observation])
    for key in baseline:
        np.testing.assert_array_equal(captured[key], baseline[key])
    attention = server.collector.finish()
    assert attention.shape == (1, 50, 50) and server.collector.layers == server.config.num_hidden_layers
    torch.manual_seed(123)
    result = server.infer(observation)
    k = result['autohorizon']['execution_horizon']
    for key in baseline:
        np.testing.assert_array_equal(result[key], baseline[key][0, :k])
    print(json.dumps({'action_parity': 'exact', 'attention_shape': list(attention.shape),
                      'diagnostics': result['autohorizon']}), flush=True)


if __name__ == '__main__':
    main()
