"""Check attention capture parity, separating production MoE numerical jitter."""
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
    warmup = server._infer_batch([observation])
    torch.manual_seed(123)
    baseline = server._infer_batch([observation])
    production_jitter = {key: float(np.max(np.abs(baseline[key] - warmup[key])))
                         for key in baseline}
    print('BASELINE_REPEAT_DELTA', production_jitter, flush=True)
    # The production robby_moe path uses atomic accumulation. Verify bitwise
    # collector parity separately on the existing reference MoE backend; do not
    # weaken the assertion or change the production evaluation configuration.
    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert as moe
    original_robby = moe.robby_moe_forward
    moe.robby_moe_forward = None
    try:
        torch.manual_seed(123)
        baseline = server._infer_batch([observation])
        flow.predict_velocity, expert.attention_interface = wrapped_velocity, wrapped_attention
        server.collector.reset()
        torch.manual_seed(123)
        captured = server._infer_batch([observation])
        for key in baseline:
            np.testing.assert_array_equal(captured[key], baseline[key])
        attention = server.collector.finish()
        assert attention.shape == (1, 50, 50)
        assert server.collector.layers == server.config.num_hidden_layers
    finally:
        moe.robby_moe_forward = original_robby
        flow.predict_velocity, expert.attention_interface = wrapped_velocity, wrapped_attention
    # Check prefix slicing against the same realized production sample, without
    # assuming that two atomic-MoE forwards are bitwise deterministic.
    original_infer_batch = server._infer_batch
    full_chunk = {}
    def record_batch(*args, **kwargs):
        actions = original_infer_batch(*args, **kwargs)
        full_chunk.update({key: value.copy() for key, value in actions.items()})
        return actions
    server._infer_batch = record_batch
    result = server.infer(observation)
    k = result['autohorizon']['execution_horizon']
    for key in full_chunk:
        np.testing.assert_array_equal(result[key], full_chunk[key][0, :k])
    print(json.dumps({'reference_moe_action_parity': 'exact',
                      'production_baseline_repeat_max_abs_delta': production_jitter,
                      'production_prefix_slicing': 'exact',
                      'attention_shape': list(attention.shape),
                      'diagnostics': result['autohorizon']}), flush=True)



if __name__ == '__main__':
    main()
