"""Training-free ChunkTrust AHS integration for synchronous RoboTwin evaluation."""
import argparse
import json
import numpy as np
import torch
from deploy.chunktrust_runtime import ChunkTrustRuntime
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, str2bool
from deploy.websocket_policy_server import WebsocketPolicyServer


class ChunkTrustServer(LingbotVLAv2Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep the trace collector eager; compile the expensive velocity network
        # using the existing sampler switch, without Python state in compiled code.
        def sample_with_trace(*args, **kwargs):
            actions, velocity = self.vla.model.sample_actions(
                *args, **kwargs, return_velocity_trace=True)
            self.velocity_trace = velocity.detach().float().cpu().numpy()
            return actions
        self.sample_actions_fn = sample_with_trace
        self.runtime = None

    def _prepare_model_input(self, observation):
        applied = super()._prepare_model_input(observation)
        self.active_mask = applied['action_joint_mask'].cpu().numpy().astype(bool)
        if self.active_mask.sum() != 14:
            raise ValueError('RoboTwin AHS expects 12 arm + 2 gripper coordinates')
        return applied

    def infer(self, observation, **kwargs):
        if observation.get('reset'):
            result = super().infer(observation)
            self.runtime = ChunkTrustRuntime(self.config.n_action_steps,
                                             seed=int(observation.get('episode_seed', 0)))
            self.replan = 0
            return result
        if self.runtime is None:
            raise ValueError('Reset the policy at the start of every episode')
        if 'batch' in observation:
            raise ValueError('Use one AHS server per synchronous environment')
        if self.runtime.pending is not None and 'executed_count' not in observation:
            raise ValueError('Client must report actually executed action count')
        self.runtime.acknowledge(observation.get('executed_count', 0))
        actions, normalized = self._infer_batch([observation], return_normalized=True)
        normalized = normalized[0].numpy()[:, self.active_mask]
        velocity = self.velocity_trace[:, 0, :, :][:, :, self.active_mask]
        k, info = self.runtime.select(velocity, normalized)
        self.replan += 1
        print('CHUNKTRUST ' + json.dumps({'replan': self.replan, 'k': k,
              'history_length': len(self.runtime.history),
              'q_mix': np.asarray(info['q_mix']).tolist()}), flush=True)
        result = {key: value[0, :k] for key, value in actions.items()}
        result['chunktrust'] = {'execution_horizon': k, 'replan': self.replan}
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--robot_norm_path', default=None)
    parser.add_argument('--use_length', type=int, default=50)
    parser.add_argument('--use_bf16', type=str2bool, default=False)
    parser.add_argument('--use_fp32', type=str2bool, default=True)
    parser.add_argument('--use_compile', type=str2bool, default=True)
    parser.add_argument('--port', type=int, default=9330)
    args = parser.parse_args()
    from pathlib import Path
    norm = args.robot_norm_path or str(Path(args.model_path).parent / 'robotwin_clean_norm.json')
    model = ChunkTrustServer(args.model_path, robot_norm_path=norm,
                            use_length=args.use_length, chunk_ret=True,
                            use_bf16=args.use_bf16, use_fp32=args.use_fp32,
                            use_compile=args.use_compile)
    WebsocketPolicyServer(model, port=args.port,
                          metadata={'method': 'chunktrust_ahs', 'executed_count_required': True}).serve_forever()


if __name__ == '__main__':
    main()
