"""AutoHorizon inference for Lingbot VLA v2 (frozen model, eager attention)."""
import argparse
import json
from pathlib import Path
from deploy.autohorizon_runtime import AutoHorizonCollector, select_horizon
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, str2bool
from deploy.websocket_policy_server import WebsocketPolicyServer


class AutoHorizonServer(LingbotVLAv2Server):
    def __init__(self, *args, attention_step=3, selector='bidir',
                 hold_threshold=0.3, entropy_quantile=0.9, **kwargs):
        if kwargs.get('use_compile', False):
            raise ValueError('AutoHorizon attention capture requires --use_compile False')
        super().__init__(*args, **kwargs)
        if not self.chunk_ret:
            raise ValueError('AutoHorizon requires chunk_ret=True')
        if self.use_length < 1 or self.use_length > self.config.n_action_steps:
            raise ValueError('use_length must lie within the model action horizon')
        self.collector = AutoHorizonCollector(self.vla.model, self.config.n_action_steps, attention_step)
        self.selector = selector
        self.hold_threshold = hold_threshold
        self.entropy_quantile = entropy_quantile
        self.replan = 0

    def infer(self, observation, **kwargs):
        if observation.get('reset'):
            if observation.get('path_to_pi_model') is not None:
                raise ValueError('Restart AutoHorizon server to change checkpoint')
            result = super().infer(observation)
            self.collector.reset()
            self.replan = 0
            return result
        if 'batch' in observation:
            raise ValueError('Use one AutoHorizon server per synchronous environment')
        self.collector.reset()
        actions = self._infer_batch([observation])
        attention = self.collector.finish()[0]
        k, info = select_horizon(attention, self.selector, self.hold_threshold, self.entropy_quantile)
        k = min(k, self.use_length)
        self.replan += 1
        diagnostics = {'execution_horizon': k, 'replan': self.replan,
                       'attention_step': self.collector.attention_step,
                       'layers': self.collector.layers, 'selector': self.selector}
        if self.selector == 'bidir':
            diagnostics.update(forward_horizon=info['N_forward'], backward_horizon=info['N_backward'],
                               join_row=info['join_row'])
        print('AUTOHORIZON ' + json.dumps(diagnostics), flush=True)
        result = {key: value[0, :k] for key, value in actions.items()}
        result['autohorizon'] = diagnostics
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--robot_norm_path')
    parser.add_argument('--use_length', type=int, default=50)
    parser.add_argument('--use_bf16', type=str2bool, default=False)
    parser.add_argument('--use_fp32', type=str2bool, default=True)
    parser.add_argument('--use_compile', type=str2bool, default=False)
    parser.add_argument('--port', type=int, default=9630)
    parser.add_argument('--attention_step', type=int, default=3)
    parser.add_argument('--selector', choices=['bidir', 'forward'], default='bidir')
    parser.add_argument('--hold_threshold', type=float, default=0.3)
    parser.add_argument('--entropy_quantile', type=float, default=0.9)
    args = parser.parse_args()
    if args.hold_threshold < 0 or not 0 < args.entropy_quantile <= 1:
        parser.error('Invalid soft-pointer thresholds')
    norm = args.robot_norm_path or str(Path(args.model_path).parent / 'robotwin_clean_norm.json')
    model = AutoHorizonServer(args.model_path, robot_norm_path=norm, use_length=args.use_length,
                             chunk_ret=True, use_bf16=args.use_bf16, use_fp32=args.use_fp32,
                             use_compile=args.use_compile, attention_step=args.attention_step,
                             selector=args.selector, hold_threshold=args.hold_threshold,
                             entropy_quantile=args.entropy_quantile)
    WebsocketPolicyServer(model, port=args.port, metadata={'method': 'autohorizon'}).serve_forever()


if __name__ == '__main__':
    main()
