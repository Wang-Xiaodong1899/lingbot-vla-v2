"""VLA-Corrector for Lingbot: per-step vision monitor and recovery-only OGG."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, str2bool
from deploy.websocket_policy_server import WebsocketPolicyServer
from lingbotvla.corrector.runtime import (base_metadata, check_base_metadata, load_corrector,
                                         visual_tokens, cosine_error, online_guidance)
from lingbotvla.corrector.monitor import RobustThresholdState, robust_threshold_step


class CorrectorServer(LingbotVLAv2Server):
    def __init__(self, model_path, corrector_checkpoint, robot_norm_path, guidance_eta=0.1,
                 guidance_every=3, **kwargs):
        if kwargs.get('use_compile') or kwargs.get('use_bf16'):
            raise ValueError('Corrector uses eager FP32 matching the feature extraction model')
        predictor, payload = load_corrector(corrector_checkpoint,'cpu')
        if payload['metadata']['smoke']:
            raise ValueError('Smoke corrector weights cannot be used for formal evaluation')
        metadata = payload['metadata']['feature_metadata']
        check_base_metadata(metadata,model_path,robot_norm_path)
        self.corrector_metadata = metadata
        os.environ['QWEN3VL_PATH'] = metadata['qwen_path']
        super().__init__(path_to_pi_model=model_path, robot_norm_path=robot_norm_path, **kwargs)
        self.vla.requires_grad_(False).eval()
        self.vla.model.config.precompute_grid_thw = False
        self.predictor = predictor.cuda()
        self.guidance_eta, self.guidance_every = guidance_eta, guidance_every
        self.k_step = payload['metadata']['k_step']
        self.corrector_step = payload['step']
        self._clear_episode()
        print('CORRECTOR_LOADED '+json.dumps(dict(checkpoint=str(Path(corrector_checkpoint).resolve()),
                                                step=self.corrector_step, k_step=self.k_step,
                                                norm_stats=robot_norm_path, guidance_eta=guidance_eta)),flush=True)

    def resize_image(self, observation):
        # Match offline embedded-PNG decoding before the base Qwen processor.
        from torchvision.transforms.v2 import Resize
        resize = Resize((self.data_config.img_size, self.data_config.img_size))
        for key in self.vla.feature_transform.org_features['images']:
            image = torch.as_tensor(np.asarray(observation[key])).permute(2,0,1).contiguous().float()
            if image.max() > 1:
                image = image / 255.0
            observation[key] = resize(image)

    def _clear_episode(self):
        self.pending = deque()
        self.plan = None
        self.cursor = 0
        self.clock = 0
        self.monitor = RobustThresholdState(cooldown_steps=10)
        self.last_diagnostics = {}
        self.events = 0
        self.replans = 0

    def _batch(self, transformed):
        return {k:v[None].cuda() if torch.is_tensor(v) else [v] for k,v in transformed.items()}

    @torch.no_grad()
    def infer(self, observation, **kwargs):
        if observation.get('reset'):
            if observation.get('path_to_pi_model') is not None:
                raise ValueError('Restart server to change the frozen base checkpoint')
            if hasattr(self,'clock') and self.clock:
                print('CORRECTOR_EPISODE '+json.dumps(dict(steps=self.clock,events=self.events,replans=self.replans)),flush=True)
            self.reset(robo_name=observation['robo_name'])
            self._clear_episode()
            seed = observation.get('episode_seed')
            if seed is not None:
                torch.manual_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))
            return dict(action=None)
        if 'batch' in observation:
            raise ValueError('One Corrector server per synchronous simulator')
        if self.clock and int(observation.get('executed_count',0)) != 1:
            raise ValueError('Corrector requires exactly one executed action and fresh observation per request')
        transformed = self._prepare_model_input(observation)
        batch = self._batch(transformed)
        z = visual_tokens(self.vla.model,batch)
        mask = batch['action_joint_mask']
        target = None
        error, threshold, trigger = None,None,False
        while self.pending and self.pending[0][0] <= self.clock:
            due, origin, prediction = self.pending.popleft()
            if due != self.clock:
                raise RuntimeError('Misaligned visual dynamics prediction queue')
            actual = z-origin
            error = cosine_error(prediction,actual)
            if error is not None:
                threshold,trigger = robust_threshold_step(error,self.monitor)
            if trigger:
                target = (prediction-actual).detach()
                self.plan = None
                self.events += 1
        guidance_stats = []
        replan = self.plan is None or self.cursor >= self.use_length
        if replan:
            args = dict(images=batch['images'].float(),img_masks=batch['img_masks'],
                        lang_tokens=batch['lang_tokens'],lang_masks=batch['lang_masks'],
                        state=batch['state'].float(),image_grid_thw=batch['image_grid_thw'])
            if target is not None and target.norm() > 1e-4:
                with online_guidance(self.vla.model,self.predictor,z,target,mask,
                                     eta=self.guidance_eta,every=self.guidance_every) as guidance_stats:
                    self.plan = self.vla.model.sample_actions(**args)
            else:
                self.plan = self.vla.model.sample_actions(**args)
            self.cursor = 0
            self.replans += 1
            if not torch.isfinite(self.plan).all():
                raise FloatingPointError('Nonfinite sampled plan')
        action = self.plan[:,self.cursor] * mask
        prediction = self.predictor(z,action).detach()
        self.pending.append((self.clock+self.k_step,z.detach(),prediction))
        # Unnormalize with the exact same transform and model-supplied stats.
        restored = self._unapply_batched_actions([transformed],action[:,None])
        result = {k:v[0,0] for k,v in restored.items()}
        diagnostics = dict(step=self.clock,corrector_step=self.corrector_step,error=error,threshold=threshold,
                           trigger=trigger,replan=replan,cursor=self.cursor,events=self.events,
                           ogg_applied=len(guidance_stats),ogg=guidance_stats)
        self.last_diagnostics = diagnostics
        if replan or error is not None:
            print('CORRECTOR '+json.dumps(diagnostics),flush=True)
        self.cursor += 1
        self.clock += 1
        result['corrector'] = diagnostics
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path',required=True)
    p.add_argument('--corrector_checkpoint',default=os.environ.get('CORRECTOR_CHECKPOINT'))
    p.add_argument('--robot_norm_path',default=os.environ.get('CORRECTOR_NORM_STATS'))
    p.add_argument('--use_length',type=int,default=50)
    p.add_argument('--use_bf16',type=str2bool,default=False)
    p.add_argument('--use_fp32',type=str2bool,default=True)
    p.add_argument('--use_compile',type=str2bool,default=False)
    p.add_argument('--guidance_eta',type=float,default=0.1)
    p.add_argument('--guidance_every',type=int,default=3)
    p.add_argument('--port',type=int,default=9830)
    a = p.parse_args()
    if not a.corrector_checkpoint:
        p.error('CORRECTOR_CHECKPOINT or --corrector_checkpoint is required')
    if not a.robot_norm_path:
        a.robot_norm_path=str(Path(a.model_path)/'robotwin_clean_norm.json')
    if not 1 <= a.use_length <= 50 or a.guidance_eta <= 0 or a.guidance_every < 1:
        p.error('Invalid horizon/guidance arguments')
    server=CorrectorServer(a.model_path,a.corrector_checkpoint,a.robot_norm_path,
                          use_length=a.use_length,chunk_ret=True,use_bf16=a.use_bf16,use_fp32=a.use_fp32,
                          use_compile=a.use_compile,guidance_eta=a.guidance_eta,guidance_every=a.guidance_every)
    WebsocketPolicyServer(server,port=a.port,metadata=dict(method='vla-corrector',per_step_observation=True)).serve_forever()


if __name__=='__main__':
    main()
