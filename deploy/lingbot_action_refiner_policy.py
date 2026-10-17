"""RoboTwin server for a frozen Lingbot plus a trained action flow refiner."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, str2bool
from deploy.websocket_policy_server import WebsocketPolicyServer
from lingbotvla.models.vla.lingbot_vla.action_refinement import (
    ActionFlowRefiner, ActionRefinerConfig, RefinedLingbotPolicy,
)


class RefinerPolicyServer(LingbotVLAv2Server):
    def __init__(self, model_path, refiner_checkpoint, **kwargs):
        checkpoint = torch.load(refiner_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported refiner checkpoint")
        metadata = checkpoint["metadata"]
        if Path(model_path).resolve() != Path(metadata["base_model"]).resolve():
            raise ValueError("Evaluation base model differs from the refiner's training base")
        digest = hashlib.sha256((Path(model_path) / "config.json").read_bytes()).hexdigest()
        if digest != metadata["base_config_sha256"]:
            raise ValueError("Base model configuration changed since refiner training")
        dtype = "bfloat16" if kwargs.get("use_bf16", False) else "float32"
        if dtype != metadata["base_dtype"]:
            raise ValueError("Use the same base inference dtype as refiner training")
        if kwargs.get("use_compile", False):
            raise ValueError("Refiner feature capture currently requires --use_compile False")
        os.environ["QWEN3VL_PATH"] = metadata["qwen_path"]
        norm_path = Path(metadata["norm_stats_path"])
        if hashlib.sha256(norm_path.read_bytes()).hexdigest() != metadata["norm_stats_sha256"]:
            raise ValueError("Normalization file changed since training")
        kwargs["robot_norm_path"] = str(norm_path)
        super().__init__(path_to_pi_model=model_path, **kwargs)
        if self.config.num_steps != metadata["base_num_steps"]:
            raise ValueError("Base sampling step count differs from refiner training")
        refiner = ActionFlowRefiner(ActionRefinerConfig(**checkpoint["refiner_config"]))
        refiner.load_state_dict(checkpoint["refiner"], strict=True)
        self.refined_policy = RefinedLingbotPolicy(self.vla, refiner.cuda()).eval()
        self.refiner_step = checkpoint["step"]
        print(json.dumps({"refiner_checkpoint": str(Path(refiner_checkpoint).resolve()),
                          "refiner_step": self.refiner_step, "base_steps": self.config.num_steps,
                          "refiner_steps": refiner.config.num_steps, "dtype": dtype}), flush=True)

    @torch.no_grad()
    def _infer_batch(self, observations, return_normalized=False):
        if not isinstance(observations, (list, tuple)) or not observations:
            raise ValueError("Expected a nonempty list of observations")
        applied = [self._prepare_model_input(obs) for obs in observations]
        batch = {}
        for key in applied[0]:
            values = [sample[key] for sample in applied]
            if torch.is_tensor(values[0]):
                batch[key] = self._pad_and_stack_tensors(values).cuda()
            else:
                batch[key] = values
        actions = self.refined_policy.sample_actions(batch)
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Refined action contains NaN/Inf")
        # Refine in normalized coordinates, then use original robot conversion.
        unnormalized = self._unapply_batched_actions(applied, actions)
        return (unnormalized, actions) if return_normalized else unnormalized

    def reset(self, robo_name, path_to_pi_model=None):
        if path_to_pi_model is not None:
            raise ValueError("Restart the refiner server to change its base checkpoint")
        super().reset(robo_name=robo_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--refiner_checkpoint", default=os.environ.get("REFINER_CHECKPOINT"))
    parser.add_argument("--use_length", type=int, default=50)
    parser.add_argument("--chunk_ret", type=str2bool, default=True)
    parser.add_argument("--port", type=int, default=9430)
    parser.add_argument("--use_bf16", type=str2bool, default=False)
    parser.add_argument("--use_fp32", type=str2bool, default=True)
    parser.add_argument("--use_compile", type=str2bool, default=False)
    args = parser.parse_args()
    if not args.refiner_checkpoint:
        parser.error("Set --refiner_checkpoint or REFINER_CHECKPOINT")
    policy = RefinerPolicyServer(
        args.model_path, args.refiner_checkpoint, use_length=args.use_length,
        chunk_ret=args.chunk_ret, use_bf16=args.use_bf16, use_fp32=args.use_fp32,
        use_compile=args.use_compile,
    )
    WebsocketPolicyServer(policy, port=args.port, metadata={"refiner_step": policy.refiner_step}).serve_forever()


if __name__ == "__main__":
    main()
