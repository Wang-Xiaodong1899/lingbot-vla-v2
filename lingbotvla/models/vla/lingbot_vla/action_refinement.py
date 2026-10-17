"""Candidate-to-GT action flow; the pretrained policy remains frozen.

All actions are in the base policy's normalized, padded action coordinates.
This is a second-stage refiner, not a history-action A2A implementation.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ActionRefinerConfig:
    action_dim: int = 55
    state_dim: int = 55
    horizon: int = 50
    context_dim: int = 2560
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_steps: int = 4
    loss_type: str = "l1"

    def __post_init__(self):
        if min(self.action_dim, self.state_dim, self.horizon, self.context_dim,
               self.hidden_dim, self.num_layers, self.num_heads, self.num_steps) <= 0:
            raise ValueError("Refiner dimensions and step counts must be positive")
        if self.hidden_dim % self.num_heads or self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even and divisible by num_heads")
        if self.loss_type not in ("l1", "mse"):
            raise ValueError("loss_type must be l1 or mse")


def action_valid_mask(actions, joint_mask=None, action_is_pad=None):
    """True for real joints and real future timesteps (never padded tails)."""
    mask = torch.ones_like(actions, dtype=torch.bool)
    if joint_mask is not None:
        joints = joint_mask.bool()
        if joints.ndim == 2:
            joints = joints[:, None, :]
        mask = mask & joints
    if action_is_pad is not None:
        mask = mask & ~action_is_pad.bool().unsqueeze(-1)
    return mask


def masked_error(prediction, target, mask, loss_type="l1"):
    # where also excludes NaNs in padding from loss and gradients.
    delta = torch.where(mask, prediction.float() - target.float(), 0.0)
    error = delta.abs() if loss_type == "l1" else delta.square()
    return error.sum(), mask.sum()


class ActionFlowRefiner(nn.Module):
    def __init__(self, config: ActionRefinerConfig):
        super().__init__()
        self.config = config
        width = config.hidden_dim
        # Fixed proposal and joint validity remain visible throughout integration.
        self.action_proj = nn.Linear(config.action_dim * 3, width)
        self.state_proj = nn.Linear(config.state_dim, width)
        self.context_proj = nn.Sequential(nn.LayerNorm(config.context_dim),
                                          nn.Linear(config.context_dim, width))
        self.time_proj = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.positions = nn.Parameter(torch.randn(1, config.horizon, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width, config.num_heads, dim_feedforward=width * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)
        # TransformerEncoder clones its first layer; initialize clones independently.
        for block in self.blocks.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, config.action_dim))
        # At initialization, refinement is exactly the base proposal.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x_t, time, proposal, state, context, joint_mask=None):
        if x_t.shape[1:] != (self.config.horizon, self.config.action_dim):
            raise ValueError(f"Unexpected action shape {tuple(x_t.shape)}")
        mask = action_valid_mask(proposal, joint_mask)
        x_t = torch.where(mask, x_t, 0.0)
        proposal = torch.where(mask, proposal, 0.0)
        half = self.config.hidden_dim // 2
        frequencies = torch.exp(torch.arange(half, device=x_t.device).float()
                                * (-math.log(10000.0) / max(half - 1, 1)))
        phases = time.float()[:, None] * frequencies[None, :] * (2 * math.pi)
        time_embedding = self.time_proj(torch.cat([phases.sin(), phases.cos()], dim=-1))
        condition = self.state_proj(state.float()) + self.context_proj(context.float()) + time_embedding
        tokens = self.action_proj(torch.cat([x_t.float(), proposal.float(), mask.float()], dim=-1))
        tokens = tokens + self.positions + condition[:, None, :]
        return self.output(self.blocks(tokens)) * mask

    def flow_loss(self, proposal, target, state, context, joint_mask=None,
                  action_is_pad=None, time=None):
        proposal, target = proposal.detach().float(), target.detach().float()
        # Inference never knows where a future episode ends. Do not expose the
        # GT temporal padding mask as a network condition.
        joints = action_valid_mask(proposal, joint_mask)
        mask = action_valid_mask(target, joint_mask, action_is_pad)
        if not mask.any():
            raise ValueError("Batch contains no valid action targets")
        if time is None:
            time = torch.rand(proposal.shape[0], device=proposal.device)
        # Unknown targets use the proposal so GT padding cannot affect real tokens.
        delta = torch.where(mask, target - proposal, 0.0)
        x_t = torch.where(joints, proposal + time[:, None, None] * delta, 0.0)
        velocity = self(x_t, time, proposal, state.detach(), context.detach(), joints)
        total, count = masked_error(velocity, delta, mask, self.config.loss_type)
        return total / count.clamp_min(1), {"loss_sum": total.detach(), "valid_count": count.detach()}

    @torch.no_grad()
    def refine(self, proposal, state, context, joint_mask=None, num_steps=None):
        steps = self.config.num_steps if num_steps is None else num_steps
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        x = proposal.float().clone()
        for step in range(steps):
            time = torch.full((x.shape[0],), step / steps, device=x.device)
            x = x + self(x, time, proposal, state, context, joint_mask).float() / steps
        return x

    def config_dict(self):
        return asdict(self.config)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Load a trusted refiner training checkpoint; base weights stay separate."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported refiner checkpoint format")
        model = cls(ActionRefinerConfig(**checkpoint["refiner_config"]))
        model.load_state_dict(checkpoint["refiner"], strict=True)
        return model.to(device).eval()


class FrozenLingbotProposal(nn.Module):
    """Run real base-policy sampling and reuse its final valid language feature.

    The final language token has attended to the image/language prefix. Capturing
    its final normalization avoids an extra visual-language forward. GT actions
    are never passed to the base policy or used to obtain this feature.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, batch, noise=None):
        self.policy.eval()
        core = self.policy.model
        context = []
        language_mask = batch["lang_masks"].bool()
        if not language_mask.any(dim=-1).all():
            raise ValueError("Every sample needs at least one valid language token")

        def capture(_module, _inputs, output):
            language_length = language_mask.shape[1]
            indices = torch.arange(language_length, device=output.device).expand_as(language_mask)
            last_valid = indices.masked_fill(~language_mask, -1).amax(dim=-1)
            last_valid = output.shape[1] - language_length + last_valid
            context.append(output[torch.arange(output.shape[0], device=output.device), last_valid].float().detach())

        handle = core.qwenvl_with_expert.qwenvl.model.language_model.norm.register_forward_hook(capture)
        try:
            dtype = next(core.parameters()).dtype
            proposal = core.sample_actions(
                images=batch["images"].to(dtype=dtype), img_masks=batch["img_masks"],
                lang_tokens=batch["lang_tokens"], lang_masks=batch["lang_masks"],
                state=batch["state"].to(dtype=dtype), noise=noise,
                image_grid_thw=batch.get("image_grid_thw"),
            )
        finally:
            handle.remove()
        if len(context) != 1:
            raise RuntimeError(f"Expected one prefix feature, captured {len(context)}")
        return proposal.float().detach(), context[0]


class RefinedLingbotPolicy(nn.Module):
    """Normalized-batch inference API; unnormalize only AFTER refinement."""

    def __init__(self, base_policy, refiner):
        super().__init__()
        self.base = FrozenLingbotProposal(base_policy)
        self.refiner = refiner

    @torch.no_grad()
    def sample_actions(self, batch, noise=None, num_steps=None):
        self.eval()
        proposal, context = self.base(batch, noise=noise)
        return self.refiner.refine(proposal, batch["state"], context,
                                   batch.get("action_joint_mask", batch.get("joint_mask")), num_steps)
