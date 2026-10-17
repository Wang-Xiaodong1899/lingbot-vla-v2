"""Read-only attention collection for Lingbot's AutoHorizon integration."""
import torch
from deploy.autohorizon_selector import bidir_soft_pointer, pick_horizon_softpointer


@torch.no_grad()
def action_attention(query, key, mask, horizon):
    """Mean over query heads, after full-key softmax and action-block slicing.

    Q/K are post-RoPE tensors [B, tokens, heads, head_dim]. Preserve the
    prefix/state/future-token softmax denominator before pooling over layers.
    """
    if query.shape[1] < horizon or key.shape[1] < horizon:
        raise ValueError('Action horizon exceeds available query/key tokens')
    heads, kv_heads = query.shape[2], key.shape[2]
    if heads % kv_heads:
        raise ValueError('Query head count must be divisible by KV head count')
    q = query[:, -horizon:].float().transpose(1, 2)
    k = key.float().repeat_interleave(heads // kv_heads, dim=2).transpose(1, 2)
    logits = torch.matmul(q, k.transpose(-1, -2)) * query.shape[-1] ** -0.5
    if mask.dtype != torch.bool or mask.ndim != 3:
        raise ValueError('Expected Lingbot boolean attention mask [B,Q,K]')
    logits = torch.where(mask[:, None, -horizon:, :], logits, -2.3819763e38)
    return logits.softmax(dim=-1)[..., -horizon:].mean(dim=1)


class AutoHorizonCollector:
    """Wrap an eager sampler without changing its attention output or velocity."""
    def __init__(self, flow_model, horizon, attention_step=3):
        if attention_step < 1 or attention_step > flow_model.config.num_steps:
            raise ValueError('attention_step must lie within the denoising schedule')
        self.horizon = horizon
        self.attention_step = attention_step
        self.original_velocity = flow_model.predict_velocity
        expert = flow_model.qwenvl_with_expert
        self.original_attention = expert.attention_interface
        self.reset()

        def velocity(*args, **kwargs):
            self.step += 1
            return self.original_velocity(*args, **kwargs)

        def attention(query, key, value, mask):
            # The original attention forward is always used for model outputs.
            output = self.original_attention(query, key, value, mask)
            if self.step == self.attention_step:
                block = action_attention(query, key, mask, self.horizon)
                self.attention_sum = block if self.attention_sum is None else self.attention_sum + block
                self.layers += 1
            return output

        flow_model.predict_velocity = velocity
        expert.attention_interface = attention

    def reset(self):
        self.step = 0
        self.layers = 0
        self.attention_sum = None

    def finish(self):
        if self.layers == 0 or self.attention_sum is None:
            raise RuntimeError('No attention captured; eager, uncompiled inference is required')
        # Upstream divides the third-step tensor by total denoising steps.
        # Its subsequent row normalization cancels this common scalar. Retain it
        # here for parity, and average over layers; batch dimension stays separate.
        return self.attention_sum / self.layers / self.step


def select_horizon(attention, method='bidir', hold_threshold=0.3, entropy_quantile=0.9):
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1] or attention.shape[0] < 2:
        raise ValueError('Expected square action attention with horizon >= 2')
    if not torch.isfinite(attention).all() or (attention < 0).any():
        raise ValueError('Attention must be finite and nonnegative')
    if (attention.sum(-1) <= 0).any():
        raise ValueError('Every action query must have nonzero action attention')
    if hold_threshold < 0 or not 0 < entropy_quantile <= 1:
        raise ValueError('Invalid soft-pointer thresholds')
    selectors = {'bidir': bidir_soft_pointer, 'forward': pick_horizon_softpointer}
    if method not in selectors:
        raise ValueError(f'Unknown selector: {method}')
    k, info = selectors[method](attention, hold_thr=hold_threshold, max_entropy_q=entropy_quantile)
    return int(k.item()), info
