"""Portable batched expert computation preserving the official fused weight layout."""
import torch
import torch.nn.functional as F

def npu_moe_forward(hidden_states, routing_weights, selected_experts, gate, up, down):
    def local(x):
        return x.to_local() if hasattr(x, "to_local") else x
    h, gate, up, down = [local(x).to(torch.bfloat16) for x in (hidden_states, gate, up, down)]
    count = gate.shape[0]
    x = h.unsqueeze(0).expand(count, -1, -1).contiguous()
    activated = F.silu(torch.bmm(x, gate.transpose(1, 2))) * torch.bmm(x, up.transpose(1, 2))
    outputs = torch.bmm(activated, down.transpose(1, 2))
    weights = (F.one_hot(selected_experts, num_classes=count).float() * routing_weights.float().unsqueeze(-1)).sum(1)
    return (outputs.float() * weights.transpose(0, 1).unsqueeze(-1)).sum(0).to(h.dtype)
