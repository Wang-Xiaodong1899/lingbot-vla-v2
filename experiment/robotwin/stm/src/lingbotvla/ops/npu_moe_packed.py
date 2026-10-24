"""Pack routed tokens into padded expert batches; three batched GEMMs."""
import torch
import torch.nn.functional as F

def npu_moe_forward(hidden_states,routing_weights,selected_experts,gate,up,down):
 def local(x):return x.to_local() if hasattr(x,'to_local') else x
 h,g,u,d=[local(x).to(torch.bfloat16) for x in (hidden_states,gate,up,down)]
 T,H=h.shape;E=g.shape[0];K=selected_experts.shape[1]
 ids=selected_experts.reshape(-1).long()
 # Expert IDs are small integers, represented exactly in FP32.
 # NPU integer argsort falls back to AiCPU; FP32 uses the device kernel.
 order=torch.argsort(ids.float(),stable=True)
 sorted_ids=ids[order]
 counts=(ids[:,None] == torch.arange(E,device=ids.device)[None,:]).float().sum(0).long()
 capacity=int(counts.max().item())
 if capacity==0:return h*0
 starts=counts.cumsum(0)-counts
 slots=torch.arange(ids.numel(),device=h.device)-starts[sorted_ids]
 packed_idx=sorted_ids*capacity+slots
 packed=h.new_zeros((E*capacity,H)).index_copy(0,packed_idx,h[order//K]).view(E,capacity,H)
 activated=F.silu(torch.bmm(packed,g.transpose(1,2)))*torch.bmm(packed,u.transpose(1,2))
 out=torch.bmm(activated,d.transpose(1,2)).reshape(E*capacity,H)[packed_idx]
 routed=out.new_zeros((T*K,H)).index_copy(0,order,out).view(T,K,H)
 return (routed.float()*routing_weights.float().unsqueeze(-1)).sum(1).to(h.dtype)
