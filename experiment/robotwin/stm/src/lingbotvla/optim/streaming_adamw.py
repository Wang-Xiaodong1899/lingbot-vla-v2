"""Single-device AdamW, CPU states for large parameters and bounded elementwise workspaces."""
import torch
from torch.optim.adamw import adamw
from torch.distributed.tensor import DTensor

class StreamingAdamW(torch.optim.AdamW):
 @torch.no_grad()
 def step(self,closure=None):
  loss=None
  if closure is not None:
   with torch.enable_grad():loss=closure()
  for group in self.param_groups:
   if group.get('capturable') or group.get('differentiable'):
    raise ValueError('CPU-state AdamW cannot be capturable/differentiable')
   for p in group['params']:
    if p.grad is None:continue
    if isinstance(p,DTensor) or not p.is_contiguous() or p.grad.is_sparse or torch.is_complex(p):
     raise ValueError('CPU-state AdamW requires local contiguous real parameters and dense gradients')
    offload=p.numel()>2**22
    state_device='cpu' if offload else p.device
    st=self.state[p]
    if not st:
     st['step']=torch.tensor(0.,device='cpu')
     st['exp_avg']=torch.zeros_like(p,device=state_device)
     st['exp_avg_sq']=torch.zeros_like(p,device=state_device)
     if group['amsgrad']:st['max_exp_avg_sq']=torch.zeros_like(p,device=state_device)
    if offload:
     for key in st:
      if isinstance(st[key],torch.Tensor) and st[key].device.type!='cpu':st[key]=st[key].cpu()
    flat=p.view(-1);grad=p.grad.reshape(-1)
    for start in range(0,p.numel(),2**22):
     sl=slice(start,start+2**22)
     m=st['exp_avg'].view(-1)[sl].to(p.device)
     v=st['exp_avg_sq'].view(-1)[sl].to(p.device)
     vmax=[st['max_exp_avg_sq'].view(-1)[sl].to(p.device)] if group['amsgrad'] else []
     local_step=st['step'].clone()
     adamw([flat[sl]],[grad[sl]],[m],[v],vmax,[local_step],foreach=False,capturable=False,differentiable=False,fused=False,
           amsgrad=group['amsgrad'],beta1=group['betas'][0],beta2=group['betas'][1],lr=group['lr'],weight_decay=group['weight_decay'],eps=group['eps'],maximize=group['maximize'])
     if offload:
      st['exp_avg'].view(-1)[sl].copy_(m.cpu());st['exp_avg_sq'].view(-1)[sl].copy_(v.cpu())
      if vmax:st['max_exp_avg_sq'].view(-1)[sl].copy_(vmax[0].cpu())
     del m,v,vmax
    st['step'].add_(1)
  return loss
