"""Single-device Muon with CPU momentum and bounded expert workspaces.
Preserves FP32 master weights, momentum dtype and Muon update mathematics.
Not a distributed optimizer. No gradient clearing or precision truncation.
"""
import torch
from torch.distributed.tensor import DTensor
from lingbotvla.optim.muon import DistributedMuon, batched_newton_schulz, _adjust_lr

class StreamingMuon(DistributedMuon):
 @torch.no_grad()
 def step(self, closure=None):
  loss=None
  if closure is not None:
   with torch.enable_grad(): loss=closure()
  for group in self.param_groups:
   lr=float(group['lr']); wd=float(group['weight_decay']); momentum=float(group['momentum'])
   for p in group['params']:
    if isinstance(p,DTensor): raise RuntimeError('StreamingMuon supports only single-device tensors')
    if p.grad is None: continue
    if p.grad.is_sparse or torch.is_complex(p): raise RuntimeError('Dense real gradients required')
    state=self.state[p]
    if 'momentum_buffer' not in state:
     state['momentum_buffer']=torch.zeros_like(p,device='cpu')
    if state['momentum_buffer'].device.type!='cpu':
     state['momentum_buffer']=state['momentum_buffer'].cpu()
    buf=state['momentum_buffer']
    adjusted_lr=_adjust_lr(lr,group['adjust_lr_fn'],p.shape[-2:])
    # Experts have independent NS normalization; chunking their leading axis
    # leaves each matrix's update unchanged.
    slices=[slice(i,i+4) for i in range(0,p.shape[0],4)] if p.ndim==3 else [slice(None)]
    for sl in slices:
     grad=p.grad[sl]; mom=buf[sl].to(p.device)
     mom.lerp_(grad,1-momentum)
     update=grad.lerp(mom,momentum) if group['nesterov'] else mom.clone()
     buf[sl].copy_(mom.cpu())
     del mom
     ortho=batched_newton_schulz(update,tuple(group['ns_coefficients']),int(group['ns_steps']),float(group['eps']))
     if wd: p[sl].mul_(1-lr*wd)
     p[sl].add_(ortho.to(p.dtype),alpha=-adjusted_lr)
     del update,ortho
  return loss
