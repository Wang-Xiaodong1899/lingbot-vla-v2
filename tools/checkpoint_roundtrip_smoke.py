"""Isolated four-GPU FSDP2/Muon save/resume check, not a VLA training run."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from lingbotvla.checkpoint.checkpointer import DistributedCheckpointer
from lingbotvla.checkpoint.integrity import validate_checkpoint
from lingbotvla.optim import build_muon_optimizer
from lingbotvla.utils.async_hf_checkpoint import AsyncHFCheckpointSaver


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    root = Path(sys.argv[1]).resolve()
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 8)).cuda()
    fully_shard(model, mesh=init_device_mesh('cuda', (dist.get_world_size(),)))
    opt = build_muon_optimizer(model, SimpleNamespace(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=.9)
    x = torch.randn(8, 256, device='cuda')
    last = None
    for step in (1, 2):
        model(x).square().mean().backward()
        opt.step()
        scheduler.step()
        opt.zero_grad()
        last = {'model':model, 'optimizer':opt, 'extra_state':{
            'global_step':step, 'lr_scheduler':scheduler.state_dict(),
            'train_dataloader':{'rank':rank,'cursor':step*8}, 'torch_rng_state':torch.get_rng_state()}}
        DistributedCheckpointer.save(str(root), last, global_steps=step)
        validate_checkpoint(root/f'global_step_{step}', dist.get_world_size())
    expected = [p.to_local().detach().clone() for p in model.parameters()]
    expected_opt = {name:{k: v.to_local().clone() if hasattr(v,'to_local') else v.clone() if torch.is_tensor(v) else v
                         for k,v in opt.state[param].items()} for name,param in model.named_parameters()}
    # Resume into a fresh optimizer, as a real process restart does.
    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 8)).cuda()
    fully_shard(model, mesh=init_device_mesh('cuda', (dist.get_world_size(),)))
    opt = build_muon_optimizer(model, SimpleNamespace(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=.9)
    restored = {'model':model,'optimizer':opt,'extra_state':{}}
    DistributedCheckpointer.load(str(root/'global_step_2'), restored)
    assert restored['extra_state']['global_step'] == 2
    assert restored['extra_state']['train_dataloader']['rank'] == rank
    scheduler.load_state_dict(restored['extra_state']['lr_scheduler'])
    assert scheduler.last_epoch == 2
    for actual, target in zip(model.parameters(),expected):
        torch.testing.assert_close(actual.to_local(),target,rtol=0,atol=0)
    for name,param in model.named_parameters():
        state, target = opt.state[param], expected_opt[name]
        for key, value in target.items():
            actual=state[key]
            if hasattr(actual,'to_local'): actual=actual.to_local()
            if torch.is_tensor(value): torch.testing.assert_close(actual,value,rtol=0,atol=0)
            else: assert actual==value
    # Interrupt between model and optimizer saving: the formal step directory
    # must remain absent and the last complete checkpoint must stay readable.
    original_save = DistributedCheckpointer._save_uncommitted.__func__
    def interrupted_save(cls, path, state, global_steps=None, save_async=False):
        original_save(cls, path, {'model':state['model']}, global_steps, save_async)
        raise RuntimeError('injected failure after model shards')
    DistributedCheckpointer._save_uncommitted = classmethod(interrupted_save)
    try:
        try:
            DistributedCheckpointer.save(str(root), restored, global_steps=3)
        except RuntimeError as exc:
            assert 'injected failure' in str(exc)
        else:
            raise AssertionError('Injected save failure was ignored')
    finally:
        DistributedCheckpointer._save_uncommitted = classmethod(original_save)
    assert not (root/'global_step_3').exists()
    validate_checkpoint(root/'global_step_2',dist.get_world_size())
    # A partial directory must never pass resume validation.
    if rank == 0:
        (root/'.global_step_3.incomplete-test'/'model').mkdir(parents=True)
        try: validate_checkpoint(root/'.global_step_3.incomplete-test',dist.get_world_size())
        except ValueError: pass
        else: raise AssertionError('Partial checkpoint was accepted')
        saver = AsyncHFCheckpointSaver(enabled=False)
        saver.submit(2,str(root/'global_step_2'),str(root),'dcp',False,False,None)
        assert (root/'global_step_2/hf_ckpt').is_dir()
        print('PASS: two successive saves; fresh FSDP2/Muon optimizer + model exact restore; scheduler/rank state restore; injected interrupted save does not publish; prior checkpoint preserved; partial checkpoint rejection; HF export',flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
