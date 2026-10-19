"""Real Lingbot sampling + OGG smoke, without changing any model weights."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from tasks.vla.train_action_refiner import read_base_config,load_base,build_data,to_device
from lingbotvla.corrector.runtime import load_corrector,visual_tokens,online_guidance,check_base_metadata,dequantize_tokens


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(0)
    predictor,payload=load_corrector(a.checkpoint,'cuda')
    m=payload['metadata']['feature_metadata']
    check_base_metadata(m,m['base_model'],m['norm_stats'])
    args=SimpleNamespace(base_model=m['base_model'],qwen_path=m['qwen_path'],base_dtype='float32',
                         norm_stats=Path(m['norm_stats']),train_list=m['train_list'],
                         robot_config_root=str(ROOT/'configs/robot_configs'))
    config=read_base_config(args);config.precompute_grid_thw=False;config.use_qwen3_fixed_grid_cache=False
    dataset,collate,_=build_data(args,config)
    batch=to_device(next(iter(DataLoader(dataset,batch_size=1,collate_fn=collate))),torch.device('cuda'))
    base=load_base(args,config,torch.device('cuda'))
    core=base.policy.model
    with torch.no_grad():
        z=visual_tokens(core,batch)
        episode=a.cache/'task00_episode000000'
        cached=dequantize_tokens(np.load(episode/'z_q.npy',mmap_mode='r')[0],np.load(episode/'z_scale.npy',mmap_mode='r')[0])
        alignment=torch.nn.functional.cosine_similarity(z.flatten(),torch.from_numpy(cached).cuda().flatten(),dim=0)
        if alignment<0.999:
            raise ValueError(f'Offline/online visual feature mismatch: {alignment.item()}')
        future=dequantize_tokens(np.load(episode/'z_q.npy',mmap_mode='r')[10],np.load(episode/'z_scale.npy',mmap_mode='r')[10])
        actual=torch.from_numpy(future).cuda()[None]-z
        prediction=predictor(z,batch['actions'][:,0]*batch['action_joint_mask'])
        target=prediction-actual
        noise=torch.randn(1,config.n_action_steps,config.max_action_dim,device='cuda')
        kwargs=dict(images=batch['images'].float(),img_masks=batch['img_masks'],lang_tokens=batch['lang_tokens'],
                    lang_masks=batch['lang_masks'],state=batch['state'].float(),noise=noise.clone(),
                    image_grid_thw=batch['image_grid_thw'])
        ordinary=core.sample_actions(**kwargs)
        kwargs['noise']=noise.clone()
        with online_guidance(core,predictor,z,target,batch['action_joint_mask']) as stats:
            guided=core.sample_actions(**kwargs)
    assert stats and any(s['gradient_norm']>0 for s in stats)
    assert torch.isfinite(guided).all() and (guided-ordinary).norm()>0
    assert all(p.grad is None for p in base.parameters())
    assert all(p.grad is None for p in predictor.parameters())
    report=dict(check='passed',base_frozen=True,corrector_frozen=True,feature_cosine=float(alignment),
                proposal_shape=list(guided.shape),visual_shape=list(z.shape),ogg_steps=len(stats),
                action_delta_norm=float((guided-ordinary).norm()),stats=stats)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
