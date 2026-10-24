"""Adapt the published STM candidate to this CUDA host; preserve provenance."""
import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get('STM_ASSETS_ROOT', ROOT.parents[2])).resolve()
BASE = Path(os.environ.get('INIT_MODEL', '/root/autodl-tmp/models/hggforget-lingbotvla-robotwin-clean-sft/global_step_20000')).resolve()
output = Path(os.environ['OUTPUT_DIR']).resolve()
launch = output / 'launch'
launch.mkdir(parents=True, exist_ok=True)
config = yaml.safe_load((ROOT / 'training/initial_candidate/train.temporal.yaml').read_text())
micro = int(os.environ.get('MICRO_BATCH_SIZE', '16'))
gpus = int(os.environ['NGPU'])
assert micro > 0 and gpus > 0 and 512 % (micro * gpus) == 0
config['model'].update(model_path=str(BASE), config_path=str(BASE),
    tokenizer_path=str(ASSETS / 'evaluation/Qwen3-VL-4B-Instruct'))
robot_dir = launch / 'robot_configs'
robot_dir.mkdir(exist_ok=True)
robot = yaml.safe_load((BASE.parent / 'launch/robot_configs/robotwin_clean.yaml').read_text())
robot['norm_stats'] = str(BASE.parent / 'robotwin_clean_norm.json')
(robot_dir / 'robotwin_clean.yaml').write_text(yaml.safe_dump(robot, sort_keys=False))
dataset = ASSETS / 'data/RoboTwin_lerobot_v30'
(launch / 'train.txt').write_text(f'robotwin_clean {dataset}\n')
config['data'].update(train_path=str(launch / 'train.txt'), robot_config_root=str(robot_dir),
    norm_stats_file=robot['norm_stats'], num_workers=4, prefetch_factor=1)
train = config['train']
train.update(output_dir=str(output), micro_batch_size=micro, gradient_accumulation_steps=512//(micro*gpus),
    data_parallel_shard_size=gpus, data_parallel_replicate_size=1,
    npu_moe_mode='dense', enable_reentrant=False, max_steps=4000, save_steps=1000,
    save_interval_seconds=int(os.environ.get('SAVE_INTERVAL_SECONDS', '7200')),
    verify_save_step=3, save_hf_weights=True, async_save_hf_weights=False, enable_resume=True)
align = train['align_params']
align['depth'].update(moge_path=str(ASSETS / 'weights/depth/moge2-vitb-normal.pt'),
    morgbd_path=str(ASSETS / 'weights/depth/model.pt'))
align['video'].update(ckpt_path=str(ASSETS / 'weights/dino_video/teacher_step_10000.pth'),
    config_path=str(ASSETS / 'weights/dino_video/config.yaml'))
required = [BASE/'config.json', BASE/'model.safetensors.index.json', Path(robot['norm_stats']), dataset/'meta/info.json']
required += [BASE / f for f in set(json.loads((BASE/'model.safetensors.index.json').read_text())['weight_map'].values())]
required += [Path(align[s][k]) for s, keys in [('depth',['moge_path','morgbd_path']),('video',['ckpt_path','config_path'])] for k in keys]
for path in required:
    assert path.is_file() and path.stat().st_size, f'Missing asset: {path}'
old = launch / 'config.yaml'
if old.exists() and any((output/'checkpoints').glob('global_step_*')):
    previous = yaml.safe_load(old.read_text())
    # Microbatch and saving cadence may change without changing the experiment.
    for item in (previous, config):
        for key in ['micro_batch_size','gradient_accumulation_steps','save_interval_seconds']:
            item['train'].pop(key, None)
    assert previous == config, 'Refuse changing experiment settings in an existing output directory'
    train.update(micro_batch_size=micro, gradient_accumulation_steps=512//(micro*gpus),
        save_interval_seconds=int(os.environ.get('SAVE_INTERVAL_SECONDS', '7200')))
old.write_text(yaml.safe_dump(config, sort_keys=False))
(launch/'provenance.json').write_text(json.dumps({
    'source_repo':'zhuojianli/icm', 'source_revision':'6cae6fc54d004c34b573cdb486c1beb9762f1c2c',
    'initial_checkpoint':str(BASE), 'dataset':str(dataset),
    'published_formal_settings':{'global_batch_size':512,'max_steps':4000,'save_steps':1000},
    'limitation':'Formal H20 CLI config is absent from the published repository. LR/optimizer/architecture follow the explicit initial candidate, not a claimed exact formal reproduction.',
    'lr':train['lr'], 'lr_min':train['lr_min'], 'optimizer':train['optimizer'],
    'augmentation':'Published shared brightness/contrast/saturation; LINGBOT_AUG_LEVEL=0 (candidate default). Formal z3a environment was not published.',
    'initial_optimizer_state':'New Muon state; baseline only supplies model weights.',
    'local_changes':['CUDA paths and 2-GPU accumulation','7200-second collective checkpoint timer plus step 3 save validation','frozen-ViT gradient audit corrected','refuse silent restart if resume fails','DCP writer thread count bounded by LINGBOT_DCP_THREADS=1'],
}, indent=2)+'\n')
print(f'Config: {old}; {gpus} GPUs × {micro} micro × {512//(micro*gpus)} accumulation = 512', flush=True)
