"""Check STM HF export, DCP metadata and per-rank training states."""
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader

checkpoint = Path(sys.argv[1]).resolve()
hf = checkpoint/'hf_ckpt'
index = json.loads((hf/'model.safetensors.index.json').read_text())
model_config = json.loads((hf/'config.json').read_text())
assert model_config['temporal_memory'] is True
assert model_config['temporal_memory_dim'] == 256 and model_config['temporal_memory_heads'] == 4
assert model_config['freeze_vision_encoder'] is True
base = Path(os.environ.get('INIT_MODEL', '/root/autodl-tmp/models/hggforget-lingbotvla-robotwin-clean-sft/global_step_20000'))
base_map = json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
new_keys = set(index['weight_map']) - set(base_map)
assert new_keys and all(k.startswith('model.temporal_memory.') for k in new_keys)
assert set(base_map).issubset(index['weight_map'])
record = {'checkpoint':str(checkpoint),'hf_shards':{},'stm_parameters':{},'training_states':{}}
for file in sorted(set(index['weight_map'].values())):
    path = hf/file
    with safe_open(path, framework='pt') as bundle:
        for key in bundle.keys():
            assert index['weight_map'][key] == file
            if key in new_keys:
                tensor = bundle.get_tensor(key).float()
                assert torch.isfinite(tensor).all(), key
                record['stm_parameters'][key] = {'shape':list(tensor.shape),'norm':tensor.norm().item()}
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    record['hf_shards'][file] = {'bytes':path.stat().st_size,'sha256':digest}
assert record['stm_parameters']['model.temporal_memory.up.weight']['norm'] > 0
for section in ['model','optimizer']:
    reader = FileSystemReader(checkpoint/section)
    metadata = reader.read_metadata()
    assert metadata.state_dict_metadata
    for storage in metadata.storage_data.values():
        assert (checkpoint/section/storage.relative_path).stat().st_size >= storage.offset+storage.length
    record[section+'_metadata_keys'] = len(metadata.state_dict_metadata)
for rank in range(2):
    state = torch.load(checkpoint/'extra_state'/f'extra_state_rank_{rank}.pt', map_location='cpu', weights_only=False)
    assert state['global_step'] == int(checkpoint.name.removeprefix('global_step_'))
    for key in ['lr_scheduler','train_dataloader','environ_meter','torch_rng_state']:
        assert key in state
    record['training_states'][str(rank)] = {'global_step':state['global_step'],'keys':sorted(state)}
record['verified'] = True
(checkpoint/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
