"""Read-only asset checks for the local clean-only PICA training recipe."""
import argparse
import json
from pathlib import Path
import re

import yaml


def check(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    missing = []
    errors = []

    def exists(path):
        if not Path(path).exists():
            missing.append(str(path))
            return False
        return True

    baseline = Path(config['model']['model_path'])
    index_path = baseline / 'model.safetensors.index.json'
    if exists(index_path):
        weights = json.loads(index_path.read_text())['weight_map']
        for shard in sorted(set(weights.values())):
            exists(baseline / shard)
        layers = {int(m.group(1)) for key in weights
                  if (m := re.search(r'qwen_expert\.model\.layers\.(\d+)\.', key))}
        if layers != set(range(36)):
            errors.append(f'Expected 36 action layers in V2 baseline, got {sorted(layers)}')
    exists(baseline / 'config.json')
    tokenizer = Path(config['model']['tokenizer_path'])
    exists(tokenizer / 'config.json')
    exists(tokenizer / 'tokenizer_config.json')
    exists(tokenizer / 'preprocessor_config.json')
    exists(config['data']['norm_stats_file'])
    align = config['train']['align_params']
    for section, names in [('depth', ['moge_path', 'morgbd_path']), ('video', ['ckpt_path', 'config_path'])]:
        for name in names:
            exists(align[section][name])
    manifest = Path(config['data']['train_path'])
    if exists(manifest):
        lines = [line.split(maxsplit=1) for line in manifest.read_text().splitlines() if line.strip()]
        if len(lines) != 50 or len({line[-1] for line in lines}) != 50:
            errors.append('Expected 50 unique clean tasks in training manifest')
        for line in lines:
            if len(line) != 2 or line[0] != 'robotwin':
                errors.append(f'Invalid task manifest entry: {line}')
                continue
            dataset = Path(line[1])
            if not exists(dataset):
                continue
            # Match the transactional conversion guard in the baseline launcher.
            marker_path, info_path = dataset / '.conversion_complete.json', dataset / 'meta/info.json'
            if not (exists(marker_path) and exists(info_path)):
                continue
            marker, info = json.loads(marker_path.read_text()), json.loads(info_path.read_text())
            if marker['episodes'] != 50 or info['total_episodes'] != 50 or info['codebase_version'] != 'v2.1':
                errors.append(f'Incomplete/incompatible clean dataset: {dataset}')
    if missing:
        errors.append(f'Missing {len(missing)} training assets/directories:\n  ' + '\n  '.join(missing))
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PICA assets ready: {baseline}; 36 action layers; 50 clean tasks / 2500 episodes.')
    print('Batch unit: observation/action chunk; teacher/student layer indices: '
          f"{config['train']['pica_params']['teacher_layer']}/{config['train']['pica_params']['student_layer']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    check(parser.parse_args().config)
