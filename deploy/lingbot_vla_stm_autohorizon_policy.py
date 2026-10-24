"""AutoHorizon with the checkpoint-matched STM implementation."""
from pathlib import Path
import sys
import deploy

STM_ROOT = Path(__file__).resolve().parents[1] / 'experiment/robotwin/stm/src'
sys.path.insert(0, str(STM_ROOT))
deploy.__path__.insert(0, str(STM_ROOT / 'deploy'))

from deploy.lingbot_vla_autohorizon_policy import AutoHorizonServer, main

_original = AutoHorizonServer._temporal_inputs

def _validated_temporal_inputs(self, applied, observations):
    result = _original(self, applied, observations)
    if not self.config.temporal_memory:
        raise ValueError('STM evaluation requires temporal_memory=True')
    if observations[0]['timestamp'] >= 1 and self.replan < 3:
        print('STM_HISTORY ages=' + str(result['history_ages'].tolist()) +
              ' valid_views=' + str(result['history_img_masks'].sum().item()), flush=True)
    return result

AutoHorizonServer._temporal_inputs = _validated_temporal_inputs

if __name__ == '__main__':
    main()
