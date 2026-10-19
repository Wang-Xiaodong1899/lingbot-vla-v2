import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('monitor',ROOT/'tools/monitor_corrector_handoff.py')
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


class HandoffTests(unittest.TestCase):
    def test_cache_requires_all_expected_episodes_and_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            metadata=dict(total_episodes=2,total_frames=24,smoke_max_episodes=0,
                          training_data=[dict(training_episodes=[0,1])])
            for e in range(2):
                path=root/f'task00_episode{e:06d}';path.mkdir()
                (path/'complete.json').write_text(json.dumps(dict(frames=12,tokens=3,token_dim=4)))
                for name,array in dict(z_q=np.zeros((12,3,4),dtype=np.int8),
                                       z_scale=np.ones((12,3),dtype=np.float16),
                                       actions=np.zeros((12,55),dtype=np.float32),
                                       mask=np.ones(55,dtype=np.bool_)).items():
                    np.save(path/f'{name}.npy',array)
            self.assertEqual(monitor.validate_cache(root,metadata),dict(episodes=2,frames=24,arrays_verified=8))
            (root/'task00_episode000001/z_q.npy').unlink()
            with self.assertRaises(FileNotFoundError):
                monitor.validate_cache(root,metadata)

    def test_partial_cache_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(AssertionError):
                monitor.validate_cache(Path(folder),dict(total_episodes=2,total_frames=24))


if __name__=='__main__':unittest.main()
