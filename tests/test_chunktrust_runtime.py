import unittest
import numpy as np
from deploy.chunktrust_runtime import ChunkTrustRuntime


class RuntimeTest(unittest.TestCase):
    def test_actual_execution_and_episode_reset(self):
        rng = np.random.default_rng(1)
        velocity = rng.normal(size=(10, 50, 14)).astype(np.float32)
        actions = rng.normal(size=(50, 14)).astype(np.float32)
        runtime = ChunkTrustRuntime(seed=12)
        k, info = runtime.select(velocity, actions)
        self.assertTrue(10 <= k <= 50)
        with self.assertRaises(ValueError):
            runtime.select(velocity, actions)
        with self.assertRaises(ValueError):
            runtime.acknowledge(k + 1)
        runtime.acknowledge(3)
        np.testing.assert_array_equal(runtime.history, actions[:3])
        runtime.select(velocity, actions)
        runtime.acknowledge(0)
        np.testing.assert_array_equal(runtime.history, actions[:3])
        fresh = ChunkTrustRuntime(seed=12)
        self.assertEqual(len(fresh.history), 0)
        self.assertEqual(fresh.select(velocity, actions)[0], k)


if __name__ == '__main__':
    unittest.main()
