"""Verify render skipping preserves every STM frame under variable horizons."""
import ast
from pathlib import Path
import random
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'experiment/robotwin/eval_policy_client_lingbotvla.py'
module = ast.parse(SOURCE.read_text())
function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'temporal_capture_required')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
capture_required = namespace['temporal_capture_required']


class CaptureTests(unittest.TestCase):
    def check_sequence(self, chunks):
        current = 0
        retained = {0}
        for chunk in chunks:
            end = current + chunk
            for step in range(current + 1, end + 1):
                if capture_required(step, end):
                    retained.add(step)
            for offset in (0, 8, 15):
                target = end - offset
                if target >= 0:
                    self.assertIn(target, retained, (current, chunk, offset))
            retained = {step for step in retained if step >= end - 15}
            current = end

    def test_every_pair_of_horizons(self):
        for previous in range(1, 51):
            for following in range(1, 51):
                self.check_sequence([previous, following])

    def test_long_variable_horizon_stream(self):
        rng = random.Random(0)
        self.check_sequence([rng.randint(1, 50) for _ in range(10000)])

    def test_only_unneeded_frames_are_skipped(self):
        self.assertEqual([i for i in range(1, 51) if capture_required(i, 50)], list(range(35, 51)))
        self.assertTrue(all(capture_required(i, 8) for i in range(1, 9)))


if __name__ == '__main__':
    unittest.main()
