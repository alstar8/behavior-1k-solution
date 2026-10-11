import unittest

import numpy as np

from b1k.shared.b1k_proprio import R1PRO_COMPACT_PROPRIOCEPTION_INDICES
from b1k.shared.normalize import NormStats
from b1k.transforms import ProprioNoise
from b1k.transforms_normalize import floor_state_std


class ProprioNoiseTest(unittest.TestCase):
    def test_perturbs_only_joints_and_base_velocity(self):
        proprio = np.tile(np.arange(61, dtype=np.float32), (20000, 1))
        out = ProprioNoise(joint_std=0.01, base_vel_std=0.02)({"observation.state": proprio})["observation.state"]
        diff = out - proprio
        indices = R1PRO_COMPACT_PROPRIOCEPTION_INDICES
        trunk = indices["trunk_qpos"]
        noisy = np.zeros(61, dtype=bool)
        for part_slice, std in (
            (indices["base_qvel"], 0.02),
            (np.s_[trunk.start : trunk.stop - 1], 0.01),
            (indices["arm_left_qpos"], 0.01),
            (indices["arm_right_qpos"], 0.01),
        ):
            noisy[part_slice] = True
            np.testing.assert_allclose(diff[:, part_slice].std(axis=0), std, rtol=0.05)
        np.testing.assert_array_equal(diff[:, ~noisy], 0.0)
        np.testing.assert_array_equal(diff[:, trunk.stop - 1], 0.0)
        np.testing.assert_array_equal(proprio[0], np.arange(61, dtype=np.float32))

    def test_rejects_legacy_proprio(self):
        with self.assertRaises(ValueError):
            ProprioNoise(joint_std=0.01)({"observation.state": np.zeros(256, dtype=np.float32)})


class FloorStateStdTest(unittest.TestCase):
    def setUp(self):
        self.stats = {
            "state": NormStats(mean=np.zeros(3), std=np.array([0.4, 0.0007, 0.06])),
            "actions": NormStats(mean=np.zeros(3), std=np.array([1e-8, 0.003, 0.1])),
        }

    def test_raises_only_small_state_std(self):
        out = floor_state_std(self.stats, 0.05)
        np.testing.assert_allclose(out["state"].std, [0.4, 0.05, 0.06])
        np.testing.assert_array_equal(out["actions"].std, self.stats["actions"].std)
        np.testing.assert_array_equal(self.stats["state"].std, [0.4, 0.0007, 0.06])

    def test_zero_floor_is_identity(self):
        self.assertIs(floor_state_std(self.stats, 0.0), self.stats)
        self.assertIsNone(floor_state_std(None, 0.05))


if __name__ == "__main__":
    unittest.main()
