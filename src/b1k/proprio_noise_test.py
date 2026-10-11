import unittest

import numpy as np

from b1k.shared.b1k_proprio import R1PRO_COMPACT_PROPRIOCEPTION_INDICES
from b1k.transforms import ProprioNoise


class ProprioNoiseTest(unittest.TestCase):
    def test_perturbs_only_joints_and_base_velocity(self):
        proprio = np.tile(np.arange(61, dtype=np.float32), (20000, 1))
        out = ProprioNoise(joint_std=0.01, base_vel_std=0.02)({"observation.state": proprio})["observation.state"]
        diff = out - proprio
        indices = R1PRO_COMPACT_PROPRIOCEPTION_INDICES
        noisy = np.zeros(61, dtype=bool)
        for key, std in (
            ("base_qvel", 0.02),
            ("trunk_qpos", 0.01),
            ("arm_left_qpos", 0.01),
            ("arm_right_qpos", 0.01),
        ):
            noisy[indices[key]] = True
            np.testing.assert_allclose(diff[:, indices[key]].std(axis=0), std, rtol=0.05)
        np.testing.assert_array_equal(diff[:, ~noisy], 0.0)
        np.testing.assert_array_equal(proprio[0], np.arange(61, dtype=np.float32))

    def test_rejects_legacy_proprio(self):
        with self.assertRaises(ValueError):
            ProprioNoise(joint_std=0.01)({"observation.state": np.zeros(256, dtype=np.float32)})


if __name__ == "__main__":
    unittest.main()
