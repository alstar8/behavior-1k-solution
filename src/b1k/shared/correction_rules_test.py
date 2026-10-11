import unittest

import numpy as np

from b1k.shared.correction_rules import LEFT_GRIPPER_IDX, RIGHT_GRIPPER_IDX, apply_correction_rules
from b1k.shared.eval_b1k_wrapper import B1KWrapperConfig


class CorrectionRulesTest(unittest.TestCase):
    def test_radio_keeps_final_stage(self):
        state = np.zeros(23, dtype=np.float32)
        state[[LEFT_GRIPPER_IDX, RIGHT_GRIPPER_IDX]] = 1.0
        actions = np.random.default_rng(0).normal(size=(30, 23)).astype(np.float32)
        out_actions, out_stage = apply_correction_rules(0, 4, state, actions)
        self.assertEqual(out_stage, 4)
        np.testing.assert_array_equal(out_actions, actions)

    def test_radio_opens_gripper_closed_before_stage_2(self):
        state = np.zeros(23, dtype=np.float32)
        state[LEFT_GRIPPER_IDX] = -1.0
        actions = np.zeros((30, 23), dtype=np.float32)
        out_actions, out_stage = apply_correction_rules(0, 1, state, actions)
        self.assertEqual(out_stage, 1)
        np.testing.assert_array_equal(out_actions[:, LEFT_GRIPPER_IDX], 1.0)

    def test_wrapper_does_not_compress_by_default(self):
        config = B1KWrapperConfig()
        self.assertEqual(config.execute_in_n_steps, config.actions_to_execute)


if __name__ == "__main__":
    unittest.main()
