import unittest

import numpy as np

from b1k.action_alignment import ACTION_SLICES
from b1k.action_alignment import D2026_LEAD
from b1k.action_alignment import RFT_LEAD
from b1k.action_alignment import aligned_action_chunks
from b1k.action_alignment import assemble_action


def _episode(length: int = 8, dim: int = 23) -> np.ndarray:
    actions = np.zeros((length, dim), dtype=np.float32)
    row = np.arange(length, dtype=np.float32)
    actions[:, 0] = 0.75 * row
    actions[:, 1] = 100.0 + row
    actions[:, 2] = 200.0 + row
    for col in range(3, dim):
        actions[:, col] = 1000.0 * col + row
    return actions


class ActionAlignmentTest(unittest.TestCase):
    def test_rft_takes_the_next_row_for_every_slice(self):
        actions = _episode()
        horizon = 3
        label = assemble_action(actions, 2, horizon, RFT_LEAD)
        self.assertIsNotNone(label)
        np.testing.assert_array_equal(label, actions[3:6])

    def test_2026_splits_base_from_arms(self):
        actions = _episode()
        horizon = 3
        t = 2
        label = assemble_action(actions, t, horizon, D2026_LEAD)
        self.assertIsNotNone(label)
        np.testing.assert_array_equal(label[:, 0:3], actions[t : t + horizon, 0:3])
        for name, (lo, hi) in ACTION_SLICES.items():
            if name == "base":
                continue
            np.testing.assert_array_equal(label[:, lo:hi], actions[t - 1 : t - 1 + horizon, lo:hi])

    def test_2026_skips_when_the_previous_row_or_the_base_window_is_missing(self):
        actions = _episode()
        horizon = 3
        self.assertIsNone(assemble_action(actions, 0, horizon, D2026_LEAD))
        self.assertIsNotNone(assemble_action(actions, 5, horizon, D2026_LEAD))
        self.assertIsNone(assemble_action(actions, 6, horizon, D2026_LEAD))

    def test_rft_skips_past_the_last_full_window(self):
        actions = _episode()
        horizon = 3
        self.assertIsNotNone(assemble_action(actions, 4, horizon, RFT_LEAD))
        self.assertIsNone(assemble_action(actions, 5, horizon, RFT_LEAD))

    def test_base_command_is_not_rescaled(self):
        actions = _episode()
        label = assemble_action(actions, 2, 3, D2026_LEAD)
        np.testing.assert_array_equal(label[:, 0], 0.75 * np.arange(2, 5, dtype=np.float32))
        rft = assemble_action(actions, 2, 3, RFT_LEAD)
        np.testing.assert_array_equal(rft[:, 0], 0.75 * np.arange(3, 6, dtype=np.float32))

    def test_cropped_block_matches_the_full_episode(self):
        actions = _episode()
        horizon = 3
        for leads, t in ((RFT_LEAD, 0), (D2026_LEAD, 2)):
            full = assemble_action(actions, t, horizon, leads)
            span0 = min(t + lead for lead in leads.values())
            span1 = max(t + lead + horizon for lead in leads.values())
            cropped = assemble_action(actions[span0:span1], t - span0, horizon, leads)
            np.testing.assert_array_equal(cropped, full)

    def test_vectorized_chunks_match_assemble(self):
        actions = _episode(length=12)
        horizon = 4
        for leads in (RFT_LEAD, D2026_LEAD):
            chunks, index = aligned_action_chunks(actions, leads, horizon)
            self.assertGreater(len(index), 0)
            for row, t in enumerate(index):
                np.testing.assert_array_equal(chunks[row], assemble_action(actions, int(t), horizon, leads))


if __name__ == "__main__":
    unittest.main()
