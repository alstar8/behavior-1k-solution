#!/usr/bin/env python3
"""Check the 2025-submission action adapter against the 2026 radio action layout.

The adapter is used only for IliaLarchenko/behavior_submission checkpoints.
"""

from __future__ import annotations

import numpy as np

from b1k.shared.convert_2025_action import convert_2025_action_to_2026


def main() -> None:
    action_2025 = np.zeros(23, dtype=np.float32)
    action_2025[3:7] = np.array([1.025, -1.45, -0.47, 0.0], dtype=np.float32)
    action_2025[14] = 1.0
    action_2025[22] = 1.0
    converted = convert_2025_action_to_2026(action_2025)
    if converted.shape != (23,):
        raise SystemExit(f"bad shape {converted.shape}")
    if not np.allclose(converted, action_2025):
        raise SystemExit("2025 and 2026 action layouts diverged")
    print("2025 action -> 2026 action: 23-d layout matches (identity)")


if __name__ == "__main__":
    main()
