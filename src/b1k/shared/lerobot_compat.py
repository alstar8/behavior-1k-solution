"""Compatibility helpers for older OmniGibson LeRobot imports."""

import dataclasses
import sys
import types


@dataclasses.dataclass(frozen=True)
class VideoEncoderConfig:
    vcodec: str
    pix_fmt: str
    g: int
    crf: int
    extra_options: dict[str, str] | None = None


@dataclasses.dataclass(frozen=True)
class DepthEncoderConfig(VideoEncoderConfig):
    depth_min: float = 0.0
    depth_max: float = 1.0
    shift: int = 0
    use_log: bool = False
    output_unit: str = "m"


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def ensure_legacy_lerobot_api() -> None:
    """Restore legacy LeRobot module paths expected by BEHAVIOR-1K."""
    import lerobot.configs as _configs
    import lerobot.constants as _constants
    import lerobot.datasets as _datasets
    import lerobot.datasets.utils as _dataset_utils
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not hasattr(_datasets, "LeRobotDataset"):
        _datasets.LeRobotDataset = LeRobotDataset

    io_utils = _ensure_module("lerobot.datasets.io_utils")
    if not hasattr(io_utils, "write_info"):
        io_utils.write_info = _dataset_utils.write_info

    constants = _ensure_module("lerobot.utils.constants")
    if not hasattr(constants, "HF_LEROBOT_HOME"):
        constants.HF_LEROBOT_HOME = _constants.HF_LEROBOT_HOME

    if not hasattr(_configs, "VideoEncoderConfig"):
        _configs.VideoEncoderConfig = VideoEncoderConfig
    if not hasattr(_configs, "DepthEncoderConfig"):
        _configs.DepthEncoderConfig = DepthEncoderConfig
