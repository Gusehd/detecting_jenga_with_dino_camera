"""Where config/, captures/ and the robot URDF are, for the uv project and the ROS 2 package alike.

The data root (holds config/ and captures/) is the first of:
  1. $JENGA_VISION_DIR
  2. the checkout this code runs from (uv editable install, or colcon --symlink-install)
  3. the current directory, if it has config/scan_setup.json
  4. the installed package share (colcon build without --symlink-install): config/ is a copy
     there, and taught poses or calibration saved into it are lost on the next build.
User-given relative paths (--out captures/x) stay relative to the current directory.

The URDF (only the arm chains are read, see kinematics.py) is the first that exists of:
  1. $JENGA_URDF
  2. <checkout>/jenga_world/jenga_bringup/bi_manipulator_config/urdf/dual_robot.urdf (the 10/29 copy on the vision PC)
  3. share/bi_manipulator_config/urdf/dual_robot.urdf of the sourced ROS 2 workspace
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE = "jenga_vision"
CHECKOUT = Path(__file__).resolve().parents[2]
URDF_IN_REPO = Path("jenga_world/jenga_bringup/bi_manipulator_config/urdf/dual_robot.urdf")


def _share(package: str) -> Path | None:
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    except ImportError:
        return None
    try:
        return Path(get_package_share_directory(package))
    except PackageNotFoundError:
        return None


def data_root() -> Path:
    if os.environ.get("JENGA_VISION_DIR"):
        return Path(os.environ["JENGA_VISION_DIR"]).expanduser()
    for root in (CHECKOUT, Path.cwd()):
        if (root / "config/scan_setup.json").exists():
            return root
    share = _share(PACKAGE)
    return share if share is not None else CHECKOUT


def find_urdf() -> Path:
    candidates = [Path(os.environ["JENGA_URDF"]).expanduser()] if os.environ.get("JENGA_URDF") else []
    candidates.append(CHECKOUT / URDF_IN_REPO)
    share = _share("bi_manipulator_config")
    if share is not None:
        candidates.append(share / "urdf/dual_robot.urdf")
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]  # missing: the error names the last place looked


ROOT = data_root()
CONFIG_DIR = ROOT / "config"
CAPTURES_DIR = ROOT / "captures"
