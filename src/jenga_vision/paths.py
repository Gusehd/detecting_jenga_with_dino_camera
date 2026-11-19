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

The game block file (the pull order jenga_game reads first, game_block.json) is the first of:
  1. $JENGA_GAME_BLOCK
  2. game_block.json in a folder above the checkout (jenga_ws/src/ros_pkgs/game_block.json on the robot PC)
  3. <the ros_pkgs folder above the checkout>/game_block.json, created by the first scan
  None when the checkout is not inside a ros_pkgs tree (e.g. the vision PC): then nothing is written there.
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


def game_block_file() -> Path | None:
    if os.environ.get("JENGA_GAME_BLOCK"):
        return Path(os.environ["JENGA_GAME_BLOCK"]).expanduser()
    for folder in CHECKOUT.parents:
        if (folder / "game_block.json").exists():
            return folder / "game_block.json"
    for folder in CHECKOUT.parents:
        if folder.name == "ros_pkgs":
            return folder / "game_block.json"
    return None


ROOT = data_root()
CONFIG_DIR = ROOT / "config"
CAPTURES_DIR = ROOT / "captures"
GAME_BLOCK_FILE = game_block_file()
