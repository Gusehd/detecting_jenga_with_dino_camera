"""Dino calibration from one shot pose taught on the robot.

The user puts the Dino by hand (manual mode / jog) so that it looks at a known
block: the gap above the end face is in the middle of the image, horizontal,
in focus. That pose (saved with jenga-arm "Save robot pose") and the tower
model give the Dino frame in the hand frame:

    tool = FK(q)^-1 @ shot(aim point of that end face)

The tool origin is the point in focus (not the Dino tip), so the focus distance
needs no measuring: planning with standoff 0 puts every end face in focus.
Written to config/scan_setup.json (dino.tool_hand.<arm>, aim = top edge,
standoff 0); then run `python -m jenga_vision.scan_plan --write` again.

    uv run jenga-calib cali_way_3 --face +x --layer 1 --slot 0           # print only
    uv run jenga-calib cali_way_3 --face +x --layer 1 --slot 0 --write

Layers and slots are the teammate's: L00 = bottom layer, S0..S2 with S1 the
middle; S0 is image left on every face (image up = world up).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from jenga_vision.arm_slider import POSES_FILE
from jenga_vision.kinematics import Chain
from jenga_vision.scan_plan import FACE_NORMALS, SETUP_FILE, Tower, arm_tool, shot_pose

AIM_TOP_EDGE = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pose", help="saved arm pose (config/arm_poses.json) of the shot")
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument("--face", choices=list(FACE_NORMALS), required=True)
    parser.add_argument("--layer", type=int, required=True, help="teammate's layer, L00 = bottom")
    parser.add_argument("--slot", type=int, choices=(0, 1, 2), required=True, help="0 = image left, 1 = middle")
    parser.add_argument("--write", action="store_true", help=f"save into {SETUP_FILE}")
    args = parser.parse_args()

    setup = json.loads(SETUP_FILE.read_text())
    q = json.loads(POSES_FILE.read_text())[args.arm][args.pose]
    tower = Tower(setup["tower"], AIM_TOP_EDGE)
    faces = [f for f in tower.end_faces(args.face, range(args.layer, args.layer + 1)) if f[1] == args.slot]
    if not faces:
        raise SystemExit(f"no end face of L{args.layer:02d} on face {args.face} "
                         f"(even layers show their ends on +-y, odd ones on +-x)")
    _, _, aim, normal = faces[0]
    # Order of the slots in the image: S0 must be image left.
    right_dir = shot_pose(aim, normal, 0)[:3, 0]
    slot_dirs = [f[2] for f in tower.end_faces(args.face, range(args.layer, args.layer + 1))]
    if (slot_dirs[2] - slot_dirs[0]) @ right_dir < 0:
        print("note: on this face S0 is on the image RIGHT (slot numbering runs right to left)")

    hand = Chain(args.arm).fk(q)
    tool = np.linalg.inv(hand) @ shot_pose(aim, normal, 0.0)
    old = arm_tool(setup["dino"], args.arm)
    np.set_printoptions(precision=1, suppress=True)
    print(f"{args.arm} arm pose '{args.pose}' -> {args.face} L{args.layer:02d}_S{args.slot} top edge, "
          f"aim point (world, mm) {aim * 1000}")
    print(f"point in focus, hand frame (mm): {tool[:3, 3] * 1000}   was {old[:3, 3] * 1000} "
          f"({np.linalg.norm(tool[:3, 3] - old[:3, 3]) * 1000:.0f} mm away)")
    ang = math.degrees(math.acos(np.clip(tool[:3, 2] @ old[:3, 2], -1, 1)))
    roll = math.degrees(math.atan2(old[:3, 1] @ tool[:3, 0], old[:3, 1] @ tool[:3, 1]))
    print(f"optical axis, hand frame: {np.round(tool[:3, 2], 3)}   {ang:.1f} deg from before, image roll {roll:+.1f} deg")
    print(f"focus distance from the hand origin: {np.linalg.norm(tool[:3, 3]) * 1000:.0f} mm")
    if not args.write:
        print("(dry: pass --write to save)")
        return
    d = setup["dino"]
    d.setdefault("tool_hand", {})[args.arm] = [[round(float(v), 6) for v in row] for row in tool]
    d["standoff_m"] = 0.0
    d.setdefault("tip_to_focus_m", 0.025)
    d.setdefault("calibration", {})[args.arm] = {
        "pose": args.pose, "q": q, "face": args.face, "layer": args.layer, "slot": args.slot,
        "aim": "top edge", "date": time.strftime("%Y-%m-%d %H:%M")}
    d["_tool_note"] = ("tool_hand: Dino frame in the hand frame from jenga-calib (origin = point in focus, "
                       "z = optical axis, y = image down). A missing arm uses the other one mirrored. "
                       "tip_to_focus_m: Dino tip behind the focus point (measured at dial 30, only for the sim clearance).")
    setup["aim"] = AIM_TOP_EDGE
    SETUP_FILE.write_text(json.dumps(setup, indent=2, ensure_ascii=False) + "\n")
    print(f"saved to {SETUP_FILE}; now: uv run python -m jenga_vision.scan_plan  (add --write to save routes)")


if __name__ == "__main__":
    main()
