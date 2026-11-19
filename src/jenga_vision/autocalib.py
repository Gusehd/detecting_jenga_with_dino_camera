"""Where the Dino really sits in the hand jig today: calibration by the robot itself (jenga_autoscan calib).

The Dino is pushed into the jig by hand, so it never sits exactly as it did at the last calibration. With the
Dino mounted (jenga_autoscan mount), the robot

  1. predicts the tool from the finger angles: the jig rides on the ring finger's first link
     (ah?_link_32; the URDF has no little finger), so the Dino moves with that link from the calibration's
     hand_q to the fingers now. 2026-11-19: ring j1 83 deg instead of 100 = focus point 37 mm, 17 deg away.
  2. goes to the calibration block (dino.calibration.<arm>: face, layer, slot; +x L07 middle) with it, searches
     focus, edge and level (Scanner.lock_on) and computes the tool that puts the point in focus on the model
     top edge with the image level
  3. goes there again with the new tool and checks (focus depth, edge row, tilt), at most CHECKS times

Measured: depth (along the optical axis), height (image down) and the turn about the optical axis. Not
measured (kept from the prediction): sideways along the gap and the tilt of the optical axis. Neither moves the
point in focus: the tool origin is that point, a tilt turns the Dino about it, and a sideways offset only
shifts the view along the 25 mm wide end face.

The tower is the reference: its model top edge is taken as the truth (as for jenga-calib), so a tower that
stands off the model moves the calibration with it.

    ros2 run jenga_vision jenga_autoscan calib                     # dry run: virtual arm, fake camera
    ros2 run jenga_vision jenga_autoscan calib --send              # real robot: measure, print, save nothing
    ros2 run jenga_vision jenga_autoscan calib --send --write      # and save into scan_setup.json
    ros2 run jenga_vision jenga_autoscan calib --send --write --hand jig_mount_20261119   # after mount --teach
    ... --no-predict      the fingers look as at the last calibration although they read other angles
                          (2026-11-19: the same hold read ring j1 72 deg, the 11-18 calibration 100.5):
                          start from the calibrated Dino as it is, not from the finger prediction

--write backs up scan_setup.json and moves <out>/.rigid.json aside (its image turn and focus belong to the
old calibration): the next run does a full sweep and saves a new one.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass

import numpy as np

from jenga_vision.kinematics import URDF, axis_angle, load_joints, transform
from jenga_vision.scan_plan import SETUP_FILE, arm_tool

CHECKS = 2  # goes with the new tool to check (and correct again)
DEPTH_OK_M, HEIGHT_OK_M, LEVEL_OK_DEG = 0.0002, 0.0001, 0.7  # a check this close needs no more correction
MAX_FROM_PREDICTION = (0.015, 15.0)  # m, deg: found further than this from the prediction -> not saved
MAX_PREDICTION = (0.06, 30.0)  # m, deg: fingers that far off -> mount again instead
JIG_LINK = "link_32"  # ring finger, after joint 31 (first phalanx)


@dataclass
class Found:
    depth: float  # m past the model face where the block is in focus
    down: float  # m below the model edge (commanded)
    roll: float  # rad of arm roll in the command (0 with --soft-level)
    turn_deg: float  # image turn that levels the edge
    edge_err_m: float  # edge this far below the image centre (+ = below)
    tilt_deg: float  # edge tilt left in the turned image


def delta(shot, f: Found, roll_sign: int = 1) -> np.ndarray:
    """Dino frame now -> Dino frame where it really is, in the Dino frame (tool @ delta = new tool)."""
    p = shot.pose(f.depth, f.down, f.roll)  # the commanded Dino pose, world
    really = shot.aim - p[:3, 1] * f.edge_err_m  # point in focus: on the model face, the edge err above the edge
    t = p[:3, :3].T @ (really - p[:3, 3])  # z: -depth (in focus = on the model face), y: -(down + edge)
    return transform(axis_angle(np.array([0, 0, 1.0]), roll_sign * math.radians(f.turn_deg + f.tilt_deg)), t)


def new_tool(tool: np.ndarray, shot, f: Found, roll_sign: int = 1) -> np.ndarray:
    return tool @ delta(shot, f, roll_sign)


def good(f: Found) -> bool:
    return (abs(f.depth) < DEPTH_OK_M and abs(f.down + f.edge_err_m) < HEIGHT_OK_M
            and abs(f.turn_deg + f.tilt_deg) < LEVEL_OK_DEG)


def link_pose(arm: str, hq, link: str = JIG_LINK) -> np.ndarray:
    """Pose of a hand link in the hand frame (ah?_base_link) for the 20 hand values `hq`."""
    joints, h = load_joints(URDF), arm[0]
    name, chain = f"ah{h}_{link}", []
    while name != f"ah{h}_base_link":
        chain.append(joints[name])
        name = joints[name].parent
    t = np.eye(4)
    for j in reversed(chain):
        t = t @ j.origin
        if j.kind != "fixed":
            f, k = int(j.name[-2]), int(j.name[-1])
            t = t @ transform(axis_angle(j.axis, hq[f * 4 + k]), [0, 0, 0])
    return t


def predict_tool(tool: np.ndarray, arm: str, ref_hq, now_hq) -> np.ndarray:
    """The tool moved with the jig's finger link from `ref_hq` (calibration) to `now_hq`."""
    return link_pose(arm, now_hq) @ np.linalg.inv(link_pose(arm, ref_hq)) @ tool


def change(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """b against a in a's Dino frame: (mm x sideways / y down / z along the axis, total deg, roll deg, tilt deg)."""
    d = np.linalg.inv(a) @ b
    total = math.degrees(math.acos(float(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))))
    roll = math.degrees(math.atan2(d[1, 0], d[0, 0]))
    tilt = math.degrees(math.acos(float(np.clip(d[2, 2], -1, 1))))
    return d[:3, 3] * 1000, total, roll, tilt


def change_text(a: np.ndarray, b: np.ndarray) -> str:
    mm, total, roll, tilt = change(a, b)
    return (f"sideways {mm[0]:+.1f}, down {mm[1]:+.1f}, along the axis {mm[2]:+.1f} mm "
            f"({np.linalg.norm(mm):.1f}); turned {total:.1f} deg (roll {roll:+.1f}, axis tilt {tilt:.1f})")


def hand_now(args, ref_hq) -> tuple[list[float], str]:
    if args.send and not args.fake:
        from jenga_vision.arm_slider import ArmNode
        node = ArmNode(args.arm, False, node_name="jenga_calib_hand")
        names = [f"ah{args.arm[0]}_joint{f}{j}" for f in range(5) for j in range(4)]
        t0 = time.time()
        while any(n not in node.positions for n in names) and time.time() - t0 < 20:
            time.sleep(0.1)
        positions = dict(node.positions)
        node.shutdown()
        if any(n not in positions for n in names):
            raise SystemExit(f"no /joint_states of the {args.arm} hand (ROS sourced? ROS_DOMAIN_ID?)")
        return [positions[n] for n in names], "/joint_states"
    if args.hand:
        from jenga_vision.hand_slider import POSES_FILE
        return json.loads(POSES_FILE.read_text())[args.arm[0]][args.hand], f"hand pose '{args.hand}' (dry run)"
    return list(ref_hq), "the calibration's hand_q (dry run)"


def calib_main(args, scan_main, run_faces: dict) -> None:
    setup = json.loads(SETUP_FILE.read_text())
    cal = setup["dino"].get("calibration", {}).get(args.arm)
    if not cal or not setup["dino"].get("tool_hand"):
        raise SystemExit("no Dino calibration to start from (jenga-calib once by hand)")
    tool_cal = arm_tool(setup["dino"], args.arm)
    ref_hq = cal.get("hand_q")
    now_hq, source = hand_now(args, ref_hq or [0.0] * 20)
    if args.no_predict:  # the fingers look as at the calibration, only their joint readings differ
        tool0 = tool_cal
        print(f"--no-predict: starting from the calibrated Dino as it is (fingers read from {source})")
    elif ref_hq:
        tool0 = predict_tool(tool_cal, args.arm, ref_hq, now_hq)
        worst = max(abs(math.degrees(a - b)) for a, b in zip(now_hq[12:20], ref_hq[12:20]))
        print(f"fingers ({source}): ring / little up to {worst:.1f} deg off the calibration's '{cal.get('hand_pose')}'")
        print(f"   predicted Dino: {change_text(tool_cal, tool0)}")
        mm, total, _, _ = change(tool_cal, tool0)
        if np.linalg.norm(mm) > MAX_PREDICTION[0] * 1000 or total > MAX_PREDICTION[1]:
            raise SystemExit("the fingers are too far from the calibration to predict the Dino: mount it again")
    else:
        tool0 = tool_cal
        print("no hand_q in the calibration: starting from the calibrated tool as it is")

    face = cal["face"].replace("+", "p").replace("-", "m")
    args.face, args.layers = face, [cal["layer"], cal["layer"]]
    args.calib_layer, args.calib_slot = cal["layer"], cal["slot"]
    args.tool, args.ignore_hand, args.fix_hand, args.calib_result = tool0, True, False, None
    for k, v in run_faces.get(cal["face"], {}).items():  # seed, via, end of the jig scan
        if k in ("seed", "via", "end"):
            setattr(args, k, v)
    args.prior_depth_mm = args.prior_down_mm = 0.0
    args.start = getattr(args, "end", None) or args.start  # the virtual arm starts where the jig scans start
    if args.send:
        print(f"the arm must be at '{args.start}' (the jig scans' start), away from the tower")
    session, _ = scan_main(args)
    res = args.calib_result
    if res is None:
        print("calibration not finished: nothing saved")
        return
    tool = res["tool"]
    print(f"\nDino against the calibration of {cal.get('date', '?')}: {change_text(tool_cal, tool)}")
    print(f"   against the prediction from the fingers: {change_text(tool0, tool)}")
    if not good(res["last"]):
        print("   the last check was not within "
              f"{DEPTH_OK_M * 1000:.1f} mm / {HEIGHT_OK_M * 1000:.1f} mm / {LEVEL_OK_DEG} deg")
    mm, total, _, _ = change(tool0, tool)
    if np.linalg.norm(mm) > MAX_FROM_PREDICTION[0] * 1000 or total > MAX_FROM_PREDICTION[1]:
        print("NOT saved: that far from the prediction looks like a wrong block or edge; look at the session images "
              f"({session})")
        return
    if not args.write:
        print("not saved (add --write)")
        return
    if args.fake or not args.send:
        print("not saved: fake camera / virtual arm")
        return
    write(args, setup, tool, now_hq, res)


def write(args, setup: dict, tool: np.ndarray, now_hq, res: dict) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = SETUP_FILE.with_name(f"{SETUP_FILE.name}.bak_{stamp}_before_autocalib")
    shutil.copy(SETUP_FILE, backup)
    d = setup["dino"]
    old = d["calibration"][args.arm]
    d["tool_hand"][args.arm] = [[round(float(v), 6) for v in row] for row in tool]
    cal = {**old, "pose": "autoscan calib", "q": [round(float(v), 6) for v in res["q"]],
           "date": time.strftime("%Y-%m-%d %H:%M"), "method": "jenga_autoscan calib (focus, edge, level; "
           "sideways and axis tilt from the finger prediction)", "hand_q": [round(float(v), 6) for v in now_hq],
           "mm_per_px": res["mm_per_px"], "previous": {k: old.get(k) for k in ("date", "pose", "hand_pose")}}
    if args.hand:  # after mount --teach: this hand pose is the mount pose from now on
        from jenga_vision.hand_slider import POSES_FILE
        hands = json.loads(POSES_FILE.read_text())[args.arm[0]]
        cal["hand_pose"] = args.hand
        cal.pop("hand_cmd", None)
        if f"{args.hand}_cmd" in hands:
            cal["hand_cmd"] = hands[f"{args.hand}_cmd"]
    d["calibration"][args.arm] = cal
    SETUP_FILE.write_text(json.dumps(setup, indent=2, ensure_ascii=False) + "\n")
    print(f"saved to {SETUP_FILE} (backup {backup.name})")
    rigid = args.out / ".rigid.json"
    if rigid.exists():
        rigid.rename(rigid.with_name(f".rigid.json.bak_{stamp}"))
        print(f"   {rigid} moved aside: the next run sweeps the faces fully and saves a new one")
