"""Where the Dino really sits in the hand jig today: calibration by the robot itself (jenga_autoscan calib).

The Dino is pushed into the jig by hand, so it never sits as it did at the last calibration, and the fingers
holding it don't say reliably where it went (2026-11-19: a hold that looked like the 11-18 one read ring j1
72 deg instead of 100.5; the finger model gave 61 mm, the robot's view needed the Dino much closer). So the
robot searches widely, the guesses only being where it starts:

  guesses   the last calibration, then (not with --no-predict) the tool moved with the ring finger's first
            link (ah?_link_32) from the calibration's hand_q to the fingers now
  depth     a sweep through -search_out_mm..+max_in_mm (45 / 40) in SWEEP_STEP_M, outside in, stopping past
            a clear peak; the next guess if there is none
  turn      any angle about the optical axis: the image turned in TURN_STEP_DEG until a TOP edge shows
            (best_turn), up / down by TURN_SEARCH_MM until one does
  height    the edge centred (Scanner.top_edge: a bottom edge -> up a layer)
  sideways  across the tower (side_positions): where the edge is seen = the layer's end faces; the middle of
            that run = the middle block. Not measured if the run is not found (then as the guess).
  check     the tool that puts the point in focus on the model top edge of the middle block, there again
            with it, corrected up to CHECKS times. The tool is only shifted; the turn is saved as
            dino.image_turn_deg (camera.DinoCamera turns every frame by it): the arm never rolls the hand.

Not measured: the tilt of the optical axis (kept from the guess). It does not move the point in focus (the
tool origin), it only turns the Dino about it. The tower is the reference: its model top edge is taken as the
truth (as for jenga-calib), so a tower that stands off the model moves the calibration with it.

    ros2 run jenga_vision jenga_autoscan calib                     # dry run: virtual arm, fake camera
    ros2 run jenga_vision jenga_autoscan calib --send              # real robot: measure, print, save nothing
    ros2 run jenga_vision jenga_autoscan calib --send --write      # and save into scan_setup.json
    ros2 run jenga_vision jenga_autoscan calib --send --write --hand jig_mount_20261119   # after mount --teach

--write backs up scan_setup.json and sets the image turn of <out>/.rigid.json to 0 (the turn is now in
image_turn_deg); its scale and focus stay, so the next run skips the first-block searches as before.
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
DEPTH_OK_M, HEIGHT_OK_M, LEVEL_OK_DEG = 0.0005, 0.0001, 0.7  # a check this close needs no more correction
# (depth: a refocus repeats within about +-0.3 mm on the robot, 2026-11-19 checks +0.34, -0.34)
SWEEP_STEP_M = 0.0015  # depth sweep step (the focus curve's s is about 2 mm)
PAST_PEAK = 0.6  # the sweep stops below this x the best so far, once the peak is clear
PEAK_RATIO = 1.6  # a clear peak: this x the median of the sweep, not at its ends
SWITCH_OUT_M = 0.06  # this far out from the face before the hand turns to the next guess
TURN_SEARCH_MM = (0, -4, 4, -8, 8, -12, 12, -16, 16, -20, 20, -24, 24)  # heights tried for an edge at any turn
SIDE_RANGE_M, SIDE_STEP_M = 0.11, 0.006  # sideways sweep: +-, step (tower 115 mm wide, middle block 38.5)
MIN_RUN_M = 0.07  # the end faces seen over at least this much (2 of 3 blocks), else sideways not measured
PREDICT_MIN_M = 0.005  # the finger guess is tried only if it differs from the last calibration this much
JIG_LINK = "link_32"  # ring finger, after joint 31 (first phalanx)


@dataclass
class Found:
    depth: float  # m past the model face where the block is in focus
    down: float  # m below the model edge (commanded)
    roll: float  # rad of arm roll in the command (0 with --soft-level)
    turn_deg: float  # image turn that levels the edge
    edge_err_m: float  # edge this far below the image centre (+ = below)
    tilt_deg: float  # edge tilt left in the turned image
    side_m: float = 0.0  # commanded this far sideways (+ = image right) from the model aim to see the middle block


def delta(shot, f: Found) -> np.ndarray:
    """Dino frame now -> where its point in focus really is, in the Dino frame (tool @ delta = new tool).
    Only a shift: the turn about the optical axis goes to dino.image_turn_deg (the camera turns every frame),
    so the arm never rolls the hand for it."""
    p = shot.pose(f.depth, f.down, f.roll)  # the commanded Dino pose, world
    p[:3, 3] += p[:3, 0] * f.side_m
    really = shot.aim - p[:3, 1] * f.edge_err_m  # point in focus: on the model face, the edge err above the edge
    t = p[:3, :3].T @ (really - p[:3, 3])  # z: -depth (in focus = on the model face), y: -(down + edge)
    return transform(np.eye(3), t)


def new_tool(tool: np.ndarray, shot, f: Found) -> np.ndarray:
    return tool @ delta(shot, f)


def good(f: Found) -> bool:
    return abs(f.depth) < DEPTH_OK_M and abs(f.down + f.edge_err_m) < HEIGHT_OK_M and abs(f.tilt_deg) < LEVEL_OK_DEG


def shifted(shot, side_m: float):
    """The shot moved `side_m` along the face (+ = image right)."""
    from dataclasses import replace
    return replace(shot, aim=shot.aim + shot.pose(0.0, 0.0)[:3, 0] * side_m)


def side_positions() -> list[float]:
    n = int(round(SIDE_RANGE_M / SIDE_STEP_M))
    return [k * SIDE_STEP_M for k in range(-n, n + 1)]


def run_around(on: list[float]) -> tuple[float, float] | None:
    """The longest run of neighbouring positions in `on` (sorted, m), if at least MIN_RUN_M long."""
    runs, start = [], None
    for k, x in enumerate(on):
        if start is None:
            start = x
        if k + 1 == len(on) or on[k + 1] - x > 2.5 * SIDE_STEP_M:  # one position missed (a seam) is still the run
            runs.append((start, x))
            start = None
    if not runs:
        return None
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    return (lo, hi) if hi - lo >= MIN_RUN_M else None


def clear_peak(seen: list[tuple[float, float]]) -> bool:
    vals = [f for _, f in seen]
    if len(vals) < 5:
        return False
    k = int(np.argmax(vals))
    return 1 <= k <= len(vals) - 2 and vals[k] > PEAK_RATIO * float(np.median(vals))


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
    guesses = [("the last calibration", tool_cal)]
    if ref_hq:
        worst = max(abs(math.degrees(x - y)) for x, y in zip(now_hq[12:20], ref_hq[12:20]))
        print(f"fingers ({source}): ring / little up to {worst:.1f} deg off the calibration's '{cal.get('hand_pose')}'")
        predicted = predict_tool(tool_cal, args.arm, ref_hq, now_hq)
        if not args.no_predict and np.linalg.norm(change(tool_cal, predicted)[0]) > PREDICT_MIN_M * 1000:
            print(f"   second guess, from the fingers: {change_text(tool_cal, predicted)}")
            guesses.append(("predicted from the fingers", predicted))
    tool0 = tool_cal
    face = cal["face"].replace("+", "p").replace("-", "m")
    args.face, args.layers = face, [cal["layer"], cal["layer"]]
    args.calib_layer, args.calib_slot = cal["layer"], cal["slot"]
    args.tool, args.ignore_hand, args.fix_hand, args.calib_result = tool0, True, False, None
    args.calib_candidates = guesses
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
    print(f"   turned about its axis {res['turn_deg']:+.1f} deg more than then (saved as an image turn, the arm "
          "does not roll for it)")
    if not np.allclose(res["start_tool"], tool_cal):
        print(f"   against the finger guess it started from: {change_text(res['start_tool'], tool)}")
    if not res["side_found"]:
        print("   sideways not measured (the tower's ends not found): kept from the guess")
    if not good(res["last"]):
        print("   the last check was not within "
              f"{DEPTH_OK_M * 1000:.1f} mm / {HEIGHT_OK_M * 1000:.1f} mm / {LEVEL_OK_DEG} deg")
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
    turned = (d.get("image_turn_deg", 0.0) + res["turn_deg"] + 180) % 360 - 180
    d["image_turn_deg"] = round(float(turned), 2)
    d["_image_turn_note"] = ("image_turn_deg: the Dino's turn about its axis in the jig on top of image_upside_down, "
                             "from jenga_autoscan calib; camera.DinoCamera turns every frame back by it")
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
    if rigid.exists():  # its image turn was the Dino's turn under the old calibration: now in image_turn_deg
        shutil.copy(rigid, rigid.with_name(f".rigid.json.bak_{stamp}"))
        data = json.loads(rigid.read_text())
        for face in data.values():
            face["turn_deg"], face["roll_deg"] = 0.0, 0.0
        rigid.write_text(json.dumps(data, indent=2))
        print(f"   {rigid}: image turn set to 0 (now in image_turn_deg), the rest kept: the next run is as fast")
