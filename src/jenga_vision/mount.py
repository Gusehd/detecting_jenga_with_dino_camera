"""Dino in the hand jig, held by the ring and little finger.

Once per jig / Dino (then `jenga_autoscan calib --send --hand <name> --write`, see autocalib.py):

    ros2 run jenga_vision jenga_autoscan mount --teach --send
        torque off -> push the Dino in, close ring and little by hand -> Enter -> torque on; the pose the hand
        then holds is saved in hand_poses.json as <name> (measured) and <name>_cmd (the command that holds it)

Every demo day:

    ros2 run jenga_vision jenga_autoscan mount --send
        torque off -> push the Dino in -> Enter -> torque on, the fingers go slowly to the calibration's
        hand_cmd (else hand_q) and are compared with hand_q
    ros2 run jenga_vision jenga_autoscan calib --send --write   (where the Dino sits today, about 1-2 min)
    ros2 run jenga_vision jenga_autoscan run --send      (checks the fingers again before it moves)

Without --send nothing is published and the torque is not touched. The torque service (std_srvs/SetBool,
the hand driver's ~/set_torque_enabled) switches the whole hand: all 20 joints go limp and all of them are
commanded. Before the torque comes back the hand's current pose is sent as the command, so nothing jumps to
an old command. Ctrl+C at any point switches the torque back on.

Why a taught command (2026-11-19): the 11-18 calibration pose jig_scan_pose was pushed by hand to ring / little
j1 100 deg (URDF limit 90); commanded there the fingers stopped at 83 / 80 deg.

Also used by `jenga_autoscan run --fix-hand`: the fingers back to the calibration (no torque change).
"""

from __future__ import annotations

import json
import math
import os
import time

from jenga_vision import hand_slider
from jenga_vision.scan_plan import SETUP_FILE

TORQUE_SERVICES = {"right": "/ahr_AllegroHandV6Right/set_torque_enabled",
                   "left": "/ahl_AllegroHandV6Left/set_torque_enabled"}
TORQUE_TIMEOUT_S = 3.0
STATE_WAIT_S = 10.0
CLOSE_SPEED = 20.0  # deg/s of the largest finger joint, as jenga-task
SETTLE_S = 1.0  # after the close, before the fingers are compared with the calibration
# jenga_pose (teammate) starts a hand move at its last command, kept in this file; stale once the torque was
# off, so it is deleted like their pose recorder does (jenga_pose/common/hand_command.py)
TEAMMATE_LAST_CMD = os.path.join(os.environ.get("JENGA_STATE_DIR", "/tmp"), "jenga_hand_cmd_{side}.json")


def calibration_hand_q(arm: str) -> list[float]:
    q = json.loads(SETUP_FILE.read_text())["dino"].get("calibration", {}).get(arm, {}).get("hand_q")
    if not q:
        raise SystemExit(f"no dino.calibration.{arm}.hand_q in {SETUP_FILE} (jenga-calib --hand)")
    return q


def calibration_hand_cmd(arm: str) -> list[float]:
    """What to command for the calibration pose: hand_cmd (taught with mount --teach: the command under which
    the hand held hand_q), else hand_q itself (11-18 calibration, a pose pushed by hand)."""
    cal = json.loads(SETUP_FILE.read_text())["dino"].get("calibration", {}).get(arm, {})
    return cal.get("hand_cmd") or calibration_hand_q(arm)


def hand_node(arm: str, send: bool, node):
    topic = f"/allegro_hand_position_effort_controller_{arm[0]}/commands"
    return hand_slider.HandNode(arm[0], send, topic, node=node)


def wait_hand(hand, wait_s: float = STATE_WAIT_S) -> list[float] | None:
    t0 = time.time()
    while hand.current() is None and time.time() - t0 < wait_s:
        time.sleep(0.05)
    return hand.current()


def close_to(hand, start: list[float], target: list[float], send: bool, speed: float = CLOSE_SPEED) -> None:
    """All 20 joints from `start` to `target`, every joint at most `speed` deg/s (as jenga-task hand_move).
    The target is not clamped to the URDF limits: the calibration pose was measured on the hand and has the
    ring / little j1 at 100 deg (jig_scan_pose, 2026-11-18), past the URDF 90. Only the start is clamped to the
    limits widened to the target."""
    def wide_clamp(i: int, v: float) -> float:
        lo, hi = hand_slider.LIMITS[hand_slider.JOINTS[i]]
        return min(max(v, min(lo, target[i])), max(hi, target[i]))

    cmd = [wide_clamp(i, v) for i, v in enumerate(start)]
    target = list(target)
    biggest = max(abs(math.degrees(t - c)) for t, c in zip(target, cmd))
    print(f"   fingers to the calibration pose: largest joint change {biggest:.1f} deg, {biggest / speed:.1f} s")
    if not send:
        return
    step = math.radians(speed) / hand_slider.RATE_HZ
    while True:
        moved = False
        for i, (c, t) in enumerate(zip(cmd, target)):
            d = max(-step, min(step, t - c))
            if d:
                cmd[i] = c + d
                moved = True
        if not moved:
            break
        hand.publish(list(cmd))
        time.sleep(1 / hand_slider.RATE_HZ)


def set_torque(node, arm: str, enabled: bool) -> None:
    from std_srvs.srv import SetBool

    client = node.create_client(SetBool, TORQUE_SERVICES[arm])
    if not client.wait_for_service(timeout_sec=TORQUE_TIMEOUT_S):
        raise RuntimeError(f"no torque service {TORQUE_SERVICES[arm]} (real hand driver running? jenga-sim has none)")
    future = client.call_async(SetBool.Request(data=enabled))
    t0 = time.time()
    while not future.done() and time.time() - t0 < TORQUE_TIMEOUT_S:
        time.sleep(0.01)
    if not future.done() or not future.result().success:
        raise RuntimeError(f"torque {'on' if enabled else 'off'} failed: "
                           f"{future.result().message if future.done() else 'no answer'}")
    try:
        os.remove(TEAMMATE_LAST_CMD.format(side=arm))
    except OSError:
        pass
    print(f"   torque {'ON' if enabled else 'OFF'} ({TORQUE_SERVICES[arm]})")


def hold_here(hand, now: list[float]) -> list[float]:
    """Command the pose the fingers are at (before the torque comes back: no jump to an old command)."""
    cmd = [float(v) for v in now]
    for _ in range(5):
        hand.publish(cmd)
        time.sleep(0.02)
    return cmd


def finger_text(q: list[float]) -> str:
    return "  ".join(f"{name} " + " ".join(f"{math.degrees(q[f * 4 + j]):6.1f}" for j in range(4))
                     for f, name in ((3, "ring"), (4, "little")))


def save_hand_pose(arm: str, name: str, measured: list[float], cmd: list[float]) -> None:
    """hand_poses.json <hand>.<name> = measured, <name>_cmd = the command that holds it (backup first)."""
    path = hand_slider.POSES_FILE
    backup = path.with_name(f"{path.name}.bak_{time.strftime('%Y%m%d_%H%M%S')}_before_{name}")
    backup.write_text(path.read_text())
    data = json.loads(path.read_text())
    data.setdefault(arm[0], {})[name] = [round(float(v), 5) for v in measured]
    data[arm[0]][f"{name}_cmd"] = [round(float(v), 5) for v in cmd]
    path.write_text(json.dumps(data, indent=2))
    print(f"   saved '{name}' and '{name}_cmd' to {path} (backup {backup.name})")


def teach_main(args, node, hand) -> None:
    """mount --teach: torque off, Dino in, the user closes the ring / little finger by hand, Enter: torque on,
    the hand holds there; the pose it settles at is saved. The Dino calibration must then be made with it."""
    name = args.name or time.strftime("jig_mount_%Y%m%d")
    input(("the hand is limp" if args.send else "(dry run) here the hand would be limp")
          + ": push the Dino in, close ring and little by hand so it sits in the hand, hold them, Enter > ")
    now = wait_hand(hand)
    if now is None:
        print("   (no /joint_states: dry run, nothing saved)")
        return
    print(f"   at Enter:     {finger_text(now)} deg")
    cmd = hold_here(hand, now)
    if args.send:
        set_torque(node, args.arm, True)
        args.torque_off = False
        print("   torque on: let go of the fingers")
        time.sleep(SETTLE_S + 1.0)
    held = hand.current() or now
    print(f"   hand holds:   {finger_text(held)} deg")
    moved = max(abs(math.degrees(a - b)) for a, b in zip(held[12:20], now[12:20]))
    if moved > 3.0:
        print(f"   the fingers moved {moved:.1f} deg after the torque came on (the hand can't hold past its limit or "
              "force): this is the pose that is saved, the one mount can make again. Check the Dino still sits right.")
    if not args.send:
        print("   (dry run: nothing saved)")
        return
    save_hand_pose(args.arm, name, held, cmd)
    print(f"next (arm at jig_ready_1118): the robot finds the Dino in this pose, predicted from the fingers\n"
          f"   jenga_autoscan calib --send --hand {name}            # measure and print\n"
          f"   jenga_autoscan calib --send --hand {name} --write    # and save: '{name}' is the mount pose from then on\n"
          f"every demo day: jenga_autoscan mount --send, jenga_autoscan calib --send --write, then run --send")


def mount_main(args) -> None:
    from jenga_vision.arm_slider import start_node, stop_node
    from jenga_vision.autoscan import HAND_STOP_DEG, HAND_WARN_DEG, hand_check

    setup = json.loads(SETUP_FILE.read_text())
    torque = not args.no_torque
    if args.teach:
        if not torque:
            raise SystemExit("--teach needs the torque off (no --no-torque)")
        print(f"teach the mount pose, {args.arm} hand: torque off -> Dino in, close ring / little by hand -> Enter -> "
              "torque on, the pose is saved")
    else:
        target, cmd_target = calibration_hand_q(args.arm), calibration_hand_cmd(args.arm)
        pose_name = setup["dino"]["calibration"][args.arm].get("hand_pose", "hand_q")
        print(f"Dino mount, {args.arm} hand: " + ("torque off -> Dino in -> Enter -> " if torque else "")
              + f"fingers to '{pose_name}' (the calibration)")
    if not args.send:
        print("DRY RUN (add --send to really do it): nothing is published, no torque change")
    node = start_node("jenga_mount")
    hand = hand_node(args.arm, args.send, node)
    args.torque_off = False
    try:
        now = wait_hand(hand)
        if now is None and args.send:
            raise RuntimeError(f"no /joint_states of the {args.arm} hand (ROS sourced? ROS_DOMAIN_ID?)")
        if torque and args.send:
            set_torque(node, args.arm, False)
            args.torque_off = True
        if args.teach:
            return teach_main(args, node, hand)
        if now is None:
            print("   (no /joint_states: dry run from the calibration pose itself)")
            now = target
        if torque:
            input(("the hand is limp" if args.send else "(dry run) here the hand would be limp")
                  + ": push the Dino into the jig, then Enter (Ctrl+C = torque back on, stop) > ")
            now = wait_hand(hand) or now
            if args.send:
                hold_here(hand, now)
                set_torque(node, args.arm, True)
                args.torque_off = False
                time.sleep(0.3)
                now = wait_hand(hand) or now
        close_to(hand, now, cmd_target, args.send, args.speed)
        if args.send:
            time.sleep(SETTLE_S)
        worst = hand_check(setup, args.arm, dict(zip(hand.names, hand.current() or target)))
        if worst is None:
            return
        if worst > HAND_STOP_DEG:
            print(f"NOT mounted: ring / little finger {worst:.1f} deg off the calibration (stop > {HAND_STOP_DEG:.0f}). "
                  "Is the Dino all the way in the jig? Run mount again. If it never gets there, the hand can't make "
                  "the calibration pose: teach one it can (mount --teach --send) and calibrate with it.")
        else:
            print(f"mounted: ring / little finger within {worst:.1f} deg of the calibration"
                  + (" (check: > warn)" if worst > HAND_WARN_DEG else "") + ". Now: jenga_autoscan calib --send --write, then run --send")
    except (KeyboardInterrupt, EOFError):
        print("\nstopped")
    except RuntimeError as e:
        print(f"\nSTOPPED: {e}")
    finally:
        if args.torque_off:
            try:
                set_torque(node, args.arm, True)
            except RuntimeError as e:
                print(f"   WARNING: {e}: switch the torque on by hand")
        stop_node(node)
