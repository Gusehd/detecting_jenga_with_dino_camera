"""Put the Dino into the hand jig: torque off, the user pushes the Dino in, Enter, the fingers close to the
calibration pose (dino.calibration.<arm>.hand_q of scan_setup.json) and hold it. Once per demo day:

    ros2 run jenga_vision jenga_autoscan mount            # dry run: what it would do
    ros2 run jenga_vision jenga_autoscan mount --send     # real hand

The torque service (std_srvs/SetBool, the hand driver's ~/set_torque_enabled) switches the whole hand, so all
20 joints go limp and all of them go to hand_q. Before the torque comes back the hand's current pose is sent
as the command, so the fingers hold where the user left them and then close slowly from there.
Ctrl+C at any point switches the torque back on.

Also used by `jenga_autoscan run --fix-hand`: close the fingers back to hand_q (no torque change) when the
ring / little finger is off before a scan.
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


def mount_main(args) -> None:
    from jenga_vision.arm_slider import start_node, stop_node
    from jenga_vision.autoscan import HAND_STOP_DEG, HAND_WARN_DEG, hand_check

    target = calibration_hand_q(args.arm)
    setup = json.loads(SETUP_FILE.read_text())
    pose_name = setup["dino"]["calibration"][args.arm].get("hand_pose", "hand_q")
    torque = not args.no_torque
    print(f"Dino mount, {args.arm} hand: " + ("torque off -> Dino in -> Enter -> " if torque else "")
          + f"fingers close to '{pose_name}' (the calibration)")
    if not args.send:
        print("DRY RUN (add --send to really do it): nothing is published, no torque change")
    node = start_node("jenga_mount")
    hand = hand_node(args.arm, args.send, node)
    torque_off = False
    try:
        now = wait_hand(hand)
        if now is None:
            if args.send:
                raise RuntimeError(f"no /joint_states of the {args.arm} hand (ROS sourced? ROS_DOMAIN_ID?)")
            print("   (no /joint_states: dry run from the calibration pose itself)")
            now = target
        if torque:
            if args.send:
                set_torque(node, args.arm, False)
                torque_off = True
            input(("the hand is limp" if args.send else "(dry run) here the hand would be limp")
                  + ": push the Dino into the jig, then Enter (Ctrl+C = torque back on, stop) > ")
            now = wait_hand(hand) or now
            if args.send:
                for _ in range(5):  # hold where the user left the fingers, whatever was commanded before
                    hand.publish([hand_slider.clamp(i, v) for i, v in enumerate(now)])
                    time.sleep(0.02)
                set_torque(node, args.arm, True)
                torque_off = False
                time.sleep(0.3)
                now = wait_hand(hand) or now
        close_to(hand, now, target, args.send, args.speed)
        if args.send:
            time.sleep(SETTLE_S)
        worst = hand_check(setup, args.arm, dict(zip(hand.names, hand.current() or target)))
        if worst is None:
            return
        if worst > HAND_STOP_DEG:
            print(f"NOT mounted: ring / little finger {worst:.1f} deg off the calibration (stop > {HAND_STOP_DEG:.0f}). "
                  "Is the Dino all the way in the jig? Run mount again.")
        else:
            print(f"mounted: ring / little finger within {worst:.1f} deg of the calibration"
                  + (" (check: > warn)" if worst > HAND_WARN_DEG else "") + ". Now: jenga_autoscan run --send")
    except (KeyboardInterrupt, EOFError):
        print("\nstopped")
    except RuntimeError as e:
        print(f"\nSTOPPED: {e}")
    finally:
        if torque_off:
            try:
                set_torque(node, args.arm, True)
            except RuntimeError as e:
                print(f"   WARNING: {e}: switch the torque on by hand")
        stop_node(node)
