"""Run a task: arm moves and hand moves, one step after the other.

Tasks are in config/tasks.json and use the saved poses of jenga-arm
(config/arm_poses.json), jenga-hand (config/hand_poses.json) and the hand
sequences (config/hand_sequences.json):

    "pick_dino": {"arm": "right", "hand": "r", "steps": [
        {"arm": "home"},                    move the arm to a saved arm pose
        {"hand": "initial_1"},              move all fingers to a saved hand pose
        {"hand_seq": "grasp_middle_1"},     run a hand sequence (finger by finger)
        {"arm": "final_grasp_1", "speed": 5},   per-step speed (deg/s) overrides
        {"route": "auto_scan_right_my"},    move through a jenga-route route (config/arm_routes.json)
        {"task": "pick_dino"},              all steps of another task (same arm and hand)
        ...]}

Each step starts when the previous one has finished. Arm moves go straight to
/xarm7_<arm>_controller/follow_joint_trajectory (no planning, no collision
check), joint-interpolated, no joint faster than --arm-speed deg/s (at least
2 s). The hand command moves at --hand-speed deg/s like jenga-hand.

By default nothing is sent (dry run: the plan with move times is printed).
--send really moves; --step waits for Enter before every step (and every route
point); --from N starts at step N. Ctrl+C cancels the arm move (the arm holds) and stops the hand
command where it is.

Don't run jenga-hand / jenga-arm with --send at the same time: their sliders
don't know what this script moved (close them, or Sync before using them).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from jenga_vision import arm_route, arm_slider, hand_slider

TASKS_FILE = Path("config/tasks.json")
STEP_KINDS = ("arm", "hand", "hand_seq", "route")


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError:
        return {}


def describe(step: dict) -> str:
    kind = next(k for k in STEP_KINDS if k in step)
    extra = f"  (speed {step['speed']})" if "speed" in step else ""
    return f"{kind:8s} {step[kind]}{extra}"


def expand(task: dict, tasks: dict, depth: int = 0) -> dict:
    """Replace {"task": name} steps by that task's steps."""
    if depth > 5:
        raise SystemExit("tasks include each other too deeply")
    steps = []
    for step in task["steps"]:
        if "task" in step:
            sub = tasks.get(step["task"])
            if sub is None:
                raise SystemExit(f"task '{step['task']}' not found in {TASKS_FILE}")
            if (sub["arm"], sub["hand"]) != (task["arm"], task["hand"]):
                raise SystemExit(f"task '{step['task']}' uses another arm/hand")
            steps += expand(sub, tasks, depth + 1)["steps"]
        else:
            steps.append(step)
    return dict(task, steps=steps)


def check_task(task: dict, arm_poses: dict, hand_poses: dict, sequences: dict, routes: dict) -> list[str]:
    errors = []
    for n, step in enumerate(task["steps"], 1):
        kinds = [k for k in STEP_KINDS if k in step]
        if len(kinds) != 1:
            errors.append(f"step {n}: needs exactly one of {' / '.join(STEP_KINDS)}: {step}")
            continue
        name = step[kinds[0]]
        if kinds[0] == "route":
            if not routes.get(name):
                errors.append(f"step {n}: route '{name}' not saved for the {task['arm']} arm")
            elif any(abs(arm_slider.clamp(i, v) - v) > 1e-6 for p in routes[name] for i, v in enumerate(p["q"])):
                errors.append(f"step {n}: route '{name}' has points outside the joint limits")
        elif kinds[0] == "arm":
            if name not in arm_poses:
                errors.append(f"step {n}: arm pose '{name}' not saved for the {task['arm']} arm")
            elif any(abs(arm_slider.clamp(i, v) - v) > 1e-6 for i, v in enumerate(arm_poses[name])):
                errors.append(f"step {n}: arm pose '{name}' is outside the joint limits")
        elif kinds[0] == "hand" and name not in hand_poses:
            errors.append(f"step {n}: hand pose '{name}' not saved for hand {task['hand']}")
        elif kinds[0] == "hand_seq":
            if name not in sequences:
                errors.append(f"step {n}: hand sequence '{name}' not found")
            else:
                missing = sorted({p for _, p in sequences[name] if p not in hand_poses})
                if missing:
                    errors.append(f"step {n}: sequence '{name}' needs hand poses {missing}")
    return errors


class Runner:
    def __init__(self, task: dict, args: argparse.Namespace):
        self.task, self.args = task, args
        self.arm_poses = load_json(arm_slider.POSES_FILE).get(task["arm"], {})
        self.hand_poses = load_json(hand_slider.POSES_FILE).get(task["hand"], {})
        self.sequences = load_json(hand_slider.SEQUENCES_FILE)
        self.routes = arm_route.load_all().get(task["arm"], {})
        errors = check_task(task, self.arm_poses, self.hand_poses, self.sequences, self.routes)
        if errors:
            raise SystemExit("task can't run:\n  " + "\n  ".join(errors))

        self.node = arm_slider.start_node("jenga_task")
        self.arm = arm_slider.ArmNode(task["arm"], args.send, node=self.node)
        topic = f"/allegro_hand_position_effort_controller_{task['hand']}/commands"
        self.hand = hand_slider.HandNode(task["hand"], args.send, topic, node=self.node)
        # Planned poses, so a dry run can time every step without the robot.
        self.arm_q: list[float] | None = None
        self.hand_cmd: list[float] | None = None

    def wait_states(self) -> None:
        for _ in range(30):
            if self.arm.current() is not None and self.hand.current() is not None:
                break
            time.sleep(0.1)
        self.arm_q, self.hand_cmd = self.arm.current(), self.hand.current()
        if self.args.send:
            if self.arm_q is None or self.hand_cmd is None:
                raise SystemExit("no /joint_states for the arm and hand (ROS sourced? robot PC connected?)")
            if not self.arm.client.wait_for_server(timeout_sec=5):
                raise SystemExit(f"action server not available: {self.arm.action_name}")
            # The hand command starts exactly where the hand is (no jump).
            self.hand_cmd = [hand_slider.clamp(i, v) for i, v in enumerate(self.hand_cmd)]
        elif self.arm_q is None or self.hand_cmd is None:
            print("(no /joint_states: dry run times the first moves from the first saved poses)")

    # --- arm ---------------------------------------------------------------------------
    def arm_move(self, target: list[float], speed: float) -> None:
        start = self.arm.current() if self.args.send else self.arm_q
        start = start or target
        biggest = max(abs(math.degrees(t - c)) for t, c in zip(target, start))
        seconds = max(arm_slider.MIN_MOVE_TIME, biggest / max(speed, 0.1))
        print(f"   largest joint change {biggest:.1f} deg, {seconds:.1f} s")
        self.arm_q = list(target)
        if not self.args.send:
            return
        self.arm.move(target, seconds)
        try:
            while not self.arm.done.wait(0.1):
                pass
        except KeyboardInterrupt:
            self.arm.stop()
            self.arm.done.wait(3)
            raise
        if not self.arm.succeeded:
            raise RuntimeError(f"arm move failed: {self.arm.result}")
        cur = self.arm.current()
        err = max(abs(math.degrees(t - c)) for t, c in zip(target, cur))
        print(f"   reached ({self.arm.result}), largest joint error {err:.2f} deg")

    # --- hand --------------------------------------------------------------------------
    def hand_move(self, values: list[float], joints: list[int], speed: float) -> None:
        if self.hand_cmd is None:  # dry run without robot state
            self.hand_cmd = [hand_slider.clamp(i, v) for i, v in enumerate(values)]
        target = list(self.hand_cmd)
        for i in joints:
            target[i] = hand_slider.clamp(i, values[i])
        biggest = max(abs(math.degrees(t - c)) for t, c in zip(target, self.hand_cmd))
        print(f"   largest finger joint change {biggest:.1f} deg, {biggest / max(speed, 0.1):.1f} s")
        if not self.args.send:
            self.hand_cmd = target
            return
        step = math.radians(max(speed, 0.1)) / hand_slider.RATE_HZ
        while True:
            moved = False
            for i, (c, t) in enumerate(zip(self.hand_cmd, target)):
                d = max(-step, min(step, t - c))
                if d:
                    self.hand_cmd[i] = c + d
                    moved = True
            if not moved:
                break
            self.hand.publish(list(self.hand_cmd))
            time.sleep(1 / hand_slider.RATE_HZ)

    def hand_pose(self, name: str, speed: float) -> None:
        self.hand_move(self.hand_poses[name], list(range(20)), speed)

    def hand_sequence(self, name: str, speed: float) -> None:
        for fingers, pose in self.sequences[name]:
            print(f"   {'+'.join(fingers)} -> {pose}")
            self.hand_move(self.hand_poses[pose], hand_slider.step_joints(fingers), speed)
            if self.args.send:
                time.sleep(hand_slider.STEP_PAUSE)

    # --- run ---------------------------------------------------------------------------
    def run(self) -> None:
        steps = self.task["steps"]
        mode = "SEND" if self.args.send else "DRY RUN (nothing is sent)"
        pace = "Enter before every step" if self.args.step else "continuous"
        print(f"task: {self.task.get('note', '')}\n{len(steps)} steps, arm {self.args.arm_speed} deg/s, "
              f"hand {self.args.hand_speed} deg/s, {pace}, {mode}")
        self.wait_states()
        if self.args.start > 1:
            print(f"starting at step {self.args.start}: the first move starts from where the robot is")
        for n, step in enumerate(steps, 1):
            if n < self.args.start:
                continue
            print(f"\n[{n}/{len(steps)}] {describe(step)}")
            if self.args.step and input("   Enter = go, q = quit > ").strip().lower() == "q":
                print("quit")
                return
            if "arm" in step:
                self.arm_move(self.arm_poses[step["arm"]], step.get("speed", self.args.arm_speed))
            elif "route" in step:
                points = self.routes[step["route"]]
                for k, point in enumerate(points, 1):
                    print(f"   route point {k}/{len(points)}: {'via ' if point['via'] else 'shot'} {point['label']}")
                    if self.args.step and k > 1 and input("   Enter = go, q = quit > ").strip().lower() == "q":
                        print("quit")
                        return
                    self.arm_move(point["q"], step.get("speed", self.args.arm_speed))
            elif "hand" in step:
                self.hand_pose(step["hand"], step.get("speed", self.args.hand_speed))
            else:
                self.hand_sequence(step["hand_seq"], step.get("speed", self.args.hand_speed))
        print("\ntask done")

    def shutdown(self) -> None:
        arm_slider.stop_node(self.node)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?", help="task name in config/tasks.json (omit to list)")
    parser.add_argument("--send", action="store_true", help="really move (default: dry run)")
    parser.add_argument("--arm-speed", type=float, default=10.0, help="max arm joint speed (deg/s), default 10")
    parser.add_argument("--hand-speed", type=float, default=20.0, help="max finger joint speed (deg/s), default 20")
    parser.add_argument("--step", action="store_true", help="wait for Enter before every step")
    parser.add_argument("--from", dest="start", type=int, default=1, help="start at step N (1-based)")
    args = parser.parse_args()

    tasks = load_json(TASKS_FILE)
    if not args.task:
        for name, task in tasks.items():
            print(f"{name}: {task.get('note', '')}")
            for n, step in enumerate(task["steps"], 1):
                print(f"  {n:2d} {describe(step)}")
        return
    if args.task not in tasks:
        raise SystemExit(f"no task '{args.task}' in {TASKS_FILE} (have: {', '.join(tasks)})")
    try:
        runner = Runner(expand(tasks[args.task], tasks), args)
    except ImportError:
        raise SystemExit("rclpy not found: run `source /opt/ros/humble/setup.bash` first")
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\nCtrl+C: arm move cancelled (arm holds), hand command stopped where it was")
    except (RuntimeError, EOFError) as e:
        print(f"\nstopped: {e}")
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
