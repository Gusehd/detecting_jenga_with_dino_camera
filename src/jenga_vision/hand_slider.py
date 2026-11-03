"""Allegro Hand V6 slider GUI: 20 joints, named poses, smooth moves.

Needs ROS 2 Humble sourced (source /opt/ros/humble/setup.bash) and the right
ROS_DOMAIN_ID. The robot side is real hardware, so by default nothing is
published (dry run): the GUI only shows the command it would send. Pass --send
to really publish.

Command: std_msgs/Float64MultiArray on
/allegro_hand_position_effort_controller_{l,r}/commands, 20 positions (rad) in
the order thumb 0-3, index 0-3, middle 0-3, ring 0-3, little 0-3
(= ahX_joint00..43). --effort E appends 20 effort values E (40 values) in case
the controller wants position + effort.

Safety:
- With --send the commanded pose starts from /joint_states; nothing is sent
  until the hand state was received (no jump from zeros).
- Sliders and poses set the target; the command moves to it at --speed deg/s,
  so loading a pose is a smooth move. STOP freezes at the current command.
- Values are clamped to the URDF limits.

Poses are saved by name in config/hand_poses.json (per hand).

Sequences (config/hand_sequences.json, shared by both hands) move a few
fingers at a time to a saved pose, one step after the other:
    {"grasp_dino": [[["ring", "little"], "grasp_dino"], [["middle"], "grasp_dino"], ...]}
A step starts when the previous one reached its target (+ STEP_PAUSE). A
finger can also be given as "thumb:01" = only joints 0 and 1 of the thumb.
STOP aborts a running sequence.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

FINGERS = ("thumb", "index", "middle", "ring", "little")
# URDF limits (rad), same for both hands, read from /robot_description 2026-10-28.
LIMITS = {
    (0, 0): (-0.0349066, 1.6580628), (0, 1): (-1.6580628, 1.6580628),
    (0, 2): (-0.1745329, 1.3089969), (0, 3): (-0.1745329, 1.3962634),
    (1, 0): (-0.3839724, 1.3089969), (2, 0): (-1.134464, 1.134464),
    (3, 0): (-1.3089969, 0.3839724), (4, 0): (-1.3089969, 0.4363323),
}
for _f in range(1, 5):
    LIMITS[(_f, 1)] = (-0.0698132, 1.5707963)
    LIMITS[(_f, 2)] = (-0.1745329, 1.3962634)
    LIMITS[(_f, 3)] = (-0.1745329, 1.3962634)
JOINTS = [(f, j) for f in range(5) for j in range(4)]
RATE_HZ = 50
POSES_FILE = Path("config/hand_poses.json")
SEQUENCES_FILE = Path("config/hand_sequences.json")
STEP_PAUSE = 0.3  # s between sequence steps


def step_joints(fingers: list[str]) -> list[int]:
    """Joint indices (0-19) of a sequence step, e.g. ["ring", "thumb:01"]."""
    out = []
    for item in fingers:
        name, _, joints = item.partition(":")
        f = FINGERS.index(name)
        out += [f * 4 + int(j) for j in (joints or "0123")]
    return out


def joint_names(hand: str) -> list[str]:
    return [f"ah{hand}_joint{f}{j}" for f, j in JOINTS]


def clamp(i: int, v: float) -> float:
    lo, hi = LIMITS[JOINTS[i]]
    return min(max(v, lo), hi)


class HandNode:
    """ROS side: joint_states subscriber and (only with send=True) the command publisher.

    Creates and spins its own node, or uses `node` (already spinning) if given.
    """

    def __init__(self, hand: str, send: bool, topic: str, node=None):
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        from jenga_vision.arm_slider import start_node

        self._msg_type = Float64MultiArray
        self._own = node is None
        self.node = start_node(f"jenga_hand_slider_{hand}") if self._own else node
        self.names = joint_names(hand)
        self.state: list[float] | None = None
        self._lock = threading.Lock()
        self.pub = self.node.create_publisher(Float64MultiArray, topic, 10) if send else None
        self.node.create_subscription(JointState, "/joint_states", self._on_state, 10)

    def _on_state(self, msg) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(n in idx for n in self.names):
            with self._lock:
                self.state = [msg.position[idx[n]] for n in self.names]

    def current(self) -> list[float] | None:
        with self._lock:
            return None if self.state is None else list(self.state)

    def publish(self, data: list[float]) -> None:
        if self.pub is not None:
            self.pub.publish(self._msg_type(data=data))

    def shutdown(self) -> None:
        if self._own:
            from jenga_vision.arm_slider import stop_node

            stop_node(self.node)


class SliderApp:
    def __init__(self, root: tk.Tk, ros: HandNode, args: argparse.Namespace):
        self.root, self.ros, self.args = root, ros, args
        self.target = [0.0] * 20
        self.command: list[float] | None = None  # last command (None = not started)
        self.vars: list[tk.DoubleVar] = []
        self._setting = False  # True while sliders are moved by code
        root.title(f"Allegro {args.hand.upper()} {'SEND' if args.send else 'DRY RUN'}")

        top = ttk.Frame(root, padding=6)
        top.pack(fill="x")
        mode = "SEND (real hardware!)" if args.send else "DRY RUN (nothing is published)"
        ttk.Label(top, text=f"hand {args.hand}  |  {mode}  |  {args.topic}",
                  foreground="red" if args.send else "gray").pack(side="left")

        grid = ttk.Frame(root, padding=6)
        grid.pack()
        for f, name in enumerate(FINGERS):
            box = ttk.LabelFrame(grid, text=f"{f} {name}", padding=4)
            box.grid(row=0, column=f, padx=3, sticky="n")
            for j in range(4):
                i = f * 4 + j
                lo, hi = (math.degrees(v) for v in LIMITS[(f, j)])
                var = tk.DoubleVar(value=0.0)
                ttk.Label(box, text=f"joint{f}{j}  [{lo:.0f}, {hi:.0f}]").pack(anchor="w")
                tk.Scale(box, variable=var, from_=hi, to=lo, resolution=0.5, length=180,
                         orient="vertical", command=lambda _v, i=i: self.on_slider(i)).pack()
                self.vars.append(var)

        bar = ttk.Frame(root, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="STOP", command=self.stop).pack(side="left")
        ttk.Button(bar, text="Sync from robot", command=self.sync).pack(side="left", padx=4)
        ttk.Label(bar, text="speed deg/s").pack(side="left", padx=(12, 2))
        self.speed = tk.DoubleVar(value=args.speed)
        ttk.Spinbox(bar, from_=1, to=120, increment=5, width=5, textvariable=self.speed).pack(side="left")

        poses = ttk.Frame(root, padding=6)
        poses.pack(fill="x")
        ttk.Label(poses, text="pose").pack(side="left")
        self.pose_name = tk.StringVar(value="grasp_dino")
        self.pose_box = ttk.Combobox(poses, textvariable=self.pose_name, width=18)
        self.pose_box.pack(side="left", padx=4)
        ttk.Button(poses, text="Go (smooth)", command=self.load_pose).pack(side="left")
        ttk.Button(poses, text="Save current target", command=self.save_pose).pack(side="left", padx=4)
        ttk.Button(poses, text="Delete", command=self.delete_pose).pack(side="left")
        self.refresh_poses()

        seq = ttk.Frame(root, padding=6)
        seq.pack(fill="x")
        ttk.Label(seq, text="sequence").pack(side="left")
        self.seq_name = tk.StringVar(value="grasp_dino")
        self.seq_box = ttk.Combobox(seq, textvariable=self.seq_name, width=18)
        self.seq_box.pack(side="left", padx=4)
        ttk.Button(seq, text="Run", command=self.run_sequence).pack(side="left")
        self.seq_box["values"] = sorted(self._sequences())
        self.running: list | None = None  # remaining steps of the running sequence
        self.step_done_at: float | None = None

        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, padding=6, font=("TkFixedFont", 9),
                  wraplength=900, justify="left").pack(fill="x")

        if not args.send:  # dry run: start from zeros unless the robot state shows up
            self.command = [0.0] * 20
        root.after(int(1000 / RATE_HZ), self.tick)

    # --- poses ---------------------------------------------------------------------
    def _all_poses(self) -> dict:
        try:
            return json.loads(POSES_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def _hand_poses(self) -> dict[str, list[float]]:
        return self._all_poses().get(self.args.hand, {})

    def refresh_poses(self) -> None:
        self.pose_box["values"] = sorted(self._hand_poses())

    def save_pose(self) -> None:
        name = self.pose_name.get().strip()
        if not name:
            return
        data = self._all_poses()
        data.setdefault(self.args.hand, {})[name] = [round(v, 5) for v in self.target]
        POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
        POSES_FILE.write_text(json.dumps(data, indent=2))
        self.refresh_poses()
        self.status.set(f"saved pose '{name}' to {POSES_FILE}")

    def delete_pose(self) -> None:
        data = self._all_poses()
        if data.get(self.args.hand, {}).pop(self.pose_name.get().strip(), None) is not None:
            POSES_FILE.write_text(json.dumps(data, indent=2))
            self.refresh_poses()

    def load_pose(self) -> None:
        pose = self._hand_poses().get(self.pose_name.get().strip())
        if pose is not None:
            self.set_target([clamp(i, v) for i, v in enumerate(pose)])

    # --- target / command ------------------------------------------------------------
    def set_target(self, values: list[float]) -> None:
        self.target = list(values)
        self._setting = True
        for var, v in zip(self.vars, values):
            var.set(round(math.degrees(v), 1))
        self._setting = False

    def on_slider(self, i: int) -> None:
        if not self._setting:
            self.target[i] = clamp(i, math.radians(self.vars[i].get()))

    # --- sequences ------------------------------------------------------------------
    def _sequences(self) -> dict:
        try:
            return json.loads(SEQUENCES_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def run_sequence(self) -> None:
        steps = self._sequences().get(self.seq_name.get().strip())
        if steps is None or self.command is None:
            return
        poses = self._hand_poses()
        missing = sorted({p for _, p in steps if p not in poses})
        if missing:
            self.status.set(f"sequence needs poses not saved for hand {self.args.hand}: {missing}")
            return
        self.running = list(steps)
        self.start_step()

    def start_step(self) -> None:
        fingers, pose = self.running.pop(0)
        values = self._hand_poses()[pose]
        target = list(self.target)
        for i in step_joints(fingers):
            target[i] = clamp(i, values[i])
        self.set_target(target)
        self.step_name = f"{'+'.join(fingers)} -> {pose}"
        self.step_done_at = None

    def advance_sequence(self, moved: bool) -> None:
        if self.running is None or moved:
            return
        now = time.monotonic()
        if self.step_done_at is None:
            self.step_done_at = now
        elif now - self.step_done_at >= STEP_PAUSE:
            if self.running:
                self.start_step()
            else:
                self.running = None

    def stop(self) -> None:
        self.running = None
        if self.command is not None:
            self.set_target(self.command)

    def sync(self) -> None:
        cur = self.ros.current()
        if cur is None:
            self.status.set("no /joint_states for this hand yet")
            return
        self.command = cur
        self.set_target([clamp(i, v) for i, v in enumerate(cur)])

    def tick(self) -> None:
        if self.command is None:
            if self.ros.current() is not None:
                self.sync()  # --send: start exactly where the hand is
            else:
                self.status.set("waiting for /joint_states of this hand ... (nothing is sent)")
        else:
            step = math.radians(max(self.speed.get(), 0.1)) / RATE_HZ
            moved = False
            for i, (c, t) in enumerate(zip(self.command, self.target)):
                d = max(-step, min(step, t - c))
                if d:
                    self.command[i] = c + d
                    moved = True
            data = list(self.command)
            if self.args.effort is not None:
                data += [self.args.effort] * 20
            if moved:
                self.ros.publish(data)
            self.advance_sequence(moved)
            sent = "sent" if self.args.send else "would send"
            step = f"   [sequence: {self.step_name}]" if self.running is not None else ""
            self.status.set(f"{sent} ({len(data)}): " + " ".join(f"{v:+.2f}" for v in data)
                            + ("   [moving]" if moved else "") + step)
        self.root.after(int(1000 / RATE_HZ), self.tick)


def _wake(root: tk.Tk) -> None:
    """Return to Python regularly so Ctrl+C is handled while Tk waits for events."""
    root.after(200, _wake, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hand", choices=("l", "r"), required=True)
    parser.add_argument("--send", action="store_true", help="really publish (default: dry run)")
    parser.add_argument("--topic", default=None,
                        help="default /allegro_hand_position_effort_controller_<hand>/commands")
    parser.add_argument("--speed", type=float, default=20.0, help="max joint speed of the command (deg/s)")
    parser.add_argument("--effort", type=float, default=None,
                        help="append 20 effort values (40-value command), if the controller wants them")
    args = parser.parse_args()
    args.topic = args.topic or f"/allegro_hand_position_effort_controller_{args.hand}/commands"
    try:
        ros = HandNode(args.hand, args.send, args.topic)
    except ImportError:
        raise SystemExit("rclpy not found: run `source /opt/ros/humble/setup.bash` first")
    root = tk.Tk()
    SliderApp(root, ros, args)
    signal.signal(signal.SIGINT, lambda *_: root.destroy())  # Ctrl+C closes the window
    _wake(root)
    try:
        root.mainloop()
    finally:
        ros.shutdown()


if __name__ == "__main__":
    main()
