"""xArm7 slider GUI: 7 joints, named poses (teaching), moves via the trajectory controller.

Needs ROS 2 Humble sourced and the right ROS_DOMAIN_ID. Moves go straight to
/xarm7_<arm>_controller/follow_joint_trajectory (no MoveIt: no planning, no
collision check), so by default nothing is sent (dry run). Pass --send to move.

How it works:
- Sliders follow the arm's pose from /joint_states (also when it is moved by
  hand in manual mode) until you touch one: then they hold your target, and
  the arm moves when you press GO. After GO (or "Robot -> sliders") they
  follow the arm again. The move time is chosen so that no joint is faster
  than --speed deg/s (at least 2 s).
- STOP cancels the goal; the controller holds where the arm is.
- "Robot -> sliders" copies the arm's actual pose into the sliders, e.g.
  after moving it by hand in manual mode (teaching).
- Poses are saved by name in config/arm_poses.json (per arm). "Save robot
  pose" stores the actual pose, "Save target" the slider values.
- Cartesian jog: each button moves the hand by the step (mm or deg) along /
  about an axis of the base or hand frame; joints come from MoveIt IK (see
  cartesian.py). With --send the arm moves right away, starting from where
  it actually is; in dry run the jog only changes the sliders (GO sends).
  The hand pose shown is the hand base link in the arm base frame.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

TWO_PI = 2 * math.pi
# URDF limits (rad) of xarm_<arm>_joint1..7, read from /robot_description 2026-10-28.
LIMITS = [(-TWO_PI, TWO_PI), (-2.059, 2.0944), (-TWO_PI, TWO_PI), (-0.19198, 3.927),
          (-TWO_PI, TWO_PI), (-1.69297, math.pi), (-TWO_PI, TWO_PI)]
MIN_MOVE_TIME = 2.0  # s
MIN_JOG_TIME = 1.0  # s
JOG_SPEED_MM = 10.0  # hand speed limits of a jog (mm/s, deg/s)
JOG_SPEED_DEG = 10.0
POSES_FILE = Path("config/arm_poses.json")


def dino_tool(arm: str):
    """Calibrated Dino frame in the hand frame (config/scan_setup.json), or None."""
    from jenga_vision.scan_plan import SETUP_FILE, arm_tool
    try:
        d = json.loads(SETUP_FILE.read_text())["dino"]
    except (OSError, KeyError, ValueError):
        return None
    return arm_tool(d, arm) if d.get("tool_hand") else None


def joint_names(arm: str) -> list[str]:
    return [f"xarm_{arm}_joint{i}" for i in range(1, 8)]


def clamp(i: int, v: float) -> float:
    lo, hi = LIMITS[i]
    return min(max(v, lo), hi)


def start_node(name: str):
    """rclpy context + node, spun in a background thread."""
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    # Our own Ctrl+C handling must still be able to send a cancel, so rclpy
    # must not shut the context down on SIGINT.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node(name)
    node._jenga_spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    node._jenga_spin.start()
    return node


def stop_node(node) -> None:
    import rclpy

    if rclpy.ok():
        rclpy.shutdown()  # stops spin first; destroying a spinning node aborts
    node._jenga_spin.join(timeout=2)
    node.destroy_node()


class ArmNode:
    """ROS side: joint_states subscriber and (only with send=True) the trajectory action client.

    Creates and spins its own node, or uses `node` (already spinning) if given.
    """

    def __init__(self, arm: str, send: bool, node_name: str = "jenga_arm_slider", node=None):
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from control_msgs.action import FollowJointTrajectory

        self._action_type = FollowJointTrajectory
        self._own = node is None
        self.node = start_node(f"{node_name}_{arm}") if self._own else node
        self.names = joint_names(arm)
        self.action_name = f"/xarm7_{arm}_controller/follow_joint_trajectory"
        self.state: list[float] | None = None
        self.result = ""  # last goal status, shown in the GUI
        self.succeeded = False  # last goal finished with SUCCEEDED
        self.done = threading.Event()  # set when the last goal has finished (or failed to start)
        self.done.set()
        self._goal = None
        self._lock = threading.Lock()
        self.client = ActionClient(self.node, FollowJointTrajectory, self.action_name) if send else None
        self.node.create_subscription(JointState, "/joint_states", self._on_state, 10)

    def _on_state(self, msg) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(n in idx for n in self.names):
            with self._lock:
                self.state = [msg.position[idx[n]] for n in self.names]

    def current(self) -> list[float] | None:
        with self._lock:
            return None if self.state is None else list(self.state)

    def move(self, target: list[float], seconds: float) -> None:
        self.move_path([target], seconds)

    def move_path(self, points: list[list[float]], seconds: float) -> None:
        """Send waypoints evenly spread over `seconds` as one trajectory."""
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint

        if self.client is None:
            return
        self.done.clear()
        self.succeeded = False
        if not self.client.server_is_ready():
            self.result = f"action server not ready: {self.action_name}"
            self.done.set()
            return
        goal = self._action_type.Goal()
        goal.trajectory.joint_names = self.names
        for i, q in enumerate(points, 1):
            t = seconds * i / len(points)
            sec = int(t)
            # Velocities must be given for all points or none; a single point stops at rest.
            goal.trajectory.points.append(JointTrajectoryPoint(
                positions=list(q), velocities=[0.0] * 7 if len(points) == 1 else [],
                time_from_start=Duration(sec=sec, nanosec=int((t - sec) * 1e9))))
        self.result = f"sent ({seconds:.1f} s) ..."
        self.client.send_goal_async(goal).add_done_callback(self._on_accepted)

    def _on_accepted(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self.result = "goal REJECTED"
            self.done.set()
            return
        self._goal = handle
        self.result = "moving ..."
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        from action_msgs.msg import GoalStatus

        status = future.result().status
        names = {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED", GoalStatus.STATUS_CANCELED: "CANCELED",
                 GoalStatus.STATUS_ABORTED: "ABORTED"}
        self.result = f"done: {names.get(status, status)} (error {future.result().result.error_code})"
        self.succeeded = status == GoalStatus.STATUS_SUCCEEDED
        self._goal = None
        self.done.set()

    def stop(self) -> None:
        if self._goal is not None:
            self._goal.cancel_goal_async()
            self.result = "cancel requested"

    def shutdown(self) -> None:
        if self._own:
            stop_node(self.node)


class ArmApp:
    def __init__(self, root: tk.Tk, ros: ArmNode, args: argparse.Namespace):
        self.root, self.ros, self.args = root, ros, args
        self.target: list[float] | None = None  # None until /joint_states arrives
        self.editing = False  # True: sliders hold the user's target; False: they follow the arm
        self.vars: list[tk.DoubleVar] = []
        self._setting = False
        root.title(f"xArm7 {args.arm} {'SEND' if args.send else 'DRY RUN'}")

        mode = "SEND (moves the arm, no collision check!)" if args.send else "DRY RUN (nothing is sent)"
        ttk.Label(root, text=f"arm {args.arm}  |  {mode}", padding=6,
                  foreground="red" if args.send else "gray").pack(anchor="w")

        grid = ttk.Frame(root, padding=6)
        grid.pack()
        self.actual_labels: list[ttk.Label] = []
        for i, (lo, hi) in enumerate(LIMITS):
            ttk.Label(grid, text=f"joint{i + 1}").grid(row=i, column=0, sticky="w")
            var = tk.DoubleVar(value=0.0)
            scale = tk.Scale(grid, variable=var, from_=math.degrees(lo), to=math.degrees(hi), resolution=0.1,
                             length=520, orient="horizontal", showvalue=True,
                             command=lambda _v, i=i: self.on_slider(i))
            scale.grid(row=i, column=1)
            scale.bind("<ButtonPress>", self._start_edit)
            spin = ttk.Spinbox(grid, textvariable=var, width=8, increment=0.5,
                               from_=math.degrees(lo), to=math.degrees(hi), command=lambda i=i: self.on_slider(i))
            spin.grid(row=i, column=2, padx=4)
            spin.bind("<Return>", lambda _e, i=i: self.on_slider(i))
            spin.bind("<Key>", self._start_edit)  # typing must not be overwritten by the robot pose
            spin.bind("<ButtonPress>", self._start_edit)
            lab = ttk.Label(grid, text="", width=22, font=("TkFixedFont", 9))
            lab.grid(row=i, column=3, sticky="w")
            self.vars.append(var)
            self.actual_labels.append(lab)

        bar = ttk.Frame(root, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="GO", command=self.go).pack(side="left")
        ttk.Button(bar, text="STOP", command=self.ros.stop).pack(side="left", padx=4)
        ttk.Button(bar, text="Robot -> sliders", command=self.sync).pack(side="left", padx=4)
        ttk.Label(bar, text="max speed deg/s").pack(side="left", padx=(12, 2))
        self.speed = tk.DoubleVar(value=args.speed)
        ttk.Spinbox(bar, from_=1, to=60, increment=1, width=5, textvariable=self.speed).pack(side="left")

        poses = ttk.Frame(root, padding=6)
        poses.pack(fill="x")
        ttk.Label(poses, text="pose").pack(side="left")
        self.pose_name = tk.StringVar(value="home")
        self.pose_box = ttk.Combobox(poses, textvariable=self.pose_name, width=20)
        self.pose_box.pack(side="left", padx=4)
        ttk.Button(poses, text="Load -> sliders", command=self.load_pose).pack(side="left")
        ttk.Button(poses, text="Save robot pose", command=lambda: self.save_pose(robot=True)).pack(side="left", padx=4)
        ttk.Button(poses, text="Save target", command=lambda: self.save_pose(robot=False)).pack(side="left")
        ttk.Button(poses, text="Delete", command=self.delete_pose).pack(side="left", padx=4)
        self.refresh_poses()

        self._build_jog(root)

        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, padding=6, font=("TkFixedFont", 9),
                  wraplength=900, justify="left").pack(fill="x")
        root.after(100, self.tick)

    # --- poses ---------------------------------------------------------------------
    def _all_poses(self) -> dict:
        try:
            return json.loads(POSES_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def _arm_poses(self) -> dict[str, list[float]]:
        return self._all_poses().get(self.args.arm, {})

    def refresh_poses(self) -> None:
        self.pose_box["values"] = sorted(self._arm_poses())

    def save_pose(self, robot: bool) -> None:
        name = self.pose_name.get().strip()
        values = self.ros.current() if robot else self.target
        if not name or values is None:
            return
        data = self._all_poses()
        data.setdefault(self.args.arm, {})[name] = [round(v, 5) for v in values]
        POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
        POSES_FILE.write_text(json.dumps(data, indent=2))
        self.refresh_poses()
        self.message = f"saved {'robot' if robot else 'target'} pose '{name}' to {POSES_FILE}"

    def delete_pose(self) -> None:
        data = self._all_poses()
        if data.get(self.args.arm, {}).pop(self.pose_name.get().strip(), None) is not None:
            POSES_FILE.write_text(json.dumps(data, indent=2))
            self.refresh_poses()

    def load_pose(self) -> None:
        pose = self._arm_poses().get(self.pose_name.get().strip())
        if pose is not None and self.target is not None:
            self.set_target([clamp(i, v) for i, v in enumerate(pose)])
            self.editing = True

    # --- Cartesian jog -----------------------------------------------------------------
    def _build_jog(self, root: tk.Tk) -> None:
        from jenga_vision.cartesian import Cartesian

        self.cart = Cartesian(self.ros.node, self.args.arm, dino_tool(self.args.arm))
        self._jogging = False
        self._jog_target: list[float] | None = None
        self.pose_text = "hand pose: waiting for /compute_fk ..."

        box = ttk.LabelFrame(root, text="Cartesian jog (MoveIt IK)", padding=6)
        box.pack(fill="x", padx=6)
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="frame").pack(side="left")
        self.jog_frame = tk.StringVar(value="hand")
        for f in ("hand", "base") + (("dino",) if self.cart.tool is not None else ()):
            ttk.Radiobutton(row, text=f, value=f, variable=self.jog_frame).pack(side="left")
        ttk.Label(row, text="step mm").pack(side="left", padx=(12, 2))
        self.step_mm = tk.DoubleVar(value=2.0)
        ttk.Spinbox(row, from_=0.1, to=50, increment=0.5, width=5, textvariable=self.step_mm).pack(side="left")
        ttk.Label(row, text="step deg").pack(side="left", padx=(12, 2))
        self.step_deg = tk.DoubleVar(value=2.0)
        ttk.Spinbox(row, from_=0.1, to=30, increment=0.5, width=5, textvariable=self.step_deg).pack(side="left")
        for kind, names in (("t", ("X", "Y", "Z")), ("r", ("Rx", "Ry", "Rz"))):
            buttons = ttk.Frame(box)
            buttons.pack(fill="x", pady=2)
            for axis, name in enumerate(names):
                for sign in (-1, 1):
                    ttk.Button(buttons, text=f"{name}{'+' if sign > 0 else '-'}", width=5,
                               command=lambda k=kind, a=axis, s=sign: self.jog(k, a, s)).pack(side="left", padx=1)
                ttk.Label(buttons, text=" ").pack(side="left")
        self.pose_label = ttk.Label(box, text="", font=("TkFixedFont", 9))
        self.pose_label.pack(anchor="w")
        threading.Thread(target=self._poll_pose, daemon=True).start()

    def _poll_pose(self) -> None:
        import time

        from jenga_vision.cartesian import JogError, rpy_deg

        while True:
            time.sleep(0.5)
            cur = self.ros.current()
            if cur is None or not self.cart.ready():
                self.pose_text = "hand pose: /compute_fk not available (move_group running?)"
                continue
            try:
                pos, rot = self.cart.fk(cur)
            except JogError as e:
                self.pose_text = f"hand pose: {e}"
                continue
            r, p, y = rpy_deg(rot)
            self.pose_text = (f"hand pose ({self.cart.base}): x {pos[0] * 1000:8.1f}  y {pos[1] * 1000:8.1f}  "
                              f"z {pos[2] * 1000:8.1f} mm   roll {r:7.1f}  pitch {p:7.1f}  yaw {y:7.1f} deg")

    def jog(self, kind: str, axis: int, sign: int) -> None:
        if self._jogging or self._jog_target is not None or not self.ros.done.is_set():
            self.message = "busy: wait for the current move to finish"
            return
        # Dry run builds on the sliders; a real move starts from the actual pose, so a
        # pose changed by hand (manual mode) never makes the arm jump back.
        start = self.ros.current() if self.args.send else self.target
        if start is None:
            self.message = "no /joint_states for this arm yet"
            return
        try:
            step = self.step_mm.get() / 1000 if kind == "t" else math.radians(self.step_deg.get())
        except (tk.TclError, ValueError):
            return
        self._jogging = True
        self.message = "jog: solving IK ..."
        threading.Thread(target=self._jog_worker, args=(list(start), kind, axis, sign * step, self.jog_frame.get()),
                         daemon=True).start()

    def _jog_worker(self, start: list[float], kind: str, axis: int, amount: float, frame: str) -> None:
        from jenga_vision.cartesian import JogError

        name = ("X", "Y", "Z")[axis] if kind == "t" else ("Rx", "Ry", "Rz")[axis]
        size = f"{amount * 1000:+.1f} mm" if kind == "t" else f"{math.degrees(amount):+.1f} deg"
        try:
            path = self.cart.plan_jog(start, kind, axis, amount, frame)
        except JogError as e:
            self.message = f"jog {name} {size} ({frame}) REFUSED: {e}"
            self._jogging = False
            return
        if any(abs(clamp(i, v) - v) > 1e-6 for i, v in enumerate(path[-1])):
            self.message = f"jog {name} {size} REFUSED: joint limit"
            self._jogging = False
            return
        change = max(abs(math.degrees(a - b)) for a, b in zip(path[-1], start))
        hand = abs(amount) * 1000 / JOG_SPEED_MM if kind == "t" else abs(math.degrees(amount)) / JOG_SPEED_DEG
        seconds = max(MIN_JOG_TIME, hand, change / max(self.speed.get(), 0.1))
        if self.args.send:
            self.ros.move_path(path, seconds)
            self.message = f"jog {name} {size} ({frame}): {len(path)} steps, {seconds:.1f} s, max joint {change:.1f} deg"
        else:
            self.message = (f"DRY RUN jog {name} {size} ({frame}): IK ok, {len(path)} steps, "
                            f"max joint {change:.1f} deg -> sliders (GO would send)")
            self._jog_target = path[-1]  # sliders hold it (tick); with --send they follow the arm
        self._jogging = False

    # --- target / moves --------------------------------------------------------------
    message = ""

    def set_target(self, values: list[float]) -> None:
        self.target = list(values)
        self._setting = True
        for var, v in zip(self.vars, values):
            var.set(round(math.degrees(v), 1))
        self._setting = False

    def _start_edit(self, _event=None) -> None:
        if self.target is not None:
            self.editing = True

    def on_slider(self, i: int) -> None:
        if self._setting or self.target is None:
            return
        self.editing = True
        try:
            self.target[i] = clamp(i, math.radians(float(self.vars[i].get())))
        except (tk.TclError, ValueError):
            pass  # half-typed spinbox value

    def sync(self) -> None:
        cur = self.ros.current()
        if cur is not None:
            self.set_target([clamp(i, v) for i, v in enumerate(cur)])
            self.editing = False

    def move_time(self, cur: list[float]) -> float:
        biggest = max(abs(t - c) for t, c in zip(self.target, cur))
        return max(MIN_MOVE_TIME, math.degrees(biggest) / max(self.speed.get(), 0.1))

    def go(self) -> None:
        cur = self.ros.current()
        if cur is None or self.target is None:
            self.message = "no /joint_states for this arm yet"
            return
        seconds = self.move_time(cur)
        if self.args.send:
            self.ros.move(self.target, seconds)
            self.editing = False  # follow the arm while it moves
            self.message = ""
        else:
            self.message = f"DRY RUN: would move in {seconds:.1f} s to " + \
                " ".join(f"{math.degrees(v):.1f}" for v in self.target)

    def tick(self) -> None:
        cur = self.ros.current()
        if cur is None:
            self.status.set("waiting for /joint_states of this arm ... (nothing is sent)")
        else:
            if self._jog_target is not None:
                self.set_target([clamp(i, v) for i, v in enumerate(self._jog_target)])
                self.editing = True
                self._jog_target = None
            if self.target is None or not self.editing:
                self.sync()
            for lab, c, t in zip(self.actual_labels, cur, self.target):
                lab.config(text=f"robot {math.degrees(c):7.1f}  d {math.degrees(t - c):+6.1f}")
            self.pose_label.config(text=self.pose_text)
            lines = ["sliders EDITED: GO moves there, 'Robot -> sliders' cancels" if self.editing
                     else "sliders follow the robot", f"move time {self.move_time(cur):.1f} s"]
            if self.ros.result:
                lines.append(self.ros.result)
            if self.message:
                lines.append(self.message)
            self.status.set("   |   ".join(lines))
        self.root.after(100, self.tick)


def _wake(root: tk.Tk) -> None:
    """Return to Python regularly so Ctrl+C is handled while Tk waits for events."""
    root.after(200, _wake, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--send", action="store_true", help="really move the arm (default: dry run)")
    parser.add_argument("--speed", type=float, default=10.0, help="max joint speed of a move (deg/s)")
    args = parser.parse_args()
    try:
        ros = ArmNode(args.arm, args.send)
    except ImportError:
        raise SystemExit("rclpy not found: run `source /opt/ros/humble/setup.bash` first")
    root = tk.Tk()
    ArmApp(root, ros, args)
    signal.signal(signal.SIGINT, lambda *_: root.destroy())  # Ctrl+C closes the window
    _wake(root)
    try:
        root.mainloop()
    finally:
        ros.shutdown()


if __name__ == "__main__":
    main()
