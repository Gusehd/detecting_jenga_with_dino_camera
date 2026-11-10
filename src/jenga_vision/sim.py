"""Virtual robot + RViz view: try tasks/routes without the real hardware.

One process on a ROS domain other than 0 (0 = real xArm7 + Allegro):
  * fake controllers: both arms' follow_joint_trajectory (moves are interpolated
    in time) and both hands' position commands (applied at once),
  * /joint_states for all of them (start: right `home`, left `auto_home`),
  * /compute_fk and /compute_ik with the real URDF kinematics (kinematics.py),
    so the Cartesian jog in jenga-arm behaves like on the robot,
  * /jenga_sim/markers: tower blocks, arm skeletons, a palm box, the Dino (guess
    from config/scan_setup.json), the end faces to shoot; the Dino turns red
    when it gets closer than --warn-mm to the tower box (no mesh collision),
  * rviz2 with config/sim.rviz (--no-rviz to skip).

The URDF meshes (xarm_description, allegro_hand_plexus_description) are not
installed here, so the robot is drawn as lines and boxes.

    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=77
    uv run jenga-sim                                  # terminal 1
    uv run jenga-task scan_right_draft --send         # terminal 2, moves the virtual robot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from jenga_vision import hand_slider
from jenga_vision.arm_slider import POSES_FILE, joint_names
from jenga_vision.cartesian import matrix_to_quat, quat_to_matrix
from jenga_vision.kinematics import URDF, Chain, axis_angle, load_joints, transform
from jenga_vision.scan_plan import FACE_NORMALS, SETUP_FILE, Tower, arm_tool

RVIZ_FILE = Path("config/sim.rviz")
ROBOT_DOMAINS = ("0", "42")  # real xArm7 + Allegro (42 since 2026-11-09): the fake controllers would answer there
ARMS = ("right", "left")
START_POSES = {"right": "home", "left": "auto_home"}
START_HAND = "initial_1"
DINO_LEN, DINO_R = 0.105, 0.016  # AM3111 body behind the tip


def link_in_world(link: str) -> np.ndarray:
    """4x4 of a link reached through fixed joints only (e.g. xarm_right_link_base)."""
    joints = load_joints(URDF)
    t = np.eye(4)
    while link in joints:
        j = joints[link]
        if j.kind != "fixed":
            raise ValueError(f"{link} is behind a moving joint")
        t = j.origin @ t
        link = j.parent
    return t


def joint_points(chain: Chain, q) -> list[np.ndarray]:
    """World positions of every moving joint and the tip, base first."""
    t, i, pts = np.eye(4), 0, []
    for j in chain.links:
        t = t @ j.origin
        if j.kind != "fixed":
            pts.append(t[:3, 3].copy())
            t = t @ transform(axis_angle(j.axis, q[i]), [0, 0, 0])
            i += 1
    pts.append(t[:3, 3].copy())
    return pts


PALM_LO, PALM_HI = np.array([-0.025, -0.055, 0.0]), np.array([0.02, 0.055, 0.095])  # hand frame, rough


def finger_lines(arm: str, hq) -> list[np.ndarray]:
    """Per URDF finger (0 thumb .. 3 ring), the link origins in the hand frame for the
    20 hand values `hq` (finger*4 + joint). The 10/15 URDF has no little finger."""
    joints = load_joints(URDF)
    h = arm[0]
    base = f"ah{h}_base_link"
    memo = {base: np.eye(4)}

    def pose(link):
        if link not in memo:
            j = joints[link]
            t = pose(j.parent) @ j.origin
            if j.kind != "fixed":
                f, k = int(j.name[-2]), int(j.name[-1])
                t = t @ transform(axis_angle(j.axis, hq[f * 4 + k]), [0, 0, 0])
            memo[link] = t
        return memo[link]

    return [np.array([pose(f"ah{h}_link_{f}{k}")[:3, 3] for k in range(6) if f"ah{h}_link_{f}{k}" in joints])
            for f in range(4)]


def dino_frame(setup: dict, arm: str) -> np.ndarray:
    """Physical Dino tip frame in the hand frame (the calibrated tool is the point in
    focus; the tip is `tip_to_focus_m` behind it along the optical axis)."""
    d = setup["dino"]
    t = arm_tool(d, arm)
    if d.get("tool_hand"):
        t = t @ transform(np.eye(3), [0, 0, -d.get("tip_to_focus_m", 0.003)])
    return t


FINGER_R = 0.012  # finger link radius for the clearance check


def geometry(chain: Chain, tool: np.ndarray, arm: str, q, hq, dino: bool = True) -> dict:
    """World geometry of one arm: hand 4x4, palm corners, finger polylines, Dino frame."""
    hand = chain.fk(q)
    grid = np.array(np.meshgrid(*[[0, 0.5, 1]] * 3)).T.reshape(-1, 3)
    palm = (PALM_LO + grid * (PALM_HI - PALM_LO)) @ hand[:3, :3].T + hand[:3, 3]
    fingers = [line @ hand[:3, :3].T + hand[:3, 3] for line in finger_lines(arm, hq)]
    g = {"hand": hand, "palm": palm, "fingers": fingers}
    if dino:
        g["dino"] = hand @ tool
    return g


def tower_distance_mm(tower: Tower, pts: np.ndarray) -> float:
    """Smallest distance (mm) of points to the tower box (whole tower as 1 box)."""
    t = tower.t
    local = (np.asarray(pts) - tower.o) @ tower.r
    half = np.array([t["block_len"] / 2, t["block_len"] / 2, t["layers"] * t["layer_h"] / 2])
    d = np.maximum(np.abs(local - [0, 0, half[2]]) - half, 0)
    return float(np.linalg.norm(d, axis=1).min() * 1000)


def clearances(tower: Tower, g: dict) -> dict[str, float]:
    """mm from the tower box per part (surfaces: finger/Dino radius subtracted)."""
    fingers = np.concatenate([np.linspace(a, b, 4) for line in g["fingers"] for a, b in zip(line, line[1:])])
    c = {"palm": tower_distance_mm(tower, g["palm"]),
         "fingers": tower_distance_mm(tower, fingers) - FINGER_R * 1000}
    if "dino" in g:
        d = g["dino"]
        c["dino_tip"] = tower_distance_mm(tower, d[None, :3, 3])
        c["dino"] = tower_distance_mm(tower, [d[:3, 3] - d[:3, 2] * s for s in np.linspace(DINO_R, DINO_LEN, 8)]) - DINO_R * 1000
    return c


class Sim:
    def __init__(self, args):
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.node import Node

        self.args = args
        rclpy.init()
        self.node = Node("jenga_sim")
        self.cg = ReentrantCallbackGroup()
        poses = json.loads(POSES_FILE.read_text())
        setup = json.loads(SETUP_FILE.read_text())
        self.tower = Tower(setup["tower"], setup.get("aim", 0.5))
        self.setup = setup
        self.chains = {a: Chain(a) for a in ARMS}
        self.base = {a: link_in_world(f"xarm_{a}_link_base") for a in ARMS}
        self.tool = {a: dino_frame(setup, a) for a in ARMS}
        self.lock = threading.Lock()
        self.q = {a: list(poses.get(a, {}).get(START_POSES[a], [0.0] * 7)) for a in ARMS}
        hand_poses = json.loads(hand_slider.POSES_FILE.read_text())
        self.hq = {h: list(hand_poses.get(h, {}).get(START_HAND, [0.0] * 20)) for h in "rl"}
        self.dino_arms = set(args.dino)
        self.min_mm = {a: math.inf for a in ARMS}
        self.warned = {a: False for a in ARMS}
        self._ros()

    # --- fake robot -------------------------------------------------------------
    def _ros(self):
        from control_msgs.action import FollowJointTrajectory
        from moveit_msgs.srv import GetPositionFK, GetPositionIK
        from rclpy.action import ActionServer, CancelResponse
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        from visualization_msgs.msg import MarkerArray
        from tf2_ros import StaticTransformBroadcaster
        from geometry_msgs.msg import TransformStamped

        n = self.node
        self.js_pub = n.create_publisher(JointState, "/joint_states", 10)
        self.mk_pub = n.create_publisher(MarkerArray, "/jenga_sim/markers", 1)
        self.JointState = JointState
        names = sum((joint_names(a) for a in ARMS), []) + sum((hand_slider.joint_names(h) for h in "rl"), [])
        self.names = names
        n.create_timer(0.05, self._publish_state)
        n.create_timer(0.1, self._publish_markers)
        for h in "rl":
            n.create_subscription(Float64MultiArray, f"/allegro_hand_position_effort_controller_{h}/commands",
                                  lambda m, h=h: self.hq.__setitem__(h, list(m.data)[:20]), 10)
        for a in ARMS:
            ActionServer(n, FollowJointTrajectory, f"/xarm7_{a}_controller/follow_joint_trajectory",
                         lambda gh, a=a: self._execute(a, gh), cancel_callback=lambda g: CancelResponse.ACCEPT,
                         callback_group=self.cg)
        n.create_service(GetPositionFK, "/compute_fk", self._fk, callback_group=self.cg)
        n.create_service(GetPositionIK, "/compute_ik", self._ik, callback_group=self.cg)
        self.static = StaticTransformBroadcaster(n)
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = "world", "jenga_tower"
        t.header.stamp = n.get_clock().now().to_msg()
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, self.tower.o)
        x, y, z, w = matrix_to_quat(self.tower.r)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = x, y, z, w
        self.static.sendTransform(t)

    def _publish_state(self):
        with self.lock:
            pos = self.q["right"] + self.q["left"] + self.hq["r"] + self.hq["l"]
        m = self.JointState(name=self.names, position=[float(v) for v in pos])
        m.header.stamp = self.node.get_clock().now().to_msg()
        self.js_pub.publish(m)

    def _execute(self, arm, gh):
        from control_msgs.action import FollowJointTrajectory
        traj = gh.request.trajectory
        order = [traj.joint_names.index(nm) for nm in joint_names(arm)]
        pts = traj.points
        sec = lambda p: p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
        print(f"SIM {arm}: {len(pts)} points, {sec(pts[-1]):.1f} s", flush=True)
        t0, prev_t, q0 = time.time(), 0.0, list(self.q[arm])
        speed = self.args.time_scale
        for p in pts:
            goal = [p.positions[k] for k in order]
            T = sec(p) / speed
            while time.time() - t0 < T:
                if gh.is_cancel_requested:
                    print(f"SIM {arm}: canceled", flush=True)
                    gh.canceled()
                    return FollowJointTrajectory.Result()
                s = min((time.time() - t0 - prev_t) / max(T - prev_t, 1e-6), 1.0)
                with self.lock:
                    self.q[arm] = [a + (b - a) * s for a, b in zip(q0, goal)]
                time.sleep(0.01)
            with self.lock:
                self.q[arm] = list(goal)
            q0, prev_t = list(goal), T
        gh.succeed()
        return FollowJointTrajectory.Result()

    @staticmethod
    def _arm_of(name: str) -> str:
        return "left" if "left" in name or name.startswith("ahl") else "right"

    def _fk(self, req, res):
        from geometry_msgs.msg import PoseStamped
        js = req.robot_state.joint_state
        res.pose_stamped, res.fk_link_names = [], []
        for link in req.fk_link_names:
            arm = self._arm_of(link)
            q = [js.position[js.name.index(nm)] for nm in joint_names(arm)]
            t = np.linalg.inv(self._frame(req.header.frame_id)) @ Chain(arm, tip=link).fk(q)
            m = PoseStamped()
            m.header.frame_id = req.header.frame_id
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, t[:3, 3])
            o = m.pose.orientation
            o.x, o.y, o.z, o.w = matrix_to_quat(t[:3, :3])
            res.pose_stamped.append(m)
            res.fk_link_names.append(link)
        res.error_code.val = 1
        return res

    def _ik(self, req, res):
        r = req.ik_request
        arm = self._arm_of(r.group_name or r.ik_link_name)
        names = joint_names(arm)
        js = r.robot_state.joint_state
        seed = [js.position[js.name.index(nm)] for nm in names]
        p = r.pose_stamped.pose
        o = p.orientation
        goal = self._frame(r.pose_stamped.header.frame_id) @ transform(
            quat_to_matrix(o.x, o.y, o.z, o.w), [p.position.x, p.position.y, p.position.z])
        sol = Chain(arm, tip=r.ik_link_name or None).ik(goal, seed)
        if sol is None:
            res.error_code.val = -31  # NO_IK_SOLUTION
            return res
        res.solution.joint_state.name = names
        res.solution.joint_state.position = [float(v) for v in sol]
        res.error_code.val = 1
        return res

    def _frame(self, frame: str) -> np.ndarray:
        return np.eye(4) if frame in ("", "world") else link_in_world(frame)

    # --- markers ----------------------------------------------------------------
    def _publish_markers(self):
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point

        now = self.node.get_clock().now().to_msg()
        out = MarkerArray()

        def mk(ns, i, kind, color, scale, pose=None):
            m = Marker()
            m.header.frame_id, m.header.stamp = "world", now
            m.ns, m.id, m.type, m.action = ns, i, kind, Marker.ADD
            m.color.r, m.color.g, m.color.b, m.color.a = map(float, color)
            m.scale.x, m.scale.y, m.scale.z = map(float, scale)
            if pose is None:
                m.pose.orientation.w = 1.0
            else:
                m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pose[:3, 3])
                o = m.pose.orientation
                o.x, o.y, o.z, o.w = matrix_to_quat(pose[:3, :3])
            out.markers.append(m)
            return m

        t = self.tower.t
        tower_frame = transform(self.tower.r, self.tower.o)
        k = 0
        for layer in range(t["layers"]):
            for slot in range(3):
                off = (slot - 1) * t["slot_pitch"]
                z = (layer + 0.5) * t["layer_h"]
                even = layer % 2 == 0  # runs along tower y
                local = transform(np.eye(3), [off, 0, z] if even else [0, off, z])
                size = (t["slot_pitch"] - 0.002, t["block_len"], t["layer_h"] - 0.001) if even else \
                       (t["block_len"], t["slot_pitch"] - 0.002, t["layer_h"] - 0.001)
                shade = 0.75 if (layer + slot) % 2 else 0.68
                mk("tower", k, Marker.CUBE, (0.85, shade, 0.45, 1.0), size, tower_frame @ local)
                k += 1
        k = 0
        lo, hi = self.setup.get("scan_layers", [1, 15])
        for face in FACE_NORMALS:
            for _layer, _slot, c, _n in self.tower.end_faces(face, range(lo, hi + 1)):
                mk("targets", k, Marker.SPHERE, (0.1, 0.4, 1.0, 0.8), (0.006,) * 3, transform(np.eye(3), c))
                k += 1

        with self.lock:
            q = {a: list(self.q[a]) for a in ARMS}
        status = []
        for i, a in enumerate(ARMS):
            pts = joint_points(self.chains[a], q[a])
            m = mk("arm", i, Marker.LINE_STRIP, (0.6, 0.6, 0.65, 1.0), (0.05, 0, 0))
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
            m = mk("joints", i, Marker.SPHERE_LIST, (0.3, 0.3, 0.35, 1.0), (0.07, 0.07, 0.07))
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts[:-1]]
            g = geometry(self.chains[a], self.tool[a], a, q[a], self.hq[a[0]], a in self.dino_arms)
            hand = g["hand"]
            mk("hand", i, Marker.CUBE, (0.2, 0.2, 0.2, 0.5), tuple(PALM_HI - PALM_LO),
               hand @ transform(np.eye(3), (PALM_LO + PALM_HI) / 2))
            for f, line in enumerate(g["fingers"]):
                m = mk("fingers", i * 4 + f, Marker.LINE_STRIP, (0.25, 0.25, 0.3, 1.0), (2 * FINGER_R, 0, 0))
                m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in line]
            c = clearances(self.tower, g)
            clear = min(c.values())
            if "dino" in g:
                tip = g["dino"]
                col = (1.0, 0.1, 0.1, 1.0) if min(c["dino_tip"], c["dino"]) < self.args.warn_mm else (0.1, 0.8, 0.3, 1.0)
                mk("dino", i, Marker.CYLINDER, col, (2 * DINO_R, 2 * DINO_R, DINO_LEN),
                   tip @ transform(np.eye(3), [0, 0, -DINO_LEN / 2]))
                mk("dino_axis", i, Marker.ARROW, (1.0, 1.0, 0.0, 1.0), (0.002, 0.004, 0.006)).points = [
                    Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
                    for v in (tip[:3, 3], tip[:3, 3] + tip[:3, 2] * 0.03)]
                status.append(f"{a}: Dino tip {c['dino_tip']:.0f} mm")
            if clear < self.args.warn_mm and not self.warned[a]:
                part = min(c, key=c.get)
                print(f"SIM WARNING {a}: {part} {clear:.1f} mm from the tower box", flush=True)
            self.min_mm[a] = min(self.min_mm[a], clear)
            if clear < self.args.warn_mm and not self.warned[a]:
                print(f"SIM WARNING {a}: {clear:.1f} mm from the tower box", flush=True)
            self.warned[a] = clear < self.args.warn_mm
            status.append(f"{a} min {self.min_mm[a]:.0f} mm")
        m = mk("status", 0, Marker.TEXT_VIEW_FACING, (1, 1, 1, 1), (0, 0, 0.025),
               transform(np.eye(3), self.tower.o + [0, 0, t["layers"] * t["layer_h"] + 0.08]))
        m.text = " | ".join(status)
        self.mk_pub.publish(out)

    def spin(self):
        from rclpy.executors import MultiThreadedExecutor
        ex = MultiThreadedExecutor()
        ex.add_node(self.node)
        ex.spin()


def check(name: str, warn_mm: float, samples: int = 40) -> None:
    """Offline (no ROS): walk a task of config/tasks.json, joint-interpolate every arm
    move like jenga-task does, print the smallest tower clearance per move."""
    from jenga_vision import arm_route, task as task_mod
    tasks = json.loads(task_mod.TASKS_FILE.read_text())
    task = task_mod.expand(tasks[name], tasks)
    arm, h = task["arm"], task["hand"]
    arm_poses = json.loads(POSES_FILE.read_text()).get(arm, {})
    hand_poses = json.loads(hand_slider.POSES_FILE.read_text()).get(h, {})
    sequences = json.loads(hand_slider.SEQUENCES_FILE.read_text())
    routes = arm_route.load_all().get(arm, {})
    setup = json.loads(SETUP_FILE.read_text())
    tower, chain, tool = Tower(setup["tower"], setup.get("aim", 0.5)), Chain(arm), dino_frame(setup, arm)
    q = np.array(arm_poses.get(START_POSES[arm], [0.0] * 7))
    hq = list(hand_poses.get(START_HAND, [0.0] * 20))
    worst = {}

    def move(label, goal):
        nonlocal q
        goal = np.array(goal, float)
        c = {}
        for s in np.linspace(0, 1, samples):
            for part, v in clearances(tower, geometry(chain, tool, arm, q + (goal - q) * s, hq)).items():
                c[part] = min(c.get(part, math.inf), v)
        q = goal
        part = min(c, key=c.get)
        for k, v in c.items():
            worst[k] = min(worst.get(k, math.inf), v)
        flag = "  <-- WARNING" if c[part] < warn_mm else ""
        print(f"  {label:42s} min {c[part]:6.1f} mm ({part}){flag}")

    print(f"task {name}: {arm} arm, Dino in hand, clearance to the tower box (whole tower, "
          f"no table/jig/other arm)")
    for n, step in enumerate(task["steps"], 1):
        if "arm" in step:
            move(f"[{n}] arm {step['arm']}", arm_poses[step["arm"]])
        elif "route" in step:
            for i, pt in enumerate(routes[step["route"]], 1):
                move(f"[{n}] {step['route']} {i}: {pt.get('label', '')}", pt["q"])
        elif "hand" in step:
            hq = list(hand_poses[step["hand"]])
        elif "hand_seq" in step:
            for fingers, pose in sequences[step["hand_seq"]]:
                for k in hand_slider.step_joints(fingers):
                    hq[k] = hand_poses[pose][k]
    print("smallest per part: " + ", ".join(f"{k} {v:.1f} mm" for k, v in worst.items()))


def rviz_env() -> dict:
    """Environment for rviz2 without what importing cv2 added (its Qt plugins make rviz2 fail on xcb)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("QT_QPA_")}
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = ":".join(p for p in env["LD_LIBRARY_PATH"].split(":") if "/cv2/" not in p)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--dino", nargs="*", default=["right", "left"], choices=ARMS,
                        help="arms that hold a Dino (drawn + clearance check), default both")
    parser.add_argument("--warn-mm", type=float, default=5.0, help="clearance warning (default 5 mm)")
    parser.add_argument("--time-scale", type=float, default=1.0, help="play moves this many times faster")
    parser.add_argument("--check", metavar="TASK", help="no ROS: print the tower clearance of every move of TASK")
    args = parser.parse_args()
    if args.check:
        check(args.check, args.warn_mm)
        return
    if os.environ.get("ROS_DOMAIN_ID", "0") in ("", *ROBOT_DOMAINS):
        raise SystemExit(f"refusing to run on ROS_DOMAIN_ID {os.environ.get('ROS_DOMAIN_ID') or 0} (real robot): "
                         "export ROS_DOMAIN_ID=77")
    sim = Sim(args)
    rviz = None if args.no_rviz else subprocess.Popen(["rviz2", "-d", str(RVIZ_FILE.resolve())], env=rviz_env())
    print(f"jenga-sim on ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']}: fake arms/hands, /compute_fk, /compute_ik, "
          "/jenga_sim/markers. Ctrl+C to quit.", flush=True)
    try:
        sim.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rviz:
            rviz.terminate()
        print("min clearance to the tower box: " + ", ".join(f"{a} {v:.1f} mm" for a, v in sim.min_mm.items()))


if __name__ == "__main__":
    main()
