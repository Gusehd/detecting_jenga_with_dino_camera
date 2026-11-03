"""Cartesian jog for the xArm7: move the hand by mm / deg, joints from MoveIt IK.

Only MoveIt's /compute_fk and /compute_ik services are used (no planning, no
collision check). A jog is cut into small steps (--step-mm / --step-deg) and
each step is solved with the previous solution as the IK seed, so the hand
moves on a nearly straight line and the arm stays in the same configuration.
A jog is refused if IK fails or any joint jumps more than MAX_STEP_DEG in one
step (IK switched to another solution).

Frames:
  base  axes of xarm_<arm>_link_base, rotation about the hand origin
  hand  axes of the hand (<ah>_base_link), e.g. "move 2 mm along the hand's z"
  dino  axes of the Dino (z = optical axis into the scene, y = image down),
        rotation about the point in focus (calibrated tool, see calib.py)
"""

from __future__ import annotations

import math
import threading

import numpy as np

MAX_STEP_DEG = 5.0  # largest joint change allowed between two IK steps
STEP_M = 0.001  # IK step along a translation (m)
STEP_RAD = math.radians(1.0)  # IK step along a rotation
CHECK_M = 0.001  # final FK position error allowed (m)


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix (Shepperd's method)."""
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
    q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return tuple(q)


def axis_rotation(axis: int, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def rpy_deg(r: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw (fixed x, y, z axes) in degrees."""
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    return (math.degrees(math.atan2(r[2, 1], r[2, 2])), math.degrees(pitch),
            math.degrees(math.atan2(r[1, 0], r[0, 0])))


def jog_pose(p: np.ndarray, r: np.ndarray, kind: str, axis: int, amount: float,
             frame: str, tool: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pose after moving `amount` (m for kind 't', rad for 'r') along/about `axis` of `frame`.
    frame "dino" needs `tool`, the 4x4 Dino frame in the hand frame."""
    if frame == "dino":
        rt, pt = tool[:3, :3], tool[:3, 3]
        if kind == "t":
            d = np.zeros(3)
            d[axis] = amount
            return p + r @ rt @ d, r
        focus = p + r @ pt
        r_new = r @ rt @ axis_rotation(axis, amount) @ rt.T
        return focus - r_new @ pt, r_new
    if kind == "t":
        d = np.zeros(3)
        d[axis] = amount
        return (p + (r @ d if frame == "hand" else d)), r
    rot = axis_rotation(axis, amount)
    return p, (r @ rot if frame == "hand" else rot @ r)


class JogError(RuntimeError):
    pass


class Cartesian:
    """FK / IK through MoveIt services, on a node that is already spinning in another thread."""

    def __init__(self, node, arm: str, tool: np.ndarray | None = None):
        from moveit_msgs.srv import GetPositionFK, GetPositionIK

        self.arm = arm
        self.tool = tool  # Dino frame in the hand frame, for frame "dino"
        self.group = f"xarm_{arm}"
        self.base = f"xarm_{arm}_link_base"
        self.tip = f"ah{arm[0]}_base_link"
        self.names = [f"xarm_{arm}_joint{i}" for i in range(1, 8)]
        self._fk = node.create_client(GetPositionFK, "/compute_fk")
        self._ik = node.create_client(GetPositionIK, "/compute_ik")

    def ready(self) -> bool:
        return self._fk.service_is_ready() and self._ik.service_is_ready()

    @staticmethod
    def _call(client, request, timeout: float = 2.0):
        if not client.service_is_ready():
            raise JogError(f"service {client.srv_name} not available (move_group running?)")
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            client.remove_pending_request(future)
            raise JogError(f"{client.srv_name} timed out")
        return future.result()

    def _robot_state(self, q: list[float]):
        from moveit_msgs.msg import RobotState

        state = RobotState()
        state.joint_state.name = list(self.names)
        state.joint_state.position = [float(v) for v in q]
        return state

    def fk(self, q: list[float]) -> tuple[np.ndarray, np.ndarray]:
        from moveit_msgs.srv import GetPositionFK

        req = GetPositionFK.Request()
        req.header.frame_id = self.base
        req.fk_link_names = [self.tip]
        req.robot_state = self._robot_state(q)
        res = self._call(self._fk, req)
        if res.error_code.val != 1 or not res.pose_stamped:
            raise JogError(f"FK failed (error {res.error_code.val})")
        pose = res.pose_stamped[0].pose
        o = pose.orientation
        return (np.array([pose.position.x, pose.position.y, pose.position.z]),
                quat_to_matrix(o.x, o.y, o.z, o.w))

    def ik(self, p: np.ndarray, r: np.ndarray, seed: list[float]) -> list[float] | None:
        from moveit_msgs.srv import GetPositionIK

        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = self.group
        ik.ik_link_name = self.tip
        ik.robot_state = self._robot_state(seed)
        ik.avoid_collisions = False
        ik.timeout.nanosec = 100_000_000
        ik.pose_stamped.header.frame_id = self.base
        ik.pose_stamped.pose.position.x, ik.pose_stamped.pose.position.y, ik.pose_stamped.pose.position.z = \
            (float(v) for v in p)
        o = ik.pose_stamped.pose.orientation
        o.x, o.y, o.z, o.w = (float(v) for v in matrix_to_quat(r))
        res = self._call(self._ik, req)
        if res.error_code.val != 1:
            return None
        js = res.solution.joint_state
        idx = {n: i for i, n in enumerate(js.name)}
        if not all(n in idx for n in self.names):
            return None
        return [js.position[idx[n]] for n in self.names]

    def plan_jog(self, q0: list[float], kind: str, axis: int, amount: float, frame: str) -> list[list[float]]:
        """Joint waypoints (excluding q0) for one jog. Raises JogError if it can't be done safely."""
        p0, r0 = self.fk(q0)
        step = STEP_M if kind == "t" else STEP_RAD
        n = max(1, math.ceil(abs(amount) / step - 1e-9))
        seed, path = list(q0), []
        for i in range(1, n + 1):
            p, r = jog_pose(p0, r0, kind, axis, amount * i / n, frame, self.tool)
            q = self.ik(p, r, seed)
            if q is None:
                raise JogError(f"IK failed at step {i}/{n} (out of reach or near a singularity)")
            jump = max(abs(math.degrees(a - b)) for a, b in zip(q, seed))
            if jump > MAX_STEP_DEG:
                raise JogError(f"joint jump {jump:.1f} deg at step {i}/{n}: IK changed configuration")
            path.append(q)
            seed = q
        p_end, _ = self.fk(path[-1])
        p_goal, _ = jog_pose(p0, r0, kind, axis, amount, frame, self.tool)
        err = float(np.linalg.norm(p_end - p_goal))
        if err > CHECK_M:
            raise JogError(f"FK check: end is {err * 1000:.1f} mm off the goal")
        return path
