"""Offline FK / IK for the xArm7 arms, straight from the URDF (no ROS, no MoveIt).

The URDF is the teammate's dual_robot.urdf (jenga_world/jenga_bringup/...).
Only the chain world -> xarm_<arm>_joint1..7 -> <tip> is used; fixed joints are
folded in. IK is damped least squares with joint limits, seeded (stays near
the seed, so it keeps the arm's configuration), good for planning poses that
are later checked and sent with the normal tools.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parents[2] / "jenga_world/jenga_bringup/bi_manipulator_config/urdf/dual_robot.urdf"


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transform(r: np.ndarray, p) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, p
    return t


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def rot_error(r_goal: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) that turns r into r_goal."""
    e = r_goal @ r.T
    angle = math.acos(max(-1.0, min(1.0, (np.trace(e) - 1) / 2)))
    if angle < 1e-9:
        return np.zeros(3)
    w = np.array([e[2, 1] - e[1, 2], e[0, 2] - e[2, 0], e[1, 0] - e[0, 1]])
    if math.sin(angle) < 1e-6:  # ~180 deg: axis from the symmetric part
        axis = np.sqrt(np.maximum((np.diag(e) + 1) / 2, 0))
        return axis * angle
    return w / (2 * math.sin(angle)) * angle


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray
    lo: float
    hi: float


# The 10/15 dual_robot.urdf mounts the hands with rpy 0; the real robot uses the
# xacro's +-90 deg about z (checked 2026-10-30: a Dino-frame jog went down instead
# of out of the face until this was applied).
RPY_OVERRIDE = {"right_qc_to_right_hand": (0.0, 0.0, 1.5708), "left_qc_to_left_hand": (0.0, 0.0, -1.5708)}


def load_joints(path: Path = URDF) -> dict[str, Joint]:
    """Joints by child link name."""
    joints = {}
    for j in ET.parse(path).getroot().iter("joint"):
        if j.find("parent") is None:
            continue  # ros2_control <joint> entries
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = list(RPY_OVERRIDE.get(j.get("name"), rpy))
        a = j.find("axis")
        lim = j.find("limit")
        child = j.find("child").get("link")
        joints[child] = Joint(
            j.get("name"), j.get("type"), j.find("parent").get("link"), child,
            transform(rpy_matrix(*rpy), xyz),
            np.array([float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()]),
            float(lim.get("lower", -math.inf)) if lim is not None else -math.inf,
            float(lim.get("upper", math.inf)) if lim is not None else math.inf)
    return joints


class Chain:
    """world -> tip through the 7 arm joints."""

    def __init__(self, arm: str, tip: str | None = None, path: Path = URDF):
        joints = load_joints(path)
        self.arm = arm
        self.tip = tip or f"ah{arm[0]}_base_link"
        self.links: list[Joint] = []
        link = self.tip
        while link in joints:
            self.links.append(joints[link])
            link = joints[link].parent
        self.links.reverse()
        self.names = [j.name for j in self.links if j.kind in ("revolute", "continuous")]
        expected = [f"xarm_{arm}_joint{i}" for i in range(1, 8)]
        if self.names != expected:
            raise ValueError(f"chain to {self.tip} has joints {self.names}, expected {expected}")
        self.lo = np.array([j.lo for j in self.links if j.kind != "fixed"])
        self.hi = np.array([j.hi for j in self.links if j.kind != "fixed"])

    def fk(self, q, tool: np.ndarray | None = None) -> np.ndarray:
        """4x4 pose of the tip (times `tool`, a 4x4 in the tip frame) in world."""
        t, i = np.eye(4), 0
        for j in self.links:
            t = t @ j.origin
            if j.kind != "fixed":
                t = t @ transform(axis_angle(j.axis, q[i]), [0, 0, 0])
                i += 1
        return t @ tool if tool is not None else t

    def jacobian(self, q, tool: np.ndarray | None = None) -> np.ndarray:
        t, i = np.eye(4), 0
        axes, origins = [], []
        for j in self.links:
            t = t @ j.origin
            if j.kind != "fixed":
                axes.append(t[:3, :3] @ j.axis)
                origins.append(t[:3, 3].copy())
                t = t @ transform(axis_angle(j.axis, q[i]), [0, 0, 0])
                i += 1
        p = (t @ tool if tool is not None else t)[:3, 3]
        jac = np.zeros((6, 7))
        for k, (a, o) in enumerate(zip(axes, origins)):
            jac[:3, k] = np.cross(a, p - o)
            jac[3:, k] = a
        return jac

    def ik(self, goal: np.ndarray, seed, tool: np.ndarray | None = None,
           iters: int = 300, tol_m: float = 1e-5, tol_rad: float = 1e-4) -> np.ndarray | None:
        """Joint values reaching the 4x4 `goal` (world), starting at `seed`; None if not reached."""
        q = np.array(seed, dtype=float)
        for _ in range(iters):
            t = self.fk(q, tool)
            e = np.concatenate([goal[:3, 3] - t[:3, 3], rot_error(goal[:3, :3], t[:3, :3])])
            if np.linalg.norm(e[:3]) < tol_m and np.linalg.norm(e[3:]) < tol_rad:
                return q
            jac = self.jacobian(q, tool)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), e)
            step = np.max(np.abs(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, self.lo, self.hi)
        return None
