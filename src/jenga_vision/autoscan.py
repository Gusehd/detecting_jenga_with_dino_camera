"""Automatic Dino scan of tower faces: approach, focus by moving, find the gap, shoot.

The tower model (config/scan_setup.json: tower pose, layer height 21.86 mm,
calibrated Dino tool) is right in height and direction, but a real tower
differs by up to ~10 mm in depth, and the Dino sits differently in the hand
after every pick (axis shift > 1 cm, ~5 deg turn). So per face and layer:

  1. go to the left end face of the layer (as seen on this face), at the
     expected depth and height
  2. focus (focus_search.find_peak): outwards first, inwards only if nothing better
     is there; a peak = clear falls on both sides of it. Never more than
     `--limit-mm` deeper than the previous block, nor `--max-in-mm` past the model.
  3. the edge (first time on a face): look up / down for the bright rim of the end
     face (a closed gap has no dark band), check it is the TOP edge (long grain
     above it), else go up a layer
  4. center the edge (mm/px measured with a 0.5 mm move), level it by turning
     the Dino about its axis, refocus
  5. shoot: FRAMES_PER_SHOT frames, gap = median of measure_end (as jenga-ends)
  6. next block of the layer: model position + the depth / height / turn found
     here, short refocus (+-`--refocus-mm`); then out 30 mm and up to the next layer

At magnification 50 the Dino is in focus only when it almost touches the
block (an automatic approach nearly pushed one, 2026-10-30): use ~30, where
its tip is about 25 mm from the block in focus (2026-11-02).

Results go to captures/autoscan_<time>/: ends.csv in the jenga-ends format
(key L<old layer>:f<face>:<screen block>, old layer = teammate layer + 1, faces
1..4 = +x, +y, -x, -y), shots.csv with the found poses, one image per shot, and
ranking.json (teammate block names jenga_LXX_SY, easiest first, jenga-ends
ranking with --mode), jenga_pick_candidates.json (the teammate's jenga_game format).
Rank several sessions together (both arms):

    uv run jenga-autoscan rank captures/autoscan_A captures/autoscan_B

Everything in one go (pick the Dino, scan -y and +x, pull order, place the Dino):

    uv run jenga-autoscan run           # virtual run (needs ROS sourced for the task dry runs)
    uv run jenga-autoscan run --send    # real robot

--fast (scan and run) is the same scan with shorter waits and fewer focus samples, see FAST.
Compare it on one tower with a normal scan before trusting it:

    uv run jenga-autoscan compare captures/autoscan_NORMAL captures/autoscan_FAST

run always picks and places the Dino with pick_dino_quick / place_dino_quick (robot-checked 2026-11-09,
about 26 s each). --quick (DRAFT) also speeds up the joint moves to and from the face (see QUICK). --motion-only skips the
layers (pick, go to the face, leave, place), to watch and time those moves in jenga-sim.

Robot / camera:
  without --send  no ROS: a virtual arm (the joints are just set) and the fake camera
  --send          real moves through /xarm7_<arm>_controller/follow_joint_trajectory
                  (on domain 77 with jenga-sim: virtual robot in RViz)
  --fake          fake camera even with --send: synthetic images from the model
                  with an unknown depth error (--fake-depth-mm), to test the logic

The arm must hold the Dino and be at a pose from which a joint move to the
face's first point is safe (home, or give --via poses). Close jenga-view first
(one process per camera). Ctrl+C stops the arm where it is; in the camera
window SPACE = take the current depth, q = abort.

    uv run jenga-autoscan scan --face px --layers 13 13                          # virtual run
    uv run jenga-autoscan scan --face px --layers 13 13 --send --via cali_way_1  # real robot
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from jenga_vision import ends
from jenga_vision.arm_slider import POSES_FILE
from jenga_vision.camera import focus_value
from jenga_vision.dashboard import DEFAULT_PORT, Dashboard, NoDashboard, replay
from jenga_vision.focus_search import find_peak
from jenga_vision.kinematics import Chain
from jenga_vision.scan_plan import (FACE_NORMALS, SETUP_FILE, Tower, arm_tool, away_from_limits, plan_face_stable,
                                    shot_pose)

FACE_NUMBER = {"+x": 1, "+y": 2, "-x": 3, "-y": 4}  # jenga-ends faces: 90 deg counterclockwise each
LINE_STEP_M = 0.002  # IK step along a straight move
LINE_ACCEL_MM_S2 = 200.0  # start and stop of a straight move
RAMP_DT_S = 0.05  # time between trajectory points
MAX_JOINT_STEP_DEG = 3.0  # per IK step; more = the arm would change configuration
RETREAT_M = 0.03  # out from the face between layers
SETTLE_S = 1.0  # wait after a move before reading frames: 0.3 s still shakes the focus value (hold test)
DEFAULT_MM_PER_PX = 0.02  # Dino at dial 30; measured on the first shot when possible
EDGE_TOLERANCE_PX = 25  # edge row this close to the image centre is good enough
BRIGHTNESS_REF = 100.0  # steady_focus is the focus value at this image median
EDGE_STEP = 40  # gray step of the rim over 6 rows: in focus 98-219, no edge in the image <= 21 (2026-11-02)
RIM_HALF_WIDTH = 40  # px, columns used for the rim row at the middle of each image third
EDGE_ROWS_AGREE = 12  # the middle third's rim row is this close to the line through the outer two
EDGE_MAX_SPAN = 160  # rows between the left and right thirds: the Dino may sit up to about 20 deg turned (13 deg seen 2026-11-10; 80 px = 10 deg hid the edge, no leveling, 12 s searches per shot)
EDGE_MARGIN = 0.03  # stripes difference closer to 0 than this: undecided, taken as the top edge
REFOCUS_MOVE_M = 0.001  # focus again after moving more than this up / down
EDGE_SEARCH_STEP_MM = 2.5  # vertical search step for the edge (the image is ~10 mm high at dial 30)
UNCERTAIN_PX = 1.0  # a layer's middle / side difference this small decides nothing (a repeat differs by 2 px)
RUN_LAYERS = [2, 14]  # floors 3..15: not floors 1, 2 nor the top two (user, 2026-11-03)
# Per face, as tested on the real robot 2026-11-03. -y: straight from home, entering at the top layer
# (cali_way_1 made the cables hit the tower; entering low passes 116 mm from the axis instead of 157).
RUN_FACES = [("-y", {"face": "my", "seed": "home", "via": [], "top_first": True, "prior_depth_mm": -10.0,
                     "prior_down_mm": -3.5, "joint_speed": 10.0}),
             ("+x", {"face": "px", "seed": "cali_way_3", "via": ["cali_way_1"], "top_first": False,
                     "prior_depth_mm": -5.0, "prior_down_mm": 0.0, "joint_speed": 20.0})]
# Replaying the focus logs of 2026-11-03 (83 searches, noise up to 8 %): 3.9 samples per search instead of 7.1,
# peak within 0.4 mm in 9 of 10. The replay can't show exposure hunting, and px against defocus was not measured
# at dial 40. 0.4 s after a move the focus value still shakes by about 6 %; the shot itself always waits SETTLE_S.
FAST = {"step_mm": 2.0, "settle_s": 0.4, "focus_fall": 0.85, "focus_falls": 1, "level_once": True,
        "line_speed": 30.0, "joint_speed": 20.0}
# Moves that are the same every time (DRAFT, 2026-11-09, sim only): the joint moves to and from a face at
# approach_speed with the Dino in the hand, at least approach_min_s per move. --approach-slow-deg would slow
# the end next to the face, but the -y far point is 225 mm from the tower box and the closest the path comes
# is 93 mm halfway (jenga-sim model, no cables), so QUICK doesn't. Default (no --quick): joint_speed, 2 s min.
QUICK = {"approach_speed": 45.0, "approach_min_s": 1.0, "far_speed": 120.0, "parallel_speed": 72.0,
         "line_accel": 300.0}  # 1.2x of the speeds first run on the robot 2026-11-10 (user)
SAFE_FROM_FACE_M = 0.025  # point in focus this far out of the face plane (the Dino tip about 9 mm more): far_speed
PARALLEL_M = 0.001  # a straight move that changes the distance to the face by less than this: parallel_speed
# Named sets of settings, used as defaults (an option given by hand still wins): --preset=v1
#   v1 (2026-11-10): --fast --quick --sweep --shot-settle-s 0.3. Both faces 2.8 min of scanning, with the quick
#   pick / place 3 min 40 s; tower of 2026-11-10 (after it was tidied at 11:50): 15 in a row by pushing.
#   v2 (DRAFT 2026-11-10, not run on the robot): v1 + 0.2 s settle, 100 mm/s along the face, 60 deg/s to and
#   from the face, one camera for both faces, the next face starts with the Dino turn / scale, narrow 2nd reference,
#   the search starts where the last scan of the face found its first block.
PRESETS = {"v1": {"sweep": True, "shot_settle_s": 0.3},
           "v2": {"sweep": True, "shot_settle_s": 0.2, "parallel_speed": 100.0, "approach_speed": 60.0,
                  "keep_camera": True, "carry_dino": True, "narrow_ref2": True, "prior_from_last": True}}
LEVEL_DEG, MAX_ROLL_DEG = 0.7, 20.0
# --blur-test: passes without stopping, to see how much moving changes the gap px (2026-11-10)
BLUR_MAX_MM_S = 60.0  # passes along the face at a fixed distance (user OK to raise from 30, 2026-11-10)
BLUR_MARGIN_M = 0.010  # along passes start / end this far beyond the outer block centres
BLUR_V_SPAN_M = 0.010  # up / down passes go this far above and below the edge
BLUR_WINDOW_MM = 8.0  # frames this close to a block centre count for that block
BLUR_EDGE_PX = 80  # and only frames with the edge this close to the image centre
GLIDE_LAG_S = 0.05  # camera frame behind the joint state while moving (33-67 ms, blur test 2026-11-10)
AE_DROP = 0.5  # a frame below this x the recent median: the auto exposure collapsed
AE_BACK = 0.85  # back after a collapse when this x the brightness before it ...
AE_RECOVER_MAX_S = 1.6  # ... or this long after it (it climbs 0.5-1.5 s, 2026-11-10)
AE_STEADY = 0.08  # the frames of one shot within this of each other
AE_MAX_WAIT_S = 2.0  # then the frames are taken anyway (and counted)
FRAME_LAG_S = 0.05  # a frame read this long after the wait was exposed after it
LAST_START_MAX_OFF_MM = 8.0  # a last edge height this far from the prior is another layer (layer 21.9 mm), not used
LAST_START_FILE = ".last_start.json"  # in --out: per face, depth and height of the first block of the last scan
FAKE_FPS = 24.0  # the Dino at 640x480 while moving (blur test 2026-11-10)
GLIDE_MIN_FRAMES = 3  # good frames per block in a glide, else that block is shot stopped  # level the gap when tilted more than this; limit of the correction


def rim_row(gray: np.ndarray) -> tuple[int, float]:
    """(row, size) of the strongest dark-above / bright-below step of a median row profile."""
    p = cv2.GaussianBlur(np.median(gray, axis=1).reshape(-1, 1).astype(np.float32), (1, 5), 0).ravel()
    d = p[6:] - p[:-6]
    r = int(d.argmax())
    return r + 3, float(d[r])


def steady_focus(frame: np.ndarray) -> float:
    """focus_value at a fixed brightness. The Dino's auto exposure can't be locked and hunts even
    with nothing moving (brightness x2 = focus value x5, 2026-11-03), so the raw value is no use."""
    return focus_value(frame) * (BRIGHTNESS_REF / max(float(np.median(frame)), 1.0)) ** 2


def rim_darkness(frame: np.ndarray) -> float:
    """Darkest row just above the rim / image median, median over strips: 0 = black line. Separates
    the shots measure_end reads as 0 px (2026-11-02: open gaps 0.01, thin lines 0.2-0.4, rest 0.4-0.8)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lo, hi = h // 4, 3 * h // 4
    bounds = np.linspace(int(w * ends.MEASURE_SPAN[0]), int(w * ends.MEASURE_SPAN[1]), ends.STRIPS + 1).astype(int)
    found = []
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        p = cv2.GaussianBlur(np.median(gray[:, s0:s1], axis=1).reshape(-1, 1).astype(np.float32), (1, 5), 0).ravel()
        r = lo + int((p[6:] - p[:-6])[lo:hi].argmax()) + 3
        found.append(float(p[max(r - 12, 0):r + 1].min()))
    return float(np.median(found)) / max(float(np.median(gray)), 1.0)


def rim_rows(frame: np.ndarray) -> list[float | None]:
    """Row of the end face's top rim in the left, middle and right thirds (None = no step there)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]
    found = [rim_row(gray[:, (2 * i + 1) * w // 6 - RIM_HALF_WIDTH: (2 * i + 1) * w // 6 + RIM_HALF_WIDTH])
             for i in range(3)]  # narrow columns: a tilted rim would smear out over a whole third
    return [float(r) if size >= EDGE_STEP else None for r, size in found]


def edge_row(frame: np.ndarray) -> float | None:
    """Row of the edge between two layers; None if no edge in the image. The bright rim, not
    the dark gap: a closed gap has no dark band, and the block is still to be measured (0 px)."""
    left, mid, right = rim_rows(frame)
    rows = [r for r in (left, mid, right) if r is not None]
    if len(rows) < 2 or max(rows) - min(rows) > EDGE_MAX_SPAN:
        return None
    if len(rows) == 3 and abs(mid - (left + right) / 2) > EDGE_ROWS_AGREE:  # a straight line, may be tilted
        return None
    return float(np.mean(rows))


def edge_seen(frame: np.ndarray) -> bool:
    return edge_row(frame) is not None


def stripes(gray: np.ndarray) -> float:
    """Vertical / horizontal gradient: > 1 for the long grain of a block's side
    (horizontal stripes), ~1 for the end grain."""
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean()
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean()
    return float(gy / (gx + 1e-6))


def edge_side(frame: np.ndarray, row: float) -> float:
    """> 0: long grain above the gap and end grain below = the TOP edge of the end face
    (the one to measure); < 0: the bottom edge. On the 2026-10-30 top-edge shots the
    stripes were 1.10-1.66 above vs 0.95-1.13 below."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r, m, n = int(row), 15, 100
    above, below = gray[max(0, r - m - n): max(0, r - m)], gray[r + m: r + m + n]
    if above.shape[0] < 20 or below.shape[0] < 20:
        return 0.0
    return stripes(above) - stripes(below)


def edge_tilt(frame: np.ndarray) -> float | None:
    """Slope (rows per column) of the edge, from its row in the left and right thirds; None if not seen."""
    left, _, right = rim_rows(frame)
    if left is None or right is None:
        return None
    return (right - left) / (2 * frame.shape[1] / 3)


class ScanError(RuntimeError):
    pass


class DepthTaken(Exception):
    """SPACE in the camera window: this depth is the focus."""

    def __init__(self, depth: float, value: float):
        super().__init__()
        self.depth, self.value = depth, value


# --- geometry ----------------------------------------------------------------------
@dataclass
class Shot:
    layer: int  # teammate layer (L00 = bottom)
    slot: int
    screen: str  # left / middle / right as seen on this face
    aim: np.ndarray  # model aim point (top edge middle of the end face), world
    normal: np.ndarray  # outward face normal, world

    def pose(self, depth_m: float, down_m: float, roll: float = 0.0) -> np.ndarray:
        """Dino frame with the point in focus `depth_m` past the model face (into the
        tower), `down_m` below the model gap, turned `roll` rad about the optical axis
        (levels the image when the Dino sits turned in the hand)."""
        from jenga_vision.kinematics import axis_angle, transform
        p = shot_pose(self.aim - self.normal * depth_m - np.array([0, 0, down_m]), self.normal, 0.0)
        return p @ transform(axis_angle(np.array([0, 0, 1.0]), roll), [0, 0, 0])


def face_shots(tower: Tower, face: str, layers: range) -> dict[int, list[Shot]]:
    """Per layer, the end faces of `face` sorted left -> right on screen."""
    faces = tower.end_faces(face, layers)
    right = shot_pose(faces[0][2], faces[0][3], 0)[:3, 0]
    out: dict[int, list[Shot]] = {}
    for layer, slot, aim, normal in faces:
        out.setdefault(layer, []).append(Shot(layer, slot, "", aim, normal))
    for row in out.values():
        row.sort(key=lambda s: s.aim @ right)
        for s, name in zip(row, ends.BLOCKS):
            s.screen = name
    return out


def first_face(layer: int) -> str:
    """The face where this teammate layer is named from (jenga-ends face 1 or 2)."""
    return "+x" if layer % 2 == 1 else "+y"


def block_slot(tower: Tower, layer: int, block: str) -> int:
    """Teammate slot of a jenga-ends block name (left/middle/right seen from the layer's first face)."""
    row = face_shots(tower, first_face(layer), range(layer, layer + 1))[layer]
    return next(s.slot for s in row if s.screen == block)


# --- arm ---------------------------------------------------------------------------
def ramped(path: list[np.ndarray], q0: np.ndarray, dist_mm: float, seconds: float,
           accel: float = LINE_ACCEL_MM_S2) -> list[np.ndarray]:
    """The same path, resampled so that evenly timed points start and stop at `accel` mm/s2
    instead of jumping to full speed (fast moves would shake the Dino for longer)."""
    if dist_mm < 1e-6:
        return path
    qs = np.vstack([q0, *path])
    a = accel
    disc = seconds ** 2 - 4 * dist_mm / a  # cruise speed from dist = v * (seconds - v / a)
    v = a * (seconds - math.sqrt(disc)) / 2 if disc > 0 else 2 * dist_mm / seconds
    ta = v / a if disc > 0 else seconds / 2
    a = v / ta
    n = max(len(path), math.ceil(seconds / RAMP_DT_S))
    out = []
    for i in range(1, n + 1):
        t = seconds * i / n
        if t < ta:
            d = a * t * t / 2
        elif t > seconds - ta:
            d = dist_mm - a * (seconds - t) ** 2 / 2
        else:
            d = v * (t - ta / 2)
        u = min(d / dist_mm, 1.0) * (len(qs) - 1)
        k = min(int(u), len(qs) - 2)
        out.append(qs[k] + (qs[k + 1] - qs[k]) * (u - k))
    return out


def approach(args) -> dict:
    """Speeds of the joint moves to and from a face (QUICK with --quick)."""
    slow = {"slow_speed": args.approach_slow_speed or args.joint_speed}
    deg = args.approach_slow_deg
    return {"speed": args.approach_speed or args.joint_speed, "min_s": args.approach_min_s,
            "slow_end": dict(slow, slow_end_deg=deg) if deg else None,
            "slow_start": dict(slow, slow_start_deg=deg) if deg else None}


class Mover:
    """Joint moves and straight Dino-frame moves; virtual (joints just set) without send."""

    def __init__(self, arm: str, tool: np.ndarray, send: bool, q0, joint_speed_deg: float, line_speed_mm: float):
        self.arm, self.tool, self.send = arm, tool, send
        self.chain = Chain(arm)
        self.joint_speed, self.line_speed = joint_speed_deg, line_speed_mm
        self.far_speed = self.parallel_speed = None  # mm/s, see speed_for
        self.line_accel = LINE_ACCEL_MM_S2
        self.face: tuple[np.ndarray, np.ndarray] | None = None  # (point on the face, outward normal)
        self.ros = None
        self.idle = None  # called while waiting for a move (camera preview)
        self.seconds = 0.0  # planned motion time so far
        if send:
            from jenga_vision.arm_slider import ArmNode
            self.ros = ArmNode(arm, True, node_name="jenga_autoscan")
            t0 = time.time()
            while self.ros.current() is None:
                if time.time() - t0 > 20:  # discovery with the Jazzy robot PC takes seconds
                    raise ScanError("no /joint_states for the arm (ROS sourced? right ROS_DOMAIN_ID?)")
                time.sleep(0.1)
            q0 = self.ros.current()
        self.q = np.array(q0, float)

    def dino(self) -> np.ndarray:
        return self.chain.fk(self.q, self.tool)

    def _send(self, path: list[np.ndarray], seconds: float) -> None:
        self.seconds += seconds
        if self.ros is not None:
            self.ros.move_path([list(map(float, q)) for q in path], seconds)
            try:
                while not self.ros.done.wait(0.002 if self.idle else 0.05):
                    if self.idle:
                        self.idle()
            except KeyboardInterrupt:
                self.ros.stop()
                self.ros.done.wait(3)
                raise
            if not self.ros.succeeded:
                raise ScanError(f"arm move failed: {self.ros.result}")
        elif self.idle:  # virtual arm: let the camera see the way
            for q in path:
                self.q = np.array(q, float)
                self.idle()
        self.q = np.array(path[-1], float)

    def joint_move(self, q, label: str = "", speed: float | None = None, slow: dict | None = None,
                   min_s: float = 2.0) -> None:
        """Straight joint move at `speed` (default joint_speed); `slow` = a task step's slow_start_deg /
        slow_end_deg / slow_speed: that part of the same path at slow_speed."""
        from jenga_vision.task import step_parts
        q = np.array(q, float)
        for part, v in step_parts(list(self.q), list(q), speed or self.joint_speed, slow or {}):
            part = np.array(part, float)
            biggest = float(np.degrees(np.max(np.abs(part - self.q))))
            seconds = max(min_s, biggest / v)
            print(f"   joint move {label}: largest joint {biggest:.0f} deg at {v:.0f} deg/s, {seconds:.1f} s")
            self._send([part], seconds)

    def speed_for(self, start: np.ndarray, goal: np.ndarray) -> tuple[float, str]:
        """mm/s of a straight move: far_speed when both ends (so the whole line) are at least
        SAFE_FROM_FACE_M out of the face plane, parallel_speed when the distance to the face stays
        within PARALLEL_M, else line_speed (towards / away from the tower)."""
        if self.face is None:
            return self.line_speed, "line"
        point, normal = self.face
        out0, out1 = float((start - point) @ normal), float((goal - point) @ normal)
        if self.far_speed and min(out0, out1) >= SAFE_FROM_FACE_M:
            return self.far_speed, "far"
        if self.parallel_speed and abs(out1 - out0) <= PARALLEL_M:
            return self.parallel_speed, "parallel"
        return self.line_speed, "line"

    def line_to(self, goal_dino: np.ndarray, min_s: float = 0.4, speed: float | None = None) -> None:
        """Straight line of the Dino frame (position; orientation interpolated) to `goal_dino`
        at `speed` mm/s (default: speed_for)."""
        start = self.dino()
        speed = speed or self.speed_for(start[:3, 3], goal_dino[:3, 3])[0]
        dp = goal_dino[:3, 3] - start[:3, 3]
        rel = start[:3, :3].T @ goal_dino[:3, :3]
        ang = math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1) / 2)))
        n = max(1, math.ceil(max(np.linalg.norm(dp) / LINE_STEP_M, ang / math.radians(1))))
        axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-9 else np.array([0, 0, 1.0])
        from jenga_vision.kinematics import axis_angle, transform
        q, path = self.q.copy(), []
        for i in range(1, n + 1):
            s = i / n
            g = transform(start[:3, :3] @ axis_angle(axis, ang * s), start[:3, 3] + dp * s)
            nq = self.chain.ik(g @ np.linalg.inv(self.tool), q)
            if nq is None:
                raise ScanError("IK failed on a straight move (out of reach?)")
            if np.degrees(np.max(np.abs(nq - q))) > MAX_JOINT_STEP_DEG:
                raise ScanError("a straight move would flip the arm configuration")
            path.append(nq)
            q = nq
        biggest = float(np.degrees(np.max(np.abs(q - self.q))))
        dist_mm = float(np.linalg.norm(dp) * 1000)
        seconds = max(min_s, dist_mm / speed + speed / self.line_accel, biggest / self.joint_speed)
        self._send(ramped(path, self.q, dist_mm, seconds, self.line_accel), seconds)

    def shutdown(self) -> None:
        if self.ros is not None:
            self.ros.shutdown()


# --- cameras -----------------------------------------------------------------------
class RealCamera:
    """The Dino, read all the time by a thread. Its auto exposure (shutter priority, the only mode
    the camera accepts) now and then collapses to black and climbs back over 0.5-1.5 s
    (2026-11-03, 2026-11-10); frames of that climb give wrong gaps. frames() therefore waits
    until the exposure is back and returns frames of steady brightness, taken after the wait."""

    def __init__(self, device: int, window: bool):
        from jenga_vision.camera import DinoCamera
        self.cam = DinoCamera(device)
        self.window = window
        self.settle = SETTLE_S
        self.cond = threading.Condition()
        self.buf: deque = deque(maxlen=64)  # (seq, time, frame, brightness)
        self.seq = 0
        self.collapse: tuple[float, float] | None = None  # (time, brightness before) of the last collapse
        self.collapses, self.waited, self.unsteady = 0, 0.0, 0
        self.last_grab = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            try:
                frame = self.cam.read()
            except RuntimeError:
                time.sleep(0.01)
                continue
            t = time.time()
            b = float(np.median(frame[::8, ::8, 1]))
            with self.cond:
                recent = [x[3] for x in list(self.buf)[-30:]]
                typical = float(np.median(recent)) if len(recent) >= 10 else None
                if typical and b < AE_DROP * typical and (self.collapse is None or t - self.collapse[0] > AE_RECOVER_MAX_S):
                    self.collapse = (t, typical)
                    self.collapses += 1
                self.seq += 1
                self.buf.append((self.seq, t, frame, b))
                self.cond.notify_all()

    def _after(self, t0: float, n: int) -> list[tuple]:
        """The first n frames read after t0 (waits for them)."""
        deadline = time.time() + 3.0
        with self.cond:
            while True:
                got = [x for x in self.buf if x[1] > t0]
                if len(got) >= n:
                    return got[:n]
                left = deadline - time.time()
                if left <= 0:
                    raise ScanError("no frames from the Dino for 3 s")
                self.cond.wait(left)

    def _exposure_ok(self, got: list[tuple]) -> bool:
        bs = [x[3] for x in got]
        if max(bs) > (1 + AE_STEADY) * max(min(bs), 1.0):
            return False
        if self.collapse is None:
            return True
        t, before = self.collapse
        return got[0][1] - t >= AE_RECOVER_MAX_S or min(bs) >= AE_BACK * before

    def frames(self, n: int, settle: float | None = None) -> list[np.ndarray]:
        wait = self.settle if settle is None else settle
        end = time.time() + wait
        while time.time() < end:  # the video keeps running on the dashboard
            if isinstance(self.dash, Dashboard) and self.buf:
                self.dash.frame(self.buf[-1][2])
            time.sleep(min(0.03, max(0.0, end - time.time())))
        t0 = end + FRAME_LAG_S
        start = None  # when the first frames were there: waiting beyond that is for the exposure
        while True:
            got = self._after(t0, n)
            start = start or time.time()
            if self._exposure_ok(got):
                break
            if time.time() - start > AE_MAX_WAIT_S:
                self.unsteady += 1
                print(f"   exposure not steady after {AE_MAX_WAIT_S:.1f} s (brightness "
                      f"{', '.join(f'{x[3]:.0f}' for x in got)}), using these frames")
                break
            t0 = got[0][1]  # slide on by one frame
        self.waited += time.time() - start
        out = [x[2] for x in got]
        self.dash.frame(out[-1])
        return out

    def grab(self) -> np.ndarray:
        """The next new frame, no settling (passes while the arm moves)."""
        with self.cond:
            while not self.buf or self.buf[-1][0] <= self.last_grab:
                if not self.cond.wait(1.0):
                    raise ScanError("no frames from the Dino for 1 s")
            self.last_grab, frame = self.buf[-1][0], self.buf[-1][2]
        self.dash.frame(frame)
        if self.window:
            self.show(frame, "pass")
        return frame

    def preview(self, text: str = "moving") -> None:
        """Live view while the arm moves or waits (window only)."""
        if not self.buf:
            return
        if self.window:
            self.show(self.buf[-1][2], text)
        elif isinstance(self.dash, Dashboard):
            self.dash.frame(self.buf[-1][2])

    def exposure_report(self) -> str:
        return (f"exposure: {self.collapses} collapses, {self.waited:.1f} s waiting for a steady exposure, "
                f"{self.unsteady} shots not steady")

    def show(self, frame: np.ndarray, text: str) -> None:
        if not self.window:
            return
        view = frame.copy()
        h, w = view.shape[:2]
        cv2.line(view, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
        cv2.line(view, (0, h // 2), (w, h // 2), (255, 255, 0), 1)
        cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("jenga autoscan", view)
        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            self.pending_key = k  # kept until key() reads it (previews run while the arm moves)

    def key(self, ms: int = 1) -> int:
        """Key pressed in the camera window since the last call (255 = none)."""
        k = getattr(self, "pending_key", 255)
        self.pending_key = 255
        if k == 255 and self.window:
            k = cv2.waitKey(ms) & 0xFF
        return k

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=2)
        self.cam.release()


class FakeCamera(RealCamera):
    """Synthetic Dino image from where the point in focus is relative to a 'true' tower:
    the model tower with every end face `depth_mm` (+ `lean_mm` per 10 layers) deeper and
    the gaps `drop_mm` lower. Blur grows with the focus error, the gap band moves with
    the height error, every block has its own gap width (seeded)."""

    def __init__(self, mover: Mover, shots: dict[int, list[Shot]], depth_mm: float, lean_mm: float,
                 drop_mm: float, window: bool, roll_deg: float = 0.0, seed: int = 1):
        self.settle = SETTLE_S
        self.dash = NoDashboard()
        self.mover, self.window = mover, window
        self.roll = math.radians(roll_deg)  # Dino turned in the hand, unknown to the scanner
        self.shots = [s for row in shots.values() for s in row]
        self.depth, self.lean, self.drop = depth_mm / 1000, lean_mm / 1000, drop_mm / 1000
        rng = np.random.default_rng(seed)
        self.gap_px = {(s.layer, s.slot): float(rng.uniform(4, 16)) for s in self.shots}
        self.top = 150 + self._texture(rng, (6.0, 1.0), 12)  # long grain of the block above
        self.bottom = 120 + self._texture(rng, (1.0, 1.0), 15)  # end grain
        self.mm_per_px = DEFAULT_MM_PER_PX * 1.3  # "unknown" to the scanner
        self.layer_px = 0.02186 / (self.mm_per_px / 1000)

    @staticmethod
    def _texture(rng, aspect, contrast) -> np.ndarray:
        """Noise at several scales, so the sharpness falls off gradually with blur (like wood)."""
        t = np.zeros((480, 640), np.float32)
        for scale in (0.7, 1.5, 3.0, 6.0):
            n = rng.normal(0, 1, (480, 640)).astype(np.float32)
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=scale * aspect[0], sigmaY=scale * aspect[1])
            t += n / n.std() * contrast
        return t

    def truth(self, shot: Shot) -> tuple[float, float]:
        return self.depth + self.lean * shot.layer / 10, self.drop

    def frames(self, n: int, settle: float | None = None) -> list[np.ndarray]:
        if self.mover.send:  # timed like the real camera on the virtual robot
            time.sleep((self.settle if settle is None else settle) + 0.2)
        return self.render(n)

    def grab(self) -> np.ndarray:
        if self.mover.send:
            time.sleep(1 / FAKE_FPS)
        return self.render(1)[0]

    def render(self, n: int) -> list[np.ndarray]:
        d = self.mover.dino()
        focus = d[:3, 3]
        shot = min(self.shots, key=lambda s: np.linalg.norm(s.aim - focus))
        depth, drop = self.truth(shot)
        face_point = shot.aim - shot.normal * depth
        defocus = float((face_point - focus) @ d[:3, 2])  # + = face is beyond the point in focus
        lateral = float((focus - shot.aim) @ d[:3, 0])
        x_true = d[:3, 0] * math.cos(self.roll) + d[:3, 1] * math.sin(self.roll)
        tilt = -math.tan(math.asin(float(np.clip(-x_true[2], -1, 1))))  # image slope of a level line
        img = self._image(shot, defocus, (focus[2] - (shot.aim[2] - drop)) / (self.mm_per_px / 1000),
                          abs(lateral) < 0.018, tilt)
        rng = np.random.default_rng()
        out = [np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8) for _ in range(n)]
        self.dash.frame(out[-1])
        return out

    def _image(self, shot: Shot, defocus: float, gap_offset_px: float, on_block: bool,
               tilt: float = 0.0) -> np.ndarray:
        h, w = 480, 640
        gray = np.full((h, w), 135.0, np.float32)
        if on_block:
            ys, xs = np.mgrid[0:h, 0:w]
            row = h / 2 + gap_offset_px + tilt * (xs - w / 2)  # camera above the gap -> gap lower
            low = row + self.layer_px  # bottom edge of the same end face
            half = self.gap_px[(shot.layer, shot.slot)] / 2
            gray = np.where(ys < row - half, self.top, np.where(ys < row + half, 15.0,
                            np.where(ys < low - 3, self.bottom, np.where(ys < low + 3, 15.0, self.top))))
            gray = gray.astype(np.float32)
        sigma = 0.3 + 1.5 * abs(defocus) * 1000  # px of blur per mm out of focus
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
        return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def preview(self, text: str = "moving") -> None:
        if self.window:
            self.show(self.frames(1)[0], text)

    def release(self) -> None:
        pass


# --- scan --------------------------------------------------------------------------
@dataclass
class FaceState:
    depth_m: float = 0.0  # last found depth past the model face
    down_m: float = 0.0  # last found gap drop
    roll: float = 0.0  # rad about the optical axis that levels the gap
    height_known: bool = False  # down_m was found on a gap of this face
    depth_known: bool = False  # depth_m was found in focus on this face
    leveled: bool = False  # the turn of the Dino is known for this face
    mm_per_px: float | None = None
    rows: list[dict] = field(default_factory=list)


class Scanner:
    def __init__(self, args, mover: Mover, cam, session: Path, carry: dict | None = None):
        self.args, self.mover, self.cam, self.session = args, mover, cam, session
        self.carry = carry or {}
        cam.settle = args.settle_s
        cam.dash = args.dash
        self.dash = args.dash
        self.face = ""
        self.max_in = args.max_in_mm / 1000
        self.last_depth = None  # depth found on the previous block: the limit follows it
        self.layer_h = json.loads(SETUP_FILE.read_text())["tower"]["layer_h"]
        self.roll = self.carry.get("roll", 0.0)  # the Dino's turn in the hand: the other face's is a start
        self.log = (session / "focus_log.csv").open("w", newline="")
        self.log_w = csv.writer(self.log)
        self.log_w.writerow(["time", "layer", "slot", "screen", "depth_mm", "down_mm", "roll_deg", "focus", "edge_row",
                             "brightness"])

    def measure(self, n: int = 3) -> tuple[float, np.ndarray]:
        frames = self.cam.frames(n)
        return float(np.median([steady_focus(f) for f in frames])), frames[-1]

    def deepest(self) -> float:
        if self.last_depth is None:
            return self.max_in
        return min(self.max_in, self.last_depth + self.args.limit_mm / 1000)

    def go(self, shot: Shot, depth: float, down: float) -> None:
        if depth > self.deepest() + 1e-6:
            raise ScanError(f"refusing to go {depth * 1000:.1f} mm past the model face "
                            f"(limit {self.deepest() * 1000:.1f} mm)")
        self.mover.line_to(shot.pose(depth, down, self.roll))

    def log_sample(self, shot: Shot, d: float, down: float, f: float, frame: np.ndarray) -> None:
        row = edge_row(frame)
        self.log_w.writerow([f"{time.time():.2f}", shot.layer, shot.slot, shot.screen, f"{d * 1000:.2f}",
                             f"{down * 1000:.2f}", f"{math.degrees(self.roll):.2f}", f"{f:.2f}",
                             "" if row is None else f"{row:.0f}", f"{np.median(frame):.0f}"])
        self.log.flush()

    def focus(self, shot: Shot, guess: float, down: float, wide: bool) -> tuple[float, float] | None:
        """(depth, focus value) of the sharpest point near `guess`, and the arm is there;
        None if there is no clear peak. SPACE in the camera window = take this depth, q = abort."""
        a = self.args
        out, inward = (a.search_out_mm, a.search_in_mm) if wide else (a.refocus_mm, a.refocus_mm)

        best = 0.0

        def measure(depth_mm: float) -> float:
            nonlocal best
            d = depth_mm / 1000
            self.go(shot, d, down)
            f, frame = self.measure()
            self.cam.show(frame, f"L{shot.layer:02d} {shot.screen} depth {depth_mm:+.2f} mm focus {f:.1f}")
            self.log_sample(shot, d, down, f, frame)
            best = max(best, f)
            self.dash.event(stage="scan", face=self.face, layer=shot.layer, slot=shot.slot, action="focus", focus=f,
                            peak=best)
            k = self.cam.key()
            if k in (ord("q"), 27):
                raise ScanError("aborted by the user (q)")
            if k == ord(" "):
                raise DepthTaken(d, f)
            return f

        g = guess * 1000
        try:
            p = find_peak(measure, g, g - out, min(g + inward, self.deepest() * 1000), a.step_mm, fall=a.focus_fall,
                          falls=a.focus_falls, min_peak=a.min_focus)
        except DepthTaken as e:
            print(f"   SPACE: depth {e.depth * 1000:+.2f} mm taken")
            return e.depth, e.value
        print(f"   {shot.screen}: {p.samples} focus samples, {'peak' if p.found else 'no peak'} at "
              f"{p.depth:+.2f} mm, value {p.value:.1f} {p.reason}")
        if not p.found:
            return None
        self.go(shot, p.depth / 1000, down)
        return p.depth / 1000, p.value

    def find_edge(self, shot: Shot, depth: float, down: float) -> float | None:
        """Camera height (down) at which the edge is in the image, or None."""
        if edge_seen(self.cam.frames(1)[0]):
            return down
        n = int(self.args.gap_search_mm // EDGE_SEARCH_STEP_MM)
        offs = [sign * k * EDGE_SEARCH_STEP_MM for k in range(1, n + 1) for sign in (-1, 1)]  # up first
        for off in offs:
            self.go(shot, depth, down + off / 1000)
            frame = self.cam.frames(1)[0]
            self.cam.show(frame, f"looking for the edge: {off:+.1f} mm")
            if edge_seen(frame):
                print(f"   edge found {off:+.1f} mm {'lower' if off > 0 else 'higher'}")
                return down + off / 1000
        self.go(shot, depth, down)
        return None

    def top_edge(self, shot: Shot, depth: float, down: float, st: FaceState) -> float | None:
        """Center the edge found, check that it is the TOP edge of the end face (long grain
        above it); a bottom edge -> move up one layer height and look again."""
        down = self.center_edge(shot, depth, down, st)
        frame = self.cam.frames(1)[0]
        row = edge_row(frame)
        if row is None:
            return down
        side = edge_side(frame, row)
        if side >= -EDGE_MARGIN:
            print(f"   top edge (grain stripes above - below = {side:+.2f})")
            return down
        up = self.layer_h
        print(f"   that is the BOTTOM edge (stripes {side:+.2f}): moving up {up * 1000:.1f} mm")
        down -= up
        self.go(shot, depth, down)
        found = self.find_edge(shot, depth, down)
        return None if found is None else self.center_edge(shot, depth, found, st)

    def level(self, shot: Shot, depth: float, down: float, st: FaceState) -> None:
        """Turn the Dino about its axis until the edge is level (the Dino sits a few deg
        turned after every pick)."""
        if self.args.level_once and st.leveled:
            return
        slope = edge_tilt(self.cam.frames(1)[0])
        if slope is None:
            return
        if abs(math.degrees(math.atan(slope))) < LEVEL_DEG:
            st.leveled = True
            return
        before = self.roll
        for sign in (1, -1):  # the sign is checked on the image, not trusted
            self.roll = float(np.clip(before + sign * math.atan(slope), -math.radians(MAX_ROLL_DEG),
                                      math.radians(MAX_ROLL_DEG)))
            self.go(shot, depth, down)
            new = edge_tilt(self.cam.frames(1)[0])
            if new is not None and abs(new) < abs(slope):
                print(f"   edge tilted {math.degrees(math.atan(slope)):+.1f} deg -> roll {math.degrees(self.roll):+.1f} deg "
                      f"(now {math.degrees(math.atan(new)):+.1f})")
                st.roll = self.roll
                st.leveled = True
                return
        self.roll = before
        self.go(shot, depth, down)
        print(f"   edge tilted {math.degrees(math.atan(slope)):+.1f} deg, leveling did not help (kept)")

    def center_edge(self, shot: Shot, depth: float, down: float, st: FaceState) -> float:
        """Move the camera up / down until the edge is near the image centre (measures the
        image scale on the first call)."""
        self.dash.event(action="edge")
        row = edge_row(self.cam.frames(1)[0])
        if row is None:
            return down
        h = 480
        if st.mm_per_px is None:  # measure the scale: move the camera 0.5 mm towards the edge
            test = 0.0005 if row > h / 2 else -0.0005
            self.go(shot, depth, down + test)
            row2 = edge_row(self.cam.frames(1)[0])
            if row2 is not None and abs(row2 - row) >= 5:
                st.mm_per_px = abs(test) * 1000 / abs(row2 - row)
                row, down = row2, down + test
            else:
                st.mm_per_px = DEFAULT_MM_PER_PX
                self.go(shot, depth, down)
            print(f"   image scale {st.mm_per_px * 1000:.1f} um/px")
        for _ in range(4):
            err = row - h / 2  # + = edge below the centre -> camera down
            if abs(err) <= EDGE_TOLERANCE_PX:
                break
            down += float(np.clip(err * st.mm_per_px / 1000, -0.003, 0.003))
            self.go(shot, depth, down)
            row = edge_row(self.cam.frames(1)[0])
            if row is None:
                break
        return down

    def shoot(self, face: str, shot: Shot, depth: float, down: float, focus: float, st: FaceState,
              settle: float = SETTLE_S) -> None:
        self.dash.event(action="shot")
        frames = self.cam.frames(ends.FRAMES_PER_SHOT, settle=settle)
        gaps = [ends.measure_end(f) for f in frames]
        self.record_shot(face, shot, frames[-1], float(np.median([g.gap for g in gaps])), gaps[-1].row, depth, down,
                         focus, st, list(self.mover.q))

    def record_shot(self, face: str, shot: Shot, frame: np.ndarray, gap: float, rim_row: int, depth: float,
                    down: float, focus: float, st: FaceState, q, note: str = "") -> None:
        """Save the image and a shots.csv row (stopped shot, or the best frame of a glide)."""
        number = FACE_NUMBER[face]
        slot = shot.slot
        if self.args.turned:  # the far ends of the blocks are in view, left and right swapped
            number, slot = (number + 1) % 4 + 1, 2 - slot
        key = f"L{shot.layer + 1:02d}:f{number}:{shot.screen}"
        name = f"{key.replace(':', '_')}.png"
        cv2.imwrite(str(self.session / name), frame)
        self.cam.show(frame, f"{key} gap {gap:.1f} px focus {focus:.0f}")
        st.rows.append({"key": key, "gap": round(gap, 2), "face": face, "layer": shot.layer, "slot": slot,
                        "screen": shot.screen, "focus": round(focus, 1), "depth_mm": round(depth * 1000, 2),
                        "down_mm": round(down * 1000, 2), "roll_deg": round(math.degrees(self.roll), 2),
                        "row": rim_row, "image": name, "brightness": round(float(np.median(frame))),
                        "rim_dark": round(rim_darkness(frame), 3),
                        "q": [round(float(v), 5) for v in q]})
        gap_mm = round(gap * (st.mm_per_px or DEFAULT_MM_PER_PX), 3)
        self.dash.shot(layer=shot.layer, slot=slot, face=face, gap_px=gap, gap_mm=gap_mm)
        self.dash.event(action="measured", gap_mm=gap_mm)
        print(f"   SHOT{note} {key} (L{shot.layer:02d}_S{slot}): gap {gap:.1f} px, rim dark {rim_darkness(frame):.2f}, focus {focus:.0f}, "
              f"depth {depth * 1000:+.1f} mm, down {down * 1000:+.1f} mm, roll {math.degrees(self.roll):+.1f} deg")

    def scan_face(self, face: str, rows: dict[int, list[Shot]], far_in, far_out) -> FaceState:
        self.face = face
        st = FaceState(depth_m=self.args.prior_depth_mm / 1000, down_m=self.args.prior_down_mm / 1000,
                       roll=self.roll, mm_per_px=self.carry.get("mm_per_px"))
        self.enter(face, far_in)
        if self.args.blur_test is not None and not self.args.motion_only:
            return self.blur_test(face, rows, far_out, st)
        if self.args.glide and not self.args.motion_only:
            return self.glide_face(face, rows, far_out, st)
        if self.args.sweep and not self.args.motion_only:
            return self.sweep_face(face, rows, far_out, st)
        for layer in sorted(rows, reverse=self.args.top_first) if not self.args.motion_only else []:
            row = rows[layer]
            print(f"\n== {face} L{layer:02d}")
            first = row[0]
            self.mover.line_to(first.pose(st.depth_m - RETREAT_M, st.down_m))
            for k, shot in enumerate(row):
                got = self.lock_on(shot, st, wide=k == 0 or not st.depth_known)
                if got is None:
                    continue
                d, _, f = got
                self.shoot(face, shot, d, st.down_m, f, st)
                self.save(st)
            last = row[-1]
            self.mover.line_to(last.pose(st.depth_m - RETREAT_M, st.down_m))
        self.leave(far_out)
        return st

    def enter(self, face: str, far_in) -> None:
        a = approach(self.args)
        for via in self.args.via:
            self.mover.joint_move(json.loads(POSES_FILE.read_text())[self.mover.arm][via], via, a["speed"],
                                  min_s=a["min_s"])
        self.mover.joint_move(far_in, f"{face} far in", a["speed"], a["slow_end"], a["min_s"])

    def leave(self, far_out) -> None:
        a = approach(self.args)
        self.mover.line_to(np.array(self.mover.chain.fk(far_out, self.mover.tool)))
        for k, via in enumerate(reversed(self.args.via)):
            self.mover.joint_move(json.loads(POSES_FILE.read_text())[self.mover.arm][via], via, a["speed"],
                                  a["slow_start"] if k == 0 else None, a["min_s"])

    def lock_on(self, shot: Shot, st: FaceState, wide: bool) -> tuple[float, float, float] | None:
        """Focus on `shot`, find (first time on the face) and centre the edge, level the Dino:
        (depth, down, focus value) with the arm there, or None (skipped)."""
        found = self.focus(shot, st.depth_m, st.down_m, wide)
        if found is None and not wide:
            found = self.focus(shot, st.depth_m, st.down_m, True)
        if found is None:
            print(f"   {shot.screen}: no focus, skipped")
            return None
        d, f = found
        st.depth_m, st.depth_known = d, True
        self.last_depth = d
        down = focused_at = st.down_m
        if not st.height_known:  # first edge of the face: search up / down for it
            down = self.find_edge(shot, d, st.down_m)
            if down is None:
                print(f"   {shot.screen}: in focus (depth {d * 1000:+.1f} mm) but no edge within "
                      f"{self.args.gap_search_mm:.0f} mm up/down, skipped")
                return None
            down = self.top_edge(shot, d, down, st)
            if down is None:
                return None
            st.height_known = True
        if edge_seen(self.cam.frames(1)[0]):
            roll_at = self.roll
            down = self.center_edge(shot, d, down, st)
            self.level(shot, d, down, st)
            down = self.center_edge(shot, d, down, st)
            if abs(down - focused_at) > REFOCUS_MOVE_M or self.roll != roll_at:  # other part of the face in view
                found = self.focus(shot, d, down, False)
                if found is not None:
                    d, f = found
                    st.depth_m = d
            self.go(shot, d, down)
        st.down_m = down
        return d, down, f

    def sweep_face(self, face: str, rows: dict[int, list[Shot]], far_out, st: FaceState) -> FaceState:
        """--sweep: the full search (lock_on) only on the middle block of the first and the last
        layer; the other shots go straight to the depth and edge height on the line between
        those two, without a search, row by row back and forth. The edge is centred again only
        when it is far from the image centre."""
        layers = sorted(rows, reverse=self.args.top_first)
        normal = rows[layers[0]][0].normal
        ref: dict[int, tuple[float, float]] = {}

        def shots(layer: int, start_mid: bool) -> list[Shot]:
            """The row from the end nearest to the arm (back and forth); a reference row
            starts at its middle block, where the arm is."""
            row = rows[layer]
            if start_mid:
                mid = row[len(row) // 2]
                return [mid] + [s for s in row if s is not mid]
            at = self.mover.dino()[:3, 3]
            return row if np.linalg.norm(row[0].aim - at) <= np.linalg.norm(row[-1].aim - at) else row[::-1]

        def reference(layer: int) -> bool:
            mid = rows[layer][len(rows[layer]) // 2]
            print(f"\n== {face} L{layer:02d} reference: {mid.screen}")
            if st.rows:
                self.back_off(normal, RETREAT_M)
            self.mover.line_to(mid.pose(st.depth_m - RETREAT_M, st.down_m, self.roll))
            got = self.lock_on(mid, st, wide=not (self.args.narrow_ref2 and ref))  # near the first one's depth
            if got is None:
                return False
            ref[layer] = got[:2]
            self.shoot_row(face, layer, shots(layer, True), got[0], got[1], got[2], st)
            return True

        if not reference(layers[0]):
            raise ScanError(f"no reference on L{layers[0]:02d}: the sweep needs it")
        if len(layers) > 1 and not reference(layers[-1]):
            print(f"   no reference on L{layers[-1]:02d}: the depth and height of L{layers[0]:02d} are used for all")
        (l0, (d0, h0)), (l1, (d1, h1)) = sorted(ref.items())[0], sorted(ref.items())[-1]
        self.last_depth = max(d0, d1)
        for layer in layers[-2:0:-1]:
            u = 0.0 if l1 == l0 else (layer - l0) / (l1 - l0)
            d, down = d0 + (d1 - d0) * u, h0 + (h1 - h0) * u
            print(f"\n== {face} L{layer:02d} sweep: depth {d * 1000:+.1f} mm, down {down * 1000:+.1f} mm")
            self.back_off(normal, self.args.sweep_retreat_mm / 1000)
            row = shots(layer, False)
            self.mover.line_to(row[0].pose(d - self.args.sweep_retreat_mm / 1000, down, self.roll))
            self.shoot_row(face, layer, row, d, down, None, st)
        self.back_off(normal, RETREAT_M)
        self.leave(far_out)
        return st

    def blur_test(self, face: str, rows: dict[int, list[Shot]], far_out, st: FaceState) -> FaceState:
        """--blur-test LAYER: stopped shots of the layer's blocks, then passes without stopping
        (along the layer at --blur-h-speeds, up / down over the middle block at --blur-v-speeds)
        with every frame measured; blur_frames.csv, blur_summary.json, every 4th frame as jpg."""
        a = self.args
        layer = a.blur_test
        if layer not in rows:
            raise ScanError(f"L{layer:02d} has no end faces on {face} (have {sorted(rows)})")
        if max(a.blur_h_speeds + a.blur_v_speeds, default=0.0) > BLUR_MAX_MM_S:
            raise ScanError(f"passes in front of the tower are limited to {BLUR_MAX_MM_S:.0f} mm/s")
        row = rows[layer]
        mid = row[len(row) // 2]
        print(f"\n== {face} L{layer:02d} blur test: reference on the {mid.screen} block")
        self.mover.line_to(mid.pose(st.depth_m - RETREAT_M, st.down_m))
        got = self.lock_on(mid, st, wide=True)
        if got is None:
            raise ScanError("no focus / edge on the middle block: the blur test needs it")
        d, down, f = got
        self.shoot_row(face, layer, [mid] + [s for s in row if s is not mid], d, down, f, st)
        stopped = {r["screen"]: r["gap"] for r in st.rows}
        right = row[-1].aim - row[0].aim
        right /= np.linalg.norm(right)
        centres = {s.screen: float((s.aim - mid.aim) @ right) * 1000 for s in row}
        frames_dir = self.session / "blur_frames"
        frames_dir.mkdir(exist_ok=True)
        records: list[dict] = []

        def shifted(pose: np.ndarray, along_m: float) -> np.ndarray:
            p = pose.copy()
            p[:3, 3] += right * along_m
            return p

        def record(kind: str, v: float, goal: np.ndarray) -> None:
            print(f"   {kind} pass at {v:.0f} mm/s ...")
            recs = self.record_pass(goal, v, mid.aim, right, lag_s=0.0)
            for r in recs:
                k = len(records)
                image = ""
                if k % 4 == 0:
                    image = f"{kind}_{v:02.0f}_{k:04d}.jpg"
                    cv2.imwrite(str(frames_dir / image), r.pop("frame"))
                r.pop("frame", None)
                records.append({"kind": kind, "speed": v, **r, "image": image})
            print(f"   {len(recs)} frames")

        start, end = shifted(row[0].pose(d, down, self.roll), -BLUR_MARGIN_M), \
            shifted(row[-1].pose(d, down, self.roll), BLUR_MARGIN_M)
        if a.blur_h_speeds:
            self.mover.line_to(start)
        for k, v in enumerate(a.blur_h_speeds):  # back and forth
            record("along", v, end if k % 2 == 0 else start)
        over = max(row, key=lambda s: stopped.get(s.screen, -1.0))  # up / down where the gap is widest
        if a.blur_v_speeds:
            print(f"   up / down passes over the {over.screen} block (stopped gap {stopped.get(over.screen)} px)")
        self.go(over, d, down)
        top, low = over.pose(d, down - BLUR_V_SPAN_M, self.roll), over.pose(d, down + BLUR_V_SPAN_M, self.roll)
        self.mover.line_to(top)
        for k, v in enumerate(a.blur_v_speeds):
            record("updown", v, low if k % 2 == 0 else top)
        self.go(mid, d, down)
        self.back_off(mid.normal, RETREAT_M)
        self.leave(far_out)

        summary = blur_summary(records, stopped, centres, over.screen)
        with (self.session / "blur_frames.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        (self.session / "blur_summary.json").write_text(json.dumps(summary, indent=2))
        print_blur_summary(summary)
        return st

    def record_pass(self, goal: np.ndarray, v: float, origin: np.ndarray, along: np.ndarray,
                    lag_s: float = GLIDE_LAG_S) -> list[dict]:
        """Straight move to `goal` at `v` mm/s without stopping, every camera frame measured:
        [{t, along_mm (from `origin` along `along`, minus what the arm moved during the camera
        lag), up_mm, gap, rim_row, edge_row, frame, q}]."""
        recs: list[dict] = []
        direction = float((goal[:3, 3] - self.mover.dino()[:3, 3]) @ along)
        shift = math.copysign(v * lag_s, direction) if direction else 0.0

        def grab() -> None:
            frame = self.cam.grab()
            q = self.mover.ros.current() if self.mover.ros is not None else list(self.mover.q)
            at = self.mover.chain.fk(np.array(q), self.mover.tool)[:3, 3]
            g = ends.measure_end(frame)
            er = edge_row(frame)
            recs.append({"t": round(time.time(), 3), "along_mm": round(float((at - origin) @ along) * 1000 - shift, 2),
                         "up_mm": round(float(at[2] - origin[2]) * 1000, 2), "gap": round(g.gap, 2),
                         "rim_row": g.row, "edge_row": "" if er is None else round(er), "frame": frame, "q": q})

        self.mover.idle = grab
        try:
            self.mover.line_to(goal, speed=v)
        finally:
            self.mover.idle = None
        return recs

    def glide_face(self, face: str, rows: dict[int, list[Shot]], far_out, st: FaceState) -> FaceState:
        """--glide: the depth and edge height like --sweep (full search only on the middle block of
        the first and the last layer, the line between them for the rest), but every layer is shot
        in one pass along it without stopping (moving along the gap does not blur its width, blur
        test 2026-11-10); per block the median gap of the frames near its centre."""
        if self.args.glide_speed > BLUR_MAX_MM_S:
            raise ScanError(f"--glide-speed is limited to {BLUR_MAX_MM_S:.0f} mm/s")
        layers = sorted(rows, reverse=self.args.top_first)
        normal = rows[layers[0]][0].normal
        ref: dict[int, tuple[float, float]] = {}
        frames_log: list[dict] = []

        def reference(layer: int) -> bool:
            mid = rows[layer][len(rows[layer]) // 2]
            print(f"\n== {face} L{layer:02d} reference: {mid.screen}")
            if st.rows:
                self.back_off(normal, RETREAT_M)
            self.mover.line_to(mid.pose(st.depth_m - RETREAT_M, st.down_m, self.roll))
            got = self.lock_on(mid, st, wide=True)
            if got is None:
                return False
            ref[layer] = got[:2]
            self.glide_row(face, layer, rows[layer], got[0], got[1], st, frames_log)
            return True

        if not reference(layers[0]):
            raise ScanError(f"no reference on L{layers[0]:02d}: the glide needs it")
        if len(layers) > 1 and not reference(layers[-1]):
            print(f"   no reference on L{layers[-1]:02d}: the depth and height of L{layers[0]:02d} are used for all")
        (l0, (d0, h0)), (l1, (d1, h1)) = sorted(ref.items())[0], sorted(ref.items())[-1]
        self.last_depth = max(d0, d1)
        for layer in layers[-2:0:-1]:
            u = 0.0 if l1 == l0 else (layer - l0) / (l1 - l0)
            d, down = d0 + (d1 - d0) * u, h0 + (h1 - h0) * u
            print(f"\n== {face} L{layer:02d} glide: depth {d * 1000:+.1f} mm, down {down * 1000:+.1f} mm")
            self.back_off(normal, self.args.sweep_retreat_mm / 1000)
            self.glide_row(face, layer, rows[layer], d, down, st, frames_log, out_m=self.args.sweep_retreat_mm / 1000)
        self.back_off(normal, RETREAT_M)
        self.leave(far_out)
        if frames_log:
            with (self.session / "glide_frames.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(frames_log[0]))
                w.writeheader()
                w.writerows(frames_log)
        return st

    def glide_row(self, face: str, layer: int, row: list[Shot], depth: float, down: float, st: FaceState,
                  frames_log: list[dict], out_m: float = 0.0) -> None:
        """One pass along the layer from the end nearest to the arm, then one shots.csv row per
        block; a block with too few good frames is shot stopped instead."""
        along = row[-1].aim - row[0].aim
        along = along / np.linalg.norm(along)
        mid = row[len(row) // 2]

        def beyond(shot: Shot, sign: float, out: float) -> np.ndarray:
            pose = shot.pose(depth - out, down, self.roll)
            pose[:3, 3] += along * sign * BLUR_MARGIN_M
            return pose

        at = self.mover.dino()[:3, 3]
        ends_ = [(row[0], -1.0), (row[-1], 1.0)]
        if np.linalg.norm(row[-1].aim - at) < np.linalg.norm(row[0].aim - at):
            ends_.reverse()
        (s0, g0), (s1, g1) = ends_
        if out_m:  # come in along the face out of reach, then in
            self.mover.line_to(beyond(s0, g0, out_m))
        self.mover.line_to(beyond(s0, g0, 0.0))
        v = self.args.glide_speed
        recs = self.record_pass(beyond(s1, g1, 0.0), v, mid.aim, along)
        for r in recs:
            frames_log.append({"layer": layer, "speed": v, **{k: x for k, x in r.items() if k not in ("frame", "q")}})
        for shot in row:
            c = float((shot.aim - mid.aim) @ along) * 1000
            good = [r for r in recs if r["edge_row"] != "" and abs(r["edge_row"] - 240) <= BLUR_EDGE_PX
                    and abs(r["along_mm"] - c) <= BLUR_WINDOW_MM]
            if len(good) < GLIDE_MIN_FRAMES:
                print(f"   {shot.screen}: {len(good)} good frames in the pass, shooting it stopped")
                self.shoot_row(face, layer, [shot], depth, down, None, st)
                continue
            best = min(good, key=lambda r: abs(r["along_mm"] - c))
            gap = float(np.median([r["gap"] for r in good]))
            self.record_shot(face, shot, best["frame"], gap, best["rim_row"], depth, down,
                             steady_focus(best["frame"]), st, best["q"], note=f" (glide, {len(good)} frames)")
        self.save(st)

    def back_off(self, normal: np.ndarray, m: float) -> None:
        """Straight out of the face by `m` before moving along it."""
        goal = self.mover.dino().copy()
        goal[:3, 3] += normal * m
        self.mover.line_to(goal)

    def shoot_row(self, face: str, layer: int, row: list[Shot], depth: float, down: float, focus: float | None,
                  st: FaceState) -> None:
        """Shoot `row` at one depth and edge height; the first shot of a reference row is
        where the arm already is."""
        for k, shot in enumerate(row):
            if not (k == 0 and focus is not None):
                self.go(shot, depth, down)
            self.dash.event(action="shot")
            frames = self.cam.frames(ends.FRAMES_PER_SHOT, settle=self.args.shot_settle_s)
            f = steady_focus(frames[-1]) if focus is None or k else focus
            row_px = edge_row(frames[-1])
            if row_px is not None and abs(row_px - 240) <= self.args.sweep_edge_px:  # these frames are the shot
                gaps = [ends.measure_end(x) for x in frames]
                self.record_shot(face, shot, frames[-1], float(np.median([g.gap for g in gaps])), gaps[-1].row,
                                 depth, down, f, st, list(self.mover.q))
            else:
                print(f"   {shot.screen}: edge {'not seen' if row_px is None else f'at row {row_px:.0f}'}, centring")
                found = down if row_px is not None else self.find_edge(shot, depth, down)
                if found is None:
                    print(f"   {shot.screen}: no edge, shot anyway")
                else:
                    down = self.center_edge(shot, depth, found, st)  # the next blocks of this row are as high
                    self.go(shot, depth, down)
                self.shoot(face, shot, depth, down, f, st, settle=self.args.shot_settle_s)
            self.save(st)

    def save(self, st: FaceState) -> None:
        with (self.session / "shots.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(st.rows[0]))
            w.writeheader()
            for r in st.rows:
                w.writerow({**r, "q": json.dumps(r["q"])})
        with (self.session / "ends.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "gap"])
            w.writerows([r["key"], f"{r['gap']:.1f}"] for r in st.rows)


def blur_summary(records: list[dict], stopped: dict[str, float], centres: dict[str, float],
                 updown_block: str = "middle") -> list[dict]:
    """Per pass and block: median gap px of the frames that see the edge near the image centre
    (along: within BLUR_WINDOW_MM of the block centre), against the stopped shot."""
    out = []
    for kind, v in dict.fromkeys((r["kind"], r["speed"]) for r in records):
        rs = [r for r in records if r["kind"] == kind and r["speed"] == v and r["edge_row"] != ""
              and abs(r["edge_row"] - 240) <= BLUR_EDGE_PX]
        blocks = centres if kind == "along" else {updown_block: centres[updown_block]}
        for screen, c in blocks.items():
            gaps = [r["gap"] for r in rs if kind != "along" or abs(r["along_mm"] - c) <= BLUR_WINDOW_MM]
            med = float(np.median(gaps)) if gaps else None
            out.append({"kind": kind, "speed": v, "block": screen, "frames": len(gaps),
                        "gap_median": None if med is None else round(med, 2),
                        "gap_spread": round(float(np.percentile(gaps, 90) - np.percentile(gaps, 10)), 2) if gaps else None,
                        "stopped": stopped.get(screen),
                        "diff": None if med is None or screen not in stopped else round(med - stopped[screen], 2)})
    return out


def print_blur_summary(summary: list[dict]) -> None:
    print("\nblur test: gap px moving vs stopped (median of the frames with the edge near the centre)")
    print(f"{'pass':7s} {'mm/s':>4s}  {'block':6s} {'frames':>6s} {'moving':>7s} {'10-90%':>7s} {'stopped':>7s} {'diff':>6s}")
    for s in summary:
        fmt = lambda x: "-" if x is None else f"{x:.1f}"
        print(f"{s['kind']:7s} {s['speed']:4.0f}  {s['block']:6s} {s['frames']:6d} {fmt(s['gap_median']):>7s} "
              f"{fmt(s['gap_spread']):>7s} {fmt(s['stopped']):>7s} {fmt(s['diff']):>6s}")


# --- ranking -----------------------------------------------------------------------
def rank(sessions: list[Path], mode: str, out: Path, candidates: Path | None = None) -> list[dict]:
    setup = json.loads(SETUP_FILE.read_text())
    tower = Tower(setup["tower"], setup.get("aim", 0.5))
    shots: dict[str, float] = {}
    for s in sessions:
        with (s / "ends.csv").open() as f:
            shots.update({r["key"]: float(r["gap"]) for r in csv.DictReader(f)})
    old_layers = sorted({int(k[1:3]) for k in shots})
    order = ends.ranking(shots, old_layers, 2, mode)
    result = []
    for n, (score, old, block) in enumerate(order, 1):
        layer = old - 1
        slot = block_slot(tower, layer, block)
        result.append({"rank": n, "block": f"jenga_L{layer:02d}_S{slot}", "layer": layer, "slot": slot,
                       "score_px": round(float(score), 2), "name_jenga_ends": f"L{old:02d} {block}"})
    out.write_text(json.dumps({"mode": mode, "sessions": [str(s) for s in sessions],
                               "note": "easiest first; block = teammate frame jenga_LXX_SY (L00 bottom)",
                               "ranking": result}, indent=2))
    (candidates or out.parent / "jenga_pick_candidates.json").write_text(json.dumps({
        "version": 1, "source": "camera", "stamp": time.time(),
        "candidates": [{"block": r["block"], "score": r["score_px"], "reason": f"jenga-ends rank {r['rank']}"}
                       for r in result]}, indent=1))
    return result


# --- pick order (빼안빼 rule) -------------------------------------------------------
def load_gaps(sessions: list[Path]) -> dict[tuple[int, int], tuple[float, str]]:
    """{(teammate layer, slot): (gap px, face)}; a later session wins on the same block."""
    gaps: dict[tuple[int, int], tuple[float, str]] = {}
    for s in sessions:
        with (s / "shots.csv").open() as f:
            for r in csv.DictReader(f):
                gaps[(int(r["layer"]), int(r["slot"]))] = (float(r["gap"]), r["face"])
    return gaps


def layer_margin(px: dict[tuple[int, int], float], layer: int) -> tuple[float, int] | None:
    """(middle px - wider side px, block to pull first) of a layer, None if it can't be compared.
    Margin >= 0: 안빼안, the middle first. Otherwise 빼안빼, the wider side first."""
    sides = {s: px[(layer, s)] for s in (0, 2) if (layer, s) in px}
    if (layer, 1) not in px or not sides:
        return None
    side = max(sides, key=sides.get)
    margin = px[(layer, 1)] - sides[side]
    return margin, 1 if margin >= 0 else side


def pick_order(px: dict[tuple[int, int], float], mode: str = "margin", prefer=None) -> list[tuple[int, int, str]]:
    """Blocks to pull: [(layer, slot, why)]. With the 빼안빼 rule each layer is 안빼안 or 빼안빼:
    the middle finishes its layer, a side is followed at once by the opposite side.
      margin  layers with the clearest middle / side difference first (default)
      rule    widest gap first, with the rule
      px      widest gap first, every block, without the rule
    Ties: upper layer, then the middle block. `prefer(block)` breaks ties first (to find the
    luckiest and unluckiest tie order)."""
    tie = prefer if prefer else lambda b: 0
    if mode == "margin":
        rows = [(m, layer, first) for layer in {b[0] for b in px} if (found := layer_margin(px, layer))
                for m, first in [found]]
        rows.sort(key=lambda r: (-abs(r[0]), tie((r[1], r[2])), -r[1]))
        firsts = [(layer, first) for _, layer, first in rows]
    else:
        firsts = sorted(px, key=lambda b: (-px[b], tie(b), -b[0], b[1] != 1, b[1]))
    if mode == "px":
        return [(*b, "gap") for b in firsts]
    seq: list[tuple[int, int, str]] = []
    done: set[int] = set()
    for layer, slot in firsts:
        if layer in done:
            continue
        done.add(layer)
        seq.append((layer, slot, "gap"))
        if slot != 1:
            seq.append((layer, 2 - slot, "opposite"))
    return seq


def parse_truth(words: list[str]) -> dict[tuple[int, int], bool]:
    """"5=빼안빼" (floor = file layer, 1 = bottom; left middle right as seen facing the
    scanned face) -> {(teammate layer, slot): pulls}."""
    truth = {}
    for w in words:
        floor, marks = w.split("=")
        if len(marks) != 3 or set(marks) - set("빼안"):
            raise SystemExit(f"--truth {w}: expected like 5=빼안빼")
        for slot, mark in enumerate(marks):
            truth[(int(floor) - 1, slot)] = mark == "빼"
    return truth


def streak(seq: list[tuple[int, int, str]], truth: dict[tuple[int, int], bool]) -> int:
    n = 0
    for layer, slot, _ in seq:
        if not truth[(layer, slot)]:
            break
        n += 1
    return n


def seq_text(seq: list[tuple[int, int, str]], truth: dict[tuple[int, int], bool]) -> str:
    names = "LMR"
    return " ".join(f"L{layer + 1:02d}{names[slot]}{'o' if truth[(layer, slot)] else 'X'}" for layer, slot, _ in seq)


def order_main(sessions: list[Path], out: Path | None) -> None:
    gaps = load_gaps(sessions)
    px = {b: g for b, (g, _) in gaps.items()}
    seq = pick_order(px)
    names = ("left", "middle", "right")
    scores = {}
    for n, (layer, slot, why) in enumerate(seq, 1):
        margin = layer_margin(px, layer)[0]
        scores[(layer, slot)] = abs(margin)
        measured = gaps[(layer, slot)] if (layer, slot) in gaps else (None, gaps[(layer, 2 - slot)][1])
        shown = "not measured" if measured[0] is None else f"{measured[0]:.1f} px"
        note = "  <- opposite side" if why == "opposite" else ""
        note += "  (difference too small to trust)" if abs(margin) <= UNCERTAIN_PX else ""
        print(f"{n:3d}  jenga_L{layer:02d}_S{slot}  floor {layer + 1:2d} {names[slot]:6s} (facing {measured[1]})  "
              f"{shown}, difference {abs(margin):.1f}{note}")
    unranked = sorted({layer for layer, _ in px} - {layer for layer, _, _ in seq})
    if unranked:
        print("not ranked, shots missing: floor " + ", ".join(str(layer + 1) for layer in unranked))
    if out:
        out.write_text(json.dumps({
            "version": 1, "source": "camera", "stamp": time.time(),
            "candidates": [{"block": f"jenga_L{layer:02d}_S{slot}", "score": scores[(layer, slot)],
                            "reason": f"빼안빼 rule {n} ({why})"} for n, (layer, slot, why) in enumerate(seq, 1)]},
            indent=1))
        print(f"wrote {out}")


def compare_main(a: Path, b: Path) -> None:
    """Two scans of one tower, block by block, and whether they lead to the same pulls."""
    ga, gb = load_gaps([a]), load_gaps([b])
    print(f"A = {a}\nB = {b}")
    diffs = []
    for block in sorted(ga.keys() | gb.keys()):
        va, vb = ga.get(block, (None,))[0], gb.get(block, (None,))[0]
        if va is not None and vb is not None:
            diffs.append(abs(va - vb))
        print(f"floor {block[0] + 1:2d} {'LMR'[block[1]]}  A {'-' if va is None else f'{va:5.1f}'}  "
              f"B {'-' if vb is None else f'{vb:5.1f}'}")
    pa, pb = {k: v[0] for k, v in ga.items()}, {k: v[0] for k, v in gb.items()}
    layers = sorted({k[0] for k in pa} | {k[0] for k in pb})
    same = trusted = trusted_same = 0
    for layer in layers:
        ma, mb = layer_margin(pa, layer), layer_margin(pb, layer)
        if ma and mb:
            agree = (ma[1] == 1) == (mb[1] == 1)
            same += agree
            if abs(ma[0]) > UNCERTAIN_PX and abs(mb[0]) > UNCERTAIN_PX:
                trusted += 1
                trusted_same += agree
    print(f"{len(diffs)} blocks in both: median difference {np.median(diffs):.1f} px, largest {max(diffs):.1f} px")
    print(f"same pattern (안빼안 or 빼안빼) on {same} of {len(layers)} layers; "
          f"where both differences are clear: {trusted_same} of {trusted}")
    seq_a, seq_b = pick_order(pa)[:8], pick_order(pb)[:8]
    print("first 8 of A: " + " ".join(f"F{x[0] + 1}{'LMR'[x[1]]}" for x in seq_a))
    print("first 8 of B: " + " ".join(f"F{x[0] + 1}{'LMR'[x[1]]}" for x in seq_b))


def score_main(sessions: list[Path], words: list[str], top: int) -> None:
    gaps = load_gaps(sessions)
    truth = parse_truth(words)
    gaps = {b: v for b, v in gaps.items() if b in truth}
    if not gaps:
        raise SystemExit("no scanned block has a truth")
    base = [truth[b] for b in gaps]
    print(f"{sum(base)} of {len(base)} scanned blocks pull")
    faces = sorted({face for _, face in gaps.values()})
    for name, use in [*((f, [f]) for f in faces), *([("merged", faces)] if len(faces) > 1 else [])]:
        px = {b: g for b, (g, face) in gaps.items() if face in use}
        for mode in ("px", "rule", "margin"):
            seq = pick_order(px, mode)
            best = streak(pick_order(px, mode, lambda b: not truth[b]), truth)
            worst = streak(pick_order(px, mode, lambda b: truth[b]), truth)
            hits = sum(truth[(layer, slot)] for layer, slot, _ in seq[:top])
            print(f"{name:6s} {mode:6s}  streak {streak(seq, truth)} (ties {worst}-{best}), "
                  f"top {min(top, len(seq))}: {hits} pull  {seq_text(seq, truth)}")


def order_for_page(gaps: dict) -> list[dict]:
    return [{"layer": layer, "slot": slot} for layer, slot, _ in pick_order({b: g for b, (g, _) in gaps.items()})]


@contextmanager
def video_during_task(args):
    """The Dino's video on the dashboard while the arm picks or places it (no camera otherwise)."""
    stop = threading.Event()
    thread = None
    if isinstance(args.dash, Dashboard) and args.send and not args.fake:
        def run() -> None:
            from jenga_vision.camera import DinoCamera
            try:
                cam = DinoCamera(args.device)
            except Exception as e:  # no video is no reason to stop the robot
                print(f"   no video during this task: {e}")
                return
            try:
                while not stop.is_set():
                    args.dash.frame(cam.read())
            finally:
                cam.release()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread:
            thread.join(timeout=3)


# --- everything in one go ------------------------------------------------------------
def run_main(args) -> None:
    """Pick the Dino, scan the faces, write the pull order, place the Dino. Any stop leaves
    the Dino in the hand and the arm where it is: placing it from there would not be safe."""
    from jenga_vision import task
    t0 = time.time()
    times: dict[str, float] = {}
    task_args = argparse.Namespace(send=args.send, arm_speed=args.task_arm_speed, hand_speed=args.task_hand_speed,
                                   step=args.step, start=1)

    def timed_task(name: str) -> bool:
        print(f"\n===== {name} =====")
        t = time.time()
        args.dash.event(stage="pick" if name.startswith("pick") else "place")
        with video_during_task(args):
            ok = task.run_task(f"{name}_quick", task_args)  # the robot-checked quick pick / place (2026-11-09)
        times[name] = time.time() - t
        return ok

    def report() -> None:
        parts = [f"{k} {v / 60:.1f} min" for k, v in times.items()]
        print(f"\ntimes: {', '.join(parts)}; total {(time.time() - t0) / 60:.1f} min")

    if args.motion_only:  # its empty sessions apart from the real scans
        cache = args.out / ".plan_cache.json"
        args.out = args.out / "motion_only"
        args.out.mkdir(parents=True, exist_ok=True)
        if cache.exists() and not (args.out / cache.name).exists():
            shutil.copy(cache, args.out / cache.name)
    setup = json.loads(SETUP_FILE.read_text())
    tower = Tower(setup["tower"], setup.get("aim", 0.5))
    layers = range(args.layers[0], args.layers[1] + 1)
    faces = [(f, s) for f, s in RUN_FACES if s["face"] in args.faces]
    args.planned_total = sum(len(r) for f, _ in faces for r in face_shots(tower, f, layers).values())
    args.dash.event(total=args.planned_total)

    if args.send:
        steps = ([] if args.no_pick else ["pick_dino"]) + [f for f, _ in faces] + ([] if args.no_place else ["place_dino"])
        print("the robot will run: " + " -> ".join(steps))
        print(f"layers {args.layers[0]}..{args.layers[1]}, dial 40, jenga-view and jenga-arm closed, arm at home")
        if input("Enter = start, q = quit > ").strip().lower() == "q":
            return
        t0 = time.time()
    if not args.no_pick and not timed_task("pick_dino"):
        print("stopped in pick_dino: nothing else runs")
        args.dash.event(stage="stopped", ended=time.time())
        return report()

    sessions = []
    args.camera_holder = {} if args.keep_camera and args.send and not args.fake else None
    args.carry = {} if args.carry_dino else None
    for face, setting in faces:
        print(f"\n===== scan {face} =====")
        scan_args = argparse.Namespace(**{**vars(args), **setting, "no_prompt": True})
        if args.joint_speed is not None:
            scan_args.joint_speed = args.joint_speed
        t = time.time()
        session, finished = scan_main(scan_args)
        times[face] = time.time() - t
        if session:
            sessions.append(session)
        if not finished:
            if args.camera_holder and args.camera_holder.get("cam"):
                args.camera_holder["cam"].release()
            print(f"\nstopped in the {face} scan: the Dino stays in the hand. Move the arm clear of the tower, then "
                  "place the Dino with jenga-task place_dino.")
            if sessions:
                print("order of what was scanned: uv run jenga-autoscan order " + " ".join(map(str, sessions)))
            args.dash.event(stage="stopped", ended=time.time())
            return report()

    if not args.motion_only and args.blur_test is None:  # no scan: keep the last pull order (also the teammate's copy)
        print("\n===== pull order =====")
        out = args.out / time.strftime("order_%Y%m%d_%H%M%S.json")
        order_main(sessions, out)
        state_dir = os.environ.get("JENGA_STATE_DIR")
        if state_dir:
            shutil.copy(out, Path(state_dir) / "jenga_pick_candidates.json")
            print(f"copied to {state_dir}/jenga_pick_candidates.json")
        print("sessions: " + " ".join(map(str, sessions)))
        args.dash.event(order=order_for_page(load_gaps(sessions)))

    if args.camera_holder and args.camera_holder.get("cam"):
        args.camera_holder["cam"].release()
    if not args.no_place and not timed_task("place_dino"):
        print("stopped in place_dino: the pull order above is still valid")
    args.dash.event(stage="done", ended=time.time())
    report()


# --- main --------------------------------------------------------------------------
def planned_ends(args, tool, tower, face: str, layers: range, seed, plan_setup: dict) -> tuple[list, list]:
    """Joints of the first and last planned point of a face. The plan takes 24 s on -y and only
    depends on these inputs, so it is kept in <out>/.plan_cache.json."""
    key = hashlib.sha256(json.dumps([args.arm, np.asarray(tool).tolist(), plan_setup, face, list(layers), list(seed)],
                                    sort_keys=True).encode()).hexdigest()
    cache_file = args.out / ".plan_cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    if key in cache:
        print("   plan from the cache")
        return cache[key]
    pts, _ = plan_face_stable(Chain(args.arm), tool, tower, face, layers, seed, plan_setup)
    for note in away_from_limits(pts, Chain(args.arm)):
        print(f"   {note}")
    cache[key] = [[float(v) for v in pts[0]["q"]], [float(v) for v in pts[-1]["q"]]]
    args.out.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))
    return cache[key]


def scan_main(args) -> tuple[Path | None, bool]:
    """(session folder, finished without a stop)."""
    setup = json.loads(SETUP_FILE.read_text())
    face = args.face.replace("p", "+").replace("m", "-")
    if face not in FACE_NORMALS:
        raise SystemExit("--face: px, py, mx or my")
    tower = Tower(setup["tower"], setup.get("aim", 0.5))
    tool = arm_tool(setup["dino"], args.arm)
    if not setup["dino"].get("tool_hand"):
        raise SystemExit("Dino not calibrated (run jenga-calib first)")
    layers = range(args.layers[0], args.layers[1] + 1)
    rows = face_shots(tower, face, layers)
    if not rows:
        raise SystemExit(f"no end faces of layers {args.layers} on face {face}")
    poses = json.loads(POSES_FILE.read_text())[args.arm]
    seed = poses[args.seed]
    plan_setup = dict(setup, dino=dict(setup["dino"], standoff_m=0.0))
    print(f"planning {face} layers {args.layers[0]}..{args.layers[1]} (IK seed '{args.seed}') ...")
    first_q, last_q = planned_ends(args, tool, tower, face, layers, seed, plan_setup)
    far_in, far_out = last_q if args.top_first else first_q, last_q

    last_file = args.out / LAST_START_FILE
    if args.prior_from_last and last_file.exists():
        last = json.loads(last_file.read_text()).get(face)
        if last and abs(last["down_mm"] - args.prior_down_mm) > LAST_START_MAX_OFF_MM:
            print(f"   the last {face} scan ({last['session']}) found its edge {last['down_mm']:+.1f} mm down, more than "
                  f"{LAST_START_MAX_OFF_MM:.0f} mm from {args.prior_down_mm:+.1f}: probably another layer, not used")
            last = None
        if last:
            print(f"   start like the last {face} scan ({last['session']}): depth {last['depth_mm']:+.1f} mm, "
                  f"down {last['down_mm']:+.1f} mm (instead of {args.prior_depth_mm:+.1f}, {args.prior_down_mm:+.1f})")
            args.prior_depth_mm, args.prior_down_mm = last["depth_mm"], last["down_mm"]
    session = args.out / time.strftime("autoscan_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    mover = Mover(args.arm, tool, args.send, poses[args.start], args.joint_speed, args.line_speed)
    mover.far_speed, mover.parallel_speed = args.far_speed, args.parallel_speed
    mover.line_accel = args.line_accel
    first_shot = next(iter(rows.values()))[0]
    mover.face = (first_shot.aim, first_shot.normal)
    fake = args.fake or not args.send
    holder = getattr(args, "camera_holder", None)  # run --keep-camera: one Dino reader for all faces
    if fake:
        cam = FakeCamera(mover, rows, args.fake_depth_mm, args.fake_lean_mm, args.fake_drop_mm, args.window,
                         args.fake_roll_deg)
    elif holder is not None and holder.get("cam"):
        cam = holder["cam"]
    else:
        cam = RealCamera(args.device, args.window)
        if holder is not None:
            holder["cam"] = cam
    keep = holder is not None and holder.get("cam") is cam
    mode = ("SEND " if args.send else "VIRTUAL ") + ("(fake camera)" if fake else "(Dino)")
    print(f"autoscan {face}, {sum(len(r) for r in rows.values())} end faces, {mode}; session {session}")
    if args.send:
        moves = " -> ".join([*args.via, f"{face} far in", "scan", *reversed(args.via), args.end])
        print(f"the arm will move: {moves}; max {args.max_in_mm:.0f} mm past the model face")
        if args.window and not args.no_prompt:
            cam.preview("waiting for Enter in the terminal")
        if not args.no_prompt and input("Dino in the hand, path clear? Enter = start, q = quit > ").strip().lower() == "q":
            cam.release()
            mover.shutdown()
            return None, False
    cam.dash = args.dash
    args.dash.event(total=getattr(args, "planned_total", None) or sum(len(r) for r in rows.values()))
    mover.idle = cam.preview if args.window or isinstance(args.dash, Dashboard) else None
    t0 = time.time()
    st = None
    finished = False
    carry = getattr(args, "carry", None)  # run --carry-dino: what the first face found about the Dino
    scanner = Scanner(args, mover, cam, session, carry)
    try:
        st = scanner.scan_face(face, rows, far_in, far_out)
        if carry is not None:
            carry.update(roll=scanner.roll, mm_per_px=st.mm_per_px)
        if st.rows:  # where the first block of this face was found: the start of the next scan
            data = json.loads(last_file.read_text()) if last_file.exists() else {}
            first = st.rows[0]
            data[face] = {"depth_mm": first["depth_mm"], "down_mm": first["down_mm"], "layer": first["layer"],
                          "session": session.name}
            last_file.write_text(json.dumps(data, indent=2))
        a = approach(args)
        mover.joint_move(poses[args.end], args.end, a["speed"], None if args.via else a["slow_start"], a["min_s"])
        finished = True
    except KeyboardInterrupt:
        print("\nCtrl+C: stopped (arm holds where it is)")
    except ScanError as e:
        print(f"\nSTOPPED: {e}")
    finally:
        if not keep:
            cam.release()
        mover.shutdown()
        cv2.destroyAllWindows()
    print(f"\n{time.time() - t0:.0f} s here, arm motion {mover.seconds / 60:.1f} min "
          f"(+ about {args.settle_s + 0.2:.1f} s of camera per focus step), session {session}")
    if isinstance(cam, RealCamera) and not isinstance(cam, FakeCamera):
        print(cam.exposure_report())
    if st and st.rows:
        if fake and isinstance(cam, FakeCamera):
            errs = [abs(r["depth_mm"] - (args.fake_depth_mm + args.fake_lean_mm * r["layer"] / 10)) for r in st.rows]
            print(f"fake camera check: depth found within {max(errs):.2f} mm of the truth "
                  f"({len(st.rows)} shots)")
        res = rank([session], args.mode, session / "ranking.json")
        print(f"ranking ({args.mode}, this session only): " + ", ".join(r["block"] for r in res[:6]) + " ...")
    return (session if st and st.rows else None), finished


def replay_main(args) -> None:
    dash = Dashboard(args.port)
    print(f"dashboard: {dash.url} (F = full screen, L = language); replaying, Ctrl+C stops")
    order = order_for_page(load_gaps(args.sessions))
    try:
        while True:
            replay(dash, args.sessions, args.speed, order)
            if not args.loop:
                input("replay finished: Enter or Ctrl+C closes it > ")
                return
            time.sleep(5 / args.speed)
    except (KeyboardInterrupt, EOFError):
        pass


def add_scan_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dashboard", action="store_true", help="stage display in the browser instead of the camera window")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="of the dashboard")
    p.add_argument("--fast", action="store_true", help="shorter waits and fewer focus samples: " + str(FAST))
    p.add_argument("--preset", default=None, help="named settings (write --preset=NAME): " + ", ".join(PRESETS))
    p.add_argument("--settle-s", type=float, default=SETTLE_S,
                   help="wait after a move while searching (the shot itself always waits the full time)")
    p.add_argument("--focus-fall", type=float, default=0.88, help="a focus sample below this x the best has fallen")
    p.add_argument("--focus-falls", type=int, default=2, help="that many samples in a row end the search on a side")
    p.add_argument("--level-once", action="store_true", help="level the edge only until it is level once per face")
    p.add_argument("--arm", choices=("right", "left"), default="right")
    p.add_argument("--layers", nargs=2, type=int, default=None, metavar=("FIRST", "LAST"),
                   help="teammate layers (L00 = bottom), default scan_layers of scan_setup.json")
    p.add_argument("--send", action="store_true", help="really move (default: virtual arm, fake camera)")
    p.add_argument("--fake", action="store_true", help="fake camera also with --send")
    p.add_argument("--start", default="home", help="saved pose the arm is at when starting (virtual run)")
    p.add_argument("--end", default="home", help="saved pose to go to at the end")
    p.add_argument("--via", nargs="*", default=[], help="saved poses passed before the face (joint moves)")
    p.add_argument("--turned", action="store_true",
                   help="the tower was turned 180 deg by hand: results are saved as the opposite face (far ends)")
    p.add_argument("--top-first", action="store_true",
                   help="enter and leave at the top layer, scan downwards (joint moves stay far from the tower)")
    p.add_argument("--seed", default="home", help="saved pose to seed the face IK (arm configuration)")
    p.add_argument("--joint-speed", type=float, default=10.0, help="deg/s")
    p.add_argument("--quick", action="store_true", help="DRAFT: quicker moves to and from the face: " + str(QUICK))
    p.add_argument("--approach-speed", type=float, default=None, help="deg/s of the joint moves to and from the face")
    p.add_argument("--approach-slow-deg", type=float, default=0.0, help="slower part next to the tower (largest joint)")
    p.add_argument("--approach-slow-speed", type=float, default=None, help="deg/s of that part")
    p.add_argument("--approach-min-s", type=float, default=2.0, help="shortest joint move")
    p.add_argument("--line-accel", type=float, default=LINE_ACCEL_MM_S2, help="mm/s2 at start and stop of straight moves")
    p.add_argument("--far-speed", type=float, default=None,
                   help="mm/s of straight moves at least 25 mm out of the face all the way (default --line-speed)")
    p.add_argument("--parallel-speed", type=float, default=None,
                   help="mm/s of straight moves along the face at a fixed distance (default --line-speed)")
    p.add_argument("--motion-only", action="store_true", help="go to the face and leave again, no layers")
    p.add_argument("--sweep", action="store_true", help="DRAFT: full search only on the middle block of the first and "
                   "last layer, the other shots on the line between them, no search")
    p.add_argument("--blur-test", type=int, default=None, metavar="LAYER",
                   help="instead of the scan: stopped shots of this layer (teammate number, L04 = floor 5), "
                        "then passes without stopping, every frame measured")
    p.add_argument("--blur-h-speeds", type=float, nargs="*", default=[5.0, 10.0, 20.0, 30.0],
                   help="mm/s of the passes along the layer (--blur-test)")
    p.add_argument("--blur-v-speeds", type=float, nargs="*", default=[2.0, 5.0, 10.0],
                   help="mm/s of the passes up / down over the block with the widest stopped gap (--blur-test)")
    p.add_argument("--keep-camera", action="store_true", help="run: one Dino reader for all faces")
    p.add_argument("--prior-from-last", action="store_true",
                   help="start the search at the depth and edge height of the last scan of this face (same place)")
    p.add_argument("--carry-dino", action="store_true",
                   help="run: start the next face with the Dino turn and image scale the face before found")
    p.add_argument("--narrow-ref2", action="store_true",
                   help="--sweep: the second reference searches near the first one's depth (wide only if that fails)")
    p.add_argument("--shot-settle-s", type=float, default=SETTLE_S,
                   help="wait after stopping at a block before its shot (--sweep, --blur-test)")
    p.add_argument("--glide", action="store_true", help="DRAFT: depth and height like --sweep, but every layer in "
                   "one pass along it without stopping, every frame measured")
    p.add_argument("--glide-speed", type=float, default=30.0, help="mm/s of the passes along a layer (--glide)")
    p.add_argument("--sweep-retreat-mm", type=float, default=10.0, help="out from the face between rows (--sweep)")
    p.add_argument("--sweep-edge-px", type=float, default=100.0,
                   help="centre the edge again when it is this far from the image centre (--sweep)")
    p.add_argument("--line-speed", type=float, default=10.0, help="mm/s of straight moves")
    p.add_argument("--prior-depth-mm", type=float, default=-5.0,
                   help="expected focus depth past the model face (+x L13, dial 30, 2026-11-02: -5)")
    p.add_argument("--limit-mm", type=float, default=6.0,
                   help="never deeper than the depth found on the previous block + this")
    p.add_argument("--prior-down-mm", type=float, default=-3.9,
                   help="expected edge height below the model (negative = higher), start of the edge search "
                        "(+x L13, 2026-11-02: -3.9)")
    p.add_argument("--gap-search-mm", type=float, default=12.5, help="look for the edge this far up / down")
    p.add_argument("--search-out-mm", type=float, default=20.0, help="first focus search of a layer: this far out")
    p.add_argument("--search-in-mm", type=float, default=15.0, help="and this far in (if nothing better outside)")
    p.add_argument("--max-in-mm", type=float, default=8.0,
                   help="never deeper than this past the model (tip about 25 mm from the block in focus at dial 30)")
    p.add_argument("--refocus-mm", type=float, default=5.0, help="+- search for the next blocks of a layer")
    p.add_argument("--step-mm", type=float, default=1.0, help="focus search step")
    p.add_argument("--min-focus", type=float, default=12.0,
                   help="a peak lower than this is not a peak (floor about 4; real peaks 45-96 at dial 30)")
    p.add_argument("--mode", choices=ends.METHODS, default=ends.METHODS[0])
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--window", action="store_true", help="show the camera while scanning")
    p.add_argument("--out", type=Path, default=Path("captures"))
    p.add_argument("--fake-depth-mm", type=float, default=-3.0, help="fake camera: true faces this much deeper")
    p.add_argument("--fake-lean-mm", type=float, default=1.5, help="fake camera: + this per 10 layers")
    p.add_argument("--fake-drop-mm", type=float, default=0.6, help="fake camera: gaps this much lower")
    p.add_argument("--fake-roll-deg", type=float, default=4.0, help="fake camera: Dino turned in the hand")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan", help="scan one tower face")
    p.add_argument("--face", required=True, help="px, py, mx, my (= +x, +y, -x, -y)")
    add_scan_options(p)
    p.set_defaults(no_prompt=False)
    r = sub.add_parser("rank", help="ranking.json from one or more scan sessions")
    r.add_argument("sessions", nargs="+", type=Path)
    r.add_argument("--mode", choices=ends.METHODS, default=ends.METHODS[0])
    r.add_argument("--out", type=Path, default=Path("captures/ranking.json"))
    r.add_argument("--candidates", type=Path, help="teammate's candidates file, e.g. "
                   "$JENGA_STATE_DIR/jenga_pick_candidates.json (default: next to --out)")
    u = sub.add_parser("run", help="pick the Dino, scan -y and +x, write the pull order, place the Dino")
    add_scan_options(u)
    u.set_defaults(search_out_mm=10.0, search_in_mm=25.0, refocus_mm=8.0, limit_mm=8.0, max_in_mm=14.0, line_speed=15.0)
    u.set_defaults(joint_speed=None)  # per face in RUN_FACES
    u.add_argument("--faces", nargs="+", choices=[s["face"] for _, s in RUN_FACES],
                   default=[s["face"] for _, s in RUN_FACES], help="my = -y, px = +x, in the order of RUN_FACES")
    u.add_argument("--no-pick", action="store_true", help="the Dino is already in the hand, arm at home")
    u.add_argument("--no-place", action="store_true", help="keep the Dino in the hand at the end")
    u.add_argument("--step", action="store_true", help="Enter before every step of pick and place")
    u.add_argument("--task-arm-speed", type=float, default=10.0, help="deg/s of pick and place")
    u.add_argument("--task-hand-speed", type=float, default=20.0, help="deg/s of the fingers in pick and place")
    o = sub.add_parser("order", help="pull order of one or more sessions: widest gap first + 빼안빼 rule")
    o.add_argument("sessions", nargs="+", type=Path)
    o.add_argument("--candidates", type=Path, help="also write the teammate's candidates file here")
    c = sub.add_parser("score", help="streaks per face and merged, rule off / on, against pulled-by-hand results")
    c.add_argument("sessions", nargs="+", type=Path)
    c.add_argument("--truth", nargs="+", required=True, metavar="FLOOR=MARKS",
                   help="e.g. 4=안빼안 5=빼안빼: floor 1 = bottom, left middle right facing the scanned face")
    c.add_argument("--top", type=int, default=8, help="also count the pulls among the first N")
    d = sub.add_parser("dashboard", help="replay saved sessions on the stage display, no robot or camera")
    d.add_argument("sessions", nargs="+", type=Path)
    d.add_argument("--speed", type=float, default=1.0, help="times real speed")
    d.add_argument("--port", type=int, default=DEFAULT_PORT)
    d.add_argument("--loop", action="store_true")
    x = sub.add_parser("compare", help="two scans of one tower, block by block")
    x.add_argument("a", type=Path)
    x.add_argument("b", type=Path)
    if "--fast" in sys.argv[1:]:  # as defaults: an option given by hand still wins
        p.set_defaults(**FAST)
        u.set_defaults(**FAST)
    if "--quick" in sys.argv[1:]:
        p.set_defaults(**QUICK)
        u.set_defaults(**QUICK)
    for name in [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--preset=")]:
        if name not in PRESETS:
            raise SystemExit(f"--preset: {', '.join(PRESETS)}")
        for o in (p, u):
            o.set_defaults(**{**FAST, **QUICK, **PRESETS[name]})
    args = parser.parse_args()
    if args.cmd == "compare":
        compare_main(args.a, args.b)
        return
    if args.cmd == "dashboard":
        return replay_main(args)
    if args.cmd == "order":
        order_main(args.sessions, args.candidates)
        return
    if args.cmd == "score":
        score_main(args.sessions, args.truth, args.top)
        return
    if args.cmd == "rank":
        res = rank(args.sessions, args.mode, args.out, args.candidates)
        for x in res:
            print(f"{x['rank']:3d}  {x['block']}  score {x['score_px']:+.1f}")
        print(f"wrote {args.out} and the candidates file")
        return
    if args.layers is None and args.cmd != "run":
        args.layers = json.loads(SETUP_FILE.read_text())["scan_layers"]
    if args.send and not args.dashboard:
        args.window = True  # SPACE / q in the camera window, and the user should see it
    args.dash = Dashboard(args.port) if args.dashboard else NoDashboard()
    if args.dashboard:
        print(f"dashboard: {args.dash.url} (F = full screen, L = language)")
    if args.cmd == "run":
        args.layers = args.layers or RUN_LAYERS
        run_main(args)
    else:
        scan_main(args)
    if args.dashboard:
        try:
            input("the dashboard stays open: Enter or Ctrl+C closes it > ")
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == "__main__":
    main()
