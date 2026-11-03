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
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from jenga_vision import ends
from jenga_vision.arm_slider import POSES_FILE
from jenga_vision.camera import focus_value
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
DISCARD_FRAMES = 3  # stale frames in the V4L2 buffer
DEFAULT_MM_PER_PX = 0.02  # Dino at dial 30; measured on the first shot when possible
EDGE_TOLERANCE_PX = 25  # edge row this close to the image centre is good enough
BRIGHTNESS_REF = 100.0  # steady_focus is the focus value at this image median
EDGE_STEP = 40  # gray step of the rim over 6 rows: in focus 98-219, no edge in the image <= 21 (2026-11-02)
RIM_HALF_WIDTH = 40  # px, columns used for the rim row at the middle of each image third
EDGE_ROWS_AGREE = 12  # the middle third's rim row is this close to the line through the outer two
EDGE_MAX_SPAN = 80  # rows between the left and right thirds: the Dino may sit up to about 10 deg turned
EDGE_MARGIN = 0.03  # stripes difference closer to 0 than this: undecided, taken as the top edge
REFOCUS_MOVE_M = 0.001  # focus again after moving more than this up / down
EDGE_SEARCH_STEP_MM = 2.5  # vertical search step for the edge (the image is ~10 mm high at dial 30)
LEVEL_DEG, MAX_ROLL_DEG = 0.7, 10.0  # level the gap when tilted more than this; limit of the correction


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
def ramped(path: list[np.ndarray], q0: np.ndarray, dist_mm: float, seconds: float) -> list[np.ndarray]:
    """The same path, resampled so that evenly timed points start and stop at LINE_ACCEL_MM_S2
    instead of jumping to full speed (fast moves would shake the Dino for longer)."""
    if dist_mm < 1e-6:
        return path
    qs = np.vstack([q0, *path])
    a = LINE_ACCEL_MM_S2
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


class Mover:
    """Joint moves and straight Dino-frame moves; virtual (joints just set) without send."""

    def __init__(self, arm: str, tool: np.ndarray, send: bool, q0, joint_speed_deg: float, line_speed_mm: float):
        self.arm, self.tool, self.send = arm, tool, send
        self.chain = Chain(arm)
        self.joint_speed, self.line_speed = joint_speed_deg, line_speed_mm
        self.ros = None
        self.idle = None  # called while waiting for a move (camera preview)
        self.seconds = 0.0  # planned motion time so far
        if send:
            from jenga_vision.arm_slider import ArmNode
            self.ros = ArmNode(arm, True, node_name="jenga_autoscan")
            t0 = time.time()
            while self.ros.current() is None:
                if time.time() - t0 > 5:
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

    def joint_move(self, q, label: str = "") -> None:
        q = np.array(q, float)
        biggest = float(np.degrees(np.max(np.abs(q - self.q))))
        seconds = max(2.0, biggest / self.joint_speed)
        print(f"   joint move {label}: largest joint {biggest:.0f} deg, {seconds:.1f} s")
        self._send([q], seconds)

    def line_to(self, goal_dino: np.ndarray, min_s: float = 0.4) -> None:
        """Straight line of the Dino frame (position; orientation interpolated) to `goal_dino`."""
        start = self.dino()
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
        seconds = max(min_s, dist_mm / self.line_speed + self.line_speed / LINE_ACCEL_MM_S2, biggest / self.joint_speed)
        self._send(ramped(path, self.q, dist_mm, seconds), seconds)

    def shutdown(self) -> None:
        if self.ros is not None:
            self.ros.shutdown()


# --- cameras -----------------------------------------------------------------------
class RealCamera:
    def __init__(self, device: int, window: bool):
        from jenga_vision.camera import DinoCamera
        self.cam = DinoCamera(device)
        self.window = window

    def frames(self, n: int) -> list[np.ndarray]:
        time.sleep(SETTLE_S)
        for _ in range(DISCARD_FRAMES):
            self.cam.read()
        return [self.cam.read() for _ in range(n)]

    def preview(self, text: str = "moving") -> None:
        """Live view while the arm moves or waits (window only)."""
        if self.window:
            self.show(self.cam.read(), text)

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
        self.cam.release()


class FakeCamera(RealCamera):
    """Synthetic Dino image from where the point in focus is relative to a 'true' tower:
    the model tower with every end face `depth_mm` (+ `lean_mm` per 10 layers) deeper and
    the gaps `drop_mm` lower. Blur grows with the focus error, the gap band moves with
    the height error, every block has its own gap width (seeded)."""

    def __init__(self, mover: Mover, shots: dict[int, list[Shot]], depth_mm: float, lean_mm: float,
                 drop_mm: float, window: bool, roll_deg: float = 0.0, seed: int = 1):
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

    def frames(self, n: int) -> list[np.ndarray]:
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
        return [np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8) for _ in range(n)]

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
    mm_per_px: float | None = None
    rows: list[dict] = field(default_factory=list)


class Scanner:
    def __init__(self, args, mover: Mover, cam, session: Path):
        self.args, self.mover, self.cam, self.session = args, mover, cam, session
        self.max_in = args.max_in_mm / 1000
        self.last_depth = None  # depth found on the previous block: the limit follows it
        self.layer_h = json.loads(SETUP_FILE.read_text())["tower"]["layer_h"]
        self.roll = 0.0
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

        def measure(depth_mm: float) -> float:
            d = depth_mm / 1000
            self.go(shot, d, down)
            f, frame = self.measure()
            self.cam.show(frame, f"L{shot.layer:02d} {shot.screen} depth {depth_mm:+.2f} mm focus {f:.1f}")
            self.log_sample(shot, d, down, f, frame)
            k = self.cam.key()
            if k in (ord("q"), 27):
                raise ScanError("aborted by the user (q)")
            if k == ord(" "):
                raise DepthTaken(d, f)
            return f

        g = guess * 1000
        try:
            p = find_peak(measure, g, g - out, min(g + inward, self.deepest() * 1000), a.step_mm, min_peak=a.min_focus)
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
        slope = edge_tilt(self.cam.frames(1)[0])
        if slope is None or abs(math.degrees(math.atan(slope))) < LEVEL_DEG:
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
                return
        self.roll = before
        self.go(shot, depth, down)
        print(f"   edge tilted {math.degrees(math.atan(slope)):+.1f} deg, leveling did not help (kept)")

    def center_edge(self, shot: Shot, depth: float, down: float, st: FaceState) -> float:
        """Move the camera up / down until the edge is near the image centre (measures the
        image scale on the first call)."""
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

    def shoot(self, face: str, shot: Shot, depth: float, down: float, focus: float, st: FaceState) -> None:
        frames = self.cam.frames(ends.FRAMES_PER_SHOT)
        gaps = [ends.measure_end(f) for f in frames]
        gap = float(np.median([g.gap for g in gaps]))
        number = FACE_NUMBER[face]
        slot = shot.slot
        if self.args.turned:  # the far ends of the blocks are in view, left and right swapped
            number, slot = (number + 1) % 4 + 1, 2 - slot
        key = f"L{shot.layer + 1:02d}:f{number}:{shot.screen}"
        name = f"{key.replace(':', '_')}.png"
        cv2.imwrite(str(self.session / name), frames[-1])
        self.cam.show(frames[-1], f"{key} gap {gap:.1f} px focus {focus:.0f}")
        st.rows.append({"key": key, "gap": round(gap, 2), "face": face, "layer": shot.layer, "slot": slot,
                        "screen": shot.screen, "focus": round(focus, 1), "depth_mm": round(depth * 1000, 2),
                        "down_mm": round(down * 1000, 2), "roll_deg": round(math.degrees(self.roll), 2),
                        "row": gaps[-1].row, "image": name, "brightness": round(float(np.median(frames[-1]))),
                        "rim_dark": round(rim_darkness(frames[-1]), 3),
                        "q": [round(float(v), 5) for v in self.mover.q]})
        print(f"   SHOT {key} (L{shot.layer:02d}_S{slot}): gap {gap:.1f} px, rim dark {rim_darkness(frames[-1]):.2f}, focus {focus:.0f}, "
              f"depth {depth * 1000:+.1f} mm, down {down * 1000:+.1f} mm, roll {math.degrees(self.roll):+.1f} deg")

    def scan_face(self, face: str, rows: dict[int, list[Shot]], far_in, far_out) -> FaceState:
        st = FaceState(depth_m=self.args.prior_depth_mm / 1000, down_m=self.args.prior_down_mm / 1000)
        for via in self.args.via:
            self.mover.joint_move(json.loads(POSES_FILE.read_text())[self.mover.arm][via], via)
        self.mover.joint_move(far_in, f"{face} far in")
        for layer in sorted(rows, reverse=self.args.top_first):
            row = rows[layer]
            print(f"\n== {face} L{layer:02d}")
            first = row[0]
            self.mover.line_to(first.pose(st.depth_m - RETREAT_M, st.down_m))
            for k, shot in enumerate(row):
                wide = k == 0 or not st.depth_known
                found = self.focus(shot, st.depth_m, st.down_m, wide)
                if found is None and not wide:
                    found = self.focus(shot, st.depth_m, st.down_m, True)
                if found is None:
                    print(f"   {shot.screen}: no focus, skipped")
                    continue
                d, f = found
                st.depth_m, st.depth_known = d, True
                self.last_depth = d
                down = focused_at = st.down_m
                if not st.height_known:  # first edge of the face: search up / down for it
                    down = self.find_edge(shot, d, st.down_m)
                    if down is None:
                        print(f"   {shot.screen}: in focus (depth {d * 1000:+.1f} mm) but no edge within "
                              f"{self.args.gap_search_mm:.0f} mm up/down, skipped")
                        continue
                    down = self.top_edge(shot, d, down, st)
                    if down is None:
                        continue
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
                self.shoot(face, shot, d, st.down_m, f, st)
                self.save(st)
            last = row[-1]
            self.mover.line_to(last.pose(st.depth_m - RETREAT_M, st.down_m))
        self.mover.line_to(np.array(self.mover.chain.fk(far_out, self.mover.tool)))
        for via in reversed(self.args.via):
            self.mover.joint_move(json.loads(POSES_FILE.read_text())[self.mover.arm][via], via)
        return st

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


def pick_order(px: dict[tuple[int, int], float], rule: bool = True,
               prefer=None) -> list[tuple[int, int, str]]:
    """Blocks to pull, widest gap first: [(layer, slot, why)]. Ties: upper layer, then the
    middle block. With the rule each layer is 안빼안 or 빼안빼: the middle finishes its
    layer, a side is followed at once by the opposite side, which finishes it too.
    `prefer(block)` breaks ties first (to find the luckiest and unluckiest tie order)."""
    order = sorted(px, key=lambda b: (-px[b], prefer(b) if prefer else 0, -b[0], b[1] != 1, b[1]))
    if not rule:
        return [(*b, "gap") for b in order]
    seq: list[tuple[int, int, str]] = []
    done: set[int] = set()
    for layer, slot in order:
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
    seq = pick_order({b: g for b, (g, _) in gaps.items()})
    names = ("left", "middle", "right")
    for n, (layer, slot, why) in enumerate(seq, 1):
        px, face = gaps[(layer, slot)] if (layer, slot) in gaps else (None, gaps[(layer, 2 - slot)][1])
        shown = "not measured" if px is None else f"{px:.1f} px"
        print(f"{n:3d}  jenga_L{layer:02d}_S{slot}  floor {layer + 1:2d} {names[slot]:6s} (facing {face})  "
              f"{shown}{'  <- opposite side' if why == 'opposite' else ''}")
    if out:
        out.write_text(json.dumps({
            "version": 1, "source": "camera", "stamp": time.time(),
            "candidates": [{"block": f"jenga_L{layer:02d}_S{slot}", "score": gaps.get((layer, slot), (0.0,))[0],
                            "reason": f"빼안빼 rule {n} ({why})"} for n, (layer, slot, why) in enumerate(seq, 1)]},
            indent=1))
        print(f"wrote {out}")


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
        for rule in (False, True):
            seq = pick_order(px, rule)
            best = streak(pick_order(px, rule, lambda b: not truth[b]), truth)
            worst = streak(pick_order(px, rule, lambda b: truth[b]), truth)
            hits = sum(truth[(layer, slot)] for layer, slot, _ in seq[:top])
            print(f"{name:6s} rule {'on ' if rule else 'off'}  streak {streak(seq, truth)} (ties {worst}-{best}), "
                  f"top {min(top, len(seq))}: {hits} pull  {seq_text(seq, truth)}")


# --- main --------------------------------------------------------------------------
def scan_main(args) -> None:
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
    pts, _ = plan_face_stable(Chain(args.arm), tool, tower, face, layers, seed, plan_setup)
    for note in away_from_limits(pts, Chain(args.arm)):
        print(f"   {note}")
    far_in, far_out = pts[-1]["q"] if args.top_first else pts[0]["q"], pts[-1]["q"]

    session = args.out / time.strftime("autoscan_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    mover = Mover(args.arm, tool, args.send, poses[args.start], args.joint_speed, args.line_speed)
    fake = args.fake or not args.send
    cam = (FakeCamera(mover, rows, args.fake_depth_mm, args.fake_lean_mm, args.fake_drop_mm, args.window,
                      args.fake_roll_deg)
           if fake else RealCamera(args.device, args.window))
    mode = ("SEND " if args.send else "VIRTUAL ") + ("(fake camera)" if fake else "(Dino)")
    print(f"autoscan {face}, {sum(len(r) for r in rows.values())} end faces, {mode}; session {session}")
    if args.send:
        moves = " -> ".join([*args.via, f"{face} far in", "scan", *reversed(args.via), args.end])
        print(f"the arm will move: {moves}; max {args.max_in_mm:.0f} mm past the model face")
        if args.window:
            cam.preview("waiting for Enter in the terminal")
        if input("Dino in the hand, path clear? Enter = start, q = quit > ").strip().lower() == "q":
            cam.release()
            mover.shutdown()
            return
    mover.idle = cam.preview if args.window else None
    t0 = time.time()
    st = None
    try:
        st = Scanner(args, mover, cam, session).scan_face(face, rows, far_in, far_out)
        mover.joint_move(poses[args.end], args.end)
    except KeyboardInterrupt:
        print("\nCtrl+C: stopped (arm holds where it is)")
    except ScanError as e:
        print(f"\nSTOPPED: {e}")
    finally:
        cam.release()
        mover.shutdown()
        cv2.destroyAllWindows()
    print(f"\n{time.time() - t0:.0f} s here, arm motion {mover.seconds / 60:.1f} min "
          f"(+ about {SETTLE_S + 0.2:.1f} s of camera per focus step), session {session}")
    if st and st.rows:
        if fake and isinstance(cam, FakeCamera):
            errs = [abs(r["depth_mm"] - (args.fake_depth_mm + args.fake_lean_mm * r["layer"] / 10)) for r in st.rows]
            print(f"fake camera check: depth found within {max(errs):.2f} mm of the truth "
                  f"({len(st.rows)} shots)")
        res = rank([session], args.mode, session / "ranking.json")
        print(f"ranking ({args.mode}, this session only): " + ", ".join(r["block"] for r in res[:6]) + " ...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan", help="scan one tower face")
    p.add_argument("--arm", choices=("right", "left"), default="right")
    p.add_argument("--face", required=True, help="px, py, mx, my (= +x, +y, -x, -y)")
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
    r = sub.add_parser("rank", help="ranking.json from one or more scan sessions")
    r.add_argument("sessions", nargs="+", type=Path)
    r.add_argument("--mode", choices=ends.METHODS, default=ends.METHODS[0])
    r.add_argument("--out", type=Path, default=Path("captures/ranking.json"))
    r.add_argument("--candidates", type=Path, help="teammate's candidates file, e.g. "
                   "$JENGA_STATE_DIR/jenga_pick_candidates.json (default: next to --out)")
    o = sub.add_parser("order", help="pull order of one or more sessions: widest gap first + 빼안빼 rule")
    o.add_argument("sessions", nargs="+", type=Path)
    o.add_argument("--candidates", type=Path, help="also write the teammate's candidates file here")
    c = sub.add_parser("score", help="streaks per face and merged, rule off / on, against pulled-by-hand results")
    c.add_argument("sessions", nargs="+", type=Path)
    c.add_argument("--truth", nargs="+", required=True, metavar="FLOOR=MARKS",
                   help="e.g. 4=안빼안 5=빼안빼: floor 1 = bottom, left middle right facing the scanned face")
    c.add_argument("--top", type=int, default=8, help="also count the pulls among the first N")
    args = parser.parse_args()
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
    if args.layers is None:
        args.layers = json.loads(SETUP_FILE.read_text())["scan_layers"]
    if args.send:
        args.window = True  # SPACE / q in the camera window, and the user should see it
    scan_main(args)


if __name__ == "__main__":
    main()
