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
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from jenga_vision import ends
from jenga_vision.arm_slider import POSES_FILE
from jenga_vision.camera import focus_value
from jenga_vision.dashboard import DEFAULT_PORT, Dashboard, NoDashboard, replay
from jenga_vision.focus_search import find_peak
from jenga_vision.kinematics import Chain
from jenga_vision.clearance import MIN_CLEAR_MM
from jenga_vision.paths import CAPTURES_DIR, CONFIG_DIR, GAME_BLOCK_FILE
from jenga_vision.scan_plan import (FACE_NORMALS, SETUP_FILE, Tower, arm_tool, away_from_limits, plan_face_stable,
                                    shot_pose, view_yaw)

FACE_NUMBER = {"+x": 1, "+y": 2, "-x": 3, "-y": 4}  # jenga-ends faces: 90 deg counterclockwise each
LINE_STEP_M = 0.002  # IK step along a straight move
LINE_ACCEL_MM_S2 = 200.0  # start and stop of a straight move
RAMP_DT_S = 0.05  # time between trajectory points
MAX_JOINT_STEP_DEG = 3.0  # per IK step; more = the arm would change configuration
RETREAT_M = 0.03  # out from the face between layers
# The arm configuration of each face, fixed (the tower stands in one place): first and last planned point of the
# jig scan (layers 2..14); the plan's ends are solved again from these, so the arm can't take another one.
#   -y: wrist j5 -15, j6 +85, j7 133 (as planned on the teammate's PC 2026-11-19). The configuration of our PC and
#       the 11-18 robot scans (j5 183, j6 -76, j7 293) has j6 at its -97 limit on the top layer's left block: on the
#       teammate's tower (3 mm deeper) the scan stopped there. This one: j6 90 deg from its limits, arm 55 mm from
#       the tower box. Its way out must not be jig_ready_1118 (that move hit the tower, -12 mm in the model):
#       -y starts from and ends at the teammate's home (169-199 mm clear).
#   +x: as on our PC and the 11-18 scans, from and to jig_ready_1118 (11 / 26 deg, 120 mm clear).
ROUTE = {("right", "-y"): ([-0.43192, -0.90688, 2.12582, 0.83385, -0.26467, 1.4791, 2.31922],
                           [0.15296, -0.94202, 1.14648, 0.67408, -0.64335, 1.55981, 1.76974]),
         ("right", "+x"): ([0.09832, 0.71072, 0.12397, 1.35043, 2.40767, 0.94618, 3.43459],
                           [0.00513, 0.6177, 0.46695, 1.7739, 2.05785, 0.3974, 3.82999])}
# run: where each face starts and ends (over run_faces of scan_setup.json), see ROUTE
FACE_ROUTE = {"-y": {"seed": "teammate_home", "via": [], "end": "teammate_home"},
              "+x": {"seed": "jig_ready_1118", "via": ["jig_ready_1118"], "end": "jig_ready_1118"}}
ROUTE_TOL_DEG = 90.0  # any joint further from the route -> stop (a flipped wrist is 150+ off; the elbow may vary)
ROUTE_VERSION = "route-2026-11-19b"  # in the plan cache key: plans made before the fixed route are not used
COLUMN_BACK_M = 0.08  # --columns: back along the lens from a column before the joint move to the next one
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
# run without options = the 2026-11-18 hand jig scan (Dino mounted once with `mount`, no pick / place);
# --preset=none for the old defaults, --pick / --place for the 11-09 quick pick / place
RUN_PRESET = "jig-full"
HOME_SPEED = 40.0  # deg/s of the last move of run to run_home (and of the first one to the scans' start); 20 was too slow (user)
START_TOL_DEG = 2.0  # run: the arm this close to the scans' start pose needs no move there
# The teammate's home for the right arm, where run ends (2026-11-19): the left arm at their home on the robot
# (/joint_states xarm_left_joint1..7 = -8.0 1.0 24.0 44.0 -74.0 93.0 -43.3 deg), mirrored (MIRROR_SIGNS).
# From jig_ready_1118 the move is up to 154 deg on joint 7 and 73 on joint 5.
TEAMMATE_HOME = {"right": [0.139598, 0.017453, -0.41889, 0.767863, 1.291499, 1.623174, 0.755739]}


def saved_poses(arm: str) -> dict[str, list[float]]:
    """arm_poses.json of `arm` plus the poses kept in the code (teammate_home)."""
    poses = json.loads(POSES_FILE.read_text())[arm]
    if arm in TEAMMATE_HOME:
        poses.setdefault("teammate_home", TEAMMATE_HOME[arm])
    return poses
RUN_OUT = "jig1118b"  # in captures/: the plan cache and per-face state of the 11-18 jig calibration
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
#   v3 (DRAFT 2026-11-16, sim only): v2 + the Dino turn levelled by turning the image (--soft-level, no arm roll,
#   so no refocus after it), a reference edge left where it is within 40 px of the centre (--edge-ok-px), sweep shots within 60 (--sweep-edge-px),
#   the focus peak from two samples plus one to confirm (--focus-predict). Offline on 492 saved shots: one image
#   turn per face matched the arm-levelled gaps up to +-10 deg; tools/soft_level_test_20261116.py.
#   v4 (DRAFT 2026-11-16, sim only; goal: both faces in 60 s with the Dino fixed in a jig on the hand): v3 + the
#   Dino where the last scan had it (--rigid: no edge search, no second reference, one row after the other) and a
#   row stopped once its side is wider than its middle (--lazy-row: the 빼안빼 call can't change; 13 % fewer
#   shots on the saved scans, tools/shortcut_rules_20261116.py).
PRESETS = {"v1": {"sweep": True, "shot_settle_s": 0.3},
           "v2": {"sweep": True, "shot_settle_s": 0.2, "parallel_speed": 100.0, "approach_speed": 60.0,
                  "keep_camera": True, "carry_dino": True, "narrow_ref2": True, "prior_from_last": True}}
# --focus-predict left out of v3 (2026-11-16 robot): twice it put the middle peak 1.5-2.3 mm short and the third
# sample still passed (its value only has to match the two samples, which were both on the slope)
PRESETS["v3"] = {**PRESETS["v2"], "soft_level": True, "edge_ok_px": 40.0, "sweep_edge_px": 60.0}
PRESETS["v4"] = {**PRESETS["v3"], "rigid": True, "lazy_row": True}
# The Dino rides on the ring and little finger in the hand jig (2026-11-18): those joints decide where it is, so
# they must be as at the calibration (dino.calibration.<arm>.hand_q, jenga-calib --hand)
HAND_FINGERS = (3, 4)  # ring, little
HAND_WARN_DEG, HAND_STOP_DEG = 3.0, 8.0
RECORD_PROBE = 30  # --record: frames timed before the video file is opened at their rate
EST_GAP_MM = 0.10  # typical gap of a block a 빼안빼 layer did not shoot (block_geometry)
# 2026-11-18, Dino in the hand jig (scan hand pose jig_scan_pose): every tower handled as a new one (nothing of the
# last tower), short focus searches as one sweep, the face's sideways lean learned row by row. Two modes:
#   jig-full  every block shot, residuals measured                       (new tower ~1.4-1.5 min, both faces)
#   jig-fast  a 빼안빼 row skips its third block, that one's residual estimated (+-0.3 mm)  (target < 1.2 min)
PRESETS["jig-full"] = {**PRESETS["v4"], "lazy_row": False, "fresh_tower": True, "sweep_focus": True,
                       "rigid_trust": True, "sweep_retreat_mm": 0.0, "out_speed": 100.0, "line_accel": 600.0,
                       "shot_settle_s": 0.15, "approach_speed": 60.0, "refocus_below": 0.6, "sweep_edge_px": 90.0}
# fast: also shoot an edge up to 115 px off the image centre instead of centring it (measured within +-120 px;
# +-100 px changed 5 % of saved frames by more than 1 px, 2026-11-16), joint moves at 90 deg/s
PRESETS["jig-fast"] = {**PRESETS["jig-full"], "lazy_row": True, "sweep_edge_px": 115.0, "approach_speed": 90.0}
RIGID_SEED_FILE = CONFIG_DIR / "rigid_seed.json"  # in git: the start on a PC that has no RIGID_FILE yet
RIGID_FILE = ".rigid.json"  # in --out: per face, the Dino turn, image scale, depth and edge height per layer
RIGID_REFOCUS = 0.8  # --rigid: a row's first shot below this x the face's median focus value is refocused: about
# 1.3 mm off (robot 2026-11-16, --defocus-test 8: focus 0.85-0.9 x at 1 mm, gaps within 0.6 px up to 1 mm, 1.6 at 1.5)
FOCUS_WIDTH_MM = 1.98  # s of the Gaussian top of the focus curve (median of 271 searches, 2026-10-30..11-10)
PREDICT_MAX_MM = 3.0  # --focus-predict: a peak predicted further from the start than this is not trusted
PREDICT_CONFIRM = 0.94  # the sample at the predicted peak must reach this x the best of the two (noise 3-6 %)
ROLL_TEST_DEGS = [0.0, -3.0, 3.0, -5.0, 5.0, -10.0, 10.0, 0.0]
REPEAT_FILE = "repeat_test.csv"  # in --out: one row per block per --repeat-test run, kept across runs
DEFOCUS_TEST_MM = [0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0, 0.0]  # + = deeper (closer to the tower)
LEVEL_DEG, MAX_ROLL_DEG = 0.7, 20.0
TURN_STEP_DEG = 10  # calib: image turns tried for the edge (any Dino turn in the jig)
# --blur-test: passes without stopping, to see how much moving changes the gap px (2026-11-10)
BLUR_MAX_MM_S = 60.0  # passes along the face at a fixed distance (user OK to raise from 30, 2026-11-10)
BLUR_MARGIN_M = 0.010  # along passes start / end this far beyond the outer block centres
BLUR_V_SPAN_M = 0.010  # up / down passes go this far above and below the edge
BLUR_WINDOW_MM = 8.0  # frames this close to a block centre count for that block
BLUR_EDGE_PX = 80  # and only frames with the edge this close to the image centre
SWEEP_HALF_MM = 4.0  # --sweep-focus: one pass from 4 mm out to 4 mm in of the guess (new towers: 3-5 mm off)
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


def turn_image(frame: np.ndarray, deg: float) -> np.ndarray:
    """The image turned `deg` about its centre (+ = counterclockwise); turn_image(f, degrees(atan(edge_tilt(f))))
    levels the edge."""
    if abs(deg) < 1e-3:
        return frame
    h, w = frame.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def best_turn(frame: np.ndarray) -> float | None:
    """Image turn (deg, -180..180) that shows a level TOP edge (long grain above it) in the middle of the
    frame, or None: any turn of the Dino about its axis (calib). Only the central square is used, which
    stays inside the image at every turn."""
    h, w = frame.shape[:2]
    half = int(min(h, w) / 2 / math.sqrt(2)) - 2
    best = None
    for deg in range(-180, 180, TURN_STEP_DEG):
        crop = turn_image(frame, deg)[h // 2 - half:h // 2 + half, w // 2 - half:w // 2 + half]
        row = edge_row(crop)
        if row is None or edge_side(crop, row) < -EDGE_MARGIN:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        size = rim_row(gray)[1]
        if best is None or size > best[1]:
            best = (deg, size)
    if best is None:
        return None
    tilt = edge_tilt(turn_image(frame, best[0]))  # finer: the full width at that turn
    fine = best[0] + (math.degrees(math.atan(tilt)) if tilt is not None and abs(tilt) < 0.2 else 0.0)
    return (fine + 180) % 360 - 180


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
    yaw_deg: float = 0.0  # the Dino looks at the face from the side (scan_plan.view_yaw)
    rot: np.ndarray | None = None  # --columns: the Dino turned as in the column's taught pose instead

    def pose(self, depth_m: float, down_m: float, roll: float = 0.0) -> np.ndarray:
        """Dino frame with the point in focus `depth_m` past the model face (into the
        tower), `down_m` below the model gap, turned `roll` rad about the optical axis
        (levels the image when the Dino sits turned in the hand)."""
        from jenga_vision.kinematics import axis_angle, transform
        p = shot_pose(self.aim - self.normal * depth_m - np.array([0, 0, down_m]), self.normal, 0.0, self.yaw_deg)
        if self.rot is not None:
            p = transform(self.rot, p[:3, 3])
        return p @ transform(axis_angle(np.array([0, 0, 1.0]), roll), [0, 0, 0])


def hand_check(setup: dict, arm: str, positions: dict[str, float]) -> float | None:
    """Print the ring / little finger joints against the calibration; the largest difference (deg), or None
    if there is nothing to compare."""
    ref = setup["dino"].get("calibration", {}).get(arm, {}).get("hand_q")
    names = [f"ah{arm[0]}_joint{f}{j}" for f in range(5) for j in range(4)]
    if not ref:
        print("   hand: no hand_q in the calibration (jenga-calib --hand), finger angles not checked")
        return None
    if any(n not in positions for n in names):
        print("   hand: no hand joints in /joint_states, finger angles not checked")
        return None
    worst = 0.0
    for f in HAND_FINGERS:
        now = [math.degrees(positions[names[f * 4 + j]]) for j in range(4)]
        was = [math.degrees(ref[f * 4 + j]) for j in range(4)]
        worst = max(worst, *(abs(a - b) for a, b in zip(now, was)))
        print(f"   hand finger {f} ({'ring' if f == 3 else 'little'}): now " + " ".join(f"{v:6.1f}" for v in now) +
              "  calibrated " + " ".join(f"{v:6.1f}" for v in was) + " deg")
    return worst


def fix_hand(mover, arm: str, setup: dict) -> float | None:
    """run --fix-hand: the fingers back to the calibration (the Dino still in the jig), then checked again."""
    from jenga_vision import mount
    hand = mount.hand_node(arm, True, mover.ros.node)
    now = mount.wait_hand(hand, 3.0)
    if now is None:
        return None
    mount.close_to(hand, now, mount.calibration_hand_cmd(arm), True)
    time.sleep(mount.SETTLE_S)
    return hand_check(setup, arm, dict(mover.ros.positions))


def column_poses(face: str, arm: str) -> dict[str, list[float]]:
    """column_poses.<face> of scan_setup.json: {left / middle / right: saved pose name or joint values}."""
    names = json.loads(SETUP_FILE.read_text()).get("column_poses", {}).get(face)
    if not names:
        raise ScanError(f"--columns: no column_poses.{face} in {SETUP_FILE}")
    saved = json.loads(POSES_FILE.read_text())[arm]
    missing = [v for v in names.values() if isinstance(v, str) and v not in saved]
    if missing:
        raise ScanError(f"--columns: pose(s) {', '.join(missing)} not in {POSES_FILE} (teach them with jenga-arm)")
    return {k: saved[v] if isinstance(v, str) else v for k, v in names.items()}


def face_shots(tower: Tower, face: str, layers: range, yaw_deg: float = 0.0) -> dict[int, list[Shot]]:
    """Per layer, the end faces of `face` sorted left -> right on screen."""
    faces = tower.end_faces(face, layers)
    right = shot_pose(faces[0][2], faces[0][3], 0)[:3, 0]
    out: dict[int, list[Shot]] = {}
    for layer, slot, aim, normal in faces:
        out.setdefault(layer, []).append(Shot(layer, slot, "", aim, normal, yaw_deg))
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
    clearance = None  # clearance.Clearance: every joint move checked against the tower model first
    """Joint moves and straight Dino-frame moves; virtual (joints just set) without send."""

    def __init__(self, arm: str, tool: np.ndarray, send: bool, q0, joint_speed_deg: float, line_speed_mm: float):
        self.arm, self.tool, self.send = arm, tool, send
        self.chain = Chain(arm)
        self.joint_speed, self.line_speed = joint_speed_deg, line_speed_mm
        self.far_speed = self.parallel_speed = self.out_speed = None  # mm/s, see speed_for
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
            near = ""
            if self.clearance is not None:  # checked before anything is sent (clearance.py)
                mm, what, at = self.clearance.along(self.q, part)
                near = f", {what} {mm:.0f} mm from the tower"
                if mm < MIN_CLEAR_MM:
                    raise ScanError(f"joint move {label} would bring the {what} {mm:.0f} mm close to the tower "
                                    f"(model) at {at:.0%} of the way: not sent")
            print(f"   joint move {label}: largest joint {biggest:.0f} deg at {v:.0f} deg/s, {seconds:.1f} s{near}")
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
        if self.out_speed and out1 - out0 > PARALLEL_M and abs(out1 - out0) >= 0.9 * float(np.linalg.norm(goal - start)):
            return self.out_speed, "out"  # straight away from the face (--out-speed)
        return self.line_speed, "line"

    def line_to(self, goal_dino: np.ndarray, min_s: float = 0.4, speed: float | None = None) -> None:
        """Straight line of the Dino frame (position; orientation interpolated) to `goal_dino`
        at `speed` mm/s (default: speed_for)."""
        start = self.dino()
        speed = speed or self.speed_for(start[:3, 3], goal_dino[:3, 3])[0]
        dp = goal_dino[:3, 3] - start[:3, 3]
        rel = start[:3, :3].T @ goal_dino[:3, :3]
        ang = math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1) / 2)))
        if np.linalg.norm(dp) < 5e-5 and ang < math.radians(0.05):  # already there (IK error): no 0.4 s empty move
            return
        n = max(1, math.ceil(max(np.linalg.norm(dp) / LINE_STEP_M, ang / math.radians(1))))
        axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-9 else np.array([0, 0, 1.0])
        from jenga_vision.kinematics import axis_angle, transform
        q, path = self.q.copy(), []
        for i in range(1, n + 1):
            s = i / n
            g = transform(start[:3, :3] @ axis_angle(axis, ang * s), start[:3, 3] + dp * s)
            nq = self.chain.ik(g @ np.linalg.inv(self.tool), q)
            if nq is None:  # at a joint limit (2026-11-19 -y L14 left: j6 at -97): bend the elbow away from it
                nq = self.chain.ik(g @ np.linalg.inv(self.tool), q, avoid_limits=True, iters=1000)
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

    def __init__(self, device: int, window: bool, record: Path | None = None):
        from jenga_vision.camera import DinoCamera
        self.cam = DinoCamera(device)
        self.window = window
        self.record, self.writer = record, None  # --record: every frame to an mp4, the latest show() text on it
        self.label = ""
        self.settle = SETTLE_S
        self.cond = threading.Condition()
        self.buf: deque = deque(maxlen=64)  # (seq, time, frame, brightness)
        self.seq = 0
        self.collapse: tuple[float, float] | None = None  # (time, brightness before) of the last collapse
        self.collapses, self.waited, self.unsteady = 0, 0.0, 0
        self.last_grab = 0
        self.turn_deg = 0.0  # --soft-level: frames() and grab() hand out images turned this much (level edge)
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
            if self.record is not None:
                self._write(frame, t)
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
        out = [turn_image(x[2], self.turn_deg) for x in got]
        self.dash.frame(out[-1])
        return out

    def grab(self) -> np.ndarray:
        """The next new frame, no settling (passes while the arm moves)."""
        with self.cond:
            while not self.buf or self.buf[-1][0] <= self.last_grab:
                if not self.cond.wait(1.0):
                    raise ScanError("no frames from the Dino for 1 s")
            self.last_grab, frame = self.buf[-1][0], self.buf[-1][2]
        frame = turn_image(frame, self.turn_deg)
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

    def _write(self, frame: np.ndarray, t: float) -> None:
        view = frame.copy()
        cv2.putText(view, f"{time.strftime('%H:%M:%S', time.localtime(t))}  {self.label}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if self.writer is None:  # the first RECORD_PROBE frames give the real frame rate (playback speed)
            self.pending = getattr(self, "pending", []) + [(t, view)]
            if len(self.pending) < RECORD_PROBE:
                return
            fps = (len(self.pending) - 1) / max(self.pending[-1][0] - self.pending[0][0], 1e-3)
            h, w = view.shape[:2]
            self.writer = cv2.VideoWriter(str(self.record), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            print(f"   recording the Dino to {self.record} ({fps:.1f} frames/s)")
            for _, v in self.pending:
                self.writer.write(v)
            self.pending = []
            return
        self.writer.write(view)

    def show(self, frame: np.ndarray, text: str) -> None:
        self.label = text
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
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class FakeCamera(RealCamera):
    """Synthetic Dino image from where the point in focus is relative to a 'true' tower:
    the model tower with every end face `depth_mm` (+ `lean_mm` per 10 layers) deeper and
    the gaps `drop_mm` lower. Blur grows with the focus error, the gap band moves with
    the height error, every block has its own gap width (seeded)."""

    def __init__(self, mover: Mover, shots: dict[int, list[Shot]], depth_mm: float, lean_mm: float,
                 drop_mm: float, window: bool, roll_deg: float = 0.0, seed: int = 1, true_tool: np.ndarray | None = None):
        self.true_tool = true_tool  # where the Dino really is in the hand (--fake-tool), else the assumed tool
        self.settle = SETTLE_S
        self.dash = NoDashboard()
        self.turn_deg = 0.0
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
        return [turn_image(f, self.turn_deg) for f in self.render(n)]

    def grab(self) -> np.ndarray:
        if self.mover.send:
            time.sleep(1 / FAKE_FPS)
        return turn_image(self.render(1)[0], self.turn_deg)

    def render(self, n: int) -> list[np.ndarray]:
        ros = self.mover.ros  # during a move the virtual robot's joint states, not the target (--sweep-focus, glide)
        q = ros.current() if ros is not None and ros.current() is not None else self.mover.q
        d = self.mover.chain.fk(np.array(q, float), self.mover.tool if self.true_tool is None else self.true_tool)
        focus = d[:3, 3]
        shot = min(self.shots, key=lambda s: np.linalg.norm(s.aim - focus))
        depth, drop = self.truth(shot)
        face_point = shot.aim - shot.normal * depth
        # + = face beyond the point in focus: to the face plane along the line of sight (a tilted Dino too)
        defocus = float((face_point - focus) @ shot.normal) / max(-float(d[:3, 2] @ shot.normal), 0.2)
        lateral = float((focus - shot.aim) @ shot.pose(0.0, 0.0)[:3, 0])  # along the face
        x_true = d[:3, 0] * math.cos(self.roll) + d[:3, 1] * math.sin(self.roll)
        y_true = -d[:3, 0] * math.sin(self.roll) + d[:3, 1] * math.cos(self.roll)
        turn = math.degrees(math.atan2(x_true[2], -y_true[2]))  # the whole picture turned (any angle)
        img = self._image(shot, defocus, (focus[2] - (shot.aim[2] - drop)) / (self.mm_per_px / 1000),
                          abs(lateral) < 0.018)
        img = turn_image(img, -turn)
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
        if self.window:  # no wait: the real camera shows its latest frame while the arm moves
            self.show(turn_image(self.render(1)[0], self.turn_deg), text)

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
        if args.soft_level and "turn_deg" in self.carry:  # the same, levelled by turning the image instead
            cam.turn_deg = self.carry["turn_deg"]
        self.slot_down: dict[int, float] = {}  # m per slot above the row's height (--rigid: the face's tilt)
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

    def sweep_focus(self, shot: Shot, guess: float, down: float) -> tuple[float, float] | None:
        """--sweep-focus: the Dino moved along the face normal through `guess` +- SWEEP_HALF_MM without stopping,
        every frame's focus value against the depth the joint states give (record_pass); one more pass on the
        side the peak ran off to. (depth, focus value) with the arm at the peak, or None (no clear peak)."""
        half, v = SWEEP_HALF_MM / 1000, self.args.sweep_focus_speed
        normal = shot.normal
        recs: list[dict] = []
        lo, hi = guess - half, min(guess + half, self.deepest())
        self.go(shot, lo, down)
        for _ in range(3):
            got = self.record_pass(shot.pose(hi, down, self.roll), v, shot.aim, -normal)
            for r in got:
                r["f"] = steady_focus(r["frame"])
                recs.append(r)
            ds = np.array([r["along_mm"] for r in recs])
            fs = np.array([r["f"] for r in recs])
            k = int(np.argmax(fs))
            span = ds.max() - ds.min()
            if span < 1.0:
                return None
            # the peak inside, 0.5 mm from both ends of what was seen: found
            if ds.min() + 0.5 <= ds[k] <= ds.max() - 0.5 or fs[k] < self.args.min_focus:
                break  # inside, or nothing seen at all (the step search takes over)
            if ds[k] > ds.max() - 0.5:  # still rising inward
                lo, hi = ds.max() / 1000, min(ds.max() / 1000 + 2 * half, self.deepest())
                if hi - lo < 0.0005:
                    break
            else:  # still rising outward: come back out past where it started
                lo, hi = ds.min() / 1000 - 2 * half, ds.min() / 1000
                self.go(shot, lo, down)
                continue
        near = [(d, f) for d, f in zip(ds, fs) if abs(d - ds[k]) <= 1.0]
        if len(near) >= 5:
            a, b, _ = np.polyfit([d for d, _ in near], [f for _, f in near], 2)
            peak = -b / (2 * a) if a < 0 else ds[k]
            peak = float(np.clip(peak, ds[k] - 0.5, ds[k] + 0.5))
        else:
            peak = float(ds[k])
        best = float(fs[k])
        if best < self.args.min_focus or best < 1.15 * float(np.median(fs)):
            print(f"   {shot.screen}: focus sweep {span:.1f} mm, {len(recs)} frames: no clear peak ({best:.0f})")
            return None
        d = min(peak / 1000, self.deepest())
        self.go(shot, d, down)
        for r in recs:
            self.log_sample(shot, r["along_mm"] / 1000, down, r["f"], r["frame"])
        print(f"   {shot.screen}: focus sweep {span:.1f} mm, {len(recs)} frames, peak at {d * 1000:+.2f} mm, value {best:.1f}")
        return d, best

    def focus(self, shot: Shot, guess: float, down: float, wide: bool) -> tuple[float, float] | None:
        """(depth, focus value) of the sharpest point near `guess`, and the arm is there;
        None if there is no clear peak. SPACE in the camera window = take this depth, q = abort."""
        a = self.args
        out, inward = (a.search_out_mm, a.search_in_mm) if wide else (a.refocus_mm, a.refocus_mm)

        best = 0.0
        seen: dict[float, float] = {}  # depth mm -> value, so the search after a failed prediction reuses them

        def measure(depth_mm: float) -> float:
            nonlocal best
            key = round(depth_mm, 3)
            if key in seen:
                return seen[key]
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
            seen[key] = f
            return f

        if a.sweep_focus and not wide:
            got = self.sweep_focus(shot, guess, down)
            if got is not None:
                return got
            print(f"   {shot.screen}: back to the step search")
        g = guess * 1000
        lo, hi = g - out, min(g + inward, self.deepest() * 1000)
        try:
            if a.focus_predict:
                got = self.predict_peak(measure, g, lo, hi)
                if got is not None:
                    d0, f0 = got
                    print(f"   {shot.screen}: 3 focus samples, peak predicted at {d0:+.2f} mm, value {f0:.1f}")
                    return d0 / 1000, f0
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

    def predict_peak(self, measure, g: float, lo: float, hi: float) -> tuple[float, float] | None:
        """--focus-predict: the start and one step out, the peak where a Gaussian of width FOCUS_WIDTH_MM
        through both puts it, one sample there to confirm. (depth mm, value) with the arm there, or None
        (the caller searches as usual, the samples are reused). Offline on 242 saved searches: within
        0.5 mm in 77 %, but only 62 % within 1 mm when the start was more than 1.5 mm off the peak."""
        a = self.args
        d2 = g - a.step_mm
        if d2 < lo:
            return None
        f1, f2 = measure(g), measure(d2)
        if min(f1, f2) < a.min_focus / 3:  # on the floor: the ratio says nothing
            return None
        s = FOCUS_WIDTH_MM
        d0 = (g + d2) / 2 + s * s * (math.log(f1) - math.log(f2)) / (g - d2)
        if not (lo <= d0 <= hi) or abs(d0 - g) > PREDICT_MAX_MM:
            print(f"   predicted peak {d0:+.2f} mm is too far from the start {g:+.2f}: searching")
            return None
        f0 = measure(d0)
        if f0 < a.min_focus or f0 < PREDICT_CONFIRM * max(f1, f2):
            print(f"   predicted peak {d0:+.2f} mm not confirmed ({f0:.1f} vs {max(f1, f2):.1f}): searching")
            return None
        return d0, f0

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
        frame = self.cam.frames(1)[0]
        slope = edge_tilt(frame)
        if slope is None:
            return
        if self.args.soft_level:  # turn the image, not the arm: nothing moves, so no refocus after it
            deg = math.degrees(math.atan(slope))
            if abs(deg) < LEVEL_DEG:
                st.leveled = True
                return
            before = self.cam.turn_deg
            turned = float(np.clip(before + deg, -MAX_ROLL_DEG, MAX_ROLL_DEG))
            new = edge_tilt(turn_image(frame, turned - before))
            if new is not None and abs(new) < abs(slope):
                self.cam.turn_deg = turned
                st.leveled = True
                print(f"   edge tilted {deg:+.1f} deg -> image turned {turned:+.1f} deg "
                      f"(now {math.degrees(math.atan(new)):+.1f})")
            else:
                print(f"   edge tilted {deg:+.1f} deg, turning the image did not help (kept {before:+.1f})")
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
        ok = self.args.edge_ok_px
        if ok is not None and abs(row - h / 2) <= ok:  # --edge-ok-px: good enough where it is, no scale move yet
            return down
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
            if abs(err) <= (ok or EDGE_TOLERANCE_PX):
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
                        "turn_deg": round(self.cam.turn_deg, 2), "row": rim_row, "image": name, "brightness": round(float(np.median(frame))),
                        "rim_dark": round(rim_darkness(frame), 3),
                        "mm_per_px": round(st.mm_per_px or DEFAULT_MM_PER_PX, 6),
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
        if self.args.columns and not self.args.motion_only:
            return self.column_face(face, rows, st, column_poses(face, self.mover.arm))
        self.enter(face, far_in)
        if getattr(self.args, "calib_slot", None) is not None:
            return self.calib_face(face, rows, far_out, st)
        if self.args.blur_test is not None and not self.args.motion_only:
            return self.blur_test(face, rows, far_out, st)
        if self.args.roll_test is not None and not self.args.motion_only:
            return self.roll_test(face, rows, far_out, st)
        if self.args.defocus_test is not None and not self.args.motion_only:
            return self.defocus_test(face, rows, far_out, st)
        if self.args.repeat_test is not None and not self.args.motion_only:
            return self.repeat_test(face, rows, far_out, st)
        if self.args.glide and not self.args.motion_only:
            return self.glide_face(face, rows, far_out, st)
        if self.args.rigid and not self.args.motion_only:
            saved = json.loads((self.args.out / RIGID_FILE).read_text()).get(face) \
                if (self.args.out / RIGID_FILE).exists() else None
            if not saved and RIGID_SEED_FILE.exists():  # a new PC: the 11-18 robot scans' state, from git
                saved = json.loads(RIGID_SEED_FILE.read_text()).get(face)
                if saved:
                    print(f"   --rigid: nothing saved in {self.args.out} yet: {RIGID_SEED_FILE.name} (11-18 scans)")
            if not saved and self.args.fresh_tower:
                # nothing saved yet (a new PC: captures/ is not in git; or after calib --write): still every row
                # of the hand jig scan, each row's first block searched (rigid_face), never the top / bottom sweep
                # with the layers between interpolated (2026-11-19 teammate PC: that path pushed the tower over)
                cal = json.loads(SETUP_FILE.read_text())["dino"].get("calibration", {}).get(self.mover.arm, {})
                saved = {"session": "none yet", "turn_deg": 0.0, "roll_deg": 0.0, "layers": {}, "focus": None,
                         "mm_per_px": cal.get("mm_per_px")}
                print(f"   --rigid: no saved Dino pose for {face} yet: every row from the calibration, searched")
            if saved:
                return self.rigid_face(face, rows, far_out, st, saved)
            print(f"   --rigid: no saved Dino pose for {face} yet: a full sweep (saved for the next scan)")
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

    def edge_now(self, n: int = 5) -> tuple[float | None, float | None]:
        """Median edge row offset from the image centre (px, + = below) and edge tilt (deg) of n frames."""
        frames = self.cam.frames(n)
        rows = [r for r in (edge_row(f) for f in frames) if r is not None]
        tilts = [t for t in (edge_tilt(f) for f in frames) if t is not None]
        return (float(np.median(rows)) - 240 if rows else None,
                math.degrees(math.atan(float(np.median(tilts)))) if tilts else None)

    def calib_face(self, face: str, rows: dict[int, list[Shot]], far_out, st: FaceState) -> FaceState:
        """calib: where the Dino really is in the hand, from the calibration block (jenga_vision.autocalib).
        Wide searches, so a Dino pushed in anyhow is found: depth (a sweep through the whole range), the turn
        about its axis (image turns, any angle), height (edge up / down), sideways (the tower's ends). Then the
        tool that puts the point in focus on the model top edge of the middle block with the image level, and
        a check there with it (corrected up to autocalib.CHECKS times)."""
        from jenga_vision import autocalib
        shot = next(s for s in rows[self.args.calib_layer] if s.slot == self.args.calib_slot)
        print(f"\n== calibration block {face} L{shot.layer:02d} S{shot.slot} ({shot.screen})")
        self.cam.turn_deg, self.roll = 0.0, 0.0
        d = None
        for k, (name, tool) in enumerate(self.args.calib_candidates):
            if k > 0:  # well out from the face before the hand turns to the next guess
                self.mover.line_to(shot.pose(-autocalib.SWITCH_OUT_M, 0.0))
                self.mover.tool = tool
            print(f"   Dino as {name}: depth sweep {-self.args.search_out_mm:+.0f}..{self.args.max_in_mm:+.0f} mm")
            self.mover.line_to(shot.pose(-self.args.search_out_mm / 1000 - RETREAT_M, 0.0))
            d = self.depth_sweep(shot, 0.0)
            if d is not None:
                break
        if d is None:
            raise ScanError("no focus on the calibration block with any guess (nothing saved)")
        down = self.find_turn(shot, d)
        got = self.focus(shot, d, down, False)
        d = got[0] if got else d
        st.depth_m, st.depth_known, self.last_depth = d, True, None
        down = self.top_edge(shot, d, down, st)
        if down is None:
            raise ScanError("no top edge on the calibration block (nothing saved)")
        self.fine_turn()
        side = self.side_sweep(shot, d, down)
        here = autocalib.shifted(shot, side or 0.0)
        got = self.focus(here, d, down, False)
        d = got[0] if got else d
        down = self.center_edge(here, d, down, st)
        self.fine_turn()

        start_tool, found = self.mover.tool, None
        for k in range(1 + autocalib.CHECKS):
            err_px, tilt = self.edge_now()
            if err_px is None or tilt is None or st.mm_per_px is None:
                raise ScanError("no edge to measure on the calibration block (nothing saved)")
            self.cam.turn_deg += tilt
            found = autocalib.Found(d, down, self.roll, self.cam.turn_deg, err_px * st.mm_per_px / 1000, tilt,
                                    side if k == 0 and side else 0.0)
            print(f"   {'found' if k == 0 else f'check {k}'}: depth {d * 1000:+.2f} mm, down {down * 1000:+.2f} mm, "
                  f"sideways {found.side_m * 1000:+.1f} mm, edge {err_px:+.1f} px, "
                  f"image turn {self.cam.turn_deg:+.1f} deg (tilt {tilt:+.1f})")
            if (k > 0 and autocalib.good(found)) or k == autocalib.CHECKS:
                break
            self.mover.tool = autocalib.new_tool(self.mover.tool, shot, found)  # a shift: no hand roll
            self.last_depth = None
            self.go(shot, 0.0, 0.0)
            got = self.focus(shot, 0.0, 0.0, False)  # short search round the new tool's focus
            if got is None:
                raise ScanError("no focus with the new tool (nothing saved)")
            d, down = got[0], 0.0
        self.args.calib_result = {"tool": self.mover.tool, "start_tool": start_tool, "last": found,
                                  "mm_per_px": st.mm_per_px, "q": list(self.mover.q), "side_found": side is not None,
                                  "turn_deg": self.cam.turn_deg}
        self.mover.line_to(shot.pose(-RETREAT_M, 0.0))
        self.leave(far_out)
        return st

    def depth_sweep(self, shot: Shot, down: float) -> float | None:
        """calib: focus value through the whole depth range, outside in, stopping once past a clear peak."""
        from jenga_vision import autocalib
        seen: list[tuple[float, float]] = []
        d = -self.args.search_out_mm / 1000
        while d <= self.max_in + 1e-9:
            self.go(shot, d, down)
            f, frame = self.measure(1)
            self.log_sample(shot, d, down, f, frame)
            seen.append((d, f))
            best = max(seen, key=lambda x: x[1])
            if best[0] < d and f < autocalib.PAST_PEAK * best[1] and autocalib.clear_peak(seen):
                break
            d += autocalib.SWEEP_STEP_M
        if not autocalib.clear_peak(seen):
            vals = [f for _, f in seen]
            print(f"   no clear focus peak ({len(seen)} samples, best {max(vals):.2f}, median {np.median(vals):.2f})")
            return None
        best = max(seen, key=lambda x: x[1])
        print(f"   sharpest at {best[0] * 1000:+.1f} mm (focus {best[1]:.2f}, median {np.median([f for _, f in seen]):.2f})")
        return best[0]

    def find_turn(self, shot: Shot, depth: float) -> float:
        """calib: up / down until an edge shows at some image turn; sets cam.turn_deg, returns the height."""
        from jenga_vision import autocalib
        for off in autocalib.TURN_SEARCH_MM:
            self.go(shot, depth, off / 1000)
            turn = best_turn(self.cam.frames(1)[0])
            if turn is not None:
                self.cam.turn_deg = turn
                print(f"   edge at {off:+.0f} mm, image turned {turn:+.1f} deg to level it")
                self.up_is_up(shot, depth, off / 1000)
                return off / 1000
        raise ScanError("no edge at any image turn within "
                        f"{max(abs(v) for v in autocalib.TURN_SEARCH_MM):.0f} mm up / down (nothing saved)")

    def up_is_up(self, shot: Shot, depth: float, down: float) -> None:
        """calib: an end face looks much the same upside down, so check by moving: with the camera 1 mm lower
        the edge must go up in the image; if it goes down, the image is turned 180 deg more."""
        row0 = edge_row(self.cam.frames(1)[0])
        self.go(shot, depth, down + 0.001)
        row1 = edge_row(self.cam.frames(1)[0])
        self.go(shot, depth, down)
        if row0 is None or row1 is None or abs(row1 - row0) < 3:
            print("   (up / down not checked: the edge did not move clearly)")
            return
        if row1 > row0:
            self.cam.turn_deg = (self.cam.turn_deg + 360) % 360 - 180
            print(f"   the edge went down with the camera: upside down, image turned {self.cam.turn_deg:+.1f} deg")

    def fine_turn(self) -> None:
        _, tilt = self.edge_now(3)
        if tilt is not None:
            self.cam.turn_deg += tilt

    def side_sweep(self, shot: Shot, depth: float, down: float) -> float | None:
        """calib: sideways across the tower; where the edge is seen = the tower's end faces of this layer.
        The middle of that run = the middle block (m along the face, + = image right), or None."""
        from jenga_vision import autocalib
        on = []
        for x in autocalib.side_positions():
            self.go(autocalib.shifted(shot, x), depth, down)
            if edge_seen(self.cam.frames(1)[0]):
                on.append(x)
        run = autocalib.run_around(on)
        if run is None:
            print(f"   sideways: the tower's ends not found (edge seen at {len(on)} positions): sideways not measured")
            self.go(shot, depth, down)
            return None
        lo, hi = run
        mid = (lo + hi) / 2
        print(f"   sideways: edge seen from {lo * 1000:+.0f} to {hi * 1000:+.0f} mm: middle block at {mid * 1000:+.1f} mm")
        return mid

    def enter(self, face: str, far_in) -> None:
        a = approach(self.args)
        for via in self.args.via:
            self.mover.joint_move(saved_poses(self.mover.arm)[via], via, a["speed"], min_s=a["min_s"])
        self.mover.joint_move(far_in, f"{face} far in", a["speed"], a["slow_end"], a["min_s"])

    def leave(self, far_out) -> None:
        a = approach(self.args)
        self.mover.line_to(np.array(self.mover.chain.fk(far_out, self.mover.tool)))
        for k, via in enumerate(reversed(self.args.via)):
            self.mover.joint_move(saved_poses(self.mover.arm)[via], via, a["speed"],
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
        if self.args.soft_level:  # the image is turned, the arm stays: one centring, no second one
            if not edge_seen(self.cam.frames(1)[0]):  # 2026-11-16 +x L13: focused on the block above the edge
                print(f"   {shot.screen}: edge not in view after focusing, looking for it")
                found_at = self.find_edge(shot, d, down)
                down = down if found_at is None else found_at
            down = self.center_edge(shot, d, down, st)
            self.level(shot, d, down, st)
            if abs(down - focused_at) > REFOCUS_MOVE_M:  # other part of the face in view
                found = self.focus(shot, d, down, False)
                if found is not None:
                    d, f = found
                    st.depth_m = d
            self.go(shot, d, down)
        elif edge_seen(self.cam.frames(1)[0]):
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
            # the row may centre the edge again (2026-11-16 +x: L13 not in view at the first one's height): the
            # sweep goes by the height the row ended at, not the one lock_on started from
            ref[layer] = self.shoot_row(face, layer, shots(layer, True), got[0], got[1], got[2], st)
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

    def learn_slot_down(self, st: FaceState, layer: int) -> None:
        """--fresh-tower: the face's sideways lean from the rows shot so far (where each edge really was against
        its row's mean), so the next rows send each block to its own height instead of centring the far ones."""
        scale = st.mm_per_px or DEFAULT_MM_PER_PX
        rows = [r for r in st.rows if r["layer"] == layer]
        if len(rows) < 2:
            return
        edges = {r["slot"]: r["down_mm"] + (r["row"] - 240) * scale for r in rows}
        mean = float(np.mean(list(edges.values())))
        self.slot_seen = getattr(self, "slot_seen", {})
        for slot, e in edges.items():
            self.slot_seen.setdefault(slot, []).append(e - mean)
        self.slot_down = {k: float(np.median(v)) / 1000 for k, v in self.slot_seen.items()}
        print("   this face so far, per block height: " +
              " ".join(f"S{k} {v * 1000:+.2f} mm" for k, v in sorted(self.slot_down.items())))

    def column_face(self, face: str, rows: dict[int, list[Shot]], st: FaceState, poses: dict) -> FaceState:
        """--columns (2026-11-18 hand jig: the wrist is next to the lens, one Dino turn can't reach all three
        columns of a face): one taught arm pose per column of end faces (left / middle / right on the face). Per
        column: a joint move to COLUMN_BACK_M behind the taught pose along the lens, straight in, the full search
        on its first block, then the other layers with the same Dino turn, only the height changing (a short
        refocus when a shot is out of focus), straight back out along the lens."""
        normal = next(iter(rows.values()))[0].normal
        layers = sorted(rows, reverse=self.args.top_first)
        chain, tool = self.mover.chain, self.mover.tool
        usual = None
        for c, screen in enumerate(s for s in ends.BLOCKS if s in poses):
            q_taught = np.array(poses[screen], float)
            taught = chain.fk(q_taught, tool)
            back = taught.copy()
            back[:3, 3] -= taught[:3, 2] * COLUMN_BACK_M
            q_back = chain.ik(back, q_taught, tool)
            if q_back is None:
                raise ScanError(f"{screen} column: no IK {COLUMN_BACK_M * 1000:.0f} mm behind the taught pose")
            col = [replace(sh, rot=taught[:3, :3]) for layer in layers for sh in rows[layer] if sh.screen == screen]
            if c % 2:  # back and forth: the next column starts at the end the last one finished
                col = col[::-1]
            print(f"\n== {face} column {screen}: {len(col)} end faces, L{col[0].layer:02d} -> L{col[-1].layer:02d}")
            a = approach(self.args)
            self.mover.joint_move(q_back, f"{screen} column, {COLUMN_BACK_M * 1000:.0f} mm back", a["speed"],
                                  a["slow_end"], a["min_s"])
            st.leveled = not self.args.soft_level  # another arm turn: the image tilt is read again
            self.mover.line_to(col[0].pose(st.depth_m - RETREAT_M, st.down_m, self.roll))
            got = self.lock_on(col[0], st, wide=not st.depth_known)
            if got is None:
                raise ScanError(f"no focus / edge on {col[0].screen} L{col[0].layer:02d}: the column needs it")
            usual = usual or got[2]
            d, down = self.shoot_row(face, col[0].layer, [col[0]], *got, st)
            for sh in col[1:]:
                d, down = self.shoot_row(face, sh.layer, [sh], d, down, None, st, refocus_below=self.args.refocus_below * usual)
            out = self.mover.dino().copy()
            out[:3, 3] -= out[:3, 2] * COLUMN_BACK_M
            self.mover.line_to(out)
        return st

    def rigid_face(self, face: str, rows: dict[int, list[Shot]], far_out, st: FaceState, saved: dict) -> FaceState:
        """--rigid: the Dino sits in the hand as in the last scan of this face (`saved`, RIGID_FILE), so its
        turn, the image scale and the edge height are known: a short focus search on the first block only,
        then every row in order (top to bottom on -y) at the saved depth and height of the layer plus what
        the rows so far found different. A row whose first shot is out of focus is refocused."""
        layers = sorted(rows, reverse=self.args.top_first)
        normal = rows[layers[0]][0].normal
        known = sorted((int(k), v) for k, v in saved["layers"].items())
        ks = [k for k, _ in known]

        # a straight line through the saved layers: one badly focused layer of the last scan must not carry its
        # error into this one (2026-11-16 robot: a 1.7 mm wrong refocus saved for 4 layers came back as offsets)
        fits = [np.polyfit(ks, [v[i] for _, v in known], 1) if len(ks) >= 3 else None for i in (0, 1)]

        def saved_at(layer: int) -> tuple[float, float]:
            if self.args.fresh_tower:  # nothing of the last tower: the model face, corrected row by row
                return self.args.prior_depth_mm / 1000, self.args.prior_down_mm / 1000
            return tuple((float(np.polyval(fit, layer)) if fit is not None
                          else float(np.interp(layer, ks, [v[i] for _, v in known]))) / 1000
                         for i, fit in enumerate(fits))

        st.mm_per_px = saved.get("mm_per_px") or st.mm_per_px
        self.slot_down = {} if self.args.fresh_tower else \
            {int(k): v / 1000 for k, v in saved.get("slot_down_mm", {}).items()}
        if self.args.fresh_tower:
            print("   --fresh-tower: the Dino as in the last scan, nothing of the last tower")
        if self.slot_down:
            print("   per block height from the last scan: " +
                  " ".join(f"S{k} {v * 1000:+.2f} mm" for k, v in sorted(self.slot_down.items())))
        st.height_known = True
        st.leveled = not self.args.soft_level  # the Dino may turn in a round jig: read the tilt again on the first
        # block (soft: on the image already there, no move); the arm-roll levelling keeps the saved roll
        self.roll = st.roll = math.radians(saved["roll_deg"])
        self.cam.turn_deg = saved["turn_deg"]
        print(f"   rigid Dino: as in {saved['session']} (image turned {saved['turn_deg']:+.1f} deg, "
              f"roll {saved['roll_deg']:+.1f} deg)")
        retreat = self.args.sweep_retreat_mm / 1000
        off_d = off_h = 0.0
        for i, layer in enumerate(layers):
            d0, h0 = saved_at(layer)
            d, down = d0 + off_d, h0 + off_h
            row = rows[layer]
            at = self.mover.dino()[:3, 3]
            if np.linalg.norm(row[-1].aim - at) < np.linalg.norm(row[0].aim - at):
                row = row[::-1]
            print(f"\n== {face} L{layer:02d} rigid: depth {d * 1000:+.1f} mm, down {down * 1000:+.1f} mm")
            if i == 0 and self.args.rigid_trust and saved.get("focus"):  # no search: a check like the other rows
                self.mover.line_to(row[0].pose(d - RETREAT_M, down, self.roll))
                self.last_depth = d
                self.go(row[0], d, down)
                self.level(row[0], d, down, st)
                d, down = self.shoot_row(face, layer, row, d, down, None, st,
                                         refocus_below=self.args.refocus_below * saved["focus"])
            elif i == 0:
                self.mover.line_to(row[0].pose(d - RETREAT_M, down, self.roll))
                st.depth_m, st.down_m = d, down
                got = self.lock_on(row[0], st, wide=False)
                if got is None:
                    raise ScanError(f"no focus on L{layer:02d} {row[0].screen} near the saved depth")
                d, down = self.shoot_row(face, layer, row, *got, st)
            else:
                self.back_off(normal, retreat)
                self.mover.line_to(row[0].pose(d - retreat, down, self.roll))
                # the saved face median, not the few shots so far (one block's value varies 30-120 at one depth)
                usual = saved.get("focus") or float(np.median([r["focus"] for r in st.rows]))
                d, down = self.shoot_row(face, layer, row, d, down, None, st, refocus_below=self.args.refocus_below * usual)
            off_d, off_h = d - d0, down - h0
            if self.args.fresh_tower:
                self.learn_slot_down(st, layer)
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

    def roll_test(self, face: str, rows: dict[int, list[Shot]], far_out, st: FaceState) -> FaceState:
        """--roll-test LAYER: level the Dino with the arm on the layer's middle block (the usual way),
        then turn it by each of --roll-test-degs about its axis, the depth and height kept. Each image is
        measured as it is (no levelling) and turned level in software (--soft-level), against the
        levelled shot; roll_test.csv and the images (as taken)."""
        a = self.args
        if a.soft_level:
            raise ScanError("--roll-test compares against levelling with the arm: leave --soft-level off")
        layer = a.roll_test
        if layer not in rows:
            raise ScanError(f"L{layer:02d} has no end faces on {face} (have {sorted(rows)})")
        row = rows[layer]
        mid = row[len(row) // 2]
        print(f"\n== {face} L{layer:02d} roll test on the {mid.screen} block")
        self.mover.line_to(mid.pose(st.depth_m - RETREAT_M, st.down_m, self.roll))
        got = self.lock_on(mid, st, wide=True)
        if got is None:
            raise ScanError("no focus / edge on the middle block: the roll test needs it")
        d, down, _ = got
        base = self.roll
        records = []
        for off in a.roll_test_degs:
            self.roll = float(np.clip(base + math.radians(off), -math.radians(MAX_ROLL_DEG), math.radians(MAX_ROLL_DEG)))
            self.go(mid, d, down)
            frames = self.cam.frames(ends.FRAMES_PER_SHOT, settle=SETTLE_S)
            raw = float(np.median([ends.measure_end(f).gap for f in frames]))
            slopes = [t for f in frames if (t := edge_tilt(f)) is not None]
            seen_deg = math.degrees(math.atan(float(np.median(slopes)))) if slopes else None
            soft = (float(np.median([ends.measure_end(turn_image(f, seen_deg)).gap for f in frames]))
                    if seen_deg is not None else None)
            name = f"roll_{off:+05.1f}.png"
            cv2.imwrite(str(self.session / name), frames[-1])
            rec = {"turn_deg": off, "roll_deg": round(math.degrees(self.roll), 2), "edge_tilt_deg":
                   "" if seen_deg is None else round(seen_deg, 2), "gap_raw": raw,
                   "gap_soft": "" if soft is None else soft, "focus": round(steady_focus(frames[-1]), 1),
                   "edge_row": "" if (r := edge_row(frames[-1])) is None else round(r), "image": name}
            records.append(rec)
            print(f"   turned {off:+5.1f} deg: edge tilt {rec['edge_tilt_deg']}, gap as is {raw:.2f} px, "
                  f"turned level {rec['gap_soft']} px, focus {rec['focus']}")
        self.roll = base
        self.go(mid, d, down)
        self.back_off(mid.normal, RETREAT_M)
        self.leave(far_out)
        with (self.session / "roll_test.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        levelled = [r["gap_raw"] for r in records if r["turn_deg"] == 0.0]
        ref = float(np.median(levelled)) if levelled else records[0]["gap_raw"]
        print(f"\nroll test, levelled gap {ref:.2f} px (repeats at 0 deg: {', '.join(f'{g:.2f}' for g in levelled)})")
        for r in records:
            soft = "-" if r["gap_soft"] == "" else f"{r['gap_soft'] - ref:+.2f}"
            print(f"   {r['turn_deg']:+5.1f} deg: as is {r['gap_raw'] - ref:+.2f} px, turned level {soft} px")
        print(f"-> {self.session / 'roll_test.csv'}")
        return st

    def repeat_test(self, face: str, rows: dict[int, list[Shot]], far_out, st: FaceState) -> FaceState:
        """--repeat-test LAYER: the full search (focus, edge, top edge, level, scale) on the layer's middle
        block, then its three blocks shot there. Appended to <out>/REPEAT_FILE with --repeat-label; run it
        again after putting the Dino back in the jig (or picking it again) and the table of all runs of this
        face and layer shows how far the Dino moved: what --rigid assumes is 0."""
        a = self.args
        layer = a.repeat_test
        if layer not in rows:
            raise ScanError(f"L{layer:02d} has no end faces on {face} (have {sorted(rows)})")
        row = rows[layer]
        mid = row[len(row) // 2]
        print(f"\n== {face} L{layer:02d} repeat test ({a.repeat_label})")
        self.mover.line_to(mid.pose(st.depth_m - RETREAT_M, st.down_m, self.roll))
        got = self.lock_on(mid, st, wide=True)
        if got is None:
            raise ScanError("no focus / edge on the middle block: the repeat test needs it")
        d, down, f = got
        self.shoot_row(face, layer, [mid] + [s for s in row if s is not mid], d, down, f, st)
        self.back_off(mid.normal, RETREAT_M)
        self.leave(far_out)
        path = a.out / REPEAT_FILE
        new = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["label", "session", "face", "layer", "block", "depth_mm", "down_mm", "turn_deg", "roll_deg",
                            "um_per_px", "edge_row", "gap", "focus"])
            for r in st.rows:
                w.writerow([a.repeat_label, self.session.name, face, layer, r["screen"], r["depth_mm"], r["down_mm"],
                            r["turn_deg"], r["roll_deg"], "" if st.mm_per_px is None else round(st.mm_per_px * 1e6, 1),
                            r["row"], r["gap"], r["focus"]])
        with path.open() as fh:
            runs = [r for r in csv.DictReader(fh) if r["face"] == face and int(r["layer"]) == layer]
        print(f"\nrepeat test {face} L{layer:02d}, all runs in {path} (middle block; gaps left / middle / right px)")
        by: dict[str, list[dict]] = {}
        for r in runs:
            by.setdefault(r["session"], []).append(r)
        mids = []
        for sess, rs in by.items():
            m = next((r for r in rs if r["block"] == "middle"), rs[0])
            g = {r["block"]: float(r["gap"]) for r in rs}
            mids.append(m)
            print(f"   {m['label'][:16]:16s} {sess[-15:]}: depth {float(m['depth_mm']):+6.2f} mm, down "
                  f"{float(m['down_mm']):+6.2f} mm, turn {float(m['turn_deg']):+5.1f} deg, roll "
                  f"{float(m['roll_deg']):+5.1f} deg, gaps " + " / ".join(f"{g.get(b, float('nan')):.1f}" for b in ends.BLOCKS))
        if len(mids) > 1:
            span = lambda k: max(float(m[k]) for m in mids) - min(float(m[k]) for m in mids)
            print(f"   spread (max - min) over {len(mids)} runs: depth {span('depth_mm'):.2f} mm, down "
                  f"{span('down_mm'):.2f} mm, turn {span('turn_deg'):.1f} deg, roll {span('roll_deg'):.1f} deg")
            print("   --rigid is fine when depth < 1 mm, down < 0.8 mm (edge within 40 px) and turn + roll < 2 deg")
        return st

    def defocus_test(self, face: str, rows: dict[int, list[Shot]], far_out, st: FaceState) -> FaceState:
        """--defocus-test LAYER: focus on the layer's middle block as usual, then per block of the row its own
        focus peak (full search, no prediction) and shots at that peak + each of --defocus-test-mm, the height
        kept: how much the gap px (and the 빼안빼 call of the layer) changes when the depth is off (what
        --rigid-trust risks), and how far apart the blocks' peaks are (one depth per row). defocus_test.csv."""
        a = self.args
        layer = a.defocus_test
        if layer not in rows:
            raise ScanError(f"L{layer:02d} has no end faces on {face} (have {sorted(rows)})")
        row = rows[layer]
        mid = row[len(row) // 2]
        print(f"\n== {face} L{layer:02d} defocus test")
        self.mover.line_to(mid.pose(st.depth_m - RETREAT_M, st.down_m, self.roll))
        got = self.lock_on(mid, st, wide=True)
        if got is None:
            raise ScanError("no focus / edge on the middle block: the defocus test needs it")
        d, down, _ = got
        a.focus_predict = False  # the reference peaks must be searched, not predicted (2026-11-16: 1.5 mm short)
        records = []
        peaks = {}
        for shot in row:
            found = self.focus(shot, d, down, False)
            peak = found[0] if found else d
            peaks[shot.screen] = peak
            print(f"   {shot.screen}: own peak {peak * 1000:+.2f} mm ({(peak - d) * 1000:+.2f} from the middle's first)"
                  + ("" if found else " (no peak found: the middle's depth used)"))
            for off in a.defocus_test_mm:
                self.go(shot, peak + off / 1000, down)
                frames = self.cam.frames(ends.FRAMES_PER_SHOT, settle=SETTLE_S)
                gap = float(np.median([ends.measure_end(f).gap for f in frames]))
                name = f"defocus_{shot.screen}_{off:+04.1f}.png"
                cv2.imwrite(str(self.session / name), frames[-1])
                records.append({"block": shot.screen, "offset_mm": off, "depth_mm": round((peak + off / 1000) * 1000, 2),
                                "gap": gap, "focus": round(steady_focus(frames[-1]), 1),
                                "edge_row": "" if (r := edge_row(frames[-1])) is None else round(r), "image": name})
                print(f"   {shot.screen:6s} {off:+4.1f} mm: gap {gap:.2f} px, focus {records[-1]['focus']}")
        self.go(row[-1], peaks[row[-1].screen], down)
        self.back_off(mid.normal, RETREAT_M)
        self.leave(far_out)
        with (self.session / "defocus_test.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        print(f"\ndefocus test L{layer:02d}: block peaks " + ", ".join(f"{b} {v * 1000:+.2f}" for b, v in peaks.items())
              + f" mm (spread {(max(peaks.values()) - min(peaks.values())) * 1000:.2f} mm)")
        print("gap px minus the in-focus gap (median of the 0 mm shots); focus value")
        names = [s.screen for s in row]
        ref = {b: float(np.median([r["gap"] for r in records if r["block"] == b and r["offset_mm"] == 0.0]))
               for b in names}
        slot = {"left": 0, "middle": 1, "right": 2}
        call0 = layer_margin({(0, slot[b]): ref[b] for b in names}, 0)
        for off in dict.fromkeys(a.defocus_test_mm):
            at = {r["block"]: r for r in records if r["offset_mm"] == off}
            call = layer_margin({(0, slot[b]): at[b]["gap"] for b in names}, 0)
            same = "same call" if call and call0 and (call[0] >= 0) == (call0[0] >= 0) else "CALL CHANGES"
            print(f"   {off:+4.1f} mm: " + "  ".join(f"{b} {at[b]['gap'] - ref[b]:+.2f} ({at[b]['focus']:.0f})"
                                                for b in names) + f"   {same}")
        print(f"-> {self.session / 'defocus_test.csv'}")
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
        if m <= 0:
            return
        goal = self.mover.dino().copy()
        goal[:3, 3] += normal * m
        self.mover.line_to(goal)

    def shoot_row(self, face: str, layer: int, row: list[Shot], depth: float, down: float, focus: float | None,
                  st: FaceState, refocus_below: float | None = None) -> tuple[float, float]:
        """Shoot `row` at one depth and edge height; the first shot of a reference row is
        where the arm already is. `refocus_below` (--rigid): the first shot's focus value under this ->
        a short focus search first. --lazy-row: a side wider than the middle makes the layer 빼안빼 whatever
        the third block is, so that one is not shot. (depth, down) at the end of the row."""
        seen: dict[str, float] = {}
        base = down  # the row's height; each block goes to base + its slot's offset (--rigid, slot_down)
        for k, shot in enumerate(row):
            off = self.slot_down.get(shot.slot, 0.0)
            down = base + off
            if not (k == 0 and focus is not None):
                self.go(shot, depth, down)
            self.dash.event(action="shot")
            frames = self.cam.frames(ends.FRAMES_PER_SHOT, settle=self.args.shot_settle_s)
            f = steady_focus(frames[-1]) if focus is None or k else focus
            if k == 0 and refocus_below is not None and f < refocus_below:
                print(f"   {shot.screen}: focus {f:.0f} below {refocus_below:.0f}, refocusing")
                found = self.focus(shot, depth, down, False)
                if found is not None and found[1] > f:  # 2026-11-16 robot: a "peak" 1.7 mm off, below the start
                    depth, f = found
                    st.depth_m = depth
                elif found is not None:
                    print(f"   {shot.screen}: the peak found ({found[1]:.0f}) is not better than where it was: kept")
                self.go(shot, depth, down)
                frames = self.cam.frames(ends.FRAMES_PER_SHOT, settle=self.args.shot_settle_s)
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
                    base = down - off
                    self.go(shot, depth, down)
                self.shoot(face, shot, depth, down, f, st, settle=self.args.shot_settle_s)
            self.save(st)
            seen[shot.screen] = st.rows[-1]["gap"]
            if self.args.lazy_row and k == 1 and len(row) == 3 and "middle" in seen:
                side = next(b for b in seen if b != "middle")
                if seen[side] > seen["middle"]:
                    print(f"   {side} {seen[side]:.1f} px > middle {seen['middle']:.1f} px: 빼안빼 whatever the "
                          f"{row[2].screen} is, not shot")
                    break
        return depth, base

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


def session_scale(session: Path, row: dict) -> tuple[float, str]:
    """(mm per px, where it came from) of a shots.csv row: its own column, else the RIGID_FILE next to the
    session when it was written by that session (scans before 2026-11-18), else the default."""
    if row.get("mm_per_px"):
        return float(row["mm_per_px"]), "measured"
    rigid = session.parent / RIGID_FILE
    if rigid.exists():
        saved = json.loads(rigid.read_text()).get(row["face"], {})
        if saved.get("session") == session.name and saved.get("mm_per_px"):
            return float(saved["mm_per_px"]), "measured"
    return DEFAULT_MM_PER_PX, "default"


def block_geometry(sessions: list[Path]) -> tuple[list[dict], list[dict], dict, dict]:
    """What the shots say about the real block sizes and where the blocks are, for the pulling code: per block
    (blocks), per layer (layers) and a straight line through the layer heights per face (fit). World frame of
    config/scan_setup.json (the teammate's tower_pose). A later session wins on the same block, like load_gaps.
      gap_mm   dark band above the end face = how much lower this block is than the one(s) the layer above
               rests on (block heights differ by tenths of a mm): the 잔차 the pull order is made of
      top_z_m  world z of the block's top at its end face: arm height of the shot + where the band sits in the
               image - half the band. Precise relative to the other blocks of the face (image, ~0.02 mm/px);
               absolute only as good as jenga-calib
      dz_mm    top_z_m - model top (z0 + (layer + 1) x layer_h of scan_setup): the error stacked up so far
      out_mm   end face past the model face, outward. From the focus depth: +-1 mm (the depth of field), and on
               sweep / --rigid rows the depth of the row, not of the block"""
    setup = json.loads(SETUP_FILE.read_text())
    t = setup["tower"]
    tower = Tower(t, 1.0)  # aim at the top edge, as the shots
    found: dict[tuple[int, int], dict] = {}
    for s in sessions:
        with (s / "shots.csv").open() as f:
            for r in csv.DictReader(f):
                layer, slot, face = int(r["layer"]), int(r["slot"]), r["face"]
                aim, normal = next((a, n) for lay, sl, a, n in tower.end_faces(face, range(layer, layer + 1))
                                   if sl == slot)
                mm, scale = session_scale(s, r)
                gap_mm = float(r["gap"]) * mm
                band_z = aim[2] - float(r["down_mm"]) / 1000 - (float(r["row"]) - 240) * mm / 1000
                top_z = band_z - gap_mm / 2000
                model_top = t["z0"] + (layer + 1) * t["layer_h"]
                end = aim - normal * float(r["depth_mm"]) / 1000
                found[(layer, slot)] = {
                    "block": f"jenga_L{layer:02d}_S{slot}", "layer": layer, "slot": slot, "face": face,
                    "gap_px": float(r["gap"]), "gap_mm": round(gap_mm, 3), "top_z_m": round(float(top_z), 5),
                    "dz_mm": round(float(top_z - model_top) * 1000, 2), "out_mm": round(-float(r["depth_mm"]), 2),
                    "end_face_top_m": [round(float(end[0]), 5), round(float(end[1]), 5), round(float(top_z), 5)],
                    "normal": [round(float(v), 4) for v in normal[:2]], "scale": scale,
                    "session": s.name, "image": r["image"]}
    blocks = [found[k] for k in sorted(found)]
    layers = []
    for layer in sorted({b["layer"] for b in blocks}):
        bs = [b for b in blocks if b["layer"] == layer]
        carry = max(bs, key=lambda b: b["top_z_m"])  # the layer above rests on the highest block
        layers.append({"layer": layer, "face": carry["face"], "blocks": len(bs), "top_z_m": carry["top_z_m"],
                       "dz_mm": carry["dz_mm"], "carries": carry["block"],
                       "spread_mm": round(max(b["top_z_m"] for b in bs) * 1000 - min(b["top_z_m"] for b in bs) * 1000, 2),
                       "out_mm": round(float(np.mean([b["out_mm"] for b in bs])), 2)})
    # per layer, the rise from the lowest scanned layer, and that rise minus the teammate's nominal one
    # (jenga_tf block height, tower.jenga_tf_layer_h): the error stacked up over the scanned layers (2026-11-18)
    nominal = t.get("jenga_tf_layer_h", t["layer_h"])
    if layers:
        base = layers[0]
        for lay in layers:
            rise = (lay["top_z_m"] - base["top_z_m"]) * 1000
            lay["rise_mm"] = round(rise, 2)
            lay["rise_residual_mm"] = round(rise - (lay["layer"] - base["layer"]) * nominal * 1000, 2)
            lay["measured"] = [b["slot"] for b in blocks if b["layer"] == lay["layer"]]
        # per block of every scanned layer, measured or not (a 빼안빼 layer skips one side): its top against an
        # ideal stack of jenga_tf blocks on the lowest scanned layer, and its bottom (= the top of the layer below)
        by_layer = {lay["layer"]: lay for lay in layers}
        for b in blocks:
            rise = (b["top_z_m"] - base["top_z_m"]) * 1000
            b["top_rise_mm"] = round(rise, 2)
            b["top_residual_mm"] = round(rise - (b["layer"] - base["layer"]) * nominal * 1000, 2)
    else:
        by_layer, base = {}, None
    block_z = {}
    measured = {(b["layer"], b["slot"]): b for b in blocks}
    # A block not shot (a 빼안빼 layer skips one side), and for comparison every shot block as if it had not been:
    #   logic: its top is at most the underside of the layer above at its place (top_max). That underside comes
    #     from the other blocks of the layer (top + gap) and the face's sideways lean (per slot, the median offset
    #     of the underside from its layer's mean over the other layers of this face; constant within +-0.1 mm on
    #     the 2026-11-18 scans).
    #   data: the gap of such a block is typically EST_GAP_MM (skipped blocks of 273 robot layers: median 0.1 mm,
    #     90 % under 0.23 mm), so top_est = top_max - EST_GAP_MM; checked on 72 skipped blocks of 15 full scans:
    #     error median 0.14 mm, 90 % 0.33 mm (the layer's highest block instead: 0.60 / 0.92 mm).
    under = {k: b["top_z_m"] * 1000 + b["gap_mm"] for k, b in measured.items()}
    face_of = {lay["layer"]: lay["face"] for lay in layers}

    def estimate(layer: int, slot: int) -> tuple[float, float] | None:
        others = [s for s in range(3) if s != slot and (layer, s) in under]
        if not others:
            return None
        offs: dict[int, list[float]] = {}
        for l in by_layer:
            if l == layer or face_of[l] != face_of[layer]:
                continue
            here = {s: under[(l, s)] for s in range(3) if (l, s) in under}
            if len(here) >= 2 and slot in here:
                mean = float(np.mean(list(here.values())))
                for s, u in here.items():
                    offs.setdefault(s, []).append(u - mean)
        off = {s: float(np.median(v)) for s, v in offs.items()}
        if slot not in off or any(s not in off for s in others):
            u = float(np.mean([under[(layer, s)] for s in others]))  # no lean known: flat
        else:
            u = float(np.mean([under[(layer, s)] - off[s] for s in others])) + off[slot]
        return u, u - EST_GAP_MM

    for layer, lay in sorted(by_layer.items()):
        below = by_layer.get(layer - 1)
        for slot in range(3):
            b = measured.get((layer, slot))
            est = estimate(layer, slot)
            if b:
                top, source = b["top_z_m"] * 1000, "measured"
            elif est:
                top, source = est[1], "estimated"
            else:
                top, source = lay["top_z_m"] * 1000, "layer"
            rise = top - base["top_z_m"] * 1000

            def residual(z: float) -> float:
                return round(z - base["top_z_m"] * 1000 - (layer - base["layer"]) * nominal * 1000, 2)

            entry = {"source": source, "top_z_m": round(top / 1000, 5), "top_rise_mm": round(rise, 2),
                     "top_residual_mm": residual(top)}
            if est:  # shot blocks too: what the estimate would have said, to judge the fast mode by
                entry.update(est_top_max_residual_mm=residual(est[0]), est_top_residual_mm=residual(est[1]),
                             est_error_mm=0.3)
            if below:
                entry["bottom_residual_mm"] = below["rise_residual_mm"]  # it sits on the layer below
            block_z[f"jenga_L{layer:02d}_S{slot}"] = entry
    fit = {}
    for face in sorted({lay["face"] for lay in layers}):
        ls = [lay for lay in layers if lay["face"] == face]
        if len(ls) >= 2:
            slope, cut = np.polyfit([lay["layer"] for lay in ls], [lay["top_z_m"] for lay in ls], 1)
            fit[face] = {"layer_h_mm": round(float(slope) * 1000, 3), "z0_m": round(float(cut - slope), 5),
                         "layers": [lay["layer"] for lay in ls]}  # top of layer L = z0 + (L + 1) x layer_h
    return blocks, layers, fit, block_z


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
    blocks, layers, fit, block_z = block_geometry(sessions)
    tower_t = json.loads(SETUP_FILE.read_text())["tower"]
    nominal_h = tower_t.get("jenga_tf_layer_h", tower_t["layer_h"])
    print(f"layer heights (top of the highest block; rise from the lowest scanned layer, residual vs "
          f"{nominal_h * 1000:.2f} mm per layer):")
    for lay in layers:
        print(f"   floor {lay['layer'] + 1:2d} ({lay['face']})  top z {lay['top_z_m'] * 1000:7.2f} mm  "
              f"rise {lay['rise_mm']:7.2f} mm  residual {lay['rise_residual_mm']:+5.2f} mm  "
              f"blocks within {lay['spread_mm']:.2f} mm ({len(lay['measured'])} shot)  end face {lay['out_mm']:+.1f} mm")
    for face, f in fit.items():
        print(f"   {face}: {f['layer_h_mm']:.3f} mm per layer")
    if out:
        out.write_text(json.dumps({
            "version": 1, "source": "camera", "stamp": time.time(),
            "candidates": [{"block": f"jenga_L{layer:02d}_S{slot}", "score": scores[(layer, slot)],
                            "reason": f"빼안빼 rule {n} ({why})"} for n, (layer, slot, why) in enumerate(seq, 1)],
            # added 2026-11-18, read by nothing in jenga_game yet (rules.load_candidates keeps them as meta)
            "geometry_note": "world frame of the tower pose. layers[]: top_z_m = top of the highest block of the "
                             "layer; rise_mm = from the lowest scanned layer; rise_residual_mm = rise_mm - layers "
                             f"x {nominal_h * 1000:.2f} mm (jenga_tf block height): the stacked-up error; measured = "
                             "slots shot (a 빼안빼 layer may skip one). block_z.<block>: every block of every scanned layer: "
                             "top_residual_mm = its top minus an ideal stack from the lowest scanned layer; source "
                             "measured / estimated (not shot: est_top_residual_mm, +-0.3 mm); est_top_max_residual_mm "
                             "= the logical upper bound (underside of the layer above); shot blocks carry the estimate "
                             "too, to compare; bottom_residual_mm = the layer below. blocks[]: gap_mm = how much lower the block is "
                             "than the highest of its layer; dz_mm vs the scan model; out_mm +-1 mm. Relative values "
                             "are 0.02 mm/px precise, top_z_m absolute only as good as the Dino calibration",
            "block_z": block_z, "blocks": blocks, "layers": layers, "layer_fit": fit},
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
    # run_faces.<face> in scan_setup.json overrides RUN_FACES (2026-11-18 hand jig: other IK seeds, no old via
    # poses, back to home_1118), so the robot-tested v2 settings stay as they were
    faces = [(f, {**s, **setup.get("run_faces", {}).get(f, {}), **FACE_ROUTE.get(f, {})})
             for f, s in RUN_FACES if s["face"] in args.faces]
    args.planned_total = sum(len(r) for f, _ in faces for r in face_shots(tower, f, layers).values())
    args.dash.event(total=args.planned_total)

    cal = setup["dino"].get("calibration", {}).get(args.arm, {})
    print(f"Dino calibration of {cal.get('date', '?')} ({cal.get('pose', '?')}), image turn "
          f"{setup['dino'].get('image_turn_deg', 0.0):+.1f} deg: {SETUP_FILE}")
    if args.send:
        steps = ([] if args.no_pick else ["pick_dino"]) + [f for f, _ in faces] + ([] if args.no_place else ["place_dino"])
        print("the robot will run: " + " -> ".join(steps))
        print(f"layers {args.layers[0]}..{args.layers[1]}, dial 40, jenga-view and jenga-arm closed, arm at home")
        if args.yes or not sys.stdin.isatty():  # ros2 launch / jenga_game: the start button was the confirmation
            print("--yes (or no terminal): starting without Enter")
        elif input("Enter = start, q = quit > ").strip().lower() == "q":
            return
        t0 = time.time()
    start = faces[0][1].get("seed") if faces and args.no_pick else None
    if start:  # the fixed route starts there (ROUTE, FACE_ROUTE): -y from the teammate's home
        go_home(args, start, saved_poses(args.arm)[start], tol_deg=START_TOL_DEG)
        args.start = start
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

    tests = (args.blur_test, args.roll_test, args.defocus_test, args.repeat_test)
    if not args.motion_only and all(t is None for t in tests):  # no scan: keep the last pull order
        print("\n===== pull order =====")
        out = args.out / time.strftime("order_%Y%m%d_%H%M%S.json")
        order_main(sessions, out)
        state_dir = os.environ.get("JENGA_STATE_DIR")
        if state_dir:
            shutil.copy(out, Path(state_dir) / "jenga_pick_candidates.json")
            print(f"copied to {state_dir}/jenga_pick_candidates.json")
        if GAME_BLOCK_FILE is not None:  # jenga_game / jenga_pick / jenga_stack read this first
            shutil.copy(out, GAME_BLOCK_FILE)
            print(f"copied to {GAME_BLOCK_FILE}")
        print("sessions: " + " ".join(map(str, sessions)))
        args.dash.event(order=order_for_page(load_gaps(sessions)))

    if args.camera_holder and args.camera_holder.get("cam"):
        args.camera_holder["cam"].release()
    if not args.no_place and not timed_task("place_dino"):
        print("stopped in place_dino: the pull order above is still valid")
        return report()
    if not args.no_home:
        name = args.home or json.loads(SETUP_FILE.read_text()).get("run_home")
        if name:
            go_home(args, name, saved_poses(args.arm)[name])
        elif args.arm in TEAMMATE_HOME:
            go_home(args, "the teammate's home", TEAMMATE_HOME[args.arm])
    args.dash.event(stage="done", ended=time.time())
    report()


def home_clearance(args):
    from jenga_vision.clearance import Clearance
    setup = json.loads(SETUP_FILE.read_text())
    return Clearance(args.arm, Tower(setup["tower"], setup.get("aim", 0.5)), arm_tool(setup["dino"], args.arm),
                     setup["dino"].get("calibration", {}).get(args.arm, {}).get("hand_q") or [0.0] * 20)


def go_home(args, name: str, target: list[float], tol_deg: float = 0.0) -> None:
    """run: the last move, a joint move from the scans' end pose (jig_ready_1118, away from the tower) to the
    teammate's home (TEAMMATE_HOME, or a saved pose: --home, run_home of scan_setup.json)."""
    if not args.send:
        q = saved_poses(args.arm)[args.start] if getattr(args, "start", None) in saved_poses(args.arm) else None
        note = "" if q is None else " ({} {:.0f} mm from the tower)".format(*home_clearance(args).along(q, target)[1::-1])
        print(f"\n(virtual) the arm would go to '{name}'{note}")
        return
    from jenga_vision.arm_slider import ArmNode
    node = ArmNode(args.arm, True, node_name="jenga_autoscan_home")
    try:
        t0 = time.time()
        while node.current() is None and time.time() - t0 < 10:
            time.sleep(0.05)
        if node.current() is None or not node.client.wait_for_server(timeout_sec=5):
            print(f"\nnot going to '{name}': no arm state or no trajectory action")
            return
        biggest = float(np.degrees(np.max(np.abs(np.array(target) - np.array(node.current())))))
        if biggest <= tol_deg:
            return
        mm, what, at = home_clearance(args).along(node.current(), target)
        if mm < MIN_CLEAR_MM:
            raise ScanError(f"the move to '{name}' would bring the {what} {mm:.0f} mm close to the tower (model) at "
                            f"{at:.0%} of the way: not sent")
        seconds = max(2.0, biggest / HOME_SPEED)
        print(f"\n===== {name} =====\n   joint move: largest joint {biggest:.0f} deg at {HOME_SPEED:.0f} deg/s, {seconds:.1f} s")
        node.move(target, seconds)
        try:
            node.done.wait(seconds + 10)
        except KeyboardInterrupt:
            node.stop()
            node.done.wait(3)
            print("Ctrl+C: stopped (arm holds where it is)")
            return
        if not node.succeeded:
            print(f"   move to '{name}' failed: {node.result}")
    finally:
        node.shutdown()


# Arms mounted mirror-image across world y = 0 (dual_robot.urdf): left joints -> right joints with these signs
# give the mirrored pose (checked on random poses, 2026-11-19: 0.03 mm)
MIRROR_SIGNS = (-1, 1, -1, 1, -1, 1, -1)


def home_pose_main(args) -> None:
    """The teammate's home for the right arm = the left arm now, mirrored (read only from /joint_states).
    Saved as arm_poses.json right.<name> and as run_home of scan_setup.json (run goes there at the end)."""
    from jenga_vision.arm_slider import ArmNode
    src, dst = ("left", "right") if args.arm == "right" else ("right", "left")
    node = ArmNode(src, False, node_name="jenga_home_pose")
    t0 = time.time()
    while node.current() is None and time.time() - t0 < 20:
        time.sleep(0.1)
    q = node.current()
    node.shutdown()
    if q is None:
        raise SystemExit(f"no /joint_states of the {src} arm (ROS sourced? ROS_DOMAIN_ID?)")
    mirrored = [round(float(v) * s, 5) for v, s in zip(q, MIRROR_SIGNS)]
    print(f"{src} arm now (deg):      " + " ".join(f"{math.degrees(v):7.1f}" for v in q))
    print(f"{dst} arm '{args.name}' (deg): " + " ".join(f"{math.degrees(v):7.1f}" for v in mirrored))
    if not args.write:
        print("not saved (add --write)")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for path in (POSES_FILE, SETUP_FILE):
        shutil.copy(path, path.with_name(f"{path.name}.bak_{stamp}_before_{args.name}"))
    poses = json.loads(POSES_FILE.read_text())
    poses.setdefault(dst, {})[args.name] = mirrored
    POSES_FILE.write_text(json.dumps(poses, indent=2))
    setup = json.loads(SETUP_FILE.read_text())
    setup["run_home"] = args.name
    SETUP_FILE.write_text(json.dumps(setup, indent=2, ensure_ascii=False) + "\n")
    print(f"saved '{args.name}' to {POSES_FILE} and run_home to {SETUP_FILE} (backups .bak_{stamp}_*): "
          "run goes there at the end (--no-home to stay)")


# --- main --------------------------------------------------------------------------
def on_route(arm: str, face: str, tool, q) -> list[float]:
    """The planned point `q` solved again from the fixed arm configuration of this face (ROUTE), so the arm takes
    the same configuration on every PC; ScanError if it can't be reached there (nothing has moved yet)."""
    refs = ROUTE.get((arm, face))
    if refs is None:
        return [float(v) for v in q]
    chain = Chain(arm)
    goal = chain.fk(np.array(q, float), tool)
    best = None
    for ref in refs:
        got = chain.ik(goal @ np.linalg.inv(tool), np.array(ref, float))
        if got is not None:
            off = float(np.degrees(np.max(np.abs(got - np.array(ref)))))
            if best is None or off < best[1]:
                best = (got, off)
    if best is None or best[1] > ROUTE_TOL_DEG:
        raise ScanError(f"{face}: no arm pose within {ROUTE_TOL_DEG:.0f} deg of the fixed route (ROUTE), nothing moved"
                        + ("" if best is None else f" (closest {best[1]:.0f} deg off)"))
    return [round(float(v), 6) for v in best[0]]


def planned_ends(args, tool, tower, face: str, layers: range, seed, plan_setup: dict) -> tuple[list, list]:
    """Joints of the first and last planned point of a face, in the fixed arm configuration (on_route). The
    plan takes 24 s on -y and only depends on these inputs, so it is kept in <out>/.plan_cache.json."""
    key = hashlib.sha256(json.dumps([args.arm, np.asarray(tool).tolist(), plan_setup, face, list(layers), list(seed),
                                     ROUTE_VERSION], sort_keys=True).encode()).hexdigest()
    cache_file = args.out / ".plan_cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    if key in cache:
        print("   plan from the cache")
        return cache[key]
    pts, _ = plan_face_stable(Chain(args.arm), tool, tower, face, layers, seed, plan_setup)
    for note in away_from_limits(pts, Chain(args.arm)):
        print(f"   {note}")
    cache[key] = [on_route(args.arm, face, tool, pts[0]["q"]), on_route(args.arm, face, tool, pts[-1]["q"])]
    args.out.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))
    return cache[key]


def save_rigid(path: Path, face: str, st: FaceState, turn_deg: float, roll_deg: float, session: str) -> None:
    """What --rigid starts the next scan of `face` with: per layer the depth and mean edge height of its shots."""
    data = json.loads(path.read_text()) if path.exists() else {}
    layers: dict[int, list[dict]] = {}
    for r in st.rows:
        layers.setdefault(r["layer"], []).append(r)
    # where each edge really was (the arm's height + its row in the image), and per slot its median offset from
    # its layer's mean: the face leans sideways, so a far block of a row would otherwise need centring (2026-11-18)
    scale = st.mm_per_px or DEFAULT_MM_PER_PX
    edge = {id(r): r["down_mm"] + (r["row"] - 240) * scale for r in st.rows}
    offs: dict[int, list[float]] = {}
    for rs in layers.values():
        if len(rs) >= 2:
            mean = float(np.mean([edge[id(r)] for r in rs]))
            for r in rs:
                offs.setdefault(r["slot"], []).append(edge[id(r)] - mean)
    slot_down = {s: round(float(np.median(v)), 2) for s, v in sorted(offs.items()) if len(v) >= 2}
    focus = round(float(np.median([r["focus"] for r in st.rows])), 1)
    was = data.get(face, {}).get("focus")
    if was and focus < 0.5 * was:  # a scan out of focus must not lower the bar of the next ones (sim 2026-11-18)
        print(f"   {face}: median focus {focus:.0f} under half the saved {was:.0f}: the saved value is kept")
        focus = was
    data[face] = {"session": session, "turn_deg": round(turn_deg, 2), "roll_deg": round(roll_deg, 2),
                  "mm_per_px": st.mm_per_px, "focus": focus,
                  "slot_down_mm": {str(k): v for k, v in slot_down.items()},
                  "layers": {str(k): [rs[0]["depth_mm"],
                                      round(float(np.mean([edge[id(r)] - slot_down.get(r["slot"], 0.0) for r in rs])), 2)]
                             for k, rs in sorted(layers.items())}}
    path.write_text(json.dumps(data, indent=2))


def scan_main(args) -> tuple[Path | None, bool]:
    """(session folder, finished without a stop)."""
    setup = json.loads(SETUP_FILE.read_text())
    face = args.face.replace("p", "+").replace("m", "-")
    if face not in FACE_NORMALS:
        raise SystemExit("--face: px, py, mx or my")
    tower = Tower(setup["tower"], setup.get("aim", 0.5))
    tool = getattr(args, "tool", None)  # calib: the tool predicted from the fingers
    tool = arm_tool(setup["dino"], args.arm) if tool is None else tool
    if not setup["dino"].get("tool_hand"):
        raise SystemExit("Dino not calibrated (run jenga-calib first)")
    layers = range(args.layers[0], args.layers[1] + 1)
    rows = face_shots(tower, face, layers, view_yaw(setup, face))
    if rows and view_yaw(setup, face):
        print(f"   the Dino looks at {face} turned {view_yaw(setup, face):+.0f} deg (view_yaw_deg in {SETUP_FILE})")
    if not rows:
        raise SystemExit(f"no end faces of layers {args.layers} on face {face}")
    poses = saved_poses(args.arm)
    seed = poses[args.seed]
    plan_setup = dict(setup, dino=dict(setup["dino"], standoff_m=0.0))
    print(f"planning {face} layers {args.layers[0]}..{args.layers[1]} (IK seed '{args.seed}') ...")
    if args.columns:  # one taught pose per column, no planned path
        far_in = far_out = None
        column_poses(face, args.arm)  # fails here, before anything moves, if one is missing
    else:
        first_q, last_q = planned_ends(args, tool, tower, face, layers, seed, plan_setup)
        far_in, far_out = last_q if args.top_first else first_q, last_q

    last_file = args.out / LAST_START_FILE
    if args.prior_from_last and last_file.exists() and not args.fresh_tower:
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
    from jenga_vision.clearance import Clearance
    mover.clearance = Clearance(args.arm, tower, tool, setup["dino"].get("calibration", {}).get(args.arm, {}).get(
        "hand_q") or [0.0] * 20)
    if mover.ros is not None and not args.fake:  # jenga-sim's hand is not the real one
        time.sleep(0.3)  # the hand joints come in the same /joint_states
        worst = hand_check(setup, args.arm, dict(mover.ros.positions))
        if worst is not None and worst > HAND_STOP_DEG and args.fix_hand:
            worst = fix_hand(mover, args.arm, setup)
        if worst is not None and worst > HAND_STOP_DEG and not args.ignore_hand:
            mover.shutdown()  # before the exit: a node destroyed while it spins aborts the process
            session.rmdir()
            raise SystemExit(f"STOPPED: ring / little finger {worst:.1f} deg off the calibration (limit "
                             f"{HAND_STOP_DEG:.0f}): the Dino is not where it was calibrated, nothing moved. Mount it "
                             "with `jenga_autoscan mount --send` (or close the fingers back: run --fix-hand)")
        if worst is not None and worst > HAND_WARN_DEG:
            print(f"   WARNING: ring / little finger {worst:.1f} deg off the calibration: focus searches may take longer")
    mover.far_speed, mover.parallel_speed, mover.out_speed = args.far_speed, args.parallel_speed, args.out_speed
    mover.line_accel = args.line_accel
    first_shot = next(iter(rows.values()))[0]
    mover.face = (first_shot.aim, first_shot.normal)
    fake = args.fake or not args.send
    holder = getattr(args, "camera_holder", None)  # run --keep-camera: one Dino reader for all faces
    if fake:
        true_tool = None
        if getattr(args, "fake_tool", None):  # mm x y z, deg about x y z, in the Dino frame
            from jenga_vision.kinematics import rpy_matrix, transform
            dx, dy, dz, rx, ry, rz = args.fake_tool
            true_tool = tool @ transform(rpy_matrix(*np.radians([rx, ry, rz])), np.array([dx, dy, dz]) / 1000)
        cam = FakeCamera(mover, rows, args.fake_depth_mm, args.fake_lean_mm, args.fake_drop_mm, args.window,
                         args.fake_roll_deg, true_tool=true_tool)
    elif holder is not None and holder.get("cam"):
        cam = holder["cam"]
    else:
        record = args.out / time.strftime("dino_%Y%m%d_%H%M%S.mp4") if args.record else None
        cam = RealCamera(args.device, args.window, record)
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
            carry.update(roll=scanner.roll, mm_per_px=st.mm_per_px, turn_deg=cam.turn_deg)
        if st.rows:  # where the first block of this face was found: the start of the next scan
            data = json.loads(last_file.read_text()) if last_file.exists() else {}
            first = st.rows[0]
            data[face] = {"depth_mm": first["depth_mm"], "down_mm": first["down_mm"], "layer": first["layer"],
                          "session": session.name}
            last_file.write_text(json.dumps(data, indent=2))
            save_rigid(args.out / RIGID_FILE, face, st, cam.turn_deg, math.degrees(scanner.roll), session.name)
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
    p.add_argument("--ignore-hand", action="store_true",
                   help=f"scan even with the ring / little finger more than {HAND_STOP_DEG:.0f} deg off the calibration")
    p.add_argument("--fix-hand", action="store_true",
                   help=f"ring / little finger more than {HAND_STOP_DEG:.0f} deg off: close all fingers back to the "
                   "calibration pose (the Dino must still be in the jig) and check again")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="of the dashboard")
    p.add_argument("--fast", action="store_true", help="shorter waits and fewer focus samples: " + str(FAST))
    p.add_argument("--preset", default=None, help="named settings (write --preset=NAME): " + ", ".join(PRESETS)
                   + f"; run uses {RUN_PRESET} without one, none = no preset")
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
    p.add_argument("--roll-test", type=int, default=None, metavar="LAYER",
                   help="instead of the scan: level on the middle block of this layer (teammate number, L08 = "
                        "floor 9), then turn the Dino by --roll-test-degs and measure each image as it is and "
                        "turned level in software (is --soft-level as good as levelling with the arm?)")
    p.add_argument("--repeat-test", type=int, default=None, metavar="LAYER",
                   help="full search on the middle block of LAYER and its row shot, appended to <out>/"
                        f"{REPEAT_FILE}: run again after re-seating the Dino, the table shows how far it moved")
    p.add_argument("--repeat-label", default="", help="name of this --repeat-test run (e.g. jig1, pick2)")
    p.add_argument("--defocus-test", type=int, default=None, metavar="LAYER",
                   help="focus on the middle block of LAYER (teammate layer, 8 = floor 9), then shoot each block of "
                        "the row at the found depth + --defocus-test-mm: gap px against depth error (defocus_test.csv)")
    p.add_argument("--defocus-test-mm", type=float, nargs="*", default=DEFOCUS_TEST_MM,
                   help="depth offsets (mm, + = deeper) of --defocus-test")
    p.add_argument("--roll-test-degs", type=float, nargs="*", default=ROLL_TEST_DEGS,
                   help="Dino turns (deg) about its axis from the levelled one (--roll-test)")
    p.add_argument("--soft-level", action="store_true",
                   help="DRAFT: level the edge by turning the image (once per face), not the arm")
    p.add_argument("--edge-ok-px", type=float, default=None,
                   help="DRAFT: a reference edge this close to the image centre is not centred (default: centre "
                        "to 25 px)")
    p.add_argument("--focus-predict", action="store_true",
                   help="DRAFT: focus peak from 2 samples + 1 to confirm, the usual search if that fails")
    p.add_argument("--keep-camera", action="store_true", help="run: one Dino reader for all faces")
    p.add_argument("--prior-from-last", action="store_true",
                   help="start the search at the depth and edge height of the last scan of this face (same place)")
    p.add_argument("--carry-dino", action="store_true",
                   help="run: start the next face with the Dino turn and image scale the face before found")
    p.add_argument("--narrow-ref2", action="store_true",
                   help="--sweep: the second reference searches near the first one's depth (wide only if that fails)")
    p.add_argument("--shot-settle-s", type=float, default=SETTLE_S,
                   help="wait after stopping at a block before its shot (--sweep, --blur-test)")
    p.add_argument("--columns", action="store_true",
                   help="one taught arm pose per column of end faces (column_poses in scan_setup.json), the "
                        "layers of a column by moving up / down only")
    p.add_argument("--glide", action="store_true", help="DRAFT: depth and height like --sweep, but every layer in "
                   "one pass along it without stopping, every frame measured")
    p.add_argument("--glide-speed", type=float, default=30.0, help="mm/s of the passes along a layer (--glide)")
    p.add_argument("--sweep-retreat-mm", type=float, default=10.0, help="out from the face between rows (--sweep)")
    p.add_argument("--rigid", action="store_true",
                   help="DRAFT: the Dino sits as in the last scan of the face (jig on the hand, or not picked again): "
                        f"its turn, scale, depths and heights from <out>/{RIGID_FILE}, one short focus search per face")
    p.add_argument("--rigid-trust", action="store_true",
                   help="DRAFT --rigid: no focus search on the first block either, only the check of every row")
    p.add_argument("--out-speed", type=float, default=None,
                   help="DRAFT: mm/s of straight moves away from the face (default: --line-speed)")
    p.add_argument("--record", action="store_true",
                   help="save the Dino's video of the scan to <out>/dino_<time>.mp4 (the last shot / search on it)")
    p.add_argument("--sweep-focus", action="store_true",
                   help="short focus searches as one move along the lens axis without stopping (every frame "
                        "measured) instead of stop-and-look steps")
    p.add_argument("--sweep-focus-speed", type=float, default=4.0, help="mm/s of --sweep-focus")
    p.add_argument("--fresh-tower", action="store_true",
                   help="--rigid without anything of the last tower (per-layer depth / height, per-block "
                        "height, the last start): only the Dino's turn, scale and usual focus value")
    p.add_argument("--refocus-below", type=float, default=RIGID_REFOCUS,
                   help="--rigid / --columns: a shot whose focus value is under this x the face's usual one gets a "
                        "short focus search (0.8 ~ 1.3 mm off; 0.6 ~ 2 mm, gaps then up to ~2 px narrower)")
    p.add_argument("--lazy-row", action=argparse.BooleanOptionalAction, default=False,
                   help="DRAFT: a side wider than the middle decides 빼안빼: the third block of that row is not shot")
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
    p.add_argument("--out", type=Path, default=CAPTURES_DIR)
    p.add_argument("--fake-depth-mm", type=float, default=-3.0, help="fake camera: true faces this much deeper")
    p.add_argument("--fake-lean-mm", type=float, default=1.5, help="fake camera: + this per 10 layers")
    p.add_argument("--fake-drop-mm", type=float, default=0.6, help="fake camera: gaps this much lower")
    p.add_argument("--fake-roll-deg", type=float, default=4.0, help="fake camera: Dino turned in the hand")
    p.add_argument("--fake-tool", type=float, nargs=6, default=None, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                   help="fake camera: the Dino really sits this far (mm) and turned (deg) in its own frame from the "
                   "assumed tool (x sideways, y image down, z along the axis)")


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
    r.add_argument("--out", type=Path, default=CAPTURES_DIR / "ranking.json")
    r.add_argument("--candidates", type=Path, help="teammate's candidates file, e.g. "
                   "$JENGA_STATE_DIR/jenga_pick_candidates.json (default: next to --out)")
    u = sub.add_parser("run", help=f"scan -y and +x with the Dino in the hand jig (--preset={RUN_PRESET}), "
                       "write the pull order")
    add_scan_options(u)
    u.set_defaults(out=CAPTURES_DIR / RUN_OUT, no_pick=True, no_place=True)
    u.set_defaults(search_out_mm=10.0, search_in_mm=25.0, refocus_mm=8.0, limit_mm=8.0, max_in_mm=14.0, line_speed=15.0)
    u.set_defaults(joint_speed=None)  # per face in RUN_FACES
    u.add_argument("--faces", nargs="+", choices=[s["face"] for _, s in RUN_FACES],
                   default=[s["face"] for _, s in RUN_FACES], help="my = -y, px = +x, in the order of RUN_FACES")
    u.add_argument("--pick", dest="no_pick", action="store_false", help="pick the Dino first (pick_dino_quick)")
    u.add_argument("--place", dest="no_place", action="store_false", help="place the Dino at the end (place_dino_quick)")
    u.add_argument("--no-pick", action="store_true", help=argparse.SUPPRESS)  # the default now; kept for old launches
    u.add_argument("--no-place", action="store_true", help=argparse.SUPPRESS)
    u.add_argument("--home", help="saved arm pose to go to at the end (default: run_home of scan_setup.json if set, "
                   "else TEAMMATE_HOME)")
    u.add_argument("--no-home", action="store_true", help="stay at the scans' end pose (jig_ready_1118)")
    u.add_argument("--yes", action="store_true",
                   help="no 'Enter = start' question with --send (also without a terminal, e.g. ros2 launch)")
    k = sub.add_parser("calib", help="after mount: the robot finds where the Dino sits in the hand today "
                       "(calibration block, focus / edge / level) and with --write saves it")
    add_scan_options(k)
    k.set_defaults(out=CAPTURES_DIR / RUN_OUT, soft_level=True, edge_ok_px=None, no_prompt=False,
                   search_out_mm=45.0, max_in_mm=40.0)  # wide: the Dino may sit anyhow (autocalib.py)
    k.add_argument("--write", action="store_true", help="save the new tool into scan_setup.json (backup first)")
    k.add_argument("--no-predict", action="store_true", help="start from the calibrated Dino as it is, not from the "
                   "finger prediction (the fingers look as at the calibration but read other angles)")
    k.add_argument("--hand", help="after mount --teach: the taught hand pose, from now on the mount pose (also the "
                   "finger angles of a dry run)")
    hp = sub.add_parser("home-pose", help="the teammate's home for this arm = the other arm now, mirrored "
                        "(read only); --write saves it and run goes there at the end")
    hp.add_argument("--arm", choices=("right", "left"), default="right", help="the arm that gets the home")
    hp.add_argument("--name", default="home_mirror", help="pose name in arm_poses.json")
    hp.add_argument("--write", action="store_true")
    m = sub.add_parser("mount", help="once per day: hand torque off, Dino into the jig, Enter, fingers close to the "
                       "calibration pose")
    m.add_argument("--arm", choices=("right", "left"), default="right")
    m.add_argument("--send", action="store_true", help="really switch the torque and move the fingers (default: dry run)")
    m.add_argument("--speed", type=float, default=20.0, help="deg/s of the largest finger joint")
    m.add_argument("--no-torque", action="store_true", help="only close the fingers (jenga-sim has no torque service)")
    m.add_argument("--teach", action="store_true", help="once per jig / Dino: close ring and little by hand with the "
                   "torque off, Enter saves the pose the hand then holds (calibrate the Dino with it after)")
    m.add_argument("--name", help="--teach: name of the saved hand pose (default jig_mount_<date>)")
    u.add_argument("--step", action="store_true", help="Enter before every step of pick and place")
    u.add_argument("--task-arm-speed", type=float, default=10.0, help="deg/s of pick and place")
    u.add_argument("--task-hand-speed", type=float, default=20.0, help="deg/s of the fingers in pick and place")
    o = sub.add_parser("order", help="pull order of one or more sessions: widest gap first + 빼안빼 rule")
    o.add_argument("sessions", nargs="+", type=Path)
    o.add_argument("--candidates", type=Path, help="also write the teammate's candidates file here")
    h = sub.add_parser("hand", help="ring / little finger now (/joint_states, read only) against the calibration")
    h.add_argument("--arm", choices=("right", "left"), default="right")
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
    names = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--preset=")]
    if not names and sys.argv[1:2] == ["run"]:
        u.set_defaults(**{**FAST, **QUICK, **PRESETS[RUN_PRESET]})
    for name in names:
        if name == "none":
            continue
        if name not in PRESETS:
            raise SystemExit(f"--preset: {', '.join(PRESETS)}, none")
        for o in (p, u):
            o.set_defaults(**{**FAST, **QUICK, **PRESETS[name]})
    args = parser.parse_args()
    if args.cmd == "mount":
        from jenga_vision.mount import mount_main
        return mount_main(args)
    if args.cmd == "home-pose":
        return home_pose_main(args)
    if args.cmd == "compare":
        compare_main(args.a, args.b)
        return
    if args.cmd == "dashboard":
        return replay_main(args)
    if args.cmd == "hand":
        from jenga_vision.arm_slider import ArmNode
        node = ArmNode(args.arm, False, node_name="jenga_hand_check")
        t0 = time.time()
        while not any(n.startswith(f"ah{args.arm[0]}_joint") for n in node.positions) and time.time() - t0 < 20:
            time.sleep(0.1)
        worst = hand_check(json.loads(SETUP_FILE.read_text()), args.arm, dict(node.positions))
        node.shutdown()
        if worst is not None:
            verdict = "OK" if worst <= HAND_WARN_DEG else ("check" if worst <= HAND_STOP_DEG else "scan refuses")
            print(f"   largest difference {worst:.1f} deg: {verdict} (warn > {HAND_WARN_DEG:.0f}, stop > {HAND_STOP_DEG:.0f})")
        return
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
    elif args.cmd == "calib":
        from jenga_vision.autocalib import calib_main
        calib_main(args, scan_main, json.loads(SETUP_FILE.read_text()).get("run_faces", {}))
    else:
        scan_main(args)
    if args.dashboard:
        try:
            input("the dashboard stays open: Enter or Ctrl+C closes it > ")
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == "__main__":
    main()
