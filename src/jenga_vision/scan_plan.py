"""Draft scan motions: Dino-Lite in the Allegro hand looks at every block end face.

Offline (no ROS): tower geometry from config/scan_setup.json (teammate's tower
pose and block sizes), arm kinematics from the teammate's URDF
(kinematics.py). For every end face on a tower face, the Dino tip is put
`standoff_m` in front of the face centre, optical axis into the tower, image
up = world up. The hand pose follows from the Dino pose in the hand frame
(`dino` in scan_setup.json: a GUESS until measured), joints from IK seeded
shot by shot, so the arm keeps one configuration along a face.

Tower frame (teammate's convention): layers L00 (bottom)..L16, slots S0..S2
(S1 middle). Even layers run along tower y -> their end faces are on the +-y
faces; odd layers run along x -> +-x faces. The right arm takes -y and +x,
the left arm -x and +y, so each arm sees one end of every block.

Writes routes auto_scan_<arm>_<face> to config/arm_routes.json (play them
with jenga-route, dry run first), a report, and a top/side view PNG.
Checks: joint limits, teammate's limits (joint4 >= 10 deg, |joint1| <= 120
deg), joint jumps between neighbouring shots. No collision check.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from jenga_vision.arm_route import ROUTES_FILE, load_all
from jenga_vision.arm_slider import POSES_FILE
from jenga_vision.kinematics import Chain, axis_angle, transform

SETUP_FILE = Path("config/scan_setup.json")
FACE_NORMALS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
ELBOW_MIN = math.radians(10)  # teammate's moveit_arm.py: joint4 >= 10 deg (xArm C22)
J1_MAX = math.radians(120)
MAX_STEP_DEG = 8.0  # largest joint change for one 5 mm / 2 deg tracking step
TRACK_M, TRACK_RAD = 0.005, math.radians(2)
N_SEEDS = 200  # random seeds tried for the first point of a face
FAR_M = 0.12  # transit point this far beyond approach_m, straight out from the face


class Tower:
    def __init__(self, t: dict, aim: float = 0.5):
        self.t = t
        self.aim = aim  # height of the aim point on an end face: 0.5 = centre, 1.0 = top edge (gap above)
        yaw = math.radians(t["yaw_deg"])
        self.r = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        self.o = np.array([t["x"], t["y"], t["z0"]])

    def world(self, p) -> np.ndarray:
        return self.o + self.r @ np.asarray(p, float)

    def end_faces(self, face: str, layers: range) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
        """(layer, slot, aim point world, outward normal world) of the end faces on `face`."""
        n2 = np.array(FACE_NORMALS[face], float)
        along_y = n2[1] != 0  # +-y faces show the blocks that run along tower y = even layers
        out = []
        for layer in layers:
            if (layer % 2 == 0) != along_y:
                continue
            for slot in range(3):
                off = (slot - 1) * self.t["slot_pitch"]
                xy = n2 * self.t["block_len"] / 2 + (np.array([off, 0]) if along_y else np.array([0, off]))
                z = (layer + self.aim) * self.t["layer_h"]
                out.append((layer, slot, self.world([xy[0], xy[1], z]), self.r @ np.array([n2[0], n2[1], 0])))
        return out


def arm_tool(d: dict, arm: str) -> np.ndarray:
    """Dino frame of `arm` in its hand frame: the calibrated `tool_hand` of that arm
    (origin = the point in focus), else the right arm's one mirrored (hand xz plane),
    else the guess from axis/tip/roll (origin = Dino tip)."""
    cal = d.get("tool_hand", {})
    if cal.get(arm) is not None:
        return np.array(cal[arm], float)
    other = "right" if arm == "left" else "left"
    if cal.get(other) is not None:
        m = np.diag([1.0, -1.0, 1.0, 1.0])
        return m @ np.array(cal[other], float) @ m
    t = dino_tool(d, arm)
    return np.diag([1.0, -1.0, 1.0, 1.0]) @ t @ np.diag([1.0, -1.0, 1.0, 1.0]) if arm == "left" else t


def dino_tool(d: dict, arm: str = "right") -> np.ndarray:
    """Guess (before calibration): 4x4 Dino frame in the hand frame: z = optical axis
    (out of the tip), y = image down.

    roll_deg (a number or per arm) turns the hand around the Dino axis; the
    real image then turns too, which the scan undoes in software."""
    z = np.array(d["axis_hand"], float)
    z /= np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0])  # palm normal: image "up" at roll 0 points away from the palm
    y = -(ref - z * (ref @ z))
    y /= np.linalg.norm(y)
    r = np.column_stack([np.cross(y, z), y, z]) @ axis_angle(np.array([0, 0, 1.0]), math.radians(d["roll_deg"][arm] if isinstance(d["roll_deg"], dict) else d["roll_deg"]))
    return transform(r, d["tip_hand"])


def shot_pose(centre: np.ndarray, normal: np.ndarray, standoff: float) -> np.ndarray:
    """Dino frame in world: tip in front of the face, looking in, image up = world up."""
    z = -normal
    y = np.array([0.0, 0.0, -1.0])
    return transform(np.column_stack([np.cross(y, z), y, z]), centre + normal * standoff)


def check_q(q: np.ndarray) -> list[str]:
    bad = []
    if q[3] < ELBOW_MIN:
        bad.append(f"j4 {math.degrees(q[3]):.0f}<10")
    if abs(q[0]) > J1_MAX:
        bad.append(f"|j1| {math.degrees(q[0]):.0f}>120")
    return bad


def first_solution(chain: Chain, goal: np.ndarray, seed, rng: np.random.Generator) -> np.ndarray | None:
    """IK from many seeds; the valid solution closest to `seed` (keeps the arm near its start pose)."""
    seed = np.array(seed, float)
    best, best_d = None, math.inf
    for k in range(N_SEEDS):
        s = seed if k == 0 else np.clip(seed + rng.normal(0, 0.6 if k < N_SEEDS // 2 else 1.5, 7), chain.lo, chain.hi)
        q = chain.ik(goal, s)
        if q is None or check_q(q):
            continue
        d = float(np.max(np.abs(q - seed)))
        if d < best_d:
            best, best_d = q, d
    return best


def track(chain: Chain, tool: np.ndarray, start_dino: np.ndarray, goal_dino: np.ndarray, q0: np.ndarray):
    """Follow the straight line start -> goal (Dino frame) with small IK steps; (q, largest step deg) or None."""
    dp = goal_dino[:3, 3] - start_dino[:3, 3]
    rel = start_dino[:3, :3].T @ goal_dino[:3, :3]
    ang = math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1) / 2)))
    n = max(1, math.ceil(max(np.linalg.norm(dp) / TRACK_M, ang / TRACK_RAD)))
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    q, worst = np.array(q0, float), 0.0
    inv_tool = np.linalg.inv(tool)
    for i in range(1, n + 1):
        r = start_dino[:3, :3] @ (axis_angle(axis, ang * i / n) if ang > 1e-9 else np.eye(3))
        step = transform(r, start_dino[:3, 3] + dp * i / n)
        q_new = chain.ik(step @ inv_tool, q)
        if q_new is None:
            return None
        worst = max(worst, float(np.degrees(np.max(np.abs(q_new - q)))))
        q = q_new
    return q, worst


def plan_face(chain: Chain, tool: np.ndarray, tower: Tower, face: str, layers: range, seed, setup: dict):
    """Route points [{label, via, q}] and a report for one face, or raises with the reason."""
    faces = tower.end_faces(face, layers)
    standoff = setup["dino"]["standoff_m"]
    # Screen order left -> right: image x (right) of the shot pose.
    right_dir = shot_pose(faces[0][2], faces[0][3], standoff)[:3, 0]
    points, report = [], []
    q_prev, goal_prev = np.array(seed, float), None
    rng = np.random.default_rng(0)

    def solve(goal_dino: np.ndarray, label: str, via: bool) -> np.ndarray:
        nonlocal q_prev, goal_prev
        if goal_prev is None:
            q = first_solution(chain, goal_dino @ np.linalg.inv(tool), q_prev, rng)
            if q is None:
                raise ValueError(f"{label}: no IK solution within the joint limits (out of reach?)")
            worst = 0.0
        else:
            res = track(chain, tool, goal_prev, goal_dino, q_prev)
            if res is None or res[1] > MAX_STEP_DEG or check_q(res[0]):
                # Straight-line tracking failed: solve the point alone, near the previous joints.
                q = first_solution(chain, goal_dino @ np.linalg.inv(tool), q_prev, rng)
                if q is None:
                    raise ValueError(f"{label}: IK lost on the way from the previous point")
                label += " (re-solved)"
            else:
                q = res[0]
        bad = check_q(q)
        jump = float(np.degrees(np.max(np.abs(q - q_prev))))
        points.append({"label": label, "via": via, "q": [round(float(v), 5) for v in q]})
        report.append((label, via, jump, bad, goal_dino[:3, 3]))
        q_prev, goal_prev = q, goal_dino
        return q

    by_layer: dict[int, list] = {}
    for f in faces:
        by_layer.setdefault(f[0], []).append(f)
    # A far point straight out from the first shot: the arm comes from the seed pose to
    # it with a joint move (well away from the tower), then goes in on a straight line.
    lay0 = min(by_layer)
    row0 = sorted(by_layer[lay0], key=lambda f: f[2] @ right_dir)
    solve(shot_pose(row0[0][2] + row0[0][3] * (setup["approach_m"] + FAR_M), row0[0][3], standoff),
          f"{face} far in", True)
    first = True
    for layer in sorted(by_layer):
        row = sorted(by_layer[layer], key=lambda f: f[2] @ right_dir)
        names = ("left", "middle", "right")
        for k, (lay, slot, centre, normal) in enumerate(row):
            if k == 0:  # come in from outside the tower at this layer's height
                out = setup["approach_m"] if first else setup["layer_retreat_m"]
                solve(shot_pose(centre + normal * out, normal, standoff), f"{face} L{lay:02d} in", True)
                first = False
            solve(shot_pose(centre, normal, standoff), f"{face} L{lay:02d}_S{slot} {names[k]}", False)
        lay, slot, centre, normal = row[-1]
        out = setup["layer_retreat_m"] if layer != max(by_layer) else setup["approach_m"]
        solve(shot_pose(centre + normal * out, normal, standoff), f"{face} L{lay:02d} out", True)
    solve(shot_pose(centre + normal * (setup["approach_m"] + FAR_M), normal, standoff), f"{face} far out", True)
    return points, report


LIMIT_MARGIN = math.radians(20)


def away_from_limits(points: list[dict], chain: Chain) -> list[str]:
    """Joints with a +-360 deg range end up at the limit when IK clips them: shift
    such a joint by 360 deg on the whole face (same hand pose) when that moves it
    off the limit. Returns notes."""
    notes = []
    for i in range(7):
        if chain.hi[i] - chain.lo[i] < 2 * math.tau - 1e-3:
            continue
        vals = np.array([p["q"][i] for p in points])
        for shift in (-math.tau, math.tau):
            near = vals.max() > chain.hi[i] - LIMIT_MARGIN or vals.min() < chain.lo[i] + LIMIT_MARGIN
            moved = vals + shift
            if near and moved.max() <= chain.hi[i] - LIMIT_MARGIN and moved.min() >= chain.lo[i] + LIMIT_MARGIN:
                for p in points:
                    p["q"][i] = round(float(p["q"][i] + shift), 5)
                notes.append(f"joint{i + 1} shifted by {math.degrees(shift):+.0f} deg (was at its limit)")
                break
    return notes


def plan_face_stable(chain: Chain, tool: np.ndarray, tower: Tower, face: str, layers: range, seed, setup: dict):
    """plan_face; if a point had to be re-solved (the arm changed configuration on the
    way), plan again starting near that configuration so the whole face uses it."""
    face_seed = seed
    for _ in range(3):
        pts, rep = plan_face(chain, tool, tower, face, layers, face_seed, setup)
        resolved = [pt for pt in pts if "re-solved" in pt["label"]]
        if not resolved:
            return pts, rep
        face_seed = resolved[0]["q"]
    raise ValueError(f"no single arm configuration for the whole face ({len(resolved)} re-solved points)")


def interp_pose(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """Straight line between two 4x4 poses, rotation about one axis."""
    rel = a[:3, :3].T @ b[:3, :3]
    ang = math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1) / 2)))
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    r = a[:3, :3] @ (axis_angle(axis, ang * s) if ang > 1e-9 else np.eye(3))
    return transform(r, a[:3, 3] + (b[:3, 3] - a[:3, 3]) * s)


def transit_clearance(chain: Chain, tool: np.ndarray, tower: Tower, qa, qb) -> float:
    """Smallest horizontal distance (m) of the Dino tip, hand base and wrist to the tower box
    (above the stand) along the joint-interpolated move qa -> qb. Negative = inside."""
    qa, qb = np.array(qa, float), np.array(qb, float)
    half = tower.t["block_len"] / 2
    top = tower.o[2] + tower.t["layers"] * tower.t["layer_h"]
    worst = math.inf
    for s in np.linspace(0, 1, 60):
        q = qa + (qb - qa) * s
        hand = chain.fk(q)
        for p in (hand[:3, 3], (hand @ tool)[:3, 3], chain.fk(q, np.linalg.inv(chain.links[-1].origin) @ np.linalg.inv(chain.links[-2].origin))[:3, 3]):
            if p[2] > top + 0.03:
                continue  # above the tower
            local = tower.r.T @ (p - tower.o)
            d = np.max(np.abs(local[:2])) - half  # square tower, outside > 0
            worst = min(worst, d)
    return worst


def draw(tower: Tower, plans: dict, path: Path) -> None:
    """Top view (left) and side view along each face (right) of the planned Dino tips."""
    img = np.full((520, 1040, 3), 255, np.uint8)
    s, ox, oy = 420, 270, 470  # top view: px per m, world origin

    def top(p):
        return int(ox + p[1] * -s), int(oy - p[0] * s)  # world +x up, +y left

    corners = [tower.world([sx * 0.0575, sy * 0.0575, 0]) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    cv2.polylines(img, [np.array([top(c) for c in corners])], True, (80, 80, 80), 2)
    for name, base in (("R", (0, -0.5)), ("L", (0, 0.5))):
        cv2.circle(img, top(base), 8, (0, 0, 0), -1)
        cv2.putText(img, f"{name} arm", (top(base)[0] - 20, top(base)[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    colors = {"-y": (0, 0, 220), "+x": (0, 140, 255), "-x": (200, 0, 0), "+y": (200, 120, 0)}
    for (arm, face), (pts, rep) in plans.items():
        for label, via, _, bad, p in rep:
            cv2.circle(img, top(p), 2 if via else 3, (150, 150, 150) if via else colors[face], -1)
        lab = rep[len(rep) // 2][4]
        cv2.putText(img, f"{arm} {face}", top(lab + (lab - tower.o) * 0.6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[face], 1)
    cv2.putText(img, "top view (world +x up, +y left), dots = Dino tip", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    # side view: height of every shot per face
    x0 = 560
    cv2.putText(img, "shots per face (height = layer)", (x0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    for k, ((arm, face), (pts, rep)) in enumerate(plans.items()):
        cx = x0 + 30 + k * 115
        cv2.putText(img, f"{arm} {face}", (cx - 10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[face], 1)
        for label, via, _, bad, p in rep:
            if via:
                continue
            j = next(k for k, w in enumerate(("left", "middle", "right")) if w in label)
            y = int(480 - (p[2] - tower.o[2]) * 1000)
            cv2.rectangle(img, (cx + j * 22, y - 4), (cx + j * 22 + 18, y + 4),
                          (0, 0, 255) if bad else colors[face], -1 if not bad else 2)
    cv2.imwrite(str(path), img)


MIRROR = np.array([-1, 1, -1, 1, -1, 1, -1])  # right arm joints -> mirror image left arm (world xz plane)
PICK_POSES = ("home", "before_grasp", "final_grasp_1", "final_grasp_1_1", "final_grasp_1_2")


def mirrored_left_poses(right_poses: dict, jig_y: float) -> dict[str, list[float]]:
    """Left arm pick poses: the right ones mirrored at world y = 0, then shifted by 2 * jig_y
    so the left hand reaches the same jig (IK keeps the orientation)."""
    chain = Chain("left")
    out, q_prev = {}, None
    for name in PICK_POSES:
        q = np.array(right_poses[name]) * MIRROR
        if abs(jig_y) > 1e-6:
            goal = chain.fk(q)
            goal[1, 3] += 2 * jig_y
            q = chain.ik(goal, q if q_prev is None else q)
            if q is None:
                raise ValueError(f"left {name}: IK failed after the jig shift")
        out[f"auto_{name}"] = [round(float(v), 5) for v in q]
        q_prev = q
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="save the routes to config/arm_routes.json")
    parser.add_argument("--png", type=Path, default=Path("captures/scan_plan.png"))
    parser.add_argument("--standoff", type=float, default=None,
                        help="extra distance (m) out from the face, overrides scan_setup.json (draft: 0.04)")
    parser.add_argument("--only", nargs=2, metavar=("ARM", "FACE"), help="plan one face only, e.g. right my (my = -y, px = +x)")
    parser.add_argument("--layers", nargs=2, type=int, metavar=("FIRST", "LAST"), help="override scan_layers")
    parser.add_argument("--seed", help="saved arm pose to start IK from (default: scan_setup.json seed, home)")
    parser.add_argument("--name", help="route name for --only (default auto_scan_<arm>_<face>)")
    args = parser.parse_args()

    setup = json.loads(SETUP_FILE.read_text())
    if args.only:
        face = args.only[1].replace("p", "+").replace("m", "-")
        if face not in FACE_NORMALS:
            raise SystemExit(f"face {args.only[1]}: use px, mx, py or my")
        setup["faces"] = {args.only[0]: [face]}
    if args.layers:
        setup["scan_layers"] = list(args.layers)
    if args.standoff is not None:
        setup["dino"]["standoff_m"] = args.standoff
    print(f"standoff {setup['dino']['standoff_m'] * 1000:.0f} mm "
          + ("(from the point in focus: 0 = in focus)" if setup["dino"].get("tool_hand") else "(Dino tip to end face)"))
    tower = Tower(setup["tower"], setup.get("aim", 0.5))
    layers = range(setup["scan_layers"][0], setup["scan_layers"][1] + 1)
    arm_poses = json.loads(POSES_FILE.read_text()) if POSES_FILE.exists() else {}
    left_pick = mirrored_left_poses(arm_poses["right"], setup.get("jig_y", 0.0))
    arm_poses.setdefault("left", {}).update({k: v for k, v in left_pick.items() if k not in arm_poses["left"]})
    plans, routes = {}, {}
    for arm, faces in setup["faces"].items():
        chain = Chain(arm)
        seed = args.seed or setup["seed"][arm]
        seed = arm_poses[arm][seed] if isinstance(seed, str) else seed
        tool_arm = arm_tool(setup["dino"], arm)
        for face in faces:
            print(f"\n== {arm} arm, tower face {face}")
            try:
                pts, rep = plan_face_stable(chain, tool_arm, tower, face, layers, seed, setup)
            except ValueError as e:
                print(f"   NOT REACHABLE with this setup: {e}")
                continue
            for note in away_from_limits(pts, chain):
                print(f"   {note}")
            plans[(arm, face)] = (pts, rep)
            routes[(arm, args.name or f"auto_scan_{arm}_{face.replace('+', 'p').replace('-', 'm')}")] = pts
            shots = [r for r in rep if not r[1]]
            bad = [r for r in rep if r[3]]
            print(f"   {len(shots)} shots, {len(rep) - len(shots)} via points; "
                  f"largest jump between points {max(r[2] for r in rep[1:]):.1f} deg")
            for label, via, jump, b, p in rep:
                if b or label.endswith(("S0 left", "S1 middle", "S2 right")) and label.split()[1] in ("L01", "L02", "L15"):
                    print(f"   {label:24s} tip {np.round(p * 1000).astype(int)} mm  {' '.join(b)}")
            if bad:
                print(f"   ! {len(bad)} points break the teammate's limits (j4>=10, |j1|<=120)")
            print(f"   first point is {np.degrees(np.max(np.abs(np.array(pts[0]['q']) - np.array(seed)))):.0f} deg "
                  f"(largest joint) from the seed pose")
            for name, qa, qb in (("seed -> first", seed, pts[0]["q"]), ("last -> seed", pts[-1]["q"], seed)):
                c = transit_clearance(chain, tool_arm, tower, qa, qb)
                warn = "  ! TOO CLOSE: add a via point" if c < 0.05 else ""
                print(f"   {name}: wrist/hand/Dino stay {c * 1000:.0f} mm outside the tower{warn}")
    args.png.parent.mkdir(parents=True, exist_ok=True)
    draw(tower, plans, args.png)
    print(f"\nview: {args.png}")
    if args.write:
        poses = json.loads(POSES_FILE.read_text())
        new = {} if args.only else {k: v for k, v in left_pick.items() if k not in poses.get("left", {})}
        poses.setdefault("left", {}).update(new)
        POSES_FILE.write_text(json.dumps(poses, indent=2))
        print(f"left pick poses (mirror of right, jig_y {setup.get('jig_y', 0.0)}): "
              f"{', '.join(new) or 'already saved, kept'}")
        data = load_all()
        for (arm, name), pts in routes.items():
            data.setdefault(arm, {})[name] = pts
        ROUTES_FILE.write_text(json.dumps(data, indent=2))
        print(f"wrote {', '.join(n for _, n in routes)} to {ROUTES_FILE}")
    else:
        print("(dry: pass --write to save the routes)")


if __name__ == "__main__":
    main()
