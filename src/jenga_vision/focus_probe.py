"""Measure the real focus curve of the Dino, moving only away from the block.

Bring the Dino to the block by hand first (jenga-arm, frame dino). This tool starts
from wherever the arm is and never goes deeper than that pose:

  hold   stay at the pose: first without moving (baseline), then after a small move
         out and back with different waits. Shows whether the focus value jumps because
         of arm vibration. Ends at the start pose.
  curve  step straight out along the optical axis (-Z of the Dino), measuring at every
         step. Start a little past the sharpest point to get both sides of the peak.
         Ends at the far point.
  sweep  does a gap measured while moving match the stopped one? Starts at home with the
         Dino in the hand, goes to a layer at the depth, height and turn found by a
         jenga-autoscan session (same pick!), runs along its end faces at each --speeds,
         prints the gaps per block next to the session's, and goes back home.

Results go to captures/focus_<time>/: samples.csv (every frame's focus value) and for
curve one image per step, so that other sharpness measures can be tried offline.

Robot / camera (as jenga-autoscan):
  without --send  no ROS: virtual arm from the saved pose --start, fake camera
  --send          real moves (on domain 77 with jenga-sim: virtual robot)
  --fake          fake camera also with --send

Ctrl+C stops the arm where it is; q in the camera window stops after the current step.
Close jenga-view first (one process per camera).

    uv run jenga-focus curve                # virtual run
    uv run jenga-focus hold --send
    uv run jenga-focus curve --send --mm 30
    uv run jenga-focus sweep --session captures/autoscan_X --layer 13 --send
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

from jenga_vision.arm_slider import POSES_FILE
from jenga_vision import ends
from jenga_vision.autoscan import (DISCARD_FRAMES, FakeCamera, Mover, RealCamera, ScanError, edge_row, face_shots)
from jenga_vision.camera import focus_value
from jenga_vision.kinematics import Chain, transform
from jenga_vision.scan_plan import SETUP_FILE, Tower, arm_tool, plan_face_stable

FRAMES = 5  # frames per measurement: tells one bad frame from a whole bad measurement
MIN_OUT_COS = 0.3  # the move out must point at least this much away from the tower axis
MAX_SWEEP_MM = 60.0
RUN_IN_MM = 29.0  # sweep from this far before the first block's center (19 half block + 10 to reach speed)
BLOCK_PITCH_MM = 38.5
BLOCK_WINDOW_MM = 14.0  # frames this close to a block's center count for it


class FakeCurveCamera(RealCamera):
    """A blurred texture, sharpest `peak_mm` out from the pose the Dino starts at."""

    def __init__(self, mover: Mover, peak_mm: float, window: bool):
        self.mover, self.window = mover, window
        self.start = mover.dino()
        self.peak_mm = peak_mm
        self.rng = np.random.default_rng(0)
        noise = self.rng.normal(0, 1, (480, 640)).astype(np.float32)
        self.texture = cv2.normalize(cv2.GaussianBlur(noise, (0, 0), 1.5), None, 60, 200, cv2.NORM_MINMAX)
        self.cam = self

    def read(self) -> np.ndarray:
        out_mm = float((self.start[:3, 3] - self.mover.dino()[:3, 3]) @ self.start[:3, 2]) * 1000
        gray = cv2.GaussianBlur(self.texture, (0, 0), 0.3 + 1.2 * abs(out_mm - self.peak_mm))
        return cv2.cvtColor(np.clip(gray + self.rng.normal(0, 2, gray.shape), 0, 255).astype(np.uint8),
                            cv2.COLOR_GRAY2BGR)

    def release(self) -> None:
        pass


def check_outward(dino: np.ndarray, tower: dict) -> float:
    """Independent of the tool's sign: moving along -Z must lead away from the tower axis."""
    out = -dino[:2, 2]
    radial = dino[:2, 3] - np.array([tower["x"], tower["y"]])
    cos = float(out @ radial / max(np.linalg.norm(out) * np.linalg.norm(radial), 1e-9))
    if cos < MIN_OUT_COS:
        raise ScanError(f"-Z of the Dino does not lead away from the tower (cos {cos:+.2f}): "
                        "is the Dino facing the tower? nothing moved")
    return cos


class Probe:
    def __init__(self, args, mover: Mover, cam, session: Path):
        self.args, self.mover, self.cam, self.session = args, mover, cam, session
        self.start = mover.dino()
        self.file = (session / "samples.csv").open("w", newline="")
        self.w = csv.writer(self.file)
        self.w.writerow(["time", "kind", "index", "out_mm", "wait_s", "median", *[f"f{i}" for i in range(FRAMES)],
                         "edge_row", "brightness"])

    def go_out(self, mm: float) -> None:
        self.mover.line_to(self.start @ transform(np.eye(3), [0, 0, -mm / 1000]))

    def sample(self, kind: str, index: int, out_mm: float, wait_s: float, save: bool = False) -> float:
        time.sleep(wait_s)
        for _ in range(DISCARD_FRAMES):
            self.cam.cam.read()
        frames = [self.cam.cam.read() for _ in range(FRAMES)]
        values = [focus_value(f) for f in frames]
        med = float(np.median(values))
        row = edge_row(frames[-1])
        self.w.writerow([f"{time.time():.2f}", kind, index, f"{out_mm:.2f}", wait_s, f"{med:.2f}",
                         *[f"{v:.2f}" for v in values], "" if row is None else f"{row:.0f}",
                         f"{cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY).mean():.1f}"])
        self.file.flush()
        if save:
            cv2.imwrite(str(self.session / f"{kind}_{index:03d}_out{out_mm:05.1f}.png"), frames[-1])
        self.cam.show(frames[-1], f"{kind} {index} out {out_mm:.1f} mm wait {wait_s:.1f} s focus {med:.1f}")
        if self.cam.key() in (ord("q"), 27):
            raise ScanError("stopped by the user (q)")
        return med

    def curve(self) -> None:
        a = self.args
        n = round(a.mm / a.step)
        out, val = [], []
        for i in range(n + 1):
            d = i * a.step
            self.go_out(d)
            val.append(self.sample("curve", i, d, a.wait, save=True))
            out.append(d)
            print(f"   out {d:5.1f} mm  focus {val[-1]:7.2f}")
        k = int(np.argmax(val))
        print(f"\nsharpest {val[k]:.1f} at {out[k]:.1f} mm out; floor (lowest) {min(val):.1f}; arm is now {a.mm:.0f} mm out")
        if k == 0:
            print("the start was the sharpest: the inner side of the peak is missing. "
                  "Start a little further in, if there is room.")
        if k == n:
            print("still rising at the end: the peak is further out. Run again with a larger --mm.")

    def hold(self) -> None:
        a = self.args
        print(f"baseline: {a.reps} measurements without moving")
        base = [self.sample("still", i, 0.0, 1.0) for i in range(a.reps)]
        print(f"   median {np.median(base):.1f}, min {min(base):.1f}, max {max(base):.1f}")
        for wait in a.waits:
            print(f"after a {a.jiggle:.1f} mm move out and back, wait {wait:.1f} s:")
            v = []
            for i in range(a.reps):
                self.go_out(a.jiggle)
                self.mover.line_to(self.start)
                v.append(self.sample("settle", i, 0.0, wait))
            print(f"   median {np.median(v):.1f}, min {min(v):.1f}, max {max(v):.1f}")
        print("arm is back at the start pose")

    def close(self) -> None:
        self.file.close()


def sweep_main(args, setup: dict, tool: np.ndarray, poses: dict) -> None:
    rows = list(csv.DictReader((args.session / "shots.csv").open()))
    rows = [r for r in rows if int(r["layer"]) == args.layer]
    if len(rows) != 3:
        raise SystemExit(f"{args.session} has {len(rows)} shots of layer {args.layer}, 3 needed")
    face = rows[0]["face"]
    depth = float(np.median([float(r["depth_mm"]) for r in rows])) / 1000
    down = float(np.median([float(r["down_mm"]) for r in rows])) / 1000
    roll = math.radians(float(rows[0]["roll_deg"]))
    tower = Tower(setup["tower"], setup.get("aim", 0.5))
    row = face_shots(tower, face, range(args.layer, args.layer + 1))[args.layer]
    far = plan_face_stable(Chain(args.arm), tool, tower, face, range(args.layer, args.layer + 1), poses[args.seed],
                           dict(setup, dino=dict(setup["dino"], standoff_m=0.0)))[0][0]["q"]
    left, right = row[0].pose(depth, down, roll), row[-1].pose(depth, down, roll)
    along = (right[:3, 3] - left[:3, 3]) / np.linalg.norm(right[:3, 3] - left[:3, 3])
    a, b = left.copy(), right.copy()
    a[:3, 3] -= along * RUN_IN_MM / 1000
    b[:3, 3] += along * RUN_IN_MM / 1000
    out_of_face = transform(np.eye(3), [0, 0, -0.03])

    mover = Mover(args.arm, tool, args.send, poses[args.start], args.joint_speed, args.line_speed)
    fake = args.fake or not args.send
    cam = FakeCamera(mover, {args.layer: row}, depth * 1000 - args.layer * 0.15, 1.5, down * 1000, args.send,
                     math.degrees(-roll)) if fake else RealCamera(args.device, True)
    session = args.out / time.strftime("sweep_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    print(f"sweep {face} L{args.layer:02d}: depth {depth * 1000:+.1f} mm, down {down * 1000:+.1f} mm, "
          f"roll {math.degrees(roll):+.1f} deg, speeds {args.speeds} mm/s; "
          f"{'SEND' if args.send else 'VIRTUAL'} {'(fake camera)' if fake else '(Dino)'}, "
          f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}, session {session}")
    print(f"the arm will move: {' -> '.join(args.via)} -> far point -> 30 mm before the layer -> sweeps -> back; "
          "Dino must be in the same pick as the session")
    grabbed: list[tuple[int, float, float, np.ndarray]] = []
    speed_now = [0]

    def grab() -> None:
        """Every frame while sweeping, with where the arm really is."""
        frame = cam.cam.read() if not fake else cam.frames(1)[0]
        q = np.array(mover.ros.current() if mover.ros is not None else mover.q, float)
        if fake:
            mover.q = q  # the fake camera looks from here
        p = mover.chain.fk(q, tool)[:3, 3]
        grabbed.append((speed_now[0], time.time(), float((p - left[:3, 3]) @ along) * 1000, frame))

    try:
        if args.send:
            cam.show(cam.cam.read(), "waiting for Enter in the terminal") if not fake else None
            if input("Dino in the hand (same pick), path clear? Enter = start, q = quit > ").strip().lower() == "q":
                return
        for via in args.via:
            mover.joint_move(poses[via], via)
        mover.joint_move(far, "far point")
        mover.line_speed = args.line_speed
        mover.line_to(a @ out_of_face)
        mover.line_to(a)
        start, end = a, b
        for v in args.speeds:
            mover.line_speed = v
            speed_now[0] = v
            mover.idle = grab
            t0 = time.time()
            mover.line_to(end)
            mover.idle = None
            print(f"   {v:.0f} mm/s: {time.time() - t0:.1f} s")
            start, end = end, start
        mover.line_speed = args.line_speed
        mover.line_to(start @ out_of_face)
        mover.joint_move(far, "far point")
        for via in reversed(args.via):
            mover.joint_move(poses[via], via)
        mover.joint_move(poses[args.end], args.end)
    except KeyboardInterrupt:
        print("\nCtrl+C: stopped (arm holds where it is)")
    except ScanError as e:
        print(f"\nSTOPPED: {e}")
    finally:
        cam.release()
        mover.shutdown()
        cv2.destroyAllWindows()
    report_sweep(grabbed, {r["screen"]: float(r["gap"]) for r in rows}, session)
    print(f"session {session}")


def report_sweep(grabbed: list, stopped: dict[str, float], session: Path) -> None:
    with (session / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speed", "time", "x_mm", "block", "gap", "edge_row", "focus", "image"])
        results: dict[tuple[float, str], list] = {}
        for i, (v, t, x, frame) in enumerate(grabbed):
            k = round(x / BLOCK_PITCH_MM)
            block = ends.BLOCKS[k] if 0 <= k <= 2 and abs(x - k * BLOCK_PITCH_MM) <= BLOCK_WINDOW_MM else ""
            g, row, foc = ends.measure_end(frame).gap, edge_row(frame), focus_value(frame)
            name = f"{v:03.0f}_{i:04d}.png"
            cv2.imwrite(str(session / name), frame)
            w.writerow([v, f"{t:.3f}", f"{x:.1f}", block, g, "" if row is None else f"{row:.1f}", f"{foc:.1f}", name])
            if block:
                results.setdefault((v, block), []).append((g, row, foc))
    print(f"\n{'speed':>6} {'block':>7} {'stopped':>8} {'frames':>7} {'gap median':>11} {'gap p25-p75':>12} "
          f"{'edge seen':>10} {'edge row std':>13} {'focus':>6}")
    for (v, block), r in results.items():
        g = [x[0] for x in r]
        rows = [x[1] for x in r if x[1] is not None]
        print(f"{v:6.0f} {block:>7} {stopped.get(block, float('nan')):8.1f} {len(r):7d} {np.median(g):11.1f} "
              f"{np.percentile(g, 25):5.1f}-{np.percentile(g, 75):<6.1f} {len(rows) / len(r):10.0%} "
              f"{np.std(rows) if rows else float('nan'):13.1f} {np.median([x[2] for x in r]):6.1f}")
    for v in sorted({x[0] for x in grabbed}):
        ts = [x[1] for x in grabbed if x[0] == v]
        if len(ts) > 1:
            print(f"   {v:.0f} mm/s: {len(ts) / (ts[-1] - ts[0]):.0f} frames/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("curve", "hold", "sweep"))
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument("--send", action="store_true", help="really move (default: virtual arm, fake camera)")
    parser.add_argument("--fake", action="store_true", help="fake camera also with --send")
    parser.add_argument("--start", default="cali_way_3", help="saved pose of the virtual arm without --send")
    parser.add_argument("--mm", type=float, default=30.0, help="curve: how far out")
    parser.add_argument("--step", type=float, default=0.5, help="curve: step in mm")
    parser.add_argument("--wait", type=float, default=0.3, help="curve: wait after a step, s (0.3 = jenga-autoscan)")
    parser.add_argument("--reps", type=int, default=15, help="hold: measurements per wait")
    parser.add_argument("--waits", type=float, nargs="+", default=[0.1, 0.3, 1.0], help="hold: waits to compare, s")
    parser.add_argument("--jiggle", type=float, default=0.5, help="hold: size of the move out and back, mm")
    parser.add_argument("--line-speed", type=float, default=5.0, help="mm/s")
    parser.add_argument("--joint-speed", type=float, default=10.0, help="deg/s")
    parser.add_argument("--session", type=Path, help="sweep: jenga-autoscan session with the depth, height and turn")
    parser.add_argument("--layer", type=int, default=13, help="sweep: teammate layer")
    parser.add_argument("--speeds", type=float, nargs="+", default=[15.0, 30.0, 60.0], help="sweep: mm/s")
    parser.add_argument("--via", nargs="*", default=["cali_way_1"], help="sweep: saved poses passed before the face")
    parser.add_argument("--seed", default="cali_way_3", help="sweep: saved pose to seed the face IK")
    parser.add_argument("--end", default="home", help="sweep: saved pose to go to at the end")
    parser.add_argument("--fake-peak-mm", type=float, default=2.0, help="fake camera: sharpest this far out")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("captures"))
    args = parser.parse_args()
    if not 0 < args.mm <= MAX_SWEEP_MM:
        raise SystemExit(f"--mm: 0..{MAX_SWEEP_MM:.0f}")
    if args.step < 0.1:
        raise SystemExit("--step: at least 0.1")

    setup = json.loads(SETUP_FILE.read_text())
    if not setup["dino"].get("tool_hand"):
        raise SystemExit("Dino not calibrated (run jenga-calib first)")
    tool = arm_tool(setup["dino"], args.arm)
    poses = json.loads(POSES_FILE.read_text())[args.arm]
    if args.cmd == "sweep":
        if args.session is None:
            raise SystemExit("sweep: --session captures/autoscan_...")
        args.line_speed = max(args.line_speed, 10.0)
        args.joint_speed = max(args.joint_speed, 10.0)
        if args.start == "cali_way_3":
            args.start = "home"
        sweep_main(args, setup, tool, poses)
        return
    q0 = poses[args.start]
    mover = Mover(args.arm, tool, args.send, q0, args.joint_speed, args.line_speed)
    fake = args.fake or not args.send
    cam = FakeCurveCamera(mover, args.fake_peak_mm, args.send) if fake else RealCamera(args.device, True)
    session = args.out / time.strftime("focus_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    probe = Probe(args, mover, cam, session)
    try:
        cos = check_outward(probe.start, setup["tower"])
        p = probe.start[:3, 3]
        print(f"{args.cmd}: {'SEND' if args.send else 'VIRTUAL'} {'(fake camera)' if fake else '(Dino)'}, "
              f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}, session {session}")
        print(f"focus point now ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m; way out = {-probe.start[:3, 2].round(3)}, "
              f"away from the tower axis (cos {cos:.2f})")
        how = f"{args.mm:.0f} mm straight out in {args.step} mm steps" if args.cmd == "curve" else \
            f"{args.jiggle} mm out and back, {args.reps * len(args.waits)} times"
        print(f"the arm will move: {how}, never deeper than here")
        if args.send:
            cam.show(cam.cam.read(), "waiting for Enter in the terminal")
            if input("Enter = start, q = quit > ").strip().lower() == "q":
                return
        mover.idle = cam.preview if args.send else None
        probe.curve() if args.cmd == "curve" else probe.hold()
    except KeyboardInterrupt:
        print("\nCtrl+C: stopped (arm holds where it is)")
    except ScanError as e:
        print(f"\nSTOPPED: {e}")
    finally:
        probe.close()
        cam.release()
        mover.shutdown()
        cv2.destroyAllWindows()
    print(f"session {session}")


if __name__ == "__main__":
    main()
