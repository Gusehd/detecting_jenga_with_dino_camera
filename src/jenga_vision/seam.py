"""Seam test: which block of a layer stands highest, seen at the joints between them.

The layer above rests on the highest block(s) of a layer, so instead of the gap
above one outer block, look at the end faces of a layer where two blocks meet
(LEFT seam = left outer | middle, RIGHT seam = middle | right outer). In one
close-up the gap above the block left of the joint is compared with the gap
above the block right of it: the lower block has the wider dark gap. Both
sides share the same light, exposure and camera tilt, so only the difference
counts. Each seam is shot from both ends of the layer.

Mini tower (12 blocks): layers 1-3 are tested, layer 4 only presses down. Put
a weight (e.g. a 500 ml bottle) on top so load-bearing blocks really grip.
Face 1 = the side where layer --first shows its three end faces. After each
face turn the tower 90 degrees, always the same direction.

Prediction per layer: block height relative to the middle = -(gap_outer -
gap_middle), averaged over both ends; a block lower than the highest one by
more than --tol px is predicted easy.

Keys:
  SPACE    take the shot / confirm the instruction on screen
  z        go back one step
  e / h    check phase: pushed block was easy / hard   (u: skip)
  q / ESC  quit; resume later with --resume <session folder>
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from jenga_vision.camera import DinoCamera
from jenga_vision.led import set_led
from jenga_vision.long_edge import CORE_WINDOW, DARK_FRACTION, _smooth

FRAMES_PER_SHOT = 3
STRIPS = 8  # per side of the seam
SEAM_MARGIN = 50  # px left out on each side of the joint (rounded block corners)
GREEN, RED, YELLOW, WHITE, CYAN = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (255, 255, 0)
BLOCKS = ("left", "middle", "right")  # as seen from the layer's first face


@dataclass
class SeamResult:
    row: int  # center row of the gap band
    seam_x: int  # column of the joint
    gap_left: float  # dark core width above the block left of the joint (px)
    gap_right: float

    @property
    def step(self) -> float:
        """> 0: the left block stands lower than the right one."""
        return self.gap_left - self.gap_right


def measure_seam(frame: np.ndarray) -> SeamResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    dark = DARK_FRACTION * float(np.median(gray))
    p_all = _smooth(np.median(gray[:, w // 10 : w - w // 10], axis=1), 9)
    lo, hi = h // 4, 3 * h // 4
    center = lo + int(np.argmin(p_all[lo:hi]))
    a, b = max(center - CORE_WINDOW, 0), min(center + CORE_WINDOW + 1, h)

    # The joint is the darkest column in the end faces below the gap band.
    below = gray[min(center + 20, h - 1) : h - 10, :]
    cols = _smooth(below.mean(axis=0), 15) if below.size else np.zeros(w)
    c0, c1 = w // 5, w - w // 5
    seam_x = c0 + int(np.argmin(cols[c0:c1]))

    def side(x0: int, x1: int) -> float:
        if x1 - x0 < STRIPS:
            return float("nan")
        bounds = np.linspace(x0, x1, STRIPS + 1).astype(int)
        widths = [float((_smooth(np.median(gray[:, s0:s1], axis=1))[a:b] < dark).sum())
                  for s0, s1 in zip(bounds[:-1], bounds[1:])]
        return float(np.median(widths))

    return SeamResult(center, seam_x, side(w // 20, seam_x - SEAM_MARGIN), side(seam_x + SEAM_MARGIN, w - w // 20))


@dataclass(frozen=True)
class Step:
    kind: str  # "shot" or "rotate"
    face: int
    layer: int = 0
    seam: str = ""  # "left" / "right" as seen on screen

    @property
    def key(self) -> str:
        return f"L{self.layer:02d}:f{self.face}:{self.seam}"

    def instruction(self) -> str:
        if self.kind == "rotate":
            return f"TURN THE TOWER 90 deg -> face {self.face}, then SPACE"
        return f"L{self.layer} end faces: {self.seam.upper()} seam"


def build_steps(first: int, last: int) -> list[Step]:
    steps: list[Step] = []
    for face in range(1, 5):
        if face > 1:
            steps.append(Step("rotate", face))
        for layer in range(first, last + 1):
            # Layers with the parity of `first` show their end faces on faces 1 and 3.
            if (layer - first) % 2 == (0 if face in (1, 3) else 1):
                steps += [Step("shot", face, layer, s) for s in ("left", "right")]
    return steps


def outer_gaps(shots: dict[str, tuple[float, float]], layer: int, first: int) -> dict[str, list[float]]:
    """gap_outer - gap_middle per outer block, one value per seam shot (> 0: outer lower)."""
    near = 1 if (layer - first) % 2 == 0 else 2  # the layer's first face
    far = near + 2
    out: dict[str, list[float]] = {"left": [], "right": []}
    for face, seam, block, outer_is_left in (
        (near, "left", "left", True), (near, "right", "right", False),
        (far, "left", "right", True), (far, "right", "left", False),
    ):
        v = shots.get(f"L{layer:02d}:f{face}:{seam}")
        if v is None:
            continue
        gl, gr = v
        out[block].append(gl - gr if outer_is_left else gr - gl)
    return out


def predict_layer(rel: dict[str, list[float]], tol: float) -> dict[str, tuple[float, str]]:
    """Height of each block relative to the middle (px, + = higher) and its prediction."""
    heights = {"middle": 0.0}
    for block in ("left", "right"):
        if rel[block]:
            heights[block] = -float(np.mean(rel[block]))
    top = max(heights.values())
    return {b: (hgt, "easy" if top - hgt > tol else "hard") for b, hgt in heights.items()}


def put(img, text, y, color=WHITE, scale=0.6):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("captures"))
    parser.add_argument("--first", type=int, default=1, help="lowest layer to test")
    parser.add_argument("--last", type=int, default=3, help="highest layer to test (needs a layer above it)")
    parser.add_argument("--tol", type=float, default=2.0, help="px lower than the highest block -> easy")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    steps = build_steps(args.first, args.last)
    session = args.resume or args.out / time.strftime("seam_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    shots_csv, result_csv = session / "seams.csv", session / "blocks.csv"
    layers = list(range(args.first, args.last + 1))
    checks = [(layer, b) for layer in layers for b in BLOCKS]

    shots: dict[str, tuple[float, float]] = {}
    if shots_csv.exists():
        with shots_csv.open() as f:
            shots = {r["key"]: (float(r["gap_left"]), float(r["gap_right"])) for r in csv.DictReader(f)}
    results: dict[str, str] = {}
    if result_csv.exists():
        with result_csv.open() as f:
            results = {f"{r['layer']}:{r['block']}": r["actual"] for r in csv.DictReader(f) if r["actual"]}

    def save_shots() -> None:
        with shots_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "gap_left", "gap_right", "step"])
            w.writerows([k, f"{gl:.1f}", f"{gr:.1f}", f"{gl - gr:.1f}"] for k, (gl, gr) in shots.items())

    def predictions() -> dict[str, tuple[float, str]]:
        out = {}
        for layer in layers:
            for block, v in predict_layer(outer_gaps(shots, layer, args.first), args.tol).items():
                out[f"{layer}:{block}"] = v
        return out

    def save_results() -> None:
        pred = predictions()
        with result_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "block", "height_px", "pred", "actual"])
            for layer, block in checks:
                hgt, p = pred.get(f"{layer}:{block}", (float("nan"), ""))
                w.writerow([layer, block, f"{hgt:.1f}", p, results.get(f"{layer}:{block}", "")])

    def report() -> None:
        pred = predictions()
        print("\nlayer  block   height  pred  actual")
        ok = n = 0
        for layer, block in checks:
            hgt, p = pred.get(f"{layer}:{block}", (float("nan"), "-"))
            a = results.get(f"{layer}:{block}", "")
            mark = ""
            if a in ("easy", "hard") and p:
                n += 1
                ok += a == p
                mark = "OK" if a == p else "MISS"
            print(f"{layer:>5}  {block:7} {hgt:6.1f}  {p:5} {a:6} {mark}")
        if n:
            print(f"{ok}/{n} correct")

    def face_of(layer: int) -> int:
        return 1 if (layer - args.first) % 2 == 0 else 2

    i = next((n for n, s in enumerate(steps) if s.kind == "shot" and s.key not in shots), len(steps))
    while i > 0 and steps[i - 1].kind == "rotate":
        i -= 1
    phase = "scan" if i < len(steps) else "check"
    idx = next((n for n, (lay, b) in enumerate(checks) if f"{lay}:{b}" not in results), len(checks))

    with DinoCamera(args.device) as cam:
        for _ in range(5):
            cam.read()
        set_led(False)
        print(f"Session: {session}   ({len(steps)} steps, layers {args.first}-{args.last})")
        try:
            while True:
                frame = cam.read()
                h, w = frame.shape[:2]
                view = frame.copy()
                if phase == "scan":
                    step = steps[i]
                    if step.kind == "shot":
                        r = measure_seam(frame)
                        cv2.line(view, (w // 2, h // 2 + 20), (w // 2, h), CYAN, 1)
                        cv2.line(view, (0, h // 2), (w // 8, h // 2), CYAN, 2)
                        cv2.line(view, (w - w // 8, h // 2), (w, h // 2), CYAN, 2)
                        cv2.line(view, (r.seam_x, r.row + 10), (r.seam_x, h), GREEN, 1)
                        put(view, f"face {step.face}/4  step {i + 1}/{len(steps)}   "
                                  f"gap L {r.gap_left:.0f}  R {r.gap_right:.0f}  step {r.step:+.0f}px", 24)
                        put(view, step.instruction(), 56, YELLOW, 0.7)
                        put(view, "joint on the cyan vertical line, gap on the side marks", 84, WHITE, 0.5)
                        if i == 0:
                            put(view, f"(face 1: L{args.first} shows its 3 end faces)", 108, WHITE, 0.5)
                    else:
                        put(view, f"step {i + 1}/{len(steps)}", 24)
                        put(view, step.instruction(), 60, YELLOW, 0.7)
                    put(view, "SPACE: shoot/ok   z: back   q: quit (resumable)", h - 12, WHITE, 0.5)
                else:
                    if idx < len(checks):
                        layer, block = checks[idx]
                        put(view, f"CHECK {idx + 1}/{len(checks)}  (weight on top!)", 24)
                        put(view, f"PUSH L{layer} {block.upper()} (seen from face {face_of(layer)})", 60, YELLOW, 0.75)
                        put(view, "e: easy  h: hard  u: skip  z: back", 92, YELLOW)
                    else:
                        put(view, "CHECK DONE - see terminal", 40, GREEN, 1.0)
                cv2.imshow("jenga seam", view)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if phase == "scan":
                    step = steps[i]
                    if key == ord("z") and i > 0:
                        i -= 1
                        if steps[i].kind == "shot":
                            shots.pop(steps[i].key, None)
                            save_shots()
                    elif key == ord(" "):
                        if step.kind == "shot":
                            frames = [frame] + [cam.read() for _ in range(FRAMES_PER_SHOT - 1)]
                            rs = [measure_seam(f) for f in frames]
                            gl = float(np.median([r.gap_left for r in rs]))
                            gr = float(np.median([r.gap_right for r in rs]))
                            shots[step.key] = (gl, gr)
                            save_shots()
                            cv2.imwrite(str(session / f"{step.key.replace(':', '_')}_{time.time() * 1000:.0f}.png"), frame)
                            print(f"[{i + 1}/{len(steps)}] {step.key}: gap L {gl:.1f}  R {gr:.1f}  step {gl - gr:+.1f}px")
                        i += 1
                        if i == len(steps):
                            phase, idx = "check", 0
                            save_results()
                            report()
                else:
                    if key in (ord("e"), ord("h"), ord("u")) and idx < len(checks):
                        layer, block = checks[idx]
                        results[f"{layer}:{block}"] = {ord("e"): "easy", ord("h"): "hard", ord("u"): "skip"}[key]
                        save_results()
                        idx += 1
                        if idx == len(checks):
                            report()
                    elif key == ord("z") and idx > 0:
                        idx -= 1
        finally:
            set_led(True)
            save_shots()
            save_results()
    cv2.destroyAllWindows()
    print(f"saved {session}")


if __name__ == "__main__":
    main()
