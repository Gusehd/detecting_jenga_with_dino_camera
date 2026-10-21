"""Seam test: which block of a layer stands highest, seen at the joints between them.

The layer above rests on the highest block(s) of a layer, so instead of the gap
above one outer block, look at the end faces of a layer where two blocks meet
(LEFT seam = left outer | middle, RIGHT seam = middle | right outer). In one
close-up the gap between the bottom edge of the long block above and the top
edges of the two end faces is measured on each side of the joint: a lower
block has a wider dark gap. Each seam is shot from both ends of the layer, so
the middle block gets 4 gap readings and each outer block 2.

Face 1 = the side where layer --first shows its three end faces. After each
face turn the tower 90 degrees, always the same direction (clockwise seen from
above).

Ranking (default --mode layer). Premise: no layer is all hard; either at least
one outer or the middle comes out. Per layer
    D = median(middle gaps) - median(outer gaps)   (4 readings each)
  D > 0: the middle floats      -> middle, score D
  D < 0: the outers float       -> the outer with the wider gap, score -D;
                                   the other outer, score its gap - middle gap
All other blocks get gap - middle gap (outers) or D (middle), mostly negative.
Blocks are listed by score, ties -> upper layer first, then wider own gap. No
threshold: count how many in a row from the top are easy.
--mode block is the legacy per-block rule (lower than the highest by --tol).

Gaps are measured on the flat top of each end face: the joint area is left out
and the low SIDE_PERCENTILE of the strip widths is used, because rounded block
corners only ever widen the gap.

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
STRIPS = 16  # per side of the seam
SIDE_PERCENTILE = 25  # of the strip widths; rounded corners only widen the gap
SEAM_MARGIN = 50  # px left out on each side of the joint (rounded block corners)
GREEN, RED, YELLOW, WHITE, CYAN = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (255, 255, 0)
BLOCKS = ("left", "middle", "right")  # as seen from the layer's first face


@dataclass
class SeamResult:
    row: int  # center row of the gap band
    seam_x: int  # column of the joint
    gap_left: float  # dark core width above the block left of the joint (px)
    gap_right: float
    left_box: tuple[int, int, int, int] = (0, 0, 0, 0)  # x0, x1, row center, measured part (for drawing)
    right_box: tuple[int, int, int, int] = (0, 0, 0, 0)

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

    def side(x0: int, x1: int) -> tuple[float, tuple[int, int, int, int]]:
        if x1 - x0 < STRIPS:
            return float("nan"), (x0, x1, center, 0)
        bounds = np.linspace(x0, x1, STRIPS + 1).astype(int)
        widths = [float((_smooth(np.median(gray[:, s0:s1], axis=1))[a:b] < dark).sum())
                  for s0, s1 in zip(bounds[:-1], bounds[1:])]
        dark_rows = np.flatnonzero(_smooth(np.median(gray[:, x0:x1], axis=1))[a:b] < dark)
        row = a + int(dark_rows.mean()) if len(dark_rows) else center
        return float(np.percentile(widths, SIDE_PERCENTILE)), (x0, x1, row, 0)

    gl, lb = side(w // 20, seam_x - SEAM_MARGIN)
    gr, rb = side(seam_x + SEAM_MARGIN, w - w // 20)
    return SeamResult(center, seam_x, gl, gr, lb, rb)


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


def block_gaps(shots: dict[str, tuple[float, float]], layer: int, first: int) -> dict[str, list[float]]:
    """Gap readings (px) above each block of a layer; blocks named as seen from its first face."""
    near = 1 if (layer - first) % 2 == 0 else 2  # the layer's first face
    far = near + 2  # opposite end: left and right swap
    out: dict[str, list[float]] = {b: [] for b in BLOCKS}
    for face, seam, on_left, on_right in (
        (near, "left", "left", "middle"), (near, "right", "middle", "right"),
        (far, "left", "right", "middle"), (far, "right", "middle", "left"),
    ):
        v = shots.get(f"L{layer:02d}:f{face}:{seam}")
        if v is not None:
            out[on_left].append(v[0])
            out[on_right].append(v[1])
    return out


def layer_scores(gaps: dict[str, list[float]]) -> dict[str, float]:
    """Score per block of one layer (higher = easier), see the module docstring."""
    if not gaps["middle"] or not (gaps["left"] or gaps["right"]):
        return {}
    mid = float(np.median(gaps["middle"]))
    d = mid - float(np.median(gaps["left"] + gaps["right"]))
    own = {b: min(gaps[b]) - mid for b in ("left", "right") if gaps[b]}
    scores = {"middle": d, **own}
    if d < 0:
        scores[max(own, key=own.get)] = -d
    return scores


def legacy_scores(shots: dict[str, tuple[float, float]], layer: int, first: int) -> dict[str, float]:
    """Old per-block rule: px below the highest block, outer height from the gap step
    against the middle in the same photo, averaged over both ends."""
    near = 1 if (layer - first) % 2 == 0 else 2
    steps: dict[str, list[float]] = {"left": [], "right": []}
    for face, seam, block, outer_is_left in (
        (near, "left", "left", True), (near, "right", "right", False),
        (near + 2, "left", "right", True), (near + 2, "right", "left", False),
    ):
        v = shots.get(f"L{layer:02d}:f{face}:{seam}")
        if v is not None:
            steps[block].append(v[0] - v[1] if outer_is_left else v[1] - v[0])
    heights = {"middle": 0.0, **{b: -float(np.mean(v)) for b, v in steps.items() if v}}
    top = max(heights.values())
    return {b: top - h for b, h in heights.items()}


@dataclass
class Pick:
    layer: int
    block: str
    score: float
    gap: float  # median own gap, tie-break


def rank_blocks(shots: dict[str, tuple[float, float]], layers: list[int], first: int,
                mode: str = "layer") -> list[Pick]:
    """All blocks, easiest first: score desc, ties -> upper layer, then wider own gap."""
    picks = []
    for layer in layers:
        gaps = block_gaps(shots, layer, first)
        scores = layer_scores(gaps) if mode == "layer" else legacy_scores(shots, layer, first)
        picks += [Pick(layer, b, sc, float(np.median(gaps[b]))) for b, sc in scores.items()]
    return sorted(picks, key=lambda p: (-p.score, -p.layer, -p.gap))


def put(img, text, y, color=WHITE, scale=0.6):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def put_at(img, text, org, color=WHITE, scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("captures"))
    parser.add_argument("--first", type=int, default=2, help="lowest layer to test")
    parser.add_argument("--last", type=int, default=16, help="highest layer to test (needs a layer above it)")
    parser.add_argument("--mode", choices=("layer", "block"), default="layer",
                        help="layer: middle-vs-outers ranking (default); block: legacy per-block rule")
    parser.add_argument("--tol", type=float, default=2.0, help="--mode block: px below the highest block -> easy")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--report", action="store_true",
                        help="no camera: print the ranking of --resume (use --remeasure after code changes)")
    parser.add_argument("--remeasure", action="store_true", help="with --report: re-measure the saved photos")
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

    def ranking() -> list[Pick]:
        return rank_blocks(shots, layers, args.first, args.mode)

    def predicted(p: Pick) -> str:
        return "easy" if p.score > (args.tol if args.mode == "block" else 0) else "hard"

    def save_results() -> None:
        with result_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "layer", "block", "score", "pred", "actual"])
            for n, p in enumerate(ranking(), 1):
                w.writerow([n, p.layer, p.block, f"{p.score:.1f}", predicted(p), results.get(f"{p.layer}:{p.block}", "")])

    def report() -> None:
        print(f"\nrank layer  block   score  pred  actual   (mode {args.mode})")
        streak, broken, ok, n = 0, False, 0, 0
        for rank, p in enumerate(ranking(), 1):
            a = results.get(f"{p.layer}:{p.block}", "")
            mark = ""
            if a in ("easy", "hard"):
                n += 1
                ok += a == predicted(p)
                mark = "OK" if a == predicted(p) else "MISS"
                if not broken:
                    streak += a == "easy"
                    broken = a == "hard"
            print(f"{rank:>4} {p.layer:>5}  {p.block:7} {p.score:6.1f}  {predicted(p):5} {a:6}   {mark}")
        if n:
            print(f"easy in a row from the top: {streak}    pred correct {ok}/{n}")

    if args.report:
        if args.remeasure:
            for key in list(shots):
                photos = sorted(session.glob(f"{key.replace(':', '_')}_*.png"))
                if photos:
                    r = measure_seam(cv2.imread(str(photos[-1])))
                    shots[key] = (r.gap_left, r.gap_right)
        report()
        return

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
                        names = ("OUTER", "MIDDLE") if step.seam == "left" else ("MIDDLE", "OUTER")
                        for (x0, x1, row, _), gap, name in ((r.left_box, r.gap_left, names[0]),
                                                            (r.right_box, r.gap_right, names[1])):
                            if gap == gap:  # not nan
                                half = max(int(gap // 2), 1)
                                cv2.rectangle(view, (x0, row - half), (x1, row + half), CYAN, 1)
                                put_at(view, f"{name} {gap:.0f}px", (x0 + 4, min(row + half + 22, h - 30)), CYAN)
                        put(view, f"face {step.face}/4  step {i + 1}/{len(steps)}", 24)
                        put(view, step.instruction(), 52, YELLOW, 0.7)
                        put(view, f"{names[0]} {r.gap_left:.0f}px | {names[1]} {r.gap_right:.0f}px   "
                                  f"(wider gap = lower block)", 80, CYAN, 0.6)
                        put(view, "joint on the cyan vertical line, gap on the side marks", 104, WHITE, 0.5)
                        if i == 0:
                            put(view, f"(face 1: L{args.first} shows its 3 end faces)", 126, WHITE, 0.5)
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
