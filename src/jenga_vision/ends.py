"""End-face test: the gap above each block's end face, one block per shot.

Like jenga-seam, but instead of the joint between two blocks each block's end
face is centered in the frame, so its rounded corners fall outside the
measured part. Per layer 3 blocks x both ends = 6 shots; layers 2-16 = 90
shots.

The gap is between the top edge of the end face and the bottom edge of the
long block above. The two ends of a layer sit under different blocks of the
layer above, so gaps are only compared within one end:
    middle score = gM - (gL + gR) / 2      (> 0: the middle floats)
    outer score  = g_outer - gM            (> 0: that outer floats)
--report prints the ranking (score desc, ties -> upper layer) for several
ways to use the ends: near end only (faces 1/2), far end only (faces 3/4),
mean of both ends, min of both ends, plus how often the two ends agree.

Face 1 = the side where layer --first shows its three end faces. Turn the
tower clockwise (seen from above), 90 degrees per face. Block names in files
are as seen from the layer's first face (left/middle/right); on the far face
the block on screen left is the layer's "right".

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
STRIPS = 16
SIDE_PERCENTILE = 25  # of the strip widths; leftover corner rounding only widens the gap
MEASURE_SPAN = (0.2, 0.8)  # fraction of the frame width measured (center of the end face)
GREEN, RED, YELLOW, WHITE, CYAN = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (255, 255, 0)
BLOCKS = ("left", "middle", "right")  # as seen from the layer's first face
METHODS = ("near", "far", "mean", "min")


@dataclass
class EndGap:
    row: int  # center row of the gap band
    gap: float  # dark core width (px)
    x0: int
    x1: int


def measure_end(frame: np.ndarray) -> EndGap:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    x0, x1 = int(w * MEASURE_SPAN[0]), int(w * MEASURE_SPAN[1])
    dark = DARK_FRACTION * float(np.median(gray))
    p_all = _smooth(np.median(gray[:, x0:x1], axis=1), 9)
    lo, hi = h // 4, 3 * h // 4
    center = lo + int(np.argmin(p_all[lo:hi]))
    a, b = max(center - CORE_WINDOW, 0), min(center + CORE_WINDOW + 1, h)
    bounds = np.linspace(x0, x1, STRIPS + 1).astype(int)
    widths = [float((_smooth(np.median(gray[:, s0:s1], axis=1))[a:b] < dark).sum())
              for s0, s1 in zip(bounds[:-1], bounds[1:])]
    dark_rows = np.flatnonzero(p_all[a:b] < dark)
    row = a + int(dark_rows.mean()) if len(dark_rows) else center
    return EndGap(row, float(np.percentile(widths, SIDE_PERCENTILE)), x0, x1)


@dataclass(frozen=True)
class Step:
    kind: str  # "shot" or "rotate"
    face: int
    layer: int = 0
    screen: str = ""  # block position on screen: left / middle / right

    @property
    def key(self) -> str:
        return f"L{self.layer:02d}:f{self.face}:{self.screen}"

    def instruction(self) -> str:
        if self.kind == "rotate":
            return f"TURN THE TOWER 90 deg CLOCKWISE -> face {self.face}, then SPACE"
        return f"L{self.layer} {self.screen.upper()} end face"


def build_steps(first: int, last: int) -> list[Step]:
    steps: list[Step] = []
    for face in range(1, 5):
        if face > 1:
            steps.append(Step("rotate", face))
        for layer in range(first, last + 1):
            # Layers with the parity of `first` show their end faces on faces 1 and 3.
            if (layer - first) % 2 == (0 if face in (1, 3) else 1):
                steps += [Step("shot", face, layer, s) for s in BLOCKS]
    return steps


def end_gaps(shots: dict[str, float], layer: int, first: int) -> dict[str, dict[str, float]]:
    """{"near": {block: gap}, "far": {...}} with blocks named from the layer's first face."""
    near = 1 if (layer - first) % 2 == 0 else 2
    swap = {"left": "right", "middle": "middle", "right": "left"}
    out: dict[str, dict[str, float]] = {"near": {}, "far": {}}
    for end, face, names in (("near", near, {b: b for b in BLOCKS}), ("far", near + 2, swap)):
        for screen in BLOCKS:
            v = shots.get(f"L{layer:02d}:f{face}:{screen}")
            if v is not None:
                out[end][names[screen]] = v
    return out


def end_scores(g: dict[str, float]) -> dict[str, float]:
    """Scores within one end (higher = floats more = easier)."""
    if len(g) < 3:
        return {}
    mid = g["middle"]
    return {"middle": mid - (g["left"] + g["right"]) / 2, "left": g["left"] - mid, "right": g["right"] - mid}


def block_scores(shots: dict[str, float], layer: int, first: int, method: str) -> dict[str, float]:
    ends = end_gaps(shots, layer, first)
    near, far = end_scores(ends["near"]), end_scores(ends["far"])
    if method == "near":
        return near
    if method == "far":
        return far
    if not near or not far:
        return near or far
    comb = np.mean if method == "mean" else min
    return {b: float(comb([near[b], far[b]])) for b in BLOCKS}


def ranking(shots: dict[str, float], layers: list[int], first: int, method: str) -> list[tuple[float, int, str]]:
    """(score, layer, block), easiest first; ties -> upper layer."""
    items = [(sc, layer, b) for layer in layers for b, sc in block_scores(shots, layer, first, method).items()]
    return sorted(items, key=lambda x: (-x[0], -x[1]))


def streak(order: list[tuple[float, int, str]], results: dict[str, str]) -> int:
    n = 0
    for _, layer, b in order:
        a = results.get(f"{layer}:{b}")
        if a == "easy":
            n += 1
        elif a == "hard":
            break
    return n


def print_report(shots: dict[str, float], results: dict[str, str], layers: list[int], first: int) -> None:
    orders = {m: ranking(shots, layers, first, m) for m in METHODS}
    print("\nrank  " + "  ".join(f"{m:>16}" for m in METHODS))
    for r in range(max(len(o) for o in orders.values())):
        cells = []
        for m in METHODS:
            if r < len(orders[m]):
                sc, layer, b = orders[m][r]
                a = results.get(f"{layer}:{b}", "")
                cells.append(f"L{layer:02d} {b[0].upper()} {sc:5.1f} {a[:1] or '.'}")
            else:
                cells.append("")
        print(f"{r + 1:>4}  " + "  ".join(f"{c:>16}" for c in cells))
    if results:
        print("\neasy in a row from the top: " + "   ".join(f"{m} {streak(orders[m], results)}" for m in METHODS))

    # Do the two ends point at the same block?
    same_top = same_type = n = 0
    print("\nlayer  near L/M/R gaps    far L/M/R gaps     near top  far top   truth")
    for layer in layers:
        ends = end_gaps(shots, layer, first)
        ns, fs = end_scores(ends["near"]), end_scores(ends["far"])
        if not ns or not fs:
            continue
        n += 1
        nt, ft = max(ns, key=ns.get), max(fs, key=fs.get)
        same_top += nt == ft
        same_type += (ns["middle"] > 0) == (fs["middle"] > 0)
        truth = "".join((results.get(f"{layer}:{b}", "?") or "?")[0].upper() for b in BLOCKS)
        print(f"{layer:>5}  " + " ".join(f"{ends['near'][b]:5.1f}" for b in BLOCKS) + "  "
              + " ".join(f"{ends['far'][b]:5.1f}" for b in BLOCKS) + f"   {nt:8}  {ft:8}  {truth}")
    if n:
        print(f"\nends agree on the most floating block: {same_top}/{n}   "
              f"on the layer type (middle floats or not): {same_type}/{n}")


def put(img, text, y, color=WHITE, scale=0.6):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("captures"))
    parser.add_argument("--first", type=int, default=2, help="lowest layer to test")
    parser.add_argument("--last", type=int, default=16, help="highest layer to test (needs a layer above it)")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--report", action="store_true", help="no camera: print the rankings of --resume")
    args = parser.parse_args()

    steps = build_steps(args.first, args.last)
    session = args.resume or args.out / time.strftime("ends_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    shots_csv, result_csv = session / "ends.csv", session / "blocks.csv"
    layers = list(range(args.first, args.last + 1))
    checks = [(layer, b) for layer in layers for b in BLOCKS]

    shots: dict[str, float] = {}
    if shots_csv.exists():
        with shots_csv.open() as f:
            shots = {r["key"]: float(r["gap"]) for r in csv.DictReader(f)}
    results: dict[str, str] = {}
    if result_csv.exists():
        with result_csv.open() as f:
            results = {f"{r['layer']}:{r['block']}": r["actual"] for r in csv.DictReader(f) if r["actual"]}

    if args.report:
        print_report(shots, results, layers, args.first)
        return

    def save_shots() -> None:
        with shots_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "gap"])
            w.writerows([k, f"{g:.1f}"] for k, g in shots.items())

    def save_results() -> None:
        scores = {m: {f"{lay}:{b}": sc for sc, lay, b in ranking(shots, layers, args.first, m)} for m in METHODS}
        with result_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "block", *[f"score_{m}" for m in METHODS], "actual"])
            for layer, block in checks:
                k = f"{layer}:{block}"
                w.writerow([layer, block, *[f"{scores[m][k]:.1f}" if k in scores[m] else "" for m in METHODS],
                            results.get(k, "")])

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
                        r = measure_end(frame)
                        half = max(int(r.gap // 2), 1)
                        cv2.rectangle(view, (r.x0, r.row - half), (r.x1, r.row + half), CYAN, 1)
                        cv2.line(view, (w // 2, h // 2 + 20), (w // 2, h), CYAN, 1)
                        cv2.line(view, (0, h // 2), (w // 8, h // 2), CYAN, 2)
                        cv2.line(view, (w - w // 8, h // 2), (w, h // 2), CYAN, 2)
                        put(view, f"face {step.face}/4  step {i + 1}/{len(steps)}   gap {r.gap:.0f}px", 24)
                        put(view, step.instruction(), 52, YELLOW, 0.7)
                        put(view, "end face centered on the cyan vertical line, gap on the side marks", 80, WHITE, 0.5)
                        if i == 0:
                            put(view, f"(face 1: L{args.first} shows its 3 end faces)", 102, WHITE, 0.5)
                    else:
                        put(view, f"step {i + 1}/{len(steps)}", 24)
                        put(view, step.instruction(), 60, YELLOW, 0.7)
                    put(view, "SPACE: shoot/ok   z: back   q: quit (resumable)", h - 12, WHITE, 0.5)
                else:
                    if idx < len(checks):
                        layer, block = checks[idx]
                        put(view, f"CHECK {idx + 1}/{len(checks)}", 24)
                        put(view, f"PUSH L{layer} {block.upper()} (seen from face {face_of(layer)})", 60, YELLOW, 0.75)
                        put(view, "e: easy  h: hard  u: skip  z: back", 92, YELLOW)
                    else:
                        put(view, "CHECK DONE - see terminal", 40, GREEN, 1.0)
                cv2.imshow("jenga ends", view)

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
                            g = float(np.median([measure_end(f).gap for f in frames]))
                            shots[step.key] = g
                            save_shots()
                            cv2.imwrite(str(session / f"{step.key.replace(':', '_')}_{time.time() * 1000:.0f}.png"), frame)
                            print(f"[{i + 1}/{len(steps)}] {step.key}: gap {g:.1f}px")
                        i += 1
                        if i == len(steps):
                            phase, idx = "check", 0
                            save_results()
                            print_report(shots, results, layers, args.first)
                else:
                    if key in (ord("e"), ord("h"), ord("u")) and idx < len(checks):
                        layer, block = checks[idx]
                        results[f"{layer}:{block}"] = {ord("e"): "easy", ord("h"): "hard", ord("u"): "skip"}[key]
                        save_results()
                        idx += 1
                        if idx == len(checks):
                            print_report(shots, results, layers, args.first)
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
