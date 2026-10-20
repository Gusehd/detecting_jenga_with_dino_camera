"""Guided whole-tower scan: the tool walks you through every shot, then the check.

Setup: camera fixed, Dino LED off (done automatically), zoom 50, cap touching.
The tower stands in a jig that hides layer 1, and layers 1 and 17 (top) are
never pulled, so layers 2-16 are scanned by default.
Face 1 = the face where layer 2 shows its three short end faces. After each
face you turn the tower 90 degrees (always the same direction), so all four
faces are shot under the same light.

On each face the scan climbs the layers that show their long side there
(every second layer): outer block, top edge, LEFT / CENTER / RIGHT.
Outer score = median of its 3 shots (--agg min: the narrowest of the 3, kept
as an option). Layers are then judged as a whole (pick.layer_ranking): if the
larger of the two outer scores is >= --sure, both outer blocks are SURE.
Middle blocks are not recommended by default: inferring them from the outer
scores failed on the second tower (--middle-below 14 turns the legacy rule
back on). They are not shot either (no signal in their end-face score);
--with-middle adds those shots back for data collection.
After the last shot the tool asks you to push the blocks in rank order.

Keys:
  SPACE    take the shot / confirm the instruction on screen
  z        go back one step (the shot there is taken again)
  e / h    check phase: pushed block was easy / hard
  u        check phase: cannot test, skip
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
from jenga_vision.long_edge import measure_single
from jenga_vision.pick import LayerPick, layer_ranking

FRAMES_PER_SHOT = 3
SEGMENTS = ("left", "center", "right")
GREEN, RED, YELLOW, WHITE, CYAN = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (255, 255, 0)


@dataclass(frozen=True)
class Step:
    kind: str  # "shot" or "rotate"
    face: int
    layer: int = 0
    block: str = ""  # e.g. "L07-P1", "L06-P2"
    part: str = ""  # left / center / right for outer, end-f<face> for middle

    @property
    def key(self) -> str:
        return f"{self.block}:{self.part}"

    def instruction(self) -> str:
        if self.kind == "rotate":
            return f"TURN THE TOWER 90 deg -> face {self.face}, then SPACE"
        if self.block.endswith("P2"):
            return f"L{self.layer} MIDDLE block END face, top edge"
        return f"L{self.layer} OUTER block long side, top edge {self.part.upper()}"


def outer_position(face: int) -> int:
    # Faces 1-2 see the P1 side of their layers, faces 3-4 the opposite P3 side.
    return 1 if face <= 2 else 3


def build_steps(first: int, last: int, with_middle: bool = False) -> list[Step]:
    steps: list[Step] = []
    for face in range(1, 5):
        if face > 1:
            steps.append(Step("rotate", face))
        long_parity = 1 if face in (1, 3) else 0  # even layers show end faces on faces 1/3
        for layer in range(first, last + 1):
            if layer % 2 == long_parity:
                block = f"L{layer:02d}-P{outer_position(face)}"
                steps += [Step("shot", face, layer, block, seg) for seg in SEGMENTS]
            elif with_middle:
                steps.append(Step("shot", face, layer, f"L{layer:02d}-P2", f"end-f{face}"))
    return steps


def block_scores(shots: dict[str, float], agg: str = "median") -> dict[str, float]:
    parts: dict[str, list[float]] = {}
    for key, px in shots.items():
        parts.setdefault(key.split(":")[0], []).append(px)
    return {
        block: (min(v) if block.endswith("P2") or agg == "min" else float(np.mean(v) if agg == "mean" else np.median(v)))
        for block, v in parts.items()
    }


def put(img, text, y, color=WHITE, scale=0.6):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def describe(block: str) -> str:
    layer, pos = int(block[1:3]), block[-2:]
    if pos == "P2":
        return f"L{layer} MIDDLE"
    face = (1 if layer % 2 else 2) + (0 if pos == "P1" else 2)
    return f"L{layer} OUTER on face {face}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("captures"))
    parser.add_argument("--first", type=int, default=2, help="lowest layer to scan (1 sits in the jig)")
    parser.add_argument("--last", type=int, default=16, help="highest layer to scan (top layer excluded)")
    parser.add_argument("--sure", type=float, default=16.0, help="layer outer px for both outers SURE")
    parser.add_argument("--middle-below", type=float, default=0.0,
                        help="legacy: layer outer px under which P2 is LIKELY (0 = off)")
    parser.add_argument("--agg", choices=("median", "mean", "min"), default="median",
                        help="outer score from its 3 shots")
    parser.add_argument("--with-middle", action="store_true", help="also shoot the middle end faces (data only)")
    parser.add_argument("--resume", type=Path, default=None, help="session folder to continue")
    args = parser.parse_args()

    steps = build_steps(args.first, args.last, args.with_middle)
    session = args.resume or args.out / time.strftime("tower_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    shots_csv, picks_csv = session / "shots.csv", session / "picks.csv"

    shots: dict[str, float] = {}
    if shots_csv.exists():
        with shots_csv.open() as f:
            shots = {r["key"]: float(r["px"]) for r in csv.DictReader(f)}
    results: dict[str, str] = {}
    if picks_csv.exists():
        with picks_csv.open() as f:
            results = {r["block"]: r["actual"] for r in csv.DictReader(f) if r["actual"]}

    def save_shots() -> None:
        with shots_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "px"])
            w.writerows([k, f"{v:.1f}"] for k, v in shots.items())

    def ranked() -> list[LayerPick]:
        return layer_ranking(block_scores(shots, args.agg), args.sure, args.middle_below)

    def save_picks() -> None:
        with picks_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "block", "where", "px", "layer_px", "tier", "actual"])
            for rank, pk in enumerate(ranked(), 1):
                w.writerow([rank, pk.block, describe(pk.block), "" if pk.px is None else f"{pk.px:.1f}",
                            f"{pk.layer_px:.1f}", pk.tier, results.get(pk.block, "")])

    def summary() -> str:
        tested = [pk for pk in ranked() if results.get(pk.block) in ("easy", "hard")]
        parts = []
        for name in ("SURE", "LIKELY"):
            group = [pk.block for pk in tested if pk.tier == name]
            if group:
                parts.append(f"{name} {sum(results[b] == 'easy' for b in group)}/{len(group)}")
        return "   ".join(parts) or "no results yet"

    def label(pk: LayerPick) -> str:
        if pk.block.endswith("P2"):
            return f"{describe(pk.block)}  (outers max {pk.layer_px:.0f}px)"
        return f"{describe(pk.block)}  ({pk.px:.0f}px, layer {pk.layer_px:.0f}px)"

    # Resume where the shots stop; rotate steps right after a finished face are replayed.
    i = next((n for n, s in enumerate(steps) if s.kind == "shot" and s.key not in shots), len(steps))
    while i > 0 and steps[i - 1].kind == "rotate":
        i -= 1
    phase = "scan" if i < len(steps) else "check"
    idx = 0
    if phase == "check":
        idx = next((n for n, pk in enumerate(ranked()) if pk.block not in results), len(ranked()))

    with DinoCamera(args.device) as cam:
        # The LED command is ignored until the camera is streaming.
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
                        band = measure_single(frame)
                        half = max(int(band.width_px // 2), 1)
                        cv2.rectangle(view, (w // 10, band.row - half), (w - w // 10, band.row + half), CYAN, 1)
                        cv2.line(view, (0, h // 2), (w // 8, h // 2), CYAN, 2)
                        cv2.line(view, (w - w // 8, h // 2), (w, h // 2), CYAN, 2)
                        put(view, f"face {step.face}/4   step {i + 1}/{len(steps)}   live {band.width_px:.0f}px", 24)
                        put(view, step.instruction(), 56, YELLOW, 0.7)
                        if i == 0:
                            put(view, f"(stand on the side where L{args.first} shows 3 short end faces)", 86, WHITE, 0.5)
                    else:
                        put(view, f"step {i + 1}/{len(steps)}", 24)
                        put(view, step.instruction(), 60, YELLOW, 0.7)
                    put(view, "SPACE: shoot/ok   z: back   q: quit (resumable)", h - 12, WHITE, 0.5)
                else:
                    order = ranked()
                    if idx < len(order):
                        pk = order[idx]
                        color = GREEN if pk.tier == "SURE" else YELLOW if pk.tier == "LIKELY" else WHITE
                        put(view, f"CHECK rank {idx + 1}/{len(order)}   {pk.tier}", 24, color)
                        put(view, f"PUSH {label(pk)}", 60, color, 0.7)
                        put(view, "e: easy  h: hard  u: skip  z: back", 92, YELLOW)
                    else:
                        put(view, "CHECK DONE", 40, GREEN, 1.0)
                    put(view, summary(), h - 12, YELLOW)
                cv2.imshow("jenga tower", view)

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
                            px = float(np.median([measure_single(f).width_px for f in frames]))
                            shots[step.key] = px
                            save_shots()
                            name = f"f{step.face}_{step.block}_{step.part}_{time.time() * 1000:.0f}.png"
                            cv2.imwrite(str(session / name), frame)
                            print(f"[{i + 1}/{len(steps)}] face {step.face} {step.block} {step.part}: {px:.1f}px")
                        i += 1
                        if i == len(steps):
                            phase, idx = "check", 0
                            save_picks()
                            print("\nranking:")
                            for rank, pk in enumerate(ranked(), 1):
                                print(f"  {rank:2d}. {pk.tier:6s} {label(pk)}")
                else:
                    order = ranked()
                    if key in (ord("e"), ord("h"), ord("u")) and idx < len(order):
                        block = order[idx].block
                        results[block] = {ord("e"): "easy", ord("h"): "hard", ord("u"): "skip"}[key]
                        save_picks()
                        print(f"rank {idx + 1}: {describe(block)} -> {results[block]}   ({summary()})")
                        idx += 1
                    elif key == ord("z") and idx > 0:
                        idx -= 1
        finally:
            set_led(True)
            save_shots()
            if shots:
                save_picks()
    cv2.destroyAllWindows()
    print(f"saved {session}")


if __name__ == "__main__":
    main()
