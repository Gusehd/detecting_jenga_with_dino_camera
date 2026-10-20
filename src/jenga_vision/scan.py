"""Whole-tower scan of the outer blocks, ranked by top-edge gap, then checked.

Phase 1 - scan (labels unknown):
  per outer block: set layer / position, shoot top edge LEFT, CENTER, RIGHT
  (SPACE x3, cap touching, zoom fixed, Dino LED off). The third shot stores the
  block and moves to layer + 2 (the next outer block on the same side).
Phase 2 - check (press p):
  blocks are shown in rank order (widest black gap first); push each one and
  press e / h. The precision of the top 5 / top 10 is shown as you go.

Keys (scan):
  [ / ]    layer -1 / +1 (1 = bottom)
  1 2 3    position in layer
  SPACE    measure + save the current segment
  z        retake the previous segment (before the third shot)
  x        discard the current block's shots
  (to redo a stored block, go back to its layer and shoot it again: it is replaced)
  p        finish scanning, start the check phase
Keys (check):
  e / h    pushed: easy / hard
  u        cannot test (skipped, not counted)
  z        back to the previous rank
Any phase:
  q / ESC  quit (everything so far is already saved)

Middle-block mode (--middle): scans P2 through its short end faces instead.
  The top edge of P2's end face is shot once per end (a / b selects the end),
  each shot moves to layer + 2. Shoot all end-A faces on one side, turn the
  tower 180 degrees, press b and shoot the end-B faces. Rank = smaller of the
  two ends (a gap must be open at both ends).
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from jenga_vision.camera import DinoCamera
from jenga_vision.led import set_led
from jenga_vision.long_edge import EASY_TOP_PX, measure_single

SEGMENTS = ("left", "center", "right")
FRAMES_PER_SHOT = 3
GREEN, RED, YELLOW, WHITE, CYAN = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (255, 255, 0)


@dataclass
class Block:
    layer: int
    position: int
    widths: list[float] = field(default_factory=list)  # long face: left, center, right
    ends: dict[str, float] = field(default_factory=dict)  # middle mode: end a / b
    actual: str = ""  # "", easy, hard, skip

    @property
    def px(self) -> float:
        if self.ends:
            return min(self.ends.values())
        return float(np.median(self.widths))


@dataclass
class Scan:
    session: Path
    middle: bool = False
    blocks: list[Block] = field(default_factory=list)

    def store(self, block: Block) -> None:
        self.blocks = [b for b in self.blocks if (b.layer, b.position) != (block.layer, block.position)]
        self.blocks.append(block)
        self.save()

    def store_end(self, layer: int, end: str, px: float) -> Block:
        block = next((b for b in self.blocks if b.layer == layer), None)
        if block is None:
            block = Block(layer, 2)
            self.blocks.append(block)
        block.ends[end] = px
        self.save()
        return block

    def ranked(self) -> list[Block]:
        return sorted(self.blocks, key=lambda b: b.px, reverse=True)

    def precision(self, k: int) -> tuple[int, int]:
        """(easy, tested) among the top-k ranks that have a result."""
        tested = [b for b in self.ranked()[:k] if b.actual in ("easy", "hard")]
        return sum(b.actual == "easy" for b in tested), len(tested)

    def save(self) -> None:
        with (self.session / "scan.csv").open("w", newline="") as f:
            w = csv.writer(f)
            parts = ["px_a", "px_b"] if self.middle else ["px_left", "px_center", "px_right"]
            w.writerow(["rank", "layer", "position", *parts, "px", "actual"])
            for rank, b in enumerate(self.ranked(), 1):
                values = [b.ends.get(e) for e in "ab"] if self.middle else b.widths
                w.writerow([rank, b.layer, b.position, *("" if v is None else f"{v:.1f}" for v in values),
                            f"{b.px:.1f}", b.actual])


def put(img, text, y, color=WHITE, scale=0.6):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def summary(scan: Scan) -> str:
    parts = []
    for k in (5, 10):
        easy, tested = scan.precision(k)
        parts.append(f"top{k}: {easy}/{tested} easy")
    return "   ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("captures"))
    parser.add_argument("--middle", action="store_true", help="scan P2 through its end faces")
    args = parser.parse_args()

    prefix = "scan_middle" if args.middle else "scan"
    session = args.out / time.strftime(f"{prefix}_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    scan = Scan(session, middle=args.middle)

    layer, position = (2, 2) if args.middle else (1, 3)
    end = "a"
    widths: list[float] = []
    phase = "scan"
    idx = 0  # rank index in the check phase
    with DinoCamera(args.device) as cam:
        # The LED command is ignored until the camera is streaming.
        for _ in range(5):
            cam.read()
        set_led(False)
        print(f"Session: {session}")
        try:
            while True:
                frame = cam.read()
                h, w = frame.shape[:2]
                view = frame.copy()

                if phase == "scan":
                    band = measure_single(frame)
                    half = max(int(band.width_px // 2), 1)
                    cv2.rectangle(view, (w // 10, band.row - half), (w - w // 10, band.row + half), CYAN, 1)
                    cv2.line(view, (0, h // 2), (w // 8, h // 2), CYAN, 2)
                    cv2.line(view, (w - w // 8, h // 2), (w, h // 2), CYAN, 2)
                    if args.middle:
                        put(view, f"MIDDLE  L{layer} P2 end-{end.upper()}   live {band.width_px:.0f}px", 24)
                        done = next((b.ends for b in scan.blocks if b.layer == layer), {})
                        put(view, "  ".join(f"{e.upper()}:{done[e]:.0f}" if e in done else f"{e.upper()}:--"
                                            for e in "ab"), 50)
                        put(view, f"SHOOT: P2 END-{end.upper()} TOP EDGE", 80, YELLOW, 0.8)
                    else:
                        put(view, f"SCAN  L{layer} P{position}   live {band.width_px:.0f}px", 24)
                        shots = "  ".join(
                            f"{s[0].upper()}:{widths[i]:.0f}" if i < len(widths) else f"{s[0].upper()}:--"
                            for i, s in enumerate(SEGMENTS)
                        )
                        put(view, shots, 50)
                        put(view, f"SHOOT: TOP {SEGMENTS[len(widths)].upper()} ({len(widths) + 1}/3)", 80, YELLOW, 0.8)
                    put(view, f"scanned {len(scan.blocks)} blocks   p: finish -> check", h - 12, YELLOW)
                else:
                    ranked = scan.ranked()
                    if idx < len(ranked):
                        b = ranked[idx]
                        color = GREEN if b.px >= EASY_TOP_PX else WHITE
                        put(view, f"CHECK rank {idx + 1}/{len(ranked)}", 24)
                        put(view, f"PUSH L{b.layer} P{b.position}  ({b.px:.0f}px)", 60, color, 1.0)
                        put(view, "e: easy  h: hard  u: skip  z: back", 92, YELLOW)
                    else:
                        put(view, "CHECK DONE", 40, GREEN, 1.0)
                    put(view, summary(scan), h - 12, YELLOW)
                cv2.imshow("jenga scan", view)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if phase == "scan":
                    if key in (ord("]"), ord("["), ord("1"), ord("2"), ord("3")):
                        # Moving to another block drops half-finished shots so they never mix.
                        widths.clear()
                    if key == ord("]"):
                        layer += 1
                    elif key == ord("[") and layer > 1:
                        layer -= 1
                    elif key in (ord("1"), ord("2"), ord("3")) and not args.middle:
                        position = key - ord("0")
                    elif args.middle and key in (ord("a"), ord("b")):
                        end = chr(key)
                    elif args.middle and key == ord(" "):
                        frames = [frame] + [cam.read() for _ in range(FRAMES_PER_SHOT - 1)]
                        px = float(np.median([measure_single(f).width_px for f in frames]))
                        name = f"L{layer:02d}_P2_end-{end}_{time.time() * 1000:.0f}.png"
                        cv2.imwrite(str(session / name), frame)
                        block = scan.store_end(layer, end, px)
                        print(f"L{layer} P2 end-{end}: {px:.1f}px  (score {block.px:.1f})")
                        layer += 2
                    elif key == ord("x"):
                        widths.clear()
                    elif key == ord("z") and widths:
                        widths.pop()
                    elif key == ord(" ") and len(widths) < len(SEGMENTS):
                        seg = SEGMENTS[len(widths)]
                        frames = [frame] + [cam.read() for _ in range(FRAMES_PER_SHOT - 1)]
                        px = float(np.median([measure_single(f).width_px for f in frames]))
                        widths.append(px)
                        name = f"L{layer:02d}_P{position}_top-{seg}_{time.time() * 1000:.0f}.png"
                        cv2.imwrite(str(session / name), frame)
                        print(f"L{layer} P{position} top-{seg}: {px:.1f}px")
                        if len(widths) == len(SEGMENTS):
                            block = Block(layer, position, list(widths))
                            scan.store(block)
                            print(f"stored L{layer} P{position}: {block.px:.1f}px  ({len(scan.blocks)} blocks)")
                            widths.clear()
                            layer += 2
                    elif key == ord("p") and scan.blocks:
                        phase, idx = "check", 0
                        print("ranking:")
                        for rank, b in enumerate(scan.ranked(), 1):
                            print(f"  {rank:2d}. L{b.layer} P{b.position}  {b.px:.1f}px")
                else:
                    ranked = scan.ranked()
                    if key in (ord("e"), ord("h"), ord("u")) and idx < len(ranked):
                        ranked[idx].actual = {ord("e"): "easy", ord("h"): "hard", ord("u"): "skip"}[key]
                        b = ranked[idx]
                        print(f"rank {idx + 1}: L{b.layer} P{b.position} {b.px:.1f}px -> {b.actual}")
                        scan.save()
                        idx += 1
                        if idx == len(ranked):
                            print(summary(scan))
                    elif key == ord("z") and idx > 0:
                        idx -= 1
        finally:
            set_led(True)
            scan.save()
    cv2.destroyAllWindows()
    print(f"saved {session / 'scan.csv'}")


if __name__ == "__main__":
    main()
