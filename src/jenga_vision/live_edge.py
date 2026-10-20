"""Live blind test on the long face: top-edge gap, Dino LED off.

Workflow per block (outer block, long face, zoom fixed, cap touching):
  1. set layer / position
  2. top edge LEFT, SPACE -> CENTER, SPACE -> RIGHT, SPACE
  3. read the prediction, push the block, press e (easy) or h (hard)

Keys:
  [ / ]    layer -1 / +1 (1 = bottom)
  1 2 3    position in layer
  SPACE    measure + save the current segment
  z        retake the previous segment
  x        discard the current block
  e / h    actual result: easy / hard (logs the block, starts the next one)
  q / ESC  quit
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from jenga_vision.camera import DinoCamera
from jenga_vision.led import set_led
from jenga_vision.long_edge import EASY_TOP_PX, measure_single

SEGMENTS = ("left", "center", "right")
FRAMES_PER_SHOT = 3
FIELDS = ["layer", "position", "px_left", "px_center", "px_right", "px", "pred", "actual", "correct", "timestamp"]
GREEN, RED, YELLOW, WHITE, CYAN = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (255, 255, 0)


def put(img, text, y, color=WHITE, scale=0.6):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def predict(px: float) -> str:
    return "easy" if px >= EASY_TOP_PX else "hard"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("captures"))
    args = parser.parse_args()

    session = args.out / time.strftime("live_edge_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    csv_path = session / "results.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(FIELDS)

    layer, position = 1, 3
    widths: list[float] = []
    done = correct = 0
    with DinoCamera(args.device) as cam:
        # The LED command is ignored until the camera is streaming.
        for _ in range(5):
            cam.read()
        set_led(False)
        print(f"Session: {session}")
        try:
            while True:
                frame = cam.read()
                band = measure_single(frame)
                h, w = frame.shape[:2]

                view = frame.copy()
                half = max(int(band.width_px // 2), 1)
                live_color = GREEN if band.width_px >= EASY_TOP_PX else RED
                cv2.rectangle(view, (w // 10, band.row - half), (w - w // 10, band.row + half), CYAN, 1)
                cv2.line(view, (0, h // 2), (w // 8, h // 2), CYAN, 2)
                cv2.line(view, (w - w // 8, h // 2), (w, h // 2), CYAN, 2)

                put(view, f"L{layer} P{position}   live {band.width_px:.0f}px", 24, live_color)
                shots = "  ".join(
                    f"{s[0].upper()}:{widths[i]:.0f}" if i < len(widths) else f"{s[0].upper()}:--"
                    for i, s in enumerate(SEGMENTS)
                )
                put(view, shots, 50)
                if len(widths) < len(SEGMENTS):
                    put(view, f"SHOOT: TOP {SEGMENTS[len(widths)].upper()} ({len(widths) + 1}/3)", 80, YELLOW, 0.8)
                else:
                    px = float(np.median(widths))
                    pred = predict(px)
                    put(view, f"PREDICT {pred.upper()} {px:.0f}px -> push, then e/h", 80,
                        GREEN if pred == "easy" else RED, 0.8)
                acc = f"{correct}/{done} correct" if done else "no results yet"
                put(view, f"{acc}   thr {EASY_TOP_PX:.0f}px   LED off", h - 12, YELLOW)
                cv2.imshow("jenga live edge", view)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("]"):
                    layer += 1
                elif key == ord("[") and layer > 1:
                    layer -= 1
                elif key in (ord("1"), ord("2"), ord("3")):
                    position = key - ord("0")
                elif key == ord("x"):
                    widths.clear()
                elif key == ord("z") and widths:
                    widths.pop()
                elif key == ord(" ") and len(widths) < len(SEGMENTS):
                    seg = SEGMENTS[len(widths)]
                    frames = [frame] + [cam.read() for _ in range(FRAMES_PER_SHOT - 1)]
                    px = float(np.median([measure_single(f).width_px for f in frames]))
                    widths.append(px)
                    cv2.imwrite(str(session / f"L{layer:02d}_P{position}_top-{seg}_{time.time() * 1000:.0f}.png"), frame)
                    print(f"L{layer} P{position} top-{seg}: {px:.1f}px")
                elif key in (ord("e"), ord("h")) and len(widths) == len(SEGMENTS):
                    actual = "easy" if key == ord("e") else "hard"
                    px = float(np.median(widths))
                    pred = predict(px)
                    done += 1
                    correct += pred == actual
                    with csv_path.open("a", newline="") as f:
                        csv.writer(f).writerow(
                            [layer, position, *(f"{v:.1f}" for v in widths), f"{px:.1f}",
                             pred, actual, int(pred == actual), time.time()]
                        )
                    print(f"L{layer} P{position}: {px:.1f}px pred {pred} actual {actual} -> "
                          f"{'OK' if pred == actual else 'MISS'}  ({correct}/{done})")
                    widths.clear()
        finally:
            set_led(True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
