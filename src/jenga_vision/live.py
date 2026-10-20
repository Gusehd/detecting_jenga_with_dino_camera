"""Live blind test: predict easy/hard from both end faces, then record the truth.

Workflow per block:
  1. set layer / position
  2. aim at end A, press a  (grabs a burst, scores it, saves frames)
  3. aim at end B, press b
  4. push the block, press e (easy) or h (hard) -> logged to results.csv

Keys:
  [ / ]    layer -1 / +1 (1 = bottom)
  1 2 3    position in layer
  a / b    capture end face A / B (press again to retake)
  e / h    actual result: easy / hard (logs the block and starts the next one)
  x        discard captures of the current block
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
from jenga_vision.gap import EASY_THRESHOLD, analyze, is_easy

BURST = 5
FIELDS = ["layer", "position", "score_a", "score_b", "score", "pred", "actual", "correct", "timestamp"]
GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)


def verdict(score: float) -> tuple[str, tuple[int, int, int]]:
    return ("EASY", GREEN) if is_easy(score) else ("HARD", RED)


def put(img, text, y, color=WHITE, scale=0.6):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("captures"))
    args = parser.parse_args()

    session = args.out / time.strftime("live_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    csv_path = session / "results.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(FIELDS)

    layer, position = 1, 1
    faces: dict[str, float] = {}
    done = correct = 0
    with DinoCamera(args.device) as cam:
        print(f"Session: {session}")
        while True:
            frame = cam.read()
            res = analyze(frame)
            h, w = frame.shape[:2]

            view = frame.copy()
            for row, s in ((res.top_row, res.top_score), (res.bottom_row, res.bottom_score)):
                color = GREEN if is_easy(s) else RED
                cv2.line(view, (w // 3, row), (2 * w // 3, row), color, 1)
                cv2.putText(view, f"{s:.2f}", (2 * w // 3 + 6, row + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            label, color = verdict(res.score)
            put(view, f"L{layer} P{position}   live {res.score:.2f} {label}", 24, color)
            face_txt = "  ".join(f"{e.upper()}:{faces[e]:.2f}" if e in faces else f"{e.upper()}:--" for e in "ab")
            put(view, face_txt, 50)
            if faces:
                block = max(faces.values())
                label, color = verdict(block)
                partial = "" if len(faces) == 2 else " (one face)"
                put(view, f"PREDICT {label} {block:.2f}{partial}", 80, color, 0.8)
            acc = f"{correct}/{done} correct" if done else "no results yet"
            put(view, f"{acc}   thr {EASY_THRESHOLD}", h - 12, YELLOW)
            cv2.imshow("jenga live", view)

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
                faces.clear()
            elif key in (ord("a"), ord("b")):
                end = chr(key)
                scores = []
                for i in range(BURST):
                    f = frame if i == 0 else cam.read()
                    scores.append(analyze(f).score)
                    cv2.imwrite(str(session / f"L{layer:02d}_P{position}_{end}_{i}.png"), f)
                faces[end] = float(np.median(scores))
                print(f"L{layer} P{position} {end.upper()}: {faces[end]:.2f}  (burst {np.round(scores, 2).tolist()})")
            elif key in (ord("e"), ord("h")) and faces:
                actual = "easy" if key == ord("e") else "hard"
                score = max(faces.values())
                pred = "easy" if is_easy(score) else "hard"
                done += 1
                correct += pred == actual
                with csv_path.open("a", newline="") as f:
                    csv.writer(f).writerow([
                        layer, position,
                        f"{faces['a']:.3f}" if "a" in faces else "",
                        f"{faces['b']:.3f}" if "b" in faces else "",
                        f"{score:.3f}", pred, actual, int(pred == actual), time.time(),
                    ])
                print(f"L{layer} P{position}: pred {pred} actual {actual} -> {'OK' if pred == actual else 'MISS'}  ({correct}/{done})")
                faces.clear()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
