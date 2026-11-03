"""Live Dino-Lite viewer for checking framing, focus and lighting.

Keys: s = save frame, g = toggle grid, o = LED on/off, e = end-face overlay,
q / ESC = quit

The end-face overlay (also --ends) shows what jenga-ends measures: the cyan
center line to put the block's end-face center on, the measured span and the
gap band with its width in px.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from jenga_vision.camera import DinoCamera, focus_value
from jenga_vision.ends import CYAN, YELLOW, measure_end
from jenga_vision.led import set_led


def draw_grid(frame, step: int = 40) -> None:
    h, w = frame.shape[:2]
    for x in range(0, w, step):
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
    for y in range(0, h, step):
        cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)


def draw_ends(view, frame) -> None:
    h, w = view.shape[:2]
    cv2.line(view, (w // 2, 0), (w // 2, h), CYAN, 1)
    g = measure_end(frame)
    half = max(g.gap / 2, 1)
    cv2.rectangle(view, (g.x0, int(g.row - half)), (g.x1, int(g.row + half)), YELLOW, 1)
    cv2.putText(view, f"gap {g.gap:.1f}px", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("captures"))
    parser.add_argument("--ends", action="store_true", help="start with the end-face overlay on")
    args = parser.parse_args()

    show_grid = False
    show_ends = args.ends
    led_on = False
    prev = time.time()
    with DinoCamera(args.device) as cam:
        w, h = cam.resolution
        print(f"Dino-Lite /dev/video{args.device} {w}x{h}")
        while True:
            frame = cam.read()
            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            view = frame.copy()
            if show_grid:
                draw_grid(view)
            if show_ends:
                draw_ends(view, frame)
            info = f"{w}x{h}  {fps:4.1f}fps  focus {focus_value(frame):6.1f}  LED {'ON' if led_on else 'OFF'}"
            cv2.putText(view, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("dino", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                led_on = not led_on
                set_led(led_on)
                print(f"LED {'on' if led_on else 'off'}")
            if key == ord("g"):
                show_grid = not show_grid
            if key == ord("e"):
                show_ends = not show_ends
            if key == ord("s"):
                args.out.mkdir(parents=True, exist_ok=True)
                path = args.out / f"{time.strftime('%Y%m%d_%H%M%S')}_{int(now * 1000) % 1000:03d}.png"
                cv2.imwrite(str(path), frame)
                print(f"saved {path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
