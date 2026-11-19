"""Live Dino-Lite viewer for checking framing, focus and lighting.

Keys: s = save frame, g = toggle grid, o = LED on/off, e = end-face overlay,
q / ESC = quit

The end-face overlay (also --ends) shows what jenga-ends measures: the cyan
center line to put the block's end-face center on, the measured span and the
gap band with its width in px.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from jenga_vision.camera import DinoCamera, focus_value
from jenga_vision.ends import CYAN, YELLOW, measure_end
from jenga_vision.led import set_led
from jenga_vision.paths import CAPTURES_DIR, CONFIG_DIR


def edge_tilt_deg(frame) -> float | None:
    """Tilt of the strongest horizontal edge in the middle half of the rows (deg, + = lower on the right), from
    8 narrow columns; None if the columns disagree. Compared with dino.calibration.<arm>.image_tilt_deg it shows
    whether the Dino went back into the hand jig turned as at the calibration."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    xs, rows = [], []
    for x0 in np.linspace(0.1 * w, 0.85 * w, 8).astype(int):
        p = cv2.GaussianBlur(np.median(gray[:, x0:x0 + w // 20], axis=1).reshape(-1, 1), (1, 9), 0).ravel()
        d = np.abs(np.diff(p))[h // 4: 3 * h // 4]
        if d.max() < 3:
            continue
        xs.append(x0 + w // 40)
        rows.append(h // 4 + int(np.argmax(d)))
    if len(xs) < 5:
        return None
    slope, cut = np.polyfit(xs, rows, 1)
    if np.median(np.abs(np.polyval([slope, cut], xs) - rows)) > 4:
        return None
    return float(np.degrees(np.arctan(slope)))


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
    parser.add_argument("--out", type=Path, default=CAPTURES_DIR)
    parser.add_argument("--ends", action="store_true", help="start with the end-face overlay on")
    args = parser.parse_args()

    show_grid = False
    show_ends = args.ends
    led_on = False
    prev = time.time()
    try:
        cal = json.loads(CONFIG_DIR / "scan_setup.json".read_text())["dino"]["calibration"]["right"]
        ref_tilt = cal.get("image_tilt_deg")
    except (OSError, KeyError, ValueError):
        ref_tilt = None
    with DinoCamera(args.device) as cam:
        if cam.upside_down:
            print("image turned 180 deg (dino.image_upside_down in config/scan_setup.json)")
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
            tilt = edge_tilt_deg(frame)
            text = "edge tilt -" if tilt is None else f"edge tilt {tilt:+.1f} deg"
            if ref_tilt is not None:
                text += f"  (calibrated {ref_tilt:+.1f})"
            cv2.putText(view, text, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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
