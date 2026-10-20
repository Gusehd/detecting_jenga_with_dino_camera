"""Labeled capture tool for Jenga block end faces.

Each save writes a PNG and appends a row to labels.csv in the session folder.
The right panel plots the mean brightness of each image row (center columns),
so horizontal gaps between layers show up as dips.

Keys:
  [ / ]    layer -1 / +1 (1 = bottom)
  1 2 3    position in layer (left to right, seen from end A)
  a / b    which end face is being viewed
  t        cycle view: face (whole end face) / long (side long face, gap mm shown live)
           / top edge / bottom edge
  e / h    label: easy (slides out) / hard (load-bearing)
  u        label: unknown (shoot first, label later in labels.csv)
  l        cycle light: front / off (room light) / back / both; Dino LED is off in 'off' and 'back'
  r        store reference frame (backlight off) and save it as *_ref.png
  d        toggle difference view: brightening vs the reference, amplified
  SPACE    save frame; in top/bottom view it then steps through the edge sequence
           top left -> center -> right -> bottom left -> center -> right -> next block
  z        step the edge sequence back one (to retake the previous shot)
  q / ESC  quit

Backlight check without turning the Dino LED off: keep the camera still,
press r with the phone off, then turn the phone on behind the block and press d.
Only light added by the phone shows up in the difference view.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from jenga_vision.camera import DinoCamera, sharpness
from jenga_vision.led import set_led
from jenga_vision.long_edge import measure, measure_single

LIGHTS = ("front", "off", "back", "both")
LED_ON_LIGHTS = ("front", "both")
VIEWS = ("face", "long", "top", "bottom")
SEGMENTS = ("left", "center", "right")  # along the long edge, close-up shots
FIELDS = [
    "file", "layer", "position", "end", "view", "segment", "label", "light", "sharpness",
    "top_mm", "bottom_mm", "edge_px", "timestamp",
]
PANEL_W = 200


def row_profile(gray: np.ndarray) -> np.ndarray:
    """Mean brightness per row over the central half of the columns."""
    w = gray.shape[1]
    return gray[:, w // 4 : 3 * w // 4].mean(axis=1)


def profile_panel(profile: np.ndarray) -> np.ndarray:
    h = len(profile)
    panel = np.full((h, PANEL_W, 3), 30, np.uint8)
    xs = (profile / 255.0 * (PANEL_W - 1)).astype(np.int32)
    pts = np.stack([xs, np.arange(h)], axis=1).reshape(-1, 1, 2)
    cv2.polylines(panel, [pts], False, (0, 255, 255), 1)
    return panel


def diff_view(frame: np.ndarray, ref: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """Brightening relative to ref, amplified and false-colored."""
    gain_img = cv2.subtract(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    )
    amplified = cv2.convertScaleAbs(gain_img, alpha=gain)
    return cv2.applyColorMap(amplified, cv2.COLORMAP_INFERNO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("captures"))
    args = parser.parse_args()

    session = args.out / time.strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    csv_path = session / "labels.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(FIELDS)

    layer, position, end, label, light, mode, seg = 1, 1, "a", "easy", 0, 0, 0
    count = 0
    ref = None
    show_diff = False

    def save(img: np.ndarray, focus: float, suffix: str = "") -> str:
        top_mm = bottom_mm = edge_px = ""
        if VIEWS[mode] == "long":
            r = measure(img)
            top_mm, bottom_mm = f"{r.top_mm:.3f}", f"{r.bottom_mm:.3f}"
        segment = ""
        if VIEWS[mode] in ("top", "bottom"):
            edge_px = f"{measure_single(img).width_px:.1f}"
            segment = SEGMENTS[seg]
        nonlocal count
        count += 1
        view_tag = f"{VIEWS[mode]}-{segment}" if segment else VIEWS[mode]
        name = f"L{layer:02d}_P{position}_{end}_{view_tag}_{label}_{LIGHTS[light]}_{count:04d}{suffix}.png"
        cv2.imwrite(str(session / name), img)
        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [name, layer, position, end, VIEWS[mode], segment, label, LIGHTS[light], f"{focus:.1f}",
                 top_mm, bottom_mm, edge_px, time.time()]
            )
        return name

    with DinoCamera(args.device) as cam:
        set_led(True)
        print(f"Session: {session}")
        while True:
            frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            focus = sharpness(frame)

            view = diff_view(frame, ref) if show_diff and ref is not None else frame.copy()
            color = {"easy": (0, 255, 0), "hard": (0, 0, 255)}.get(label, (0, 255, 255))
            ref_state = "DIFF" if show_diff and ref is not None else ("ref set" if ref is not None else "no ref")
            lines = [
                f"L{layer} P{position} end-{end.upper()} {VIEWS[mode].upper()}  {label.upper()}  light:{LIGHTS[light]}",
                f"focus {focus:6.1f}   saved {count}   {ref_state}",
            ]
            if VIEWS[mode] == "long":
                r = measure(frame)
                w = view.shape[1]
                for band in (r.top, r.bottom):
                    half = max(int(band.width_px // 2), 1)
                    cv2.rectangle(view, (w // 10, band.row - half), (w - w // 10, band.row + half), (255, 255, 0), 1)
                lines.append(f"gap top {r.top_mm:.2f}mm  bottom {r.bottom_mm:.2f}mm  (face {r.face_height_px:.0f}px)")
            elif VIEWS[mode] != "face":
                h, w = view.shape[:2]
                cv2.line(view, (0, h // 2), (w // 8, h // 2), (255, 255, 0), 2)
                cv2.line(view, (w - w // 8, h // 2), (w, h // 2), (255, 255, 0), 2)
                band = measure_single(frame)
                half = max(int(band.width_px // 2), 1)
                cv2.rectangle(view, (w // 10, band.row - half), (w - w // 10, band.row + half), (255, 255, 0), 1)
                lines.append(f"gap {band.width_px:.0f}px  (keep zoom dial fixed)")
                step = (mode - VIEWS.index("top")) * len(SEGMENTS) + seg + 1
                lines.append(f"SHOOT: {VIEWS[mode].upper()} {SEGMENTS[seg].upper()}  ({step}/6)")
            for i, text in enumerate(lines):
                cv2.putText(view, text, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("capture", np.hstack([view, profile_panel(row_profile(gray))]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("]"):
                layer += 1
            elif key == ord("[") and layer > 1:
                layer -= 1
            elif key in (ord("1"), ord("2"), ord("3")):
                position = key - ord("0")
            elif key in (ord("a"), ord("b")):
                end = chr(key)
            elif key == ord("e"):
                label = "easy"
            elif key == ord("h"):
                label = "hard"
            elif key == ord("u"):
                label = "unknown"
            elif key == ord("t"):
                mode = (mode + 1) % len(VIEWS)
                seg = 0
            elif key == ord("z") and VIEWS[mode] in ("top", "bottom"):
                if seg > 0:
                    seg -= 1
                elif VIEWS[mode] == "bottom":
                    mode, seg = VIEWS.index("top"), len(SEGMENTS) - 1
            elif key == ord("l"):
                light = (light + 1) % len(LIGHTS)
                set_led(LIGHTS[light] in LED_ON_LIGHTS)
            elif key == ord("d"):
                show_diff = not show_diff
            elif key == ord("r"):
                ref = np.mean([frame] + [cam.read() for _ in range(7)], axis=0).astype(np.uint8)
                print(f"reference {save(ref, focus, '_ref')}")
            elif key == ord(" "):
                print(f"saved {save(frame, focus)}")
                if VIEWS[mode] in ("top", "bottom"):
                    seg += 1
                    if seg == len(SEGMENTS):
                        seg = 0
                        if VIEWS[mode] == "top":
                            mode = VIEWS.index("bottom")
                        else:
                            mode = VIEWS.index("top")
                            print(f"L{layer} P{position} done -> set the next block")
        set_led(True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
