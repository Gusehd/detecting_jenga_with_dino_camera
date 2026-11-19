"""Dino-Lite (AM3111) capture via OpenCV/V4L2."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from jenga_vision.led import set_led
from jenga_vision.paths import CONFIG_DIR

SETUP_FILE = CONFIG_DIR / "scan_setup.json"  # the same file as scan_plan.SETUP_FILE


def mounted_upside_down() -> bool:
    """dino.image_upside_down in SETUP_FILE: the Dino sits turned 180 deg about its axis (hand jig, 2026-11-18:
    a finger moved down in front of the lens went up on screen). Every frame is then turned back, so image up =
    world up again for everything that reads the camera (scan, jenga-view, dashboard)."""
    try:
        return bool(json.loads(SETUP_FILE.read_text())["dino"].get("image_upside_down", False))
    except (OSError, KeyError, ValueError):
        return False


class DinoCamera:
    """`led` is the Dino LED state set on open (off by default); None leaves it alone. `upside_down` turns
    every frame 180 deg (None: from SETUP_FILE, see mounted_upside_down)."""

    def __init__(self, device: int = 0, width: int = 640, height: int = 480,
                 led: bool | None = False, upside_down: bool | None = None):
        self.device = device
        self.upside_down = mounted_upside_down() if upside_down is None else upside_down
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if led is not None:
            # The LED command is ignored until the camera is streaming.
            for _ in range(5):
                self.read()
            try:
                set_led(led)
            except RuntimeError as e:
                print(f"warning: LED not set ({e})")

    @property
    def resolution(self) -> tuple[int, int]:
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Frame read failed")
        return cv2.rotate(frame, cv2.ROTATE_180) if self.upside_down else frame

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> DinoCamera:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class GeminiCamera(DinoCamera):
    """Orbbec Gemini 305 color stream (UVC), center crop as a digital zoom.

    The Gemini sees ~13 cm across at 7 cm, so a block edge is tiny in the full
    frame. read() returns the central `roi` fraction of the frame, upscaled
    `zoom` times with nearest neighbor (no smoothing), so the Dino tools can
    run unchanged. px values are then `zoom` x native pixels.
    """

    def __init__(self, device: int = 0, width: int = 1280, height: int = 800,
                 roi: float = 0.25, zoom: int = 2):
        super().__init__(device, width, height, led=None, upside_down=False)
        self.roi, self.zoom = roi, zoom

    def read(self) -> np.ndarray:
        frame = super().read()
        h, w = frame.shape[:2]
        ch, cw = int(h * self.roi), int(w * self.roi)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        crop = frame[y0 : y0 + ch, x0 : x0 + cw]
        return cv2.resize(crop, (cw * self.zoom, ch * self.zoom), interpolation=cv2.INTER_NEAREST)


def find_gemini_color() -> int:
    """/dev/video index of the Gemini 305 left RGB stream (numbers change on replug)."""
    for node in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        iface = node / "device" / "interface"
        index = node / "index"
        if (iface.exists() and "RGB Camera Left" in iface.read_text()
                and index.exists() and index.read_text().strip() == "0"):
            return int(node.name[5:])
    raise RuntimeError("Gemini 305 RGB stream not found (is it plugged in?)")


def sharpness(frame: np.ndarray) -> float:
    """Variance of Laplacian: higher means better focus."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def focus_value(frame: np.ndarray) -> float:
    """Band-pass sharpness (variance of a difference of Gaussians, sigma 1 and 3 px).
    The plain variance of Laplacian (jenga-view) is dominated by sensor noise: on the
    2026-10-30 shots it was 741-1619 in focus and 293-572 blurred; this is 44-135 vs 2-4."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.var(cv2.GaussianBlur(gray, (0, 0), 1.0) - cv2.GaussianBlur(gray, (0, 0), 3.0)))
