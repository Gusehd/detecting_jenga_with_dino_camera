"""Dino-Lite (AM3111) capture via OpenCV/V4L2."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class DinoCamera:
    def __init__(self, device: int = 0, width: int = 640, height: int = 480):
        self.device = device
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
        return frame

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
        super().__init__(device, width, height)
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
