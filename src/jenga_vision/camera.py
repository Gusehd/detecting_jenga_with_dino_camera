"""Dino-Lite (AM3111) capture via OpenCV/V4L2."""

from __future__ import annotations

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


def sharpness(frame: np.ndarray) -> float:
    """Variance of Laplacian: higher means better focus."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
