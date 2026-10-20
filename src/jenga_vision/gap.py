"""Gap score for a Jenga block end face seen head-on by the Dino-Lite.

The end face is assumed to fill the middle of the frame, with the layer above
and below visible at the top and bottom. A loose (easy) block shows a thin,
very dark line at its top or bottom boundary; a load-bearing (hard) block shows
only the soft shadow of the rounded edge.

Score per boundary = depth of the narrowest dark valley near that boundary,
relative to its surroundings (0 = none, 1 = black). Frame score = max of the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Chosen on the first 10 labeled blocks (easy 0.31-0.88, hard 0.15-0.34).
EASY_THRESHOLD = 0.4
SEARCH = 15  # rows around a detected edge to look for the gap line
VALLEY_KERNEL = 21  # rows; dark lines narrower than this count as gaps


@dataclass
class GapResult:
    top_row: int
    bottom_row: int
    top_score: float
    bottom_score: float

    @property
    def score(self) -> float:
        return max(self.top_score, self.bottom_score)


def row_profile(gray: np.ndarray) -> np.ndarray:
    """Median brightness per row over the central third of the columns."""
    w = gray.shape[1]
    return np.median(gray[:, w // 3 : 2 * w // 3], axis=1).astype(np.float32)


def analyze(frame: np.ndarray) -> GapResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    p = row_profile(gray)
    h = len(p)

    smooth = cv2.GaussianBlur(p.reshape(-1, 1), (1, 9), 0).ravel()
    grad = np.abs(np.gradient(smooth))
    mid = h // 2
    top = 10 + int(np.argmax(grad[10 : mid - 5]))
    bottom = mid + 5 + int(np.argmax(grad[mid + 5 : h - 10]))

    kernel = np.ones((VALLEY_KERNEL, 1), np.uint8)
    closed = cv2.morphologyEx(p.reshape(-1, 1), cv2.MORPH_CLOSE, kernel).ravel()
    valley = (closed - p) / np.maximum(closed, 1)

    scores = []
    for edge in (top, bottom):
        lo, hi = max(edge - SEARCH, 0), min(edge + SEARCH + 1, h)
        i = lo + int(np.argmax(valley[lo:hi]))
        scores.append((i, float(valley[i])))
    (top_row, top_score), (bottom_row, bottom_score) = scores
    return GapResult(top_row, bottom_row, top_score, bottom_score)


def is_easy(score: float) -> bool:
    return score >= EASY_THRESHOLD
