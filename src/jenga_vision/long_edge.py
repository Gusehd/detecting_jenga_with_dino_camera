"""Gap width along the long edges of an outer block, seen from the side.

The block's long face (75 x 15 mm) fills the middle of the frame, Dino LED off,
room light only. A gap at the top or bottom edge shows up as a dark band; the
wider the band, the looser the block.

Per edge, the band width is measured in vertical strips along the edge and the
median is taken. Widths are converted to mm using the block's face height
(15 mm) in pixels, so magnification does not have to match between shots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

BLOCK_THICKNESS_MM = 15.0
STRIPS = 16
WINDOW = 30  # rows around a detected edge searched for the dark band
# Close-up single edge (measure_single). Chosen on 6 blocks, 2026-10-20:
# top-edge core width easy 18-29 px, hard 4.5-9 px at Dino zoom 50, cap touching.
DARK_FRACTION = 0.25  # 'black' = darker than this fraction of the frame median
CORE_WINDOW = 40  # rows around the darkest row counted as part of the band
EASY_TOP_PX = 13.0


@dataclass
class EdgeBand:
    row: int  # center row of the dark band
    width_px: float  # median band width over strips
    strip_widths: list[float] = field(default_factory=list)


@dataclass
class LongEdgeResult:
    top: EdgeBand
    bottom: EdgeBand

    @property
    def face_height_px(self) -> float:
        return float(self.bottom.row - self.top.row)

    def mm(self, px: float) -> float:
        return px / max(self.face_height_px, 1.0) * BLOCK_THICKNESS_MM

    @property
    def top_mm(self) -> float:
        return self.mm(self.top.width_px)

    @property
    def bottom_mm(self) -> float:
        return self.mm(self.bottom.width_px)


def _smooth(p: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(p.astype(np.float32).reshape(-1, 1), (1, k), 0).ravel()


def _find_edges(p: np.ndarray) -> tuple[int, int]:
    """Strongest brightness change above and below the frame center."""
    grad = np.abs(np.gradient(_smooth(p, 9)))
    h, mid = len(p), len(p) // 2
    top = 10 + int(np.argmax(grad[10 : mid - 5]))
    bottom = mid + 5 + int(np.argmax(grad[mid + 5 : h - 10]))
    return top, bottom


def _band_width(p: np.ndarray, edge: int, face_level: float, outward: int) -> tuple[int, float]:
    """Width of the dark run around the darkest row near `edge`.

    `outward` is -1 for the top edge (neighbor above) and +1 for the bottom.
    The threshold sits halfway between the band minimum and the darker of the
    block face and the neighbor beyond the band.
    """
    h = len(p)
    lo, hi = max(edge - WINDOW, 0), min(edge + WINDOW + 1, h)
    i = lo + int(np.argmin(p[lo:hi]))
    if outward < 0:
        beyond = p[max(lo - 20, 0) : lo]
    else:
        beyond = p[hi : min(hi + 20, h)]
    neighbor = float(np.median(beyond)) if len(beyond) else face_level
    thr = p[i] + 0.5 * (min(face_level, neighbor) - p[i])
    a = i
    while a > 0 and p[a - 1] < thr:
        a -= 1
    b = i
    while b < h - 1 and p[b + 1] < thr:
        b += 1
    return (a + b) // 2, float(b - a + 1)


def measure(frame: np.ndarray) -> LongEdgeResult:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    x0, x1 = w // 10, w - w // 10
    top_e, bot_e = _find_edges(np.median(gray[:, x0:x1], axis=1))

    bounds = np.linspace(x0, x1, STRIPS + 1).astype(int)
    bands: dict[int, list[tuple[int, float]]] = {-1: [], 1: []}
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        p = _smooth(np.median(gray[:, s0:s1], axis=1))
        inner = p[top_e + WINDOW : bot_e - WINDOW]
        face_level = float(np.median(inner)) if len(inner) else float(np.median(p))
        bands[-1].append(_band_width(p, top_e, face_level, -1))
        bands[1].append(_band_width(p, bot_e, face_level, 1))

    def summarize(items: list[tuple[int, float]]) -> EdgeBand:
        rows, widths = zip(*items)
        return EdgeBand(int(np.median(rows)), float(np.median(widths)), list(widths))

    return LongEdgeResult(summarize(bands[-1]), summarize(bands[1]))


def measure_single(frame: np.ndarray) -> EdgeBand:
    """Close-up of one edge: width in px of the near-black core of the gap band.

    Dino LED off, room light only. A real gap gets no light and is near black;
    a closed edge only shows a grey shadow. So only rows darker than
    DARK_FRACTION of the frame median count, which also cancels auto exposure.
    With the Dino cap touching the block the working distance is fixed, so px
    widths are comparable as long as the magnification dial is not changed.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    x0, x1 = w // 10, w - w // 10
    dark = DARK_FRACTION * float(np.median(gray))
    p_all = _smooth(np.median(gray[:, x0:x1], axis=1), 9)
    lo, hi = h // 4, 3 * h // 4
    center = lo + int(np.argmin(p_all[lo:hi]))
    a, b = max(center - CORE_WINDOW, 0), min(center + CORE_WINDOW + 1, h)

    bounds = np.linspace(x0, x1, STRIPS + 1).astype(int)
    widths = []
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        p = _smooth(np.median(gray[:, s0:s1], axis=1))
        widths.append(float((p[a:b] < dark).sum()))
    dark_rows = np.flatnonzero(p_all[a:b] < dark)
    row = a + int(dark_rows.mean()) if len(dark_rows) else center
    return EdgeBand(row, float(np.median(widths)), widths)
