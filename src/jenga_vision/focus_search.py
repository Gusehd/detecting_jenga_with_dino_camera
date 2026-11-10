"""Find the sharpest depth along the optical axis, without a robot or a camera.

Depths are in mm, + = towards the block. A single focus value says nothing, and its
size depends on what is in the image (45-96 on one layer, 2026-11-02), so a peak is
only called when the values clearly fall on both sides of it. Going out is always
safe, so the search looks outwards first and goes in only if nothing better is there:

    arrived on the peak     out 3 steps, back, in 3 steps: both sides fall, done
    peak further out        out until it falls behind the peak; never goes in
    peak further in         out until the values have fallen OUT_DROPS times in a row (the
                            outside only gets worse), back, in until it falls

Falling twice in a row going out cannot happen when the start is deeper than the peak (the
values rise going out), also not with a dip of 0.8 in one sample, so it never leads inwards
from too deep. Before 2026-11-10 it walked out to the limit: 5 samples (5.5 s) per search.

Real curve at dial 30: half value about 4 mm from the peak on both sides, 4-7 % down
at 1 mm, 15-25 % at 2 mm, nearly flat from 15 mm on, floor about 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

STEP_MM = 1.0
FALL = 0.88  # a sample below this x the best one has clearly fallen (hold test: +- 3 % when settled)
FALLS = 2  # that many samples in a row
RISEN = 0.75  # the start below this x the best one: truly lower, not just one low sample
MIN_PEAK = 12.0  # floor about 4: lower than this is not a peak, wherever it stands
OUT_DROP = 0.88  # going out, each sample below this x the one before ... (0.85 missed 1.85 -> 1.61 on the robot)
OUT_DROPS = 2  # ... that many times in a row: nothing better further out


@dataclass(frozen=True)
class Peak:
    found: bool
    depth: float  # mm; best sample if not found
    value: float
    samples: int
    reason: str = ""


def find_peak(measure: Callable[[float], float], start: float, out_limit: float, in_limit: float,
              step: float = STEP_MM, fall: float = FALL, falls: int = FALLS, min_peak: float = MIN_PEAK) -> Peak:
    """`measure(depth)` moves there and returns the focus value. Stays within
    out_limit <= depth <= in_limit, and ends wherever the last sample was."""
    seen: dict[float, float] = {}

    def best() -> float:
        return max(seen, key=seen.get)

    def fallen(sign: int) -> bool:
        """`falls` samples in a row beyond the best one, on this side, are clearly lower.
        Never on the flat floor, where it would only be noise."""
        b = best()
        if seen[b] < min_peak:
            return False
        side = sorted((d for d in seen if (d - b) * sign > 0), key=lambda d: abs(d - b))
        run = 0
        for d in side:
            run = run + 1 if seen[d] < fall * seen[b] else 0
            if run >= falls:
                return True
        return False

    def only_worse_out() -> bool:
        out = sorted((d for d in seen if d <= start), reverse=True)  # start, then outwards
        drops = [seen[b] < OUT_DROP * seen[a] for a, b in zip(out, out[1:])]
        return len(drops) >= OUT_DROPS and all(drops[-OUT_DROPS:])

    def walk(sign: int, limit: float) -> None:
        d = start
        while not fallen(sign) and not (sign < 0 and only_worse_out()):
            d += sign * step
            if (d - limit) * sign > 1e-9:
                return
            seen[d] = measure(d)

    seen[start] = measure(start)
    walk(-1, out_limit)
    if not fallen(-1) and seen[start] < RISEN * seen[best()]:  # clearly higher out there: do not go in
        return _result(seen, False, f"still rising at the outer limit {out_limit:+.1f} mm")
    walk(+1, in_limit)
    if not fallen(+1):
        return _result(seen, False, f"no fall before the inner limit {in_limit:+.1f} mm")
    if not fallen(-1):
        return _result(seen, False, f"no fall before the outer limit {out_limit:+.1f} mm")
    return _result(seen, True, "", step)


def _result(seen: dict[float, float], found: bool, reason: str, step: float = 0.0) -> Peak:
    b = max(seen, key=seen.get)
    depth = b
    if found:  # the top is flat: the samples around it place it better than the highest one
        near = sorted(d for d in seen if abs(d - b) <= 2 * step + 1e-9)
        if len(near) >= 3:
            a, c, _ = np.polyfit(near, np.log([seen[d] for d in near]), 2)
            if a < 0:
                depth = float(np.clip(-c / (2 * a), b - step, b + step))
    return Peak(found, float(depth), seen[b], len(seen), reason)
