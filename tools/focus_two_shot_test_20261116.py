"""Offline check (2026-11-16): can two focus samples place the peak, instead of a search?

Near the top the focus curve is close to a Gaussian: log f(d) = log A - (d - d0)^2 / (2 s^2).
The scale A depends on what is in the image (45-96 on one layer), but the ratio of two samples
does not, so with the width s known
    d0 = (d1 + d2) / 2 + s^2 * (log f1 - log f2) / (d1 - d2)
Every search in captures/autoscan_*/focus_log.csv is one test: the peak of the full search
(log-parabola over the samples around the best, like focus_search._result) is the answer,
s is fit on the other sessions (leave one session out), and the prediction uses only the
first two samples the search took (start and the first step out).

    uv run python tools/focus_two_shot_test_20261116.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

GAP_S = 3.0  # a pause longer than this starts a new search
FLOOR = 12.0  # focus_search.MIN_PEAK: below this the value is noise


def searches() -> list[dict]:
    out = []
    for log in sorted(Path("captures").glob("autoscan_*/focus_log.csv")):
        rows = list(csv.DictReader(log.open()))
        cur: list[dict] = []
        for r in rows:
            if cur and (float(r["time"]) - float(cur[-1]["time"]) > GAP_S or r["down_mm"] != cur[-1]["down_mm"]
                        or r["layer"] != cur[-1]["layer"] or r["roll_deg"] != cur[-1]["roll_deg"]):
                out.append({"session": log.parent.name, "rows": cur})
                cur = []
            cur.append(r)
        if cur:
            out.append({"session": log.parent.name, "rows": cur})
    for s in out:
        s["d"] = np.array([float(r["depth_mm"]) for r in s["rows"]])
        s["f"] = np.array([float(r["focus"]) for r in s["rows"]])
    return out


def peak_fit(d: np.ndarray, f: np.ndarray) -> tuple[float, float] | None:
    """(peak depth, s) from the samples within 2 steps of the best one; None if not a clear top."""
    b = int(np.argmax(f))
    if f[b] < FLOOR:
        return None
    near = np.abs(d - d[b]) <= 2.0 * np.min(np.diff(np.unique(d))) + 1e-9 if len(np.unique(d)) > 1 else None
    if near is None or near.sum() < 3:
        return None
    a, c, _ = np.polyfit(d[near], np.log(f[near]), 2)
    if a >= 0:
        return None
    d0 = -c / (2 * a)
    if not (d[near].min() <= d0 <= d[near].max()):  # top not bracketed
        return None
    return float(d0), float(np.sqrt(-1 / (2 * a)))


def main() -> None:
    ss = searches()
    fits = []
    for s in ss:
        p = peak_fit(s["d"], s["f"])
        if p is not None and len(s["d"]) >= 3:
            fits.append({**s, "d0": p[0], "s": p[1]})
    print(f"{len(ss)} searches, {len(fits)} with a clear, bracketed top; "
          f"samples per search median {np.median([len(s['d']) for s in ss]):.0f}")
    widths = np.array([x["s"] for x in fits])
    print(f"curve width s: median {np.median(widths):.2f} mm, p10 {np.percentile(widths, 10):.2f}, "
          f"p90 {np.percentile(widths, 90):.2f}")

    err, err_far, used = [], [], 0
    for x in fits:
        others = [y["s"] for y in fits if y["session"] != x["session"]]
        s = float(np.median(others))
        d1, d2, f1, f2 = x["d"][0], x["d"][1], x["f"][0], x["f"][1]
        if min(f1, f2) < FLOOR / 3 or d1 == d2:
            continue
        used += 1
        pred = (d1 + d2) / 2 + s * s * (np.log(f1) - np.log(f2)) / (d1 - d2)
        e = pred - x["d0"]
        err.append(e)
        if abs(d1 - x["d0"]) > 1.5:
            err_far.append(e)
    a = np.abs(err)
    print(f"\ntwo samples (start + first step), s from the other sessions: {used} searches")
    print(f"  |error| median {np.median(a):.2f} mm, p90 {np.percentile(a, 90):.2f}, max {a.max():.2f}, "
          f"within 0.5 mm {np.mean(a <= 0.5) * 100:.0f}%, within 1.0 mm {np.mean(a <= 1.0) * 100:.0f}%")
    if err_far:
        b = np.abs(err_far)
        print(f"  start more than 1.5 mm from the peak ({len(b)}): median {np.median(b):.2f}, p90 "
              f"{np.percentile(b, 90):.2f}, within 1.0 mm {np.mean(b <= 1.0) * 100:.0f}%")
    # three samples (the first three the search took), d0 and A fit with s fixed:
    # log f = log A - (d - d0)^2 / (2 s^2)  ->  linear least squares in (log A - d0^2/2s^2, d0/s^2)
    e3, e3_far, n3 = [], [], 0
    for x in fits:
        if len(x["d"]) < 4:  # the full search must have more than the three used here
            continue
        s = float(np.median([y["s"] for y in fits if y["session"] != x["session"]]))
        d, f = x["d"][:3], x["f"][:3]
        if f.min() < FLOOR / 3:
            continue
        n3 += 1
        y = np.log(f) + d * d / (2 * s * s)
        k, _ = np.polyfit(d, y, 1)
        e = k * s * s - x["d0"]
        e3.append(e)
        if abs(d[0] - x["d0"]) > 1.5:
            e3_far.append(e)
    a3 = np.abs(e3)
    print(f"\nthree samples, s fixed: {n3} searches that took 4 or more")
    print(f"  |error| median {np.median(a3):.2f} mm, p90 {np.percentile(a3, 90):.2f}, max {a3.max():.2f}, "
          f"within 0.5 mm {np.mean(a3 <= 0.5) * 100:.0f}%, within 1.0 mm {np.mean(a3 <= 1.0) * 100:.0f}%")
    if e3_far:
        b = np.abs(e3_far)
        print(f"  start more than 1.5 mm from the peak ({len(b)}): median {np.median(b):.2f}, p90 "
              f"{np.percentile(b, 90):.2f}, within 1.0 mm {np.mean(b <= 1.0) * 100:.0f}%")
    print(f"  samples the full search took on these: median {np.median([len(x['d']) for x in fits if len(x['d']) >= 4]):.0f}")
    start_off = np.abs([x["d"][0] - x["d0"] for x in fits])
    print(f"  (for scale: the start itself was off by median {np.median(start_off):.2f} mm, "
          f"p90 {np.percentile(start_off, 90):.2f})")


if __name__ == "__main__":
    main()
