"""Offline check (2026-11-16): can the Dino roll be handled in software instead of turning the arm?

The robot shots are taken after `level` turned the Dino until the edge is level. Here each saved
shot is turned by a known angle (a Dino left unleveled), then measured three ways:
  raw      measure_end on the turned image (no leveling at all)
  exact    turned back by exactly the same angle (only the cost of resampling twice)
  soft     turned back by the angle edge_tilt reads from the turned image (the real proposal)
and compared with the saved shot. A second test shifts the image up / down (the edge left off the
image centre) and measures without centring.

Per layer, the 빼안빼 call (layer_margin) is compared too: a flip there changes the pick order.

    uv run python tools/soft_level_test_20261116.py [captures/autoscan_* ...]
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from jenga_vision import ends
from jenga_vision.autoscan import edge_tilt, layer_margin

ANGLES = (-10, -5, -3, -2, -1, 1, 2, 3, 5, 10)
SHIFTS = (-100, -60, -30, 30, 60, 100)


def turn(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def shift(img: np.ndarray, dy: int) -> np.ndarray:
    m = np.float32([[1, 0, 0], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def soft_level(img: np.ndarray) -> np.ndarray | None:
    slope = edge_tilt(img)
    if slope is None:
        return None
    return turn(img, math.degrees(math.atan(slope)))


def gap(img: np.ndarray) -> float:
    return ends.measure_end(img).gap


# 2026-11-10 14:05: the Dino sat 13 deg turned and the saved shots are not level (no leveling ran),
# so they can't serve as the leveled reference here
NOT_LEVEL = {"autoscan_20261110_140528", "autoscan_20261110_140907"}


def sessions_from_args() -> list[Path]:
    if len(sys.argv) > 1:
        return [Path(p) for p in sys.argv[1:]]
    return [p for p in _all_sessions() if p.name not in NOT_LEVEL]


def _all_sessions() -> list[Path]:
    return sorted(p for p in Path("captures").glob("autoscan_2026111*") if (p / "shots.csv").exists()) + \
        sorted(p for p in Path("captures").glob("autoscan_202611[0][3-9]*") if (p / "shots.csv").exists())


def stats(d: list[float]) -> str:
    a = np.abs(np.array(d))
    return (f"n {len(a):4d}  median {np.median(a):4.2f}  p90 {np.percentile(a, 90):4.2f}  "
            f"max {a.max():5.2f}  >1px {np.mean(a > 1) * 100:4.1f}%")


def main() -> None:
    shots = []  # (session, layer, slot, image)
    for s in sessions_from_args():
        with (s / "shots.csv").open() as f:
            for r in csv.DictReader(f):
                p = s / r["image"]
                if p.exists():
                    shots.append((s.name, int(r["layer"]), int(r["slot"]), p))
    print(f"{len(shots)} shots from {len({x[0] for x in shots})} sessions")

    base, tilt0 = {}, []
    imgs = {}
    for key in shots:
        img = cv2.imread(str(key[3]))
        imgs[key] = img
        base[key] = gap(img)
        t = edge_tilt(img)
        if t is not None:
            tilt0.append(math.degrees(math.atan(t)))
    print(f"saved shots: residual tilt |deg| median {np.median(np.abs(tilt0)):.2f}, max {np.max(np.abs(tilt0)):.2f}")

    def layer_calls(px: dict) -> dict:
        out = {}
        for sess in {k[0] for k in px}:
            sub = {(k[1], k[2]): v for k, v in px.items() if k[0] == sess}
            for layer in {k[0] for k in sub}:
                m = layer_margin(sub, layer)
                if m is not None:
                    out[(sess, layer)] = m
        return out

    calls0 = layer_calls(base)

    def flips(px: dict) -> tuple[int, int, int]:
        """(layers whose first block changed, of them with |margin| > 1 px at the saved shot, layers)."""
        c = layer_calls(px)
        changed = [k for k in calls0 if k in c and c[k][1] != calls0[k][1]]
        return len(changed), sum(abs(calls0[k][0]) > 1 for k in changed), len(calls0)

    rows = []
    print("\n== turned by a known angle (deg): |gap - saved gap| px")
    for deg in ANGLES:
        res = {"raw": {}, "exact": {}, "soft": {}, "face": {}}
        miss = 0
        turned = {key: turn(img, deg) for key, img in imgs.items()}
        per_face: dict[str, list[float]] = {}
        for key, t in turned.items():
            res["raw"][key] = gap(t)
            res["exact"][key] = gap(turn(t, -deg))
            slope = edge_tilt(t)
            if slope is None:
                miss += 1
                res["soft"][key] = res["raw"][key]
            else:
                a = math.degrees(math.atan(slope))
                per_face.setdefault(key[0], []).append(a)
                res["soft"][key] = gap(turn(t, a))
        # the roll is set by the grip, the same for a whole face: one angle per session (median)
        face_deg = {s: float(np.median(v)) for s, v in per_face.items()}
        for key, t in turned.items():
            res["face"][key] = gap(turn(t, face_deg.get(key[0], 0.0)))
        err = [face_deg[s] + deg for s in face_deg]
        print(f"      face angle read: |error| median {np.median(np.abs(err)):.2f} deg, max {np.max(np.abs(err)):.2f}")
        for kind, px in res.items():
            d = [px[k] - base[k] for k in imgs]
            f = flips(px)
            print(f"{deg:+4d}  {kind:5s}  {stats(d)}  call flips {f[0]:2d}/{f[2]} (of them margin>1px: {f[1]})"
                  + (f"  no tilt read: {miss}" if kind == "soft" else ""))
            rows.append({"test": "turn", "amount": deg, "kind": kind, "median": np.median(np.abs(d)),
                         "p90": np.percentile(np.abs(d), 90), "max": np.max(np.abs(d)),
                         "over1": np.mean(np.abs(d) > 1), "flips": f[0], "flips_clear": f[1], "layers": f[2],
                         "no_tilt": miss if kind == "soft" else ""})

    print("\n== shifted up/down (px, + = edge lower): measured without centring")
    for dy in SHIFTS:
        px = {key: gap(shift(img, dy)) for key, img in imgs.items()}
        d = [px[k] - base[k] for k in imgs]
        f = flips(px)
        print(f"{dy:+4d}  shift  {stats(d)}  call flips {f[0]:2d}/{f[2]} (of them margin>1px: {f[1]})")
        rows.append({"test": "shift", "amount": dy, "kind": "shift", "median": np.median(np.abs(d)),
                     "p90": np.percentile(np.abs(d), 90), "max": np.max(np.abs(d)), "over1": np.mean(np.abs(d) > 1),
                     "flips": f[0], "flips_clear": f[1], "layers": f[2], "no_tilt": ""})

    out = Path("captures/soft_level_test_20261116.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
