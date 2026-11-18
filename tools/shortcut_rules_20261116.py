"""Offline check (2026-11-16): which shots of a face can be skipped without changing the pull order?

The 빼안빼 rule decides a layer by its middle gap against the wider side gap (layer_margin):
middle >= wider side -> 안빼안 (pull the middle), else 빼안빼 (the wider side, then the other side).
Candidate shortcuts, each judged against measuring all three blocks:
  mid0     shoot the middle first; middle 0 px -> 빼안빼 at once, sides not shot
  side0    shoot one side first; 0 px -> 안빼안 at once (the other two not shot)
  sides0   shoot both sides first; both 0 px -> 안빼안 (the middle not shot)
  enough   stop the scan once K pulls from layers with |margin| > T are in hand (the rest of the face,
           or the whole second face, not shot)

Data: robot scans 2026-11-03..11-10 (no pull truth saved: agreement with the full rule only), and the
hand scan tower of 2026-10-21 (ends_20261021_142127, all 45 blocks pushed: truth), one end at a time.

    uv run python tools/shortcut_rules_20261116.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from jenga_vision.autoscan import layer_margin, pick_order

C = Path("captures")
# same tower, -y then +x (2026-11-03 A/B/C, 2026-11-10 morning x3, new tower); 14:05 (13 deg turned) and
# 13:58 (tower stacked the other way) left out
PAIRS = [("20261103_142245", "20261103_144017"),
         ("20261103_175150", "20261103_180148"), ("20261110_115356", "20261110_115529"),
         ("20261110_120702", "20261110_120832"), ("20261110_133822", "20261110_133938"),
         ("20261110_141611", "20261110_141726")]


def robot_px(name: str) -> dict[tuple[int, int], float]:
    with (C / f"autoscan_{name}" / "shots.csv").open() as f:
        return {(int(r["layer"]), int(r["slot"])): float(r["gap"]) for r in csv.DictReader(f)}


def layers_of(px: dict) -> list[int]:
    return sorted({l for l, s in px if all((l, k) in px for k in (0, 1, 2))})


def decision(px: dict, layer: int) -> tuple[float, int]:
    return layer_margin(px, layer)


def robot_part() -> None:
    print("== robot scans: how often each shortcut fires and whether it agrees with all three shots")
    n = mid0 = sides0 = allzero = 0
    side0_fire = side0_wrong = 0
    mid0_wrong = 0
    per_face = []
    for a, b in PAIRS:
        for name in (a, b):
            px = robot_px(name)
            for layer in layers_of(px):
                L, M, R = px[(layer, 0)], px[(layer, 1)], px[(layer, 2)]
                margin, first = decision(px, layer)
                n += 1
                if L == M == R == 0:
                    allzero += 1
                if M == 0:
                    mid0 += 1
                    if first == 1:  # the full rule says 안빼안 (only when all three are 0)
                        mid0_wrong += 1
                if L == 0 and R == 0:
                    sides0 += 1  # the full rule then always says 안빼안 (margin = M >= 0)
                for side, other in ((L, R), (R, L)):
                    if side == 0:
                        side0_fire += 1
                        if first != 1:  # other side wider than the middle: really 빼안빼
                            side0_wrong += 1
            per_face.append(name)
    print(f"   {n} layer readings ({len(per_face)} faces)")
    print(f"   mid0:   middle 0 px in {mid0} ({mid0 / n:.0%}); disagrees with the full rule {mid0_wrong} "
          f"(all three 0: {allzero}); saves 2 shots each")
    print(f"   sides0: both sides 0 px in {sides0} ({sides0 / n:.0%}); always agrees; saves 1 shot each")
    print(f"   side0:  a side 0 px {side0_fire} times (either side shot first); full rule says 빼안빼 anyway in "
          f"{side0_wrong} ({side0_wrong / max(side0_fire, 1):.0%}) -> not safe")

    print("\n== 'enough': pulls from layers with |margin| > T, per tower (-y face alone / both faces)")
    for T in (1.0, 2.0, 3.0):
        row = []
        for a, b in PAIRS:
            counts = []
            for names in ((a,), (a, b)):
                px = {}
                for nm in names:
                    px.update(robot_px(nm))
                seq = pick_order(px)
                sure = {l for l in layers_of(px) if abs(decision(px, l)[0]) > T}
                counts.append(sum(1 for l, _, _ in seq if l in sure))
            row.append(f"{counts[0]:2d}/{counts[1]:2d}")
        print(f"   T={T:.0f} px: " + "  ".join(row))
    print("   (towers: 11-03 B, C, 11-10 morning x3, 11-10 new; first number = the -y face alone)")


def lazy(L: float, M: float, R: float, side_first: int) -> tuple[int, dict]:
    """Shots of a layer when it stops as soon as the 빼안빼 call is certain: the nearer side first, then the
    middle, the far side only if still needed. (shots taken, the gaps seen)."""
    s1, s2 = (L, R) if side_first == 0 else (R, L)
    k1, k2 = side_first, 2 - side_first
    if s1 > M:  # that side is wider than the middle: 빼안빼, whatever the far side is
        return 2, {k1: s1, 1: M}
    return 3, {k1: s1, 1: M, k2: s2}


def lazy_part() -> None:
    print("\n== lazy layer: near side, middle, far side only if needed (the call is the same by construction)")
    full = taken = 0
    topk = []
    for a, b in PAIRS:
        for name in (a, b):
            px = robot_px(name)
            seen = {}
            for i, layer in enumerate(layers_of(px)):
                L, M, R = px[(layer, 0)], px[(layer, 1)], px[(layer, 2)]
                n, g = lazy(L, M, R, i % 2 * 2)  # zigzag: start side alternates
                full += 3
                taken += n
                for k, v in g.items():
                    seen[(layer, k)] = v
            # the pull order from what was seen: a skipped far side counts as 0 (margin = lower bound)
            filled = {(l, k): seen.get((l, k), 0.0) for l in layers_of(px) for k in (0, 1, 2)}
            o_full = [(l, s) for l, s, _ in pick_order(px)]
            o_lazy = [(l, s) for l, s, _ in pick_order(filled)]
            for K in (8, 10):
                topk.append((K, len(set(o_full[:K]) & set(o_lazy[:K]))))
    print(f"   shots {taken} of {full} ({1 - taken / full:.0%} fewer)")
    for K in (8, 10):
        v = [x for k, x in topk if k == K]
        print(f"   first {K} pulls: the same blocks as the full order in {sum(x == K for x in v)} of {len(v)} faces, "
              f"min overlap {min(v)}/{K}")


def hand_part() -> None:
    d = C / "ends_20261021_142127"
    with (d / "ends.csv").open() as f:
        shots = {r["key"]: float(r["gap"]) for r in csv.DictReader(f)}
    with (d / "blocks.csv").open() as f:
        easy = {(int(r["layer"]), r["block"]): r["actual"] == "easy" for r in csv.DictReader(f)}
    from jenga_vision.ends import end_gaps
    first = 2
    readings = []  # (layer, end, {block: gap})
    for layer in range(2, 17):
        for end, g in end_gaps(shots, layer, first).items():
            if len(g) == 3:
                readings.append((layer, end, g))
    print(f"\n== hand-scan tower 2026-10-21 (truth for all 45): {len(readings)} end readings of 15 layers")

    def truth_kind(layer: int) -> str:
        t = "".join("빼" if easy[(layer, b)] else "안" for b in ("left", "middle", "right"))
        return t

    m0 = [(l, e) for l, e, g in readings if g["middle"] == 0]
    m0_sides_easy = sum(easy[(l, "left")] and easy[(l, "right")] for l, e in m0)
    m0_one_side = sum(easy[(l, "left")] or easy[(l, "right")] for l, e in m0)
    print(f"   middle 0 px in {len(m0)} readings: both sides easy {m0_sides_easy}, at least one {m0_one_side}, "
          f"middle easy {sum(easy[(l, 'middle')] for l, e in m0)}")
    s0 = [(l, e) for l, e, g in readings if g["left"] == 0 and g["right"] == 0]
    print(f"   both sides 0 px in {len(s0)} readings: middle easy {sum(easy[(l, 'middle')] for l, e in s0)}")
    one = [(l, e, g) for l, e, g in readings if (g["left"] == 0) != (g["right"] == 0)]
    ok = sum(easy[(l, "middle")] for l, e, g in one)
    print(f"   exactly one side 0 px in {len(one)} readings: middle easy {ok} (side0 would call 안빼안)")
    print("   layer truths: " + " ".join(f"{l}={truth_kind(l)}" for l in range(2, 17)))


if __name__ == "__main__":
    robot_part()
    lazy_part()
    hand_part()
