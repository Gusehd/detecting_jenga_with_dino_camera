"""Merge scan sessions of one tower into a single ranked pick list.

All sessions must come from the same tower under the same lighting (turn the
tower, not the camera), so px values are comparable across faces. Blocks are
ranked by px and tagged with a confidence tier:
  SURE    px >= --sure    (14 px: 100% on the first tower, 9/13 on the second)
  LIKELY  px >= --likely  (12 px: about 3 in 4)
  -       below that, not recommended
jenga-tower uses the per-layer rule in layer_ranking() instead.

If the sessions already contain push results, precision per tier and for the
top-k is printed as well.

Usage:
  jenga-pick captures/scan_A captures/scan_B captures/scan_middle_C
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Pick:
    layer: int
    position: int
    px: float
    source: str
    actual: str

    @property
    def face(self) -> str:
        if self.position == 2:
            return "middle"
        return f"{'odd' if self.layer % 2 else 'even'}-P{self.position}"


def load(sessions: list[Path]) -> list[Pick]:
    """Read every scan.csv; a block scanned twice keeps the later session's value."""
    picks: dict[tuple[int, int], Pick] = {}
    for session in sorted(sessions, key=lambda p: p.name.split("_")[-2:]):
        with (session / "scan.csv").open() as f:
            for row in csv.DictReader(f):
                key = (int(row["layer"]), int(row["position"]))
                picks[key] = Pick(*key, float(row["px"]), session.name, row["actual"])
    return list(picks.values())


def tier(px: float, sure: float, likely: float) -> str:
    if px >= sure:
        return "SURE"
    if px >= likely:
        return "LIKELY"
    return "-"


@dataclass
class LayerPick:
    block: str  # e.g. "L07-P1", "L07-P2"
    px: float | None  # the block's own score (None: middle block not shot)
    layer_px: float  # larger of the layer's two outer scores
    tier: str
    partner_px: float = 0.0  # smaller of the layer's two outer scores


def layer_ranking(scores: dict[str, float], sure: float, middle_below: float) -> list[LayerPick]:
    """Judge whole layers from the outer blocks' top-edge gaps.

    Both outer blocks of a layer behave the same way (11 of 11 layers on the
    2026-10-20 tower): either the middle block carries the layer above and both
    outers slide, or the outers carry it and the middle slides. So the layer is
    judged by the larger of its two outer scores:
      >= sure          both outer blocks SURE, middle not recommended
      <  middle_below  middle block LIKELY
      in between       nothing recommended in this layer
    The middle block's own end-face score is not used (no signal on that tower).
    """
    layers: dict[int, dict[str, float]] = {}
    for block, px in scores.items():
        layers.setdefault(int(block[1:3]), {})[block[-2:]] = px
    picks = []
    for layer, pos in layers.items():
        outer = [px for p, px in pos.items() if p != "P2"]
        if not outer:
            continue
        layer_px, partner_px = max(outer), min(outer)
        for p in ("P1", "P2", "P3"):
            if p == "P2":
                t = "LIKELY" if layer_px < middle_below else "-"
            elif p in pos:
                t = "SURE" if layer_px >= sure else "-"
            else:
                continue
            picks.append(LayerPick(f"L{layer:02d}-{p}", pos.get(p), layer_px, t, partner_px))

    def order(pk: LayerPick) -> tuple:
        # Outers by layer score, then own score: the partner of a wide gap goes
        # before narrower gaps elsewhere (streak 7 -> 12, 3-5 -> 7, 3 -> 7 on the
        # 2026-10-20 scans). Equal layer scores: the layer whose other outer is
        # wider first, then the upper layer (less weight on it). SURE first,
        # then LIKELY middles (weakest outers first), then the remaining outers
        # in the same layer order, middles last.
        layer = int(pk.block[1:3])
        outer_key = (-pk.layer_px, -pk.partner_px, -layer, -(pk.px or 0.0))
        if pk.tier == "SURE":
            return (0, *outer_key)
        if pk.tier == "LIKELY":
            return (1, pk.layer_px)
        if not pk.block.endswith("P2"):
            return (2, *outer_key)
        return (3, -(pk.px if pk.px is not None else -1.0))

    return sorted(picks, key=order)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", type=Path, help="scan session folders of one tower")
    parser.add_argument("--sure", type=float, default=14.0)
    parser.add_argument("--likely", type=float, default=12.0)
    parser.add_argument("--out", type=Path, default=None, help="csv path (default: picks.csv next to the sessions)")
    args = parser.parse_args()

    picks = sorted(load(args.sessions), key=lambda p: p.px, reverse=True)
    out = args.out or args.sessions[0].parent / "picks.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "layer", "position", "face", "px", "tier", "actual", "source"])
        for rank, p in enumerate(picks, 1):
            w.writerow([rank, p.layer, p.position, p.face, f"{p.px:.1f}",
                        tier(p.px, args.sure, args.likely), p.actual, p.source])

    print(f"{'rank':>4}  {'block':8} {'face':8} {'px':>5}  {'tier':6}  actual")
    for rank, p in enumerate(picks, 1):
        t = tier(p.px, args.sure, args.likely)
        if t == "-" and not p.actual:
            continue
        print(f"{rank:>4}  L{p.layer:<2} P{p.position}   {p.face:8} {p.px:5.1f}  {t:6}  {p.actual}")
    n_sure = sum(tier(p.px, args.sure, args.likely) == "SURE" for p in picks)
    n_likely = sum(tier(p.px, args.sure, args.likely) == "LIKELY" for p in picks)
    print(f"\n{len(picks)} blocks scanned: {n_sure} SURE, {n_likely} LIKELY   -> {out}")

    tested = [p for p in picks if p.actual in ("easy", "hard")]
    if tested:
        print("\nchecked against push results:")
        for name in ("SURE", "LIKELY", "-"):
            group = [p for p in tested if tier(p.px, args.sure, args.likely) == name]
            if group:
                print(f"  {name:6}: {sum(p.actual == 'easy' for p in group)}/{len(group)} easy")
        ranked = [p for p in picks if p.actual in ("easy", "hard")]
        for k in (5, 10, 15):
            if len(ranked) >= k:
                print(f"  top{k:<2}  : {sum(p.actual == 'easy' for p in ranked[:k])}/{k} easy")


if __name__ == "__main__":
    main()
