"""Check the middle-block inference layer by layer on a pushed tower session.

The theory has two parts, checked separately:
  A. physics (push results only, no camera): in each layer the two outer
     blocks behave alike and the middle block does the opposite.
       middle carries  -> P1 easy, P3 easy, P2 hard
       outers carry    -> P1 hard, P3 hard, P2 easy
     anything else (all three hard, mixed outers) breaks the rule.
  B. inference (camera): layer_px = larger outer score.
       layer_px >= --sure          predict "middle carries" (outers SURE)
       layer_px <  --middle-below  predict "outers carry"   (P2 LIKELY)
     compared against what the pushes showed.
Then a sweep of both thresholds shows how sensitive the result is.

For a full check push every block in the check phase, including the "-" ones
(u = skip leaves a hole in the table).

Usage:
  jenga-layers captures/tower_20261020_120701
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from jenga_vision.pick import layer_ranking

MARK = {"easy": "E", "hard": "H"}


def load(session: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Outer scores and push results from picks.csv (old and new column layouts)."""
    scores, results = {}, {}
    with (session / "picks.csv").open() as f:
        for row in csv.DictReader(f):
            block = row["block"]
            if not block.endswith("P2") and row["px"]:
                scores[block] = float(row["px"])
            if row["actual"] in MARK:
                results[block] = row["actual"]
    return scores, results


def physics(p1: str | None, p2: str | None, p3: str | None) -> str:
    """Layer state from push results: middle / outers / broken / ? (incomplete)."""
    known = [x for x in (p1, p2, p3) if x]
    if p1 == p3 == "easy" and p2 in ("hard", None) or p2 == "hard" and "easy" in (p1, p3) and None in (p1, p3):
        return "middle" if len(known) == 3 else "middle?"
    if p1 == p3 == "hard" and p2 in ("easy", None) or p2 == "easy" and "hard" in (p1, p3) and None in (p1, p3):
        return "outers" if len(known) == 3 else "outers?"
    if len(known) < 2:
        return "?"
    if p2 is None and p1 != p3 and None not in (p1, p3):
        return "broken"  # mixed outers can never fit, whatever P2 is
    return "broken" if len(known) == 3 or (p2 and p2 == p1 or p2 and p2 == p3) else "?"


def precision(scores, results, sure, middle_below) -> tuple[int, int, int, int]:
    """(easy SURE, tested SURE, easy LIKELY, tested LIKELY) for one threshold pair."""
    counts = {"SURE": [0, 0], "LIKELY": [0, 0]}
    for pk in layer_ranking(scores, sure, middle_below):
        if pk.tier in counts and pk.block in results:
            counts[pk.tier][0] += results[pk.block] == "easy"
            counts[pk.tier][1] += 1
    return (*counts["SURE"], *counts["LIKELY"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--sure", type=float, default=16.0)
    parser.add_argument("--middle-below", type=float, default=14.0)
    args = parser.parse_args()

    scores, results = load(args.session)
    layers = sorted({int(b[1:3]) for b in scores})

    print("A + B per layer   (E easy, H hard, . not pushed)")
    print(f"{'layer':>5}  {'P1px':>5} {'P3px':>5} {'layer':>5}  P1 P2 P3  {'physics':8} {'predict':8} verdict")
    tally = {"physics": {}, "predict": {}}
    for layer in layers:
        px = [scores.get(f"L{layer:02d}-P{p}") for p in (1, 3)]
        layer_px = max(v for v in px if v is not None)
        act = [results.get(f"L{layer:02d}-P{p}") for p in (1, 2, 3)]
        state = physics(*act)
        if layer_px >= args.sure:
            pred = "middle"
        elif layer_px < args.middle_below:
            pred = "outers"
        else:
            pred = "-"
        if pred == "-" or state.startswith("?"):
            verdict = ""
        elif state.rstrip("?") == pred:
            verdict = "OK" + ("?" if state.endswith("?") else "")
        else:
            verdict = "MISS" + ("?" if state.endswith("?") else "")
        tally["physics"][state.rstrip("?")] = tally["physics"].get(state.rstrip("?"), 0) + 1
        if verdict:
            tally["predict"][verdict.rstrip("?")] = tally["predict"].get(verdict.rstrip("?"), 0) + 1
        cells = "  ".join(MARK.get(a, ".") for a in act)
        fmt = lambda v: f"{v:5.1f}" if v is not None else "    -"
        print(f"{layer:>5}  {fmt(px[0])} {fmt(px[1])} {layer_px:5.1f}  {cells}   {state:8} {pred:8} {verdict}")

    print("\n(a trailing ? means one block of the layer was not pushed: the state is inferred)")
    print("A physics: " + ", ".join(f"{k} {v}" for k, v in sorted(tally["physics"].items())))
    print("B predict: " + ", ".join(f"{k} {v}" for k, v in sorted(tally["predict"].items())))
    s_easy, s_n, l_easy, l_n = precision(scores, results, args.sure, args.middle_below)
    print(f"picks:     SURE {s_easy}/{s_n}   LIKELY {l_easy}/{l_n}")

    print("\nthreshold sweep   (rows --sure, cols --middle-below; SURE easy/tested | LIKELY easy/tested)")
    cols = [10, 12, 14, 16]
    print(" " * 6 + "".join(f"{c:>16}" for c in cols))
    for sure in (12, 14, 16, 18, 20):
        cells = []
        for mb in cols:
            if mb > sure:
                cells.append(f"{'':>16}")
                continue
            se, sn, le, ln = precision(scores, results, sure, mb)
            cells.append(f"{se:>3}/{sn:<3}|{le:>3}/{ln:<3}".rjust(16))
        print(f"{sure:>6}" + "".join(cells))


if __name__ == "__main__":
    main()
