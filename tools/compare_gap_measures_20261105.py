"""Offline: do softer gap measures predict better than measure_end? Candidates fixed before looking."""
import csv, glob, itertools, json, sys
from pathlib import Path
import cv2, numpy as np
from jenga_vision import ends
from jenga_vision.long_edge import CORE_WINDOW, DARK_FRACTION, _smooth
from jenga_vision.autoscan import pick_order, layer_margin, parse_truth, rim_darkness, streak

C = Path("captures")
def strips(gray):
    h, w = gray.shape
    x0, x1 = int(w * ends.MEASURE_SPAN[0]), int(w * ends.MEASURE_SPAN[1])
    med = float(np.median(gray))
    p_all = _smooth(np.median(gray[:, x0:x1], axis=1), 9)
    lo, hi = h // 4, 3 * h // 4
    c = lo + int(np.argmin(p_all[lo:hi]))
    a, b = max(c - CORE_WINDOW, 0), min(c + CORE_WINDOW + 1, h)
    bounds = np.linspace(x0, x1, ends.STRIPS + 1).astype(int)
    prof = [_smooth(np.median(gray[:, s0:s1], axis=1))[a:b] for s0, s1 in zip(bounds[:-1], bounds[1:])]
    return np.array(prof), med

def feats(path):
    gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    p, med = strips(gray)
    w = (p < DARK_FRACTION * med).sum(axis=1).astype(float)
    def soft(T):  # darkness-weighted width: thin grey lines count partly
        return float(np.mean(np.clip((T * med - p) / (T * med), 0, 1).sum(axis=1)))
    return dict(base=float(np.percentile(w, ends.SIDE_PERCENTILE)), p50=float(np.median(w)), mean=float(w.mean()),
                frac=float((w > 0).mean()), soft50=soft(.5), soft75=soft(.75),
                w40=float(np.percentile((p < .4 * med).sum(axis=1), 25)),
                rim=1 - rim_darkness(cv2.imread(str(path))))

FEATS = ["base", "p50", "mean", "frac", "soft50", "soft75", "w40", "rim"]
def load_robot(names):
    out = {}
    for n in names:
        d = C / f"autoscan_20261103_{n}"
        for r in csv.DictReader(open(d / "shots.csv")):
            out[(int(r["layer"]), int(r["slot"]))] = (r["face"], feats(d / r["image"]), float(r["gap"]))
    return out

A = "4=안빼안 6=안빼안 8=빼안빼 10=안빼안 12=빼안빼 14=안빼안".split()
B = "3=안빼안 4=안빼안 5=빼안빼 6=안빼안 7=안빼안 8=빼안빼 9=안빼안 10=안빼안 11=빼안빼 12=빼안빼 13=빼안빼 14=안빼안 15=안빼안".split()
Cc = "5=빼안빼 6=안빼안 7=안빼안 8=빼안빼 9=안빼안 10=빼안빼 11=빼안빼 13=빼안빼".split()
SETS = [("A front", ["114143"], A), ("A back", ["115306"], A), ("B", ["142245", "144017"], B), ("C", ["175150", "180148"], Cc)]

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return float(np.mean([(x > y) + .5 * (x == y) for x in pos for y in neg]))

print("-- robot, dial 40. pattern = layers with right 안빼안/빼안빼; streak = margin order with rule (ties worst-best)")
tot = {f: [0, 0, 0] for f in FEATS}
rows = {}
for name, sess, words in SETS:
    data = load_robot(sess); truth = parse_truth(words)
    data = {k: v for k, v in data.items() if k in truth}
    rows[name] = (data, truth)
    print(f"{name}: {len(data)} blocks, {sum(truth[k] for k in data)} pull")
    for f in FEATS:
        px = {k: v[1][f] for k, v in data.items()}
        layers = sorted({k[0] for k in px})
        good = n = 0
        for l in layers:
            m = layer_margin(px, l)
            if m:
                n += 1
                good += (m[1] == 1) == truth[(l, 1)]
        seq = pick_order(px)
        s = streak(seq, truth)
        lo = streak(pick_order(px, "margin", lambda b: truth[b]), truth)
        hi = streak(pick_order(px, "margin", lambda b: not truth[b]), truth)
        a = auc([px[k] for k in px if truth[k]], [px[k] for k in px if not truth[k]])
        tot[f][0] += good; tot[f][1] += n; tot[f][2] += s
        print(f"  {f:7s} pattern {good:2d}/{n:2d}  streak {s:2d} ({lo}-{hi})  blockAUC {a:.2f}")
print("total robot:")
for f in FEATS:
    print(f"  {f:7s} pattern {tot[f][0]}/{tot[f][1]}  streak sum {tot[f][2]}")

print("\n-- tower 4 (hand held, dial 50, all 45 pushed): by end, using file names; both ends mean of feature")
d = C / "ends_20261021_142127"
truth4 = {(int(r["layer"]), r["block"]): r["actual"] == "easy" for r in csv.DictReader(open(d / "blocks.csv"))}
imgs = {}
for p in d.glob("L*_f*_*.png"):
    a = p.stem.split("_")
    imgs[f"{a[0]}:{a[1]}:{a[2]}"] = feats(p)  # a later retake overwrites only if sorted; checked below
print("shots", len(imgs), "of ends.csv", sum(1 for _ in open(d / "ends.csv")) - 1)
cs = {r["key"]: float(r["gap"]) for r in csv.DictReader(open(d / "ends.csv"))}
print("base vs ends.csv differ on", sum(abs(imgs[k]["base"] - cs[k]) > .01 for k in cs if k in imgs))
for f in FEATS:
    shots = {k: v[f] for k, v in imgs.items()}
    order = ends.ranking(shots, list(range(2, 17)), 2, "mean")
    res = {f"{l}:{b}": "easy" if v else "hard" for (l, b), v in truth4.items()}
    s = ends.streak(order, res)
    a = auc([x[0] for x in order if truth4[(x[1], x[2])]], [x[0] for x in order if not truth4[(x[1], x[2])]])
    # robot-like: only faces 1 and 4 (+x, -y)
    robot = {k: v for k, v in shots.items() if ":f1:" in k or ":f4:" in k}
    o2 = ends.ranking(robot, list(range(3, 16)), 2, "mean")
    s2 = ends.streak(o2, res)
    a2 = auc([x[0] for x in o2 if truth4[(x[1], x[2])]], [x[0] for x in o2 if not truth4[(x[1], x[2])]])
    print(f"  {f:7s} four faces: streak {s:2d} AUC {a:.2f}   f1+f4 L3-15: streak {s2:2d} AUC {a2:.2f}")
