"""Nominal Dino tool from the hand-jig design (JENGA_ZIG, 2026-11-17) -> plan both faces, clearances.

Lens tip and axis below are in the hand base frame (ahr_base_link), from JENGA_ZIG out/design.json; change
them for a new jig design.

    uv run python tools/jig_reach_20261117.py
"""
import json, math, numpy as np
from jenga_vision.kinematics import Chain, axis_angle, transform
from jenga_vision.scan_plan import Tower, arm_tool, plan_face_stable, away_from_limits, transit_clearance, SETUP_FILE
from jenga_vision.arm_slider import POSES_FILE
from jenga_vision.sim import clearances, geometry
from jenga_vision.autoscan import RUN_FACES, RUN_LAYERS

setup = json.loads(SETUP_FILE.read_text())
tower = Tower(setup["tower"], setup.get("aim", 0.5))
old = arm_tool(setup["dino"], "right")
tip = np.array([-95.5, -57.9, 84.7]) / 1000
z = np.array([-0.999, -0.037, 0.0]); z /= np.linalg.norm(z)
zo = old[:3, 2]
ax = np.cross(zo, z); ang = math.asin(min(1, np.linalg.norm(ax)))
R = axis_angle(ax / np.linalg.norm(ax), ang) @ old[:3, :3]
new = transform(R, tip + z * setup["dino"].get("tip_to_focus_m", 0.025))
print(f"axis turn {math.degrees(ang):.1f} deg, focus point old {np.round(old[:3,3]*1000,1)} new {np.round(new[:3,3]*1000,1)} mm")
poses = json.loads(POSES_FILE.read_text())["right"]
chain = Chain("right")
plan_setup = dict(setup, dino=dict(setup["dino"], standoff_m=0.0))
layers = range(RUN_LAYERS[0], RUN_LAYERS[1] + 1)
for name, tool in (("now", old), ("jig", new)):
    print(f"== {name}")
    for face, s in RUN_FACES:
        f = face
        try:
            pts, _ = plan_face_stable(chain, tool, tower, f, layers, poses[s["seed"]], plan_setup)
        except ValueError as e:
            print(f"   {f}: PLAN FAILED: {e}"); continue
        notes = away_from_limits(pts, chain)
        hq = np.zeros(20)
        worst = {}
        for p in pts:
            c = clearances(tower, geometry(chain, tool, "right", p["q"], hq))
            for k in ("palm", "dino", "fingers"):
                if not p["via"] or k != "dino":
                    worst[k] = min(worst.get(k, 1e9), c[k])
        first, last = pts[0]["q"], pts[-1]["q"]
        start = poses["home"] if not s["via"] else poses[s["via"][0]]
        tin = transit_clearance(chain, tool, tower, start, last if s["top_first"] else first)
        tout = transit_clearance(chain, tool, tower, last, start)
        big = max(abs(math.degrees(a - b)) for a, b in zip(start, last if s["top_first"] else first))
        print(f"   {f}: {len(pts)} points ok {notes}; palm min {worst['palm']:.0f} mm, dino body min {worst['dino']:.0f}, open fingers min {worst['fingers']:.0f} mm "
              f"(shots); transit in {tin*1000:.0f} mm, out {tout*1000:.0f} mm; biggest joint in {big:.0f} deg")
