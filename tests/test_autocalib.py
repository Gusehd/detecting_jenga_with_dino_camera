import json
import math
from argparse import Namespace

import numpy as np

from jenga_vision import autocalib
from jenga_vision.autoscan import Shot
from jenga_vision.scan_plan import SETUP_FILE, arm_tool

SETUP = json.loads(SETUP_FILE.read_text())
TOOL = arm_tool(SETUP["dino"], "right")
HAND_Q = SETUP["dino"]["calibration"]["right"]["hand_q"]


def test_prediction_same_fingers_is_the_calibration():
    assert np.allclose(autocalib.predict_tool(TOOL, "right", HAND_Q, HAND_Q), TOOL)


def test_prediction_of_2026_11_19():
    # ring j1 83.2 instead of 100.5 deg (the teammate's mount): the focus point moves about 37 mm, 17 deg
    now = list(HAND_Q)
    now[12:16] = [math.radians(v) for v in (-2.8, 83.2, -8.2, 75.1)]
    mm, total, _, _ = autocalib.change(TOOL, autocalib.predict_tool(TOOL, "right", HAND_Q, now))
    assert 35 < np.linalg.norm(mm) < 40
    assert 16 < total < 19


def test_delta_puts_the_focus_on_the_model_edge():
    shot = Shot(7, 1, "middle", np.array([0.5, 0.0, 0.2]), np.array([1.0, 0, 0]))
    f = autocalib.Found(depth=0.003, down=-0.0015, roll=0.0, turn_deg=0.0, edge_err_m=0.0003, tilt_deg=0.0)
    really = shot.pose(f.depth, f.down) @ autocalib.delta(shot, f)
    # edge 0.3 mm below the image centre: the point in focus is 0.3 mm above the (model) edge
    assert np.allclose(really[:3, 3], shot.aim + np.array([0, 0, f.edge_err_m]), atol=1e-9)
    assert np.allclose(really[:3, :3], shot.pose(0, 0)[:3, :3])


def test_delta_roll_only():
    shot = Shot(7, 1, "middle", np.array([0.5, 0.0, 0.2]), np.array([1.0, 0, 0]))
    f = autocalib.Found(0.0, 0.0, 0.0, turn_deg=4.0, edge_err_m=0.0, tilt_deg=1.0)
    _, total, roll, tilt = autocalib.change(np.eye(4), autocalib.delta(shot, f, roll_sign=-1))
    assert math.isclose(roll, -5.0, abs_tol=1e-9) and math.isclose(total, 5.0, abs_tol=1e-6) and tilt < 1e-6


def test_good():
    assert autocalib.good(autocalib.Found(0.0001, 0.0, 0.0, 0.0, 0.00005, 0.3))
    assert not autocalib.good(autocalib.Found(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert not autocalib.good(autocalib.Found(0.0, 0.0005, 0.0, 0.0, 0.0, 0.0))


def test_write(tmp_path, monkeypatch):
    setup_file = tmp_path / "scan_setup.json"
    setup_file.write_text(SETUP_FILE.read_text())
    monkeypatch.setattr(autocalib, "SETUP_FILE", setup_file)
    out = tmp_path / "captures"
    out.mkdir()
    (out / ".rigid.json").write_text("{}")
    tool = TOOL @ autocalib.transform(np.eye(3), [0.0, 0.001, -0.002])
    res = {"q": [0.1] * 7, "mm_per_px": 2.6e-5}
    autocalib.write(Namespace(arm="right", hand=None, out=out), json.loads(setup_file.read_text()), tool, HAND_Q, res)
    saved = json.loads(setup_file.read_text())["dino"]
    assert np.allclose(saved["tool_hand"]["right"], tool, atol=1e-6)
    cal = saved["calibration"]["right"]
    assert cal["hand_pose"] == "jig_scan_pose" and cal["previous"]["pose"] == "px_col_mid"
    assert cal["face"] == "+x" and cal["layer"] == 7 and cal["slot"] == 1
    assert len(list(tmp_path.glob("scan_setup.json.bak_*"))) == 1
    assert not (out / ".rigid.json").exists() and len(list(out.glob(".rigid.json.bak_*"))) == 1
