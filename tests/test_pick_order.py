"""Pull order with the 빼안빼 rule, and its streak against pulled-by-hand results."""

from jenga_vision.autoscan import layer_margin, parse_truth, pick_order, streak


def test_middle_finishes_its_layer():
    px = {(5, 0): 1.0, (5, 1): 9.0, (5, 2): 2.0, (6, 1): 3.0}
    assert [b[:2] for b in pick_order(px, "rule")] == [(5, 1), (6, 1)]


def test_side_is_followed_by_the_opposite_side():
    px = {(5, 0): 9.0, (5, 1): 4.0, (5, 2): 0.0, (6, 1): 3.0}
    assert pick_order(px, "rule") == [(5, 0, "gap"), (5, 2, "opposite"), (6, 1, "gap")]


def test_opposite_side_comes_even_if_it_was_not_measured():
    assert pick_order({(5, 2): 7.0, (5, 1): 1.0}, "rule") == [(5, 2, "gap"), (5, 0, "opposite")]


def test_ties_take_the_upper_layer_then_the_middle():
    px = {(4, 0): 4.0, (4, 1): 4.0, (4, 2): 4.0, (6, 0): 4.0, (6, 1): 4.0, (6, 2): 4.0}
    assert [b[:2] for b in pick_order(px, "rule")] == [(6, 1), (4, 1)]


def test_without_the_rule_every_block_is_listed():
    px = {(4, 0): 1.0, (4, 1): 4.0, (4, 2): 2.0}
    assert [b[:2] for b in pick_order(px, "px")] == [(4, 1), (4, 2), (4, 0)]


def test_truth_floor_is_one_based():
    truth = parse_truth(["5=빼안빼"])
    assert truth == {(4, 0): True, (4, 1): False, (4, 2): True}


def test_streak_stops_at_the_first_failure():
    truth = parse_truth(["5=빼안빼", "6=안빼안"])
    assert streak([(4, 0, "gap"), (4, 2, "opposite"), (5, 0, "gap"), (5, 1, "gap")], truth) == 2


def test_margin_takes_the_clearest_layer_first():
    px = {(5, 0): 20.0, (5, 1): 19.0, (5, 2): 0.0, (6, 0): 0.0, (6, 1): 8.0, (6, 2): 0.0}
    assert [b[:2] for b in pick_order(px)] == [(6, 1), (5, 0), (5, 2)]


def test_margin_tie_between_middle_and_side_is_the_middle():
    assert layer_margin({(5, 0): 16.0, (5, 1): 16.0, (5, 2): 10.0}, 5) == (0.0, 1)


def test_a_layer_without_a_middle_or_sides_is_not_ranked():
    assert layer_margin({(5, 2): 7.0}, 5) is None
    assert pick_order({(5, 2): 7.0, (6, 1): 3.0, (6, 0): 0.0}) == [(6, 1, "gap")]


def test_margin_order_on_the_tower_of_2026_11_03():
    rows = {3: (3, 15, 6), 4: (4, 7, 7), 5: (14, 9, 25), 6: (0, 3, 1), 7: (2, 12, 10), 8: (7, 2, 10), 9: (4.75, 15, 10),
            10: (0, 18, 0), 11: (16, 16, 10), 12: (None, None, 7), 13: (4, 3, 9), 14: (None, 2, 3), 15: (4, 6, 7)}
    px = {(floor - 1, slot): v for floor, row in rows.items() for slot, v in enumerate(row) if v is not None}
    truth = parse_truth(["3=안빼안", "4=안빼안", "5=빼안빼", "6=안빼안", "7=안빼안", "8=빼안빼", "9=안빼안", "10=안빼안",
                         "11=빼안빼", "12=빼안빼", "13=빼안빼", "14=안빼안", "15=안빼안"])
    assert streak(pick_order(px), truth) == 11


def test_geometry_from_the_shots(tmp_path):
    """A block 10 px under the image centre is 10 px x scale lower; its top is half the gap under the band."""
    import csv
    import json

    from jenga_vision.autoscan import SETUP_FILE, block_geometry, order_main
    t = json.loads(SETUP_FILE.read_text())["tower"]
    s = tmp_path / "autoscan_x"
    s.mkdir()
    rows = [dict(key=f"L09:f4:{n}", gap=g, face="-y", layer=8, slot=k, screen=n, depth_mm=3.0, down_mm=-1.0,
                 row=r, image=f"{n}.png", mm_per_px=0.02) for k, (n, g, r) in
            enumerate([("left", 0.0, 240), ("middle", 10.0, 250), ("right", 0.0, 240)])]
    with (s / "shots.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    blocks, layers, _, block_z = block_geometry([s])
    model_top = t["z0"] + 9 * t["layer_h"]
    assert abs(blocks[0]["dz_mm"] - 1.0) < 1e-6  # 1 mm above the model gap, no band
    assert blocks[1]["gap_mm"] == 0.2
    assert abs(blocks[1]["top_z_m"] - (model_top + 0.001 - 0.0002 - 0.0001)) < 1e-5
    assert blocks[1]["out_mm"] == -3.0
    assert layers[0]["carries"] in ("jenga_L08_S0", "jenga_L08_S2") and abs(layers[0]["spread_mm"] - 0.3) < 1e-6
    order_main([s], tmp_path / "order.json")
    d = json.loads((tmp_path / "order.json").read_text())
    assert d["candidates"][0] == {"block": "jenga_L08_S1", "score": 10.0, "reason": "빼안빼 rule 1 (gap)"}
    assert len(d["blocks"]) == 3
    assert d["block_z"]["jenga_L08_S1"]["source"] == "measured" and d["block_z"]["jenga_L08_S1"]["top_residual_mm"] < 0
