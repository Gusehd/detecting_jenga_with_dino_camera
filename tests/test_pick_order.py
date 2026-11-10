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
