"""Pull order with the 빼안빼 rule, and its streak against pulled-by-hand results."""

from jenga_vision.autoscan import parse_truth, pick_order, streak


def test_middle_finishes_its_layer():
    px = {(5, 0): 1.0, (5, 1): 9.0, (5, 2): 2.0, (6, 1): 3.0}
    assert [b[:2] for b in pick_order(px)] == [(5, 1), (6, 1)]


def test_side_is_followed_by_the_opposite_side():
    px = {(5, 0): 9.0, (5, 1): 4.0, (5, 2): 0.0, (6, 1): 3.0}
    assert pick_order(px) == [(5, 0, "gap"), (5, 2, "opposite"), (6, 1, "gap")]


def test_opposite_side_comes_even_if_it_was_not_measured():
    assert pick_order({(5, 2): 7.0, (5, 1): 1.0}) == [(5, 2, "gap"), (5, 0, "opposite")]


def test_ties_take_the_upper_layer_then_the_middle():
    px = {(4, 0): 4.0, (4, 1): 4.0, (4, 2): 4.0, (6, 0): 4.0, (6, 1): 4.0, (6, 2): 4.0}
    assert [b[:2] for b in pick_order(px)] == [(6, 1), (4, 1)]


def test_without_the_rule_every_block_is_listed():
    px = {(4, 0): 1.0, (4, 1): 4.0, (4, 2): 2.0}
    assert [b[:2] for b in pick_order(px, rule=False)] == [(4, 1), (4, 2), (4, 0)]


def test_truth_floor_is_one_based():
    truth = parse_truth(["5=빼안빼"])
    assert truth == {(4, 0): True, (4, 1): False, (4, 2): True}


def test_streak_stops_at_the_first_failure():
    truth = parse_truth(["5=빼안빼", "6=안빼안"])
    assert streak([(4, 0, "gap"), (4, 2, "opposite"), (5, 0, "gap"), (5, 1, "gap")], truth) == 2
