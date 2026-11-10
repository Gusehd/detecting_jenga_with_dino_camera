"""Fast far from the Dino, slow near it, on one straight joint path."""

import math

from jenga_vision.task import step_parts

START = [0.0] * 7
TARGET = [math.radians(100)] + [math.radians(50)] * 6


def biggest(a, b):
    return max(abs(math.degrees(x - y)) for x, y in zip(a, b))


def test_a_plain_step_is_one_move():
    assert step_parts(START, TARGET, 40, {}) == [(TARGET, 40)]


def test_slow_end():
    parts = step_parts(START, TARGET, 40, {"slow_end_deg": 15, "slow_speed": 10})
    assert [speed for _, speed in parts] == [40, 10]
    assert parts[-1][0] == TARGET
    assert abs(biggest(parts[0][0], TARGET) - 15) < 1e-6
    assert abs(parts[0][0][1] / parts[0][0][0] - 0.5) < 1e-9  # still on the straight joint path


def test_slow_start():
    parts = step_parts(START, TARGET, 40, {"slow_start_deg": 15, "slow_speed": 10})
    assert [speed for _, speed in parts] == [10, 40]
    assert abs(biggest(parts[0][0], START) - 15) < 1e-6
    assert parts[-1][0] == TARGET


def test_a_short_move_is_all_slow():
    near = [math.radians(5)] * 7
    assert step_parts(START, near, 40, {"slow_end_deg": 15, "slow_speed": 8}) == [(near, 8)]
