"""find_peak on the focus curves recorded on the robot (dial 30, 2026-11-02)."""

from pathlib import Path

import numpy as np
import pytest

from jenga_vision.focus_search import find_peak

DATA = Path(__file__).parent / "data"
OFFSETS = (-15, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15)  # arrival from the peak, + = deeper


class Curve:
    """A recorded curve. Only 1.4-3.8 mm of the inner side was recorded: further in, its
    outer side is mirrored (both sides measured about 4 mm to half value)."""

    def __init__(self, name: str, seed: int = 0, noise: float = 0.0, dip: float = 0.0):
        d = np.genfromtxt(DATA / f"focus_curve_{name}.csv", delimiter=",", names=True)
        self.out, self.val = d["out_mm"], d["focus"]
        self.peak = float(self.out[self.val.argmax()])
        self.rng, self.noise, self.dip = np.random.default_rng(seed), noise, dip
        self.deepest = -1e9
        self.moved = 0.0
        self.at = None

    def value(self, depth: float) -> float:
        """depth: mm from the peak, + = deeper."""
        out = self.peak - depth
        if out < 0:
            out = self.peak + abs(depth)
        return float(np.interp(out, self.out, self.val))

    def measure(self, depth: float) -> float:
        self.deepest = max(self.deepest, depth)
        self.moved += 0.0 if self.at is None else abs(depth - self.at)
        self.at = depth
        v = self.value(depth) * (1 + self.rng.normal(0, self.noise))
        return v * self.rng.uniform(0.8, 0.95) if self.rng.random() < self.dip else v


def search(c: Curve, off: float, out=20.0, inward=20.0):
    return find_peak(c.measure, float(off), off - out, off + inward)


@pytest.mark.parametrize("name", ["line", "plain"])
@pytest.mark.parametrize("off", OFFSETS)
def test_finds_the_peak_from_any_arrival(name, off):
    c = Curve(name)
    p = search(c, off)
    assert p.found, p.reason
    assert abs(p.depth) <= 1.0


@pytest.mark.parametrize("name", ["line", "plain"])
@pytest.mark.parametrize("off", OFFSETS)
def test_with_settled_noise_and_some_dips(name, off):
    """Noise as in the hold test after 1 s, with dips like the 0.3 s ones on top."""
    bad = 0
    for seed in range(100):
        c = Curve(name, seed, noise=0.02, dip=0.1)
        p = search(c, off)
        bad += not p.found or abs(p.depth) > 1.5
    assert bad <= 5


@pytest.mark.parametrize("off", [0, 1, -1])
def test_arriving_on_the_peak_is_quick(off):
    c = Curve("plain")
    p = search(c, off)
    assert p.samples <= 9
    assert c.deepest <= 4.0  # at most a few mm past the peak


@pytest.mark.parametrize("off", [-3, -2, -1, 0])
def test_never_more_than_a_bit_past_the_peak(off):
    c = Curve("line")
    search(c, off)
    assert c.deepest <= 4.0


@pytest.mark.parametrize("off", [1, 3, 5, 10, 15])
def test_arriving_too_deep_never_goes_deeper_by_much(off):
    """The accident of 2026-10-30: arriving past the peak must lead out, not in."""
    c = Curve("line")
    search(c, off)
    assert c.deepest <= off + 3.0


def test_a_dip_on_the_first_sample_does_not_lead_inwards():
    """A low first value used to cancel the 'only falling' check of the old search."""
    c = Curve("plain")
    values = iter([0.5])

    def measure(depth):
        v = c.measure(depth)
        return v * next(values, 1.0)

    p = find_peak(measure, 5.0, -15.0, 20.0)
    assert p.found and abs(p.depth) <= 1.0
    assert c.deepest <= 8.0


def test_flat_image_is_no_peak():
    p = find_peak(lambda d: 4.0, 0.0, -20.0, 8.0)
    assert not p.found


def test_stays_within_the_limits():
    seen = []

    def measure(d):
        seen.append(d)
        return 4.0

    find_peak(measure, 0.0, -5.0, 3.0)
    assert min(seen) >= -5.0 and max(seen) <= 3.0


def test_peak_too_far_in_is_reported_not_chased():
    c = Curve("line")
    p = search(c, -15, inward=15.0)  # the peak is right at the limit: its far side cannot be seen
    assert not p.found and "inner limit" in p.reason
    assert c.deepest <= 0.0


def test_peak_beyond_the_outer_limit_does_not_go_in():
    c = Curve("line")
    seen = []

    def measure(d):
        seen.append(d)
        return c.value(d)

    p = find_peak(measure, 12.0, 8.0, 20.0)  # peak at 0, out limit 8
    assert not p.found and "rising" in p.reason
    assert max(seen) <= 12.0


# Dial 40 on the robot, -y L14 middle, 2026-11-10 (autoscan_20261110_113003): first search, 2 mm steps,
# depth (mm, + = deeper) -> focus. Its peak is near -3.3; the inner side is mirrored.
DIAL40 = {-20: 0.69, -18: 0.71, -16: 0.76, -14: 0.86, -12: 1.07, -10: 1.69, -8: 3.85, -6: 11.36, -4: 22.92,
          -3.3: 30.0, -2: 16.86}


def dial40(depth: float) -> float:
    d = depth if depth <= -3.3 else -3.3 - (depth + 3.3)
    xs = sorted(DIAL40)
    return float(np.interp(d, xs, [DIAL40[x] for x in xs]))


@pytest.mark.parametrize("start", [-10.0, -8.0, -6.0])
def test_peak_further_in_does_not_walk_out_to_the_limit(start):
    """Dial 40 falls all the way out: two drops in a row end the outward walk (it used to walk
    5 more samples out to the limit)."""
    seen = []

    def measure(d):
        seen.append(d)
        return dial40(d)

    p = find_peak(measure, start, start - 10.0, start + 25.0, step=2.0, fall=0.85, falls=1)
    assert p.found and abs(p.depth + 3.3) <= 1.0
    assert min(seen) >= start - 2 * 2.0 - 1e-9  # two steps out, not five


@pytest.mark.parametrize("off", [-6, -5, -4])
def test_dial30_falling_side_also_stops_early(off):
    c = Curve("line")
    seen = []

    def measure(d):
        seen.append(d)
        return c.measure(d)

    p = find_peak(measure, float(off), off - 20.0, off + 20.0)
    assert p.found and abs(p.depth) <= 1.0
    assert min(seen) >= off - 2 * 1.0 - 1e-9


def test_dips_while_rising_out_never_stop_the_outward_walk():
    """From too deep the values rise going out; even dips of 0.8 cannot fake two drops."""
    for seed in range(200):
        c = Curve("line", seed, noise=0.02, dip=0.3)
        search(c, 8.0)
        assert c.deepest <= 8.0 + 3.0
