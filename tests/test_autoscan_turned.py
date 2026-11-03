"""A face seen with the tower turned 180 deg counts as the far ends of the same blocks."""

from jenga_vision import ends
from jenga_vision.autoscan import FACE_NUMBER


def opposite(face: str) -> int:
    return (FACE_NUMBER[face] + 1) % 4 + 1


def test_opposite_faces():
    assert opposite("+x") == FACE_NUMBER["-x"]
    assert opposite("-x") == FACE_NUMBER["+x"]
    assert opposite("+y") == FACE_NUMBER["-y"]
    assert opposite("-y") == FACE_NUMBER["+y"]


def test_screen_right_of_the_turned_tower_is_the_left_block():
    shots = {"L04:f1:left": 7.0, "L04:f1:middle": 5.0, "L04:f1:right": 0.0,
             "L04:f3:left": 1.0, "L04:f3:middle": 2.0, "L04:f3:right": 3.0}
    g = ends.end_gaps(shots, 4, 2)
    assert g["near"] == {"left": 7.0, "middle": 5.0, "right": 0.0}
    assert g["far"] == {"left": 3.0, "middle": 2.0, "right": 1.0}
