"""Distance of the arm, hand, fingers and Dino to the tower box along a joint move, checked before it is sent.

The joint moves to and from the faces are straight in joint space, not in space: on 2026-11-19 (teammate's PC)
one that started at the -y face with the wrist the other way round swept the tower over (in the model the Dino
goes 9-12 mm into the tower box on it). Every joint move of the scan is now sampled first; closer than
MIN_CLEAR_MM -> ScanError, nothing is sent. The box is the model tower (scan_setup.json tower) and the parts are
rough (jenga_vision.sim: palm box, finger lines, Dino cylinder; arm links from the elbow on as LINK_R cylinders).
"""

from __future__ import annotations

import numpy as np

from jenga_vision.kinematics import Chain
from jenga_vision.sim import clearances, geometry, joint_points, tower_distance_mm

MIN_CLEAR_MM = 5.0
LINK_R = 0.05  # xArm forearm / wrist radius, rough
SAMPLES = 40


class Clearance:
    def __init__(self, arm: str, tower, tool: np.ndarray, hand_q):
        self.arm, self.tower, self.tool, self.hand_q = arm, tower, tool, hand_q
        self.chain = Chain(arm)

    def at(self, q) -> tuple[float, str]:
        """(mm, part) of the closest part at joints `q`."""
        q = np.asarray(q, float)
        c = clearances(self.tower, geometry(self.chain, self.tool, self.arm, q, self.hand_q))
        pts = joint_points(self.chain, q)
        seg = np.concatenate([np.linspace(a, b, 6) for a, b in zip(pts[3:], pts[4:])])  # elbow onwards
        c["arm"] = tower_distance_mm(self.tower, seg) - LINK_R * 1000
        part = min(c, key=c.get)
        return c[part], part

    def along(self, a, b) -> tuple[float, str, float]:
        """(mm, part, fraction of the move) of the closest point of the straight joint move a -> b."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        worst = (float("inf"), "", 0.0)
        for s in np.linspace(0.0, 1.0, SAMPLES):
            mm, part = self.at(a + (b - a) * s)
            if mm < worst[0]:
                worst = (mm, part, float(s))
        return worst
