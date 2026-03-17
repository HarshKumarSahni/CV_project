from __future__ import annotations

import math
from collections import defaultdict

from .models import Landmark


def distance(point_a: Landmark, point_b: Landmark) -> float:
    return math.hypot(point_b.x - point_a.x, point_b.y - point_a.y)


def calculate_angle(point_a: Landmark, point_b: Landmark, point_c: Landmark) -> float:
    ab_x = point_a.x - point_b.x
    ab_y = point_a.y - point_b.y
    cb_x = point_c.x - point_b.x
    cb_y = point_c.y - point_b.y
    angle = math.degrees(math.atan2(cb_y, cb_x) - math.atan2(ab_y, ab_x))
    angle = abs(angle)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def deviation_from_vertical(point_a: Landmark, point_b: Landmark) -> float:
    dx = point_b.x - point_a.x
    dy = point_b.y - point_a.y
    angle = abs(math.degrees(math.atan2(dy, dx)))
    if angle > 180.0:
        angle = angle % 180.0
    if angle > 90.0:
        angle = 180.0 - angle
    return abs(90.0 - angle)


def average_landmarks(samples: list[dict[str, Landmark]]) -> dict[str, Landmark]:
    if not samples:
        return {}

    sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0, "count": 0.0}
    )
    for sample in samples:
        for name, landmark in sample.items():
            sums[name]["x"] += landmark.x
            sums[name]["y"] += landmark.y
            sums[name]["z"] += landmark.z
            sums[name]["visibility"] += landmark.visibility
            sums[name]["count"] += 1.0

    averaged: dict[str, Landmark] = {}
    for name, total in sums.items():
        count = total["count"] or 1.0
        averaged[name] = Landmark(
            x=total["x"] / count,
            y=total["y"] / count,
            z=total["z"] / count,
            visibility=total["visibility"] / count,
        )
    return averaged
