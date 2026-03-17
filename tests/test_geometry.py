import pytest

from cv_pose.geometry import calculate_angle, deviation_from_vertical
from cv_pose.models import Landmark
from cv_pose.pose import build_pose_frame


def test_calculate_angle_returns_right_angle() -> None:
    point_a = Landmark(0.0, 1.0)
    point_b = Landmark(0.0, 0.0)
    point_c = Landmark(1.0, 0.0)
    assert calculate_angle(point_a, point_b, point_c) == pytest.approx(90.0)


def test_deviation_from_vertical_is_small_for_upright_segment() -> None:
    lower = Landmark(0.0, 1.0)
    upper = Landmark(0.0, 0.0)
    assert deviation_from_vertical(lower, upper) == pytest.approx(0.0)


def test_build_pose_frame_derives_common_metrics() -> None:
    landmarks = {
        "left_shoulder": Landmark(0.20, 0.20),
        "left_elbow": Landmark(0.16, 0.34),
        "left_wrist": Landmark(0.12, 0.48),
        "left_hip": Landmark(0.22, 0.50),
        "left_knee": Landmark(0.24, 0.74),
        "left_ankle": Landmark(0.25, 0.94),
        "right_shoulder": Landmark(0.80, 0.20, visibility=0.1),
        "right_elbow": Landmark(0.84, 0.34, visibility=0.1),
        "right_wrist": Landmark(0.88, 0.48, visibility=0.1),
        "right_hip": Landmark(0.78, 0.50, visibility=0.1),
        "right_knee": Landmark(0.76, 0.74, visibility=0.1),
        "right_ankle": Landmark(0.75, 0.94, visibility=0.1),
    }

    frame = build_pose_frame(landmarks, timestamp=1.0)

    assert frame.dominant_side == "left"
    assert frame.angle("knee") is not None
    assert frame.angle("elbow") is not None
    assert frame.metric("torso_tilt") is not None
    assert frame.metric("knee_forward") is not None
