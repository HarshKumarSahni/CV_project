from __future__ import annotations

from collections import deque

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional at import time
    cv2 = None

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover - optional at import time
    mp = None

from .geometry import average_landmarks, calculate_angle, deviation_from_vertical, distance
from .models import Landmark, PoseFrame

SIDES = ("left", "right")
POSE_JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "ear")
VISIBLE_NAMES = [f"{side}_{joint}" for side in SIDES for joint in POSE_JOINTS]


def select_dominant_side(landmarks: dict[str, Landmark]) -> str:
    side_visibility = {}
    for side in SIDES:
        side_visibility[side] = sum(
            landmarks.get(f"{side}_{joint}", Landmark(0.0, 0.0, visibility=0.0)).visibility
            for joint in ("shoulder", "hip", "knee", "ankle", "elbow")
        )
    return max(side_visibility, key=side_visibility.get, default="left")


def _joint_angle(landmarks: dict[str, Landmark], side: str, joint_a: str, joint_b: str, joint_c: str) -> float | None:
    names = (f"{side}_{joint_a}", f"{side}_{joint_b}", f"{side}_{joint_c}")
    if not all(name in landmarks for name in names):
        return None
    return calculate_angle(landmarks[names[0]], landmarks[names[1]], landmarks[names[2]])


def derive_pose_metrics(landmarks: dict[str, Landmark]) -> tuple[dict[str, float], dict[str, float]]:
    angles: dict[str, float] = {}
    metrics: dict[str, float] = {}

    for side in SIDES:
        shoulder_name = f"{side}_shoulder"
        hip_name = f"{side}_hip"
        knee_name = f"{side}_knee"
        ankle_name = f"{side}_ankle"

        elbow_angle = _joint_angle(landmarks, side, "shoulder", "elbow", "wrist")
        if elbow_angle is not None:
            angles[f"elbow_{side}"] = elbow_angle

        hip_angle = _joint_angle(landmarks, side, "shoulder", "hip", "knee")
        if hip_angle is not None:
            angles[f"hip_{side}"] = hip_angle

        knee_angle = _joint_angle(landmarks, side, "hip", "knee", "ankle")
        if knee_angle is not None:
            angles[f"knee_{side}"] = knee_angle

        body_line = _joint_angle(landmarks, side, "shoulder", "hip", "ankle")
        if body_line is not None:
            angles[f"body_line_{side}"] = body_line

        if shoulder_name in landmarks and hip_name in landmarks:
            metrics[f"torso_tilt_{side}"] = deviation_from_vertical(landmarks[hip_name], landmarks[shoulder_name])
            metrics[f"torso_length_{side}"] = distance(landmarks[hip_name], landmarks[shoulder_name])

        if knee_name in landmarks and ankle_name in landmarks:
            metrics[f"shin_tilt_{side}"] = deviation_from_vertical(landmarks[knee_name], landmarks[ankle_name])

        if knee_name in landmarks and ankle_name in landmarks and shoulder_name in landmarks and hip_name in landmarks:
            torso_length = max(distance(landmarks[hip_name], landmarks[shoulder_name]), 1e-3)
            metrics[f"knee_forward_{side}"] = abs(landmarks[knee_name].x - landmarks[ankle_name].x) / torso_length

        if shoulder_name in landmarks and hip_name in landmarks and ankle_name in landmarks:
            torso_length = max(distance(landmarks[hip_name], landmarks[shoulder_name]), 1e-3)
            shoulder_ankle_mid_y = (landmarks[shoulder_name].y + landmarks[ankle_name].y) / 2.0
            metrics[f"hip_height_offset_{side}"] = (landmarks[hip_name].y - shoulder_ankle_mid_y) / torso_length

    return angles, metrics


def build_pose_frame(landmarks: dict[str, Landmark], timestamp: float, confidence: float | None = None) -> PoseFrame:
    if confidence is None:
        visibility_values = [landmark.visibility for landmark in landmarks.values()]
        confidence = sum(visibility_values) / len(visibility_values) if visibility_values else 0.0
    dominant_side = select_dominant_side(landmarks)
    angles, metrics = derive_pose_metrics(landmarks)
    return PoseFrame(
        timestamp=timestamp,
        landmarks=landmarks,
        confidence=confidence,
        dominant_side=dominant_side,
        angles=angles,
        metrics=metrics,
    )


class PoseEstimator:
    """Thin wrapper around MediaPipe pose estimation with smoothing."""

    MP_LANDMARKS = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_ear": 7,
        "right_ear": 8,
    }

    def __init__(
        self,
        smoothing_window: int = 5,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if cv2 is None or mp is None:
            raise RuntimeError(
                "OpenCV and MediaPipe are required to run the webcam app. Install dependencies from requirements.txt."
            )
        if not hasattr(mp, "solutions"):
            version = getattr(mp, "__version__", "unknown")
            raise RuntimeError(
                "This app uses MediaPipe's legacy pose API, but the installed version "
                f"({version}) does not expose 'mediapipe.solutions'. "
                "Reinstall with a compatible version, for example: "
                "'pip uninstall -y mediapipe && pip install \"mediapipe==0.10.14\"'."
            )

        self._pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._history: deque[dict[str, Landmark]] = deque(maxlen=smoothing_window)

    def process(self, frame_bgr, timestamp: float) -> PoseFrame | None:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb_frame)
        if not result.pose_landmarks:
            self._history.clear()
            return None

        landmarks: dict[str, Landmark] = {}
        for name, index in self.MP_LANDMARKS.items():
            point = result.pose_landmarks.landmark[index]
            landmarks[name] = Landmark(
                x=float(point.x),
                y=float(point.y),
                z=float(point.z),
                visibility=float(point.visibility),
            )

        self._history.append(landmarks)
        smoothed = average_landmarks(list(self._history))
        return build_pose_frame(smoothed, timestamp)

    def close(self) -> None:
        self._pose.close()

