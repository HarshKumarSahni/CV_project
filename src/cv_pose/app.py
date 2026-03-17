from __future__ import annotations

import time

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None

from .pose import PoseEstimator, VISIBLE_NAMES
from .workout import WorkoutEngine

STATUS_COLORS = {
    "good": (72, 201, 176),
    "warn": (0, 191, 255),
    "bad": (52, 52, 235),
    "info": (200, 200, 200),
}

CONNECTIONS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_shoulder", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_shoulder", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
]

EXERCISE_KEYS = {
    ord("1"): "squat",
    ord("2"): "push-up",
    ord("3"): "plank",
}


def _exercise_label(name: str) -> str:
    return name.replace("-", " ").title()


def _to_pixel(frame_shape, x: float, y: float) -> tuple[int, int]:
    height, width = frame_shape[:2]
    px = int(max(0.0, min(1.0, x)) * width)
    py = int(max(0.0, min(1.0, y)) * height)
    return px, py


def draw_stickman(frame, pose_frame) -> None:
    if cv2 is None or pose_frame is None:
        return

    for start_name, end_name in CONNECTIONS:
        start = pose_frame.landmarks.get(start_name)
        end = pose_frame.landmarks.get(end_name)
        if start is None or end is None:
            continue
        if min(start.visibility, end.visibility) < 0.35:
            continue
        cv2.line(frame, _to_pixel(frame.shape, start.x, start.y), _to_pixel(frame.shape, end.x, end.y), (80, 255, 255), 2)

    for name in VISIBLE_NAMES:
        landmark = pose_frame.landmarks.get(name)
        if landmark is None or landmark.visibility < 0.35:
            continue
        cv2.circle(frame, _to_pixel(frame.shape, landmark.x, landmark.y), 4, (255, 255, 255), -1)


def draw_panel(frame, top_left: tuple[int, int], size: tuple[int, int], alpha: float = 0.5) -> None:
    if cv2 is None:
        return
    overlay = frame.copy()
    x, y = top_left
    width, height = size
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (24, 24, 24), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_lines(frame, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.6) -> None:
    if cv2 is None:
        return
    x, y = origin
    line_height = int(28 * scale)
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += line_height


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remaining:02d}"


def draw_hud(frame, engine: WorkoutEngine, result, saved_message: str = "") -> None:
    if cv2 is None:
        return

    state = engine.state
    status_color = STATUS_COLORS.get(result.feedback_level, STATUS_COLORS["info"])
    header_lines = [
        f"Exercise: {_exercise_label(engine.selected_exercise)}",
        f"Session: {'Running' if state.is_running else 'Idle'}",
        f"Phase: {state.phase}",
        f"Correct reps: {state.correct_reps}",
        f"Attempted reps: {state.attempted_reps}",
        f"Valid plank time: {format_duration(state.valid_hold_seconds)}",
        f"Elapsed: {format_duration(state.elapsed_seconds)}",
    ]
    guidance_lines = [
        "Controls:",
        "1 Squat  2 Push-up  3 Plank",
        "S Start/Stop session",
        "R Reset without saving",
        "Q Quit",
        "Camera: side view works best",
    ]

    draw_panel(frame, (16, 16), (360, 240), alpha=0.55)
    draw_panel(frame, (16, frame.shape[0] - 190), (430, 160), alpha=0.55)

    draw_lines(frame, header_lines, (30, 48), (240, 240, 240), scale=0.62)
    draw_lines(frame, guidance_lines, (30, frame.shape[0] - 154), (210, 210, 210), scale=0.55)

    feedback_text = [f"Feedback: {result.primary_feedback}"]
    if saved_message:
        feedback_text.append(saved_message)
    draw_lines(frame, feedback_text, (410, 48), status_color, scale=0.78)


def run_app(camera_index: int = 0) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install dependencies from requirements.txt first.")

    estimator = PoseEstimator()
    engine = WorkoutEngine()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        estimator.close()
        raise RuntimeError("Could not open webcam 0. Check camera permissions or change the camera index.")

    saved_message = ""
    saved_message_expires_at = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            timestamp = time.monotonic()
            pose_frame = estimator.process(frame, timestamp)
            result = engine.analyze_frame(pose_frame)

            draw_stickman(frame, pose_frame)
            if time.monotonic() > saved_message_expires_at:
                saved_message = ""
            draw_hud(frame, engine, result, saved_message)

            cv2.imshow("CV Pose Fitness Coach", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in EXERCISE_KEYS:
                next_exercise = EXERCISE_KEYS[key]
                if next_exercise != engine.selected_exercise:
                    saved = engine.switch_exercise(next_exercise)
                    if saved is not None:
                        saved_message = f"Saved: {saved.json_path}"
                        saved_message_expires_at = time.monotonic() + 6.0
                continue

            if key == ord("s"):
                if engine.state.is_running:
                    saved = engine.finish_session()
                    if saved is not None:
                        saved_message = f"Saved: {saved.json_path}"
                        saved_message_expires_at = time.monotonic() + 6.0
                else:
                    engine.start_session(engine.selected_exercise)
                continue

            if key == ord("r"):
                engine.reset_session()
                saved_message = "Session reset"
                saved_message_expires_at = time.monotonic() + 3.0
                continue

            if key == ord("q"):
                if engine.state.is_running:
                    saved = engine.finish_session()
                    if saved is not None:
                        saved_message = f"Saved: {saved.json_path}"
                break
    finally:
        capture.release()
        estimator.close()
        cv2.destroyAllWindows()


def main() -> None:
    run_app()
