import json

import pytest

from cv_pose.models import PoseFrame
from cv_pose.reporting import ReportWriter
from cv_pose.workout import WorkoutEngine


@pytest.fixture()
def engine(tmp_path):
    return WorkoutEngine(report_writer=ReportWriter(tmp_path))


def make_frame(
    timestamp: float,
    *,
    knee: float = 170.0,
    hip: float = 160.0,
    elbow: float = 170.0,
    body_line: float = 175.0,
    torso_tilt: float = 18.0,
    knee_forward: float = 0.22,
    hip_offset: float = 0.0,
    confidence: float = 1.0,
) -> PoseFrame:
    return PoseFrame(
        timestamp=timestamp,
        landmarks={},
        confidence=confidence,
        dominant_side="left",
        angles={
            "knee_left": knee,
            "hip_left": hip,
            "elbow_left": elbow,
            "body_line_left": body_line,
        },
        metrics={
            "torso_tilt_left": torso_tilt,
            "knee_forward_left": knee_forward,
            "hip_height_offset_left": hip_offset,
        },
    )


def test_squat_counts_only_correct_rep(engine) -> None:
    engine.start_session("squat")
    for frame in [
        make_frame(0.0, knee=170.0),
        make_frame(0.4, knee=130.0),
        make_frame(0.8, knee=88.0, torso_tilt=25.0, knee_forward=0.3),
        make_frame(1.2, knee=170.0, torso_tilt=18.0),
    ]:
        engine.analyze_frame(frame)

    assert engine.state.attempted_reps == 1
    assert engine.state.correct_reps == 1
    assert engine.state.active_feedback == "Correct squat rep"


def test_squat_rejects_bad_back_posture(engine) -> None:
    engine.start_session("squat")
    for frame in [
        make_frame(0.0, knee=170.0),
        make_frame(0.5, knee=126.0, torso_tilt=70.0),
        make_frame(1.0, knee=90.0, torso_tilt=72.0),
        make_frame(1.5, knee=170.0, torso_tilt=18.0),
    ]:
        engine.analyze_frame(frame)

    assert engine.state.attempted_reps == 1
    assert engine.state.correct_reps == 0
    assert engine.state.error_counts["straighten_back"] >= 1


def test_pushup_requires_body_alignment(engine) -> None:
    engine.start_session("push-up")
    for frame in [
        make_frame(0.0, elbow=170.0, body_line=175.0),
        make_frame(0.5, elbow=120.0, body_line=150.0, hip_offset=0.35),
        make_frame(1.0, elbow=85.0, body_line=148.0, hip_offset=0.38),
        make_frame(1.4, elbow=170.0, body_line=176.0),
    ]:
        engine.analyze_frame(frame)

    assert engine.state.attempted_reps == 1
    assert engine.state.correct_reps == 0
    assert any(code in engine.state.error_counts for code in ("keep_hips_level", "lift_hips"))


def test_plank_tracks_only_valid_hold_time(engine) -> None:
    engine.start_session("plank")
    for frame in [
        make_frame(0.0, body_line=176.0, hip_offset=0.0),
        make_frame(1.0, body_line=176.0, hip_offset=0.0),
        make_frame(2.0, body_line=176.0, hip_offset=0.0),
        make_frame(3.0, body_line=150.0, hip_offset=0.42),
    ]:
        engine.analyze_frame(frame)

    assert engine.state.valid_hold_seconds == pytest.approx(2.0)
    assert engine.state.phase == "broken"


def test_finish_session_writes_json_and_csv(engine, tmp_path) -> None:
    engine.start_session("squat")
    for frame in [
        make_frame(0.0, knee=170.0),
        make_frame(0.4, knee=130.0),
        make_frame(0.8, knee=88.0, torso_tilt=25.0, knee_forward=0.3),
        make_frame(1.2, knee=170.0, torso_tilt=18.0),
    ]:
        engine.analyze_frame(frame)

    saved = engine.finish_session()

    assert saved is not None
    assert tmp_path.joinpath(saved.report.session_id + ".json").exists()
    assert tmp_path.joinpath("session_summary.csv").exists()

    data = json.loads(tmp_path.joinpath(saved.report.session_id + ".json").read_text(encoding="utf-8"))
    assert data["correct_reps"] == 1
    assert data["attempted_reps"] == 1
