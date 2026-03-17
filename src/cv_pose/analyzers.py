from __future__ import annotations

from dataclasses import dataclass

from .models import ExerciseResult, PoseFrame, SessionState

ISSUE_MESSAGES = {
    "pose_missing": "Step into a side view so shoulder, hip, knee, and ankle stay visible",
    "straighten_back": "Straighten back",
    "steady_knees": "Keep knees steadier",
    "go_lower": "Go lower",
    "keep_hips_level": "Keep hips level",
    "lift_hips": "Lift hips slightly",
    "lower_hips": "Lower hips slightly",
    "extend_fully": "Reset fully at the top",
}

ISSUE_LABELS = {
    "pose_missing": "Pose not visible",
    "straighten_back": "Straighten back",
    "steady_knees": "Steady knees",
    "go_lower": "More depth needed",
    "keep_hips_level": "Keep hips level",
    "lift_hips": "Lift hips slightly",
    "lower_hips": "Lower hips slightly",
    "extend_fully": "Reset fully at top",
}

ISSUE_LEVELS = {
    "pose_missing": "bad",
    "straighten_back": "bad",
    "steady_knees": "warn",
    "go_lower": "warn",
    "keep_hips_level": "bad",
    "lift_hips": "warn",
    "lower_hips": "warn",
    "extend_fully": "warn",
}


@dataclass
class ExerciseAnalyzer:
    exercise_name: str

    def analyze(self, pose_frame: PoseFrame, session_state: SessionState) -> ExerciseResult:
        raise NotImplementedError

    def _result(
        self,
        *,
        phase: str,
        form_valid: bool,
        feedback: str,
        feedback_level: str,
        rep_increment: int = 0,
        attempted_rep_increment: int = 0,
        hold_increment: float = 0.0,
        warnings: list[str] | None = None,
        diagnostics: dict | None = None,
        issue_code: str | None = None,
        event_type: str | None = None,
        event_message: str | None = None,
    ) -> ExerciseResult:
        return ExerciseResult(
            phase=phase,
            form_valid=form_valid,
            primary_feedback=feedback,
            feedback_level=feedback_level,
            rep_increment=rep_increment,
            attempted_rep_increment=attempted_rep_increment,
            hold_increment=hold_increment,
            warnings=warnings or [],
            diagnostics=diagnostics or {},
            issue_code=issue_code,
            event_type=event_type,
            event_message=event_message,
        )

    def _feedback(self, issue_code: str | None, default_feedback: str, form_valid: bool) -> tuple[str, str]:
        if issue_code:
            return ISSUE_MESSAGES[issue_code], ISSUE_LEVELS.get(issue_code, "warn")
        return default_feedback, "good" if form_valid else "info"


class SquatAnalyzer(ExerciseAnalyzer):
    descent_threshold = 135.0
    depth_threshold = 95.0
    stand_threshold = 160.0
    max_torso_tilt = 62.0
    max_knee_forward = 0.95

    def __init__(self) -> None:
        super().__init__(exercise_name="squat")

    def analyze(self, pose_frame: PoseFrame, session_state: SessionState) -> ExerciseResult:
        knee_angle = pose_frame.angle("knee", default=180.0) or 180.0
        hip_angle = pose_frame.angle("hip", default=180.0) or 180.0
        torso_tilt = pose_frame.metric("torso_tilt", default=90.0) or 90.0
        knee_forward = pose_frame.metric("knee_forward", default=0.0) or 0.0
        diagnostics = {
            "knee_angle": round(knee_angle, 2),
            "hip_angle": round(hip_angle, 2),
            "torso_tilt": round(torso_tilt, 2),
            "knee_forward": round(knee_forward, 2),
        }

        cycle_started = bool(session_state.context.get("cycle_started"))
        if knee_angle < self.descent_threshold:
            if not cycle_started:
                session_state.context["cycle_started"] = True
                session_state.context["cycle_form_valid"] = True
                session_state.context["min_primary_angle"] = knee_angle
                session_state.context["rejection_issue"] = None
            else:
                session_state.context["min_primary_angle"] = min(
                    session_state.context.get("min_primary_angle", knee_angle),
                    knee_angle,
                )

            if torso_tilt > self.max_torso_tilt:
                session_state.context["cycle_form_valid"] = False
                session_state.context["rejection_issue"] = session_state.context.get("rejection_issue") or "straighten_back"
            elif knee_forward > self.max_knee_forward:
                session_state.context["cycle_form_valid"] = False
                session_state.context["rejection_issue"] = session_state.context.get("rejection_issue") or "steady_knees"

        phase = "up" if knee_angle >= self.stand_threshold else "down"
        live_issue = None
        if torso_tilt > self.max_torso_tilt:
            live_issue = "straighten_back"
        elif knee_forward > self.max_knee_forward:
            live_issue = "steady_knees"
        elif phase == "down" and knee_angle > self.depth_threshold:
            live_issue = "go_lower"

        form_valid = live_issue is None

        if cycle_started and knee_angle >= self.stand_threshold:
            attempted_increment = 1
            min_knee = session_state.context.get("min_primary_angle", knee_angle)
            cycle_form_valid = bool(session_state.context.get("cycle_form_valid", True))
            rejection_issue = session_state.context.get("rejection_issue")
            if min_knee > self.depth_threshold:
                cycle_form_valid = False
                rejection_issue = rejection_issue or "go_lower"

            session_state.context["cycle_started"] = False
            session_state.context["cycle_form_valid"] = True
            session_state.context["min_primary_angle"] = 180.0
            session_state.context["rejection_issue"] = None

            if cycle_form_valid:
                return self._result(
                    phase="up",
                    form_valid=True,
                    feedback="Correct squat rep",
                    feedback_level="good",
                    rep_increment=1,
                    attempted_rep_increment=attempted_increment,
                    diagnostics=diagnostics,
                    event_type="rep_counted",
                    event_message="Correct squat rep counted",
                )

            rejection_issue = rejection_issue or live_issue or "go_lower"
            return self._result(
                phase="up",
                form_valid=False,
                feedback=ISSUE_MESSAGES[rejection_issue],
                feedback_level=ISSUE_LEVELS.get(rejection_issue, "warn"),
                attempted_rep_increment=attempted_increment,
                diagnostics=diagnostics,
                issue_code=rejection_issue,
                warnings=[ISSUE_MESSAGES[rejection_issue]],
                event_type="rep_rejected",
                event_message=f"Squat rep not counted: {ISSUE_LABELS[rejection_issue]}",
            )

        default_feedback = "Drive up" if phase == "down" and knee_angle <= self.depth_threshold else "Ready to squat"
        feedback, feedback_level = self._feedback(live_issue, default_feedback, form_valid)
        warnings = [ISSUE_MESSAGES[live_issue]] if live_issue else []
        return self._result(
            phase=phase,
            form_valid=form_valid,
            feedback=feedback,
            feedback_level=feedback_level,
            diagnostics=diagnostics,
            issue_code=live_issue,
            warnings=warnings,
        )


class PushUpAnalyzer(ExerciseAnalyzer):
    descent_threshold = 125.0
    depth_threshold = 90.0
    top_threshold = 155.0
    min_body_line = 160.0
    max_hip_offset = 0.3

    def __init__(self) -> None:
        super().__init__(exercise_name="push-up")

    def analyze(self, pose_frame: PoseFrame, session_state: SessionState) -> ExerciseResult:
        elbow_angle = pose_frame.angle("elbow", default=180.0) or 180.0
        body_line = pose_frame.angle("body_line", default=180.0) or 180.0
        hip_offset = pose_frame.metric("hip_height_offset", default=0.0) or 0.0
        diagnostics = {
            "elbow_angle": round(elbow_angle, 2),
            "body_line": round(body_line, 2),
            "hip_offset": round(hip_offset, 2),
        }

        cycle_started = bool(session_state.context.get("cycle_started"))
        if elbow_angle < self.descent_threshold:
            if not cycle_started:
                session_state.context["cycle_started"] = True
                session_state.context["cycle_form_valid"] = True
                session_state.context["min_primary_angle"] = elbow_angle
                session_state.context["rejection_issue"] = None
            else:
                session_state.context["min_primary_angle"] = min(
                    session_state.context.get("min_primary_angle", elbow_angle),
                    elbow_angle,
                )

            if body_line < self.min_body_line or abs(hip_offset) > self.max_hip_offset:
                session_state.context["cycle_form_valid"] = False
                session_state.context["rejection_issue"] = session_state.context.get("rejection_issue") or "keep_hips_level"

        phase = "up" if elbow_angle >= self.top_threshold else "down"
        live_issue = None
        if body_line < self.min_body_line:
            live_issue = "keep_hips_level"
        elif hip_offset > self.max_hip_offset:
            live_issue = "lift_hips"
        elif hip_offset < -self.max_hip_offset:
            live_issue = "lower_hips"
        elif phase == "down" and elbow_angle > self.depth_threshold:
            live_issue = "go_lower"

        form_valid = live_issue is None

        if cycle_started and elbow_angle >= self.top_threshold:
            attempted_increment = 1
            min_elbow = session_state.context.get("min_primary_angle", elbow_angle)
            cycle_form_valid = bool(session_state.context.get("cycle_form_valid", True))
            rejection_issue = session_state.context.get("rejection_issue")
            if min_elbow > self.depth_threshold:
                cycle_form_valid = False
                rejection_issue = rejection_issue or "go_lower"

            session_state.context["cycle_started"] = False
            session_state.context["cycle_form_valid"] = True
            session_state.context["min_primary_angle"] = 180.0
            session_state.context["rejection_issue"] = None

            if cycle_form_valid:
                return self._result(
                    phase="up",
                    form_valid=True,
                    feedback="Correct push-up rep",
                    feedback_level="good",
                    rep_increment=1,
                    attempted_rep_increment=attempted_increment,
                    diagnostics=diagnostics,
                    event_type="rep_counted",
                    event_message="Correct push-up rep counted",
                )

            rejection_issue = rejection_issue or live_issue or "go_lower"
            return self._result(
                phase="up",
                form_valid=False,
                feedback=ISSUE_MESSAGES[rejection_issue],
                feedback_level=ISSUE_LEVELS.get(rejection_issue, "warn"),
                attempted_rep_increment=attempted_increment,
                diagnostics=diagnostics,
                issue_code=rejection_issue,
                warnings=[ISSUE_MESSAGES[rejection_issue]],
                event_type="rep_rejected",
                event_message=f"Push-up rep not counted: {ISSUE_LABELS[rejection_issue]}",
            )

        default_feedback = "Press through" if phase == "down" and elbow_angle <= self.depth_threshold else "Ready for push-up"
        feedback, feedback_level = self._feedback(live_issue, default_feedback, form_valid)
        warnings = [ISSUE_MESSAGES[live_issue]] if live_issue else []
        return self._result(
            phase=phase,
            form_valid=form_valid,
            feedback=feedback,
            feedback_level=feedback_level,
            diagnostics=diagnostics,
            issue_code=live_issue,
            warnings=warnings,
        )


class PlankAnalyzer(ExerciseAnalyzer):
    min_body_line = 165.0
    max_hip_offset = 0.28

    def __init__(self) -> None:
        super().__init__(exercise_name="plank")

    def analyze(self, pose_frame: PoseFrame, session_state: SessionState) -> ExerciseResult:
        body_line = pose_frame.angle("body_line", default=180.0) or 180.0
        hip_offset = pose_frame.metric("hip_height_offset", default=0.0) or 0.0
        diagnostics = {
            "body_line": round(body_line, 2),
            "hip_offset": round(hip_offset, 2),
        }

        issue_code = None
        if hip_offset > self.max_hip_offset:
            issue_code = "lift_hips"
        elif hip_offset < -self.max_hip_offset:
            issue_code = "lower_hips"
        elif body_line < self.min_body_line:
            issue_code = "keep_hips_level"

        now = pose_frame.timestamp
        previous_timestamp = session_state.last_timestamp
        delta = max(0.0, now - previous_timestamp) if previous_timestamp is not None else 0.0
        form_valid = issue_code is None

        if form_valid:
            event_type = None
            event_message = None
            if session_state.phase != "hold":
                event_type = "hold_started"
                event_message = "Plank hold locked in"
            return self._result(
                phase="hold",
                form_valid=True,
                feedback="Hold steady",
                feedback_level="good",
                hold_increment=delta,
                diagnostics=diagnostics,
                event_type=event_type,
                event_message=event_message,
            )

        event_type = None
        event_message = None
        if session_state.phase == "hold":
            event_type = "hold_broken"
            event_message = f"Plank form broke: {ISSUE_LABELS[issue_code or 'keep_hips_level']}"
        return self._result(
            phase="broken",
            form_valid=False,
            feedback=ISSUE_MESSAGES[issue_code or "keep_hips_level"],
            feedback_level=ISSUE_LEVELS.get(issue_code or "keep_hips_level", "warn"),
            diagnostics=diagnostics,
            issue_code=issue_code or "keep_hips_level",
            warnings=[ISSUE_MESSAGES[issue_code or "keep_hips_level"]],
            event_type=event_type,
            event_message=event_message,
        )
