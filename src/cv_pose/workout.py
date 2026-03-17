from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from .analyzers import ISSUE_LABELS, ISSUE_MESSAGES, PlankAnalyzer, PushUpAnalyzer, SquatAnalyzer
from .models import ExerciseResult, PoseFrame, SavedReport, SessionEvent, SessionReport, SessionState, utc_now
from .reporting import ReportWriter


class WorkoutEngine:
    def __init__(self, report_writer: ReportWriter | None = None) -> None:
        self.report_writer = report_writer or ReportWriter(Path("reports"))
        self.analyzers = {
            "squat": SquatAnalyzer(),
            "push-up": PushUpAnalyzer(),
            "plank": PlankAnalyzer(),
        }
        self.selected_exercise = "squat"
        self.state = self._build_idle_state(self.selected_exercise)

    def _build_idle_state(self, exercise: str) -> SessionState:
        return SessionState(
            exercise=exercise,
            active_feedback=f"Press S to start {exercise}",
            feedback_level="info",
        )

    def switch_exercise(self, exercise: str) -> SavedReport | None:
        if exercise not in self.analyzers:
            raise ValueError(f"Unsupported exercise: {exercise}")

        saved_report = None
        if self.state.is_running and self.state.exercise != exercise:
            saved_report = self.finish_session()

        self.selected_exercise = exercise
        self.state = self._build_idle_state(exercise)
        return saved_report

    def start_session(self, exercise: str | None = None) -> SessionState:
        exercise_name = exercise or self.selected_exercise
        if exercise_name not in self.analyzers:
            raise ValueError(f"Unsupported exercise: {exercise_name}")

        self.selected_exercise = exercise_name
        self.state = SessionState(
            exercise=exercise_name,
            phase="ready",
            is_running=True,
            started_at=utc_now(),
            active_feedback=f"{exercise_name.title()} session started",
            feedback_level="info",
            context={
                "camera_view": "side",
                "feedback_mode": "text+color",
                "app_version": "0.1.0",
            },
        )
        self.log_event("session_started", f"{exercise_name.title()} session started", 0.0)
        return self.state

    def analyze_frame(self, pose_frame: PoseFrame | None) -> ExerciseResult:
        if not self.state.is_running:
            return self.preview_feedback(pose_frame)

        if pose_frame is None or pose_frame.confidence < 0.35:
            self.state.active_feedback = ISSUE_MESSAGES["pose_missing"]
            self.state.feedback_level = "bad"
            self.state.active_warnings = [ISSUE_MESSAGES["pose_missing"]]
            self._track_issue("pose_missing", None)
            self.state.last_timestamp = None
            return ExerciseResult(
                phase=self.state.phase,
                form_valid=False,
                primary_feedback=ISSUE_MESSAGES["pose_missing"],
                feedback_level="bad",
                warnings=[ISSUE_MESSAGES["pose_missing"]],
                issue_code="pose_missing",
            )

        analyzer = self.analyzers[self.state.exercise]
        result = analyzer.analyze(pose_frame, self.state)
        self._apply_result(result, pose_frame.timestamp)
        self.state.last_timestamp = pose_frame.timestamp
        return result

    def preview_feedback(self, pose_frame: PoseFrame | None) -> ExerciseResult:
        if pose_frame is None or pose_frame.confidence < 0.35:
            return ExerciseResult(
                phase="idle",
                form_valid=False,
                primary_feedback=ISSUE_MESSAGES["pose_missing"],
                feedback_level="bad",
                warnings=[ISSUE_MESSAGES["pose_missing"]],
                issue_code="pose_missing",
            )

        return ExerciseResult(
            phase="idle",
            form_valid=True,
            primary_feedback=f"Press S to start {self.selected_exercise}",
            feedback_level="info",
            diagnostics=pose_frame.metrics,
        )

    def finish_session(self) -> SavedReport | None:
        if not self.state.is_running:
            return None

        self.state.ended_at = utc_now()
        report = self._build_report()
        saved = self.report_writer.write(report)
        exercise_name = self.state.exercise
        self.state = self._build_idle_state(exercise_name)
        return saved

    def reset_session(self) -> None:
        exercise_name = self.state.exercise
        self.state = self._build_idle_state(exercise_name)

    def log_event(self, event_type: str, message: str, timestamp: float, diagnostics: dict | None = None) -> None:
        offset = 0.0
        first_frame_timestamp = self.state.context.get("first_frame_timestamp")
        if first_frame_timestamp is not None:
            offset = max(0.0, timestamp - first_frame_timestamp)
        self.state.event_log.append(
            SessionEvent(
                timestamp_seconds=round(offset, 2),
                event_type=event_type,
                message=message,
                diagnostics=diagnostics or {},
            )
        )

    def _apply_result(self, result: ExerciseResult, timestamp: float) -> None:
        if self.state.context.get("first_frame_timestamp") is None:
            self.state.context["first_frame_timestamp"] = timestamp
        self.state.context["last_frame_timestamp"] = timestamp
        first_frame_timestamp = self.state.context.get("first_frame_timestamp", timestamp)
        self.state.elapsed_seconds = max(0.0, timestamp - first_frame_timestamp)

        previous_phase = self.state.phase
        self.state.phase = result.phase
        self.state.correct_reps += result.rep_increment
        self.state.attempted_reps += result.attempted_rep_increment
        self.state.valid_hold_seconds += result.hold_increment
        self.state.active_feedback = result.primary_feedback
        self.state.feedback_level = result.feedback_level
        self.state.active_warnings = result.warnings

        if previous_phase != result.phase:
            self.log_event("phase_changed", f"{previous_phase} -> {result.phase}", timestamp)

        self._track_issue(result.issue_code, timestamp)

        if result.event_type:
            self.log_event(result.event_type, result.event_message or result.primary_feedback, timestamp, result.diagnostics)

    def _track_issue(self, issue_code: str | None, timestamp: float | None) -> None:
        previous_issue = self.state.context.get("last_issue_code")
        if issue_code == previous_issue:
            return

        self.state.context["last_issue_code"] = issue_code
        if issue_code is None:
            return

        self.state.error_counts[issue_code] = self.state.error_counts.get(issue_code, 0) + 1
        if timestamp is not None:
            self.log_event("feedback", ISSUE_MESSAGES.get(issue_code, issue_code), timestamp, {"issue_code": issue_code})

    def _build_report(self) -> SessionReport:
        started_at = self.state.started_at or utc_now()
        ended_at = self.state.ended_at or utc_now()
        duration = self.state.elapsed_seconds
        if duration <= 0.0:
            duration = max(0.0, (ended_at - started_at).total_seconds())

        top_mistakes = [
            {
                "issue_code": issue_code,
                "label": ISSUE_LABELS.get(issue_code, issue_code.replace("_", " ").title()),
                "count": count,
            }
            for issue_code, count in sorted(self.state.error_counts.items(), key=lambda item: (-item[1], item[0]))
        ]

        correct = self.state.correct_reps
        attempted = self.state.attempted_reps
        success_rate = (correct / attempted * 100.0) if attempted else 0.0

        return SessionReport(
            session_id=f"{self.state.exercise}-{uuid4().hex[:8]}",
            exercise=self.state.exercise,
            started_at=started_at.isoformat(),
            ended_at=ended_at.isoformat(),
            duration_seconds=round(duration, 2),
            attempted_reps=attempted,
            correct_reps=correct,
            success_rate=round(success_rate, 2),
            valid_hold_seconds=round(self.state.valid_hold_seconds, 2),
            top_posture_mistakes=top_mistakes,
            event_log=[event.to_dict() for event in self.state.event_log],
            metadata={
                "camera_view": "side",
                "feedback_mode": "text+color",
                "app_version": self.state.context.get("app_version", "0.1.0"),
            },
        )
