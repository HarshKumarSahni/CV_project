from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class PoseFrame:
    timestamp: float
    landmarks: dict[str, Landmark]
    confidence: float
    dominant_side: str
    angles: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def angle(self, name: str, side: str | None = None, default: float | None = None) -> float | None:
        key = name if name in self.angles else f"{name}_{side or self.dominant_side}"
        return self.angles.get(key, default)

    def metric(self, name: str, side: str | None = None, default: float | None = None) -> float | None:
        key = name if name in self.metrics else f"{name}_{side or self.dominant_side}"
        return self.metrics.get(key, default)


@dataclass
class ExerciseResult:
    phase: str
    form_valid: bool
    primary_feedback: str
    feedback_level: str
    rep_increment: int = 0
    attempted_rep_increment: int = 0
    hold_increment: float = 0.0
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    issue_code: str | None = None
    event_type: str | None = None
    event_message: str | None = None


@dataclass
class SessionEvent:
    timestamp_seconds: float
    event_type: str
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionState:
    exercise: str
    phase: str = "idle"
    attempted_reps: int = 0
    correct_reps: int = 0
    valid_hold_seconds: float = 0.0
    active_feedback: str = "Press S to start"
    feedback_level: str = "info"
    active_warnings: list[str] = field(default_factory=list)
    error_counts: dict[str, int] = field(default_factory=dict)
    event_log: list[SessionEvent] = field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    is_running: bool = False
    elapsed_seconds: float = 0.0
    last_timestamp: float | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionReport:
    session_id: str
    exercise: str
    started_at: str
    ended_at: str
    duration_seconds: float
    attempted_reps: int
    correct_reps: int
    success_rate: float
    valid_hold_seconds: float
    top_posture_mistakes: list[dict[str, Any]]
    event_log: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def csv_row(self) -> dict[str, Any]:
        top_issue = self.top_posture_mistakes[0] if self.top_posture_mistakes else {}
        return {
            "session_id": self.session_id,
            "exercise": self.exercise,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "attempted_reps": self.attempted_reps,
            "correct_reps": self.correct_reps,
            "success_rate": round(self.success_rate, 2),
            "valid_hold_seconds": round(self.valid_hold_seconds, 2),
            "top_issue": top_issue.get("label", ""),
            "top_issue_count": top_issue.get("count", 0),
        }


@dataclass
class SavedReport:
    report: SessionReport
    json_path: str
    csv_path: str
