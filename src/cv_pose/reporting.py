from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import SavedReport, SessionReport


class ReportWriter:
    def __init__(self, output_dir: str | Path = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, report: SessionReport) -> SavedReport:
        json_path = self.output_dir / f"{report.session_id}.json"
        csv_path = self.output_dir / "session_summary.csv"

        with json_path.open("w", encoding="utf-8") as json_file:
            json.dump(report.to_dict(), json_file, indent=2)

        row = report.csv_row()
        needs_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

        return SavedReport(report=report, json_path=str(json_path), csv_path=str(csv_path))
