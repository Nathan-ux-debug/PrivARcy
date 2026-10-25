"""D4 - Correction Log: persistent storage for human reviewer decisions."""

import datetime
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from .exceptions import DataStoreError
from .models import CorrectionRecord

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id TEXT NOT NULL,
    bbox TEXT NOT NULL,
    decision TEXT NOT NULL,
    detected_class TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    context TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_frame ON corrections(frame_id);
CREATE INDEX IF NOT EXISTS idx_corrections_decision ON corrections(decision);
"""

VALID_DECISIONS = {"accepted", "rejected"}


class CorrectionLogDB:
    """SQLite-backed store for human review decisions on flagged detections (D4).

    Parameters:
        db_path: Path to the SQLite file. Created if it doesn't exist.
                  Pass ":memory:" for an ephemeral in-process database.
    """

    def __init__(self, db_path: str | Path = "correction_log.db") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record_decision(
        self,
        frame_id: str,
        bbox: List[float],
        decision: str,
        detected_class: str = "",
        confidence: float = 0.0,
        context: str = "",
        reviewer: str = "",
    ) -> int:
        """Record one reviewer decision. Returns the new row id.

        This is the function a frontend "record review action" endpoint
        calls directly — see api.py for a thin FastAPI wrapper around it.

        Raises:
            DataStoreError: if decision isn't "accepted"/"rejected", or
                             bbox doesn't have exactly 4 values.
        """
        if decision not in VALID_DECISIONS:
            raise DataStoreError(f"decision must be one of {VALID_DECISIONS}, got '{decision}'")
        if not frame_id:
            raise DataStoreError("frame_id must be non-empty.")
        if len(bbox) != 4:
            raise DataStoreError(f"bbox must have exactly 4 values (x_min,y_min,x_max,y_max), got {len(bbox)}")

        record = CorrectionRecord(
            frame_id=str(frame_id),
            bbox=list(bbox),
            decision=decision,
            detected_class=detected_class,
            confidence=float(confidence),
            context=context,
            reviewer=reviewer,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        try:
            cursor = self._conn.execute(
                """INSERT INTO corrections
                   (frame_id, bbox, decision, detected_class, confidence, context, reviewer, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (record.frame_id, json.dumps(record.bbox), record.decision, record.detected_class,
                 record.confidence, record.context, record.reviewer, record.created_at),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise DataStoreError(f"Failed to record correction: {exc}") from exc

        logger.info("Recorded '%s' decision for frame '%s' (row id=%d).", decision, frame_id, cursor.lastrowid)
        return cursor.lastrowid

    def get_decisions(
        self,
        frame_id: Optional[str] = None,
        decision: Optional[str] = None,
        limit: int = 100,
    ) -> List[CorrectionRecord]:
        """Query recorded decisions, optionally filtered by frame or decision type."""
        query = "SELECT id, frame_id, bbox, decision, detected_class, confidence, context, reviewer, created_at FROM corrections WHERE 1=1"
        params: list = []
        if frame_id is not None:
            query += " AND frame_id = ?"
            params.append(str(frame_id))
        if decision is not None:
            if decision not in VALID_DECISIONS:
                raise DataStoreError(f"decision must be one of {VALID_DECISIONS}, got '{decision}'")
            query += " AND decision = ?"
            params.append(decision)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        try:
            rows = self._conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise DataStoreError(f"Failed to query corrections: {exc}") from exc

        return [
            CorrectionRecord(
                id=r[0], frame_id=r[1], bbox=json.loads(r[2]), decision=r[3],
                detected_class=r[4], confidence=r[5], context=r[6], reviewer=r[7], created_at=r[8],
            )
            for r in rows
        ]

    def count(self, decision: Optional[str] = None) -> int:
        if decision is None:
            return self._conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        return self._conn.execute("SELECT COUNT(*) FROM corrections WHERE decision = ?", (decision,)).fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CorrectionLogDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
