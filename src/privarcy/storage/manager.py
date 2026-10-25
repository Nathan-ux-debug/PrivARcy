"""Storage layout for the four PrivARcy data stores.

All under one data directory (see ``privarcy.paths.data_dir``):

    D1  Video File Store      <data>/jobs/<job_id>.json    metadata (source, size, fps, status, output, counts)
                              <data>/sources/              OPTIONAL imported copies (off by default)
        Redacted outputs      <data>/outputs/
    D3  Processed Frames      <data>/review_frames/<job_id>/   review crops, temporary
    D4  Correction Log        <data>/correction_log.db   (+ <data>/rules/ for rule versions)

Lifecycle rules
---------------
* D3 is temporary. It is deleted when the job's review is resolved, when the job
  is discarded, or after ``d3_retention_days`` (default 7) at startup.
* A job found as ``running`` at startup was interrupted: its manifest is marked
  ``interrupted`` and its partial output stays resumable.
* Raw OCR text is never written to any store.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class JobManifest:
    job_id: str
    source_path: str
    source_name: str
    status: str = "created"          # created | running | completed | cancelled | failed | interrupted
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    output_path: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0
    source_size_bytes: int = 0
    frames_processed: int = 0
    detections: int = 0
    redactions: int = 0
    flagged: int = 0
    pending_review: int = 0
    processing_time: float = 0.0
    avg_fps: float = 0.0
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class StorageManager:
    def __init__(self, data_dir: Path | str, d3_retention_days: float = 7.0, copy_sources: bool = False) -> None:
        self.data_dir = Path(data_dir)
        self.d3_retention_days = d3_retention_days
        self.copy_sources = copy_sources

        self.root = self.data_dir
        self.d1_dir = self.root
        self.d1_jobs = self.root / "jobs"
        self.d1_sources = self.root / "sources"
        self.d3_dir = self.root / "review_frames"
        self.rules_dir = self.root / "rules"
        self.outputs_dir = self.root / "outputs"
        self.live_measurements_path = self.root / "live_measurements.json"
        self.d4_path = self.root / "correction_log.db"
        for directory in (self.d1_jobs, self.d3_dir, self.rules_dir, self.outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if self.copy_sources:
            self.d1_sources.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ jobs
    def create_job(self, source_path: str | Path, probe: Optional[dict] = None) -> JobManifest:
        source = Path(source_path)
        job = JobManifest(job_id=uuid.uuid4().hex[:12], source_path=str(source.resolve()), source_name=source.name)
        try:
            job.source_size_bytes = source.stat().st_size
        except OSError:
            pass
        if probe:
            job.width, job.height = int(probe.get("width", 0)), int(probe.get("height", 0))
            job.fps, job.total_frames = float(probe.get("fps", 0.0)), int(probe.get("total_frames", 0))
        if self.copy_sources and source.is_file():
            target = self.d1_sources / f"{job.job_id}_{source.name}"
            shutil.copy2(source, target)
            job.source_path = str(target)
        self.save_job(job)
        return job

    def save_job(self, job: JobManifest) -> None:
        job.updated_at = time.time()
        path = self.d1_jobs / f"{job.job_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    def load_job(self, job_id: str) -> Optional[JobManifest]:
        path = self.d1_jobs / f"{job_id}.json"
        if not path.is_file():
            return None
        try:
            return JobManifest(**json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Unreadable job manifest: %s", path)
            return None

    def list_jobs(self, limit: int = 50) -> List[JobManifest]:
        jobs = [j for j in (self.load_job(p.stem) for p in self.d1_jobs.glob("*.json")) if j is not None]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def find_job_for_output(self, output_path: str) -> Optional[JobManifest]:
        for job in self.list_jobs(500):
            if job.output_path == str(output_path):
                return job
        return None

    # ------------------------------------------------------------------- D3
    def job_frames_dir(self, job_id: str) -> Path:
        path = self.d3_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_review_items(self, job_id: str, items: List[dict]) -> None:
        """Persist the review queue so it survives a restart. Items hold boxes, class,
        confidence and crop paths - never recognised text."""
        path = self.job_frames_dir(job_id) / "review_items.json"
        path.write_text(json.dumps(items), encoding="utf-8")

    def load_review_items(self, job_id: str) -> List[dict]:
        path = self.d3_dir / job_id / "review_items.json"
        if not path.is_file():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def set_review_status(self, job_id: str, item_id: str, status: str) -> int:
        """Mark one item confirmed/rejected; returns how many items remain pending."""
        items = self.load_review_items(job_id)
        for item in items:
            if item.get("id") == item_id:
                item["status"] = status
        self.save_review_items(job_id, items)
        return sum(1 for i in items if i.get("status") == "pending")

    def release_job_frames(self, job_id: str) -> bool:
        path = self.d3_dir / job_id
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            return True
        return False

    def d3_usage_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.d3_dir.rglob("*") if f.is_file())

    # -------------------------------------------------------------- lifecycle
    def startup_cleanup(self) -> dict:
        """Run once at app start: mark interrupted jobs, purge expired D3 data,
        and clear the legacy review_frames folder."""
        report = {"interrupted": 0, "d3_purged": 0, "legacy_review_frames_removed": 0}
        for job in self.list_jobs(1000):
            if job.status == "running":
                job.status = "interrupted"
                job.error = "Application closed while this job was running."
                self.save_job(job)
                report["interrupted"] += 1

        cutoff = time.time() - self.d3_retention_days * 86400
        for child in self.d3_dir.iterdir():
            if child.is_dir():
                job = self.load_job(child.name)
                expired = child.stat().st_mtime < cutoff
                if job is None or expired:
                    shutil.rmtree(child, ignore_errors=True)
                    report["d3_purged"] += 1

        legacy = self.outputs_dir / "review_frames"
        if legacy.is_dir():
            # These crops were written by the old pipeline next to the videos.
            # They are temporary artifacts (D3 material); the old pipeline's
            # review queue does not survive a restart, so they are orphaned.
            for file in legacy.glob("*.jpg"):
                try:
                    file.unlink()
                    report["legacy_review_frames_removed"] += 1
                except OSError:
                    pass
        return report

    def delete_job(self, job_id: str, delete_output: bool = False) -> None:
        job = self.load_job(job_id)
        self.release_job_frames(job_id)
        if job and delete_output and job.output_path:
            Path(job.output_path).unlink(missing_ok=True)
        (self.d1_jobs / f"{job_id}.json").unlink(missing_ok=True)

    def layout(self) -> dict:
        """Human-readable description of what lives where (shown in Settings)."""
        return {
            "Job manifests": str(self.d1_jobs),
            "Redacted outputs": str(self.outputs_dir),
            "Review frames (temporary)": str(self.d3_dir),
            "Correction log": str(self.d4_path),
        }
