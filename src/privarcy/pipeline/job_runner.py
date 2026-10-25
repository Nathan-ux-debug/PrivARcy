"""Runs one recorded-video job end to end and keeps D1/D3 consistent.

Responsibilities (previously scattered through the Qt worker):
  * register the job in D1 (manifest with real source metadata),
  * choose the output path and the resume state (cancel -> resume later),
  * give the pipeline a per-job D3 folder for review crops,
  * update the manifest on every outcome (completed / cancelled / failed),
  * clean partial files on failure.
It has no Qt dependency so it can be unit-tested and reused by a CLI.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2

logger = logging.getLogger(__name__)


@dataclass
class JobResult:
    job_id: str
    output_path: str
    frames: int
    redactions: int
    detections: list
    total_detections: int
    flagged: int
    processing_time: float
    total_frames: int
    avg_fps: float
    video_duration: float
    cancelled: bool = False
    summary: dict = field(default_factory=dict)
    class_map: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    resumed_from: int = 0


def probe_video(path: str | Path) -> dict:
    """Open the video once and report what OpenCV can actually read."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open this file as a video: {path}")
        info = {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "total_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        ok, _ = capture.read()
        if not ok:
            raise RuntimeError("The video opened but no frame could be decoded (unsupported codec?).")
        info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0.0
        return info
    finally:
        capture.release()


class JobRunner:
    def __init__(self, pipeline, storage) -> None:
        self.pipeline = pipeline
        self.storage = storage
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True
        self.pipeline.cancel()

    def run(self, video_path: str | Path, config=None, on_frame: Optional[Callable] = None,
            on_detections: Optional[Callable] = None, on_flagged: Optional[Callable] = None,
            on_redactions: Optional[Callable] = None, plan=None, face_policy=None) -> JobResult:
        self._cancel = False
        video_path = Path(video_path)
        probe = probe_video(video_path)
        job = self.storage.create_job(video_path, probe)

        outputs = self.storage.outputs_dir
        output_path = outputs / f"{video_path.stem}_redacted.mp4"
        resume_path = outputs / f"{video_path.stem}.resume.json"

        start_frame, prefix_output = 0, None
        if resume_path.exists() and output_path.exists():
            try:
                state = json.loads(resume_path.read_text(encoding="utf-8"))
                if state.get("input_path") == str(video_path.resolve()):
                    start_frame = int(state.get("next_frame", 0))
                    prefix_output = output_path
            except Exception:
                resume_path.unlink(missing_ok=True)
        processing_output = output_path.with_name(f"{output_path.stem}_resume.mp4") if prefix_output else output_path

        job.status, job.output_path = "running", str(output_path)
        self.storage.save_job(job)
        review_dir = self.storage.job_frames_dir(job.job_id)

        try:
            raw = self.pipeline.process_video(
                video_path, processing_output,
                redaction_method=getattr(config, "redaction_method", "Blur"),
                enabled_classes=getattr(config, "enabled_classes", None),
                start_frame=start_frame, prefix_output=prefix_output,
                frame_callback=on_frame, detection_callback=on_detections,
                flagged_callback=on_flagged, redaction_callback=on_redactions,
                cancel_requested=lambda: self._cancel, review_dir=review_dir,
                config=config, plan=plan, face_policy=face_policy,
            )
        except Exception as exc:
            job.status, job.error = "failed", str(exc)
            self.storage.save_job(job)
            self.storage.release_job_frames(job.job_id)
            if processing_output != output_path:
                Path(processing_output).unlink(missing_ok=True)
            elif not prefix_output:
                output_path.unlink(missing_ok=True)       # never leave a half-written file that looks finished
            raise

        cancelled = bool(raw["cancelled"])
        if cancelled:
            if raw["frames_processed"] == start_frame and prefix_output is None:
                processing_output.unlink(missing_ok=True)
                resume_path.unlink(missing_ok=True)
            else:
                if processing_output != output_path and processing_output.exists():
                    os.replace(processing_output, output_path)
                resume_path.write_text(json.dumps({
                    "input_path": str(video_path.resolve()), "next_frame": raw["frames_processed"],
                }), encoding="utf-8")
        else:
            if processing_output != output_path and processing_output.exists():
                os.replace(processing_output, output_path)
            resume_path.unlink(missing_ok=True)

        records = raw.get("detection_records", [])
        for record in records:
            record["output_path"] = str(output_path)
            record["job_id"] = job.job_id

        pending = sum(1 for r in records if r.get("status") == "pending")
        warnings: List[str] = []     # model class-map notes are static: shown in Settings, not repeated per job
        low = raw["summary"].get("low_confidence_unreviewed", {})
        if low:
            warnings.append(
                "Low-confidence detections (< 0.50) were not redacted and not reviewed: "
                + ", ".join(f"{k}: {v}" for k, v in sorted(low.items()))
                + ". Low confidence is not proof the frame is safe."
            )
        ocr = raw["summary"].get("ocr", {})
        if not ocr.get("available"):
            warnings.append("OCR was not available for this job: sensitive text on documents was not detected.")

        job.status = "cancelled" if cancelled else "completed"
        job.frames_processed = raw["frames_processed"]
        job.total_frames = raw["total_frames"] or job.total_frames
        job.detections, job.redactions, job.flagged = raw["detections"], raw["redactions"], raw["flagged"]
        job.pending_review = pending
        job.processing_time = raw["processing_time"]
        job.avg_fps = raw["frames_processed"] / max(raw["processing_time"], 1e-6)
        job.warnings, job.summary = warnings, raw["summary"]
        self.storage.save_job(job)
        if pending == 0:
            self.storage.release_job_frames(job.job_id)    # nothing to review -> no need to keep crops
        else:
            self.storage.save_review_items(job.job_id, [r for r in records if r.get("status") == "pending"])

        return JobResult(
            job_id=job.job_id, output_path=str(output_path), frames=raw["frames_processed"],
            redactions=raw["redactions"], detections=records, total_detections=raw["detections"],
            flagged=raw["flagged"], processing_time=raw["processing_time"], total_frames=raw["total_frames"],
            avg_fps=job.avg_fps, video_duration=raw["video_duration"], cancelled=cancelled,
            summary=raw["summary"], class_map=raw.get("class_map", {}), warnings=warnings, resumed_from=start_frame,
        )
