# backend/pipeline/video_pipeline.py
"""Recorded-video processing.

Per-frame logic lives in :class:`FrameEngine` (shared with the live camera);
this class owns the video I/O: open -> loop -> write -> progress callbacks ->
cancel / resume, and returns the same result dictionary as before (plus new
``summary`` / ``class_map`` keys).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2

from privarcy.paths import model_path
from .config import (
    AUTO_REDACT_CONFIDENCE, REVIEW_MIN_CONFIDENCE, PipelineConfig, box_iou, model_class_names, normalize_method,
    verify_class_map,
)
from .frame_engine import FrameEngine
from .redact_ops import redact_region

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = model_path()


class VideoPipeline:
    """Complete PrivARcy video-processing pipeline.

    Video -> frames -> FrameEngine (YOLO, decisions, face policy, OCR text,
    smoothing/tracking, redaction) -> reconstructed video.
    """

    # Kept for backwards compatibility with older callers.
    CLASS_NAMES = {0: "id_card", 1: "document", 2: "screen", 3: "face", 4: "credit_card", 5: "license_plate"}
    DEFAULT_ENABLED_CLASSES = {"id_card", "screen", "face", "credit_card", "license_plate"}
    AUTO_REDACT_CONFIDENCE = AUTO_REDACT_CONFIDENCE
    REVIEW_MIN_CONFIDENCE = REVIEW_MIN_CONFIDENCE

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        confidence_threshold=0.80,
        face_policy=None,
        review_iou_threshold=0.50,
        review_track_max_gap=5,
        temporal_smoothing=3,
        model=None,
        config: PipelineConfig | None = None,
        ocr_provider=None,
    ):
        self.model_path = str(model_path)
        self.face_policy = face_policy
        self.ocr_provider = ocr_provider

        self.config = config or PipelineConfig()
        if config is None:
            self.config.iou_threshold = float(review_iou_threshold)
            self.config.track_max_gap = int(review_track_max_gap)
            self.config.temporal_smoothing = max(1, int(temporal_smoothing))
        self.confidence_threshold = float(confidence_threshold)
        self.review_iou_threshold = self.config.iou_threshold
        self.review_track_max_gap = self.config.track_max_gap
        self.temporal_smoothing = self.config.temporal_smoothing

        self._cancel_requested = False
        self.engine = None

        if model is None:
            from ultralytics import YOLO

            logger.info("Loading YOLO model: %s", self.model_path)
            model = YOLO(self.model_path)
        self.model = model

        self.class_map = verify_class_map(model_class_names(self.model))
        for warning in self.class_map["warnings"]:
            logger.warning("Class map: %s", warning)
        logger.info("Model classes: %s", self.class_map["names"])

    # ------------------------------------------------------------------
    def cancel(self):
        self._cancel_requested = True

    def reset_cancel(self):
        self._cancel_requested = False

    # ------------------------------------------------------------------
    def process_video(
        self,
        input_path,
        output_path,
        redaction_method="Blur",
        enabled_classes=None,
        low_threshold=0.50,
        start_frame=0,
        prefix_output=None,
        progress_callback=None,
        frame_callback=None,
        fps_callback=None,
        eta_callback=None,
        detection_callback=None,
        flagged_callback=None,
        redaction_callback=None,
        cancel_requested=None,
        review_dir=None,
        config: PipelineConfig | None = None,
        plan=None,
        face_policy=None,
    ):
        """Process a video from beginning to end; returns a statistics dict."""
        self.reset_cancel()
        input_path, output_path = Path(input_path), Path(output_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found:\n{input_path}")

        cfg = (config or self.config).copy()
        cfg.redaction_method = normalize_method(redaction_method or cfg.redaction_method)
        if enabled_classes is not None:
            cfg.enabled_classes = set(enabled_classes)
        cfg.enabled_classes.discard("document")   # OCR region only, never redacted whole

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video:\n{input_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 30.0
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Could not determine video dimensions.")
        logger.info("Processing %s: %dx%d @ %.2f FPS, %d frames", input_path.name, width, height, fps, total_frames)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video:\n{output_path}")

        if prefix_output:
            prefix_capture = cv2.VideoCapture(str(prefix_output))
            try:
                while True:
                    ok, prefix_frame = prefix_capture.read()
                    if not ok:
                        break
                    writer.write(prefix_frame)
            finally:
                prefix_capture.release()

        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        review_dir = Path(review_dir) if review_dir else output_path.parent / "review_frames"
        engine = FrameEngine(
            self.model, cfg, source_name=input_path.name, face_policy=face_policy or self.face_policy, plan=plan,
            ocr_provider=self.ocr_provider, review_dir=review_dir,
        )
        self.engine = engine

        start_time = time.time()
        frame_number = start_frame
        cancelled = False
        try:
            while True:
                if cancel_requested and cancel_requested():
                    self.cancel()
                if self._cancel_requested:
                    cancelled = True
                    break

                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1

                output = engine.process(frame, frame_number, fps)
                writer.write(output.frame)

                elapsed = time.time() - start_time
                done = frame_number - start_frame
                current_fps = done / elapsed if elapsed > 0 else 0
                percent = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                eta = max(0, total_frames - frame_number) / current_fps if current_fps > 0 else 0

                if progress_callback:
                    progress_callback(percent)
                if frame_callback:
                    frame_callback(frame_number, total_frames)
                if fps_callback:
                    fps_callback(current_fps)
                if eta_callback:
                    eta_callback(eta)
                if detection_callback:
                    detection_callback(engine.total_detections)
                if flagged_callback:
                    flagged_callback(engine.total_flagged)
                if redaction_callback:
                    redaction_callback(engine.total_redactions)
        finally:
            capture.release()
            writer.release()

        processing_time = time.time() - start_time
        summary = engine.summary()
        result = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "frames_processed": frame_number,
            "total_frames": total_frames,
            "video_duration": total_frames / fps if fps > 0 else 0.0,
            "detections": engine.total_detections,
            "redactions": engine.total_redactions,
            "flagged": engine.total_flagged,
            "detection_records": engine.records,
            "processing_time": processing_time,
            "cancelled": cancelled,
            "summary": summary,
            "class_map": self.class_map,
            "width": width,
            "height": height,
            "fps": fps,
        }
        logger.info(
            "Finished %s: %d frames, %d detections, %d redactions, %.1fs, cancelled=%s",
            input_path.name, frame_number, engine.total_detections, engine.total_redactions, processing_time, cancelled,
        )
        return result

    # ------------------------------------------------------------------
    # Backwards-compatible helpers
    # ------------------------------------------------------------------
    def _redact_region(self, frame, x1, y1, x2, y2, method):
        return redact_region(frame, x1, y1, x2, y2, method)

    @staticmethod
    def _box_iou(first_box, second_box):
        return box_iou(first_box, second_box)
