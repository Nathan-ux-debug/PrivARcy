"""BackendService - the single facade the desktop app (and scripts) talk to.

No Qt in here. Everything the UI shows must come from a method on this class,
so the UI can never display numbers the backend did not produce.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from privarcy.capture import CameraSource, ProbeResult, build_phone_url, discover_local_cameras, local_ip_addresses, probe_source
from privarcy.capture.phone_pairing import PhoneDeviceStore, PhonePairingIdentity, PhonePairingService
from privarcy.faces import FaceBook, FaceEmbedder, FaceMode, FacePolicy, build_face_detector
from privarcy.faces.scanner import FaceScan, scan_faces
from privarcy.ocr.provider import OCRProvider
from privarcy.paths import data_dir as default_data_dir
from privarcy.pipeline.config import REDACTION_METHODS, PipelineConfig
from privarcy.pipeline.job_runner import JobResult, JobRunner, probe_video
from privarcy.pipeline.live_session import LiveSession
from privarcy.pipeline.plan import EditPlan
from privarcy.pipeline.preview import PreviewEngine
from privarcy.pipeline.review_apply import apply_review_redactions
from privarcy.pipeline.video_pipeline import VideoPipeline
from privarcy.rules import RuleStore, evidence_from_log, propose_rules, validate_proposal
from privarcy.storage import CorrectionLogDB, StorageManager

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".webm")


class BackendService:
    def __init__(self, data_dir: Optional[Path] = None, model=None) -> None:
        self.data_dir = Path(data_dir) if data_dir else default_data_dir()
        self.storage = StorageManager(self.data_dir)
        self.startup_report = self.storage.startup_cleanup()

        self.config = PipelineConfig()
        self.rules = RuleStore(self.storage.rules_dir)
        self._apply_active_rules()

        self.correction_db = CorrectionLogDB(self.storage.d4_path)
        try:
            from privarcy.faces.deep_embedder import DeepFaceEmbedder
            self.face_embedder = DeepFaceEmbedder()
        except Exception as exc:          # missing package / model download failed
            logger.warning("Deep embedder unavailable (%s); using the HOG baseline.", exc)
        self.face_embedder = FaceEmbedder()
        self.ocr_provider = OCRProvider(model_name=self.config.ocr_model)

        self.pipeline = VideoPipeline(model=model, config=self.config, ocr_provider=self.ocr_provider)
        self.face_detector, self.face_detector_notes = build_face_detector(self.pipeline.model)
        self.pipeline.face_detector = self.face_detector

        self.job_runner: Optional[JobRunner] = None
        self.live_session: Optional[LiveSession] = None
        self.last_live_recording: Optional[str] = None
        self.phone_pairing: Optional[PhonePairingService] = None
        self._phone_pairing_identity: PhonePairingIdentity | None = None
        self._phone_pairing_store: PhoneDeviceStore | None = None
        self._scan_cancel = False
        self._restore_phone_pairing()

    def _restore_phone_pairing(self) -> None:
        directory = self.data_dir / "phone_pairing"
        if not (directory / "paired-devices.dpapi").exists():
            return
        self._phone_pairing_identity = PhonePairingIdentity.load_or_create(directory)
        self._phone_pairing_store = PhoneDeviceStore(
            directory, self._phone_pairing_identity.device_id
        )
        if not self._phone_pairing_store.list_devices():
            return
        try:
            self.phone_pairing = PhonePairingService(
                identity=self._phone_pairing_identity,
                device_store=self._phone_pairing_store,
            ).start()
        except RuntimeError as exc:
            logger.warning("Could not restore the paired-phone listener: %s", exc)

    # ------------------------------------------------------------------ rules
    def _apply_active_rules(self) -> None:
        self.config.class_auto_redact_confidence = dict(self.rules.active().class_auto_redact_confidence)

    def update_settings(self, **values) -> None:
        if "ocr_model" in values and values["ocr_model"] != self.config.ocr_model:
            self.ocr_provider = OCRProvider(model_name=values["ocr_model"])
            self.pipeline.ocr_provider = self.ocr_provider
        self.config.update(**values)

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict:
        names = self.pipeline.class_map
        torch = sys.modules.get("torch")      # already imported by ultralytics when the real model is loaded
        try:
            device = ("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else "n/a"
        except Exception:
            device = "unknown"
        ocr = self.ocr_provider.status()
        return {
            "detector": {
                "loaded": self.pipeline.model is not None, "device": device,
                "classes": names["names"], "class_map_ok": names["ok"], "warnings": names["warnings"],
            },
            "ocr": {
                "available": ocr["available"], "loaded": ocr["loaded"], "model": ocr["model"], "reason": ocr["reason"],
                "integrated": True, "localizer": "OpenCV morphological text-line localiser",
            },
            "face_recognition": {
                "loaded": self.face_detector is not None,
                "backends": getattr(self.face_detector, "backends", []),
                "notes": self.face_detector_notes,
                "embedding": "HOG (baseline, not deep-learning)",
            },
            "redaction": {"methods": list(REDACTION_METHODS), "active": self.config.redaction_method},
            "storage": self.storage.layout(),
            "rules": {"active_version": self.rules.active_version_number(),
                      "overrides": dict(self.config.class_auto_redact_confidence)},
            "opencv": {"version": getattr(cv2, "__version__", "?"), "file": getattr(cv2, "__file__", "?"),
                       "has_cascade": hasattr(cv2, "CascadeClassifier")},
            "startup": self.startup_report,
        }

    def overview(self) -> dict:
        jobs = self.storage.list_jobs(500)
        return {
            "completed_jobs": sum(1 for j in jobs if j.status == "completed"),
            "total_jobs": len(jobs),
            "pending_review": sum(j.pending_review for j in jobs if j.status in ("completed", "cancelled")),
            "corrections_logged": self.correction_db.count(),
        }

    def get_history(self, limit: int = 50) -> list:
        return self.storage.list_jobs(limit)

    # ------------------------------------------------------------------ video
    @staticmethod
    def validate_video(path: str) -> dict:
        p = Path(path)
        if not p.is_file():
            return {"ok": False, "error": "File not found."}
        if p.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            return {"ok": False, "error": f"Unsupported extension '{p.suffix}'. Supported: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"}
        try:
            info = probe_video(p)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        info.update(ok=True, size_bytes=p.stat().st_size, name=p.name)
        return info

    def run_job(self, video_path: str, on_frame=None, on_detections=None, on_flagged=None, on_redactions=None,
                plan: Optional[EditPlan] = None, face_policy: Optional[FacePolicy] = None) -> JobResult:
        self.job_runner = JobRunner(self.pipeline, self.storage)
        try:
            return self.job_runner.run(video_path, config=self.config, on_frame=on_frame, on_detections=on_detections,
                                       on_flagged=on_flagged, on_redactions=on_redactions, plan=plan, face_policy=face_policy)
        finally:
            self.job_runner = None

    def cancel_job(self) -> None:
        if self.job_runner is not None:
            self.job_runner.cancel()

    # ----------------------------------------------------------------- review
    def _log_decision(self, item: dict, decision: str, note: str = "") -> int:
        context = {
            "job_id": item.get("job_id"), "source": item.get("source"), "frame": item.get("frame"),
            "model_class": item.get("label"), "status_at_review": item.get("status"),
            "source_class": item.get("source_class"), "ocr_patterns": item.get("ocr_patterns"),
            "note": note, "rules_version": self.rules.active_version_number(),
        }
        return self.correction_db.record_decision(
            frame_id=str(item.get("id", "")), bbox=[float(v) for v in item.get("bbox", [0, 0, 0, 0])],
            decision=decision, detected_class=item.get("label") or item.get("classification") or "",
            confidence=float(item.get("confidence") or 0.0), context=json.dumps(context),
        )

    def confirm_detection(self, item: dict, note: str = "") -> dict:
        """Confirm = 'yes, redact this'. Applies the redaction to the output video, then logs."""
        frames = apply_review_redactions(item)
        row = self._log_decision(item, "accepted", note)
        return {"frames_redacted": frames, "log_id": row}

    def reject_detection(self, item: dict, note: str = "") -> dict:
        """Reject = 'false positive, leave visible'. Logged; the video is untouched."""
        return {"log_id": self._log_decision(item, "rejected", note)}

    def pending_review_items(self) -> list:
        """Every unresolved review item across jobs, newest job first."""
        items = []
        for job in self.storage.list_jobs(200):
            if job.pending_review <= 0:
                continue
            for item in self.storage.load_review_items(job.job_id):
                if item.get("status") == "pending":
                    item.setdefault("job_id", job.job_id)
                    item.setdefault("output_path", job.output_path)
                    item["job_name"] = job.source_name
                    items.append(item)
        return items

    def resolve_review_item(self, item: dict, decision: str, note: str = "") -> dict:
        """decision: 'confirmed' (redact) or 'rejected' (leave visible)."""
        result = self.confirm_detection(item, note) if decision == "confirmed" else self.reject_detection(item, note)
        job_id = item.get("job_id")
        if job_id:
            remaining = self.storage.set_review_status(job_id, item["id"], decision)
            job = self.storage.load_job(job_id)
            if job:
                job.pending_review = remaining
                self.storage.save_job(job)
            if remaining == 0:
                self.storage.release_job_frames(job_id)      # review finished -> temporary crops no longer needed
        return result

    def release_job_review_frames(self, job_id: str) -> None:
        job = self.storage.load_job(job_id)
        self.storage.release_job_frames(job_id)
        if job:
            job.pending_review = 0
            self.storage.save_job(job)

    def list_corrections(self, limit: int = 500) -> list:
        return self.correction_db.get_decisions(limit=limit)

    # ------------------------------------------------------------------- rules
    def rule_evidence(self):
        return evidence_from_log(self.correction_db)

    def rule_proposals(self) -> dict:
        evidence = self.rule_evidence()
        current = dict(self.config.class_auto_redact_confidence)
        out = []
        for proposal in propose_rules(evidence, current):
            report = validate_proposal(proposal, evidence, current)
            out.append({"proposal": proposal, "validation": report})
        return {"evidence": evidence, "items": out}

    def approve_proposal(self, proposal, report, approver: str = "reviewer"):
        overrides = dict(self.config.class_auto_redact_confidence)
        overrides[proposal.class_name] = proposal.proposed_threshold
        version = self.rules.activate(overrides, approver, proposal.summary(), self.rule_evidence().to_dict(), report.to_dict())
        self._apply_active_rules()
        return version

    def reject_proposal(self, proposal, approver: str = "reviewer") -> None:
        self.rules.reject(proposal.summary(), approver)

    def rollback_rules(self, version: int, approver: str = "reviewer"):
        v = self.rules.rollback(version, approver)
        self._apply_active_rules()
        return v

    # -------------------------------------------------------------------- faces
    # ------------------------------------------------------------------ faces (who gets blurred)
    def new_face_policy(self, mode=FaceMode.AUTO, live: bool = False, book: Optional[FaceBook] = None) -> FacePolicy:
        """A fresh policy. Live sessions label people T-01, T-02 ...; recorded video uses Face 01, Face 02 ..."""
        if book is None:
            book = FaceBook(self.face_embedder, label_prefix="T", id_prefix="T", label_sep="-") if live else FaceBook(self.face_embedder)
        return FacePolicy(book, mode)

    def scan_faces(self, video_path: str, on_progress=None, stride_seconds: float = 0.5) -> FaceScan:
        """One quick pass that lists the distinct people in a video (for the picker)."""
        if self.face_detector is None:
            raise RuntimeError("No face detector is available: " + "; ".join(self.face_detector_notes or ["unknown reason"]))
        self._scan_cancel = False
        return scan_faces(video_path, self.face_detector, FaceBook(self.face_embedder), stride_seconds=stride_seconds,
                          on_progress=on_progress, cancel=lambda: self._scan_cancel)

    def cancel_scan(self) -> None:
        self._scan_cancel = True

    def create_preview(self, face_policy: Optional[FacePolicy] = None) -> PreviewEngine:
        """Editor preview: analyse a frame once, re-render instantly when options change. Uses the live config object."""
        return PreviewEngine(self.pipeline.model, self.config, face_policy=face_policy)

    # ------------------------------------------------------------------ cameras
    def discover_cameras(self, max_index: int = 5) -> list:
        return discover_local_cameras(max_index)

    @staticmethod
    def phone_stream_url(preset: str, host: str, port: Optional[int] = None) -> str:
        return build_phone_url(preset, host, port)

    @staticmethod
    def lan_addresses() -> list:
        return local_ip_addresses()

    @staticmethod
    def probe_camera(source: CameraSource, timeout: float = 8.0) -> ProbeResult:
        return probe_source(source, timeout)

    def start_live(self, source, width: int = 640, height: int = 480, fps: float = 30.0,
                   face_mode=FaceMode.AUTO, plan: Optional[EditPlan] = None,
                   capture_factory: Optional[Callable] = None, record: bool = True) -> LiveSession:
        """``source`` is a camera index or a ``CameraSource`` (USB, phone app stream, RTSP/HTTP camera)."""
        self.stop_live()
        path = self.storage.outputs_dir / f"live_{int(time.time())}.mp4" if record else None
        cfg = self.config.copy()
        ocr = self.ocr_provider if cfg.live_ocr_enabled else None
        policy = self.new_face_policy(face_mode, live=True)
        session = LiveSession(self.pipeline.model, cfg, source, width, height, fps, output_path=path,
                              ocr_provider=ocr, face_policy=policy, plan=plan, capture_factory=capture_factory)
        self.live_session = session
        return session

    def start_phone_pairing(self) -> PhonePairingService:
        if self.phone_pairing is not None and not self.phone_pairing._stopped.is_set():
            self.phone_pairing.renew_qr()
            return self.phone_pairing
        directory = self.data_dir / "phone_pairing"
        identity = self._phone_pairing_identity or PhonePairingIdentity.load_or_create(directory)
        device_store = self._phone_pairing_store or PhoneDeviceStore(
            directory, identity.device_id
        )
        pairing = PhonePairingService(identity=identity, device_store=device_store).start()
        self._phone_pairing_identity = identity
        self._phone_pairing_store = device_store
        self.phone_pairing = pairing
        return pairing

    def paired_phone_devices(self) -> list[dict]:
        if self._phone_pairing_store is None:
            return []
        connected = set()
        if self.phone_pairing is not None:
            with self.phone_pairing._status_lock:
                connected = set(self.phone_pairing._active_device_ids)
        return [
            {**device, "connected": device["device_id"] in connected}
            for device in self._phone_pairing_store.list_devices()
        ]

    def revoke_phone_device(self, device_id: str) -> bool:
        if self.phone_pairing is not None:
            revoked = self.phone_pairing.revoke_device(device_id)
        elif self._phone_pairing_store is not None:
            revoked = self._phone_pairing_store.revoke(device_id)
        else:
            return False
        if revoked and self.live_session is not None and self.live_session.source.kind == "phone":
            self.stop_live()
        return revoked

    def disconnect_phone_devices(self) -> None:
        if self.phone_pairing is not None:
            self.phone_pairing.disconnect_paired_devices()

    def stop_phone_pairing(self, force: bool = False) -> None:
        if not force and self._phone_pairing_store is not None and self._phone_pairing_store.list_devices():
            return
        pairing, self.phone_pairing = self.phone_pairing, None
        if pairing is not None:
            pairing.stop()

    def stop_live(self) -> Optional[str]:
        session, self.live_session = self.live_session, None
        if session is None:
            return None
        session.stop()
        self._record_live_measurement(session)
        if session.output_path and session.output_path.exists() and session.stats["frames"] > 0:
            self.last_live_recording = str(session.output_path)
        elif session.output_path:
            session.output_path.unlink(missing_ok=True)
        return self.last_live_recording

    def _record_live_measurement(self, session: LiveSession) -> None:
        """Keep REAL measurements from real sessions (>= 50 frames) as verification evidence."""
        status = session.status()
        if session.stats["frames"] < 50 or session.actual is None:
            return
        entry = {
            "when": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": session.stats["frames"],
            "avg_fps": round(status["fps"], 2), "latency_ms": round(status["latency_ms"] or 0, 1),
            "inference_ms": round(status["inference_ms"] or 0, 1), "resolution": list(session.actual),
            "detection_interval": session.config.detection_interval, "ocr": bool(session.config.live_ocr_enabled),
            "ended_with_error": session.error,
        }
        path = self.storage.live_measurements_path
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except Exception:
            data = []
        data.append(entry)
        path.write_text(json.dumps(data[-50:], indent=2), encoding="utf-8")

    def live_measurements(self) -> list:
        path = self.storage.live_measurements_path
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        except Exception:
            return []

    def shutdown(self) -> None:
        self.stop_live()
        self.stop_phone_pairing(force=True)
        self._phone_pairing_identity = None
        self.cancel_job()
