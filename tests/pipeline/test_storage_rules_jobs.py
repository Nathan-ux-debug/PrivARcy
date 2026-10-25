import os
import time
from pathlib import Path

import cv2
import pytest

from privarcy.storage import CorrectionLogDB
from privarcy.pipeline.config import PipelineConfig
from privarcy.pipeline.job_runner import JobRunner, probe_video
from privarcy.pipeline.review_apply import apply_review_redactions
from privarcy.pipeline.video_pipeline import VideoPipeline
from privarcy.rules import RuleStore, evidence_from_log, propose_rules, validate_proposal
from privarcy.storage import StorageManager
from tests.support.fakes import FakeYolo, gradient_frame


def make_video(path, frames=30, w=320, h=240):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(frames):
        wr.write(gradient_frame(w, h))
    wr.release()
    return path


def frames_in(path):
    c = cv2.VideoCapture(str(path)); n = 0
    while c.read()[0]:
        n += 1
    c.release()
    return n


# ---------------------------------------------------------------- video + jobs
def test_recorded_video_n_frames_in_n_frames_out_and_manifest(tmp_path):
    src = make_video(tmp_path / "in.mp4", 30)
    model = FakeYolo([[("credit_card", 0.9, (50, 50, 120, 120))]] * 30)
    st = StorageManager(tmp_path / "Backend")
    cfg = PipelineConfig(redaction_method="Solid Mask", ocr_enabled=False)
    res = JobRunner(VideoPipeline(model=model, config=cfg), st).run(src, config=cfg)
    assert res.frames == 30 and frames_in(res.output_path) == 30
    job = st.load_job(res.job_id)
    assert job.status == "completed" and job.frames_processed == 30 and job.width == 320 and job.redactions == 30
    assert any("OCR was not available" in w for w in job.warnings)
    out = cv2.VideoCapture(res.output_path); ok, f = out.read(); out.release()
    assert f[60:110, 60:110].mean() < 20          # solid-masked (mp4 compression -> not exactly 0)


def test_failed_job_marked_failed_and_partial_output_removed(tmp_path):
    src = make_video(tmp_path / "in.mp4", 5)
    class Boom(FakeYolo):
        def predict(self, *a, **k): raise RuntimeError("model exploded")
    st = StorageManager(tmp_path / "Backend")
    runner = JobRunner(VideoPipeline(model=Boom([]), config=PipelineConfig(ocr_enabled=False)), st)
    with pytest.raises(RuntimeError):
        runner.run(src)
    job = st.list_jobs()[0]
    assert job.status == "failed" and "exploded" in job.error
    assert not (st.outputs_dir / "in_redacted.mp4").exists()


def test_cancel_then_resume_produces_full_length_output(tmp_path):
    src = make_video(tmp_path / "in.mp4", 30)
    st = StorageManager(tmp_path / "Backend")
    cfg = PipelineConfig(ocr_enabled=False)
    pipe = VideoPipeline(model=FakeYolo([]), config=cfg)
    runner = JobRunner(pipe, st)
    res = runner.run(src, config=cfg, on_frame=lambda c, t: runner.cancel() if c == 10 else None)
    assert res.cancelled and 10 <= res.frames < 30
    runner2 = JobRunner(VideoPipeline(model=FakeYolo([]), config=cfg), st)
    res2 = runner2.run(src, config=cfg)
    assert not res2.cancelled and res2.resumed_from == res.frames and frames_in(res2.output_path) == 30


def test_probe_rejects_non_video(tmp_path):
    bad = tmp_path / "x.mp4"; bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError):
        probe_video(bad)


def test_confirm_applies_redaction_to_output_video(tmp_path):
    src = make_video(tmp_path / "in.mp4", 10)
    st = StorageManager(tmp_path / "Backend")
    cfg = PipelineConfig(redaction_method="Solid Mask", ocr_enabled=False)
    model = FakeYolo([[("id_card", 0.6, (50, 50, 120, 120))]] * 10)
    res = JobRunner(VideoPipeline(model=model, config=cfg), st).run(src, config=cfg)
    item = next(r for r in res.detections if r["status"] == "pending")
    assert len(item["track_observations"]) == 10
    c = cv2.VideoCapture(res.output_path); f = c.read()[1]; c.release()
    assert f[60:110, 60:110].mean() > 40                       # not redacted before review
    assert apply_review_redactions(item) == 10
    c = cv2.VideoCapture(res.output_path); f = c.read()[1]; c.release()
    assert f[60:110, 60:110].mean() < 20 and frames_in(res.output_path) == 10


# ------------------------------------------------------------------- storage
def test_storage_layout_and_separation(tmp_path):
    st = StorageManager(tmp_path / "B")
    for d in (st.d1_jobs, st.d3_dir, st.rules_dir, st.outputs_dir):
        assert d.is_dir()
    assert len({str(st.d1_jobs), str(st.d3_dir), str(st.d4_path), str(st.outputs_dir), str(st.rules_dir)}) == 5
    src = tmp_path / "v.mp4"; make_video(src, 3)
    job = st.create_job(src)
    assert Path(job.source_path) == src.resolve()          # reference, not a duplicate copy
    assert not any(st.d1_dir.rglob("*.mp4"))


def test_startup_cleanup_interrupted_expired_and_legacy(tmp_path):
    st = StorageManager(tmp_path / "B")
    j = st.create_job(__file__); j.status = "running"; st.save_job(j)
    d3 = st.job_frames_dir(j.job_id); (d3 / "a.jpg").write_bytes(b"x")
    old = st.job_frames_dir("oldjob"); (old / "b.jpg").write_bytes(b"x")
    j2 = st.create_job(__file__); j2.status = "completed"; st.save_job(j2)
    old_job = st.d1_jobs / "ghost.json"                    # D3 folder without manifest -> orphan
    orphan = st.job_frames_dir("ghost")
    legacy = st.outputs_dir / "review_frames"; legacy.mkdir(); (legacy / "1.jpg").write_bytes(b"x")
    exp = st.job_frames_dir(j2.job_id); (exp / "c.jpg").write_bytes(b"x")
    past = time.time() - 9 * 86400; os.utime(exp, (past, past))
    rep = StorageManager(tmp_path / "B").startup_cleanup()
    assert rep["interrupted"] == 1 and rep["legacy_review_frames_removed"] == 1
    assert rep["d3_purged"] == 3                            # oldjob + ghost (no manifest) + expired
    assert st.load_job(j.job_id).status == "interrupted" and (d3 / "a.jpg").exists()
    assert not exp.exists() and not orphan.exists()


def test_release_frames_when_review_resolved(tmp_path):
    st = StorageManager(tmp_path / "B"); d = st.job_frames_dir("abc"); (d / "x.jpg").write_bytes(b"1")
    assert st.release_job_frames("abc") and not d.exists() and not st.release_job_frames("abc")


# --------------------------------------------------------- rule refinement
def log(db, cls, conf, decision, n):
    for i in range(n):
        db.record_decision(f"f{cls}{conf}{i}", [1, 2, 3, 4], decision, cls, conf)


def test_unusable_legacy_records_are_not_evidence(tmp_path):
    db = CorrectionLogDB(":memory:")
    for i in range(74):
        db.record_decision(f"x{i}", [0, 0, 0, 0], "accepted", "face", 0.0)    # exactly what the old build wrote
    ev = evidence_from_log(db)
    assert ev.unusable_records == 74 and propose_rules(ev, {}) == []


def test_proposal_validate_activate_audit_rollback(tmp_path):
    db = CorrectionLogDB(":memory:")
    log(db, "credit_card", 0.62, "accepted", 6); log(db, "credit_card", 0.74, "accepted", 8)
    log(db, "face", 0.70, "accepted", 6); log(db, "face", 0.60, "rejected", 6)    # mixed -> no proposal
    ev = evidence_from_log(db)
    props = propose_rules(ev, {})
    assert [p.class_name for p in props] == ["credit_card"] and props[0].proposed_threshold == 0.60
    report = validate_proposal(props[0], ev, {})
    assert report.passed

    store = RuleStore(tmp_path / "rules")
    assert store.active_version_number() == 0
    with pytest.raises(ValueError):
        store.activate({"credit_card": 0.6}, "me", "x", {}, {"passed": False})      # no unvalidated change
    with pytest.raises(ValueError):
        store.activate({"credit_card": 0.3}, "me", "x", {}, {"passed": True})       # invariant violation
    v1 = store.activate({"credit_card": 0.6}, "alice", props[0].summary(), ev.to_dict(), report.to_dict())
    assert store.active_version_number() == 1 and store.active().class_auto_redact_confidence == {"credit_card": 0.6}
    v2 = store.rollback(0, "alice", "testing")
    assert store.active_version_number() == 2 and store.active().class_auto_redact_confidence == {}
    assert len(store.history()) == 3                           # nothing deleted
    assert [a["action"] for a in store.audit()][:3] == ["rollback", "activate", "init"]


def test_validation_fails_when_reviewers_rejected_newly_auto_items():
    db = CorrectionLogDB(":memory:")
    log(db, "id_card", 0.55, "rejected", 3); log(db, "id_card", 0.65, "accepted", 40)
    ev = evidence_from_log(db)
    from privarcy.rules.proposals import Proposal
    bad = Proposal("id_card", 0.8, 0.55, 43, 0.93, "forced")
    assert not validate_proposal(bad, ev, {}).passed


def test_active_rule_changes_pipeline_behaviour_and_rollback_restores(tmp_path):
    db = CorrectionLogDB(":memory:")
    log(db, "credit_card", 0.62, "accepted", 12)
    ev = evidence_from_log(db); p = propose_rules(ev, {})[0]
    store = RuleStore(tmp_path / "r")
    store.activate({"credit_card": p.proposed_threshold}, "a", "n", {}, validate_proposal(p, ev, {}).to_dict())
    from privarcy.pipeline.frame_engine import FrameEngine
    for overrides, expected in (({"credit_card": 0.6}, "auto_redacted"), ({}, "pending")):
        cfg = PipelineConfig(class_auto_redact_confidence=overrides, ocr_enabled=False)
        out = FrameEngine(FakeYolo([[("credit_card", 0.65, (10, 10, 80, 60))]]), cfg).process(gradient_frame(), 1, 30)
        assert out.records[0]["status"] == expected


def test_review_queue_persists_and_resolves_and_frees_d3(tmp_path):
    from privarcy.services.backend import BackendService
    src = make_video(tmp_path / "in.mp4", 10)
    svc = BackendService(tmp_path / "Backend", model=FakeYolo([[("id_card", 0.6, (50, 50, 120, 120))]] * 10))
    svc.update_settings(ocr_enabled=False, redaction_method="Solid Mask")
    res = svc.run_job(str(src))
    svc2 = BackendService(tmp_path / "Backend", model=FakeYolo([]))          # "restart"
    items = svc2.pending_review_items()
    assert len(items) == 1 and items[0]["job_name"] == "in.mp4" and Path(items[0]["image_path"]).exists()
    out = svc2.resolve_review_item(items[0], "confirmed", "ok")
    assert out["frames_redacted"] == 10 and svc2.pending_review_items() == []
    assert not (svc2.storage.d3_dir / res.job_id).exists()                    # D3 freed after review
    rec = svc2.list_corrections()[0]
    assert rec.decision == "accepted" and rec.detected_class == "id_card" and rec.confidence == pytest.approx(0.6)
    assert rec.bbox == [50.0, 50.0, 120.0, 120.0]                             # evidence the OLD build never recorded
    assert "in.mp4" in rec.context and "John" not in rec.context


def test_app_starts_and_detects_faces_when_opencv_has_no_cascadeclassifier(tmp_path, monkeypatch):
    """Reproduces the reported startup failure condition (cv2 without CascadeClassifier)."""
    import cv2
    from privarcy.faces import FacePrivacyError, FaceDetector
    from privarcy.services.backend import BackendService
    monkeypatch.delattr(cv2, "CascadeClassifier")
    with pytest.raises(FacePrivacyError) as err:
        FaceDetector()
    assert "interpreter=" in str(err.value) and "cv2.__file__" in str(err.value)     # evidence-based message
    svc = BackendService(tmp_path / "B", model=FakeYolo([[("face", 0.9, (40, 40, 120, 130))]] * 3))
    assert svc.face_detector is not None and svc.face_detector.backends == ["YoloFaceDetector"]
    assert svc.diagnostics()["opencv"]["has_cascade"] is False
    faces = svc.face_detector.detect(gradient_frame())
    assert len(faces) == 1 and faces[0].bbox == (40.0, 40.0, 120.0, 130.0)
