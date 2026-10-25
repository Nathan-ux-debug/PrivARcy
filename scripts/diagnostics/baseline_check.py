"""Real-model regression baseline on a video of YOUR choice (nothing mocked).

    python scripts\\diagnostics\\baseline_check.py path\\to\\video.mp4 [--no-ocr]

Prints: the model's real class map, frames in vs frames out, detections, redactions, items for review,
low-confidence (unreviewed) counts, OCR availability. Use it before/after changes to prove nothing regressed.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch  # noqa: F401
except Exception:
    pass

import cv2  # noqa: E402

from privarcy.services.backend import BackendService  # noqa: E402


def count_frames(path: str) -> int:
    cap, n = cv2.VideoCapture(path), 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()
    svc = BackendService()
    svc.update_settings(ocr_enabled=not a.no_ocr)
    d = svc.diagnostics()
    print("Model classes :", d["detector"]["classes"])
    for w in d["detector"]["warnings"]:
        print("  note:", w)
    print("OCR           :", d["ocr"]["available"], d["ocr"]["reason"])
    print("Face backends :", d["face_recognition"]["backends"], d["face_recognition"]["notes"])
    frames_in = count_frames(a.video)
    res = svc.run_job(a.video, on_frame=lambda c, t: print(f"\r  frame {c}/{t}", end="", flush=True))
    frames_out = count_frames(res.output_path)
    print()
    report = {
        "frames_in": frames_in, "frames_out": frames_out, "frames_match": frames_in == frames_out,
        "detections": res.total_detections, "redactions": res.redactions, "pending_review": sum(1 for r in res.detections if r["status"] == "pending"),
        "seconds": round(res.processing_time, 1), "fps": round(res.avg_fps, 1), "summary": res.summary, "warnings": res.warnings,
        "output": res.output_path,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["frames_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
