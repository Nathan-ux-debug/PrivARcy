"""PHYSICAL webcam test - produces real measurements, nothing mocked.

    python scripts\\diagnostics\\webcam_check.py [--camera 0] [--seconds 20] [--width 640 --height 480] [--ocr]

Runs the real YOLO model through LiveSession for N seconds, then prints measured FPS /
latency / inference time and saves a redacted snapshot + a JSON report under the data folder (outputs/).
Hold a face / card / ID in front of the camera and check the snapshot shows it covered.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch  # noqa: F401  (import before Qt/ultralytics on Windows)
except Exception:
    pass

import cv2  # noqa: E402

from privarcy.services.backend import BackendService  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--ocr", action="store_true")
    a = ap.parse_args()

    svc = BackendService()
    svc.update_settings(live_ocr_enabled=a.ocr)
    print("Opening camera", a.camera, "…")
    session = svc.start_live(a.camera, a.width, a.height, 30)
    if session.state == "error":
        print("FAILED TO START:", session.error)
        return 2
    print("Camera opened:", session.actual, "- measuring for", a.seconds, "s")
    t0, samples = time.time(), []
    while time.time() - t0 < a.seconds and session.state == "live":
        time.sleep(1.0)
        st = session.status()
        samples.append(st)
        print(f"  {st['fps']:5.1f} fps | latency {st['latency_ms'] or 0:6.1f} ms | inference {st['inference_ms'] or 0:6.1f} ms | "
              f"frames {st['frames']} | redactions {session.stats['redacted']}")
    snapshot = session.latest_frame()
    final = session.status()
    error = session.error
    svc.stop_live()

    out = svc.storage.outputs_dir
    if snapshot is not None:
        cv2.imwrite(str(out / "webcam_check_snapshot.jpg"), snapshot)
    report = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"), "camera": a.camera, "resolution": list(session.actual or []),
        "frames": final["frames"], "avg_fps_last_window": round(final["fps"], 2),
        "latency_ms": round(final["latency_ms"] or 0, 1), "inference_ms": round(final["inference_ms"] or 0, 1),
        "meets_15fps_target": final["fps"] >= 15.0, "ended_with_error": error, "ocr": a.ocr,
        "redactions": session.stats["redacted"], "held_redactions": session.stats["held"],
        "low_confidence_detections": session.stats["low_confidence"],
    }
    (out / "webcam_check_report.json").write_text(json.dumps(report, indent=2))
    print("\nREPORT:", json.dumps(report, indent=2))
    print("\nOpen", out / "webcam_check_snapshot.jpg", "and CONFIRM the sensitive object is covered. Then restart/stop with Ctrl+C if needed.")
    return 0 if not error else 3


if __name__ == "__main__":
    sys.exit(main())
