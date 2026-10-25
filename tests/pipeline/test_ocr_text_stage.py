"""OCR localisation tests.

REAL: the text localiser (OpenCV) and coordinate mapping.
MOCK: the TrOCR recogniser (transformers is not available in CI) - it returns
scripted text for the line whose position it is asked to read, which lets us
prove that localised boxes land on the right pixels of the ORIGINAL frame.
"""
import cv2
import numpy as np

from privarcy.ocr.localizer import TextLocalizer
from privarcy.ocr.models import OCRResult
from privarcy.pipeline.config import PipelineConfig
from privarcy.pipeline.frame_engine import FrameEngine
from tests.support.fakes import FakeYolo

DOC_ORIGIN = (200, 150)          # document is NOT at (0,0): proves crop offsets are handled
LINES = [  # (text, y-baseline relative to doc, sensitive?)
    ("Name: John Smith", 60, True),
    ("Weather report for the week", 120, False),
    ("Account Number: 1234 5678 9012", 180, True),
]


def make_frame():
    frame = np.full((520, 900, 3), 90, np.uint8)
    x0, y0 = DOC_ORIGIN
    cv2.rectangle(frame, (x0, y0), (x0 + 520, y0 + 260), (245, 245, 245), -1)
    for text, y, _ in LINES:
        cv2.putText(frame, text, (x0 + 24, y0 + y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return frame


def truth_boxes():
    x0, y0 = DOC_ORIGIN
    out = []
    for text, y, sens in LINES:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        out.append(((x0 + 24, y0 + y - h, x0 + 24 + w, y0 + y + 6), sens, text))
    return out


def iou(a, b):
    from privarcy.pipeline.config import box_iou
    return box_iou(a, b)


class ScriptedOCR:
    """MOCK recogniser: identifies the line by the centre of the box it is asked to read."""
    def __init__(self):
        self.reads = 0

    def read_region(self, frame, bbox, frame_id=None, source_class=None):
        self.reads += 1
        cy = (bbox[1] + bbox[3]) / 2
        for box, _sens, text in truth_boxes():
            if box[1] - 8 <= cy <= box[3] + 8:
                return OCRResult(text=text, confidence=0.93, bbox=bbox)
        return OCRResult(text="", confidence=0.1, bbox=bbox)


class Provider:
    def __init__(self, engine): self.engine = engine
    def get(self): return self.engine
    def status(self): return {"available": True, "loaded": True, "model": "mock", "reason": ""}


def test_localiser_finds_each_text_line_in_original_coordinates():
    frame = make_frame()
    x0, y0 = DOC_ORIGIN
    crop = frame[y0:y0 + 260, x0:x0 + 520]
    regions = TextLocalizer().locate(crop, offset=(x0, y0))
    for box, _s, text in truth_boxes():
        best = max((iou(box, r.bbox) for r in regions), default=0)
        assert best > 0.5, f"line not localised: {text!r} (best IoU {best:.2f})"


def test_localiser_empty_and_blank_inputs():
    assert TextLocalizer().locate(np.zeros((0, 0, 3), np.uint8)) == []
    assert TextLocalizer().locate(np.full((100, 200, 3), 128, np.uint8)) == []


def test_sensitive_lines_redacted_harmless_line_untouched_and_no_text_stored(tmp_path):
    frame = make_frame()
    x0, y0 = DOC_ORIGIN
    doc = (x0, y0, x0 + 520, y0 + 260)
    cfg = PipelineConfig(redaction_method="Solid Mask", box_padding={}, hold_missed_detections=False)
    ocr = ScriptedOCR()
    eng = FrameEngine(FakeYolo([[("document", 0.9, doc)], [("document", 0.9, doc)]]), cfg, ocr_provider=Provider(ocr))
    out = eng.process(frame.copy(), 1, 30.0)

    for box, sens, text in truth_boxes():
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        region = out.frame[box[1] + 4:box[3] - 4, box[0] + 4:box[2] - 4]
        if sens:
            assert (region == 0).mean() > 0.9, f"sensitive line not covered: {text!r}"
        else:
            assert np.array_equal(region, frame[box[1] + 4:box[3] - 4, box[0] + 4:box[2] - 4]), "harmless line altered"
    assert out.text_redactions == 2
    # the whole document region is NOT blanked
    assert np.array_equal(out.frame[y0 + 240:y0 + 255, x0 + 5:x0 + 500], frame[y0 + 240:y0 + 255, x0 + 5:x0 + 500])

    # no recognised text is persisted in any record
    blob = repr(eng.records)
    assert "John" not in blob and "1234" not in blob
    assert any(r.get("ocr_patterns") for r in eng.records)

    # second frame: cached lines -> no new OCR calls but still redacted
    reads = ocr.reads
    out2 = eng.process(frame.copy(), 2, 30.0)
    assert ocr.reads == reads and out2.text_redactions == 2


def test_ocr_unavailable_pipeline_keeps_working():
    class Off(Provider):
        def get(self): return None
        def status(self): return {"available": False, "loaded": False, "model": "x", "reason": "no transformers"}
    eng = FrameEngine(FakeYolo([[("credit_card", 0.9, (10, 10, 90, 60))]]), PipelineConfig(), ocr_provider=Off(None))
    out = eng.process(make_frame(), 1, 30.0)
    assert out.redactions == 1 and eng.summary()["ocr"]["available"] is False


def test_screen_policy_ocr_only_redacts_text_not_whole_screen():
    frame = make_frame()
    x0, y0 = DOC_ORIGIN
    scr = (x0, y0, x0 + 520, y0 + 260)
    cfg = PipelineConfig(redaction_method="Solid Mask", box_padding={}, screen_policy="ocr_only")
    eng = FrameEngine(FakeYolo([[("screen", 0.95, scr)]]), cfg, ocr_provider=Provider(ScriptedOCR()))
    out = eng.process(frame.copy(), 1, 30.0)
    assert out.text_redactions == 2 and not (out.frame[y0 + 10:y0 + 250, x0 + 10:x0 + 510] == 0).all()


def test_screen_policy_falls_back_to_redact_all_when_ocr_missing():
    class Off(Provider):
        def get(self): return None
        def status(self): return {"available": False}
    cfg = PipelineConfig(redaction_method="Solid Mask", box_padding={}, screen_policy="ocr_only")
    eng = FrameEngine(FakeYolo([[("screen", 0.95, (10, 10, 100, 80))]]), cfg, ocr_provider=Off(None))
    out = eng.process(make_frame(), 1, 30.0)
    assert out.redactions == 1
