"""End to end on a REAL face (scikit-image astronaut photo) with the real Haar detector, real HOG identities, real video
files in and out. Only the YOLO model is a stand-in (it reports the Haar boxes as class 'face'); the YOLO weights are not
needed to verify identity tracking, the three face modes and the written output."""
import cv2
import numpy as np
import pytest

from privarcy.faces import FaceDetector, FaceMode
from privarcy.services.backend import BackendService
from tests.support.fakes import FakeYolo, _Box, _Result

skimage_data = pytest.importorskip("skimage.data")
pytestmark = pytest.mark.slow


class HaarYolo(FakeYolo):
    """Stand-in for the YOLO model: reports real Haar face detections as class 'face'."""
    def __init__(self):
        super().__init__([])
        self.haar = FaceDetector()

    def predict(self, source=None, conf=0.25, imgsz=640, verbose=False):
        self.calls += 1
        boxes = [_Box(self._ids["face"], 0.92, f.bbox) for f in self.haar.detect(source)]
        return [_Result(boxes, self.names)]


def two_person_scene():
    a = cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR)
    b = cv2.flip(a, 1)[:, :, ::-1].copy()                      # a different-looking "second person" (mirrored, colours shifted)
    return np.hstack([a, b])


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("v") / "in.mp4"
    scene = two_person_scene()
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (scene.shape[1], scene.shape[0]))
    for _ in range(24):
        w.write(scene)
    w.release()
    return path


def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def last_frame(path):
    cap = cv2.VideoCapture(str(path)); frame = None
    while True:
        ok, f = cap.read()
        if not ok: break
        frame = f
    cap.release()
    return frame


def make_service(tmp_path):
    svc = BackendService(data_dir=tmp_path / "data", model=HaarYolo())
    svc.config.update(ocr_enabled=False, redaction_method="Blur", detection_interval=1)
    return svc


def face_boxes(svc, video):
    cap = cv2.VideoCapture(str(video)); ok, f = cap.read(); cap.release()
    return [tuple(int(v) for v in b.bbox) for b in svc.face_detector.detect(f)]


def test_scan_finds_both_people_with_thumbnails(tmp_path, video):
    svc = make_service(tmp_path)
    scan = svc.scan_faces(str(video))
    ids = scan.identities()
    assert len(ids) == 2 and all(i.thumbnail is not None and i.sightings >= 3 for i in ids)


@pytest.mark.parametrize("mode,expect_left_blurred,expect_right_blurred", [
    (FaceMode.EVERY, True, True), (FaceMode.AUTO, False, True), (FaceMode.SELECTED, True, False)])
def test_processing_output_matches_the_chosen_face_mode(tmp_path, video, mode, expect_left_blurred, expect_right_blurred):
    svc = make_service(tmp_path)
    scan = svc.scan_faces(str(video))
    left, right = sorted(scan.identities(), key=lambda i: i.face_id)
    # which identity is on the left of the picture? the one first seen there (box x position in the first sample)
    first = scan.samples[min(scan.samples)]
    x_of = {s["face_id"]: s["bbox"][0] for s in first}
    left_id, right_id = sorted(x_of, key=x_of.get)
    policy = svc.new_face_policy(mode, book=scan.book)
    if mode == FaceMode.AUTO:
        policy.set_keep_visible({left_id})                    # keep the left person visible, blur everyone else
    if mode == FaceMode.SELECTED:
        policy.set_selected({left_id})                        # blur ONLY the left person
    result = svc.run_job(str(video), face_policy=policy)
    assert not result.cancelled and result.frames == 24, result
    out = last_frame(result.output_path)
    original = two_person_scene()
    (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2) = sorted(face_boxes(svc, video))
    left_face, right_face = out[ly1:ly2, lx1:lx2], out[ry1:ry2, rx1:rx2]
    ref_l, ref_r = sharpness(original[ly1:ly2, lx1:lx2]), sharpness(original[ry1:ry2, rx1:rx2])
    assert (sharpness(left_face) < 0.5 * ref_l) == expect_left_blurred
    assert (sharpness(right_face) < 0.5 * ref_r) == expect_right_blurred
    corner = out[400:500, 20:120]                              # background far from any face stays untouched
    assert abs(sharpness(corner) - sharpness(original[400:500, 20:120])) < 0.35 * sharpness(original[400:500, 20:120]) + 1
