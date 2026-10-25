"""Face scan over a real (synthetic) video file: distinct people, stable ids, cancel, progress."""
import cv2
import pytest

from privarcy.faces import FaceBook
from privarcy.faces.scanner import scan_faces
from tests.support.fakes import ColorEmbedder, ColorFaceDetector, face_frame


def make_video(path, frames, fps=10, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for f in frames:
        w.write(f)
    w.release()
    return path


def test_scan_lists_distinct_people_and_keeps_ids_stable_while_they_move(tmp_path):
    frames = []
    for i in range(40):
        a = (20 + i * 3, 30, 80 + i * 3, 90)                      # person 1 drifts right
        frames.append(face_frame([(a, 50), ((200, 120, 260, 180), 100)]))   # person 2 stays
    video = make_video(tmp_path / "v.mp4", frames)
    scan = scan_faces(video, ColorFaceDetector(), FaceBook(ColorEmbedder()), stride_seconds=0.3)
    assert [i.label for i in scan.identities()] == ["Face 01", "Face 02"]
    ids_per_sample = [tuple(sorted(s["face_id"] for s in v)) for v in scan.samples.values()]
    assert set(ids_per_sample) == {("F01", "F02")} and scan.frames_scanned >= 10
    assert all(i.thumbnail is not None for i in scan.identities())


def test_person_who_leaves_and_comes_back_is_the_same_identity(tmp_path):
    present = face_frame([((40, 40, 100, 100), 50)])
    frames = [present] * 10 + [face_frame([])] * 10 + [face_frame([((180, 90, 240, 150), 50)])] * 10
    scan = scan_faces(make_video(tmp_path / "v.mp4", frames), ColorFaceDetector(), FaceBook(ColorEmbedder()), stride_seconds=0.3)
    assert len(scan.identities()) == 1 and scan.identities()[0].sightings >= 6


def test_two_faces_in_one_frame_are_always_two_people_even_if_they_look_identical(tmp_path):
    frames = [face_frame([((20, 20, 80, 80), 50), ((150, 20, 210, 80), 50)])] * 12
    scan = scan_faces(make_video(tmp_path / "v.mp4", frames), ColorFaceDetector(), FaceBook(ColorEmbedder()), stride_seconds=0.3)
    assert len(scan.identities()) == 2


def test_progress_and_cancel(tmp_path):
    video = make_video(tmp_path / "v.mp4", [face_frame([((20, 20, 80, 80), 50)])] * 60)
    seen = []
    scan = scan_faces(video, ColorFaceDetector(), FaceBook(ColorEmbedder()), stride_seconds=0.2,
                      on_progress=lambda d, t: seen.append((d, t)), cancel=lambda: len(seen) >= 3)
    assert scan.cancelled and len(seen) == 3 and seen[-1][1] > 3


def test_unreadable_video_raises(tmp_path):
    with pytest.raises(RuntimeError):
        scan_faces(tmp_path / "missing.mp4", ColorFaceDetector(), FaceBook(ColorEmbedder()))
