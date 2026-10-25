"""FrameEngine + FacePolicy end to end on synthetic frames (MOCK YOLO + MOCK embedder; redaction is real)."""
from privarcy.faces import FaceBook, FaceMode, FacePolicy
from privarcy.pipeline.config import PipelineConfig
from privarcy.pipeline.frame_engine import FrameEngine
from privarcy.pipeline.plan import EditPlan, ManualRegion
from tests.support.fakes import ColorEmbedder, FakeYolo, face_frame, is_masked

A, B, C = (20, 20, 80, 80), (120, 20, 180, 80), (220, 20, 280, 80)


def cfg(**kw):
    c = PipelineConfig(redaction_method="Solid Mask", box_padding={}, cover_raw_box=False, ocr_enabled=False, temporal_smoothing=1)
    c.update(**kw)
    return c


def engine(script, mode, **kw):
    policy = FacePolicy(FaceBook(ColorEmbedder()), mode, **kw)
    return FrameEngine(FakeYolo(script), cfg(), source_name="t", face_policy=policy), policy


def frames(n, faces):
    return [face_frame(faces) for _ in range(n)]


def test_every_mode_blurs_all_faces_even_low_confidence_ones_in_the_review_band():
    eng, _ = engine([[("face", 0.95, A), ("face", 0.60, B)]] * 2, FaceMode.EVERY)
    out = eng.process(face_frame([(A, 50), (B, 100)]), 1)
    assert is_masked(out.frame, A) and is_masked(out.frame, B) and len(out.faces) == 2 and all(f.blurred for f in out.faces)
    assert out.flagged_new == 0                                   # faces never wait in the review queue under the policy


def test_auto_mode_keeps_chosen_face_visible_and_blurs_new_arrivals():
    script = [[("face", 0.9, A), ("face", 0.9, B)]] * 2 + [[("face", 0.9, A), ("face", 0.9, B), ("face", 0.9, C)]] * 2
    eng, policy = engine(script, FaceMode.AUTO)
    f = face_frame([(A, 50), (B, 100)])
    out = eng.process(f.copy(), 1)
    ids = {x.label: x.face_id for x in out.faces}
    policy.set_keep_visible({ids["Face 01"]})
    out2 = eng.process(face_frame([(A, 50), (B, 100)]), 2)
    assert not is_masked(out2.frame, A) and is_masked(out2.frame, B)
    eng.process(face_frame([(A, 50), (B, 100)]), 3)
    out4 = eng.process(face_frame([(A, 50), (B, 100), (C, 150)]), 4)
    assert is_masked(out4.frame, C) and not is_masked(out4.frame, A)          # newly appeared person is blurred


def test_selected_mode_blurs_only_selected_and_selection_persists_across_frames():
    script = [[("face", 0.9, A), ("face", 0.9, B)]] * 6
    eng, policy = engine(script, FaceMode.SELECTED)
    out = eng.process(face_frame([(A, 50), (B, 100)]), 1)
    assert not any(f.blurred for f in out.faces)                  # nobody selected yet -> nothing blurred
    ids = {f.label: f.face_id for f in out.faces}
    policy.set_selected({ids["Face 02"]})
    for n in range(2, 7):
        o = eng.process(face_frame([(A, 50), (B, 100)]), n)
        assert is_masked(o.frame, B) and not is_masked(o.frame, A)


def test_person_who_leaves_and_returns_keeps_the_same_choice():
    gap = [[]] * 8                                                # longer than track_max_gap -> track expires
    script = [[("face", 0.9, A)]] + gap + [[("face", 0.9, (150, 100, 210, 160))]]
    eng, policy = engine(script, FaceMode.SELECTED)
    first = eng.process(face_frame([(A, 50)]), 1).faces[0].face_id
    policy.set_selected({first})
    for n in range(2, 10):
        eng.process(face_frame([]), n)
    back = eng.process(face_frame([((150, 100, 210, 160), 50)]), 10)
    assert back.faces[0].face_id == first and back.faces[0].blurred


def test_other_classes_keep_their_existing_behaviour_under_any_face_mode():
    card = (200, 120, 300, 200)
    eng, _ = engine([[("credit_card", 0.95, card), ("face", 0.9, A)]], FaceMode.AUTO)
    eng.policy_unused = True
    out = eng.process(face_frame([(A, 50)]), 1)
    assert is_masked(out.frame, card) and is_masked(out.frame, A)


def test_face_class_disabled_means_faces_are_not_touched_at_all():
    policy = FacePolicy(FaceBook(ColorEmbedder()), FaceMode.EVERY)
    c = cfg(); c.enabled_classes.discard("face")
    eng = FrameEngine(FakeYolo([[("face", 0.95, A)]]), c, source_name="t", face_policy=policy)
    out = eng.process(face_frame([(A, 50)]), 1)
    assert not is_masked(out.frame, A) and out.faces == []


def test_manual_region_is_redacted_only_inside_its_time_window_and_ellipse_keeps_corners():
    plan = EditPlan(manual=[ManualRegion(0.5, 0.5, 0.9, 0.9, start=1.0, end=2.0),
                            ManualRegion(0.05, 0.05, 0.35, 0.45, shape="ellipse")])
    eng = FrameEngine(FakeYolo([[]] * 100), cfg(), source_name="t", plan=plan)
    f0 = face_frame([])
    early = eng.process(f0.copy(), 1, fps=10.0)                   # t = 0.1 s : rectangle not active yet
    assert not is_masked(early.frame, (160, 120, 288, 216))
    mid = eng.process(f0.copy(), 15, fps=10.0)                    # t = 1.5 s : active
    assert is_masked(mid.frame, (170, 130, 280, 210))
    h, w = f0.shape[:2]
    cx, cy = int(0.2 * w), int(0.25 * h)
    assert mid.frame[cy, cx].sum() == 0                           # ellipse centre masked
    corner = (int(0.05 * w) + 1, int(0.05 * h) + 1)
    assert mid.frame[corner[1], corner[0]].sum() != 0             # ellipse corner untouched (shape, not box)
