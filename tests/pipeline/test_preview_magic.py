"""Instant preview (no second inference) and Magic Blur shape/tracking."""
import cv2
import numpy as np

from privarcy.faces import FaceBook, FaceMode, FacePolicy
from privarcy.pipeline.config import PipelineConfig
from privarcy.pipeline.magic import MagicTracker, colour_signature, object_mask
from privarcy.pipeline.plan import EditPlan, ManualRegion, MagicTarget
from privarcy.pipeline.preview import PreviewEngine
from tests.support.fakes import ColorEmbedder, FakeYolo, face_frame, gradient_frame

A, B = (20, 20, 80, 80), (120, 20, 180, 80)
SCREEN = (200, 120, 300, 200)


def cfg(**kw):
    c = PipelineConfig(redaction_method="Solid Mask", box_padding={}, cover_raw_box=False, ocr_enabled=False)
    c.update(**kw)
    return c


def make(script, mode=FaceMode.AUTO):
    policy = FacePolicy(FaceBook(ColorEmbedder()), mode)
    model = FakeYolo(script)
    c = cfg()
    return PreviewEngine(model, c, face_policy=policy), policy, model, c


def solid_changed(a, b, box):
    x1, y1, x2, y2 = box
    return not np.array_equal(a[y1:y2, x1:x2], b[y1:y2, x1:x2])


def test_toggling_options_re_renders_without_running_the_model_again():
    pv, policy, model, c = make([[("face", 0.9, A), ("face", 0.9, B), ("screen", 0.95, SCREEN)]] * 5)
    frame = face_frame([(A, 50), (B, 100)])
    analysis = pv.analyze(frame, 0)
    assert model.calls == 1 and len(analysis.objects) == 3
    r1 = pv.render(analysis)
    assert not np.array_equal(r1.image[20:80, 20:80], frame[20:80, 20:80]) and r1.redactions == 3
    c.enabled_classes.discard("screen")                                   # chip off
    r2 = pv.render(analysis)
    assert np.array_equal(r2.image[120:200, 200:300], frame[120:200, 200:300]) and r2.redactions == 2
    policy.set_keep_visible({analysis.objects[0].face_id})                # keep Face 01 visible
    r3 = pv.render(analysis)
    assert np.array_equal(r3.image[20:80, 20:80], frame[20:80, 20:80]) and r3.redactions == 1
    policy.set_mode("every")
    assert pv.render(analysis).redactions == 2
    assert model.calls == 1                                               # never re-detected


def test_preview_matches_processing_decision_rules_review_band_is_not_blurred():
    pv, *_ = make([[("credit_card", 0.65, SCREEN)]])
    analysis = pv.analyze(gradient_frame(), 0)
    r = pv.render(analysis)
    assert r.redactions == 0 and r.overlays[0].state == "review"


def test_overlays_carry_face_ids_and_state_for_click_selection():
    pv, policy, *_ = make([[("face", 0.9, A), ("face", 0.9, B)]], FaceMode.SELECTED)
    analysis = pv.analyze(face_frame([(A, 50), (B, 100)]), 0)
    policy.set_selected({analysis.objects[1].face_id})
    states = {o.label: o.state for o in pv.render(analysis).overlays}
    assert states == {"Face 01": "visible", "Face 02": "blurred"}


def test_object_at_prefers_the_smallest_box():
    pv, *_ = make([[("screen", 0.95, (0, 0, 300, 220)), ("face", 0.9, A)]])
    a = pv.analyze(face_frame([(A, 50)]), 0)
    assert a.object_at(40, 40).name == "face" and a.object_at(250, 200).name == "screen" and a.object_at(5, 235) is None


def test_manual_region_and_magic_target_appear_in_preview():
    pv, *_ = make([[("screen", 0.95, SCREEN)]])
    frame = gradient_frame()
    analysis = pv.analyze(frame, 0)
    target = pv.magic_target_at(analysis, 250, 150)
    assert target is not None and target.name == "screen" and target.signature is not None
    plan = EditPlan(manual=[ManualRegion(0.05, 0.6, 0.3, 0.9)], magic=[target])
    pv.config.enabled_classes.discard("screen")                         # magic works even if the class chip is off
    r = pv.render(analysis, plan)
    assert solid_changed(frame, r.image, (205, 125, 295, 195)) and solid_changed(frame, r.image, (20, 150, 90, 210))
    assert {o.kind for o in r.overlays} >= {"magic", "region"}


def test_magic_click_on_empty_space_gives_no_target():
    pv, *_ = make([[("screen", 0.95, SCREEN)]])
    assert pv.magic_target_at(pv.analyze(gradient_frame(), 0), 10, 10) is None


# ----------------------------------------------------------------------------- shape + tracking
def blob_frame(center, color=(30, 40, 220), size=(320, 240), radius=(35, 25)):
    f = np.full((size[1], size[0], 3), 90, np.uint8)
    cv2.ellipse(f, center, radius, 0, 0, 360, color, -1)
    return f


def test_object_mask_follows_the_shape_not_the_whole_box():
    f = blob_frame((160, 120))
    mask = object_mask(f, (115, 85, 205, 155))
    inside = mask[120, 160] > 0
    corner = mask[88, 118] > 0                                      # corner of the box, outside the ellipse
    box_area = 90 * 70
    assert inside and not corner and 0.35 * box_area < (mask > 0).sum() < 0.8 * box_area


def test_tracker_follows_same_looking_object_and_ignores_a_different_one():
    f0 = blob_frame((100, 120))
    box0 = (55, 90, 150, 150)
    target = MagicTarget("screen", box0, colour_signature(f0, box0), "screen")
    tracker = MagicTracker([target])
    f1 = blob_frame((200, 120))
    f1_other = f1.copy(); cv2.ellipse(f1_other, (60, 200), (20, 15), 0, 0, 360, (200, 200, 30), -1)
    dets = [{"name": "screen", "box": (155, 90, 250, 150)}, {"name": "screen", "box": (40, 185, 80, 215)},
            {"name": "face", "box": (155, 90, 250, 150)}]
    masks = tracker.update(f1_other, dets, 1)
    assert len(masks) == 1 and masks[0][120, 200] > 0 and masks[0][200, 60] == 0


def test_tracker_drops_target_after_it_is_gone_for_too_long():
    f0 = blob_frame((100, 120)); box0 = (55, 90, 150, 150)
    tracker = MagicTracker([MagicTarget("screen", box0, colour_signature(f0, box0))], max_gap=2)
    assert len(tracker.update(f0, [{"name": "screen", "box": box0}], 1)) == 1
    for run in range(2, 6):
        masks = tracker.update(f0, [], run)
    assert masks == []
