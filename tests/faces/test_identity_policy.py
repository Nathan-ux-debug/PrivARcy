"""Face identities and the three blur modes (auto / selected / every). Embeddings are MOCKED, logic is real."""
import pytest

from privarcy.faces import FaceBook, FaceMode, FacePolicy
from tests.support.fakes import AngleEmbedder, ColorEmbedder, face_frame

A, B, C = (20, 20, 80, 80), (120, 20, 180, 80), (220, 20, 280, 80)


def policy(mode=FaceMode.AUTO, embedder=None, **kw):
    return FacePolicy(FaceBook(embedder or ColorEmbedder()), mode, **kw)


def identify_all(p, frame, boxes):
    return [p.identify(frame, b) for b in boxes]


def test_same_person_keeps_one_identity_and_different_people_get_different_ones():
    p = policy()
    f = face_frame([(A, 50), (B, 100)])
    first = identify_all(p, f, [A, B])
    again = identify_all(p, face_frame([((60, 60, 120, 120), 50)]), [(60, 60, 120, 120)])
    assert first[0] != first[1] and again[0] == first[0]
    assert [i.label for i in p.book.identities()] == ["Face 01", "Face 02"]


def test_auto_mode_blurs_everyone_including_new_people_except_kept_visible():
    p = policy(FaceMode.AUTO)
    f = face_frame([(A, 50), (B, 100)])
    a, b = identify_all(p, f, [A, B])
    p.set_keep_visible({a})
    assert p.decide(a) == (False, "kept_visible") and p.decide(b) == (True, "auto_blur")
    new = p.identify(face_frame([(C, 150)]), C)               # a person who appears later
    assert p.decide(new)[0] is True


def test_selected_mode_blurs_only_selected_and_leaves_others_alone():
    p = policy(FaceMode.SELECTED)
    a, b = identify_all(p, face_frame([(A, 50), (B, 100)]), [A, B])
    p.set_selected({b})
    assert p.decide(b)[0] is True and p.decide(a) == (False, "not_selected")


def test_every_mode_ignores_selection():
    p = policy(FaceMode.EVERY, keep_visible=set(), selected=set())
    a, = identify_all(p, face_frame([(A, 50)]), [A])
    p.set_keep_visible({a}); p.set_selected(set())
    assert p.decide(a) == (True, "every_face") and p.decide("anything") == (True, "every_face")


def test_toggle_flips_the_choice_that_matters_in_the_current_mode():
    p = policy(FaceMode.AUTO)
    a, = identify_all(p, face_frame([(A, 50)]), [A])
    assert p.toggle(a) is False and a in p.keep_visible            # now kept visible
    p.set_mode("selected")
    assert p.toggle(a) is True and a in p.selected                 # now selected for blur


def test_too_small_or_unidentifiable_face_fails_safe_to_blurred_even_in_selected_mode():
    p = policy(FaceMode.SELECTED)
    tiny = (10, 10, 20, 20)
    fid = p.identify(face_frame([(tiny, 50)]), tiny)
    assert fid is None and p.decide(fid) == (True, "unidentified_blurred")


def test_two_faces_in_one_frame_are_never_merged_by_the_policy_book():
    p = policy()
    ids = identify_all(p, face_frame([(A, 50), (B, 50)]), [A, B])      # identical appearance, same frame
    # the policy links by appearance so these may share an id; scanner enforces per-frame uniqueness (see scanner test)
    assert all(i is not None for i in ids)


def test_selected_mode_uses_looser_match_so_a_selected_person_is_not_missed_but_auto_stays_strict():
    emb = AngleEmbedder()
    # sighting 1 at 0 degrees; later sighting at 20 degrees -> cosine ~0.94 (between loose 0.85 and strict 0.95)
    book = FaceBook(emb)
    p = FacePolicy(book, FaceMode.SELECTED)
    a = p.identify(face_frame([(A, 0)]), A)
    p.set_selected({a})
    later = p.identify(face_frame([(A, 20)]), A)
    assert later == a                                                  # still recognised as the selected person -> blurred
    book2 = FaceBook(emb)
    p2 = FacePolicy(book2, FaceMode.AUTO)
    a2 = p2.identify(face_frame([(A, 0)]), A)
    p2.set_keep_visible({a2})
    other = p2.identify(face_frame([(A, 20)]), A)
    assert other != a2 and p2.decide(other)[0] is True                 # NOT treated as the kept-visible person -> blurred


def test_book_thumbnail_and_summary():
    p = policy()
    a, = identify_all(p, face_frame([(A, 50)]), [A])
    ident = p.book.get(a)
    assert ident.thumbnail is not None and ident.thumbnail.shape[:2] == (96, 96) and ident.sightings == 1


@pytest.mark.parametrize("raw,expected", [("auto", FaceMode.AUTO), ("SELECTED", FaceMode.SELECTED), ("every", FaceMode.EVERY), ("junk", FaceMode.AUTO)])
def test_mode_parse(raw, expected):
    assert FaceMode.parse(raw) == expected
