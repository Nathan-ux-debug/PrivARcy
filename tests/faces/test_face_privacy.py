"""
Unit tests for the face_privacy package.

Note on face detection tests: Haar cascades are trained on real face
statistics and generally do NOT reliably fire on procedurally-drawn
"cartoon" faces, so detector tests here focus on error handling and
config validation rather than asserting a synthetic image is detected
as a face. Validate detection accuracy against real photos separately
before relying on it (see README).

Run with:
    python -m unittest tests.faces.test_face_privacy -v
"""

import unittest

import numpy as np

from privarcy.faces import FaceDetector, FaceEmbedder, FaceMatcher, FaceBox, FacePrivacyError


def _random_face_crop(seed: int, size=(80, 80)) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (*size, 3), dtype=np.uint8)


class TestFaceDetector(unittest.TestCase):
    def test_invalid_frame_raises(self):
        detector = FaceDetector()
        with self.assertRaises(FacePrivacyError):
            detector.detect(None)
        with self.assertRaises(FacePrivacyError):
            detector.detect(np.array([]))

    def test_blank_frame_returns_no_faces(self):
        detector = FaceDetector()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        self.assertEqual(faces, [])

    def test_invalid_scale_factor_raises(self):
        with self.assertRaises(FacePrivacyError):
            FaceDetector(scale_factor=1.0)

    def test_invalid_min_neighbors_raises(self):
        with self.assertRaises(FacePrivacyError):
            FaceDetector(min_neighbors=0)

    def test_bad_cascade_path_raises(self):
        with self.assertRaises(FacePrivacyError):
            FaceDetector(cascade_path="/nonexistent/path.xml")

    def test_facebox_exposes_class_name_for_redaction_compat(self):
        box = FaceBox(bbox=(0, 0, 10, 10), confidence=0.9)
        self.assertEqual(box.class_name, "face")  # redaction.FrameRedactor reads this attribute


class TestFaceEmbedder(unittest.TestCase):
    def test_embedding_is_normalized_and_consistent_shape(self):
        embedder = FaceEmbedder()
        crop = _random_face_crop(1)
        vec = embedder.embed(crop)
        self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=5)

    def test_same_crop_gives_identical_embedding(self):
        embedder = FaceEmbedder()
        crop = _random_face_crop(2)
        vec1 = embedder.embed(crop)
        vec2 = embedder.embed(crop.copy())
        self.assertTrue(np.allclose(vec1, vec2))

    def test_different_size_crops_give_same_length_embedding(self):
        embedder = FaceEmbedder()
        vec_small = embedder.embed(_random_face_crop(3, size=(40, 40)))
        vec_large = embedder.embed(_random_face_crop(3, size=(200, 200)))
        self.assertEqual(vec_small.shape, vec_large.shape)

    def test_invalid_crop_raises(self):
        embedder = FaceEmbedder()
        with self.assertRaises(FacePrivacyError):
            embedder.embed(None)
        with self.assertRaises(FacePrivacyError):
            embedder.embed(np.array([]))

    def test_grayscale_input_accepted(self):
        embedder = FaceEmbedder()
        gray_crop = np.random.randint(0, 255, (80, 80), dtype=np.uint8)
        vec = embedder.embed(gray_crop)  # should not raise
        self.assertGreater(len(vec), 0)


class TestFaceMatcher(unittest.TestCase):
    def test_identical_embedding_matches_with_high_similarity(self):
        matcher = FaceMatcher(similarity_threshold=0.75)
        embedding = np.array([1.0, 0.0, 0.0])
        face = FaceBox(bbox=(0, 0, 10, 10), confidence=0.9)
        result = matcher.match(face, embedding, enrolled=[("Alice", embedding)])
        self.assertTrue(result.is_known)
        self.assertEqual(result.person_name, "Alice")
        self.assertAlmostEqual(result.similarity, 1.0, places=5)

    def test_orthogonal_embedding_does_not_match(self):
        matcher = FaceMatcher(similarity_threshold=0.75)
        face = FaceBox(bbox=(0, 0, 10, 10), confidence=0.9)
        query = np.array([1.0, 0.0, 0.0])
        enrolled_vec = np.array([0.0, 1.0, 0.0])  # orthogonal -> similarity 0
        result = matcher.match(face, query, enrolled=[("Bob", enrolled_vec)])
        self.assertFalse(result.is_known)
        self.assertIsNone(result.person_name)

    def test_empty_registry_means_unknown(self):
        matcher = FaceMatcher()
        face = FaceBox(bbox=(0, 0, 10, 10), confidence=0.9)
        result = matcher.match(face, np.array([1.0, 0.0]), enrolled=[])
        self.assertFalse(result.is_known)

    def test_picks_closest_of_multiple_people(self):
        matcher = FaceMatcher(similarity_threshold=0.5)
        face = FaceBox(bbox=(0, 0, 10, 10), confidence=0.9)
        query = np.array([1.0, 0.0, 0.0])
        enrolled = [
            ("Alice", np.array([0.0, 1.0, 0.0])),   # similarity 0
            ("Bob", np.array([0.9, 0.1, 0.0]) / np.linalg.norm([0.9, 0.1, 0.0])),  # close to query
        ]
        result = matcher.match(face, query, enrolled)
        self.assertEqual(result.person_name, "Bob")

    def test_invalid_threshold_raises(self):
        with self.assertRaises(FacePrivacyError):
            FaceMatcher(similarity_threshold=1.5)

    def test_shape_mismatch_raises(self):
        matcher = FaceMatcher()
        face = FaceBox(bbox=(0, 0, 10, 10), confidence=0.9)
        with self.assertRaises(FacePrivacyError):
            matcher.match(face, np.array([1.0, 0.0]), enrolled=[("Alice", np.array([1.0, 0.0, 0.0]))])

    def test_match_many_batch(self):
        matcher = FaceMatcher(similarity_threshold=0.9)
        face1 = FaceBox(bbox=(0, 0, 10, 10), confidence=0.9)
        face2 = FaceBox(bbox=(20, 20, 30, 30), confidence=0.9)
        emb = np.array([1.0, 0.0])
        results = matcher.match_many([(face1, emb), (face2, emb)], enrolled=[("Alice", emb)])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.is_known for r in results))


if __name__ == "__main__":
    unittest.main()
