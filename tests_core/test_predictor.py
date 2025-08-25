import time
import unittest

import dlib
import numpy as np

from package import predictor


class TestPredictor(unittest.TestCase):
    def setUp(self):
        dlib_predictor = dlib.shape_predictor("models/dlib/shape_predictor_68_face_landmarks.dat")
        dlib_recognition_model = dlib.face_recognition_model_v1("models/dlib/dlib_face_recognition_resnet_model_v1.dat")
        features = np.array(
            [
                -0.0694541335105896,
                0.06430532783269882,
                0.0677134096622467,
                -0.02568257413804531,
                0.012633268721401691,
            ]
        )
        self.face_descriptor = {}
        self.face_descriptor["test_user"] = features
        self.sensitivity = 0.4
        data = np.load("./tests_core/test_data/test_image.npz")
        self.image = data["test_img"]
        self.predictor = predictor.Predictor(
            dlib_predictor, dlib_recognition_model, self.face_descriptor, self.sensitivity
        )

    # test feature_extraction
    def test_feature_extraction_correctness(self):
        face_descriptor, _ = self.predictor.feature_extraction(self.image)
        self.assertEqual(len(face_descriptor), 128)

    def test_feature_extraction_output_type(self):
        face_descriptor, _ = self.predictor.feature_extraction(self.image)
        self.assertIsInstance(face_descriptor, np.ndarray)

    def test_feature_extraction_invalid_input(self):
        face_descriptor, feature_coordinates = self.predictor.feature_extraction("invalid_input")
        self.assertEqual(face_descriptor, None)
        self.assertEqual(feature_coordinates, None)

    def test_feature_extraction_boundary_zero(self):
        image = np.zeros((774, 800, 3))
        face_descriptor, feature_coordinates = self.predictor.feature_extraction(image)
        self.assertEqual(face_descriptor, None)
        self.assertEqual(feature_coordinates, None)

    def test_feature_extraction_performance(self):
        start_time = time.time()
        self.predictor.feature_extraction(self.image)
        elapsed_time = time.time() - start_time
        # 性能判斷, 假設期望在 1 秒內完成
        self.assertLess(elapsed_time, 1, "Performance degraded, took too long to process.")

    # tett euclidean_distance
    def test_euclidean_distance_correctness(self):
        min_dist, matched_name = self.predictor.euclidean_distance(self.face_descriptor["test_user"])
        self.assertLessEqual(min_dist, self.sensitivity)
        features = np.array(
            [
                -1.0694541335105896,
                0.06430532783269882,
                0.0677134096622467,
                -0.02568257413804531,
                0.012633268721401691,
            ]
        )
        face_descriptor = {}
        face_descriptor["test_user"] = features
        min_dist, matched_name = self.predictor.euclidean_distance(face_descriptor["test_user"])
        self.assertGreaterEqual(min_dist, self.sensitivity)

    def test_euclidean_distance_output_type(self):
        min_dist, matched_name = self.predictor.euclidean_distance(self.face_descriptor["test_user"])
        self.assertIsInstance(min_dist, float)

    def test_euclidean_distance_invalid_input(self):
        min_dist, matched_name = self.predictor.euclidean_distance("invalid_input")
        self.assertEqual(min_dist, None)

    def test_euclidean_distance_performance(self):
        start_time = time.time()
        self.predictor.euclidean_distance(self.face_descriptor["test_user"])
        elapsed_time = time.time() - start_time
        # 性能判斷, 假設期望在 0.1 秒內完成
        self.assertLess(elapsed_time, 0.1, "Performance degraded, took too long to process.")
