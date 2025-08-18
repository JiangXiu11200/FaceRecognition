import time
import unittest

from package import coordinate_detection


class TesttestCoordinateDetection(unittest.TestCase):
    def setUp(self):
        self.coordinate_detection = coordinate_detection.CoordinateDetection([370, 200], [910, 520], 0.8, 0.4)
        self.face_box_center = [657, 373]
        self.bounding_box_height = 0.56
        self.detection_score = 0.92

    def test_face_box_in_roi_correctness(self):
        result = self.coordinate_detection.face_box_in_roi(
            self.face_box_center, self.bounding_box_height, self.detection_score
        )
        self.assertEqual(result, True)
        face_box_center = [649, 396]
        bounding_box_height = 0.34
        detection_score = 0.95
        result = self.coordinate_detection.face_box_in_roi(face_box_center, bounding_box_height, detection_score)
        self.assertEqual(result, False)

    def test_face_box_in_roi_output_type(self):
        result = self.coordinate_detection.face_box_in_roi(
            self.face_box_center, self.bounding_box_height, self.detection_score
        )
        self.assertIsInstance(result, bool)

    def test_face_box_in_roi_invalid_input(self):
        result = self.coordinate_detection.face_box_in_roi("invalid_input", "invalid_input", "invalid_input")
        self.assertEqual(result, False)

    def test_face_box_in_roi_boundary_zero(self):
        result = self.coordinate_detection.face_box_in_roi(0, 0, 0)
        self.assertEqual(result, False)

    def test_face_box_in_roi_performance(self):
        start_time = time.time()
        self.coordinate_detection.face_box_in_roi(
            self.face_box_center[0], self.bounding_box_height, self.detection_score
        )
        elapsed_time = time.time() - start_time
        self.assertLess(elapsed_time, 0.1, "Performance degraded, took too long to process.")
