import time
import unittest

import cv2
import mediapipe as mp
import numpy

from package import calculation


class TestCalculation(unittest.TestCase):
    def setUp(self):
        self.calculation = calculation.Calculation(774, 800)
        data = numpy.load("./tests/test_data/test_image.npz")
        self.test_img = data["test_img"]
        self.detection_mp, self.bounding_box_mp = self.setup_mediapipe()
        self.bounding_box_height = round(self.bounding_box_mp.height, 2)

        # test eyes_pre_treatmentsing
        for_testing_bounding_eye_left, for_testing_bounding_eye_righ = self.setup_eyes_boundingbox()
        self.grayscale_value = 80
        self.eye_left_roi = self.test_img[
            for_testing_bounding_eye_left[0][1] : for_testing_bounding_eye_left[1][1],
            for_testing_bounding_eye_left[0][0] : for_testing_bounding_eye_left[1][0],
        ]
        self.eye_right_roi = self.test_img[
            for_testing_bounding_eye_righ[0][1] : for_testing_bounding_eye_righ[1][1],
            for_testing_bounding_eye_righ[0][0] : for_testing_bounding_eye_righ[1][0],
        ]

        # test blink_detect
        self.eyes_blink_true = [
            [810, 852, 868, 887, 837, 535, 288, 217, 347, 422, 549, 692, 764, 778, 818, 806],
            [883, 887, 861, 846, 800, 572, 332, 232, 337, 489, 535, 705, 692, 744, 745, 762],
        ]
        self.eyes_blink_false = [
            [576, 261, 166, 231, 466, 653, 758, 863, 873, 866, 886, 886, 874, 879, 888, 918],
            [596, 299, 229, 223, 431, 519, 644, 663, 662, 691, 703, 719, 762, 765, 756, 751],
        ]
        self.blink_count_true = 176
        self.blink_count_false = 177
        self.left_median = 677
        self.right_median = 688

    def setup_mediapipe(self):
        mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        frame_bgr = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2RGB)
        with mp_face_detection as face_detection:
            results = face_detection.process(frame_bgr)
            for detection_mp in results.detections:
                if results.detections:
                    bounding_box_mp = detection_mp.location_data.relative_bounding_box
                    return detection_mp, bounding_box_mp

    def setup_eyes_boundingbox(self):
        return self.calculation.get_eyes_boundingbox(self.detection_mp, self.bounding_box_height)

    # test get_face_boundingbox
    def test_get_face_boundingbox_correctness(self):
        face_bounding_box, center = self.calculation.get_face_boundingbox(self.bounding_box_mp)
        self.assertEqual(face_bounding_box, [[170, 184], [472, 506]])
        self.assertEqual(center, [321, 345])

    def test_get_face_boundingbox_output_type(self):
        face_bounding_box, center = self.calculation.get_face_boundingbox(self.bounding_box_mp)
        self.assertIsInstance(face_bounding_box, list)
        self.assertIsInstance(center, list)

    def test_process_all_cameras_invalid_input(self):
        with self.assertRaises(Exception):
            self.calculation.get_face_boundingbox("invalid_input")
        with self.assertRaises(Exception):
            self.calculation.get_face_boundingbox(None)
        with self.assertRaises(Exception):
            self.calculation.get_face_boundingbox(0.12345)

    def test_process_all_cameras_boundary_zero(self):
        with self.assertRaises(Exception):
            self.calculation.get_face_boundingbox(0)

    def test_process_all_cameras_performance(self):
        start_time = time.time()
        self.calculation.get_face_boundingbox(self.bounding_box_mp)
        elapsed_time = time.time() - start_time
        # 性能判斷, 假設期望在 0.1 秒內完成
        self.assertLess(elapsed_time, 0.1, "Performance degraded, took too long to process.")

    # test get_eyes_boundingbox
    def test_get_get_eyes_boundingbox_correctness(self):
        bounding_eye_left, bounding_eye_right = self.calculation.get_eyes_boundingbox(self.detection_mp, self.bounding_box_height)
        self.assertEqual(bounding_eye_left, [[241, 238], [292, 289]])
        self.assertEqual(bounding_eye_right, [[369, 238], [420, 289]])

    def test_get_eyes_boundingbox_output_type(self):
        bounding_eye_left, bounding_eye_right = self.calculation.get_eyes_boundingbox(self.detection_mp, self.bounding_box_height)
        self.assertIsInstance(bounding_eye_left, list)
        self.assertIsInstance(bounding_eye_right, list)

    def test_get_eyes_boundingbox_invalid_input(self):
        with self.assertRaises(Exception):
            self.calculation.get_eyes_boundingbox("invalid_input", "invalid_input")
        with self.assertRaises(Exception):
            self.calculation.get_eyes_boundingbox(None, self.bounding_box_height)
        with self.assertRaises(Exception):
            self.calculation.get_eyes_boundingbox(0.12345, 123.321)

    def test_get_eyes_boundingbox_boundary_zero(self):
        with self.assertRaises(Exception):
            self.calculation.get_eyes_boundingbox(0)

    def test_get_eyes_boundingbox_performance(self):
        start_time = time.time()
        self.calculation.get_eyes_boundingbox(self.detection_mp, self.bounding_box_height)
        elapsed_time = time.time() - start_time
        # 性能判斷, 假設期望在 0.1 秒內完成
        self.assertLess(elapsed_time, 0.1, "Performance degraded, took too long to process.")

    # test eyes_pre_treatmentsing
    def test_eyes_pre_treatmentsing_correctness(self):
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions(self.eye_left_roi, self.eye_right_roi, self.grayscale_value)
        self.assertEqual(left_eye_gary[30][30], 255)
        self.assertEqual(left_eye_gary.shape, (51, 51))
        self.assertEqual(right_eye_gary[30][30], 255)
        self.assertEqual(right_eye_gary.shape, (51, 51))

    def test_get_eyes_boundingbox_output_type(self):
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions(self.eye_left_roi, self.eye_right_roi, self.grayscale_value)
        self.assertIsInstance(left_eye_gary, numpy.ndarray)
        self.assertIsInstance(right_eye_gary, numpy.ndarray)

    def test_eyes_pre_treatmentsing_invalid_input(self):
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions("invalid_input", "invalid_input", "invalid_input")
        self.assertEqual(left_eye_gary, None)
        self.assertEqual(right_eye_gary, None)
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions(None, None, None)
        self.assertEqual(left_eye_gary, None)
        self.assertEqual(right_eye_gary, None)
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions(
            self.eye_left_roi, self.eye_right_roi, str(self.grayscale_value)
        )
        self.assertEqual(left_eye_gary, None)
        self.assertEqual(right_eye_gary, None)

    def test_eyes_pre_treatmentsing_boundary_zero(self):
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions([], [], [])
        self.assertEqual(left_eye_gary, None)
        self.assertEqual(right_eye_gary, None)

    def test_eyes_pre_treatmentsing_performance(self):
        start_time = time.time()
        calculation.Calculation.preprocess_eye_regions(self.eye_left_roi, self.eye_right_roi, self.grayscale_value)
        elapsed_time = time.time() - start_time
        # 性能判斷, 假設期望在 0.1 秒內完成
        self.assertLess(elapsed_time, 0.1, "Performance degraded, took too long to process.")

    # blink_detect
    def test_eyes_pre_treatmentsing_correctness(self):
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions(self.eye_left_roi, self.eye_right_roi, self.grayscale_value)
        self.assertEqual(left_eye_gary[30][30], 255)
        self.assertEqual(left_eye_gary.shape, (51, 51))
        self.assertEqual(right_eye_gary[30][30], 255)
        self.assertEqual(right_eye_gary.shape, (51, 51))

    def test_get_eyes_boundingbox_output_type(self):
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions(self.eye_left_roi, self.eye_right_roi, self.grayscale_value)
        self.assertIsInstance(left_eye_gary, numpy.ndarray)
        self.assertIsInstance(right_eye_gary, numpy.ndarray)

    def test_eyes_pre_treatmentsing_invalid_input(self):
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions("invalid_input", "invalid_input", "invalid_input")
        self.assertEqual(left_eye_gary, None)
        self.assertEqual(right_eye_gary, None)
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions(None, None, None)
        self.assertEqual(left_eye_gary, None)
        self.assertEqual(right_eye_gary, None)
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions(
            self.eye_left_roi, self.eye_right_roi, str(self.grayscale_value)
        )
        self.assertEqual(left_eye_gary, None)
        self.assertEqual(right_eye_gary, None)

    def test_eyes_pre_treatmentsing_boundary_zero(self):
        left_eye_gary, right_eye_gary = calculation.Calculation.preprocess_eye_regions([], [], [])
        self.assertEqual(left_eye_gary, None)
        self.assertEqual(right_eye_gary, None)

    def test_eyes_pre_treatmentsing_performance(self):
        start_time = time.time()
        calculation.Calculation.preprocess_eye_regions(self.eye_left_roi, self.eye_right_roi, self.grayscale_value)
        elapsed_time = time.time() - start_time
        # 性能判斷, 假設期望在 0.1 秒內完成
        self.assertLess(elapsed_time, 0.1, "Performance degraded, took too long to process.")

    # blink_detect
    def test_blink_detect_correctness(self):
        blink_state, left_median, right_median = calculation.Calculation.blink_detect(
            self.eyes_blink_true, self.blink_count_true, self.left_median, self.right_median
        )
        self.assertEqual(blink_state, True)
        self.assertEqual(left_median, 677)
        self.assertEqual(right_median, 688)
        blink_state, left_median, right_median = calculation.Calculation.blink_detect(
            self.eyes_blink_false, self.blink_count_false, self.left_median, self.right_median
        )
        self.assertEqual(blink_state, False)
        self.assertEqual(left_median, 677)
        self.assertEqual(right_median, 688)

    def test_blink_detect_output_type(self):
        blink_state, left_median, right_median = calculation.Calculation.blink_detect(
            self.eyes_blink_true, self.blink_count_true, self.left_median, self.right_median
        )
        self.assertIsInstance(blink_state, bool)
        self.assertIsInstance(left_median, int)
        self.assertIsInstance(right_median, int)

    def test_blink_detect_invalid_input(self):
        blink_state, left_median, right_median = calculation.Calculation.blink_detect(
            "invalid_input", "invalid_input", "invalid_input", "invalid_input"
        )
        self.assertEqual(blink_state, None)
        self.assertEqual(left_median, None)
        self.assertEqual(right_median, None)
        blink_state, left_median, right_median = calculation.Calculation.blink_detect(
            None, self.blink_count_true, self.left_median, self.right_median
        )
        self.assertEqual(blink_state, None)
        self.assertEqual(left_median, None)
        self.assertEqual(right_median, None)
        blink_state, left_median, right_median = calculation.Calculation.blink_detect(1, 1, 1, "60")
        self.assertEqual(blink_state, None)
        self.assertEqual(left_median, None)
        self.assertEqual(right_median, None)

    def test_blink_detect_boundary_zero(self):
        blink_state, left_median, right_median = calculation.Calculation.blink_detect(0, 0, 0, 0)
        self.assertEqual(blink_state, None)
        self.assertEqual(left_median, None)
        self.assertEqual(right_median, None)
        blink_state, left_median, right_median = calculation.Calculation.blink_detect(
            self.eyes_blink_true, self.blink_count_true, self.left_median, str(0)
        )
        self.assertEqual(blink_state, None)
        self.assertEqual(left_median, None)
        self.assertEqual(right_median, None)

    def test_blink_detect_performance(self):
        start_time = time.time()
        calculation.Calculation.preprocess_eye_regions(self.eye_left_roi, self.eye_right_roi, self.grayscale_value)
        elapsed_time = time.time() - start_time
        # 性能判斷, 假設期望在 0.1 秒內完成
        self.assertLess(elapsed_time, 0.1, "Performance degraded, took too long to process.")
