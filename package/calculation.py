import cv2
import mediapipe
import numpy as np

import package.config as config


class Calculation:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    def get_face_boundingbox(self, bounding_box: mediapipe):
        """
        Get the face bounding box coordinates converted by mediapipe data format to the actual image size.

        Parameters:
            bounding_box (mediapipe.bounding_box): The bounding box coordinates of the face.

        Returns:
            bounding_box (list): The bounding box coordinates of the face.
            center (list): The center coordinates of the face.
        """
        bounding_x1, bounding_y1 = int(bounding_box.xmin * self.image_width), int(bounding_box.ymin * self.image_height)
        bounding_x2, bounding_y2 = (
            int(bounding_x1 + bounding_box.width * self.image_width),
            int(bounding_y1 + bounding_box.height * self.image_height),
        )
        center_x = (bounding_x1 + bounding_x2) // 2
        center_y = (bounding_y1 + bounding_y2) // 2
        bounding_box = [[bounding_x1, bounding_y1], [bounding_x2, bounding_y2]]
        center = [center_x, center_y]
        return bounding_box, center

    def get_eyes_boundingbox(self, detection: mediapipe, bounding_height: mediapipe):
        """
        Get the eyes bounding box coordinates converted by mediapipe data format to the actual image size.

        Parameters:
            detection (mediapipe.detection): The detection data of the face.
            bounding_height (mediapipe.bounding_box): The bounding box height of the face.

        Returns:
            bounding_eye_left (list): The left eye bounding box coordinates.
            bounding_eye_right (list): The right eye bounding box coordinates.
        """
        eye_left = detection.location_data.relative_keypoints[0]
        eye_right = detection.location_data.relative_keypoints[1]
        eye_left_x, eye_left_y = int(eye_left.x * self.image_width), int(eye_left.y * self.image_height)
        eye_right_x, eye_right_y = int(eye_right.x * self.image_width), int(eye_right.y * self.image_height)
        eye_proportion = (bounding_height * self.image_height) * 0.08
        bounding_eye_left_x1, bounding_eye_left_y1, bounding_eye_left_x2, bounding_eye_left_y2 = (
            int(eye_left_x - eye_proportion),
            int(eye_left_y - eye_proportion),
            int(eye_left_x + eye_proportion),
            int(eye_left_y + eye_proportion),
        )
        bounding_eye_right_x1, bounding_eye_right_y1, bounding_eye_right_x2, bounding_eye_right_y2 = (
            int(eye_right_x - eye_proportion),
            int(eye_right_y - eye_proportion),
            int(eye_right_x + eye_proportion),
            int(eye_right_y + eye_proportion),
        )

        bounding_eye_left = [[bounding_eye_left_x1, bounding_eye_left_y1], [bounding_eye_left_x2, bounding_eye_left_y2]]
        bounding_eye_right = [
            [bounding_eye_right_x1, bounding_eye_right_y1],
            [bounding_eye_right_x2, bounding_eye_right_y2],
        ]
        return bounding_eye_left, bounding_eye_right

    @staticmethod
    def preprocess_eye_regions(eye_left_roi: np.ndarray, eye_right_roi: np.ndarray, threshold_value: int):
        """
        Grayscale the eyes ROI and image processing.

        Parameters:
            eye_left_roi (np.ndarray): The left eye ROI.
            eye_right_roi (np.ndarray): The right eye ROI.
            threshold_value (int): The threshold value for grayscale.

        Returns:
            left_eye_gary (np.ndarray): The grayscaled left eye ROI.
            right_eye_gary (np.ndarray): The grayscaled right eye ROI.
        """
        try:
            if eye_left_roi.size == 0:
                eye_left_roi = eye_right_roi
            elif eye_right_roi.size == 0:
                eye_right_roi = eye_left_roi
            left_eye_gary = cv2.cvtColor(eye_left_roi, cv2.COLOR_BGR2GRAY)
            right_eye_gary = cv2.cvtColor(eye_right_roi, cv2.COLOR_BGR2GRAY)
            left_eye_gary = cv2.GaussianBlur(left_eye_gary, (3, 3), 0)
            right_eye_gary = cv2.GaussianBlur(right_eye_gary, (3, 3), 0)
            ret, left_eye_gary = cv2.threshold(left_eye_gary, threshold_value, 255, cv2.THRESH_BINARY)
            ret, right_eye_gary = cv2.threshold(right_eye_gary, threshold_value, 255, cv2.THRESH_BINARY)
            return left_eye_gary, right_eye_gary
        except Exception as e:
            config.logger.debug(f"Calculate eyes grayscale area error: {e}")
            return None, None

    @staticmethod
    def blink_detect(eyes_blink: list, blink_count: int, left_median: int, right_median: int):
        """
        Detect blinking of both eyes.

        Parameters:
            eyes_blink (list): The eyes blink list.
            blink_count (int): The data length of the eyes_blink.
            left_median (int): The left eye median.
            right_median (int): The right eye median.

        Returns:
            blink_state (bool): The blink state.
            left_median (int): The left eye median.
            right_median (int): The right eye median.
        """
        try:
            eyes_blink[0].pop(0)
            eyes_blink[1].pop(0)
            left_blink = np.array(eyes_blink[0])
            right_blink = np.array(eyes_blink[1])
            if blink_count % 15 == 0:
                left_median = int(np.median(eyes_blink[0]) * 0.8)
                right_median = int(np.median(eyes_blink[1]) * 0.8)
                blink_state = False
            left_blink = (left_blink > left_median).astype(int)
            right_blink = (right_blink > right_median).astype(int)
            left_blink_state = Calculation._easy_eye_list_calculation(left_blink)
            right_blink_state = Calculation._easy_eye_list_calculation(right_blink)
            if left_blink_state and right_blink_state:  # both eyes blink
                blink_state = True
            else:
                blink_state = False
            return blink_state, left_median, right_median
        except Exception as err:
            print(f"blink_detect error: {err}")
            return None, None, None

    @staticmethod
    def _easy_eye_list_calculation(blink_list: np.ndarray):
        """
        Check whether to blink.

        Parameters:
            blink_list (np.ndarray): The blink list.

        Returns:
            state (bool): The blink state.
        """
        state = False
        if (
            0 in blink_list
            and np.count_nonzero(blink_list == 0) >= 3
            and np.count_nonzero(blink_list == 1) >= 3
            and blink_list[0] != 0
            and blink_list[-1] != 0
        ):
            first_zero_index = np.argmax(blink_list == 0)
            last_zero_index = len(blink_list) - 1 - np.argmax(np.flip(blink_list) == 0)
            if (last_zero_index - first_zero_index + 1) == np.count_nonzero(blink_list == 0):
                state = True
        else:
            state = False
        return state
