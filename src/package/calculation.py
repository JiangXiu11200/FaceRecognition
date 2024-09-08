import csv
import time
import traceback

import cv2
import dlib
import mediapipe
import numpy as np

import src.package.config as config


class Calculation:
    def get_face_boundingbox(bounding_box: mediapipe, width: int, height: int):
        '''
        Get the face bounding box coordinates converted by mediapipe data format to the actual image size.

        Args:
            bounding_box (mediapipe.bounding_box): The bounding box coordinates of the face.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            bounding_box (list): The bounding box coordinates of the face.
            center (list): The center coordinates of the face.
        '''
        bounding_x1, bounding_y1 = int(bounding_box.xmin * width), int(bounding_box.ymin * height)
        bounding_x2, bounding_y2 = int(bounding_x1 + bounding_box.width * width), int(bounding_y1 + bounding_box.height * height)
        center_x = (bounding_x1 + bounding_x2) // 2
        center_y = (bounding_y1 + bounding_y2) // 2        
        bounding_box = [[bounding_x1, bounding_y1], [bounding_x2, bounding_y2]]
        center = [center_x, center_y]
        return bounding_box, center

    def get_eyes_boundingbox(detection: mediapipe, bounding_height: mediapipe, width: int, height: int):
        '''
        Get the eyes bounding box coordinates converted by mediapipe data format to the actual image size.
        
        Args:
            detection (mediapipe.detection): The detection data of the face.
            bounding_height (mediapipe.bounding_box): The bounding box height of the face.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            bounding_eye_left (list): The left eye bounding box coordinates.
            bounding_eye_right (list): The right eye bounding box coordinates.
        '''
        eye_left = detection.location_data.relative_keypoints[0]
        eye_right = detection.location_data.relative_keypoints[1]
        eye_left_x, eye_left_y = int(eye_left.x * width), int(eye_left.y * height)
        eye_right_x, eye_right_y = int(eye_right.x * width), int(eye_right.y * height)
        eye_proportion = (bounding_height * height) * 0.08
        bounding_eye_left_x1, bounding_eye_left_y1, \
        bounding_eye_left_x2, bounding_eye_left_y2 = int(eye_left_x - eye_proportion), int(eye_left_y - eye_proportion), \
                                                    int(eye_left_x + eye_proportion), int(eye_left_y + eye_proportion)
        bounding_eye_right_x1, bounding_eye_right_y1, \
        bounding_eye_right_x2, bounding_eye_right_y2 = int(eye_right_x - eye_proportion), int(eye_right_y - eye_proportion), \
                                                    int(eye_right_x + eye_proportion), int(eye_right_y + eye_proportion)
        
        bounding_eye_left = [[bounding_eye_left_x1, bounding_eye_left_y1], [bounding_eye_left_x2, bounding_eye_left_y2]]
        bounding_eye_right = [[bounding_eye_right_x1, bounding_eye_right_y1], [bounding_eye_right_x2, bounding_eye_right_y2]]
        return bounding_eye_left, bounding_eye_right

    def grayscale_area(eye_left_roi: np.ndarray, eye_right_roi: np.ndarray, threshold_value: int):
        '''
        Grayscale the eyes ROI and image pre-processing blur.

        Args:
            eye_left_roi (np.ndarray): The left eye ROI.
            eye_right_roi (np.ndarray): The right eye ROI.
            threshold_value (int): The threshold value for grayscale.

        Returns:
            left_eye_gary (np.ndarray): The grayscaled left eye ROI.
            right_eye_gary (np.ndarray): The grayscaled right eye ROI.
        '''
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

    def blink_detect(eyes_blink: list, blink_count: int, left_median: int, right_median: int):
        '''
        Detect blinking of both eyes.

        Args:
            eyes_blink (list): The eyes blink list.
            blink_count (int): The data length of the eyes_blink.
            left_median (int): The left eye median.
            right_median (int): The right eye median.

        Returns:
            blink_state (bool): The blink state.
            left_median (int): The left eye median.
            right_median (int): The right eye median.
        '''
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
            left_blink_state = Calculation.easy_blink_detect(left_blink)
            right_blink_state = Calculation.easy_blink_detect(right_blink)
            if left_blink_state and right_blink_state:  # both eyes blink
                blink_state = True
            else:
                blink_state = False
            return blink_state, left_median, right_median
        except Exception as err:
            print(f"blink_detect error: {err}")
            return None

    def detection_range(detct_start: list, detct_end: list, face_box_center_x, face_box_center_y, bounding_box_height, detection_score, \
                        minimum_bounding_box_height, minimum_face_detection_score):
        return detct_start[0] < face_box_center_x < detct_end[0] and detct_start[1] < face_box_center_y < detct_end[1] and \
                bounding_box_height > minimum_bounding_box_height and detection_score > minimum_face_detection_score

    def easy_blink_detect(blink_list: np.ndarray):
        '''
        Check whether to blink.

        Args:
            blink_list (np.ndarray): The blink list.

        Returns:
            state (bool): The blink state.
        '''
        state = False
        if 0 in blink_list and np.count_nonzero(blink_list == 0) >= 3 and np.count_nonzero(blink_list == 1) >= 3 and blink_list[0] != 0 and blink_list[-1] != 0:
            first_zero_index = np.argmax(blink_list == 0)
            last_zero_index = len(blink_list) - 1 - np.argmax(np.flip(blink_list) == 0)
            if (last_zero_index - first_zero_index + 1) == np.count_nonzero(blink_list == 0):
                state = True
        else:
            state = False
        return state

    def save_feature(out_put_path: str, face_descriptor: np.ndarray):
        '''
        Save face descriptor.
        
        Args:
            out_put_path (str): The output path.
            face_descriptor (np.ndarray): The face descriptor.

        Returns:
            result (bool): Save status.
        '''
        try:
            with open(out_put_path, mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(face_descriptor)
            return True
        except Exception as err:
            print("Save feature save_feature mode error: ", err)
            return False

    def face_prediction(face_roi: np.ndarray, dlib_predictor: dlib.shape_predictor, dlib_face_reco_model: dlib.face_recognition_model_v1, \
                        registered_face_descriptor: np.ndarray, sensitivity: float):
        '''
        Predict faces.
        
        Args:
            face_roi (np.ndarray): The face ROI.
            dlib_predictor (dlib.shape_predictor): The dlib predictor.
            dlib_face_reco (dlib.face_recognition_model_v1): The dlib face recognition model.
            registered_face_descriptor (np.ndarray): The registered face descriptor. (face model data)

        Returns:
            result (bool): Prediction results.
        '''
        try:
            start_time = time.time()
            result = False
            current_face_descriptor, _ = Calculation.feature_extraction(face_roi, dlib_predictor, dlib_face_reco_model)
            distance = Calculation.euclidean_distance(registered_face_descriptor, current_face_descriptor)
            if distance <= sensitivity:
                config.logger.info("pass.")
                result = True
            else:
                config.logger.info("fail.")
                result = False
            end_time = time.time()
            execution_time = round(end_time - start_time, 3)
            config.logger.info(f"Face Recognition Time: {execution_time} sec")
            return result
        except Exception as err:
            print("face_prediction error: ", err)
            config.logger.debug(traceback.print_exc())
            return False

    def feature_extraction(face_roi: np.ndarray, dlib_predictor: dlib.shape_predictor, dlib_face_reco_model: dlib.face_recognition_model_v1):
        '''
        Get face descriptor by dlib.

        Args:
            face_roi (np.ndarray): The face ROI.
            predictor (dlib.shape_predictor): The dlib predictor.
            dlib_face_reco (dlib.face_recognition_model_v1): The dlib face recognition model.
            face_features (np.ndarray): The face features.

        Returns:
            face_descriptor (np.ndarray): The face descriptor.
        '''
        try:
            landmarks_frame = cv2.cvtColor(face_roi, cv2. COLOR_BGR2RGB)
            dlib_coordinate = dlib.rectangle(0, 0, face_roi.shape[0], face_roi.shape[1])
            feature_coordinates = dlib_predictor(landmarks_frame, dlib_coordinate)
            current_face_descriptor = np.array(dlib_face_reco_model.compute_face_descriptor(face_roi, feature_coordinates))
            return current_face_descriptor, feature_coordinates
        except Exception as err:
            print("feature_extraction error: ", err)
            config.logger.debug(traceback.print_exc())
            return False, False

    def euclidean_distance(registered_face_descriptor: np.ndarray, current_face_descriptor: np.ndarray):
        '''
        Calculate the Euclidean distance between the current face descriptor and the loaded model.
        
        Args:
            registered_face_descriptor (np.ndarray): Registered face descriptors imported by the model.
            current_face_descriptor (np.ndarray): The current face descriptor.

        Returns:
            result (float): The Euclidean distance.
        '''
        dist_list = []
        for original_features in registered_face_descriptor:
            dist = np.sqrt(np.sum(np.square(current_face_descriptor - original_features)))
            dist_list.append(dist)
        result = min(dist_list)
        config.logger.debug(f"Minimum euclidean distance: {result}")
        return result