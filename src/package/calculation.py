import csv
import time
import traceback

import cv2
import dlib
import mediapipe
import numpy as np

import package.config as config


class calculation:
    def get_face_boundingbox(self, bounding_box: mediapipe, width: int, height: int):
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

    def get_eyes_boundingbox(self, detection: mediapipe, bounding_height: mediapipe, width: int, height: int):
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

    def grayscale_area(self, eye_left_roi: np.ndarray, eye_right_roi: np.ndarray, threshold_value: int):
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

    def detection_range(self, detct_start: list, detct_end: list, face_box_center_x, face_box_center_y, bounding_box_height, detection_score, minimum_bounding_box_height, minimum_face_detection_score):
        return detct_start[0] < face_box_center_x < detct_end[0] and detct_start[1] < face_box_center_y < detct_end[1] and \
                bounding_box_height > minimum_bounding_box_height and detection_score > minimum_face_detection_score

    def easy_blink_detect(self, blink_list: np.ndarray):
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

    def save_feature(self, out_put_path: str, face_descriptor: np.ndarray):
        try:
            with open(out_put_path, mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(face_descriptor)
            return True
        except Exception as err:
            print("Sve feature save_feature mode error: ", err)
            return False

    def feature_extraction(self, face_roi, predictor, face_reco, face_features): # -> np.ndarray:
        start_time = time.time()
        try:
            landmarks_frame = cv2.cvtColor(face_roi, cv2. COLOR_BGR2RGB)
            dlib_coordinate = dlib.rectangle(0, 0, face_roi.shape[0], face_roi.shape[1])
            shape = predictor(landmarks_frame, dlib_coordinate)
            face_descriptor = np.array(face_reco.compute_face_descriptor(face_roi, shape))
            # if debug:
            #     # 繪製dlib特徵點
            #     for i in range(68):
            #         cv2.circle(face_roi,(shape.part(i).x, shape.part(i).y), 3, (0, 0, 255), 2)
            #         cv2.putText(face_roi, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            #     cv2.imshow("face_roi", face_roi)
            distance = self.euclidean_distance(face_features, face_descriptor)
            if distance <= 0.4:
                config.logger.info("pass.")
            else:
                config.logger.info("fail.")
            end_time = time.time()
            execution_time = round(end_time - start_time, 3)
            config.logger.info(f"Face Recognition Time: {execution_time} sec")
            return face_descriptor
        except Exception as e:
            config.logger.debug(traceback.print_exc())
            return False

    def euclidean_distance(self, face_features: np.ndarray, face_descriptor: np.ndarray):
        dist_list = []
        for original_features in face_features:
            dist = np.sqrt(np.sum(np.square(face_descriptor - original_features)))
            dist_list.append(dist)
        result = min(dist_list)
        config.logger.debug(f"Minimum euclidean distance: {result}")
        return result