import csv
import time
import traceback

import cv2
import dlib
import numpy as np

from src import config


class calculation:
    def get_face_boundingbox(self, frame, bounding_box, width, height):       
        # face bounding box
        bounding_x1, bounding_y1 = int(bounding_box.xmin * width), int(bounding_box.ymin * height)
        bounding_x2, bounding_y2 = int(bounding_x1 + bounding_box.width * width), int(bounding_y1 + bounding_box.height * height)
        center_x = (bounding_x1 + bounding_x2) // 2
        center_y = (bounding_y1 + bounding_y2) // 2        
        if config.debug:
            cv2.rectangle(frame, (bounding_x1, bounding_y1), (bounding_x2, bounding_y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 225), -1)
        return bounding_x1, bounding_y1, bounding_x2, bounding_y2, center_x, center_y

    def get_eyes_boundingbox(self, frame, detection, bounding_height, width, height):
        # eyes bounding box
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
        if config.debug:
            cv2.rectangle(frame, (bounding_eye_left_x1, bounding_eye_left_y1), (bounding_eye_left_x2, bounding_eye_left_y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (bounding_eye_right_x1, bounding_eye_right_y1), (bounding_eye_right_x2, bounding_eye_right_y2), (0, 255, 0), 2)
        return (
            bounding_eye_left_x1,
            bounding_eye_left_y1,
            bounding_eye_left_x2,
            bounding_eye_left_y2,
            bounding_eye_right_x1,
            bounding_eye_right_y1,
            bounding_eye_right_x2,
            bounding_eye_right_y2
            )

    def grayscale_area(self, eye_left_roi, eye_right_roi, threshold_value):
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
            if config.debug:
                cv2.imshow("eyes_left", left_eye_gary)
                cv2.imshow("eyes_right", right_eye_gary)
            return left_eye_gary, right_eye_gary
        except Exception as e:
            config.logger.debug(f"Calculate eyes grayscale area error: {e}")
            return None, None

    def detection_range(self, face_box_center_x, face_box_center_y, bounding_box_height, detection_score, minimum_bounding_box_height, minimum_face_detection_score):
        return self.detct_start[0] < face_box_center_x < self.detct_end[0] and self.detct_start[1] < face_box_center_y < self.detct_end[1] and \
                bounding_box_height > minimum_bounding_box_height and detection_score > minimum_face_detection_score

    def easy_blink_detect(self, blink_list):
        state = False
        if 0 in blink_list and np.count_nonzero(blink_list == 0) >= 3 and np.count_nonzero(blink_list == 1) >= 3 and blink_list[0] != 0 and blink_list[-1] != 0:
            first_zero_index = np.argmax(blink_list == 0)
            last_zero_index = len(blink_list) - 1 - np.argmax(np.flip(blink_list) == 0)
            if (last_zero_index - first_zero_index + 1) == np.count_nonzero(blink_list == 0):
                state = True
        else:
            state = False
        return state

    def save_feature(self, face_descriptor):
        try:
            with open(config.face_model, mode='a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(face_descriptor)
            return True
        except Exception as e:
            print(e)
            return False

    def feature_extraction(self, face_roi, predictor, face_reco):
        start_time = time.time()
        try:
            landmarks_frame = cv2.cvtColor(face_roi, cv2. COLOR_BGR2RGB)
            dlib_coordinate = dlib.rectangle(0, 0, face_roi.shape[0], face_roi.shape[1])
            shape = predictor(landmarks_frame, dlib_coordinate)
            face_descriptor = np.array(face_reco.compute_face_descriptor(face_roi, shape))
            if config.debug:
                # 繪製dlib特徵點
                for i in range(68):
                    cv2.circle(face_roi,(shape.part(i).x, shape.part(i).y), 3, (0, 0, 255), 2)
                    cv2.putText(face_roi, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imshow("face_roi", face_roi)
            if config.FACE_RECOGNITION_MODE:
                distance = self.euclidean_distance(face_descriptor)
                if distance <= 0.4:
                    config.logger.info("pass.")
                else:
                    config.logger.info("fail.")
                end_time = time.time()
                execution_time = round(end_time - start_time, 3)
                config.logger.info(f"Face Recognition Time: {execution_time} sec")
                return face_descriptor
            elif config.SETTING_MODE == 2:
                self.save_feature(face_descriptor)
                return True
        except Exception as e:
            config.logger.debug(traceback.print_exc())
            return False

    def euclidean_distance(self, face_descriptor):
        dist_list = []
        for original_features in config.face_features:
            dist = np.sqrt(np.sum(np.square(face_descriptor - original_features)))
            dist_list.append(dist)
        result = min(dist_list)
        config.logger.debug(f"Minimum euclidean distance: {result}")
        return result