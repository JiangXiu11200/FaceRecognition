import csv
import time
import traceback
from multiprocessing import Queue

import cv2
import dlib
import numpy as np

import package.config as config


class Predictor:
    def __init__(
        self,
        dlib_predictor: dlib.shape_predictor,
        dlib_recognition_model: dlib.face_recognition_model_v1,
        registered_face_descriptor: np.ndarray,
        sensitivity: float,
    ):
        self.dlib_predictor = dlib_predictor
        self.dlib_recognition_model = dlib_recognition_model
        self.registered_face_descriptor = registered_face_descriptor
        self.sensitivity = sensitivity

    @staticmethod
    def save_feature(out_put_path: str, face_descriptor: np.ndarray):
        """
        Save face descriptor.

        Parameters:
            out_put_path (str): The output path.
            face_descriptor (np.ndarray): The face descriptor.

        Returns:
            result (bool): Save status.
        """
        try:
            with open(out_put_path, mode="a+", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(face_descriptor)
            return True
        except Exception as err:
            print("Save feature save_feature mode error: ", err)
            return False

    def face_prediction(self, face_roi: np.ndarray, detection_results: Queue):
        """
        Predict faces.

        Parameters:
            face_roi (np.ndarray): The face ROI.

        Returns:
            result (bool): Prediction results.
        """
        try:
            start_time = time.time()
            result = False
            current_face_descriptor, _ = self.feature_extraction(face_roi)
            distance = self.euclidean_distance(current_face_descriptor)
            if distance <= self.sensitivity:
                config.logger.info("pass.")
                detection_results.put([True, distance])
                result = True
            else:
                config.logger.info("fail.")
                detection_results.put([False, distance])
                result = False
            end_time = time.time()
            execution_time = round(end_time - start_time, 3)
            config.logger.info(f"Face Recognition Time: {execution_time} sec.")
            return result
        except Exception as err:
            print("face_prediction error: ", err)
            config.logger.debug(traceback.print_exc())
            return False

    def feature_extraction(self, face_roi: np.ndarray):
        """
        Get face descriptor by dlib.

        Parameters:
            face_roi (np.ndarray): The face ROI.

        Returns:
            face_descriptor (np.ndarray): The face descriptor.
        """
        try:
            if np.mean(face_roi) < 10:
                config.logger.error("The image is incorrect.")
                return None, None
            landmarks_frame = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            dlib_coordinate = dlib.rectangle(0, 0, face_roi.shape[0], face_roi.shape[1])
            feature_coordinates = self.dlib_predictor(landmarks_frame, dlib_coordinate)
            current_face_descriptor = np.array(self.dlib_recognition_model.compute_face_descriptor(face_roi, feature_coordinates))
            return current_face_descriptor, feature_coordinates
        except Exception as err:
            print("feature_extraction error: ", err)
            config.logger.debug(traceback.print_exc())
            return None, None

    def euclidean_distance(self, current_face_descriptor: np.ndarray):
        """
        Calculate the Euclidean distance between the current face descriptor and the loaded model.

        Parameters:
            current_face_descriptor (np.ndarray): The current face descriptor.

        Returns:
            result (float): The Euclidean distance.
        """
        try:
            dist_list = []
            for original_features in self.registered_face_descriptor:
                dist = np.sqrt(np.sum(np.square(current_face_descriptor - original_features)))
                dist_list.append(dist)
            result = min(dist_list)
            config.logger.debug(f"Minimum euclidean distance: {result}")
            return result
        except Exception as err:
            print("euclidean_distance error: ", err)
            config.logger.debug(traceback.print_exc())
            return None
