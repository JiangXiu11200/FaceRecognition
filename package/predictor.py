import csv
import time
import traceback
from multiprocessing import Queue
from typing import Optional

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
    def save_feature(out_put_path: str, face_descriptor: np.ndarray, user_name: str = "User") -> bool:
        """
        Save face descriptor to CSV file. \n
        CSV format: name,f1,f2,...,f128

        Parameters:
            out_put_path (str): The output path.
            face_descriptor (np.ndarray): The face descriptor.
            user_name (str): The name of the user.

        Returns:
            result (bool): Save status.
        """
        try:
            if isinstance(face_descriptor, np.ndarray):
                face_descriptor = face_descriptor.tolist()
            with open(out_put_path, mode="a+", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([user_name] + face_descriptor)
            config.logger.info(f"Feature saved successfully for {user_name}.")
            return True
        except Exception as err:
            config.logger.error(f"Unable to save feature for {user_name}. Error: {err}")
            config.logger.debug(traceback.print_exc())
            return False

    def face_prediction(self, face_roi: np.ndarray, detection_results: Queue) -> bool:
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
            distance, name = self.euclidean_distance(current_face_descriptor)
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

    def feature_extraction(
        self, face_roi: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[dlib.full_object_detection]]:
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
            current_face_descriptor = np.array(
                self.dlib_recognition_model.compute_face_descriptor(face_roi, feature_coordinates)
            )
            return current_face_descriptor, feature_coordinates
        except Exception as err:
            print("feature_extraction error: ", err)
            config.logger.debug(traceback.print_exc())
            return None, None

    def euclidean_distance(self, current_face_descriptor: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between the current face descriptor and the loaded model.

        Parameters:
            current_face_descriptor (np.ndarray): The current face descriptor.

        Returns:
            result (float): The Euclidean distance.
        """
        try:
            if len(self.registered_face_descriptor) == 0:
                config.logger.error("Model data is empty.")
                return 999, "Unknown"

            min_dist = float("inf")
            matched_name = None

            for name, features in self.registered_face_descriptor.items():
                dist = np.linalg.norm(current_face_descriptor - features)
                if dist < min_dist:
                    min_dist = dist
                    matched_name = name

            config.logger.debug(f"Minimum distance: {min_dist}, matched: {matched_name}")
            return min_dist, matched_name
        except Exception as err:
            print("euclidean_distance error: ", err)
            config.logger.debug(traceback.print_exc())
            return None
