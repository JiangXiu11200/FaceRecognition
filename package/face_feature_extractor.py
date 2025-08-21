"""
Provide FastAPI registered user face feature functionality.
"""

import csv
import hashlib
import time
import traceback
from typing import Optional

import cv2
import dlib
import numpy as np

# TODO: 1. 未來將 face features 改成 key-value db
# TODO: 2. 修正每次註冊都要重新載入 dlib 模型的問題


class FaceFeatureExtractor:
    """
    A class to extract face features using dlib. \n
    It can save the extracted features to a CSV file. \n
    CSV format: name,f1,f2,...,f128
    """

    def __init__(
        self,
        feature_csv_path: str,
        dlib_predictor_path: str,
        dlib_recognition_model_path: str,
        user_name: str = None,
    ):
        self.feature_csv_path = feature_csv_path
        self.user_name = (
            user_name if user_name is not None else "User_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        )
        self.dlib_predictor = dlib.shape_predictor(dlib_predictor_path)
        self.dlib_recognition_model = dlib.face_recognition_model_v1(dlib_recognition_model_path)

    def get_face_roi(self, image: np.ndarray) -> tuple[bool, Optional[dict]]:
        """
        Get the face region of interest (ROI) from the image.

        Parameters:
            image (np.ndarray): The input image.

        Returns:
            status (bool): True if face ROI is found, False otherwise.
            message (Optional[dict]): Contains the face ROI if found, otherwise an error message.
        """
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detector = dlib.get_frontal_face_detector()
            faces = detector(gray_image, 1)

            if len(faces) > 0:
                return True, {"face_roi": image[faces[0].top() : faces[0].bottom(), faces[0].left() : faces[0].right()]}
        except Exception as e:
            return False, {"error": f"Unable to get face ROI. Error: {e}"}

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
            if face_roi is None or face_roi.size == 0:
                return False, {"error": "No face ROI provided for feature extraction."}
            landmarks_frame = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            dlib_coordinate = dlib.rectangle(0, 0, face_roi.shape[0], face_roi.shape[1])
            feature_coordinates = self.dlib_predictor(landmarks_frame, dlib_coordinate)
            current_face_descriptor = np.array(
                self.dlib_recognition_model.compute_face_descriptor(face_roi, feature_coordinates)
            )
            save_status, message = self._save_feature(current_face_descriptor)

            return save_status, message
        except Exception as err:
            print(traceback.format_exc())
            return False, {"error": f"Unable to extract features. Error: {err}"}

    def _save_feature(self, face_descriptor: np.ndarray) -> bool:
        """
        Save face descriptor to CSV file. \n
        CSV format: name,f1,f2,...,f128

        Parameters:
            face_descriptor (np.ndarray): The face descriptor.

        Returns:
            result (bool): Save status.
        """
        try:
            if isinstance(face_descriptor, np.ndarray):
                face_descriptor = face_descriptor.tolist()
            with open(self.feature_csv_path, mode="a+", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self.user_name] + face_descriptor)
            return True, {"message": f"Feature saved successfully for {self.user_name}."}
        except Exception as err:
            return False, {"error": f"Unable to save feature for {self.user_name}. Error: {err}"}

    @staticmethod
    def delete_feature(feature_csv_path: str, user_name: str) -> tuple[bool, Optional[dict]]:
        """
        Delete face feature from CSV file.

        Parameters:
            user_name (str): The name of the user whose feature is to be deleted.
            feature_csv_path (str): The path to the CSV file containing face features.

        Returns:
            result (bool): True if deletion was successful, False otherwise.
            message (Optional[dict]): Contains a success message or an error message.
        """
        deleted = False
        try:
            with open(feature_csv_path) as file:
                rows = list(csv.reader(file))
            with open(feature_csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                for row in rows:
                    if row and row[0] == user_name:
                        deleted = True
                    if row and row[0] != user_name:
                        writer.writerow(row)
            if deleted:
                return True, {"message": f"User: {user_name} feature deleted successfully."}
            else:
                return False, {"error": f"No feature found for User: {user_name}."}
        except Exception as err:
            return False, {"error": f"Unable to delete feature for {user_name}. Error: {err}"}
