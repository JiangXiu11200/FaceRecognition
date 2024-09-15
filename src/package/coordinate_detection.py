class CoordinateDetection:
    def __init__(
        self,
        detection_range_start_point: list,
        detection_range_end_point: list,
        minimum_face_detection_score: float,
        minimum_bounding_box_height: int,
    ):
        self.detection_range_start_point = detection_range_start_point
        self.detection_range_end_point = detection_range_end_point
        self.minimum_face_detection_score = minimum_face_detection_score
        self.minimum_bounding_box_height = minimum_bounding_box_height

    def face_box_in_roi(self, face_box_center_x, face_box_center_y, bounding_box_height, detection_score):
        if (
            self.detection_range_start_point[0] < face_box_center_x < self.detection_range_end_point[0]
            and self.detection_range_start_point[1] < face_box_center_y < self.detection_range_end_point[1]
            and bounding_box_height > self.minimum_bounding_box_height
            and detection_score > self.minimum_face_detection_score
        ):
            return True
        return False
