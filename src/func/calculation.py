import cv2
import config

class calculation:
    def get_face_boundingbox(frame, bounding_box, width, height):
        # face bounding box
        bounding_x1, bounding_y1 = int(bounding_box.xmin * width), int(bounding_box.ymin * height)
        bounding_x2, bounding_y2 = int(bounding_x1 + bounding_box.width * width), int(bounding_y1 + bounding_box.height * height)
        if config.debug:
            cv2.rectangle(frame, (bounding_x1, bounding_y1), (bounding_x2, bounding_y2), (0, 255, 0), 2)
        return bounding_x1, bounding_y1, bounding_x2, bounding_y2

    def get_eyes_boundingbox(frame, detection, bounding_height, width, height):
        # eyes bounding box
        eye_left = detection.location_data.relative_keypoints[0]
        eye_right = detection.location_data.relative_keypoints[1]
        eye_left_x, eye_left_y = int(eye_left.x * width), int(eye_left.y * height)
        eye_right_x, eye_right_y = int(eye_right.x * width), int(eye_right.y * height)
        eye_proportion = (bounding_height * height) * 0.1
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

    def grayscale_area(eye_left_roi, eye_right_roi):
        if eye_left_roi.size == 0:
            eye_left_roi = eye_right_roi
        elif eye_right_roi.size == 0:
            eye_right_roi = eye_left_roi
        left_eye_gary = cv2.cvtColor(eye_left_roi, cv2.COLOR_BGR2GRAY)
        right_eye_gary = cv2.cvtColor(eye_right_roi, cv2.COLOR_BGR2GRAY)
        ret, left_eye_gary = cv2.threshold(left_eye_gary, 80, 255, cv2.THRESH_BINARY)
        ret, right_eye_gary = cv2.threshold(right_eye_gary, 80, 255, cv2.THRESH_BINARY)
        if config.debug:
            cv2.imshow("eyes_left", left_eye_gary)
            cv2.imshow("eyes_right", right_eye_gary)
        return left_eye_gary, right_eye_gary