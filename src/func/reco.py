import cv2
import csv
import dlib
import numpy as np
import time
import config

def easy_blink_detect(blink_list):
    state = False
    if 0 in blink_list and np.count_nonzero(blink_list == 0) >= 3 and np.count_nonzero(blink_list == 1) >= 3 and blink_list[0] != 0 and blink_list[-1] != 0:
        first_zero_index = np.argmax(blink_list == 0)
        last_zero_index = len(blink_list) - 1 - np.argmax(np.flip(blink_list) == 0)
        if (last_zero_index - first_zero_index + 1) == np.count_nonzero(blink_list == 0):
            state = True
    else:
        state = False
    return state

def save_feature(face_descriptor):
    try:
        with open("./model.csv", mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(face_descriptor)
        return True
    except:
        return False

def feature_extraction(face_roi, predictor, face_reco):
    start_time = time.time()
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
    if config.mode == 1:
        distance = euclidean_distance(face_descriptor)
        if distance <= 0.4:
            config.logger.info("pass.")
        else:
            config.logger.info("fail.")
        end_time = time.time()
        execution_time = round(end_time - start_time, 3)
        config.logger.info(f"Face Recognition Time: {execution_time} sec")
        return face_descriptor
    elif config.mode == 2:
        save_feature(face_descriptor)
        return True

def euclidean_distance(face_descriptor):
    dist_list = []
    for original_features in config.face_model:
        dist = np.sqrt(np.sum(np.square(face_descriptor - original_features)))
        dist_list.append(dist)
    result = min(dist_list)
    config.logger.debug(f"Minimum euclidean distance: {result}")
    return result