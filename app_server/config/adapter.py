from ..database import SessionLocal
from ..models import FaceRecognitionConfig, SystemConfig, VideoConfig
from .schema import RecoConfigData, SystemConfigData, VideoConfigData


class ConfigAdapter:
    """Adapter to convert database config to FaceApp config format"""

    def __init__(self):
        self.load_from_db()

    def load_from_db(self):
        """Load configuration from database"""
        db = SessionLocal()
        try:
            sys_config = db.query(SystemConfig).first()
            if sys_config:
                self.system_config = SystemConfigData(debug=sys_config.debug)
            else:
                raise ValueError("Load system config failed, please check database.")

            video_config = db.query(VideoConfig).first()
            if video_config:
                self.video_config = VideoConfigData(
                    rtsp=video_config.rtsp,
                    web_camera=video_config.web_camera,
                    image_height=video_config.image_height,
                    image_width=video_config.image_width,
                    detection_range_start_point=[
                        video_config.detection_range_start_point_x,
                        video_config.detection_range_start_point_y,
                    ],
                    detection_range_end_point=[
                        video_config.detection_range_end_point_x,
                        video_config.detection_range_end_point_y,
                    ],
                )
            else:
                raise ValueError("Load video config failed, please check database.")

            reco_config = db.query(FaceRecognitionConfig).first()
            if reco_config:
                self.reco_config = RecoConfigData(
                    enable=reco_config.enable,
                    set_mode=reco_config.set_mode,  # FIXME: FastAPI 會報錯
                    enable_blink_detection=reco_config.enable_blink_detection,
                    dlib_predictor=reco_config.dlib_predictor_path,
                    dlib_recognition_model=reco_config.dlib_recognition_model_path,
                    face_model=reco_config.face_model,
                    minimum_bounding_box_height=reco_config.minimum_bounding_box_height,
                    minimum_face_detection_score=reco_config.minimum_face_detection_score,
                    eyes_detection_brightness_threshold=reco_config.eyes_detection_brightness_threshold,
                    eyes_detection_brightness_value=[
                        reco_config.eyes_detection_brightness_value_min,
                        reco_config.eyes_detection_brightness_value_max,
                    ],
                    sensitivity=reco_config.sensitivity,
                    consecutive_prediction_intervals=reco_config.consecutive_prediction_intervals_frame,
                )
            else:
                raise ValueError("Load face recognition config failed, please check database.")

        except Exception as e:
            print(f"Error loading configuration from database: {e}")
            raise ValueError("Failed to load configuration from database.")

        finally:
            db.close()
