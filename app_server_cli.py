import typer

from app_server import models
from app_server.database import SessionLocal

app = typer.Typer()


@app.command()
def init_data():
    db = SessionLocal()
    system_config = models.SystemConfig(debug=True)
    db.add(system_config)
    db.commit()
    typer.echo("- System configuration initialized!")

    video_config = models.VideoConfig(
        rtsp=None,
        web_camera=0,
        image_height=720,
        image_width=1280,
        detection_range_start_point_x=420,
        detection_range_start_point_y=160,
        detection_range_end_point_x=820,
        detection_range_end_point_y=560,
    )
    db.add(video_config)
    db.commit()
    typer.echo("- Video configuration initialized!")

    face_recognition_config = models.FaceRecognitionConfig(
        enable=True,
        set_mode=False,  # FIXME: FastAPI 會報錯
        dlib_predictor_path="models/dlib/shape_predictor_68_face_landmarks.dat",
        dlib_recognition_model_path="models/dlib/dlib_face_recognition_resnet_model_v1.dat",
        face_model="models/face_recognition/model.csv",
        minimum_bounding_box_height=0.4,
        minimum_face_detection_score=0.6,
        eyes_detection_brightness_threshold=120,
        eyes_detection_brightness_value_min=50,
        eyes_detection_brightness_value_max=20,
        sensitivity=0.4,
        consecutive_prediction_intervals_frame=90,
    )
    db.add(face_recognition_config)
    db.commit()
    typer.echo("- Face recognition configuration initialized!")


if __name__ == "__main__":
    app()
