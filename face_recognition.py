from src.face_detection import FaceApp


class FaceRecognition(FaceApp):
    def main(self):
        self.run()

if __name__ == "__main__":
    app = FaceRecognition()
    app.main()