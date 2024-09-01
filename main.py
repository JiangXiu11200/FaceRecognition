from src import face_detection


class FaceRecognition(face_detection.FaceApp):
    def main(self):
        '''Start the facial recognition main program.'''
        self.run()

if __name__ == "__main__":
    app = FaceRecognition()
    app.main()