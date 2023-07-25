import cv2
from random import randrange

class FaceDetector:
    def __init__(self, cascade_file_path):
        # Initialize the FaceDetector with the Haar cascade classifier XML file
        self.trained_face_data = cv2.CascadeClassifier(cascade_file_path)

    def detect_faces(self, image_path):
        # Load the image and convert it to grayscale
        img = cv2.imread(image_path)
        grey_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use the Haar cascade classifier to detect faces in the grayscale image
        face_coordinates = self.trained_face_data.detectMultiScale(grey_scaled_img, scaleFactor=1.1, minNeighbors=2, minSize=(80, 80))

        return img, face_coordinates

class FaceDrawer:
    def draw_rectangles(self, img, face_coordinates):
        # Draw rectangles around the detected faces on the image
        for x, y, w, h in face_coordinates:
            cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(255), randrange(255), randrange(255)), 1)

        # Return the image with rectangles drawn around the faces
        return img

class LiveFaceRecognizer:
    def __init__(self, cascade_file_path):
        # Initialize the LiveFaceRecognizer with the FaceDetector and FaceDrawer
        self.face_detector = FaceDetector(cascade_file_path)
        self.face_drawer = FaceDrawer()

        # Create a named window for the video stream with a specific size (e.g., 800x600)
        cv2.namedWindow("Live Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Face Recognition", 800, 600)

    def recognize_faces(self):
        # Open a video stream from the default camera (usually the webcam)
        video_stream = cv2.VideoCapture(0)
        while True:
            # Read a frame from the video stream
            success_frame_read, frame = video_stream.read()
            if not success_frame_read:
                break

            # Convert the frame to grayscale and detect faces using FaceDetector
            greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coordinates = self.face_detector.detect_faces(greyscale_img)

            # Draw rectangles around the detected faces using FaceDrawer
            frame_with_rectangles = self.face_drawer.draw_rectangles(frame, face_coordinates)

            # Display the frame with the rectangles around the detected faces
            cv2.imshow("Live Face Recognition", frame_with_rectangles)

            # Check if the space key is pressed to stop the video stream
            pressed_key = cv2.waitKey(1)
            if pressed_key == 32:
                break

        # Release the video stream and close the window
        video_stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cascade_file_path = 'haarcascade_frontalface_default.xml'
    image_path = "Family-Picnic-88-1024x683.jpg"

    # Detect faces and draw rectangles on the image using FaceDetector and FaceDrawer
    face_detector = FaceDetector(cascade_file_path)
    img, face_coordinates = face_detector.detect_faces(image_path)

    face_drawer = FaceDrawer()
    img_with_rectangles = face_drawer.draw_rectangles(img, face_coordinates)

    # Display the image with rectangles around the detected faces
    cv2.imshow("Detected Faces", img_with_rectangles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Perform live face recognition using LiveFaceRecognizer
    live_face_recognizer = LiveFaceRecognizer(cascade_file_path)
    live_face_recognizer.recognize_faces()
