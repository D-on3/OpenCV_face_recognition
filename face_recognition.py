import cv2
from random import randrange


# Abstract class for Face Detector
class FaceDetectorBase:
    def detect_faces(self, image):
        raise NotImplementedError(
            "detect_faces method must be implemented in derived classes.")


# Concrete class for Haar Cascade Face Detector
class HaarCascadeFaceDetector(FaceDetectorBase):
    def __init__(self, cascade_file_path):
        # Initialize the Haar Cascade classifier
        self._face_cascade = cv2.CascadeClassifier(cascade_file_path)

    def detect_faces(self, image):
        # Convert the image to grayscale for face detection
        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces using the cascade classifier
        return self._face_cascade.detectMultiScale(grey_img, scaleFactor=1.1,
                                                   minNeighbors=2,
                                                   minSize=(80, 80))

    # Getter for face_cascade attribute
    def get_face_cascade(self):
        return self._face_cascade


# Class for drawing rectangles around detected faces
class FaceDrawer:
    @staticmethod
    def draw_rectangles(img, face_coordinates):
        # Draw rectangles around the detected faces
        for x, y, w, h in face_coordinates:
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          (randrange(255), randrange(255), randrange(255)), 2)
        return img


# Abstract class for Face Detector
class FaceDetectorBase:
    def detect_faces(self, image):
        raise NotImplementedError(
            "detect_faces method must be implemented in derived classes.")


# Class for live face recognition
class LiveFaceRecognizer(FaceDetectorBase):
    def __init__(self, cascade_file_path):
        # Initialize the Haar Cascade Face Detector and Face Drawer
        self._face_detector = HaarCascadeFaceDetector(cascade_file_path)
        self._face_drawer = FaceDrawer()

    def recognize_faces(self):
        # Open the webcam stream
        video_stream = cv2.VideoCapture(0)
        # Set the video stream resolution to 640x480
        video_stream.set(3, 640)
        video_stream.set(4, 480)
        # Resize the window to 640x480
        cv2.namedWindow("Detected Faces", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detected Faces", 640, 480)

        while True:
            # Read a frame from the webcam
            success, frame = video_stream.read()
            if not success:
                break

            # Detect faces in the current frame
            face_coordinates = self._face_detector.detect_faces(frame)

            # Draw rectangles around the detected faces
            frame_with_rectangles = self._face_drawer.draw_rectangles(frame,
                                                                      face_coordinates)

            # Display the result
            cv2.imshow("Detected Faces", frame_with_rectangles)

            # Stop the webcam stream if 'q' or space key is pressed
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord(' ')):
                break

        # Release the webcam and close all OpenCV windows
        video_stream.release()
        cv2.destroyAllWindows()

    # Getter for face_detector attribute
    def get_face_detector(self):
        return self._face_detector

    # Getter for face_drawer attribute
    def get_face_drawer(self):
        return self._face_drawer


if __name__ == "__main__":
    cascade_file_path = 'haarcascade_frontalface_default.xml'

    image_path = "Family-Picnic-88-1024x683.jpg"

    # Detect faces and draw rectangles on the image using HaarCascadeFaceDetector and FaceDrawer
    face_detector = HaarCascadeFaceDetector(cascade_file_path)
    img = cv2.imread(image_path)
    face_coordinates = face_detector.detect_faces(img)

    face_drawer = FaceDrawer()
    img_with_rectangles = face_drawer.draw_rectangles(img, face_coordinates)

    # Resize the window to 640x480
    cv2.namedWindow("Detected Faces", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Faces", 640, 480)
    # Display the image with rectangles around the detected faces
    cv2.imshow("Detected Faces", img_with_rectangles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create the LiveFaceRecognizer instance and start the face recognition
    live_face_recognizer = LiveFaceRecognizer(cascade_file_path)
    live_face_recognizer.recognize_faces()

