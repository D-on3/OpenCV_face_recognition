import cv2
from random import randrange

class FaceDetector:
    def __init__(self, cascade_file_path):
        # Initialize the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cascade_file_path)

    def detect_faces(self, image):
        # Convert the image to grayscale for face detection
        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces using the cascade classifier
        return self.face_cascade.detectMultiScale(grey_img, scaleFactor=1.1, minNeighbors=2, minSize=(80, 80))

class FaceDrawer:
    def draw_rectangles(self, img, face_coordinates):
        # Draw rectangles around the detected faces
        for x, y, w, h in face_coordinates:
            cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(255), randrange(255), randrange(255)), 2)
        return img

class LiveFaceRecognizer:
    def __init__(self, cascade_file_path):
        # Initialize the face detector and face drawer
        self.face_detector = FaceDetector(cascade_file_path)
        self.face_drawer = FaceDrawer()

    def recognize_faces(self):
        # Open the webcam stream
        video_stream = cv2.VideoCapture(0)

        while True:
            # Read a frame from the webcam
            success, frame = video_stream.read()
            if not success:
                break

            # Detect faces in the current frame
            face_coordinates = self.face_detector.detect_faces(frame)

            # Draw rectangles around the detected faces
            frame_with_rectangles = self.face_drawer.draw_rectangles(frame, face_coordinates)

            # Display the result
            cv2.imshow("Detected Faces", frame_with_rectangles)

            # Stop the webcam stream if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Stop the webcam stream if space key is pressed
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        # Release the webcam and close all OpenCV windows
        video_stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cascade_file_path = 'haarcascade_frontalface_default.xml'
    image_path = "Family-Picnic-88-1024x683.jpg"

    # Detect faces and draw rectangles on the image using FaceDetector and FaceDrawer
    face_detector = FaceDetector(cascade_file_path)
    img = cv2.imread(image_path)
    face_coordinates = face_detector.detect_faces(img)

    face_drawer = FaceDrawer()
    img_with_rectangles = face_drawer.draw_rectangles(img, face_coordinates)

    # Display the image with rectangles around the detected faces
    cv2.imshow("Detected Faces", img_with_rectangles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create the LiveFaceRecognizer instance and start the face recognition
    live_face_recognizer = LiveFaceRecognizer(cascade_file_path)
    live_face_recognizer.recognize_faces()
