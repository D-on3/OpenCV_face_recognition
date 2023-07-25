# Face Detection and Recognition with OpenCV

This Python script uses OpenCV to perform face detection and recognition using the Haar cascade classifier. It includes functionalities for detecting faces in an image, drawing rectangles around the detected faces, and performing live face recognition from a webcam stream.

## Requirements

- Python 3.10
- OpenCV (cv2) library (Install with `pip install opencv-python`)

## Usage

1. Make sure you have Python and the required libraries installed.
2. Download or clone the repository.
3. Ensure you have the `haarcascade_frontalface_default.xml` file in the same directory as the script. This file contains the pre-trained Haar cascade classifier for face detection.
4. Place the image you want to process in the same directory as the script, or prepare a webcam for live face recognition.
5. Run the script:


```bash
   python face_detection_recognition.py
```

6. Depending on the chosen mode (image or live video), the script will either display the image with rectangles around the detected faces or perform real-time face recognition from the webcam stream.

## Classes

The code is organized into three classes:

1. **`FaceDetector`** (Class responsible for detecting faces)

This class encapsulates the functionality for detecting faces in an image using the Haar cascade classifier. It loads the pre-trained classifier from the `haarcascade_frontalface_default.xml` file and provides the `detect_faces` method to detect faces in an input image.

**Methods**:
- `__init__(self, cascade_file_path)`: Constructor that initializes the `FaceDetector` object with the path to the Haar cascade classifier file.
- `detect_faces(self, image_path)`: Detects faces in an input image and returns the image with face rectangles and the coordinates of the detected faces.

2. **`FaceDrawer`** (Class responsible for drawing rectangles around detected faces)

This class handles drawing rectangles around the detected faces on the image.

**Methods**:
- `draw_rectangles(self, img, face_coordinates)`: Draws rectangles around the detected faces on the input image and returns the modified image.

3. **`LiveFaceRecognizer`** (Class for live face recognition)

This class utilizes the `FaceDetector` and `FaceDrawer` classes to perform live face recognition from a webcam stream.

**Methods**:
- `__init__(self, cascade_file_path)`: Constructor that initializes the `LiveFaceRecognizer` object with the path to the Haar cascade classifier file.
- `recognize_faces(self)`: Performs live face recognition from the webcam stream, displaying rectangles around the detected faces in real-time. The recognition stops when the space key is pressed.

## Face Recognition Parameters

You can fine-tune the face detection parameters, such as `scaleFactor`, `minNeighbors`, and `minSize`, within the `FaceDetector` class to adjust the sensitivity and accuracy of face detection. These parameters are found in the `detectMultiScale` method.

## Example

The script can be executed with the provided example image `Family-Picnic-88-1024x683.jpg` for image face detection and recognition. It will display the image with rectangles around the detected faces. Additionally, the script can perform live face recognition using the webcam.

Feel free to customize the image or use the live face recognition mode to detect and recognize faces in real-time.

**Note**: Ensure that you have access to a webcam for live face recognition mode.
