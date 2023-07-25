<!DOCTYPE html>
<html>
<head>
  <title>Face Detection and Recognition with OpenCV</title>
</head>
<body>
  <h1>Face Detection and Recognition with OpenCV</h1>
  
  <h2>Requirements</h2>
  <ul>
    <li>Python 3.x</li>
    <li>OpenCV (cv2) library (Install with <code>pip install opencv-python</code>)</li>
  </ul>
  
  <h2>Usage</h2>
  <ol>
    <li>Make sure you have Python and the required libraries installed.</li>
    <li>Download or clone the repository.</li>
    <li>Ensure you have the <code>haarcascade_frontalface_default.xml</code> file in the same directory as the script. This file contains the pre-trained Haar cascade classifier for face detection.</li>
    <li>Place the image you want to process in the same directory as the script, or prepare a webcam for live face recognition.</li>
    <li>Run the script:<br><code>python face_detection_recognition.py</code></li>
    <li>Depending on the chosen mode (image or live video), the script will either display the image with rectangles around the detected faces or perform real-time face recognition from the webcam stream.</li>
  </ol>

  <h2>Classes</h2>
  <p>The code is organized into three classes:</p>
  
  <ol>
    <li><strong>FaceDetector</strong> (Class responsible for detecting faces)</li>
    <p>This class encapsulates the functionality for detecting faces in an image using the Haar cascade classifier. It loads the pre-trained classifier from the <code>haarcascade_frontalface_default.xml</code> file and provides the <code>detect_faces</code> method to detect faces in an input image.</p>
  
    <li><strong>FaceDrawer</strong> (Class responsible for drawing rectangles around detected faces)</li>
    <p>This class handles drawing rectangles around the detected faces on the image.</p>
  
    <li><strong>LiveFaceRecognizer</strong> (Class for live face recognition)</li>
    <p>This class utilizes the <code>FaceDetector</code> and <code>FaceDrawer</code> classes to perform live face recognition from a webcam stream.</p>
  </ol>
  
  <h2>Face Recognition Parameters</h2>
  <p>You can fine-tune the face detection parameters, such as <code>scaleFactor</code>, <code>minNeighbors</code>, and <code>minSize</code>, within the <code>FaceDetector</code> class to adjust the sensitivity and accuracy of face detection. These parameters are found in the <code>detectMultiScale</code> method.</p>
  
  <h2>Example</h2>
  <p>The script can be executed with the provided example image <code>Family-Picnic-88-1024x683.jpg</code> for image face detection and recognition. It will display the image with rectangles around the detected faces. Additionally, the script can perform live face recognition using the webcam.</p>
  <p>Feel free to customize the image or use the live face recognition mode to detect and recognize faces in real-time.</p>
  
  <p><strong>Note:</strong> Ensure that you have access to a webcam for live face recognition mode.</p>
</body>
</html>
