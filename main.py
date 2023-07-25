import cv2
from random import randrange

def limit_false_positives(image_path):
    '''
    Function to perform facial recognition with OpenCV and limit false positives.

    Parameters:
        image_path (str): Path to the input image for facial recognition.

    Returns:
        None (displays the image with rectangles around detected faces).
    '''

    ''' Load some pre-trained data on face frontals from OpenCV 
    (Haar cascade algorithm)'''
    trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    ''' choose an image to detect face in '''
    img = cv2.imread(image_path)

    ''' Convert to grayscale for better face detection '''
    grey_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ''' Feed the data to the algorithm = detect faces
    detectMultiScale = no matter the scale it looks for similar shape faces '''
    # Detect faces in the grayscale image using Haar cascades.
    # Tuning scaleFactor, minNeighbors, and minSize to reduce false positives.
    face_coordinates = trained_face_data.detectMultiScale(grey_scaled_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    ''' Print coordinates of the detected faces '''
    # Print the detected face coordinates (x, y, w, h).
    print(face_coordinates)

    ''' Draw rectangles around the faces
    The indexes of the numpy array correspond to the faces that are detected '''
    # Loop through the detected faces and draw rectangles around them.
    # Randomly colored rectangles are drawn to distinguish multiple faces.
    for x, y, w, h in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(255), randrange(255), randrange(255)), 1)

    # Display the result
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "Family-Picnic-88-1024x683.jpg"
    limit_false_positives(image_path)
