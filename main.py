import cv2

''' Load some pre-trained data on face frontals from opencv 
(har cascade algorithm)'''
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
''' choose an image to detect face in '''
img = cv2.imread("Face-Recognition.jpg")

''' Convert to greyscale check the different filters  on
https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/
https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html '''
grey_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

''' feed the data to the algorithm = detect faces
detectMultiScale = no matter the scale it look to face similar shape
'''
face_coordinates = trained_face_data.detectMultiScale(grey_scaled_img)
'''print coordinates of the face'''
print(face_coordinates)

''' Draw rectangles arround the faces'''

for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+h , y +h),(0,255,0),1)

cv2.imshow("My face detector", img)
cv2.waitKey()

print("d-On3")
