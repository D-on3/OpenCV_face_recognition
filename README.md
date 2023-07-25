# Facial Recognition with OpenCV

This Python script performs facial recognition using OpenCV's Haar cascade classifier and limits false positives to improve accuracy. The code detects faces in an input image and draws rectangles around them. Randomly colored rectangles are used to distinguish multiple detected faces.

## Requirements

- Python 3.10
- OpenCV (cv2) library (Install with `pip install opencv-python`)

## Usage

1. Ensure you have Python and the required libraries installed.

2. Download or clone the repository.

3. Place the image you want to process in the same directory as the `main.py` script.

4. In the terminal or command prompt, navigate to the directory containing the `main.py` script.

5. Run the script and provide the image filename as an argument:


Replace `<image_filename>` with the name of your input image file.

6. The script will display the image with rectangles around the detected faces.

## Customization

You can adjust the parameters of the `detectMultiScale` function in the `limit_false_positives` function to fine-tune the sensitivity of the detector and reduce false positives:

- `scaleFactor`: Controls how much the image size is reduced at each image scale. Lower values make detection slower but more accurate.
- `minNeighbors`: Specifies the number of neighbors a region should have to be considered as a face. Higher values reduce false positives but may miss some faces.
- `minSize`: Specifies the minimum size of the detected face. Faces smaller than this size are ignored.

Experiment with different parameter values to find the best configuration for your specific images.

## Example

The example usage of the `limit_false_positives` function is shown in the `__main__` block of the `facial_recognition.py` script. You can run the example by executing the script without any arguments.

