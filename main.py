import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# create HandLandmarker object
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

img = cv.imread('test-pictures/peace.jpg')
if img is None:
    print("No picture found. Please check if the path is correct.")
else:
    image = mp.Image.create_from_file("test-pictures/peace.jpg")
    detection_result = detector.detect(image)
    conv_image = image.numpy_view()
    cv.imshow('window', cv.cvtColor(conv_image, cv.COLOR_RGB2BGR))
    print(detection_result)
    cv.waitKey(0)
    cv.destroyAllWindows()
