import sys

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PyQt6.QtWidgets import QApplication, QFileDialog

# create HandLandmarker object
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Landmarker variables
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def select_image():
    _ = QApplication.instance() or QApplication(sys.argv)

    image_filter = (
        "JPEG (*.jpg, *.jpeg);;"
        "Bilder (*.png, *.jpg, *.jpeg, *.gif, *.webp, *.bmp);;"
        "PNG (*.png);;"
        "Alle Dateien (*.*)"
    )

    file_path, _ = QFileDialog.getOpenFileName(
        parent=None, caption="Bild auswählen", directory="", filter=image_filter
    )

    return file_path


def check_for_image(img):
    return img is None


def recognize_image(selected):
    image = mp.Image.create_from_file(selected)
    detection_result = detector.detect(image)
    if not detection_result.hand_landmarks:
        print("no landmarks detected")
        return None

    lmarks = detection_result.hand_landmarks[0]
    return lmarks


def convert_to_pixel(lmark, width, height):
    x_pixel = int(lmark.x * width)
    y_pixel = int(lmark.y * height)
    return (x_pixel, y_pixel)


selected = select_image()

if not selected:
    print("No image selected")
    sys.exit()

img = cv.imread(selected)

if check_for_image(img):
    print("No image found. Please check if the path is correct.")
else:
    height, width = img.shape[:2]
    landmarks = recognize_image(selected)
    if landmarks is None:
        print("No hand detected")
    else:
        cv.circle(
            img,
            (convert_to_pixel(landmarks[INDEX_FINGER_TIP], width, height)),
            50,
            (0, 0, 255),
            10,
        )
        cv.circle(
            img,
            (convert_to_pixel(landmarks[MIDDLE_FINGER_TIP], width, height)),
            50,
            (0, 0, 255),
            10,
        )
        annotated_image = cv.circle(
            img,
            (convert_to_pixel(landmarks[WRIST], width, height)),
            50,
            (0, 0, 255),
            10,
        )

        cv.namedWindow("window", cv.WINDOW_NORMAL)
        cv.imshow("window", annotated_image)
        print(landmarks)

    cv.waitKey(0)
    cv.destroyAllWindows()
