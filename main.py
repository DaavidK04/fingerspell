import sys
from typing import NamedTuple

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PyQt6.QtWidgets import QApplication, QFileDialog

from gestures import check_thumb_direction
from landmarks import PINKY_MCP, THUMB_TIP, WRIST

# create HandLandmarker object
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)


class Point(NamedTuple):
    x: int
    y: int


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
    return Point(x_pixel, y_pixel)


def convert_all_to_pixel(landmarks, width, height):
    return [convert_to_pixel(lmark, width, height) for lmark in landmarks]


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
        points = convert_all_to_pixel(landmarks, width, height)
        print("landmark:", landmarks[WRIST])
        cv.circle(
            img,
            (points[THUMB_TIP]),
            50,
            (0, 0, 255),
            10,
        )
        cv.circle(
            img,
            (points[PINKY_MCP]),
            50,
            (0, 0, 255),
            10,
        )
        annotated_image = cv.circle(
            img,
            (points[WRIST]),
            50,
            (0, 0, 255),
            10,
        )

        cv.namedWindow("window", cv.WINDOW_NORMAL)
        cv.imshow("window", annotated_image)
        # for finger in LONG_FINGERS:
        # print("stretched", is_stretched(points, finger))
        print("Daumen: ", check_thumb_direction(points))
    cv.waitKey(0)
    cv.destroyAllWindows()
