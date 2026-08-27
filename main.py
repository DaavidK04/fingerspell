import math
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
THUMB = (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP)
INDEX_FINGER = (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP)
MIDDLE_FINGER = (
    MIDDLE_FINGER_MCP,
    MIDDLE_FINGER_PIP,
    MIDDLE_FINGER_DIP,
    MIDDLE_FINGER_TIP,
)
RING_FINGER = (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP)
PINKY_FINGER = (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP)
LONG_FINGERS = (INDEX_FINGER, MIDDLE_FINGER, RING_FINGER, PINKY_FINGER)

THUMB_SPREAD_THRESHOLD = 0.73


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


def is_stretched(points, finger):
    _mcp, pip, _dip, tip = finger
    return math.dist(points[tip], points[WRIST]) > math.dist(points[pip], points[WRIST])


def thumb_ratio(points):
    wrist_to_pinkymcp_scale = math.dist(points[WRIST], points[PINKY_MCP])
    thumb_measure = math.dist(points[THUMB_TIP], points[INDEX_FINGER_MCP])
    ratio = thumb_measure / wrist_to_pinkymcp_scale

    return ratio


def is_thumb_spread(points):
    return thumb_ratio(points) > THUMB_SPREAD_THRESHOLD


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
        # print(is_stretched(landmarks[INDEX_FINGER_TIP], landmarks[INDEX_FINGER_PIP], landmarks[WRIST]))
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
        # print(is_stretched(landmarks, MIDDLE_FINGER, width, height))
        for finger in LONG_FINGERS:
            print("stretched", is_stretched(points, finger))
        # print("Distance: %s", math.dist(points[WRIST], points[PINKY_MCP]))
        _, scale, measure = thumb_ratio(points)
        print(f"Scale: {scale}")
        print(f"Measure: {measure}")
    cv.waitKey(0)
    cv.destroyAllWindows()
