from typing import NamedTuple

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# create HandLandmarker object
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)


class Point(NamedTuple):
    x: int
    y: int


def recognize_image(image):
    rgb_frame = convert_bgr_to_rgb(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
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


def convert_bgr_to_rgb(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def draw_circle_around_landmarks(points, img):
    for point in points:
        cv.circle(
            img,
            (point),
            30,
            (0, 0, 255),
            5,
        )
