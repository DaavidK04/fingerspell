import sys

import cv2 as cv
from PyQt6.QtWidgets import QApplication, QFileDialog

from detection import (
    convert_all_to_pixel,
    draw_circle_around_landmarks,
    recognize_image,
)
from gestures import recognize_gesture


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


selected = select_image()

if not selected:
    print("No image selected")
    sys.exit()

img = cv.imread(selected)

if check_for_image(img):
    print("No image found. Please check if the path is correct.")
else:
    height, width = img.shape[:2]
    landmarks = recognize_image(img)
    if landmarks is None:
        print("No hand detected")
    else:
        points = convert_all_to_pixel(landmarks, width, height)
        draw_circle_around_landmarks(points, img)
        cv.namedWindow("window", cv.WINDOW_NORMAL)
        cv.imshow("window", img)
        print(recognize_gesture(points))
    cv.waitKey(0)
    cv.destroyAllWindows()
