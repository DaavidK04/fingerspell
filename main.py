import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
from PyQt6.QtWidgets import QApplication, QFileDialog


# create HandLandmarker object
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)


def select_image():
    app = QApplication.instance() or QApplication(sys.argv) 
    
    image_filter = (
        "JPEG (*.jpg, *.jpeg);;"
        "Bilder (*.png, *.jpg, *.jpeg, *.gif, *.webp, *.bmp);;"
        "PNG (*.png);;"
        "Alle Dateien (*.*)"
    )
    
    file_path, _ = QFileDialog.getOpenFileName(
        parent=None,
        caption="Bild auswählen",
        directory="",
        filter= image_filter
    )
    
    return file_path


def check_for_image(img):
    return img is None
    

def recognize_image(selected, width, height):
    image = mp.Image.create_from_file(selected)
    detection_result = detector.detect(image)
    if not detection_result.hand_landmarks:
        print("no landmarks detected")
        return None, None, None, None, None, None, None
    
    lmarks = detection_result.hand_landmarks[0]
    index_x = int(lmarks[8].x * width)
    index_y = int(lmarks[8].y * height)
    wrist_x = int(lmarks[0].x * width)
    wrist_y = int(lmarks[0].y * height)
    middle_x = int(lmarks[12].x * width)
    middle_y = int(lmarks[12].y * height)
                
    return index_x, index_y, wrist_x, wrist_y, middle_x, middle_y, detection_result

selected = select_image()

if not selected:
    print("No image selected")
    sys.exit()

img = cv.imread(selected)

if check_for_image(img):
    print("No image found. Please check if the path is correct.")
else:
    height, width = img.shape[:2]
    index_x, index_y, wrist_x, wrist_y, middle_x, middle_y, detection_result = recognize_image(selected, width, height)
    if index_x is None:
        print("No hand detected")
    else:
        cv.circle(img, (wrist_x,wrist_y), 50, (0, 0, 255), 10)
        cv.circle(img, (middle_x,middle_y), 50, (0, 0, 255), 10)
        annotated_image = cv.circle(img, (index_x,index_y), 50, (0, 0, 255), 10)
        
        cv.namedWindow('window', cv.WINDOW_NORMAL)
        cv.imshow('window', annotated_image)
        print(detection_result)
        
    if index_y < wrist_y:
        print("index up")
    else:
        print("index down")
    cv.waitKey(0)
    cv.destroyAllWindows()
    