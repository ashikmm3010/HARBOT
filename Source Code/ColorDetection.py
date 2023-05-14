#https://pysource.com/2021/10/19/simple-color-recognition-with-opencv-and-python/
import cv2
from test1 import *
from Stepper import *




def color():
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    x=0
    red=0
    while x<75:
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = frame.shape

        cx = int(width / 2)
        cy = int(height / 2)

        # Pick pixel value
        pixel_center = hsv_frame[cy, cx]
        hue_value = pixel_center[0]

        color = "Undefined"
        if hue_value < 5:
            color = "RED"
            print(x)
            red=1
        elif hue_value < 22:
            color = "ORANGE"
            red=1
        elif hue_value < 33:
            color = "YELLOW"
        elif hue_value < 78:
            color = "GREEN"
        elif hue_value < 131:
            color = "BLUE"
        elif hue_value < 170:
            color = "VIOLET"
        else:
            color = "RED"
            print(x)
            red=1

        pixel_center_bgr = frame[cy, cx]
        b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])

        cv2.rectangle(frame, (cx - 220, 10), (cx + 200, 120), (255, 255, 255), -1)
        cv2.putText(frame, color, (cx - 200, 100), 0, 3, (b, g, r), 5)
        cv2.circle(frame, (cx, cy), 5, (25, 25, 25), 3)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        x=x+1
    cap.release()
    return red

color()

cv2.destroyAllWindows()