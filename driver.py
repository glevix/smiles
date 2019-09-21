import numpy as np
import cv2


def process(frame):
    
    return frame


cap = cv2.VideoCapture(0)  # 0 for camera, filename for video

while True:
    ret, im = cap.read()  # get current frame
    if not ret:
        break
    processed = process(im)
    cv2.imshow('frame', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
