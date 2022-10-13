import sys
import cv2
import numpy as np
from generate_labels import image_to_vp

if len(sys.argv) > 1:
  VIDEO = sys.argv[1]
else:
  raise RuntimeError('No video provided')

left_prev_lanes = []
right_prev_lanes = []

cap = cv2.VideoCapture(VIDEO)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_copy = np.copy(frame)
        vp, img = image_to_vp(frame_copy, left_prev_lanes, right_prev_lanes, debug=True)
        cv2.imshow('lined', img)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
    