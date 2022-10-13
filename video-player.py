import cv2
import numpy as np
from process import process_image

# This is the dumb way to do it:
# training_data = np.load('data/training.npy', allow_pickle=True) # shape: (frame, calibration)

def run_video_player():
    left_prev_lanes = []
    right_prev_lanes = []
    
    cap = cv2.VideoCapture('unlabeled/9.hevc')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_copy = np.copy(frame)
            vp, img = process_image(frame_copy, left_prev_lanes, right_prev_lanes)
            cv2.imshow('lined', img)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    
run_video_player()