import cv2
import numpy as np

training_data = np.load('data/training.npy', allow_pickle=True) # shape: (frame, calibration)

for i in range(100):
    vp = (0, 0) # debug_process_image(training_data[i][0])
    frame = training_data[i][0]
    cv2.imshow('img', frame)
    cv2.waitKey(50)