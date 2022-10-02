import numpy as np
import cv2


def get_ground_truth_labels(i):
    return np.loadtxt(f'labeled/{i}.txt')

def get_video_frames(i):    
    frames = []
    path = f'labeled/{i}.hevc'
    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
    return np.stack(frames, axis=0) # dimensions (T, H, W, C)

def get_training_data():
    X = []
    y = []
    for i in range(5):
        X.extend(get_video_frames(i))
        y.extend(get_ground_truth_labels(i))
    return np.array(list(zip(X, y)), dtype=object)

training_data = get_training_data()
