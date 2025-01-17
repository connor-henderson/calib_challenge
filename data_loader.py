import numpy as np
import cv2

def get_ground_truth_labels(i):
    return np.loadtxt(f'labeled/{i}.txt', ndmin=2)

def get_video_frames(i):
    frames = []
    folder = 'labeled' if i < 5 else 'unlabeled'
    path = f'{folder}/{i}.hevc'
    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
    return np.stack(frames, axis=0) # dimensions (T, H, W, C)

def get_and_save_data(save_filename, video_indices):
    X = []
    for i in video_indices:
        X.extend(get_video_frames(i))
    test_data = np.array(X)
    np.save(save_filename, test_data)
    
get_and_save_data('data/training.npy', range(5))
get_and_save_data('data/test.npy', range(5, 10))