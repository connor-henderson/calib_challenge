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

def get_and_save_training_data():
    X = []
    y = []
    for i in range(5):
        X.extend(get_video_frames(i))
        y.extend(get_ground_truth_labels(i))
    training_data = np.array(list(zip(X, y)), dtype=object)
    np.save('data/training.npy', training_data)

def get_and_save_test_data():
    X = []
    for i in range(5, 10):
        X.extend(get_video_frames(i))
    test_data = np.array(X)
    np.save('data/test.npy', test_data)
    

get_and_save_training_data()
get_and_save_test_data()




# "To rotate from one frame into another with euler angles the convention is to 
# rotate around roll, then pitch and then yaw, while rotating around the rotated 
# axes, not the original axes."

# ecef = earth centered earth fixed

# Test to add for going from vp -> _roll, pitch, yaw: make the vp the center of the camera and see if the MSE is 100%
# (because 0 roll, pitch, and yaw) would be the same as the camera being mounted perfectly


# ecef_from_local = rot_from_quat(quats_ecef[0])
# local_from_ecef = ecef_from_local.T
# positions_local = np.einsum('ij,kj->ki', local_from_ecef, postions_ecef - positions_ecef[0])
# rotations_global = rot_from_quat(quats_ecef)
# rotations_local = np.einsum('ij,kjl->kil', local_from_ecef, rotations_global)
# eulers_local = euler_from_rot(rotations_local)