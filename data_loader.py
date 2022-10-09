import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


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
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred_gray_img = cv2.GaussianBlur(gray_img,(5,5),0) # additional image smoothing
            
            # Masking with a 4-sided polygon
            canny = cv2.Canny(blurred_gray_img,25,75) #minVal, maxVal chosen from trial and error + here: https://s3.us-east-2.amazonaws.com/elasticbeanstalk-us-east-2-856193192518/files/ece493/Assignment1-Lane+Detection.pdf
            
            
            mask = np.zeros_like(canny)
            ignore_mask_color = 255
            
            imshape = canny.shape
            y = imshape[0]
            x = imshape[1]
            y_offset = -10
            x_offset = 100
            
            roi_vertices = np.array([[(200, y - 200),
                                      (x / 2 - x_offset, y / 2 + y_offset),
                                      (x / 2 + x_offset, y / 2 + y_offset),
                                      (x - 200, y - 200)]],
                                      dtype=np.int32)
            
            # cv2.fillPoly(img, pts=[roi_vertices], color=(0, 255, 0))
            cv2.fillPoly(mask, roi_vertices, ignore_mask_color)
            
            
            masked_edges = cv2.bitwise_and(canny, mask)
            # masked_edges_img = np.dstack((masked_edges, masked_edges, masked_edges))
            # plt.imshow(masked_edges_img)
            
            
            # plt.subplot(121),plt.imshow(blurred_gray_img,cmap = 'gray')
            # plt.title('blurred gray Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(masked_edges,cmap = 'gray')
            # plt.title('masked Image'), plt.xticks([]), plt.yticks([])
            # plt.show()
            # break
            
            rho = 1
            theta = 31 * np.pi/180
            threshold = 10
            min_line_length = 5
            max_line_gap = 20
            lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
            
            line_image = np.copy(img) * 0
            for line in lines:
                for x0, y0, x1, y1 in line:
                    cv2.line(line_image, (x0, y0), (x1, y1), (0, 0, 255), 4)
            
            plt.subplot(121),plt.imshow(line_image,cmap = 'gray')
            plt.title('Line Image'), plt.xticks([]), plt.yticks([])
            plt.show()
            
            frames.append(masked_edges)
            break
    return
    return np.stack(frames, axis=0) # dimensions (T, H, W)

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
    

# get_and_save_training_data()
# get_and_save_test_data()

for i in range(0, 3):
    get_video_frames(i)





# "To rotate from one frame into another with euler angles the convention is to 
# rotate around roll, then pitch and then yaw, while rotating around the rotated 
# axes, not the original axes."

# ecef = earth centered earth fixed



# ecef_from_local = rot_from_quat(quats_ecef[0])
# local_from_ecef = ecef_from_local.T
# positions_local = np.einsum('ij,kj->ki', local_from_ecef, postions_ecef - positions_ecef[0])
# rotations_global = rot_from_quat(quats_ecef)
# rotations_local = np.einsum('ij,kjl->kil', local_from_ecef, rotations_global)
# eulers_local = euler_from_rot(rotations_local)