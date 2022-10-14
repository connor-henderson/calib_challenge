import numpy as np
import cv2
import matplotlib.pyplot as plt
from types import SimpleNamespace

def get_roi_from_img(img, roi=None): 
    if roi is None:
        raise RuntimeError('No roi params provided')
    mask = np.zeros_like(img)
    ignore_mask_color = 255
    y, x = img.shape
    # roi_vertices = np.array([[(roi['x_bottom_offset'], y - roi['y_bottom_offset']),
    #                           (x / 2 - roi['x_top_offset'], y / 1.8 + roi['y_top_offset']),
    #                           (x / 2 + roi['x_top_offset'], y / 1.8 + roi['y_top_offset']),
    #                           (x - roi['x_bottom_offset'], y - roi['y_bottom_offset'])]],
    #                           dtype=np.int32)
    y_offset = -10
    x_offset = 300
    roi_vertices = np.array([[(50, y - 150),
                              (x / 2 - x_offset, y / 1.8 + y_offset),
                              (x / 2 + x_offset, y / 1.8 + y_offset),
                              (x - 50, y - 150)]],
                              dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(img, img, mask=mask)
    return masked_edges

def get_hough_lines_p(img, hough=None):
    if hough is None:
        raise RuntimeError('No hough params provided')
    h = SimpleNamespace(**hough)
    return cv2.HoughLinesP(img, h.rho, h.theta, h.threshold, np.array([]), h.min_line_length, h.max_line_gap)

def calc_point(y_crop, slope, intercept):
    return (int((y_crop - intercept)/slope), y_crop)

def add_line_to_image(image, line, crop_lane=False):
    if line is None:
        return image
    slope, intercept = line
    
    y_crop_start = 750 if crop_lane else 0
    y_crop_end = 400 if crop_lane else image.shape[0]
    
    start_point = calc_point(y_crop_start, slope, intercept)
    end_point = calc_point(y_crop_end, slope, intercept)
    cv2.line(image, start_point, end_point, (0, 255, 0), 10)
    return image

def draw_ans_for_debug(img, left_lane, right_lane, vp):
    add_line_to_image(img, left_lane, crop_lane=True)
    add_line_to_image(img, right_lane, crop_lane=True)
    cv2.circle(img, (int(vp[0]), int(vp[1])), 5, (0, 0, 255), 8)
    return img

def show(imgs, cmap=None):
    rows = (len(imgs)+1)//2
    plt.figure(figsize=(10, 11))
    for i, img in enumerate(imgs):
        plt.subplot(rows, 2, i+1)
        cmap = 'gray' if len(img.shape)==2 else cmap
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=8)
    plt.show()

