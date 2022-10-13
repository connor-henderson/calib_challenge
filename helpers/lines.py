import numpy as np
from helpers.const import *

def filter_lines_to_points(hough_lines):
    left_pts = []
    right_pts = []
    for line in hough_lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope = (y2-y1)/((x2-x1) + 0.0000001) # avoid division by zero
        intercept = y1 - (slope * x1)

        left_intercept_prop = intercept > 800 and intercept < 1300
        right_intercept_prop = intercept > -50 and intercept < 300
        
        left_slope_prop = slope < -MIN_LANE_SLOPE and slope > -MAX_LANE_SLOPE
        right_slope_prop = slope > MIN_LANE_SLOPE and slope < MAX_LANE_SLOPE
        
        if left_slope_prop and left_slope_prop:
            left_pts.append((x1, y1))
            left_pts.append((x2, y2))
        elif right_slope_prop and right_intercept_prop:
            right_pts.append((x1, y1))
            right_pts.append((x2, y2))
    return left_pts, right_pts

def left_lane_min_max_y(x, y):
    min_y = -x + 864
    max_y = -x + 1196
    return min_y <= y <= max_y 
    

def right_lane_min_max_y(x, y):
    min_y = x - 300
    max_y = x
    return min_y <= x <= max_y

def filter_points(pts, lane):
    new_pts = []
    for x, y in pts:
        func = left_lane_min_max_y if lane == 'left' else right_lane_min_max_y
        if func(x, y):
            new_pts.append((x, y))
    return new_pts 

def get_pts_close_to_adj_line(pts, slope, intercept):
    new_pts = []
    for x, y in pts:
        if abs(y - (slope*x + intercept)) < 100:
            new_pts.append((x, y))
    return new_pts

def get_intersection(line1, line2):
    if line1 is None or line2 is None:
        return None
    m1, b1 = line1
    m2, b2 = line2
    xi = (b1-b2) / (m2-m1)
    yi = m1 * xi + b1
    return np.array([abs(xi), abs(yi)])

def lines_to_filtered_pts(hough_lines):
    if hough_lines is None:
        return [], []
    left_pts, right_pts = filter_lines_to_points(hough_lines)
    
    left_filtered_pts = filter_points(left_pts, 'left')
    right_filtered_pts = filter_points(right_pts, 'right')
    
    return left_filtered_pts, right_filtered_pts

