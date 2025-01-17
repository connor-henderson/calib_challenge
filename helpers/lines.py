import numpy as np
from helpers.lanes import has_lane_slope

def filter_lines_to_points(hough_lines, min_lane_slope, max_lane_slope):
    left_pts = []
    right_pts = []
    for line in hough_lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope = (y2-y1)/((x2-x1) + 1e-15) # avoid division by zero
        
        if has_lane_slope('left', slope, min_lane_slope, max_lane_slope):
            left_pts.append((x1, y1))
            left_pts.append((x2, y2))
        elif has_lane_slope('right', slope, min_lane_slope, max_lane_slope):
            right_pts.append((x1, y1))
            right_pts.append((x2, y2))
    return left_pts, right_pts

def left_lane_min_max_y(x, y):
    min_y = -x + 864
    max_y = -x + 1196
    return min_y <= y <= max_y 

def right_lane_min_max_y(x, y):
    min_y = x - 350
    max_y = x + 50
    return min_y <= y <= max_y

def filter_points(pts, lane):
    new_pts = []
    for x, y in pts:
        func = left_lane_min_max_y if lane == 'left' else right_lane_min_max_y
        if func(x, y):
            new_pts.append((x, y))
    return new_pts

def get_intersection(line1, line2):
    if line1 is None or line2 is None:
        raise RuntimeError('line1 or line2 is None')
    m1, b1 = line1
    m2, b2 = line2
    xi = (b1-b2) / ((m2-m1) + 1e-15) # avoid division by zero
    yi = m1 * xi + b1
    return np.array([abs(xi), abs(yi)])

def lines_to_filtered_pts(hough_lines, min_lane_slope=0.5, max_lane_slope=1.5):
    if hough_lines is None:
        return [], []
    left_pts, right_pts = filter_lines_to_points(hough_lines, min_lane_slope, max_lane_slope)

    left_filtered_pts = filter_points(left_pts, 'left')
    right_filtered_pts = filter_points(right_pts, 'right')
    
    return left_filtered_pts, right_filtered_pts

