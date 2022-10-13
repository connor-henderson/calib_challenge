import numpy as np
import cv2
import scipy

def add_line_to_image(image, line):
    if line is None:
        return image
    slope, intercept = line
    start_point = (int((image.shape[0] - intercept)/slope), image.shape[0])
    end_point = (int((0 - intercept)/slope), 0)
    cv2.line(image, start_point, end_point, (0, 255, 0), 10)
    return image

def get_roi_from_img(img):
    mask = np.zeros_like(img)
    ignore_mask_color = 255
    y, x = img.shape
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

def get_hough_lines_p(img, rho=2, theta=31*np.pi/180, threshold=10, min_line_length=5, max_line_gap=20):
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

min_lane_slope = 0.5
max_lane_slope = 1.5
def filter_lines(copy_original_img, hough_lines):
    left_filtered_lines = []
    right_filtered_lines = []
    for line in hough_lines:
        x1, y1, x2, y2 = line.reshape(4)
        # length = np.sqrt((x2-x1)**2 + (y2-y1)**2) # not used
        slope = (y2-y1)/((x2-x1) + 0.0000001) # avoid division by zero
        intercept = y1 - (slope * x1)

        left_intercept_prop = intercept > 800 and intercept < 1300
        right_intercept_prop = intercept > -50 and intercept < 300
        
        left_slope_prop = slope < -min_lane_slope and slope > -max_lane_slope
        right_slope_prop = slope > min_lane_slope and slope < max_lane_slope
        
        if left_slope_prop and left_slope_prop:
            left_filtered_lines.append(line)
        elif right_slope_prop and right_intercept_prop: #slope > min_lane_slope and slope < max_lane_slope
            right_filtered_lines.append(line)
    return left_filtered_lines, right_filtered_lines

def lines_to_points(lines):
    pts = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            pt1, pt2 = (x1, y1), (x2, y2)
            pts.append(pt1)
            pts.append(pt2)
    return pts

def left_lane_min_max_y(x, y):
    min_y = -x + 864
    max_y = -x + 1196
    return min_y <= y <= max_y 

def right_lane_min_max_y(x, y):
    min_y = x - 300
    max_y = x
    return min_y <= x <= max_y

def filter_points(pts, lane, img):
    new_pts = []
    for x, y in pts:
        func = left_lane_min_max_y if lane == 'left' else right_lane_min_max_y
        if func(x, y):
            new_pts.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    return new_pts, img

def lines_to_filtered_pts(image, hough_lines):
    left_filtered_lines, right_filtered_lines = filter_lines(image, hough_lines)
    left_pts = lines_to_points(left_filtered_lines)
    right_pts = lines_to_points(right_filtered_lines)
    left_filtered_pts, left_pts_img  = filter_points(left_pts, 'left', np.copy(image))
    right_filtered_pts, right_pts_image = filter_points(right_pts, 'right', np.copy(image))
    return left_filtered_pts, right_filtered_pts, left_pts_img, right_pts_image

# NEW LANE FILTERING METHOD
standard_left_lane = (-0.71, 840) # (slope, intercept)
standard_right_lane = (0.71, 0) # (slope, intercept)

def matches_left_lane_props(slope):
    return slope < -min_lane_slope and slope > -max_lane_slope

def matches_right_lane_props(slope):
    return slope > min_lane_slope and slope < max_lane_slope

def get_pts_close_to_adj_line(pts, slope, intercept, img):
    new_pts = []
    for x, y in pts:
        if abs(y - (slope*x + intercept)) < 100:
            new_pts.append((x, y))
            # cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    return new_pts, img

def pts_to_lane(pts, side, img):
    if len(pts) < 3: 
        return None
    
    slope, intercept = standard_left_lane if side == 'left' else standard_right_lane
    pts_close_to_line, new_img = get_pts_close_to_adj_line(pts, slope, intercept, img)
    
    x = [pt[0] for pt in pts_close_to_line]
    y = [pt[1] for pt in pts_close_to_line]
    if len(pts_close_to_line) < 3:
        return None
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
    
    has_lane_props = matches_left_lane_props(slope) if side == 'left' else matches_right_lane_props(slope)
    if has_lane_props:
        # print('slope: ', slope, 'matches left lane props', matches_left_lane_props(slope))
        # start_point = (int((img.shape[0] - intercept)/slope), img.shape[0])
        # end_point = (int((0 - intercept)/slope), 0)
        # cv2.line(new_img, start_point, end_point, (0, 255, 0), 10)
        return (slope, intercept)
    return None

def add_line_to_image(image, line):
    if line is None:
        return image
    slope, intercept = line
    start_point = (int((image.shape[0] - intercept)/slope), image.shape[0])
    end_point = (int((0 - intercept)/slope), 0)
    cv2.line(image, start_point, end_point, (0, 255, 0), 10)
    return image

def draw_ans_for_debug(img, left_lane, right_lane):
    copy_img = np.copy(img)
    if left_lane is not None:
        slope, intercept = left_lane
        start_point = (int((copy_img.shape[0] - intercept)/slope), copy_img.shape[0])
        end_point = (int((0 - intercept)/slope), 0)
        cv2.line(copy_img, start_point, end_point, (0, 255, 0), 10)
    else:
        slope, intercept = standard_left_lane
        start_point = (int((copy_img.shape[0] - intercept)/slope), copy_img.shape[0])
        end_point = (int((0 - intercept)/slope), 0)
        cv2.line(copy_img, start_point, end_point, (0, 0, 255), 10)
    if right_lane is not None:
        slope, intercept = right_lane
        start_point = (int((copy_img.shape[0] - intercept)/slope), copy_img.shape[0])
        end_point = (int((0 - intercept)/slope), 0)
        cv2.line(copy_img, start_point, end_point, (0, 255, 0), 10)
    else:
        slope, intercept = standard_right_lane
        start_point = (int((copy_img.shape[0] - intercept)/slope), copy_img.shape[0])
        end_point = (int((0 - intercept)/slope), 0)
        cv2.line(copy_img, start_point, end_point, (0, 0, 255), 10)
    return copy_img
    
def lane_with_mom_calc(lane, prev_lanes):
    if lane is None: return lane
    prev_lanes.append(lane)
    if len(prev_lanes) < 50:
        return lane
    
    momentum = 0.05
    prev_lane_avg = np.mean(np.array(prev_lanes), axis=0)
    slope_with_momentum = momentum * lane[0] + (1 - momentum) * prev_lane_avg[0]
    intercept_with_momentum = lane[1] * momentum + prev_lane_avg[1] * (1 - momentum)
    return slope_with_momentum, intercept_with_momentum

def get_intersection(line1, line2):
    if line1 is None or line2 is None:
        return None
    m1, b1 = line1
    m2, b2 = line2
    xi = (b1-b2) / (m2-m1)
    yi = m1 * xi + b1
    return np.array([abs(xi), abs(yi)])

def get_intersection_with_standard_lanes(left_lane, right_lane):
    left_line = left_lane if left_lane is not None else standard_left_lane
    right_line = right_lane if right_lane is not None else standard_right_lane
    return get_intersection(left_line, right_line)

def process_image(image, prev_left_lanes, prev_right_lanes, prev_canny_imgs=None):
    copy_img = np.copy(image)
    gray_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(copy_img, 50, 100)
    if prev_canny_imgs is not None:
        canny = combine_with_previous_second_canny(canny, prev_canny_imgs)

    masked_edges = get_roi_from_img(canny)
    hough_lines = get_hough_lines_p(masked_edges)
    # hough_img = draw_lines(np.copy(image), hough_lines)
    
    # take the hough lines, give me a set of acceptable points
    left_pts, right_pts, left_pts_img, right_pts_image = lines_to_filtered_pts(copy_img, hough_lines)

    # take the acceptable points, remove the ones super far away from a "standard lane" and return a best fit line
    test_img = np.copy(image)
    left_lane = pts_to_lane(left_pts, 'left', test_img)
    right_lane = pts_to_lane(right_pts, 'right', test_img)

    # append the calc'ed lanes to the prev lanes if they exist and calc the momentum
    left_lane_with_mom = lane_with_mom_calc(left_lane, prev_left_lanes)
    right_lane_with_mom = lane_with_mom_calc(right_lane, prev_right_lanes)
    new_img = draw_ans_for_debug(image, left_lane_with_mom, right_lane_with_mom)
    
    # draw_ans_for_debug(np.copy(image), left_lane, right_lane)
    vp = get_intersection_with_standard_lanes(left_lane_with_mom, right_lane_with_mom)
    
    return vp, new_img
