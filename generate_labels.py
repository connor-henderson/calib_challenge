import cv2
import numpy as np
from helpers.images import draw_ans_for_debug, get_hough_lines_p, get_roi_from_img
from helpers.lanes import lane_with_mom_calc, pts_to_lane
from helpers.lines import get_intersection, lines_to_filtered_pts
from openpilot.transformations import get_calib_from_vp


params = {
'standard_left_lane':  (-0.71, 840), # (slope, intercept)
'standard_right_lane':  (0.71, 0), # (slope, intercept)
'min_lane_slope':  0.5,
'max_lane_slope':  1.5,
'momentum':  0.95,
'scaling_constant':  10,
'canny_low_threshold':  50,
'canny_high_threshold':  100,
'hough': { 'rho': 1,  'theta': 15*np.pi/180,  'threshold': 5,  'min_line_length': 5,  'max_line_gap': 5 },
'roi':  { 'x_bottom_offset': 50, 'x_top_offset': 300, 'y_bottom_offset': 150, 'y_top_offset': -10 },
}
# def get_hough_lines_p_test(img, rho=2, theta=31*np.pi/180, threshold=10, min_line_length=5, max_line_gap=20):
#     return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)



def image_to_vp(image, prev_left_lanes, prev_right_lanes, params=params, debug=False):
    # process the image from color -> grayscale -> canny -> masked by ROI -> probabilistic hough lines
    copy_img = np.copy(image)
    gray_img = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(copy_img, params['canny_low_threshold'], params['canny_high_threshold'])
    masked_edges = get_roi_from_img(canny, **params['roi'])
    hough_lines = get_hough_lines_p(masked_edges, **params['hough'])
    
    # take the hough lines, filter them by slope and intercept into a set of acceptable points
    left_pts, right_pts = lines_to_filtered_pts(hough_lines, params['min_lane_slope'], params['max_lane_slope'])

    # take the acceptable points, remove the ones super far away from a "standard lane" and return a best fit line
    left_lane = pts_to_lane(left_pts, 'left', params['standard_left_lane'], params['min_lane_slope'], params['max_lane_slope'])
    right_lane = pts_to_lane(right_pts, 'right', params['standard_right_lane'], params['min_lane_slope'], params['max_lane_slope'])

    # append the calc'ed lanes to the prev lanes if they exist and calc the lane line with momentum
    left_lane_with_mom = lane_with_mom_calc(left_lane, prev_left_lanes, params['momentum'], params['scaling_constant'])
    right_lane_with_mom = lane_with_mom_calc(right_lane, prev_right_lanes, params['momentum'], params['scaling_constant'])
    
    # use these lanes to calc the vanishing point
    vp = get_intersection(left_lane_with_mom, right_lane_with_mom)
    
    if debug:
        debug_img = draw_ans_for_debug(np.copy(image), left_lane_with_mom, right_lane_with_mom, vp)
        return vp, debug_img
    return vp

def generate_and_write_labels(filename, frames):
  prev_left_lanes = []
  prev_right_lanes = []

  with open(filename, 'w') as f:
      for frame in frames:
          vp = image_to_vp(frame, prev_left_lanes, prev_right_lanes)
          _roll_calib, pitch_calib, yaw_calib = get_calib_from_vp(vp)
          
          f.write(f'{pitch_calib} {yaw_calib}\n')
      f.close()

# 