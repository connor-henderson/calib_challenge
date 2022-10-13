import cv2
import numpy as np
import importlib
from openpilot.transformations import get_calib_from_vp
from helpers import lines, lanes, images
importlib.reload(lines)
importlib.reload(lanes)
importlib.reload(images)
lines_to_filtered_pts = lines.lines_to_filtered_pts
get_intersection = lines.get_intersection
pts_to_lane = lanes.pts_to_lane
lane_with_mom_calc = lanes.lane_with_mom_calc
get_roi_from_img = images.get_roi_from_img
get_hough_lines_p = images.get_hough_lines_p
draw_ans_for_debug = images.draw_ans_for_debug

def image_to_vp(image, prev_left_lanes, prev_right_lanes, debug=False):
    # process the image from color -> grayscale -> canny -> masked by ROI -> probabilistic hough lines
    gray_img = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray_img, 25, 100)
    masked_edges = get_roi_from_img(canny)
    hough_lines = get_hough_lines_p(masked_edges)
    
    # take the hough lines, filter them by slope and intercept into a set of acceptable points
    left_pts, right_pts = lines_to_filtered_pts(hough_lines)

    # take the acceptable points, remove the ones super far away from a "standard lane" and return a best fit line
    left_lane = pts_to_lane(left_pts, 'left')
    right_lane = pts_to_lane(right_pts, 'right')

    # append the calc'ed lanes to the prev lanes if they exist and calc the lane line with momentum
    left_lane_with_mom = lane_with_mom_calc(left_lane, prev_left_lanes)
    right_lane_with_mom = lane_with_mom_calc(right_lane, prev_right_lanes)
    
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