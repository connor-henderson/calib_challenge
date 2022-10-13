import numpy as np
import cv2
import scipy
from helpers.const import MAX_LANE_SLOPE, MIN_LANE_SLOPE, MOMENTUM, STANDARD_LEFT_LANE, STANDARD_RIGHT_LANE
from helpers.lines import get_pts_close_to_adj_line

def matches_left_lane_props(slope):
    return slope < -MIN_LANE_SLOPE and slope > -MAX_LANE_SLOPE

def matches_right_lane_props(slope):
    return slope > MIN_LANE_SLOPE and slope < MAX_LANE_SLOPE

def pts_to_lane(pts, side):
    standard_lane = STANDARD_LEFT_LANE if side == 'left' else STANDARD_RIGHT_LANE
        
    if len(pts) < 3: 
        return standard_lane
    
    pts_close_to_line = get_pts_close_to_adj_line(pts, standard_lane[0], standard_lane[1])
    x = [pt[0] for pt in pts_close_to_line]
    y = [pt[1] for pt in pts_close_to_line]
    
    if len(pts_close_to_line) < 3:
        return standard_lane
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
    
    has_lane_props = matches_left_lane_props(slope) if side == 'left' else matches_right_lane_props(slope)
    if has_lane_props:
        return (slope, intercept)
    return standard_lane

def momentum_calc(prev, curr, scaled_momentum):
    return prev * scaled_momentum + curr * (1 - scaled_momentum)

def lane_with_mom_calc(lane, prev_lanes):
    prev_lanes.append(lane)
    prev_lane_avg = np.mean(np.array(prev_lanes, dtype=object), axis=0)
    
    scaling_factor = len(prev_lanes)/(len(prev_lanes) + 5)
    scaled_momentum = MOMENTUM * scaling_factor
    
    slope_with_scaled_momentum = momentum_calc(prev_lane_avg[0], lane[0], scaled_momentum)
    intercept_with_scaled_momentum = momentum_calc(prev_lane_avg[1], lane[1], scaled_momentum)
    return slope_with_scaled_momentum, intercept_with_scaled_momentum