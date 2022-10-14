import numpy as np
import scipy

def has_lane_slope(side, slope, min_lane_slope, max_lane_slope):
    if side == 'left':
        return slope < -min_lane_slope and slope > -max_lane_slope
    return slope > min_lane_slope and slope < max_lane_slope

def get_pts_close_to_std_line(pts, slope, intercept):
    new_pts = []
    for x, y in pts:
        if abs(y - (slope*x + intercept)) < 100: # this could be a param, just pass it as a prop
            new_pts.append((x, y))
    return new_pts

def pts_to_lane(pts, side, standard_lane, min_lane_slope, max_lane_slope):
    if len(pts) < 3:
        return standard_lane

    pts_close_to_line = get_pts_close_to_std_line(pts, standard_lane[0], standard_lane[1])
    x = [pt[0] for pt in pts_close_to_line]
    y = [pt[1] for pt in pts_close_to_line]
    
    if len(pts_close_to_line) < 3:
        return standard_lane
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
    
    if has_lane_slope(side, slope, min_lane_slope, max_lane_slope):
        return (slope, intercept)
    return standard_lane

def momentum_calc(prev, curr, scaled_momentum):
    return prev * scaled_momentum + curr * (1 - scaled_momentum)

def lane_with_mom_calc(lane, prev_lanes, momentum=0.95, scaling_constant=5):
    prev_lanes.append(lane)
    prev_lane_avg = np.mean(np.array(prev_lanes, dtype=object), axis=0)
    
    scaling_factor = len(prev_lanes)/(len(prev_lanes) + scaling_constant)
    scaled_momentum = momentum * scaling_factor
    
    slope_with_scaled_momentum = momentum_calc(prev_lane_avg[0], lane[0], scaled_momentum)
    intercept_with_scaled_momentum = momentum_calc(prev_lane_avg[1], lane[1], scaled_momentum)
    return slope_with_scaled_momentum, intercept_with_scaled_momentum