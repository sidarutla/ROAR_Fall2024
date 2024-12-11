import numpy as np
import math
from typing import List, Tuple, Optional
import roar_py_interface
from scipy.interpolate import CubicSpline
from dataclasses import dataclass

@dataclass
class PathPoint:
    x: float
    y: float
    yaw: float
    curvature: float
    velocity: float

class RPPController:
    def __init__(self):
        self.k = 0.55  # Increased from 0.45
        self.Lfc = 5.0  # Reduced from 6.0 for tighter following
        self.Kp = 1.2   # Increased from 0.8
        self.Ki = 0.08  # Increased from 0.05
        self.Kd = 0.03  # Increased from 0.02
        self.error_integral = 0
        self.prev_error = 0
        self.dt = 0.05
        self.wheel_base = 4.7
        self.max_steer = np.deg2rad(35.0)  # Reduced from 45.0 for stability
        
        # Path smoothing parameters
        self.smooth_path_resolution = 0.6  # Reduced from 0.8
        self.path_smoothing_window = 5    # Reduced from 7
        
        # Dynamic lookahead parameters
        self.min_lookahead = 5.0   # Reduced from 6.0
        self.max_lookahead = 25.0  # Reduced from 30.0
        self.curvature_factor = 1.8 # Increased from 1.5
        self.speed_factor = 0.15   # Increased from 0.12
        
        # Speed control parameters
        self.max_speed = 320.0     # Increased from 300.0
        self.min_speed = 35.0      # Increased from 30.0
        self.max_accel = 5.0       # Increased from 4.0
        self.max_decel = -7.0      # More aggressive from -6.0
        self.safety_margin = 1.3   # Reduced from 1.4

    def smooth_path(self, waypoints: List[roar_py_interface.RoarPyWaypoint]) -> List[PathPoint]:
        x = np.array([wp.location[0] for wp in waypoints])
        y = np.array([wp.location[1] for wp in waypoints])
        
        # Racing line optimization using path shortening
        for _ in range(3):  # Multiple iterations for better optimization
            for i in range(1, len(x)-1):
                # Check if we can cut the corner
                prev_point = np.array([x[i-1], y[i-1]])
                curr_point = np.array([x[i], y[i]])
                next_point = np.array([x[i+1], y[i+1]])
                
                # Calculate optimal racing line point
                v1 = curr_point - prev_point
                v2 = next_point - curr_point
                angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                
                # More aggressive corner cutting at higher speeds
                cut_factor = min(0.3, abs(angle) * 0.15)  
                optimal_point = curr_point + (next_point - prev_point) * cut_factor
                
                # Apply optimization while maintaining safety margin
                x[i] = optimal_point[0]
                y[i] = optimal_point[1]
        
        # Enhanced spline fitting with tension parameter
        t = np.arange(len(x))
        cs_x = CubicSpline(t, x, bc_type='natural')
        cs_y = CubicSpline(t, y, bc_type='natural')
        
        # Generate dense path points with dynamic resolution
        num_points = int(len(x) * 2)  # Increased density for better accuracy
        t_dense = np.linspace(0, len(x)-1, num_points)
        x_dense = cs_x(t_dense)
        y_dense = cs_y(t_dense)
        
        # Calculate curvature and optimal velocities
        dx = cs_x.derivative()(t_dense)
        dy = cs_y.derivative()(t_dense)
        ddx = cs_x.derivative(2)(t_dense)
        ddy = cs_y.derivative(2)(t_dense)
        
        curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**(3/2)
        
        # Optimize speed profile based on curvature
        max_lateral_accel = 12.0  # Increased from 9.81
        v_curve = np.sqrt(max_lateral_accel / (curvature + 1e-10)) * 3.6
        
        # Dynamic speed adjustment based on section characteristics
        v_curve = self.optimize_speed_profile(v_curve, curvature)
        
        return [PathPoint(x=x_dense[i], y=y_dense[i], 
                         yaw=np.arctan2(dy[i], dx[i]),
                         curvature=curvature[i], 
                         velocity=v_curve[i]) 
                for i in range(len(x_dense))]

    def optimize_speed_profile(self, velocities: np.ndarray, curvatures: np.ndarray) -> np.ndarray:
        # Optimize entry and exit speeds for corners
        v_optimized = velocities.copy()
        
        # Find corner entry and exit points
        curvature_threshold = 0.01
        corner_indices = np.where(curvatures > curvature_threshold)[0]
        
        for i in range(len(corner_indices)):
            if i > 0 and corner_indices[i] - corner_indices[i-1] > 1:
                # Corner entry optimization
                entry_idx = corner_indices[i]
                v_entry = min(velocities[entry_idx] * 1.2, self.max_speed)
                v_optimized[max(0, entry_idx-5):entry_idx] = np.linspace(
                    v_entry, velocities[entry_idx], min(5, entry_idx))
                
                # Corner exit optimization
                exit_idx = corner_indices[i-1]
                v_exit = min(velocities[exit_idx] * 1.3, self.max_speed)
                v_optimized[exit_idx:min(exit_idx+5, len(velocities))] = np.linspace(
                    velocities[exit_idx], v_exit, min(5, len(velocities)-exit_idx))
        
        return v_optimized

    def calculate_velocity_profile(self, curvatures: np.ndarray) -> np.ndarray:
        # Calculate maximum velocity based on lateral acceleration limit
        max_lateral_accel = 9.81 * 2.0  # g-force limit
        v_curve = np.sqrt(max_lateral_accel / (curvatures + 1e-10)) * 3.6  # Convert to km/h
        
        # Apply velocity constraints
        v_curve = np.clip(v_curve, self.min_speed, self.max_speed)
        
        # Forward pass for acceleration limits
        v_forward = np.copy(v_curve)
        for i in range(1, len(v_forward)):
            v_prev = v_forward[i-1]
            v_max = np.sqrt(v_prev**2 + 2*self.max_accel*self.smooth_path_resolution)
            v_forward[i] = min(v_forward[i], v_max)
        
        # Backward pass for deceleration limits
        v_backward = np.copy(v_forward)
        for i in range(len(v_backward)-2, -1, -1):
            v_next = v_backward[i+1]
            v_max = np.sqrt(v_next**2 - 2*self.max_decel*self.smooth_path_resolution)
            v_backward[i] = min(v_backward[i], v_max)
        
        return v_backward

    def get_target_point(self, current_pos: np.ndarray, current_speed: float, 
                        path_points: List[PathPoint]) -> Tuple[PathPoint, float]:
        # Calculate dynamic lookahead distance
        current_curvature = min(1.0, path_points[0].curvature)
        lookahead = self.k * current_speed + self.Lfc
        lookahead *= (1.0 - self.curvature_factor * current_curvature)
        lookahead = np.clip(lookahead, self.min_lookahead, self.max_lookahead)
        
        # Find target point
        min_dist = float('inf')
        target_point = None
        target_dist = 0
        
        for point in path_points:
            dist = np.hypot(point.x - current_pos[0], point.y - current_pos[1])
            if dist > lookahead and dist < min_dist:
                min_dist = dist
                target_point = point
                target_dist = dist
                
        return target_point, target_dist

    def calculate_control(self, current_pos: np.ndarray, current_yaw: float, 
                         current_speed: float, target_point: PathPoint, 
                         target_dist: float) -> Tuple[float, float, float]:
        # Calculate steering angle using pure pursuit
        alpha = math.atan2(target_point.y - current_pos[1], target_point.x - current_pos[0]) - current_yaw
        steer_angle = math.atan2(2.0 * self.wheel_base * math.sin(alpha), target_dist)
        steer_angle = np.clip(steer_angle, -self.max_steer, self.max_steer)
        
        # Calculate throttle and brake using PID
        speed_error = target_point.velocity - current_speed
        self.error_integral += speed_error * self.dt
        error_derivative = (speed_error - self.prev_error) / self.dt
        
        control_output = (self.Kp * speed_error + 
                         self.Ki * self.error_integral + 
                         self.Kd * error_derivative)
        
        self.prev_error = speed_error
        
        # Convert control output to throttle and brake
        if control_output >= 0:
            throttle = min(1.0, control_output)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -control_output)
            
        return steer_angle, throttle, brake 