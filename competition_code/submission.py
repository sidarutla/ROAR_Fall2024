"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

#!/usr/bin/env python3

import numpy as np
from math import atan2, hypot, sin, cos, pi

class RegulatedPurePursuitController:
    def __init__(self):
        # Controller parameters
        self.lookahead_distance = 0.5  # meters
        self.max_linear_velocity = 0.5  # m/s
        self.min_linear_velocity = 0.1  # m/s
        self.max_angular_velocity = 1.0  # rad/s
        self.min_lookahead_distance = 0.3
        self.max_lookahead_distance = 0.9
        self.curvature_scaling_factor = 2.0
        
    def compute_velocity(self, robot_pose, path, current_velocity=None):
        """
        Compute the velocity commands to follow the path.
        
        Args:
            robot_pose: [x, y, theta] - Robot's current pose
            path: [[x1,y1], [x2,y2], ...] - List of waypoints
            current_velocity: [v, w] - Current linear and angular velocity
            
        Returns:
            v, w - Linear and angular velocity commands
        """
        if len(path) < 2:
            return 0.0, 0.0
            
        # Convert path to numpy array for easier manipulation
        path = np.array(path)
        
        # Find lookahead point
        lookahead_point = self._find_lookahead_point(robot_pose, path)
        if lookahead_point is None:
            return 0.0, 0.0
            
        # Calculate curvature and velocities
        curvature = self._compute_curvature(robot_pose, lookahead_point)
        v, w = self._compute_velocity_commands(curvature)
        
        return v, w
        
    def _find_lookahead_point(self, robot_pose, path):
        """Find the lookahead point on the path"""
        robot_x, robot_y, _ = robot_pose
        
        # Find closest point on path
        distances = np.linalg.norm(path - [robot_x, robot_y], axis=1)
        closest_idx = np.argmin(distances)
        
        # Search for lookahead point
        for i in range(closest_idx, len(path)):
            point = path[i]
            distance = hypot(point[0] - robot_x, point[1] - robot_y)
            
            if distance >= self.lookahead_distance:
                return point
                
        # If no point found, return the last point
        return path[-1] if len(path) > 0 else None
        
    def _compute_curvature(self, robot_pose, lookahead_point):
        """Compute the curvature to reach the lookahead point"""
        robot_x, robot_y, robot_theta = robot_pose
        
        # Transform lookahead point to robot frame
        dx = lookahead_point[0] - robot_x
        dy = lookahead_point[1] - robot_y
        
        # Calculate angle to lookahead point
        alpha = atan2(dy, dx) - robot_theta
        
        # Normalize alpha to [-pi, pi]
        while alpha > pi:
            alpha -= 2 * pi
        while alpha < -pi:
            alpha += 2 * pi

def create_controller():
    """Factory function to create the controller"""
    return RegulatedPurePursuitController()
