"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

#!/usr/bin/env python3

import numpy as np
from math import atan2, hypot, sin, cos, pi

class RegulatedPurePursuitController:
    def __init__(self):
        # Core parameters
        self.lookahead_distance = 0.5
        self.max_linear_velocity = 0.5
        self.min_linear_velocity = 0.1
        self.max_angular_velocity = 1.0
        self.min_lookahead_distance = 0.3
        self.max_lookahead_distance = 0.9
        
        # Velocity regulation parameters
        self.curvature_scaling_factor = 2.0
        self.acceleration_limit = 0.5  # m/s^2
        self.deceleration_limit = 0.8  # m/s^2
        self.angular_acceleration_limit = 1.5  # rad/s^2
        self.approach_velocity_scaling_dist = 1.0
        self.approach_velocity_scaling_factor = 0.5
        
        # Controller state
        self.previous_linear_velocity = 0.0
        self.previous_angular_velocity = 0.0
        self.last_time = None

    def compute_velocity(self, robot_pose, path, current_velocity=None, dt=0.1):
        """
        Compute regulated velocity commands to follow the path.
        
        Args:
            robot_pose: [x, y, theta] - Robot's current pose
            path: [[x1,y1], [x2,y2], ...] - List of waypoints
            current_velocity: [v, w] - Current linear and angular velocity
            dt: Time step since last update
        """
        if len(path) < 2:
            return 0.0, 0.0

        path = np.array(path)
        
        # Find lookahead point with adaptive distance
        lookahead_point = self._find_lookahead_point(robot_pose, path)
        if lookahead_point is None:
            return 0.0, 0.0

        # Compute base velocities
        curvature = self._compute_curvature(robot_pose, lookahead_point)
        base_linear_vel, base_angular_vel = self._compute_velocity_commands(curvature)

        # Apply sophisticated velocity regulation
        linear_vel, angular_vel = self._regulate_velocity(
            base_linear_vel,
            base_angular_vel,
            robot_pose,
            path,
            dt
        )

        # Update controller state
        self.previous_linear_velocity = linear_vel
        self.previous_angular_velocity = angular_vel

        return linear_vel, angular_vel

    def _regulate_velocity(self, linear_vel, angular_vel, robot_pose, path, dt):
        """
        Advanced velocity regulation considering multiple factors
        """
        # 1. Apply acceleration limits
        linear_vel = self._apply_acceleration_limits(
            linear_vel, 
            self.previous_linear_velocity, 
            dt
        )
        angular_vel = self._apply_acceleration_limits(
            angular_vel,
            self.previous_angular_velocity,
            dt,
            is_angular=True
        )

        # 2. Scale velocity based on path features
        linear_vel = self._apply_path_scaling(
            linear_vel,
            robot_pose,
            path
        )

        # 3. Scale velocity when approaching goal
        linear_vel = self._apply_goal_approach_scaling(
            linear_vel,
            robot_pose,
            path
        )

        # 4. Ensure minimum velocity constraints
        linear_vel = max(linear_vel, self.min_linear_velocity)
        angular_vel = np.clip(
            angular_vel,
            -self.max_angular_velocity,
            self.max_angular_velocity
        )

        return linear_vel, angular_vel

    def _apply_acceleration_limits(self, target_vel, current_vel, dt, is_angular=False):
        """Apply acceleration limits to velocity changes"""
        acc_limit = self.angular_acceleration_limit if is_angular else self.acceleration_limit
        dec_limit = acc_limit * 1.5  # Higher deceleration limit for safety

        max_acc = acc_limit * dt
        max_dec = dec_limit * dt

        if target_vel > current_vel:
            return min(target_vel, current_vel + max_acc)
        else:
            return max(target_vel, current_vel - max_dec)

    def _apply_path_scaling(self, linear_vel, robot_pose, path):
        """Scale velocity based on path characteristics"""
        # Calculate local path curvature
        local_curvature = self._estimate_path_curvature(path)
        
        # Scale velocity based on curvature
        curvature_scaling = 1.0 / (1.0 + self.curvature_scaling_factor * abs(local_curvature))
        
        return linear_vel * curvature_scaling

    def _apply_goal_approach_scaling(self, linear_vel, robot_pose, path):
        """Scale velocity when approaching the goal"""
        if len(path) < 2:
            return linear_vel

        # Distance to goal
        dist_to_goal = hypot(
            path[-1][0] - robot_pose[0],
            path[-1][1] - robot_pose[1]
        )

        # Apply scaling when approaching goal
        if dist_to_goal < self.approach_velocity_scaling_dist:
            scaling = max(
                self.approach_velocity_scaling_factor,
                dist_to_goal / self.approach_velocity_scaling_dist
            )
            return linear_vel * scaling

        return linear_vel

    def _estimate_path_curvature(self, path, window_size=3):
        """Estimate local path curvature using a sliding window"""
        if len(path) < window_size:
            return 0.0

        # Use Menger curvature for estimation
        curvatures = []
        for i in range(len(path) - window_size + 1):
            points = path[i:i + window_size]
            curvature = self._menger_curvature(points)
            curvatures.append(abs(curvature))

        return np.mean(curvatures)

    @staticmethod
    def _menger_curvature(points):
        """Calculate Menger curvature for three points"""
        if len(points) != 3:
            return 0.0

        p1, p2, p3 = points
        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
        
        d1 = hypot(p2[0] - p1[0], p2[1] - p1[1])
        d2 = hypot(p3[0] - p2[0], p3[1] - p2[1])
        d3 = hypot(p1[0] - p3[0], p1[1] - p3[1])
        
        try:
            return 4 * area / (d1 * d2 * d3)
        except ZeroDivisionError:
            return 0.0

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
