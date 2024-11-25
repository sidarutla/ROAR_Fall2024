"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

#!/usr/bin/env python3

import numpy as np
from math import atan2, hypot, sin, cos, pi, exp
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist

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
        self.acceleration_limit = 0.5
        self.deceleration_limit = 0.8
        self.angular_acceleration_limit = 1.5
        self.approach_velocity_scaling_dist = 1.0
        self.approach_velocity_scaling_factor = 0.5
        
        # Obstacle avoidance parameters
        self.obstacle_influence_radius = 0.5
        self.safety_radius = 0.3
        self.obstacle_weight = 1.0
        self.repulsive_field_strength = 0.5
        self.max_obstacle_influence = 0.8
        
        # Controller state
        self.previous_linear_velocity = 0.0
        self.previous_angular_velocity = 0.0
        self.last_time = None
        
        # Path preprocessing parameters
        self.path_resolution = 0.1  # meters between points
        self.smoothing_factor = 0.1  # B-spline smoothing
        self.max_path_points = 1000
        self.min_path_points = 10
        self.pruning_distance = 0.5  # meters
        self.look_ahead_points = 5
        
        # Path memory
        self.processed_path = None
        self.path_curvatures = None
        self.current_path_index = 0

    def set_path(self, path):
        """
        Preprocess and store a new path
        
        Args:
            path: [[x1,y1], [x2,y2], ...] - List of waypoints
        """
        if len(path) < 2:
            return False
            
        # Preprocess the path
        self.processed_path = self._preprocess_path(path)
        
        # Calculate path properties
        self.path_curvatures = self._calculate_path_curvatures(self.processed_path)
        self.current_path_index = 0
        
        return True

    def _preprocess_path(self, path):
        """
        Comprehensive path preprocessing pipeline
        """
        # 1. Remove redundant points
        path = self._remove_redundant_points(path)
        
        # 2. Smooth the path using B-splines
        path = self._smooth_path(path)
        
        # 3. Resample path at desired resolution
        path = self._resample_path(path)
        
        # 4. Ensure path doesn't exceed maximum points
        if len(path) > self.max_path_points:
            path = self._downsample_path(path)
            
        return path

    def _remove_redundant_points(self, path):
        """Remove points that are too close together"""
        if len(path) < 2:
            return path
            
        filtered_path = [path[0]]
        for point in path[1:]:
            if hypot(point[0] - filtered_path[-1][0],
                    point[1] - filtered_path[-1][1]) > self.pruning_distance:
                filtered_path.append(point)
                
        return np.array(filtered_path)

    def _smooth_path(self, path):
        """Smooth path using B-splines"""
        if len(path) < 3:
            return path
            
        try:
            # Fit B-spline to path
            tck, u = splprep([path[:, 0], path[:, 1]], s=self.smoothing_factor, k=3)
            
            # Generate new points along spline
            u_new = np.linspace(0, 1, len(path) * 2)
            smoothed_path = np.column_stack(splev(u_new, tck))
            
            return smoothed_path
        except Exception:
            # Fallback if spline fitting fails
            return path

    def _resample_path(self, path):
        """Resample path at regular intervals"""
        resampled_path = []
        cumulative_distance = 0.0
        current_point = path[0]
        resampled_path.append(current_point)
        
        for point in path[1:]:
            distance = hypot(point[0] - current_point[0],
                           point[1] - current_point[1])
            
            while cumulative_distance + distance > self.path_resolution:
                # Interpolate new point
                ratio = (self.path_resolution - cumulative_distance) / distance
                new_point = [
                    current_point[0] + ratio * (point[0] - current_point[0]),
                    current_point[1] + ratio * (point[1] - current_point[1])
                ]
                resampled_path.append(new_point)
                
                # Update for next iteration
                current_point = new_point
                distance = hypot(point[0] - current_point[0],
                               point[1] - current_point[1])
                cumulative_distance = 0.0
                
            cumulative_distance += distance
            current_point = point
            
        # Add final point
        resampled_path.append(path[-1])
        return np.array(resampled_path)

    def _downsample_path(self, path):
        """Downsample path to maximum allowed points"""
        if len(path) <= self.max_path_points:
            return path
            
        indices = np.linspace(0, len(path)-1, self.max_path_points, dtype=int)
        return path[indices]

    def _calculate_path_curvatures(self, path):
        """Calculate curvature at each path point"""
        if len(path) < 3:
            return np.zeros(len(path))
            
        curvatures = np.zeros(len(path))
        
        for i in range(1, len(path)-1):
            curvatures[i] = self._estimate_curvature(
                path[i-1],
                path[i],
                path[i+1]
            )
            
        # Set end points curvature same as neighbors
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        
        return curvatures

    def _estimate_curvature(self, p1, p2, p3):
        """Estimate curvature using three points"""
        # Use Menger curvature formula
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

    def _update_current_path_index(self, robot_pose):
        """Update the current position along the path"""
        if self.processed_path is None:
            return
            
        # Find closest point on path
        distances = cdist(
            self.processed_path,
            [[robot_pose[0], robot_pose[1]]]
        ).flatten()
        
        closest_idx = np.argmin(distances)
        
        # Look ahead a few points to avoid backtracking
        self.current_path_index = min(
            closest_idx + self.look_ahead_points,
            len(self.processed_path) - 1
        )

    def compute_velocity(self, robot_pose, path, obstacles=None, current_velocity=None, dt=0.1):
        """
        Enhanced compute_velocity with path preprocessing
        """
        # Update or initialize processed path
        if self.processed_path is None:
            if not self.set_path(path):
                return 0.0, 0.0
                
        # Update current position along path
        self._update_current_path_index(robot_pose)
        
        # Get relevant path segment
        active_path = self.processed_path[self.current_path_index:]
        
        # Use existing velocity computation with processed path
        return super().compute_velocity(
            robot_pose,
            active_path,
            obstacles,
            current_velocity,
            dt
        )

    def _apply_path_scaling(self, linear_vel, robot_pose, path):
        """Enhanced path scaling using preprocessed curvature"""
        if self.path_curvatures is None:
            return super()._apply_path_scaling(linear_vel, robot_pose, path)
            
        # Use pre-calculated curvatures
        local_curvature = self.path_curvatures[self.current_path_index]
        
        # Scale velocity based on curvature
        curvature_scaling = 1.0 / (1.0 + self.curvature_scaling_factor * abs(local_curvature))
        
        return linear_vel * curvature_scaling

    def _regulate_velocity(self, linear_vel, angular_vel, robot_pose, path, obstacles, dt):
        """
        Advanced velocity regulation with obstacle avoidance
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

        # 3. Apply obstacle avoidance
        if obstacles is not None:
            linear_vel, angular_vel = self._apply_obstacle_avoidance(
                linear_vel,
                angular_vel,
                robot_pose,
                obstacles
            )

        # 4. Scale velocity when approaching goal
        linear_vel = self._apply_goal_approach_scaling(
            linear_vel,
            robot_pose,
            path
        )

        # 5. Ensure minimum velocity constraints
        linear_vel = max(linear_vel, self.min_linear_velocity)
        angular_vel = np.clip(
            angular_vel,
            -self.max_angular_velocity,
            self.max_angular_velocity
        )

        return linear_vel, angular_vel

    def _apply_obstacle_avoidance(self, linear_vel, angular_vel, robot_pose, obstacles):
        """Apply obstacle avoidance behavior"""
        if len(obstacles) == 0:
            return linear_vel, angular_vel

        # Calculate repulsive forces from obstacles
        total_force_x = 0
        total_force_y = 0
        robot_x, robot_y, robot_theta = robot_pose

        for obstacle in obstacles:
            # Calculate distance to obstacle
            dx = obstacle[0] - robot_x
            dy = obstacle[1] - robot_y
            distance = hypot(dx, dy)

            if distance < self.obstacle_influence_radius:
                # Calculate repulsive force
                force_magnitude = self._calculate_repulsive_force(distance)
                
                # Calculate force components
                force_x = -force_magnitude * dx / distance
                force_y = -force_magnitude * dy / distance
                
                total_force_x += force_x
                total_force_y += force_y

        # Convert forces to velocity adjustments
        if abs(total_force_x) > 0 or abs(total_force_y) > 0:
            # Calculate desired heading change
            desired_heading = atan2(total_force_y, total_force_x)
            heading_error = self._normalize_angle(desired_heading - robot_theta)
            
            # Adjust angular velocity based on heading error
            angular_vel += self.obstacle_weight * heading_error
            
            # Scale linear velocity based on obstacle proximity
            force_magnitude = hypot(total_force_x, total_force_y)
            velocity_scaling = 1.0 - min(force_magnitude, self.max_obstacle_influence)
            linear_vel *= velocity_scaling

        return linear_vel, angular_vel

    def _calculate_repulsive_force(self, distance):
        """Calculate repulsive force based on distance to obstacle"""
        if distance < self.safety_radius:
            return self.repulsive_field_strength
        elif distance < self.obstacle_influence_radius:
            # Exponential decay of force
            d_scaled = (distance - self.safety_radius) / (self.obstacle_influence_radius - self.safety_radius)
            return self.repulsive_field_strength * exp(-5 * d_scaled)
        return 0.0

    def _detect_collision_risk(self, robot_pose, obstacles):
        """Detect if there's a risk of collision"""
        if obstacles is None:
            return False

        robot_x, robot_y, _ = robot_pose
        for obstacle in obstacles:
            distance = hypot(obstacle[0] - robot_x, obstacle[1] - robot_y)
            if distance < self.safety_radius:
                return True
        return False

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

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
