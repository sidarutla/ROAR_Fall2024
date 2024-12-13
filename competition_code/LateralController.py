import numpy as np
import math
from typing import Tuple


def normalize_rad(rad: float):
    return rad % (2 * np.pi)


class LatController:
    def __init__(self):
        # Lookahead tuning
        self.use_velocity_scaled_lookahead = True
        self.lookahead_time = 1.5         # How many seconds to look ahead (higher = smoother but less responsive)
        self.min_lookahead_dist = 3.0     # Minimum lookahead in meters (too low = unstable)
        self.max_lookahead_dist = 15.0    # Maximum lookahead in meters (too high = cuts corners)
        self.base_lookahead_dist = 5.0    # Fixed lookahead if not using velocity scaling

        # Steering tuning
        self.use_regulated_linear_velocity_scaling = True
        self.regulated_linear_scaling_min_radius = 0.9  # Minimum turning radius for regulation
        self.steering_gain = 0.5          # Overall steering aggressiveness

    def calculate_curvature(self, waypoint_vector):
        """Calculate path curvature for speed regulation"""
        dist_squared = (waypoint_vector[0] * waypoint_vector[0]) + (waypoint_vector[1] * waypoint_vector[1])
        if dist_squared > 0.001:
            return 2.0 * waypoint_vector[1] / dist_squared
        return 0.0

    def get_lookahead_distance(self, speed_kmh):
        """Calculate adaptive lookahead distance based on velocity"""
        if self.use_velocity_scaled_lookahead:
            # Convert km/h to m/s
            speed_ms = speed_kmh / 3.6
            lookahead = speed_ms * self.lookahead_time
            return np.clip(lookahead, self.min_lookahead_dist, self.max_lookahead_dist)
        return self.base_lookahead_dist

    def run(self, vehicle_location, vehicle_rotation, next_waypoint, current_speed_kmh=0) -> Tuple[float, float]:
        """
        Calculates the steering command using regulated pure pursuit.
        
        Returns:
            Tuple[float, float]: (steering_command, path_curvature)
        """
        # Calculate vector to waypoint
        waypoint_vector = np.array(next_waypoint.location) - np.array(vehicle_location)
        
        # Get distance to waypoint
        distance_to_waypoint = np.linalg.norm(waypoint_vector)
        if distance_to_waypoint == 0:
            return 0, 0

        # Get adaptive lookahead distance
        lookahead = self.get_lookahead_distance(current_speed_kmh)
        
        # Normalize waypoint vector
        waypoint_vector_normalized = waypoint_vector / distance_to_waypoint

        # Calculate heading error
        alpha = normalize_rad(vehicle_rotation[2]) - normalize_rad(
            math.atan2(waypoint_vector_normalized[1], waypoint_vector_normalized[0])
        )

        # Calculate curvature
        curvature = self.calculate_curvature(waypoint_vector)
        
        # Apply curvature-based steering regulation
        if self.use_regulated_linear_velocity_scaling:
            steering_command = self.steering_gain * math.atan2(
                2.0 * lookahead * math.sin(alpha) / max(distance_to_waypoint, self.regulated_linear_scaling_min_radius), 
                1.0
            )
        else:
            steering_command = self.steering_gain * math.atan2(
                2.0 * lookahead * math.sin(alpha) / distance_to_waypoint, 
                1.0
            )

        return float(steering_command), float(curvature)
