#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from nav2_core.controller import Controller
from nav2_core.exceptions import ControllerException

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.msg import VoxelGrid
from rclpy.node import Node
from nav2_core.controller import Controller


class RegulatedPurePursuitController(Controller):
    def __init__(self):
        self.name = 'RegulatedPurePursuitController'
        super().__init__()
        self.params = {
            'desired_linear_vel': 0.5,
            'lookahead_dist': 0.6,
            'min_lookahead_dist': 0.3,
            'max_lookahead_dist': 0.9,
            'use_velocity_scaled_lookahead_dist': True,
            'use_collision_detection': True,
            'transform_tolerance': 0.1,
            'regulated_linear_scaling_min_radius': 0.9,
            'regulated_linear_scaling_min_speed': 0.25,
        }
        
    def configure(self, node, name, tf, costmap):
        """Configure the controller with ROS parameters"""
        self.node = node
        self.tf = tf
        self.costmap = costmap
        self.name = name

        # Get parameters from ROS param server
        self.get_parameters()
        return True

    def get_parameters(self):
        """Get parameters from ROS parameter server"""
        self.lookahead_distance = self.node.get_parameter('lookahead_dist').value
        self.max_linear_velocity = self.node.get_parameter('max_linear_vel').value
        self.min_linear_velocity = self.node.get_parameter('min_linear_vel').value
        self.max_angular_velocity = self.node.get_parameter('max_angular_vel').value
        # ... (add other parameters)

    def setPath(self, path):
        """ROS2 interface for setting the path"""
        return self.set_path(path.poses)

    def computeVelocityCommands(self, pose, velocity):
        """ROS2 interface for computing velocity commands"""
        try:
            if self.processed_path is None:
                raise ControllerException('Path not set!')

            cmd_vel = self.compute_velocity(
                self._pose_to_array(pose),
                self.processed_path,
                None,  # obstacles
                self._twist_to_array(velocity)
            )
            
            return self._create_twist_message(cmd_vel)
        except Exception as e:
            raise ControllerException(str(e))

    # Helper methods for ROS message conversion
    def _pose_to_array(self, pose):
        """Convert ROS pose to [x, y, theta]"""
        # Implementation here

    def _twist_to_array(self, twist):
        """Convert ROS twist to [v, w]"""
        # Implementation here

    def _create_twist_message(self, velocities):
        """Convert [v, w] to ROS twist message"""
        # Implementation here
        
    def findLookaheadPoint(self, pose):
        """Find lookahead point on path"""
        # Calculate lookahead distance
        lookahead_dist = self.params['lookahead_dist']
        if self.params['use_velocity_scaled_lookahead_dist']:
            lookahead_dist = min(
                max(
                    pose.velocity * self.params['lookahead_dist'],
                    self.params['min_lookahead_dist']
                ),
                self.params['max_lookahead_dist']
            )
            
        # Find closest point on path at lookahead distance
        closest_point = None
        min_dist = float('inf')
        
        for point in self.global_path.poses:
            dist = self.euclidean_distance(pose.position, point.pose.position)
            if abs(dist - lookahead_dist) < min_dist:
                min_dist = abs(dist - lookahead_dist)
                closest_point = point
                
        return closest_point
        
    def calculateVelocityCommands(self, pose, lookahead_point):
        """Calculate velocity commands to reach lookahead point"""
        # Calculate angle to lookahead point
        alpha = math.atan2(
            lookahead_point.pose.position.y - pose.position.y,
            lookahead_point.pose.position.x - pose.position.x
        ) - pose.orientation.z
        
        # Calculate curvature
        lookahead_dist = self.euclidean_distance(
            pose.position,
            lookahead_point.pose.position
        )
        curvature = 2.0 * math.sin(alpha) / lookahead_dist
        
        # Calculate velocities
        linear_vel = self.params['desired_linear_vel']
        angular_vel = linear_vel * curvature
        
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel
        
        return cmd_vel
        
    def applyVelocityScaling(self, cmd_vel, pose, lookahead_point):
        """Apply velocity scaling based on curvature"""
        # Calculate path curvature
        curvature = self.calculatePathCurvature(pose, lookahead_point)
        
        # Scale velocity based on curvature
        if abs(curvature) > 1.0 / self.params['regulated_linear_scaling_min_radius']:
            scaling = self.params['regulated_linear_scaling_min_radius'] * abs(curvature)
            cmd_vel.linear.x *= scaling
            
        # Ensure minimum speed
        if cmd_vel.linear.x < self.params['regulated_linear_scaling_min_speed']:
            cmd_vel.linear.x = self.params['regulated_linear_scaling_min_speed']
            
        return cmd_vel
        
    def checkCollision(self, pose, cmd_vel):
        """Check for potential collisions"""
        # Project robot position forward
        dt = 0.1  # Time step
        num_steps = 10
        
        for i in range(num_steps):
            # Calculate projected position
            projected_x = pose.position.x + cmd_vel.linear.x * dt * math.cos(pose.orientation.z)
            projected_y = pose.position.y + cmd_vel.linear.x * dt * math.sin(pose.orientation.z)
            projected_theta = pose.orientation.z + cmd_vel.angular.z * dt
            
            # Check if projected position is in collision
            if self.costmap.getCost(projected_x, projected_y) >= 253:  # Lethal cost
                return True
                
        return False
        
    @staticmethod
    def euclidean_distance(pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt(
            (pos1.x - pos2.x) ** 2 +
            (pos1.y - pos2.y) ** 2
        ) 