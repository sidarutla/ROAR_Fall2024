#!/usr/bin/env python3

import numpy as np
from math import atan2, hypot, sin, cos, pi
from nav2_core.controller import Controller
from nav2_core.exceptions import ControllerException
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav2_costmap_2d.costmap_2d import Costmap2D
from tf2_ros.buffer import Buffer
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

class RegulatedPurePursuitController(Controller):
    def __init__(self):
        self.name = 'RegulatedPurePursuitController'
        self.costmap = None
        self.tf = None
        self.node = None
        self.parameters = {}
        self.global_path = None
        self.robot_pose = None
        self.transform_tolerance = None
        self.processed_path = None
        self.path_curvatures = None
        self.current_path_index = 0

    def configure(self, node: Node, name: str, tf: Buffer, costmap: Costmap2D) -> bool:
        """Configure the controller with ROS parameters"""
        self.node = node
        self.tf = tf
        self.costmap = costmap
        self.name = name

        self.get_parameters()
        self.setup_publishers()
        
        return True

    def cleanup(self) -> None:
        """Clean up controller resources"""
        self.global_path = None
        self.processed_path = None

    def activate(self) -> None:
        """Activate the controller"""
        self.global_path = None
        self.processed_path = None

    def deactivate(self) -> None:
        """Deactivate the controller"""
        self.global_path = None
        self.processed_path = None

    def get_parameters(self) -> None:
        """Get parameters from ROS parameter server"""
        self.parameters = {
            'lookahead_dist': self.node.declare_parameter('lookahead_dist', 0.5).value,
            'min_lookahead_dist': self.node.declare_parameter('min_lookahead_dist', 0.3).value,
            'max_lookahead_dist': self.node.declare_parameter('max_lookahead_dist', 0.9).value,
            'max_linear_vel': self.node.declare_parameter('max_linear_vel', 0.5).value,
            'min_linear_vel': self.node.declare_parameter('min_linear_vel', 0.1).value,
            'max_angular_vel': self.node.declare_parameter('max_angular_vel', 1.0).value,
            'curvature_scaling_factor': self.node.declare_parameter('curvature_scaling_factor', 2.0).value,
            'smoothing_factor': self.node.declare_parameter('smoothing_factor', 0.1).value,
            'path_resolution': self.node.declare_parameter('path_resolution', 0.1).value,
            'transform_tolerance': self.node.declare_parameter('transform_tolerance', 0.1).value,
            'use_velocity_scaled_lookahead_dist': self.node.declare_parameter('use_velocity_scaled_lookahead_dist', True).value,
            'min_approach_linear_velocity': self.node.declare_parameter('min_approach_linear_velocity', 0.05).value,
            'approach_velocity_scaling_dist': self.node.declare_parameter('approach_velocity_scaling_dist', 0.6).value,
            'max_allowed_time_to_collision': self.node.declare_parameter('max_allowed_time_to_collision', 1.0).value,
            'use_collision_detection': self.node.declare_parameter('use_collision_detection', True).value,
        }

    def setPath(self, path: Path) -> bool:
        """Set and process the global path"""
        if not path or len(path.poses) < 2:
            return False

        self.global_path = path
        self.processed_path = self._preprocess_path(
            [(pose.pose.position.x, pose.pose.position.y) for pose in path.poses]
        )
        return True

    def computeVelocityCommands(self, 
                              pose: PoseStamped,
                              velocity: Twist) -> Twist:
        """Compute velocity commands to follow path"""
        if not self.global_path:
            raise ControllerException('Global path not set!')

        self.robot_pose = pose

        # Transform global path to robot frame
        transformed_path = self._transform_global_path()
        if not transformed_path:
            raise ControllerException('Failed to transform global path')

        # Find lookahead point
        lookahead_point = self._find_lookahead_point(transformed_path)
        if lookahead_point is None:
            raise ControllerException('Failed to find lookahead point')

        # Compute velocities
        cmd_vel = self._compute_velocity_commands(lookahead_point, velocity)

        # Check for collisions if enabled
        if self.parameters['use_collision_detection']:
            if self._check_collision(cmd_vel):
                self.node.get_logger().warn('Possible collision detected!')
                return self._create_zero_velocity()

        return cmd_vel

    def _transform_global_path(self) -> list:
        """Transform global path to robot frame"""
        try:
            transform = self.tf.lookup_transform(
                'base_link',
                self.global_path.header.frame_id,
                rclpy.time.Time(),
                Duration(seconds=self.parameters['transform_tolerance'])
            )
            
            # Transform path points
            transformed_path = []
            for pose in self.global_path.poses:
                transformed_pose = self.tf.transform(pose, 'base_link')
                transformed_path.append(transformed_pose)
            
            return transformed_path
        except Exception as e:
            self.node.get_logger().error(f'Transform failed: {str(e)}')
            return None

    def _check_collision(self, cmd_vel: Twist) -> bool:
        """Check for potential collisions"""
        if not self.costmap:
            return False

        # Project robot's path forward
        num_points = 10
        dt = self.parameters['max_allowed_time_to_collision'] / num_points
        
        x, y = 0.0, 0.0  # Robot's current position in base_link frame
        theta = 0.0
        
        for _ in range(num_points):
            # Update position
            x += cmd_vel.linear.x * cos(theta) * dt
            y += cmd_vel.linear.x * sin(theta) * dt
            theta += cmd_vel.angular.z * dt
            
            # Check cost at projected point
            cost = self.costmap.getCost(int(x), int(y))
            if cost >= 253:  # Lethal cost threshold
                return True
                
        return False

    def _create_zero_velocity(self) -> Twist:
        """Create a zero velocity command"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        return cmd_vel

    # ... (keeping previous path processing and velocity computation methods) 