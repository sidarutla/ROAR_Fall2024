"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from collections import deque
import math
from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    #Takes in the current car location, the index of the waypoints list, and the waypoint list to get the closest waypoint
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 2:
            return i % len(waypoints)
    return current_idx

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints, 
        )

        # Creates the section dividers and sets the current zone to 0
        # Zone 0: Start - after turn 4
        # Zone 1: Turn 5 - Before turn 6
        # Zone 2: Turn6 - Turn 7
        # Zone 3: After turn 7 - Before turn 9 (Long 180 degree turn)
        # Zone 4: Turns 9 and 10 (Sharp S-turn after long straightaway)
        self.regions = [[740, 720], [100, 220], [-80, -160], [-345, 0], [-290, 400]] # (-290, 400) is the start of the track
        self.currentRegion = 0

    
    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints, 
        )
        # Generates the waypoint to follow based on the vehicle's speed
        waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + int(vehicle_velocity_norm / 2.75) + 4) % len(self.maneuverable_waypoints)]

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Proportional controller to steer the vehicle towards the target waypoint
        steer_control = (
            -8.0 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        # Calculates the distance to the final two turns of the track 

        distance = math.sqrt((self.regions[self.currentRegion % len(self.regions)][0] - waypoint_to_follow.location[0]) ** 2 + (self.regions[self.currentRegion % len(self.regions)][1] - waypoint_to_follow.location[1]) ** 2)

        # Calculates the appropriate throttle response based on the speed and angle to the next waypoint, as well as it 'sector'
        # FIXME: Add adaptive throttle and braking, as well as better zone management
        
        if distance < 10:
            self.currentRegion += 1

        normalizedRegion = self.currentRegion % len(self.regions)
        
        if normalizedRegion < 3:
            if (abs(delta_heading) > 0.01175 and vehicle_velocity_norm > 36.5):
                throttle = 1
                brake = 1
                reverse = 1
                handBrake = 1
            else:
                throttle = 0.75 + (0.7 / delta_heading - vehicle_velocity_norm) / 20
                brake = 0
                reverse = 0
                handBrake = 0
        elif normalizedRegion == 4:
            # Adjust these values to get braking performance needed
            if (abs(delta_heading) > 0.00008 and vehicle_velocity_norm > 22.5):
                throttle = 1
                brake = 1
                reverse = 1
                handBrake = 1
            else:
                throttle = 1
                brake = 0
                reverse = 0
                handBrake = 0
        else:
            if (abs(delta_heading) > 0.007 and vehicle_velocity_norm > 50):
                throttle = 1
                brake = 1
                reverse = 1
                handBrake = 1
            else:
                throttle = 0.75 + (1 / delta_heading - vehicle_velocity_norm) / 20
                brake = 0
                reverse = 0
                handBrake = 0

        gear = max(1, (int)((vehicle_velocity_norm) / 20))

        control = {
            "throttle": throttle,
            "steer": steer_control,
            "brake": brake,
            "hand_brake": handBrake,
            "reverse": reverse,
            "target_gear": gear
        }

        print(f"Current Speed: {vehicle_velocity_norm}\nBrake Value: {brake}\nCurrent region: {normalizedRegion}\nLap Number: {self.currentRegion // len(self.regions) + 1}\nDelta Heading: {delta_heading}")

        await self.vehicle.apply_action(control)
        return control