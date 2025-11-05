"""
Mission Executor

Simplified test execution using ArduPilot missions instead of programmatic commands.
Much more reliable and easier to debug than manual arm/takeoff/land sequences.
"""

import time
import logging
from typing import Tuple, Dict
from pymavlink import mavutil
import numpy as np

from mission_loader import MissionLoader

logger = logging.getLogger(__name__)


class MissionExecutor:
    """
    Executes ArduPilot missions and collects telemetry.

    This is simpler and more reliable than programmatic test sequences
    because it uses ArduPilot's built-in mission logic.
    """

    def __init__(self, connection: mavutil.mavlink_connection):
        """
        Initialize mission executor.

        Args:
            connection: MAVLink connection to vehicle
        """
        self.connection = connection
        self.mission_loader = MissionLoader(connection)

    def run_mission(self, mission_file: str, timeout: float = 120.0) -> Tuple[bool, Dict]:
        """
        Load and execute a mission, collecting telemetry.

        Args:
            mission_file: Path to .waypoints mission file
            timeout: Maximum mission duration

        Returns:
            (success, telemetry_dict)
        """
        logger.info(f"Starting mission: {mission_file}")

        # Upload mission
        if not self.mission_loader.load_and_upload(mission_file):
            logger.error("Failed to upload mission")
            return False, {}

        # Verify mission uploaded
        count = self.mission_loader.verify_mission()
        if count <= 0:
            logger.error("Mission verification failed")
            return False, {}

        logger.info(f"Mission uploaded successfully ({count} waypoints)")

        # Wait for EKF to initialize
        logger.info("Waiting for EKF initialization...")
        if not self._wait_for_ekf(timeout=30):
            logger.error("EKF initialization failed")
            return False, {}

        # Arm vehicle
        logger.info("Arming vehicle...")
        if not self._arm_vehicle():
            logger.error("Failed to arm vehicle")
            return False, {}

        # Set AUTO mode to start mission
        logger.info("Setting AUTO mode to start mission...")
        if not self._set_mode("AUTO"):
            logger.error("Failed to set AUTO mode")
            self._disarm_vehicle()
            return False, {}

        # Monitor mission and collect telemetry
        logger.info("Mission running - collecting telemetry...")
        success, telemetry = self._monitor_mission(timeout)

        # Mission complete (or failed)
        if success:
            logger.info("✓ Mission completed successfully")
        else:
            logger.warning("Mission did not complete successfully")

        return success, telemetry

    def _wait_for_ekf(self, timeout: float = 30.0) -> bool:
        """Wait for EKF to initialize and provide position data."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Try to get position
            msg = self.connection.recv_match(
                type='GLOBAL_POSITION_INT',
                blocking=True,
                timeout=1
            )

            if msg:
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                alt = msg.relative_alt / 1000.0

                if lat != 0 and lon != 0:
                    logger.info(f"✓ EKF ready - Position: ({lat:.6f}, {lon:.6f}, {alt:.2f}m)")
                    return True

            elapsed = int(time.time() - start_time)
            if elapsed % 5 == 0 and elapsed > 0:
                logger.debug(f"Waiting for EKF... ({elapsed}s)")

        logger.error("EKF initialization timeout")
        return False

    def _arm_vehicle(self, timeout: float = 10.0) -> bool:
        """Arm the vehicle."""
        # Send arm command
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,  # confirmation
            1,  # arm
            0, 0, 0, 0, 0, 0
        )

        # Wait for armed confirmation
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if msg and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                logger.info("✓ Vehicle armed")
                return True

        logger.error("Arming timeout")
        return False

    def _disarm_vehicle(self):
        """Disarm the vehicle."""
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,  # confirmation
            0,  # disarm
            0, 0, 0, 0, 0, 0
        )
        logger.info("Vehicle disarmed")

    def _set_mode(self, mode_name: str, timeout: float = 5.0) -> bool:
        """Set vehicle mode."""
        # Get mode ID
        mode_mapping = self.connection.mode_mapping()
        if mode_name not in mode_mapping:
            logger.error(f"Unknown mode: {mode_name}")
            return False

        mode_id = mode_mapping[mode_name]

        # Send mode change
        self.connection.mav.set_mode_send(
            self.connection.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )

        # Wait for confirmation
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if msg and msg.custom_mode == mode_id:
                logger.info(f"✓ Mode set to {mode_name}")
                return True

        logger.warning(f"Mode change to {mode_name} not confirmed")
        return True  # Continue anyway, mode might have changed

    def _monitor_mission(self, timeout: float) -> Tuple[bool, Dict]:
        """
        Monitor mission execution and collect telemetry.

        Returns:
            (success, telemetry_dict)
        """
        start_time = time.time()

        # Telemetry storage
        telemetry = {
            'time': [],
            'latitude': [],
            'longitude': [],
            'altitude': [],
            'roll': [],
            'pitch': [],
            'yaw': [],
            'roll_rate': [],
            'pitch_rate': [],
            'yaw_rate': [],
            'vx': [],
            'vy': [],
            'vz': [],
        }

        mission_complete = False
        last_waypoint = 0

        while time.time() - start_time < timeout:
            current_time = time.time() - start_time

            # Get position
            pos_msg = self.connection.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
            if pos_msg:
                telemetry['time'].append(current_time)
                telemetry['latitude'].append(pos_msg.lat / 1e7)
                telemetry['longitude'].append(pos_msg.lon / 1e7)
                telemetry['altitude'].append(pos_msg.relative_alt / 1000.0)
                telemetry['vx'].append(pos_msg.vx / 100.0)
                telemetry['vy'].append(pos_msg.vy / 100.0)
                telemetry['vz'].append(pos_msg.vz / 100.0)

            # Get attitude
            att_msg = self.connection.recv_match(type='ATTITUDE', blocking=False)
            if att_msg:
                telemetry['roll'].append(np.degrees(att_msg.roll))
                telemetry['pitch'].append(np.degrees(att_msg.pitch))
                telemetry['yaw'].append(np.degrees(att_msg.yaw))
                telemetry['roll_rate'].append(np.degrees(att_msg.rollspeed))
                telemetry['pitch_rate'].append(np.degrees(att_msg.pitchspeed))
                telemetry['yaw_rate'].append(np.degrees(att_msg.yawspeed))

            # Check mission progress
            mission_msg = self.connection.recv_match(type='MISSION_CURRENT', blocking=False)
            if mission_msg:
                if mission_msg.seq > last_waypoint:
                    logger.info(f"Reached waypoint {mission_msg.seq}")
                    last_waypoint = mission_msg.seq

            # Check if mission completed
            mission_item_reached = self.connection.recv_match(
                type='MISSION_ITEM_REACHED',
                blocking=False
            )
            if mission_item_reached:
                logger.debug(f"Waypoint {mission_item_reached.seq} reached")

            # Check if landed
            heartbeat = self.connection.recv_match(type='HEARTBEAT', blocking=False)
            if heartbeat:
                # Check if disarmed (mission complete and landed)
                if not (heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                    logger.info("Vehicle disarmed - mission complete")
                    mission_complete = True
                    break

            # Check for crash (very low altitude while armed)
            if telemetry['altitude'] and telemetry['altitude'][-1] < -0.5:
                logger.error("Crash detected (negative altitude)")
                break

            time.sleep(0.05)  # 20Hz telemetry collection

        # Convert lists to numpy arrays
        for key in telemetry:
            telemetry[key] = np.array(telemetry[key])

        # Add metrics for performance evaluator
        if len(telemetry['time']) > 0:
            telemetry['metrics'] = {
                'duration': telemetry['time'][-1] if len(telemetry['time']) > 0 else 0,
                'max_altitude': np.max(telemetry['altitude']) if len(telemetry['altitude']) > 0 else 0,
                'max_roll': np.max(np.abs(telemetry['roll'])) if len(telemetry['roll']) > 0 else 0,
                'max_pitch': np.max(np.abs(telemetry['pitch'])) if len(telemetry['pitch']) > 0 else 0,
                'crashed': not mission_complete,
            }

        return mission_complete, telemetry


# Convenience function
def run_mission_test(connection: mavutil.mavlink_connection,
                     mission_file: str,
                     timeout: float = 120.0) -> Tuple[bool, Dict]:
    """
    Convenience function to run a mission test.

    Args:
        connection: MAVLink connection
        mission_file: Path to mission file
        timeout: Timeout in seconds

    Returns:
        (success, telemetry)
    """
    executor = MissionExecutor(connection)
    return executor.run_mission(mission_file, timeout)
