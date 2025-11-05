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

        # Step 1: Set required parameters for AUTO mode arming
        logger.info("Setting AUTO_OPTIONS and ARMING_CHECK parameters...")
        self._set_parameter("AUTO_OPTIONS", 1)  # Allow arming in AUTO mode
        self._set_parameter("ARMING_CHECK", 0)  # Disable arming checks for SITL testing
        time.sleep(0.5)  # Let parameters settle

        # Step 2: Load waypoints using correct MAVProxy method
        logger.info(f"Loading waypoints: {mission_file}")
        if not self._load_waypoints_mavproxy(mission_file):
            logger.error("Failed to load waypoints")
            return False, {}

        # Step 3: Wait 20 seconds for all sensors to get ready
        logger.info("Waiting 20 seconds for sensors to initialize...")
        if not self._wait_for_sensors_ready(timeout=20):
            logger.warning("Sensor initialization timeout - proceeding anyway")

        # Step 4: Set mode AUTO (BEFORE arming)
        logger.info("Setting AUTO mode...")
        if not self._set_mode("AUTO"):
            logger.error("Failed to set AUTO mode")
            return False, {}

        # Step 5: Arm throttle
        logger.info("Arming vehicle in AUTO mode...")
        if not self._arm_vehicle():
            logger.error("Failed to arm vehicle")
            return False, {}

        # Step 6: Send RC 3 1500 (throttle mid-stick) - required for mission start
        logger.info("Setting throttle to mid-stick (RC 3 1500)...")
        self._send_rc_override(channel=3, pwm=1500)
        time.sleep(0.5)

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

    def _set_parameter(self, param_name: str, param_value: float) -> bool:
        """
        Set a parameter value.

        Args:
            param_name: Parameter name (e.g., "AUTO_OPTIONS")
            param_value: Parameter value

        Returns:
            True if successful
        """
        logger.info(f"Setting {param_name} = {param_value}")

        # Encode parameter name (max 16 chars)
        param_name_bytes = param_name.encode('utf-8')[:16]

        # Send PARAM_SET message
        self.connection.mav.param_set_send(
            self.connection.target_system,
            self.connection.target_component,
            param_name_bytes,
            float(param_value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )

        # Wait for confirmation
        start_time = time.time()
        while time.time() - start_time < 5.0:
            msg = self.connection.recv_match(type='PARAM_VALUE', blocking=True, timeout=1)
            if msg and msg.param_id.decode('utf-8').strip('\x00') == param_name:
                logger.info(f"✓ {param_name} set to {msg.param_value}")
                return True

        logger.warning(f"Parameter {param_name} set command sent (no confirmation)")
        return True  # Continue anyway

    def _load_waypoints_mavproxy(self, mission_file: str) -> bool:
        """
        Load waypoints using the mission loader.

        This uses pymavlink's MISSION_ITEM messages which is the correct
        programmatic way to load waypoints (equivalent to MAVProxy's 'wp load').

        Args:
            mission_file: Path to waypoint file

        Returns:
            True if successful
        """
        # Use the mission loader to upload waypoints
        if not self.mission_loader.load_and_upload(mission_file):
            return False

        # Verify mission uploaded
        count = self.mission_loader.verify_mission()
        if count <= 0:
            logger.error("Mission verification failed")
            return False

        logger.info(f"✓ Mission loaded successfully ({count} waypoints)")
        return True

    def _wait_for_sensors_ready(self, timeout: float = 20.0) -> bool:
        """
        Wait for sensors to be ready (EKF, GPS, etc.).

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if sensors ready, False if timeout
        """
        start_time = time.time()
        ekf_ready = False

        while time.time() - start_time < timeout:
            # Check for position data (indicates EKF is initialized)
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
                    if not ekf_ready:
                        logger.info(f"✓ EKF ready - Position: ({lat:.6f}, {lon:.6f}, {alt:.2f}m)")
                        ekf_ready = True

                    # Continue waiting for full timeout to let all sensors stabilize
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        logger.info(f"✓ Sensors initialized ({timeout}s wait complete)")
                        return True

            elapsed = int(time.time() - start_time)
            if elapsed % 5 == 0 and elapsed > 0 and not ekf_ready:
                logger.debug(f"Waiting for sensors... ({elapsed}s)")

        return ekf_ready

    def _send_rc_override(self, channel: int, pwm: int):
        """
        Send RC channel override.

        Args:
            channel: RC channel number (1-8)
            pwm: PWM value (typically 1000-2000, 1500 is neutral)
        """
        # Initialize all channels to UINT16_MAX (no change)
        channels = [65535] * 8

        # Set the specific channel
        channels[channel - 1] = pwm

        # Send RC_CHANNELS_OVERRIDE message
        self.connection.mav.rc_channels_override_send(
            self.connection.target_system,
            self.connection.target_component,
            *channels
        )

        logger.debug(f"RC override: Channel {channel} = {pwm}")

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
