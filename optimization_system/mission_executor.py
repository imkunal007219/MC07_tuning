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

from .mission_loader import MissionLoader
from .early_crash_detection import EarlyCrashDetector
from .config import EARLY_CRASH_DETECTION_CONFIG

logger = logging.getLogger(__name__)


class MissionExecutor:
    """
    Executes ArduPilot missions and collects telemetry.

    This is simpler and more reliable than programmatic test sequences
    because it uses ArduPilot's built-in mission logic.
    """

    def __init__(self, connection: mavutil.mavlink_connection,
                 enable_early_crash_detection: bool = None):
        """
        Initialize mission executor.

        Args:
            connection: MAVLink connection to vehicle
            enable_early_crash_detection: Enable early crash detection (default: from config)
        """
        self.connection = connection
        self.mission_loader = MissionLoader(connection)

        # Initialize early crash detection
        if enable_early_crash_detection is None:
            enable_early_crash_detection = EARLY_CRASH_DETECTION_CONFIG.get('enabled', True)

        self.early_crash_detection_enabled = enable_early_crash_detection
        if self.early_crash_detection_enabled:
            self.crash_detector = EarlyCrashDetector(
                check_interval=EARLY_CRASH_DETECTION_CONFIG.get('check_interval', 0.5),
                min_samples=EARLY_CRASH_DETECTION_CONFIG.get('min_samples', 50),
                sensitivity=EARLY_CRASH_DETECTION_CONFIG.get('sensitivity', 'medium')
            )
            logger.info("Early crash detection enabled (saves ~75% crash testing time)")
        else:
            self.crash_detector = None
            logger.info("Early crash detection disabled")

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

        # Step 3: Wait for all sensors to get ready (increased timeout for fresh SITL)
        logger.info("Waiting 30 seconds for sensors to initialize...")
        if not self._wait_for_sensors_ready(timeout=30):
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
            logger.info("✓✓✓ MISSION COMPLETED SUCCESSFULLY! ✓✓✓")
            if 'metrics' in telemetry:
                logger.info(f"Flight duration: {telemetry['metrics']['duration']:.1f}s")
                logger.info(f"Max altitude: {telemetry['metrics']['max_altitude']:.2f}m")
        else:
            logger.error("✗✗✗ MISSION FAILED ✗✗✗")
            if 'metrics' in telemetry and telemetry['metrics'].get('crashed', False):
                logger.error("Reason: Vehicle crashed")
            else:
                logger.error("Reason: Mission timeout (vehicle did not land/disarm within timeout)")

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

        logger.debug("ARM command sent, waiting for response...")

        # Wait for armed confirmation or command acknowledgement
        start_time = time.time()
        arm_ack_received = False

        while time.time() - start_time < timeout:
            # Check all message types
            msg = self.connection.recv_match(blocking=True, timeout=1)

            if msg is None:
                continue

            msg_type = msg.get_type()

            # Check for command acknowledgement
            if msg_type == 'COMMAND_ACK':
                if msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    arm_ack_received = True
                    if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        logger.debug("ARM command accepted by vehicle")
                    else:
                        # Arm command was rejected
                        result_str = {
                            mavutil.mavlink.MAV_RESULT_DENIED: "DENIED",
                            mavutil.mavlink.MAV_RESULT_UNSUPPORTED: "UNSUPPORTED",
                            mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED: "TEMPORARILY_REJECTED",
                            mavutil.mavlink.MAV_RESULT_FAILED: "FAILED"
                        }.get(msg.result, f"UNKNOWN({msg.result})")
                        logger.error(f"ARM command rejected: {result_str}")
                        logger.error("Check MAVProxy console for pre-arm check failures")
                        return False

            # Check if vehicle actually armed
            elif msg_type == 'HEARTBEAT':
                if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    logger.info("✓ Vehicle armed")
                    return True

        # Timeout - provide helpful error message
        if arm_ack_received:
            logger.error("Arming timeout - command was accepted but vehicle did not arm")
            logger.error("Vehicle may still be completing pre-arm checks")
        else:
            logger.error("Arming timeout - no response from vehicle")
            logger.error("Check SITL is running and connection is stable")

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
            if msg:
                # Handle both string and bytes (pymavlink version differences)
                param_id = msg.param_id
                if isinstance(param_id, bytes):
                    param_id = param_id.decode('utf-8')
                param_id = param_id.strip('\x00')

                if param_id == param_name:
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

    def _process_telemetry_metrics(self, telemetry: Dict, mission_complete: bool) -> Dict:
        """Process raw telemetry data into numpy arrays and calculate metrics."""
        # Convert lists to numpy arrays
        for key in telemetry:
            if isinstance(telemetry[key], list):
                telemetry[key] = np.array(telemetry[key])

        # Add metrics for performance evaluator
        if len(telemetry.get('time', [])) > 0:
            telemetry['metrics'] = {
                'duration': telemetry['time'][-1] if len(telemetry['time']) > 0 else 0,
                'max_altitude': np.max(telemetry['altitude']) if len(telemetry['altitude']) > 0 else 0,
                'max_roll': np.max(np.abs(telemetry['roll'])) if len(telemetry['roll']) > 0 else 0,
                'max_pitch': np.max(np.abs(telemetry['pitch'])) if len(telemetry['pitch']) > 0 else 0,
                'crashed': not mission_complete,
            }
        else:
            # No telemetry data collected
            telemetry['metrics'] = {
                'duration': 0,
                'max_altitude': 0,
                'max_roll': 0,
                'max_pitch': 0,
                'crashed': not mission_complete,
            }

        return telemetry

    def _prepare_telemetry_for_crash_detection(self, telemetry: Dict) -> list:
        """
        Convert telemetry dict to format expected by crash detector.

        Args:
            telemetry: Raw telemetry dict with lists

        Returns:
            List of dicts with time-series data for crash detection
        """
        # Convert to list of dict entries (one per timestep)
        telemetry_data = []

        # Find the minimum length across all arrays (they might not be perfectly aligned)
        min_len = min(
            len(telemetry.get('time', [])),
            len(telemetry.get('roll', [])),
            len(telemetry.get('pitch', [])),
            len(telemetry.get('altitude', []))
        )

        for i in range(min_len):
            data_point = {
                'time': telemetry['time'][i] if i < len(telemetry['time']) else 0,
                'roll': telemetry['roll'][i] if i < len(telemetry['roll']) else 0,
                'pitch': telemetry['pitch'][i] if i < len(telemetry['pitch']) else 0,
                'yaw': telemetry['yaw'][i] if i < len(telemetry['yaw']) else 0,
                'altitude': telemetry['altitude'][i] if i < len(telemetry['altitude']) else 0,
                'roll_rate': telemetry['roll_rate'][i] if i < len(telemetry['roll_rate']) else 0,
                'pitch_rate': telemetry['pitch_rate'][i] if i < len(telemetry['pitch_rate']) else 0,
                'yaw_rate': telemetry['yaw_rate'][i] if i < len(telemetry['yaw_rate']) else 0,
            }
            telemetry_data.append(data_point)

        return telemetry_data

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
        is_armed = True  # Track arming state
        last_heartbeat_check = time.time()
        last_crash_check_time = 0.0  # Track when we last checked for crashes

        while time.time() - start_time < timeout:
            current_time = time.time() - start_time

            # Early crash detection (check periodically)
            if (self.early_crash_detection_enabled and
                self.crash_detector is not None and
                current_time - last_crash_check_time >= self.crash_detector.check_interval):

                # Only check if we have enough data
                if len(telemetry['time']) >= self.crash_detector.min_samples:
                    # Prepare data for crash detector
                    telemetry_data = self._prepare_telemetry_for_crash_detection(telemetry)

                    # Check for impending crash
                    is_unstable, reason = self.crash_detector.check_stability(telemetry_data)

                    if is_unstable:
                        logger.warning(f"⚠ Early crash detected: {reason}")
                        logger.warning(f"Aborting mission early (saved ~{timeout - current_time:.1f}s)")

                        # Mark as crashed and return early
                        telemetry = self._process_telemetry_metrics(telemetry, mission_complete=False)
                        telemetry['metrics']['crashed'] = True
                        telemetry['metrics']['crash_reason'] = f"Early detection: {reason}"
                        telemetry['metrics']['early_abort_time'] = current_time
                        return False, telemetry

                    last_crash_check_time = current_time

            # Get all available messages in queue
            while True:
                msg = self.connection.recv_match(blocking=False)
                if msg is None:
                    break

                msg_type = msg.get_type()

                # Process different message types
                if msg_type == 'GLOBAL_POSITION_INT':
                    telemetry['time'].append(current_time)
                    telemetry['latitude'].append(msg.lat / 1e7)
                    telemetry['longitude'].append(msg.lon / 1e7)
                    telemetry['altitude'].append(msg.relative_alt / 1000.0)
                    telemetry['vx'].append(msg.vx / 100.0)
                    telemetry['vy'].append(msg.vy / 100.0)
                    telemetry['vz'].append(msg.vz / 100.0)

                elif msg_type == 'ATTITUDE':
                    telemetry['roll'].append(np.degrees(msg.roll))
                    telemetry['pitch'].append(np.degrees(msg.pitch))
                    telemetry['yaw'].append(np.degrees(msg.yaw))
                    telemetry['roll_rate'].append(np.degrees(msg.rollspeed))
                    telemetry['pitch_rate'].append(np.degrees(msg.pitchspeed))
                    telemetry['yaw_rate'].append(np.degrees(msg.yawspeed))

                elif msg_type == 'MISSION_CURRENT':
                    if msg.seq > last_waypoint:
                        logger.info(f"Reached waypoint {msg.seq}")
                        last_waypoint = msg.seq

                elif msg_type == 'MISSION_ITEM_REACHED':
                    logger.debug(f"Waypoint {msg.seq} reached")

                elif msg_type == 'HEARTBEAT':
                    # Check arming status
                    was_armed = is_armed
                    is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

                    # Detect disarm event
                    if was_armed and not is_armed:
                        logger.info("Vehicle disarmed - mission complete")
                        mission_complete = True
                        # Give it a bit more time to collect final telemetry
                        time.sleep(1.0)
                        # Process telemetry before returning
                        telemetry = self._process_telemetry_metrics(telemetry, mission_complete)
                        return True, telemetry

            # Check for crash (very low altitude while armed)
            if is_armed and telemetry['altitude'] and telemetry['altitude'][-1] < -0.5:
                logger.error("Crash detected (negative altitude)")
                break

            time.sleep(0.05)  # 20Hz telemetry collection

        # Process telemetry before returning
        telemetry = self._process_telemetry_metrics(telemetry, mission_complete)

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
