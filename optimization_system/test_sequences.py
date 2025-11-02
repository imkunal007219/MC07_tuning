"""
Automated Test Sequences for Drone Parameter Optimization
Implements comprehensive test missions for evaluating PID parameters
"""

import time
import math
import numpy as np
from pymavlink import mavutil
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Results from a test sequence"""
    success: bool
    duration: float
    max_error: float
    rms_error: float
    overshoot: float
    settling_time: float
    rise_time: float
    oscillations: int
    crashed: bool
    motor_saturation_count: int
    test_data: Dict


class TestSequence:
    """Base class for all test sequences"""

    def __init__(self, connection: mavutil.mavlink_connection, timeout: float = 60.0):
        self.connection = connection
        self.timeout = timeout
        self.start_time = None

    def _convert_to_tuple(self, result: TestResult) -> Tuple[bool, Dict]:
        """
        Convert TestResult to (success, telemetry) tuple format
        This format is expected by the optimizer and performance evaluator
        """
        # Convert test data to proper telemetry format
        telemetry = {}

        if 'positions' in result.test_data and result.test_data['positions']:
            positions = result.test_data['positions']
            # Extract time series data
            telemetry['time'] = np.array([p[0] for p in positions])
            telemetry['latitude'] = np.array([p[1] for p in positions]) if len(positions[0]) > 1 else None
            telemetry['longitude'] = np.array([p[2] for p in positions]) if len(positions[0]) > 2 else None
            telemetry['altitude'] = np.array([p[3] for p in positions]) if len(positions[0]) > 3 else None

        if 'attitudes' in result.test_data and result.test_data['attitudes']:
            attitudes = result.test_data['attitudes']
            if 'time' not in telemetry:
                telemetry['time'] = np.array([a[0] for a in attitudes])
            telemetry['attitude'] = np.array([[a[1], a[2], a[3]] for a in attitudes])  # roll, pitch, yaw
            telemetry['roll'] = np.array([a[1] for a in attitudes])
            telemetry['pitch'] = np.array([a[2] for a in attitudes])
            telemetry['yaw'] = np.array([a[3] for a in attitudes])

        if 'velocities' in result.test_data and result.test_data['velocities']:
            velocities = result.test_data['velocities']
            telemetry['velocity'] = np.array([[v[1], v[2], v[3]] for v in velocities])  # vx, vy, vz
            telemetry['vx'] = np.array([v[1] for v in velocities])
            telemetry['vy'] = np.array([v[2] for v in velocities])
            telemetry['vz'] = np.array([v[3] for v in velocities])

        if 'responses' in result.test_data:
            responses = result.test_data['responses']
            if 'time' not in telemetry:
                telemetry['time'] = np.array([r[0] for r in responses])
            telemetry['response_value'] = np.array([r[1] for r in responses])

        # Add target values if available
        if 'target_value' in result.test_data:
            target = result.test_data['target_value']
            telemetry['attitude_target'] = np.full_like(telemetry.get('attitude', np.array([0])), target)
            telemetry['altitude_target'] = np.full_like(telemetry.get('altitude', np.array([0])), target)

        # Add performance metrics
        telemetry['metrics'] = {
            'max_error': result.max_error,
            'rms_error': result.rms_error,
            'overshoot': result.overshoot,
            'settling_time': result.settling_time,
            'rise_time': result.rise_time,
            'oscillations': result.oscillations,
            'crashed': result.crashed,
            'duration': result.duration
        }

        # Add motor saturation data (placeholder)
        if 'time' in telemetry:
            telemetry['motor_outputs'] = np.ones((len(telemetry['time']), 4)) * 0.5  # Placeholder

        # Add rates (placeholder - should be collected from actual telemetry)
        if 'attitude' in telemetry:
            # Approximate rates from attitude differences
            if len(telemetry['attitude']) > 1:
                dt = np.diff(telemetry['time'])
                attitude_diff = np.diff(telemetry['attitude'], axis=0)
                rates = attitude_diff / dt[:, np.newaxis]
                # Pad to match length
                rates = np.vstack([rates, rates[-1]])
                telemetry['rates'] = rates
                telemetry['roll_rate'] = rates[:, 0]
                telemetry['pitch_rate'] = rates[:, 1]
                telemetry['yaw_rate'] = rates[:, 2]
            else:
                telemetry['rates'] = np.zeros_like(telemetry['attitude'])
                telemetry['roll_rate'] = np.zeros(len(telemetry['attitude']))
                telemetry['pitch_rate'] = np.zeros(len(telemetry['attitude']))
                telemetry['yaw_rate'] = np.zeros(len(telemetry['attitude']))

        # Add position targets (for hover test, target is initial position)
        if 'latitude' in telemetry and telemetry['latitude'] is not None:
            telemetry['position_target'] = np.array([
                [telemetry['latitude'][0], telemetry['longitude'][0], telemetry['altitude'][0]]
            ] * len(telemetry['time']))
            telemetry['position'] = np.column_stack([
                telemetry['latitude'], telemetry['longitude'], telemetry['altitude']
            ])

        return (not result.crashed and result.success, telemetry)

    def wait_heartbeat(self):
        """Wait for heartbeat from vehicle"""
        self.connection.wait_heartbeat()

    def get_position(self) -> Tuple[float, float, float]:
        """Get current position (lat, lon, alt)"""
        msg = self.connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
        if msg:
            return msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0
        return None, None, None

    def get_attitude(self) -> Tuple[float, float, float]:
        """Get current attitude (roll, pitch, yaw) in degrees"""
        msg = self.connection.recv_match(type='ATTITUDE', blocking=True, timeout=5)
        if msg:
            return (math.degrees(msg.roll),
                   math.degrees(msg.pitch),
                   math.degrees(msg.yaw))
        return None, None, None

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current velocity (vx, vy, vz) in m/s"""
        msg = self.connection.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=5)
        if msg:
            return msg.vx, msg.vy, msg.vz
        return None, None, None

    def arm_and_takeoff(self, target_altitude: float = 10.0) -> bool:
        """Arm and takeoff to target altitude"""
        logger.info(f"Arming and taking off to {target_altitude}m")

        # Set mode to GUIDED
        mode = 'GUIDED'
        mode_id = self.connection.mode_mapping()[mode]
        self.connection.mav.set_mode_send(
            self.connection.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)

        time.sleep(1)

        # Arm
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0)

        # Wait for arming
        time.sleep(2)

        # Takeoff
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, target_altitude)

        # Wait for altitude
        start_time = time.time()
        while time.time() - start_time < 30:
            _, _, alt = self.get_position()
            if alt and alt >= target_altitude * 0.95:
                logger.info(f"Reached target altitude: {alt}m")
                return True
            time.sleep(0.5)

        logger.error("Takeoff timeout")
        return False

    def land_and_disarm(self) -> bool:
        """Land and disarm"""
        logger.info("Landing")

        # Land command
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0)

        time.sleep(10)  # Wait for landing

        # Disarm
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0)

        return True


class HoverStabilityTest(TestSequence):
    """Test basic hover stability"""

    def __init__(self, connection, duration: float = 30.0, altitude: float = 10.0):
        super().__init__(connection, timeout=duration + 60)
        self.test_duration = duration
        self.altitude = altitude

    def run(self) -> Tuple[bool, Dict]:
        """Execute hover stability test and return (success, telemetry) tuple"""
        result = self._run_test()
        return self._convert_to_tuple(result)

    def _run_test(self) -> TestResult:
        """Execute hover stability test"""
        logger.info(f"Running hover stability test for {self.test_duration}s at {self.altitude}m")

        # Takeoff
        if not self.arm_and_takeoff(self.altitude):
            return TestResult(False, 0, 0, 0, 0, 0, 0, 0, True, 0, {})

        # Record data during hover
        start_time = time.time()
        positions = []
        attitudes = []
        velocities = []

        while time.time() - start_time < self.test_duration:
            lat, lon, alt = self.get_position()
            roll, pitch, yaw = self.get_attitude()
            vx, vy, vz = self.get_velocity()

            if alt is not None:
                positions.append((time.time() - start_time, lat, lon, alt))
                attitudes.append((time.time() - start_time, roll, pitch, yaw))
                velocities.append((time.time() - start_time, vx, vy, vz))

                # Check for crash (altitude too low)
                if alt < 1.0:
                    logger.error("Crash detected!")
                    return TestResult(False, time.time() - start_time, 0, 0, 0, 0, 0, 0, True, 0, {})

            time.sleep(0.1)

        # Calculate metrics
        duration = time.time() - start_time

        # Altitude error
        alt_errors = [abs(p[3] - self.altitude) for p in positions]
        max_alt_error = max(alt_errors) if alt_errors else 0
        rms_alt_error = np.sqrt(np.mean(np.array(alt_errors)**2)) if alt_errors else 0

        # Attitude oscillations (count zero crossings)
        roll_angles = [a[1] for a in attitudes]
        pitch_angles = [a[2] for a in attitudes]
        oscillations = self._count_oscillations(roll_angles) + self._count_oscillations(pitch_angles)

        # Land
        self.land_and_disarm()

        test_data = {
            'positions': positions,
            'attitudes': attitudes,
            'velocities': velocities,
            'altitude_errors': alt_errors
        }

        success = max_alt_error < 1.0 and oscillations < 10

        return TestResult(
            success=success,
            duration=duration,
            max_error=max_alt_error,
            rms_error=rms_alt_error,
            overshoot=0,
            settling_time=0,
            rise_time=0,
            oscillations=oscillations,
            crashed=False,
            motor_saturation_count=0,
            test_data=test_data
        )

    def _count_oscillations(self, signal: List[float]) -> int:
        """Count number of oscillations in signal"""
        if len(signal) < 2:
            return 0

        zero_crossings = 0
        mean = np.mean(signal)
        for i in range(1, len(signal)):
            if (signal[i-1] - mean) * (signal[i] - mean) < 0:
                zero_crossings += 1

        return zero_crossings // 2  # Full oscillation = 2 zero crossings


class StepResponseTest(TestSequence):
    """Test step response for attitude and altitude"""

    def __init__(self, connection, axis: str = 'roll', step_size: float = 10.0):
        super().__init__(connection, timeout=60)
        self.axis = axis  # 'roll', 'pitch', 'yaw', or 'altitude'
        self.step_size = step_size

    def run(self) -> TestResult:
        """Execute step response test"""
        logger.info(f"Running step response test: {self.axis} {self.step_size}")

        # Takeoff to 10m
        if not self.arm_and_takeoff(10.0):
            return TestResult(False, 0, 0, 0, 0, 0, 0, 0, True, 0, {})

        time.sleep(2)  # Stabilize

        # Record initial state
        start_time = time.time()
        responses = []

        if self.axis == 'altitude':
            target_alt = 10.0 + self.step_size
            initial_value = 10.0

            # Send position setpoint
            self.connection.mav.set_position_target_local_ned_send(
                0,  # time_boot_ms
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,  # type_mask (only positions enabled)
                0, 0, -target_alt,  # x, y, z
                0, 0, 0,  # vx, vy, vz
                0, 0, 0,  # afx, afy, afz
                0, 0)  # yaw, yaw_rate

            # Record response
            for _ in range(100):  # 10 seconds at 10Hz
                _, _, alt = self.get_position()
                if alt is not None:
                    responses.append((time.time() - start_time, alt))
                time.sleep(0.1)

        elif self.axis in ['roll', 'pitch']:
            initial_roll, initial_pitch, _ = self.get_attitude()
            initial_value = initial_roll if self.axis == 'roll' else initial_pitch

            # Send attitude setpoint
            if self.axis == 'roll':
                self.connection.mav.set_attitude_target_send(
                    0,  # time_boot_ms
                    self.connection.target_system,
                    self.connection.target_component,
                    0b00000000,  # type_mask
                    self._euler_to_quaternion(math.radians(self.step_size), 0, 0),
                    0, 0, 0,  # body roll rate, pitch rate, yaw rate
                    0.5)  # thrust
            else:  # pitch
                self.connection.mav.set_attitude_target_send(
                    0,
                    self.connection.target_system,
                    self.connection.target_component,
                    0b00000000,
                    self._euler_to_quaternion(0, math.radians(self.step_size), 0),
                    0, 0, 0,
                    0.5)

            # Record response
            for _ in range(100):
                roll, pitch, _ = self.get_attitude()
                value = roll if self.axis == 'roll' else pitch
                if value is not None:
                    responses.append((time.time() - start_time, value))
                time.sleep(0.1)

        # Calculate step response metrics
        times = [r[0] for r in responses]
        values = [r[1] for r in responses]

        if self.axis == 'altitude':
            target_value = 10.0 + self.step_size
        else:
            target_value = self.step_size

        rise_time = self._calculate_rise_time(times, values, initial_value, target_value)
        settling_time = self._calculate_settling_time(times, values, target_value)
        overshoot = self._calculate_overshoot(values, initial_value, target_value)

        errors = [abs(v - target_value) for v in values]
        max_error = max(errors) if errors else 0
        rms_error = np.sqrt(np.mean(np.array(errors)**2)) if errors else 0

        oscillations = self._count_oscillations(values, target_value)

        # Land
        self.land_and_disarm()

        test_data = {
            'responses': responses,
            'target_value': target_value,
            'initial_value': initial_value
        }

        success = overshoot < 20.0 and settling_time < 5.0 and oscillations < 5

        return TestResult(
            success=success,
            duration=time.time() - start_time,
            max_error=max_error,
            rms_error=rms_error,
            overshoot=overshoot,
            settling_time=settling_time,
            rise_time=rise_time,
            oscillations=oscillations,
            crashed=False,
            motor_saturation_count=0,
            test_data=test_data
        )

    def _euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion [w, x, y, z]"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [w, x, y, z]

    def _calculate_rise_time(self, times, values, initial, target):
        """Calculate 10%-90% rise time"""
        threshold_10 = initial + 0.1 * (target - initial)
        threshold_90 = initial + 0.9 * (target - initial)

        t_10 = None
        t_90 = None

        for t, v in zip(times, values):
            if t_10 is None and v >= threshold_10:
                t_10 = t
            if t_90 is None and v >= threshold_90:
                t_90 = t
                break

        if t_10 is not None and t_90 is not None:
            return t_90 - t_10
        return 0

    def _calculate_settling_time(self, times, values, target, tolerance=0.02):
        """Calculate settling time (2% tolerance)"""
        threshold = target * tolerance

        for i in range(len(values) - 1, -1, -1):
            if abs(values[i] - target) > threshold:
                return times[i] if i < len(times) else times[-1]

        return 0

    def _calculate_overshoot(self, values, initial, target):
        """Calculate percentage overshoot"""
        max_value = max(values)
        overshoot = ((max_value - target) / (target - initial)) * 100
        return max(0, overshoot)

    def _count_oscillations(self, signal, target):
        """Count oscillations around target"""
        if len(signal) < 2:
            return 0

        zero_crossings = 0
        for i in range(1, len(signal)):
            if (signal[i-1] - target) * (signal[i] - target) < 0:
                zero_crossings += 1

        return zero_crossings // 2


class TrajectoryTrackingTest(TestSequence):
    """Test trajectory tracking (square, circle, figure-8)"""

    def __init__(self, connection, trajectory_type: str = 'square', size: float = 20.0):
        super().__init__(connection, timeout=120)
        self.trajectory_type = trajectory_type  # 'square', 'circle', 'figure8'
        self.size = size

    def run(self) -> TestResult:
        """Execute trajectory tracking test"""
        logger.info(f"Running trajectory test: {self.trajectory_type}")

        # Takeoff
        if not self.arm_and_takeoff(10.0):
            return TestResult(False, 0, 0, 0, 0, 0, 0, 0, True, 0, {})

        time.sleep(2)

        # Get home position
        home_msg = self.connection.recv_match(type='HOME_POSITION', blocking=True, timeout=5)
        if not home_msg:
            logger.error("Could not get home position")
            return TestResult(False, 0, 0, 0, 0, 0, 0, 0, True, 0, {})

        home_lat = home_msg.latitude / 1e7
        home_lon = home_msg.longitude / 1e7

        # Generate waypoints
        waypoints = self._generate_waypoints(home_lat, home_lon)

        # Track trajectory
        start_time = time.time()
        actual_positions = []
        tracking_errors = []

        for wp in waypoints:
            target_lat, target_lon, target_alt = wp

            # Send waypoint
            self.connection.mav.mission_item_send(
                self.connection.target_system,
                self.connection.target_component,
                0,  # seq
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                2,  # current (guided mode waypoint)
                0,  # autocontinue
                0, 0, 0, 0,  # params 1-4
                target_lat, target_lon, target_alt)

            # Wait to reach waypoint
            while True:
                lat, lon, alt = self.get_position()
                if lat is None:
                    continue

                actual_positions.append((time.time() - start_time, lat, lon, alt))

                # Calculate distance to waypoint
                dist = self._haversine_distance(lat, lon, target_lat, target_lon)
                tracking_errors.append(dist)

                if dist < 2.0:  # Within 2m of waypoint
                    break

                if time.time() - start_time > self.timeout:
                    logger.error("Trajectory timeout")
                    break

                time.sleep(0.1)

        # Calculate metrics
        duration = time.time() - start_time
        max_error = max(tracking_errors) if tracking_errors else 0
        rms_error = np.sqrt(np.mean(np.array(tracking_errors)**2)) if tracking_errors else 0

        # Land
        self.land_and_disarm()

        test_data = {
            'waypoints': waypoints,
            'actual_positions': actual_positions,
            'tracking_errors': tracking_errors
        }

        success = max_error < 5.0 and rms_error < 2.0

        return TestResult(
            success=success,
            duration=duration,
            max_error=max_error,
            rms_error=rms_error,
            overshoot=0,
            settling_time=0,
            rise_time=0,
            oscillations=0,
            crashed=False,
            motor_saturation_count=0,
            test_data=test_data
        )

    def _generate_waypoints(self, home_lat, home_lon):
        """Generate waypoints for trajectory"""
        alt = 10.0

        if self.trajectory_type == 'square':
            # Square pattern
            half_size = self.size / 2
            return [
                (home_lat + half_size/111000, home_lon + half_size/111000, alt),
                (home_lat + half_size/111000, home_lon - half_size/111000, alt),
                (home_lat - half_size/111000, home_lon - half_size/111000, alt),
                (home_lat - half_size/111000, home_lon + half_size/111000, alt),
                (home_lat, home_lon, alt)
            ]

        elif self.trajectory_type == 'circle':
            # Circle pattern (8 points)
            waypoints = []
            for i in range(8):
                angle = 2 * math.pi * i / 8
                lat_offset = (self.size / 2) * math.cos(angle) / 111000
                lon_offset = (self.size / 2) * math.sin(angle) / 111000
                waypoints.append((home_lat + lat_offset, home_lon + lon_offset, alt))
            return waypoints

        elif self.trajectory_type == 'figure8':
            # Figure-8 pattern
            waypoints = []
            for i in range(16):
                t = 2 * math.pi * i / 16
                x = self.size * math.sin(t) / 111000
                y = self.size * math.sin(2 * t) / 2 / 111000
                waypoints.append((home_lat + x, home_lon + y, alt))
            return waypoints

        return [(home_lat, home_lon, alt)]

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS coordinates in meters"""
        R = 6371000  # Earth radius in meters

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c


class DisturbanceRejectionTest(TestSequence):
    """Test disturbance rejection with simulated wind"""

    def __init__(self, connection, wind_speed: float = 5.0):
        super().__init__(connection, timeout=60)
        self.wind_speed = wind_speed

    def run(self) -> TestResult:
        """Execute disturbance rejection test"""
        logger.info(f"Running disturbance rejection test with {self.wind_speed} m/s wind")

        # Set wind parameters in SITL
        self.connection.mav.param_set_send(
            self.connection.target_system,
            self.connection.target_component,
            b'SIM_WIND_SPD',
            self.wind_speed,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

        time.sleep(1)

        # Takeoff
        if not self.arm_and_takeoff(10.0):
            return TestResult(False, 0, 0, 0, 0, 0, 0, 0, True, 0, {})

        # Hover with wind disturbance
        start_time = time.time()
        positions = []
        attitudes = []

        for _ in range(300):  # 30 seconds at 10Hz
            lat, lon, alt = self.get_position()
            roll, pitch, yaw = self.get_attitude()

            if lat is not None:
                positions.append((time.time() - start_time, lat, lon, alt))
                attitudes.append((time.time() - start_time, roll, pitch, yaw))

            time.sleep(0.1)

        # Reset wind
        self.connection.mav.param_set_send(
            self.connection.target_system,
            self.connection.target_component,
            b'SIM_WIND_SPD',
            0.0,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

        # Calculate metrics
        alt_errors = [abs(p[3] - 10.0) for p in positions]
        max_error = max(alt_errors) if alt_errors else 0
        rms_error = np.sqrt(np.mean(np.array(alt_errors)**2)) if alt_errors else 0

        # Attitude deviations
        roll_angles = [abs(a[1]) for a in attitudes]
        pitch_angles = [abs(a[2]) for a in attitudes]
        max_attitude_error = max(max(roll_angles), max(pitch_angles))

        # Land
        self.land_and_disarm()

        test_data = {
            'positions': positions,
            'attitudes': attitudes,
            'max_attitude_error': max_attitude_error
        }

        success = max_error < 2.0 and max_attitude_error < 30.0

        return TestResult(
            success=success,
            duration=time.time() - start_time,
            max_error=max_error,
            rms_error=rms_error,
            overshoot=0,
            settling_time=0,
            rise_time=0,
            oscillations=0,
            crashed=False,
            motor_saturation_count=0,
            test_data=test_data
        )


class TestSequenceManager:
    """Manages and executes all test sequences"""

    def __init__(self, connection_string: str = 'tcp:127.0.0.1:5760'):
        self.connection_string = connection_string
        self.connection = None

    def connect(self):
        """Connect to SITL"""
        logger.info(f"Connecting to {self.connection_string}")
        self.connection = mavutil.mavlink_connection(self.connection_string)
        self.connection.wait_heartbeat()
        logger.info("Connected")

    def run_all_tests(self) -> Dict[str, TestResult]:
        """Run all test sequences"""
        results = {}

        # Hover stability test
        logger.info("=" * 60)
        logger.info("HOVER STABILITY TEST")
        logger.info("=" * 60)
        hover_test = HoverStabilityTest(self.connection, duration=30.0, altitude=10.0)
        results['hover_stability'] = hover_test.run()

        time.sleep(5)  # Pause between tests

        # Step response tests
        for axis in ['roll', 'pitch', 'altitude']:
            logger.info("=" * 60)
            logger.info(f"STEP RESPONSE TEST - {axis.upper()}")
            logger.info("=" * 60)
            step_size = 10.0 if axis == 'altitude' else 15.0
            step_test = StepResponseTest(self.connection, axis=axis, step_size=step_size)
            results[f'step_response_{axis}'] = step_test.run()
            time.sleep(5)

        # Trajectory tracking tests
        for traj_type in ['square', 'circle']:
            logger.info("=" * 60)
            logger.info(f"TRAJECTORY TEST - {traj_type.upper()}")
            logger.info("=" * 60)
            traj_test = TrajectoryTrackingTest(self.connection, trajectory_type=traj_type, size=20.0)
            results[f'trajectory_{traj_type}'] = traj_test.run()
            time.sleep(5)

        # Disturbance rejection test
        logger.info("=" * 60)
        logger.info("DISTURBANCE REJECTION TEST")
        logger.info("=" * 60)
        disturbance_test = DisturbanceRejectionTest(self.connection, wind_speed=5.0)
        results['disturbance_rejection'] = disturbance_test.run()

        return results

    def print_summary(self, results: Dict[str, TestResult]):
        """Print test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)

        for test_name, result in results.items():
            status = "PASS" if result.success else "FAIL"
            logger.info(f"{test_name}: {status}")
            logger.info(f"  Duration: {result.duration:.2f}s")
            logger.info(f"  Max Error: {result.max_error:.2f}")
            logger.info(f"  RMS Error: {result.rms_error:.2f}")
            if result.overshoot > 0:
                logger.info(f"  Overshoot: {result.overshoot:.1f}%")
            if result.settling_time > 0:
                logger.info(f"  Settling Time: {result.settling_time:.2f}s")
            if result.oscillations > 0:
                logger.info(f"  Oscillations: {result.oscillations}")
            logger.info("")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example usage
    manager = TestSequenceManager('tcp:127.0.0.1:5760')
    manager.connect()
    results = manager.run_all_tests()
    manager.print_summary(results)
