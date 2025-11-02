"""
Safety Constraints and Crash Detection System
Monitors flight parameters and detects unsafe conditions during optimization
"""

import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SafetyViolationType(Enum):
    """Types of safety violations"""
    CRASH = "crash"
    ALTITUDE_TOO_LOW = "altitude_too_low"
    ALTITUDE_TOO_HIGH = "altitude_too_high"
    EXCESSIVE_TILT = "excessive_tilt"
    EXCESSIVE_RATE = "excessive_rate"
    MOTOR_SATURATION = "motor_saturation"
    OSCILLATION = "oscillation"
    POSITION_ERROR = "position_error"
    VELOCITY_EXCESSIVE = "velocity_excessive"
    TIMEOUT = "timeout"
    GEOFENCE = "geofence_violation"
    LOW_BATTERY = "low_battery"
    INSTABILITY = "instability"


@dataclass
class SafetyLimits:
    """Safety constraint limits for drone operation"""

    # Altitude limits (meters)
    min_altitude: float = 0.5
    max_altitude: float = 100.0
    altitude_error_max: float = 10.0

    # Attitude limits (degrees)
    max_roll: float = 45.0
    max_pitch: float = 45.0
    max_tilt: float = 50.0  # Combined roll/pitch

    # Rate limits (deg/s)
    max_roll_rate: float = 360.0
    max_pitch_rate: float = 360.0
    max_yaw_rate: float = 180.0

    # Position limits (meters)
    max_horizontal_error: float = 20.0
    max_velocity_xy: float = 15.0
    max_velocity_z: float = 5.0

    # Geofence (meters from home)
    geofence_radius: float = 100.0

    # Motor limits
    motor_saturation_threshold: float = 0.95  # 95% of max
    motor_saturation_duration_max: float = 2.0  # seconds

    # Oscillation detection
    oscillation_threshold: float = 5.0  # degrees
    oscillation_frequency_min: float = 0.5  # Hz
    oscillation_count_max: int = 10

    # Stability
    stability_check_window: float = 2.0  # seconds
    stability_variance_max: float = 5.0  # degrees

    # Battery
    min_battery_voltage: float = 42.0  # 12S LiPo minimum

    # Timeout
    max_test_duration: float = 300.0  # 5 minutes per test


@dataclass
class SafetyViolation:
    """Record of a safety violation"""
    violation_type: SafetyViolationType
    timestamp: float
    severity: str  # 'critical', 'warning', 'info'
    description: str
    telemetry_snapshot: Dict
    recovery_possible: bool


class SafetyMonitor:
    """
    Real-time safety monitoring system
    Detects crashes, instabilities, and constraint violations
    """

    def __init__(self, limits: Optional[SafetyLimits] = None):
        self.limits = limits or SafetyLimits()
        self.violations = []
        self.telemetry_history = []
        self.start_time = None
        self.home_position = None
        self.motor_saturation_start = None

        # Running statistics
        self.attitude_buffer = []
        self.rate_buffer = []
        self.buffer_size = 20  # 2 seconds at 10Hz

    def reset(self):
        """Reset monitor for new test"""
        self.violations = []
        self.telemetry_history = []
        self.start_time = time.time()
        self.home_position = None
        self.motor_saturation_start = None
        self.attitude_buffer = []
        self.rate_buffer = []

    def set_home_position(self, lat: float, lon: float, alt: float):
        """Set home position for geofence"""
        self.home_position = (lat, lon, alt)
        logger.info(f"Home position set: {lat:.6f}, {lon:.6f}, {alt:.2f}m")

    def check_telemetry(self, telemetry: Dict) -> Tuple[bool, List[SafetyViolation]]:
        """
        Check telemetry data for safety violations
        Returns: (is_safe, violations_detected)
        """
        current_violations = []
        current_time = time.time() - self.start_time if self.start_time else 0

        # Store telemetry
        self.telemetry_history.append({
            'timestamp': current_time,
            'data': telemetry.copy()
        })

        # Extract telemetry data
        altitude = telemetry.get('altitude', 0)
        roll = telemetry.get('roll', 0)
        pitch = telemetry.get('pitch', 0)
        yaw = telemetry.get('yaw', 0)
        roll_rate = telemetry.get('roll_rate', 0)
        pitch_rate = telemetry.get('pitch_rate', 0)
        yaw_rate = telemetry.get('yaw_rate', 0)
        lat = telemetry.get('lat', 0)
        lon = telemetry.get('lon', 0)
        vx = telemetry.get('vx', 0)
        vy = telemetry.get('vy', 0)
        vz = telemetry.get('vz', 0)
        battery_voltage = telemetry.get('battery_voltage', 50.0)
        motor_outputs = telemetry.get('motor_outputs', [0.5, 0.5, 0.5, 0.5])

        # Update buffers
        self.attitude_buffer.append((roll, pitch, yaw))
        self.rate_buffer.append((roll_rate, pitch_rate, yaw_rate))
        if len(self.attitude_buffer) > self.buffer_size:
            self.attitude_buffer.pop(0)
        if len(self.rate_buffer) > self.buffer_size:
            self.rate_buffer.pop(0)

        # === CRITICAL CHECKS ===

        # 1. Crash detection (altitude too low)
        if altitude < self.limits.min_altitude:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.CRASH,
                timestamp=current_time,
                severity='critical',
                description=f"Crash detected: altitude {altitude:.2f}m < {self.limits.min_altitude}m",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=False
            )
            current_violations.append(violation)
            logger.error(violation.description)

        # 2. Altitude too high
        if altitude > self.limits.max_altitude:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.ALTITUDE_TOO_HIGH,
                timestamp=current_time,
                severity='critical',
                description=f"Altitude too high: {altitude:.2f}m > {self.limits.max_altitude}m",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=True
            )
            current_violations.append(violation)
            logger.error(violation.description)

        # 3. Excessive tilt
        tilt = math.sqrt(roll**2 + pitch**2)
        if abs(roll) > self.limits.max_roll:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.EXCESSIVE_TILT,
                timestamp=current_time,
                severity='critical',
                description=f"Excessive roll: {roll:.1f}° > {self.limits.max_roll}°",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=False
            )
            current_violations.append(violation)
            logger.error(violation.description)

        if abs(pitch) > self.limits.max_pitch:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.EXCESSIVE_TILT,
                timestamp=current_time,
                severity='critical',
                description=f"Excessive pitch: {pitch:.1f}° > {self.limits.max_pitch}°",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=False
            )
            current_violations.append(violation)
            logger.error(violation.description)

        if tilt > self.limits.max_tilt:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.EXCESSIVE_TILT,
                timestamp=current_time,
                severity='critical',
                description=f"Excessive tilt: {tilt:.1f}° > {self.limits.max_tilt}°",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=False
            )
            current_violations.append(violation)
            logger.error(violation.description)

        # 4. Excessive rates
        if abs(roll_rate) > self.limits.max_roll_rate:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.EXCESSIVE_RATE,
                timestamp=current_time,
                severity='warning',
                description=f"Excessive roll rate: {roll_rate:.1f}°/s > {self.limits.max_roll_rate}°/s",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=True
            )
            current_violations.append(violation)
            logger.warning(violation.description)

        if abs(pitch_rate) > self.limits.max_pitch_rate:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.EXCESSIVE_RATE,
                timestamp=current_time,
                severity='warning',
                description=f"Excessive pitch rate: {pitch_rate:.1f}°/s > {self.limits.max_pitch_rate}°/s",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=True
            )
            current_violations.append(violation)
            logger.warning(violation.description)

        # 5. Excessive velocity
        velocity_xy = math.sqrt(vx**2 + vy**2)
        if velocity_xy > self.limits.max_velocity_xy:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.VELOCITY_EXCESSIVE,
                timestamp=current_time,
                severity='warning',
                description=f"Excessive XY velocity: {velocity_xy:.1f} m/s > {self.limits.max_velocity_xy} m/s",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=True
            )
            current_violations.append(violation)
            logger.warning(violation.description)

        if abs(vz) > self.limits.max_velocity_z:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.VELOCITY_EXCESSIVE,
                timestamp=current_time,
                severity='warning',
                description=f"Excessive Z velocity: {abs(vz):.1f} m/s > {self.limits.max_velocity_z} m/s",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=True
            )
            current_violations.append(violation)
            logger.warning(violation.description)

        # 6. Geofence violation
        if self.home_position:
            distance = self._haversine_distance(
                lat, lon,
                self.home_position[0], self.home_position[1]
            )
            if distance > self.limits.geofence_radius:
                violation = SafetyViolation(
                    violation_type=SafetyViolationType.GEOFENCE,
                    timestamp=current_time,
                    severity='critical',
                    description=f"Geofence violation: {distance:.1f}m > {self.limits.geofence_radius}m",
                    telemetry_snapshot=telemetry.copy(),
                    recovery_possible=True
                )
                current_violations.append(violation)
                logger.error(violation.description)

        # 7. Motor saturation
        max_motor = max(motor_outputs) if motor_outputs else 0
        if max_motor > self.limits.motor_saturation_threshold:
            if self.motor_saturation_start is None:
                self.motor_saturation_start = current_time
            elif current_time - self.motor_saturation_start > self.limits.motor_saturation_duration_max:
                violation = SafetyViolation(
                    violation_type=SafetyViolationType.MOTOR_SATURATION,
                    timestamp=current_time,
                    severity='warning',
                    description=f"Sustained motor saturation: {max_motor:.2f} > {self.limits.motor_saturation_threshold}",
                    telemetry_snapshot=telemetry.copy(),
                    recovery_possible=True
                )
                current_violations.append(violation)
                logger.warning(violation.description)
                self.motor_saturation_start = None  # Reset
        else:
            self.motor_saturation_start = None

        # 8. Oscillation detection
        if len(self.attitude_buffer) >= self.buffer_size:
            oscillations = self._detect_oscillations()
            if oscillations > self.limits.oscillation_count_max:
                violation = SafetyViolation(
                    violation_type=SafetyViolationType.OSCILLATION,
                    timestamp=current_time,
                    severity='critical',
                    description=f"Excessive oscillations detected: {oscillations} > {self.limits.oscillation_count_max}",
                    telemetry_snapshot=telemetry.copy(),
                    recovery_possible=False
                )
                current_violations.append(violation)
                logger.error(violation.description)

        # 9. Instability detection
        if len(self.attitude_buffer) >= self.buffer_size:
            is_unstable = self._detect_instability()
            if is_unstable:
                violation = SafetyViolation(
                    violation_type=SafetyViolationType.INSTABILITY,
                    timestamp=current_time,
                    severity='critical',
                    description="Instability detected: excessive attitude variance",
                    telemetry_snapshot=telemetry.copy(),
                    recovery_possible=False
                )
                current_violations.append(violation)
                logger.error(violation.description)

        # 10. Battery voltage
        if battery_voltage < self.limits.min_battery_voltage:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.LOW_BATTERY,
                timestamp=current_time,
                severity='warning',
                description=f"Low battery: {battery_voltage:.1f}V < {self.limits.min_battery_voltage}V",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=True
            )
            current_violations.append(violation)
            logger.warning(violation.description)

        # 11. Timeout
        if current_time > self.limits.max_test_duration:
            violation = SafetyViolation(
                violation_type=SafetyViolationType.TIMEOUT,
                timestamp=current_time,
                severity='warning',
                description=f"Test timeout: {current_time:.1f}s > {self.limits.max_test_duration}s",
                telemetry_snapshot=telemetry.copy(),
                recovery_possible=True
            )
            current_violations.append(violation)
            logger.warning(violation.description)

        # Store violations
        self.violations.extend(current_violations)

        # Determine if safe
        critical_violations = [v for v in current_violations if v.severity == 'critical']
        is_safe = len(critical_violations) == 0

        return is_safe, current_violations

    def _detect_oscillations(self) -> int:
        """Detect oscillations in attitude data"""
        if len(self.attitude_buffer) < 4:
            return 0

        # Extract roll and pitch
        roll_data = [att[0] for att in self.attitude_buffer]
        pitch_data = [att[1] for att in self.attitude_buffer]

        # Count zero crossings around mean
        roll_oscillations = self._count_zero_crossings(roll_data)
        pitch_oscillations = self._count_zero_crossings(pitch_data)

        return max(roll_oscillations, pitch_oscillations)

    def _count_zero_crossings(self, data: List[float]) -> int:
        """Count zero crossings in signal"""
        mean = np.mean(data)
        crossings = 0

        for i in range(1, len(data)):
            if (data[i-1] - mean) * (data[i] - mean) < 0:
                crossings += 1

        return crossings // 2  # Full oscillation = 2 crossings

    def _detect_instability(self) -> bool:
        """Detect instability from attitude variance"""
        if len(self.attitude_buffer) < self.buffer_size:
            return False

        # Calculate variance
        roll_data = [att[0] for att in self.attitude_buffer]
        pitch_data = [att[1] for att in self.attitude_buffer]

        roll_variance = np.var(roll_data)
        pitch_variance = np.var(pitch_data)

        # Check if variance is increasing (trending unstable)
        if roll_variance > self.limits.stability_variance_max or \
           pitch_variance > self.limits.stability_variance_max:
            return True

        return False

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between GPS coordinates in meters"""
        R = 6371000  # Earth radius in meters

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    def get_violation_summary(self) -> Dict:
        """Get summary of all violations"""
        summary = {
            'total_violations': len(self.violations),
            'critical_violations': len([v for v in self.violations if v.severity == 'critical']),
            'warning_violations': len([v for v in self.violations if v.severity == 'warning']),
            'violation_types': {},
            'first_critical_violation': None,
            'is_safe': len([v for v in self.violations if v.severity == 'critical']) == 0
        }

        # Count by type
        for violation in self.violations:
            vtype = violation.violation_type.value
            if vtype not in summary['violation_types']:
                summary['violation_types'][vtype] = 0
            summary['violation_types'][vtype] += 1

        # Find first critical violation
        critical = [v for v in self.violations if v.severity == 'critical']
        if critical:
            summary['first_critical_violation'] = {
                'type': critical[0].violation_type.value,
                'timestamp': critical[0].timestamp,
                'description': critical[0].description
            }

        return summary

    def should_abort_test(self) -> bool:
        """Determine if test should be aborted"""
        # Abort on any critical, non-recoverable violation
        for violation in self.violations:
            if violation.severity == 'critical' and not violation.recovery_possible:
                return True

        return False

    def calculate_safety_score(self) -> float:
        """
        Calculate safety score (0-1, higher is safer)
        Used as penalty in fitness function
        """
        if not self.violations:
            return 1.0

        # Penalties
        critical_penalty = 0.5
        warning_penalty = 0.1

        score = 1.0

        for violation in self.violations:
            if violation.severity == 'critical':
                score -= critical_penalty
            elif violation.severity == 'warning':
                score -= warning_penalty

        return max(0.0, score)


class ParameterSafetyChecker:
    """Check if parameter values are within safe ranges before testing"""

    def __init__(self):
        self.safe_ranges = self._define_safe_ranges()

    def _define_safe_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Define absolute safe ranges for parameters"""
        return {
            # Rate controllers - must be positive, reasonable bounds
            'ATC_RAT_RLL_P': (0.01, 1.0),
            'ATC_RAT_RLL_I': (0.01, 1.0),
            'ATC_RAT_RLL_D': (0.0, 0.05),
            'ATC_RAT_PIT_P': (0.01, 1.0),
            'ATC_RAT_PIT_I': (0.01, 1.0),
            'ATC_RAT_PIT_D': (0.0, 0.05),
            'ATC_RAT_YAW_P': (0.01, 2.0),
            'ATC_RAT_YAW_I': (0.001, 0.5),

            # Attitude controllers
            'ATC_ANG_RLL_P': (1.0, 20.0),
            'ATC_ANG_PIT_P': (1.0, 20.0),
            'ATC_ANG_YAW_P': (1.0, 20.0),

            # Filters
            'INS_GYRO_FILTER': (5.0, 100.0),
            'INS_ACCEL_FILTER': (5.0, 50.0),

            # Motor parameters
            'MOT_THST_HOVER': (0.1, 0.8),
            'MOT_SPIN_MIN': (0.0, 0.3),
            'MOT_SPIN_MAX': (0.8, 1.0),

            # Position controllers
            'PSC_POSXY_P': (0.1, 5.0),
            'PSC_POSZ_P': (0.1, 5.0),
            'PSC_VELXY_P': (0.1, 10.0),
            'PSC_VELZ_P': (0.1, 15.0),
        }

    def check_parameters(self, params: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Check if parameters are safe to test
        Returns: (is_safe, list of violations)
        """
        violations = []

        for param_name, value in params.items():
            if param_name in self.safe_ranges:
                min_val, max_val = self.safe_ranges[param_name]

                if value < min_val:
                    violations.append(f"{param_name}={value:.4f} < min safe value {min_val}")
                elif value > max_val:
                    violations.append(f"{param_name}={value:.4f} > max safe value {max_val}")

                # Check for NaN or inf
                if math.isnan(value) or math.isinf(value):
                    violations.append(f"{param_name}={value} is invalid (NaN or Inf)")

        is_safe = len(violations) == 0

        if not is_safe:
            logger.error(f"Parameter safety check FAILED: {len(violations)} violations")
            for v in violations:
                logger.error(f"  - {v}")
        else:
            logger.info("Parameter safety check PASSED")

        return is_safe, violations


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example usage
    monitor = SafetyMonitor()
    monitor.reset()
    monitor.set_home_position(37.7749, -122.4194, 0)

    # Simulate telemetry
    test_telemetry = {
        'altitude': 10.0,
        'roll': 5.0,
        'pitch': 3.0,
        'yaw': 0.0,
        'roll_rate': 10.0,
        'pitch_rate': 8.0,
        'yaw_rate': 5.0,
        'lat': 37.7749,
        'lon': -122.4194,
        'vx': 1.0,
        'vy': 0.5,
        'vz': -0.2,
        'battery_voltage': 48.0,
        'motor_outputs': [0.5, 0.5, 0.5, 0.5]
    }

    is_safe, violations = monitor.check_telemetry(test_telemetry)
    logger.info(f"Safety check: {'PASS' if is_safe else 'FAIL'}")
    logger.info(f"Violations: {len(violations)}")

    # Test parameter checker
    checker = ParameterSafetyChecker()
    test_params = {
        'ATC_RAT_RLL_P': 0.15,
        'ATC_RAT_PIT_P': 0.15,
        'ATC_ANG_RLL_P': 4.5,
    }

    is_safe, violations = checker.check_parameters(test_params)
    logger.info(f"Parameter check: {'PASS' if is_safe else 'FAIL'}")
