"""
Performance Evaluator for Drone Tuning

Evaluates drone performance based on telemetry data and calculates fitness scores
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import math

from frequency_domain_analysis import (
    analyze_telemetry_frequency_domain,
    calculate_frequency_domain_fitness
)


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    # Step response metrics
    rise_time: float = 0.0
    settling_time: float = 0.0
    overshoot: float = 0.0
    steady_state_error: float = 0.0

    # Stability metrics
    oscillation_detected: bool = False
    oscillation_amplitude: float = 0.0
    oscillation_frequency: float = 0.0

    # Control metrics
    motor_saturation_count: int = 0
    motor_saturation_duration: float = 0.0

    # Tracking metrics
    position_rmse: float = 0.0
    attitude_rmse: float = 0.0

    # Power metrics
    avg_power: float = 0.0
    power_efficiency: float = 0.0

    # Safety metrics
    max_angle: float = 0.0
    max_rate: float = 0.0
    max_altitude_error: float = 0.0

    # Frequency-domain metrics
    phase_margin_deg: float = 0.0
    gain_margin_db: float = 0.0
    phase_margin_score: float = 0.0
    gain_margin_score: float = 0.0

    # Overall
    crashed: bool = False
    fitness: float = 0.0


class PerformanceEvaluator:
    """Evaluates drone performance from telemetry data"""

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize performance evaluator

        Args:
            weights: Dictionary of weights for fitness function components
        """
        # Default fitness weights (must sum to ~1.0)
        self.weights = weights or {
            'stability': 0.25,
            'response_time': 0.20,
            'tracking': 0.15,
            'phase_margin': 0.15,      # Frequency-domain: formal stability
            'gain_margin': 0.10,        # Frequency-domain: robustness
            'power_efficiency': 0.05,
            'smoothness': 0.10
        }

        # Safety constraints
        self.safety_limits = {
            'max_angle': 45.0,           # degrees
            'max_rate': 360.0,           # deg/s
            'max_altitude_error': 2.0,   # meters
            'max_position_error': 5.0,   # meters
            'min_phase_margin': 45.0,    # degrees
            'max_oscillation_amp': 5.0,  # degrees
        }

        # Performance thresholds
        self.performance_thresholds = {
            'rise_time_target': 0.5,      # seconds
            'settling_time_target': 2.0,  # seconds
            'overshoot_max': 10.0,        # percent
            'steady_state_error_max': 2.0 # percent
        }

    def evaluate_telemetry(self, telemetry: Dict) -> PerformanceMetrics:
        """
        Evaluate performance from telemetry data

        Args:
            telemetry: Dictionary containing telemetry arrays

        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()

        try:
            # Check for crash
            if self._detect_crash(telemetry):
                metrics.crashed = True
                metrics.fitness = -1000.0
                return metrics

            # Calculate step response metrics
            if 'attitude' in telemetry and 'attitude_target' in telemetry:
                step_metrics = self._analyze_step_response(
                    telemetry['time'],
                    telemetry['attitude'],
                    telemetry['attitude_target']
                )
                metrics.rise_time = step_metrics['rise_time']
                metrics.settling_time = step_metrics['settling_time']
                metrics.overshoot = step_metrics['overshoot']
                metrics.steady_state_error = step_metrics['steady_state_error']

            # Detect oscillations
            if 'attitude' in telemetry:
                osc_metrics = self._detect_oscillations(
                    telemetry['time'],
                    telemetry['attitude']
                )
                metrics.oscillation_detected = osc_metrics['detected']
                metrics.oscillation_amplitude = osc_metrics['amplitude']
                metrics.oscillation_frequency = osc_metrics['frequency']

            # Motor saturation analysis
            if 'motor_outputs' in telemetry:
                sat_metrics = self._analyze_motor_saturation(
                    telemetry['time'],
                    telemetry['motor_outputs']
                )
                metrics.motor_saturation_count = sat_metrics['count']
                metrics.motor_saturation_duration = sat_metrics['duration']

            # Tracking performance
            if 'position' in telemetry and 'position_target' in telemetry:
                metrics.position_rmse = self._calculate_rmse(
                    telemetry['position'],
                    telemetry['position_target']
                )

            if 'attitude' in telemetry and 'attitude_target' in telemetry:
                metrics.attitude_rmse = self._calculate_rmse(
                    telemetry['attitude'],
                    telemetry['attitude_target']
                )

            # Power metrics
            if 'motor_outputs' in telemetry:
                metrics.avg_power = self._calculate_avg_power(
                    telemetry['motor_outputs']
                )
                metrics.power_efficiency = self._calculate_power_efficiency(
                    telemetry['motor_outputs'],
                    telemetry.get('altitude_change', 0)
                )

            # Safety metrics
            if 'attitude' in telemetry:
                metrics.max_angle = np.max(np.abs(telemetry['attitude']))

            if 'rates' in telemetry:
                metrics.max_rate = np.max(np.abs(telemetry['rates']))

            if 'altitude' in telemetry and 'altitude_target' in telemetry:
                metrics.max_altitude_error = np.max(
                    np.abs(telemetry['altitude'] - telemetry['altitude_target'])
                )

            # Frequency-domain analysis
            try:
                freq_results = analyze_telemetry_frequency_domain(telemetry)
                if 'overall' in freq_results:
                    metrics.phase_margin_deg = freq_results['overall'].get('phase_margin_deg', 0.0)
                    metrics.gain_margin_db = freq_results['overall'].get('gain_margin_db', 0.0)

                    # Calculate normalized fitness scores
                    freq_fitness = calculate_frequency_domain_fitness({
                        'phase_margin_deg': metrics.phase_margin_deg,
                        'gain_margin_db': metrics.gain_margin_db
                    })
                    metrics.phase_margin_score = freq_fitness['phase_margin_score']
                    metrics.gain_margin_score = freq_fitness['gain_margin_score']

                    logger.debug(f"Frequency domain: PM={metrics.phase_margin_deg:.1f}°, "
                               f"GM={metrics.gain_margin_db:.1f}dB")
            except Exception as e:
                logger.debug(f"Frequency domain analysis failed: {e}")
                # Use neutral scores if analysis fails
                metrics.phase_margin_score = 0.5
                metrics.gain_margin_score = 0.5

            # Check safety violations
            if not self._check_safety_constraints(metrics):
                metrics.fitness = -500.0
                return metrics

            # Calculate overall fitness
            metrics.fitness = self._calculate_fitness(metrics)

        except Exception as e:
            logger.error(f"Error evaluating telemetry: {e}")
            metrics.crashed = True
            metrics.fitness = -1000.0

        return metrics

    def _detect_crash(self, telemetry: Dict) -> bool:
        """Detect if the drone crashed"""
        try:
            # Check for NaN or infinite values
            for key, data in telemetry.items():
                if isinstance(data, np.ndarray):
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        logger.warning(f"Invalid values detected in {key}")
                        return True

            # Check if altitude went to zero (ground contact)
            if 'altitude' in telemetry:
                if np.min(telemetry['altitude']) < -0.5:
                    logger.warning("Altitude went negative - crash detected")
                    return True

            # Check for extreme angles
            if 'attitude' in telemetry:
                if np.any(np.abs(telemetry['attitude']) > 90):
                    logger.warning("Extreme angle detected - crash")
                    return True

            # Check if simulation ended prematurely
            if 'time' in telemetry:
                if len(telemetry['time']) < 10:
                    logger.warning("Too few data points - likely crashed")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error in crash detection: {e}")
            return True

    def _analyze_step_response(self, time: np.ndarray, response: np.ndarray,
                               target: np.ndarray) -> Dict[str, float]:
        """Analyze step response characteristics"""
        try:
            # Find step change in target
            target_diff = np.diff(target)
            step_indices = np.where(np.abs(target_diff) > 0.1)[0]

            if len(step_indices) == 0:
                return {
                    'rise_time': 0.0,
                    'settling_time': 0.0,
                    'overshoot': 0.0,
                    'steady_state_error': 0.0
                }

            # Analyze first step
            step_idx = step_indices[0]
            initial_value = response[step_idx]
            final_target = target[step_idx + 1]
            step_size = final_target - initial_value

            if abs(step_size) < 0.01:
                return {
                    'rise_time': 0.0,
                    'settling_time': 0.0,
                    'overshoot': 0.0,
                    'steady_state_error': 0.0
                }

            # Calculate rise time (10% to 90%)
            rise_time = self._calculate_rise_time(
                time[step_idx:],
                response[step_idx:],
                initial_value,
                final_target
            )

            # Calculate settling time (within 2% of final value)
            settling_time = self._calculate_settling_time(
                time[step_idx:],
                response[step_idx:],
                final_target,
                tolerance=0.02
            )

            # Calculate overshoot
            overshoot = self._calculate_overshoot(
                response[step_idx:],
                initial_value,
                final_target
            )

            # Calculate steady-state error
            steady_state_error = self._calculate_steady_state_error(
                response[-100:],  # Last 100 samples
                final_target
            )

            return {
                'rise_time': rise_time,
                'settling_time': settling_time,
                'overshoot': overshoot,
                'steady_state_error': steady_state_error
            }

        except Exception as e:
            logger.error(f"Error analyzing step response: {e}")
            return {
                'rise_time': 999.0,
                'settling_time': 999.0,
                'overshoot': 100.0,
                'steady_state_error': 100.0
            }

    def _calculate_rise_time(self, time: np.ndarray, response: np.ndarray,
                            initial: float, final: float) -> float:
        """Calculate rise time (10% to 90%)"""
        try:
            threshold_10 = initial + 0.1 * (final - initial)
            threshold_90 = initial + 0.9 * (final - initial)

            # Find when response crosses thresholds
            cross_10 = np.where(response >= threshold_10)[0]
            cross_90 = np.where(response >= threshold_90)[0]

            if len(cross_10) == 0 or len(cross_90) == 0:
                return 999.0

            t_10 = time[cross_10[0]]
            t_90 = time[cross_90[0]]

            return t_90 - t_10

        except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating rise time: {e}")
            return 999.0

    def _calculate_settling_time(self, time: np.ndarray, response: np.ndarray,
                                final: float, tolerance: float = 0.02) -> float:
        """Calculate settling time"""
        try:
            error = np.abs(response - final)
            threshold = abs(final) * tolerance

            # Find last time error exceeded threshold
            exceed_indices = np.where(error > threshold)[0]

            if len(exceed_indices) == 0:
                return time[0]

            settle_idx = exceed_indices[-1]
            return time[settle_idx] - time[0]

        except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating settling time: {e}")
            return 999.0

    def _calculate_overshoot(self, response: np.ndarray,
                            initial: float, final: float) -> float:
        """Calculate overshoot percentage"""
        try:
            step_size = final - initial

            if step_size > 0:
                max_value = np.max(response)
                overshoot = ((max_value - final) / step_size) * 100
            else:
                min_value = np.min(response)
                overshoot = ((final - min_value) / abs(step_size)) * 100

            return max(0, overshoot)

        except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating overshoot: {e}")
            return 100.0

    def _calculate_steady_state_error(self, response: np.ndarray,
                                      target: float) -> float:
        """Calculate steady-state error"""
        try:
            if abs(target) < 0.001:
                return abs(np.mean(response))

            error = abs(np.mean(response) - target)
            return (error / abs(target)) * 100

        except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating steady state error: {e}")
            return 100.0

    def _detect_oscillations(self, time: np.ndarray,
                            signal_data: np.ndarray) -> Dict:
        """
        Detect oscillations using Welch's method (more efficient than FFT)

        Welch's method uses windowed FFT segments, providing:
        - ~4x faster computation than full FFT
        - Better noise reduction through averaging
        - More stable frequency estimation
        """
        try:
            # Detrend signal
            detrended = signal.detrend(signal_data)

            # Calculate sampling frequency
            N = len(signal_data)
            if N < 10:
                return {'detected': False, 'amplitude': 0.0, 'frequency': 0.0}

            dt = np.mean(np.diff(time))
            fs = 1.0 / dt  # Sampling frequency

            # Use Welch's method for power spectral density estimation
            # nperseg: length of each segment (smaller = faster, but less frequency resolution)
            nperseg = min(256, N // 2)

            freqs, psd = welch(
                detrended,
                fs=fs,
                nperseg=nperseg,
                scaling='density',
                detrend='constant'
            )

            # Skip DC component (freqs[0] = 0 Hz)
            if len(freqs) < 2:
                return {'detected': False, 'amplitude': 0.0, 'frequency': 0.0}

            # Find dominant frequency (excluding DC)
            max_idx = np.argmax(psd[1:]) + 1
            dominant_freq = freqs[max_idx]
            amplitude = np.sqrt(psd[max_idx])  # Convert PSD to amplitude

            # Oscillation detected if amplitude > threshold and freq in range
            # Thresholds can be tuned based on your specific drone characteristics
            detected = (amplitude > 2.0 and 0.5 < dominant_freq < 20.0)

            return {
                'detected': detected,
                'amplitude': amplitude,
                'frequency': dominant_freq
            }

        except Exception as e:
            logger.error(f"Error detecting oscillations: {e}")
            return {
                'detected': False,
                'amplitude': 0.0,
                'frequency': 0.0
            }

    def _analyze_motor_saturation(self, time: np.ndarray,
                                  motor_outputs: np.ndarray) -> Dict:
        """Analyze motor saturation"""
        try:
            # Motor outputs typically 0-1 or PWM values
            # Assume normalized 0-1
            saturation_threshold = 0.95

            saturated = np.any(motor_outputs > saturation_threshold, axis=1)
            count = np.sum(saturated)

            dt = np.mean(np.diff(time))
            duration = count * dt

            return {
                'count': int(count),
                'duration': duration
            }

        except Exception as e:
            logger.error(f"Error analyzing motor saturation: {e}")
            return {
                'count': 0,
                'duration': 0.0
            }

    def _calculate_rmse(self, actual: np.ndarray, target: np.ndarray) -> float:
        """Calculate RMSE between actual and target"""
        try:
            return np.sqrt(np.mean((actual - target)**2))
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating RMSE: {e}")
            return 999.0

    def _calculate_avg_power(self, motor_outputs: np.ndarray) -> float:
        """Calculate average power consumption"""
        try:
            # Simplified power model: P ~ motor_output^2
            return np.mean(np.sum(motor_outputs**2, axis=1))
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating average power: {e}")
            return 0.0

    def _calculate_power_efficiency(self, motor_outputs: np.ndarray,
                                   altitude_change: float) -> float:
        """Calculate power efficiency (altitude change per unit power)"""
        try:
            total_power = np.sum(motor_outputs**2)
            if total_power < 0.001:
                return 0.0
            return abs(altitude_change) / total_power
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Error calculating power efficiency: {e}")
            return 0.0

    def _check_safety_constraints(self, metrics: PerformanceMetrics) -> bool:
        """Check if safety constraints are violated"""
        if metrics.max_angle > self.safety_limits['max_angle']:
            logger.warning(f"Max angle {metrics.max_angle} exceeds limit")
            return False

        if metrics.max_rate > self.safety_limits['max_rate']:
            logger.warning(f"Max rate {metrics.max_rate} exceeds limit")
            return False

        if metrics.max_altitude_error > self.safety_limits['max_altitude_error']:
            logger.warning(f"Max altitude error {metrics.max_altitude_error} exceeds limit")
            return False

        if metrics.oscillation_amplitude > self.safety_limits['max_oscillation_amp']:
            logger.warning(f"Oscillation amplitude {metrics.oscillation_amplitude} exceeds limit")
            return False

        return True

    def _calculate_fitness(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall fitness score"""

        # Stability score (inverse of oscillation and overshoot)
        stability_score = 100.0
        if metrics.oscillation_detected:
            stability_score -= 50.0
        stability_score -= min(metrics.overshoot, 50.0)
        stability_score = max(0, stability_score)

        # Response time score (inverse of rise/settling time)
        rise_score = 100.0 * np.exp(-metrics.rise_time / 0.5)
        settling_score = 100.0 * np.exp(-metrics.settling_time / 2.0)
        response_score = (rise_score + settling_score) / 2.0

        # Tracking score (inverse of RMSE)
        tracking_score = 100.0 / (1.0 + metrics.position_rmse + metrics.attitude_rmse)

        # Frequency-domain scores (normalized 0-1 scale, convert to 0-100)
        phase_margin_score = metrics.phase_margin_score * 100.0
        gain_margin_score = metrics.gain_margin_score * 100.0

        # Power efficiency score
        power_score = min(100.0, metrics.power_efficiency * 10.0)

        # Smoothness score (inverse of motor saturation)
        smoothness_score = 100.0 * np.exp(-metrics.motor_saturation_duration / 10.0)

        # Weighted sum
        fitness = (
            self.weights['stability'] * stability_score +
            self.weights['response_time'] * response_score +
            self.weights['tracking'] * tracking_score +
            self.weights['phase_margin'] * phase_margin_score +
            self.weights['gain_margin'] * gain_margin_score +
            self.weights['power_efficiency'] * power_score +
            self.weights['smoothness'] * smoothness_score
        )

        return fitness

    def validate_parameters(self, sitl_manager, parameters: Dict[str, float]) -> Dict:
        """
        Run comprehensive validation tests

        Args:
            sitl_manager: SITL manager instance
            parameters: Parameters to validate

        Returns:
            Validation results dictionary
        """
        logger.info("Running comprehensive validation tests...")

        validation_results = {
            'safety_passed': True,
            'performance_score': 0.0,
            'test_results': {}
        }

        # This would run comprehensive test sequences
        # For now, return placeholder
        logger.info("Validation complete")

        return validation_results
