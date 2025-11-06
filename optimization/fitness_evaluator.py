#!/usr/bin/env python3
"""
Fitness Evaluation System for Drone PID Tuning

Evaluates parameter candidates by running test sequences in SITL
and calculating performance metrics.

Fitness function combines multiple objectives:
- Stability (no oscillations or crashes)
- Response time (rise time, settling time)
- Tracking accuracy (overshoot, steady-state error)
- Smoothness (control effort, motor saturation)

Author: MC07 Tuning System
Date: 2025-11-05
"""

import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from dronekit import connect, VehicleMode, LocationGlobalRelative
    from pymavlink import mavutil
    DRONEKIT_AVAILABLE = True
except ImportError:
    DRONEKIT_AVAILABLE = False
    logging.warning("dronekit not available - fitness evaluation will be limited")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStage(Enum):
    """Test stages for hierarchical optimization"""
    RATE = 1          # Rate controller tests
    ATTITUDE = 2      # Attitude controller tests
    POSITION = 3      # Position controller tests


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    # Stability metrics
    crashed: bool = False
    oscillation_detected: bool = False
    max_oscillation_amplitude: float = 0.0

    # Response metrics
    rise_time: float = 0.0           # Time to reach 90% of target
    settling_time: float = 0.0       # Time to settle within 2% of target
    overshoot: float = 0.0           # Maximum overshoot percentage
    steady_state_error: float = 0.0  # Final tracking error

    # Tracking metrics
    rms_tracking_error: float = 0.0  # Root mean square tracking error
    max_tracking_error: float = 0.0  # Maximum tracking error

    # Control effort metrics
    control_smoothness: float = 0.0  # Variance in control outputs
    motor_saturation_count: int = 0  # Number of times motors saturated

    # Test execution
    test_completed: bool = False
    test_duration: float = 0.0
    stage: TestStage = TestStage.RATE

    # Raw data
    attitude_history: List[Dict] = field(default_factory=list)
    rate_history: List[Dict] = field(default_factory=list)
    control_history: List[Dict] = field(default_factory=list)


class FitnessEvaluator:
    """
    Evaluates fitness of parameter candidates
    """

    def __init__(self,
                 stage: TestStage = TestStage.RATE,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize fitness evaluator

        Args:
            stage: Test stage (RATE, ATTITUDE, or POSITION)
            weights: Optional custom weights for fitness components
        """
        self.stage = stage

        # Default fitness weights (can be customized)
        self.weights = weights or self._get_default_weights(stage)

        logger.info(f"FitnessEvaluator initialized for stage: {stage.name}")

    def _get_default_weights(self, stage: TestStage) -> Dict[str, float]:
        """
        Get default fitness weights for each stage

        Args:
            stage: Test stage

        Returns:
            Dictionary of weights
        """
        if stage == TestStage.RATE:
            # Rate controller: prioritize stability and fast response
            return {
                'stability': 0.40,      # No oscillations/crashes
                'response_speed': 0.25, # Fast rise/settling time
                'tracking': 0.20,       # Accurate rate tracking
                'smoothness': 0.15,     # Smooth control outputs
            }
        elif stage == TestStage.ATTITUDE:
            # Attitude controller: balance stability and accuracy
            return {
                'stability': 0.35,
                'response_speed': 0.20,
                'tracking': 0.30,
                'smoothness': 0.15,
            }
        else:  # POSITION
            # Position controller: prioritize accuracy
            return {
                'stability': 0.25,
                'response_speed': 0.15,
                'tracking': 0.45,
                'smoothness': 0.15,
            }

    def calculate_fitness(self, metrics: PerformanceMetrics) -> float:
        """
        Calculate overall fitness score from performance metrics

        Args:
            metrics: PerformanceMetrics object

        Returns:
            Fitness score (0-100, higher is better)
        """
        # Immediate disqualification for crashes
        if metrics.crashed:
            logger.warning("Candidate crashed - fitness = 0")
            return 0.0

        # Heavily penalize oscillations
        if metrics.oscillation_detected:
            logger.warning(f"Oscillations detected (amplitude: {metrics.max_oscillation_amplitude:.2f}°)")
            return 5.0  # Very low score but not zero

        # Component scores (0-100 each)
        stability_score = self._calculate_stability_score(metrics)
        response_score = self._calculate_response_score(metrics)
        tracking_score = self._calculate_tracking_score(metrics)
        smoothness_score = self._calculate_smoothness_score(metrics)

        # Weighted combination
        fitness = (
            self.weights['stability'] * stability_score +
            self.weights['response_speed'] * response_score +
            self.weights['tracking'] * tracking_score +
            self.weights['smoothness'] * smoothness_score
        )

        logger.debug(f"Fitness components: "
                    f"stability={stability_score:.1f}, "
                    f"response={response_score:.1f}, "
                    f"tracking={tracking_score:.1f}, "
                    f"smoothness={smoothness_score:.1f}, "
                    f"TOTAL={fitness:.1f}")

        return fitness

    def _calculate_stability_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate stability component score"""
        score = 100.0

        # Penalize oscillations (shouldn't reach here if oscillation_detected, but just in case)
        if metrics.max_oscillation_amplitude > 0:
            # Exponential penalty for oscillation amplitude
            penalty = min(50, metrics.max_oscillation_amplitude * 10)
            score -= penalty

        # Penalize motor saturation
        if metrics.motor_saturation_count > 0:
            # Each saturation event reduces score
            penalty = min(30, metrics.motor_saturation_count * 2)
            score -= penalty

        return max(0, score)

    def _calculate_response_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate response speed component score"""
        score = 100.0

        # Target times (stage-dependent)
        if self.stage == TestStage.RATE:
            target_rise_time = 0.15  # 150ms
            target_settling_time = 0.40  # 400ms
        elif self.stage == TestStage.ATTITUDE:
            target_rise_time = 0.25  # 250ms
            target_settling_time = 0.60  # 600ms
        else:  # POSITION
            target_rise_time = 0.50  # 500ms
            target_settling_time = 1.00  # 1s

        # Penalize slow rise time
        if metrics.rise_time > target_rise_time:
            penalty = min(40, (metrics.rise_time - target_rise_time) * 100)
            score -= penalty

        # Penalize slow settling time
        if metrics.settling_time > target_settling_time:
            penalty = min(40, (metrics.settling_time - target_settling_time) * 50)
            score -= penalty

        # Penalize excessive overshoot
        if metrics.overshoot > 10:  # More than 10%
            penalty = min(20, (metrics.overshoot - 10) * 2)
            score -= penalty

        return max(0, score)

    def _calculate_tracking_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate tracking accuracy component score"""
        score = 100.0

        # Target tracking errors (stage-dependent)
        if self.stage == TestStage.RATE:
            target_rms = 2.0   # 2 deg/s RMS error
            target_max = 10.0  # 10 deg/s max error
        elif self.stage == TestStage.ATTITUDE:
            target_rms = 1.0   # 1 degree RMS error
            target_max = 5.0   # 5 degree max error
        else:  # POSITION
            target_rms = 0.5   # 0.5 meter RMS error
            target_max = 2.0   # 2 meter max error

        # Penalize high RMS error
        if metrics.rms_tracking_error > target_rms:
            penalty = min(50, (metrics.rms_tracking_error - target_rms) * 20)
            score -= penalty

        # Penalize high max error
        if metrics.max_tracking_error > target_max:
            penalty = min(30, (metrics.max_tracking_error - target_max) * 5)
            score -= penalty

        # Penalize steady-state error
        if metrics.steady_state_error > target_rms / 2:
            penalty = min(20, metrics.steady_state_error * 10)
            score -= penalty

        return max(0, score)

    def _calculate_smoothness_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate control smoothness component score"""
        score = 100.0

        # Penalize high control variance (jerky control)
        if metrics.control_smoothness > 0:
            # Normalize control smoothness (arbitrary scale, tune based on data)
            penalty = min(50, metrics.control_smoothness * 10)
            score -= penalty

        return max(0, score)

    def run_test_sequence(self, connection_string: str, timeout: int = 60) -> PerformanceMetrics:
        """
        Run complete test sequence and collect metrics

        Args:
            connection_string: MAVLink connection string (e.g., "127.0.0.1:14550")
            timeout: Maximum test duration in seconds

        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics(stage=self.stage)

        if not DRONEKIT_AVAILABLE:
            logger.error("dronekit not available - cannot run test sequence")
            metrics.crashed = True
            return metrics

        try:
            logger.info(f"Connecting to vehicle on {connection_string}...")
            vehicle = connect(connection_string, wait_ready=True, timeout=timeout)

            logger.info("Running test sequence...")
            start_time = time.time()

            # Run stage-specific tests
            if self.stage == TestStage.RATE:
                metrics = self._run_rate_test(vehicle, metrics)
            elif self.stage == TestStage.ATTITUDE:
                metrics = self._run_attitude_test(vehicle, metrics)
            else:  # POSITION
                metrics = self._run_position_test(vehicle, metrics)

            metrics.test_duration = time.time() - start_time
            metrics.test_completed = True

            logger.info(f"Test completed in {metrics.test_duration:.1f}s")

            # Close connection
            vehicle.close()

        except Exception as e:
            logger.error(f"Test sequence failed: {e}")
            metrics.crashed = True
            metrics.test_completed = False

        return metrics

    def _run_rate_test(self, vehicle, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """
        Run rate controller test sequence

        Tests:
        1. Hover stability test (5s)
        2. Roll rate step response
        3. Pitch rate step response
        4. Yaw rate step response
        5. Combined maneuver

        Args:
            vehicle: Dronekit vehicle object
            metrics: PerformanceMetrics to populate

        Returns:
            Updated metrics
        """
        logger.info("Rate controller test sequence")

        # TODO: Implement actual test sequence with dronekit
        # For now, placeholder implementation
        time.sleep(5)

        # Placeholder metrics
        metrics.rise_time = 0.18
        metrics.settling_time = 0.35
        metrics.overshoot = 8.5
        metrics.rms_tracking_error = 1.5
        metrics.max_tracking_error = 5.0
        metrics.steady_state_error = 0.5

        return metrics

    def _run_attitude_test(self, vehicle, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """
        Run attitude controller test sequence

        Args:
            vehicle: Dronekit vehicle object
            metrics: PerformanceMetrics to populate

        Returns:
            Updated metrics
        """
        logger.info("Attitude controller test sequence")

        # TODO: Implement attitude test
        time.sleep(5)

        return metrics

    def _run_position_test(self, vehicle, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """
        Run position controller test sequence

        Args:
            vehicle: Dronekit vehicle object
            metrics: PerformanceMetrics to populate

        Returns:
            Updated metrics
        """
        logger.info("Position controller test sequence")

        # TODO: Implement position test
        time.sleep(5)

        return metrics


if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("FITNESS EVALUATOR DEMONSTRATION")
    print("="*70 + "\n")

    # Test for each stage
    for stage in TestStage:
        print(f"\n{stage.name} Controller Evaluation")
        print("-" * 70)

        evaluator = FitnessEvaluator(stage=stage)

        # Create sample metrics (simulating test results)
        metrics = PerformanceMetrics(
            stage=stage,
            crashed=False,
            oscillation_detected=False,
            rise_time=0.18,
            settling_time=0.42,
            overshoot=7.5,
            steady_state_error=0.3,
            rms_tracking_error=1.2,
            max_tracking_error=4.5,
            control_smoothness=0.05,
            motor_saturation_count=2,
            test_completed=True,
            test_duration=30.0
        )

        fitness = evaluator.calculate_fitness(metrics)

        print(f"Fitness weights: {evaluator.weights}")
        print(f"\nPerformance Metrics:")
        print(f"  Rise time: {metrics.rise_time:.3f}s")
        print(f"  Settling time: {metrics.settling_time:.3f}s")
        print(f"  Overshoot: {metrics.overshoot:.1f}%")
        print(f"  RMS error: {metrics.rms_tracking_error:.2f}")
        print(f"  Max error: {metrics.max_tracking_error:.2f}")
        print(f"\n  OVERALL FITNESS: {fitness:.2f}/100")

    print("\n" + "="*70 + "\n")
