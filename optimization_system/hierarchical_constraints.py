"""
Hierarchical Control Constraints for Cascade Loop Optimization

Enforces bandwidth separation rules to ensure stable cascade control:
- Inner loop must be 5-10x faster than outer loop
- Rate loop > 4-15x faster than Attitude loop
- Attitude loop > 3-8x faster than Position loop

This prevents optimizer from finding "solutions" that work individually
but fail when cascaded together.
"""

import logging
from typing import Dict, Optional, Tuple
from .physics_based_seeding import (
    estimate_closed_loop_bandwidth,
    estimate_attitude_loop_bandwidth,
    estimate_position_loop_bandwidth,
    check_bandwidth_separation,
    validate_cascade_bandwidths
)

logger = logging.getLogger(__name__)


class HierarchicalConstraintValidator:
    """
    Validates and enforces hierarchical control constraints during optimization
    """

    def __init__(self, drone_params: Dict[str, float],
                 enforce_constraints: bool = True,
                 penalty_multiplier: float = 1000.0):
        """
        Initialize constraint validator

        Args:
            drone_params: Drone physical parameters (mass, inertia, etc.)
            enforce_constraints: Whether to apply hard constraints
            penalty_multiplier: Penalty value for constraint violations
        """
        self.drone_params = drone_params
        self.enforce_constraints = enforce_constraints
        self.penalty_multiplier = penalty_multiplier

        # Store optimized parameters from previous phases
        self.optimized_rate_params = None
        self.optimized_attitude_params = None
        self.optimized_position_params = None

        # Calculate motor gain for bandwidth estimation
        max_thrust = drone_params.get('max_thrust_per_motor', 75.0)
        arm_length = drone_params.get('arm_length', 0.75)
        self.motor_gain = max_thrust * arm_length
        self.inertia_roll = drone_params.get('Ixx', 6.0)
        self.inertia_pitch = drone_params.get('Iyy', 6.0)

        logger.info("HierarchicalConstraintValidator initialized")
        logger.info(f"  Enforce constraints: {enforce_constraints}")
        logger.info(f"  Motor gain: {self.motor_gain:.2f} N·m")
        logger.info(f"  Inertia (roll/pitch): {self.inertia_roll:.2f} kg·m²")

    def set_optimized_phase(self, phase_name: str, parameters: Dict[str, float]):
        """
        Store optimized parameters from a completed phase

        Args:
            phase_name: Name of the phase ('phase1_rate', 'phase2_attitude', etc.)
            parameters: Optimized parameter dictionary
        """
        if phase_name == 'phase1_rate':
            self.optimized_rate_params = parameters
            logger.info("✓ Stored optimized rate controller parameters")
        elif phase_name == 'phase2_attitude':
            self.optimized_attitude_params = parameters
            logger.info("✓ Stored optimized attitude controller parameters")
        elif phase_name == 'phase3_position':
            self.optimized_position_params = parameters
            logger.info("✓ Stored optimized position controller parameters")

    def validate_phase1_rate(self, params: Dict[str, float]) -> Tuple[bool, str, float]:
        """
        Validate rate controller parameters (Phase 1)

        No inter-loop constraints for the innermost loop.

        Args:
            params: Rate controller parameters

        Returns:
            Tuple of (is_valid, message, estimated_bandwidth_hz)
        """
        # Estimate rate loop bandwidth
        P = params.get('ATC_RAT_RLL_P', 0.15)
        I = params.get('ATC_RAT_RLL_I', 0.15)
        D = params.get('ATC_RAT_RLL_D', 0.005)

        rate_bw = estimate_closed_loop_bandwidth(P, I, D, self.inertia_roll, self.motor_gain)

        # Basic sanity checks
        if rate_bw < 5.0:
            msg = f"Rate loop bandwidth too low: {rate_bw:.2f} Hz (minimum: 5 Hz)"
            return False, msg, rate_bw

        if rate_bw > 30.0:
            msg = f"Rate loop bandwidth too high: {rate_bw:.2f} Hz (maximum: 30 Hz, may amplify noise)"
            return False, msg, rate_bw

        msg = f"Rate loop bandwidth: {rate_bw:.2f} Hz (valid)"
        return True, msg, rate_bw

    def validate_phase2_attitude(self, params: Dict[str, float]) -> Tuple[bool, str, Dict]:
        """
        Validate attitude controller parameters (Phase 2)

        Enforces: Rate BW > 4-15x Attitude BW

        Args:
            params: Attitude controller parameters

        Returns:
            Tuple of (is_valid, message, bandwidth_dict)
        """
        if self.optimized_rate_params is None:
            logger.warning("Rate parameters not set - cannot validate attitude loop separation")
            return True, "Rate parameters not available", {}

        # Get rate loop bandwidth
        rate_P = self.optimized_rate_params.get('ATC_RAT_RLL_P', 0.15)
        rate_I = self.optimized_rate_params.get('ATC_RAT_RLL_I', 0.15)
        rate_D = self.optimized_rate_params.get('ATC_RAT_RLL_D', 0.005)
        rate_bw = estimate_closed_loop_bandwidth(rate_P, rate_I, rate_D,
                                                 self.inertia_roll, self.motor_gain)

        # Get attitude loop bandwidth
        attitude_P = params.get('ATC_ANG_RLL_P', 4.5)
        attitude_bw = estimate_attitude_loop_bandwidth(attitude_P, rate_bw)

        # Check separation
        is_valid, ratio = check_bandwidth_separation(
            rate_bw, attitude_bw, min_ratio=4.0, max_ratio=15.0
        )

        bw_dict = {
            'rate': rate_bw,
            'attitude': attitude_bw,
            'ratio': ratio
        }

        if not is_valid:
            msg = f"Rate/Attitude bandwidth separation violated: {ratio:.2f}x (required: 4-15x)"
            return False, msg, bw_dict

        msg = f"Rate/Attitude separation: {ratio:.2f}x (rate: {rate_bw:.2f} Hz, attitude: {attitude_bw:.2f} Hz)"
        return True, msg, bw_dict

    def validate_phase3_position(self, params: Dict[str, float]) -> Tuple[bool, str, Dict]:
        """
        Validate position controller parameters (Phase 3)

        Enforces: Attitude BW > 3-8x Position BW

        Args:
            params: Position controller parameters

        Returns:
            Tuple of (is_valid, message, bandwidth_dict)
        """
        if self.optimized_attitude_params is None:
            logger.warning("Attitude parameters not set - cannot validate position loop separation")
            return True, "Attitude parameters not available", {}

        # Get rate loop bandwidth (for reference)
        if self.optimized_rate_params:
            rate_P = self.optimized_rate_params.get('ATC_RAT_RLL_P', 0.15)
            rate_I = self.optimized_rate_params.get('ATC_RAT_RLL_I', 0.15)
            rate_D = self.optimized_rate_params.get('ATC_RAT_RLL_D', 0.005)
            rate_bw = estimate_closed_loop_bandwidth(rate_P, rate_I, rate_D,
                                                     self.inertia_roll, self.motor_gain)
        else:
            rate_bw = 15.0  # Default

        # Get attitude loop bandwidth
        attitude_P = self.optimized_attitude_params.get('ATC_ANG_RLL_P', 4.5)
        attitude_bw = estimate_attitude_loop_bandwidth(attitude_P, rate_bw)

        # Get position loop bandwidth
        position_P = params.get('PSC_POSXY_P', 1.0)
        velocity_P = params.get('PSC_VELXY_P', 2.0)
        position_bw = estimate_position_loop_bandwidth(position_P, velocity_P, attitude_bw)

        # Check separation
        is_valid, ratio = check_bandwidth_separation(
            attitude_bw, position_bw, min_ratio=3.0, max_ratio=8.0
        )

        bw_dict = {
            'rate': rate_bw,
            'attitude': attitude_bw,
            'position': position_bw,
            'ratio': ratio
        }

        if not is_valid:
            msg = f"Attitude/Position bandwidth separation violated: {ratio:.2f}x (required: 3-8x)"
            return False, msg, bw_dict

        msg = f"Attitude/Position separation: {ratio:.2f}x (attitude: {attitude_bw:.2f} Hz, position: {position_bw:.2f} Hz)"
        return True, msg, bw_dict

    def validate_parameters(self, phase_name: str,
                           params: Dict[str, float]) -> Tuple[bool, str, Dict]:
        """
        Validate parameters for a given phase with hierarchical constraints

        Args:
            phase_name: Phase identifier ('phase1_rate', 'phase2_attitude', etc.)
            params: Parameter dictionary to validate

        Returns:
            Tuple of (is_valid, message, details_dict)
        """
        if not self.enforce_constraints:
            return True, "Constraints not enforced", {}

        if phase_name == 'phase1_rate':
            is_valid, msg, bw = self.validate_phase1_rate(params)
            return is_valid, msg, {'bandwidth': bw}

        elif phase_name == 'phase2_attitude':
            return self.validate_phase2_attitude(params)

        elif phase_name == 'phase3_position':
            return self.validate_phase3_position(params)

        elif phase_name == 'phase4_advanced':
            # No hierarchical constraints for advanced parameters
            return True, "No hierarchical constraints for phase 4", {}

        else:
            logger.warning(f"Unknown phase: {phase_name}")
            return True, "Unknown phase", {}

    def get_constraint_penalty(self, phase_name: str,
                              params: Dict[str, float]) -> float:
        """
        Calculate penalty for constraint violations

        Returns:
            Penalty value (0 if valid, large negative if invalid)
        """
        is_valid, msg, details = self.validate_parameters(phase_name, params)

        if is_valid:
            return 0.0
        else:
            # Large negative penalty for constraint violations
            penalty = -self.penalty_multiplier
            logger.debug(f"Constraint penalty applied: {penalty} ({msg})")
            return penalty

    def get_full_system_validation(self) -> Dict:
        """
        Validate complete cascade system with all optimized parameters

        Returns:
            Validation results dictionary
        """
        if not all([self.optimized_rate_params,
                   self.optimized_attitude_params,
                   self.optimized_position_params]):
            return {
                'valid': False,
                'message': 'Not all phases have been optimized yet'
            }

        # Use the comprehensive validation function
        results = validate_cascade_bandwidths(
            rate_params=self.optimized_rate_params,
            attitude_params=self.optimized_attitude_params,
            position_params=self.optimized_position_params,
            drone_params=self.drone_params,
            strict_mode=self.enforce_constraints
        )

        return results


# Convenience function for standalone validation
def validate_hierarchical_parameters(rate_params: Dict[str, float],
                                     attitude_params: Dict[str, float],
                                     position_params: Dict[str, float],
                                     drone_params: Dict[str, float]) -> Dict:
    """
    Standalone function to validate hierarchical parameter set

    Args:
        rate_params: Rate controller parameters
        attitude_params: Attitude controller parameters
        position_params: Position controller parameters
        drone_params: Drone physical parameters

    Returns:
        Validation results dictionary
    """
    return validate_cascade_bandwidths(
        rate_params=rate_params,
        attitude_params=attitude_params,
        position_params=position_params,
        drone_params=drone_params,
        strict_mode=True
    )
