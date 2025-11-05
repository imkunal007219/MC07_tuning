"""
Physics-Based Population Seeding for Drone PID Tuning

Uses control theory principles to initialize optimization population
near stable regions instead of pure random initialization.

Based on:
- Natural frequency and damping calculations
- Bandwidth separation rules (inner loop 5-10x faster than outer)
- Ziegler-Nichols tuning formulas
- Classical control theory for cascaded loops
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
from math import pi, sqrt

logger = logging.getLogger(__name__)


class PhysicsBasedSeeder:
    """
    Generate initial parameter estimates using control theory
    """

    def __init__(self, drone_params: Dict):
        """
        Initialize seeder with drone physical parameters

        Args:
            drone_params: Dictionary containing:
                - mass: Total vehicle mass (kg)
                - Ixx: Roll axis inertia (kg·m²)
                - Iyy: Pitch axis inertia (kg·m²)
                - Izz: Yaw axis inertia (kg·m²)
                - max_thrust_per_motor: Maximum thrust per motor (N)
                - arm_length: Distance from center to motor (m)
        """
        self.mass = drone_params['mass']
        self.Ixx = drone_params['Ixx']
        self.Iyy = drone_params['Iyy']
        self.Izz = drone_params['Izz']
        self.max_thrust_per_motor = drone_params['max_thrust_per_motor']
        self.arm_length = drone_params['arm_length']

        # Calculate effective motor gain (torque per throttle command)
        # For a quad, torque = force * arm_length
        # Assuming 4 motors, max differential thrust = max_thrust_per_motor
        self.motor_torque_gain = self.max_thrust_per_motor * self.arm_length

        # Gravity constant
        self.g = 9.81  # m/s²

        logger.info("PhysicsBasedSeeder initialized")
        logger.info(f"  Mass: {self.mass} kg")
        logger.info(f"  Inertia - Ixx: {self.Ixx}, Iyy: {self.Iyy}, Izz: {self.Izz} kg·m²")
        logger.info(f"  Motor torque gain: {self.motor_torque_gain:.2f} N·m")

    def calculate_rate_controller_gains(self,
                                        desired_bandwidth_hz: float = 15.0,
                                        desired_damping: float = 0.707,
                                        axis: str = 'roll') -> Dict[str, float]:
        """
        Calculate rate (innermost) loop PID gains using control theory

        The rate controller is the fastest loop and directly controls angular velocity.

        Control law: τ = I * α_desired
        With PID: τ = Kp*e + Ki*∫e + Kd*ė

        For a second-order system: ω_n = sqrt(Kp/I), ζ = Kd/(2*sqrt(Kp*I))
        Bandwidth ≈ ω_n for critically damped systems

        Args:
            desired_bandwidth_hz: Desired closed-loop bandwidth in Hz (typically 10-20 Hz)
            desired_damping: Desired damping ratio (0.707 = critically damped)
            axis: 'roll', 'pitch', or 'yaw'

        Returns:
            Dictionary with P, I, D gains
        """
        # Select appropriate inertia
        if axis == 'roll':
            inertia = self.Ixx
        elif axis == 'pitch':
            inertia = self.Iyy
        elif axis == 'yaw':
            inertia = self.Izz
        else:
            raise ValueError(f"Unknown axis: {axis}")

        # Convert bandwidth to rad/s
        omega_n = 2 * pi * desired_bandwidth_hz

        # Calculate P gain from desired natural frequency
        # ω_n = sqrt(Kp * motor_gain / I)
        # Kp = I * ω_n² / motor_gain
        Kp = inertia * omega_n**2 / self.motor_torque_gain

        # Calculate D gain for desired damping
        # ζ = Kd * motor_gain / (2 * sqrt(Kp * motor_gain * I))
        # Simplify: ζ = Kd / (2 * sqrt(Kp * I))
        # Kd = 2 * ζ * sqrt(Kp * I)
        Kd = 2 * desired_damping * sqrt(Kp * inertia)

        # Calculate I gain (typically 10-30% of P for good disturbance rejection)
        # Higher I gain improves steady-state but can cause overshoot
        Ki = Kp * 0.2  # 20% of P gain

        logger.info(f"Rate controller ({axis}): P={Kp:.3f}, I={Ki:.3f}, D={Kd:.4f}")
        logger.info(f"  Natural frequency: {omega_n/(2*pi):.1f} Hz, Damping: {desired_damping:.3f}")

        return {'P': Kp, 'I': Ki, 'D': Kd}

    def calculate_attitude_controller_gains(self,
                                            rate_bandwidth_hz: float = 15.0,
                                            bandwidth_ratio: float = 5.0) -> Dict[str, float]:
        """
        Calculate attitude (middle) loop gains using bandwidth separation

        The attitude controller is a P-only controller that commands rate.
        It should be 3-5x slower than the rate loop for stability.

        For a P controller: rate_cmd = Kp * angle_error
        The effective bandwidth is approximately: BW = Kp (in rad/s)

        Args:
            rate_bandwidth_hz: Inner rate loop bandwidth in Hz
            bandwidth_ratio: Ratio of rate/attitude bandwidth (typically 4-6)

        Returns:
            Dictionary with P gain
        """
        # Attitude loop should be slower than rate loop
        attitude_bandwidth_hz = rate_bandwidth_hz / bandwidth_ratio

        # For a P controller, the gain directly relates to bandwidth
        # Kp ≈ 2π * BW (in rad/s per rad)
        # But in ArduPilot units, it's deg/s per degree, so no conversion needed
        Kp = 2 * pi * attitude_bandwidth_hz

        logger.info(f"Attitude controller: P={Kp:.3f}")
        logger.info(f"  Bandwidth: {attitude_bandwidth_hz:.1f} Hz (ratio: 1:{bandwidth_ratio:.1f} vs rate)")

        return {'P': Kp}

    def calculate_position_controller_gains(self,
                                           attitude_bandwidth_hz: float = 3.0,
                                           bandwidth_ratio: float = 4.0) -> Dict[str, float]:
        """
        Calculate position (outermost) loop gains using bandwidth separation

        Position control is cascaded: Position -> Velocity -> Acceleration -> Attitude
        Should be 3-5x slower than attitude loop.

        Args:
            attitude_bandwidth_hz: Middle attitude loop bandwidth in Hz
            bandwidth_ratio: Ratio of attitude/position bandwidth (typically 3-5)

        Returns:
            Dictionary with position and velocity controller gains
        """
        # Position loop should be slowest
        position_bandwidth_hz = attitude_bandwidth_hz / bandwidth_ratio

        # Position P gain (converts position error to velocity command)
        PSC_POSXY_P = 2 * pi * position_bandwidth_hz

        # Velocity loop - typically 2x faster than position loop
        velocity_bandwidth_hz = position_bandwidth_hz * 2.0
        omega_n_vel = 2 * pi * velocity_bandwidth_hz

        # Velocity controller PID gains
        # These convert velocity error to acceleration command
        PSC_VELXY_P = omega_n_vel / self.g  # Normalized by gravity
        PSC_VELXY_I = PSC_VELXY_P * 0.5     # I gain for wind rejection
        PSC_VELXY_D = PSC_VELXY_P * 0.1     # D gain for damping

        # Altitude controller (similar structure but separate tuning)
        PSC_POSZ_P = 2 * pi * position_bandwidth_hz
        PSC_VELZ_P = 2 * pi * velocity_bandwidth_hz

        # Acceleration controller for Z axis
        PSC_ACCZ_P = 0.5  # Typical value for thrust/accel conversion
        PSC_ACCZ_I = 1.0  # Higher I for hover thrust learning
        PSC_ACCZ_D = 0.0  # Usually zero to avoid noise amplification

        logger.info(f"Position controller gains:")
        logger.info(f"  POSXY_P: {PSC_POSXY_P:.3f} (BW: {position_bandwidth_hz:.2f} Hz)")
        logger.info(f"  VELXY: P={PSC_VELXY_P:.3f}, I={PSC_VELXY_I:.3f}, D={PSC_VELXY_D:.3f}")
        logger.info(f"  POSZ_P: {PSC_POSZ_P:.3f}, VELZ_P: {PSC_VELZ_P:.3f}")
        logger.info(f"  ACCZ: P={PSC_ACCZ_P:.3f}, I={PSC_ACCZ_I:.3f}")

        return {
            'PSC_POSXY_P': PSC_POSXY_P,
            'PSC_VELXY_P': PSC_VELXY_P,
            'PSC_VELXY_I': PSC_VELXY_I,
            'PSC_VELXY_D': PSC_VELXY_D,
            'PSC_POSZ_P': PSC_POSZ_P,
            'PSC_VELZ_P': PSC_VELZ_P,
            'PSC_ACCZ_P': PSC_ACCZ_P,
            'PSC_ACCZ_I': PSC_ACCZ_I,
            'PSC_ACCZ_D': PSC_ACCZ_D,
        }

    def calculate_filter_values(self,
                               bandwidth_hz: float,
                               filter_ratio: float = 5.0) -> Dict[str, float]:
        """
        Calculate filter cutoff frequencies

        Filters should be set 5-10x higher than controller bandwidth
        to avoid phase lag while filtering noise.

        Args:
            bandwidth_hz: Controller bandwidth in Hz
            filter_ratio: Ratio of filter/controller bandwidth (typically 5-10)

        Returns:
            Dictionary with filter cutoff frequencies
        """
        # Filter cutoff should be higher than bandwidth to minimize phase lag
        filter_cutoff_hz = bandwidth_hz * filter_ratio

        return {
            'FLTD': filter_cutoff_hz,  # D-term filter
            'FLTE': filter_cutoff_hz * 0.5,  # Error filter (more aggressive)
            'FLTT': filter_cutoff_hz,  # Target filter
        }

    def calculate_motor_parameters(self) -> Dict[str, float]:
        """
        Calculate motor-related parameters

        Returns:
            Dictionary with motor parameters
        """
        # Hover throttle estimation
        # Hover thrust = mass * g
        # Per motor = (mass * g) / 4
        hover_thrust_per_motor = (self.mass * self.g) / 4.0
        hover_throttle = hover_thrust_per_motor / self.max_thrust_per_motor

        # Ensure reasonable bounds
        hover_throttle = np.clip(hover_throttle, 0.2, 0.6)

        # Motor spin values
        MOT_SPIN_MIN = 0.15  # Minimum throttle for motor stability
        MOT_SPIN_MAX = 0.95  # Maximum throttle (leave headroom)
        MOT_THST_EXPO = 0.65  # Thrust curve expo (0.65 typical for large props)

        logger.info(f"Motor parameters:")
        logger.info(f"  Hover throttle: {hover_throttle:.3f} ({hover_throttle*100:.1f}%)")
        logger.info(f"  Spin min/max: {MOT_SPIN_MIN:.2f} / {MOT_SPIN_MAX:.2f}")

        return {
            'MOT_THST_HOVER': hover_throttle,
            'MOT_SPIN_MIN': MOT_SPIN_MIN,
            'MOT_SPIN_MAX': MOT_SPIN_MAX,
            'MOT_THST_EXPO': MOT_THST_EXPO,
        }

    def generate_initial_parameters(self) -> Dict[str, float]:
        """
        Generate complete set of initial parameters using control theory

        This creates a single "seed" parameter set that should be near
        stable operating point.

        Returns:
            Dictionary of all tunable parameters with physics-based initial values
        """
        logger.info("\n" + "="*60)
        logger.info("Generating physics-based initial parameters")
        logger.info("="*60)

        params = {}

        # ===== PHASE 1: RATE CONTROLLERS =====
        logger.info("\nPHASE 1: Rate Controllers (Inner Loop)")

        rate_bandwidth_hz = 15.0  # Target 15 Hz bandwidth for rate loop

        # Roll rate controller
        roll_gains = self.calculate_rate_controller_gains(
            rate_bandwidth_hz, desired_damping=0.707, axis='roll'
        )
        params['ATC_RAT_RLL_P'] = roll_gains['P']
        params['ATC_RAT_RLL_I'] = roll_gains['I']
        params['ATC_RAT_RLL_D'] = roll_gains['D']

        # Pitch rate controller (same as roll for symmetric quad)
        pitch_gains = self.calculate_rate_controller_gains(
            rate_bandwidth_hz, desired_damping=0.707, axis='pitch'
        )
        params['ATC_RAT_PIT_P'] = pitch_gains['P']
        params['ATC_RAT_PIT_I'] = pitch_gains['I']
        params['ATC_RAT_PIT_D'] = pitch_gains['D']

        # Yaw rate controller (typically slower, lower bandwidth)
        yaw_gains = self.calculate_rate_controller_gains(
            rate_bandwidth_hz * 0.7, desired_damping=0.707, axis='yaw'
        )
        params['ATC_RAT_YAW_P'] = yaw_gains['P']
        params['ATC_RAT_YAW_I'] = yaw_gains['I']

        # Rate controller filters
        rate_filters = self.calculate_filter_values(rate_bandwidth_hz, filter_ratio=5.0)
        params['ATC_RAT_RLL_FLTD'] = rate_filters['FLTD']
        params['ATC_RAT_RLL_FLTE'] = rate_filters['FLTE']
        params['ATC_RAT_RLL_FLTT'] = rate_filters['FLTT']
        params['ATC_RAT_PIT_FLTD'] = rate_filters['FLTD']
        params['ATC_RAT_PIT_FLTE'] = rate_filters['FLTE']
        params['ATC_RAT_PIT_FLTT'] = rate_filters['FLTT']
        params['ATC_RAT_YAW_FLTE'] = rate_filters['FLTE']
        params['ATC_RAT_YAW_FLTT'] = rate_filters['FLTT']

        # Gyro filter (should be high to minimize delay)
        params['INS_GYRO_FILTER'] = rate_bandwidth_hz * 4.0  # 4x rate bandwidth

        # ===== PHASE 2: ATTITUDE CONTROLLERS =====
        logger.info("\nPHASE 2: Attitude Controllers (Middle Loop)")

        attitude_bandwidth_hz = rate_bandwidth_hz / 5.0  # 5x slower than rate

        attitude_gains = self.calculate_attitude_controller_gains(
            rate_bandwidth_hz, bandwidth_ratio=5.0
        )
        params['ATC_ANG_RLL_P'] = attitude_gains['P']
        params['ATC_ANG_PIT_P'] = attitude_gains['P']
        params['ATC_ANG_YAW_P'] = attitude_gains['P'] * 0.8  # Yaw slightly lower

        # Acceleration limits (deg/s²) - based on motor capability
        # Max angular acceleration ≈ max_torque / inertia
        max_angular_accel_roll = (self.motor_torque_gain * 4) / self.Ixx  # rad/s²
        max_angular_accel_roll_deg = max_angular_accel_roll * (180/pi) * 100  # cdeg/s²

        params['ATC_ACCEL_R_MAX'] = np.clip(max_angular_accel_roll_deg, 50000, 180000)
        params['ATC_ACCEL_P_MAX'] = np.clip(max_angular_accel_roll_deg, 50000, 180000)
        params['ATC_ACCEL_Y_MAX'] = np.clip(max_angular_accel_roll_deg * 0.5, 10000, 40000)
        params['ATC_SLEW_YAW'] = 3000  # Yaw slew rate (cdeg/s²)

        # ===== PHASE 3: POSITION CONTROLLERS =====
        logger.info("\nPHASE 3: Position Controllers (Outer Loop)")

        position_gains = self.calculate_position_controller_gains(
            attitude_bandwidth_hz, bandwidth_ratio=4.0
        )
        params.update(position_gains)

        # Position controller filters
        pos_vel_bandwidth = attitude_bandwidth_hz / 2.0
        pos_filters = self.calculate_filter_values(pos_vel_bandwidth, filter_ratio=3.0)
        params['PSC_VELXY_FLTD'] = pos_filters['FLTD']
        params['PSC_VELXY_FLTE'] = pos_filters['FLTE']
        params['PSC_ACCZ_FLTD'] = pos_filters['FLTD']
        params['PSC_ACCZ_FLTE'] = pos_filters['FLTE']

        # ===== PHASE 4: MOTOR PARAMETERS =====
        logger.info("\nPHASE 4: Motor and Advanced Parameters")

        motor_params = self.calculate_motor_parameters()
        params.update(motor_params)

        # Additional parameters
        params['ATC_INPUT_TC'] = 0.15  # Input time constant (smoothing)
        params['ATC_THR_MIX_MAN'] = 0.5  # Throttle mix for manual flight

        logger.info("\n" + "="*60)
        logger.info("Physics-based parameter generation complete")
        logger.info(f"Generated {len(params)} parameters")
        logger.info("="*60 + "\n")

        return params

    def generate_population(self,
                           parameters: List[str],
                           bounds: Dict[str, Tuple[float, float]],
                           population_size: int,
                           seed_ratio: float = 0.3,
                           diversity_sigma: float = 0.15) -> List[List[float]]:
        """
        Generate initial population with physics-based seeding

        Strategy:
        - seed_ratio of population uses physics-based values with small variation
        - Remaining population is uniformly random for diversity

        Args:
            parameters: List of parameter names to optimize
            bounds: Dictionary of (min, max) bounds for each parameter
            population_size: Number of individuals to generate
            seed_ratio: Fraction of population to seed (0.0-1.0)
            diversity_sigma: Std dev for Gaussian noise around seed (as fraction of range)

        Returns:
            List of individuals (each individual is list of normalized [0,1] values)
        """
        logger.info(f"\nGenerating population of {population_size} individuals")
        logger.info(f"  Seeded individuals: {int(population_size * seed_ratio)} ({seed_ratio*100:.0f}%)")
        logger.info(f"  Random individuals: {population_size - int(population_size * seed_ratio)}")

        # Generate physics-based seed values
        seed_params = self.generate_initial_parameters()

        population = []
        n_seeded = int(population_size * seed_ratio)

        # Create seeded individuals
        for i in range(n_seeded):
            individual = []
            for param_name in parameters:
                # Get physics-based value
                if param_name in seed_params:
                    seed_value = seed_params[param_name]
                else:
                    # If parameter not in seed, use middle of range
                    min_val, max_val = bounds[param_name]
                    seed_value = (min_val + max_val) / 2.0
                    logger.warning(f"Parameter {param_name} not in seed, using midpoint")

                # Normalize to [0, 1]
                min_val, max_val = bounds[param_name]
                param_range = max_val - min_val
                normalized_seed = (seed_value - min_val) / param_range

                # Add Gaussian noise for diversity
                noise = np.random.normal(0, diversity_sigma)
                normalized_value = normalized_seed + noise

                # Clip to [0, 1]
                normalized_value = np.clip(normalized_value, 0.0, 1.0)
                individual.append(normalized_value)

            population.append(individual)

        # Create random individuals for diversity
        for i in range(population_size - n_seeded):
            individual = [np.random.random() for _ in parameters]
            population.append(individual)

        logger.info(f"Population generation complete: {len(population)} individuals")

        return population


def apply_ziegler_nichols(ultimate_gain: float,
                          ultimate_period: float,
                          method: str = 'classic') -> Dict[str, float]:
    """
    Apply Ziegler-Nichols tuning rules

    Classic ZN method requires determining ultimate gain (Ku) and period (Tu)
    where system oscillates at steady state.

    Args:
        ultimate_gain: Ku - gain at which system oscillates
        ultimate_period: Tu - period of oscillation (seconds)
        method: 'classic' or 'modified' ZN rules

    Returns:
        Dictionary with P, I, D gains
    """
    if method == 'classic':
        # Classic Ziegler-Nichols PID rules
        Kp = 0.6 * ultimate_gain
        Ki = 1.2 * ultimate_gain / ultimate_period
        Kd = 0.075 * ultimate_gain * ultimate_period
    elif method == 'modified':
        # Modified ZN (less aggressive)
        Kp = 0.33 * ultimate_gain
        Ki = 0.66 * ultimate_gain / ultimate_period
        Kd = 0.11 * ultimate_gain * ultimate_period
    else:
        raise ValueError(f"Unknown ZN method: {method}")

    return {'P': Kp, 'I': Ki, 'D': Kd}


def calculate_settling_time(natural_freq: float, damping_ratio: float) -> float:
    """
    Calculate 2% settling time for second-order system

    t_s ≈ 4 / (ζ * ω_n) for 2% criterion

    Args:
        natural_freq: Natural frequency (rad/s)
        damping_ratio: Damping ratio (dimensionless)

    Returns:
        Settling time in seconds
    """
    if damping_ratio <= 0 or natural_freq <= 0:
        return float('inf')

    return 4.0 / (damping_ratio * natural_freq)


def calculate_phase_margin(Kp: float, Ki: float, Kd: float,
                          inertia: float, frequency: float) -> float:
    """
    Calculate phase margin at given frequency for PID controller

    Simplified calculation for loop stability assessment

    Args:
        Kp, Ki, Kd: PID gains
        inertia: System inertia
        frequency: Frequency to evaluate (rad/s)

    Returns:
        Phase margin in degrees
    """
    # Transfer function: G(s) = (Kd*s^2 + Kp*s + Ki) / (I*s^2)
    # At frequency ω: G(jω)

    s = 1j * frequency

    # PID numerator
    num = Kd * s**2 + Kp * s + Ki

    # Plant denominator (double integrator)
    den = inertia * s**2

    # Loop transfer function
    L = num / den

    # Phase margin = 180° + phase(L)
    phase_L = np.angle(L, deg=True)
    phase_margin = 180 + phase_L

    return phase_margin


def generate_narrow_bounds(drone_params: Dict,
                          tolerance: float = 0.3,
                          apply_stability_rules: bool = True) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Generate narrow parameter bounds based on physics calculations

    Creates tighter bounds centered on physics-based expected values,
    reducing search space while maintaining stability margins.

    Args:
        drone_params: Dictionary of drone physical parameters
        tolerance: Fractional tolerance around expected value (0.3 = ±30%)
        apply_stability_rules: Whether to enforce additional stability constraints

    Returns:
        Dictionary organized by phase with narrow bounds for each parameter
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING NARROW BOUNDS FROM PHYSICS")
    logger.info("="*80)
    logger.info(f"Tolerance: ±{tolerance*100:.0f}% around physics-based values")

    # Generate physics-based initial parameters
    seeder = PhysicsBasedSeeder(drone_params)
    expected_params = seeder.generate_initial_parameters()

    narrow_bounds = {}

    # ===== PHASE 1: RATE CONTROLLERS =====
    phase1_bounds = {}

    # Roll/Pitch rate P gains
    P_expected = expected_params['ATC_RAT_RLL_P']
    phase1_bounds['ATC_RAT_RLL_P'] = (
        P_expected * (1 - tolerance),
        P_expected * (1 + tolerance)
    )
    phase1_bounds['ATC_RAT_PIT_P'] = (
        P_expected * (1 - tolerance),
        P_expected * (1 + tolerance)
    )

    # Roll/Pitch rate I gains (constrained to 0.5-2.0 × P gain)
    I_expected = expected_params['ATC_RAT_RLL_I']
    if apply_stability_rules:
        I_min = max(I_expected * (1 - tolerance), P_expected * 0.5 * (1 - tolerance))
        I_max = min(I_expected * (1 + tolerance), P_expected * 2.0 * (1 + tolerance))
    else:
        I_min = I_expected * (1 - tolerance)
        I_max = I_expected * (1 + tolerance)
    phase1_bounds['ATC_RAT_RLL_I'] = (I_min, I_max)
    phase1_bounds['ATC_RAT_PIT_I'] = (I_min, I_max)

    # Roll/Pitch rate D gains (constrained to < P gain / 10 for stability)
    D_expected = expected_params['ATC_RAT_RLL_D']
    if apply_stability_rules:
        D_max = min(D_expected * (1 + tolerance), P_expected / 10.0)
        D_min = D_expected * (1 - tolerance)
    else:
        D_min = D_expected * (1 - tolerance)
        D_max = D_expected * (1 + tolerance)
    phase1_bounds['ATC_RAT_RLL_D'] = (D_min, D_max)
    phase1_bounds['ATC_RAT_PIT_D'] = (D_min, D_max)

    # Yaw rate gains (typically 0.7x roll/pitch)
    yaw_P_expected = expected_params['ATC_RAT_YAW_P']
    phase1_bounds['ATC_RAT_YAW_P'] = (
        yaw_P_expected * (1 - tolerance),
        yaw_P_expected * (1 + tolerance)
    )
    yaw_I_expected = expected_params['ATC_RAT_YAW_I']
    phase1_bounds['ATC_RAT_YAW_I'] = (
        yaw_I_expected * (1 - tolerance),
        yaw_I_expected * (1 + tolerance)
    )

    # Filters - must respect hierarchy: gyro_filter > d_filter > error_filter
    gyro_filter = expected_params['INS_GYRO_FILTER']
    phase1_bounds['INS_GYRO_FILTER'] = (
        gyro_filter * (1 - tolerance * 0.5),  # Tighter tolerance on gyro filter
        gyro_filter * (1 + tolerance * 0.5)
    )

    # D-term filter (must be < 0.5 × gyro filter)
    d_filter = expected_params['ATC_RAT_RLL_FLTD']
    if apply_stability_rules:
        d_filter_max = min(d_filter * (1 + tolerance), gyro_filter * 0.5)
    else:
        d_filter_max = d_filter * (1 + tolerance)
    phase1_bounds['ATC_RAT_RLL_FLTD'] = (d_filter * (1 - tolerance), d_filter_max)
    phase1_bounds['ATC_RAT_PIT_FLTD'] = (d_filter * (1 - tolerance), d_filter_max)

    # Error filter (typically lower than D filter)
    e_filter = expected_params['ATC_RAT_RLL_FLTE']
    if apply_stability_rules:
        e_filter_max = min(e_filter * (1 + tolerance), d_filter * 0.8)
    else:
        e_filter_max = e_filter * (1 + tolerance)
    phase1_bounds['ATC_RAT_RLL_FLTE'] = (0.0, e_filter_max)  # Can be 0
    phase1_bounds['ATC_RAT_PIT_FLTE'] = (0.0, e_filter_max)
    phase1_bounds['ATC_RAT_YAW_FLTE'] = (0.0, e_filter_max)

    # Target filter (similar to D filter)
    t_filter = expected_params['ATC_RAT_RLL_FLTT']
    phase1_bounds['ATC_RAT_RLL_FLTT'] = (t_filter * (1 - tolerance), t_filter * (1 + tolerance))
    phase1_bounds['ATC_RAT_PIT_FLTT'] = (t_filter * (1 - tolerance), t_filter * (1 + tolerance))
    phase1_bounds['ATC_RAT_YAW_FLTT'] = (t_filter * (1 - tolerance), t_filter * (1 + tolerance))

    narrow_bounds['phase1_rate'] = phase1_bounds

    # ===== PHASE 2: ATTITUDE CONTROLLERS =====
    phase2_bounds = {}

    ang_p_expected = expected_params['ATC_ANG_RLL_P']
    phase2_bounds['ATC_ANG_RLL_P'] = (
        ang_p_expected * (1 - tolerance),
        ang_p_expected * (1 + tolerance)
    )
    phase2_bounds['ATC_ANG_PIT_P'] = (
        ang_p_expected * (1 - tolerance),
        ang_p_expected * (1 + tolerance)
    )
    yaw_ang_p = expected_params['ATC_ANG_YAW_P']
    phase2_bounds['ATC_ANG_YAW_P'] = (
        yaw_ang_p * (1 - tolerance),
        yaw_ang_p * (1 + tolerance)
    )

    # Acceleration limits
    accel_r = expected_params['ATC_ACCEL_R_MAX']
    phase2_bounds['ATC_ACCEL_R_MAX'] = (
        accel_r * (1 - tolerance * 0.5),  # Tighter tolerance
        accel_r * (1 + tolerance * 0.5)
    )
    phase2_bounds['ATC_ACCEL_P_MAX'] = (
        accel_r * (1 - tolerance * 0.5),
        accel_r * (1 + tolerance * 0.5)
    )
    accel_y = expected_params['ATC_ACCEL_Y_MAX']
    phase2_bounds['ATC_ACCEL_Y_MAX'] = (
        accel_y * (1 - tolerance * 0.5),
        accel_y * (1 + tolerance * 0.5)
    )
    slew_yaw = expected_params['ATC_SLEW_YAW']
    phase2_bounds['ATC_SLEW_YAW'] = (
        slew_yaw * (1 - tolerance),
        slew_yaw * (1 + tolerance)
    )

    narrow_bounds['phase2_attitude'] = phase2_bounds

    # ===== PHASE 3: POSITION CONTROLLERS =====
    phase3_bounds = {}

    for param_name in ['PSC_POSXY_P', 'PSC_VELXY_P', 'PSC_VELXY_I', 'PSC_VELXY_D',
                       'PSC_POSZ_P', 'PSC_VELZ_P', 'PSC_ACCZ_P', 'PSC_ACCZ_I', 'PSC_ACCZ_D']:
        expected = expected_params[param_name]
        phase3_bounds[param_name] = (
            expected * (1 - tolerance),
            expected * (1 + tolerance)
        )

    # Position filters
    velxy_fltd = expected_params['PSC_VELXY_FLTD']
    phase3_bounds['PSC_VELXY_FLTD'] = (
        velxy_fltd * (1 - tolerance),
        velxy_fltd * (1 + tolerance)
    )
    phase3_bounds['PSC_VELXY_FLTE'] = (
        velxy_fltd * (1 - tolerance),
        velxy_fltd * (1 + tolerance)
    )
    accz_fltd = expected_params['PSC_ACCZ_FLTD']
    phase3_bounds['PSC_ACCZ_FLTD'] = (
        accz_fltd * (1 - tolerance),
        accz_fltd * (1 + tolerance)
    )
    phase3_bounds['PSC_ACCZ_FLTE'] = (
        accz_fltd * (1 - tolerance),
        accz_fltd * (1 + tolerance)
    )

    narrow_bounds['phase3_position'] = phase3_bounds

    # ===== PHASE 4: MOTOR PARAMETERS =====
    phase4_bounds = {}

    for param_name in ['MOT_THST_HOVER', 'MOT_SPIN_MIN', 'MOT_SPIN_MAX',
                       'MOT_THST_EXPO', 'ATC_INPUT_TC', 'ATC_THR_MIX_MAN']:
        expected = expected_params[param_name]
        # Motor params get tighter tolerance (±20%)
        param_tolerance = tolerance * 0.67  # ~20% if tolerance is 30%
        phase4_bounds[param_name] = (
            expected * (1 - param_tolerance),
            expected * (1 + param_tolerance)
        )

    narrow_bounds['phase4_advanced'] = phase4_bounds

    # Log summary
    logger.info("\nNarrow bounds generated:")
    for phase_name, bounds_dict in narrow_bounds.items():
        logger.info(f"\n{phase_name}: {len(bounds_dict)} parameters")
        for param, (min_val, max_val) in sorted(bounds_dict.items())[:3]:  # Show first 3
            range_pct = ((max_val - min_val) / min_val) * 100 if min_val > 0 else 0
            logger.info(f"  {param}: [{min_val:.4f}, {max_val:.4f}] (±{range_pct/2:.1f}%)")

    logger.info("\n" + "="*80)
    logger.info(f"Narrow bounds complete: {sum(len(b) for b in narrow_bounds.values())} parameters")
    logger.info("="*80 + "\n")

    return narrow_bounds


def estimate_closed_loop_bandwidth(Kp: float, Ki: float, Kd: float,
                                   inertia: float,
                                   motor_gain: float = 1.0) -> float:
    """
    Estimate closed-loop bandwidth for a PID-controlled second-order system

    For a PID controller on a double integrator (typical for attitude control):
    - Open loop: G(s) = motor_gain / (I * s^2)
    - Controller: C(s) = Kd*s^2 + Kp*s + Ki
    - Closed-loop bandwidth ≈ natural frequency ω_n for critically damped

    Args:
        Kp: Proportional gain
        Ki: Integral gain
        Kd: Derivative gain
        inertia: System inertia (kg·m²)
        motor_gain: Motor torque gain (N·m per control unit)

    Returns:
        Estimated closed-loop bandwidth in Hz
    """
    # For PID on double integrator: ω_n ≈ sqrt(Kp * motor_gain / inertia)
    if Kp <= 0 or inertia <= 0:
        return 0.0

    omega_n = sqrt(Kp * motor_gain / inertia)  # rad/s
    bandwidth_hz = omega_n / (2 * pi)

    return bandwidth_hz


def estimate_attitude_loop_bandwidth(Kp: float, rate_loop_bandwidth_hz: float) -> float:
    """
    Estimate attitude loop bandwidth

    Attitude loop is a P-only controller that commands rate.
    The closed-loop bandwidth is approximately Kp for well-tuned systems.

    Args:
        Kp: Attitude P gain (deg/s per degree)
        rate_loop_bandwidth_hz: Inner rate loop bandwidth in Hz

    Returns:
        Estimated closed-loop bandwidth in Hz
    """
    if Kp <= 0:
        return 0.0

    # For P controller: BW ≈ Kp / (2π) in Hz
    # But limited by inner rate loop bandwidth
    attitude_bw_hz = Kp / (2 * pi)

    # Cannot exceed rate loop bandwidth / 5 (cascade rule)
    max_attitude_bw = rate_loop_bandwidth_hz / 5.0
    attitude_bw_hz = min(attitude_bw_hz, max_attitude_bw)

    return attitude_bw_hz


def estimate_position_loop_bandwidth(position_P: float,
                                    velocity_P: float,
                                    attitude_loop_bandwidth_hz: float) -> float:
    """
    Estimate position loop bandwidth

    Position loop: Position P controller -> Velocity PID -> Attitude

    Args:
        position_P: Position P gain (m/s per meter)
        velocity_P: Velocity P gain
        attitude_loop_bandwidth_hz: Middle attitude loop bandwidth in Hz

    Returns:
        Estimated closed-loop bandwidth in Hz
    """
    if position_P <= 0 or velocity_P <= 0:
        return 0.0

    # Position loop BW ≈ position_P / (2π)
    position_bw_hz = position_P / (2 * pi)

    # Cannot exceed attitude loop bandwidth / 4 (cascade rule)
    max_position_bw = attitude_loop_bandwidth_hz / 4.0
    position_bw_hz = min(position_bw_hz, max_position_bw)

    return position_bw_hz


def check_bandwidth_separation(inner_bw_hz: float,
                               outer_bw_hz: float,
                               min_ratio: float = 5.0,
                               max_ratio: float = 10.0) -> Tuple[bool, float]:
    """
    Check if bandwidth separation meets cascade control requirements

    For stable cascade control, inner loop must be significantly faster
    than outer loop (typically 5-10x).

    Args:
        inner_bw_hz: Inner loop bandwidth in Hz
        outer_bw_hz: Outer loop bandwidth in Hz
        min_ratio: Minimum required ratio (default 5.0)
        max_ratio: Maximum recommended ratio (default 10.0)

    Returns:
        Tuple of (is_valid, actual_ratio)
    """
    if inner_bw_hz <= 0 or outer_bw_hz <= 0:
        return False, 0.0

    ratio = inner_bw_hz / outer_bw_hz

    # Check if ratio is within acceptable range
    is_valid = min_ratio <= ratio <= max_ratio

    return is_valid, ratio


def validate_cascade_bandwidths(rate_params: Dict[str, float],
                                attitude_params: Dict[str, float],
                                position_params: Dict[str, float],
                                drone_params: Dict[str, float],
                                strict_mode: bool = True) -> Dict[str, any]:
    """
    Validate bandwidth separation across all cascade control loops

    Args:
        rate_params: Rate controller parameters (ATC_RAT_RLL_P, I, D, etc.)
        attitude_params: Attitude controller parameters (ATC_ANG_RLL_P, etc.)
        position_params: Position controller parameters (PSC_POSXY_P, PSC_VELXY_P, etc.)
        drone_params: Drone physical parameters (Ixx, Iyy, etc.)
        strict_mode: If True, enforce strict separation constraints

    Returns:
        Dictionary with validation results and bandwidth estimates
    """
    logger.info("Validating cascade control bandwidth separation...")

    # Calculate motor torque gain
    max_thrust = drone_params.get('max_thrust_per_motor', 75.0)
    arm_length = drone_params.get('arm_length', 0.75)
    motor_gain = max_thrust * arm_length

    results = {
        'valid': True,
        'violations': [],
        'bandwidths': {},
        'ratios': {}
    }

    # ===== RATE LOOP BANDWIDTH =====
    rate_P = rate_params.get('ATC_RAT_RLL_P', 0.15)
    rate_I = rate_params.get('ATC_RAT_RLL_I', 0.15)
    rate_D = rate_params.get('ATC_RAT_RLL_D', 0.005)
    inertia_roll = drone_params.get('Ixx', 6.0)

    rate_bw = estimate_closed_loop_bandwidth(rate_P, rate_I, rate_D, inertia_roll, motor_gain)
    results['bandwidths']['rate'] = rate_bw

    logger.info(f"  Rate loop bandwidth: {rate_bw:.2f} Hz")

    # ===== ATTITUDE LOOP BANDWIDTH =====
    attitude_P = attitude_params.get('ATC_ANG_RLL_P', 4.5)

    attitude_bw = estimate_attitude_loop_bandwidth(attitude_P, rate_bw)
    results['bandwidths']['attitude'] = attitude_bw

    logger.info(f"  Attitude loop bandwidth: {attitude_bw:.2f} Hz")

    # Check rate/attitude separation
    rate_attitude_valid, rate_attitude_ratio = check_bandwidth_separation(
        rate_bw, attitude_bw, min_ratio=4.0, max_ratio=15.0
    )
    results['ratios']['rate_to_attitude'] = rate_attitude_ratio

    if not rate_attitude_valid:
        msg = f"Rate/Attitude bandwidth separation violated: {rate_attitude_ratio:.2f}x (required: 4-15x)"
        results['violations'].append(msg)
        logger.warning(f"  ✗ {msg}")
        if strict_mode:
            results['valid'] = False
    else:
        logger.info(f"  ✓ Rate/Attitude separation: {rate_attitude_ratio:.2f}x")

    # ===== POSITION LOOP BANDWIDTH =====
    position_P = position_params.get('PSC_POSXY_P', 1.0)
    velocity_P = position_params.get('PSC_VELXY_P', 2.0)

    position_bw = estimate_position_loop_bandwidth(position_P, velocity_P, attitude_bw)
    results['bandwidths']['position'] = position_bw

    logger.info(f"  Position loop bandwidth: {position_bw:.2f} Hz")

    # Check attitude/position separation
    attitude_position_valid, attitude_position_ratio = check_bandwidth_separation(
        attitude_bw, position_bw, min_ratio=3.0, max_ratio=8.0
    )
    results['ratios']['attitude_to_position'] = attitude_position_ratio

    if not attitude_position_valid:
        msg = f"Attitude/Position bandwidth separation violated: {attitude_position_ratio:.2f}x (required: 3-8x)"
        results['violations'].append(msg)
        logger.warning(f"  ✗ {msg}")
        if strict_mode:
            results['valid'] = False
    else:
        logger.info(f"  ✓ Attitude/Position separation: {attitude_position_ratio:.2f}x")

    # Overall summary
    if results['valid']:
        logger.info("✓ All bandwidth separation constraints satisfied")
    else:
        logger.error(f"✗ Bandwidth validation failed: {len(results['violations'])} violations")

    return results
