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
