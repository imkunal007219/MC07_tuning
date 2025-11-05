"""
Configuration file for automated drone tuning system
"""

import numpy as np
import os

# Get project root directory (parent of optimization_system)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# DRONE PHYSICAL PARAMETERS
# ============================================================================
DRONE_PARAMS = {
    'mass': 30.0,              # kg
    'Ixx': 6.0,                # kg·m²
    'Iyy': 6.0,                # kg·m²
    'Izz': 10.0,               # kg·m²
    'arm_length': 0.75,        # m
    'frame_height': 0.1,       # m
    'disc_area': 1.5,          # m²
    'thrust_coefficient': 2.86e-5,  # N/(rad/s)²
    'torque_coefficient': 4.77e-7,  # Nm/(rad/s)²
    'prop_expo': 0.75,
    'max_thrust_per_motor': 75.0,  # N
}

# ============================================================================
# OPTIMIZATION CONFIGURATION
# ============================================================================
OPTIMIZATION_CONFIG = {
    'algorithm': 'genetic',    # 'genetic' or 'bayesian'
    'parallel_instances': 10,  # Number of parallel SITL instances
    'max_generations': 100,    # For genetic algorithm
    'population_size': 50,     # For genetic algorithm
    'mutation_rate': 0.1,
    'crossover_rate': 0.7,
    'elite_size': 5,
    'timeout_per_sim': 120,    # seconds
}

# ============================================================================
# HIERARCHICAL OPTIMIZATION PHASES
# ============================================================================
OPTIMIZATION_PHASES = {
    'phase1_rate': {
        'name': 'Rate Controllers (Inner Loop)',
        'priority': 1,
        'parameters': [
            # Roll Rate
            'ATC_RAT_RLL_P',
            'ATC_RAT_RLL_I',
            'ATC_RAT_RLL_D',
            'ATC_RAT_RLL_FLTD',
            'ATC_RAT_RLL_FLTE',
            'ATC_RAT_RLL_FLTT',
            # Pitch Rate
            'ATC_RAT_PIT_P',
            'ATC_RAT_PIT_I',
            'ATC_RAT_PIT_D',
            'ATC_RAT_PIT_FLTD',
            'ATC_RAT_PIT_FLTE',
            'ATC_RAT_PIT_FLTT',
            # Yaw Rate
            'ATC_RAT_YAW_P',
            'ATC_RAT_YAW_I',
            'ATC_RAT_YAW_FLTE',
            'ATC_RAT_YAW_FLTT',
            # Gyro Filter
            'INS_GYRO_FILTER',
        ],
        'bounds': {
            'ATC_RAT_RLL_P': (0.05, 0.5),
            'ATC_RAT_RLL_I': (0.05, 0.5),
            'ATC_RAT_RLL_D': (0.001, 0.02),
            'ATC_RAT_RLL_FLTD': (5.0, 50.0),
            'ATC_RAT_RLL_FLTE': (0.0, 5.0),
            'ATC_RAT_RLL_FLTT': (5.0, 50.0),
            'ATC_RAT_PIT_P': (0.05, 0.5),
            'ATC_RAT_PIT_I': (0.05, 0.5),
            'ATC_RAT_PIT_D': (0.001, 0.02),
            'ATC_RAT_PIT_FLTD': (5.0, 50.0),
            'ATC_RAT_PIT_FLTE': (0.0, 5.0),
            'ATC_RAT_PIT_FLTT': (5.0, 50.0),
            'ATC_RAT_YAW_P': (0.1, 1.0),
            'ATC_RAT_YAW_I': (0.01, 0.2),
            'ATC_RAT_YAW_FLTE': (0.0, 5.0),
            'ATC_RAT_YAW_FLTT': (5.0, 50.0),
            'INS_GYRO_FILTER': (10.0, 80.0),
        }
    },
    'phase2_attitude': {
        'name': 'Attitude Controllers (Middle Loop)',
        'priority': 2,
        'parameters': [
            'ATC_ANG_RLL_P',
            'ATC_ANG_PIT_P',
            'ATC_ANG_YAW_P',
            'ATC_ACCEL_R_MAX',
            'ATC_ACCEL_P_MAX',
            'ATC_ACCEL_Y_MAX',
            'ATC_SLEW_YAW',
        ],
        'bounds': {
            'ATC_ANG_RLL_P': (2.0, 10.0),
            'ATC_ANG_PIT_P': (2.0, 10.0),
            'ATC_ANG_YAW_P': (2.0, 10.0),
            'ATC_ACCEL_R_MAX': (50000, 200000),
            'ATC_ACCEL_P_MAX': (50000, 200000),
            'ATC_ACCEL_Y_MAX': (10000, 50000),
            'ATC_SLEW_YAW': (1000, 6000),
        }
    },
    'phase3_position': {
        'name': 'Position Controllers (Outer Loop)',
        'priority': 3,
        'parameters': [
            # Position XY
            'PSC_POSXY_P',
            'PSC_VELXY_P',
            'PSC_VELXY_I',
            'PSC_VELXY_D',
            'PSC_VELXY_FLTD',
            'PSC_VELXY_FLTE',
            # Position Z
            'PSC_POSZ_P',
            'PSC_VELZ_P',
            'PSC_ACCZ_P',
            'PSC_ACCZ_I',
            'PSC_ACCZ_D',
            'PSC_ACCZ_FLTD',
            'PSC_ACCZ_FLTE',
        ],
        'bounds': {
            'PSC_POSXY_P': (0.5, 3.0),
            'PSC_VELXY_P': (1.0, 6.0),
            'PSC_VELXY_I': (0.2, 2.0),
            'PSC_VELXY_D': (0.1, 1.0),
            'PSC_VELXY_FLTD': (2.0, 10.0),
            'PSC_VELXY_FLTE': (2.0, 10.0),
            'PSC_POSZ_P': (0.5, 3.0),
            'PSC_VELZ_P': (1.0, 10.0),
            'PSC_ACCZ_P': (0.1, 1.0),
            'PSC_ACCZ_I': (0.5, 3.0),
            'PSC_ACCZ_D': (0.0, 0.1),
            'PSC_ACCZ_FLTD': (5.0, 20.0),
            'PSC_ACCZ_FLTE': (5.0, 20.0),
        }
    },
    'phase4_advanced': {
        'name': 'Advanced Parameters',
        'priority': 4,
        'parameters': [
            'MOT_THST_HOVER',
            'MOT_SPIN_MIN',
            'MOT_SPIN_MAX',
            'MOT_THST_EXPO',
            'ATC_INPUT_TC',
            'ATC_THR_MIX_MAN',
        ],
        'bounds': {
            'MOT_THST_HOVER': (0.2, 0.5),
            'MOT_SPIN_MIN': (0.10, 0.20),
            'MOT_SPIN_MAX': (0.90, 0.99),
            'MOT_THST_EXPO': (0.5, 0.8),
            'ATC_INPUT_TC': (0.1, 0.5),
            'ATC_THR_MIX_MAN': (0.1, 0.9),
        }
    }
}

# ============================================================================
# FITNESS FUNCTION WEIGHTS
# ============================================================================
# Weights for fitness function components (must sum to ~1.0)
# Frequency-domain weights ensure formal control-theoretic stability
FITNESS_WEIGHTS = {
    'stability': 0.25,              # Time-domain stability (oscillations, overshoot)
    'response_time': 0.20,          # Rise and settling time
    'tracking': 0.15,               # Position and attitude tracking accuracy
    'phase_margin': 0.15,           # Frequency-domain: formal stability margin (target: 45-60°)
    'gain_margin': 0.10,            # Frequency-domain: robustness to gain variations (target: >6 dB)
    'power_efficiency': 0.05,       # Power consumption
    'smoothness': 0.10,             # Control smoothness (motor saturation)
}

# ============================================================================
# PERFORMANCE METRICS THRESHOLDS
# ============================================================================
PERFORMANCE_THRESHOLDS = {
    # Time-domain thresholds
    'max_rise_time': 1.5,          # seconds
    'max_settling_time': 3.0,      # seconds
    'max_overshoot': 20.0,         # percent
    'max_steady_state_error': 2.0, # percent
    'max_oscillation_amplitude': 5.0,  # degrees

    # Frequency-domain thresholds (formal stability criteria)
    'min_phase_margin': 45.0,      # degrees (textbook: 30-60° for satisfactory performance)
    'target_phase_margin': 52.5,   # degrees (optimal: middle of range)
    'min_gain_margin': 6.0,        # dB (textbook: >6 dB for satisfactory performance)
    'target_gain_margin': 9.0,     # dB (optimal: healthy margin)
}

# ============================================================================
# SAFETY CONSTRAINTS
# ============================================================================
SAFETY_CONSTRAINTS = {
    'max_angle': 45,               # degrees
    'max_rate': 360,               # deg/s
    'max_altitude_error': 2.0,     # meters
    'max_position_error': 5.0,     # meters
    'min_phase_margin': 45,        # degrees
    'max_oscillation_amp': 5.0,    # degrees
    'max_motor_output': 0.95,      # 0-1
    'min_battery_voltage': 42.0,   # volts
}

# ============================================================================
# TEST SEQUENCE CONFIGURATION
# ============================================================================
TEST_SEQUENCES = {
    'basic_stability': {
        'duration': 5.0,           # seconds
        'altitude': 5.0,           # meters
        'expected_error': 0.5,     # meters
    },
    'step_response_roll': {
        'amplitude': 10.0,         # degrees
        'duration': 5.0,
    },
    'step_response_pitch': {
        'amplitude': 10.0,         # degrees
        'duration': 5.0,
    },
    'step_response_yaw': {
        'amplitude': 45.0,         # degrees
        'duration': 5.0,
    },
    'step_response_altitude': {
        'amplitude': 2.0,          # meters
        'duration': 5.0,
    },
    'frequency_sweep': {
        'freq_min': 0.1,           # Hz
        'freq_max': 10.0,          # Hz
        'duration': 30.0,
    },
    'trajectory_figure8': {
        'radius': 5.0,             # meters
        'altitude': 5.0,
        'velocity': 2.0,           # m/s
    },
    'disturbance_rejection': {
        'wind_speed': 5.0,         # m/s
        'gust_duration': 2.0,      # seconds
        'altitude': 5.0,
    },
}

# ============================================================================
# SITL CONFIGURATION
# ============================================================================
SITL_CONFIG = {
    'ardupilot_path': os.path.join(PROJECT_ROOT, 'ardupilot'),
    'vehicle': 'ArduCopter',
    'frame': 'quad',
    'home_location': '-35.363261,149.165230,584,353',  # Canberra
    'speedup': 1,              # Real-time for initial testing
    'defaults': 'copter-30kg.parm',
    'model': 'quad',
    'console': False,
    'map': False,
}

# ============================================================================
# MAVLINK CONFIGURATION
# ============================================================================
MAVLINK_CONFIG = {
    'connection_timeout': 30,
    'heartbeat_timeout': 10,
    'command_timeout': 10,
    'default_altitude': 5.0,
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    'log_dir': os.path.join(PROJECT_ROOT, 'logs'),
    'save_all_telemetry': True,
    'save_performance_metrics': True,
    'save_crash_logs': True,
    'log_level': 'INFO',
}

# ============================================================================
# CONVERGENCE CRITERIA
# ============================================================================
CONVERGENCE_CONFIG = {
    'min_generations': 20,
    'fitness_improvement_threshold': 0.01,  # 1% improvement
    'stagnation_generations': 10,            # Stop if no improvement
    'target_fitness': 0.95,                  # Stop if reached
}

# ============================================================================
# EARLY CRASH DETECTION CONFIGURATION
# ============================================================================
EARLY_CRASH_DETECTION_CONFIG = {
    'enabled': True,
    'check_interval': 0.5,          # seconds
    'min_samples': 50,               # minimum data points before checking
    'sensitivity': 'medium',         # 'low', 'medium', 'high'
    'expected_time_savings': 0.75,   # 75% reduction in crash time (2 min → 30 sec)
}

# ============================================================================
# INTELLIGENT TEST SEQUENCING CONFIGURATION
# ============================================================================
INTELLIGENT_TEST_SEQUENCING_CONFIG = {
    'enabled': True,
    'min_pass_score': 60.0,          # minimum score to pass a stage (0-100)
    'enable_optional_tests': True,    # run optional tests (frequency, trajectory)
    'early_termination': True,        # stop testing if fundamentals fail
    'expected_time_savings': 0.40,    # 40% reduction in evaluation time
}

# ============================================================================
# BOUNDS MODE CONFIGURATION
# ============================================================================
# The bounds in OPTIMIZATION_PHASES above are WIDE bounds for comprehensive search
# Narrow bounds can be generated dynamically based on physics calculations

BOUNDS_MODE_CONFIG = {
    'wide': {
        'description': 'Wide bounds for comprehensive search (original)',
        'tolerance': None,  # Use bounds from OPTIMIZATION_PHASES as-is
        'apply_stability_rules': False,
    },
    'narrow': {
        'description': 'Narrow bounds ±30% around physics-based values',
        'tolerance': 0.3,  # ±30% around expected values
        'apply_stability_rules': True,  # Enforce D < P/10, I = 0.5-2.0×P, filter hierarchy
    },
    'adaptive': {
        'description': 'Start narrow, expand if needed (not yet implemented)',
        'tolerance': 0.3,
        'apply_stability_rules': True,
        'expansion_rate': 1.2,  # Expand by 20% each adaptation
    }
}


def get_optimization_bounds(bounds_mode: str = 'wide'):
    """
    Get parameter bounds based on selected mode

    Args:
        bounds_mode: 'wide', 'narrow', or 'adaptive'

    Returns:
        Dictionary of optimization phases with bounds for each parameter
    """
    if bounds_mode == 'wide':
        # Return original wide bounds from OPTIMIZATION_PHASES
        return OPTIMIZATION_PHASES

    elif bounds_mode == 'narrow':
        # Generate narrow bounds dynamically using physics
        from physics_based_seeding import generate_narrow_bounds

        config = BOUNDS_MODE_CONFIG['narrow']
        narrow_bounds_dict = generate_narrow_bounds(
            drone_params=DRONE_PARAMS,
            tolerance=config['tolerance'],
            apply_stability_rules=config['apply_stability_rules']
        )

        # Merge narrow bounds with phase metadata from OPTIMIZATION_PHASES
        result = {}
        for phase_name, phase_config in OPTIMIZATION_PHASES.items():
            result[phase_name] = {
                'name': phase_config['name'],
                'priority': phase_config['priority'],
                'parameters': phase_config['parameters'],
                'bounds': narrow_bounds_dict[phase_name]
            }

        return result

    elif bounds_mode == 'adaptive':
        # Start with narrow bounds (adaptive expansion not yet implemented)
        # Future: Expand bounds during optimization if convergence stalls
        return get_optimization_bounds('narrow')

    else:
        raise ValueError(f"Unknown bounds_mode: {bounds_mode}. Must be 'wide', 'narrow', or 'adaptive'")
